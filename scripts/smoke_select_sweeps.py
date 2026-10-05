#!/usr/bin/env python3
"""Smoke for the sweep selection scripts (lane L-AM16-ALPHA d2; the lambda rule of AM-2 / DL-06; AM-7a
exclusion of diverged candidates, lane L-KD-HARDEN).

Synthetic inputs only: rule units on candidate records, and end-to-end runs of scripts/select_alpha.py
and scripts/select_lambda.py on synthetic run directories (best.json, run_meta, telemetry and a tiny
checkpoint) under a temp dir. No dataset, no real checkpoint, no GPU; nothing is written in the repo.

Checks (docs/lane_specs/part2.md lane 2 (d) d2, and the cross-lane selection contract):
  alpha  clear winner; tie containing 50 -> 50; tie without 50 -> the smallest alpha; band from s
         above and below the 0.5 pp floor; the band edge is inclusive (decimal); boundary 25/100
         flagged; missing band -> refuse; two candidates -> shortfall_am19_item2 naming AM-19 item 2
         (exit 3).
  lambda clear winner; tie -> the smallest lambda; boundary 0.25/4 flagged; fewer than five finished
         candidates -> shortfall_am19_item2 naming AM-19 item 2 (exit 3), never a selection.
  inputs a run with no run_end record (it died during its final validation) or a torn last
         telemetry line is a shortfall (exit 3); refused (exit 2, never a traceback): a torn middle
         telemetry line, a run_end that disagrees with best.json, a run whose own checks failed, a
         dry run, another seed, --allow-offgrid, a Logit-KD semantics override, a second run_meta
         row, a checkpoint whose best value disagrees with best.json, an unreadable checkpoint (a
         missing one leaves the run finished, ckpt_sha256 null: PL-6), a path under a 'test'
         directory, a band file that is not an object, a missing
         lambda selection file, alpha runs whose lambda is not the lambda selection's winner, an
         existing selection file, a rule file that disagrees with configs/distill.py.
  AM-7a  rule units: the trace line per diverged candidate, a sole finished candidate wins with no band
         and no tie, the edge of the finished set is a boundary (grid_end wins when both), finished and
         diverged together must cover the grid. End to end: lambda 4 diverged under AM-7 (b) without
         best.json -> selected among four, one excluded_am7a entry; lambda 0.25 diverged and 0.5 wins ->
         edge_of_finished_set; all four non-defaults diverged -> lambda 1; lambda 1 diverged -> exit 2
         default_candidate_diverged; an input_nonfinite cause, a val_nonfinite rule, or a nonfinite train
         row (or a val row with a non-finite all-class mIoU) with no run_abort -> exit 2
         run_aborted_other; AM-7 (b) at iter 300, a run_abort after a run_end, or at an iteration other
         than the last train row -> exit 2 abort_record_invalid; a diverged directory left out -> exit 3;
         the same diverged lambda twice -> duplicate_candidate; alpha 25 and 100 diverged -> 50; alpha 50
         diverged -> exit 2; alpha 100 diverged with 25 tied with 50 -> 50. No case prints a traceback,
         and only exit 0 writes a file.
  records (Q2, Q4, Q3; commit 8) an AM-7 record train_distill cannot have written -> exit 2
         abort_record_invalid: AM-7 (b) at iter ramp_iters + 100 (that window only seeds the minimum),
         a ratio "nan" or other than window_mean / running_min, a threshold other than 5 x running_min,
         a window mean not above 5 x the minimum, and (load_candidate units) a non-finite, string,
         boolean, missing or negative field, an integer beyond the float range, or a finite ratio where
         the quotient overflows; AM-7 (a) with a finite loss and grad_norm, at iter 1 (step1_checks
         under R8-1), or whose train row's nonfinite map does not name the non-finite key. Accepted as
         diverged: AM-7 (b) at ramp_iters + 101, a float ratio of exactly 5.0 while window_mean > 5 x
         running_min holds (the pair found by search, both properties asserted), running_min 0 with
         ratio "inf", an overflowing quotient written "inf", and the AM-7 (a) norm case. A diverged
         default candidate beside a num_workers mismatch, or a teacher checkpoint, config or resolved
         model-config hash mismatch -> recipe_mismatch_across_candidates, not default_candidate_diverged.
  10b    the recipe of record: grad_clip_norm 1.0, batch 8, VAL every 2000 or capped, no ImageNet init,
         a float horizon or no teacher provenance -> recipe_mismatch naming the field; num_workers or
         teacher checkpoint, config or model-config hashes differing across the candidates ->
         recipe_mismatch_across_candidates (K2: PL-11, R5).
  cg     L-CKPT-GUARD: a teacher config_sha256 or model_cfg_sha256 absent, None, empty or not a string
         -> recipe_mismatch; select_alpha's five K8-2(a) cases (no run loaded or three unfinished:
         shortfall, exit 3; one wrong-lambda run beside two missing: lambda_mismatch; the default
         diverged at the selection's lambda: default_candidate_diverged; at another lambda:
         lambda_mismatch); the AM-7 comment names R8-1 (K8-2(b)); an AM-7 (a) loss record whose
         grad_norm is not null in its detail (1.7, "nan", 0, False) or its train row (1.7, 0, False,
         key absent) -> abort_record_invalid, end to end and unit (K8-2(c)).
  AM-19  lane K2 (PL-5 to PL-22; PL-39 cases 1-29): the decision dates and the time rule, the schedule
         file, the launch log and its agreement with the run directories, each grid value's status at
         the decision date (finished, diverged, cut, waiting), the on-course test, repeats, the decision
         record, the rule with cut candidates, the alpha decision date (bases a, b, c), the refusal
         classes and the gate's records source. These run in scratch git repositories under the system
         temp dir, each a copy of one template whose commit is the code pin, with GIT_DIR, GIT_WORK_TREE
         and GIT_INDEX_FILE removed and no user or system git configuration (PL-22); no git command runs
         in the checkout. Pre-K2 cases run in such a repository too, launched before both dates, so their
         expectations hold. Every scratch directory the smoke creates is removed at the end.
         --section <name> (repeatable) runs only the named sections (mutation runs).
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import configs.distill as distill_cfg  # noqa: E402
import scripts.select_alpha as sa  # noqa: E402
import scripts.select_lambda as sl  # noqa: E402
import src.training.sweep_select as ss  # noqa: E402
from configs.distill import LOGIT_KD_SEMANTICS  # noqa: E402
from src.training.sweep_select import (RULES_PATH, SelectionRefused, apply_rule,  # noqa: E402
                                       dl27_band, load_candidate, load_rules, sha256_file)

RULES = load_rules()
ALPHA, LAMBDA = RULES["alpha_cwd"], RULES["lambda_logit"]
E1_VALS = {"42": 0.36314016580581665, "43": 0.3598, "44": 0.3662}
LAMBDA_GRID = [0.25, 0.5, 1, 2, 4]
RAMP = 335                               # a first-epoch ramp length; AM-7 (b) is evaluated from RAMP + 100
ABORT_ITER = 12000
TEACHER_SHA = "ab" * 32
# every run_meta carries the three teacher hashes the candidates of a sweep share (L-CKPT-GUARD)
TEACHER_PROV = {"ckpt_sha256": TEACHER_SHA, "config_sha256": "c0" * 32, "model_cfg_sha256": "d0" * 32}
AUTO = object()                          # make_run: the trainer's own train-row grad_norm (see there)
results: list[tuple[str, bool, str]] = []
# AM-19 (lane K2). Times are Unix seconds. Auto-mode launches start at T0, one minute apart, so every pre-K2
# case runs before both decision dates (item 2(f)) with its expectations unchanged.
T0 = 1791244800.0                        # 2026-10-06T00:00:00Z
NOW_BEFORE = 1791590400.0                # 2026-10-10T00:00:00Z: before both decision dates
LAMBDA_C = 1792425600.0                  # the end of 2026-10-19 (Schedule T): 24:00 Asia/Manila = 16:00:00Z
ALPHA_C = 1792684800.0                   # the end of 2026-10-22 (Schedule T)
DAY = 86400.0
_LAUNCHES = itertools.count()
_SCRATCH = itertools.count()
_SCRATCH_DIRS: list = []                 # every directory scratch_dir() created; main() removes them at the end
GIT_ENV_DATE = "2026-10-01T00:00:00Z"


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def cands(pairs) -> list[dict]:
    return [{"status": "finished", "value": float(v), "run_id": f"run_{v:g}", "best_val": b,
             "ckpt_sha256": f"sha{v:g}"} for v, b in pairs]


def refused(fn, *args, **kwargs) -> str | None:
    try:
        fn(*args, **kwargs)
    except SelectionRefused as e:
        return e.code
    return None


# ------------------------------------------------------------------------------ rule units
def test_alpha_rule() -> None:
    r = apply_rule(cands([(25, 0.40), (50, 0.43), (100, 0.41)]), ALPHA, 0.005, "alpha")
    check("alpha_clear_winner", r["winner"]["value"] == 50 and not r["tie"] and not r["boundary"], str(r["tied"]))
    r = apply_rule(cands([(25, 0.430), (50, 0.428), (100, 0.431)]), ALPHA, 0.005, "alpha")
    check("alpha_tie_with_50_goes_to_50", r["winner"]["value"] == 50 and r["tie"]
          and r["tied"] == [25.0, 50.0, 100.0], str(r["tied"]))
    r = apply_rule(cands([(25, 0.430), (50, 0.420), (100, 0.432)]), ALPHA, 0.005, "alpha")
    check("alpha_tie_without_50_goes_to_smallest", r["winner"]["value"] == 25 and r["tie"]
          and r["tied"] == [25.0, 100.0] and r["boundary"], str(r["tied"]))
    r = apply_rule(cands([(25, 0.40), (50, 0.41), (100, 0.43)]), ALPHA, 0.005, "alpha")
    check("alpha_boundary_100_flagged", r["winner"]["value"] == 100 and r["boundary"] and not r["tie"])
    r = apply_rule(cands([(25, 0.44), (50, 0.43), (100, 0.41)]), ALPHA, 0.005, "alpha")
    check("alpha_boundary_25_flagged", r["winner"]["value"] == 25 and r["boundary"] and not r["tie"])
    r = apply_rule(cands([(25, 0.40), (50, 0.395), (100, 0.39)]), ALPHA, 0.005, "alpha")
    check("alpha_band_edge_inclusive_in_decimal", r["winner"]["value"] == 50 and r["tied"] == [25.0, 50.0]
          and (0.40 - 0.395) > 0.005, "0.40 - 0.395 is 0.0050000000000000044 in binary floats")
    # band from s below and above the 0.5 pp floor, and its effect on the same three values
    low = dict(E1_VALS, **{})
    s_low = __import__("statistics").stdev(low.values())
    s_doc = {"e1_best_val": low, "s": s_low}
    s, band, _ = dl27_band(s_doc, 0.005)
    check("band_floor_when_sqrt2_s_below", math.sqrt(2) * s < 0.005 and band == 0.005, f"s={s!r}")
    wide = {"42": 0.36314016580581665, "43": 0.3431, "44": 0.3802}
    s_w = __import__("statistics").stdev(wide.values())
    s2, band2, _ = dl27_band({"e1_best_val": wide, "s": s_w, "band": math.sqrt(2) * s_w}, 0.005)
    check("band_sqrt2_s_when_above", band2 == math.sqrt(2) * s2 and band2 > 0.005, f"band={band2!r}")
    vals = [(25, 0.430), (50, 0.421), (100, 0.431)]
    narrow = apply_rule(cands(vals), ALPHA, band, "alpha")["winner"]["value"]
    widened = apply_rule(cands(vals), ALPHA, band2, "alpha")["winner"]["value"]
    check("band_width_changes_the_tie", narrow == 25 and widened == 50, f"{narrow} vs {widened}")
    check("band_s_mismatch_refused", refused(dl27_band, {"e1_best_val": low, "s": s_low + 1e-6}, 0.005)
          == "band_s_mismatch")
    check("band_value_mismatch_refused", refused(dl27_band, {"e1_best_val": low, "s": s_low,
                                                             "band": 0.006}, 0.005) == "band_mismatch")
    check("band_missing_seed_refused", refused(dl27_band, {"e1_best_val": {"42": 0.36, "43": 0.36},
                                                           "s": 0.0}, 0.005) == "band_format")
    check("alpha_two_candidates_refused",
          refused(apply_rule, cands([(25, 0.4), (50, 0.4)]), ALPHA, 0.005, "alpha") == "partial_input")
    check("alpha_offgrid_candidate_refused",
          refused(apply_rule, cands([(25, 0.4), (30, 0.4), (100, 0.4)]), ALPHA, 0.005, "alpha")
          == "partial_input")


def test_lambda_rule() -> None:
    grid = [0.25, 0.5, 1, 2, 4]
    r = apply_rule(cands(zip(grid, [0.40, 0.41, 0.43, 0.42, 0.40])), LAMBDA, 0.005, "lambda")
    check("lambda_clear_winner", r["winner"]["value"] == 1 and not r["tie"] and not r["boundary"])
    r = apply_rule(cands(zip(grid, [0.40, 0.428, 0.43, 0.427, 0.41])), LAMBDA, 0.005, "lambda")
    check("lambda_tie_goes_to_smallest", r["winner"]["value"] == 0.5 and r["tied"] == [0.5, 1.0, 2.0])
    r = apply_rule(cands(zip(grid, [0.44, 0.41, 0.42, 0.40, 0.40])), LAMBDA, 0.005, "lambda")
    check("lambda_boundary_0.25_flagged", r["winner"]["value"] == 0.25 and r["boundary"])
    r = apply_rule(cands(zip(grid, [0.40, 0.41, 0.42, 0.40, 0.43])), LAMBDA, 0.005, "lambda")
    check("lambda_boundary_4_flagged", r["winner"]["value"] == 4 and r["boundary"])
    check("lambda_four_candidates_refused",
          refused(apply_rule, cands(zip(grid[:4], [0.4] * 4)), LAMBDA, 0.005, "lambda") == "partial_input")


# ------------------------------------------------------------------- synthetic run dirs
def abort_record(it: int, rule: str = "AM-7(b)", cause: str = "student_divergence", detail=None,
                 input_finite=True, teacher_finite=True, ts: float = 1.0) -> dict:
    """train_distill's run_abort row (AM-7a section A); fields not computed at the abort point are null
    and a non-finite float is written as a string. The AM-7 (b) detail is one the trainer can write
    (window_mean 3.12 > 5 x running_min 0.6, threshold 5 x 0.6, ratio 3.12 / 0.6); the AM-7 (a) detail
    records a NaN loss. `detail` overrides fields."""
    if rule == "AM-7(b)":
        base = {"loss": 3.12, "grad_norm": 1.7, "window_mean": 3.12, "running_min": 0.6, "ratio": 3.12 / 0.6,
                "threshold": 5.0 * 0.6}
    else:
        base = {"loss": "nan", "grad_norm": None, "window_mean": None, "running_min": 0.6, "ratio": None,
                "threshold": None}
    return {"event": "run_abort", "iter": it, "rule": rule, "cause": cause,
            "detail": {**base, **(detail or {})}, "input_finite": input_finite,
            "teacher_finite": teacher_finite, "params_finite": True, "n_val": it // 4000, "wall_clock": ts}


def make_run(root: Path, stage: str, value: float, best_val: float, *, name: str | None = None,
             finished=True, mode="real", seed=42, offgrid=False, rows=1, ckpt_best=None,
             with_ckpt=True, lam=1.0, override=False, run_end=True, checks_passed=True,
             end_best=None, torn="", corrupt_ckpt=False, abort=None, abort_after_end=False,
             record_iter=None, nonfinite=False, with_best=True, recipe=None, val_nonfinite=None,
             row_grad_norm=AUTO, launch=None) -> Path:
    """A synthetic run directory. `abort` (run_abort record kwargs) ends the telemetry at the abort
    iteration with a run_abort row and no run_end; `abort_after_end` appends it after a run_end instead;
    `nonfinite` (True for the loss, or a {key: tag} map) flags the last train row as train_distill does
    (null + nonfinite map); an abort at iteration 1 has that single train row; `val_nonfinite`
    (a val-row key) adds a val row at the last iteration with that value non-finite; `recipe` overrides
    run_meta's recipe fields. A train row whose loss is non-finite holds grad_norm null, as the trainer
    writes it (the loss stop precedes backward); `row_grad_norm` sets another value, or None to drop the
    key (K8-2(c)). Every row carries the trainer's timestamps (AM-19): the run launches at `launch` (default:
    the next auto launch, T0 + n minutes), its train row at iteration i at launch + i s, a val row 0.25 s
    after its train row, an abort 0.5 s after, run_end 1 s after the last train row."""
    sk = stage.lower()
    launch = T0 + 60.0 * next(_LAUNCHES) if launch is None else float(launch)
    d = root / (name or (f"{sk}_s42_alpha{value:g}" if stage == "E3" else f"{sk}_s42_lambda{value:g}"))
    d.mkdir(parents=True)
    terms = {"logit_kd": True, "cwd_feat": stage == "E3", "cwd_logit": stage == "E3"}
    meta = {"event": "run_meta", "stage": stage, "mode": mode, "seed": seed, "terms": terms,
            "projection_params": 51200 if stage == "E3" else 0, "lambda_logit": lam if stage == "E3" else value,
            "logit_kd_semantics": LOGIT_KD_SEMANTICS, "logit_kd_semantics_declared": None,
            "logit_kd_semantics_override_used": override, "max_iters": 80000, "wall_clock": launch}
    if stage == "E3":
        meta.update({"alpha_cwd": value, "alpha_offgrid": offgrid, "beta_cwd": 3})
    meta.update({"num_workers": 12, "batch_size": 16, "val_interval": 4000, "max_val_batches": None,
                 "poly_horizon": 80000, "grad_clip_norm": None, "used_pretrained": True, "ramp_iters": RAMP,
                 "teacher_provenance": dict(TEACHER_PROV)})
    meta.update(recipe or {})
    (d / f"{sk}_run_meta.jsonl").write_text("".join(json.dumps(meta) + "\n" for _ in range(rows)),
                                            encoding="utf-8")
    last = 80000 if finished else 79999
    if abort is not None and not abort_after_end:
        last = abort.get("it", ABORT_ITER)
    ck_name = f"{sk}_student_best_iter76000.pt"
    named = ({"loss": "nan"} if nonfinite is True else dict(nonfinite)) if nonfinite else {}
    last_row = {"event": "train", "iter": last, "loss": 1.0, **{k: None for k in named},
                "wall_clock": launch + last, "iter_seconds": 1.0}
    if "loss" in named:
        last_row.setdefault("grad_norm", None)
    if row_grad_norm is None:
        last_row.pop("grad_norm", None)
    elif row_grad_norm is not AUTO:
        last_row["grad_norm"] = row_grad_norm
    if named:
        last_row["nonfinite"] = named
    lines = ([json.dumps({"event": "train", "iter": last - 1, "loss": 1.0, "wall_clock": launch + last - 1,
                          "iter_seconds": 1.0})] if last > 1 else []) + [json.dumps(last_row)]
    if val_nonfinite is not None:
        val = {"event": "val", "iter": last, "all_class_miou": best_val,
               "disease_only_miou_PROVISIONAL": best_val, "val_seconds": 0.25, "wall_clock": launch + last + 0.25}
        lines.append(json.dumps({**val, val_nonfinite: None, "nonfinite": {val_nonfinite: "nan"}}))
    if torn == "middle":
        lines.insert(1, '{"event": "train", "iter": 7')
    if finished and run_end and (abort is None or abort_after_end):
        lines.append(json.dumps({"event": "run_end", "iter": 80000,
                                 "best_val_miou_all_class": best_val if end_best is None else end_best,
                                 "best_ckpt": ck_name, "checks_passed": checks_passed,
                                 "wall_clock_start": launch, "wall_clock_end": launch + 80001.0}))
    if abort is not None:
        kw = {k: v for k, v in abort.items() if k != "it"}
        lines.append(json.dumps(abort_record(last if record_iter is None else record_iter,
                                             ts=launch + last + 0.5, **kw)))
    text = "\n".join(lines) + "\n"
    if torn == "last":
        text += '{"event": "train", "it'
    (d / f"{sk}_telemetry.jsonl").write_text(text, encoding="utf-8")
    if with_ckpt and corrupt_ckpt:
        (d / ck_name).write_bytes(b"\x00not a checkpoint")
    elif with_ckpt:
        payload = {"stage": stage, "iter": 76000,
                   "best_val_miou_all_class": best_val if ckpt_best is None else ckpt_best,
                   ("alpha_cwd" if stage == "E3" else "lambda_logit"): value}
        torch.save(payload, d / ck_name)
    if with_best:
        (d / "best.json").write_text(json.dumps({"best_ckpt": f"/workspace/{d.name}/{ck_name}",
                                                 "best_val_miou_all_class": best_val}, indent=2),
                                     encoding="utf-8")
    return d


def band_file(root: Path) -> Path:
    s = __import__("statistics").stdev(E1_VALS.values())
    p = root / "dl27_band.json"
    p.write_text(json.dumps({"e1_best_val": E1_VALS, "s": s}), encoding="utf-8")
    return p


def lambda_selection_file(root: Path, lam: float = 1.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / "lambda_selection.json"
    p.write_text(json.dumps({"format": "lambda_selection/1",
                             "winner": {"lambda": lam, "run_id": "e2_s42_lambda1"}}), encoding="utf-8")
    return p


# ------------------------------------------------------------- scratch repositories (lane K2)
def scratch_dir(prefix: str) -> Path:
    """A new directory under the system temp dir, named <prefix>_<pid>_<n>: no random suffix can spell
    'test' (SL-1), and it lies outside the repository."""
    while True:
        p = Path(tempfile.gettempdir()) / f"{prefix}_{os.getpid()}_{next(_SCRATCH)}"
        if "test" in str(p).lower():
            raise SystemExit(f"refusing to create {p}: its path contains 'test' (SL-1); set TMPDIR elsewhere")
        if REPO == p.resolve() or REPO in p.resolve().parents:
            raise SystemExit(f"refusing to create {p} inside the repository")
        try:
            p.mkdir()
        except FileExistsError:
            continue
        _SCRATCH_DIRS.append(p)
        return p


def git(repo: Path, *args: str) -> str:
    """A git command in a scratch repository (PL-22): -C <scratch>, never the checkout, with GIT_DIR,
    GIT_WORK_TREE and GIT_INDEX_FILE removed and no user or system configuration."""
    repo = Path(repo).resolve()
    assert repo != REPO and REPO not in repo.parents, repo
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_DATE=GIT_ENV_DATE,
               GIT_COMMITTER_DATE=GIT_ENV_DATE)
    r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=k2-smoke", "-c", "user.email=k2-smoke@localhost",
                        "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=master", *args],
                       capture_output=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {r.stderr.decode(errors='replace')[-300:]}")
    return r.stdout.decode("utf-8", "replace").strip()


def commit_files(repo: Path, files: dict, msg: str) -> str:
    """Write {repository path: bytes or str} into `repo`, stage those paths by name and commit; returns HEAD."""
    for rel, data in files.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    if files:
        git(repo, "add", "--", *files)
    git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


DL_TEXT = ("# Decision log (synthetic; smoke_select_sweeps)\n\n| ID | Status | Decision | Where | Ch4 |\n"
           "|---|---|---|---|---|\n| DL-70 | RECORDED | AM-19 (synthetic row) | - | yes |\n"
           "| DL-90 | RECORDED | Schedule R: the adviser's reply (synthetic row) | - | no |\n")
_TEMPLATE: list = []


def code_files() -> dict:
    """What the scratch repositories commit at their code pin: the running modules' bytes (so the code the
    selection runs is committed there, PL-16), the rule and schedule files, and a synthetic decision log."""
    return {ss.SWEEP_SELECT_REL: Path(ss.__file__).read_bytes(),
            ss.SELECT_SCRIPT_REL["lambda_logit"]: Path(sl.__file__).read_bytes(),
            ss.SELECT_SCRIPT_REL["alpha_cwd"]: Path(sa.__file__).read_bytes(),
            ss.DISTILL_REL: Path(distill_cfg.__file__).read_bytes(),
            ss.RULES_REL: (REPO / ss.RULES_REL).read_bytes(),
            ss.SCHEDULE_REL: (REPO / ss.SCHEDULE_REL).read_bytes(),
            ss.DECISION_LOG_REL: DL_TEXT}


def template() -> tuple[Path, str]:
    """A scratch repository whose one commit (the code pin) holds code_files(); built once, then copied."""
    if not _TEMPLATE:
        root = scratch_dir("k2_template")
        git(root, "init", "-q")
        _TEMPLATE.extend([root, commit_files(root, code_files(), "pin")])
    return _TEMPLATE[0], _TEMPLATE[1]


def new_repo() -> Path:
    root = scratch_dir("k2_repo")
    shutil.copytree(template()[0], root, dirs_exist_ok=True)
    return root


def launch_log_text(lines: list) -> bytes:
    return "".join(json.dumps(x) + "\n" for x in lines).encode("utf-8")


def launch_line(run_id: str, stage: str, value: float, lam, alpha, attempt: int, pin: str, *, report=None,
                sweep=None, seed=42, horizon=80000, schedule="T") -> dict:
    return {"format": ss.LAUNCH_LOG_FORMAT, "event": "launch", "run_id": run_id, "stage": stage, "seed": seed,
            "horizon": horizon, "sweep": sweep or ("lambda_logit" if stage == "E2" else "alpha_cwd"),
            "value": value, "lambda_logit": lam, "alpha_cwd": alpha, "attempt": attempt, "schedule": schedule,
            "code_pin": pin, "ckpt_dir": f"/workspace/kd_ckpts/{run_id}", "am8a_report": report}


def set_meta(meta_p: Path, **fields) -> None:
    """Set fields in every row of a run_meta file (the rows stay identical to one another)."""
    rows = [json.loads(ln) for ln in meta_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta_p.write_text("".join(json.dumps({**r, **fields}) + "\n" for r in rows), encoding="utf-8")


def launched_line(run_dir: Path, stage: str, key: str, value: float, pin: str, records: str, **over) -> dict:
    meta_p = run_dir / f"{stage.lower()}_run_meta.jsonl"
    first = json.loads(meta_p.read_text(encoding="utf-8").splitlines()[0])
    pos = [float(v) for v in ss.LAUNCH_ORDER[key]].index(float(value)) + 1
    line = {"event": "launched", "run_id": run_dir.name, "run_meta_sha256": sha256_file(meta_p),
            "launch_time_utc": first["wall_clock"], "git_head": pin, "records_commit": records,
            "decision_date": ss.AM19_DECISION_DATES[key]["T"], "launch_order_position": pos}
    line.update(over)
    return line


def g_lines(pin: str, records: str, launch_ts: float) -> tuple[dict, dict]:
    """A KD launch outside the sweeps (stage G, seed 42): its launch line and its launched line."""
    line = {"format": ss.LAUNCH_LOG_FORMAT, "event": "launch", "run_id": "g_s42_a1", "stage": "G", "seed": 42,
            "horizon": 80000, "sweep": None, "value": None, "lambda_logit": None, "alpha_cwd": None, "attempt": 1,
            "schedule": "T", "code_pin": pin, "ckpt_dir": "/workspace/kd_ckpts/g_s42_a1", "am8a_report": None}
    launched = {"event": "launched", "run_id": "g_s42_a1", "run_meta_sha256": "0" * 64,
                "launch_time_utc": float(launch_ts), "git_head": pin, "records_commit": records,
                "decision_date": None, "launch_order_position": None}
    return line, launched


def _parse_argv(argv) -> dict:
    opts, cur = {}, None
    for tok in argv:
        if tok.startswith("--"):
            cur = tok
            opts.setdefault(cur, [])
        else:
            opts[cur].append(tok)
    return opts


def auto_repo(module, argv) -> tuple[Path, list]:
    """Auto mode: the repository a pre-K2 case needs, built from its arguments. The given band and lambda
    selection files are committed at their paths of record (a missing one stays missing); each existing
    --runs directory gets a launch line (the sweep's stage, seed and horizon; its own value, lambda and alpha;
    attempts numbered per value) in a records commit; its run_meta gets git_head (the pin) and
    records_commit; a launched line follows, at HEAD. A path containing 'test' is never touched (SL-1)."""
    key = "lambda_logit" if module is sl else "alpha_cwd"
    stage = "E2" if key == "lambda_logit" else "E3"
    repo, pin = new_repo(), template()[1]
    opts = _parse_argv(argv)
    files = {}
    for opt, rel in (("--band", ss.BAND_REL), ("--lambda-selection", ss.LAMBDA_SELECTION_REL)):
        if opt in opts:
            given = Path(opts[opt][0])
            if "test" not in str(given).lower() and given.is_file():
                files[rel] = given.read_bytes()
            opts[opt] = [str(repo / rel)]
    runs, seen, groups, lines = [], set(), {}, []
    for d in opts.get("--runs", []):
        p = Path(d)
        if "test" in str(p).lower() or not p.is_dir() or p.resolve() in seen:
            continue
        seen.add(p.resolve())
        try:
            meta = json.loads((p / f"{stage.lower()}_run_meta.jsonl").read_text(encoding="utf-8").splitlines()[0])
        except (OSError, IndexError, json.JSONDecodeError):
            meta = {}
        value = meta.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = float(p.name.split("lambda" if key == "lambda_logit" else "alpha")[1].split("_")[0])
        lam = float(value) if key == "lambda_logit" else meta.get("lambda_logit", 1.0)
        alpha = None if key == "lambda_logit" else float(value)
        groups[(lam, alpha)] = groups.get((lam, alpha), 0) + 1
        lines.append(launch_line(p.name, stage, float(value), lam, alpha, groups[(lam, alpha)], pin))
        runs.append(p)
    records = commit_files(repo, {**files, ss.LAUNCH_LOG_REL: launch_log_text(lines)}, "launch lines")
    events = []
    for p, line in zip(runs, lines):
        meta_p = p / f"{stage.lower()}_run_meta.jsonl"
        if meta_p.is_file():
            set_meta(meta_p, git_head=pin, records_commit=records)
            events.append(launched_line(p, stage, key, line["value"], pin, records))
    commit_files(repo, {ss.LAUNCH_LOG_REL: launch_log_text(lines + events)}, "launched lines")
    return repo, [tok for opt, vals in opts.items() for tok in (opt, *vals)]


def run_cli(module, argv, *, now: float = NOW_BEFORE, repo: Path | None = None) -> tuple[int, str]:
    """main(argv) in-process, with sweep_select.GIT_ROOT at a scratch repository (auto mode builds it from
    the arguments when `repo` is None) and sweep_select.now_utc() returning `now`."""
    if repo is None:
        repo, argv = auto_repo(module, list(argv))
    saved = ss.GIT_ROOT, ss.now_utc
    ss.GIT_ROOT, ss.now_utc = repo, (lambda: now)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = module.main(argv)
    finally:
        ss.GIT_ROOT, ss.now_utc = saved
    return rc, out.getvalue() + err.getvalue()


def test_alpha_cli(tmp: Path) -> None:
    root = tmp / "alpha_ok"
    runs = [make_run(root, "E3", a, b) for a, b in ((25, 0.430), (50, 0.428), (100, 0.431))]
    band, lam, out = band_file(root), lambda_selection_file(root), tmp / "alpha_selection.json"
    common = ["--band", str(band), "--lambda-selection", str(lam)]
    rc, log = run_cli(sa, ["--runs", *map(str, runs), *common, "--out", str(out)])
    doc = json.loads(out.read_text()) if out.exists() else {}
    want_keys = {"candidates", "s", "band", "rule_trace", "winner", "tie", "boundary"}
    check("alpha_cli_selects", rc == 0 and want_keys <= set(doc) and doc["winner"]["alpha"] == 50
          and doc["tie"] and doc["band"] == 0.005 and doc["lambda_logit"] == 1.0
          and doc.get("lambda_selection_file", {}).get("lambda") == 1.0, log[-200:])
    check("alpha_cli_candidates_record_existing_checkpoints",
          [c["alpha"] for c in doc.get("candidates", [])] == [25.0, 50.0, 100.0]
          and all(c["ckpt_sha256"] == sha256_file(r / "e3_student_best_iter76000.pt")
                  for c, r in zip(doc.get("candidates", []), runs))
          and doc.get("winner", {}).get("run_id") == "e3_s42_alpha50")
    rc, log = run_cli(sa, ["--runs", *map(str, runs), *common, "--out", str(out)])
    check("alpha_cli_never_overwrites", rc == 2 and "output_exists" in log)
    rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(root / "absent.json"),
                           "--lambda-selection", str(lam), "--out", str(tmp / "a_nob.json")])
    check("alpha_cli_missing_band_refused", rc == 2 and "band_missing" in log)
    listed = root / "band_list.json"
    listed.write_text("[0.005]", encoding="utf-8")
    rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(listed), "--lambda-selection",
                           str(lam), "--out", str(tmp / "a_list.json")])
    check("alpha_cli_band_not_an_object_refused", rc == 2 and "band_format" in log)
    rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(band), "--lambda-selection",
                           str(root / "absent_lambda.json"), "--out", str(tmp / "a_nol.json")])
    check("alpha_cli_missing_lambda_selection_refused", rc == 2 and "lambda_selection_missing" in log)
    rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(band), "--lambda-selection",
                           str(lambda_selection_file(tmp / "lam_half", 0.5)), "--out",
                           str(tmp / "a_lam.json")])
    check("alpha_cli_lambda_other_than_selection_refused", rc == 2 and "lambda_mismatch" in log)
    rc, log = run_cli(sa, ["--runs", *map(str, runs[:2]), *common, "--out", str(tmp / "a_two.json")])
    # K2 (E-44): renamed from alpha_cli_two_runs_refused_lane2_stop; exit 3 unchanged, the text names AM-19 item 2
    check("alpha_cli_two_runs_refused_naming_am19_item2", rc == 3 and "AM-19 item 2" in log
          and "shortfall_am19_item2" in log and not (tmp / "a_two.json").exists())
    cases = {"unfinished": (dict(finished=False), 3),
             "no_run_end_died_in_final_val": (dict(run_end=False), 3),
             "torn_last_telemetry_line": (dict(run_end=False, torn="last"), 3),
             "torn_middle_telemetry_line": (dict(torn="middle"), 2),
             "run_end_best_mismatch": (dict(end_best=0.2), 2),
             "run_checks_failed": (dict(checks_passed=False), 2),
             "dry_run": (dict(mode="dry"), 2), "seed_43": (dict(seed=43), 2),
             "offgrid": (dict(offgrid=True), 2), "two_run_meta_rows": (dict(rows=2), 2),
             "semantics_override": (dict(override=True), 2),
             "checkpoint_value_mismatch": (dict(ckpt_best=0.1), 2),
             "checkpoint_unreadable": (dict(corrupt_ckpt=True), 2),
             "mixed_lambda": (dict(lam=0.5), 2)}
    for label, (kw, code) in cases.items():
        r2 = tmp / f"alpha_{label}"
        bad = [make_run(r2, "E3", 25, 0.43), make_run(r2, "E3", 50, 0.42),
               make_run(r2, "E3", 100, 0.41, **kw)]
        o = tmp / f"a_{label}.json"
        rc, log = run_cli(sa, ["--runs", *map(str, bad), *common, "--out", str(o)])
        check(f"alpha_cli_refuses_{label}", rc == code and not o.exists() and "Traceback" not in log,
              f"rc={rc} {log.strip()[-160:]}")
    # K2 (PL-6, AM-19 item 2(b): finished is final): replaces alpha_cli_refuses_checkpoint_missing (was exit 2).
    # A finished run whose checkpoint is absent stays finished, with ckpt_sha256 null and checkpoint_present false.
    r2 = tmp / "alpha_checkpoint_missing"
    fin = [make_run(r2, "E3", 25, 0.43), make_run(r2, "E3", 50, 0.42), make_run(r2, "E3", 100, 0.41, with_ckpt=False)]
    o = tmp / "a_checkpoint_missing.json"
    rc, log = run_cli(sa, ["--runs", *map(str, fin), *common, "--out", str(o)])
    doc = json.loads(o.read_text()) if o.exists() else {}
    c100 = next((c for c in doc.get("candidates", []) if c["alpha"] == 100), {})
    check("alpha_cli_finished_without_checkpoint_is_final", rc == 0 and c100.get("ckpt_sha256") is None
          and c100.get("checkpoint_present") is False and doc.get("winner", {}).get("alpha") == 25,
          f"rc={rc} {c100} {log.strip()[-160:]}")
    # PL-4 (SL-1): three run paths under a directory named 'test' that are never created. The
    # selection refuses them by path (refuse_test_path) before its is_dir() check, so nothing is
    # created or written.
    trap = [tmp / "x" / "test" / f"e3_s42_alpha{a}" for a in (25, 50, 100)]
    rc, log = run_cli(sa, ["--runs", *map(str, trap), *common, "--out", str(tmp / "a_t.json")])
    check("alpha_cli_refuses_test_path", rc == 2 and "test_path" in log)


def test_lambda_cli(tmp: Path) -> None:
    root = tmp / "lambda_ok"
    grid = [0.25, 0.5, 1, 2, 4]
    runs = [make_run(root, "E2", v, b) for v, b in zip(grid, [0.40, 0.428, 0.43, 0.427, 0.41])]
    out = tmp / "lambda_selection.json"
    rc, log = run_cli(sl, ["--runs", *map(str, runs), "--out", str(out)])
    doc = json.loads(out.read_text()) if out.exists() else {}
    check("lambda_cli_selects", rc == 0 and doc.get("winner", {}).get("lambda") == 0.5
          and doc.get("tie") and doc.get("band") == 0.005 and "rule_trace" in doc, log[-200:])
    # K2 (E-44): the four lambda shortfall checks below were named *_naming_am17_item9 and looked for
    # "AM-17 item 9"; exit 3 is unchanged and the text names AM-19 item 2.
    rc, log = run_cli(sl, ["--runs", *map(str, runs[:4]), "--out", str(tmp / "l_four.json")])
    check("lambda_cli_four_runs_refused_naming_am19_item2", rc == 3 and "AM-19 item 2" in log
          and "AM-17 item 9" not in log and not (tmp / "l_four.json").exists(), log.strip()[-160:])
    r2 = tmp / "lambda_unfinished"
    runs2 = [make_run(r2, "E2", v, 0.4, finished=(v != 4)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs2), "--out", str(tmp / "l_unf.json")])
    check("lambda_cli_unfinished_refused_naming_am19_item2", rc == 3 and "AM-19 item 2" in log
          and "run_unfinished" in log and not (tmp / "l_unf.json").exists())
    r3 = tmp / "lambda_override"
    runs3 = [make_run(r3, "E2", v, 0.4, override=(v == 1)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs3), "--out", str(tmp / "l_ovr.json")])
    check("lambda_cli_semantics_override_refused", rc == 2 and "semantics" in log)
    r4 = tmp / "lambda_died_in_final_val"
    runs4 = [make_run(r4, "E2", v, 0.4, run_end=(v != 2)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs4), "--out", str(tmp / "l_end.json")])
    check("lambda_cli_no_run_end_refused_naming_am19_item2", rc == 3 and "AM-19 item 2" in log
          and not (tmp / "l_end.json").exists(), log.strip()[-160:])
    r5 = tmp / "lambda_torn_last"
    runs5 = [make_run(r5, "E2", v, 0.4, run_end=(v != 1), torn=("last" if v == 1 else "")) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs5), "--out", str(tmp / "l_torn.json")])
    check("lambda_cli_torn_last_line_is_shortfall_naming_am19_item2", rc == 3 and "AM-19 item 2" in log
          and "Traceback" not in log, log.strip()[-160:])


def test_rules_file(tmp: Path) -> None:
    check("rules_file_agrees_with_config", RULES["format"] == "sweep_rules/1"
          and ALPHA["grid"] == [25, 50, 100] and LAMBDA["grid"] == [0.25, 0.5, 1, 2, 4])
    doc = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    doc["alpha_cwd"]["grid"] = [25, 50, 75]
    bad = tmp / "rules_bad.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    check("rules_file_disagreeing_with_config_refused", refused(load_rules, bad) == "rules_mismatch")
    check("rules_file_default_candidates", LAMBDA["default_candidate"] == 1
          and ALPHA["default_candidate"] == 50
          and all("a diverged non-default candidate is not a shortfall: its directory is supplied and it is "
                  "excluded (AM-7a)" in RULES[k]["shortfall"] for k in ("lambda_logit", "alpha_cwd")))
    for name, value, code in (("lambda_logit", 2, "rules_mismatch"), ("alpha_cwd", 25, "rules_mismatch"),
                              ("lambda_logit", None, "rules_format"), ("alpha_cwd", "null", "rules_format"),
                              ("lambda_logit", "one", "rules_format"), ("alpha_cwd", True, "rules_format")):
        doc = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        if value is None:
            del doc[name]["default_candidate"]
        else:
            doc[name]["default_candidate"] = None if value == "null" else value
        bad = tmp / f"rules_dc_{name}_{value}.json"
        bad.write_text(json.dumps(doc), encoding="utf-8")
        check(f"rules_file_default_candidate_{name}_{value}_refused", refused(load_rules, bad) == code,
              str(refused(load_rules, bad)))
    import src.training.sweep_select as ss_mod
    saved_window, ss_mod.AM7B_WINDOW = ss_mod.AM7B_WINDOW, 50
    try:
        ss_mod.load_rules(RULES_PATH)
        got = ("accepted", "")
    except SelectionRefused as e:
        got = (e.code, str(e))
    finally:
        ss_mod.AM7B_WINDOW = saved_window
    check("rules_am7b_constants_must_match_config", got[0] == "rules_mismatch"
          and "AM7_DIVERGENCE" in got[1] and "sweep_rules" not in got[1], str(got)[:200])


# ------------------------------------------------------------------------ AM-7a (L-KD-HARDEN)
def div(v, it: int = ABORT_ITER, rule: str = "AM-7(b)") -> dict:
    """A diverged candidate record as load_candidate returns it (rule units)."""
    ab = abort_record(it, rule)
    return {"status": "diverged", "value": float(v), "run_id": f"run_{v:g}",
            "abort": {k: ab[k] for k in ("iter", "rule", "cause", "detail")}, "n_val": ab["n_val"],
            "best_val_partial": None, "telemetry_sha256": "t", "run_meta_sha256": "m"}


def test_am7a_rule() -> None:
    fin = cands([(0.5, 0.44), (1, 0.42), (2, 0.41), (4, 0.40)])
    r = apply_rule(fin, LAMBDA, 0.005, "lambda", diverged=[div(0.25)])
    check("am7a_rule_edge_of_finished_set_is_a_boundary", r["winner"]["value"] == 0.5 and r["boundary"]
          and r["boundary_kind"] == "edge_of_finished_set" and r["n_finished"] == 4 and r["n_diverged"] == 1,
          f"{r['winner']['value']} {r['boundary_kind']}")
    check("am7a_rule_trace_names_the_excluded_run",
          "lambda=0.25 run=run_0.25: diverged (AM-7a) at iter 12000 under AM-7(b); excluded from the "
          "selection" in r["rule_trace"], str(r["rule_trace"][:2]))
    check("am7a_rule_excluded_entry_schema",
          [sorted(e) for e in r["excluded_am7a"]] == [sorted(["value", "run_id", "abort", "n_val",
                                                              "best_val_partial", "telemetry_sha256",
                                                              "run_meta_sha256"])]
          and sorted(r["excluded_am7a"][0]["abort"]) == ["cause", "detail", "iter", "rule"])
    r = apply_rule(cands([(1, 0.40)]), LAMBDA, 0.005, "lambda", diverged=[div(v) for v in (0.25, 0.5, 2, 4)])
    check("am7a_rule_sole_finished_wins_without_band_or_tie", r["winner"]["value"] == 1 and not r["tie"]
          and r["tied"] == [1.0] and r["n_finished"] == 1 and r["n_diverged"] == 4
          and any(t.startswith("sole finished candidate (AM-7a): no band, no tie") for t in r["rule_trace"])
          and sum("diverged (AM-7a)" in t for t in r["rule_trace"]) == 4)
    # K2 (PL-18): apply_rule now refuses a finished set without the default (a diverged default refuses the
    # selection), so grid_end's precedence over the edge flag is checked on the boundary helper itself.
    b, kind, removed = ss.boundary_kind(4.0, LAMBDA, [4.0], [(v, "AM-7a") for v in (0.25, 0.5, 1.0, 2.0)])
    check("am7a_rule_grid_end_wins_over_edge", b and kind == "grid_end" and [e["value"] for e in removed]
          == [0.25, 0.5, 1.0, 2.0], f"{b} {kind} {removed}")
    check("pl18_apply_rule_refuses_a_finished_set_without_the_default",
          refused(apply_rule, cands([(4, 0.40)]), LAMBDA, 0.005, "lambda",
                  diverged=[div(v) for v in (0.25, 0.5, 1, 2)]) == "partial_input")
    r = apply_rule(cands(zip(LAMBDA_GRID, [0.40, 0.41, 0.43, 0.42, 0.40])), LAMBDA, 0.005, "lambda")
    check("am7a_rule_without_divergence_unchanged", r["winner"]["value"] == 1 and r["boundary_kind"] is None
          and r["excluded_am7a"] == [] and r["n_finished"] == 5 and r["n_diverged"] == 0)
    check("am7a_rule_all_diverged_refused",
          refused(apply_rule, [], LAMBDA, 0.005, "lambda", diverged=[div(v) for v in LAMBDA_GRID])
          == "partial_input")
    check("am7a_rule_finished_and_diverged_must_cover_the_grid",
          refused(apply_rule, fin, LAMBDA, 0.005, "lambda") == "partial_input"
          and refused(apply_rule, fin, LAMBDA, 0.005, "lambda", diverged=[div(0.25), div(4)])
          == "partial_input")


DIVERGED = dict(abort={}, with_best=False)          # AM-7 (b) at ABORT_ITER, ratio 5.2, no best.json


def result_line(log: str) -> str:
    return next((ln for ln in log.splitlines() if ln.startswith("RESULT:")), "")


def lambda_case(tmp: Path, label: str, over: dict, vals=(0.40, 0.428, 0.43, 0.427, 0.41), extra=()):
    """Five E2 runs; `over` maps a lambda to make_run kwargs and `extra` adds (lambda, kwargs) runs.
    Returns (rc, log, the selection doc or None, the output path, the run dirs)."""
    root = tmp / f"am7a_{label}"
    runs = [make_run(root, "E2", v, b, **over.get(v, {})) for v, b in zip(LAMBDA_GRID, vals)]
    runs += [make_run(root, "E2", v, 0.4, name=f"e2_s42_lambda{v:g}_{i}", **kw)
             for i, (v, kw) in enumerate(extra)]
    out = tmp / f"l_{label}.json"
    rc, log = run_cli(sl, ["--runs", *map(str, runs), "--out", str(out)])
    return rc, log, (json.loads(out.read_text()) if out.exists() else None), out, runs


def test_am7a_lambda_cli(tmp: Path) -> None:
    rc, log, doc, out, runs = lambda_case(tmp, "l4_diverged", {4: DIVERGED})
    ex = (doc or {}).get("excluded_am7a", [])
    check("am7a_lambda4_diverged_selects_among_four",
          rc == 0 and doc is not None and doc["winner"]["lambda"] == 0.5 and doc["n_finished"] == 4
          and doc["n_diverged"] == 1 and len(ex) == 1 and ex[0]["value"] == 4.0
          and ex[0]["abort"]["rule"] == "AM-7(b)" and ex[0]["abort"]["iter"] == ABORT_ITER
          and ex[0]["abort"]["cause"] == "student_divergence" and ex[0]["best_val_partial"] is None
          and ex[0]["telemetry_sha256"] == sha256_file(runs[4] / "e2_telemetry.jsonl")
          and ex[0]["run_meta_sha256"] == sha256_file(runs[4] / "e2_run_meta.jsonl")
          and [c["lambda"] for c in doc["candidates"]] == [0.25, 0.5, 1.0, 2.0]
          and doc["amendments"] == ["AM-2", "AM-7a"] and doc["boundary_kind"] is None
          and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")
    check("am7a_lambda4_result_line",
          result_line(log).startswith("RESULT: SELECTED lambda_logit = 0.5 (run e2_s42_lambda0.5, tie=True, "
                                      "boundary=False/None, diverged=[4])"), result_line(log))
    check("am7a_lambda4_trace_line",
          "lambda=4 run=e2_s42_lambda4: diverged (AM-7a) at iter 12000 under AM-7(b); excluded from the "
          "selection" in (doc or {}).get("rule_trace", []))
    rc, log, doc, out, _ = lambda_case(tmp, "l025_diverged", {0.25: DIVERGED},
                                       vals=(0.45, 0.44, 0.42, 0.41, 0.40))
    check("am7a_lambda025_diverged_edge_of_finished_set",
          rc == 0 and doc is not None and doc["winner"]["lambda"] == 0.5 and doc["boundary"] is True
          and doc["boundary_kind"] == "edge_of_finished_set"
          and "boundary=True/edge_of_finished_set, diverged=[0.25]" in result_line(log), result_line(log))
    rc, log, doc, out, _ = lambda_case(tmp, "l_all_nondefault", {v: DIVERGED for v in (0.25, 0.5, 2, 4)})
    check("am7a_all_nondefault_diverged_selects_lambda1",
          rc == 0 and doc is not None and doc["winner"]["lambda"] == 1.0 and doc["n_finished"] == 1
          and doc["n_diverged"] == 4 and doc["tie"] is False
          and "diverged=[0.25, 0.5, 2, 4]" in result_line(log), result_line(log))
    rc, log, doc, out, _ = lambda_case(tmp, "l1_diverged", {1: DIVERGED})
    check("am7a_default_lambda1_diverged_refused", rc == 2 and doc is None and not out.exists()
          and "default_candidate_diverged" in log and "AM-7 applies in full" in log
          and "Traceback" not in log, log.strip()[-200:])
    other = {"input_nonfinite": dict(abort=dict(rule="AM-7(a)", cause="input_nonfinite", input_finite=False),
                                     nonfinite=True),
             "teacher_nonfinite": dict(abort=dict(rule="AM-7(a)", cause="teacher_nonfinite",
                                                  teacher_finite=False), nonfinite=True),
             "val_nonfinite": dict(abort=dict(rule="val_nonfinite", cause="val_nonfinite")),
             "step1_checks": dict(abort=dict(rule="step1_checks", cause="checks_failed")),
             "nonfinite_row_without_abort": dict(finished=False, run_end=False, nonfinite=True),
             "nonfinite_row_torn_abort": dict(finished=False, run_end=False, nonfinite=True, torn="last"),
             "val_nonfinite_row_without_abort": dict(finished=False, run_end=False,
                                                     val_nonfinite="all_class_miou"),
             "val_nonfinite_row_torn_abort": dict(finished=False, run_end=False,
                                                  val_nonfinite="all_class_miou", torn="last")}
    for label, kw in other.items():
        rc, log, doc, out, _ = lambda_case(tmp, f"l2_{label}", {2: kw})
        check(f"am7a_{label}_is_run_aborted_other", rc == 2 and doc is None and "run_aborted_other" in log
              and "STOP: investigate; AM-8a governs a repeat; never excluded, never a shortfall" in log
              and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")
    # only the all-class mIoU aborts a run: a finished run whose provisional disease-only mIoU is
    # non-finite in a val row is still a finished candidate
    rc, log, doc, out, _ = lambda_case(tmp, "l2_val_disease_only_nonfinite",
                                       {2: dict(val_nonfinite="disease_only_miou_PROVISIONAL")})
    check("am7a_val_disease_only_nonfinite_is_still_finished", rc == 0 and doc is not None
          and doc.get("n_finished") == 5, f"rc={rc} {log.strip()[-200:]}")
    invalid = {"am7b_at_iter_300": dict(abort=dict(it=300), with_best=False),
               "am7b_ratio_not_the_quotient": dict(abort=dict(detail={"ratio": 5.0}), with_best=False),
               "abort_after_run_end": dict(abort={}, abort_after_end=True),
               "abort_iter_not_last_train_row": dict(abort={}, record_iter=ABORT_ITER - 1, with_best=False)}
    for label, kw in invalid.items():
        rc, log, doc, out, _ = lambda_case(tmp, f"l2_{label}", {2: kw})
        check(f"am7a_{label}_is_abort_record_invalid", rc == 2 and doc is None
              and "abort_record_invalid" in log and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")
    root = tmp / "am7a_omitted"
    runs = [make_run(root, "E2", v, b, **({4: DIVERGED}).get(v, {}))
            for v, b in zip(LAMBDA_GRID, (0.40, 0.428, 0.43, 0.427, 0.41))]
    out = tmp / "l_omitted.json"
    rc, log = run_cli(sl, ["--runs", *map(str, runs[:4]), "--out", str(out)])
    check("am7a_diverged_directory_omitted_is_shortfall", rc == 3 and not out.exists()
          and "AM-19 item 2" in log                     # K2 (E-44): was "AM-17 item 9"
          and "diverged candidates are not shortfalls: pass their directories" in log, log.strip()[-200:])
    rc, log = run_cli(sl, ["--runs", *map(str, runs), str(runs[4]), "--out", str(out)])
    check("am7a_same_diverged_directory_twice_is_duplicate", rc == 2 and not out.exists()
          and "duplicate_candidate" in log, log.strip()[-160:])
    rc, log, doc, out, _ = lambda_case(tmp, "l4_twice", {4: DIVERGED}, extra=[(4, DIVERGED)])
    check("am7a_second_diverged_lambda4_run_is_duplicate", rc == 2 and doc is None
          and "duplicate_candidate" in log, log.strip()[-160:])
    recipe = {"grad_clip_norm_1": dict(recipe={"grad_clip_norm": 1.0}),
              "batch_size_8": dict(recipe={"batch_size": 8}),
              "val_interval_2000": dict(recipe={"val_interval": 2000}),
              "max_val_batches_10": dict(recipe={"max_val_batches": 10}),
              "not_pretrained": dict(recipe={"used_pretrained": False}),
              "poly_horizon_float": dict(recipe={"poly_horizon": 80000.0}),
              "num_workers_differ": dict(recipe={"num_workers": 8}),
              "teacher_ckpt_differs": dict(recipe={"teacher_provenance": {**TEACHER_PROV,
                                                                          "ckpt_sha256": "cd" * 32}}),
              "teacher_config_differs": dict(recipe={"teacher_provenance": {**TEACHER_PROV,
                                                                            "config_sha256": "c1" * 32}}),
              "teacher_model_cfg_differs": dict(recipe={"teacher_provenance": {
                  **TEACHER_PROV, "model_cfg_sha256": "d1" * 32}}),
              "teacher_provenance_absent": dict(recipe={"teacher_provenance": None}),
              "diverged_run_num_workers_differ": dict(abort={}, with_best=False, recipe={"num_workers": 8})}
    for label, kw in recipe.items():
        rc, log, doc, out, _ = lambda_case(tmp, f"l2_{label}", {2: kw})
        named = {"num_workers_differ": "differ in num_workers",
                 "diverged_run_num_workers_differ": "differ in num_workers",
                 "teacher_ckpt_differs": "differ in teacher_ckpt_sha256",
                 "teacher_config_differs": "differ in teacher_config_sha256",
                 "teacher_model_cfg_differs": "differ in teacher_model_cfg_sha256"}.get(label, "")
        # K2 (PL-11, R5): a difference across the candidates refuses with its own sweep-level code
        want = "recipe_mismatch_across_candidates" if named else "recipe_mismatch"
        check(f"item10b_recipe_{label}_refused", rc == 2 and doc is None and code_of(log) == want
              and named in log and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")
    rc, log, doc, out, _ = lambda_case(tmp, "l2_am7a_with_nonfinite_row",
                                       {2: dict(abort=dict(rule="AM-7(a)"), nonfinite=True)})
    check("am7a_am7a_divergence_with_nonfinite_row_is_excluded",
          rc == 0 and doc is not None and [e["value"] for e in doc["excluded_am7a"]] == [2.0]
          and doc["excluded_am7a"][0]["abort"]["detail"]["loss"] == "nan", f"rc={rc} {log.strip()[-200:]}")
    rc, log, doc, out, _ = lambda_case(tmp, "l4_with_best", {4: dict(abort={})})
    check("am7a_diverged_run_best_json_is_partial_value",
          rc == 0 and doc is not None and doc["excluded_am7a"][0]["best_val_partial"] == 0.41,
          f"rc={rc} {log.strip()[-200:]}")


def test_am7a_alpha_cli(tmp: Path) -> None:
    def alpha_case(label: str, over: dict, vals=(0.430, 0.428, 0.431)):
        root = tmp / f"am7a_alpha_{label}"
        runs = [make_run(root, "E3", a, b, **over.get(a, {})) for a, b in zip((25, 50, 100), vals)]
        band, lam, out = band_file(root), lambda_selection_file(root), tmp / f"a_{label}.json"
        rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(band), "--lambda-selection", str(lam),
                               "--out", str(out)])
        return rc, log, (json.loads(out.read_text()) if out.exists() else None), out

    rc, log, doc, out = alpha_case("25_100_diverged", {25: DIVERGED, 100: DIVERGED})
    check("am7a_alpha_25_and_100_diverged_selects_50",
          rc == 0 and doc is not None and doc["winner"]["alpha"] == 50.0 and doc["n_finished"] == 1
          and doc["n_diverged"] == 2 and doc["amendments"] == ["AM-16 item 2", "AM-7a"]
          and "diverged=[25, 100]" in result_line(log), f"rc={rc} {result_line(log)} {log.strip()[-160:]}")
    rc, log, doc, out = alpha_case("50_diverged", {50: DIVERGED})
    check("am7a_alpha_default_50_diverged_refused", rc == 2 and doc is None and not out.exists()
          and "default_candidate_diverged" in log and "Traceback" not in log, log.strip()[-200:])
    rc, log, doc, out = alpha_case("100_diverged_25_tied", {100: DIVERGED}, vals=(0.430, 0.428, 0.40))
    check("am7a_alpha_100_diverged_25_tied_with_50_selects_50",
          rc == 0 and doc is not None and doc["winner"]["alpha"] == 50.0 and doc["tie"] is True
          and doc["tied"] == [25.0, 50.0] and doc["n_finished"] == 2 and doc["n_diverged"] == 1
          and doc["boundary_kind"] == "edge_of_finished_set", f"rc={rc} {result_line(log)}")
    rc, log, doc, out = alpha_case("100_diverged_other_lambda", {100: dict(DIVERGED, lam=0.5)})
    check("am7a_alpha_lambda_consistency_includes_diverged_runs", rc == 2 and doc is None
          and "lambda_mismatch" in log, log.strip()[-200:])


def outcome(fn, *args):
    """fn's result, the code of a SelectionRefused, or the name and text of any other exception: a unit
    whose check raises unexpectedly FAILs by name instead of stopping the smoke."""
    try:
        return fn(*args)
    except SelectionRefused as e:
        return e.code
    except Exception as e:                        # noqa: BLE001 - reported as the check's detail
        return f"{type(e).__name__}: {e}"


def ratio_exactly_factor_pair() -> tuple[float, float]:
    """(window_mean, running_min) for which the trainer's test window_mean > 5.0 x running_min holds in
    floats while the float quotient window_mean / running_min is exactly 5.0. Found by search: running_min
    steps up one float at a time from 0.75, window_mean is the float just above 5.0 x running_min."""
    rm = 0.75
    for _ in range(100000):
        rm = math.nextafter(rm, 1.0)
        wm = math.nextafter(5.0 * rm, math.inf)
        if wm > 5.0 * rm and wm / rm == 5.0:
            return wm, rm
    raise RuntimeError("no (window_mean, running_min) pair found")


def test_am7_records(tmp: Path) -> None:
    """Q2, Q4 (commit 8): an AM-7 run_abort record train_distill cannot have written is
    abort_record_invalid; Q3: the recipe is checked before a divergence is read."""
    wm, rm = ratio_exactly_factor_pair()
    check("q2_search_pair_window_mean_above_5x_and_ratio_exactly_5", wm > 5.0 * rm and wm / rm == 5.0,
          f"window_mean={wm!r} running_min={rm!r}")
    exactly_5 = {"window_mean": wm, "running_min": rm, "ratio": wm / rm, "threshold": 5.0 * rm}
    min_0 = {"window_mean": 0.5, "running_min": 0.0, "ratio": "inf", "threshold": 0.0}
    norm_case = {"loss": 3.1, "grad_norm": "nan"}
    accepted = {"am7b_ratio_rounds_to_exactly_5": dict(abort=dict(detail=exactly_5)),
                "am7b_running_min_0_ratio_inf": dict(abort=dict(detail=min_0)),
                "am7b_first_firing_iter_ramp_plus_101": dict(abort=dict(it=RAMP + 101)),
                "am7a_finite_loss_nonfinite_grad_norm": dict(abort=dict(rule="AM-7(a)", detail=norm_case),
                                                             nonfinite={"grad_norm": "nan",
                                                                        "grad_norm_student": "nan"})}
    for label, kw in accepted.items():
        rc, log, doc, out, _ = lambda_case(tmp, f"q_{label}", {2: dict(kw, with_best=False)})
        check(f"q_{label}_accepted_as_diverged", rc == 0 and doc is not None
              and [e["value"] for e in doc["excluded_am7a"]] == [2.0], f"rc={rc} {log.strip()[-200:]}")
    invalid = {"am7b_at_ramp_plus_100": dict(abort=dict(it=RAMP + 100)),
               "am7b_ratio_nan": dict(abort=dict(detail={"ratio": "nan"})),
               "am7b_threshold_not_5x_running_min": dict(abort=dict(detail={"threshold": 3.1})),
               "am7b_window_mean_not_above_5x": dict(abort=dict(detail={"window_mean": 3.0,
                                                                        "ratio": 3.0 / 0.6})),
               "am7a_finite_loss_and_finite_grad_norm": dict(abort=dict(rule="AM-7(a)", detail={
                   "loss": 3.1, "grad_norm": 1.7}), nonfinite={"loss": "nan", "grad_norm": "nan"}),
               "am7a_at_iter_1": dict(abort=dict(it=1, rule="AM-7(a)"), nonfinite=True),
               "am7a_loss_case_row_names_only_grad_norm": dict(abort=dict(rule="AM-7(a)"),
                                                               nonfinite={"grad_norm": "nan"}),
               "am7a_grad_norm_case_row_without_nonfinite_map": dict(abort=dict(rule="AM-7(a)", detail={
                   "loss": 3.1, "grad_norm": "inf"}))}
    for label, kw in invalid.items():
        rc, log, doc, out, _ = lambda_case(tmp, f"q_{label}", {2: dict(kw, with_best=False)})
        check(f"q_{label}_is_abort_record_invalid", rc == 2 and doc is None and "abort_record_invalid" in log
              and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")
    # the other values a record can carry, one field off a record that holds otherwise (load_candidate)
    units = {"am7b_running_min_nan": ("AM-7(b)", {"running_min": "nan"}, False),
             "am7b_window_mean_inf_string": ("AM-7(b)", {"window_mean": "inf", "ratio": "inf"}, False),
             "am7b_window_mean_numeric_string": ("AM-7(b)", {"window_mean": "3.12"}, False),
             "am7b_window_mean_missing": ("AM-7(b)", {"window_mean": None}, False),
             "am7b_running_min_bool": ("AM-7(b)", {"window_mean": 6.0, "running_min": True, "ratio": 6.0,
                                                   "threshold": 5.0}, False),
             "am7b_running_min_negative": ("AM-7(b)", {"running_min": -0.1, "ratio": 3.12 / -0.1,
                                                       "threshold": 5.0 * -0.1}, False),
             "am7b_ratio_numeric_string": ("AM-7(b)", {"ratio": "5.2"}, False),
             "am7b_ratio_inf_with_positive_running_min": ("AM-7(b)", {"ratio": "inf"}, False),
             "am7b_threshold_numeric_string": ("AM-7(b)", {"threshold": "3.0"}, False),
             "am7b_running_min_0_finite_ratio": ("AM-7(b)", {"window_mean": 0.5, "running_min": 0.0,
                                                             "ratio": 1e308, "threshold": 0.0}, False),
             "am7b_running_min_integer_beyond_float": ("AM-7(b)", {"running_min": 10 ** 400}, False),
             "am7b_overflowing_quotient_written_finite": ("AM-7(b)", {"window_mean": 1.0,
                                                                      "running_min": 1e-310, "ratio": 1e308,
                                                                      "threshold": 5.0 * 1e-310}, False),
             "am7a_loss_tag_other_spelling": ("AM-7(a)", {"loss": "NaN"}, {"loss": "nan"}),
             "am7a_loss_bool_grad_norm_nan": ("AM-7(a)", {"loss": True, "grad_norm": "nan"},
                                              {"grad_norm": "nan"}),
             "am7a_grad_norm_bool": ("AM-7(a)", {"loss": 3.1, "grad_norm": True}, {"grad_norm": "nan"}),
             "am7a_loss_integer_beyond_float": ("AM-7(a)", {"loss": 10 ** 400, "grad_norm": "nan"},
                                                {"grad_norm": "nan"})}
    for label, (rule, detail, named) in units.items():
        d = make_run(tmp / f"q_unit_{label}", "E2", 2, 0.4, abort=dict(rule=rule, detail=detail),
                     nonfinite=named, with_best=False)
        got = outcome(load_candidate, d, LAMBDA, "lambda_logit")
        check(f"q_unit_{label}_is_abort_record_invalid", got == "abort_record_invalid", str(got)[:160])
    control = make_run(tmp / "q_unit_control", "E2", 2, 0.4, abort={}, with_best=False)
    got = outcome(load_candidate, control, LAMBDA, "lambda_logit")
    check("q_unit_control_record_is_diverged", isinstance(got, dict) and got.get("status") == "diverged",
          str(got)[:160])
    # the trainer writes a quotient that overflows (running_min > 0) as "inf", as for running_min 0
    over = make_run(tmp / "q_unit_overflow", "E2", 2, 0.4, with_best=False, abort=dict(detail={
        "window_mean": 1.0, "running_min": 1e-310, "ratio": "inf", "threshold": 5.0 * 1e-310}))
    got = outcome(load_candidate, over, LAMBDA, "lambda_logit")
    check("q_unit_am7b_overflowing_quotient_inf_is_diverged", 1.0 / 1e-310 == math.inf
          and isinstance(got, dict) and got.get("status") == "diverged", str(got)[:160])
    for label, recipe, field in (
            ("num_workers", {"num_workers": 8}, "num_workers"),
            ("teacher_ckpt", {"teacher_provenance": {**TEACHER_PROV, "ckpt_sha256": "cd" * 32}},
             "teacher_ckpt_sha256"),
            ("teacher_config", {"teacher_provenance": {**TEACHER_PROV, "config_sha256": "c1" * 32}},
             "teacher_config_sha256"),
            ("teacher_model_cfg", {"teacher_provenance": {**TEACHER_PROV, "model_cfg_sha256": "d1" * 32}},
             "teacher_model_cfg_sha256")):
        rc, log, doc, out, _ = lambda_case(tmp, f"q3_default_diverged_{label}",
                                           {1: DIVERGED, 2: dict(recipe=recipe)})
        # K2 (PL-11, R5): the across-candidates code, recipe_mismatch_across_candidates
        check(f"q3_diverged_default_with_{label}_mismatch_is_recipe_mismatch", rc == 2 and doc is None
              and code_of(log) == "recipe_mismatch_across_candidates" and f"differ in {field}" in log
              and "default_candidate_diverged" not in log
              and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")


def test_ckpt_guard(tmp: Path) -> None:
    """L-CKPT-GUARD: the teacher's config and resolved model-config hashes join RECIPE_IDENTICAL; K8-2(a)
    the shared-lambda refusal before default_candidate_diverged in select_alpha; K8-2(b) the AM-7 comment
    names R8-1; K8-2(c) an AM-7 (a) loss record needs grad_norm null in its detail and its train row."""
    # each new teacher key absent, None, empty or not a string: the run lacks a comparable value
    for key in ("config_sha256", "model_cfg_sha256"):
        for label, value in (("absent", KeyError), ("none", None), ("empty", ""), ("int", 123)):
            prov = {k: v for k, v in TEACHER_PROV.items() if not (k == key and value is KeyError)}
            if value is not KeyError:
                prov[key] = value
            rc, log, doc, out, _ = lambda_case(tmp, f"cg_{key}_{label}",
                                               {2: dict(recipe={"teacher_provenance": prov})})
            check(f"cg_teacher_{key}_{label}_is_recipe_mismatch", rc == 2 and doc is None
                  and code_of(log) == "recipe_mismatch" and "lacks a comparable" in log and "Traceback" not in log,
                  f"rc={rc} {log.strip()[-200:]}")

    # K8-2(a): the five lambda cases of select_alpha (selection lambda 1)
    def alpha_runs(label: str, spec: dict, vals=(0.430, 0.428, 0.431)):
        """spec maps alpha -> make_run kwargs, or None for a run directory that does not exist."""
        root = tmp / f"cg_alpha_{label}"
        root.mkdir(parents=True)
        runs = []
        for a, b in zip((25, 50, 100), vals):
            if spec.get(a, {}) is None:
                runs.append(root / f"e3_s42_alpha{a}_absent")
            else:
                runs.append(make_run(root, "E3", a, b, **spec.get(a, {})))
        band, lam, out = band_file(root), lambda_selection_file(root), tmp / f"cg_a_{label}.json"
        rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(band), "--lambda-selection",
                               str(lam), "--out", str(out)])
        return rc, log, out

    # K2 (E-44): the shortfall code is shortfall_am19_item2 (was shortfall_lane2_stop); exit 3 unchanged
    rc, log, out = alpha_runs("no_run_loaded", {25: None, 50: None, 100: None})
    check("k82a_no_run_loaded_is_shortfall_exit3", rc == 3 and "shortfall_am19_item2" in log
          and "lambda_mismatch" not in log and not out.exists(), f"rc={rc} {log.strip()[-160:]}")
    unfinished = dict(finished=False)
    rc, log, out = alpha_runs("three_unfinished", {25: unfinished, 50: unfinished, 100: unfinished})
    check("k82a_three_unfinished_is_shortfall_exit3", rc == 3 and "shortfall_am19_item2" in log
          and "lambda_mismatch" not in log, f"rc={rc} {log.strip()[-160:]}")
    rc, log, out = alpha_runs("wrong_lambda_two_missing", {25: dict(lam=0.5), 50: None, 100: None})
    check("k82a_one_wrong_lambda_run_two_missing_is_lambda_mismatch", rc == 2 and "lambda_mismatch" in log
          and "shortfall" not in log, f"rc={rc} {log.strip()[-160:]}")
    rc, log, out = alpha_runs("default_diverged_right_lambda_one_missing", {50: DIVERGED, 100: None})
    check("k82a_default_diverged_right_lambda_one_missing_is_default_candidate_diverged", rc == 2
          and "default_candidate_diverged" in log and "shortfall" not in log, f"rc={rc} {log.strip()[-160:]}")
    rc, log, out = alpha_runs("default_diverged_wrong_lambda", {50: dict(DIVERGED, lam=0.5)})
    check("k82a_lambda_mismatch_fires_before_default_candidate_diverged", rc == 2 and "lambda_mismatch" in log
          and "default_candidate_diverged" not in log and not out.exists(), f"rc={rc} {log.strip()[-160:]}")

    # K8-2(b): the AM-7 comment block of configs/distill.py names the iteration-1 exception (R8-1)
    text = (REPO / "configs" / "distill.py").read_text(encoding="utf-8")
    block = text[text.index("AM-7 divergence rule (b)"):text.index("AM7_DIVERGENCE = {")]
    check("k82b_am7_comment_names_r8_1_step1_checks", "R8-1" in block and "step1_checks" in block
          and "iteration 1" in block, block[:200])

    # K8-2(c): the loss case of AM-7 (a) with grad_norm anything but null, in the detail or the train row
    loss_case = dict(abort=dict(rule="AM-7(a)"), nonfinite=True, with_best=False)
    for label, kw in (("detail_grad_norm_1_7", dict(abort=dict(rule="AM-7(a)", detail={"grad_norm": 1.7}))),
                      ("detail_grad_norm_nan", dict(abort=dict(rule="AM-7(a)", detail={"grad_norm": "nan"}))),
                      ("detail_grad_norm_0", dict(abort=dict(rule="AM-7(a)", detail={"grad_norm": 0}))),
                      ("detail_grad_norm_false", dict(abort=dict(rule="AM-7(a)", detail={"grad_norm": False}))),
                      ("row_grad_norm_1_7", dict(row_grad_norm=1.7)),
                      ("row_grad_norm_0", dict(row_grad_norm=0)),
                      ("row_grad_norm_false", dict(row_grad_norm=False)),
                      ("row_without_grad_norm", dict(row_grad_norm=None))):
        rc, log, doc, out, _ = lambda_case(tmp, f"cg_{label}", {2: {**loss_case, **kw}})
        check(f"k82c_am7a_loss_record_{label}_is_abort_record_invalid", rc == 2 and doc is None
              and "abort_record_invalid" in log and "Traceback" not in log, f"rc={rc} {log.strip()[-200:]}")
        d = make_run(tmp / f"cg_unit_{label}", "E2", 2, 0.4, **{**loss_case, **kw})
        got = outcome(load_candidate, d, LAMBDA, "lambda_logit")
        check(f"k82c_unit_{label}_is_abort_record_invalid", got == "abort_record_invalid", str(got)[:160])
    control = make_run(tmp / "cg_unit_loss_case_control", "E2", 2, 0.4, **loss_case)
    got = outcome(load_candidate, control, LAMBDA, "lambda_logit")
    check("k82c_unit_loss_case_with_null_grad_norms_is_diverged",
          isinstance(got, dict) and got.get("status") == "diverged", str(got)[:160])


# ------------------------------------------------------------------ AM-19 scenarios (lane K2)
NOW_L = LAMBDA_C + DAY                   # after the lambda decision date
NOW_MID_L = LAMBDA_C - 12 * 3600.0       # before it, after every run of BASE_L has ended
NOW_A = ALPHA_C + DAY                    # after the alpha decision date (basis (a))
NOW_MID_A = ALPHA_C - 12 * 3600.0
BASE_L = LAMBDA_C - 4 * DAY              # the first lambda launch of the AM-19 scenarios
T_IN = 1791763200.0                      # 2026-10-12T00:00:00Z: the lambda selection's last input timestamp
BASE_A = 1791849600.0                    # 2026-10-13T00:00:00Z: the alpha default's launch
ITER_S, VAL_S = 1.0, 100.0
CKN = {"E2": "e2_student_best_iter80000.pt", "E3": "e3_student_best_iter80000.pt"}


def utc(s: str) -> float:
    from datetime import datetime, timezone
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def T(it, ts, iter_s=ITER_S) -> dict:
    """A train row with the trainer's wall_clock and iter_seconds."""
    return {"event": "train", "iter": it, "loss": 1.0, "wall_clock": float(ts), "iter_seconds": iter_s}


def V(it, ts, miou=0.40, val_s=VAL_S) -> dict:
    return {"event": "val", "iter": it, "all_class_miou": miou, "disease_only_miou_PROVISIONAL": miou,
            "val_seconds": val_s, "wall_clock": float(ts)}


def END(ts, best=0.40, checks=True, stage="E2", start=None) -> dict:
    """A run_end as the trainer writes it: wall_clock_start (the run's start; 80000 s before the end unless
    given) and wall_clock_end, no wall_clock."""
    return {"event": "run_end", "iter": 80000, "best_val_miou_all_class": best, "best_ckpt": CKN[stage],
            "checks_passed": checks, "wall_clock_start": float(ts) - 80000.0 if start is None else float(start),
            "wall_clock_end": float(ts)}


def course(L, upto, *, val_at=(40000, 80000), miou=0.40, best=None, end=None, checks=True, stage="E2",
           iter_s=ITER_S, val_s=VAL_S) -> list:
    """The rows of a run launched at L that trained to iteration `upto`: train rows at 1, at each VAL
    iteration up to `upto` and at `upto` (iter_s per iteration), a VAL row val_s after each VAL iteration
    (`best` at 80000), and a run_end 1 s after the last row when `end` is True, or at `end`."""
    rows, n_val = [], 0
    for it in sorted({1, upto} | {i for i in val_at if i <= upto}):
        t = L + it * iter_s + n_val * val_s
        rows.append(T(it, t, iter_s))
        if it in val_at:
            n_val += 1
            rows.append(V(it, t + val_s, best if (best is not None and it == 80000) else miou, val_s))
    if end is not None and end is not False:
        rows.append(END(rows[-1]["wall_clock"] + 1.0 if end is True else end, miou if best is None else best,
                        checks, stage, start=L))
    return rows


def fin(L, best=0.40, **kw) -> list:
    """A run launched at L that finished, its best VAL `best` at 80000."""
    return course(L, 80000, best=best, miou=kw.pop("miou", best), end=kw.pop("end", True), **kw)


def oc_stop_rows() -> list:
    """An on-course stop s of a lambda candidate: 0.25 s per iteration, VAL at 4000 and 8000 (50 s), stopped
    at 10000 at C - 27400; its projection C - 9000 meets the decision date."""
    return course(LAMBDA_C - 30000.0, 10000, val_at=(4000, 8000), iter_s=0.25, val_s=50.0)


class Sweep:
    """One AM-19 scenario: a scratch repository copied from the template (its commit is the code pin),
    attempts added in launch-log order, run directories outside the repository. build() commits the launch
    lines (with the AM-8a reports) as the records commit, writes the run directories (run_meta with git_head
    and records_commit), then commits the launched, not_launched and stopped lines (with their evidence and
    stop reports) as HEAD."""

    def __init__(self, label: str, key: str = "lambda_logit", *, lam: float = 1.0, files: dict | None = None):
        self.key, self.label, self.lam = key, label, lam
        self.stage = "E2" if key == "lambda_logit" else "E3"
        self.module = sl if key == "lambda_logit" else sa
        self.repo, self.pin = new_repo(), template()[1]
        self.root = scratch_dir("k2_runs")
        self.atts: list = []
        self.files = dict(files or {})
        self.lines: list = []
        self.events: list = []
        self.records = self.head = None
        self.schedule = "T"                  # the schedule the launch lines name (the one in force at launch)
        self._n = itertools.count()

    def add(self, value, rows=None, *, launch=None, state="launched", stopped=False, report="auto", meta=None,
            best_json=True, ckpt=True, lam=None, code_pin=None, launched_over=None) -> str:
        value, lam = float(value), (self.lam if lam is None else lam)
        n = 1 + sum(1 for a in self.atts if a["value"] == value and a["lam"] == lam)
        rid = f"{self.stage.lower()}_s42_{'lambda' if self.key == 'lambda_logit' else 'alpha'}{value:g}_a{n}"
        if launch is None and rows:
            launch = rows[0]["wall_clock"] - rows[0].get("iter_seconds", ITER_S)
        self.atts.append({"value": value, "lam": lam, "attempt": n, "run_id": rid, "rows": rows or [],
                          "launch": launch, "state": state, "stopped": stopped, "report": report,
                          "meta": meta or {}, "best_json": best_json, "ckpt": ckpt, "code_pin": code_pin,
                          "launched_over": launched_over or {}})
        return rid

    def pin_files(self, files: dict, msg: str = "pin") -> str:
        """Commit `files` before any launch and make that commit the code pin."""
        self.pin = commit_files(self.repo, files, msg)
        return self.pin

    def dir(self, rid: str) -> Path:
        return self.root / rid

    def report(self, rel: str, text: str) -> dict:
        data = text.encode("utf-8")
        self.files[rel] = data
        return {"path": rel, "sha256": hashlib.sha256(data).hexdigest()}

    def build(self) -> "Sweep":
        launched = set()
        for a in self.atts:
            a["code_pin"] = a["code_pin"] or self.pin
            rep = a["value"] in launched if a["report"] == "auto" else a["report"]
            ref = self.report(f"reports/am8a/{a['run_id']}.md", f"AM-8a report before {a['run_id']} (synthetic)\n") \
                if rep is True else (rep or None)
            self.lines.append(launch_line(a["run_id"], self.stage, a["value"],
                                          a["value"] if self.key == "lambda_logit" else a["lam"],
                                          None if self.key == "lambda_logit" else a["value"], a["attempt"],
                                          a["code_pin"], report=ref, schedule=self.schedule))
            if a["state"] == "launched":
                launched.add(a["value"])
        self.records = commit_files(self.repo, {**self.files, ss.LAUNCH_LOG_REL: launch_log_text(self.lines)},
                                    "launch lines")
        self.files = {}
        for a in self.atts:
            if a["state"] == "launched":
                self._write_run(a)
                ev = launched_line(self.dir(a["run_id"]), self.stage, self.key, a["value"], a["code_pin"],
                                   self.records)
                ev.update(a["launched_over"])
                self.events.append(ev)
            elif a["state"] == "not_launched":
                ref = self.report(f"reports/not_launched/{a['run_id']}.txt", f"{a['run_id']}: start refused\n")
                self.events.append({"event": "not_launched", "run_id": a["run_id"],
                                    "reason": "the start was refused (synthetic)", "evidence": ref})
        for a in self.atts:
            if a["stopped"]:
                ref = self.report(f"reports/stops/{a['run_id']}.md", f"{a['run_id']} was stopped (synthetic)\n")
                self.events.append({"event": "stopped", "run_id": a["run_id"], "report": ref})
        self.head = self.commit("launched lines")
        return self

    def commit(self, msg: str = "update", files: dict | None = None, log: bytes | None = None) -> str:
        self.files.update(files or {})
        out = {**self.files, ss.LAUNCH_LOG_REL: launch_log_text(self.lines + self.events) if log is None else log}
        self.files = {}
        return commit_files(self.repo, out, msg)

    def _write_run(self, a: dict) -> None:
        d, v, sk = self.dir(a["run_id"]), a["value"], self.stage.lower()
        d.mkdir(parents=True)
        meta = {"event": "run_meta", "stage": self.stage, "mode": "real", "seed": 42,
                "terms": {"logit_kd": True, "cwd_feat": self.stage == "E3", "cwd_logit": self.stage == "E3"},
                "lambda_logit": v if self.key == "lambda_logit" else a["lam"],
                "logit_kd_semantics": LOGIT_KD_SEMANTICS, "logit_kd_semantics_declared": None,
                "logit_kd_semantics_override_used": False, "max_iters": 80000, "num_workers": 12,
                "batch_size": 16, "val_interval": 4000, "max_val_batches": None, "poly_horizon": 80000,
                "grad_clip_norm": None, "used_pretrained": True, "ramp_iters": RAMP,
                "teacher_provenance": dict(TEACHER_PROV), "wall_clock": a["launch"], "git_head": a["code_pin"],
                "records_commit": self.records}
        if self.stage == "E3":
            meta.update({"alpha_cwd": v, "alpha_offgrid": False, "beta_cwd": 3})
        meta.update(a["meta"])
        (d / f"{sk}_run_meta.jsonl").write_text(json.dumps(meta) + "\n", encoding="utf-8")
        (d / f"{sk}_telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a["rows"]), encoding="utf-8")
        end = a["rows"][-1] if a["rows"] and a["rows"][-1].get("event") == "run_end" else None
        if end is not None and end.get("checks_passed") is True:
            best, ck = end["best_val_miou_all_class"], d / CKN[self.stage]
            if a["ckpt"] == "corrupt":
                ck.write_bytes(b"\x00not a checkpoint")
            elif a["ckpt"]:
                torch.save({"stage": self.stage, "iter": 80000, "best_val_miou_all_class": best, self.key: v}, ck)
            if a["best_json"]:
                (d / "best.json").write_text(json.dumps({"best_ckpt": f"/workspace/kd_ckpts/{a['run_id']}/"
                                                                      f"{CKN[self.stage]}",
                                                         "best_val_miou_all_class": best}), encoding="utf-8")

    def runs(self) -> list:
        return [self.dir(a["run_id"]) for a in self.atts if a["state"] == "launched"]

    def select(self, now, *, runs=None, extra=()) -> tuple:
        out = self.root / f"selection_{next(self._n)}.json"
        argv = ["--runs", *map(str, self.runs() if runs is None else runs), *extra, "--out", str(out)]
        rc, log = run_cli(self.module, argv, now=now, repo=self.repo)
        return rc, log, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)

    def record(self, now, *, edit=None, dl=True, commit=True, commit_record=True) -> tuple:
        """--write-decision-record; then the record and a decision-log row naming its sha256 are committed
        (commit_record=False commits the row only: the record file is left uncommitted)."""
        path = self.repo / ss.DECISION_RECORD_REL[self.key]
        rc, log = run_cli(self.module, ["--runs", *map(str, self.runs()), "--write-decision-record", str(path)],
                          now=now, repo=self.repo)
        if rc != 0:
            return rc, log, None
        if edit is not None:
            doc = json.loads(path.read_text(encoding="utf-8"))
            edit(doc)
            path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        sha = sha256_file(path)
        if commit:
            files = {ss.DECISION_RECORD_REL[self.key]: path.read_bytes()} if commit_record else {}
            if dl:
                files[ss.DECISION_LOG_REL] = (self.repo / ss.DECISION_LOG_REL).read_text(encoding="utf-8") \
                    + f"| DL-91 | RECORDED | {self.key} decision record, sha256 {sha} (synthetic) | - | no |\n"
            self.commit("decision record", files)
        return rc, log, sha


def lam_sweep(label: str, *, best=(0.42, 0.41, 0.40, 0.39), over: dict | None = None) -> Sweep:
    """lambda 1, 0.5, 2 and 0.25 launched in that order from BASE_L and finished well before the decision
    date, with the scores `best` (`over` maps a value to add() keyword arguments); the case adds lambda 4."""
    sw = Sweep(label)
    for i, (v, b) in enumerate(zip((1, 0.5, 2, 0.25), best)):
        kw = dict((over or {}).get(v, {}))
        sw.add(v, kw.pop("rows", None) or fin(BASE_L + 600 * i, b), **kw)
    return sw


def alpha_files(lam: float = 1.0, t_in: float = T_IN) -> dict:
    band = {"e1_best_val": E1_VALS, "s": __import__("statistics").stdev(E1_VALS.values())}
    lsel = {"format": "lambda_selection/1", "winner": {"lambda": lam, "run_id": "e2_s42_lambda1_a1"},
            "inputs_last_timestamp_utc": t_in}
    return {ss.BAND_REL: json.dumps(band), ss.LAMBDA_SELECTION_REL: json.dumps(lsel)}


def code_of(log: str) -> str:
    line = result_line(log)
    return line[line.find("(") + 1:line.find(")")] if "REFUSED" in line else line


def edited_rules(tmp: Path, label: str, fn) -> str | None:
    doc = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    fn(doc)
    p = tmp / f"rules_am19_{label}.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return refused(load_rules, p)


def test_am19_rules_time(tmp: Path) -> None:
    """R19 (P8) and PL-39 case 23: the AM-19 constants and the rule pins are kept in two places that change
    together (sweep_rules.json and sweep_select.py)."""
    check("r19_rules_file_carries_the_am19_keys", RULES["instant"] == ss.INSTANT
          and RULES["lambda_logit"]["decision_dates"] == {"T": "2026-10-19", "R": "2026-12-07"}
          and RULES["alpha_cwd"]["launch_order"] == [50, 25, 100] and RULES["stopped_early"] == ss.STOPPED_EARLY)
    check("r19_cutoff_instant_2026_10_19_is_1792425600", ss.cutoff_instant("2026-10-19") == 1792425600 == LAMBDA_C)
    check("r19_meets_is_strictly_earlier", ss.meets(LAMBDA_C - 0.001, LAMBDA_C) and not ss.meets(LAMBDA_C, LAMBDA_C))
    check("r19_manila_day_turns_at_16_00_utc", str(ss.manila_day(LAMBDA_C - 1)) == "2026-10-19"
          and str(ss.manila_day(LAMBDA_C)) == "2026-10-20")
    for label, fn, code in (
            ("lambda_T_date_10_20", lambda d: d["lambda_logit"]["decision_dates"].update(T="2026-10-20"),
             "rules_mismatch"),
            ("alpha_R_date_missing", lambda d: d["alpha_cwd"]["decision_dates"].pop("R"), "rules_format"),
            ("lambda_launch_order_1_2_05", lambda d: d["lambda_logit"].update(launch_order=[1, 2, 0.5, 0.25, 4]),
             "rules_mismatch"),
            ("alpha_launch_order_missing", lambda d: d["alpha_cwd"].pop("launch_order"), "rules_format"),
            ("end_of_date_15_59_59", lambda d: d["instant"].update(end_of_date_utc="15:59:59"), "rules_mismatch"),
            ("stopped_early_600", lambda d: d["stopped_early"].update(min_seconds=600), "rules_mismatch"),
            ("alpha_date_rule_2_days", lambda d: d["alpha_cwd"]["decision_date_rule"].update(lambda_selection_days=2),
             "rules_mismatch")):
        got = edited_rules(tmp, label, fn)
        check(f"r19_rules_{label}_is_{code}", got == code, str(got))
    for label, fn in (("tie", lambda d: d["lambda_logit"].update(tie="default_if_tied_else_smallest")),
                      ("floor", lambda d: d["alpha_cwd"]["band"].update(floor=0.006)),
                      ("boundary", lambda d: d["lambda_logit"].update(boundary=[0.5, 4])),
                      ("default", lambda d: d["alpha_cwd"].update(default=25))):
        got = edited_rules(tmp, f"pin_{label}", fn)
        check(f"pl39_23_rules_with_another_{label}_is_rules_mismatch", got == "rules_mismatch", str(got))
    # keys a sweep does not use are pinned as absent; only the pin refuses these (PL-16)
    for label, fn in (("lambda_default", lambda d: d["lambda_logit"].update(default=1)),
                      ("lambda_band_file", lambda d: d["lambda_logit"]["band"].update(file="x.json")),
                      ("lambda_band_floor", lambda d: d["lambda_logit"]["band"].update(floor=0.9)),
                      ("alpha_band_value", lambda d: d["alpha_cwd"]["band"].update(value=0.5))):
        got = edited_rules(tmp, f"pin_{label}", fn)
        check(f"pl16_rules_with_a_{label}_is_rules_mismatch", got == "rules_mismatch", str(got))
    lam_text, alpha_text = Path(sl.__file__).read_text(encoding="utf-8"), Path(sa.__file__).read_text(encoding="utf-8")
    check("r19_e44_texts_name_am19_item2", "is NOT implemented" not in lam_text and "Lane 2 STOP" not in alpha_text
          and "AM-19 item 2" in lam_text and "AM-19 item 2" in alpha_text and "AM-17 item 9 (at the" not in lam_text
          and all("AM-19 item 2" in RULES[k]["shortfall"] for k in ("lambda_logit", "alpha_cwd")))


def five_finished(label: str, over: dict | None = None) -> Sweep:
    """All five lambda candidates finished before the decision date (AM-19 item 2(f)); the winner is 0.5.
    `over` maps a value to add() keyword arguments."""
    sw = Sweep(label)
    for i, (v, b) in enumerate(zip(LAMBDA_GRID, (0.40, 0.428, 0.43, 0.427, 0.41))):
        sw.add(v, fin(BASE_L + 600 * i, b), **(over or {}).get(v, {}))
    return sw


def test_am19_schedule(tmp: Path) -> None:
    """SCH (P8) and PL-39 case 20: configs/kd_schedule.json (PL-12)."""
    sw = five_finished("sch_t").build()
    rc, log, doc = sw.select(NOW_MID_L)
    sch = (doc or {}).get("schedule", {})
    check("sch_T_uses_2026_10_19", rc == 0 and sch.get("in_force") == "T" and sch.get("decision_date") == "2026-10-19"
          and sch.get("cutoff_utc") == LAMBDA_C and sch.get("decision_log_entry") == "DL-70", f"rc={rc} {sch} {log[-200:]}")
    sw = five_finished("sch_r")
    sw.pin_files({ss.SCHEDULE_REL: json.dumps({"format": "kd_schedule/1", "in_force": "R",
                                               "decision_log_entry": "DL-90", "basis": "synthetic"})})
    sw.schedule = "R"
    rc, log, doc = sw.build().select(NOW_MID_L)
    sch = (doc or {}).get("schedule", {})
    check("sch_R_with_its_own_row_uses_2026_12_07", rc == 0 and sch.get("in_force") == "R"
          and sch.get("decision_date") == "2026-12-07" and sch.get("cutoff_utc") == ss.cutoff_instant("2026-12-07"),
          f"rc={rc} {sch} {log[-200:]}")
    r_doc = lambda entry, force="R": json.dumps({"format": "kd_schedule/1", "in_force": force,  # noqa: E731
                                                 "decision_log_entry": entry, "basis": "synthetic"})
    for label, files, code in (("value_S", {ss.SCHEDULE_REL: r_doc("DL-70", "S")}, "schedule_format"),
                               ("R_without_a_row", {ss.SCHEDULE_REL: r_doc("DL-99")}, "schedule_format"),
                               ("pl39_20d_R_naming_am19_row", {ss.SCHEDULE_REL: r_doc("DL-70")}, "schedule_format"),
                               ("pl39_20a_R_after_the_first_launch", {ss.SCHEDULE_REL: r_doc("DL-90")},
                                "schedule_changed_after_first_launch")):
        sw = five_finished(f"sch_{label}").build()
        sw.commit("schedule change", files)
        rc, log, doc = sw.select(NOW_MID_L)
        check(f"sch_{label}_is_{code}", rc == 2 and code_of(log) == code, f"rc={rc} {result_line(log)}")
    # PL-12 covers every launched line: G launched under the first schedule file, which then changed (still
    # T) before the sweep launched at a new pin; the schedule check is the only guard here
    sw = five_finished("sch_other_stage")
    g_launch, g_launched = g_lines(sw.pin, "0" * 40, BASE_L - DAY)
    g_launched["records_commit"] = commit_files(sw.repo, {ss.LAUNCH_LOG_REL: launch_log_text([g_launch])}, "G launch")
    sw.pin_files({ss.SCHEDULE_REL: (sw.repo / ss.SCHEDULE_REL).read_text(encoding="utf-8").replace(
        "AM-19 item 1(b):", "AM-19 item 1(b) (edited after G's launch):")}, "schedule edited")
    sw.lines.append(g_launch)
    sw.events.append(g_launched)
    rc, log, _ = sw.build().select(NOW_MID_L)
    check("sch_pl39_20a_changed_after_another_stages_launch_is_schedule_changed_after_first_launch",
          rc == 2 and code_of(log) == "schedule_changed_after_first_launch", result_line(log))
    sw = five_finished("sch_absent").build()
    git(sw.repo, "rm", "-q", "--", ss.SCHEDULE_REL)
    git(sw.repo, "commit", "-q", "-m", "no schedule")
    rc, log, doc = sw.select(NOW_MID_L)
    check("sch_absent_is_schedule_missing", rc == 2 and code_of(log) == "schedule_missing", result_line(log))
    sw = five_finished("sch_uncommitted").build()
    (sw.repo / ss.SCHEDULE_REL).write_text(r_doc("DL-70", "T") + "\n", encoding="utf-8")
    rc, log, doc = sw.select(NOW_MID_L)
    check("sch_uncommitted_is_schedule_uncommitted", rc == 2 and code_of(log) == "schedule_uncommitted",
          result_line(log))
    sw = five_finished("sch_head", over={2: dict(launched_over={"git_head": "1" * 40})}).build()
    rc, log, doc = sw.select(NOW_MID_L)
    check("pl39_20b_launch_head_unreadable_is_launch_log_head_unavailable",
          rc == 2 and code_of(log) == "launch_log_head_unavailable", result_line(log))
    sw = five_finished("sch_shallow").build()
    shallow = scratch_dir("k2_shallow")
    git(shallow, "clone", "-q", "--depth", "1", f"file://{sw.repo}", ".")
    rc, log = run_cli(sl, ["--runs", *map(str, sw.runs()), "--out", str(sw.root / "sh.json")], now=NOW_MID_L,
                      repo=shallow)
    check("pl39_20c_shallow_repository_is_repository_shallow", rc == 2 and code_of(log) == "repository_shallow",
          result_line(log))


def test_am19_launch_log(tmp: Path) -> None:
    """LL (P8) and PL-39 cases 7, 17, 18, 19 and 21: the launch log (PL-13, PL-14) and the code pin (PL-16)."""
    sw = five_finished("ll_absent").build()
    git(sw.repo, "rm", "-q", "--", ss.LAUNCH_LOG_REL)
    git(sw.repo, "commit", "-q", "-m", "no log")
    rc, log, _ = sw.select(NOW_MID_L)
    check("ll_absent_is_launch_log_missing", rc == 2 and code_of(log) == "launch_log_missing", result_line(log))
    sw = five_finished("ll_uncommitted").build()
    with open(sw.repo / ss.LAUNCH_LOG_REL, "ab") as fh:
        fh.write(b"\n")
    rc, log, _ = sw.select(NOW_MID_L)
    check("ll_uncommitted_is_launch_log_uncommitted", rc == 2 and code_of(log) == "launch_log_uncommitted",
          result_line(log))
    sw = five_finished("ll_torn").build()
    sw.commit("torn", log=launch_log_text(sw.lines + sw.events)[:-1])
    rc, log, _ = sw.select(NOW_MID_L)
    check("ll_torn_last_line_is_launch_log_format", rc == 2 and code_of(log) == "launch_log_format", result_line(log))
    sw = five_finished("ll_extra").build()
    extra = make_run(sw.root, "E2", 1, 0.4, name="e2_s42_lambda1_extra", launch=BASE_L)
    rc, log, _ = sw.select(NOW_MID_L, runs=sw.runs() + [extra])
    check("ll_directory_without_a_launch_line_is_launch_log_mismatch",
          rc == 2 and code_of(log) == "launch_log_mismatch", result_line(log))
    sw = five_finished("ll_17").build()
    some = [d for d in sw.runs() if "lambda2_" not in d.name]
    rc, log, _ = sw.select(NOW_MID_L, runs=some)
    check("pl39_17a_launched_line_without_its_directory_waits_before_the_date",
          rc == 3 and "shortfall_am19_item2" in log and "candidate_missing" in log, result_line(log))
    rc, log, _ = sw.select(NOW_L, runs=some)
    check("pl39_17b_launched_line_without_its_directory_is_launch_log_mismatch_after_the_date",
          rc == 2 and code_of(log) == "launch_log_mismatch", result_line(log))
    # PL-14 holds for every attempt of a value, not only its latest: before the date an earlier attempt
    # whose launched directory is withheld, or whose launch line has no outcome, makes the value wait
    sw = five_finished("ll_14_finished")
    sw.add(2, fin(BASE_L + 90000.0, 0.45))
    sw.build()
    rc, log, _ = sw.select(NOW_MID_L)
    check("pl14_two_finished_attempts_supplied_is_duplicate_candidate",
          rc == 2 and code_of(log) == "duplicate_candidate", result_line(log))
    rest = [d for d in sw.runs() if d.name != "e2_s42_lambda2_a1"]
    rc, log, _ = sw.select(NOW_MID_L, runs=rest)
    check("pl14_earlier_finished_attempt_withheld_waits_before_the_date",
          rc == 3 and "candidate_missing" in log and "e2_s42_lambda2_a1" in log, result_line(log))
    sw = Sweep("ll_14_stopped")
    for i, v in enumerate(LAMBDA_GRID):
        if v == 2:
            sw.add(2, course(BASE_L + 600 * i, 10000, val_at=()), stopped=True)
            sw.add(2, fin(BASE_L + 600 * i + 20000, 0.45))
        else:
            sw.add(v, fin(BASE_L + 600 * i, 0.40))
    sw.build()
    rc, log, doc = sw.select(NOW_MID_L)
    rest = [d for d in sw.runs() if d.name != "e2_s42_lambda2_a1"]
    rc2, log2, _ = sw.select(NOW_MID_L, runs=rest)
    check("pl14_earlier_stopped_attempt_withheld_waits_before_the_date",
          rc == 0 and doc["winner"]["lambda"] == 2.0 and rc2 == 3 and "candidate_missing" in log2,
          f"{result_line(log)} {result_line(log2)}")
    sw = Sweep("ll_14_pending")
    for i, v in enumerate(LAMBDA_GRID):
        if v == 2:
            sw.add(2, state="pending", launch=BASE_L + 600 * i)
            sw.add(2, fin(BASE_L + 600 * i + 300, 0.45))
        else:
            sw.add(v, fin(BASE_L + 600 * i, 0.40))
    sw.build()
    rc, log, _ = sw.select(NOW_MID_L)
    check("pl14_earlier_launch_line_without_an_outcome_waits_before_the_date",
          rc == 3 and "candidate_missing" in log and "neither a launched nor a not_launched" in log, result_line(log))
    rc, log, _ = sw.select(NOW_L)
    check("pl14_launch_line_without_an_outcome_is_launch_log_incomplete_after_the_date",
          rc == 2 and code_of(log) == "launch_log_incomplete", result_line(log))
    sw = five_finished("ll_rid").build()
    sw.events.append({"event": "stopped", "run_id": [], "report": None})
    sw.commit("a stopped line whose run_id is a list")
    rc, log, _ = sw.select(NOW_MID_L)
    check("ll_event_line_run_id_not_a_string_is_launch_log_format",
          rc == 2 and code_of(log) == "launch_log_format", result_line(log))
    sw = five_finished("ll_records_without_log")
    g_launch, g_launched = g_lines(sw.pin, sw.pin, BASE_L - DAY)
    sw.lines.append(g_launch)
    sw.events.append(g_launched)
    rc, log, _ = sw.build().select(NOW_MID_L)
    check("pl14_launched_line_whose_records_commit_holds_no_log_is_launch_log_rewritten",
          rc == 2 and code_of(log) == "launch_log_rewritten" and "holds no" in log, result_line(log))
    sw = five_finished("ll_event").build()
    sw.events.append({"event": [], "run_id": "e2_s42_lambda2_a1"})
    sw.commit("a line whose event is a list")
    rc, log, _ = sw.select(NOW_MID_L)
    check("ll_event_not_a_string_is_launch_log_format", rc == 2 and code_of(log) == "launch_log_format",
          result_line(log))

    # fixcode-1 and fixsmoke-6: the PL-14 wait comes after every refusal (exit 2) the latest attempts decide
    def withheld_first(label, value, second, **kw):
        sw = Sweep(f"ll_14_{label}")
        for i, v in enumerate(LAMBDA_GRID):
            L = BASE_L + 600 * i
            if v == value:
                sw.add(v, course(L, 10000, val_at=()), stopped=True)
                sw.add(v, second(L + 20000), **kw)
            else:
                sw.add(v, fin(L, 0.40))
        sw.build()
        rid = f"e2_s42_lambda{value:g}_a1"
        return sw.select(NOW_MID_L, runs=[d for d in sw.runs() if d.name != rid])
    rc, log, _ = withheld_first("default_diverged", 1, lambda L: course(L, 12000, val_at=(4000, 8000))
                                + [ABORT_ROW(12000, L + 12200.5)])
    check("pl14_withheld_earlier_attempt_of_a_diverged_default_is_default_candidate_diverged",
          rc == 2 and code_of(log) == "default_candidate_diverged", result_line(log))
    rc, log, _ = withheld_first("across", 2, lambda L: fin(L, 0.45), meta={"num_workers": 8})
    check("pl14_withheld_earlier_attempt_beside_a_recipe_difference_is_recipe_mismatch_across_candidates",
          rc == 2 and code_of(log) == "recipe_mismatch_across_candidates", result_line(log))
    rc, log, _ = withheld_first("refused", 2, lambda L: fin(L, 0.45), meta={"grad_clip_norm": 1.0})
    check("pl14_withheld_earlier_attempt_of_a_refused_value_is_that_refusal",
          rc == 2 and code_of(log) == "recipe_mismatch", result_line(log))
    sw = five_finished("ll_seed", over={2: dict(meta={"seed": 43})}).build()
    rc, log, _ = sw.select(NOW_MID_L)
    check("ll_launch_line_differs_from_run_meta_is_launch_log_mismatch",
          rc == 2 and code_of(log) == "launch_log_mismatch" and "seed" in log, result_line(log))
    sw = five_finished("ll_18a").build()
    # a run_meta that is still one valid row, edited after its launched line was printed
    set_meta(sw.dir("e2_s42_lambda2_a1") / "e2_run_meta.jsonl", edited_after_launch=True)
    rc, log, _ = sw.select(NOW_MID_L)
    check("pl39_18a_run_meta_sha256_differs_from_the_launched_line_is_launch_log_mismatch",
          rc == 2 and code_of(log) == "launch_log_mismatch" and "sha256" in log, result_line(log))
    rc, log, _ = sw.select(NOW_L)
    check("pl39_18a_run_meta_sha256_differs_after_the_date_is_launch_log_mismatch",
          rc == 2 and code_of(log) == "launch_log_mismatch" and "sha256" in log, result_line(log))
    sw = five_finished("ll_18b").build()
    sw.events.append(dict(next(e for e in sw.events if e["run_id"] == "e2_s42_lambda2_a1")))
    sw.commit("second launched line")
    rc, log, _ = sw.select(NOW_MID_L)
    check("pl39_18b_two_launched_lines_for_one_run_is_launch_log_format",
          rc == 2 and code_of(log) == "launch_log_format", result_line(log))
    sw = five_finished("ll_19").build()
    g_line = {"format": ss.LAUNCH_LOG_FORMAT, "event": "launch", "run_id": "g_s42_a1", "stage": "G", "seed": 42,
              "horizon": 80000, "sweep": None, "value": None, "lambda_logit": None, "alpha_cwd": None, "attempt": 1,
              "schedule": "T", "code_pin": sw.pin, "ckpt_dir": "/workspace/kd_ckpts/g_s42_a1", "am8a_report": None}
    sw.lines.insert(0, g_line)
    sw.commit("a line inserted before the sweep's lines")
    rc, log, _ = sw.select(NOW_MID_L)
    check("pl39_19_log_not_a_prefix_of_heads_is_launch_log_rewritten",
          rc == 2 and code_of(log) == "launch_log_rewritten", result_line(log))
    sw = Sweep("ll_07")
    for i, v in enumerate((1, 0.5, 2, 0.25, 4)):
        if v == 2:
            sw.add(2, state="not_launched", launch=BASE_L + 600 * i)
            sw.add(2, fin(BASE_L + 600 * i + 300, 0.40))
        else:
            sw.add(v, fin(BASE_L + 600 * i, 0.39 + 0.01 * i))
    rc, log, doc = sw.build().select(NOW_MID_L)
    check("pl39_07_attempt_after_a_not_launched_start_needs_no_report",
          rc == 0 and (doc or {}).get("directories", {}).get("2") == ["e2_s42_lambda2_a2"]
          and sw.lines[3]["am8a_report"] is None, f"rc={rc} {result_line(log)}")
    nl_dir = make_run(sw.root, "E2", 2, 0.4, name="e2_s42_lambda2_a1", launch=BASE_L + 1200)
    rc, log, _ = sw.select(NOW_MID_L, runs=sw.runs() + [nl_dir])
    check("ll_not_launched_attempt_supplied_with_a_run_meta_is_launch_log_mismatch",
          rc == 2 and code_of(log) == "launch_log_mismatch", result_line(log))
    # PL-39 case 21: code pins
    sw = five_finished("ll_21a")
    p2 = commit_files(sw.repo, {"docs/notes.md": "a second commit with the same code\n"}, "second pin")
    sw.atts[2]["code_pin"] = p2
    rc, log, _ = sw.build().select(NOW_MID_L)
    check("pl39_21a_two_code_pins_in_one_sweep_is_code_pin_mismatch",
          rc == 2 and code_of(log) == "code_pin_mismatch", result_line(log))
    sw = five_finished("ll_21b")
    lam_rel = ss.SELECT_SCRIPT_REL["lambda_logit"]
    sw.pin_files({lam_rel: Path(sl.__file__).read_bytes() + b"# a pinned variant\n"})
    sw.build()
    sw.commit("the running select script", {lam_rel: Path(sl.__file__).read_bytes()})
    rc, log, _ = sw.select(NOW_MID_L)
    check("pl39_21b_selection_code_differing_from_the_pin_is_code_pin_mismatch",
          rc == 2 and code_of(log) == "code_pin_mismatch" and "select_lambda.py" in log, result_line(log))
    sw = five_finished("ll_21c").build()
    sw.commit("unrelated", {"docs/notes.md": "an unrelated change after the pin\n"})
    rc, log, doc = sw.select(NOW_MID_L)
    check("pl39_21c_unrelated_file_changed_after_the_pin_is_accepted", rc == 0 and doc is not None,
          f"rc={rc} {result_line(log)}")
    sw = five_finished("ll_code").build()
    sw.commit("edited code", {ss.SWEEP_SELECT_REL: Path(ss.__file__).read_bytes() + b"# edited\n"})
    rc, log, _ = sw.select(NOW_MID_L)
    check("pl16_running_code_not_the_head_blob_is_code_uncommitted",
          rc == 2 and code_of(log) == "code_uncommitted", result_line(log))
    sw = five_finished("ll_order").build()
    o1, o2 = sw.root / "fwd.json", sw.root / "rev.json"
    rc1, _ = run_cli(sl, ["--runs", *map(str, sw.runs()), "--out", str(o1)], now=NOW_MID_L, repo=sw.repo)
    rc2, _ = run_cli(sl, ["--runs", *map(str, reversed(sw.runs())), "--out", str(o2)], now=NOW_MID_L, repo=sw.repo)
    check("ll_reversed_runs_order_writes_an_identical_file", rc1 == rc2 == 0 and o1.read_bytes() == o2.read_bytes())


def test_am19_status(tmp: Path) -> None:
    """ST (P8) and PL-39 cases 1, 2, 12, 14, 15, 16 and 28: the status of record at the decision date."""
    sw = lam_sweep("st_c1")
    l4 = LAMBDA_C - 50000.0
    sw.add(4, course(l4, 79000, val_at=(40000,), miou=0.60), stopped=True)
    rc0, log0, sha = sw.build().record(NOW_L)
    rc, log, doc = sw.select(NOW_L)
    cut = (doc or {}).get("cut_am19", [])
    d0 = cut[0]["directories"][0] if cut and cut[0]["directories"] else {}
    check("pl39_01_cut_candidate_with_the_highest_partial_val_does_not_win",
          rc0 == 0 and rc == 0 and doc["winner"]["lambda"] == 1.0 and [c["value"] for c in cut] == [4.0]
          and cut[0]["reason"] == "still_running" and d0.get("val_before_cutoff") == [[40000, 0.60]]
          and cut[0]["winner_val_same_iters"] == {"40000": 0.42}
          and abs(cut[0]["gpu_hours_to_cutoff"] - 40100 / 3600) < 1e-9
          and abs(cut[0]["gpu_hours_total"] - 79100 / 3600) < 1e-9
          and doc["decision_record"]["sha256"] == sha and doc["amendments"] == ["AM-2", "AM-7a", "AM-19 item 2"]
          and "cut=[4]" in result_line(log), f"rc0={rc0} rc={rc} {result_line(log0)} {result_line(log)} {cut[:1]}")
    sw = lam_sweep("st_c2")
    sw.add(4, course(LAMBDA_C - 80300.0, 80000, best=0.60, end=LAMBDA_C))
    rc0, log0, _ = sw.build().record(NOW_L)
    rc, log, doc = sw.select(NOW_L)
    cut = (doc or {}).get("cut_am19", [])
    check("pl39_02_run_end_exactly_at_the_cutoff_does_not_win",
          rc0 == 0 and rc == 0 and doc["winner"]["lambda"] == 1.0 and [(c["value"], c["reason"]) for c in cut]
          == [(4.0, "finished_after_date")], f"rc0={rc0} rc={rc} {result_line(log0)} {result_line(log)}")

    def after(label, rows4, *, expect_reason, stopped=False, meta=None, extra_check=None):
        sw = lam_sweep(f"st_{label}")
        sw.add(4, rows4, stopped=stopped, meta=meta, launch=None if rows4 else LAMBDA_C)
        rc0, log0, _ = sw.build().record(NOW_L)
        rc, log, doc = sw.select(NOW_L)
        cut = (doc or {}).get("cut_am19", [])
        ok = rc0 == 0 and rc == 0 and [(c["value"], c["reason"]) for c in cut] == [(4.0, expect_reason)]
        check(f"st_{label}", ok and (extra_check is None or extra_check(doc, cut)),
              f"rc0={rc0} rc={rc} {result_line(log0)} {result_line(log)} {[(c['value'], c['reason']) for c in cut]}")
        return sw

    l4 = BASE_L + 3000.0
    after("other_abort_is_run_aborted_other_rule_cause",
          course(l4, 5000, val_at=(4000,)) + [ABORT_ROW(5000, l4 + 5100.5, rule="val_nonfinite", cause="val_nonfinite")],
          expect_reason="run_aborted_other:val_nonfinite/val_nonfinite")
    after("checks_failed_is_run_checks_failed", fin(l4, 0.50, checks=False), expect_reason="run_checks_failed")
    after("diverged_after_the_date_is_aborted_after_date_and_am7a_item_8",
          [T(1, LAMBDA_C - 10999), T(12000, LAMBDA_C + 1000), ABORT_ROW(12000, LAMBDA_C + 1000.5)],
          expect_reason="aborted_after_date", extra_check=lambda doc, cut: cut[0]["am7a_item_8"] is True)
    after("launched_at_the_cutoff_is_never_launched", [T(1, LAMBDA_C + 1), T(2000, LAMBDA_C + 2000)],
          expect_reason="never_launched", stopped=True)
    after("recipe_expect_violated_is_cut_after_the_date", fin(l4, 0.50), expect_reason="recipe_mismatch",
          meta={"grad_clip_norm": 1.0})
    after("pl39_14c_backstep_across_the_cutoff_is_cut_telemetry_clock_straddle",
          [T(1, LAMBDA_C - 9999), T(1000, LAMBDA_C + 100), T(2000, LAMBDA_C - 100)],
          expect_reason="telemetry_clock_straddle", stopped=True)
    after("pl39_28a_unreported_unrepeated_abort_is_stopped_early",
          course(l4, 20000, val_at=(4000,)) + [ABORT_ROW(20000, l4 + 20100.5, rule="val_nonfinite",
                                                         cause="val_nonfinite")],
          expect_reason="run_aborted_other:val_nonfinite/val_nonfinite",
          extra_check=lambda doc, cut: doc["stopped_early_unexplained"] == [4.0] and cut[0]["stopped_early_unexplained"])
    after("pl39_28b_stop_with_a_stopped_line_is_not_stopped_early", course(l4, 20000, val_at=()),
          expect_reason="stopped_no_finished_repeat", stopped=True,
          extra_check=lambda doc, cut: doc["stopped_early_unexplained"] == [] and not cut[0]["stopped_early_unexplained"])

    sw = lam_sweep("st_12a")
    sw.add(4, course(LAMBDA_C - 10000, 1000, val_at=(), iter_s=0.05), stopped=True)
    sw.add(4, fin(LAMBDA_C + 3600, 0.60, iter_s=0.05, val_s=10.0))
    rc0, log0, _ = sw.build().record(NOW_L)
    rc, log, doc = sw.select(NOW_L)
    cut = (doc or {}).get("cut_am19", [])
    check("pl39_12a_stop_without_a_val_row_is_not_on_course",
          rc0 == 0 and rc == 0 and doc["winner"]["lambda"] == 1.0
          and [(c["value"], c["reason"]) for c in cut] == [(4.0, "stopped_no_finished_repeat")],
          f"rc0={rc0} rc={rc} {result_line(log0)} {result_line(log)}")

    sw = lam_sweep("st_14a")
    l4 = BASE_L + 3000.0
    sw.add(4, [T(1, l4 + 1), T(40000, l4 + 40000), V(40000, l4 + 40100, 0.38), T(70000, l4 + 70100),
               T(75000, l4 + 69100), T(80000, l4 + 74100), V(80000, l4 + 74200, 0.38), END(l4 + 74201, 0.38, start=l4)])
    rc, log, doc = sw.build().select(NOW_MID_L)
    check("pl39_14a_backstep_not_crossing_the_cutoff_is_accepted_and_recorded",
          rc == 0 and (doc or {}).get("clock", {}).get("e2_s42_lambda4_a1")
          == {"clock_backsteps": 1, "max_backstep_seconds": 1000.0}, f"rc={rc} {result_line(log)}")
    sw = lam_sweep("st_14b")
    sw.add(4, [T(1, LAMBDA_C - 9999), T(1000, LAMBDA_C + 100), T(2000, LAMBDA_C - 100)])
    rc, log, _ = sw.build().select(NOW_MID_L)
    check("pl39_14b_backstep_across_the_cutoff_is_refused_before_the_date",
          rc == 2 and code_of(log) == "telemetry_clock_straddle", result_line(log))
    sw = lam_sweep("st_15")
    sw.add(4, [T(1, LAMBDA_C - 999), T(1000, NOW_L + 3600)])
    rc, log, _ = sw.build().select(NOW_L)
    check("pl39_15_timestamp_later_than_now_after_the_date_is_clock_inconsistent",
          rc == 2 and code_of(log) == "clock_inconsistent", result_line(log))
    # the non-default 0.5 has the best score: a refusal must not turn into a cut (PL-6, R2)
    sw = lam_sweep("st_16a", best=(0.40, 0.45, 0.39, 0.38), over={0.5: dict(ckpt="corrupt")})
    sw.add(4, fin(BASE_L + 2400, 0.37))
    rc, log, _ = sw.build().select(NOW_L)
    check("pl39_16a_checkpoint_unreadable_after_the_date_is_refused",
          rc == 2 and code_of(log) == "checkpoint_unreadable", result_line(log))
    sw = lam_sweep("st_16b", best=(0.40, 0.45, 0.39, 0.38), over={0.5: dict(ckpt=False)})
    sw.add(4, fin(BASE_L + 2400, 0.37))
    rc, log, doc = sw.build().select(NOW_L)
    check("pl39_16b_run_without_its_checkpoint_is_finished_and_can_win",
          rc == 0 and doc["winner"]["lambda"] == 0.5 and doc["winner"]["ckpt_sha256"] is None
          and doc["winner"]["checkpoint_present"] is False and doc["n_cut"] == 0
          and "checkpoint is absent" in result_line(log), f"rc={rc} {result_line(log)}")
    sw = lam_sweep("st_nobest", over={1: dict(best_json=False)})
    sw.add(4, fin(BASE_L + 2400, 0.38))
    rc, log, doc = sw.build().select(NOW_MID_L)
    one = next((x for x in (doc or {}).get("inputs", []) if x["run_dir"].endswith("lambda1_a1")), {})
    check("st_finished_without_best_json_is_finished", rc == 0 and one.get("best_json_sha256") is None
          and doc["winner"]["checkpoint_present"] is True, f"rc={rc} {result_line(log)} {one}")
    for when, now in (("before", NOW_MID_L), ("after", NOW_L)):
        sw = lam_sweep(f"st_bestjson_{when}")
        sw.add(4, fin(BASE_L + 2400, 0.38))
        sw.build()
        (sw.dir("e2_s42_lambda4_a1") / "best.json").write_text(json.dumps(
            {"best_ckpt": f"/workspace/kd_ckpts/e2_s42_lambda4_a1/{CKN['E2']}", "best_val_miou_all_class": 0.2}),
            encoding="utf-8")
        rc, log, _ = sw.select(now)
        check(f"st_best_json_disagreeing_is_refused_{when}_the_date", rc == 2 and code_of(log) == "run_end_mismatch",
              result_line(log))
    sw = lam_sweep("st_recipe_before")
    sw.add(4, fin(BASE_L + 2400, 0.38), meta={"grad_clip_norm": 1.0})
    rc, log, _ = sw.build().select(NOW_MID_L)
    check("st_recipe_expect_violated_is_refused_before_the_date", rc == 2 and code_of(log) == "recipe_mismatch",
          result_line(log))
    sw = lam_sweep("st_recipe_default", over={1: dict(meta={"grad_clip_norm": 1.0})})
    sw.add(4, fin(BASE_L + 2400, 0.38))
    rc, log, _ = sw.build().select(NOW_L)
    check("st_recipe_expect_violated_by_the_default_is_refused_after_the_date",
          rc == 2 and code_of(log) == "recipe_mismatch", result_line(log))
    sw = lam_sweep("st_identical")
    sw.add(4, fin(BASE_L + 2400, 0.38), meta={"num_workers": 8})
    rc, log, _ = sw.build().select(NOW_L)
    check("st_recipe_identical_difference_refuses_after_the_date",
          rc == 2 and code_of(log) == "recipe_mismatch_across_candidates" and "differ in num_workers" in log,
          result_line(log))
    sw = lam_sweep("st_wall_clock")
    rows = fin(BASE_L + 2400, 0.38)
    rows[1] = {k: v for k, v in rows[1].items() if k != "wall_clock"}
    sw.add(4, rows, launch=BASE_L + 2400)
    rc, log, _ = sw.build().select(NOW_MID_L)
    check("st_row_without_a_timestamp_is_telemetry_clock_format",
          rc == 2 and code_of(log) == "telemetry_clock_format", result_line(log))
    sw = lam_sweep("st_default_after", over={1: dict(rows=fin(LAMBDA_C - 40000.0, 0.45))})
    sw.add(4, fin(BASE_L + 2400, 0.38))
    rc, log, doc = sw.build().select(NOW_L)
    check("st_default_finished_after_the_date_is_finished", rc == 0 and doc["winner"]["lambda"] == 1.0
          and doc["n_cut"] == 0, f"rc={rc} {result_line(log)}")
    sw = lam_sweep("st_default_running", over={1: dict(rows=course(LAMBDA_C - 40000.0, 60000))})
    sw.add(4, fin(BASE_L + 2400, 0.38))
    rc, log, _ = sw.build().select(NOW_L)
    check("st_default_without_an_end_record_waits_naming_item_2d", rc == 3 and "item 2(d)" in log
          and "shortfall_am19_item2" in log, result_line(log))
    sw = lam_sweep("st_diverged_before")
    sw.add(4, course(BASE_L + 3000, 12000, val_at=(4000, 8000)) + [ABORT_ROW(12000, BASE_L + 3000 + 12200.5)])
    rc, log, doc = sw.build().select(NOW_L)
    check("st_diverged_before_the_date_is_excluded_after_it", rc == 0 and doc["n_cut"] == 0
          and [e["value"] for e in doc["excluded_am7a"]] == [4.0], f"rc={rc} {result_line(log)}")
    # PL-7: an on-course test that fails is recorded with its inputs too (selection file and record)
    sw = lam_sweep("st_not_on_course")
    sw.add(4, course(LAMBDA_C - 50000.0, 10000, val_at=(4000,)), stopped=True)
    rc0, log0, _ = sw.build().record(NOW_L)
    rec = json.loads((sw.repo / ss.DECISION_RECORD_REL["lambda_logit"]).read_text(encoding="utf-8")) if rc0 == 0 else {}
    rc, log, doc = sw.select(NOW_L)
    cut = (doc or {}).get("cut_am19", [])
    ev = cut[0]["on_course"] if cut else None
    check("st_stop_not_on_course_is_recorded_with_its_inputs",
          rc0 == 0 and rc == 0 and [(c["value"], c["reason"]) for c in cut] == [(4.0, "stopped_no_finished_repeat")]
          and isinstance(ev, list) and len(ev) == 1 and ev[0]["run_id"] == "e2_s42_lambda4_a1"
          and ev[0]["on_course"] is False and "projection" in ev[0] and doc["on_course"]["4"] == ev
          and next(e for e in rec["values"] if e["value"] == 4.0)["on_course"] == ev,
          f"{result_line(log0)} {result_line(log)} {ev}")
    # runs-5: a non-finite row with no abort record is that run's ending (interpretation 5), so it is no
    # on-course stop: its finished-after-the-date repeat does not make the value finished
    sw = lam_sweep("st_nonfinite_stop")
    a1 = oc_stop_rows()
    a1[-1].update(loss=None, nonfinite={"loss": "nan"})
    sw.add(4, a1, stopped=True)
    sw.add(4, fin(LAMBDA_C - 5000.0, 0.60, iter_s=0.25, val_s=50.0))
    rc0, log0, _ = sw.build().record(NOW_L)
    rc, log, doc = sw.select(NOW_L)
    check("st_nonfinite_row_without_an_abort_record_is_no_on_course_stop",
          rc0 == 0 and rc == 0 and doc["winner"]["lambda"] == 1.0
          and [(c["value"], c["reason"]) for c in doc["cut_am19"]] == [(4.0, "finished_after_date")],
          f"{result_line(log0)} {result_line(log)}")
    # alpha-7: a repeat launched after the date has no iteration before it; the cut's last iteration is s's
    sw = lam_sweep("st_last_iter")
    sw.add(4, oc_stop_rows(), stopped=True)
    sw.add(4, [T(1, LAMBDA_C + 3601), ABORT_ROW(1, LAMBDA_C + 3601.5, rule="step1_checks", cause="checks_failed")])
    rc0, log0, _ = sw.build().record(NOW_L)
    rc, log, doc = sw.select(NOW_L)
    cut = (doc or {}).get("cut_am19", [])
    rep = (doc or {}).get("on_course_repeat", {}).get("4", {})
    check("st_cut_last_iteration_is_the_largest_before_the_date_and_both_runs_vals_are_reported",
          rc0 == 0 and rc == 0 and [(c["value"], c["reason"]) for c in cut]
          == [(4.0, "run_aborted_other:step1_checks/checks_failed")] and cut[0]["last_iter"] == 10000
          and cut[0]["directories"][1]["last_iter_before_cutoff"] is None
          and rep.get("stopped") == "e2_s42_lambda4_a1" and rep.get("repeat") == "e2_s42_lambda4_a2"
          and rep.get("stopped_val") == [[4000, 0.40], [8000, 0.40]] and rep.get("repeat_val") == [],
          f"{result_line(log0)} {result_line(log)} {cut[:1]} {rep}")


def ABORT_ROW(it, ts, **kw) -> dict:
    return abort_record(it, ts=float(ts), **kw)


def run_rows(rows: list) -> "ss.Run":
    r = ss.Run(Path("unit_run"))
    r.rows, r.row_ts = rows, [ss._row_ts(x) for x in rows]
    return r


def val_course(train_at, val_at, *, iter_s_at=None, val_s_at=None, t0=1000.0) -> list:
    """Unit rows: a train row at each iteration of train_at (1 s per iteration unless iter_s_at names
    another), a VAL row after each iteration of val_at (300 s unless val_s_at names another)."""
    rows, t, prev = [], t0, 0
    for it in sorted(set(train_at)):
        s = (iter_s_at or {}).get(it, 1.0)
        t += (it - prev) * 1.0
        prev = it
        rows.append(T(it, t, s))
        if it in val_at:
            vs = (val_s_at or {}).get(it, 300.0)
            t += vs
            rows.append(V(it, t, val_s=vs))
    return rows


def test_am19_on_course(tmp: Path) -> None:
    """PL-7 and PL-39 cases 12, 13; PL-8 and case 28 at the unit level (P8 DR's 1200 s rows)."""
    vals = list(range(4000, 80000, 4000))
    rows = val_course(vals + [80000], vals, val_s_at={8000: 600.0})
    t_last = rows[-1]["wall_clock"]
    ok, inp = ss.on_course(run_rows(rows), t_last + 450.0, 80000)
    check("pl39_13a_pending_final_validation_counts_with_the_largest_val_seconds",
          ok is False and inp["remaining_validations"] == 1 and inp["remaining_iterations"] == 0
          and inp["max_val_seconds"] == "600.0", str(inp))
    ok, inp = ss.on_course(run_rows(rows), t_last + 600.5, 80000)
    check("pl39_13b_stop_during_the_final_validation_leaves_one_validation",
          ok is True and inp["remaining_validations"] == 1 and inp["projection"] == str(Decimal(repr(t_last)) + 600),
          str(inp))
    rows = val_course(vals[:-1] + [76000, 79000], vals)
    t_last = rows[-1]["wall_clock"]
    ok, inp = ss.on_course(run_rows(rows), t_last + 2000.0, 80000)
    check("pl39_13c_remaining_iterations_come_from_the_last_train_row", ok is True
          and inp["remaining_iterations"] == 1000 and inp["remaining_validations"] == 1, str(inp))
    rows = val_course(vals, vals, iter_s_at={76000: 13.0})
    t_last = rows[-1]["wall_clock"]
    ok, inp = ss.on_course(run_rows(rows), t_last + 5000.0, 80000)
    check("st_on_course_uses_the_median_iter_seconds_not_the_mean", ok is True
          and inp["median_iter_seconds"] == "1.0", str(inp))
    rows = val_course(vals, vals)
    t_last = rows[-1]["wall_clock"]
    ok, inp = ss.on_course(run_rows(rows), t_last + 4000 + 300, 80000)
    check("st_projection_exactly_at_the_cutoff_is_not_on_course", ok is False
          and inp["projection"] == str(Decimal(repr(t_last)) + 4300), str(inp))
    ok, inp = ss.on_course(run_rows([T(1, 1001.0), T(1000, 2000.0)]), 1e12, 80000)
    check("pl39_12_no_val_row_is_not_on_course", ok is False and inp["reason"] == "no VAL row", str(inp))
    ok, inp = ss.on_course(run_rows([V(4000, 5000.0)]), 1e12, 80000)
    check("pl39_12_no_train_row_is_not_on_course", ok is False and inp["reason"] == "no train row", str(inp))
    # r2-2: on_course() is total: a last row without a finite timestamp, or a VAL row whose iter is no
    # integer, is not on course (no crash)
    rows = val_course([1000, 4000], [4000])
    rows[-1] = dict(rows[-1], wall_clock=None)
    got = outcome(ss.on_course, run_rows(rows), 1e12, 80000)
    check("st_on_course_last_row_without_a_finite_timestamp_is_not_on_course",
          isinstance(got, tuple) and got[0] is False and "timestamp" in got[1]["reason"], str(got))
    rows = val_course([1000, 4000], [4000])
    rows[-1] = dict(rows[-1], iter=[4000])
    got = outcome(ss.on_course, run_rows(rows), 1e12, 80000)
    check("st_on_course_val_iter_not_an_integer_is_not_on_course",
          isinstance(got, tuple) and got[0] is False and "integer" in got[1]["reason"], str(got))

    def se(rows, cutoff, stopped=None, later=()):
        return ss.stopped_early(SimpleNamespace(run=run_rows(rows), stopped=stopped, run_id="unit_run"), cutoff,
                                list(later))[0]
    rows = [T(1, 1000.0), T(2, 1001.0)]
    check("dr_stopped_early_1201_seconds_is_marked", se(rows, 1001.0 + 1201) is True)
    check("dr_stopped_early_1200_seconds_is_not_marked", se(rows, 1001.0 + 1200) is False)
    rows2 = rows + [V(2, 1002.0, val_s=700.0)]
    check("dr_stopped_early_threshold_is_twice_the_largest_val_seconds", se(rows2, 1002.0 + 1300) is False
          and se(rows2, 1002.0 + 1401) is True)
    check("pl39_28c_not_marked_with_a_stopped_line_or_a_later_attempt",
          se(rows, 1001.0 + 5000, stopped={"report": {}}) is False and se(rows, 1001.0 + 5000, later=[object()]) is False)


def test_am19_repeats(tmp: Path) -> None:
    """RP (P8) and PL-39 cases 3-6 and 8-11: attempts and repeats (PL-9)."""
    l4 = BASE_L + 3000.0

    def case(label, attempts, now, *, record=False, default=None):
        sw = lam_sweep(f"rp_{label}", over=default or {})
        for rows, kw in attempts:
            sw.add(4, rows, **kw)
        sw.build()
        rc0 = log0 = None
        if record:
            rc0, log0, _ = sw.record(now)
        rc, log, doc = sw.select(now)
        return sw, rc, log, doc, rc0, log0

    stop = course(l4, 10000, val_at=())
    sw, rc, log, doc, _, _ = case("finished_repeat", [(stop, dict(stopped=True)), (fin(l4 + 20000, 0.38), {})], NOW_MID_L)
    check("rp_stopped_attempt_then_finished_repeat_with_its_report_is_finished",
          rc == 0 and doc["directories"]["4"] == ["e2_s42_lambda4_a1", "e2_s42_lambda4_a2"]
          and sw.lines[-1]["am8a_report"] is not None, f"rc={rc} {result_line(log)}")
    _, rc, log, _, _, _ = case("no_report", [(stop, dict(stopped=True)), (fin(l4 + 20000, 0.38), dict(report=False))],
                               NOW_MID_L)
    check("rp_repeat_without_its_report_is_repeat_report_missing",
          rc == 2 and code_of(log) == "repeat_report_missing", result_line(log))
    sw = lam_sweep("rp_bad_report")
    sw.files["reports/am8a/bad.md"] = b"a report\n"
    sw.add(4, stop, stopped=True)
    sw.add(4, fin(l4 + 20000, 0.38), report={"path": "reports/am8a/bad.md", "sha256": "0" * 64})
    rc, log, _ = sw.build().select(NOW_MID_L)
    check("rp_repeat_report_with_another_sha256_is_repeat_report_mismatch",
          rc == 2 and code_of(log) == "repeat_report_mismatch", result_line(log))
    _, rc, log, _, _, _ = case("c3", [(fin(l4, 0.38), {}), (course(LAMBDA_C + 3600, 5000), {})], NOW_L)
    check("pl39_03_later_attempt_of_a_finished_candidate_after_the_date_is_repeat_after_end_not_a_cut",
          rc == 2 and code_of(log) == "repeat_after_end", result_line(log))
    div_rows = course(l4, 12000, val_at=(4000, 8000)) + [ABORT_ROW(12000, l4 + 12200.5)]
    _, rc, log, _, _, _ = case("div_then_running", [(div_rows, {}), (course(l4 + 20000, 5000), {})], NOW_MID_L)
    check("rp_diverged_attempt_then_a_repeat_is_repeat_after_end",
          rc == 2 and code_of(log) == "repeat_after_end", result_line(log))
    div2 = course(l4 + 20000, 12000, val_at=(4000, 8000)) + [ABORT_ROW(12000, l4 + 20000 + 12200.5)]
    _, rc, log, _, _, _ = case("two_diverged", [(div_rows, {}), (div2, {})], NOW_MID_L)
    check("rp_two_diverged_attempts_is_duplicate_candidate",
          rc == 2 and code_of(log) == "duplicate_candidate", result_line(log))
    failed = fin(l4, 0.38, checks=False)
    _, rc, log, doc, _, _ = case("c4", [(failed, {}), (fin(l4 + 90000, 0.38), {})], NOW_MID_L)
    check("pl39_04_failed_check_then_finished_repeat_with_its_report_is_finished",
          rc == 0 and doc["directories"]["4"] == ["e2_s42_lambda4_a1", "e2_s42_lambda4_a2"], result_line(log))
    _, rc, log, doc, rc0, log0 = case("c5", [(failed, {}), (fin(l4 + 90000, 0.38, checks=False), {})], NOW_L,
                                      record=True)
    check("pl39_05_failed_check_twice_is_cut_aborted_twice_run_end_checks_failed",
          rc0 == 0 and rc == 0 and [(c["value"], c["reason"]) for c in doc["cut_am19"]]
          == [(4.0, "aborted_twice:run_end/checks_failed")], f"{result_line(log0 or '')} {result_line(log)}")
    ab1 = course(l4, 5000, val_at=(4000,)) + [ABORT_ROW(5000, l4 + 5100.5, rule="val_nonfinite", cause="val_nonfinite")]
    ab2 = course(l4 + 9000, 5000, val_at=(4000,)) + [ABORT_ROW(5000, l4 + 14100.5, rule="val_nonfinite",
                                                               cause="val_nonfinite")]
    _, rc, log, doc, rc0, log0 = case("abort_twice", [(ab1, {}), (ab2, {})], NOW_L, record=True)
    check("rp_same_abort_twice_is_cut_aborted_twice_rule_cause",
          rc0 == 0 and rc == 0 and [(c["value"], c["reason"]) for c in doc["cut_am19"]]
          == [(4.0, "aborted_twice:val_nonfinite/val_nonfinite")], f"{result_line(log0 or '')} {result_line(log)}")
    _, rc, log, _, _, _ = case("c6", [(ab1, {}), (ab2, {}), (course(l4 + 20000, 1000), {})], NOW_MID_L)
    check("pl39_06_third_attempt_after_aborted_twice_is_repeat_after_aborted_twice",
          rc == 2 and code_of(log) == "repeat_after_aborted_twice", result_line(log))
    sw = Sweep("rp_default_twice")
    sw.add(1, course(BASE_L, 5000, val_at=(4000,)) + [ABORT_ROW(5000, BASE_L + 5100.5, rule="val_nonfinite",
                                                                cause="val_nonfinite")])
    sw.add(1, course(BASE_L + 9000, 5000, val_at=(4000,)) + [ABORT_ROW(5000, BASE_L + 14100.5, rule="val_nonfinite",
                                                                       cause="val_nonfinite")])
    for i, v in enumerate((0.5, 2, 0.25, 4)):
        sw.add(v, fin(BASE_L + 20000 + 600 * i, 0.40))
    rc, log, _ = sw.build().select(NOW_MID_L)
    check("rp_default_ending_the_same_way_twice_is_default_aborted_twice",
          rc == 2 and code_of(log) == "default_aborted_twice", result_line(log))
    _, rc, log, _, _, _ = case("c8", [(stop, dict(stopped=True)), (fin(l4 + 5000, 0.38), {})], NOW_MID_L)
    check("pl39_08_attempts_overlapping_in_time_is_repeat_overlap",
          rc == 2 and code_of(log) == "repeat_overlap", result_line(log))
    # an on-course stop s: 0.25 s per iteration, VAL at 4000 and 8000 (50 s), stopped at 10000 at C - 27400;
    # its projection C - 9000 meets the date
    ls = LAMBDA_C - 30000.0
    s_rows = course(ls, 10000, val_at=(4000, 8000), iter_s=0.25, val_s=50.0)
    s = (s_rows, dict(stopped=True))
    _, rc, log, _, _, _ = case("c9", [s, (course(LAMBDA_C - 5000, 40000, val_at=(4000,), iter_s=0.25, val_s=50.0), {})],
                               NOW_L)
    check("pl39_09_on_course_stop_with_its_repeat_still_running_waits", rc == 3 and "on-course" in log
          and "shortfall_am19_item2" in log, result_line(log))
    r10 = [T(1, LAMBDA_C - 4999), ABORT_ROW(1, LAMBDA_C - 4998.5, rule="step1_checks", cause="checks_failed")]
    _, rc, log, doc, rc0, log0 = case("c10", [s, (r10, {})], NOW_L, record=True)
    check("pl39_10_on_course_stop_whose_repeat_aborts_is_cut_with_that_code",
          rc0 == 0 and rc == 0 and [(c["value"], c["reason"]) for c in doc["cut_am19"]]
          == [(4.0, "run_aborted_other:step1_checks/checks_failed")], f"{result_line(log0 or '')} {result_line(log)}")
    r11 = course(LAMBDA_C - 5000, 20000, val_at=(4000,), iter_s=0.25, val_s=50.0)
    _, rc, log, _, _, _ = case("c11", [s, (r11, dict(stopped=True)), (course(LAMBDA_C + 2000, 3000), {})], NOW_L)
    check("pl39_11_on_course_stop_whose_repeat_stopped_with_a_later_attempt_running_is_cut",
          rc == 3 and "on_course_repeat_stopped" in log, result_line(log))
    _, rc, log, doc, _, _ = case("on_course_finished",
                                 [s, (fin(LAMBDA_C - 5000, 0.60, iter_s=0.25, val_s=50.0), {})], NOW_L)
    check("rp_on_course_stop_whose_repeat_finishes_after_the_date_is_finished",
          rc == 0 and doc["winner"]["lambda"] == 4.0 and doc["n_cut"] == 0, f"rc={rc} {result_line(log)}")
    _, rc, log, _, _, _ = case("on_course_no_repeat", [s], NOW_L)
    check("rp_on_course_stop_without_a_repeat_waits", rc == 3 and "stopped on course" in log, result_line(log))
    # runs-3: a repeat with its run_meta and no telemetry yet has no ending: it waits, or is cut as stopped
    for label, stopped in (("waits", False), ("stopped_is_cut_on_course_repeat_stopped", True)):
        sw = lam_sweep(f"rp_oc_no_telemetry_{int(stopped)}")
        sw.add(4, oc_stop_rows(), stopped=True)
        sw.add(4, [], launch=LAMBDA_C - 5000.0, stopped=stopped)
        sw.build()
        (sw.dir("e2_s42_lambda4_a2") / "e2_telemetry.jsonl").unlink()
        rc0, log0, _ = sw.record(NOW_L)
        rec = json.loads((sw.repo / ss.DECISION_RECORD_REL["lambda_logit"]).read_text(encoding="utf-8")) \
            if rc0 == 0 else {"values": []}
        st4 = next((e["status"] for e in rec["values"] if e["value"] == 4.0), None)
        rc, log, doc = sw.select(NOW_L)
        ok = (rc0 == 0 and rc == 0 and [(c["value"], c["reason"]) for c in doc["cut_am19"]]
              == [(4.0, "on_course_repeat_stopped")]) if stopped else (
            rc0 == 0 and st4 == ss.RUNNING_ON_COURSE and rc == 3 and "on-course repeat" in log)
        check(f"rp_on_course_repeat_without_its_telemetry_{label}", ok, f"{result_line(log0)} {result_line(log)}")
    # PL-9(e)(3): a later attempt cuts the on-course repeat even without a stopped line (fixsmoke-7)
    sw = lam_sweep("rp_oc_later_attempt")
    sw.add(4, oc_stop_rows(), stopped=True)
    sw.add(4, course(LAMBDA_C - 5000.0, 4000, val_at=(4000,), iter_s=0.25, val_s=50.0))
    sw.add(4, course(LAMBDA_C + 2000.0, 3000))
    rc, log, _ = sw.build().select(NOW_L)
    check("rp_on_course_repeat_followed_by_a_later_attempt_is_cut_on_course_repeat_stopped",
          rc == 3 and "on_course_repeat_stopped" in log, result_line(log))
    # runs-6 (interpretation 13): before the date, a non-default value that ended the same way twice waits
    # for the date, where it is cut; it is not repeated again
    _, rc, log, _, _, _ = case("c5_before", [(failed, {}), (fin(l4 + 90000, 0.38, checks=False), {})], NOW_MID_L)
    check("rp_failed_check_twice_before_the_date_waits", rc == 3 and "aborted_twice:run_end/checks_failed" in log,
          result_line(log))

    # r2-1 (interpretations 13 and 15): before the date a non-default value's single abort or failed check
    # waits, not refused, while the attempt before it is incomplete and some completion of it could make this
    # the second same ending, which waits (exit 3)
    def before_c(label, attempts, *, withhold=(), unlink=()):
        sw = lam_sweep(f"rp_{label}")
        for rows, kw in attempts:
            sw.add(4, rows, **kw)
        sw.build()
        for rid in unlink:
            (sw.dir(rid) / "e2_telemetry.jsonl").unlink()
        return sw.select(NOW_MID_L, runs=[d for d in sw.runs() if d.name not in withhold])
    a1 = "e2_s42_lambda4_a1"
    rc, log, _ = before_c("abort_twice_first_withheld", [(ab1, {}), (ab2, {})], withhold=(a1,))
    check("rp_same_abort_twice_with_the_first_withheld_waits_before_the_date",
          rc == 3 and "candidate_missing" in log and a1 in log, result_line(log))
    rc, log, _ = before_c("failed_twice_first_withheld", [(failed, {}), (fin(l4 + 90000, 0.38, checks=False), {})],
                          withhold=(a1,))
    check("rp_failed_check_twice_with_the_first_withheld_waits_before_the_date",
          rc == 3 and "candidate_missing" in log and a1 in log, result_line(log))
    rc, log, _ = before_c("abort_after_pending", [([], dict(state="pending", launch=l4)), (ab2, dict(report=True))])
    check("rp_abort_after_a_launch_line_without_an_outcome_waits_before_the_date",
          rc == 3 and "candidate_missing" in log and a1 in log, result_line(log))
    rc, log, _ = before_c("abort_after_no_telemetry", [([], dict(launch=l4)), (ab2, {})], unlink=(a1,))
    check("rp_abort_after_an_attempt_without_telemetry_waits_before_the_date",
          rc == 3 and "candidate_incomplete" in log and a1 in log, result_line(log))
    late_ab = course(l4 + 20000, 5000, val_at=(4000,)) + [ABORT_ROW(5000, l4 + 25100.5, rule="val_nonfinite",
                                                                    cause="val_nonfinite")]
    rc, log, _ = before_c("abort_after_a_stop", [(stop, dict(stopped=True)), (late_ab, {})])
    check("rp_abort_after_a_stopped_attempt_is_run_aborted_other_before_the_date",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))
    # r3-1: no wait where every completion of the log refuses: a stopped attempt before it (supplied without
    # telemetry, which is then final, or withheld while the latest ended in a trainer record), a latest that
    # names no AM-8a report, or an attempt before the withheld one that ended the same way. r3-3: only the last
    # earlier attempt without a not_launched line counts
    rc, log, _ = before_c("abort_after_stopped_no_telemetry", [([], dict(launch=l4, stopped=True)), (ab2, {})],
                          unlink=(a1,))
    check("rp_abort_after_a_stopped_attempt_without_telemetry_is_run_aborted_other_before_the_date",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))
    rc, log, _ = before_c("abort_after_stopped_withheld", [(stop, dict(stopped=True)), (late_ab, {})], withhold=(a1,))
    check("rp_abort_after_a_stopped_attempt_withheld_is_run_aborted_other_before_the_date",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))
    # a non-finite row without its abort record is no trainer record, so a stopped attempt could end so too:
    # it waits while that attempt is withheld, not once it is supplied (its telemetry is then final)
    nf2 = course(l4 + 9000, 5000, val_at=(4000,))
    nf2[-1].update(loss=None, nonfinite={"loss": "nan"})
    nf3 = course(l4 + 20000, 5000, val_at=(4000,))
    nf3[-1].update(loss=None, nonfinite={"loss": "nan"})
    rc, log, _ = before_c("nonfinite_after_stopped_no_telemetry", [([], dict(launch=l4, stopped=True)), (nf2, {})],
                          unlink=(a1,))
    check("rp_nonfinite_row_after_a_stopped_attempt_without_telemetry_is_run_aborted_other_before_the_date",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))
    rc, log, _ = before_c("nonfinite_after_stopped_withheld", [(stop, dict(stopped=True)), (nf3, {})], withhold=(a1,))
    check("rp_nonfinite_row_after_a_stopped_attempt_withheld_waits_before_the_date",
          rc == 3 and "candidate_missing" in log and a1 in log, result_line(log))
    rc, log, _ = before_c("abort_after_pending_no_report", [([], dict(state="pending", launch=l4)), (ab2, {})])
    check("rp_abort_naming_no_report_after_a_launch_line_without_an_outcome_is_run_aborted_other_before_the_date",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))
    a2 = "e2_s42_lambda4_a2"
    rc, log, _ = before_c("same_before_withheld", [(ab1, {}), (course(l4 + 9000, 1000, val_at=()), {}), (late_ab, {})],
                          withhold=(a2,))
    check("rp_abort_after_a_withheld_attempt_whose_predecessor_ended_the_same_way_is_run_aborted_other_before_the_date",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))

    def step1(L):
        return [T(1, L + 1), ABORT_ROW(1, L + 1.5, rule="step1_checks", cause="checks_failed")]
    rc, log, _ = before_c("other_then_withheld", [(step1(l4), {}), (ab2, {}), (late_ab, {})], withhold=(a2,))
    check("rp_abort_after_a_withheld_attempt_whose_predecessor_ended_otherwise_waits_before_the_date",
          rc == 3 and "candidate_missing" in log and a2 in log, result_line(log))
    rc, log, _ = before_c("withheld_then_other", [(ab1, {}), (step1(l4 + 9000), {}), (late_ab, {})], withhold=(a1,))
    check("rp_abort_after_a_complete_attempt_with_a_withheld_one_before_it_is_run_aborted_other_before_the_date",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))
    rc, log, _ = before_c("not_launched_between", [(stop, dict(stopped=True)),
                                                   ([], dict(state="not_launched", launch=l4 + 15000)), (late_ab, {})])
    check("rp_abort_after_a_not_launched_start_is_run_aborted_other_before_the_date",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))

    # r2-5 and the default (interpretation 15): the default's refusals come before the PL-14 wait; its last
    # two attempts ending the same way refuse with an earlier one withheld, and so does a single abort
    def default_case(label, attempts, withhold):
        sw = Sweep(f"rp_{label}")
        for rows, kw in attempts:
            sw.add(1, rows, **kw)
        for i, v in enumerate((0.5, 2, 0.25, 4)):
            sw.add(v, fin(BASE_L + 60000 + 600 * i, 0.40))
        sw.build()
        return sw.select(NOW_MID_L, runs=[d for d in sw.runs() if d.name not in withhold])
    d_stop = (course(BASE_L, 10000, val_at=()), dict(stopped=True))
    d_ab = [(course(BASE_L + t, 5000, val_at=(4000,)) + [ABORT_ROW(5000, BASE_L + t + 5100.5, rule="val_nonfinite",
                                                                   cause="val_nonfinite")], {}) for t in (20000, 30000)]
    rc, log, _ = default_case("default_twice_first_withheld", [d_stop, *d_ab], ("e2_s42_lambda1_a1",))
    check("rp_default_ending_the_same_way_twice_with_an_earlier_attempt_withheld_is_default_aborted_twice",
          rc == 2 and code_of(log) == "default_aborted_twice", result_line(log))
    # an earlier attempt without a stopped line could still end the same way, as for a non-default value
    d_run = (course(BASE_L, 10000, val_at=()), {})
    rc, log, _ = default_case("default_abort_previous_withheld", [d_run, d_ab[0]], ("e2_s42_lambda1_a1",))
    check("rp_default_single_abort_with_the_attempt_before_it_withheld_is_run_aborted_other",
          rc == 2 and code_of(log) == "run_aborted_other", result_line(log))


def test_am19_record(tmp: Path) -> None:
    """DR (P8) and PL-39 case 22: the decision record (AM-19 item 2(h); PL-15)."""
    sw = lam_sweep("dr_none").build()
    rc, log, _ = sw.select(NOW_L)
    check("dr_cut_without_a_record_is_a_shortfall", rc == 3 and "shortfall_am19_item2" in log
          and "--write-decision-record" in log, result_line(log))
    rc0, log0, sha = sw.record(NOW_L)
    rec = json.loads((sw.repo / ss.DECISION_RECORD_REL["lambda_logit"]).read_text(encoding="utf-8"))
    rc, log, doc = sw.select(NOW_L)
    check("dr_valid_record_selects_with_the_cut_listed",
          rc0 == 0 and rc == 0 and doc["n_cut"] == 1 and doc["cut_am19"][0]["reason"] == "never_launched"
          and doc["decision_record"] == {"path": ss.DECISION_RECORD_REL["lambda_logit"], "sha256": sha}
          and rec["never_launched"] == [4.0] and rec["alpha_sweep_cut"] is False
          and [v["status"] for v in rec["values"]] == ["finished", "finished", "finished", "finished", "cut"]
          and "shortfall_disclosure" in doc and doc["shortfall_disclosure"], f"{result_line(log0)} {result_line(log)}")
    check("pl39_29_amendments_extended_with_a_cut", doc is not None
          and doc["amendments"] == ["AM-2", "AM-7a", "AM-19 item 2"])
    sw = lam_sweep("dr_before").build()
    rc, log, _ = sw.record(NOW_MID_L, commit=False)
    check("dr_write_decision_record_before_the_date_is_decision_date_not_ended",
          rc == 2 and code_of(log) == "decision_date_not_ended", result_line(log))
    sw = lam_sweep("dr_unended")
    sw.add(4, course(LAMBDA_C - 3000, 10000))
    rc, log, _ = sw.build().record(NOW_L, commit=False)
    check("dr_write_decision_record_with_a_run_unended_is_refused",
          rc == 2 and code_of(log) == "decision_record_runs_unended", result_line(log))

    def edited(label, edit, code, **kw):
        sw = lam_sweep(f"dr_{label}").build()
        rc0, log0, _ = sw.record(NOW_L, edit=edit, **kw)
        rc, log, _ = sw.select(NOW_L)
        check(f"dr_{label}_is_{code}", rc0 == 0 and rc == 2 and code_of(log) == code,
              f"{result_line(log0)} {result_line(log)}")

    def value(d, v):
        return next(e for e in d["values"] if e["value"] == v)
    # its decision-log row committed, the record file not (an explicit-path commit that left it out)
    edited("uncommitted", None, "decision_record_uncommitted", commit_record=False)
    edited("sha_not_in_decision_log", None, "decision_record_not_in_decision_log", dl=False)
    edited("status_differs", lambda d: value(d, 4.0).update(reason="still_running"), "decision_record_status_mismatch")
    edited("hash_differs", lambda d: value(d, 1.0)["directories"][0].update(run_meta_sha256="0" * 64),
           "decision_record_hash_mismatch")
    edited("launch_entry_missing", lambda d: value(d, 1.0).update(directories=[]), "decision_record_launch_log_mismatch")
    edited("directory_without_an_entry", lambda d: value(d, 4.0)["directories"].append(
        dict(value(d, 1.0)["directories"][0], run_id="e2_s42_lambda4_a1")), "decision_record_launch_log_mismatch")
    edited("launch_log_not_a_prefix", lambda d: d["launch_log"].update(sha256="0" * 64),
           "decision_record_launch_log_mismatch")
    edited("malformed", lambda d: d.pop("values"), "decision_record_format")
    edited("wrong_sweep", lambda d: d.update(sweep="alpha_cwd"), "decision_record_format")
    edited("pl39_22_other_decision_date", lambda d: d.update(decision_date="2026-10-20"),
           "decision_record_schedule_mismatch")
    edited("pl39_22_other_schedule", lambda d: d["schedule"].update(in_force="R"), "decision_record_schedule_mismatch")
    edited("pl39_22_other_cutoff", lambda d: d.update(cutoff_utc=LAMBDA_C + 1), "decision_record_schedule_mismatch")
    edited("on_course_differs", lambda d: value(d, 4.0).update(on_course=None), "decision_record_status_mismatch")
    # fixsmoke-1, -2 and -5: a running status only where it can hold; the last iteration and the launch time
    edited("running_default_on_a_non_default_value", lambda d: value(d, 0.5).update(status=ss.RUNNING_DEFAULT),
           "decision_record_status_mismatch")
    edited("running_on_course_on_a_never_launched_value",
           lambda d: value(d, 4.0).update(status=ss.RUNNING_ON_COURSE, reason=None), "decision_record_status_mismatch")
    edited("finished_last_iter_differs", lambda d: value(d, 1.0)["directories"][0].update(last_iter_before_cutoff=1),
           "decision_record_status_mismatch")
    edited("launch_time_differs", lambda d: value(d, 1.0)["directories"][0].update(
        launch_time_utc=value(d, 1.0)["directories"][0]["launch_time_utc"] + 1), "decision_record_launch_log_mismatch")
    sw = five_finished("dr_not_ended").build()
    rc0, log0, _ = sw.record(NOW_L)
    rc, log, _ = sw.select(NOW_MID_L)
    check("dr_record_present_before_the_date_is_decision_date_not_ended",
          rc0 == 0 and rc == 2 and code_of(log) == "decision_date_not_ended", f"{result_line(log0)} {result_line(log)}")
    # the default still running when the record is written: "running (item 2(d))" accepts its later finish
    l1 = LAMBDA_C - 40000.0
    sw = lam_sweep("dr_default_running", over={1: dict(rows=course(l1, 60000, val_at=(40000,), miou=0.45))}).build()
    rc0, log0, _ = sw.record(NOW_L)
    rec = json.loads((sw.repo / ss.DECISION_RECORD_REL["lambda_logit"]).read_text(encoding="utf-8"))
    d1 = sw.dir("e2_s42_lambda1_a1")
    with open(d1 / "e2_telemetry.jsonl", "a", encoding="utf-8") as fh:
        for row in (T(80000, l1 + 80100), V(80000, l1 + 80200, 0.45), END(l1 + 80201, 0.45, start=l1)):
            fh.write(json.dumps(row) + "\n")
    torch.save({"stage": "E2", "iter": 80000, "best_val_miou_all_class": 0.45, "lambda_logit": 1.0}, d1 / CKN["E2"])
    rc, log, doc = sw.select(NOW_L + DAY)
    check("dr_default_running_item_2d_then_finished_selects",
          rc0 == 0 and value(rec, 1.0)["status"] == ss.RUNNING_DEFAULT and rc == 0 and doc["winner"]["lambda"] == 1.0
          and doc["n_cut"] == 1, f"{result_line(log0)} {result_line(log)}")
    # a running directory's last iteration before the date may only grow (a recorded value above the derived
    # one is refused)
    sw = lam_sweep("dr_default_shrinks", over={1: dict(rows=course(l1, 60000, val_at=(40000,), miou=0.45))}).build()
    rc0, log0, _ = sw.record(NOW_L, edit=lambda d: value(d, 1.0)["directories"][0].update(last_iter_before_cutoff=50000))
    with open(sw.dir("e2_s42_lambda1_a1") / "e2_telemetry.jsonl", "a", encoding="utf-8") as fh:
        for row in (T(80000, l1 + 80100), V(80000, l1 + 80200, 0.45), END(l1 + 80201, 0.45, start=l1)):
            fh.write(json.dumps(row) + "\n")
    rc, log, _ = sw.select(NOW_L + DAY)
    check("dr_running_default_last_iteration_above_the_derived_is_status_mismatch",
          rc0 == 0 and rc == 2 and code_of(log) == "decision_record_status_mismatch",
          f"{result_line(log0)} {result_line(log)}")
    # R7 (interpretation 14): the on-course repeat still running is waited for as the default is; the record
    # reads "running (on-course repeat, item 2(b))" and accepts the repeat's later finish
    r_full = fin(LAMBDA_C - 5000.0, 0.60, iter_s=0.25, val_s=50.0)

    def oc_sweep(label, *, s_stopped=True, repeat=True, r_part=None):
        sw = Sweep(f"dr_{label}")
        for i, v in enumerate((1, 0.5, 2)):
            sw.add(v, fin(BASE_L + 600 * i, 0.40))
        sw.add(4, oc_stop_rows(), stopped=s_stopped)
        if repeat:
            sw.add(4, r_full[:2] if r_part is None else r_part)
        return sw.build()

    def append_rows(sw, rows):
        with open(sw.dir("e2_s42_lambda4_a2") / "e2_telemetry.jsonl", "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    # fixcode-2: the repeat, launched before the date, was copied while it ran and later shows it finished
    # before the date: the value is finished under PL-9(e)(1) and the record still holds (PL-15)
    r_early = fin(LAMBDA_C - 27000.0, 0.60, iter_s=0.25, val_s=50.0)
    sw = oc_sweep("oc_found_finished", r_part=r_early[:3])
    rc0, log0, _ = sw.record(NOW_L)
    append_rows(sw, r_early[3:])
    rc, log, doc = sw.select(NOW_L)
    check("dr_on_course_repeat_recorded_running_then_found_finished_before_the_date_is_accepted",
          rc0 == 0 and rc == 0 and doc["winner"]["lambda"] == 4.0 and doc["on_course"]["4"] == [],
          f"{result_line(log0)} {result_line(log)}")
    # fixsmoke-4: an input of a recorded evaluation edited: refused whether the exception decides the value
    # (the evaluations are derived again) or the repeat's own ending does (they are recomputed)
    for label, part, rest in (("", None, r_full[2:]), ("_found_finished", r_early[:3], r_early[3:])):
        sw = oc_sweep(f"oc_projection_edited{label}", r_part=part)
        rc0, log0, _ = sw.record(NOW_L, edit=lambda d: value(d, 4.0)["on_course"][-1].update(projection="1"))
        append_rows(sw, rest)
        rc, log, _ = sw.select(NOW_L)
        check(f"dr_on_course_evaluation_edited{label}_is_status_mismatch",
              rc0 == 0 and rc == 2 and code_of(log) == "decision_record_status_mismatch",
              f"{result_line(log0)} {result_line(log)}")
    sw = oc_sweep("oc_entry_without_a_run_id", r_part=r_early[:3])
    rc0, log0, _ = sw.record(NOW_L, edit=lambda d: value(d, 4.0)["on_course"].insert(0, {"run_id": [], "on_course": False}))
    append_rows(sw, r_early[3:])
    rc, log, _ = sw.select(NOW_L)
    check("dr_on_course_evaluation_entry_whose_run_id_is_not_a_string_is_status_mismatch",
          rc0 == 0 and rc == 2 and code_of(log) == "decision_record_status_mismatch",
          f"{result_line(log0)} {result_line(log)}")
    # r2-3: the recorded evaluations equal those derived again entry for entry, also when the repeat's own
    # ending decides the value (a duplicated entry is refused)
    sw = oc_sweep("oc_evaluation_duplicated", r_part=r_early[:3])
    rc0, log0, _ = sw.record(NOW_L, edit=lambda d: value(d, 4.0)["on_course"].append(dict(value(d, 4.0)["on_course"][-1])))
    append_rows(sw, r_early[3:])
    rc, log, _ = sw.select(NOW_L)
    check("dr_on_course_evaluation_duplicated_is_status_mismatch",
          rc0 == 0 and rc == 2 and code_of(log) == "decision_record_status_mismatch",
          f"{result_line(log0)} {result_line(log)}")
    # r2-4: "running (on-course repeat)" only with s and its repeat as the last two launched directories: the
    # writer's cut (the repeat stopped, a later attempt stopped too) edited to running is refused
    seen: dict = {}
    sw = Sweep("dr_oc_forged_running")
    for i, v in enumerate((1, 0.5, 2)):
        sw.add(v, fin(BASE_L + 600 * i, 0.40))
    sw.add(4, oc_stop_rows(), stopped=True)
    sw.add(4, course(LAMBDA_C - 5000.0, 20000, val_at=(4000,), iter_s=0.25, val_s=50.0), stopped=True)
    sw.add(4, course(LAMBDA_C + 2000.0, 3000), stopped=True)
    rc0, log0, _ = sw.build().record(NOW_L, edit=lambda d: (seen.update(value(d, 4.0)), value(d, 4.0).update(
        status=ss.RUNNING_ON_COURSE, reason=None)))
    rc, log, _ = sw.select(NOW_L)
    check("dr_running_on_course_with_a_later_launched_attempt_is_status_mismatch",
          rc0 == 0 and (seen.get("status"), seen.get("reason")) == ("cut", "on_course_repeat_stopped")
          and rc == 2 and code_of(log) == "decision_record_status_mismatch", f"{result_line(log0)} {result_line(log)}")
    # the same with a later attempt that ended after the date (no stopped line) and a record whose hash for that
    # directory is another (a running directory's telemetry is not compared): only s and its repeat as the last
    # two launched directories refuse it
    seen = {}
    sw = Sweep("dr_oc_forged_running_ended_later")
    for i, v in enumerate((1, 0.5, 2)):
        sw.add(v, fin(BASE_L + 600 * i, 0.40))
    sw.add(4, oc_stop_rows(), stopped=True)
    sw.add(4, course(LAMBDA_C - 5000.0, 20000, val_at=(4000,), iter_s=0.25, val_s=50.0), stopped=True)
    sw.add(4, [T(1, LAMBDA_C + 2001.0), ABORT_ROW(1, LAMBDA_C + 2001.5, rule="step1_checks", cause="checks_failed")])

    def edit_ended_later(d):
        seen.update(value(d, 4.0))
        value(d, 4.0).update(status=ss.RUNNING_ON_COURSE, reason=None)
        value(d, 4.0)["directories"][-1]["telemetry_sha256"] = "0" * 64
    rc0, log0, _ = sw.build().record(NOW_L, edit=edit_ended_later)
    rc, log, _ = sw.select(NOW_L)
    check("dr_running_on_course_with_a_later_launched_attempt_that_ended_is_status_mismatch",
          rc0 == 0 and (seen.get("status"), seen.get("reason")) == ("cut", "on_course_repeat_stopped")
          and rc == 2 and code_of(log) == "decision_record_status_mismatch", f"{result_line(log0)} {result_line(log)}")
    # r3-2: a running entry the writer cannot have produced: a reason; a repeat stopped in the record's launch
    # log (its stop report kept or nulled) or ended in a telemetry file unchanged since; a later directory
    sw = oc_sweep("oc_reason")
    rc0, log0, _ = sw.record(NOW_L, edit=lambda d: value(d, 4.0).update(reason="on_course_repeat_stopped"))
    append_rows(sw, r_full[2:])
    rc, log, _ = sw.select(NOW_L)
    check("dr_running_status_with_a_reason_is_status_mismatch",
          rc0 == 0 and rc == 2 and code_of(log) == "decision_record_status_mismatch",
          f"{result_line(log0)} {result_line(log)}")

    def forged_running(label, r_rows, *, r_stopped=False, null_stop_report=False):
        """The writer's record of s (on course) and its repeat r, the value's status edited to running."""
        was: dict = {}
        sw = Sweep(f"dr_{label}")
        for i, v in enumerate((1, 0.5, 2)):
            sw.add(v, fin(BASE_L + 600 * i, 0.40))
        sw.add(4, oc_stop_rows(), stopped=True)
        sw.add(4, r_rows, stopped=r_stopped)

        def edit(d):
            was.update(value(d, 4.0))
            value(d, 4.0).update(status=ss.RUNNING_ON_COURSE, reason=None)
            if null_stop_report:
                value(d, 4.0)["directories"][-1]["stop_report_sha256"] = None
        rc0, log0, _ = sw.build().record(NOW_L, edit=edit)
        rc, log, _ = sw.select(NOW_L)
        return rc0, log0, rc, log, (was.get("status"), was.get("reason"))
    r_stop = course(LAMBDA_C - 5000.0, 4000, val_at=(4000,), iter_s=0.25, val_s=50.0)
    for label, null in (("", False), ("_its_stop_report_nulled", True)):
        rc0, log0, rc, log, was = forged_running(f"oc_forged_stopped{label}", r_stop, r_stopped=True,
                                                 null_stop_report=null)
        check(f"dr_running_on_course_whose_repeat_was_stopped_in_the_records_log{label}_is_status_mismatch",
              rc0 == 0 and was == ("cut", "on_course_repeat_stopped") and rc == 2
              and code_of(log) == "decision_record_status_mismatch", f"{result_line(log0)} {result_line(log)}")
    r_ended = [T(1, LAMBDA_C - 4999), ABORT_ROW(1, LAMBDA_C - 4998.5, rule="step1_checks", cause="checks_failed")]
    rc0, log0, rc, log, was = forged_running("oc_forged_ended", r_ended)
    check("dr_running_on_course_whose_repeat_had_ended_is_status_mismatch",
          rc0 == 0 and was == ("cut", "run_aborted_other:step1_checks/checks_failed") and rc == 2
          and code_of(log) == "decision_record_status_mismatch", f"{result_line(log0)} {result_line(log)}")
    # a later directory: the record of a running repeat, edited to list an attempt the log gained afterwards
    # (a later attempt cuts the repeat, PL-9(e)(3))
    sw = oc_sweep("oc_later_directory")
    t_id = "e2_s42_lambda4_a3"
    t_ref = sw.report(f"reports/am8a/{t_id}.md", f"AM-8a report before {t_id} (synthetic)\n")
    t_dir = {"run_id": t_id, "attempt": 3, "launched": False, "launch_time_utc": None,
             "last_iter_before_cutoff": None, "run_meta_sha256": None, "telemetry_sha256": None,
             "best_json_sha256": None, "best_ckpt_sha256": None, "am8a_report_sha256": t_ref["sha256"],
             "stop_report_sha256": None}
    rc0, log0, _ = sw.record(NOW_L, edit=lambda d: value(d, 4.0)["directories"].append(dict(t_dir)))
    t_nl = {"event": "not_launched", "run_id": t_id, "reason": "the start was refused (synthetic)",
            "evidence": sw.report(f"reports/not_launched/{t_id}.txt", f"{t_id}: start refused\n")}
    sw.commit("a later attempt", log=launch_log_text(sw.lines + sw.events + [
        launch_line(t_id, "E2", 4.0, 4.0, None, 3, sw.pin, report=t_ref), t_nl]))
    rc, log, _ = sw.select(NOW_L)
    check("dr_running_on_course_with_a_later_directory_is_status_mismatch",
          rc0 == 0 and rc == 2 and code_of(log) == "decision_record_status_mismatch",
          f"{result_line(log0)} {result_line(log)}")
    # r3-4: no on-course stop (a failed check, then a stop that was not on course): running is refused, with
    # the writer's evaluations or none
    for label, oc in (("", None), ("_or_evaluations", [])):
        sw = lam_sweep(f"dr_oc_none{label}")
        sw.add(4, fin(BASE_L + 3000.0, 0.38, checks=False))
        sw.add(4, course(BASE_L + 90000.0, 10000, val_at=()), stopped=True)

        def edit(d, oc=oc):
            value(d, 4.0).update(status=ss.RUNNING_ON_COURSE, reason=None)
            if oc is not None:
                value(d, 4.0)["on_course"] = oc
        rc0, log0, _ = sw.build().record(NOW_L, edit=edit)
        rc, log, _ = sw.select(NOW_L)
        check(f"dr_running_on_course_without_an_on_course_stop{label}_is_status_mismatch",
              rc0 == 0 and rc == 2 and code_of(log) == "decision_record_status_mismatch",
              f"{result_line(log0)} {result_line(log)}")
    sw = oc_sweep("oc_running")
    rc0, log0, sha = sw.record(NOW_L)
    rec = json.loads((sw.repo / ss.DECISION_RECORD_REL["lambda_logit"]).read_text(encoding="utf-8")) if rc0 == 0 else {}
    with open(sw.dir("e2_s42_lambda4_a2") / "e2_telemetry.jsonl", "a", encoding="utf-8") as fh:
        for row in r_full[2:]:
            fh.write(json.dumps(row) + "\n")
    rc, log, doc = sw.select(NOW_L)
    rep = (doc or {}).get("on_course_repeat", {}).get("4", {})
    check("dr_on_course_repeat_running_is_recorded_running_and_its_finish_is_accepted",
          rc0 == 0 and value(rec, 4.0)["status"] == ss.RUNNING_ON_COURSE
          and [x["run_id"] for x in value(rec, 4.0)["directories"]] == ["e2_s42_lambda4_a1", "e2_s42_lambda4_a2"]
          and rc == 0 and doc["winner"]["lambda"] == 4.0 and doc["decision_record"]["sha256"] == sha
          and [(c["value"], c["reason"]) for c in doc["cut_am19"]] == [(0.25, "never_launched")]
          and doc["on_course"]["4"][0]["on_course"] is True and rep.get("repeat_val") == [[40000, 0.60], [80000, 0.60]],
          f"{result_line(log0)} {result_line(log)} {rep}")
    # the running repeat is stopped after the record: its stop report appears and the value is cut
    sw = oc_sweep("oc_running_stopped")
    rc0, log0, _ = sw.record(NOW_L)
    sw.events.append({"event": "stopped", "run_id": "e2_s42_lambda4_a2",
                      "report": sw.report("reports/stops/e2_s42_lambda4_a2.md", "stopped after the record (synthetic)\n")})
    sw.commit("the on-course repeat stopped")
    rc, log, doc = sw.select(NOW_L)
    check("dr_on_course_repeat_stopped_after_the_record_is_cut_against_it",
          rc0 == 0 and rc == 0 and sorted((c["value"], c["reason"]) for c in doc["cut_am19"])
          == [(0.25, "never_launched"), (4.0, "on_course_repeat_stopped")], f"{result_line(log0)} {result_line(log)}")
    rc, log, _ = oc_sweep("oc_no_repeat", repeat=False).record(NOW_L)
    check("dr_on_course_stop_without_its_repeat_refuses_the_record",
          rc == 2 and code_of(log) == "decision_record_runs_unended", result_line(log))
    rc, log, _ = oc_sweep("oc_s_unstopped", s_stopped=False).record(NOW_L)
    check("dr_on_course_repeat_running_with_s_lacking_its_stopped_line_refuses_the_record",
          rc == 2 and code_of(log) == "decision_record_runs_unended" and "e2_s42_lambda4_a1" in log, result_line(log))


def test_am19_rule(tmp: Path) -> None:
    """AR (P8), PL-18 and PL-39 case 29: the rule with cut candidates."""
    fin4 = cands([(0.5, 0.44), (1, 0.42), (2, 0.41), (4, 0.40)])
    cut025 = [{"value": 0.25, "reason": "never_launched", "run_id": None}]
    r = outcome(lambda: apply_rule(fin4, LAMBDA, 0.005, "lambda", cut=cut025))
    check("ar_edge_flag_names_am19", isinstance(r, dict) and r["winner"]["value"] == 0.5
          and r["boundary_kind"] == "edge_of_finished_set"
          and r["edge_removed"] == [{"value": 0.25, "rule": "AM-19"}] and r["n_cut"] == 1
          and "lambda=0.25: cut at the decision date (AM-19), reason never_launched; excluded from the selection"
          in r["rule_trace"], str(r["edge_removed"] if isinstance(r, dict) else r))
    check("ar_grid_precondition_includes_cuts", refused(apply_rule, fin4, LAMBDA, 0.005, "lambda") == "partial_input"
          and refused(apply_rule, fin4, LAMBDA, 0.005, "lambda", cut=cut025 + [{"value": 4.0, "reason": "x"}])
          == "partial_input" and isinstance(outcome(lambda: apply_rule(fin4, LAMBDA, 0.005, "lambda", cut=cut025)), dict))
    sole = outcome(lambda: apply_rule(cands([(1, 0.40)]), LAMBDA, 0.005, "lambda", diverged=[div(0.25), div(2)],
                                      cut=[{"value": 0.5, "reason": "still_running"},
                                           {"value": 4.0, "reason": "never_launched"}]))
    check("ar_edges_on_both_sides_name_am7a_and_am19", isinstance(sole, dict) and sole["winner"]["value"] == 1.0
          and sole["edge_removed"] == [{"value": 0.25, "rule": "AM-7a"}, {"value": 0.5, "rule": "AM-19"},
                                       {"value": 2.0, "rule": "AM-7a"}, {"value": 4.0, "rule": "AM-19"}],
          str(sole["edge_removed"] if isinstance(sole, dict) else sole))
    four = outcome(lambda: apply_rule(cands([(1, 0.40)]), LAMBDA, 0.005, "lambda",
                                      cut=[{"value": v, "reason": "never_launched"} for v in (0.25, 0.5, 2, 4)]))
    check("ar_sole_finished_default_with_four_cuts", isinstance(four, dict) and four["winner"]["value"] == 1.0
          and not four["tie"] and four["n_cut"] == 4, "" if isinstance(four, dict) else str(four))
    check("pl18_cut_entry_with_a_best_val_is_partial_input",
          refused(apply_rule, fin4, LAMBDA, 0.005, "lambda", cut=[dict(cut025[0], best_val=0.6)]) == "partial_input")
    check("pl18_diverged_entry_with_a_best_val_is_partial_input",
          refused(apply_rule, fin4, LAMBDA, 0.005, "lambda", diverged=[dict(div(0.25), best_val=0.6)])
          == "partial_input")
    check("pl18_finished_entry_without_status_is_partial_input",
          refused(apply_rule, [{k: v for k, v in c.items() if k != "status"} for c in fin4], LAMBDA, 0.005,
                  "lambda", cut=cut025) == "partial_input")
    b, kind, removed = ss.boundary_kind(4.0, LAMBDA, [4.0], [(0.25, "AM-7a"), (2.0, "AM-19")])
    check("pl39_29_edge_removed_present_under_grid_end", b and kind == "grid_end"
          and removed == [{"value": 0.25, "rule": "AM-7a"}, {"value": 2.0, "rule": "AM-19"}], f"{kind} {removed}")
    sw = five_finished("ar_no_cut").build()
    rc, log, doc = sw.select(NOW_MID_L)
    want = {"schedule", "launch_log", "decision_record", "n_cut", "cut_am19", "stopped_early_unexplained",
            "edge_removed", "directories", "code", "inputs_last_timestamp_utc", "shortfall_disclosure", "clock"}
    check("pl39_29_amendments_unchanged_without_a_cut", rc == 0 and doc["amendments"] == ["AM-2", "AM-7a"]
          and doc["n_cut"] == 0, result_line(log))
    check("sf_new_fields_with_unchanged_format_strings", rc == 0 and doc["format"] == "lambda_selection/1"
          and want <= set(doc) and doc["decision_record"] is None and doc["cut_am19"] == [] and doc["edge_removed"] == []
          and doc["launch_log"]["path"] == ss.LAUNCH_LOG_REL and doc["launch_log"]["lines"] == 10
          and doc["code"]["pin"] == template()[1] and len(doc["code"]["sweep_select_sha256"]) == 64
          and doc["inputs_last_timestamp_utc"] == max(json.loads(ln)["wall_clock_end"] for d in sw.runs()
                                                      for ln in (d / "e2_telemetry.jsonl").read_text().splitlines()
                                                      if "wall_clock_end" in ln)
          and doc["directories"] == {f"{v:g}": [f"e2_s42_lambda{v:g}_a1"] for v in LAMBDA_GRID},
          f"rc={rc} {sorted(set(doc or {}))}")
    check("cl_cli_uses_the_patched_clock", rc == 0 and doc["generated_utc"] == "2026-10-19T04:00:00Z",
          str((doc or {}).get("generated_utc")))
    check("cl_default_clock_is_time_time", abs(ss.now_utc() - __import__("time").time()) < 5)


def test_am19_alpha(tmp: Path) -> None:
    """AL (P8) and PL-39 cases 24 and 26: select_alpha under AM-19 (PL-16, PL-20)."""
    def alpha(label, specs, *, files=None):
        sw = Sweep(f"al_{label}", "alpha_cwd", files=alpha_files() if files is None else files)
        for a, rows, kw in specs:
            sw.add(a, rows, **kw)
        return sw

    three = [(50, fin(BASE_A, 0.428, stage="E3"), {}), (25, fin(BASE_A + 600, 0.430, stage="E3"), {}),
             (100, fin(BASE_A + 1200, 0.431, stage="E3"), {})]
    sw = alpha("ok", three).build()
    rc, log, doc = sw.select(NOW_MID_A)
    check("al_all_finished_before_the_date_selects", rc == 0 and doc["winner"]["alpha"] == 50.0 and doc["tie"]
          and doc["schedule"]["decision_date"] is None and doc["lambda_logit"] == 1.0, f"rc={rc} {result_line(log)}")
    sw = alpha("24a", three).build()
    (sw.repo / ss.BAND_REL).write_text(json.dumps({"e1_best_val": E1_VALS, "s": 0.0}), encoding="utf-8")
    rc, log, _ = sw.select(NOW_MID_A)
    check("pl39_24a_band_file_uncommitted_is_refused", rc == 2 and code_of(log) == "band_uncommitted", result_line(log))
    sw = alpha("24b", three).build()
    s = __import__("statistics").stdev(E1_VALS.values())
    sw.commit("another band", {ss.BAND_REL: json.dumps({"e1_best_val": E1_VALS, "s": s, "band": 0.005})})
    rc, log, _ = sw.select(NOW_MID_A)
    check("pl39_24b_band_blob_differing_from_the_alpha_defaults_records_commit_is_refused",
          rc == 2 and code_of(log) == "band_changed_after_launch", result_line(log))
    sw = alpha("24c", three).build()
    os.chmod(sw.repo / ss.BAND_REL, 0o755)
    rc, log, doc = sw.select(NOW_MID_A)
    check("pl39_24c_committed_file_with_a_changed_mode_bit_and_equal_bytes_is_accepted", rc == 0 and doc is not None,
          result_line(log))
    sw = alpha("cut_lambda", [(50, fin(BASE_A, 0.428, stage="E3"), {}),
                              (25, course(BASE_A + 600, 3000, val_at=(), stage="E3"), dict(lam=0.5, stopped=True)),
                              (100, fin(BASE_A + 1200, 0.431, stage="E3"), {})]).build()
    rc, log, _ = sw.select(NOW_A)
    check("al_cut_run_at_another_lambda_is_lambda_mismatch", rc == 2 and code_of(log) == "lambda_mismatch",
          result_line(log))
    running = course(BASE_A, 60000, val_at=(40000,), stage="E3", iter_s=14.0)
    sw = alpha("26a", [(50, running, {})]).build()
    rc, log, _ = sw.select(NOW_A)
    check("al_alpha_cut_without_a_record_is_a_shortfall", rc == 3 and "AM-19 item 2(g)" in log, result_line(log))
    rc0, log0, sha = sw.record(NOW_A)
    rec = json.loads((sw.repo / ss.DECISION_RECORD_REL["alpha_cwd"]).read_text(encoding="utf-8"))
    rc, log, doc = sw.select(NOW_A)
    check("pl39_26a_exit_5_with_the_default_still_running",
          rc0 == 0 and rc == 5 and doc is None and rec["alpha_sweep_cut"] is True and rec["never_launched"] == [25.0, 100.0]
          and rec["alpha_cutoff_basis"]["basis"] == "a" and f"decision record {sha}" in result_line(log)
          and result_line(log).startswith("RESULT: CUT (AM-16 item 2; AM-19 item 2(g)) alpha_cwd = 50"),
          f"{result_line(log0)} {result_line(log)}")
    sw.commit("an alpha selection beside the cut", {ss.ALPHA_SELECTION_REL: json.dumps({"format": "alpha_selection/1"})})
    rc, log, _ = sw.select(NOW_A)
    check("pl39_26c_alpha_selection_beside_a_cut_record_is_alpha_cut_conflict",
          rc == 2 and code_of(log) == "alpha_cut_conflict", result_line(log))
    diverged = course(BASE_A, 12000, val_at=(4000, 8000), stage="E3") + [ABORT_ROW(12000, BASE_A + 12200.5)]
    sw = alpha("26b_no_record", [(50, diverged, {})]).build()
    rc, log, _ = sw.select(NOW_A)
    check("al_diverged_default_without_a_record_is_default_candidate_diverged",
          rc == 2 and code_of(log) == "default_candidate_diverged", result_line(log))
    # PL-20(b): a committed alpha-cut record that would give exit 5, then the default diverges
    part = course(BASE_A, 40000, val_at=(40000,), stage="E3", iter_s=14.0)
    sw = alpha("26b", [(50, part, {})]).build()
    rc0, log0, _ = sw.record(NOW_A)
    rc1, log1, _ = sw.select(NOW_A)
    with open(sw.dir("e3_s42_alpha50_a1") / "e3_telemetry.jsonl", "a", encoding="utf-8") as fh:
        for row in (T(45000, BASE_A + 45000 * 14.0 + 100), ABORT_ROW(45000, BASE_A + 45000 * 14.0 + 100.5)):
            fh.write(json.dumps(row) + "\n")
    rc, log, _ = sw.select(NOW_A)
    check("pl39_26b_alpha_cut_refused_when_the_default_diverged",
          rc0 == 0 and rc1 == 5 and rc == 2 and code_of(log) == "default_candidate_diverged",
          f"{result_line(log0)} {result_line(log1)} {result_line(log)}")
    # alpha-4: the record is written from a copy of the running default's telemetry; a later copy adds rows
    # before the date (and after it): the record still holds, and exit 5 stays
    sw = alpha("default_copy", [(50, part, {})]).build()
    rc0, log0, _ = sw.record(NOW_A)
    rc1, log1, _ = sw.select(NOW_A)
    with open(sw.dir("e3_s42_alpha50_a1") / "e3_telemetry.jsonl", "a", encoding="utf-8") as fh:
        for it in (55000, 60000):
            fh.write(json.dumps(T(it, BASE_A + it * 14.0 + 100, 14.0)) + "\n")
    rc, log, _ = sw.select(NOW_A)
    check("al_default_running_record_accepts_rows_added_before_the_date",
          rc0 == 0 and rc1 == 5 and rc == 5 and BASE_A + 55000 * 14.0 + 100 < ALPHA_C < BASE_A + 60000 * 14.0,
          f"{result_line(log0)} {result_line(log1)} {result_line(log)}")
    # PL-15: the record's alpha basis is compared with the derived one
    sw = alpha("basis_edited", [(50, part, {})]).build()
    rc0, log0, _ = sw.record(NOW_A, edit=lambda d: d["alpha_cutoff_basis"].update(
        T_lo_utc=d["alpha_cutoff_basis"]["T_lo_utc"] + 1))
    rc, log, _ = sw.select(NOW_A)
    check("al_record_with_another_alpha_basis_is_decision_record_schedule_mismatch",
          rc0 == 0 and rc == 2 and code_of(log) == "decision_record_schedule_mismatch",
          f"{result_line(log0)} {result_line(log)}")
    sw = alpha("cut_mismatch", [(50, fin(BASE_A, 0.428, stage="E3"), {}),
                                (25, course(BASE_A + 600, 3000, val_at=(), stage="E3"), dict(stopped=True))]).build()
    rc0, log0, _ = sw.record(NOW_A, edit=lambda d: d.update(alpha_sweep_cut=True))
    rc, log, _ = sw.select(NOW_A)
    check("al_record_cutting_the_sweep_with_a_non_default_launched_is_status_mismatch",
          rc0 == 0 and rc == 2 and code_of(log) == "decision_record_status_mismatch",
          f"{result_line(log0)} {result_line(log)}")


def test_am19_alpha_cutoff(tmp: Path) -> None:
    """PL-17 and PL-39 case 25: the alpha decision date (bases (a), (b), (c); R10)."""
    repo = new_repo()
    pin = template()[1]
    lsel = json.dumps({"format": "lambda_selection/1", "winner": {"lambda": 1.0}, "inputs_last_timestamp_utc": 0.0})
    added = commit_files(repo, {ss.LAMBDA_SELECTION_REL: lsel}, "lambda selection")
    src = ss.HeadSource(repo)

    def entry(after, ts, eid):
        return {"id": eid, "node_id": f"PSH_{eid}", "before": pin, "after": after, "ref": "refs/heads/master",
                "timestamp": ts, "activity_type": "push", "actor": {"login": "k2-smoke", "id": 1, "type": "User"}}

    def cut(t_in, launch, evidence=None, *, source=src, records=added):
        if evidence is not None:
            commit_files(source.root, {ss.PUSH_EVIDENCE_REL: json.dumps(evidence)}, "push evidence")
        atts = [] if launch is None else [SimpleNamespace(launched={"records_commit": records,
                                                                    "launch_time_utc": launch}, launch_ts=launch)]
        try:
            return ss.alpha_cutoff(source, "2026-10-22", lambda_doc={"inputs_last_timestamp_utc": t_in},
                                   default_attempts=atts)
        except SelectionRefused as e:
            return e.code

    got = cut(utc("2026-10-12T00:00:00Z"), utc("2026-10-13T00:00:00Z"))
    check("pl39_25a_basis_a_keeps_the_calendar_date", isinstance(got, dict) and got["basis"]["basis"] == "a"
          and got["decision_date"] == "2026-10-22" and got["cutoff_utc"] == ALPHA_C, str(got))
    got = cut(utc("2026-10-20T01:00:00Z"), utc("2026-10-20T05:00:00Z"))
    check("pl39_25b_basis_b_moves_the_date_to_D_plus_3", isinstance(got, dict) and got["basis"]["basis"] == "b"
          and got["basis"]["D"] == "2026-10-20" and got["decision_date"] == "2026-10-23", str(got))
    old = entry(pin, "2026-10-18T00:00:00Z", 1)
    t_in, launch = utc("2026-10-19T15:00:00Z"), utc("2026-10-21T02:00:00Z")
    got = cut(t_in, launch)
    check("pl39_25_basis_c_without_the_evidence_is_lambda_push_evidence_missing",
          got == "lambda_push_evidence_missing", str(got))
    got = cut(t_in, launch, [old, entry(added, "2026-10-20T03:00:00Z", 2)])
    check("pl39_25c_basis_c_dates_the_push_from_the_activity_record",
          isinstance(got, dict) and got["basis"]["basis"] == "c" and got["basis"]["D"] == "2026-10-20"
          and got["decision_date"] == "2026-10-23" and got["basis"]["evidence"]["commit"] == added, str(got))
    got = cut(t_in, launch, [old, entry(added, "2026-10-21T05:00:00Z", 3)])
    check("pl39_25_evidence_outside_the_bounds_is_refused", got == "lambda_push_evidence_mismatch", str(got))
    got = cut(t_in, launch, [old, entry(added, "2026-10-20T17:00:00Z", 5), entry(added, "2026-10-20T03:00:00Z", 4)])
    check("pl39_25_the_earliest_of_two_entries_wins", isinstance(got, dict) and got["basis"]["D"] == "2026-10-20"
          and got["decision_date"] == "2026-10-23" and got["basis"]["evidence"]["entry_id"] == 4, str(got))
    got = cut(t_in, launch, [entry(added, "2026-10-20T03:00:00Z", 6)])
    check("pl39_25_response_not_reaching_back_to_t_lo_is_refused", got == "lambda_push_evidence_mismatch", str(got))
    got = cut(utc("2026-10-20T15:30:00Z"), utc("2026-10-20T16:30:00Z"), [old, entry(added, "2026-10-20T16:10:00Z", 7)])
    check("al_day_boundary_is_the_manila_day_not_the_utc_day", isinstance(got, dict) and got["basis"]["basis"] == "c"
          and got["basis"]["D"] == "2026-10-21" and got["decision_date"] == "2026-10-24", str(got))
    got = cut(t_in, None, [old, entry(added, "2026-10-20T03:00:00Z", 8)])
    check("al_no_alpha_default_launched_uses_the_evidence_from_t_lo", isinstance(got, dict)
          and got["basis"]["basis"] == "c" and got["basis"]["T_hi_utc"] is None and got["decision_date"] == "2026-10-23",
          str(got))
    repo2 = new_repo()
    first = commit_files(repo2, {ss.LAMBDA_SELECTION_REL: lsel}, "lambda selection")
    second = commit_files(repo2, {ss.LAMBDA_SELECTION_REL: lsel.replace("1.0", "0.5")}, "changed after it was added")
    src2 = ss.HeadSource(repo2)
    got = cut(t_in, launch, [old, entry(second, "2026-10-20T03:00:00Z", 9)], source=src2, records=second)
    check("pl39_25_a_commit_that_did_not_add_the_file_is_refused", got == "lambda_push_evidence_mismatch", str(got))
    got = cut(t_in, launch, source=src2, records=first)
    check("al_lambda_selection_changed_since_the_alpha_defaults_records_commit_is_refused",
          got == "lambda_selection_changed", str(got))
    # PL-17(c): an entry whose `after` this repository cannot resolve refuses inside [T_lo, T_hi] only
    got = cut(t_in, launch, [old, entry("f" * 40, "2026-10-20T02:00:00Z", 10), entry(added, "2026-10-20T03:00:00Z", 11)])
    check("pl17_unresolvable_after_inside_the_bounds_is_refused", got == "lambda_push_evidence_mismatch", str(got))
    got = cut(t_in, launch, [old, entry("f" * 40, "2026-10-17T00:00:00Z", 12), entry(added, "2026-10-20T03:00:00Z", 13)])
    check("pl17_unresolvable_after_outside_the_bounds_is_ignored", isinstance(got, dict)
          and got["basis"]["basis"] == "c" and got["basis"]["D"] == "2026-10-20", str(got))
    # alpha-3 (PL-17 "the gate uses the same function", PL-28): through the gate's RecordsSource, whose
    # checkout is the pin and whose records come from records commit H, the function gives the selection's
    # bases; nothing is read at the pin's HEAD
    repo3 = new_repo()
    git(repo3, "checkout", "-q", "-b", "records")
    h3 = commit_files(repo3, {ss.LAMBDA_SELECTION_REL: lsel}, "lambda selection (records)")
    ev = [entry(pin, "2026-10-18T00:00:00Z", 20), entry(h3, "2026-10-20T03:00:00Z", 21)]
    h4 = commit_files(repo3, {ss.PUSH_EVIDENCE_REL: json.dumps(ev)}, "push evidence (records)")
    git(repo3, "checkout", "-q", "master")
    folder = scratch_dir("k2_records")
    (folder / "reports" / "derived").mkdir(parents=True)
    (folder / ss.LAMBDA_SELECTION_REL).write_text(lsel, encoding="utf-8")
    (folder / ss.PUSH_EVIDENCE_REL).write_text(json.dumps(ev), encoding="utf-8")
    rs4 = ss.RecordsSource(folder, h4, root=repo3)
    got_a = cut(utc("2026-10-12T00:00:00Z"), utc("2026-10-13T00:00:00Z"), source=rs4, records=h4)
    got_c = cut(t_in, launch, source=rs4, records=h4)
    check("pl17_the_gates_records_source_gives_the_selections_bases",
          ss.HeadSource(repo3).head() == pin and isinstance(got_a, dict) and got_a["basis"]["basis"] == "a"
          and got_a["decision_date"] == "2026-10-22" and isinstance(got_c, dict) and got_c["basis"]["basis"] == "c"
          and got_c["basis"]["D"] == "2026-10-20" and got_c["basis"]["evidence"]["commit"] == h3,
          f"{got_a} | {got_c}")


CODE_PARAMS = ("code", "missing", "uncommitted", "mismatch")


def raised_codes() -> tuple[dict, list]:
    """Every refusal code the three selection modules can raise: SelectionRefused's literal first argument,
    a literal passed as code=, missing=, uncommitted= or mismatch=, and those parameters' literal defaults.
    A SelectionRefused whose first argument is neither a literal nor one of those parameters is listed."""
    found, opaque = {}, []
    for path in (Path(ss.__file__), Path(sl.__file__), Path(sa.__file__)):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                if name == "SelectionRefused":
                    arg = node.args[0] if node.args else None
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        found.setdefault(arg.value, path.name)
                    elif not (isinstance(arg, ast.Name) and arg.id in CODE_PARAMS):
                        opaque.append(f"{path.name}:{node.lineno}")
                for kw in node.keywords:
                    if kw.arg in CODE_PARAMS and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        found.setdefault(kw.value.value, path.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                pos = a.posonlyargs + a.args
                pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + list(zip(a.kwonlyargs, a.kw_defaults))
                for arg, default in pairs:
                    if arg.arg in CODE_PARAMS and isinstance(default, ast.Constant) and isinstance(default.value, str):
                        found.setdefault(default.value, path.name)
    return found, opaque


def test_am19_classes(tmp: Path) -> None:
    """PL-5 and PL-39 case 27: every code the three modules can raise sits in exactly one class tuple."""
    found, opaque = raised_codes()
    classes = (ss.SHORTFALL_CODES, ss.CANDIDATE_LEVEL_CODES, ss.SWEEP_LEVEL_CODES)
    off = sorted(c for c in found if sum(c in t for t in classes) != 1)
    twice = sorted({c for t in classes for c in t if sum(c in u for u in classes) > 1})
    never = sorted(c for t in classes for c in t if c not in found)
    check("pl39_27_every_code_sits_in_exactly_one_class", not off and not twice and not opaque and len(found) >= 60,
          f"unclassified {off}; in two classes {twice}; opaque {opaque}; {len(found)} found; never raised {never}")
    check("pl5_candidate_level_and_shortfall_classes_as_listed",
          set(ss.CANDIDATE_LEVEL_CODES) == {"run_meta_rows", "run_meta_mismatch", "offgrid", "semantics",
                                           "recipe_mismatch", "unreadable_input", "telemetry_clock_format",
                                           "telemetry_clock_straddle", "run_end_format", "abort_record_invalid",
                                           "run_aborted_other", "run_checks_failed"}
          and set(ss.SHORTFALL_CODES) == {"candidate_missing", "candidate_incomplete", "run_unfinished",
                                         "shortfall_am19_item2"})


def test_am19_records_source(tmp: Path) -> None:
    """PL-16 and PL-28: the accessor's second source, a records folder verified by blob id (the gate's)."""
    sw = five_finished("rs").build()
    h = sw.records
    data = ss.HeadSource(sw.repo).show(h, ss.LAUNCH_LOG_REL)
    folder = scratch_dir("k2_records")
    (folder / "reports").mkdir()
    (folder / ss.LAUNCH_LOG_REL).write_bytes(data)
    rs = ss.RecordsSource(folder, h, root=sw.repo)
    rel, got = rs.read(ss.LAUNCH_LOG_REL, missing="launch_log_missing", uncommitted="launch_log_uncommitted")
    _, rules = rs.read(ss.RULES_REL, missing="rules_missing", uncommitted="rules_uncommitted")
    check("pl16_records_source_reads_a_records_file_by_blob_id_and_code_at_head",
          rel == ss.LAUNCH_LOG_REL and got == data and rules == (sw.repo / ss.RULES_REL).read_bytes())
    (folder / ss.LAUNCH_LOG_REL).write_bytes(data + b"\n")
    check("pl16_records_file_with_another_blob_id_is_records_mismatch",
          refused(rs.read, ss.LAUNCH_LOG_REL, missing="launch_log_missing", uncommitted="x") == "records_mismatch")
    check("pl16_records_file_absent_from_the_folder_but_at_the_commit_is_records_mismatch",
          refused(rs.read_at_head, ss.DECISION_LOG_REL, missing="decision_log_missing") == "records_mismatch")
    check("pl16_records_file_absent_at_the_commit_is_its_missing_code",
          refused(rs.read, ss.PUSH_EVIDENCE_REL, missing="lambda_push_evidence_missing",
                  uncommitted="lambda_push_evidence_mismatch") == "lambda_push_evidence_missing")
    (folder / ss.LAUNCH_LOG_REL).write_bytes(data)
    via = outcome(lambda: ss.require_committed(ss.LAUNCH_LOG_REL, src=rs, missing="launch_log_missing",
                                               uncommitted="launch_log_uncommitted"))
    saved_root, ss.GIT_ROOT = ss.GIT_ROOT, sw.repo         # the default source, at the scratch repository (PL-22)
    try:
        head = outcome(lambda: ss.require_committed(ss.RULES_REL, missing="rules_missing",
                                                    uncommitted="rules_uncommitted"))
    finally:
        ss.GIT_ROOT = saved_root
    check("pl16_require_committed_reads_through_either_source",
          via == (ss.LAUNCH_LOG_REL, data) and isinstance(head, tuple) and head[0] == ss.RULES_REL, f"{via!r:.80} {head!r:.80}")
    # inputs-6: a report the launch log names is a records file wherever it lies (PL-28); the pin's checkout
    # does not hold one committed after it
    note = b"an AM-8a report outside reports/ (synthetic)\n"
    h2 = commit_files(sw.repo, {"notes/am8a/x.md": note}, "a report outside reports/")
    (folder / "notes" / "am8a").mkdir(parents=True)
    (folder / "notes" / "am8a" / "x.md").write_bytes(note)
    (sw.repo / "notes" / "am8a" / "x.md").unlink()
    rs2 = ss.RecordsSource(folder, h2, root=sw.repo)
    got = outcome(lambda: rs2.read("notes/am8a/x.md", missing="repeat_report_missing",
                                   uncommitted="repeat_report_mismatch", record=True))
    check("pl28_a_report_the_log_names_is_read_from_the_records_folder_wherever_it_lies",
          got == ("notes/am8a/x.md", note) and refused(rs2.read, "notes/am8a/x.md", missing="repeat_report_missing",
                                                       uncommitted="repeat_report_mismatch") == "repeat_report_missing"
          and refused(ss._check_report, rs2, {"path": "notes/am8a/x.md", "sha256": hashlib.sha256(note).hexdigest()},
                      missing="repeat_report_missing", mismatch="repeat_report_mismatch") is None,
          str(got)[:120])
    # fixsmoke-3: alpha_inputs and load_record through the gate's source read the records at H, with the
    # checkout at the pin (which holds none of them)
    repo4 = new_repo()
    git(repo4, "checkout", "-q", "-b", "records")
    afiles = alpha_files()
    h5 = commit_files(repo4, afiles, "band and lambda selection (records)")
    cut_rec = json.dumps({"alpha_sweep_cut": True})
    h6 = commit_files(repo4, {ss.ALPHA_SELECTION_REL: json.dumps({"format": "alpha_selection/1"}),
                              ss.DECISION_RECORD_REL["alpha_cwd"]: cut_rec}, "an alpha selection beside a cut record")
    git(repo4, "checkout", "-q", "master")
    folder4 = scratch_dir("k2_records")
    (folder4 / "reports" / "derived").mkdir(parents=True)
    for rel, text in afiles.items():
        (folder4 / rel).write_text(text, encoding="utf-8")
    atts = {float(ALPHA["default_candidate"]): [SimpleNamespace(launched={"records_commit": h5})]}
    ok5 = outcome(lambda: ss.alpha_inputs(ss.RecordsSource(folder4, h5, root=repo4), ALPHA, atts))
    (folder4 / ss.ALPHA_SELECTION_REL).write_text(json.dumps({"format": "alpha_selection/1"}), encoding="utf-8")
    (folder4 / ss.DECISION_RECORD_REL["alpha_cwd"]).write_text(cut_rec, encoding="utf-8")
    rs6 = ss.RecordsSource(folder4, h6, root=repo4)
    conflict = outcome(lambda: ss.alpha_inputs(rs6, ALPHA, atts))
    loaded = outcome(lambda: ss.load_record({"src": rs6, "key": "alpha_cwd", "statuses": {}, "sweep": ALPHA}))
    check("pl28_alpha_inputs_and_load_record_read_the_records_commit_through_the_gates_source",
          isinstance(ok5, dict) and ok5["lambda"] == 1.0 and conflict == "alpha_cut_conflict"
          and loaded == "decision_record_format" and not ss.HeadSource(repo4).has(ss.DECISION_RECORD_REL["alpha_cwd"]),
          f"{ok5!r:.120} {conflict!r} {loaded!r}")


UNIT_SECTIONS = ("test_alpha_rule", "test_lambda_rule", "test_am7a_rule")
TMP_SECTIONS = ("test_alpha_cli", "test_lambda_cli", "test_rules_file", "test_am7a_lambda_cli", "test_am7a_alpha_cli",
                "test_am7_records", "test_ckpt_guard")
AM19_SECTIONS = ("test_am19_rules_time", "test_am19_schedule", "test_am19_launch_log", "test_am19_status",
                 "test_am19_on_course", "test_am19_repeats", "test_am19_record", "test_am19_rule", "test_am19_alpha",
                 "test_am19_alpha_cutoff", "test_am19_classes", "test_am19_records_source")


def main(argv=None) -> int:
    """All sections, or only those named with --section (repeatable; mutation runs). A section that raises
    fails as <section>_completed instead of stopping the smoke."""
    argv = sys.argv[1:] if argv is None else argv
    only = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--section"]
    print("=" * 78)
    print("SWEEP SELECTION SMOKE — synthetic inputs only; nothing written in the repository")
    print("=" * 78)
    tmp = scratch_dir("k2_select")
    try:
        for name in UNIT_SECTIONS + TMP_SECTIONS + AM19_SECTIONS:
            if only and name not in only:
                continue
            fn = globals()[name]
            try:
                fn() if name in UNIT_SECTIONS else fn(tmp)
            except Exception as e:  # noqa: BLE001 - reported as a failed check
                import traceback
                check(f"{name}_completed", False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}")
    finally:
        for p in _SCRATCH_DIRS:          # only directories this process created (scratch_dir)
            shutil.rmtree(p, ignore_errors=True)
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:56}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
