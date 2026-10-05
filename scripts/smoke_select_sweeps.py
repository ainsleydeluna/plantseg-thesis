#!/usr/bin/env python3
"""Smoke for the sweep selection scripts (lane L-AM16-ALPHA d2; the lambda rule of AM-2 / DL-06; AM-7a
exclusion of diverged candidates, lane L-KD-HARDEN).

Synthetic inputs only: rule units on candidate records, and end-to-end runs of scripts/select_alpha.py
and scripts/select_lambda.py on synthetic run directories (best.json, run_meta, telemetry and a tiny
checkpoint) under a temp dir. No dataset, no real checkpoint, no GPU; nothing is written in the repo.

Checks (docs/lane_specs/part2.md lane 2 (d) d2, and the cross-lane selection contract):
  alpha  clear winner; tie containing 50 -> 50; tie without 50 -> the smallest alpha; band from s
         above and below the 0.5 pp floor; the band edge is inclusive (decimal); boundary 25/100
         flagged; missing band -> refuse; two candidates -> refuse (lane 2 STOP, exit 3).
  lambda clear winner; tie -> the smallest lambda; boundary 0.25/4 flagged; fewer than five finished
         candidates -> refusal naming AM-17 item 9 (exit 3), never a selection.
  inputs a run with no run_end record (it died during its final validation) or a torn last
         telemetry line is a shortfall (exit 3); refused (exit 2, never a traceback): a torn middle
         telemetry line, a run_end that disagrees with best.json, a run whose own checks failed, a
         dry run, another seed, --allow-offgrid, a Logit-KD semantics override, a second run_meta
         row, a checkpoint whose best value disagrees with best.json, a missing or unreadable
         checkpoint, a path under a 'test' directory, a band file that is not an object, a missing
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
         model-config hash mismatch -> recipe_mismatch, not default_candidate_diverged.
  10b    the recipe of record: grad_clip_norm 1.0, batch 8, VAL every 2000 or capped, no ImageNet init,
         a float horizon, differing num_workers or teacher checkpoint, config or model-config hashes, or
         no teacher provenance -> recipe_mismatch naming the field.
  cg     L-CKPT-GUARD: a teacher config_sha256 or model_cfg_sha256 absent, None, empty or not a string
         -> recipe_mismatch; select_alpha's five K8-2(a) cases (no run loaded or three unfinished:
         shortfall, exit 3; one wrong-lambda run beside two missing: lambda_mismatch; the default
         diverged at the selection's lambda: default_candidate_diverged; at another lambda:
         lambda_mismatch); the AM-7 comment names R8-1 (K8-2(b)); an AM-7 (a) loss record whose
         grad_norm is not null in its detail (1.7, "nan", 0, False) or its train row (1.7, 0, False,
         key absent) -> abort_record_invalid, end to end and unit (K8-2(c)).
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import scripts.select_alpha as sa  # noqa: E402
import scripts.select_lambda as sl  # noqa: E402
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


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def cands(pairs) -> list[dict]:
    return [{"value": float(v), "run_id": f"run_{v:g}", "best_val": b, "ckpt_sha256": f"sha{v:g}"}
            for v, b in pairs]


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
                 input_finite=True, teacher_finite=True) -> dict:
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
            "teacher_finite": teacher_finite, "params_finite": True, "n_val": it // 4000, "wall_clock": 1.0}


def make_run(root: Path, stage: str, value: float, best_val: float, *, name: str | None = None,
             finished=True, mode="real", seed=42, offgrid=False, rows=1, ckpt_best=None,
             with_ckpt=True, lam=1.0, override=False, run_end=True, checks_passed=True,
             end_best=None, torn="", corrupt_ckpt=False, abort=None, abort_after_end=False,
             record_iter=None, nonfinite=False, with_best=True, recipe=None, val_nonfinite=None,
             row_grad_norm=AUTO) -> Path:
    """A synthetic run directory. `abort` (run_abort record kwargs) ends the telemetry at the abort
    iteration with a run_abort row and no run_end; `abort_after_end` appends it after a run_end instead;
    `nonfinite` (True for the loss, or a {key: tag} map) flags the last train row as train_distill does
    (null + nonfinite map); an abort at iteration 1 has that single train row; `val_nonfinite`
    (a val-row key) adds a val row at the last iteration with that value non-finite; `recipe` overrides
    run_meta's recipe fields. A train row whose loss is non-finite holds grad_norm null, as the trainer
    writes it (the loss stop precedes backward); `row_grad_norm` sets another value, or None to drop the
    key (K8-2(c))."""
    sk = stage.lower()
    d = root / (name or (f"{sk}_s42_alpha{value:g}" if stage == "E3" else f"{sk}_s42_lambda{value:g}"))
    d.mkdir(parents=True)
    terms = {"logit_kd": True, "cwd_feat": stage == "E3", "cwd_logit": stage == "E3"}
    meta = {"event": "run_meta", "stage": stage, "mode": mode, "seed": seed, "terms": terms,
            "projection_params": 51200 if stage == "E3" else 0, "lambda_logit": lam if stage == "E3" else value,
            "logit_kd_semantics": LOGIT_KD_SEMANTICS, "logit_kd_semantics_declared": None,
            "logit_kd_semantics_override_used": override, "max_iters": 80000}
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
    last_row = {"event": "train", "iter": last, "loss": 1.0, **{k: None for k in named}}
    if "loss" in named:
        last_row.setdefault("grad_norm", None)
    if row_grad_norm is None:
        last_row.pop("grad_norm", None)
    elif row_grad_norm is not AUTO:
        last_row["grad_norm"] = row_grad_norm
    if named:
        last_row["nonfinite"] = named
    lines = ([json.dumps({"event": "train", "iter": last - 1, "loss": 1.0})] if last > 1 else []) \
        + [json.dumps(last_row)]
    if val_nonfinite is not None:
        val = {"event": "val", "iter": last, "all_class_miou": best_val,
               "disease_only_miou_PROVISIONAL": best_val}
        lines.append(json.dumps({**val, val_nonfinite: None, "nonfinite": {val_nonfinite: "nan"}}))
    if torn == "middle":
        lines.insert(1, '{"event": "train", "iter": 7')
    if finished and run_end and (abort is None or abort_after_end):
        lines.append(json.dumps({"event": "run_end", "iter": 80000,
                                 "best_val_miou_all_class": best_val if end_best is None else end_best,
                                 "best_ckpt": ck_name, "checks_passed": checks_passed}))
    if abort is not None:
        kw = {k: v for k, v in abort.items() if k != "it"}
        lines.append(json.dumps(abort_record(last if record_iter is None else record_iter, **kw)))
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


def run_cli(module, argv) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = module.main(argv)
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
    check("alpha_cli_two_runs_refused_lane2_stop", rc == 3 and "Lane 2 STOP" in log
          and not (tmp / "a_two.json").exists())
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
             "checkpoint_missing": (dict(with_ckpt=False), 2),
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
    rc, log = run_cli(sl, ["--runs", *map(str, runs[:4]), "--out", str(tmp / "l_four.json")])
    check("lambda_cli_four_runs_refused_naming_am17_item9", rc == 3 and "AM-17 item 9" in log
          and not (tmp / "l_four.json").exists(), log.strip()[-160:])
    r2 = tmp / "lambda_unfinished"
    runs2 = [make_run(r2, "E2", v, 0.4, finished=(v != 4)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs2), "--out", str(tmp / "l_unf.json")])
    check("lambda_cli_unfinished_refused_naming_am17_item9", rc == 3 and "AM-17 item 9" in log
          and "run_unfinished" in log and not (tmp / "l_unf.json").exists())
    r3 = tmp / "lambda_override"
    runs3 = [make_run(r3, "E2", v, 0.4, override=(v == 1)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs3), "--out", str(tmp / "l_ovr.json")])
    check("lambda_cli_semantics_override_refused", rc == 2 and "semantics" in log)
    r4 = tmp / "lambda_died_in_final_val"
    runs4 = [make_run(r4, "E2", v, 0.4, run_end=(v != 2)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs4), "--out", str(tmp / "l_end.json")])
    check("lambda_cli_no_run_end_refused_naming_am17_item9", rc == 3 and "AM-17 item 9" in log
          and not (tmp / "l_end.json").exists(), log.strip()[-160:])
    r5 = tmp / "lambda_torn_last"
    runs5 = [make_run(r5, "E2", v, 0.4, run_end=(v != 1), torn=("last" if v == 1 else "")) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs5), "--out", str(tmp / "l_torn.json")])
    check("lambda_cli_torn_last_line_is_shortfall_naming_am17_item9", rc == 3 and "AM-17 item 9" in log
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
    r = apply_rule(cands([(4, 0.40)]), LAMBDA, 0.005, "lambda", diverged=[div(v) for v in (0.25, 0.5, 1, 2)])
    check("am7a_rule_grid_end_wins_over_edge", r["boundary"] and r["boundary_kind"] == "grid_end")
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
          and "AM-17 item 9" in log
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
        check(f"item10b_recipe_{label}_refused", rc == 2 and doc is None and "recipe_mismatch" in log
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
        check(f"q3_diverged_default_with_{label}_mismatch_is_recipe_mismatch", rc == 2 and doc is None
              and "recipe_mismatch" in log and f"differ in {field}" in log
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
                  and "recipe_mismatch" in log and "lacks a comparable" in log and "Traceback" not in log,
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

    rc, log, out = alpha_runs("no_run_loaded", {25: None, 50: None, 100: None})
    check("k82a_no_run_loaded_is_shortfall_exit3", rc == 3 and "shortfall_lane2_stop" in log
          and "lambda_mismatch" not in log and not out.exists(), f"rc={rc} {log.strip()[-160:]}")
    unfinished = dict(finished=False)
    rc, log, out = alpha_runs("three_unfinished", {25: unfinished, 50: unfinished, 100: unfinished})
    check("k82a_three_unfinished_is_shortfall_exit3", rc == 3 and "shortfall_lane2_stop" in log
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


def main() -> int:
    print("=" * 78)
    print("SWEEP SELECTION SMOKE — synthetic inputs only; nothing written in the repository")
    print("=" * 78)
    tmp = Path(tempfile.mkdtemp(prefix="k1_select_"))
    for fn in (test_alpha_rule, test_lambda_rule, test_am7a_rule):
        fn()
    for fn in (test_alpha_cli, test_lambda_cli, test_rules_file, test_am7a_lambda_cli, test_am7a_alpha_cli,
               test_am7_records, test_ckpt_guard):
        fn(tmp)
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:56}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
