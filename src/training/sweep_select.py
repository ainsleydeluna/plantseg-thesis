"""Sweep selection for lambda_logit (AM-2 / DL-06) and alpha_cwd (AM-16 item 2 / DL-27), with AM-7a.

Shared by scripts/select_lambda.py and scripts/select_alpha.py (lane L-AM16-ALPHA; hardened in lane
L-KD-HARDEN). The cross-lane selection contract (docs/lane_specs/part2.md): refuse partial inputs; read
the band and rule from a committed JSON (configs/sweep_rules.json; alpha's band from
reports/derived/dl27_band.json); write a JSON with a `rule_trace`; never read TEST; the winner is always
an existing run.

A candidate is one REAL training run of the sweep, given as its checkpoint directory. Only these files
are opened:
  <stage>_run_meta.jsonl     exactly one row: mode real, the sweep's seed, stage and terms, max_iters
                             equal to the sweep's iterations, the swept value, the current Logit-KD
                             semantics without an override, the recipe of record (RECIPE_EXPECT), the
                             carriers every candidate must share (RECIPE_IDENTICAL) and ramp_iters
  <stage>_telemetry.jsonl    its last row decides the candidate's status (below)
  best.json                  train_e1's schema {best_ckpt, best_val_miou_all_class}: required for a
                             finished run; read when present for a diverged one (best_val_partial)
  the checkpoint best.json names (by basename when the pod path is absent), finished runs only: it
                             must record the same best value and the same stage
Any input path with a component named `test` is refused. A file that cannot be parsed is refused by
name (never a traceback), except a torn last telemetry line, which marks a run that did not finish.

Candidate status (load_candidate):
  finished   the telemetry ends with train_distill's run_end record: the last train row reaches
             max_iters, run_end (written after the final validation and checkpoint) agrees with best.json
             and the checkpoint, and the run's own checks passed.
  diverged   AM-7a: the telemetry ends with the trainer's own run_abort record for a STUDENT divergence
             under AM-7 (a) or (b): a rule in DIVERGENCE_RULES, cause DIVERGENCE_CAUSE, finite input and
             teacher outputs, at the last train iteration. The run is excluded from the selection and never
             relaunched; its directory is a required selection input (it is not a shortfall).
  shortfall  SHORTFALL_CODES: a run that is absent, has not written its files yet, or has neither run_end
             nor run_abort (still running, or cut mid-write). Not a malformed input.
Refusal codes (exit 2 in the select scripts), besides the format and consistency refusals:
  default_candidate_diverged  the sweep's default candidate (sweep_rules default_candidate: lambda 1,
                              alpha 50) diverged: AM-7 applies in full (stop the stage, apply the AM-7a
                              item 5 clipping value, rerun the FP32 stages); no selection is made.
  run_aborted_other           a run_abort that is not a student divergence (another rule, a non-finite
                              input or teacher output), or a train row flagged `nonfinite` (or a val row
                              whose all-class mIoU is) with no run_abort after it (the abort record is
                              missing or torn). STOP: investigate; AM-8a governs a repeat; never excluded,
                              never a shortfall.
  abort_record_invalid        a run_abort record that is malformed, is not the last telemetry row, is
                              accompanied by a run_end, is not at the last train iteration, or is an
                              AM-7 (b) record that cannot hold (before ramp_iters + AM7B_WINDOW, or a
                              ratio not above AM7B_FACTOR).
  recipe_mismatch             a run_meta that violates RECIPE_EXPECT or lacks a RECIPE_IDENTICAL value,
                              or candidates whose RECIPE_IDENTICAL values differ.

Tie arithmetic. "Within the band of the best" is inclusive (best - value <= band) and is evaluated in
decimal on the recorded values (Decimal(repr(x))), so an exact 0.5 pp gap between two recorded values
counts as tied, as it does by hand. Only finished candidates are compared; a sole finished candidate
wins with no band and no tie (AM-7a). A winner is a boundary result at a grid end, or at an edge of the
finished set next to a diverged value (boundary_kind grid_end / edge_of_finished_set).
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RULES_PATH = REPO / "configs" / "sweep_rules.json"
BEST_JSON = "best.json"
BEST_JSON_KEYS = ["best_ckpt", "best_val_miou_all_class"]
STAGE_TERMS = {"E2": {"logit_kd": True, "cwd_feat": False, "cwd_logit": False},
               "E3": {"logit_kd": True, "cwd_feat": True, "cwd_logit": True}}
# A run that is absent, has not written its files yet, or has not finished (no run_end and no
# run_abort record) is a SHORTFALL, not a malformed input.
SHORTFALL_CODES = ("candidate_missing", "candidate_incomplete", "run_unfinished")
UNEXPECTED_ERROR_EXIT = 4        # the select scripts: exit 0 selected, 2 refused, 3 shortfall, 4 error
# AM-7a: only the trainer's own run_abort record decides that a run diverged.
DIVERGENCE_RULES = ("AM-7(a)", "AM-7(b)")
DIVERGENCE_CAUSE = "student_divergence"
AM7B_WINDOW = 100                # configs/distill.py AM7_DIVERGENCE; load_rules requires agreement
AM7B_FACTOR = 5.0
ABORT_DETAIL_KEYS = ("loss", "grad_norm", "window_mean", "running_min", "ratio", "threshold")
# L-KD-HARDEN item 10b: the recipe of record of every candidate, and the carriers that must be
# identical across the candidates of one sweep.
RECIPE_EXPECT = {"grad_clip_norm": None, "batch_size": 16, "val_interval": 4000, "max_val_batches": None,
                 "used_pretrained": True, "poly_horizon": 80000}
RECIPE_IDENTICAL = ("num_workers", "teacher_provenance.ckpt_sha256")


class SelectionRefused(RuntimeError):
    """A named refusal; `code` makes every guard testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def dec(x) -> Decimal:
    return Decimal(repr(float(x)))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def refuse_test_path(p: Path) -> None:
    parts = [part.lower() for part in Path(p).resolve().parts]
    if "test" in parts:
        raise SelectionRefused("test_path", f"refusing {p}: a selection never reads anything under a "
                                            "directory named 'test'")


def read_json(path: Path, code: str = "unreadable_input"):
    """json.loads of a file, refused by name when it is not valid JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SelectionRefused(code, f"{path} is not valid JSON ({e})") from e


def _finite_unit(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) \
            or not 0.0 <= float(x) <= 1.0:
        raise SelectionRefused("bad_value", f"{what} must be a finite number in [0, 1], got {x!r}")
    return float(x)


def _record_number(x):
    """A number from a run_abort record: finite floats as written, "nan"/"inf"/"-inf" as their floats
    (the trainer's strict-JSON encoding), anything else None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if x in ("nan", "inf", "-inf"):
        return float(x)
    return None


def _dotted(m: dict, dotted: str):
    head, _, rest = dotted.partition(".")
    v = m.get(head)
    if rest:
        return v.get(rest) if isinstance(v, dict) else None
    return v


# ------------------------------------------------------------------------------------ rules
def load_rules(path: Path = RULES_PATH) -> dict:
    """Read the committed rule file and require it to agree with configs/distill.py."""
    from configs.distill import AM7_DIVERGENCE, DISTILL

    path = Path(path)
    if not path.is_file():
        raise SelectionRefused("rules_missing", f"rule file not found: {path}")
    doc = read_json(path, "rules_format")
    if not isinstance(doc, dict) or doc.get("format") != "sweep_rules/1":
        raise SelectionRefused("rules_format", f"{path}: format {doc.get('format')!r} != 'sweep_rules/1'")
    lam, alp = doc.get("lambda_logit"), doc.get("alpha_cwd")
    if not isinstance(lam, dict) or not isinstance(alp, dict):
        raise SelectionRefused("rules_format", f"{path}: needs 'lambda_logit' and 'alpha_cwd' blocks")
    for name, sweep in (("lambda_logit", lam), ("alpha_cwd", alp)):
        for key in ("stage", "seed", "iterations", "grid", "band", "tie", "boundary", "winner_is",
                    "default_candidate"):
            if key not in sweep:
                raise SelectionRefused("rules_format", f"{path}: '{name}' lacks '{key}'")
        if sweep["stage"] not in STAGE_TERMS:
            raise SelectionRefused("rules_format", f"{path}: '{name}' stage {sweep['stage']!r}")
        dc = sweep["default_candidate"]
        if isinstance(dc, bool) or not isinstance(dc, (int, float)) \
                or float(dc) not in [float(v) for v in sweep["grid"]]:
            raise SelectionRefused("rules_format", f"{path}: '{name}' default_candidate {dc!r} is not a "
                                                   "value on its grid")
    lk, cwd = DISTILL["logit_kd"], DISTILL["cwd"]
    expect = [
        ("lambda grid", [float(v) for v in lam.get("grid", [])],
         [float(v) for v in lk["lambda_logit_sweep_grid"]]),
        ("lambda band", lam.get("band", {}).get("value"), lk["lambda_sweep_tie_band_pp"] / 100.0),
        ("lambda iterations", lam.get("iterations"), lk["lambda_sweep_iters_per_candidate"]),
        ("lambda seed", lam.get("seed"), lk["sweep_seed"]),
        ("lambda default candidate", lam.get("default_candidate"), lk["lambda_default_candidate"]),
        ("alpha grid", [float(v) for v in alp.get("grid", [])], [float(v) for v in cwd["alpha_cwd_grid"]]),
        ("alpha default", alp.get("default"), cwd["alpha_cwd_feature_map"]),
        ("alpha default candidate", alp.get("default_candidate"), cwd["alpha_cwd_feature_map"]),
        ("alpha iterations", alp.get("iterations"), cwd["alpha_sweep_iters_per_candidate"]),
    ]
    bad = [(what, got, want) for what, got, want in expect if got != want]
    if bad:
        raise SelectionRefused("rules_mismatch", f"{path} disagrees with configs/distill.py: {bad}")
    am7b = {"window": (AM7B_WINDOW, AM7_DIVERGENCE["window"]),
            "factor": (AM7B_FACTOR, AM7_DIVERGENCE["factor"])}
    if any(mine != cfg for mine, cfg in am7b.values()):
        raise SelectionRefused("rules_mismatch", f"sweep_select's AM-7 (b) constants disagree with "
                                                 f"configs/distill.py AM7_DIVERGENCE: {am7b} (here, config)")
    return doc


def dl27_band(doc: dict, floor: float) -> tuple[float, float, list[str]]:
    """(s, band, trace) from the DL-27 band entry: band = max(floor, sqrt(2) * s), s the n=3 sample
    SD of E1's best VAL all-class mIoU over seeds 42, 43, 44 (AM-16 item 2).

    Required keys: e1_best_val {"42", "43", "44"} and s. s is recomputed from the three values and must
    agree to 1e-12; a recorded band must agree with the recomputed one to 1e-12."""
    if not isinstance(doc, dict):
        raise SelectionRefused("band_format", f"the band file must hold a JSON object, got "
                                              f"{type(doc).__name__}")
    vals = doc.get("e1_best_val")
    if not isinstance(vals, dict) or sorted(vals) != ["42", "43", "44"]:
        raise SelectionRefused("band_format", "the band file needs e1_best_val for seeds '42', '43', "
                                              f"'44'; got {sorted(vals) if isinstance(vals, dict) else vals!r}")
    xs = [_finite_unit(vals[k], f"e1_best_val[{k}]") for k in ("42", "43", "44")]
    s_file = doc.get("s")
    if isinstance(s_file, bool) or not isinstance(s_file, (int, float)) or not math.isfinite(s_file) \
            or s_file < 0:
        raise SelectionRefused("band_format", f"the band file's s must be finite and >= 0, got {s_file!r}")
    s_calc = statistics.stdev(xs)
    if abs(s_calc - float(s_file)) > 1e-12:
        raise SelectionRefused("band_s_mismatch", f"recorded s {s_file!r} != the sample SD of "
                                                  f"{xs} = {s_calc!r}")
    s = float(s_file)
    band = max(float(floor), math.sqrt(2.0) * s)
    if "band" in doc and abs(float(doc["band"]) - band) > 1e-12:
        raise SelectionRefused("band_mismatch", f"recorded band {doc['band']!r} != max({floor}, "
                                                f"sqrt(2)*{s!r}) = {band!r}")
    trace = [f"band: s = {s!r} (sample SD, n=3, E1 seeds 42-44 = {xs}); sqrt(2)*s = "
             f"{math.sqrt(2.0) * s!r}; band = max({floor}, sqrt(2)*s) = {band!r}"
             + (" (the 0.5 pp floor applies)" if band == float(floor) else "")]
    return s, band, trace


# ------------------------------------------------------------------------------- candidates
def _read_run_meta(run_dir: Path, meta_p: Path, sweep: dict, key: str) -> dict:
    rows = []
    for n, ln in enumerate(meta_p.read_text(encoding="utf-8").splitlines(), 1):
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as e:
                raise SelectionRefused("unreadable_input", f"{meta_p}: line {n} is not JSON ({e})") from e
    if len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get("event") != "run_meta":
        raise SelectionRefused("run_meta_rows", f"{meta_p}: {len(rows)} run_meta rows; exactly one is "
                                                "required (more means an appended relaunch)")
    m = rows[0]
    stage = sweep["stage"]
    want = {"mode": "real", "seed": sweep["seed"], "stage": stage, "max_iters": sweep["iterations"],
            "terms": STAGE_TERMS[stage]}
    wrong = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if wrong:
        raise SelectionRefused("run_meta_mismatch", f"{run_dir.name}: run_meta {wrong} (got, want)")
    if key not in m:
        raise SelectionRefused("run_meta_mismatch", f"{run_dir.name}: run_meta has no {key!r}")
    if key == "alpha_cwd" and m.get("alpha_offgrid"):
        raise SelectionRefused("offgrid", f"{run_dir.name}: trained with --allow-offgrid (tests only)")
    if STAGE_TERMS[stage]["logit_kd"]:
        # lambda_logit is only comparable under one Logit-KD semantics (B32c-2), in either sweep
        from configs.distill import LOGIT_KD_SEMANTICS
        if m.get("logit_kd_semantics") != LOGIT_KD_SEMANTICS or m.get("logit_kd_semantics_override_used"):
            raise SelectionRefused("semantics", f"{run_dir.name}: Logit-KD semantics "
                                                f"{m.get('logit_kd_semantics')!r} / override "
                                                f"{m.get('logit_kd_semantics_override_used')!r}")
    bad = {k: (m.get(k, "<absent>"), v) for k, v in RECIPE_EXPECT.items()
           if k not in m or m[k] != v or type(m[k]) is not type(v)}
    if bad:
        raise SelectionRefused("recipe_mismatch", f"{run_dir.name}: run_meta departs from the recipe of "
                                                  f"record {bad} (got, want)")
    shared = {f: _dotted(m, f) for f in RECIPE_IDENTICAL}
    if not (isinstance(shared["num_workers"], int) and not isinstance(shared["num_workers"], bool)
            and isinstance(shared["teacher_provenance.ckpt_sha256"], str)
            and shared["teacher_provenance.ckpt_sha256"]):
        raise SelectionRefused("recipe_mismatch", f"{run_dir.name}: run_meta lacks a comparable "
                                                  f"{list(RECIPE_IDENTICAL)}: {shared}")
    ramp = m.get("ramp_iters")
    if isinstance(ramp, bool) or not isinstance(ramp, int) or ramp < 1:
        raise SelectionRefused("run_meta_mismatch", f"{run_dir.name}: run_meta ramp_iters {ramp!r}")
    return m


def _scan_telemetry(tel_p: Path) -> tuple[list[dict], bool]:
    """(rows, torn_last). A torn LAST line marks a run cut mid-write; a torn line elsewhere is a
    damaged file and is refused."""
    lines = tel_p.read_text(encoding="utf-8").splitlines()
    rows, torn_last = [], False
    for n, ln in enumerate(lines, 1):
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError as e:
            if n == len(lines):
                torn_last = True
                break
            raise SelectionRefused("unreadable_input", f"{tel_p}: line {n} is not JSON ({e})") from e
        rows.append(row if isinstance(row, dict) else {})
    return rows, torn_last


def _diverged(run_dir: Path, rows: list[dict], torn_last: bool, ramp_iters: int) -> dict:
    """Validate the run's run_abort record (AM-7a); return it if it is an excluded divergence."""
    events = [r.get("event") for r in rows]
    aborts = [i for i, e in enumerate(events) if e == "run_abort"]
    train_iters = [r.get("iter") for r in rows if r.get("event") == "train"]
    ab = rows[aborts[-1]]
    if len(aborts) != 1 or aborts[0] != len(rows) - 1 or torn_last or "run_end" in events:
        raise SelectionRefused("abort_record_invalid", f"{run_dir.name}: a run_abort record must be the "
                                                       "single last telemetry row, never beside a run_end")
    detail = ab.get("detail")
    if not (isinstance(ab.get("iter"), int) and isinstance(ab.get("rule"), str)
            and isinstance(ab.get("cause"), str) and isinstance(detail, dict)
            and all(k in detail for k in ABORT_DETAIL_KEYS)):
        raise SelectionRefused("abort_record_invalid", f"{run_dir.name}: malformed run_abort record "
                                                       f"{json.dumps(ab)[:200]}")
    if not train_iters or ab["iter"] != train_iters[-1]:
        raise SelectionRefused("abort_record_invalid", f"{run_dir.name}: run_abort at iter {ab['iter']} "
                                                       f"but the last train row is iter "
                                                       f"{train_iters[-1] if train_iters else None}")
    if ab["rule"] == "AM-7(b)":
        ratio = _record_number(detail.get("ratio"))
        if ab["iter"] < ramp_iters + AM7B_WINDOW or ratio is None or not ratio > AM7B_FACTOR:
            raise SelectionRefused("abort_record_invalid",
                                   f"{run_dir.name}: an AM-7 (b) abort at iter {ab['iter']} with ratio "
                                   f"{detail.get('ratio')!r} cannot hold (it needs iter >= ramp_iters + "
                                   f"{AM7B_WINDOW} = {ramp_iters + AM7B_WINDOW} and ratio > {AM7B_FACTOR})")
    if ab["rule"] not in DIVERGENCE_RULES or ab["cause"] != DIVERGENCE_CAUSE \
            or ab.get("input_finite") is not True or ab.get("teacher_finite") is not True:
        raise SelectionRefused("run_aborted_other",
                               f"{run_dir.name}: the run aborted at iter {ab['iter']} under rule "
                               f"{ab['rule']!r} with cause {ab['cause']!r} (input_finite="
                               f"{ab.get('input_finite')!r}, teacher_finite={ab.get('teacher_finite')!r}). "
                               "This is not a student divergence under AM-7 (a)/(b). STOP: investigate; "
                               "AM-8a governs a repeat; never excluded, never a shortfall.")
    return ab


def load_candidate(run_dir: Path, sweep: dict, key: str) -> dict:
    """Validate one real sweep run; return {status: finished|diverged, value, run_id, ...}."""
    run_dir = Path(run_dir)
    refuse_test_path(run_dir)
    if not run_dir.is_dir():
        raise SelectionRefused("candidate_missing", f"candidate directory not found: {run_dir}")
    stage = sweep["stage"]
    sk = stage.lower()
    best_p, meta_p, tel_p = run_dir / BEST_JSON, run_dir / f"{sk}_run_meta.jsonl", \
        run_dir / f"{sk}_telemetry.jsonl"
    for p in (meta_p, tel_p):
        if not p.is_file():
            raise SelectionRefused("candidate_incomplete", f"{run_dir.name}: {p.name} is missing")
    m = _read_run_meta(run_dir, meta_p, sweep, key)
    value = float(m[key])
    common = {"value": value, "run_id": run_dir.name, "run_dir": str(run_dir),
              "run_meta_sha256": sha256_file(meta_p), "telemetry_sha256": sha256_file(tel_p),
              "lambda_logit": m.get("lambda_logit"), "num_workers": m["num_workers"],
              "teacher_ckpt_sha256": m["teacher_provenance"]["ckpt_sha256"], "ramp_iters": m["ramp_iters"]}

    rows, torn_last = _scan_telemetry(tel_p)
    if any(r.get("event") == "run_abort" for r in rows):
        ab = _diverged(run_dir, rows, torn_last, m["ramp_iters"])
        best_val_partial = best_sha = None
        if best_p.is_file():
            best = read_json(best_p)
            if not isinstance(best, dict) or sorted(best) != BEST_JSON_KEYS:
                got = sorted(best) if isinstance(best, dict) else best
                raise SelectionRefused("best_json_format", f"{best_p}: keys {got!r} != {BEST_JSON_KEYS}")
            best_val_partial = _finite_unit(best["best_val_miou_all_class"],
                                            f"{best_p} best_val_miou_all_class")
            best_sha = sha256_file(best_p)
        return {"status": "diverged", **common,
                "abort": {"iter": ab["iter"], "rule": ab["rule"], "cause": ab["cause"],
                          "detail": ab["detail"]},
                "n_val": ab.get("n_val"), "best_val_partial": best_val_partial, "best_json_sha256": best_sha}
    # A row the trainer writes just before an abort (a train row with a `nonfinite` map, or a val row whose
    # all-class mIoU is non-finite) with no run_abort record after it: the record is missing or torn.
    bad = [(r.get("event"), r.get("iter")) for r in rows
           if "nonfinite" in r and (r.get("event") == "train"
                                    or (r.get("event") == "val" and "all_class_miou" in r["nonfinite"]))]
    if bad:
        raise SelectionRefused("run_aborted_other",
                               f"{run_dir.name}: row(s) {bad} carry the non-finite values of an abort, with "
                               "no run_abort record after them (the abort record is missing or torn). STOP: "
                               "investigate; AM-8a governs a repeat; never excluded, never a shortfall.")
    if torn_last:
        raise SelectionRefused("run_unfinished", f"{run_dir.name}: the last telemetry line is torn (the run "
                                                 "was cut mid-write)")

    # Finished = the last train row reaches max_iters AND the last row is train_distill's run_end
    # record, written after the final validation and checkpoint save.
    last_iter, run_end = None, None
    for row in rows:
        event = row.get("event")
        if event == "train":
            last_iter, run_end = row.get("iter"), None
        elif event == "run_end":
            run_end = row
    if last_iter != sweep["iterations"] or run_end is None or run_end.get("iter") != sweep["iterations"]:
        raise SelectionRefused("run_unfinished", f"{run_dir.name}: last train iteration {last_iter}, "
                                                 f"run_end {'absent' if run_end is None else run_end.get('iter')}"
                                                 f"; a finished run reaches {sweep['iterations']} and ends "
                                                 "with a run_end record")
    if not best_p.is_file():
        raise SelectionRefused("candidate_incomplete", f"{run_dir.name}: {best_p.name} is missing")
    best = read_json(best_p)
    if not isinstance(best, dict) or sorted(best) != BEST_JSON_KEYS:
        raise SelectionRefused("best_json_format", f"{best_p}: keys "
                                                   f"{sorted(best) if isinstance(best, dict) else best!r} "
                                                   f"!= {BEST_JSON_KEYS}")
    best_val = _finite_unit(best["best_val_miou_all_class"], f"{best_p} best_val_miou_all_class")
    if run_end.get("checks_passed") is not True:
        raise SelectionRefused("run_checks_failed", f"{run_dir.name}: the run's own checks did not pass "
                                                    f"(run_end checks_passed={run_end.get('checks_passed')!r})")
    if run_end.get("best_val_miou_all_class") != best_val \
            or run_end.get("best_ckpt") != Path(best["best_ckpt"]).name:
        raise SelectionRefused("run_end_mismatch", f"{run_dir.name}: run_end records best "
                                                   f"{run_end.get('best_val_miou_all_class')!r} / "
                                                   f"{run_end.get('best_ckpt')!r}, best.json "
                                                   f"{best_val!r} / {Path(best['best_ckpt']).name!r}")

    ck_path = Path(best["best_ckpt"])
    if not ck_path.is_file():
        ck_path = run_dir / ck_path.name
    refuse_test_path(ck_path)
    if not ck_path.is_file():
        raise SelectionRefused("checkpoint_missing", f"{run_dir.name}: the best checkpoint "
                                                     f"{Path(best['best_ckpt']).name} is not present")
    import torch
    try:
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 — any unreadable checkpoint is refused by name
        raise SelectionRefused("checkpoint_unreadable", f"{run_dir.name}: {ck_path.name} could not be "
                                                        f"read ({type(e).__name__}: {e})") from e
    if not isinstance(ck, dict) or ck.get("stage") != stage \
            or ck.get("best_val_miou_all_class") != best_val:
        raise SelectionRefused("checkpoint_mismatch", f"{run_dir.name}: checkpoint stage "
                                                      f"{ck.get('stage') if isinstance(ck, dict) else '?'!r} / best "
                                                      f"{ck.get('best_val_miou_all_class') if isinstance(ck, dict) else '?'!r}"
                                                      f" do not match {stage} / {best_val!r}")
    if key in ck and float(ck[key]) != value:
        raise SelectionRefused("checkpoint_mismatch", f"{run_dir.name}: checkpoint {key} {ck[key]!r} "
                                                      f"!= run_meta {value!r}")
    return {"status": "finished", **common, "best_val": best_val, "ckpt_sha256": sha256_file(ck_path),
            "checkpoint": ck_path.name, "best_json_sha256": sha256_file(best_p)}


def collect_candidates(run_dirs, sweep: dict, key: str) -> tuple[list[dict], list[dict], list[dict]]:
    """(finished, diverged, shortfall). A malformed input refuses at once; a missing or unfinished run
    is collected as shortfall so the caller can name the rule that governs it."""
    finished, diverged, shortfall = [], [], []
    for d in run_dirs:
        try:
            c = load_candidate(Path(d), sweep, key)
        except SelectionRefused as e:
            if e.code in SHORTFALL_CODES:
                shortfall.append({"run_dir": str(d), "code": e.code, "reason": str(e)})
                continue
            raise
        (finished if c["status"] == "finished" else diverged).append(c)
    values = [c["value"] for c in finished + diverged]
    dup = sorted({v for v in values if values.count(v) > 1})
    if dup:
        raise SelectionRefused("duplicate_candidate", f"more than one run for {key} {dup}")
    grid = [float(v) for v in sweep["grid"]]
    off = sorted(v for v in values if v not in grid)
    if off:
        raise SelectionRefused("grid_mismatch", f"{key} {off} not in the registered grid {grid}")
    default = float(sweep["default_candidate"])
    for c in diverged:
        if c["value"] == default:
            raise SelectionRefused(
                "default_candidate_diverged",
                f"the default candidate {key} = {default:g} (run {c['run_id']}) diverged at iter "
                f"{c['abort']['iter']} under {c['abort']['rule']}: AM-7 applies in full: stop the stage, "
                "apply the AM-7a item 5 clipping value, rerun the FP32 stages; no selection is made")
    for field in ("num_workers", "teacher_ckpt_sha256"):
        seen = {repr(c[field]) for c in finished + diverged}
        if len(seen) > 1:
            raise SelectionRefused("recipe_mismatch", f"the candidates differ in {field}: "
                                                      f"{sorted(seen)} (a sweep's runs share one value)")
    return finished, diverged, shortfall


def missing_grid_values(cands: list[dict], sweep: dict) -> list[float]:
    have = {c["value"] for c in cands}
    return [float(v) for v in sweep["grid"] if float(v) not in have]


# -------------------------------------------------------------------------------------- rule
def apply_rule(cands: list[dict], sweep: dict, band: float, name: str, *, diverged=()) -> dict:
    """Highest best value among the FINISHED candidates wins; candidates within `band` of the best
    (inclusive, decimal) are tied; a tie goes to the registered default when the rule has one and it is
    tied, else to the smallest value; a sole finished candidate wins with no band and no tie (AM-7a).
    The finished and diverged values together must be the full grid, each once. A winner at a grid end,
    or at an edge of the finished set next to a diverged value, is flagged as a boundary result."""
    diverged = list(diverged)
    grid = [float(v) for v in sweep["grid"]]
    values = sorted([c["value"] for c in cands] + [c["value"] for c in diverged])
    if values != sorted(grid):
        raise SelectionRefused("partial_input", f"{name} candidates {values} (finished and diverged) != "
                                                f"the full grid {grid}")
    if not cands:
        raise SelectionRefused("partial_input", f"no finished {name} candidate")
    trace = []
    for c in sorted(diverged, key=lambda c: c["value"]):
        trace.append(f"{name}={c['value']:g} run={c['run_id']}: diverged (AM-7a) at iter "
                     f"{c['abort']['iter']} under {c['abort']['rule']}; excluded from the selection")
    if len(cands) == 1:
        winner = cands[0]
        tied = [winner["value"]]
        best = dec(winner["best_val"])
        trace.append(f"sole finished candidate (AM-7a): no band, no tie -> {name} = {winner['value']:g} "
                     f"(run={winner['run_id']} best_val={winner['best_val']!r})")
    else:
        best = max(dec(c["best_val"]) for c in cands)
        band_d = dec(band)
        trace.append(f"best {name}-candidate value = {float(best)!r}; tie band = {band!r} (inclusive: "
                     f"best - value <= band, decimal on the recorded values)")
        tied_c = []
        for c in sorted(cands, key=lambda c: c["value"]):
            gap = best - dec(c["best_val"])
            inside = gap <= band_d
            trace.append(f"{name}={c['value']:g} run={c['run_id']} best_val={c['best_val']!r} gap={gap} "
                         f"-> {'tied' if inside else 'outside the band'}")
            if inside:
                tied_c.append(c)
        tied = [c["value"] for c in tied_c]
        rule = sweep["tie"]
        default = float(sweep["default"]) if sweep.get("default") is not None else None
        if rule == "default_if_tied_else_smallest" and default in tied:
            winner = next(c for c in tied_c if c["value"] == default)
            trace.append(f"the tied set {tied} contains the default {default:g} -> {name} = {default:g}")
        elif rule in ("smallest", "default_if_tied_else_smallest"):
            winner = tied_c[0]
            trace.append(f"tied set {tied} -> the smallest {name} = {winner['value']:g}"
                         if len(tied_c) > 1 else f"a single candidate is within the band -> {name} = "
                                                 f"{winner['value']:g}")
        else:
            raise SelectionRefused("rules_format", f"unknown tie rule {rule!r}")
    fin = sorted(c["value"] for c in cands)
    div = [c["value"] for c in diverged]
    grid_end = winner["value"] in [float(b) for b in sweep["boundary"]]
    edge = ((winner["value"] == fin[0] and any(v < winner["value"] for v in div))
            or (winner["value"] == fin[-1] and any(v > winner["value"] for v in div)))
    boundary = grid_end or edge
    kind = "grid_end" if grid_end else ("edge_of_finished_set" if edge else None)
    trace.append(f"boundary winner: {boundary} ({kind}; grid ends {sweep['boundary']}, finished set "
                 f"{fin}); the grid is not extended")
    trace.append(f"winner run {winner['run_id']} is {sweep['winner_is']} (an existing run; checkpoint "
                 f"sha256 {winner['ckpt_sha256']})")
    excluded = [{"value": c["value"], "run_id": c["run_id"],
                 "abort": {k: c["abort"][k] for k in ("iter", "rule", "cause", "detail")},
                 "n_val": c.get("n_val"), "best_val_partial": c.get("best_val_partial"),
                 "telemetry_sha256": c.get("telemetry_sha256"), "run_meta_sha256": c.get("run_meta_sha256")}
                for c in sorted(diverged, key=lambda c: c["value"])]
    return {"winner": winner, "tie": len(tied) > 1, "tied": tied, "boundary": boundary,
            "boundary_kind": kind, "best_val": float(best), "rule_trace": trace,
            "excluded_am7a": excluded, "n_finished": len(cands), "n_diverged": len(diverged)}


def write_selection(path: Path, doc: dict) -> Path:
    """Write the selection once. An existing selection file is never overwritten."""
    path = Path(path)
    refuse_test_path(path.parent)
    if path.exists():
        raise SelectionRefused("output_exists", f"{path} already exists; a sweep is selected once")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
