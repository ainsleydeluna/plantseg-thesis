"""Sweep selection for lambda_logit (AM-2 / DL-06) and alpha_cwd (AM-16 item 2 / DL-27).

Shared by scripts/select_lambda.py and scripts/select_alpha.py (lane L-AM16-ALPHA). The cross-lane
selection contract (docs/lane_specs/part2.md): refuse partial inputs; read the band and rule from a
committed JSON (configs/sweep_rules.json; alpha's band from reports/derived/dl27_band.json); write a
JSON with a `rule_trace`; never read TEST; the winner is always an existing run.

A candidate is one FINISHED REAL training run, given as its checkpoint directory. Only these files
are opened:
  best.json                  train_e1's schema {best_ckpt, best_val_miou_all_class}: the run's
                             best-checkpoint VAL all-class mIoU (strict >, earliest tie, as E1)
  <stage>_run_meta.jsonl     exactly one row: mode real, the sweep's seed, stage and terms,
                             max_iters equal to the sweep's iterations, the swept value, and the
                             current Logit-KD semantics without an override
  <stage>_telemetry.jsonl    finished: its last train row reaches max_iters and its last row is the
                             `run_end` record train_distill writes after the final validation and
                             checkpoint (checks passed, same best value and checkpoint as best.json)
  the checkpoint best.json names (by basename when the pod path is absent), which must record the
                             same best value and the same stage
Any input path with a component named `test` is refused. A file that cannot be parsed is refused by
name (never a traceback), except a torn last telemetry line, which marks a run that did not finish.

Tie arithmetic. "Within the band of the best" is inclusive (best - value <= band) and is evaluated in
decimal on the recorded values (Decimal(repr(x))), so an exact 0.5 pp gap between two recorded values
counts as tied, as it does by hand.
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
# A run that is absent, has not written its files yet, or has not finished (no run_end record) is a
# SHORTFALL, not a malformed input.
SHORTFALL_CODES = ("candidate_missing", "candidate_incomplete", "run_unfinished")
UNEXPECTED_ERROR_EXIT = 4        # the select scripts: exit 0 selected, 2 refused, 3 shortfall, 4 error


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


# ------------------------------------------------------------------------------------ rules
def load_rules(path: Path = RULES_PATH) -> dict:
    """Read the committed rule file and require it to agree with configs/distill.py."""
    from configs.distill import DISTILL

    path = Path(path)
    if not path.is_file():
        raise SelectionRefused("rules_missing", f"rule file not found: {path}")
    doc = read_json(path, "rules_format")
    if not isinstance(doc, dict) or doc.get("format") != "sweep_rules/1":
        raise SelectionRefused("rules_format", f"{path}: format {doc.get('format')!r} != 'sweep_rules/1'")
    lam, alp = doc.get("lambda_logit"), doc.get("alpha_cwd")
    if not isinstance(lam, dict) or not isinstance(alp, dict):
        raise SelectionRefused("rules_format", f"{path}: needs 'lambda_logit' and 'alpha_cwd' blocks")
    lk, cwd = DISTILL["logit_kd"], DISTILL["cwd"]
    expect = [
        ("lambda grid", [float(v) for v in lam.get("grid", [])],
         [float(v) for v in lk["lambda_logit_sweep_grid"]]),
        ("lambda band", lam.get("band", {}).get("value"), lk["lambda_sweep_tie_band_pp"] / 100.0),
        ("lambda iterations", lam.get("iterations"), lk["lambda_sweep_iters_per_candidate"]),
        ("lambda seed", lam.get("seed"), lk["sweep_seed"]),
        ("alpha grid", [float(v) for v in alp.get("grid", [])], [float(v) for v in cwd["alpha_cwd_grid"]]),
        ("alpha default", alp.get("default"), cwd["alpha_cwd_feature_map"]),
        ("alpha iterations", alp.get("iterations"), cwd["alpha_sweep_iters_per_candidate"]),
    ]
    bad = [(what, got, want) for what, got, want in expect if got != want]
    if bad:
        raise SelectionRefused("rules_mismatch", f"{path} disagrees with configs/distill.py: {bad}")
    for name, sweep in (("lambda_logit", lam), ("alpha_cwd", alp)):
        for key in ("stage", "seed", "iterations", "grid", "band", "tie", "boundary", "winner_is"):
            if key not in sweep:
                raise SelectionRefused("rules_format", f"{path}: '{name}' lacks '{key}'")
        if sweep["stage"] not in STAGE_TERMS:
            raise SelectionRefused("rules_format", f"{path}: '{name}' stage {sweep['stage']!r}")
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
def load_candidate(run_dir: Path, sweep: dict, key: str) -> dict:
    """Validate one finished real run and return {value, run_id, best_val, ckpt_sha256, ...}."""
    run_dir = Path(run_dir)
    refuse_test_path(run_dir)
    if not run_dir.is_dir():
        raise SelectionRefused("candidate_missing", f"candidate directory not found: {run_dir}")
    stage = sweep["stage"]
    sk = stage.lower()
    best_p, meta_p, tel_p = run_dir / BEST_JSON, run_dir / f"{sk}_run_meta.jsonl", \
        run_dir / f"{sk}_telemetry.jsonl"
    for p in (best_p, meta_p, tel_p):
        if not p.is_file():
            raise SelectionRefused("candidate_incomplete", f"{run_dir.name}: {p.name} is missing")
    best = read_json(best_p)
    if not isinstance(best, dict) or sorted(best) != BEST_JSON_KEYS:
        raise SelectionRefused("best_json_format", f"{best_p}: keys "
                                                   f"{sorted(best) if isinstance(best, dict) else best!r} "
                                                   f"!= {BEST_JSON_KEYS}")
    best_val = _finite_unit(best["best_val_miou_all_class"], f"{best_p} best_val_miou_all_class")

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
    want = {"mode": "real", "seed": sweep["seed"], "stage": stage, "max_iters": sweep["iterations"],
            "terms": STAGE_TERMS[stage]}
    wrong = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if wrong:
        raise SelectionRefused("run_meta_mismatch", f"{run_dir.name}: run_meta {wrong} (got, want)")
    if key not in m:
        raise SelectionRefused("run_meta_mismatch", f"{run_dir.name}: run_meta has no {key!r}")
    value = float(m[key])
    if key == "alpha_cwd" and m.get("alpha_offgrid"):
        raise SelectionRefused("offgrid", f"{run_dir.name}: trained with --allow-offgrid (tests only)")
    if STAGE_TERMS[stage]["logit_kd"]:
        # lambda_logit is only comparable under one Logit-KD semantics (B32c-2), in either sweep
        from configs.distill import LOGIT_KD_SEMANTICS
        if m.get("logit_kd_semantics") != LOGIT_KD_SEMANTICS or m.get("logit_kd_semantics_override_used"):
            raise SelectionRefused("semantics", f"{run_dir.name}: Logit-KD semantics "
                                                f"{m.get('logit_kd_semantics')!r} / override "
                                                f"{m.get('logit_kd_semantics_override_used')!r}")

    # Finished = the last train row reaches max_iters AND the last row is train_distill's run_end
    # record, written after the final validation and checkpoint save. A torn last line is a run that
    # was cut mid-write: unfinished. A torn line anywhere else is a damaged file: refused.
    last_iter, run_end = None, None
    lines = tel_p.read_text(encoding="utf-8").splitlines()
    for n, ln in enumerate(lines, 1):
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError as e:
            if n == len(lines):
                raise SelectionRefused("run_unfinished", f"{run_dir.name}: the last telemetry line is "
                                                         "torn (the run was cut mid-write)") from e
            raise SelectionRefused("unreadable_input", f"{tel_p}: line {n} is not JSON ({e})") from e
        event = row.get("event") if isinstance(row, dict) else None
        if event == "train":
            last_iter, run_end = row.get("iter"), None
        elif event == "run_end":
            run_end = row
    if last_iter != sweep["iterations"] or run_end is None or run_end.get("iter") != sweep["iterations"]:
        raise SelectionRefused("run_unfinished", f"{run_dir.name}: last train iteration {last_iter}, "
                                                 f"run_end {'absent' if run_end is None else run_end.get('iter')}"
                                                 f"; a finished run reaches {sweep['iterations']} and ends "
                                                 "with a run_end record")
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
    return {"value": value, "run_id": run_dir.name, "best_val": best_val,
            "ckpt_sha256": sha256_file(ck_path), "checkpoint": ck_path.name, "run_dir": str(run_dir),
            "best_json_sha256": sha256_file(best_p), "run_meta_sha256": sha256_file(meta_p),
            "lambda_logit": m.get("lambda_logit")}


def collect_candidates(run_dirs, sweep: dict, key: str) -> tuple[list[dict], list[dict]]:
    """(finished candidates, shortfall entries). A malformed input refuses at once; a missing or
    unfinished run is collected as shortfall so the caller can name the rule that governs it."""
    finished, shortfall = [], []
    for d in run_dirs:
        try:
            finished.append(load_candidate(Path(d), sweep, key))
        except SelectionRefused as e:
            if e.code in SHORTFALL_CODES:
                shortfall.append({"run_dir": str(d), "code": e.code, "reason": str(e)})
            else:
                raise
    values = [c["value"] for c in finished]
    dup = sorted({v for v in values if values.count(v) > 1})
    if dup:
        raise SelectionRefused("duplicate_candidate", f"more than one run for {key} {dup}")
    grid = [float(v) for v in sweep["grid"]]
    off = sorted(v for v in values if v not in grid)
    if off:
        raise SelectionRefused("grid_mismatch", f"{key} {off} not in the registered grid {grid}")
    return finished, shortfall


def missing_grid_values(finished: list[dict], sweep: dict) -> list[float]:
    have = {c["value"] for c in finished}
    return [float(v) for v in sweep["grid"] if float(v) not in have]


# -------------------------------------------------------------------------------------- rule
def apply_rule(cands: list[dict], sweep: dict, band: float, name: str) -> dict:
    """Highest best value wins; candidates within `band` of the best (inclusive, decimal) are tied;
    a tie goes to the registered default when the rule has one and it is tied, else to the smallest
    value; a winner on the grid boundary is flagged. Requires the full grid, each value once."""
    grid = [float(v) for v in sweep["grid"]]
    values = sorted(c["value"] for c in cands)
    if values != sorted(grid):
        raise SelectionRefused("partial_input", f"{name} candidates {values} != the full grid {grid}")
    best = max(dec(c["best_val"]) for c in cands)
    band_d = dec(band)
    trace = [f"best {name}-candidate value = {float(best)!r}; tie band = {band!r} (inclusive: "
             f"best - value <= band, decimal on the recorded values)"]
    tied = []
    for c in sorted(cands, key=lambda c: c["value"]):
        gap = best - dec(c["best_val"])
        inside = gap <= band_d
        trace.append(f"{name}={c['value']:g} run={c['run_id']} best_val={c['best_val']!r} gap={gap} "
                     f"-> {'tied' if inside else 'outside the band'}")
        if inside:
            tied.append(c)
    tied_values = [c["value"] for c in tied]
    rule = sweep["tie"]
    default = float(sweep["default"]) if sweep.get("default") is not None else None
    if rule == "default_if_tied_else_smallest" and default in tied_values:
        winner = next(c for c in tied if c["value"] == default)
        trace.append(f"the tied set {tied_values} contains the default {default:g} -> {name} = {default:g}")
    elif rule in ("smallest", "default_if_tied_else_smallest"):
        winner = tied[0]
        trace.append(f"tied set {tied_values} -> the smallest {name} = {winner['value']:g}"
                     if len(tied) > 1 else f"a single candidate is within the band -> {name} = "
                                           f"{winner['value']:g}")
    else:
        raise SelectionRefused("rules_format", f"unknown tie rule {rule!r}")
    boundary = winner["value"] in [float(b) for b in sweep["boundary"]]
    trace.append(f"boundary winner: {boundary} (grid ends {sweep['boundary']}); the grid is not "
                 f"extended")
    trace.append(f"winner run {winner['run_id']} is {sweep['winner_is']} (an existing run; checkpoint "
                 f"sha256 {winner['ckpt_sha256']})")
    return {"winner": winner, "tie": len(tied) > 1, "tied": tied_values, "boundary": boundary,
            "best_val": float(best), "rule_trace": trace}


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
