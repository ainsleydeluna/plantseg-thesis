#!/usr/bin/env python3
"""AM-17 item 3 MDE entry and the AM-16 item 2 / DL-27 alpha tie band, written by one invocation (lane
L-AM17-MDE, S4; docs/lane_specs/part1.md lane 6; DL-27; AM-17 item 3: "in the same decision-log entry as
AM-16 item 2's alpha tie band").

    python -B scripts/mde_entry.py --s42 DIR --s43 DIR --s44 DIR \\
        --best-json-42 FILE --best-json-43 FILE --best-json-44 FILE --out-dir DIR [--expect-k-val K] \\
        [--expect-manifest-sha256-42 HEX --expect-manifest-sha256-43 HEX --expect-manifest-sha256-44 HEX] \\
        (--script-commit SHA --script-commit-dl-id DL-n | --synthetic-inputs [--generated-utc T])

Inputs are explicit paths; nothing is globbed and nothing is listed except the loader's one-level check of each
artifact directory (src/eval/artifacts.py verify_artifact). Seed 42 is the DL-17 B66 re-score (run1), seeds 43
and 44 the in-chain A40 re-scores; the three best.json files are E1's (src/training/train_e1.py
write_best_pointer). TEST is never read.

Checks, in order; the first failure exits:
  1  every path string containing "test" (case-insensitive) is refused before any filesystem call
     (teacher_diag.refuse_test_names), then again once resolved (teacher_diag.refuse_test_path)
  2  flags: a real run needs --script-commit and --script-commit-dl-id and refuses --generated-utc
  3  --out-dir resolves outside the repository; neither output exists yet
  4  real run: HEAD == --script-commit, the code files present, `git status -- src configs scripts` empty, and
     the running stack equal to src/stats/artifact.py PINNED_ENVIRONMENT
  5  the inputs exist
  6  each artifact loads through src/stats/val_artifacts.py (MANIFEST and strict JSON, canvas only,
     Policy.REHEARSAL: VAL, 846 rows, a real-run artifact, AM-5 flags read or derived) as stage E1, student,
     fp32; three distinct directories, run_ids and checkpoints; each pair comparable; --synthetic-inputs equal
     to what the dataset names say
  7  provenance, every check recorded in provenance_checks: one evaluate_model.py commit; one device class;
     each re-score with its inputs on the model device and the determinism policy applied; one non-null image
     digest; seed 42 a valid DL-17 B66 re-score within DL17_BAND of the reference; the optional MANIFEST pins
  8  AM-5: identical excluded sets across the seeds (else STOP (lane 6 (f))), then one identical image order
  9  the stored per-image disease-only mIoU equals its re-derivation from the sufficient statistics (exit 1)
  10 the band inputs: the rule file's DL-27 band (kind, file, floor 0.005); each best.json exactly train_e1's
     keys and its value equal to its artifact's recorded checkpoint value (checked before the long run)
  11 the three pairs (align_runs under REHEARSAL, delta = candidate - baseline) and the MDE (src/stats/mde.py);
     a pair with no qualifying delta is a STOP (exit 1; the three rejection-count curves are printed)
  12 the band file, read back through its reader (sweep_select.dl27_band) before anything is written (exit 1
     on disagreement), and the entry, checked by the report layer's validator (report.validate_mde_entry;
     exit 1 on a problem)
Outputs, serialized first and created exclusively, both or neither ("synthetic_" prefix for synthetic inputs):
  <out-dir>/mde_entry_<UTC>.json  the AM-17 item 3 entry: lane 6 (c)'s fields plus provenance
  <out-dir>/dl27_band.json        {"e1_best_val": {"42", "43", "44"}, "s", "band"}: exactly the reader's keys
No output records an out-dir, a hostname or a wall time, so a rerun with the same inputs, code and
--generated-utc is byte-identical.

Exit codes: 0 written; 1 a failed check; 2 a refusal (including the lane 6 (f) STOPs); 4 an unexpected error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from scripts import compare_eval_artifacts as CEA  # noqa: E402
from src.eval import teacher_diag as td  # noqa: E402
from src.eval.artifacts import MANIFEST_NAME  # noqa: E402
from src.eval.evaluate import F32_TOL  # noqa: E402
from src.stats import mde as M  # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY, AlignmentError, align_runs  # noqa: E402
from src.stats.artifact import software_environment_block  # noqa: E402
from src.stats.ingest import EXPECTED_ROWS_VAL, Policy  # noqa: E402
from src.stats.report import TAU_P, validate_mde_entry  # noqa: E402
from src.stats.val_artifacts import (ValArtifactError, code_provenance, load_val_artifact,  # noqa: E402
                                     require_comparable, require_role)
from src.training import sweep_select as SS  # noqa: E402

SCRIPT = "scripts/mde_entry.py"
LANE = "L-AM17-MDE"
SCHEMA = "plantseg-mde-entry/1.0.0"
AUTHORITY = ("AM-17 item 3 (docs/PREREGISTRATION_AMENDMENTS.md:393-418); docs/lane_specs/part1.md lane 6; "
             "DL-27 / AM-16 item 2 (the alpha tie band)")
SEEDS = ("42", "43", "44")
PAIRS = (("43-42", "42", "43"), ("44-42", "42", "44"), ("44-43", "43", "44"))    # (name, baseline, candidate)
ENTRY_PREFIX, BAND_NAME, SYNTHETIC_PREFIX = "mde_entry_", "dl27_band.json", "synthetic_"
BAND_RULE_FILE = "reports/derived/dl27_band.json"    # configs/sweep_rules.json alpha_cwd.band.file
BAND_FLOOR = 0.005                                   # DL-27: "the larger of 0.5 pp and sqrt(2) x ... SD"
PATH_FLAGS = ("s42", "s43", "s44", "best_json_42", "best_json_43", "best_json_44", "out_dir")
LABEL = ("planning proxy — seed-pair differences approximate noise, not the spread of between-recipe "
         "differences, so the true MDE may be larger. Reported with the results; it changes no test or "
         "decision.")
TIES_NOTE = ("lane 6 (f): the constant-shift null removes Pratt zeros (the pre-registered approximation): every "
             "exact-zero difference becomes +delta for delta > 0, so exact ties deflate MDE_W; read MDE_W with "
             "each pair's share_ties and n_zero")
EQUIVALENCE = ("default_rng(42) re-seeded per (pair, delta): the draw depends neither on delta nor on the "
               "values, so default_rng(42).choice(d + delta, size=(B, n), replace=True) == d[idx] + delta")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: every repository file whose code shapes a number in the outputs (recorded with its sha256)
CODE_FILES = ("scripts/mde_entry.py", "src/stats/mde.py", "src/stats/val_artifacts.py", "src/stats/ingest.py",
              "src/stats/align.py", "src/stats/report.py", "src/stats/artifact.py", "src/eval/artifacts.py",
              "src/eval/evaluate.py", "src/eval/metrics.py", "src/eval/teacher_diag.py",
              "src/training/sweep_select.py", "scripts/compare_eval_artifacts.py", "configs/sweep_rules.json",
              "configs/distill.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AM-17 item 3 MDE entry and the DL-27 alpha tie band (L-AM17-MDE).")
    for s in SEEDS:
        p.add_argument(f"--s{s}", required=True, help=f"E1 seed-{s} VAL canvas artifact directory")
    for s in SEEDS:
        p.add_argument(f"--best-json-{s}", required=True, help=f"E1 seed-{s} best.json")
    p.add_argument("--out-dir", required=True, help="outside the repository")
    p.add_argument("--expect-k-val", type=int, help="optional: refuse unless K_val (AM-5) equals it")
    for s in SEEDS:
        p.add_argument(f"--expect-manifest-sha256-{s}",
                       help=f"optional: the sha256 of the seed-{s} artifact's MANIFEST.sha256 (exit 2 otherwise)")
    p.add_argument("--script-commit", help="real runs: 40-hex HEAD of the checkout")
    p.add_argument("--script-commit-dl-id", help="real runs: DL-<n> logging that commit")
    p.add_argument("--synthetic-inputs", action="store_true",
                   help="synthetic (smoke) inputs: no binding, outputs prefixed synthetic_")
    p.add_argument("--generated-utc", help="synthetic runs only: pin the output stamp (YYYY-MM-DDTHH:MM:SSZ)")
    return p


# --------------------------------------------------------------------------------------------------
# 1-5 paths, flags, outputs, binding, environment
# --------------------------------------------------------------------------------------------------
def guard_paths(paths: dict) -> None:
    """Every path as given, by its string only (no filesystem call), then each one resolved."""
    td.refuse_test_names([str(v) for v in paths.values()], "path arguments")
    for flag, value in paths.items():
        td.refuse_test_path(value, "--" + flag.replace("_", "-"))


def check_flags(args, synthetic: bool) -> None:
    td.check_common_flags(args, real=not synthetic)
    for s in SEEDS:
        pin = getattr(args, f"expect_manifest_sha256_{s}")
        if pin is not None and not SHA256_RE.fullmatch(pin):
            raise td.Refused(f"--expect-manifest-sha256-{s} must be 64 lowercase hex characters")
    k = args.expect_k_val
    if k is not None and not 0 <= k < EXPECTED_ROWS_VAL:
        raise td.Refused(f"--expect-k-val must be in 0..{EXPECTED_ROWS_VAL - 1}, got {k}")


def output_names(args, synthetic: bool, start_utc: str) -> tuple[str, str]:
    prefix = SYNTHETIC_PREFIX if synthetic else ""
    stamp = td.utc_stamp(args.generated_utc or start_utc)
    return f"{prefix}{ENTRY_PREFIX}{stamp}.json", f"{prefix}{BAND_NAME}"


def commit_binding(args) -> dict:
    """A real run is a run of record: HEAD is --script-commit, the code files are present and src, configs and
    scripts are clean (the run-of-record convention of scripts/hash_split_files.py)."""
    head = td.git_head()
    if head is None:
        raise td.Refused("git HEAD could not be read; a real run needs the checkout at its pin")
    if head != args.script_commit:
        raise td.Refused(f"HEAD {head} != --script-commit {args.script_commit}")
    missing = [p for p in CODE_FILES if not (REPO / p).is_file()]
    if missing:
        raise td.Refused(f"{len(missing)} code file(s) missing at HEAD: {missing}")
    tree = td.git_status(("src", "configs", "scripts"))
    if tree is None or tree.strip():
        n = "?" if tree is None else len(tree.splitlines())
        raise td.Refused(f"git status --porcelain=v1 -- src configs scripts lists {n} entr(ies); a real run "
                         "needs them clean at HEAD")
    return {"head": head, "script_commit": args.script_commit, "script_commit_dl_id": args.script_commit_dl_id}


def environment_check(synthetic: bool) -> dict:
    env = software_environment_block()
    if not synthetic and env.get("matches_pinned") is not True:
        raise td.Refused(f"the running stack {env.get('observed')} is not the pinned {env.get('pinned')} "
                         "(src/stats/artifact.py PINNED_ENVIRONMENT): the pre-registered call is version-specific")
    return env


def require_inputs(args) -> None:
    for s in SEEDS:
        if not Path(getattr(args, f"s{s}")).is_dir():
            raise td.Refused(f"--s{s}: the artifact directory does not exist: {getattr(args, f's{s}')}")
    for s in SEEDS:
        if not Path(getattr(args, f"best_json_{s}")).is_file():
            raise td.Refused(f"--best-json-{s}: the file does not exist: {getattr(args, f'best_json_{s}')}")


# --------------------------------------------------------------------------------------------------
# 6-8 artifacts, provenance, AM-5
# --------------------------------------------------------------------------------------------------
def load_artifacts(args, synthetic: bool) -> dict:
    arts = {}
    for s in SEEDS:
        label = f"s{s}"
        try:
            art = load_val_artifact(Path(getattr(args, f"s{s}")), label=label)
            require_role(art, stage="E1", model_role="student", precision="fp32", label=label)
        except ValArtifactError as e:
            raise td.Refused(str(e)) from e
        arts[s] = art
    for what, get in (("directories", lambda a: str(a.path.resolve())), ("run_ids", lambda a: a.run_id),
                      ("checkpoints", lambda a: a.summary["run"].get("checkpoint_sha256"))):
        vals = [get(arts[s]) for s in SEEDS]
        if None in vals or len(set(vals)) != len(SEEDS):
            raise td.Refused(f"the three seeds must be three runs: their {what} are {vals}")
    try:
        for name, b, c in PAIRS:
            require_comparable(arts[b], arts[c])
    except ValArtifactError as e:
        raise td.Refused(str(e)) from e
    from src.eval.adapters import DATASET_NAME
    names = sorted({arts[s].summary["dataset"]["name"] for s in SEEDS})
    is_synthetic = any(n != DATASET_NAME for n in names)
    if is_synthetic != synthetic:
        raise td.Refused(f"--synthetic-inputs is {synthetic} but the inputs' dataset names {names} say "
                         f"{is_synthetic} (synthetic iff a name is not the evaluator's {DATASET_NAME!r})")
    return arts


def _runtime(art) -> dict:
    rt = art.summary["run"].get("eval_runtime")
    return rt if isinstance(rt, dict) else {}


def device_class(art) -> list:
    """(run.env.device, the device type of eval_runtime.model_device, eval_runtime.gpu_name)."""
    run, rt = art.summary["run"], _runtime(art)
    md = rt.get("model_device")
    return [(run.get("env") or {}).get("device"), md.split(":", 1)[0] if isinstance(md, str) else None,
            rt.get("gpu_name")]


def check_commit(arts: dict) -> list:
    commits = {s: arts[s].summary["run"].get("repo_commit") for s in SEEDS}
    return ["one evaluate_model.py commit (run.repo_commit) across the three re-scores (AM-17 item 3(a): 'at one "
            "pinned commit')",
            len(set(commits.values())) == 1 and all(isinstance(c, str) and HEX40.fullmatch(c) is not None
                                                    for c in commits.values()), commits]


def check_device_class(arts: dict) -> list:
    classes = {s: device_class(arts[s]) for s in SEEDS}
    return ["one device class (run.env.device, model device type, gpu_name) across the three (AM-17 item 3(a): "
            "'on the same device')",
            len({tuple(v) for v in classes.values()}) == 1 and all(None not in v for v in classes.values()),
            classes]


def check_inputs_on_model_device(arts: dict, s: str) -> list:
    rt = _runtime(arts[s])
    md = rt.get("model_device")
    return [f"s{s}: inputs on the model device (input_devices == [model_device]); a consistency check implied by "
            "'one pinned commit': check_post_eval enforces it (src/eval/eval_runtime.py:195-205)",
            isinstance(md, str) and rt.get("input_devices") == [md],
            {"model_device": md, "input_devices": rt.get("input_devices")}]


def check_policy_applied(arts: dict, s: str) -> list:
    rt = _runtime(arts[s])
    return [f"s{s}: the determinism policy applied; a consistency check implied by 'one pinned commit': the policy "
            "is unconditional for an FP32 student on CUDA (src/eval/eval_runtime.py:88-91)",
            rt.get("determinism_policy_applied") is True, rt.get("determinism_policy_applied")]


def check_image_digest(arts: dict) -> list:
    digests = {s: _runtime(arts[s]).get("image_digest") for s in SEEDS}
    return ["one non-null image digest across the three (lane 6 (a) 'in-chain A40 re-scores' and DL-17's pinned "
            "image)",
            len(set(digests.values())) == 1 and all(isinstance(v, str) and v for v in digests.values()), digests]


def check_dl17(arts: dict) -> list:
    problems = CEA.validity_problems("s42", arts["42"].summary)
    return ["s42 is a valid DL-17 B66 re-score (scripts/compare_eval_artifacts.py validity_problems)",
            not problems, problems]


def check_dl17_delta(arts: dict) -> list:
    miou = arts["42"].summary["dataset_level"].get("all_class_miou")
    delta = abs(miou - CEA.DL17_REFERENCE_MIOU) if type(miou) is float and math.isfinite(miou) else None
    return [f"s42 all-class VAL mIoU within DL17_BAND {CEA.DL17_BAND} of the DL-17 reference "
            f"{CEA.DL17_REFERENCE_MIOU!r} (DL-17 PASS)",
            delta is not None and delta <= CEA.DL17_BAND, {"all_class_miou": miou, "abs_delta": delta}]


def check_manifest_pin(arts: dict, s: str, pin: str) -> list:
    got = arts[s].file_sha256s[MANIFEST_NAME]
    return [f"s{s}: the sha256 of MANIFEST.sha256 equals --expect-manifest-sha256-{s}", got == pin,
            {"got": got, "expected": pin}]


def provenance_checks(arts: dict, args) -> list:
    """Decision (ii): one evaluate_model.py commit, one device class and one image digest across the three, with
    their reasons; seed 42 the DL-17 B66 re-score; the optional MANIFEST pins. Each is [name, passed, detail]."""
    out = [check_commit(arts), check_device_class(arts)]
    for s in SEEDS:
        out += [check_inputs_on_model_device(arts, s), check_policy_applied(arts, s)]
    out += [check_image_digest(arts), check_dl17(arts), check_dl17_delta(arts)]
    for s in SEEDS:
        pin = getattr(args, f"expect_manifest_sha256_{s}")
        if pin is not None:
            out.append(check_manifest_pin(arts, s, pin))
    return out


def require_passed(checks: list, what: str) -> None:
    bad = [c[0] for c in checks if c[1] is not True]
    if bad:
        raise td.Refused(f"{what}: {len(bad)} check(s) failed (no re-score without a ruling): {bad}")


def am5_and_order(arts: dict, args) -> dict:
    """Lane 6 (a): identical excluded sets (lane 6 (f) STOP otherwise), then one identical image order."""
    a5 = {s: arts[s].run.am5 for s in SEEDS}
    if len({a5[s].excluded_ids for s in SEEDS}) != 1:
        detail = {f"s{s}": {"excluded": a5[s].excluded_count, "sha256": a5[s].excluded_ids_sha256} for s in SEEDS}
        raise td.Refused(f"STOP (lane 6 (f)): the AM-5 excluded sets differ across the seeds (a manifest "
                         f"mismatch): {detail}")
    orders = {tuple(r.image_id for r in arts[s].run.records) for s in SEEDS}
    if len(orders) != 1:
        raise td.Refused("the included images are not in one identical order across the three seeds (lane 6 (a))")
    k, included = a5["42"].excluded_count, a5["42"].included_count
    if args.expect_k_val is not None and k != args.expect_k_val:
        raise td.Refused(f"K_val is {k}, --expect-k-val says {args.expect_k_val}")
    return {"K_excluded": k, "n_included": included, "excluded_ids_sha256": a5["42"].excluded_ids_sha256,
            "am5_rule": a5["42"].rule, "am5_source": {s: a5[s].source for s in SEEDS}}


# --------------------------------------------------------------------------------------------------
# 9 the field of item 3(a), re-derived
# --------------------------------------------------------------------------------------------------
def derivation_check(arts: dict) -> dict:
    """Decision (i): per image, the GT-present mean IoU over disease classes 1..115 from the sufficient
    statistics (float32 tp / (gt + pred - tp), as metrics._miou_from_cm) equals the stored disease_only_miou
    within F32_TOL, and its class count equals n_eligible_disease_only."""
    worst, rows, bad = 0.0, 0, []
    for s in SEEDS:
        run = arts[s].run
        st = run.stats
        pos = {iid: i for i, iid in enumerate(st.manifest_ids)}
        keep = (st.class_id >= 1) & (st.gt > 0)
        img = st.image_index[keep]
        tp, gt, pr = (a[keep].astype(np.float32) for a in (st.tp, st.gt, st.pred))
        iou = (tp / (gt + pr - tp)).astype(np.float64)
        size = len(st.manifest_ids)
        sums = np.bincount(img, weights=iou, minlength=size)
        cnt = np.bincount(img, minlength=size)
        for r in run.records:
            p = pos[r.image_id]
            if cnt[p] != r.n_eligible_disease_only or cnt[p] == 0:
                bad.append((f"s{s}", r.image_id, "class count", int(cnt[p]), r.n_eligible_disease_only))
                continue
            diff = abs(float(sums[p] / cnt[p]) - float(r.disease_only_miou))
            rows += 1
            worst = max(worst, diff)
            if diff > F32_TOL:
                bad.append((f"s{s}", r.image_id, "value", diff))
    result = {"rule": "per image: mean over classes 1..115 with gt > 0 of float32 tp / (gt + pred - tp), from "
                      "sufficient_stats.npz, against per_image.jsonl disease_only_miou",
              "rows_checked": rows, "max_abs_diff": worst, "tolerance": F32_TOL, "mismatches": len(bad)}
    if bad:
        raise td.Stop(f"the stored per-image disease-only mIoU disagrees with its re-derivation from the "
                      f"sufficient statistics ({len(bad)} row(s), e.g. {bad[:3]}): decision (i) fails")
    return result


# --------------------------------------------------------------------------------------------------
# 11 the pairs and the MDE
# --------------------------------------------------------------------------------------------------
def pair_vectors(arts: dict) -> dict:
    """The three pairs, delta = candidate - baseline in align_runs' sorted-id order (AM-5 rows dropped)."""
    out = {}
    for name, b, c in PAIRS:
        try:
            pv = align_runs(arts[b].run, arts[c].run, policy=Policy.REHEARSAL, metric=METRIC_DISEASE_ONLY)
        except AlignmentError as e:
            raise td.Refused(f"pair {name}: {e}") from e
        if pv.alignment_status != "ok" or pv.am5 is None:
            raise td.Refused(f"pair {name}: alignment status {pv.alignment_status!r}, am5 {pv.am5!r}")
        out[name] = pv
    if len({pv.image_ids for pv in out.values()}) != 1:
        raise td.Refused("the three pairs do not share one image order")
    return out


def compute_mde(arts: dict, included: int) -> tuple[dict, int, str]:
    vectors = pair_vectors(arts)
    n = M.n_planning(included)
    pairs = {}
    for name, b, c in PAIRS:
        pv = vectors[name]
        if pv.n != included:
            raise td.Refused(f"pair {name}: {pv.n} paired images, n_included is {included}")
        pairs[name] = {"baseline": f"s{b}", "candidate": f"s{c}", **M.pair_summary(pv.delta, n)}
    ids_sha = hashlib.sha256("\n".join(next(iter(vectors.values())).image_ids).encode("utf-8")).hexdigest()
    missing = [name for name in pairs if pairs[name]["mde_w"] is None]
    if missing:
        for name in pairs:
            print(f"rejection counts {name} (of {M.B}; need {pairs[name]['power_min_count']} at three "
                  f"consecutive points): {pairs[name]['rejections']}")
        raise td.Stop(f"MDE_W is not determined on the pre-registered grid for {missing}: no delta in "
                      "0.000..0.048 has power >= 0.80 at it and at the next two grid points (decision (v); a "
                      "ruling is needed; nothing is written)")
    return pairs, n, ids_sha


# --------------------------------------------------------------------------------------------------
# 10 and 12 the DL-27 band: its inputs, then its file read back through its reader
# --------------------------------------------------------------------------------------------------
def band_sd(xs) -> float:
    """The reader's s: the n = 3 sample SD (sweep_select.dl27_band)."""
    return statistics.stdev(xs)


def band_value(s: float, floor: float) -> float:
    """The reader's band: max(floor, sqrt(2) x s)."""
    return max(float(floor), math.sqrt(2.0) * s)


def build_band(values: dict, floor: float) -> tuple[dict, bytes]:
    xs = [values[s] for s in SEEDS]
    s = band_sd(xs)
    band = band_value(s, floor)
    doc = {"e1_best_val": {k: values[k] for k in SEEDS}, "s": s, "band": band}
    return doc, td.json_bytes(doc)


def band_floor() -> float:
    try:
        rules = SS.load_rules()
    except SS.SelectionRefused as e:
        raise td.Refused(f"configs/sweep_rules.json: {e.code}: {e}") from e
    band = rules["alpha_cwd"]["band"]
    floor = band.get("floor")
    if band.get("kind") != "dl27" or band.get("file") != BAND_RULE_FILE or isinstance(floor, bool) \
            or not isinstance(floor, (int, float)) or float(floor) != BAND_FLOOR:
        raise td.Refused(f"configs/sweep_rules.json alpha_cwd.band {band} is not the DL-27 rule (kind dl27, file "
                         f"{BAND_RULE_FILE}, floor {BAND_FLOOR})")
    return float(floor)


def read_best_json(path) -> dict:
    try:
        doc = SS.read_json(Path(path), code="best_json_format")
    except SS.SelectionRefused as e:
        raise td.Refused(f"{path}: {e}") from e
    if not isinstance(doc, dict) or sorted(doc) != SS.BEST_JSON_KEYS:
        got = sorted(doc) if isinstance(doc, dict) else type(doc).__name__
        raise td.Refused(f"{path}: keys {got} != {SS.BEST_JSON_KEYS} (train_e1's best.json)")
    v = doc["best_val_miou_all_class"]
    if type(v) is not float or not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise td.Refused(f"{path}: best_val_miou_all_class {v!r} is not a finite float in [0, 1]")
    if not isinstance(doc["best_ckpt"], str) or not doc["best_ckpt"]:
        raise td.Refused(f"{path}: best_ckpt {doc['best_ckpt']!r} is not a path")
    return doc


def _basename(p) -> str | None:
    return re.split(r"[\\/]", p)[-1] if isinstance(p, str) and p else None


def band_inputs(args, arts: dict) -> tuple[dict, list, dict]:
    """The three E1 best values, gated on equality with the artifact's own recorded checkpoint value (the
    trainer's float in both places, train_e1.py:505-509); the basename comparison is recorded, not gated."""
    values, checks, files = {}, [], {}
    for s in SEEDS:
        path = Path(getattr(args, f"best_json_{s}"))
        doc = read_best_json(path)
        v = doc["best_val_miou_all_class"]
        rec = _runtime(arts[s]).get("checkpoint_best_val_miou_all_class")
        checks.append([f"s{s}: best.json best_val_miou_all_class == the artifact's eval_runtime."
                       "checkpoint_best_val_miou_all_class", type(rec) is float and rec == v,
                       {"best_json": v, "artifact": rec}])
        a, b = _basename(doc["best_ckpt"]), _basename(arts[s].summary["run"].get("checkpoint_path"))
        values[s] = v
        files[s] = {"sha256": td.file_sha256(path), "best_ckpt_basename": a,
                    "best_ckpt_basename_matches_checkpoint_path": a is not None and a == b}
    require_passed(checks, "the band inputs")
    return values, checks, files


def band_self_check(band_bytes: bytes, floor: float, s: float, band: float) -> list:
    try:
        s2, band2, trace = SS.dl27_band(json.loads(band_bytes.decode("utf-8")), floor)
    except SS.SelectionRefused as e:
        raise td.Stop(f"the band file is refused by its reader sweep_select.dl27_band: {e.code}: {e}") from e
    if repr(s2) != repr(s) or repr(band2) != repr(band):
        raise td.Stop(f"the reader recomputes (s, band) = ({s2!r}, {band2!r}), the writer has ({s!r}, {band!r})")
    return trace


# --------------------------------------------------------------------------------------------------
# 12 the entry
# --------------------------------------------------------------------------------------------------
def artifact_record(art, s: str) -> dict:
    run, rt = art.summary["run"], _runtime(art)
    return {"run_id": art.run_id, "dir_name": art.path.name, "sha256s": dict(art.file_sha256s),
            "checkpoint_sha256": run.get("checkpoint_sha256"), "repo_commit": run.get("repo_commit"),
            "metric_impl_sha256": run.get("metric_impl_sha256"), "device_class": device_class(art),
            "image_digest": rt.get("image_digest"), "batch_size": rt.get("batch_size"),
            "forward_batches": rt.get("forward_batches"), "cudnn_version": rt.get("cudnn_version"),
            "torch_cuda": rt.get("torch_cuda"), "all_class_miou": art.summary["dataset_level"].get("all_class_miou"),
            "checkpoint_best_val_miou_all_class": rt.get("checkpoint_best_val_miou_all_class"),
            "artifact_schema_version": art.summary.get("artifact_schema_version"),
            "role": "DL-17 B66 re-score (run1)" if s == "42" else "in-chain A40 re-score"}


def _g(x) -> str:
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def decision_log_line(entry: dict, entry_name: str, entry_sha: str | None, band_name: str, band_sha: str) -> str:
    p = entry["pairs"]
    names = [name for name, _, _ in PAIRS]
    b = entry["dl27_band"]
    vals = b["e1_best_val"]
    return (f"AM-17 item 3 MDE entry {entry_name}" + (f" sha256 {entry_sha}" if entry_sha else "")
            + f": n_included {entry['n_included']} (K {entry['K_excluded']}); n_planning {entry['n_planning']}; "
            f"shifted-null MDE_W {_g(entry['mde_w'])} ("
            + ", ".join(f"{n.replace('-', '−')} {_g(p[n]['mde_w'])}" for n in names)
            + "; share of exact-zero differences " + "/".join(_g(p[n]["share_ties"]) for n in names)
            + ", n_zero " + "/".join(str(p[n]["n_zero"]) for n in names)
            + f"; B 2,000 resamples of size n from default_rng(42) — one index matrix per pair, identical to "
            f"re-seeding per (pair, δ), so the three guard points share rows; the pre-registered Wilcoxon call, "
            f"α 0.00625, p < α, 3-point guard; scipy {entry['scipy_version']}, numpy {entry['numpy_version']}); "
            f"SD_Δ {_g(entry['sd_delta'])}, dz_MDE {_g(entry['dz_mde'])}, MDE_t {_g(entry['mde_t'])} "
            f"[{_g(entry['mde_t_range'][0])}, {_g(entry['mde_t_range'][1])}]; τ_P 0.010; power_caveat "
            f"{entry['power_caveat']}; label: planning proxy (item 3(e)); lane 6 (f): exact ties deflate MDE_W "
            f"under the constant-shift null. Band: {band_name} sha256 {band_sha}; E1 best VAL 42/43/44 "
            f"{vals['42']!r}/{vals['43']!r}/{vals['44']!r} (best.json sha256s in the entry); s {b['s']!r}; band "
            f"= max(0.005, √2·s) = {b['band']!r}. Code {entry['git_commit']} ({entry['script_commit_dl_id']}).")


def assemble(*, args, synthetic, generated, arts, checks, am5, derivation, pairs, n, ids_sha, band_doc,
             band_sha, band_files, band_checks, band_trace, floor, env, binding, band_name) -> dict:
    import scipy
    mde_w, mde_pair = M.overall(pairs)
    ana = M.analytic([pairs[name]["sd"] for name, _, _ in PAIRS], n)
    return {
        "schema": SCHEMA, "lane": LANE, "script": SCRIPT, "authority": AUTHORITY,
        "artifact_status": "smoke" if synthetic else "provisional", "synthetic_input_data": synthetic,
        "generated_utc": generated,
        "artifacts": {s: artifact_record(arts[s], s) for s in SEEDS},
        "provenance_checks": checks,
        "n_val": EXPECTED_ROWS_VAL, "n_included": am5["n_included"], "K_excluded": am5["K_excluded"],
        "excluded_ids_sha256": am5["excluded_ids_sha256"], "am5_rule": am5["am5_rule"],
        "am5_source": am5["am5_source"], "included_ids_sha256": ids_sha,
        "included_ids_order": "sorted image_id (src/stats/align.py align_runs)",
        "n_planning": n, "n_planning_rule": M.n_planning_rule(am5["n_included"]),
        "grid": {"delta": "k / 1000 for k = 0..50 (0.000 ... 0.050)", "points": len(M.GRID),
                 "candidates": "k = 0..48 (the guard needs k + 1 and k + 2 on the grid)"},
        "resampling": {"B": M.B, "size": n, "rng": M.INDEX_CALL, "drawn_once_per_pair": True,
                       "equivalent_to": EQUIVALENCE, "guard_points_share_rows": True,
                       "numpy_version": np.__version__},
        "test": {"call": M.WILCOXON_CALL, "alpha": M.ALPHA, "rejection": "p < alpha", "power": "rejections / B",
                 "guard": "power >= 0.80 at delta and at the next two grid points (MC noise guard over shared rows)",
                 "power_min_count": M.min_count(M.B), "centring": "none: d + delta (item 3(c); lane 6 (a))"},
        "pairs": pairs,
        "mde_w": mde_w, "mde_w_pair": mde_pair,
        "sd_delta": ana["sd_delta"], "dz_mde": ana["dz_mde"], "dz_formula": ana["dz_formula"],
        "mde_t": ana["mde_t"], "mde_t_range": ana["mde_t_range"],
        "tau_p": TAU_P, "power_caveat": M.power_caveat(mde_w),
        "label": LABEL, "ties_note": TIES_NOTE,
        "derivation_check": derivation,
        "dl27_band": {"file": band_name, "sha256": band_sha, "e1_best_val": band_doc["e1_best_val"],
                      "s": band_doc["s"], "band": band_doc["band"], "floor": floor,
                      "rule": "band = max(floor, sqrt(2) * s); s = statistics.stdev of the three values "
                              "(sweep_select.dl27_band)",
                      "best_json": band_files, "cross_checks": band_checks, "reader_trace": band_trace},
        "scipy_version": scipy.__version__, "numpy_version": np.__version__, "software_environment": env,
        "seed": M.RNG_SEED, "git_commit": td.git_head(), "code": code_provenance(CODE_FILES),
        "script_commit": (binding or {}).get("script_commit", args.script_commit),
        "script_commit_dl_id": (binding or {}).get("script_commit_dl_id", args.script_commit_dl_id),
    }


# --------------------------------------------------------------------------------------------------
def run(args) -> int:
    return td.run_with_exit_codes(_run, args)


def _run(args) -> int:
    start_utc = td.utc_now()
    synthetic = bool(args.synthetic_inputs)
    guard_paths({f: getattr(args, f) for f in PATH_FLAGS})                    # 1: no filesystem call before
    check_flags(args, synthetic)                                              # 2
    out = td.require_outside_repo(args.out_dir, "--out-dir")                  # 3
    entry_name, band_name = output_names(args, synthetic, start_utc)
    existing = [name for name in (entry_name, band_name) if os.path.lexists(out / name)]
    if existing:
        raise td.Refused(f"--out-dir already holds {existing}; an output is written once")
    binding = None if synthetic else commit_binding(args)                     # 4
    env = environment_check(synthetic)
    require_inputs(args)                                                      # 5
    arts = load_artifacts(args, synthetic)                                    # 6
    checks = provenance_checks(arts, args)                                    # 7
    require_passed(checks, "provenance")
    am5 = am5_and_order(arts, args)                                           # 8
    derivation = derivation_check(arts)                                       # 9
    floor = band_floor()                                                      # 10 (before the long run)
    values, band_checks, band_files = band_inputs(args, arts)
    pairs, n, ids_sha = compute_mde(arts, am5["n_included"])                  # 11
    band_doc, band_bytes = build_band(values, floor)                          # 12
    band_trace = band_self_check(band_bytes, floor, band_doc["s"], band_doc["band"])
    band_sha = hashlib.sha256(band_bytes).hexdigest()
    entry = assemble(args=args, synthetic=synthetic, generated=args.generated_utc or start_utc, arts=arts,
                     checks=checks, am5=am5, derivation=derivation, pairs=pairs, n=n, ids_sha=ids_sha,
                     band_doc=band_doc, band_sha=band_sha, band_files=band_files, band_checks=band_checks,
                     band_trace=band_trace, floor=floor, env=env, binding=binding, band_name=band_name)
    entry["decision_log_line"] = decision_log_line(entry, entry_name, None, band_name, band_sha)
    entry_bytes = td.json_bytes(entry)
    problems = validate_mde_entry(json.loads(entry_bytes.decode("utf-8")))     # 12
    if problems:
        raise td.Stop(f"the entry fails the report layer's validator: {list(problems)}")
    td.write_files_exclusive([(out / entry_name, entry_bytes), (out / band_name, band_bytes)])
    entry_sha = hashlib.sha256(entry_bytes).hexdigest()
    print(f"written: {entry_name} sha256 {entry_sha}")
    print(f"written: {band_name} sha256 {band_sha}")
    for name, _, _ in PAIRS:
        p = pairs[name]
        print(f"pair {name}: m {p['m']}, n {p['n']}, MDE_W {p['mde_w']}, share_ties {p['share_ties']:.6g}, "
              f"n_zero {p['n_zero']}, mean {p['mean']:.6g}, sd {p['sd']:.6g}")
    print(f"MDE_W {entry['mde_w']} ({entry['mde_w_pair']}); MDE_t {entry['mde_t']:.6g}; power_caveat "
          f"{entry['power_caveat']}; band {band_doc['band']!r}")
    print("DL line: " + decision_log_line(entry, entry_name, entry_sha, band_name, band_sha))
    return td.EXIT_OK


def main(argv=None) -> int:
    return td.cli_main(build_parser(), run, argv)


if __name__ == "__main__":
    sys.exit(main())
