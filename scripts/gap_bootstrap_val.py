#!/usr/bin/env python3
"""L-AM17B-GAP: the teacher of record minus E1 seed 42 on VAL -- paired image-level BCa intervals of the
dataset-level mIoU gap under the union-present and GT-present rules, and the per-image difference
summary (AM-17b item 2; docs/lane_specs/part1.md lane 7; DL-35 G3 and G5). Descriptive: it gates,
selects and replaces nothing.

    python -B scripts/gap_bootstrap_val.py --teacher TEACHER_CPU_ARTIFACT --e1 E1_CPU_ARTIFACT \\
        --eligibility-variants TEACHER_ELIGIBILITY.json E1_ELIGIBILITY.json \\
        [--e1-tolerance {1e-7,1e-5}] [--out-dir reports/derived] [--generated-utc T]
    python -B scripts/gap_bootstrap_val.py --check-artifact RESCORE_DIR --role {e1,teacher}

Inputs: the two AM-17b item 2(c) CPU re-scores and the L-AM17-GTPRESENT outputs
(scripts/eligibility_variants.py) for the same two artifacts. No dataset, no model, no TEST.

1. Pairing of record (exit 2 when violated; not a verdict). Both artifacts verify and are read under
   Policy.REHEARSAL (VAL, 846 rows, real-run) through src/stats/val_artifacts.py: canvas only (an
   upstream-protocol artifact is refused), a layout plantseg-eval-artifact/1.x at or after 1.1.0 (the
   ingest reader's rule; the re-scores are written after the L-AM5 merge). Per artifact
   (`pairing_checks`): the role's stage, model role and fp32; its checkpoint of record (teacher
   8c0e649a..., E1 seed 42 cf0879f7...); a run.eval_runtime record; the CPU (env.device, model and
   input devices); batch size 1; its pinned image (eval_runtime.image_digest: teacher cb413304...,
   student b80b645d...); clean governed paths; a recorded repo_commit. For the pair: one repo_commit,
   the same VAL manifest and metric implementation, identical per-image ground truth.
2. Validity (lane 7 (a); STOP, exit 1, nothing written): |E1 all-class mIoU - 0.36307525634765625|
   (the Q12 CPU value) <= --e1-tolerance, and |teacher all-class mIoU - 0.38576993346214294| <= 1e-5
   (R3; a different CPU/BLAS than the pod). --e1-tolerance accepts 1e-7 (the default) or 1e-5 only
   (ruling 2026-09-28): 1e-7 when Q12 ran at batch size 1; 1e-5 when it did not or its batch size is
   unknown, recorded as an erratum before the re-score runs. Any other value is refused (exit 2). The
   tolerance used and both deltas are printed and recorded.
3. d5 cross-check (STOP, exit 1): each point estimate equals the difference of the two L-AM17-GTPRESENT
   `rules_float64` values within 1e-12, for both rules and both class subsets (S1 ruling). Each
   eligibility file must name the input artifact's own four file hashes (exit 2 otherwise).
4. Four bootstraps (src/stats/gap.py: B = 10,000, default_rng(42), BCa; the percentile interval of the
   same replicates when the jackknife acceleration is non-finite, when BCa raises, or when BCa returns
   a non-finite bound -- SciPy warns DegenerateDataWarning and returns NaN instead of raising; each
   interval's fallback_reason names the condition) and the per-image summary on the AM-5-included
   images. Written once to <out-dir>/gap_val_<UTC>.json, serialized before the file is created, so a
   failed write leaves no partial file; an existing output is never overwritten. The output records
   the analysis code's commit, whether those files were clean at it, and each file's sha256.

--check-artifact DIR --role {e1,teacher} (ruling 2026-09-28) validates ONE re-score's pairing fields
in seconds and computes nothing: canvas, VAL with 846 rows, a real-run artifact, layout 1.x at or after
1.1.0, every per-artifact check of step 1 for the role, repo_commit equal to this checkout's HEAD (the
pin both re-scores use), and a clean read by the statistics ingest. The runbook runs it right after
the E1 re-score and again after the teacher re-score, so a missing image digest is caught after the
4-minute E1 run rather than after the 1-hour teacher run. Its validity delta is printed for
information only; the gap run gates it.

Exit codes: 0 written / PASS; 1 STOP (a validity delta above its tolerance, or the d5 cross-check
failed: evaluator regression or an unfaithful reconstruction -- report it) or, for --check-artifact, a
pairing field that fails (fix the configuration before the next re-score); 2 refused or unreadable
input, a usage error, or any other error (never a verdict).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2
LANE = "L-AM17B-GAP"
DEFAULT_OUT_DIR = REPO / "reports" / "derived"
OUT_PREFIX = "gap_val_"
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
D5_TOLERANCE = 1e-12
DEVICE, BATCH_SIZE, VAL_ROWS = "cpu", 1, 846
ROLES = ("e1", "teacher")
RULE_KEYS, SCOPE_KEYS = ("union_present", "gt_present"), ("all_class", "disease_only")
#: --e1-tolerance (ruling 2026-09-28): 1e-7 when Q12 ran at batch size 1 (the default); 1e-5 when it
#: did not or its batch size is unknown, recorded as an erratum before the re-score runs.
E1_TOLERANCES = (1e-7, 1e-5)
E1_TOLERANCE_RULE = ("1e-7 when the Q12 E1 CPU re-score ran at batch size 1; 1e-5 when it did not or "
                     "its batch size is unknown, recorded as an erratum before the re-score runs "
                     "(ruling 2026-09-28)")
#: the analysis code recorded with every output (commit, clean state and sha256 of each file)
CODE_FILES = ("scripts/gap_bootstrap_val.py", "src/stats/gap.py", "src/stats/val_artifacts.py",
              "src/stats/eligibility.py", "src/stats/noninferiority.py", "src/stats/align.py",
              "src/stats/ingest.py", "src/stats/tests.py")
BOOTSTRAP_CALL = ("scipy.stats.bootstrap((idx,), stat, n_resamples=10000, method='BCa', "
                  "random_state=numpy.random.default_rng(42), vectorized=False); the percentile interval "
                  "of the same replicates when the jackknife acceleration is non-finite, when BCa raises, "
                  "or when BCa returns a non-finite bound (SciPy warns DegenerateDataWarning and returns "
                  "NaN instead of raising); fallback_reason names the condition")

#: The pairing of record (AM-17 item 1(a), DL-17, DL-21, AM-17b item 2(c)) and the lane 7 (a) validity
#: references. E1's tolerance is --e1-tolerance; the teacher's is fixed.
PAIRING = {
    "teacher": {"stage": "teacher", "model_role": "teacher",
                "checkpoint_sha256": "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e",
                "image_digest": "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b",
                "reference": 0.38576993346214294, "tolerance": 1e-5,
                "reference_source": "R3, the teacher of record on VAL (CPU, batch 1, evaluator 3c43f89)"},
    "e1": {"stage": "E1", "model_role": "student",
           "checkpoint_sha256": "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03",
           "image_digest": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
           "reference": 0.36307525634765625,
           "reference_source": "the B66-prep Q12 CPU re-score of E1 seed 42"},
}


class Refused(RuntimeError):
    """The inputs are not the specified pairing: not a verdict."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkout_head() -> str | None:
    """HEAD of this checkout: the pin at which both re-scores must be made (None without Git)."""
    from src.stats.val_artifacts import head_commit
    return head_commit()


# --------------------------------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------------------------------
def pairing_checks(summary: dict, role: str) -> list[tuple[str, bool, str]]:
    """Every per-artifact pairing field of `role`, as [(check, passed, detail)]."""
    spec, run = PAIRING[role], summary.get("run") or {}
    rt = run.get("eval_runtime") if isinstance(run.get("eval_runtime"), dict) else {}
    got_role = (run.get("stage"), run.get("model_role"), run.get("precision"))
    devices = ((run.get("env") or {}).get("device"), rt.get("model_device"), rt.get("input_devices"))
    commit = run.get("repo_commit")
    return [
        (f"role {role}: stage {spec['stage']}, model_role {spec['model_role']}, precision fp32",
         got_role == (spec["stage"], spec["model_role"], "fp32"), f"got {got_role}"),
        (f"checkpoint sha256 of record ({spec['checkpoint_sha256'][:12]}...)",
         run.get("checkpoint_sha256") == spec["checkpoint_sha256"], f"got {run.get('checkpoint_sha256')!r}"),
        ("run.eval_runtime present (the AM-17b item 2(c) re-score writes one)", bool(rt), ""),
        ("CPU re-score: env.device, model_device and input_devices are cpu",
         devices == (DEVICE, DEVICE, [DEVICE]), f"got {devices}"),
        ("batch size 1", rt.get("batch_size") == BATCH_SIZE, f"got {rt.get('batch_size')!r}"),
        (f"image digest present and equal to the pinned {role} image",
         rt.get("image_digest") == spec["image_digest"],
         f"got {rt.get('image_digest')!r}, expected {spec['image_digest']} "
         "(docker run -e PLANTSEG_IMAGE_DIGEST=...)"),
        ("governed paths clean at scoring time", run.get("governed_paths_clean") is True,
         f"got {run.get('governed_paths_clean')!r}"),
        ("repo_commit recorded (40-hex)", isinstance(commit, str) and bool(_HEX40.match(commit)),
         f"got {commit!r}"),
    ]


def check_pairing(art_t, art_e) -> None:
    from src.stats.val_artifacts import require_comparable

    for role, art in (("teacher", art_t), ("e1", art_e)):
        for name, ok, detail in pairing_checks(art.summary, role):
            if not ok:
                raise Refused(f"{role}: {name} -- {detail}")
    pins = (art_t.summary["run"]["repo_commit"], art_e.summary["run"]["repo_commit"])
    if pins[0] != pins[1]:
        raise Refused(f"the two re-scores were made at different commits {pins}; both run at one pin")
    require_comparable(art_e, art_t)


def artifact_checks(run_dir: Path, summary: dict, role: str) -> list[tuple[str, bool, str]]:
    """The fourteen --check-artifact checks (A1-A14) of one re-score whose manifest verified."""
    from src.eval.protocols import CANVAS_PROTOCOL_ID
    from src.stats.ingest import _layout
    from src.stats.val_artifacts import AM5_LAYOUT, load_val_artifact

    run, ds = summary.get("run") or {}, summary.get("dataset") or {}
    results: list[tuple[str, bool, str]] = []
    results.append(("canvas protocol: no summary.protocol block, preprocess_protocol core_preprocess/1.0.0",
                    "protocol" not in summary and ds.get("preprocess_protocol") == CANVAS_PROTOCOL_ID,
                    f"got preprocess_protocol {ds.get('preprocess_protocol')!r}"))
    results.append((f"VAL split with {VAL_ROWS} rows",
                    ds.get("split") == "val" and ds.get("expected_rows") == ds.get("actual_rows") == VAL_ROWS,
                    f"got split {ds.get('split')!r}, rows {ds.get('actual_rows')!r}"))
    results.append(("a real-run artifact: provisional or official, not random-init",
                    run.get("artifact_status") in ("provisional", "official")
                    and run.get("random_init") is False,
                    f"got {run.get('artifact_status')!r}, random_init {run.get('random_init')!r}"))
    try:
        layout = _layout(summary)
    except Exception:                                 # noqa: BLE001 -- another major: a failed check
        layout = None
    results.append(("layout plantseg-eval-artifact/1.x at or after 1.1.0",
                    layout is not None and layout >= AM5_LAYOUT,
                    f"got {summary.get('artifact_schema_version')!r}"))
    results += pairing_checks(summary, role)
    head = checkout_head()
    results.append(("repo_commit equals this checkout's HEAD (the pin both re-scores use)",
                    head is not None and run.get("repo_commit") == head,
                    f"artifact {run.get('repo_commit')!r}, HEAD {head!r}"))
    try:
        load_val_artifact(run_dir, label=role, min_layout=AM5_LAYOUT)
        results.append(("the statistics ingest reads it under REHEARSAL (rows, AM-5 flags, "
                         "sufficient statistics)", True, ""))
    except Exception as e:                            # noqa: BLE001 -- reported as a failed check
        results.append(("the statistics ingest reads it under REHEARSAL (rows, AM-5 flags, "
                        "sufficient statistics)", False, f"{type(e).__name__}: {e}"))
    return results


def check_artifact(run_dir: Path, role: str) -> int:
    """--check-artifact: every pairing field of one re-score. 0 PASS, 1 FAIL, 2 unreadable or malformed."""
    from src.stats.ingest import verify_run_manifest

    try:
        summary = verify_run_manifest(run_dir)
    except Exception as e:                            # noqa: BLE001 -- unreadable, never a verdict
        print(f"RESULT: ERROR -- {run_dir} cannot be read: {type(e).__name__}: {e}")
        return EXIT_REFUSED
    try:
        results = artifact_checks(run_dir, summary, role)
    except Exception as e:                            # noqa: BLE001 -- a malformed summary, never a verdict
        print(f"RESULT: ERROR -- {run_dir} cannot be checked (malformed summary.json): "
              f"{type(e).__name__}: {e}")
        return EXIT_REFUSED
    run = summary.get("run") or {}
    print(f"check-artifact {role}: {run_dir} (run_id {run.get('run_id')!r}, repo_commit "
          f"{run.get('repo_commit')!r})")
    for i, (name, ok, detail) in enumerate(results, 1):
        print(f"  [{'PASS' if ok else 'FAIL'}] A{i} {name}")
        if detail and not ok:
            print(f"         {detail}")
    level = summary.get("dataset_level")
    v = level.get("all_class_miou") if isinstance(level, dict) else None
    if isinstance(v, float):
        ref = PAIRING[role]["reference"]
        tols = E1_TOLERANCES if role == "e1" else (PAIRING["teacher"]["tolerance"],)
        within = ", ".join(f"within {t:g}: {'yes' if abs(v - ref) <= t else 'no'}" for t in tols)
        print(f"  info (not gated here): all-class mIoU {v!r} vs reference {ref!r}, delta {v - ref:+.3e}; "
              f"{within}")
    good = sum(1 for _, ok, _ in results if ok)
    allok = good == len(results)
    print(f"RESULT: {'ARTIFACT PAIRING PASS' if allok else 'ARTIFACT PAIRING FAIL'} ({role}, "
          f"{good}/{len(results)})")
    return EXIT_OK if allok else EXIT_STOP


# --------------------------------------------------------------------------------------------------
# validity, eligibility files, d5
# --------------------------------------------------------------------------------------------------
def validity(art_t, art_e, e1_tolerance: float = E1_TOLERANCES[0]) -> dict:
    if e1_tolerance not in E1_TOLERANCES:
        raise Refused(f"--e1-tolerance must be one of {E1_TOLERANCES}, got {e1_tolerance!r}")
    tolerance = {"e1": e1_tolerance, "teacher": PAIRING["teacher"]["tolerance"]}
    out = {}
    for label, art in (("e1", art_e), ("teacher", art_t)):
        spec = PAIRING[label]
        v = art.summary["dataset_level"]["all_class_miou"]
        delta = v - spec["reference"]
        out[label] = {"value": v, "reference": spec["reference"], "delta": delta, "abs_delta": abs(delta),
                      "tolerance": tolerance[label], "within": abs(delta) <= tolerance[label],
                      "reference_source": spec["reference_source"]}
    return out


def load_eligibility(path: Path, art, label: str) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rf = doc.get("rules_float64")
    if doc.get("lane") != "L-AM17-GTPRESENT" or not isinstance(rf, dict):
        raise Refused(f"{label}: {path} is not an L-AM17-GTPRESENT eligibility-variants file")
    for rule in RULE_KEYS:
        for scope in SCOPE_KEYS:
            v = rf[rule].get(scope) if isinstance(rf.get(rule), dict) else None
            if not isinstance(v, float) or not math.isfinite(v):
                raise Refused(f"{label}: {path} rules_float64.{rule}.{scope} is {v!r}, not a finite number")
    if doc.get("artifact_sha256s") != art.file_sha256s:
        raise Refused(f"{label}: {path} was computed from other artifact files than {art.path}")
    return doc


def d5_cross_check(stages, elig_t: dict, elig_e: dict, background_index: int) -> tuple[dict, list[str]]:
    from src.stats.eligibility import RULES
    from src.stats.gap import SCOPES, RuleGap, scope_classes

    rec, problems = {}, []
    for scope in SCOPES:
        rec[scope] = {}
        for rule in RULES:
            pt = RuleGap(stages, rule, scope_classes(scope, stages.num_classes, background_index)).point()
            t, e = elig_t["rules_float64"][rule][scope], elig_e["rules_float64"][rule][scope]
            want = t - e
            diff = abs(pt["point"] - want)
            rec[scope][rule] = {"point": pt["point"], "rules_float64_difference": want, "abs_diff": diff,
                                "teacher_equal": pt["teacher"] == t, "e1_equal": pt["e1"] == e,
                                "holds": diff <= D5_TOLERANCE}
            if diff > D5_TOLERANCE:
                problems.append(f"{scope} {rule}: point {pt['point']!r} vs rules_float64 difference {want!r} "
                                f"(|diff| {diff!r} > {D5_TOLERANCE})")
    return rec, problems


def write_json(target: Path, doc: dict) -> None:
    """Serialize first, then create the file exclusively; a failed write leaves no partial file."""
    data = (json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    fh = open(target, "xb")                       # refuses an existing output; nothing created before
    try:
        with fh:
            fh.write(data)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:  # noqa: C901
    p = argparse.ArgumentParser(description="L-AM17B-GAP: teacher - E1 VAL gap (AM-17b item 2).")
    p.add_argument("--teacher", type=Path, help="the teacher of record's CPU re-score")
    p.add_argument("--e1", type=Path, help="E1 seed 42's CPU re-score")
    p.add_argument("--eligibility-variants", nargs=2, type=Path, metavar=("TEACHER_JSON", "E1_JSON"),
                   help="scripts/eligibility_variants.py outputs for the same two artifacts")
    p.add_argument("--e1-tolerance", type=float, default=E1_TOLERANCES[0], choices=E1_TOLERANCES,
                   metavar="{1e-7,1e-5}",
                   help="E1 validity tolerance: 1e-7 (default) when Q12 ran at batch size 1, else 1e-5 "
                        "(recorded as an erratum before the re-score runs); no other value is accepted")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--generated-utc", default=None,
                   help="pin the timestamp (for example 2026-10-01T00:00:00Z)")
    p.add_argument("--check-artifact", type=Path, default=None, metavar="RESCORE_DIR",
                   help="validate one re-score's pairing fields (seconds) instead of computing the gap")
    p.add_argument("--role", choices=ROLES, default=None, help="the pairing role of --check-artifact")
    args = p.parse_args(argv)
    if args.check_artifact is not None:
        if args.role is None:
            p.error("--check-artifact needs --role {e1,teacher}")
        if args.teacher is not None or args.e1 is not None or args.eligibility_variants is not None:
            p.error("--check-artifact checks one re-score; it takes no --teacher, --e1 or "
                    "--eligibility-variants")
        return check_artifact(args.check_artifact, args.role)
    if args.role is not None:
        p.error("--role applies to --check-artifact only")
    missing = [flag for flag, v in (("--teacher", args.teacher), ("--e1", args.e1),
                                    ("--eligibility-variants", args.eligibility_variants)) if v is None]
    if missing:
        p.error("the gap run needs --teacher, --e1 and --eligibility-variants (missing: "
                f"{', '.join(missing)})")
    stamp = args.generated_utc or datetime.now(timezone.utc).strftime(UTC_FORMAT)
    if not _UTC.match(stamp):
        print(f"error: --generated-utc must look like 2026-10-01T00:00:00Z, got {stamp!r}", file=sys.stderr)
        return EXIT_REFUSED
    target = args.out_dir / f"{OUT_PREFIX}{stamp.replace('-', '').replace(':', '')}.json"

    try:
        import numpy as np
        import scipy
        import torch

        from src.stats.align import METRIC_DISEASE_ONLY, align_runs
        from src.stats.gap import DIRECTION, gap_rules, paired_totals_identical_gt, per_image_summary
        from src.stats.noninferiority import PooledStages
        from src.stats.val_artifacts import AM5_LAYOUT, POLICY, code_provenance, load_val_artifact

        if target.exists():
            raise Refused(f"refusing to overwrite {target}")
        art_t = load_val_artifact(args.teacher, label="teacher", min_layout=AM5_LAYOUT)
        art_e = load_val_artifact(args.e1, label="e1", min_layout=AM5_LAYOUT)
        check_pairing(art_t, art_e)
        stages = PooledStages.from_runs(art_e.run, art_t.run)       # baseline E1, candidate teacher
        paired_totals_identical_gt(stages)
        elig_t = load_eligibility(args.eligibility_variants[0], art_t, "teacher")
        elig_e = load_eligibility(args.eligibility_variants[1], art_e, "e1")
    except Exception as e:                            # noqa: BLE001 -- refused input is never a verdict
        print(f"RESULT: REFUSED -- {type(e).__name__}: {e}")
        return EXIT_REFUSED

    ident = art_t.run.identity
    try:
        valid = validity(art_t, art_e, args.e1_tolerance)
        d5, problems = d5_cross_check(stages, elig_t, elig_e, ident.background_index)
    except Exception as e:                            # noqa: BLE001 -- an error is never a verdict
        print(f"RESULT: ERROR -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    for label in ("e1", "teacher"):
        v = valid[label]
        print(f"validity {label}: {v['value']!r} vs {v['reference']!r} -> delta {v['delta']:+.3e} "
              f"(tolerance {v['tolerance']:g}) {'OK' if v['within'] else 'EXCEEDED'}")
    if not all(v["within"] for v in valid.values()):
        print("RESULT: STOP -- a validity delta exceeds its tolerance (evaluator regression); nothing "
              "written")
        return EXIT_STOP
    if problems:
        for msg in problems:
            print(f"  d5: {msg}")
        print("RESULT: STOP -- the point estimates do not reproduce the L-AM17-GTPRESENT values; nothing "
              "written")
        return EXIT_STOP

    try:
        pv = align_runs(art_e.run, art_t.run, policy=POLICY, metric=METRIC_DISEASE_ONLY)   # teacher - E1
        per_image = per_image_summary(pv.delta).as_dict()
        intervals = gap_rules(stages, background_index=ident.background_index)
    except Exception as e:                            # noqa: BLE001 -- an error is never a verdict
        print(f"RESULT: ERROR -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    doc = {
        "lane": LANE,
        "authority": "AM-17b item 2 (DRAFT, adviser approval pending); docs/lane_specs/part1.md lane 7; "
                     "DL-35 G3 and G5",
        "artifact_status": "provisional",
        "descriptive": "reported before any KD result; it gates, selects and replaces nothing",
        "direction": DIRECTION,
        "inputs": {
            "teacher": art_t.describe(), "e1": art_e.describe(),
            "pairing": {"split": ident.split, "n_images": stages.n_images,
                        "split_manifest_sha256": ident.split_manifest_sha256,
                        "metric_impl_sha256": ident.metric_impl_sha256,
                        "repo_commit": art_t.summary["run"]["repo_commit"],
                        "preprocess_protocol": ident.preprocess_protocol, "device": DEVICE,
                        "batch_size": BATCH_SIZE},
            "eligibility_variants": {
                "teacher": {"path": args.eligibility_variants[0].as_posix(),
                            "sha256": _sha256(args.eligibility_variants[0])},
                "e1": {"path": args.eligibility_variants[1].as_posix(),
                       "sha256": _sha256(args.eligibility_variants[1])}},
        },
        "rules": {rule: intervals["all_class"][rule] for rule in ("union_present", "gt_present")},
        "disease_only": {rule: intervals["disease_only"][rule] for rule in ("union_present", "gt_present")},
        "per_image": {**per_image, "metric": "per-image disease-only mIoU (GT-present), teacher - E1",
                      "am5_rule": "no_disease_gt", "n_excluded_am5": pv.am5.n_excluded_am5,
                      "n_included": pv.am5.n_included, "excluded_ids_sha256": pv.am5.excluded_ids_sha256},
        "validity_deltas": valid,
        "e1_tolerance": {"value": args.e1_tolerance, "allowed": list(E1_TOLERANCES),
                         "rule": E1_TOLERANCE_RULE},
        "d5_cross_check": {"tolerance": D5_TOLERANCE, "source": "L-AM17-GTPRESENT rules_float64", **d5},
        "bootstrap_call": BOOTSTRAP_CALL,
        "code": code_provenance(CODE_FILES),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "torch": torch.__version__},
        "generated_utc": stamp,
    }
    try:
        write_json(target, doc)
    except Exception as e:                            # noqa: BLE001
        print(f"RESULT: REFUSED -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    for scope, block in (("all-class", doc["rules"]), ("disease-only", doc["disease_only"])):
        for rule, r in block.items():
            print(f"{scope:12s} {rule:13s} gap {r['point_pp']:+.4f} pp  95% CI [{r['ci_low_pp']:+.4f}, "
                  f"{r['ci_high_pp']:+.4f}] ({r['method']}, B={r['B']}, seed {r['seed']}; "
                  f"n classes teacher {r['n_eligible_teacher']}, E1 {r['n_eligible_e1']})")
    pi = doc["per_image"]
    print(f"per-image (n={pi['n']}, {pi['n_excluded_am5']} AM-5 excluded): median {pi['median']:+.5f}, "
          f"IQR {pi['iqr']:.5f}, mean {pi['mean']:+.5f}, SD {pi['sd']:.5f}, teacher better "
          f"{pi['share_teacher_better']:.4f}, ties {pi['share_ties']:.4f}, HL {pi['hl_shift']:+.5f}")
    print(f"  -> {target} (sha256 {_sha256(target)})")
    print("RESULT: GAP WRITTEN (validity within tolerance; d5 cross-check holds)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
