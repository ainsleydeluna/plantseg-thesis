#!/usr/bin/env python3
"""L-AM17B-GAP: the teacher of record minus E1 seed 42 on VAL -- paired image-level BCa intervals of the
dataset-level mIoU gap under the union-present and GT-present rules, and the per-image difference
summary (AM-17b item 2; docs/lane_specs/part1.md lane 7; DL-35 G3 and G5). Descriptive: it gates,
selects and replaces nothing.

    python -B scripts/gap_bootstrap_val.py --teacher TEACHER_CPU_ARTIFACT --e1 E1_CPU_ARTIFACT \\
        --eligibility-variants TEACHER_ELIGIBILITY.json E1_ELIGIBILITY.json \\
        [--out-dir reports/derived] [--generated-utc T]

Inputs: the two AM-17b item 2(c) CPU re-scores and the L-AM17-GTPRESENT outputs
(scripts/eligibility_variants.py) for the same two artifacts. No dataset, no model, no TEST.

1. Pairing of record (exit 2 when violated; not a verdict). Both artifacts verify and are read under
   Policy.REHEARSAL (VAL, 846 rows, real-run) through src/stats/val_artifacts.py: canvas only (an
   upstream-protocol artifact is refused), a layout plantseg-eval-artifact/1.x at or after 1.1.0 (the
   ingest reader's rule; the re-scores are written after the L-AM5 merge). The teacher is the teacher
   of record (checkpoint 8c0e649a...) and E1 is seed 42 (cf0879f7...); both on the same VAL manifest
   with the same metric implementation and identical per-image ground truth; both at the same pin
   (repo_commit) with clean governed paths; both on the CPU (env.device, eval_runtime model and input
   devices) at batch size 1; each in its pinned image (eval_runtime.image_digest: teacher cb413304...,
   student b80b645d...).
2. Validity (lane 7 (a); STOP, exit 1, nothing written): |E1 all-class mIoU - 0.36307525634765625|
   <= 1e-7 (the Q12 CPU value) and |teacher all-class mIoU - 0.38576993346214294| <= 1e-5 (R3; a
   different CPU/BLAS than the pod). Both deltas are printed and recorded.
3. d5 cross-check (STOP, exit 1): each point estimate equals the difference of the two L-AM17-GTPRESENT
   `rules_float64` values within 1e-12, for both rules and both class subsets (S1 ruling). Each
   eligibility file must name the input artifact's own four file hashes (exit 2 otherwise).
4. Four bootstraps (src/stats/gap.py: B = 10,000, default_rng(42), BCa with the percentile fallback)
   and the per-image summary on the AM-5-included images. Written once to
   <out-dir>/gap_val_<UTC>.json; an existing output is never overwritten.

Exit codes: 0 written; 1 STOP (a validity delta above its tolerance, or the d5 cross-check failed:
evaluator regression or an unfaithful reconstruction -- report it); 2 refused or unreadable input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
D5_TOLERANCE = 1e-12
DEVICE, BATCH_SIZE = "cpu", 1

#: The pairing of record (AM-17 item 1(a), DL-17, DL-21, AM-17b item 2(c)) and the lane 7 (a) validity
#: references.
PAIRING = {
    "teacher": {"stage": "teacher", "model_role": "teacher",
                "checkpoint_sha256": "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e",
                "image_digest": "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b",
                "reference": 0.38576993346214294, "tolerance": 1e-5,
                "reference_source": "R3, the teacher of record on VAL (CPU, batch 1, evaluator 3c43f89)"},
    "e1": {"stage": "E1", "model_role": "student",
           "checkpoint_sha256": "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03",
           "image_digest": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf",
           "reference": 0.36307525634765625, "tolerance": 1e-7,
           "reference_source": "the B66-prep Q12 CPU re-score of E1 seed 42"},
}


class Refused(RuntimeError):
    """The inputs are not the specified pairing: not a verdict."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_pairing(art_t, art_e) -> None:
    from src.stats.val_artifacts import require_comparable, require_role

    for label, art in (("teacher", art_t), ("e1", art_e)):
        spec, run = PAIRING[label], art.summary["run"]
        require_role(art, stage=spec["stage"], model_role=spec["model_role"],
                     checkpoint_sha256=spec["checkpoint_sha256"], label=label)
        rt = run.get("eval_runtime")
        if not isinstance(rt, dict):
            raise Refused(f"{label}: no run.eval_runtime record; the AM-17b item 2(c) re-score writes one")
        devices = (run["env"].get("device"), rt.get("model_device"), rt.get("input_devices"))
        if devices != (DEVICE, DEVICE, [DEVICE]):
            raise Refused(f"{label}: not a CPU re-score (env.device, model_device, input_devices = "
                          f"{devices})")
        if rt.get("batch_size") != BATCH_SIZE:
            raise Refused(f"{label}: batch size {rt.get('batch_size')!r}, the re-score runs at batch 1")
        if rt.get("image_digest") != spec["image_digest"]:
            raise Refused(f"{label}: image digest {rt.get('image_digest')!r} is not the pinned image "
                          f"{spec['image_digest']} (docker run -e PLANTSEG_IMAGE_DIGEST=...)")
        if run.get("governed_paths_clean") is not True:
            raise Refused(f"{label}: governed paths were dirty at scoring time; the re-score runs at a "
                          "clean pin")
    pins = (art_t.summary["run"]["repo_commit"], art_e.summary["run"]["repo_commit"])
    if pins[0] != pins[1]:
        raise Refused(f"the two re-scores were made at different commits {pins}; both run at one pin")
    require_comparable(art_e, art_t)


def validity(art_t, art_e) -> dict:
    out = {}
    for label, art in (("e1", art_e), ("teacher", art_t)):
        spec = PAIRING[label]
        v = art.summary["dataset_level"]["all_class_miou"]
        delta = v - spec["reference"]
        out[label] = {"value": v, "reference": spec["reference"], "delta": delta, "abs_delta": abs(delta),
                      "tolerance": spec["tolerance"], "within": abs(delta) <= spec["tolerance"],
                      "reference_source": spec["reference_source"]}
    return out


def load_eligibility(path: Path, art, label: str) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("lane") != "L-AM17-GTPRESENT" or not isinstance(doc.get("rules_float64"), dict):
        raise Refused(f"{label}: {path} is not an L-AM17-GTPRESENT eligibility-variants file")
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


def main(argv=None) -> int:  # noqa: C901
    p = argparse.ArgumentParser(description="L-AM17B-GAP: teacher - E1 VAL gap (AM-17b item 2).")
    p.add_argument("--teacher", required=True, type=Path, help="the teacher of record's CPU re-score")
    p.add_argument("--e1", required=True, type=Path, help="E1 seed 42's CPU re-score")
    p.add_argument("--eligibility-variants", required=True, nargs=2, type=Path,
                   metavar=("TEACHER_JSON", "E1_JSON"),
                   help="scripts/eligibility_variants.py outputs for the same two artifacts")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--generated-utc", default=None,
                   help="pin the timestamp (for example 2026-10-01T00:00:00Z)")
    args = p.parse_args(argv)
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
        from src.stats.val_artifacts import AM5_LAYOUT, POLICY, load_val_artifact

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
    valid = validity(art_t, art_e)
    for label in ("e1", "teacher"):
        v = valid[label]
        print(f"validity {label}: {v['value']!r} vs {v['reference']!r} -> delta {v['delta']:+.3e} "
              f"(tolerance {v['tolerance']:g}) {'OK' if v['within'] else 'EXCEEDED'}")
    if not all(v["within"] for v in valid.values()):
        print("RESULT: STOP -- a validity delta exceeds its tolerance (evaluator regression); nothing "
              "written")
        return EXIT_STOP
    d5, problems = d5_cross_check(stages, elig_t, elig_e, ident.background_index)
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
        "d5_cross_check": {"tolerance": D5_TOLERANCE, "source": "L-AM17-GTPRESENT rules_float64", **d5},
        "bootstrap_call": "scipy.stats.bootstrap((idx,), stat, n_resamples=10000, method='BCa', "
                          "random_state=numpy.random.default_rng(42), vectorized=False); percentile on "
                          "the same replicates when the jackknife acceleration is non-finite or BCa "
                          "fails (fallback_reason)",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "torch": torch.__version__},
        "generated_utc": stamp,
    }
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
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
