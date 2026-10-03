#!/usr/bin/env python3
"""D4 reducer: duplicate-free VAL scores and margin_dedup (AM-18 item 1(d); item 8(b) with --single; lane
L-TEACHER-DIAG).

Pair mode (the teacher of record and E1 seed 42, both AM-17b CPU re-scores):
  --teacher DIR --e1 DIR --gap-output gap_val_<UTC>.json --duplicates val_train_duplicates.json
  --duplicates-sha256 <64 hex>
Refused (P23) unless both artifacts' file hashes equal the artifact_sha256s S3's gap output records for that
role; S3's check_pairing, validity() (at the gap output's E1 tolerance) and paired_totals_identical_gt
pass; the duplicates file is the hasher's provisional output, its sha256 equals the flag and its VAL
manifest hash equals both artifacts' split_manifest_sha256; every duplicate id is in each artifact's
manifest_ids and the rows dropped are exactly the ids listed. A smoke artifact is refused (P2).

Arithmetic (P24): rows are dropped by id through each artifact's own manifest_ids; the remaining
(class_id, tp, gt, pred) triplets go to src.stats.eligibility.class_totals (no dataset_* arrays) and
dataset_miou, so eligibility is recomputed on the subset under both rules. The full set comes first: its
float32 value must equal each summary.json bit for bit, and the float64 teacher - E1 must equal S3's
union-present all-class point within 1e-12 (else STOP). margin_dedup = the float64 (math.fsum) teacher - E1,
union-present, all-class, and margin_dedup_le_0 is computed from it alone. Beside it: the float32-arithmetic
margin and the exact sign from fractions.Fraction; if the three signs differ the status is "sign
disagreement" and the exit code 1 (the halt branch, Q1). Also written per model: n_eligible under both
rules on the full set and the subset, the classes whose eligibility changed, the GT-present margin, and
n_total, n_dup, n_kept.

--single DIR (P25): the duplicate-free mIoU of one VAL artifact by the same code path, without pairing; it
gates nothing.

Outputs: teacher_d4_dedup_<UTC>.json or val_dedup_single_<UTC>.json in --out-dir (outside the repository).
Every run is of record (--script-commit, --script-commit-dl-id; --generated-utc refused). Exit codes: 0
written, 1 STOP or sign disagreement, 2 refusal or usage, 4 unexpected exception.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval import teacher_diag as td  # noqa: E402

SCRIPT = "scripts/dedup_val_scores.py"
HASHER = "scripts/hash_split_files.py"
HASHER_SCHEMA = "plantseg-val-train-duplicates/1.0.0"
GAP_LANE = "L-AM17B-GAP"
SCHEMA_PAIR, SCHEMA_SINGLE = "plantseg-teacher-d4-dedup/1.0.0", "plantseg-val-dedup-single/1.0.0"
KIND_PAIR, KIND_SINGLE = "teacher_d4_dedup", "val_dedup_single"
POINT_TOLERANCE = 1e-12
NUM_CLASSES, BACKGROUND = 116, 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D4: duplicate-free VAL scores and margin_dedup.")
    td.add_common_flags(p)
    p.add_argument("--teacher", help="pair mode: the teacher of record's VAL CPU re-score")
    p.add_argument("--e1", help="pair mode: E1 seed 42's VAL CPU re-score")
    p.add_argument("--gap-output", help="pair mode: S3's gap_val_<UTC>.json")
    p.add_argument("--single", help="one VAL artifact (P25; no pairing, gates nothing)")
    p.add_argument("--duplicates", required=True, help="the hasher's val_train_duplicates.json")
    p.add_argument("--duplicates-sha256", required=True, help="that file's sha256")
    return p


# --------------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------------
def read_duplicates(path, sha_flag: str) -> tuple[list[str], dict]:
    from src.eval.artifacts import am5_excluded_ids_sha256
    if not td.SHA256_RE.match(sha_flag or ""):
        raise td.Refused("--duplicates-sha256 must match ^[0-9a-f]{64}$")
    p = Path(path)
    if not p.is_file():
        raise td.Refused(f"--duplicates not found: {p}")
    sha = td.file_sha256(p)
    if sha != sha_flag:
        raise td.Refused(f"--duplicates sha256 {sha} != --duplicates-sha256 {sha_flag}")
    doc = td.read_json_strict(p)
    if (doc.get("lane"), doc.get("script"), doc.get("schema")) != (td.LANE, HASHER, HASHER_SCHEMA):
        raise td.Refused("--duplicates is not the D4 hasher's output")
    if doc.get("artifact_status") != "provisional" or doc.get("status") != "written":
        raise td.Refused(f"--duplicates has artifact_status {doc.get('artifact_status')!r}; provisional is required")
    ids = doc.get("duplicate_val_ids")
    if not isinstance(ids, list) or ids != sorted(set(ids)) or not all(isinstance(i, str) for i in ids):
        raise td.Refused("--duplicates: duplicate_val_ids must be a sorted list of unique ids")
    if am5_excluded_ids_sha256(ids) != doc.get("duplicate_val_ids_sha256"):
        raise td.Refused("--duplicates: duplicate_val_ids do not hash to duplicate_val_ids_sha256")
    manifest = (doc.get("val_manifest") or {}).get("hash_split_manifest")
    return ids, {"path": str(p), "sha256": sha, "n_ids": len(ids), "ids_sha256": doc.get("duplicate_val_ids_sha256"),
                 "val_manifest_sha256": manifest}


def load(path, label: str):
    from src.stats.val_artifacts import AM5_LAYOUT, ValArtifactError, load_val_artifact
    try:
        art = load_val_artifact(path, label=label, min_layout=AM5_LAYOUT)
    except ValArtifactError as e:
        raise td.Refused(str(e)) from e
    if art.summary["run"].get("artifact_status") == "smoke":
        raise td.Refused(f"{label}: a smoke artifact is refused (P2)")
    return art


def drop_positions(art, ids: list[str], label: str) -> list[int]:
    mids = list(art.run.stats.manifest_ids)
    pos = {m: i for i, m in enumerate(mids)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise td.Refused(f"{label}: {len(missing)} duplicate id(s) are not in its manifest_ids")
    return sorted(pos[i] for i in ids)


# --------------------------------------------------------------------------------------------------
# arithmetic (P24)
# --------------------------------------------------------------------------------------------------
def subset_totals(stats, drop: list[int]):
    """(ClassTotals of the kept rows, number of images whose rows were dropped)."""
    import numpy as np

    from src.stats.eligibility import class_totals
    keep = ~np.isin(stats.image_index, np.asarray(drop, dtype=stats.image_index.dtype))
    dropped_images = int(np.unique(stats.image_index[~keep]).size)
    tot = class_totals({"class_id": stats.class_id[keep], "tp": stats.tp[keep], "gt": stats.gt[keep],
                        "pred": stats.pred[keep]}, num_classes=NUM_CLASSES)
    return tot, dropped_images


def exact_mean(tot, classes, rule: str) -> Fraction | None:
    from src.stats.eligibility import UNION_PRESENT
    sel = set(int(c) for c in classes)
    terms = []
    for c in sorted(sel):
        tp, gt, pred = int(tot.tp[c]), int(tot.gt[c]), int(tot.pred[c])
        union = gt + pred - tp
        if (union > 0) if rule == UNION_PRESENT else (gt > 0):
            terms.append(Fraction(tp, union))
    return (sum(terms, Fraction(0)) / len(terms)) if terms else None


def scores(tot) -> dict:
    from src.stats.eligibility import GT_PRESENT, UNION_PRESENT, dataset_miou
    out = {}
    for rule in (UNION_PRESENT, GT_PRESENT):
        out[rule] = {}
        for scope, classes in (("all_class", range(NUM_CLASSES)),
                               ("disease_only", [c for c in range(NUM_CLASSES) if c != BACKGROUND])):
            dm = dataset_miou(tot, classes, rule)
            out[rule][scope] = {"value": dm.value, "value_float64": dm.value_float64, "n_eligible": dm.n_classes,
                                "eligible": list(dm.eligible)}
    return out


def eligibility_changes(full: dict, sub: dict) -> dict:
    return {rule: {"dropped": sorted(set(full[rule]["all_class"]["eligible"]) - set(sub[rule]["all_class"]["eligible"])),
                   "added": sorted(set(sub[rule]["all_class"]["eligible"]) - set(full[rule]["all_class"]["eligible"]))}
            for rule in full}


def sign(x) -> int:
    return (x > 0) - (x < 0)


def float32_margin(t_value: float, e_value: float) -> float:
    import numpy as np
    return float(np.float32(t_value) - np.float32(e_value))


def margin_block(t_tot, e_tot, t_sub: dict, e_sub: dict) -> dict:
    """margin_dedup (float64, fsum), its float32 twin and the exact sign; the three signs must agree."""
    from src.stats.eligibility import UNION_PRESENT
    t_u, e_u = t_sub[UNION_PRESENT]["all_class"], e_sub[UNION_PRESENT]["all_class"]
    if t_u["value_float64"] is None or e_u["value_float64"] is None:
        raise td.Stop("no eligible class on the duplicate-free subset")
    m64 = t_u["value_float64"] - e_u["value_float64"]
    m32 = float32_margin(t_u["value"], e_u["value"])
    exact = exact_mean(t_tot, range(NUM_CLASSES), UNION_PRESENT) - exact_mean(e_tot, range(NUM_CLASSES), UNION_PRESENT)
    return decide(m64, m32, exact)


def decide(m64: float, m32: float, exact: Fraction) -> dict:
    signs = {"float64": sign(m64), "float32": sign(m32), "exact": sign(exact)}
    agree = len(set(signs.values())) == 1
    return {"margin_dedup": m64, "margin_dedup_le_0": m64 <= 0, "float32_margin": m32,
            "exact_margin": f"{exact.numerator}/{exact.denominator}", "signs": signs, "signs_agree": agree,
            "margin_status": "ok" if agree else "sign disagreement",
            "rule": "teacher - E1, union-present, all-class, on VAL without the duplicates; margin_dedup is the "
                    "float64 (math.fsum) value and alone decides margin_dedup_le_0"}


def model_block(art, ids, label: str) -> tuple[dict, object, dict, dict]:
    """(record, subset totals, full scores, subset scores) of one artifact; the full set must reproduce."""
    from src.stats.eligibility import class_totals
    stats = art.run.stats
    drop = drop_positions(art, ids, label)
    full_tot = class_totals({"class_id": stats.class_id, "tp": stats.tp, "gt": stats.gt, "pred": stats.pred},
                            num_classes=NUM_CLASSES)
    sub_tot, dropped = subset_totals(stats, drop)
    if dropped != len(ids):
        raise td.Refused(f"{label}: rows of {dropped} image(s) were dropped for {len(ids)} listed id(s)")
    full, sub = scores(full_tot), scores(sub_tot)
    dl = art.summary["dataset_level"]
    for scope, key in (("all_class", "all_class_miou"), ("disease_only", "disease_only_miou")):
        if full["union_present"][scope]["value"] != dl.get(key):
            raise td.Stop(f"{label}: the full-set {scope} value {full['union_present'][scope]['value']!r} != "
                          f"summary.json {key} {dl.get(key)!r} (bitwise)")
    n_total = len(stats.manifest_ids)
    rec = {"artifact": art.describe(), "n_total": n_total, "n_dup": len(ids), "n_kept": n_total - len(ids),
           "full": full, "subset": sub, "eligibility_changed": eligibility_changes(full, sub),
           "full_float32_equals_summary": True}
    return rec, sub_tot, full, sub


# --------------------------------------------------------------------------------------------------
def run(args) -> int:
    return td.run_with_exit_codes(_run, args)


def _run(args) -> int:
    start, t0 = td.utc_now(), time.monotonic()
    single = args.single is not None
    pair_flags = (args.teacher, args.e1, args.gap_output)
    if single and any(x is not None for x in pair_flags):
        raise td.Refused("--single takes no --teacher, --e1 or --gap-output")
    if not single and any(x is None for x in pair_flags):
        raise td.Refused("pair mode requires --teacher, --e1 and --gap-output (or use --single)")
    td.check_common_flags(args, real=True)
    out_dir = td.require_outside_repo(args.out_dir, "--out-dir")
    code = td.require_commit_binding(args.script_commit, args.script_commit_dl_id)
    kind = KIND_SINGLE if single else KIND_PAIR
    repeat = td.require_single_output(out_dir, kind, args)
    out_path = out_dir / f"{kind}_{td.output_stamp(args, start)}.json"
    if out_path.exists():
        raise td.Refused(f"{out_path.name} already exists")
    ids, dup_info = read_duplicates(args.duplicates, args.duplicates_sha256)

    if single:
        art = load(args.single, "single")
        if art.summary["dataset"]["split_manifest_sha256"] != dup_info["val_manifest_sha256"]:
            raise td.Refused("the duplicates file's VAL manifest hash differs from the artifact's")
        rec, _, _, _ = model_block(art, ids, "single")
        doc = td.base_document(SCRIPT, args, stub=False, start_utc=start, code=code, extra={
            "schema": SCHEMA_SINGLE, "repeat": repeat, "gates": "nothing (AM-18 item 8(b); descriptive)",
            "inputs": {"duplicates": dup_info}, "model": rec, "status": "written"})
        td.finish_output(out_path, doc, start, t0)
        return td.EXIT_OK

    import scripts.gap_bootstrap_val as gap
    from src.stats.gap import GapError, paired_totals_identical_gt
    from src.stats.noninferiority import PooledStages
    from src.stats.val_artifacts import ValArtifactError
    gpath = Path(args.gap_output)
    if not gpath.is_file():
        raise td.Refused(f"--gap-output not found: {gpath}")
    gdoc, gsha = td.read_json_strict(gpath), td.file_sha256(gpath)
    if gdoc.get("lane") != GAP_LANE or gdoc.get("artifact_status") != "provisional":
        raise td.Refused(f"--gap-output is not a provisional {GAP_LANE} output")
    art_t, art_e = load(args.teacher, "teacher"), load(args.e1, "e1")
    for role, art in (("teacher", art_t), ("e1", art_e)):
        if ((gdoc.get("inputs") or {}).get(role) or {}).get("artifact_sha256s") != art.file_sha256s:
            raise td.Refused(f"{role}: the artifact's file hashes are not the ones S3's gap output records")
    try:
        gap.check_pairing(art_t, art_e)
        stages = PooledStages.from_runs(art_e.run, art_t.run)
        paired_totals_identical_gt(stages)
        valid = gap.validity(art_t, art_e, (gdoc.get("e1_tolerance") or {}).get("value"))
    except (gap.Refused, ValArtifactError, GapError) as e:
        raise td.Refused(f"pairing: {type(e).__name__}: {e}") from e
    if not all(v["within"] for v in valid.values()):
        raise td.Refused("validity(): a re-score is outside its tolerance of the reference")
    for role, art in (("teacher", art_t), ("e1", art_e)):
        if art.summary["dataset"]["split_manifest_sha256"] != dup_info["val_manifest_sha256"]:
            raise td.Refused(f"{role}: the duplicates file's VAL manifest hash differs from the artifact's")

    rec_t, tot_t, full_t, sub_t = model_block(art_t, ids, "teacher")
    rec_e, tot_e, full_e, sub_e = model_block(art_e, ids, "e1")
    point = (((gdoc.get("rules") or {}).get("union_present") or {}).get("point"))
    full_diff = (full_t["union_present"]["all_class"]["value_float64"]
                 - full_e["union_present"]["all_class"]["value_float64"])
    if not isinstance(point, float) or abs(full_diff - point) > POINT_TOLERANCE:
        raise td.Stop(f"the full-set float64 teacher - E1 {full_diff!r} differs from S3's point {point!r} "
                      f"by more than {POINT_TOLERANCE}")
    margin = margin_block(tot_t, tot_e, sub_t, sub_e)
    gt_present = (sub_t["gt_present"]["all_class"]["value_float64"] - sub_e["gt_present"]["all_class"]["value_float64"])
    doc = td.base_document(SCRIPT, args, stub=False, start_utc=start, code=code, extra={
        "schema": SCHEMA_PAIR, "repeat": repeat,
        "gates": "margin_dedup_le_0 feeds AM-18's pre-set halt rule for every KD run",
        "detection": "VAL images byte-identical to a TRAIN image (the hasher); re-encoded copies are not detected",
        "inputs": {"duplicates": dup_info, "gap_output": {"path": str(gpath), "sha256": gsha, "point": point},
                   "validity": valid},
        "teacher": rec_t, "e1": rec_e,
        "full_set": {"float64_teacher_minus_e1": full_diff, "s3_point": point, "abs_diff": abs(full_diff - point),
                     "tolerance": POINT_TOLERANCE},
        **margin,
        "gt_present_margin": gt_present,
        "status": "written" if margin["signs_agree"] else "sign disagreement",
    })
    td.finish_output(out_path, doc, start, t0)
    if not margin["signs_agree"]:
        print(f"STOP: sign disagreement {margin['signs']} (the halt branch, Q1)", file=sys.stderr)
        return td.EXIT_STOP
    return td.EXIT_OK


def main(argv=None) -> int:
    return td.cli_main(build_parser(), run, argv)


if __name__ == "__main__":
    sys.exit(main())
