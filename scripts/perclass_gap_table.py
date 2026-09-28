#!/usr/bin/env python3
"""L-AM17-PERCLASS: per-class VAL IoU of the teacher of record minus E1 seed 42, by TRAIN stratum
(AM-17 item 1(f); docs/lane_specs/part1.md lane 4). Descriptive: no test, no decision, selects nothing.

    python -B scripts/perclass_gap_table.py --teacher R3_ARTIFACT_DIR --e1 DL17_RUN1_ARTIFACT_DIR \\
        --strata reports/strata/train_strata_v1.json [--out-dir reports/derived] [--generated-utc T]

Inputs (read only; no dataset, no model, no TEST): the R3 teacher artifact and the DL-17 run1 E1
artifact, each verified and read by the statistics ingest under Policy.REHEARSAL (VAL, 846 rows,
real-run artifact) through src/stats/val_artifacts.py -- canvas artifacts only (an upstream-protocol
artifact is refused), layouts by the ingest reader's rule (pre-lane or any 1.x) -- plus the registered
strata file of lane L-AM17-STRATA. The teacher must be the teacher of record (checkpoint 8c0e649a...)
and E1 seed 42 (cf0879f7...), scored on the same VAL manifest with the same metric implementation and
the same per-class ground truth.

Output (src/stats/perclass.py): <out-dir>/perclass_gap_val_<UTC>.json and .csv -- 115 disease rows
(stratum, rare, VAL gt_support, iou_teacher, iou_e1, delta, both iou_status; class 69 with no VAL
ground truth is not_evaluable) plus the background line, the per-stratum / rare / overall summaries,
the input sha256s (the four files of each artifact and the strata file) and the d2 integrity record.
The files are written only when the lane 4 d2 integrity holds: each model's table IoUs reproduce its
recorded mIoU bitwise in the evaluator's arithmetic, 115 rows plus the background line, and every
class sits in exactly one stratum. An existing output is never overwritten.

Exit codes: 0 written; 1 STOP -- the d2 integrity failed (nothing written); 2 refused or unreadable
input (not the specified pair, upstream protocol, TEST split, strata file, existing output).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2
LANE = "L-AM17-PERCLASS"
DEFAULT_OUT_DIR = REPO / "reports" / "derived"
OUT_PREFIX = "perclass_gap_val_"
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: The pair of record (AM-17 items 1(a) and 1(f); DL-17): the R3 teacher artifact and DL-17 run1.
EXPECTED = {
    "teacher": {"stage": "teacher", "model_role": "teacher",
                "checkpoint_sha256": "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e"},
    "e1": {"stage": "E1", "model_role": "student",
           "checkpoint_sha256": "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03"},
}


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def csv_bytes(rows, background) -> bytes:
    from src.stats.perclass import ROW_COLUMNS
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(ROW_COLUMNS)
    for r in [*rows, background]:
        w.writerow([_cell(r[k]) for k in ROW_COLUMNS])
    return buf.getvalue().encode("utf-8")


def build(teacher_dir, e1_dir, strata_path, generated_utc: str) -> tuple[dict, bytes, list[str]]:
    """(document, CSV bytes, integrity problems). Raises ValArtifactError / PerclassError on bad input."""
    from src.stats.perclass import (PerclassError, integrity, load_strata, perclass_rows, strata_index,
                                    summarize)
    from src.stats.val_artifacts import load_val_artifact, require_comparable, require_role

    art_t = load_val_artifact(teacher_dir, label="teacher")
    art_e = load_val_artifact(e1_dir, label="E1")
    require_role(art_t, label="teacher", **EXPECTED["teacher"])
    require_role(art_e, label="E1", **EXPECTED["e1"])
    require_comparable(art_e, art_t)
    strata_doc, strata_sha = load_strata(strata_path)
    if strata_doc.get("artifact_status") != "provisional":
        raise PerclassError(f"the strata file is not the registered build (artifact_status "
                            f"{strata_doc.get('artifact_status')!r}; scripts/build_train_strata.py --check)")
    ident = art_t.run.identity
    diseases = [c for c in range(ident.num_classes) if c != ident.background_index]
    strata = strata_index(strata_doc, diseases)
    rows, background = perclass_rows(art_t.summary["per_class"], art_e.summary["per_class"], strata,
                                     background_index=ident.background_index)
    summaries = summarize(rows)
    record, problems = integrity(art_t.summary, art_e.summary, rows, background, strata,
                                 background_index=ident.background_index)
    csv_b = csv_bytes(rows, background)
    doc = {
        "lane": LANE,
        "authority": "AM-17 item 1(f) (DRAFT, adviser approval pending); docs/lane_specs/part1.md lane 4",
        "artifact_status": "provisional",
        "descriptive": "diagnostic from existing artifacts: no test, no decision, selects nothing",
        "direction": "delta = iou_teacher - iou_e1 (positive: the teacher is better)",
        "split": ident.split,
        "inputs": {
            "teacher": art_t.describe(),
            "e1": art_e.describe(),
            "strata": {"path": Path(strata_path).as_posix(), "sha256": strata_sha,
                       "split_list_sha256": strata_doc.get("split_list_sha256"),
                       "artifact_status": strata_doc.get("artifact_status"),
                       "script_commit": strata_doc.get("script_commit"),
                       "tercile_sizes": strata_doc.get("tercile_sizes")},
        },
        "not_evaluable_rule": "a class with no VAL ground truth (gt_support == 0): delta null, excluded "
                              "from every summary",
        "rows": rows,
        "background": background,
        "summaries": summaries,
        "integrity": record,
        "csv_sha256": hashlib.sha256(csv_b).hexdigest(),
        "generated_utc": generated_utc,
    }
    return doc, csv_b, problems


def write(out_dir: Path, doc: dict, csv_b: bytes) -> tuple[Path, Path]:
    stamp = doc["generated_utc"].replace("-", "").replace(":", "")
    targets = (out_dir / f"{OUT_PREFIX}{stamp}.json", out_dir / f"{OUT_PREFIX}{stamp}.csv")
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(targets[1], "xb") as fh:
        fh.write(csv_b)
    try:
        with open(targets[0], "xb") as fh:
            fh.write((json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    except BaseException:
        targets[1].unlink()
        raise
    return targets


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="L-AM17-PERCLASS: teacher - E1 per-class VAL IoU by stratum.")
    p.add_argument("--teacher", required=True, type=Path, help="the R3 teacher artifact directory")
    p.add_argument("--e1", required=True, type=Path, help="the DL-17 run1 E1 seed-42 artifact directory")
    p.add_argument("--strata", required=True, type=Path, help="reports/strata/train_strata_v1.json")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--generated-utc", default=None,
                   help="pin the timestamp (for example 2026-10-01T00:00:00Z)")
    args = p.parse_args(argv)
    stamp = args.generated_utc or datetime.now(timezone.utc).strftime(UTC_FORMAT)
    if not _UTC.match(stamp):
        print(f"error: --generated-utc must look like 2026-10-01T00:00:00Z, got {stamp!r}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        doc, csv_b, problems = build(args.teacher, args.e1, args.strata, stamp)
    except Exception as e:                            # noqa: BLE001 -- refused input is never a verdict
        print(f"RESULT: REFUSED -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    integ = doc["integrity"]
    for label in ("teacher", "e1"):
        a = integ[label]["all_class"]
        print(f"{label}: summary mIoU {a['summary']!r} | table mean (evaluator arithmetic) "
              f"{a['reproduced']!r} bitwise={a['bitwise_equal']} n={a['n_union_present']} | float64 mean "
              f"{a['float64_mean_of_table_iou']!r}")
    for g, s in doc["summaries"].items():
        mean = "-" if s["mean_delta"] is None else f"{100 * s['mean_delta']:+.3f} pp"
        med = "-" if s["median_delta"] is None else f"{100 * s['median_delta']:+.3f} pp"
        print(f"  {g:8s} n={s['n_classes']:3d} (not evaluable {s['n_not_evaluable']}) mean {mean} "
              f"median {med} | teacher better {s['teacher_better']}, E1 better {s['e1_better']}, "
              f"ties {s['ties']}")
    print(f"  not evaluable (no VAL ground truth): {integ['not_evaluable_classes']}")
    if problems:
        for msg in problems:
            print(f"  INTEGRITY: {msg}")
        print("RESULT: STOP -- lane 4 d2 integrity failed; nothing written")
        return EXIT_STOP
    try:
        json_path, csv_path = write(args.out_dir, doc, csv_b)
    except Exception as e:                            # noqa: BLE001
        print(f"RESULT: REFUSED -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    print(f"  -> {json_path} (sha256 {hashlib.sha256(json_path.read_bytes()).hexdigest()})")
    print(f"  -> {csv_path} (sha256 {doc['csv_sha256']})")
    print(f"RESULT: PERCLASS TABLE WRITTEN (d2 integrity holds: {integ['n_rows']} rows + background)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
