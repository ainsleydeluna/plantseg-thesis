#!/usr/bin/env python3
"""L-AM16-CALIB (AM-16 item 5): the range over the four calibration sets of one PTQ stage. Reads no data.

    python -B scripts/ptq_calibration_range.py --stage E4 \\
        --pair /out/e4_s42_cal42_a1/e4_ptq_record.json EVAL_42/summary.json \\
        --pair /out/e4_s42_cal43_a1/e4_ptq_record.json EVAL_43/summary.json \\
        --pair /out/e4_s42_cal44_a1/e4_ptq_record.json EVAL_44/summary.json \\
        --pair /out/e4_s42_cal45_a1/e4_ptq_record.json EVAL_45/summary.json \\
        --out-dir OUT

AM-16 item 5: "E4 and E7 at seed 42 are also calibrated on three further 128-image TRAIN subsets, drawn by the
AM-10 procedure with seeds 43, 44 and 45. Descriptive: reported per subset, with the range over all four
calibration sets. The official E4/E7 models keep the AM-10 list." Each pair is one PTQ record
(src/quant/ptq_record.py, ptq_record/1; re-hashed with every file it lists) and the scripts/evaluate_model.py
summary.json of that record's artifact on clean TEST in the canvas protocol. It refuses (exit 2, nothing
written), in this order:
  output_dir_inside_repo   --out-dir is inside the repository (checked first, before any input is read)
  range_input_invalid  a record that is not an intact ptq_record/1, or a record or summary without a field read
                       here or with one of the wrong type
  range_stage          a record's stage or a summary's run.stage differs from --stage
  range_lists          the lists are not seeds 42, 43, 44 and 45 once each, the seed-42 record's role is not
                       of_record or another's is not calibration_sensitivity
  range_parent         the records name more than one parent checkpoint, or a seed other than 42
  range_artifact       a summary's run.checkpoint_sha256 is not its record's TorchScript artifact
  range_status         a summary's artifact_status is not official (the seed-42 list) or provisional (43-45)
  range_protocol       a summary is not clean TEST in the canvas protocol (core_preprocess/1.0.0)
  range_rows           a summary's actual_rows differ from its expected_rows, or the four row counts differ
  range_metric         dataset_level.all_class_miou or disease_only_miou is absent or not finite
  range_output_exists  the output file exists (it is created exclusively: written once)
It writes ptq_calibration_range_<stage>.json: per subset the list seed, its checksum, the record and summary
sha256s, the headline all-class mIoU and the disease-only mIoU alongside, and range = max - min over the four
for each; descriptive. No subset is selected, and none is scored on VAL.

Exit codes: 0 written; 2 refused (nothing written). Nothing here trains, downloads, uses a GPU or reads a dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_REFUSED = 0, 2
FORMAT = "ptq_calibration_range/1"
STAGES = ("E4", "E7")
LIST_SEEDS = (42, 43, 44, 45)                      # AM-10 (42) and AM-16 item 5 (43-45)
SEED = 42                                          # AM-16 item 5: E4 and E7 at seed 42
METRIC = "dataset_level.all_class_miou"            # the headline metric (checkpoint and reporting)
ALONGSIDE = "dataset_level.disease_only_miou"      # reported alongside, never the headline
SPLIT, CONDITION = "test", {"type": "clean", "name": None, "severity": None}
OUT_NAME = "ptq_calibration_range_{stage}.json"
AUTHORITY = "AM-16 item 5 (descriptive: per subset, with the range over all four calibration sets); lane L-AM16-CALIB"


class RangeRefused(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _refuse(code: str, message: str):
    raise RangeRefused(code, message)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _typed(value, kind: type, what: str):
    if not isinstance(value, kind) or isinstance(value, bool):
        raise TypeError(f"{what} is {type(value).__name__}, not {kind.__name__}")
    return value


def read_pair(record_path, summary_path) -> dict:
    """One record and its summary: every summary field and nested record field compute() reads, each of the type
    it needs (read_record has checked the record's top-level keys); anything else refuses as range_input_invalid."""
    from src.quant.ptq_record import read_record
    try:
        rec, rec_sha = read_record(record_path)
        raw = Path(summary_path).read_bytes()
        summary = json.loads(raw.decode("utf-8"))
        run, ds, dl = (_typed(summary[k], dict, k) for k in ("run", "dataset", "dataset_level"))
        fields = {"stage": run["stage"], "artifact_status": run["artifact_status"],
                  "checkpoint_sha256": run["checkpoint_sha256"], "split": ds["split"],
                  "condition": ds["condition"], "protocol": ds["preprocess_protocol"],
                  "expected_rows": _typed(ds["expected_rows"], int, "dataset.expected_rows"),
                  "actual_rows": _typed(ds["actual_rows"], int, "dataset.actual_rows"),
                  "all_class_miou": dl.get("all_class_miou"), "disease_only_miou": dl.get("disease_only_miou"),
                  "list_seed": _typed(rec["calibration"]["seed"], int, "calibration.seed"),
                  "list_checksum": _typed(rec["calibration"]["checksum_sha256"], str, "calibration.checksum_sha256"),
                  "parent_sha256": _typed(rec["parent"]["checkpoint_sha256"], str, "parent.checkpoint_sha256"),
                  "torchscript_sha256": rec["artifacts"]["torchscript"]["sha256"]}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RangeRefused("range_input_invalid", f"{record_path} / {summary_path}: {type(e).__name__}: {e}") from e
    return {"record": rec, "record_path": str(Path(record_path).resolve()), "record_sha256": rec_sha,
            "summary_path": str(Path(summary_path).resolve()), "summary_sha256": hashlib.sha256(raw).hexdigest(),
            **fields}


def compute(stage: str, pairs: list) -> dict:
    from src.eval.protocols import CANVAS_PROTOCOL_ID
    from src.quant.ptq_record import ROLE_OF_RECORD, ROLE_SENSITIVITY
    rows = [read_pair(r, s) for r, s in pairs]
    wrong = [p["record_path"] for p in rows if p["record"]["stage"] != stage or p["stage"] != stage]
    if wrong:
        _refuse("range_stage", f"--stage {stage}: a record or summary of another stage: {wrong}")
    seeds = sorted(p["list_seed"] for p in rows)
    roles = [(p["list_seed"], p["record"]["role"]) for p in rows]
    if seeds != list(LIST_SEEDS) or any(r != (ROLE_OF_RECORD if s == SEED else ROLE_SENSITIVITY) for s, r in roles):
        _refuse("range_lists", f"the four records are lists {seeds} with roles {roles}; the range needs lists "
                               f"{list(LIST_SEEDS)} once each, seed {SEED} of_record and the others "
                               f"{ROLE_SENSITIVITY}")
    parents = {p["parent_sha256"] for p in rows}
    if len(parents) != 1 or any(p["record"]["seed"] != SEED for p in rows):
        _refuse("range_parent", f"the records name {len(parents)} parent checkpoints or a seed other than {SEED}: "
                                "the four calibration sets quantize one seed-42 parent")
    foreign = [p["summary_path"] for p in rows if p["checkpoint_sha256"] != p["torchscript_sha256"]]
    if foreign:
        _refuse("range_artifact", f"summaries of another artifact than their record's TorchScript file: {foreign}")
    status = [(p["list_seed"], p["artifact_status"]) for p in rows
              if p["artifact_status"] != ("official" if p["record"]["role"] == ROLE_OF_RECORD else "provisional")]
    if status:
        _refuse("range_status", f"artifact_status {status}: the list of record's score is official, the "
                                "sensitivity scores provisional")
    off = [p["summary_path"] for p in rows
           if p["split"] != SPLIT or p["condition"] != CONDITION or p["protocol"] != CANVAS_PROTOCOL_ID]
    if off:
        _refuse("range_protocol", f"not clean {SPLIT.upper()} in the canvas protocol {CANVAS_PROTOCOL_ID}: {off}")
    counts = {(p["expected_rows"], p["actual_rows"]) for p in rows}
    if len(counts) != 1 or any(e != a for e, a in counts):
        _refuse("range_rows", f"(expected_rows, actual_rows) {sorted(counts)}: one full row count for all four")
    nonfinite = [p["summary_path"] for p in rows if not (_finite(p["all_class_miou"])
                                                         and _finite(p["disease_only_miou"]))]
    if nonfinite:
        _refuse("range_metric", f"{METRIC} or {ALONGSIDE} absent or not finite in {nonfinite}")
    rows.sort(key=lambda p: p["list_seed"])
    per = [{"list_seed": p["list_seed"], "checksum_sha256": p["list_checksum"], "role": p["record"]["role"],
            "artifact_status": p["artifact_status"], "record": p["record_path"],
            "record_sha256": p["record_sha256"], "summary": p["summary_path"], "summary_sha256": p["summary_sha256"],
            "torchscript_sha256": p["checkpoint_sha256"], "all_class_miou": p["all_class_miou"],
            "disease_only_miou": p["disease_only_miou"]} for p in rows]

    def span(key):
        vals = [x[key] for x in per]
        return {"min": min(vals), "max": max(vals), "range": max(vals) - min(vals)}
    return {"format": FORMAT, "stage": stage, "seed": SEED, "metric": METRIC, "alongside": ALONGSIDE,
            "split": SPLIT, "condition": CONDITION, "protocol": CANVAS_PROTOCOL_ID,
            "rows": rows[0]["actual_rows"], "parent_checkpoint_sha256": parents.pop(), "per_subset": per,
            "all_class_miou": span("all_class_miou"), "disease_only_miou": span("disease_only_miou"),
            "descriptive": True, "authority": AUTHORITY,
            "writer": {"path": "scripts/ptq_calibration_range.py",
                       "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def out_location(out_dir, stage: str) -> Path:
    """The output file in --out-dir, which must lie outside the repository (src/quant/runner.py's guard)."""
    from src.quant.runner import QuantRunError, check_output_dir
    try:
        out = check_output_dir(out_dir, create=False)
    except QuantRunError as e:
        raise RangeRefused(e.code, str(e)) from e
    return out / OUT_NAME.format(stage=stage)


def write(path: Path, doc: dict) -> Path:
    data = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as fh:
            fh.write(data)
    except FileExistsError as e:
        raise RangeRefused("range_output_exists", f"{path} exists; the range is written once") from e
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AM-16 item 5: the range over the four calibration sets (descriptive).")
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--pair", nargs=2, action="append", metavar=("RECORD", "SUMMARY"), required=True,
                   help="a <stage>_ptq_record.json and the evaluator summary.json of its artifact (four pairs)")
    p.add_argument("--out-dir", required=True, help="directory OUTSIDE the repository")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = out_location(args.out_dir, args.stage)
        doc = compute(args.stage, args.pair)
        write(path, doc)
    except RangeRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")
        return EXIT_REFUSED
    for x in doc["per_subset"]:
        print(f"  list seed {x['list_seed']}  {x['role']:24} {x['artifact_status']:12} "
              f"all-class mIoU {x['all_class_miou']:.6f}  disease-only {x['disease_only_miou']:.6f}")
    print(f"  range all-class {doc['all_class_miou']['range']:.6f}, disease-only "
          f"{doc['disease_only_miou']['range']:.6f}  -> {path}")
    print(f"RESULT: CALIBRATION RANGE WRITTEN ({args.stage}, {len(doc['per_subset'])} calibration sets, descriptive)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
