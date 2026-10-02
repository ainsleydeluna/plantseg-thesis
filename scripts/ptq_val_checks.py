#!/usr/bin/env python3
"""Lane 8 (L-AM10) VAL checks of a PTQ run. LOCAL ONLY: these read VAL images, never TEST.

    python -B scripts/ptq_val_checks.py parity  --provenance RUN_META [--n 64]
    python -B scripts/ptq_val_checks.py backend --provenance RUN_META --out-dir OUT
    python -B scripts/ptq_val_checks.py lists   --list-dir configs/calibration

parity: the TorchScript artifact of record, loaded exactly as the evaluator loads it, and its state_dict
companion (sha256-checked against the run provenance) must give bitwise-equal outputs on n evenly spaced
VAL images (the evaluator's deterministic_subset, clean core preprocessing, one image per forward). The
same comparison ran on the 128 calibration images inside scripts/run_ptq.py; any mismatch is a STOP.

backend (lane 8 d5; Ch3 p. 154): the QNNPACK artifact and the x86 latency copy are each scored on the
full VAL split (846 rows) by the governed clean evaluator core (src/eval/evaluate.py evaluate_model),
each under its own engine, and the dataset-level metrics are differenced by src/eval/efficiency.py
backend_parity_deltas (x86 minus qnnpack). Descriptive: no threshold, never an official stage result,
never a selection input. The record goes to OUT (outside the repository, never an evaluator artifact
directory). --max-samples exists for synthetic smokes only and marks the record capped, which is not
the d5 result.

lists (lane 8 d1, its VAL half): the VAL file names (the evaluator's filename listing; no image is
opened) must be disjoint from every calibration list in the directory.

Exit codes: 0 PASS / written; 1 STOP (a parity mismatch, a list touching VAL); 2 refused or unreadable.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2
VAL_ROWS = 846
SPLIT = "val"                                        # never "test"
RECORD_SCHEMA = "plantseg-ptq-val-backend-parity/1.0.0"


class Refused(RuntimeError):
    """Refused or unreadable input: not a verdict."""


def _resolve(provenance: str) -> tuple[dict, Path]:
    from src.eval.stage_artifacts import validate_int8_artifact
    p = Path(provenance)
    try:
        prov = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read {p}: {e}") from e
    if prov.get("artifact_format") != "torchscript" or prov.get("quantization_method") != "ptq":
        raise Refused(f"{p} is not a scripts/run_ptq.py run provenance (artifact_format="
                      f"{prov.get('artifact_format')!r}, method={prov.get('quantization_method')!r})")
    return validate_int8_artifact(str(prov.get("stage")), p), p


def _companion(resolved: dict, prov_path: Path, key: str) -> Path:
    from src.eval.model_loading import sha256_file
    rec = resolved["provenance"][key]
    path = prov_path.resolve().parent / rec["path"]
    if not path.is_file() or sha256_file(path) != rec["sha256"]:
        raise Refused(f"{path} is missing or does not match the sha256 its run provenance records")
    return path


def _val_loader(indices):
    from src.eval.adapters import PlantSegEvalDataset, build_eval_loader
    return build_eval_loader(PlantSegEvalDataset(SPLIT, indices), batch_size=1, num_workers=0)


def cmd_parity(args) -> int:
    from src.eval.adapters import deterministic_subset
    from src.eval.model_loading import load_int8_student, rebuild_int8_from_state_dict
    from src.quant.ptq import output_parity

    resolved, prov_path = _resolve(args.provenance)
    stage = resolved["spec"]["stage"]
    sd_path = _companion(resolved, prov_path, "state_dict_artifact")
    ts, info = load_int8_student(resolved, require_qnnpack=True)          # selects QNNPACK first
    sd = rebuild_int8_from_state_dict(sd_path, stage=stage, method="ptq")
    indices = deterministic_subset(VAL_ROWS, args.n)
    result = output_parity({"torchscript": ts, "state_dict": sd},
                           (batch.images for batch in _val_loader(indices)),
                           [("torchscript", "state_dict")])
    record = {"check": "lane 8 ruling 4: TorchScript vs state_dict rebuild on a VAL subset",
              "stage": stage, "split": SPLIT, "n": len(indices), "source_indices": indices,
              "artifact_sha256": info.sha256, "state_dict_sha256": resolved["provenance"]
              ["state_dict_artifact"]["sha256"], "parity": result}
    print(json.dumps(record, indent=2))
    ok = result["all_equal"] and result["inputs"] == len(indices)
    print(f"RESULT: {'PTQ VAL PARITY PASS' if ok else 'PTQ VAL PARITY FAIL -- STOP'} "
          f"({result['pairs']['torchscript == state_dict']['equal']}/{len(indices)} bitwise equal)")
    return EXIT_OK if ok else EXIT_STOP


def _score(model, indices):
    from src.eval import Condition, evaluate_model
    from src.eval.adapters import build_expected_manifest_for
    manifest = build_expected_manifest_for(SPLIT, indices)
    res = evaluate_model(model, _val_loader(indices), expected_manifest=manifest,
                         condition=Condition("clean", None, None), num_classes=116,
                         background_index=0, ignore_index=255)
    if len(res.rows) != len(indices):
        raise Refused(f"evaluated {len(res.rows)} rows, expected {len(indices)}")
    return res


def cmd_backend(args) -> int:
    from src.eval.adapters import deterministic_subset
    from src.eval.efficiency import PARITY_METRICS, backend_parity_deltas
    from src.eval.model_loading import load_int8_student
    from src.quant.ptq import load_x86_latency_torchscript

    out = Path(args.out_dir).resolve()
    if out == REPO or REPO in out.parents:
        raise Refused("the descriptive record is never written inside the repository")
    resolved, prov_path = _resolve(args.provenance)
    stage = resolved["spec"]["stage"]
    x86_path = _companion(resolved, prov_path, "x86_latency_copy")
    capped = args.max_samples is not None
    indices = (deterministic_subset(VAL_ROWS, args.max_samples) if capped else list(range(VAL_ROWS)))
    target = out / f"ptq_backend_parity_val_{stage}.json"
    if target.exists():
        raise Refused(f"refusing to overwrite {target}")

    q_model, info = load_int8_student(resolved, require_qnnpack=True)
    q_res = _score(q_model, indices)
    x_model, x_meta = load_x86_latency_torchscript(x86_path, stage=stage)   # selects the x86 engine
    x_res = _score(x_model, indices)
    from src.quant.qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    q_level = {k: q_res.dataset_level.get(k) for k in PARITY_METRICS}
    x_level = {k: x_res.dataset_level.get(k) for k in PARITY_METRICS}
    record = {
        "schema": RECORD_SCHEMA,
        "evaluation_role": "descriptive_x86_backend_parity_val",
        "lane_check": "lane 8 d5 (Ch3 p. 154)",
        "is_official_stage_accuracy": False, "inferential_use": False, "model_selection_use": False,
        "threshold": None, "is_pass_fail_gate": False,
        "capped": capped, "is_the_d5_result": not capped,
        "stage": stage, "split": SPLIT, "rows": len(indices),
        "qnnpack_artifact": {"path": str(resolved["artifact_path"]), "sha256": info.sha256,
                             "engine": "qnnpack", "artifact_role": "accuracy"},
        "x86_artifact": {"path": str(x86_path), "sha256": resolved["provenance"]["x86_latency_copy"]["sha256"],
                         "engine": x_meta["engine"], "artifact_role": x_meta["artifact_role"]},
        "source_checkpoint_sha256": resolved["provenance"].get("source_checkpoint_sha256"),
        "calibration_checksum_sha256": resolved["provenance"]["calibration"]["checksum_sha256"],
        "qnnpack_dataset_level": q_level, "x86_dataset_level": x_level,
        "deltas": backend_parity_deltas(q_level, x_level),
        "batch_size": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    d = record["deltas"]["all_class_miou"]
    print(json.dumps({k: record[k] for k in ("stage", "rows", "capped", "qnnpack_dataset_level",
                                             "x86_dataset_level")}, indent=2))
    print(f"  all-class mIoU x86 - qnnpack = {d['delta_pp']} pp  -> {target}")
    print(f"RESULT: PTQ BACKEND PARITY WRITTEN ({'capped smoke, not d5' if capped else 'd5'}, "
          f"{len(indices)} VAL rows)")
    return EXIT_OK


def cmd_lists(args) -> int:
    from src.eval.adapters import list_split_stems
    from src.quant.ptq import load_calibration_list

    d = Path(args.list_dir)
    files = sorted(d.glob("ptq_calibration_seed*.json"))
    if not files:
        raise Refused(f"no ptq_calibration_seed*.json in {d}")
    val = set(list_split_stems(SPLIT))                  # names only; the locked 846 count is enforced
    touching = {}
    for f in files:
        doc, _ = load_calibration_list(f)
        hit = sorted(set(doc["selected_ids"]) & val)
        if hit:
            touching[f.name] = hit[:5]
        print(f"  {f.name}: seed {doc['seed']} ({doc['role']}) {len(doc['selected_ids'])} ids, "
              f"{len(hit)} in VAL")
    ok = not touching
    print(f"RESULT: {'CALIBRATION LISTS DISJOINT FROM VAL' if ok else 'A CALIBRATION LIST TOUCHES VAL -- STOP'} "
          f"({len(files)} lists, {len(val)} VAL names){'' if ok else f' {touching}'}")
    return EXIT_OK if ok else EXIT_STOP


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Lane 8 local VAL checks of a PTQ run (never TEST).")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("parity")
    a.add_argument("--provenance", required=True)
    a.add_argument("--n", type=int, default=64)
    b = sub.add_parser("backend")
    b.add_argument("--provenance", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--max-samples", type=int, default=None, help="synthetic smokes only (capped record)")
    c = sub.add_parser("lists")
    c.add_argument("--list-dir", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "parity" and not (1 <= args.n <= VAL_ROWS):
            raise Refused(f"--n must lie in 1..{VAL_ROWS}")
        return {"parity": cmd_parity, "backend": cmd_backend, "lists": cmd_lists}[args.cmd](args)
    except Refused as e:
        print(f"RESULT: REFUSED -- {e}")
        return EXIT_REFUSED
    except Exception as e:                            # noqa: BLE001 -- an error is never a verdict
        print(f"RESULT: ERROR -- {type(e).__name__}: {e}")
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
