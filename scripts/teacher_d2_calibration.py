#!/usr/bin/env python3
"""D2 VAL: the teacher's confidence and calibration on VAL under the R3 evaluation procedure (AM-18 item
1(b); lane L-TEACHER-DIAG).

The pass is scripts/evaluate_model.py run()'s teacher path, call for call and in the same order, with one
difference: the evaluator core gets `batch_forward=CalibratingBatchForward(ModelDeviceForward(cpu))`
instead of `forward=ModelDeviceForward(cpu)`, because the ECE needs the targets. The hook calls the same
forward, accumulates from the tensor the core argmaxes and hands that tensor back unchanged
(src/eval/calibration.py). Canvas protocol, CPU, batch size 1, the VAL manifest of record, the M4-V NMF
stream seeded 42 begun by load_teacher_model.

Roles and purposes (P3, P4):
  --teacher-role record --purpose item1     the item 1(b) pass        teacher_d2_<UTC>.json
  --teacher-role record --purpose control   AM-18 item 5's re-score   teacher_d2_control_<UTC>.json
                                            (--of-record-output: the item 1(b) JSON; reproduction_check)
  --teacher-role arm --arm-id R1|R2 ...     an arm teacher, gates nothing   teacher_d2_arm-<id>_<UTC>.json
Every role takes --val-reference (the R3-equivalent teacher CPU re-score), read summary-first; its
manifest hash must equal this pass's. Record and control: |all-class mIoU - R3| <= 1e-5 is checked before
the evaluator artifact is written; a failure writes no artifact, a JSON with status "not reproduced",
and exits 1 (P18). The evaluator artifact goes to --artifact-dir/<run id> (outside the repository); its
four file hashes are recorded, with per_image_equal and npz_equal against --val-reference as information.

Exit codes: 0 written, 1 STOP, 2 refusal or usage, 4 unexpected exception. Real mode (the CLI) needs
--script-commit and --script-commit-dl-id and refuses every run until K-part (P8). The stub teacher is
reachable only from scripts/smoke_teacher_calibration.py, which calls run(args, model_factory=...).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval import teacher_diag as td  # noqa: E402

SCRIPT = "scripts/teacher_d2_calibration.py"
SCHEMA = "plantseg-teacher-d2/1.0.0"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D2 VAL: teacher confidence and calibration (AM-18 item 1(b)).")
    td.add_common_flags(p)
    td.add_teacher_flags(p)
    p.add_argument("--purpose", help="record role: item1 or control (P4)")
    p.add_argument("--of-record-output", help="control: the item 1(b) teacher_d2 JSON")
    p.add_argument("--val-reference", required=True, help="the teacher-stage canvas VAL artifact (R5)")
    p.add_argument("--artifact-dir", required=True,
                   help="parent of the evaluator artifact this pass writes; outside the repository")
    p.add_argument("--max-samples", type=int, default=None, help="stub mode only: evenly spaced VAL subset")
    return p


def check_d2_flags(args, *, real: bool) -> None:
    if args.teacher_role == "record":
        if args.purpose not in td.PURPOSES:
            raise td.Refused(f"the record role requires --purpose in {td.PURPOSES}")
        if (args.purpose == "control") != (args.of_record_output is not None):
            raise td.Refused("--of-record-output goes with --purpose control, and only with it")
    elif args.purpose is not None or args.of_record_output is not None:
        raise td.Refused("--purpose and --of-record-output belong to the record role")
    if args.max_samples is not None:
        if real:
            raise td.Refused("--max-samples is a stub-mode flag; a real run scores all 846 VAL images")
        if args.max_samples < 1 or args.max_samples > td.VAL_ROWS:
            raise td.Refused(f"--max-samples must be in 1..{td.VAL_ROWS}")


def output_kind(args) -> str:
    if args.teacher_role == "arm":
        return f"teacher_d2_arm-{args.arm_id}"
    return "teacher_d2_control" if args.purpose == "control" else "teacher_d2"


def artifact_equalities(a: Path, b: Path) -> dict:
    """Information only: per_image.jsonl byte equality and NPZ array equality (EVALUATION_CONTRACT 10(d))."""
    import numpy as np
    per_image = (a / "per_image.jsonl").read_bytes() == (b / "per_image.jsonl").read_bytes()
    with np.load(a / "sufficient_stats.npz", allow_pickle=False) as za, \
            np.load(b / "sufficient_stats.npz", allow_pickle=False) as zb:
        npz = sorted(za.files) == sorted(zb.files) and all(
            za[k].dtype == zb[k].dtype and za[k].shape == zb[k].shape and za[k].tobytes() == zb[k].tobytes()
            for k in za.files)
    return {"per_image_equal": bool(per_image), "npz_equal": bool(npz),
            "rule": "per_image.jsonl byte-equal; NPZ keys, dtypes, shapes and raw bytes equal (information)"}


def run(args, *, model_factory=None) -> int:
    return td.run_with_exit_codes(_run, args, model_factory=model_factory)


def _run(args, *, model_factory=None) -> int:
    stub = model_factory is not None
    real = not stub
    start, t0 = td.utc_now(), time.monotonic()

    # ---- flags (no file read) ----
    td.check_common_flags(args, real=real)
    td.check_teacher_flags(args)
    check_d2_flags(args, real=real)
    td.require_provenance_field_count(real)
    out_dir = td.require_outside_repo(args.out_dir, "--out-dir")
    art_root = td.require_outside_repo(args.artifact_dir, "--artifact-dir")

    # ---- commit binding, one output per kind ----
    code = td.require_commit_binding(args.script_commit, args.script_commit_dl_id) if real else td.code_provenance()
    kind = output_kind(args)
    repeat = td.require_single_output(out_dir, kind, args, exempt=args.purpose == "control")
    stamp = td.output_stamp(args, start)
    out_path = out_dir / f"{kind}_{stamp}.json"
    if out_path.exists():
        raise td.Refused(f"{out_path.name} already exists")

    # ---- inputs read before anything unpickles: the reference (summary first), the of-record output,
    # the teacher files ----
    from src.eval.adapters import deterministic_subset
    source_indices = (deterministic_subset(td.VAL_ROWS, args.max_samples) if args.max_samples is not None
                      else list(range(td.VAL_ROWS)))
    ref = td.check_val_reference(args.val_reference, stub=stub, expected_rows=len(source_indices))
    of_record = None
    if args.purpose == "control":
        doc, sha = td.read_diag_output(args.of_record_output, script=SCRIPT, stub=stub, what="--of-record-output")
        if doc.get("purpose") != "item1" or (doc.get("teacher") or {}).get("role") != "record":
            raise td.Refused("--of-record-output is not the record role's item 1(b) output")
        of_record = {"doc": doc, "path": str(args.of_record_output), "sha256": sha}
    inputs = td.verify_teacher_inputs(args, stub=stub)

    # ---- data root: M11, then the evaluator's own construction order (evaluate_model.run) ----
    from configs.data import DATA
    td.refuse_test_path(DATA["root"], "PLANTSEG_DATA_ROOT")
    m11 = td.check_m11(DATA["root"])

    from src.eval import Condition, DatasetMeta, RunMeta, evaluate_model
    from src.eval.adapters import (DATASET_DOI, DATASET_NAME, build_eval_loader, build_expected_manifest_for,
                                   build_protocol_adapter, load_class_map)
    from src.eval.artifacts import (ARTIFACT_FILES, MANIFEST_NAME, prepare_artifact_request,
                                    validate_artifact_request, verify_artifact, write_artifact)
    from src.eval.calibration import CalibratingBatchForward, CalibrationAccumulator
    from src.eval.eval_runtime import (EVAL_NUM_WORKERS, ModelDeviceForward, build_eval_runtime_record,
                                       check_post_eval, determinism_required, evaluate_capturing_warnings,
                                       resolve_model_device)
    from src.eval.protocols import CANVAS, PROTOCOL_IDS

    class_map = load_class_map()
    run_id = f"{kind}_{stamp}"
    expected_manifest = build_expected_manifest_for("val", source_indices)
    request = prepare_artifact_request(
        out_dir=art_root / run_id,
        artifact_status="smoke" if stub else "provisional",
        run_id=run_id,
        run=RunMeta(stage="teacher", model_role="teacher", precision="fp32", quant_backend=None,
                    checkpoint_path=str(inputs.ckpt), checkpoint_sha256=inputs.sha256,
                    random_init=False, device="cpu"),
        dataset=DatasetMeta(name=DATASET_NAME, doi=DATASET_DOI, split="val", condition=Condition("clean", None, None),
                            preprocess_protocol=PROTOCOL_IDS[CANVAS], expected_rows=len(source_indices)),
        expected_manifest=expected_manifest,
        class_map=list(class_map.entries),
        repo_root=REPO)
    provenance = validate_artifact_request(request)
    if provenance.split_manifest_sha256 != ref["split_manifest_sha256"]:
        raise td.Refused("this pass's VAL manifest hash differs from --val-reference's")
    policy_applied = determinism_required("teacher", "fp32", "cpu")      # False: the teacher runs on the CPU
    adapter = build_protocol_adapter(CANVAS, "val", source_indices)

    # ---- the load (evaluator form), after-load checks, the one provenance call site ----
    loaded = td.load_teacher("evaluator", inputs, model_factory=model_factory)
    checks = td.after_load_checks(loaded, inputs, stub=stub)
    teacher = td.teacher_record(loaded, inputs, checks, stub=stub)
    if of_record is not None:
        diffs = td.same_teacher(teacher, of_record["doc"].get("teacher") or {})
        if diffs:
            raise td.Refused(f"the control re-score's teacher differs from the of-record output's: {diffs}")
    model = loaded.eval_model

    # ---- the pass ----
    model_device = resolve_model_device("cpu", cpu_only=True)
    fwd = ModelDeviceForward(model_device)
    watch = td.RngWatchForward(fwd, total=len(source_indices), label="d2 val")
    acc = CalibrationAccumulator()
    hook = CalibratingBatchForward(watch, acc)
    loader = build_eval_loader(adapter, 1, num_workers=EVAL_NUM_WORKERS)
    result, warn_summary = evaluate_capturing_warnings(
        evaluate_model, model, loader, expected_manifest=expected_manifest,
        condition=request.dataset.condition, num_classes=116,
        background_index=0, ignore_index=255, batch_forward=hook)
    check_post_eval(model=model, model_device=model_device, fwd=fwd, policy_applied=policy_applied)
    runtime = build_eval_runtime_record(
        model=model, model_device=model_device, fwd=fwd, batch_size=1,
        forward_batches=result.forward_batches, policy_applied=policy_applied,
        policy_info=None, ckpt_info=None, warn_summary=warn_summary)

    # ---- integrity checks ----
    n_rows = len(source_indices)
    rng_state = watch.require_unchanged()
    if loaded.adapter.nmf_stream is not loaded.stream or loaded.stream.draws != n_rows:
        raise td.Stop(f"M4-V stream: {loaded.stream.draws} draws for {n_rows} images, or the stream object changed")
    cal = acc.result()
    sum_tp, sum_gt = int(result.dataset_tp.sum()), int(result.dataset_gt.sum())
    if cal["correct"] != sum_tp or cal["n"] != sum_gt:
        raise td.Stop(f"calibration totals disagree with the core: correct {cal['correct']} vs TP {sum_tp}, "
                      f"n {cal['n']} vs GT {sum_gt}")
    if hook.calls != result.forward_batches or result.forward_batches != n_rows:
        raise td.Stop(f"{hook.calls} hook calls, {result.forward_batches} forward batches, {n_rows} rows")

    miou = result.dataset_level["all_class_miou"]
    if args.teacher_role == "record":
        reference = ref["all_class_miou"] if stub else td.R3_VAL_MIOU
        delta = None if miou is None else abs(miou - reference)
        passed = delta is not None and delta <= td.R3_TOLERANCE
        gate = {"rule": "|all_class_miou - reference| <= tolerance, before the artifact is written (P18)",
                "reference": reference, "reference_source": "--val-reference (stub mode)" if stub else "R3",
                "tolerance": td.R3_TOLERANCE, "value": miou, "delta": delta, "passed": passed}
    else:
        gate, passed = "nothing", True

    doc = td.base_document(SCRIPT, args, stub=stub, start_utc=start, code=code, extra={
        "schema": SCHEMA, "role": args.teacher_role, "arm_id": args.arm_id, "purpose": args.purpose,
        "repeat": repeat, "gates": gate, "teacher": teacher,
        "inputs": {"val_reference": ref,
                   "of_record_output": None if of_record is None else
                   {"path": of_record["path"], "sha256": of_record["sha256"]},
                   "data_root": str(DATA["root"]), "m11": m11},
        "pass": {"split": "val", "rows": n_rows, "protocol": PROTOCOL_IDS[CANVAS], "device": "cpu", "batch_size": 1,
                 "split_manifest_sha256": provenance.split_manifest_sha256, "run_id": run_id,
                 "all_class_miou": miou, "eval_warnings": runtime["eval_warnings"],
                 "accumulation_warnings": list(hook.accumulation_warnings)},
        "calibration": cal,
        "checks": {"sum_correct_equals_sum_tp": cal["correct"] == sum_tp, "n_equals_sum_gt": cal["n"] == sum_gt,
                   "sum_tp": sum_tp, "sum_gt": sum_gt,
                   "hook_calls": hook.calls, "forward_batches": result.forward_batches,
                   "rng_state_unchanged": watch.unchanged(), "rng_state_sha256": rng_state,
                   "rng_rule": "the caller's CPU RNG hash before the first forward == after the last, and "
                               "unchanged at every forward in between (P9)",
                   "nmf_stream": {"begin": loaded.stream_description, "end": loaded.stream.describe(),
                                  "draws": loaded.stream.draws, "expected_draws": n_rows,
                                  "object_is_adapter_stream": loaded.adapter.nmf_stream is loaded.stream}},
    })
    if args.purpose == "control":
        rec_cal = of_record["doc"].get("calibration") or {}
        doc["reproduction_check"] = True
        doc["control"] = {"ece_b": cal["ece"], "ece_of_record": rec_cal.get("ece"),
                          "ece_disease_b": cal["ece_disease"], "ece_disease_of_record": rec_cal.get("ece_disease"),
                          "all_class_miou_b": miou,
                          "all_class_miou_of_record": (of_record["doc"].get("pass") or {}).get("all_class_miou")}

    if not passed:                                       # P18: no artifact on a failed gate
        doc["status"] = "not reproduced"
        doc["artifact"] = None
        td.finish_output(out_path, doc, start, t0)
        print(f"STOP: not reproduced: all_class_miou {miou!r}, delta {gate['delta']!r} > {td.R3_TOLERANCE}",
              file=sys.stderr)
        return td.EXIT_STOP

    art_dir = write_artifact(result, request, provenance, eval_runtime=runtime)
    verify_artifact(art_dir)
    doc["artifact"] = {"dir": str(art_dir), "run_id": run_id, "artifact_status": request.artifact_status,
                       "files_sha256": {n: td.file_sha256(art_dir / n) for n in sorted([*ARTIFACT_FILES, MANIFEST_NAME])},
                       "against_val_reference": artifact_equalities(art_dir, Path(args.val_reference))}
    doc["status"] = "written"
    td.finish_output(out_path, doc, start, t0)
    return td.EXIT_OK


def main(argv=None) -> int:
    return td.cli_main(build_parser(), run, argv)


if __name__ == "__main__":
    sys.exit(main())
