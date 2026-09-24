#!/usr/bin/env python3
"""Stage-neutral evaluation CLI (A2b).

Drives the proven A2a flow in the frozen order:

    build metadata + class map
      -> prepare_artifact_request
      -> validate_artifact_request        <-- NOTHING is constructed before this succeeds
      -> apply the B6 evaluation policy   (FP32 student on CUDA only; EVALUATION_CONTRACT section 10)
      -> construct dataset adapter + model
      -> evaluate_model                   (inputs moved to the model device; warnings captured)
      -> write_artifact                   (with the run.eval_runtime record)

The CLI and its metadata are STAGE-NEUTRAL (any stage / role / precision / condition can be
*described*). Implemented construction paths are the FP32 student (E1/E2/E3, `--checkpoint`), the
converted INT8 student (E4-E7, `--provenance`, CPU/QNNPACK only) and the teacher (M4-V, CPU,
`--checkpoint` + `--teacher-config`). Corruption construction remains rejected explicitly --
metadata neutrality is never misrepresented as runtime support.

INT8 model-source validation (stage, source stage, artifact hash, backend) happens BEFORE the
dataset adapter exists, so a tampered or mismatched artifact can never touch the test split.

Importing this module has no side effects: everything happens inside `main()`.

Example (capped validation smoke, random-init, no checkpoint):
    python scripts/evaluate_model.py --stage E1 --split val --random-init \\
        --artifact-status smoke --max-samples 4 --batch-size 2 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SUPPORTED_ROLES = ("student", "teacher")
SUPPORTED_PRECISIONS = ("fp32", "int8_ptq", "int8_qat")
INT8_PRECISIONS = ("int8_ptq", "int8_qat")
SUPPORTED_CONDITIONS = ("clean",)
EXPECTED_SPLIT_ROWS = {"val": 846, "test": 1561}


class CliError(RuntimeError):
    """Argument/mode violation. Raised before any dataset or model construction."""


@dataclass(frozen=True)
class Counters:
    """Injectable construction counters so tests can prove ordering."""
    dataset: list
    model: list


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-neutral PlantSeg evaluation runner.")
    p.add_argument("--stage", default="E1",
                   choices=["teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7"])
    p.add_argument("--model-role", default="student", choices=["teacher", "student"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "int8_ptq", "int8_qat"])
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--condition", default="clean")
    p.add_argument("--corruption-severity", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--teacher-config", default=None,
                   help="thesis teacher mmseg config (required for --model-role teacher; its "
                        "IsolatedNMFLightHamHead implements M4-V)")
    p.add_argument("--provenance", default=None,
                   help="E4-E7 run-provenance JSON written by src/quant/runner.py (INT8 stages)")
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--artifact-status", default="smoke",
                   choices=["official", "provisional", "smoke"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=None,
                   help="deterministic sample cap; applied to the DATASET before the loader")
    p.add_argument("--confirm-test-split", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--device", default="cpu")
    return p


def validate_cli_args(args) -> None:
    """Pure validation. No filesystem, no dataset, no model -- safe to call in tests.

    Everything knowable from arguments alone is rejected here, before anything is constructed.
    """
    # --- implemented-support guards (metadata neutrality != runtime support) ---
    if args.model_role not in SUPPORTED_ROLES:
        raise CliError(
            f"model_role={args.model_role!r} is not supported. Supported: {SUPPORTED_ROLES}.")
    # stage and role must agree: the teacher stage is the teacher model, and only that.
    is_teacher = args.stage == "teacher" or args.model_role == "teacher"
    if is_teacher and not (args.stage == "teacher" and args.model_role == "teacher"):
        raise CliError(
            f"stage={args.stage!r} and model_role={args.model_role!r} disagree; the teacher is "
            "evaluated as stage=teacher with model_role=teacher")
    if is_teacher:
        if args.precision != "fp32":
            raise CliError(f"the teacher is evaluated in fp32; got precision={args.precision!r}")
        if args.random_init:
            raise CliError(
                "the teacher requires its fine-tuned checkpoint; --random-init is refused (no "
                "random teacher, and no ADE20K-only substitution)")
        if not args.checkpoint:
            raise CliError("the teacher stage requires --checkpoint")
        if not getattr(args, "teacher_config", None):
            raise CliError("the teacher stage requires --teacher-config (the thesis teacher config)")
        # M4-V (B61 §4): one complete pass, batch size 1, frozen manifest order, NMF seed 42.
        if args.batch_size != 1:
            raise CliError(f"M4-V evaluates the teacher with --batch-size 1; got {args.batch_size}")
    if args.precision not in SUPPORTED_PRECISIONS:
        raise CliError(
            f"precision={args.precision!r} is not supported. Supported: {SUPPORTED_PRECISIONS}.")
    if args.condition not in SUPPORTED_CONDITIONS:
        raise CliError(
            f"condition={args.condition!r} is not implemented in A2b. Supported: "
            f"{SUPPORTED_CONDITIONS}. Corruption generation does not exist yet.")

    # --- INT8 model-source consistency (E4-E7 consume a run provenance, not a raw checkpoint) ---
    # `--provenance` is newer than this function's other flags, so it is read defensively: callers
    # that predate it (existing tests, programmatic args objects) keep working unchanged.
    provenance = getattr(args, "provenance", None)
    if args.precision in INT8_PRECISIONS:
        if args.random_init:
            raise CliError(
                f"precision={args.precision!r} forbids --random-init: a quantized artifact always "
                "derives from a trained source")
        if not provenance:
            raise CliError(
                f"precision={args.precision!r} requires --provenance (the E4-E7 run-provenance "
                "JSON). A raw checkpoint cannot identify a quantized artifact.")
        if args.checkpoint:
            raise CliError(
                "--checkpoint and --provenance both identify a model source; for INT8 stages pass "
                "--provenance only")
        if args.device != "cpu":
            raise CliError(
                f"converted eager INT8 models are CPU-only; got --device {args.device!r}")
    else:
        if provenance:
            raise CliError(
                f"--provenance applies to INT8 stages only; precision={args.precision!r} takes "
                "--checkpoint")
        # --- mode / checkpoint consistency (FP32, unchanged) ---
        if args.random_init and args.checkpoint:
            raise CliError("--random-init and --checkpoint are mutually exclusive")
        if not args.random_init and not args.checkpoint:
            raise CliError("exactly one of --random-init or --checkpoint is required")
    if args.random_init and args.artifact_status != "smoke":
        raise CliError(
            f"--random-init requires --artifact-status smoke, got {args.artifact_status!r}")

    if args.batch_size < 1:
        raise CliError("--batch-size must be >= 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise CliError("--max-samples must be >= 1")

    # --- test-split guards (contract section 7) ---
    if args.split == "test":
        if not args.confirm_test_split:
            raise CliError(
                "--split test requires --confirm-test-split (the test split is not touched "
                "casually)")
        if args.random_init:
            raise CliError("--split test forbids --random-init")
        if args.max_samples is not None:
            raise CliError(
                "--split test forbids any sample cap; the full official test split is required")
        # A trained model source is mandatory; which flag supplies it depends on the precision.
        if args.precision in INT8_PRECISIONS:
            if not provenance:
                raise CliError("--split test requires --provenance for an INT8 stage")
        elif not args.checkpoint:
            raise CliError("--split test requires --checkpoint")
        if args.artifact_status == "smoke":
            raise CliError("--split test refuses artifact_status=smoke")


def run(args, *, counters: Counters | None = None, teacher_builder=None) -> Path:
    """Execute the frozen A2a flow. Returns the finalised artifact directory."""
    from src.eval import Condition, DatasetMeta, RunMeta, evaluate_model
    from src.eval.adapters import (DATASET_DOI, DATASET_NAME, PREPROCESS_PROTOCOL,
                                   PlantSegEvalDataset, build_eval_loader,
                                   build_expected_manifest_for, deterministic_subset,
                                   load_class_map)
    from src.eval.artifacts import (prepare_artifact_request, validate_artifact_request,
                                    write_artifact)
    from src.eval.eval_runtime import (EVAL_NUM_WORKERS, EvalRuntimeError, ModelDeviceForward,
                                       apply_eval_determinism, build_eval_runtime_record,
                                       check_post_eval, determinism_required,
                                       evaluate_capturing_warnings, resolve_model_device)
    from src.eval.model_loading import build_fp32_student, load_student_checkpoint

    validate_cli_args(args)

    # ---- static metadata + AUTHORITATIVE class map (verified against the pinned hash) ----
    class_map = load_class_map()

    n_total = EXPECTED_SPLIT_ROWS[args.split]
    n_rows = args.max_samples if args.max_samples is not None else n_total
    # Cap is resolved to explicit SOURCE INDICES here -- before any loader exists.
    source_indices = (deterministic_subset(n_total, n_rows)
                      if args.max_samples is not None else list(range(n_total)))

    # ---- model SOURCE validation: provenance/stage/hash, before any dataset or model exists ----
    resolved = quant_backend = None
    ckpt_path = ckpt_sha = None
    if args.precision in INT8_PRECISIONS:
        from src.eval.model_loading import select_int8_backend
        from src.eval.stage_artifacts import resolve_evaluation_source
        # A tampered artifact, a wrong source stage or a stage mismatch fails HERE.
        resolved = resolve_evaluation_source(args.stage, provenance=args.provenance)
        quant_backend = select_int8_backend(require_qnnpack=True)   # refuses a non-QNNPACK build
        ckpt_path = str(resolved["artifact_path"])
        ckpt_sha = resolved["artifact_sha256"]
    elif args.model_role == "teacher":
        from src.eval.stage_artifacts import validate_teacher_artifact
        # Teacher checkpoint structure/identity is proven before any dataset exists.
        resolved = validate_teacher_artifact(args.checkpoint)
        ckpt_path = str(resolved["checkpoint_path"])
        ckpt_sha = resolved["checkpoint_sha256"]
    elif args.checkpoint:
        from src.eval.model_loading import sha256_file
        from src.eval.stage_artifacts import validate_fp32_artifact
        validate_fp32_artifact(args.stage, args.checkpoint)        # stage metadata must agree
        ckpt_path = str(args.checkpoint)
        ckpt_sha = sha256_file(Path(args.checkpoint))

    run_id = args.run_id or (
        f"{args.stage}_{args.split}_{args.condition}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

    # Expected manifest from a FILENAME LISTING only -- no dataset adapter, no image/mask opened.
    expected_manifest = build_expected_manifest_for(args.split, source_indices)

    request = prepare_artifact_request(
        out_dir=Path(args.out_dir),
        artifact_status=args.artifact_status,
        run_id=run_id,
        run=RunMeta(stage=args.stage, model_role=args.model_role, precision=args.precision,
                    quant_backend=quant_backend, checkpoint_path=ckpt_path,
                    checkpoint_sha256=ckpt_sha,
                    random_init=bool(args.random_init), device=args.device),
        dataset=DatasetMeta(
            name=DATASET_NAME, doi=DATASET_DOI, split=args.split,
            condition=Condition(args.condition,
                                None if args.condition == "clean" else args.condition,
                                None if args.condition == "clean" else args.corruption_severity),
            preprocess_protocol=PREPROCESS_PROTOCOL,
            expected_rows=len(source_indices)),
        expected_manifest=expected_manifest,
        class_map=list(class_map.entries),
        repo_root=REPO)

    provenance = validate_artifact_request(request)     # <-- refusal happens HERE, before step 5

    # ---- evaluation determinism policy (contract section 10): an FP32 student on CUDA gets E1's
    # B6 settings, without a reseed, before anything can initialise CUDA. Everything else applies
    # nothing and records the inherited state.
    cpu_only = args.model_role == "teacher" or args.precision in INT8_PRECISIONS
    policy_applied = determinism_required(args.model_role, args.precision, args.device)
    policy_info = apply_eval_determinism() if policy_applied else None
    ck_info = None

    # ---- step 5: only now may the dataset adapter and the model be constructed ----
    if counters is not None:
        counters.dataset.append(("adapter", tuple(source_indices)))
    adapter = PlantSegEvalDataset(args.split, source_indices)

    if counters is not None:
        counters.model.append(("model", args.random_init))
    if args.model_role == "teacher":
        from src.eval.model_loading import load_teacher_model
        model = load_teacher_model(resolved, builder=teacher_builder,
                                   config_path=getattr(args, "teacher_config", None))[0]
    elif resolved is not None:
        from src.eval.model_loading import load_int8_student
        model = load_int8_student(resolved, require_qnnpack=True)[0]
    elif args.random_init:
        model = build_fp32_student(device=args.device)
    else:
        model, ck_info = load_student_checkpoint(args.checkpoint, map_location=args.device)
        if ck_info.sha256 != ckpt_sha:
            raise EvalRuntimeError(
                "checkpoint_changed_after_validation",
                f"checkpoint bytes changed between validation ({ckpt_sha}) and load "
                f"({ck_info.sha256})")

    # Inputs go to the model's construction device, exactly as train_e1.validate moves them.
    model_device = resolve_model_device(args.device, cpu_only=cpu_only)
    fwd = ModelDeviceForward(model_device)
    loader = build_eval_loader(adapter, args.batch_size, num_workers=EVAL_NUM_WORKERS)
    result, warn_summary = evaluate_capturing_warnings(
        evaluate_model, model, loader, expected_manifest=expected_manifest,
        condition=request.dataset.condition, num_classes=116,
        background_index=0, ignore_index=255, forward=fwd)
    check_post_eval(model=model, model_device=model_device, fwd=fwd,
                    policy_applied=policy_applied)
    runtime = build_eval_runtime_record(
        model=model, model_device=model_device, fwd=fwd, batch_size=args.batch_size,
        forward_batches=result.forward_batches, policy_applied=policy_applied,
        policy_info=policy_info, ckpt_info=ck_info, warn_summary=warn_summary)
    return write_artifact(result, request, provenance, eval_runtime=runtime)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = run(args)
    except Exception as e:                                # noqa: BLE001 -- reported, not swallowed
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"artifact written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
