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
*described*). Implemented construction paths are the FP32 student (E1/E2/E3 and the descriptive arms
A/F/G, `--checkpoint`), the converted INT8 student (E4-E7, `--provenance`, CPU/QNNPACK only) and the
teacher (M4-V, CPU, `--checkpoint` + `--teacher-config` + `--teacher-ckpt-sha256`, the checkpoint's
SHA-256, compared before the file is parsed; R6). Corruption construction remains rejected
explicitly -- metadata neutrality is never misrepresented as runtime support.

INT8 model-source validation (stage, source stage, artifact hash, backend) happens BEFORE the
dataset adapter exists, so a tampered or mismatched artifact can never touch the test split.

IF-T1 (TL-6(b), TL-1; lane LEFF B-1b): `--expected-checkpoint-sha256` (the SHA-256 of the handed file,
`--checkpoint` or an INT8 stage's `--provenance`, compared before its first parse and again after
validation), `--expected-converted-sha256` (an INT8 stage's converted artifact, compared after validation
and again before the load) and `--threads` (torch.set_num_threads before anything is loaded, checked
against run.eval_runtime before the write) are required with `--split test --artifact-status official`,
optional elsewhere, and hash, compare and set nothing when absent.

`--protocol` (lane L-AM13; EVALUATION_CONTRACT section 11) selects the geometry for any model:
`canvas` (default) is the 512x512 canvas, `core_preprocess/1.0.0`, the protocol of every inferential
analysis; `upstream` is the tqwei05/PlantSeg test geometry, `upstream/1.0.0` (aspect-preserving
rescale to short side 512 / long side <= 2048, pad to a multiple of 32, batch size 1, logits resized
back and scored against the original-resolution mask), descriptive only. The test-split guards are
identical under both.

Importing this module has no side effects: everything happens inside `main()`.

Example (capped validation smoke, random-init, no checkpoint):
    python scripts/evaluate_model.py --stage E1 --split val --random-init \\
        --artifact-status smoke --max-samples 4 --batch-size 2 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
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
SUPPORTED_PROTOCOLS = ("canvas", "upstream")         # src/eval/protocols.py PROTOCOL_NAMES
EXPECTED_SPLIT_ROWS = {"val": 846, "test": 1561}


class CliError(RuntimeError):
    """Argument/mode violation. Raised before any dataset or model construction."""


@dataclass(frozen=True)
class Counters:
    """Injectable construction counters so tests can prove ordering."""
    dataset: list
    model: list


def build_parser() -> argparse.ArgumentParser:
    from src.eval.artifacts import STAGES                # the artifact schema's stages are the choices (AQ25)
    p = argparse.ArgumentParser(description="Stage-neutral PlantSeg evaluation runner.")
    p.add_argument("--stage", default="E1", choices=list(STAGES))
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
    p.add_argument("--teacher-ckpt-sha256", default=None,
                   help="the teacher checkpoint's SHA-256, 64 lowercase hex characters (required for "
                        "--stage teacher, refused on any other stage); compared before the file is "
                        "parsed")
    p.add_argument("--expected-checkpoint-sha256", default=None,
                   help="the sha256 of the file this run is handed (--checkpoint, or --provenance for an "
                        "INT8 stage); compared before its first parse and again after validation; required "
                        "with --split test and --artifact-status official, refused with --random-init "
                        "(IF-T1)")
    p.add_argument("--expected-converted-sha256", default=None,
                   help="INT8 only: the sha256 of the converted artifact the provenance names; compared "
                        "after validation and again before the load; required with --split test and "
                        "--artifact-status official (IF-T1)")
    p.add_argument("--threads", default=None,
                   help="torch.set_num_threads(T) before anything is loaded or built, read back, and "
                        "checked against the recorded run.eval_runtime.torch_num_threads before the "
                        "artifact is written; required with --split test and --artifact-status official "
                        "(TL-1)")
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
    p.add_argument("--protocol", default="canvas", choices=list(SUPPORTED_PROTOCOLS),
                   help="canvas: the 512x512 canvas (core_preprocess/1.0.0; default). upstream: the "
                        "tqwei05/PlantSeg test geometry scored at original resolution "
                        "(upstream/1.0.0; descriptive; requires --batch-size 1)")
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
    # --- evaluation protocol (lane L-AM13). Read defensively, like --provenance below: callers that
    # predate the flag evaluate the canvas protocol, unchanged. ---
    protocol = getattr(args, "protocol", "canvas")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise CliError(f"protocol={protocol!r} is not supported. Supported: {SUPPORTED_PROTOCOLS}.")
    if protocol == "upstream" and args.batch_size != 1:
        raise CliError(
            "the upstream protocol runs whole-image inference at batch size 1; got --batch-size "
            f"{args.batch_size}")

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

    # --- teacher checkpoint identity (R6, L-CKPT-GUARD). Read defensively, like --provenance: an args
    # object without the attribute is a non-teacher caller. ---
    teacher_sha = getattr(args, "teacher_ckpt_sha256", None)
    if args.stage == "teacher":
        if teacher_sha is None:
            raise CliError("[teacher_ckpt_sha256_required] the teacher stage requires "
                           "--teacher-ckpt-sha256, the checkpoint's SHA-256, compared before the "
                           "file is parsed")
        from src.distill.teacher import sha256_format_error
        sha_error = sha256_format_error(teacher_sha)
        if sha_error is not None:
            raise CliError(f"[teacher_ckpt_sha256_format] --teacher-ckpt-sha256 {sha_error}")
    elif teacher_sha is not None:
        raise CliError(f"[teacher_ckpt_sha256_not_teacher] --teacher-ckpt-sha256 applies to the teacher "
                       f"stage only; stage={args.stage!r} takes none")

    # --- IF-T1 (TL-6(b), TL-1; lane LEFF B-1b): the handed file, the converted artifact and the thread
    # count. Each flag is required under R (--split test with --artifact-status official), optional
    # elsewhere, and refused where it does not apply. Appended after every guard above, which keep
    # precedence; read defensively, like --provenance, so an args object without them behaves as before. ---
    in_r = args.split == "test" and args.artifact_status == "official"
    expected = getattr(args, "expected_checkpoint_sha256", None)
    if expected is not None and args.random_init:
        raise CliError("[expected_checkpoint_sha256_not_applicable] --expected-checkpoint-sha256 names the "
                       "file a run is handed; --random-init is handed none")
    if expected is not None:
        from src.distill.teacher import sha256_format_error
        sha_error = sha256_format_error(expected)
        if sha_error is not None:
            raise CliError(f"[expected_checkpoint_sha256_format] --expected-checkpoint-sha256 {sha_error}")
    if expected is None and in_r:
        raise CliError("[expected_checkpoint_sha256_required] --split test with --artifact-status official "
                       "requires --expected-checkpoint-sha256, the sha256 of the file this run is handed "
                       "(--checkpoint, or --provenance for an INT8 stage)")
    if args.stage == "teacher" and expected is not None and expected != teacher_sha:
        raise CliError(f"[expected_checkpoint_sha256_teacher_disagrees] --expected-checkpoint-sha256 "
                       f"{expected} and --teacher-ckpt-sha256 {teacher_sha} differ; for the teacher both "
                       "name --checkpoint and must be equal")
    converted = getattr(args, "expected_converted_sha256", None)
    if converted is not None and (args.precision not in INT8_PRECISIONS or args.random_init):
        raise CliError(f"[expected_converted_sha256_not_int8] --expected-converted-sha256 applies to an INT8 "
                       f"stage handed --provenance; this run (precision={args.precision!r}, "
                       f"random_init={bool(args.random_init)}) takes none")
    if converted is not None:
        from src.distill.teacher import sha256_format_error
        sha_error = sha256_format_error(converted)
        if sha_error is not None:
            raise CliError(f"[expected_converted_sha256_format] --expected-converted-sha256 {sha_error}")
    if converted is None and in_r and args.precision in INT8_PRECISIONS:
        raise CliError("[expected_converted_sha256_required] --split test with --artifact-status official "
                       "requires --expected-converted-sha256 for an INT8 stage, the sha256 of the converted "
                       "artifact its provenance names")
    if parse_threads(getattr(args, "threads", None)) is None and in_r:
        raise CliError("[threads_required] --split test with --artifact-status official requires --threads, "
                       "the torch thread count of the execution context of record")


def parse_threads(value) -> int | None:
    """`--threads` (TL-1): a positive integer, from the CLI's text or a programmatic int; None when not given.

    L-CORR-B2's rule for scripts/evaluate_corruptions.py (parse_threads, lane/l-corr-eval at da8469c), so both
    evaluators answer one input alike: an int that is not a bool, or an ASCII digit string, greater than 0.
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and value.isascii() and int(value) > 0:
        return int(value)
    raise CliError(f"[threads_invalid] --threads must be a positive integer, got {value!r}")


def apply_threads(args) -> int | None:
    """TL-1: torch.set_num_threads(T) with a read-back, before anything is loaded or built. None: not set."""
    threads = parse_threads(getattr(args, "threads", None))
    if threads is None:
        return None
    import torch
    torch.set_num_threads(threads)
    got = torch.get_num_threads()
    if got != threads:
        raise CliError(f"[threads_not_applied] torch.set_num_threads({threads}) left torch.get_num_threads() "
                       f"at {got}")
    return threads


def handed_file(args) -> Path:
    """IF-T1: the file this run is handed: --provenance for an INT8 stage, else --checkpoint. The one place
    that names it."""
    return Path(getattr(args, "provenance", None) if args.precision in INT8_PRECISIONS else args.checkpoint)


def identity_sha256(path) -> str:
    """IF-T1's one hashing helper (P1, P2's re-hash, P3). The evaluator's own hash sites are unchanged."""
    from src.eval.model_loading import sha256_file
    return sha256_file(Path(path))


def hash_before_parse(args) -> Path | None:
    """IF-T1 P1: the handed file's sha256 before its first parse, compared with --expected-checkpoint-sha256.

    Without the flag nothing is hashed and None is returned. A missing file is left to the validator's own
    refusal, as before the flag existed.
    """
    expected = getattr(args, "expected_checkpoint_sha256", None)
    if expected is None:
        return None
    path = handed_file(args)
    if path.is_file():
        actual = identity_sha256(path)
        if actual != expected:
            raise CliError(f"[expected_checkpoint_sha256_mismatch] {path} has sha256 {actual}, expected "
                           f"{expected}; the file was not parsed")
    return path


def run(args, *, counters: Counters | None = None, teacher_builder=None) -> Path:
    """Execute the frozen A2a flow. Returns the finalised artifact directory."""
    from src.eval import Condition, DatasetMeta, RunMeta, evaluate_model
    from src.eval.adapters import (DATASET_DOI, DATASET_NAME, build_eval_loader,
                                   build_expected_manifest_for, build_protocol_adapter,
                                   deterministic_subset, load_class_map,
                                   protocol_forward_kwargs)
    from src.eval.artifacts import (prepare_artifact_request, validate_artifact_request,
                                    write_artifact)
    from src.eval.eval_runtime import (EVAL_NUM_WORKERS, EvalRuntimeError, ModelDeviceForward,
                                       apply_eval_determinism, build_eval_runtime_record,
                                       check_post_eval, determinism_required,
                                       evaluate_capturing_warnings, resolve_model_device)
    from src.eval.model_loading import build_fp32_student, load_student_checkpoint
    from src.eval.protocols import PROTOCOL_IDS

    validate_cli_args(args)
    threads = apply_threads(args)        # IF-T1 (TL-1): before the class map, the source checks and any build
    protocol = getattr(args, "protocol", "canvas")      # lane L-AM13; see validate_cli_args

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
    handed = hash_before_parse(args)     # IF-T1 P1: the handed file, before any validator parses it
    if args.precision in INT8_PRECISIONS:
        from src.eval.model_loading import select_int8_backend
        from src.eval.stage_artifacts import resolve_evaluation_source
        # A tampered artifact, a wrong source stage or a stage mismatch fails HERE.
        resolved = resolve_evaluation_source(args.stage, provenance=args.provenance)
        if args.artifact_status == "official":
            # an official E4/E7 needs the committed AM-10 list of record (L-AM10) and its PTQ record of
            # role of_record (LQP); sensitivity and synthetic lists, d3 replicates, arm directories and
            # legacy state_dict PTQ artifacts are scored descriptively only
            from src.eval.stage_artifacts import int8_official_error
            official_error = int8_official_error(resolved, repo_root=REPO)
            if official_error:
                raise CliError(official_error)
        quant_backend = select_int8_backend(require_qnnpack=True)   # refuses a non-QNNPACK build
        ckpt_path = str(resolved["artifact_path"])
        ckpt_sha = resolved["artifact_sha256"]
    elif args.model_role == "teacher":
        from src.eval.stage_artifacts import validate_teacher_artifact
        # Teacher checkpoint identity (hash first, R6) and structure are proven before any dataset
        # exists.
        resolved = validate_teacher_artifact(args.checkpoint,
                                             expected_sha256=getattr(args, "teacher_ckpt_sha256", None))
        ckpt_path = str(resolved["checkpoint_path"])
        ckpt_sha = resolved["checkpoint_sha256"]
    elif args.checkpoint:
        from src.eval.model_loading import sha256_file
        from src.eval.stage_artifacts import validate_fp32_artifact
        validate_fp32_artifact(args.stage, args.checkpoint)        # stage metadata must agree
        ckpt_path = str(args.checkpoint)
        ckpt_sha = sha256_file(Path(args.checkpoint))

    # ---- IF-T1 P2: after validation the handed file is still the one P1 hashed. FP32's ckpt_sha was
    # hashed after validate_fp32_artifact's parse; the teacher's recorded hash is its pre-parse hash and
    # INT8's is the converted artifact's, so those two re-hash the handed file. Then INT8's converted
    # artifact, after the official gate. ----
    expected = getattr(args, "expected_checkpoint_sha256", None)
    if expected is not None:
        after = ckpt_sha if resolved is None else identity_sha256(handed)
        if after != expected:
            raise CliError(f"[source_changed_during_validation] {handed} had sha256 {expected} before "
                           f"validation and {after} after it")
    expected_converted = getattr(args, "expected_converted_sha256", None)
    if expected_converted is not None and resolved["artifact_sha256"] != expected_converted:
        raise CliError(f"[expected_converted_sha256_mismatch] the converted artifact "
                       f"{resolved['artifact_path']} has sha256 {resolved['artifact_sha256']}, expected "
                       f"{expected_converted}; it was not loaded")

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
            preprocess_protocol=PROTOCOL_IDS[protocol],
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
    adapter = build_protocol_adapter(protocol, args.split, source_indices)

    if counters is not None:
        counters.model.append(("model", args.random_init))
    if args.model_role == "teacher":
        from src.eval.model_loading import load_teacher_model
        model = load_teacher_model(resolved, builder=teacher_builder,
                                   config_path=getattr(args, "teacher_config", None),
                                   expected_sha256=getattr(args, "teacher_ckpt_sha256", None))[0]
    elif resolved is not None:
        from src.eval.model_loading import load_int8_student
        if expected_converted is not None:          # IF-T1 P3: load_int8_student does not re-hash
            actual = identity_sha256(resolved["artifact_path"])
            if actual != expected_converted:
                raise EvalRuntimeError(
                    "converted_changed_before_load",
                    f"the converted artifact {resolved['artifact_path']} had sha256 {expected_converted} at "
                    f"validation and {actual} before the load; it was not loaded")
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

    # Inputs go to the model's construction device, exactly as train_e1.validate moves them. Under the
    # upstream protocol the same hook runs inside the upstream batch forward (src/eval/protocols.py).
    model_device = resolve_model_device(args.device, cpu_only=cpu_only)
    fwd = ModelDeviceForward(model_device)
    loader = build_eval_loader(adapter, args.batch_size, num_workers=EVAL_NUM_WORKERS)
    result, warn_summary = evaluate_capturing_warnings(
        evaluate_model, model, loader, expected_manifest=expected_manifest,
        condition=request.dataset.condition, num_classes=116,
        background_index=0, ignore_index=255, **protocol_forward_kwargs(protocol, fwd))
    check_post_eval(model=model, model_device=model_device, fwd=fwd,
                    policy_applied=policy_applied)
    runtime = build_eval_runtime_record(
        model=model, model_device=model_device, fwd=fwd, batch_size=args.batch_size,
        forward_batches=result.forward_batches, policy_applied=policy_applied,
        policy_info=policy_info, ckpt_info=ck_info, warn_summary=warn_summary)
    if threads is not None and runtime["torch_num_threads"] != threads:        # IF-T1 (TL-1)
        raise EvalRuntimeError(
            "threads_drifted",
            f"run.eval_runtime.torch_num_threads is {runtime['torch_num_threads']}, --threads is {threads}; "
            "no artifact was written")
    return write_artifact(result, request, provenance, eval_runtime=runtime)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = run(args)
    except Exception as e:                                # noqa: BLE001 -- reported, not swallowed
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"artifact written: {out}")
    summary = json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))
    # AM-5 (lane L-AM5): the zero-disease count is reported with every artifact.
    am5 = summary.get("am5") or {}
    print(f"am5 ({am5.get('rule')}): excluded_count={am5.get('excluded_count')} "
          f"included_count={am5.get('included_count')}")
    # Lane L-AM13: the protocol and the union-present eligible-class count, with every artifact.
    level = summary.get("dataset_level") or {}
    print(f"protocol {summary['dataset']['preprocess_protocol']}: all_class_miou="
          f"{level.get('all_class_miou')!r} (union-present, "
          f"n_eligible={level.get('all_class_miou_n_eligible')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
