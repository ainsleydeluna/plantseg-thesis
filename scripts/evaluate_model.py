#!/usr/bin/env python3
"""Stage-neutral evaluation CLI (A2b).

Drives the proven A2a flow in the frozen order:

    build metadata + class map
      -> prepare_artifact_request
      -> validate_artifact_request        <-- NOTHING is constructed before this succeeds
      -> construct dataset adapter + model
      -> evaluate_model
      -> write_artifact

The CLI and its metadata are STAGE-NEUTRAL (any stage / role / precision / condition can be
*described*), but A2b implements only the FP32-student, clean-condition construction path.
Teacher, PTQ, QAT and corruption construction are rejected explicitly -- metadata neutrality is
never misrepresented as runtime support.

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

SUPPORTED_ROLES = ("student",)
SUPPORTED_PRECISIONS = ("fp32",)
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
            f"model_role={args.model_role!r} is not implemented in A2b. Supported: "
            f"{SUPPORTED_ROLES}. Teacher construction does not exist yet.")
    if args.precision not in SUPPORTED_PRECISIONS:
        raise CliError(
            f"precision={args.precision!r} is not implemented in A2b. Supported: "
            f"{SUPPORTED_PRECISIONS}. INT8 PTQ/QAT construction does not exist yet.")
    if args.condition not in SUPPORTED_CONDITIONS:
        raise CliError(
            f"condition={args.condition!r} is not implemented in A2b. Supported: "
            f"{SUPPORTED_CONDITIONS}. Corruption generation does not exist yet.")

    # --- mode / checkpoint consistency ---
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
        if not args.checkpoint:
            raise CliError("--split test requires --checkpoint")
        if args.artifact_status == "smoke":
            raise CliError("--split test refuses artifact_status=smoke")


def run(args, *, counters: Counters | None = None) -> Path:
    """Execute the frozen A2a flow. Returns the finalised artifact directory."""
    from src.eval import Condition, DatasetMeta, RunMeta, evaluate_model
    from src.eval.adapters import (DATASET_DOI, DATASET_NAME, PREPROCESS_PROTOCOL,
                                   PlantSegEvalDataset, build_eval_loader,
                                   build_expected_manifest_for, deterministic_subset,
                                   load_class_map)
    from src.eval.artifacts import (prepare_artifact_request, validate_artifact_request,
                                    write_artifact)
    from src.eval.model_loading import build_fp32_student, load_student_checkpoint

    validate_cli_args(args)

    # ---- static metadata + AUTHORITATIVE class map (verified against the pinned hash) ----
    class_map = load_class_map()

    n_total = EXPECTED_SPLIT_ROWS[args.split]
    n_rows = args.max_samples if args.max_samples is not None else n_total
    # Cap is resolved to explicit SOURCE INDICES here -- before any loader exists.
    source_indices = (deterministic_subset(n_total, n_rows)
                      if args.max_samples is not None else list(range(n_total)))

    ckpt_path = ckpt_sha = None
    if args.checkpoint:
        from src.eval.model_loading import sha256_file
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
                    quant_backend=None, checkpoint_path=ckpt_path, checkpoint_sha256=ckpt_sha,
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

    # ---- step 5: only now may the dataset adapter and the model be constructed ----
    if counters is not None:
        counters.dataset.append(("adapter", tuple(source_indices)))
    adapter = PlantSegEvalDataset(args.split, source_indices)

    if counters is not None:
        counters.model.append(("model", args.random_init))
    model = (build_fp32_student(device=args.device) if args.random_init
             else load_student_checkpoint(args.checkpoint, map_location=args.device)[0])

    loader = build_eval_loader(adapter, args.batch_size, num_workers=0)
    result = evaluate_model(model, loader, expected_manifest=expected_manifest,
                            condition=request.dataset.condition, num_classes=116,
                            background_index=0, ignore_index=255)
    return write_artifact(result, request, provenance)


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
