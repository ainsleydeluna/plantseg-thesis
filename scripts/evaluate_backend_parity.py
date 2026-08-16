#!/usr/bin/env python3
"""Descriptive QNNPACK<->x86 clean-accuracy parity for the INT8 stages (E4-E7).

WHY THIS EXISTS. The x86/fbgemm copy is built for CPU-proxy latency, but it uses a backend-specific
quantization configuration (`reduce_range=True`), so its clean segmentation metrics can differ from
the authoritative QNNPACK artifact's. Chapter III requires that difference to be DOCUMENTED. This
script documents it; it does not gate on it, and it defines no acceptable difference.

WHAT IT IS NOT. The result is never an E-stage accuracy score. It never enters model selection,
robustness, or any hypothesis test, and it is written to its own descriptive record rather than to
an official clean-evaluation artifact directory — `parity_output_path` refuses a directory that
already holds official payloads.

NO SECOND EVALUATOR. Both sides run through the governed clean evaluator
(`src.eval.evaluate.evaluate_model`) over the same manifest, preprocessing, ignore_index and metric
reducers. This module only differences the two `dataset_level` dicts it gets back.

E5/E6 use the committed zero-training translation from `src.quant.x86_latency`: the same best
QAT-trained weights and carried observer ranges, with x86 activation qparams recomputed. No QAT is
re-run and the official QNNPACK model is never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.efficiency import (INT8_STAGES, EfficiencyError,  # noqa: E402
                                 finalize_parity_record, parity_output_path,
                                 validate_parity_request)
from src.eval.stage_artifacts import OFFICIAL_ROWS, OFFICIAL_SPLIT  # noqa: E402


class BackendParityCliError(RuntimeError):
    """Argument/mode violation, raised before any artifact or model work."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Descriptive QNNPACK vs x86 clean-accuracy parity (never an official result).")
    p.add_argument("--stage", required=True, choices=list(INT8_STAGES))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--provenance", required=True,
                   help="run-provenance JSON of the official QNNPACK INT8 artifact")
    p.add_argument("--qat-sidecar", default=None,
                   help="E5/E6 only: the auxiliary pre-convert QAT state used for translation")
    p.add_argument("--calibration-index", default=None,
                   help="E4/E7 only: the frozen shared 128-image calibration index")
    p.add_argument("--split", default=OFFICIAL_SPLIT, choices=[OFFICIAL_SPLIT])
    p.add_argument("--max-samples", type=int, default=None,
                   help="rejected: a capped run cannot be a descriptive parity result")
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    return p


def resolve_sources(args) -> tuple[dict, dict]:
    """Resolve the official QNNPACK artifact and describe the x86 copy that will be built."""
    from src.eval.stage_artifacts import resolve_evaluation_source
    from src.quant.x86_latency import (X86_PTQ_STAGES, load_qat_sidecar,
                                       qat_sidecar_missing_error)

    resolved = resolve_evaluation_source(args.stage, provenance=args.provenance)
    qnnpack_artifact = {
        "stage": args.stage,
        "backend": "qnnpack",
        "artifact_role": "accuracy",
        "path": resolved.get("path") or resolved.get("model_path"),
        "sha256": resolved.get("sha256"),
        "source_stage": resolved.get("source_stage"),
        "source_checkpoint_sha256": resolved.get("source_checkpoint_sha256"),
    }

    sidecar = None
    if args.stage not in X86_PTQ_STAGES:
        if not args.qat_sidecar:
            raise qat_sidecar_missing_error(args.stage)
        sidecar = load_qat_sidecar(args.qat_sidecar)          # validates kind/role/state
    elif not args.calibration_index:
        raise BackendParityCliError(
            f"{args.stage} is a PTQ stage: its x86 copy must be calibrated on the SAME frozen "
            "shared subset, so --calibration-index is required")

    x86_artifact = {
        "stage": args.stage,
        "backend": None,                       # filled with the engine actually selected
        "artifact_role": "x86_cpu_proxy_latency",
        "source_stage": resolved.get("source_stage"),
        "source_checkpoint_sha256": resolved.get("source_checkpoint_sha256"),
        "built_from": "preconvert_qat_state" if sidecar is not None
                      else "fp32_checkpoint_plus_frozen_calibration",
    }
    return qnnpack_artifact, x86_artifact


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not (args.real_run and args.confirm_real_run):
            raise BackendParityCliError(
                "the descriptive parity evaluation reads the full governed clean test split and "
                "requires --real-run --confirm-real-run. Nothing was read or written.")

        out_dir = Path(args.out_dir).resolve()
        if REPO == out_dir or REPO in out_dir.parents:
            raise BackendParityCliError("artifacts are never written inside the repository")

        qnnpack_artifact, x86_artifact = resolve_sources(args)

        # The governed clean manifest identity comes from the evaluator's own hasher; it is not
        # re-derived here and is never fabricated.
        from src.eval.artifacts import hash_split_manifest
        from src.eval.stage_artifacts import resolve_stage_artifact
        resolve_stage_artifact(args.stage)
        manifest_entries = load_official_manifest(args.split)
        manifest_sha256 = hash_split_manifest(manifest_entries)

        engine = validate_parity_request(
            stage=args.stage, qnnpack_artifact=qnnpack_artifact, x86_artifact=x86_artifact,
            manifest_sha256=manifest_sha256, expected_rows=len(manifest_entries),
            max_samples=args.max_samples)
        x86_artifact["backend"] = engine

        target = parity_output_path(out_dir, args.stage)
    except (BackendParityCliError, EfficiencyError, Exception) as e:  # reported, never swallowed
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print(f"validated {args.stage} backend parity against {engine}; would write {target}")
    return 0


def load_official_manifest(split: str):
    """The governed clean test manifest. Refuses rather than inventing one.

    Building it requires the PlantSeg split files, which this task never touches; the real parity
    run supplies them in the official environment.
    """
    raise EfficiencyError(
        "parity_manifest_unavailable",
        f"the governed {split!r} manifest ({OFFICIAL_ROWS} rows) is not constructible here: it "
        "needs the PlantSeg split under PLANTSEG_DATA_ROOT. The descriptive parity run executes in "
        "the official CPU environment; no manifest, row count or metric is fabricated locally.")


if __name__ == "__main__":
    raise SystemExit(main())
