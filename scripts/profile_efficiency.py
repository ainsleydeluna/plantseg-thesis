#!/usr/bin/env python3
"""Descriptive CPU-proxy efficiency profiling for the teacher and E1-E7.

No dataset. Efficiency needs no PlantSeg image: every measurement runs on a synthetic
1x3x512x512 tensor, so this entry point never reads the test split and never touches
`PLANTSEG_DATA_ROOT`.

Reuses the existing resolver (`src.eval.stage_artifacts.resolve_evaluation_source`) and the existing
loaders (`src.eval.model_loading`) — there is no second checkpoint-loading system here. The source
artifact is validated BEFORE any model is constructed.

Everything measured is DESCRIPTIVE and CPU PROXY. No inferential test is reachable from this script
and no ARM/on-device latency is claimed. Measurements that need an environment this host does not
have (fvcore, a real quantization backend, the fbgemm/x86 latency copy) are recorded under
`deferred` with their reason rather than silently approximated.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.efficiency import (ARTIFACT_ROLE_ACCURACY, ARTIFACT_ROLE_X86_LATENCY,  # noqa: E402
                                 ARTIFACT_ROLES, INPUT_SHAPE, STAGES, EfficiencyError,
                                 architecture_parameter_count, efficiency_record,
                                 measure_latency, memory_delta, profile_macs, provenance,
                                 require_quantized_backend, rss_bytes, serialized_artifact,
                                 validate_artifact_role, weight_footprint, x86_latency_copy_gate)
from src.eval.stage_artifacts import resolve_evaluation_source  # noqa: E402

METRICS = ("params", "size", "flops", "latency", "memory")
INT8_KINDS = ("int8_artifact",)


class EfficiencyCliError(RuntimeError):
    """Argument/mode violation, raised before any artifact or model work."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Descriptive CPU-proxy efficiency profiling.")
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--provenance", default=None)
    p.add_argument("--metrics", default=",".join(METRICS),
                   help=f"comma-separated subset of {METRICS}")
    p.add_argument("--artifact-role", default=ARTIFACT_ROLE_ACCURACY, choices=list(ARTIFACT_ROLES))
    p.add_argument("--threads", type=int, default=None,
                   help="fixed torch thread count; all compared stages must use the same value")
    return p


def parse_metrics(raw: str) -> tuple[str, ...]:
    requested = tuple(m.strip() for m in raw.split(",") if m.strip())
    unknown = [m for m in requested if m not in METRICS]
    if unknown:
        raise EfficiencyCliError(f"unknown metrics {unknown}; registered metrics are {list(METRICS)}")
    if not requested:
        raise EfficiencyCliError("no metrics requested")
    return requested


def load_model(resolved: dict, stage: str):
    """Dispatch to the EXISTING loader for this artifact kind. Builds no new loading system."""
    from src.eval.model_loading import (load_int8_student, load_student_checkpoint,
                                        load_teacher_model)

    kind = resolved.get("kind") or resolved.get("artifact_kind")
    if kind == "teacher_checkpoint" or stage == "teacher":
        return load_teacher_model(resolved), "teacher", "fp32"
    if kind in INT8_KINDS:
        return load_int8_student(resolved), "student", "int8"
    return load_student_checkpoint(resolved["path"]), "student", "fp32"


def fp32_reference_for(stage: str):
    """FP32 architecture used for parameter counts and theoretical compute of INT8 stages."""
    from src.eval.model_loading import build_fp32_student
    return build_fp32_student()


def profile(resolved: dict, *, stage: str, metrics, artifact_role: str, threads: int | None) -> dict:
    model, model_role, precision = load_model(resolved, stage)
    backend = resolved.get("backend") if precision == "int8" else None
    if precision == "int8":
        backend = backend or "qnnpack"
        validate_artifact_role(artifact_role, backend)

    source_path = resolved.get("path") or resolved.get("model_path")
    prov = provenance(stage=stage, model_role=model_role, precision=precision,
                      artifact_role=artifact_role, backend=backend, source_path=source_path,
                      source_sha256=resolved.get("sha256"))

    parameters = footprint = artifact = compute = latency = memory = None
    deferred: dict[str, str] = {}

    reference = fp32_reference_for(stage) if precision == "int8" else None

    if "params" in metrics:
        parameters = architecture_parameter_count(model, fp32_reference=reference)
    if "size" in metrics:
        footprint = weight_footprint(model)
        if source_path:
            artifact = serialized_artifact(source_path)
            # Contract line 412 names the QNNPACK copy "the reported accuracy/size artifact", so
            # serialized size stays with it even when latency is timed on the x86 copy. Line 407's
            # "same artifact used for latency" is honoured by labelling which artifact was measured
            # rather than silently switching size onto the latency copy.
            artifact["size_artifact_role"] = ARTIFACT_ROLE_ACCURACY
            artifact["size_measured_from_latency_copy"] = False
    if "flops" in metrics:
        try:
            compute = profile_macs(reference if reference is not None else model,
                                   input_shape=INPUT_SHAPE)
        except EfficiencyError as e:
            deferred["flops"] = f"[{e.code}] {e}"
    if "latency" in metrics:
        try:
            if artifact_role == ARTIFACT_ROLE_X86_LATENCY:
                # selects a real x86/fbgemm engine or refuses; never times the QNNPACK copy
                latency_backend = x86_latency_copy_gate(stage)
                prov["latency_backend"] = latency_backend
            elif precision == "int8":
                require_quantized_backend(backend)
            latency = measure_latency(model, thread_count=threads)
        except EfficiencyError as e:
            deferred["latency"] = f"[{e.code}] {e}"
    if "memory" in metrics:
        try:
            baseline = rss_bytes()
            with_peak = rss_bytes()
            memory = memory_delta(baseline, with_peak)
        except EfficiencyError as e:
            deferred["memory"] = f"[{e.code}] {e}"

    return efficiency_record(provenance_block=prov, parameters=parameters, footprint=footprint,
                             artifact=artifact, compute=compute, latency=latency, memory=memory,
                             deferred=deferred)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = parse_metrics(args.metrics)

        out_dir = Path(args.out_dir).resolve()
        if REPO == out_dir or REPO in out_dir.parents:
            raise EfficiencyCliError("efficiency artifacts are never written inside the repository")

        # ---- source artifact validated BEFORE any model construction ----
        resolved = resolve_evaluation_source(args.stage, checkpoint=args.checkpoint,
                                             provenance=args.provenance)

        record = profile(resolved, stage=args.stage, metrics=metrics,
                         artifact_role=args.artifact_role, threads=args.threads)
    except Exception as e:  # noqa: BLE001 - reported, never swallowed
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"efficiency_{args.stage}_{args.artifact_role}_{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    if record["deferred"]:
        for metric, reason in record["deferred"].items():
            print(f"DEFERRED {metric}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
