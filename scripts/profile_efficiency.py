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

L-RESIDUAL-SHA's efficiency half (LEFF B-1: E-35, DL-59; the LEFF Phase A plan §2 as the LEFF-B1 GO amends it):
  * --teacher-ckpt-sha256 is required for --stage teacher and refused on any other stage, with
    scripts/evaluate_model.py's codes; --teacher-config is required for the teacher. The checkpoint is hashed and
    compared before it is parsed, and again before the teacher is built.
  * The loaders dispatch on the resolver's spec kind and are unpacked to their modules (F1). FP32: the checkpoint's
    sha256 is compared again at the load ([checkpoint_changed_after_validation]); INT8: the converted artifact is
    re-hashed immediately before it is loaded ([converted_changed_before_load]).
  * No path argument names "test", as given or resolved, before anything is opened ([path_forbidden_name]); nor
    do the converted artifact and the state_dict companion a run provenance names, refused before the profiler
    reads or loads them (the resolver's whole-file hash is the one read SL-1 allows); --out-dir is outside the
    repository and absent or empty ([out_dir_not_empty]); the record is created exclusively.
  * Size: the file of record's bytes are re-hashed against the resolver's sha256 before any model is built; an FP32
    stage also records its model-only serialization, a labelled companion (LEFF-AUDIT-RULE-1 Q7), measured after
    the memory phase. The INT8 weight footprint is read from the artifact's own packed params and cross-checked
    against its state_dict companion ([state_dict_companion_mismatch]; A13).
  * Peak memory (§3): VmHWM over the protocol's 120 untimed forwards minus VmRSS at the phase start, the high-water
    mark reset through /proc/self/clear_refs after the load and gc.collect(), at --threads when it is given; in a
    combined invocation it runs first, right after the load, before any other measurement's allocation. The
    process-start values are recorded for §3's parent test. The record is plantseg-efficiency/2.0.0.
Every refusal exits 2 and prints `[code] message`, except an error a loader raises while it reads a validated file
(model_loading.CheckpointError carries no code), printed as `CheckpointError: message`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _process_start_memory() -> dict | None:
    """ru_maxrss and /proc/self/status at process start, read before torch is imported (§3's parent test)."""
    try:
        import resource
        values = {}
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("VmRSS", "VmHWM") and rest.split():
                    values[key] = int(rest.split()[0]) * 1024
        return {"ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "vm_hwm_bytes": values["VmHWM"], "vm_rss_bytes": values["VmRSS"]}
    except (ImportError, OSError, KeyError, ValueError):
        return None


PROCESS_START_MEMORY = _process_start_memory()

from src.eval.efficiency import (ARTIFACT_ROLE_ACCURACY, ARTIFACT_ROLE_X86_LATENCY,  # noqa: E402
                                 ARTIFACT_ROLES, FOOTPRINT_SOURCE_ARTIFACT, FOOTPRINT_SOURCE_COMPANION,
                                 FOOTPRINT_SOURCE_MODEL, INPUT_SHAPE, STAGES, EfficiencyError,
                                 architecture_parameter_count, efficiency_record, footprint_counts_equal,
                                 measure_latency, measure_peak_memory, model_only_serialization,
                                 packed_params_footprint, parent_contamination, profile_macs, provenance,
                                 require_quantized_backend, serialized_artifact, validate_artifact_role,
                                 weight_footprint, x86_latency_copy_gate)
from src.eval.stage_artifacts import resolve_evaluation_source  # noqa: E402

METRICS = ("params", "size", "flops", "latency", "memory")
# One group per process of record (§2): memory always alone, in a fresh process (§3)
METRIC_GROUPS = {"static": ("params", "size", "flops"), "latency": ("latency",), "memory": ("memory",)}
FORBIDDEN_NAME = "test"           # SL-1: a path whose given or resolved form names it is refused before it is opened
PATH_FLAGS = ("out_dir", "checkpoint", "provenance", "teacher_config")


class EfficiencyCliError(RuntimeError):
    """Argument/mode violation, raised before any artifact or model work. `code` names the refusal."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


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
    p.add_argument("--teacher-ckpt-sha256", default=None,
                   help="--stage teacher only (DL-59): the checkpoint's SHA-256, 64 lowercase hex characters, "
                        "compared before the file is parsed and again before the teacher is built")
    p.add_argument("--teacher-config", default=None,
                   help="--stage teacher only: the thesis teacher config (its IsolatedNMFLightHamHead implements "
                        "M4-V)")
    return p


def parse_metrics(raw: str) -> tuple[str, ...]:
    requested = tuple(m.strip() for m in raw.split(",") if m.strip())
    unknown = [m for m in requested if m not in METRICS]
    if unknown:
        raise EfficiencyCliError(f"unknown metrics {unknown}; registered metrics are {list(METRICS)}",
                                 code="metrics_unknown")
    if not requested:
        raise EfficiencyCliError("no metrics requested", code="metrics_none")
    return requested


def metric_group_of(metrics) -> str | None:
    """The metric group the requested metrics form exactly, or None for any other combination."""
    return next((g for g, members in METRIC_GROUPS.items() if set(members) == set(metrics)), None)


# ------------------------------------------------------------------ checks before anything is opened
def validate_teacher_identity(args) -> None:
    """DL-59 for the profiler (E-35): scripts/evaluate_model.py's codes and messages (:213-224), and the teacher's
    config as the evaluator requires it."""
    teacher_sha = getattr(args, "teacher_ckpt_sha256", None)
    if args.stage == "teacher":
        if teacher_sha is None:
            raise EfficiencyCliError("the teacher stage requires --teacher-ckpt-sha256, the checkpoint's SHA-256, "
                                     "compared before the file is parsed", code="teacher_ckpt_sha256_required")
        from src.distill.teacher import sha256_format_error
        sha_error = sha256_format_error(teacher_sha)
        if sha_error is not None:
            raise EfficiencyCliError(f"--teacher-ckpt-sha256 {sha_error}", code="teacher_ckpt_sha256_format")
        if not getattr(args, "teacher_config", None):
            raise EfficiencyCliError("the teacher stage requires --teacher-config (the thesis teacher config)",
                                     code="teacher_config_required")
    elif teacher_sha is not None:
        raise EfficiencyCliError(f"--teacher-ckpt-sha256 applies to the teacher stage only; stage={args.stage!r} "
                                 "takes none", code="teacher_ckpt_sha256_not_teacher")


def forbidden_path_error(label: str, value) -> str | None:
    """None when neither the given nor the resolved form of a path names "test" (SL-1); else the refusal text."""
    for form in (str(value), os.path.realpath(value)):
        if FORBIDDEN_NAME in form.lower():
            return (f"{label} {form!r} names 'test' (SL-1); it was not opened, and nothing under TEST is named, "
                    "opened or listed")
    return None


def refuse_forbidden_paths(args) -> None:
    for flag in PATH_FLAGS:
        value = getattr(args, flag, None)
        if value:
            error = forbidden_path_error(f"--{flag.replace('_', '-')}", value)
            if error:
                raise EfficiencyCliError(error, code="path_forbidden_name")


def refuse_forbidden_converted(resolved: dict) -> None:
    """SL-1 for the converted artifact a run provenance names: the resolver hashed it whole (the one read SL-1
    allows); a path naming "test", as recorded or resolved, is refused before the profiler reads or loads it."""
    if resolved["spec"]["kind"] == "int8_artifact" and forbidden_path_error("", resolved["artifact_path"]):
        raise EfficiencyError("path_forbidden_name",
                              f"the converted artifact {str(resolved['artifact_path'])!r} its run provenance names "
                              "names 'test' (SL-1): the resolver hashed it whole, the one read SL-1 allows, and the "
                              "profiler neither reads nor loads it")


def check_out_dir(raw) -> Path:
    out_dir = Path(raw).resolve()
    if REPO == out_dir or REPO in out_dir.parents:
        raise EfficiencyCliError("efficiency artifacts are never written inside the repository",
                                 code="out_dir_inside_repository")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise EfficiencyCliError(f"--out-dir {out_dir} exists and is not an empty directory; a profile is written "
                                 "into a fresh one", code="out_dir_not_empty")
    return out_dir


# ------------------------------------------------------------------ the source and its loaders
def source_identity(resolved: dict) -> tuple[Path, str]:
    """The file the resolver validated, by its own keys: artifact_path and artifact_sha256 for an INT8 stage,
    checkpoint_path and checkpoint_sha256 for an FP32 stage and the teacher."""
    if resolved["spec"]["kind"] == "int8_artifact":
        return Path(resolved["artifact_path"]), resolved["artifact_sha256"]
    return Path(resolved["checkpoint_path"]), resolved["checkpoint_sha256"]


def changed_error(resolved: dict, path, actual: str, when: str) -> EfficiencyError:
    """The file of record changed after the resolver hashed it: INT8 in B-1b's words, FP32 in the evaluator's."""
    expected = source_identity(resolved)[1]
    if resolved["spec"]["kind"] == "int8_artifact":
        return EfficiencyError("converted_changed_before_load",
                               f"the converted artifact {path} had sha256 {expected} at validation and {actual} "
                               "before the load; it was not loaded")
    return EfficiencyError("checkpoint_changed_after_validation",
                           f"checkpoint bytes changed between validation ({expected}) and {when} ({actual})")


def load_model(resolved: dict, *, stage: str, teacher_config: str | None = None,
               teacher_sha256: str | None = None, teacher_builder=None):
    """Dispatch on the resolver's spec kind to the EXISTING loader; return its module, never its tuple (F1)."""
    from src.eval.model_loading import (load_int8_student, load_student_checkpoint, load_teacher_model,
                                        sha256_file)

    kind = resolved["spec"]["kind"]
    if kind == "teacher_checkpoint":
        # hashed again and compared before any parse or build (R6); the real builder needs the thesis config
        model, _resolved = load_teacher_model(resolved, builder=teacher_builder, config_path=teacher_config,
                                              expected_sha256=teacher_sha256)
        return model
    if kind == "int8_artifact":
        path = resolved["artifact_path"]
        actual = sha256_file(path)                  # P3: load_int8_student does not re-hash
        if actual != resolved["artifact_sha256"]:
            raise changed_error(resolved, path, actual, "the load")
        return load_int8_student(resolved)[0]
    model, info = load_student_checkpoint(resolved["checkpoint_path"])
    if info.sha256 != resolved["checkpoint_sha256"]:
        raise changed_error(resolved, resolved["checkpoint_path"], info.sha256, "load")
    return model


def fp32_reference_for(stage: str):
    """FP32 architecture used for parameter counts and theoretical compute of INT8 stages."""
    from src.eval.model_loading import build_fp32_student
    return build_fp32_student()


# ------------------------------------------------------------------ size and footprint
def measure_artifact_size(resolved: dict) -> dict:
    """The file of record's serialized size, its bytes re-hashed against the resolver's before any model is built."""
    source_path, source_sha256 = source_identity(resolved)
    artifact = serialized_artifact(source_path)
    if artifact["sha256"] != source_sha256:
        raise changed_error(resolved, source_path, artifact["sha256"], "the size read")
    # Contract line 412 names the QNNPACK copy "the reported accuracy/size artifact", so
    # serialized size stays with it even when latency is timed on the x86 copy. Line 407's
    # "same artifact used for latency" is honoured by labelling which artifact was measured
    # rather than silently switching size onto the latency copy.
    artifact["size_artifact_role"] = ARTIFACT_ROLE_ACCURACY
    artifact["size_measured_from_latency_copy"] = False
    return artifact


def model_only_companion(resolved: dict) -> dict | None:
    """An FP32 stage's model-only serialization (Q7), a labelled companion to its serialized size. It parses the
    whole checkpoint, so profile() measures it after the memory phase (§3); an INT8 stage has none."""
    kind = resolved["spec"]["kind"]
    if kind == "int8_artifact":
        return None
    source_path, source_sha256 = source_identity(resolved)
    return model_only_serialization(source_path, expected_sha256=source_sha256, teacher=kind == "teacher_checkpoint")


def companion_path(resolved: dict, provenance_path) -> tuple[Path, str]:
    """The state_dict companion the run provenance records, resolved against the provenance's directory (as
    scripts/ptq_val_checks.py resolves it) and re-hashed."""
    from src.eval.model_loading import sha256_file

    rec = resolved["provenance"].get("state_dict_artifact") or {}
    if not rec.get("path") or not rec.get("sha256"):
        raise EfficiencyError("state_dict_companion_mismatch",
                              "the run provenance records no state_dict_artifact path and sha256; the INT8 "
                              "footprint is cross-checked against it")
    path = Path(rec["path"])
    if not path.is_absolute():
        base = (Path(provenance_path).resolve().parent if provenance_path
                else Path(resolved["artifact_path"]).resolve().parent)
        path = base / path
    error = forbidden_path_error("the state_dict companion", path)
    if error:
        raise EfficiencyError("path_forbidden_name", error)
    if not path.is_file() or sha256_file(path) != rec["sha256"]:
        raise EfficiencyError("state_dict_companion_mismatch",
                              f"{path} is missing or does not match the sha256 its run provenance records "
                              f"({rec['sha256']})")
    return path, rec["sha256"]


def companion_footprint(resolved: dict, provenance_path) -> dict:
    """The INT8 footprint counted on the state_dict companion, rebuilt eager (the cross-check, A13)."""
    from src.eval.model_loading import rebuild_int8_from_state_dict

    path, sha256 = companion_path(resolved, provenance_path)
    require_quantized_backend("qnnpack")             # the companion is the QNNPACK artifact's converted state
    eager = rebuild_int8_from_state_dict(path, stage=resolved["spec"]["stage"], method=resolved["spec"]["method"])
    counts = weight_footprint(eager)
    counts.update({"counted_from": "state_dict_companion_eager_rebuild", "path": str(path), "sha256": sha256})
    return counts


def _counts(footprint: dict) -> list:
    return [footprint[k] for k in ("int8_weight_elements", "fp32_weight_elements", "metadata_elements")]


def measure_footprint(model, resolved: dict, provenance_path) -> dict:
    """The parameter-byte footprint per the which-artifact rule, naming the count it reports (`footprint_source`)."""
    if resolved["spec"]["kind"] != "int8_artifact":
        return {**weight_footprint(model), "footprint_source": FOOTPRINT_SOURCE_MODEL}
    if resolved["artifact_format"] != "torchscript":
        # src/quant/runner.py's state_dict artifact: the eager rebuild is the artifact itself
        return {**weight_footprint(model), "footprint_source": FOOTPRINT_SOURCE_ARTIFACT}
    from_artifact = packed_params_footprint(model)
    from_companion = companion_footprint(resolved, provenance_path)
    if from_artifact["packed_params_unreadable"]:
        # LF26: an unreadable packed-params object is named, and the companion's count is reported
        reported, source = from_companion, FOOTPRINT_SOURCE_COMPANION
    elif not footprint_counts_equal(from_artifact, from_companion):
        raise EfficiencyError("state_dict_companion_mismatch",
                              "the artifact's packed params and its state_dict companion count different weights "
                              f"(INT8, FP32, metadata elements: artifact {_counts(from_artifact)}, companion "
                              f"{_counts(from_companion)})")
    else:
        reported, source = from_artifact, FOOTPRINT_SOURCE_ARTIFACT
    return {**reported, "footprint_source": source, "artifact_count": from_artifact,
            "companion_count": from_companion}


# ------------------------------------------------------------------ peak memory (§3)
def measure_memory(model) -> dict:
    """Peak inference memory (efficiency.measure_peak_memory) with this process's start values and §3's
    parent-contamination test on them (recorded here; the run of record refuses a contaminated process, B-2)."""
    out = measure_peak_memory(model)
    start = PROCESS_START_MEMORY
    out["process_start"] = start
    out["parent_contamination"] = (parent_contamination(start["ru_maxrss_bytes"], start["vm_hwm_bytes"])
                                   if start else None)
    return out


# ------------------------------------------------------------------ the profile
def profile(resolved: dict, *, stage: str, metrics, artifact_role: str, threads: int | None,
            provenance_path=None, teacher_config: str | None = None, teacher_sha256: str | None = None,
            teacher_builder=None) -> dict:
    wall_start = datetime.now(timezone.utc).timestamp()
    kind = resolved["spec"]["kind"]
    precision = "int8" if kind == "int8_artifact" else "fp32"
    model_role = "teacher" if kind == "teacher_checkpoint" else "student"
    backend = None
    if precision == "int8":
        backend = resolved["provenance"].get("backend") or "qnnpack"
        validate_artifact_role(artifact_role, backend)       # before anything is loaded

    source_path, source_sha256 = source_identity(resolved)
    # ---- the file of record: its bytes re-hashed against the resolver's, before any model is built ----
    artifact = measure_artifact_size(resolved) if "size" in metrics else None

    model = load_model(resolved, stage=stage, teacher_config=teacher_config, teacher_sha256=teacher_sha256,
                       teacher_builder=teacher_builder)
    prov = provenance(stage=stage, model_role=model_role, precision=precision,
                      artifact_role=artifact_role, backend=backend, source_path=source_path,
                      source_sha256=source_sha256, expected_sha256=teacher_sha256)

    parameters = footprint = compute = latency = memory = None
    deferred: dict[str, str] = {}

    if "memory" in metrics:
        # first, right after the load (§3): no other measurement's allocations precede the phase
        try:
            memory = measure_memory(model)
        except EfficiencyError as e:
            deferred["memory"] = f"[{e.code}] {e}"
    if artifact is not None and kind != "int8_artifact":
        artifact["model_only_companion"] = model_only_companion(resolved)     # parses the checkpoint: after memory

    reference = fp32_reference_for(stage) if precision == "int8" else None

    if "params" in metrics:
        parameters = architecture_parameter_count(model, fp32_reference=reference)
    if "size" in metrics:
        footprint = measure_footprint(model, resolved, provenance_path)
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

    return efficiency_record(provenance_block=prov, parameters=parameters, footprint=footprint,
                             artifact=artifact, compute=compute, latency=latency, memory=memory,
                             deferred=deferred, of_record=False, metric_group=metric_group_of(metrics),
                             wall_clock=(wall_start, datetime.now(timezone.utc).timestamp()))


def write_record(out_dir: Path, record: dict, *, stage: str, artifact_role: str) -> Path:
    """Create the record exclusively: an existing file is never replaced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    group = record.get("metric_group") or "mixed"
    out_path = out_dir / f"efficiency_{stage}_{artifact_role}_{group}_{stamp}.json"
    with open(out_path, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(record, indent=2, sort_keys=True))
    return out_path


def run(args, *, teacher_builder=None) -> tuple[Path, dict]:
    """Every check before anything is opened, then the resolver, the profile and the record. Raises on a refusal.

    `teacher_builder` is the injectable teacher factory of src.eval.model_loading.load_teacher_model, for synthetic
    smokes only; the command line never sets it.
    """
    metrics = parse_metrics(args.metrics)
    validate_teacher_identity(args)
    refuse_forbidden_paths(args)
    out_dir = check_out_dir(args.out_dir)
    if args.threads is not None:
        import torch
        torch.set_num_threads(int(args.threads))     # before any model: every measurement at it, memory included (§3)

    # ---- source artifact validated BEFORE any model construction; the teacher hashed before any parse ----
    resolved = resolve_evaluation_source(args.stage, checkpoint=args.checkpoint,
                                         provenance=args.provenance,
                                         expected_sha256=args.teacher_ckpt_sha256)
    refuse_forbidden_converted(resolved)

    record = profile(resolved, stage=args.stage, metrics=metrics, artifact_role=args.artifact_role,
                     threads=args.threads, provenance_path=args.provenance,
                     teacher_config=args.teacher_config, teacher_sha256=args.teacher_ckpt_sha256,
                     teacher_builder=teacher_builder)
    return write_record(out_dir, record, stage=args.stage, artifact_role=args.artifact_role), record


def refusal_text(e: BaseException) -> str:
    """`[code] message` for a coded refusal (a message that already leads with its code is kept as it is)."""
    code, message = getattr(e, "code", None), str(e)
    if not code or message.startswith(f"[{code}]"):
        return message
    return f"[{code}] {message}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out_path, record = run(args)
    except Exception as e:  # noqa: BLE001 - reported, never swallowed
        print(f"{type(e).__name__}: {refusal_text(e)}", file=sys.stderr)
        return 2

    print(f"wrote {out_path}")
    if record["deferred"]:
        for metric, reason in record["deferred"].items():
            print(f"DEFERRED {metric}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
