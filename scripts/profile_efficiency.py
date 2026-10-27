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

The run of record (LEFF B-2, L-LAT-THREADS: the plan §1-§2 as the GO's LF30-LF33 amend them), --of-record:
  * one stage of the efficiency analysis, one artifact role and one metric group per process;
  * every identity flag required and compared, with B-1b's codes and texts: --expected-checkpoint-sha256 (the handed
    file, hashed before its first parse and again after validation), --expected-converted-sha256 (INT8) and
    --expected-x86-sha256 (the x86 latency copy); an INT8 artifact passes stage_artifacts.int8_official_error (R2);
  * the governed paths of the running checkout clean ([governed_paths_dirty], [governed_paths_unavailable]); the KD
    image of record by digest, its PL-26 fingerprint (scripts/preflight_distill.py, imported here only) and the
    pinned fvcore;
  * the thread count of record (LEFF-AUDIT-RULE-1 Q1): inter-op set first in main(), intra-op set before any model
    and read back, the Timer at that count with its probe in setup ([timed_threads_mismatch]); the host's facts
    recorded and gated, with --host-label;
  * nothing deferred.
On every run, an INT8 stage's latency and peak memory are measured on its x86 latency copy (the which-artifact rule),
bound through its run record (E4/E7) or its eval directory's qat_finalize.json (E5/E6) and loaded with its embedded
identity; its parameters, size and FLOPs come from its QNNPACK artifact of record.
Every refusal exits 2 and prints `[code] message`, except an error a loader raises while it reads a validated file
(model_loading.CheckpointError carries no code), printed as `CheckpointError: message`.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
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

import torch  # noqa: E402

from src.eval.efficiency import (ARTIFACT_ROLE_ACCURACY, ARTIFACT_ROLE_X86_LATENCY,  # noqa: E402
                                 ARTIFACT_ROLES, EFFICIENCY_ANALYSIS_STAGES, FOOTPRINT_SOURCE_ARTIFACT,
                                 FOOTPRINT_SOURCE_COMPANION, FOOTPRINT_SOURCE_MODEL, FVCORE_PINNED, INPUT_SHAPE,
                                 INTEROP_THREADS_OF_RECORD, STAGES, THREADS_OF_RECORD, EfficiencyError,
                                 architecture_parameter_count, efficiency_record, footprint_counts_equal,
                                 host_facts, host_thread_error, measure_latency, measure_peak_memory,
                                 model_only_serialization, packed_params_footprint, parent_contamination,
                                 profile_macs, provenance, require_quantized_backend, serialized_artifact,
                                 set_interop_threads, validate_artifact_role, weight_footprint,
                                 x86_latency_copy_gate)
from src.eval.model_loading import sha256_file  # noqa: E402
from src.eval.stage_artifacts import resolve_evaluation_source, resolve_stage_artifact  # noqa: E402

METRICS = ("params", "size", "flops", "latency", "memory")
# One group per process of record (§2): memory always alone, in a fresh process (§3)
METRIC_GROUPS = {"static": ("params", "size", "flops"), "latency": ("latency",), "memory": ("memory",)}
FORBIDDEN_NAME = "test"           # SL-1: a path whose given or resolved form names it is refused before it is opened
PATH_FLAGS = ("out_dir", "checkpoint", "provenance", "teacher_config", "qat_finalize")
X86_PTQ_STAGES = ("E4", "E7")     # = src.quant.x86_latency.X86_PTQ_STAGES (restated: this import stays light)
X86_QAT_STAGES = ("E5", "E6")     # = src.quant.x86_latency.X86_QAT_STAGES
QAT_FINALIZE_FORMAT = "qat_finalize/1"     # = scripts/qat_epoch_eval.py FINALIZE_FORMAT
THREAD_RULING = "LEFF-AUDIT-RULE-1 Q1; its DL row is proposed in docs/lane_reports/leff.md"
GOVERNED_ROOT = REPO              # the running checkout, whose governed paths the run of record requires clean (A7)
CODE_FILES = ("scripts/profile_efficiency.py", "src/eval/efficiency.py")


class EfficiencyCliError(RuntimeError):
    """Argument/mode violation, raised before any artifact or model work. `code` names the refusal."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


def refuse(code: str, message: str):
    raise EfficiencyCliError(message, code=code)


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
    p.add_argument("--of-record", action="store_true",
                   help="the run of record: one stage, role and metric group per process; every identity flag "
                        "required; the thread count, host, image and governed paths enforced")
    p.add_argument("--expected-checkpoint-sha256", default=None,
                   help="the sha256 of the file this run is handed (--checkpoint, or --provenance for an INT8 "
                        "stage), compared before its first parse")
    p.add_argument("--expected-converted-sha256", default=None,
                   help="INT8: the sha256 of the converted artifact its provenance names (E5/E6: the finalize "
                        "record's artifact_of_record TorchScript)")
    p.add_argument("--expected-x86-sha256", default=None,
                   help="--artifact-role x86_cpu_proxy_latency: the sha256 of the x86 latency copy its run record "
                        "names")
    p.add_argument("--qat-finalize", default=None,
                   help="E5/E6 with --artifact-role x86_cpu_proxy_latency: the eval directory's qat_finalize.json")
    p.add_argument("--host-label", default=None, help="the run of record's host label (the pod, or the laptop)")
    p.add_argument("--host-note", default=None,
                   help="optional: the pod id, CPU type name, region and price; or the laptop's Windows build and "
                        "power mode")
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


# ------------------------------------------------------------------ §2 step 1: the command line
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


def validate_cli(args, metrics) -> None:
    """§2 step 1, in the plan's order; nothing is opened. Of record, one metric group; the role each precision and
    metric group takes (the which-artifact rule); the teacher's identity, and the handed file's flag against it
    (D14); the finalize record's stages; the identity flags' form (B-1b's codes and texts, LF32); --threads."""
    from src.distill.teacher import sha256_format_error

    if args.of_record and metric_group_of(metrics) is None:
        refuse("metric_group_not_of_record",
               f"the run of record measures exactly one metric group per process ({METRIC_GROUPS}); got "
               f"{list(metrics)}")
    spec = resolve_stage_artifact(args.stage)
    int8 = spec["kind"] == "int8_artifact"
    x86 = args.artifact_role == ARTIFACT_ROLE_X86_LATENCY
    if x86 and not int8:
        refuse("fp32_has_no_x86_copy",
               f"{args.stage} is an FP32 stage: one artifact serves its size and latency (the which-artifact rule), "
               "and it has no x86 latency copy")
    if int8 and not x86 and set(metrics) & {"latency", "memory"}:
        refuse("int8_latency_needs_x86_copy",
               f"{args.stage}'s latency and peak memory are measured on its x86 latency copy (the which-artifact "
               f"rule; DL-b): pass --artifact-role {ARTIFACT_ROLE_X86_LATENCY}")
    if int8 and x86 and set(metrics) & {"params", "size", "flops"}:
        refuse("int8_static_needs_accuracy_artifact",
               f"{args.stage}'s parameters, size and FLOPs come from its QNNPACK artifact of record (the "
               f"which-artifact rule): pass --artifact-role {ARTIFACT_ROLE_ACCURACY}")
    validate_teacher_identity(args)
    expected = args.expected_checkpoint_sha256
    if expected is not None:
        sha_error = sha256_format_error(expected)
        if sha_error is not None:
            refuse("expected_checkpoint_sha256_format", f"--expected-checkpoint-sha256 {sha_error}")
        if args.stage == "teacher" and expected != args.teacher_ckpt_sha256:
            refuse("expected_checkpoint_sha256_teacher_disagrees",
                   f"--expected-checkpoint-sha256 {expected} and --teacher-ckpt-sha256 {args.teacher_ckpt_sha256} "
                   "differ; for the teacher both name --checkpoint and must be equal")
    if args.qat_finalize is not None and args.stage not in X86_QAT_STAGES:
        refuse("qat_finalize_not_qat",
               f"--qat-finalize binds the x86 latency copy of E5 or E6; stage={args.stage!r} takes none")
    if x86 and args.stage in X86_QAT_STAGES and args.qat_finalize is None:
        refuse("qat_finalize_required",
               f"{args.stage}'s x86 latency copy is bound through its eval directory's qat_finalize.json: pass "
               "--qat-finalize")
    converted = args.expected_converted_sha256
    if converted is not None and not int8:
        refuse("expected_converted_sha256_not_int8",
               "--expected-converted-sha256 applies to an INT8 stage handed --provenance; this run "
               f"(precision={spec['precision']!r}, random_init=False) takes none")
    if converted is not None:
        sha_error = sha256_format_error(converted)
        if sha_error is not None:
            refuse("expected_converted_sha256_format", f"--expected-converted-sha256 {sha_error}")
    x86_sha = args.expected_x86_sha256
    if x86_sha is not None and not x86:
        refuse("expected_x86_sha256_not_applicable",
               f"--expected-x86-sha256 applies to --artifact-role {ARTIFACT_ROLE_X86_LATENCY}; this run "
               f"(stage={args.stage!r}, artifact_role={args.artifact_role!r}) takes none")
    if x86_sha is not None:
        sha_error = sha256_format_error(x86_sha)
        if sha_error is not None:
            refuse("expected_x86_sha256_format", f"--expected-x86-sha256 {sha_error}")
    if args.threads is not None and args.threads < 1:
        refuse("threads_invalid", f"--threads must be a positive integer, got {args.threads!r}")


# ------------------------------------------------------------------ §2 step 2: paths
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


# ------------------------------------------------------------------ §2 step 3: the run of record
def governed_gate(root) -> dict:
    """The §7.1 governed-path gate on the running checkout (stage_artifacts.governed_paths_error), then the porcelain
    the record keeps, which must show the same clean governed paths. A checkout without a usable git worktree
    refuses, never raises (A7)."""
    from src.eval.artifacts import GOVERNED_PREFIXES, git_porcelain_bytes, parse_porcelain_paths
    from src.eval.stage_artifacts import governed_paths_error

    root = Path(root)
    try:
        error = governed_paths_error(root)
        porcelain = git_porcelain_bytes(root)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True,
                              timeout=60, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        if head.returncode != 0:
            raise RuntimeError(f"git rev-parse HEAD failed: {head.stderr.strip()[:200]}")
    except Exception as e:  # noqa: BLE001 - no usable worktree is a refusal (A7)
        refuse("governed_paths_unavailable",
               f"the running checkout {root} has no usable git worktree ({type(e).__name__}: "
               f"{str(e).splitlines()[0][:200] if str(e) else ''}); the run of record proves its governed paths "
               "clean from a clone with its .git (A7)")
    if error:
        refuse("governed_paths_dirty", error)
    paths = parse_porcelain_paths(porcelain)
    dirty = sorted(p for p in paths if p.startswith(tuple(GOVERNED_PREFIXES)))
    if dirty:
        refuse("governed_paths_dirty", f"governed paths became dirty or untracked during the gate -> {dirty}")
    allowed = sorted(p for p in paths if not p.startswith(tuple(GOVERNED_PREFIXES)))
    return {"repo_commit": head.stdout.strip(), "governed_paths_clean": True,
            "worktree_state_sha256": hashlib.sha256(porcelain).hexdigest(), "dirty_allowlisted": allowed,
            "files_sha256": {rel: sha256_file(REPO / rel) for rel in CODE_FILES}}


def image_fingerprint() -> tuple:
    """PL-26 in a child process, through scripts/preflight_distill.py's own `_fingerprint` (imported here only)."""
    import scripts.preflight_distill as PD
    return PD._child_json([Path(PD.__file__).resolve(), "_fingerprint"], "FINGERPRINT_JSON", 600)


def fvcore_version() -> str | None:
    import importlib.metadata as md
    try:
        return md.version("fvcore")
    except md.PackageNotFoundError:
        return None


def image_gate() -> dict:
    """The KD image of record (DL-87) by its digest, its PL-26 fingerprint and the pinned fvcore. A fingerprint child
    that fails, runs past its timeout or prints a line that does not parse is refused [image_fingerprint_mismatch]
    with its cause, as a child that exits non-zero is."""
    import scripts.preflight_distill as PD

    digest = os.environ.get("PLANTSEG_IMAGE_DIGEST", "").strip()
    if digest != PD.KD_IMAGE_DIGEST:
        refuse("image_not_of_record",
               f"PLANTSEG_IMAGE_DIGEST {digest or '(unset)'} is not the KD image of record {PD.KD_IMAGE_DIGEST} "
               "(DL-87)")
    try:
        rc, fp, out = image_fingerprint()
    except (subprocess.SubprocessError, OSError, ValueError) as e:      # TimeoutExpired; a FINGERPRINT_JSON line
        rc, fp, out = None, None, f"{type(e).__name__}: {e}"             # that does not parse (JSONDecodeError)
    problems = (PD.fingerprint_problems(fp) if rc == 0 and isinstance(fp, dict)
                else [f"the fingerprint child failed: rc={rc} {str(out)[-200:]}"])
    if problems:
        refuse("image_fingerprint_mismatch", f"the image differs from the fingerprint of record (PL-26): {problems[:6]}")
    version = fvcore_version()
    if version != FVCORE_PINNED:
        refuse("fvcore_not_pinned", f"fvcore {version or '(absent)'} is not the pinned {FVCORE_PINNED}")
    return {"digest": digest, "fingerprint": fp, "fvcore": version}


def of_record_gates(args, metrics, interop: dict) -> dict:
    """§2 step 3 in the plan's order, before the source is opened: the stage; the identity flags and the host label
    required; the governed paths; the image; §1's thread and host checks; then §3's parent test for the memory
    group. Returns the facts the record keeps."""
    stage = args.stage
    if stage not in EFFICIENCY_ANALYSIS_STAGES:
        refuse("stage_not_in_efficiency_analysis",
               f"{stage} takes no efficiency quantity (TL §9; DL-c; AM-17b item 1(e); AM-18 item 3(a); AM-23 items 1 "
               "and 3(f); LEFF-AUDIT-RULE-1 A1): the run of record profiles the teacher and E1-E7 only")
    int8 = resolve_stage_artifact(stage)["kind"] == "int8_artifact"
    if args.expected_checkpoint_sha256 is None:
        refuse("expected_checkpoint_sha256_required",
               "--of-record requires --expected-checkpoint-sha256, the sha256 of the file this run is handed "
               "(--checkpoint, or --provenance for an INT8 stage)")
    if int8 and args.expected_converted_sha256 is None:
        refuse("expected_converted_sha256_required",
               "--of-record requires --expected-converted-sha256 for an INT8 stage, the sha256 of the converted "
               "artifact its provenance names")
    if args.artifact_role == ARTIFACT_ROLE_X86_LATENCY and args.expected_x86_sha256 is None:
        refuse("expected_x86_sha256_required",
               f"--of-record with --artifact-role {ARTIFACT_ROLE_X86_LATENCY} requires --expected-x86-sha256, the "
               "sha256 of the x86 latency copy its run record names")
    if not args.host_label:
        refuse("host_label_required",
               "--of-record requires --host-label, the host's name in the session (the RunPod pod, or the laptop)")
    code = governed_gate(GOVERNED_ROOT)
    image = image_gate()
    # ---- §1: the thread counts of record, then the host ----
    if args.threads is None:
        refuse("threads_required_of_record",
               f"--of-record requires --threads, the thread count of record ({THREADS_OF_RECORD}; {THREAD_RULING})")
    if args.threads != THREADS_OF_RECORD:
        refuse("threads_not_of_record",
               f"--threads {args.threads} is not the thread count of record {THREADS_OF_RECORD} ({THREAD_RULING})")
    if interop.get("status") != "set":
        refuse("interop_threads_locked",
               f"torch.set_num_interop_threads({INTEROP_THREADS_OF_RECORD}) did not take ({interop.get('status')}: "
               f"{interop.get('error')}); the run of record sets it first in main(), once per process")
    observed = torch.get_num_interop_threads()
    if observed != INTEROP_THREADS_OF_RECORD:
        refuse("interop_threads_not_of_record",
               f"torch.get_num_interop_threads() reads {observed}, not the inter-op count of record "
               f"{INTEROP_THREADS_OF_RECORD} ({THREAD_RULING})")
    facts = host_facts()
    error = host_thread_error(facts, args.threads)
    if error:
        refuse(*error)
    if "memory" in metrics:
        start = PROCESS_START_MEMORY
        if start is None:
            refuse("memory_sampler_unavailable", "the process-start memory values could not be read (§3)")
        verdict = parent_contamination(start["ru_maxrss_bytes"], start["vm_hwm_bytes"])
        if verdict["contaminated"]:
            refuse("memory_parent_contaminated",
                   f"ru_maxrss at process start ({start['ru_maxrss_bytes']} B) exceeds VmHWM at process start "
                   f"({start['vm_hwm_bytes']} B) by more than 64 MiB: the process inherited a large parent's "
                   "high-water mark; start it from a shell, one stage per process (§3)")
    return {"code": code, "image": image, "host_facts": facts}


def apply_threads(args) -> int | None:
    """torch.set_num_threads(T) before any host fact is read and any model is built, so every measurement runs at it
    (peak memory included, §3), and read back (B-1b's [threads_not_applied]). None when --threads is not given."""
    if args.threads is None:
        return None
    torch.set_num_threads(args.threads)
    got = torch.get_num_threads()
    if got != args.threads:
        refuse("threads_not_applied", f"torch.set_num_threads({args.threads}) left torch.get_num_threads() at {got}")
    return args.threads


# ------------------------------------------------------------------ §2 step 4: the source, hashed before its parse
def handed_file(args) -> Path | None:
    """The file this run is handed: --provenance for an INT8 stage, else --checkpoint."""
    raw = args.provenance if resolve_stage_artifact(args.stage)["kind"] == "int8_artifact" else args.checkpoint
    return Path(raw) if raw else None


def hash_before_parse(args) -> tuple:
    """P1 (B-1b): the handed file's sha256 before its first parse, compared with --expected-checkpoint-sha256. A
    missing file is left to the resolver's own refusal."""
    path = handed_file(args)
    if path is None or not path.is_file():
        return path, None
    actual = sha256_file(path)
    expected = args.expected_checkpoint_sha256
    if expected is not None and actual != expected:
        refuse("expected_checkpoint_sha256_mismatch",
               f"{path} has sha256 {actual}, expected {expected}; the file was not parsed")
    return path, actual


def require_unchanged(path, before: str | None) -> None:
    """P2 (B-1b; A5): the handed file after validation is the file that was hashed before it."""
    if before is None:
        return
    after = sha256_file(path)
    if after != before:
        refuse("source_changed_during_validation", f"{path} had sha256 {before} before validation and {after} after it")


def source_gates(args, resolved: dict, handed_sha: str | None) -> dict:
    """After the resolver: R2 (an INT8 artifact of record passes int8_official_error), H2, and an AM-18 arm
    payload refused of record. Returns the binding facts the record keeps."""
    kind = resolved["spec"]["kind"]
    official = None
    if kind == "int8_artifact":
        if args.of_record:
            from src.eval.stage_artifacts import int8_official_error
            error = int8_official_error(resolved, repo_root=REPO)
            if error:
                refuse("int8_not_official", error)
            official = True
        converted = args.expected_converted_sha256
        if converted is not None and resolved["artifact_sha256"] != converted:
            refuse("expected_converted_sha256_mismatch",
                   f"the converted artifact {resolved['artifact_path']} has sha256 {resolved['artifact_sha256']}, "
                   f"expected {converted}; it was not loaded")
    elif kind == "fp32_checkpoint" and args.of_record:
        keys = checkpoint_keys(resolved)
        if "arm" in keys:
            refuse("arm_artifact_not_of_record",
                   f"{resolved['checkpoint_path']} carries the trainer's `arm` key: an AM-18 arm student takes no "
                   "efficiency number (AM-18 item 3(a); scripts/run_ptq_arm.py :20-22)")
    return {"run_provenance_sha256": handed_sha if kind == "int8_artifact" else None, "int8_official": official}


def checkpoint_keys(resolved: dict) -> list:
    """The top-level keys of a validated FP32 checkpoint, parsed from the bytes it hashes: they must be the bytes the
    resolver validated ([checkpoint_changed_after_validation], as at the load and the size read)."""
    path = resolved["checkpoint_path"]
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != resolved["checkpoint_sha256"]:
        raise changed_error(resolved, path, actual, "the arm-key read")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    return sorted(str(k) for k in payload) if isinstance(payload, dict) else []


# ------------------------------------------------------------------ §2 step 5: the x86 latency copy (§4; LF33)
def bind_x86_copy(args, resolved: dict, handed_sha: str | None) -> dict:
    """The x86 latency copy of an INT8 stage, through its record (§2 step 5, in the plan's order): E4/E7 the run
    provenance's x86_latency_copy (resolved as scripts/ptq_val_checks.py's _companion resolves it), E5/E6 the eval
    directory's qat_finalize.json (scripts/qat_epoch_eval.py finalize). The file is hashed against its record and
    against --expected-x86-sha256, and its recorded role is an x86 latency copy's."""
    stage = resolved["spec"]["stage"]
    finalize_path = finalize_sha = None
    if stage in X86_PTQ_STAGES:
        rec = resolved["provenance"].get("x86_latency_copy")
        if not isinstance(rec, dict) or not rec.get("path") or not rec.get("sha256"):
            refuse("x86_copy_unrecorded",
                   f"the run provenance {args.provenance} records no x86_latency_copy path and sha256: only "
                   "scripts/run_ptq.py's PTQ writes one (src/quant/runner.py's legacy artifacts have none)")
        base = Path(args.provenance).resolve().parent
        role_fields, usable = rec, [rec.get("usable_for_accuracy")]
    else:
        fpath = Path(args.qat_finalize)
        try:
            raw = fpath.read_bytes()
            fin = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError) as e:
            refuse("qat_finalize_mismatch", f"{fpath} cannot be read as a {QAT_FINALIZE_FORMAT} record: {e}")
        if not isinstance(fin, dict) or fin.get("format") != QAT_FINALIZE_FORMAT or fin.get("stage") != stage:
            refuse("qat_finalize_mismatch",
                   f"{fpath} is not a {QAT_FINALIZE_FORMAT} record of {stage} (format "
                   f"{fin.get('format') if isinstance(fin, dict) else None!r})")
        if fin.get("status") != "finalized":
            refuse("qat_finalize_not_finalized",
                   f"{fpath} has status {fin.get('status')!r}; only a finalized record binds an x86 latency copy")
        aor = fin.get("artifact_of_record") or {}
        bound = ((aor.get("provenance") or {}).get("sha256"), (aor.get("torchscript") or {}).get("sha256"))
        if bound != (handed_sha, resolved["artifact_sha256"]):
            refuse("qat_finalize_mismatch",
                   f"{fpath} binds provenance {bound[0]} and TorchScript {bound[1]}; this run is handed "
                   f"{handed_sha} and its converted artifact is {resolved['artifact_sha256']}")
        rec = fin.get("x86_latency_copy") or {}
        checks = rec.get("checks") or {}
        failed = sorted(k for k, ok in checks.items() if ok is not True)
        if not checks or failed:
            refuse("x86_copy_checks_failed", f"{fpath}'s x86 latency copy did not pass its checks: {failed or 'none'}")
        if not rec.get("path") or not rec.get("sha256"):
            refuse("x86_copy_unrecorded", f"{fpath} records no x86_latency_copy path and sha256")
        base = fpath.resolve().parent
        role_fields = rec.get("provenance") or {}
        usable = [rec.get("usable_for_accuracy"), role_fields.get("usable_for_accuracy")]
        finalize_path, finalize_sha = str(fpath), hashlib.sha256(raw).hexdigest()
    path = Path(rec["path"]) if Path(rec["path"]).is_absolute() else base / rec["path"]
    error = forbidden_path_error("the x86 latency copy", path)
    if error:
        refuse("path_forbidden_name", error)
    if not path.is_file():
        refuse("x86_copy_missing", f"{path}, the x86 latency copy its record names, is missing")
    actual = sha256_file(path)
    if actual != rec["sha256"]:
        refuse("x86_copy_hash_mismatch", f"{path} has sha256 {actual}; its record names {rec['sha256']}")
    expected = args.expected_x86_sha256
    if expected is not None and actual != expected:
        refuse("expected_x86_sha256_mismatch",
               f"the x86 latency copy {path} has sha256 {actual}, expected {expected}; it was not loaded")
    if (role_fields.get("artifact_role") != ARTIFACT_ROLE_X86_LATENCY or role_fields.get("reduce_range") is not True
            or any(u is not False for u in usable)):
        refuse("x86_copy_role",
               f"{path}'s record says artifact_role {role_fields.get('artifact_role')!r}, reduce_range "
               f"{role_fields.get('reduce_range')!r}, usable_for_accuracy {usable}; an x86 latency copy is "
               f"{ARTIFACT_ROLE_X86_LATENCY}, reduce_range True and never usable for accuracy")
    return {"x86_copy": {"path": str(path), "sha256": actual, "engine": rec.get("engine")},
            "qat_finalize_path": finalize_path, "qat_finalize_sha256": finalize_sha}


def load_x86_copy(binding: dict, resolved: dict):
    """The bound copy under the x86 engine, its embedded identity checked (load_int8_torchscript)."""
    from src.eval.model_loading import CheckpointError, load_int8_torchscript

    spec = resolved["spec"]
    engine = x86_latency_copy_gate(spec["stage"])        # selects x86 or fbgemm, or refuses
    try:
        model, _meta = load_int8_torchscript(binding["x86_copy"]["path"], stage=spec["stage"], method=spec["method"],
                                             engine=engine, artifact_role=ARTIFACT_ROLE_X86_LATENCY)
    except CheckpointError as e:
        refuse("x86_copy_identity", f"{binding['x86_copy']['path']}: {e}")
    return model, engine


# ------------------------------------------------------------------ §2 step 6: the loaders (B-1; F1, LF24)
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
    from src.eval.model_loading import load_int8_student, load_student_checkpoint, load_teacher_model

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


# ------------------------------------------------------------------ size and footprint (B-1; LF26, LF27)
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
    parent-contamination test on them (the run of record refuses a contaminated process before the source)."""
    out = measure_peak_memory(model)
    start = PROCESS_START_MEMORY
    out["process_start"] = start
    out["parent_contamination"] = (parent_contamination(start["ru_maxrss_bytes"], start["vm_hwm_bytes"])
                                   if start else None)
    return out


# ------------------------------------------------------------------ §2 step 7: the profile
def profile(resolved: dict, *, stage: str, metrics, artifact_role: str, threads: int | None,
            provenance_path=None, teacher_config: str | None = None, teacher_sha256: str | None = None,
            teacher_builder=None, context: dict | None = None) -> dict:
    """Measure one stage in one role and assemble its record. `context` carries run()'s own facts (the gates of
    record, the handed file, the identity flags, the source gates, the x86 binding, the inter-op status and the host
    label); without it the record is the plain one a direct caller gets."""
    ctx = context or {}
    of_record, binding, gates = bool(ctx.get("of_record")), ctx.get("binding"), ctx.get("gates") or {}
    wall_start = datetime.now(timezone.utc).timestamp()
    kind = resolved["spec"]["kind"]
    precision = "int8" if kind == "int8_artifact" else "fp32"
    model_role = "teacher" if kind == "teacher_checkpoint" else "student"
    source_path, source_sha256 = source_identity(resolved)
    # ---- the file of record: its bytes re-hashed against the resolver's, before any model is built ----
    artifact = measure_artifact_size(resolved) if "size" in metrics else None

    if binding is not None:
        model, backend = load_x86_copy(binding, resolved)
        validate_artifact_role(artifact_role, backend)
    else:
        backend = (resolved["provenance"].get("backend") or "qnnpack") if precision == "int8" else None
        if precision == "int8":
            validate_artifact_role(artifact_role, backend)   # before anything is loaded
        model = load_model(resolved, stage=stage, teacher_config=teacher_config, teacher_sha256=teacher_sha256,
                           teacher_builder=teacher_builder)
    prov = provenance(stage=stage, model_role=model_role, precision=precision,
                      artifact_role=artifact_role, backend=backend, source_path=source_path,
                      source_sha256=source_sha256, expected_sha256=teacher_sha256)
    prov["handed"] = ctx.get("handed")
    prov["expected"] = ctx.get("expected")
    prov["binding"] = {"run_provenance_sha256": None, "qat_finalize_path": None, "qat_finalize_sha256": None,
                       "x86_copy": None, "state_dict_companion_sha256": None, "int8_official": None,
                       **(ctx.get("source") or {}), **(binding or {})}

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
        if footprint.get("companion_count"):
            prov["binding"]["state_dict_companion_sha256"] = footprint["companion_count"]["sha256"]
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
                prov["latency_backend"] = x86_latency_copy_gate(stage)
            latency = measure_latency(model, thread_count=threads)
        except EfficiencyError as e:
            deferred["latency"] = f"[{e.code}] {e}"
    if of_record and latency is not None and latency["intra_op_observed"] != threads:
        refuse("timed_threads_mismatch",
               f"the Timer's probe saw {latency['intra_op_observed']} intra-op threads, not the thread count of "
               f"record {threads}; nothing was written")
    if of_record and deferred:
        refuse("metric_deferred_of_record",
               f"the run of record defers nothing; {sorted(deferred)} could not be measured: {deferred}; nothing "
               "was written")

    interop = ctx.get("interop") or {}
    facts = gates.get("host_facts") or host_facts()
    thread_policy = {"intra_op_requested": threads,
                     "intra_op_observed": (latency or {}).get("intra_op_observed", torch.get_num_threads()),
                     "inter_op": {**interop, "observed": torch.get_num_interop_threads()},
                     "constant": THREADS_OF_RECORD, "inter_op_constant": INTEROP_THREADS_OF_RECORD,
                     "dl_row": THREAD_RULING}
    host = {"label": ctx.get("host_label"), "note": ctx.get("host_note"), "facts": facts,
            "fingerprint": facts["host_type_fingerprint"]}
    return efficiency_record(provenance_block=prov, parameters=parameters, footprint=footprint,
                             artifact=artifact, compute=compute, latency=latency, memory=memory,
                             deferred=deferred, of_record=of_record, metric_group=metric_group_of(metrics),
                             wall_clock=(wall_start, datetime.now(timezone.utc).timestamp()),
                             thread_policy=thread_policy, host=host, code=gates.get("code"),
                             image=gates.get("image"))


def write_record(out_dir: Path, record: dict, *, stage: str, artifact_role: str) -> Path:
    """Create the record exclusively: an existing file is never replaced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    group = record.get("metric_group") or "mixed"
    out_path = out_dir / f"efficiency_{stage}_{artifact_role}_{group}_{stamp}.json"
    with open(out_path, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(record, indent=2, sort_keys=True))
    return out_path


def run(args, *, teacher_builder=None, interop: dict | None = None) -> tuple[Path, dict]:
    """§2's steps in order, then the record. Raises on a refusal; nothing is written before the record.

    `teacher_builder` is the injectable teacher factory of src.eval.model_loading.load_teacher_model, for synthetic
    smokes only. `interop` is main()'s set_interop_threads() result; a caller passing none set no inter-op count in
    this call, which the run of record refuses ([interop_threads_locked]).
    """
    interop = interop or {"requested": INTEROP_THREADS_OF_RECORD, "status": "not set by this call", "error": None}
    metrics = parse_metrics(args.metrics)
    validate_cli(args, metrics)
    refuse_forbidden_paths(args)
    out_dir = check_out_dir(args.out_dir)
    threads = apply_threads(args)        # before any host fact and any model: every measurement at it (§3)
    gates = of_record_gates(args, metrics, interop) if args.of_record else {}

    # ---- §2 step 4, the source validated BEFORE any model construction: hashed before its first parse (P1), the
    # teacher's file compared with its flag before any parse, the gates of record, and the handed file re-hashed
    # after validation (P2) ----
    handed, handed_sha = hash_before_parse(args)
    resolved = resolve_evaluation_source(args.stage, checkpoint=args.checkpoint,
                                         provenance=args.provenance,
                                         expected_sha256=args.teacher_ckpt_sha256)
    refuse_forbidden_converted(resolved)
    source = source_gates(args, resolved, handed_sha)
    require_unchanged(handed, handed_sha)
    binding = (bind_x86_copy(args, resolved, handed_sha) if args.artifact_role == ARTIFACT_ROLE_X86_LATENCY
               else None)
    context = {"of_record": args.of_record, "gates": gates, "binding": binding, "source": source,
               "interop": interop, "host_label": args.host_label, "host_note": args.host_note,
               "handed": {"path": str(handed) if handed else None, "sha256": handed_sha},
               "expected": {"checkpoint_sha256": args.expected_checkpoint_sha256,
                            "converted_sha256": args.expected_converted_sha256,
                            "x86_sha256": args.expected_x86_sha256}}

    record = profile(resolved, stage=args.stage, metrics=metrics, artifact_role=args.artifact_role,
                     threads=threads, provenance_path=args.provenance,
                     teacher_config=args.teacher_config, teacher_sha256=args.teacher_ckpt_sha256,
                     teacher_builder=teacher_builder, context=context)
    return write_record(out_dir, record, stage=args.stage, artifact_role=args.artifact_role), record


def refusal_text(e: BaseException) -> str:
    """`[code] message` for a coded refusal (a message that already leads with its code is kept as it is)."""
    code, message = getattr(e, "code", None), str(e)
    if not code or message.startswith(f"[{code}]"):
        return message
    return f"[{code}] {message}"


def main(argv=None) -> int:
    interop = set_interop_threads(INTEROP_THREADS_OF_RECORD)    # first, once per process, before any model (LF30)
    args = build_parser().parse_args(argv)
    try:
        out_path, record = run(args, interop=interop)
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
