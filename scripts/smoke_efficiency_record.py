#!/usr/bin/env python3
"""Synthetic smoke of LEFF B-1: L-RESIDUAL-SHA's efficiency half, peak memory and the profiler's record.

The LEFF Phase A plan §9's B-1 smoke as the LEFF-B1 GO amends it (LF23-LF28, LF35): the stage sets (E-35;
LEFF-AUDIT-RULE-1 A1), the teacher's identity (DL-59), the loader repairs (F1) with the converted artifact's re-hash
before its load (B-1b's P3 twin), the serialized size with its model-only companion (Q7), the INT8 footprint read
from the artifact's own packed params and cross-checked against its state_dict companion (A13), peak memory (§3 with
A6's recorded slack) and the plantseg-efficiency/2.0.0 record.

Synthetic only (DL-41): random-weight students, a stub teacher, a synthetic PTQ run directory; no dataset, no
checkpoint of record, no GPU, no install. Temporary roots come from scripts/synthetic_ptq_fixtures.safe_tmpdir; no
file or folder named with "test" is created (a refused path is a never-created one, or a symlink whose never-created
target names it); the repository is unchanged (repo_state); no_forbidden_named_path_created is the last check. Each
high-water measurement runs in a fresh child process (`--child`), as a run of record does.

    python -B scripts/smoke_efficiency_record.py [--only CHECK[,CHECK...]]

Prints one line per check and `RESULT: PASS (n/n)`; exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True    # no __pycache__ in the repository, with or without -B

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import scripts.profile_efficiency as PE  # noqa: E402
from scripts.synthetic_ptq_fixtures import repo_state, safe_tmpdir, state_verdict  # noqa: E402
from src.eval import efficiency as EFF  # noqa: E402
from src.eval import model_loading as ML  # noqa: E402
from src.eval import stage_artifacts as SA  # noqa: E402

NC = 116
MiB = 1024 * 1024
TMP: Path = None                     # set by main(); a --child process creates no temporary root
TEACHER_CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
AM23 = ("E6-E2", "E7-E2", "E6-R1", "E6-R2")
LF23_STAGES = ("teacher", "E1", "E2", "E3", "A", "F", "G", "E4", "E5", "E6", "E7")
FORBIDDEN = "test"                   # the token no created path may hold, in any case (with ":")

RESULTS: list[tuple[str, bool, str]] = []
CASES: dict[str, object] = {}
FIX: dict[str, object] = {}
_COUNTER = [0]


def case(name: str):
    def register(fn):
        if name in CASES or FORBIDDEN in name.lower():
            raise RuntimeError(f"duplicate or forbidden case name {name!r}")
        CASES[name] = fn
        return fn
    return register


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


def fresh(tag: str) -> Path:
    """A never-created path under the smoke's temporary root."""
    _COUNTER[0] += 1
    return TMP / "work" / f"{tag}_{_COUNTER[0]:03d}"


@contextlib.contextmanager
def patched(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


class Counter:
    """Wraps a callable and counts its calls."""

    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, *a, **kw):
        self.n += 1
        return self.fn(*a, **kw)


def code_of(fn, *a, **kw):
    """The refusal code `fn` raises (its `code`, else its type name), or None when it returns."""
    try:
        fn(*a, **kw)
    except Exception as e:                                   # noqa: BLE001
        return getattr(e, "code", None) or type(e).__name__
    return None


def parser_exit(argv) -> object:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            PE.build_parser().parse_args(argv)
    except SystemExit as e:
        return e.code
    return None


def run_profile(argv, **kw):
    """(code, out_path, record): the refusal code, or None with the written record."""
    args = PE.build_parser().parse_args([str(a) for a in argv])
    try:
        out_path, record = PE.run(args, **kw)
    except Exception as e:                                   # noqa: BLE001
        return (getattr(e, "code", None) or type(e).__name__), None, None
    return None, out_path, record


def main_rc(argv) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = PE.main([str(a) for a in argv])
    return rc, err.getvalue()


# ---------------------------------------------------------------- fixtures (synthetic, built once)
class StubMSCAN(nn.Module):
    def __init__(self, dims=(64, 128, 320, 512), strides=(4, 8, 16, 32)):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(3, d, 1, bias=False) for d in dims])
        self.strides = strides

    def forward(self, x):
        return tuple(p(torch.nn.functional.adaptive_avg_pool2d(
            x, (max(x.shape[-2] // s, 1), max(x.shape[-1] // s, 1))))
            for p, s in zip(self.projs, self.strides))


class StubHead(nn.Module):
    def __init__(self, num_classes=NC):
        super().__init__()
        self.num_classes = num_classes
        self.squeeze = nn.Conv2d(128, 512, 1, bias=False)
        self.conv_seg = nn.Conv2d(512, num_classes, 1)

    def forward(self, feats):
        return self.conv_seg(self.squeeze(feats[1]))


class StubSegNeXt(nn.Module):
    def __init__(self, num_classes=NC):
        super().__init__()
        self.backbone = StubMSCAN()
        self.decode_head = StubHead(num_classes)

    def extract_feat(self, x):
        return self.backbone(x)


def stub_builder(_path):
    """scripts/smoke_eval_teacher.py's stub teacher (no MMSeg stack)."""
    from src.distill.segnext_teacher import SegNeXtTeacherAdapter
    return SegNeXtTeacherAdapter(StubSegNeXt())


def _student_state(seed: int) -> dict:
    from src.models.student import build_student
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in build_student(pretrained=False).state_dict().items()}


def write_fp32_checkpoint(path: Path, *, stage: str | None = None, seed: int = 7) -> Path:
    """A random-weight student in the trainers' schema, with Adam-like optimizer state (F5: the file of record holds
    it, so the model-only companion is the smaller number)."""
    state = _student_state(seed)
    floats = [v for v in state.values() if v.is_floating_point()]
    moments = {i: {"exp_avg": t.clone(), "exp_avg_sq": t.clone()} for i, t in enumerate(floats)}
    payload = {"iter": 80000, "model_state_dict": state,
               "optimizer_state_dict": {"state": moments, "param_groups": [{"lr": 0.01, "params": sorted(moments)}]},
               "num_classes": NC, "best_val_miou_all_class": 0.0}
    if stage is not None:
        payload["stage"] = stage
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def write_teacher_checkpoint(path: Path, *, seed: int = 3) -> Path:
    """scripts/smoke_eval_teacher.py's synthetic teacher file, with an `optimizer` entry as an MMSeg checkpoint
    of a fine-tune carries one."""
    g = torch.Generator().manual_seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": {"mmseg_version": "1.2.2"},
                "state_dict": {"backbone.projs.0.weight": torch.randn(4, 3, 1, 1, generator=g),
                               "decode_head.conv_seg.weight": torch.randn(NC, 8, 1, 1, generator=g),
                               "decode_head.conv_seg.bias": torch.randn(NC, generator=g)},
                "optimizer": {"state": {}, "param_groups": [{"lr": 6e-5}]}}, path)
    return path


def build_int8_run(run_dir: Path, *, stage: str = "E4", seed: int = 0) -> Path:
    """A synthetic E4 PTQ run directory: the QNNPACK TorchScript artifact with its embedded identity, its state_dict
    companion and the run provenance fields src/eval/stage_artifacts.validate_int8_artifact reads. Random weights,
    calibrated on two random inputs: never of record."""
    from src.models.student import build_student
    from src.quant.prepare import calibrate, convert_model, prepare_ptq
    from src.quant.ptq import state_dict_bytes, torchscript_bytes

    torch.backends.quantized.engine = "qnnpack"
    torch.manual_seed(seed)
    prepared = prepare_ptq(build_student(pretrained=False).eval(), select_backend=False)
    g = torch.Generator().manual_seed(seed + 1)
    calibrate(prepared, [torch.randn(1, 3, 64, 64, generator=g) for _ in range(2)])
    converted = convert_model(prepared).eval()
    key = stage.lower()
    names = {"torchscript": f"{key}_int8_qnnpack.torchscript.pt", "state_dict": f"{key}_int8_student.pt",
             "run_meta": f"{key}_ptq_run_meta.json"}
    meta = {"schema": ML.INT8_TORCHSCRIPT_SCHEMA, "stage": stage, "quantization": "ptq", "num_classes": NC,
            "engine": "qnnpack", "artifact_role": "accuracy"}
    ts_b, _notes = torchscript_bytes(converted, meta)
    sd_b = state_dict_bytes(converted, stage, "ptq")
    run_dir.mkdir(parents=True)
    (run_dir / names["torchscript"]).write_bytes(ts_b)
    (run_dir / names["state_dict"]).write_bytes(sd_b)
    prov = {"stage": stage, "source_stage": "E1", "quantization_method": "ptq", "random_init": False,
            "num_classes": NC, "backend": "qnnpack", "artifact_format": "torchscript",
            "converted_artifact": names["torchscript"], "converted_artifact_sha256": sha256_bytes(ts_b),
            "state_dict_artifact": {"path": names["state_dict"], "sha256": sha256_bytes(sd_b)},
            "ptq_run_role": "smoke", "synthetic": "LEFF B-1 smoke fixture: random weights, never of record"}
    (run_dir / names["run_meta"]).write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return run_dir / names["run_meta"]


def copy_run(src_meta: Path, tag: str) -> Path:
    """A copy of a run directory (the original stays intact for the other cases)."""
    dst = fresh(tag)
    dst.mkdir(parents=True)
    for name in sorted(os.listdir(src_meta.parent)):
        (dst / name).write_bytes((src_meta.parent / name).read_bytes())
    return dst / src_meta.name


def fixture(name: str):
    if name not in FIX:
        base = TMP / "fixtures"
        if name == "e1":
            FIX[name] = write_fp32_checkpoint(base / "e1_best.pt")
        elif name == "arm_a":
            FIX[name] = write_fp32_checkpoint(base / "a_best.pt", stage="A", seed=9)
        elif name == "teacher":
            path = write_teacher_checkpoint(base / "teacher_iter.pth")
            FIX[name] = (path, sha256_path(path))
        elif name == "e4":
            FIX[name] = build_int8_run(base / "e4_run")
        else:
            raise KeyError(name)
    return FIX[name]


def teacher_argv(out: Path, *, sha="__file__", config=True, ckpt=None, metrics="params,size") -> list:
    path, digest = fixture("teacher")
    argv = ["--stage", "teacher", "--checkpoint", ckpt or path, "--metrics", metrics, "--out-dir", out]
    if sha is not None:
        argv += ["--teacher-ckpt-sha256", digest if sha == "__file__" else sha]
    if config:
        argv += ["--teacher-config", TEACHER_CONFIG]
    return argv


def e1_argv(out: Path, *, ckpt=None, metrics="params,size") -> list:
    return ["--stage", "E1", "--checkpoint", ckpt or fixture("e1"), "--metrics", metrics, "--out-dir", out]


def e4_argv(out: Path, *, prov=None, metrics="params,size") -> list:
    return ["--stage", "E4", "--provenance", prov or fixture("e4"), "--metrics", metrics, "--out-dir", out]


def size_record(tag: str):
    """One `size` run per synthetic source, shared by the size checks."""
    key = f"size_{tag}"
    if key not in FIX:
        argv = {"e1": e1_argv, "e4": e4_argv}[tag](fresh(key), metrics="size")
        FIX[key] = run_profile(argv)
    return FIX[key]


# ---------------------------------------------------------------- 1. stage sets (LF23; E-35; A1)
@case("stages_include_arms")
def _stages():
    choices = list(next(a for a in PE.build_parser()._actions if a.dest == "stage").choices)
    ok = (EFF.STAGES == LF23_STAGES and EFF.INT8_STAGES == ("E4", "E5", "E6", "E7")
          and choices == list(EFF.STAGES) and {"A", "F", "G"} <= set(EFF.STAGES))
    return ok, f"STAGES={EFF.STAGES}; --stage choices={choices}"


@case("analysis_stages_exclude_arms")
def _analysis():
    ana = EFF.EFFICIENCY_ANALYSIS_STAGES
    out = {"A", "F", "G", "E6-KD", *AM23}
    ok = (ana == ("teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7") and not out & set(ana)
          and set(ana) <= set(EFF.STAGES))
    return ok, f"EFFICIENCY_ANALYSIS_STAGES={ana}"


@case("arm_stages_resolve_fp32")
def _arms():
    kinds = {s: SA.resolve_stage_artifact(s)["kind"] for s in ("A", "F", "G")}
    resolved = SA.resolve_evaluation_source("A", checkpoint=str(fixture("arm_a")))
    model = PE.load_model(resolved, stage="A")
    as_arm = code_of(SA.resolve_evaluation_source, "A", checkpoint=str(fixture("e1")))   # E1 records no stage
    ok = (set(kinds.values()) == {"fp32_checkpoint"} and isinstance(model, nn.Module)
          and resolved["declared_stage"] == "A" and as_arm == "stage_undeclared")
    return ok, f"{kinds}; A checkpoint -> {type(model).__name__}; an E1 checkpoint as A -> {as_arm}"


@case("e6kd_not_profiled")
def _e6kd():
    out = fresh("e6kd")
    rc = parser_exit(["--stage", "E6-KD", "--out-dir", str(out)])
    rc_am23 = [parser_exit(["--stage", s, "--out-dir", str(out)]) for s in AM23]
    prov = code_of(EFF.provenance, stage="E6-KD", model_role="student", precision="int8", artifact_role="accuracy",
                   backend="qnnpack")
    gate = code_of(EFF.x86_latency_copy_gate, "E6-KD")
    ok = (rc == 2 and rc_am23 == [2] * 4 and prov == "unknown_stage" and gate == "x86_latency_copy_not_int8_stage"
          and "E6-KD" not in EFF.STAGES and not os.path.lexists(out))
    return ok, f"parser exit {rc} (AM-23 {rc_am23}); provenance {prov}; x86 gate {gate}"


# ---------------------------------------------------------------- 2. the teacher's identity (LF25; DL-59)
def _teacher_refused(argv, want: str, text: str):
    """Refused with `want` and scripts/evaluate_model.py's message (:213-224), `[code] text`, nothing created."""
    out = Path(argv[argv.index("--out-dir") + 1])
    code, _p, _r = run_profile(argv, teacher_builder=stub_builder)
    rc, err = main_rc(argv)
    ok = code == want and rc == 2 and f"[{want}] {text}" in err and not os.path.lexists(out)
    return ok, f"{code}; main exit {rc}: {err.strip()[:160]}"


def _format_text(value) -> str:
    from src.distill.teacher import sha256_format_error
    return f"--teacher-ckpt-sha256 {sha256_format_error(value)}"


@case("teacher_sha_required")
def _t_required():
    return _teacher_refused(teacher_argv(fresh("t_req"), sha=None), "teacher_ckpt_sha256_required",
                            "the teacher stage requires --teacher-ckpt-sha256, the checkpoint's SHA-256, compared "
                            "before the file is parsed")


@case("teacher_sha_format_uppercase")
def _t_upper():
    upper = fixture("teacher")[1].upper()
    return _teacher_refused(teacher_argv(fresh("t_up"), sha=upper), "teacher_ckpt_sha256_format", _format_text(upper))


@case("teacher_sha_format_empty")
def _t_empty():
    return _teacher_refused(teacher_argv(fresh("t_empty"), sha=""), "teacher_ckpt_sha256_format", _format_text(""))


@case("teacher_sha_not_teacher")
def _t_not_teacher():
    argv = e1_argv(fresh("t_not")) + ["--teacher-ckpt-sha256", fixture("teacher")[1]]
    return _teacher_refused(argv, "teacher_ckpt_sha256_not_teacher",
                            "--teacher-ckpt-sha256 applies to the teacher stage only; stage='E1' takes none")


@case("teacher_config_required")
def _t_config():
    return _teacher_refused(teacher_argv(fresh("t_cfg"), config=False), "teacher_config_required",
                            "the teacher stage requires --teacher-config (the thesis teacher config)")


@case("teacher_hash_before_parse")
def _t_before_parse():
    garbage = fresh("t_garbage").with_suffix(".pth")
    garbage.parent.mkdir(parents=True, exist_ok=True)
    garbage.write_bytes(b"not a checkpoint: " + os.urandom(64))
    loads, builds = Counter(torch.load), Counter(stub_builder)
    with patched(torch, "load", loads):
        code, _p, _r = run_profile(teacher_argv(fresh("t_parse"), ckpt=garbage, sha="0" * 64), teacher_builder=builds)
    return (code == "teacher_hash_mismatch" and loads.n == 0 and builds.n == 0,
            f"{code}; torch.load calls {loads.n}; builder calls {builds.n}")


@case("teacher_hash_rechecked_at_load")
def _t_at_load():
    src, digest = fixture("teacher")
    moved = fresh("t_moved").with_suffix(".pth")
    moved.parent.mkdir(parents=True, exist_ok=True)
    moved.write_bytes(src.read_bytes())
    resolved = SA.resolve_evaluation_source("teacher", checkpoint=str(moved), expected_sha256=digest)
    write_teacher_checkpoint(moved, seed=4)                  # the file changes between validation and load
    builds = Counter(stub_builder)
    code = code_of(PE.load_model, resolved, stage="teacher", teacher_config=str(TEACHER_CONFIG),
                   teacher_sha256=digest, teacher_builder=builds)
    return code == "teacher_ckpt_sha256_mismatch" and builds.n == 0, f"{code}; builder calls {builds.n}"


@case("teacher_record_written")
def _t_record():
    _src, digest = fixture("teacher")
    code, path, rec = run_profile(teacher_argv(fresh("t_rec")), teacher_builder=stub_builder)
    if code is not None:
        return False, f"refused: {code}"
    prov = rec["provenance"]
    comp = rec["artifact"]["model_only_companion"]
    ok = (path.is_file() and prov["stage"] == "teacher" and prov["model_role"] == "teacher"
          and prov["source_artifact_sha256"] == digest == prov["expected_sha256"]
          and rec["artifact"]["sha256"] == digest and comp["state_key"] == "state_dict"
          and comp["checkpoint_holds_optimizer_state"] is True and rec["parameters"]["total"] > 0)
    return ok, f"{path.name}; source {prov['source_artifact_sha256'][:12]} expected {str(prov['expected_sha256'])[:12]}"


# ---------------------------------------------------------------- 3. the loaders (LF24; F1)
@case("fp32_dispatch_no_keyerror")
def _fp32_dispatch():
    resolved = SA.resolve_evaluation_source("E1", checkpoint=str(fixture("e1")))
    try:
        model = PE.load_model(resolved, stage="E1")
    except Exception as e:                                   # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return isinstance(model, nn.Module) and not isinstance(model, tuple), type(model).__name__


@case("int8_dispatch_no_keyerror")
def _int8_dispatch():
    resolved = SA.resolve_evaluation_source("E4", provenance=str(fixture("e4")))
    try:
        model = PE.load_model(resolved, stage="E4")
    except Exception as e:                                   # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return isinstance(model, torch.jit.ScriptModule), type(model).__name__


@case("teacher_dispatch_by_spec_kind")
def _teacher_dispatch():
    _src, digest = fixture("teacher")
    resolved = SA.resolve_evaluation_source("teacher", checkpoint=str(fixture("teacher")[0]), expected_sha256=digest)
    teacher = PE.load_model(resolved, stage="teacher", teacher_config=str(TEACHER_CONFIG), teacher_sha256=digest,
                            teacher_builder=stub_builder)
    e1 = SA.resolve_evaluation_source("E1", checkpoint=str(fixture("e1")))
    student = PE.load_model(e1, stage="teacher")             # the spec kind decides, not the stage argument
    ok = isinstance(teacher, ML.TeacherEvalModel) and not isinstance(student, ML.TeacherEvalModel)
    return ok, f"teacher kind -> {type(teacher).__name__}; fp32 kind (stage arg 'teacher') -> {type(student).__name__}"


@case("loaders_unpacked_to_modules")
def _unpacked():
    _src, digest = fixture("teacher")
    got = {
        "teacher": PE.load_model(SA.resolve_evaluation_source("teacher", checkpoint=str(fixture("teacher")[0]),
                                                              expected_sha256=digest),
                                 stage="teacher", teacher_config=str(TEACHER_CONFIG), teacher_sha256=digest,
                                 teacher_builder=stub_builder),
        "fp32": PE.load_model(SA.resolve_evaluation_source("E1", checkpoint=str(fixture("e1"))), stage="E1"),
        "int8": PE.load_model(SA.resolve_evaluation_source("E4", provenance=str(fixture("e4"))), stage="E4"),
    }
    ok = all(isinstance(m, nn.Module) and not isinstance(m, tuple) for m in got.values())
    return ok, str({k: type(v).__name__ for k, v in got.items()})


@case("fp32_student_counted_as_fp32")
def _fp32_counted():
    from src.models.student import build_student
    student = build_student(pretrained=False).eval()
    floats = sum(1 for m in student.modules() if type(m).__name__ == "FloatFunctional")
    params = EFF.architecture_parameter_count(student)
    macs = EFF.profile_macs(student, counter=lambda _m, _i: (1, {}, "injected"))
    torch.backends.quantized.engine = "qnnpack"
    meta = json.loads(Path(fixture("e4")).read_text(encoding="utf-8"))
    converted = ML.rebuild_int8_from_state_dict(Path(fixture("e4")).parent / meta["state_dict_artifact"]["path"],
                                                stage="E4", method="ptq")
    ok = (floats > 0 and not EFF.is_quantized(student) and params["counted_from"] == "model" and params["total"] > 0
          and macs["macs"] == 1 and EFF.is_quantized(converted))
    return ok, (f"{floats} FloatFunctional modules; the FP32 student counted from its model ({params['total']} "
                f"parameters) and its FLOPs path open; the converted student is quantized")


@case("source_identity_fp32")
def _identity_fp32():
    path = fixture("e1")
    code, _p, rec = run_profile(e1_argv(fresh("id_fp32"), metrics="params"))
    if code is not None:
        return False, f"refused: {code}"
    prov = rec["provenance"]
    ok = (prov["source_artifact_path"] == str(Path(path)) and prov["source_artifact_sha256"] == sha256_path(path)
          and prov["expected_sha256"] is None)
    return ok, f"{Path(prov['source_artifact_path']).name} {prov['source_artifact_sha256'][:12]}"


@case("source_identity_int8")
def _identity_int8():
    resolved = SA.resolve_evaluation_source("E4", provenance=str(fixture("e4")))
    code, _p, rec = run_profile(e4_argv(fresh("id_int8"), metrics="params"))
    if code is not None:
        return False, f"refused: {code}"
    prov = rec["provenance"]
    ok = (prov["source_artifact_path"] == str(resolved["artifact_path"])
          and prov["source_artifact_sha256"] == resolved["artifact_sha256"] == sha256_path(resolved["artifact_path"])
          and prov["backend"] == "qnnpack" and rec["parameters"]["counted_from"] == "fp32_reference_architecture")
    return ok, f"{Path(prov['source_artifact_path']).name} {prov['source_artifact_sha256'][:12]}"


@case("checkpoint_changed_after_validation")
def _fp32_changed():
    path = fresh("ck_changed").with_suffix(".pt")
    write_fp32_checkpoint(path, seed=21)
    resolved = SA.resolve_evaluation_source("E1", checkpoint=str(path))
    write_fp32_checkpoint(path, seed=22)                     # a valid checkpoint, other bytes
    code = code_of(PE.load_model, resolved, stage="E1")
    return code == "checkpoint_changed_after_validation", str(code)


@case("int8_converted_changed_before_load")
def _int8_changed():
    meta = copy_run(fixture("e4"), "conv_changed")
    resolved = SA.resolve_evaluation_source("E4", provenance=str(meta))
    art = Path(resolved["artifact_path"])
    art.write_bytes(art.read_bytes() + b"\0")                # the converted artifact changes after validation
    loads = Counter(ML.load_int8_student)
    with patched(ML, "load_int8_student", loads):
        try:
            PE.load_model(resolved, stage="E4")
            code, text = None, ""
        except Exception as e:                               # noqa: BLE001
            code, text = getattr(e, "code", None) or type(e).__name__, str(e)
    want = (f"the converted artifact {art} had sha256 {resolved['artifact_sha256']} at validation and "
            f"{sha256_path(art)} before the load; it was not loaded")
    ok = code == "converted_changed_before_load" and loads.n == 0 and text == want
    return ok, f"{code}; load_int8_student calls {loads.n}"


# ---------------------------------------------------------------- 4. size and footprint (LF26, LF27; A13, Q7)
@case("serialized_size_measured_fp32")
def _size_fp32():
    code, _p, rec = size_record("e1")
    if code is not None:
        return False, f"refused: {code}"
    path = Path(fixture("e1"))
    art, comp = rec["artifact"], rec["artifact"]["model_only_companion"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    buf = io.BytesIO()
    torch.save(payload["model_state_dict"], buf)
    ok = (art["serialized_artifact_bytes"] == path.stat().st_size and art["sha256"] == sha256_path(path)
          and art["size_measured_from_latency_copy"] is False and comp["state_key"] == "model_state_dict"
          and comp["model_only_serialized_bytes"] == len(buf.getbuffer())
          and 0 < comp["model_only_serialized_bytes"] < art["serialized_artifact_bytes"]
          and comp["checkpoint_holds_optimizer_state"] is True and comp["source_sha256"] == art["sha256"]
          and rec["footprint"]["footprint_source"] == "model")
    return ok, (f"file {art['serialized_artifact_bytes']} B; model-only companion "
                f"{comp['model_only_serialized_bytes']} B; keys {comp['checkpoint_top_level_keys']}")


@case("serialized_size_measured_int8")
def _size_int8():
    code, _p, rec = size_record("e4")
    if code is not None:
        return False, f"refused: {code}"
    resolved = SA.resolve_evaluation_source("E4", provenance=str(fixture("e4")))
    art = rec["artifact"]
    ok = (art["serialized_artifact_bytes"] == Path(resolved["artifact_path"]).stat().st_size
          and art["sha256"] == resolved["artifact_sha256"] and art["size_artifact_role"] == "accuracy"
          and art["size_measured_from_latency_copy"] is False and "model_only_companion" not in art)
    return ok, f"QNNPACK artifact {art['serialized_artifact_bytes']} B"


@case("traced_module_hides_packed_weights")
def _f4():
    model = PE.load_model(SA.resolve_evaluation_source("E4", provenance=str(fixture("e4"))), stage="E4")
    hidden = EFF.weight_footprint(model)["int8_weight_elements"]
    read = EFF.packed_params_footprint(model)["int8_weight_elements"]
    return hidden == 0 and read > 0, f"weight_footprint on the TorchScript artifact {hidden}; packed params {read}"


@case("int8_footprint_from_artifact_equals_companion")
def _footprint():
    code, _p, rec = size_record("e4")
    if code is not None:
        return False, f"refused: {code}"
    fp = rec["footprint"]
    meta = json.loads(Path(fixture("e4")).read_text(encoding="utf-8"))
    torch.backends.quantized.engine = "qnnpack"
    eager = ML.rebuild_int8_from_state_dict(Path(fixture("e4")).parent / meta["state_dict_artifact"]["path"],
                                            stage="E4", method="ptq")
    independent = EFF.weight_footprint(eager)
    keys = ("int8_weight_elements", "fp32_weight_elements", "metadata_elements")
    ok = (fp["footprint_source"] == "artifact" and fp["artifact_count"]["packed_params_unreadable"] == []
          and fp["artifact_count"]["packed_params_unpacked"] > 0 and fp["int8_weight_elements"] > 0
          and all(fp["artifact_count"][k] == fp["companion_count"][k] == independent[k] == fp[k] for k in keys))
    return ok, (f"source {fp['footprint_source']}; INT8 {fp['int8_weight_elements']}, FP32 "
                f"{fp['fp32_weight_elements']}, metadata {fp['metadata_elements']} elements; "
                f"{fp['artifact_count']['packed_params_unpacked']} packed params unpacked")


@case("state_dict_companion_mismatch")
def _companion_mismatch():
    meta = copy_run(fixture("e4"), "comp_bad")
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["state_dict_artifact"]["sha256"] = "f" * 64
    meta.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    code, _p, _r = run_profile(e4_argv(fresh("comp_bad_out"), prov=meta, metrics="size"))
    return code == "state_dict_companion_mismatch", str(code)


@case("int8_footprint_counts_disagree_refused")
def _counts_disagree():
    real = PE.companion_footprint

    def off_by_one(resolved, provenance_path):
        counts = dict(real(resolved, provenance_path))
        counts["int8_weight_elements"] += 1
        return counts
    with patched(PE, "companion_footprint", off_by_one):
        code, _p, _r = run_profile(e4_argv(fresh("disagree"), metrics="size"))
    return code == "state_dict_companion_mismatch", str(code)


@case("int8_footprint_fallback_named")
def _fallback():
    real, calls = EFF._unpack_packed_params, [0]

    def first_unreadable(packed):
        calls[0] += 1
        if calls[0] == 1:
            raise ValueError("RuntimeError: injected: this packed-params type cannot be unpacked")
        return real(packed)
    with patched(EFF, "_unpack_packed_params", first_unreadable):
        code, _p, rec = run_profile(e4_argv(fresh("fallback"), metrics="size"))
    if code is not None:
        return False, f"refused: {code}"
    fp = rec["footprint"]
    unreadable = fp["artifact_count"]["packed_params_unreadable"]
    keys = ("int8_weight_elements", "fp32_weight_elements", "metadata_elements")
    ok = (fp["footprint_source"] == "state_dict_companion" and len(unreadable) == 1
          and unreadable[0]["module"] and "injected" in unreadable[0]["error"]
          and all(fp[k] == fp["companion_count"][k] for k in keys))
    return ok, f"source {fp['footprint_source']}; unreadable {[u['module'] for u in unreadable]}"


@case("size_read_rehash_refused")
def _size_rehash():
    path = fresh("size_fp32").with_suffix(".pt")
    write_fp32_checkpoint(path, seed=31)
    resolved = SA.resolve_evaluation_source("E1", checkpoint=str(path))
    write_fp32_checkpoint(path, seed=32)
    fp32 = code_of(PE.measure_artifact_size, resolved)
    meta = copy_run(fixture("e4"), "size_int8")
    resolved8 = SA.resolve_evaluation_source("E4", provenance=str(meta))
    art = Path(resolved8["artifact_path"])
    art.write_bytes(art.read_bytes() + b"\0")
    int8 = code_of(PE.measure_artifact_size, resolved8)
    return (fp32 == "checkpoint_changed_after_validation" and int8 == "converted_changed_before_load",
            f"FP32 {fp32}; INT8 {int8}")


@case("size_companion_hashed_before_parse")
def _companion_before_parse():
    loads = Counter(torch.load)
    with patched(torch, "load", loads):
        code = code_of(EFF.model_only_serialization, fixture("e1"), expected_sha256="0" * 64)
    return code == "checkpoint_changed_after_validation" and loads.n == 0, f"{code}; torch.load calls {loads.n}"


# ---------------------------------------------------------------- 5. peak memory (LF28; §3; A6)
CHILD_TAG = "CHILD_JSON:"
ENV_CAUSES = ("is absent", "is not writable")    # LF28: environment-caused only when clear_refs cannot be written


def child_forward() -> dict:
    """A forward that allocates 64 MiB, every page touched: the phase's high-water mark must see it."""
    class Alloc64(nn.Module):
        def forward(self, x):
            block = torch.ones(16 * 1024 * 1024)
            return x.mean() + block[-1]
    return PE.measure_memory(Alloc64())


def child_spike() -> dict:
    """128 MiB touched and freed before the phase (the load spike), then an allocation-free forward."""
    spike = torch.ones(32 * 1024 * 1024)
    del spike

    class Quiet(nn.Module):
        def forward(self, x):
            return x[:, :1, :1, :1]
    return PE.measure_memory(Quiet())


def child_reset() -> dict:
    """128 MiB touched and freed, then the reset: the high-water mark must come back to the resident set."""
    spike = torch.ones(32 * 1024 * 1024)
    del spike
    gc.collect()
    before = EFF.read_proc_status()
    slack = EFF.hwm_slack_bytes(torch.get_num_threads(), os.sysconf("SC_PAGE_SIZE"))
    reset = EFF.reset_rss_high_water(slack_bytes=slack)
    return {"before": before, "after": EFF.read_proc_status(), "reset": reset, "slack": slack}


CHILDREN = {"forward": child_forward, "spike": child_spike, "reset": child_reset}


def run_child(name: str) -> dict:
    """One high-water measurement in a fresh process, as a run of record is: the smoke's own mark cannot confound it."""
    key = f"child_{name}"
    if key not in FIX:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--child", name], cwd=str(REPO),
                           env=env, capture_output=True, text=True, timeout=900)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith(CHILD_TAG)]
        FIX[key] = (json.loads(lines[-1][len(CHILD_TAG):]) if lines
                    else {"error": f"exit {p.returncode}: {(p.stdout + p.stderr)[-400:]}"})
    return FIX[key]


def environment_caused(res: dict) -> str | None:
    if res.get("refused") == "memory_hwm_reset_unavailable" and any(c in res.get("message", "") for c in ENV_CAUSES):
        return (f"ENVIRONMENT-CAUSED [memory_hwm_reset_unavailable] {res['message'][:160]} (LB-0 proves it in the "
                "KD image)")
    return None


def child_failed(res: dict) -> str | None:
    return None if not ("error" in res or "refused" in res) else str(res)[:300]


def write_status(rss_kb: int, hwm_kb: int) -> Path:
    """A synthetic /proc/self/status with the two lines read_proc_status reads."""
    p = fresh("status_probe")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"Name:\tpython3\nVmHWM:\t{hwm_kb:8d} kB\nVmRSS:\t{rss_kb:8d} kB\n", encoding="ascii")
    return p


def writable_clear_refs() -> Path:
    """A regular file standing in for /proc/self/clear_refs (the write succeeds; the status file decides)."""
    p = fresh("clear_refs_probe")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="ascii")
    return p


def _calls(tree, name: str) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)]


@case("adjacent_samples_removed")
def _adjacent():
    runner = ast.parse((REPO / "scripts" / "profile_efficiency.py").read_text(encoding="utf-8"))
    eff = ast.parse((REPO / "src" / "eval" / "efficiency.py").read_text(encoding="utf-8"))
    fn = next(n for n in eff.body if isinstance(n, ast.FunctionDef) and n.name == "measure_peak_memory")
    samples = [i for i, st in enumerate(fn.body) if _calls(st, "read_proc_status")]
    loops = [i for i, st in enumerate(fn.body)
             if any(isinstance(n, ast.For) and _calls(n, "model") for n in ast.walk(st))]
    ok = (not _calls(runner, "rss_bytes") and len(samples) >= 2 and len(loops) == 1
          and samples[0] < loops[0] < samples[-1])
    return ok, (f"profile_efficiency.py rss_bytes() calls {len(_calls(runner, 'rss_bytes'))}; measure_peak_memory "
                f"statements: samples {samples}, forward loop {loops}")


@case("hwm_reset_effective")
def _reset_effective():
    res = run_child("reset")
    env = environment_caused(res)
    if env:
        return True, env
    bad = child_failed(res)
    if bad:
        return False, bad
    b, a, slack = res["before"], res["after"], res["slack"]
    ok = (b["vm_hwm_bytes"] >= b["vm_rss_bytes"] + 100 * MiB and a["vm_hwm_bytes"] <= a["vm_rss_bytes"] + slack
          and res["reset"]["slack_bytes"] == slack)
    return ok, (f"VmHWM - VmRSS before the reset {(b['vm_hwm_bytes'] - b['vm_rss_bytes']) / MiB:.1f} MiB, after "
                f"{(a['vm_hwm_bytes'] - a['vm_rss_bytes']) / MiB:.2f} MiB (slack {slack / MiB:.2f} MiB)")


@case("memory_sees_forward_allocation")
def _sees_forward():
    res = run_child("forward")
    env = environment_caused(res)
    if env:
        return True, env
    bad = child_failed(res)
    if bad:
        return False, bad
    delta = res["peak_rss_delta_bytes"]
    return 60 * MiB <= delta <= 96 * MiB, f"a 64 MiB forward allocation: delta {delta / MiB:.1f} MiB (60-96)"


@case("memory_ignores_load_spike")
def _ignores_spike():
    res = run_child("spike")
    env = environment_caused(res)
    if env:
        return True, env
    bad = child_failed(res)
    if bad:
        return False, bad
    delta = res["peak_rss_delta_bytes"]
    return delta < 16 * MiB, f"128 MiB freed before the phase, an allocation-free forward: delta {delta / MiB:.2f} MiB"


@case("ru_maxrss_recorded")
def _ru_maxrss():
    res = run_child("forward")
    env = environment_caused(res)
    if env:
        return True, env
    bad = child_failed(res)
    if bad:
        return False, bad
    start = res["process_start"] or {}
    ok = (res["ru_maxrss_phase_start_bytes"] > 0 and res["ru_maxrss_end_bytes"] >= res["ru_maxrss_phase_start_bytes"]
          and isinstance(res["ru_maxrss_consistent"], bool)
          and all(isinstance(start.get(k), int) and start[k] > 0 for k in ("ru_maxrss_bytes", "vm_hwm_bytes",
                                                                             "vm_rss_bytes"))
          and res["parent_contamination"]["ru_maxrss_start_bytes"] == start["ru_maxrss_bytes"])
    return ok, (f"ru_maxrss phase start {res['ru_maxrss_phase_start_bytes'] / MiB:.0f} MiB, end "
                f"{res['ru_maxrss_end_bytes'] / MiB:.0f} MiB, consistent {res['ru_maxrss_consistent']}; process "
                f"start {start}")


@case("parent_contamination_detected")
def _parent():
    hwm = 200 * MiB
    at = EFF.parent_contamination(hwm + 64 * MiB, hwm)
    over = EFF.parent_contamination(hwm + 64 * MiB + 1, hwm)
    clean = EFF.parent_contamination(hwm, hwm)
    ok = (at["contaminated"] is False and over["contaminated"] is True and clean["contaminated"] is False
          and over["threshold_bytes"] == 64 * MiB and over["excess_bytes"] == 64 * MiB + 1)
    return ok, "excess 64 MiB: not contaminated; 64 MiB + 1 B: contaminated"


@case("hwm_reset_unavailable_refused")
def _unavailable():
    absent = TMP / "nowhere_proc" / "clear_refs"        # its folder is never created; opened without O_CREAT
    a_dir = fresh("proc_dir")
    a_dir.mkdir(parents=True)
    got = {}
    for cause, kw in (("is absent", {"clear_refs": absent}), ("is not writable", {"clear_refs": a_dir}),
                      ("did not take", {"clear_refs": writable_clear_refs(),
                                        "status": write_status(100_000, 100_000 + 200 * 1024)})):
        try:
            EFF.reset_rss_high_water(slack_bytes=MiB, **kw)
            got[cause] = None
        except EFF.EfficiencyError as e:
            got[cause] = e.code if cause in str(e) else f"{e.code} (other text: {str(e)[:80]})"
    ok = set(got.values()) == {"memory_hwm_reset_unavailable"} and not os.path.lexists(absent.parent)
    return ok, str(got)


@case("hwm_slack_recorded")
def _slack():
    rules = EFF.hwm_slack_bytes(1, 4096) == MiB and EFF.hwm_slack_bytes(7, 4096) == 2 * MiB
    at = code_of(EFF.reset_rss_high_water, slack_bytes=MiB, clear_refs=writable_clear_refs(),
                 status=write_status(100_000, 100_000 + 1024))           # VmHWM = VmRSS + the slack: accepted
    over = code_of(EFF.reset_rss_high_water, slack_bytes=MiB, clear_refs=writable_clear_refs(),
                   status=write_status(100_000, 100_000 + 1025))         # one kB above it: refused
    res = run_child("forward")
    env = environment_caused(res)
    if env:
        recorded, detail = True, env
    elif child_failed(res):
        recorded, detail = False, child_failed(res)
    else:
        recorded = (res["slack_bytes"] == EFF.hwm_slack_bytes(res["threads"], res["page_size_bytes"])
                    == res["reset"]["slack_bytes"])
        detail = f"recorded slack {res['slack_bytes']} B at {res['threads']} threads"
    ok = rules and at is None and over == "memory_hwm_reset_unavailable" and recorded
    return ok, f"max(1 MiB, 64 pages x (threads + 1)); at the slack {at}, above it {over}; {detail}"


# ---------------------------------------------------------------- 6. paths and the record (§2 step 2; LF28)
@case("schema_2_0_0")
def _schema():
    code, path, rec = run_profile(e1_argv(fresh("schema_e1"), metrics="params,size,flops"))
    code_a, _pa, rec_a = run_profile(["--stage", "A", "--checkpoint", fixture("arm_a"), "--metrics", "memory",
                                      "--out-dir", fresh("schema_a")])
    if code is not None or code_a is not None:
        return False, f"refused: {code}, {code_a}"
    keys = {"schema", "measurement_role", "cpu_proxy", "on_device", "no_arm_claim", "of_record",
            "efficiency_analysis_member", "metric_group", "provenance", "parameters", "footprint", "footprint_source",
            "artifact", "compute", "latency", "memory", "deferred", "created_utc", "wall_clock_start",
            "wall_clock_end"}
    env = rec["provenance"]["environment"]
    ok = (rec["schema"] == rec_a["schema"] == "plantseg-efficiency/2.0.0" and keys <= set(rec)
          and rec["of_record"] is False and rec["efficiency_analysis_member"] is True
          and rec["metric_group"] == "static" and "_static_" in path.name and rec["footprint_source"] == "model"
          and rec_a["efficiency_analysis_member"] is False and rec_a["metric_group"] == "memory"
          and {"active_engine", "parallel_info"} <= set(env) and "expected_sha256" in rec["provenance"]
          and rec["wall_clock_start"] <= rec["wall_clock_end"] and rec["created_utc"].endswith("Z"))
    return ok, f"{path.name}; A member {rec_a['efficiency_analysis_member']}, group {rec_a['metric_group']}"



@case("forbidden_path_name_refused")
def _forbidden():
    probe = TMP / "nowhere" / "plantseg_TEST_probe"           # never created
    absent = TMP / "nowhere_ck" / "e1.pt"                     # never created: a mutant without the guard stops at
    link = fresh("link_probe")                                # the resolver and writes nothing
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(TMP / "nowhere" / "plantseg_TeSt_target.pt", link)   # resolved form only; the target is never created
    got = {
        "--out-dir": run_profile(e1_argv(probe / "out", ckpt=absent))[0],
        "--checkpoint": run_profile(e1_argv(fresh("fb_ck"), ckpt=probe / "e1.pt"))[0],
        "--checkpoint (resolved)": run_profile(e1_argv(fresh("fb_link"), ckpt=link))[0],
        "--provenance": run_profile(e4_argv(fresh("fb_prov"), prov=probe / "e4_ptq_run_meta.json"))[0],
        "--teacher-config": run_profile(["--stage", "teacher", "--checkpoint", fixture("teacher")[0],
                                         "--teacher-ckpt-sha256", fixture("teacher")[1], "--teacher-config",
                                         probe / "teacher_cfg.py", "--out-dir", fresh("fb_cfg")],
                                        teacher_builder=stub_builder)[0],
    }
    rc, err = main_rc(e1_argv(fresh("fb_main"), ckpt=probe / "e1.pt"))
    ok = (set(got.values()) == {"path_forbidden_name"} and rc == 2 and "[path_forbidden_name]" in err
          and not os.path.lexists(probe.parent) and not os.path.lexists(absent.parent))
    return ok, f"{got}; main exit {rc}"


@case("out_dir_not_empty_refused")
def _not_empty():
    full = fresh("full_out")
    full.mkdir(parents=True)
    (full / "earlier.json").write_text("{}", encoding="utf-8")
    as_file = fresh("file_out")
    as_file.parent.mkdir(parents=True, exist_ok=True)
    as_file.write_text("x", encoding="utf-8")
    code_dir = run_profile(e1_argv(full, metrics="params"))[0]
    code_file = run_profile(e1_argv(as_file, metrics="params"))[0]
    ok = (code_dir == code_file == "out_dir_not_empty" and os.listdir(full) == ["earlier.json"]
          and (full / "earlier.json").read_text(encoding="utf-8") == "{}")
    return ok, f"non-empty directory {code_dir}; a file {code_file}"


@case("out_dir_inside_repository_refused")
def _inside_repo():
    inside = REPO / "leff_b1_inside_probe"                   # never created
    absent = TMP / "nowhere_ck" / "e1.pt"                     # a mutant without the guard stops at the resolver
    code = run_profile(e1_argv(inside, ckpt=absent, metrics="params"))[0]
    rc, err = main_rc(e1_argv(inside, ckpt=absent, metrics="params"))
    ok = code == "out_dir_inside_repository" and rc == 2 and "[out_dir_inside_repository]" in err
    return ok and not os.path.lexists(inside), f"{code}; main exit {rc}"


# ---------------------------------------------------------------- 7. hygiene (always last)
BEFORE = None


@case("repo_state_unchanged")
def _repo():
    verdict = state_verdict(BEFORE, repo_state(REPO))
    return verdict == "unchanged", verdict


@case("no_forbidden_named_path_created")
def _no_forbidden():
    bad = []
    for root, dirs, files in os.walk(TMP):
        for name in dirs + files:
            if FORBIDDEN in name.lower() or ":" in name:
                bad.append(os.path.join(root, name))
    return not bad, f"{len(bad)} under {TMP}" + (f": {bad[:3]}" if bad else "")


def child_main(name: str) -> int:
    """`--child NAME`: one memory measurement in this fresh process, printed as one CHILD_JSON line."""
    try:
        out = CHILDREN[name]()
    except Exception as e:                                   # noqa: BLE001 - the parent judges it
        out = {"refused": getattr(e, "code", None) or type(e).__name__, "message": str(e)}
    print(CHILD_TAG + json.dumps(out, default=str), flush=True)
    return 0


def main(argv=None) -> int:
    global BEFORE, TMP
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated check names (the mutation runner's)")
    ap.add_argument("--child", default=None, choices=sorted(CHILDREN), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        return child_main(args.child)
    only = None if args.only is None else [n for n in args.only.split(",") if n]
    unknown = sorted(set(only or ()) - set(CASES))
    if unknown:
        print(f"unknown checks {unknown}")
        return 2
    TMP = safe_tmpdir("leff_b1_")
    BEFORE = repo_state(REPO)
    threads = torch.get_num_threads()
    print("=" * 78)
    print("LEFF B-1 smoke: synthetic only; no dataset, no checkpoint of record, no GPU, no install")
    print(f"torch {torch.__version__} | engines {list(torch.backends.quantized.supported_engines)} | temp {TMP}")
    print("=" * 78)
    t0 = time.time()
    for name, fn in CASES.items():
        if only is not None and name not in only:
            continue
        try:
            ok, detail = fn()
        except Exception as e:                               # noqa: BLE001 - a case that raises fails
            ok, detail = False, f"raised {type(e).__name__}: {str(e)[:300]}"
        finally:
            torch.set_num_threads(threads)
        check(name, ok, detail)
        print(f"  {name:46}: {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n({time.time() - t0:.0f} s)")
    print(f"RESULT: {'PASS' if passed == len(RESULTS) else 'FAIL'} ({passed}/{len(RESULTS)})")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
