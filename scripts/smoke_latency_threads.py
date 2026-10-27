#!/usr/bin/env python3
"""Synthetic smoke of LEFF B-2: L-LAT-THREADS, the run-of-record mode, the x86-copy binding and the table builder.

The LEFF Phase A plan §9's B-2 smoke as the supplement's D28-D31 and the LEFF-B1 GO (LF30-LF35) amend it: the thread
policy (THREADS_OF_RECORD; the Timer at the requested count with its probe in setup, untimed; inter-op set first in
main(), proven in fresh child processes), every refusal of the run of record reached with every earlier gate passing,
the x86 latency copy bound through a real scripts/run_ptq.py run record (E4, in its own process, as run_ptq requires)
and through a hand-built qat_finalize.json (E5), the host facts and their type fingerprint, and
scripts/efficiency_table.py over synthetic sittings.

Synthetic only (DL-41): random-weight students, a stub teacher, a synthetic TRAIN tree for the PTQ run (no TEST folder
is ever created), a synthetic git checkout for the governed-path gate; the KD image's PL-26 fingerprint is injected
(this venv is not the image; LB-0 runs this smoke in it). Temporary roots come from synthetic_ptq_fixtures.safe_tmpdir;
no created file or folder names "test" in any case or holds ":"; the repository is unchanged; the thread counts are
restored after each check; no_forbidden_named_path_created is the last check.

    python -B scripts/smoke_latency_threads.py [--only CHECK[,CHECK...]]

Prints one line per check and `RESULT: PASS (n/n)`; exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import signal
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

import scripts.efficiency_table as EFT  # noqa: E402
import scripts.profile_efficiency as PE  # noqa: E402
from scripts.synthetic_ptq_fixtures import (make_e1_checkpoint, make_tree, repo_state, safe_tmpdir,  # noqa: E402
                                            state_verdict)
from src.eval import efficiency as EFF  # noqa: E402
from src.eval import model_loading as ML  # noqa: E402

NC = 116
TMP: Path = None
FORBIDDEN = "test"
CHILD_TAG = "CHILD_JSON:"
HOST_LABEL = "smoke-host"
SMALL = (1, 3, 64, 64)            # the policy cases' latency input: batch 1 (only the FLOP count fixes 512x512)
RESULTS: list[tuple[str, bool, str]] = []
CASES: dict[str, object] = {}
FIX: dict[str, object] = {}
INTEROP: dict = {}
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


@contextlib.contextmanager
def env(**values):
    """Set (a string) or unset (None) environment variables for the block, and restore them."""
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Counter:
    """Wraps a callable and counts its calls."""

    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, *a, **kw):
        self.n += 1
        return self.fn(*a, **kw)


def run_profile(argv, **kw):
    """(code, out_path, record): the refusal code, or None with the written record."""
    args = PE.build_parser().parse_args([str(a) for a in argv])
    kw.setdefault("interop", INTEROP)
    try:
        out_path, record = PE.run(args, **kw)
    except Exception as e:                                   # noqa: BLE001
        return (getattr(e, "code", None) or type(e).__name__), None, None
    return None, out_path, record


def refused_text(argv, **kw) -> tuple:
    """(code, message) of a refusal; (None, "") when the run wrote its record."""
    args = PE.build_parser().parse_args([str(a) for a in argv])
    kw.setdefault("interop", INTEROP)
    try:
        PE.run(args, **kw)
    except Exception as e:                                   # noqa: BLE001
        return getattr(e, "code", None) or type(e).__name__, str(e)
    return None, ""


# ---------------------------------------------------------------- fixtures (synthetic, built once)
class TinyNet(nn.Module):
    def __init__(self, ch: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(3, ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class Counting(nn.Module):
    """Counts its forwards."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.calls = inner, 0

    def forward(self, x):
        self.calls += 1
        return self.inner(x)


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


def write_fp32(path: Path, *, stage=None, seed=7, extra=None) -> Path:
    """A random-weight student in the trainers' schema."""
    from src.models.student import build_student
    torch.manual_seed(seed)
    state = {k: v.detach().clone() for k, v in build_student(pretrained=False).state_dict().items()}
    payload = {"iter": 80000, "model_state_dict": state, "optimizer_state_dict": {}, "num_classes": NC}
    if stage is not None:
        payload["stage"] = stage
    payload.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def write_teacher(path: Path, seed=3) -> Path:
    """scripts/smoke_eval_teacher.py's synthetic teacher file."""
    g = torch.Generator().manual_seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": {"mmseg_version": "1.2.2"},
                "state_dict": {"backbone.projs.0.weight": torch.randn(4, 3, 1, 1, generator=g),
                               "decode_head.conv_seg.weight": torch.randn(NC, 8, 1, 1, generator=g),
                               "decode_head.conv_seg.bias": torch.randn(NC, generator=g)}}, path)
    return path


def ptq_run() -> Path:
    """One real unbound scripts/run_ptq.py PTQ (src/quant/ptq.run_ptq) of a random E1 student on a synthetic TRAIN
    tree with build_calibration_lists' synthetic list, in its own process (run_ptq refuses a process that already
    traced the student): its run record names the x86 latency copy (E4). Built once."""
    if "e4" not in FIX:
        from scripts.build_calibration_lists import FILENAME, main as build_lists
        root, lists = TMP / "data", TMP / "lists"
        make_tree(root, n_train=140, n_val=0, seed=2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = build_lists(["--data-root", str(root), "--out-dir", str(lists), "--expected-images", "140",
                              "--generated-utc", "2026-10-01T00:00:00Z"])
        if rc != 0:
            raise RuntimeError(f"build_calibration_lists exit {rc}")
        lst = lists / FILENAME.format(seed=42)
        (TMP / "fixtures").mkdir(parents=True, exist_ok=True)
        e1 = make_e1_checkpoint(TMP / "fixtures" / "e1_src.pt", seed=5)
        out = TMP / "fixtures" / "e4_run"
        argv = ["--real-run", "--confirm-real-run", "--stage", "E4", "--source-ckpt", e1,
                "--expect-source-sha256", sha256_path(e1), "--calibration-list", lst,
                "--expect-list-checksum", json.loads(lst.read_text(encoding="utf-8"))["checksum_sha256"],
                "--data-root", root, "--out-dir", out]
        child_env = {k: v for k, v in os.environ.items() if not k.startswith("PLANTSEG_")}
        p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "run_ptq.py"), *map(str, argv)],
                           cwd=str(REPO), env={**child_env, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True,
                           text=True, timeout=1800)
        meta = out / "e4_ptq_run_meta.json"
        if p.returncode != 0 or not meta.is_file():
            raise RuntimeError(f"scripts/run_ptq.py exit {p.returncode}: {(p.stdout + p.stderr)[-300:]}")
        FIX["e4"] = meta
    return FIX["e4"]


def x86_torchscript(stage: str, method: str, *, seed: int = 0) -> bytes:
    """A synthetic x86 latency copy: a random student converted under the x86 engine (src/quant/x86_latency's PTQ
    path), traced with its embedded identity (artifact_role x86_cpu_proxy_latency)."""
    from src.models.student import build_student
    from src.quant.ptq import torchscript_bytes
    from src.quant.x86_latency import build_x86_latency_copy
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    copy_ = build_x86_latency_copy("E4", build_student(pretrained=False).eval(),
                                   calibration_batches=[torch.randn(1, 3, 64, 64, generator=g) for _ in range(2)])
    meta = {"schema": ML.INT8_TORCHSCRIPT_SCHEMA, "stage": stage, "quantization": method, "num_classes": NC,
            "engine": torch.backends.quantized.engine, "artifact_role": EFF.ARTIFACT_ROLE_X86_LATENCY}
    raw, _ = torchscript_bytes(copy_.eval(), meta)
    return raw


def qnnpack_torchscript(stage: str, method: str, *, seed: int = 0) -> tuple[bytes, bytes]:
    """A synthetic QNNPACK artifact (TorchScript with its embedded identity) and its state_dict companion."""
    from src.models.student import build_student
    from src.quant.prepare import calibrate, convert_model, prepare_ptq
    from src.quant.ptq import state_dict_bytes, torchscript_bytes
    torch.backends.quantized.engine = "qnnpack"
    torch.manual_seed(seed)
    prepared = prepare_ptq(build_student(pretrained=False).eval(), select_backend=False)
    g = torch.Generator().manual_seed(seed + 1)
    calibrate(prepared, [torch.randn(1, 3, 64, 64, generator=g) for _ in range(2)])
    converted = convert_model(prepared).eval()
    meta = {"schema": ML.INT8_TORCHSCRIPT_SCHEMA, "stage": stage, "quantization": method, "num_classes": NC,
            "engine": "qnnpack", "artifact_role": "accuracy"}
    ts, _ = torchscript_bytes(converted, meta)
    return ts, state_dict_bytes(converted, stage, method)


def e5_eval_dir() -> dict:
    """A hand-built E5 eval directory: converted/e07's QNNPACK TorchScript, its state_dict and run provenance, the x86
    latency copy under final/, and a qat_finalize/1 record with scripts/qat_epoch_eval.py finalize's fields."""
    if "e5" not in FIX:
        ev = TMP / "fixtures" / "e5_eval"
        (ev / "converted").mkdir(parents=True)
        (ev / "final").mkdir(parents=True)
        ts, sd = qnnpack_torchscript("E5", "qat", seed=3)
        (ev / "converted" / "e07_int8_qnnpack.torchscript.pt").write_bytes(ts)
        (ev / "converted" / "e07_int8_student.pt").write_bytes(sd)
        prov = {"stage": "E5", "source_stage": "E1", "quantization_method": "qat", "random_init": False,
                "num_classes": NC, "backend": "qnnpack", "artifact_format": "torchscript",
                "converted_artifact": "e07_int8_qnnpack.torchscript.pt", "converted_artifact_sha256": sha256_bytes(ts),
                "state_dict_artifact": {"path": "e07_int8_student.pt", "sha256": sha256_bytes(sd)},
                "synthetic": "LEFF B-2 smoke fixture: random weights, never of record"}
        prov_path = ev / "converted" / "e07_run_meta.json"
        prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
        x86 = x86_torchscript("E5", "qat", seed=4)
        (ev / "final" / "e07_int8_x86_latency.torchscript.pt").write_bytes(x86)
        fin = {"format": "qat_finalize/1", "stage": "E5", "run_id": "synthetic", "seed": 42, "status": "finalized",
               "artifact_of_record": {
                   "epoch": 7, "torchscript": {"path": "converted/e07_int8_qnnpack.torchscript.pt",
                                               "sha256": sha256_bytes(ts)},
                   "provenance": {"path": "converted/e07_run_meta.json", "sha256": sha256_path(prov_path)},
                   "state_dict_companion": {"path": "converted/e07_int8_student.pt", "sha256": sha256_bytes(sd)}},
               "x86_latency_copy": {
                   "path": "final/e07_int8_x86_latency.torchscript.pt", "sha256": sha256_bytes(x86),
                   "engine": torch.backends.quantized.engine, "usable_for_accuracy": False,
                   "provenance": {"artifact_role": EFF.ARTIFACT_ROLE_X86_LATENCY, "reduce_range": True,
                                  "usable_for_accuracy": False, "trained_with_x86_qat": False},
                   "checks": {"zero_optimizer_steps": True, "parity_x86_copy": True, "x86_copy_float_boundary": True,
                              "x86_weights_per_channel_int8": True, "file_matches_serialization": True}}}
        (ev / "qat_finalize.json").write_text(json.dumps(fin, indent=2), encoding="utf-8")
        FIX["e5"] = {"dir": ev, "prov": prov_path, "finalize": ev / "qat_finalize.json"}
    return FIX["e5"]


def copy_tree(src: Path, tag: str) -> Path:
    """A copy of a fixture directory (so a case may edit it; the original stays intact)."""
    dst = fresh(tag)
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            (dst / rel / name).write_bytes((Path(root) / name).read_bytes())
    return dst


def make_checkout(root: Path) -> Path:
    """A synthetic git checkout with one governed file committed: the governed-path gate's clean worktree. No
    template and no hooks."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "module_a.py").write_text("X = 1\n", encoding="utf-8")
    g = ["git", "-C", str(root), "-c", "user.name=leff-smoke", "-c", "user.email=leff-smoke@invalid",
         "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    for argv in (["init", "-q", "--template="], ["add", "--", "src/module_a.py"], ["commit", "-q", "-m", "synthetic"]):
        subprocess.run(g + argv, check=True, capture_output=True, timeout=60)
    return root


def fixture(name: str):
    if name not in FIX:
        base = TMP / "fixtures"
        if name == "e1":
            FIX[name] = write_fp32(base / "e1_best.pt")
        elif name in ("arm_a", "arm_f", "arm_g"):
            arm = name[-1].upper()
            FIX[name] = write_fp32(base / f"{arm.lower()}_best.pt", stage=arm, seed=9)
        elif name == "e3_arm":
            FIX[name] = write_fp32(base / "e3_arm_best.pt", stage="E3", seed=13,
                                   extra={"arm": "E3-R1", "target_construction": {"K": 1, "views": 1}})
        elif name == "teacher":
            p = write_teacher(base / "teacher_iter.pth")
            FIX[name] = (p, sha256_path(p))
        elif name == "git_clean":
            FIX[name] = make_checkout(base / "checkout_clean")
        else:
            raise KeyError(name)
    return FIX[name]


def fingerprint_of_record() -> dict:
    """A PL-26 fingerprint that preflight_distill.fingerprint_problems accepts (the KD image's values of record),
    injected: this venv is not the image."""
    import scripts.preflight_distill as PD
    fp = dict(PD.FINGERPRINT_OF_RECORD)
    fp["imported"] = {"torch": ["torch"], "numpy": ["numpy"], "PIL": ["pillow"]}
    fp["import mmseg.apis"] = True
    fp["import ftfy"] = True
    return fp


@contextlib.contextmanager
def of_record_world(**over):
    """The environment a run of record passes in: the KD image's digest, its fingerprint (injected), a clean governed
    checkout (synthetic), no thread variables. Each keyword replaces one part."""
    import scripts.preflight_distill as PD
    fp = over.pop("fingerprint", None) or fingerprint_of_record()
    root = over.pop("governed_root", None) or fixture("git_clean")
    digest = over.pop("digest", PD.KD_IMAGE_DIGEST)
    omp = over.pop("omp", None)
    with env(PLANTSEG_IMAGE_DIGEST=digest, OMP_NUM_THREADS=omp, MKL_NUM_THREADS=None), \
            patched(PE, "image_fingerprint", lambda: (0, fp, "")), patched(PE, "GOVERNED_ROOT", root):
        yield


def e1_record_argv(out, *, metrics="params,size,flops", threads="1", h="__file__", label=HOST_LABEL,
                   ckpt=None) -> list:
    """An E1 run of record (the static group unless named)."""
    ck = ckpt or fixture("e1")
    argv = ["--of-record", "--stage", "E1", "--checkpoint", ck, "--metrics", metrics, "--out-dir", out]
    if h is not None:
        argv += ["--expected-checkpoint-sha256", sha256_path(ck) if h == "__file__" else h]
    if threads is not None:
        argv += ["--threads", threads]
    if label is not None:
        argv += ["--host-label", label]
    return argv


def e4_argv(out, *, metrics="latency", role="x86_cpu_proxy_latency", of_record=False, prov=None, h="__file__",
            h2="__resolved__", h86="__recorded__", threads="1") -> list:
    """An E4 run handed the PTQ run record (the x86 copy's latency unless named)."""
    meta = prov or ptq_run()
    doc = json.loads(Path(meta).read_text(encoding="utf-8"))
    argv = ["--stage", "E4", "--provenance", meta, "--metrics", metrics, "--artifact-role", role, "--out-dir", out]
    if of_record:
        argv += ["--of-record", "--host-label", HOST_LABEL]
    if h is not None:
        argv += ["--expected-checkpoint-sha256", sha256_path(meta) if h == "__file__" else h]
    if h2 is not None:
        argv += ["--expected-converted-sha256", doc["converted_artifact_sha256"] if h2 == "__resolved__" else h2]
    if h86 is not None and role == EFF.ARTIFACT_ROLE_X86_LATENCY:
        argv += ["--expected-x86-sha256", (doc.get("x86_latency_copy") or {}).get("sha256", "0" * 64)
                 if h86 == "__recorded__" else h86]
    if threads is not None:
        argv += ["--threads", threads]
    return argv


def e5_argv(out, *, ev=None, metrics="latency", finalize="__default__", h86="__recorded__") -> list:
    """An E5 run of the x86 copy, bound through its eval directory's qat_finalize.json."""
    ev = ev or e5_eval_dir()
    fin = json.loads(Path(ev["finalize"]).read_text(encoding="utf-8"))
    argv = ["--stage", "E5", "--provenance", ev["prov"], "--metrics", metrics, "--artifact-role",
            "x86_cpu_proxy_latency", "--threads", "1", "--out-dir", out,
            "--expected-checkpoint-sha256", sha256_path(ev["prov"])]
    if finalize is not None:
        argv += ["--qat-finalize", ev["finalize"] if finalize == "__default__" else finalize]
    if h86 is not None:
        argv += ["--expected-x86-sha256", fin["x86_latency_copy"]["sha256"] if h86 == "__recorded__" else h86]
    return argv


# ---------------------------------------------------------------- 1. the thread policy (LF30)
@case("threads_of_record_constants")
def _constants():
    src = (REPO / "src" / "eval" / "efficiency.py").read_text(encoding="utf-8")
    cited = "LEFF-AUDIT-RULE-1 Q1; its DL row is proposed in docs/lane_reports/leff.md" in src
    ok = EFF.THREADS_OF_RECORD == 1 and EFF.INTEROP_THREADS_OF_RECORD == 1 and cited
    return ok, (f"THREADS_OF_RECORD {EFF.THREADS_OF_RECORD}, INTEROP_THREADS_OF_RECORD "
                f"{EFF.INTEROP_THREADS_OF_RECORD}; ruling cited {cited}")


@case("timer_runs_at_requested_threads")
def _timer_threads():
    lat = EFF.measure_latency(TinyNet(), input_shape=SMALL, thread_count=2)
    probe = lat["thread_probe"]
    ok = lat["intra_op_observed"] == 2 and probe["values"] == [2] and probe["probes"] >= 100 and \
        lat["protocol"]["num_threads"] == 2
    return ok, f"probe {probe}; observed {lat['intra_op_observed']} (F2: the Timer's default would be 1)"


@case("observed_threads_recorded")
def _observed():
    from torch.utils.benchmark.utils import common

    real = common.set_torch_threads

    @contextlib.contextmanager
    def three(_n):
        with real(3):
            yield
    with patched(common, "set_torch_threads", three):
        lat = EFF.measure_latency(TinyNet(), input_shape=SMALL, thread_count=2)
    ok = lat["intra_op_observed"] == 3 and lat["thread_count"] == 2
    return ok, f"requested 2, the Timer ran at 3: recorded intra_op_observed {lat['intra_op_observed']}"


@case("timed_threads_mismatch")
def _timed_mismatch():
    from torch.utils.benchmark.utils import common

    real = common.set_torch_threads

    @contextlib.contextmanager
    def two(_n):
        with real(2):
            yield
    out = fresh("timed_mm")
    with of_record_world(), patched(common, "set_torch_threads", two):
        code, msg = refused_text(e1_record_argv(out, metrics="latency"))
    return code == "timed_threads_mismatch" and not os.path.lexists(out), f"{code}: {msg[:120]}"


@case("untimed_executions_counted")
def _untimed():
    net = Counting(TinyNet())
    lat = EFF.measure_latency(net, input_shape=SMALL, thread_count=1)
    u = lat["protocol"]["untimed_executions_per_observation"]
    ok = isinstance(u, int) and u >= 0 and net.calls == 20 + 100 * (1 + u) == lat["protocol"]["forwards_executed"]
    return ok, f"untimed executions per observation {u} (measured); forwards executed {net.calls}"


def run_child(name: str, *argv) -> dict:
    p = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--child", name, *map(str, argv)],
                       cwd=str(REPO), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True,
                       text=True, timeout=900)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(CHILD_TAG)]
    return json.loads(lines[-1][len(CHILD_TAG):]) if lines else {"error": f"exit {p.returncode}: {p.stderr[-300:]}"}


def run_raw_child(name: str, *argv) -> subprocess.CompletedProcess:
    """A fresh child whose exit status is the evidence (it may end on a signal): its raw process result."""
    return subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--child", name, *map(str, argv)],
                          cwd=str(REPO), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True,
                          text=True, timeout=300)


def _forked_neg(x: torch.Tensor) -> torch.Tensor:
    """Scripted, its torch.jit.fork runs on the inter-op pool (at::launch): inter-op work."""
    return torch.jit.wait(torch.jit.fork(torch.neg, x))


@case("interop_set_at_start")
def _interop_start():
    res = run_child("interop_start", fixture("e1"), fresh("interop_start"))
    inter = (res.get("thread_policy") or {}).get("inter_op") or {}
    ok = res.get("rc") == 0 and inter.get("status") == "set" and inter.get("observed") == 1
    return ok, f"fresh process: main() exit {res.get('rc')}, inter-op {inter}" + (f" {res['error']}" if
                                                                                  "error" in res else "")


@case("interop_threads_locked_by_injection")
def _interop_locked():
    out = fresh("interop_locked")
    res = run_child("interop_locked", fixture("e1"), out, fixture("git_clean"))
    ok = res.get("rc") == 2 and "[interop_threads_locked]" in res.get("stderr", "") and not os.path.lexists(out)
    return ok, (f"a torch whose set_num_interop_threads raises (injected), then a run of record: exit "
                f"{res.get('rc')}; {res.get('stderr', res.get('error', ''))[-140:]}")


@case("interop_second_call_refused")
def _interop_twice():
    first, second = fresh("interop_first"), fresh("interop_second")
    res = run_child("interop_twice", fixture("e1"), first, second, fixture("git_clean"))
    err = res.get("stderr", res.get("error", ""))
    ok = (res.get("first_rc") == 0 and (res.get("first_inter_op") or {}).get("status") == "set"
          and res.get("rc") == 2 and "[interop_threads_locked]" in err and "already called in this process" in err
          and not os.path.lexists(second))
    return ok, (f"main() twice in one process: exit {res.get('first_rc')} (inter-op "
                f"{(res.get('first_inter_op') or {}).get('status')}), then exit {res.get('rc')} without a second torch "
                f"call; {err[-120:]}")


@case("torch_second_interop_call_aborts")
def _torch_twice():
    p = run_raw_child("torch_twice")
    # the first call returned (its marker), the second ended the process before its own marker
    aborted = (p.returncode == -signal.SIGABRT and f'{CHILD_TAG}{{"first": "returned"}}' in p.stdout
               and '"second"' not in p.stdout)
    tail = [ln for ln in p.stderr.splitlines() if "interop" in ln][:1]
    return aborted, (f"torch {torch.__version__}: the first torch.set_num_interop_threads returned, the second ended "
                     f"the process (exit {p.returncode}, SIGABRT), where the plan inferred a RuntimeError; {tail}")


@case("torch_interop_call_after_work_aborts")
def _torch_after_work():
    p = run_raw_child("torch_after_work")
    # a scripted fork ran on the inter-op pool (its marker), then a first set_num_interop_threads ended the process
    aborted = (p.returncode == -signal.SIGABRT and f'{CHILD_TAG}{{"work": "done"}}' in p.stdout
               and '"set"' not in p.stdout)
    return aborted, (f"torch {torch.__version__}: after inter-op work (a scripted torch.jit.fork), a first "
                     f"torch.set_num_interop_threads ended the process (exit {p.returncode}, SIGABRT)")


@case("interop_threads_not_of_record")
def _interop_not():
    out = fresh("interop_not")
    with of_record_world(), patched(torch, "get_num_interop_threads", lambda: 2):
        code, msg = refused_text(e1_record_argv(out))
    return code == "interop_threads_not_of_record" and not os.path.lexists(out), f"{code}: {msg[:120]}"


@case("timer_probe_in_setup_untimed")
def _probe_untimed():
    real = EFF._thread_probe

    def slow(store):
        time.sleep(0.05)
        real(store)
    with patched(EFF, "_thread_probe", slow):
        lat = EFF.measure_latency(TinyNet(), input_shape=SMALL, thread_count=1)
    ok = lat["latency_median_ms"] < 25.0 and lat["thread_probe"]["values"] == [1]
    return ok, f"a 50 ms probe in the Timer's setup; median {lat['latency_median_ms']:.2f} ms (untimed)"


@case("threads_not_applied_by_injection")
def _not_applied():
    with patched(torch, "get_num_threads", lambda: 7):
        code, msg = refused_text(["--stage", "E1", "--checkpoint", fixture("e1"), "--metrics", "params",
                                  "--threads", "1", "--out-dir", fresh("not_applied")])
    return code == "threads_not_applied", f"{code}: {msg[:120]}"


@case("threads_invalid")
def _invalid():
    code, msg = refused_text(["--stage", "E1", "--checkpoint", fixture("e1"), "--metrics", "params", "--threads", "0",
                              "--out-dir", fresh("threads_zero")])
    return code == "threads_invalid", f"{code}: {msg[:120]}"


# ---------------------------------------------------------------- 2. refusals of the run of record (§1, §2; LF32)
def _refused(argv, want, *, world=None, **kw):
    """The run refuses `want`, with every gate before it passing, and leaves --out-dir uncreated."""
    out = Path(argv[argv.index("--out-dir") + 1])
    with (world if world is not None else of_record_world()):
        code, msg = refused_text(argv, **kw)
    return code == want and not os.path.lexists(out), f"{code}: {msg[:150]}"


@case("threads_required_of_record")
def _threads_required():
    return _refused(e1_record_argv(fresh("thr_req"), threads=None), "threads_required_of_record")


@case("threads_not_of_record")
def _threads_not():
    return _refused(e1_record_argv(fresh("thr_not"), threads="2"), "threads_not_of_record")


@case("thread_env_conflict")
def _env_conflict():
    return _refused(e1_record_argv(fresh("env_conf")), "thread_env_conflict", world=of_record_world(omp="4"))


def _host_injected(**over):
    facts = dict(EFF.host_facts())
    facts.update(over)
    return patched(PE, "host_facts", lambda: facts)


@case("affinity_below_threads_by_injection")
def _affinity():
    with _host_injected(affinity_cpus=0):
        return _refused(e1_record_argv(fresh("affinity")), "affinity_below_threads")


@case("cpu_quota_below_threads")
def _quota():
    sysfs = fresh("sysfs") / "sys"
    (sysfs / "fs" / "cgroup").mkdir(parents=True)
    (sysfs / "fs" / "cgroup" / "cpu.max").write_text("50000 100000\n", encoding="ascii")
    parsed = EFF.cgroup_cpu_quota(str(sysfs), str(fresh("no_proc")))      # no /proc/self/cgroup: the mount roots
    with _host_injected(cgroup_cpu_quota=parsed):
        ok, detail = _refused(e1_record_argv(fresh("quota")), "cpu_quota_below_threads")
    return ok and parsed == 0.5, f"cpu.max '50000 100000' -> {parsed} CPUs; {detail}"


@case("host_below_threads_by_injection")
def _cores():
    with _host_injected(physical_cores=0):
        return _refused(e1_record_argv(fresh("cores")), "host_below_threads")


@case("host_not_x86")
def _not_x86():
    with _host_injected(machine="aarch64", vendor="ARM"):
        return _refused(e1_record_argv(fresh("arm_host")), "host_not_x86")


@case("host_label_required")
def _label():
    return _refused(e1_record_argv(fresh("label"), label=None), "host_label_required")


@case("stage_not_in_efficiency_analysis")
def _stage_out():
    got = []
    for name in ("arm_a", "arm_f", "arm_g"):
        ck = fixture(name)
        argv = ["--of-record", "--stage", name[-1].upper(), "--checkpoint", ck, "--metrics", "params,size,flops",
                "--threads", "1", "--host-label", HOST_LABEL, "--expected-checkpoint-sha256", sha256_path(ck),
                "--out-dir", fresh(f"stage_{name[-1]}")]
        got.append(_refused(argv, "stage_not_in_efficiency_analysis"))
    return all(ok for ok, _ in got), "; ".join(f"{s}: {d[:60]}" for s, (_ok, d) in zip("AFG", got))


@case("expected_checkpoint_sha256_required")
def _ecs_required():
    return _refused(e1_record_argv(fresh("ecs_req"), h=None), "expected_checkpoint_sha256_required")


@case("expected_checkpoint_sha256_format")
def _ecs_format():
    return _refused(e1_record_argv(fresh("ecs_fmt"), h=sha256_path(fixture("e1")).upper()),
                    "expected_checkpoint_sha256_format")


@case("expected_checkpoint_sha256_mismatch")
def _ecs_mismatch():
    loads = Counter(torch.load)
    with patched(torch, "load", loads):
        ok, detail = _refused(e1_record_argv(fresh("ecs_mm"), h="0" * 64), "expected_checkpoint_sha256_mismatch")
    return ok and loads.n == 0, f"{detail}; torch.load calls {loads.n}"


@case("source_changed_during_validation")
def _changed():
    ck = fresh("changed").with_suffix(".pt")
    write_fp32(ck, seed=31)
    argv = e1_record_argv(fresh("changed_out"), ckpt=ck)          # its sha256 is taken here, before the change
    real = PE.resolve_evaluation_source

    def change_then_resolve(*a, **kw):
        write_fp32(ck, seed=32)          # the handed file changes after its first hash, before the resolver reads it
        return real(*a, **kw)
    with patched(PE, "resolve_evaluation_source", change_then_resolve):
        return _refused(argv, "source_changed_during_validation")


@case("expected_checkpoint_sha256_teacher_disagrees")
def _teacher_disagrees():
    path, digest = fixture("teacher")
    argv = ["--stage", "teacher", "--checkpoint", path, "--teacher-ckpt-sha256", digest, "--teacher-config",
            REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py",
            "--expected-checkpoint-sha256", "0" * 64, "--metrics", "params", "--out-dir", fresh("t_disagree")]
    return _refused(argv, "expected_checkpoint_sha256_teacher_disagrees", teacher_builder=stub_builder)


@case("expected_converted_sha256_required")
def _ecv_required():
    return _refused(e4_argv(fresh("ecv_req"), metrics="params,size,flops", role="accuracy", of_record=True, h2=None),
                    "expected_converted_sha256_required")


@case("expected_converted_sha256_format")
def _ecv_format():
    return _refused(e4_argv(fresh("ecv_fmt"), metrics="params", role="accuracy", h2="xyz"),
                    "expected_converted_sha256_format")


@case("expected_converted_sha256_not_int8")
def _ecv_not_int8():
    argv = ["--stage", "E1", "--checkpoint", fixture("e1"), "--metrics", "params", "--expected-converted-sha256",
            "0" * 64, "--out-dir", fresh("ecv_fp32")]
    return _refused(argv, "expected_converted_sha256_not_int8")


@case("expected_converted_sha256_mismatch")
def _ecv_mismatch():
    return _refused(e4_argv(fresh("ecv_mm"), metrics="params", role="accuracy", h2="0" * 64),
                    "expected_converted_sha256_mismatch")


@case("expected_x86_sha256_required")
def _x86_required():
    return _refused(e4_argv(fresh("x86_req"), of_record=True, h86=None), "expected_x86_sha256_required")


@case("expected_x86_sha256_format")
def _x86_format():
    return _refused(e4_argv(fresh("x86_fmt"), h86="ABC"), "expected_x86_sha256_format")


@case("expected_x86_sha256_not_applicable")
def _x86_na():
    argv = ["--stage", "E1", "--checkpoint", fixture("e1"), "--metrics", "params", "--expected-x86-sha256", "0" * 64,
            "--out-dir", fresh("x86_na")]
    return _refused(argv, "expected_x86_sha256_not_applicable")


@case("governed_paths_dirty")
def _dirty():
    root = make_checkout(fresh("checkout_dirty"))
    (root / "src" / "module_b.py").write_text("Y = 2\n", encoding="utf-8")      # untracked, governed
    return _refused(e1_record_argv(fresh("dirty_out")), "governed_paths_dirty",
                    world=of_record_world(governed_root=root))


@case("governed_paths_unavailable")
def _no_worktree():
    plain = fresh("no_git_here")
    plain.mkdir(parents=True)
    return _refused(e1_record_argv(fresh("nogit_out")), "governed_paths_unavailable",
                    world=of_record_world(governed_root=plain))


@case("governed_porcelain_recheck_by_injection")
def _porcelain_recheck():
    import src.eval.stage_artifacts as SA
    root = make_checkout(fresh("checkout_race"))
    (root / "src" / "module_c.py").write_text("Z = 3\n", encoding="utf-8")      # untracked, governed
    out = fresh("race_out")
    with of_record_world(governed_root=root), patched(SA, "governed_paths_error", lambda _repo=None: None):
        code, msg = refused_text(e1_record_argv(out))                         # the gate's own read saw it clean
    ok = code == "governed_paths_dirty" and "during the gate" in msg and not os.path.lexists(out)
    return ok, f"{code}: {msg[:140]}"


@case("image_not_of_record")
def _image():
    return _refused(e1_record_argv(fresh("image_out")), "image_not_of_record", world=of_record_world(digest=None))


@case("image_fingerprint_mismatch")
def _fingerprint():
    fp = {**fingerprint_of_record(), "python": "0.0.0"}
    return _refused(e1_record_argv(fresh("fp_out")), "image_fingerprint_mismatch",
                    world=of_record_world(fingerprint=fp))


@case("image_fingerprint_child_failure_coded")
def _fingerprint_child_failure():
    def times_out():
        raise subprocess.TimeoutExpired(cmd="preflight_distill.py _fingerprint", timeout=600)

    def garbled():
        return json.loads("{not a fingerprint")                       # a FINGERPRINT_JSON line that does not parse
    got = []
    for label, fake, cause in (("timeout", times_out, "TimeoutExpired"), ("garbled", garbled, "JSONDecodeError"),
                               ("exit 1", lambda: (1, None, "Traceback: boom"), "rc=1")):
        out = fresh("fp_child")
        with of_record_world(), patched(PE, "image_fingerprint", fake):
            code, msg = refused_text(e1_record_argv(out))
        got.append((code == "image_fingerprint_mismatch" and cause in msg and not os.path.lexists(out),
                    f"{label}: {code}"))
    return all(ok for ok, _ in got), "; ".join(d for _ok, d in got)


@case("fvcore_not_pinned")
def _fvcore():
    with patched(PE, "fvcore_version", lambda: "0.1.4"):
        return _refused(e1_record_argv(fresh("fvcore_out")), "fvcore_not_pinned")


@case("metric_group_not_of_record")
def _group():
    return _refused(e1_record_argv(fresh("group_out"), metrics="params,latency"), "metric_group_not_of_record")


@case("metric_deferred_of_record")
def _deferred():
    def unavailable(model, inputs):
        raise EFF.EfficiencyError("fvcore_unavailable", "injected: the counter is unavailable")
    with patched(EFF, "_fvcore_counter", unavailable):
        return _refused(e1_record_argv(fresh("deferred_out")), "metric_deferred_of_record")


@case("memory_parent_contaminated")
def _parent():
    start = {"ru_maxrss_bytes": 900 * 1024 * 1024, "vm_hwm_bytes": 100 * 1024 * 1024, "vm_rss_bytes": 90 * 1024 * 1024}
    with patched(PE, "PROCESS_START_MEMORY", start):
        return _refused(e1_record_argv(fresh("parent_out"), metrics="memory"), "memory_parent_contaminated")


@case("memory_start_unavailable_by_injection")
def _start_unavailable():
    with patched(PE, "PROCESS_START_MEMORY", None):
        return _refused(e1_record_argv(fresh("start_none_out"), metrics="memory"), "memory_sampler_unavailable")


@case("arm_artifact_not_of_record")
def _arm():
    ck = fixture("e3_arm")
    argv = ["--of-record", "--stage", "E3", "--checkpoint", ck, "--expected-checkpoint-sha256", sha256_path(ck),
            "--metrics", "params,size,flops", "--threads", "1", "--host-label", HOST_LABEL, "--out-dir", fresh("arm")]
    return _refused(argv, "arm_artifact_not_of_record")


@case("arm_key_read_rehash_refused")
def _arm_rehash():
    ck = fresh("arm_rehash").with_suffix(".pt")
    write_fp32(ck, seed=41)
    real = PE.checkpoint_keys
    loads = Counter(torch.load)

    def change_then_read(resolved):
        write_fp32(Path(resolved["checkpoint_path"]), seed=42)     # the bytes change after the resolver validated them
        before = loads.n
        with patched(torch, "load", loads):
            try:
                return real(resolved)
            finally:
                parsed.append(loads.n - before)
    parsed: list = []
    with patched(PE, "checkpoint_keys", change_then_read):
        ok, detail = _refused(e1_record_argv(fresh("arm_rehash_out"), ckpt=ck), "checkpoint_changed_after_validation")
    return ok and parsed == [0], f"{detail}; torch.load calls at the arm-key read {parsed}"


@case("fp32_has_no_x86_copy")
def _fp32_x86():
    argv = ["--stage", "E1", "--checkpoint", fixture("e1"), "--metrics", "latency", "--artifact-role",
            "x86_cpu_proxy_latency", "--out-dir", fresh("fp32_x86")]
    return _refused(argv, "fp32_has_no_x86_copy")


@case("int8_latency_needs_x86_copy")
def _int8_latency():
    return _refused(e4_argv(fresh("int8_lat"), metrics="latency", role="accuracy"), "int8_latency_needs_x86_copy")


@case("int8_static_needs_accuracy_artifact")
def _int8_static():
    return _refused(e4_argv(fresh("int8_static"), metrics="params"), "int8_static_needs_accuracy_artifact")


BINDING_KEYS = {"run_provenance_sha256", "qat_finalize_path", "qat_finalize_sha256", "x86_copy",
                "state_dict_companion_sha256", "int8_official"}
# §2's field list of a 2.0.0 record of record, block by block (LF28)
RECORD_KEYS = {"schema", "measurement_role", "cpu_proxy", "on_device", "no_arm_claim", "of_record",
               "efficiency_analysis_member", "metric_group", "thread_policy", "host", "code", "image", "provenance",
               "parameters", "footprint", "footprint_source", "artifact", "compute", "latency", "memory", "deferred",
               "created_utc", "wall_clock_start", "wall_clock_end"}
THREAD_POLICY_KEYS = {"intra_op_requested", "intra_op_observed", "inter_op", "constant", "inter_op_constant", "dl_row"}
CODE_KEYS = {"repo_commit", "governed_paths_clean", "worktree_state_sha256", "dirty_allowlisted", "files_sha256"}
IMAGE_KEYS = {"digest", "fingerprint", "fvcore"}
HOST_BLOCK_KEYS = {"label", "note", "facts", "fingerprint"}
PROTOCOL_KEYS = {"warmup", "measured", "untimed_executions_per_observation", "forwards_executed", "timer",
                 "num_threads"}
ENVIRONMENT_KEYS = {"torch", "torchvision", "python", "platform", "processor", "machine", "quant_backends_available",
                    "active_engine", "thread_count", "interop_thread_count", "parallel_info"}


@case("record_of_record_complete")
def _record_complete():
    """An E1 run of record in the world of record, its static group and then its latency group: every block of
    §2's field list (each block's key set, and the values the run of record fixes), nothing deferred."""
    import scripts.preflight_distill as PD
    with of_record_world():
        code_s, _ps, st = run_profile(e1_record_argv(fresh("rec_static")))
        code_l, _pl, la = run_profile(e1_record_argv(fresh("rec_latency"), metrics="latency"))
    if code_s is not None or code_l is not None:
        return False, f"refused: {code_s}, {code_l}"
    head = subprocess.run(["git", "-C", str(fixture("git_clean")), "rev-parse", "HEAD"], capture_output=True,
                          text=True, timeout=60).stdout.strip()
    tp, host, code, image, prov, lat = (st["thread_policy"], st["host"], st["code"], st["image"], st["provenance"],
                                        la["latency"])
    proto = lat["protocol"]
    keys = (set(st) == RECORD_KEYS == set(la) and set(tp) == THREAD_POLICY_KEYS == set(la["thread_policy"])
            and set(code) == CODE_KEYS == set(la["code"]) and set(image) == IMAGE_KEYS == set(la["image"])
            and set(host) == HOST_BLOCK_KEYS == set(la["host"]) and HOST_KEYS <= set(host["facts"])
            and set(proto) == PROTOCOL_KEYS and set(prov["environment"]) == ENVIRONMENT_KEYS
            == set(la["provenance"]["environment"]))
    ok = (keys and st["of_record"] is True and la["of_record"] is True and st["deferred"] == {} == la["deferred"]
          and st["efficiency_analysis_member"] is True and (st["metric_group"], la["metric_group"]) == (
              "static", "latency") and st["footprint_source"] is not None and bool(st["created_utc"])
          and st["wall_clock_start"] <= st["wall_clock_end"] and image["fingerprint"] == fingerprint_of_record()
          and host["note"] is None and proto["timer"] == EFF.TIMER
          and tp["intra_op_requested"] == 1 == tp["intra_op_observed"] == la["thread_policy"]["intra_op_observed"]
          and tp["inter_op"]["status"] == "set" and tp["inter_op"]["observed"] == 1
          and tp["constant"] == 1 == tp["inter_op_constant"] and tp["dl_row"] == PE.THREAD_RULING
          and host["label"] == HOST_LABEL and host["fingerprint"] == EFF.host_type_fingerprint(host["facts"])
          and code["repo_commit"] == head and code["governed_paths_clean"] is True and code["dirty_allowlisted"] == []
          and set(code["files_sha256"]) == set(PE.CODE_FILES)
          and image["digest"] == PD.KD_IMAGE_DIGEST and image["fvcore"] == EFF.FVCORE_PINNED
          and prov["handed"]["sha256"] == sha256_path(fixture("e1")) == prov["expected"]["checkpoint_sha256"]
          and set(prov["binding"]) == BINDING_KEYS
          and lat["intra_op_observed"] == 1 and proto["num_threads"] == 1 and proto["warmup"] == 20
          and proto["measured"] == 100 and proto["forwards_executed"] == 20 + 100 * (
              1 + proto["untimed_executions_per_observation"]))
    return ok, (f"static and latency of record, key sets {keys}: threads {tp['intra_op_observed']}, inter-op "
                f"{tp['inter_op']['observed']}, commit {code['repo_commit'][:12]}, host {host['fingerprint'][:12]}, "
                f"forwards {proto['forwards_executed']}")


# ---------------------------------------------------------------- 3. the x86 latency copy (LF33; §4)
NO_WORLD = contextlib.nullcontext


@case("restated_constants_equal_their_sources")
def _restated():
    import ast
    from src.quant import x86_latency as XL
    tree = ast.parse((REPO / "scripts" / "qat_epoch_eval.py").read_text(encoding="utf-8"))
    fmt = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and [getattr(t, "id", None) for t in n.targets] == ["FINALIZE_FORMAT"])
    ok = (PE.X86_PTQ_STAGES == XL.X86_PTQ_STAGES and PE.X86_QAT_STAGES == XL.X86_QAT_STAGES
          and PE.QAT_FINALIZE_FORMAT == fmt)
    return ok, (f"{PE.X86_PTQ_STAGES} {PE.X86_QAT_STAGES} {PE.QAT_FINALIZE_FORMAT!r} = src.quant.x86_latency's and "
                "scripts/qat_epoch_eval.py's")


@case("e4_x86_copy_bound")
def _e4_bound():
    meta = ptq_run()
    doc = json.loads(meta.read_text(encoding="utf-8"))
    code, _path, rec = run_profile(e4_argv(fresh("e4_bound")))
    if code is not None:
        return False, f"refused: {code}"
    bind = rec["provenance"]["binding"]["x86_copy"]
    ok = (bind["sha256"] == doc["x86_latency_copy"]["sha256"] and rec["provenance"]["backend"] in ("x86", "fbgemm")
          and rec["provenance"]["artifact_role"] == "x86_cpu_proxy_latency" and rec["latency"]["latency_samples"] == 100
          and rec["provenance"]["binding"]["run_provenance_sha256"] == sha256_path(meta))
    return ok, f"bound {Path(bind['path']).name} {bind['sha256'][:12]} under {rec['provenance']['backend']}"


def _e4_edited(tag, edit):
    run = copy_tree(ptq_run().parent, tag)
    meta = run / "e4_ptq_run_meta.json"
    doc = json.loads(meta.read_text(encoding="utf-8"))
    edit(run, doc)
    meta.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return meta


@case("x86_copy_unrecorded")
def _unrecorded():
    meta = _e4_edited("unrec", lambda run, d: d.pop("x86_latency_copy"))
    return _refused(e4_argv(fresh("unrec_out"), prov=meta, h86="0" * 64), "x86_copy_unrecorded", world=NO_WORLD())


@case("x86_copy_missing")
def _missing():
    meta = _e4_edited("missing", lambda run, d: d["x86_latency_copy"].__setitem__("path", "e4_absent_copy.pt"))
    return _refused(e4_argv(fresh("missing_out"), prov=meta), "x86_copy_missing", world=NO_WORLD())


@case("x86_copy_hash_mismatch")
def _hash():
    meta = _e4_edited("hash", lambda run, d: d["x86_latency_copy"].__setitem__("sha256", "e" * 64))
    return _refused(e4_argv(fresh("hash_out"), prov=meta, h86="e" * 64), "x86_copy_hash_mismatch", world=NO_WORLD())


@case("expected_x86_sha256_mismatch")
def _x86_mm():
    return _refused(e4_argv(fresh("x86mm_out"), h86="d" * 64), "expected_x86_sha256_mismatch", world=NO_WORLD())


@case("x86_copy_role")
def _role():
    meta = _e4_edited("role", lambda run, d: d["x86_latency_copy"].__setitem__("reduce_range", False))
    return _refused(e4_argv(fresh("role_out"), prov=meta), "x86_copy_role", world=NO_WORLD())


@case("x86_copy_identity")
def _identity():
    def swap(run, d):
        raw = x86_torchscript("E7", "ptq", seed=6)              # another stage's embedded identity
        (run / "e4_int8_x86_latency.torchscript.pt").write_bytes(raw)
        d["x86_latency_copy"]["sha256"] = sha256_bytes(raw)
    meta = _e4_edited("identity", swap)
    return _refused(e4_argv(fresh("identity_out"), prov=meta), "x86_copy_identity", world=NO_WORLD())


@case("x86_copy_forbidden_name_refused")
def _x86_forbidden():
    meta = _e4_edited("x86_forbid", lambda run, d: d["x86_latency_copy"].__setitem__(
        "path", "plantseg_TEST_x86_copy.pt"))
    # the named copies are never created: a mutant without the guard stops at [x86_copy_missing], having opened nothing
    e4 = _refused(e4_argv(fresh("x86_forbid_out"), prov=meta), "path_forbidden_name", world=NO_WORLD())
    ev = _e5_edited("x86_forbid_e5", lambda f: f["x86_latency_copy"].__setitem__(
        "path", "final/plantseg_TEST_x86_copy.pt"))
    e5 = _refused(e5_argv(fresh("x86_forbid_e5_out"), ev=ev), "path_forbidden_name", world=NO_WORLD())
    return e4[0] and e5[0], f"E4 run record: {e4[1]}; E5 finalize record: {e5[1]}"


@case("e5_x86_copy_bound")
def _e5_bound():
    ev = e5_eval_dir()
    code, _path, rec = run_profile(e5_argv(fresh("e5_bound")))
    if code is not None:
        return False, f"refused: {code}"
    bind = rec["provenance"]["binding"]
    ok = (bind["qat_finalize_sha256"] == sha256_path(ev["finalize"])
          and Path(bind["qat_finalize_path"]).name == "qat_finalize.json"
          and bind["x86_copy"]["sha256"] == sha256_path(ev["dir"] / "final" / "e07_int8_x86_latency.torchscript.pt")
          and bind["run_provenance_sha256"] == sha256_path(ev["prov"])
          and rec["provenance"]["backend"] in ("x86", "fbgemm") and rec["latency"]["latency_samples"] == 100)
    return ok, f"bound through {Path(bind['qat_finalize_path']).name} under {rec['provenance']['backend']}"


def _e5_edited(tag, edit) -> dict:
    d = copy_tree(e5_eval_dir()["dir"], tag)
    fin = json.loads((d / "qat_finalize.json").read_text(encoding="utf-8"))
    edit(fin)
    (d / "qat_finalize.json").write_text(json.dumps(fin, indent=2), encoding="utf-8")
    return {"dir": d, "prov": d / "converted" / "e07_run_meta.json", "finalize": d / "qat_finalize.json"}


@case("qat_finalize_mismatch")
def _fin_mm():
    ev = _e5_edited("fin_mm", lambda f: f["artifact_of_record"]["provenance"].__setitem__("sha256", "a" * 64))
    return _refused(e5_argv(fresh("fin_mm_out"), ev=ev), "qat_finalize_mismatch", world=NO_WORLD())


@case("qat_finalize_not_finalized")
def _fin_not():
    ev = _e5_edited("fin_stop", lambda f: f.__setitem__("status", "STOP"))
    return _refused(e5_argv(fresh("fin_stop_out"), ev=ev), "qat_finalize_not_finalized", world=NO_WORLD())


@case("x86_copy_checks_failed")
def _checks_failed():
    ev = _e5_edited("fin_checks", lambda f: f["x86_latency_copy"]["checks"].__setitem__("parity_x86_copy", False))
    return _refused(e5_argv(fresh("fin_checks_out"), ev=ev), "x86_copy_checks_failed", world=NO_WORLD())


@case("qat_finalize_forbidden_name_refused")
def _fin_forbidden():
    probe = TMP / "nowhere" / "plantseg_TEST_qat_finalize.json"      # never created
    ok, detail = _refused(e5_argv(fresh("fin_forbid_out"), finalize=str(probe)), "path_forbidden_name",
                          world=NO_WORLD())
    return ok and not os.path.lexists(probe.parent), detail


@case("qat_finalize_unreadable_or_other_stage")
def _fin_other():
    other = _e5_edited("fin_stage", lambda f: f.__setitem__("stage", "E6"))
    garbled = _e5_edited("fin_garbled", lambda f: None)
    runs = [(fresh("fin_other_out"), other, "is not a qat_finalize/1 record of E5"),
            (fresh("fin_garbled_out"), garbled, "cannot be read")]
    runs = [(out, e5_argv(out, ev=ev), text) for out, ev, text in runs]    # e5_argv reads H86 from the record
    garbled["finalize"].write_text("{not json", encoding="utf-8")
    got = []
    for out, argv, text in runs:
        code, msg = refused_text(argv)
        got.append((code == "qat_finalize_mismatch" and text in msg and not os.path.lexists(out),
                    f"{code}: {msg[:70]}"))
    return all(ok for ok, _ in got), "; ".join(d for _ok, d in got)


@case("x86_copy_unrecorded_in_finalize")
def _fin_unrecorded():
    ev = _e5_edited("fin_unrec", lambda f: f["x86_latency_copy"].pop("path"))
    return _refused(e5_argv(fresh("fin_unrec_out"), ev=ev), "x86_copy_unrecorded", world=NO_WORLD())


@case("x86_copy_role_reads_every_usable_flag")
def _fin_usable():
    ev = _e5_edited("fin_usable",
                    lambda f: f["x86_latency_copy"]["provenance"].__setitem__("usable_for_accuracy", True))
    return _refused(e5_argv(fresh("fin_usable_out"), ev=ev), "x86_copy_role", world=NO_WORLD())


@case("qat_finalize_required")
def _fin_required():
    return _refused(e5_argv(fresh("fin_req"), finalize=None), "qat_finalize_required", world=NO_WORLD())


@case("qat_finalize_not_qat")
def _fin_not_qat():
    argv = e4_argv(fresh("fin_e4")) + ["--qat-finalize", e5_eval_dir()["finalize"]]
    return _refused(argv, "qat_finalize_not_qat", world=NO_WORLD())


@case("int8_not_official")
def _not_official():
    out = fresh("not_official")
    with of_record_world():
        code, msg = refused_text(e4_argv(out, metrics="params,size,flops", role="accuracy", of_record=True))
    ok = code == "int8_not_official" and "refusing an official E4 score" in msg and not os.path.lexists(out)
    return ok, f"{code}: {msg[:160]}"


# ---------------------------------------------------------------- 4. the table builder (LF34; A2)
HOST_FP = "f" * 64
PINS = {s: ({"checkpoint_sha256": format(i, "x") * 64} if s not in EFF.INT8_STAGES else
            {"checkpoint_sha256": format(i, "x") * 64, "converted_sha256": format(i + 1, "x") * 64,
             "x86_sha256": format(i + 2, "x") * 64}) for i, s in enumerate(EFF.EFFICIENCY_ANALYSIS_STAGES)}


def synthetic_record(stage, group, *, label=HOST_LABEL) -> dict:
    """A plantseg-efficiency/2.0.0 record of record, with the fields the table reads."""
    int8 = stage in EFF.INT8_STAGES
    role = "x86_cpu_proxy_latency" if int8 and group != "static" else "accuracy"
    pins = PINS[stage]
    return {"schema": EFF.EFFICIENCY_SCHEMA, "of_record": True, "efficiency_analysis_member": True,
            "metric_group": group, "deferred": {},
            "thread_policy": {"intra_op_requested": 1, "intra_op_observed": 1, "inter_op": {"observed": 1}},
            "host": {"fingerprint": HOST_FP, "label": label, "note": None},
            "image": {"digest": "sha256:" + "c" * 64, "fingerprint": {"python": "3.11.16"}, "fvcore": "x"},
            "code": {"repo_commit": "b" * 40},
            "provenance": {"stage": stage, "artifact_role": role, "handed": {"sha256": pins["checkpoint_sha256"]},
                           "source_artifact_sha256": pins.get("converted_sha256", "9" * 64),
                           "binding": {"x86_copy": {"sha256": pins.get("x86_sha256")}} if role != "accuracy" else {}},
            "parameters": {"total": 1}, "footprint": {"theoretical_weight_bytes": 4}, "footprint_source": "model",
            "artifact": {"serialized_artifact_bytes": 10}, "compute": {"macs": 1},
            "latency": {"latency_median_ms": 1.0, "latency_iqr_ms": 0.1, "latency_p95_ms": 1.2,
                        "intra_op_observed": 1},
            "memory": {"peak_rss_delta_bytes": 1024, "negative_delta": False}}


def write_session(tag, *, skip=(), extra=(), edit=None, slots=None) -> Path:
    """A sitting: <session>/<stage>/<group>/efficiency_*.json, one record per slot unless edited."""
    root = fresh(tag)
    n = 0
    for stage in EFF.EFFICIENCY_ANALYSIS_STAGES:
        for group in ("static", "latency", "memory"):
            if (stage, group) in skip or (slots is not None and (stage, group) not in slots):
                continue
            rec = synthetic_record(stage, group)
            if edit:
                edit(stage, group, rec)
            d = root / stage / group
            d.mkdir(parents=True, exist_ok=True)
            n += 1
            (d / f"efficiency_{stage}_{group}_{n:03d}.json").write_text(json.dumps(rec), encoding="utf-8")
    for stage, group in extra:
        d = root / stage / group
        d.mkdir(parents=True, exist_ok=True)
        (d / "efficiency_extra.json").write_text(json.dumps(synthetic_record(stage, group)), encoding="utf-8")
    return root


def write_pins(tag, pins=None) -> Path:
    p = fresh(tag).with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"format": EFT.PINS_FORMAT, "stages": pins or PINS}), encoding="utf-8")
    return p


def table(sessions, pins=None, out=None) -> tuple:
    out = out or fresh("table_out")
    try:
        path, doc = EFT.build([str(s) for s in sessions], str(pins or write_pins("pins")), str(out))
    except EFT.TableRefused as e:
        return e.code, None, None, out
    return None, path, doc, out


def _table_refused(want, sessions, pins=None):
    code, _p, _d, out = table(sessions, pins)
    return code == want and not os.path.lexists(out / EFT.TABLE_NAME), str(code)


def _mixed(field_edit):
    def edit(stage, group, rec):
        if (stage, group) == ("E7", "memory"):
            field_edit(rec)
    return edit


@case("table_builds_uniform_session")
def _table_ok():
    code, path, doc, _out = table([write_session("sess_ok")])
    if code is not None:
        return False, f"refused: {code}"
    cited = all(r[k]["record_sha256"] and r[k]["record"] for r in doc["rows"]
                for k in ("parameters", "footprint", "serialized_size", "flops", "latency", "peak_memory"))
    ok = (doc["format"] == "efficiency_table/1"
          and [r["stage"] for r in doc["rows"]] == list(EFF.EFFICIENCY_ANALYSIS_STAGES)
          and len(doc["records"]) == 24 and cited and doc["threads"] == 1 and path.name == "efficiency_table.json")
    return ok, f"{len(doc['rows'])} rows, {len(doc['records'])} records; every quantity cites its record's sha256"


@case("table_threads_mixed")
def _t_threads():
    return _table_refused("table_threads_mixed", [write_session("t_thr", edit=_mixed(
        lambda r: r["thread_policy"].__setitem__("intra_op_observed", 2)))])


@case("table_interop_mixed")
def _t_interop():
    return _table_refused("table_interop_mixed", [write_session("t_int", edit=_mixed(
        lambda r: r["thread_policy"].__setitem__("inter_op", {"observed": 4})))])


@case("table_host_mixed")
def _t_host():
    return _table_refused("table_host_mixed", [write_session("t_host", edit=_mixed(
        lambda r: r["host"].__setitem__("fingerprint", "e" * 64)))])


@case("table_image_mixed")
def _t_image():
    return _table_refused("table_image_mixed", [write_session("t_img", edit=_mixed(
        lambda r: r["image"].__setitem__("digest", "sha256:" + "d" * 64)))])


@case("table_commit_mixed")
def _t_commit():
    return _table_refused("table_commit_mixed", [write_session("t_commit", edit=_mixed(
        lambda r: r["code"].__setitem__("repo_commit", "a" * 40)))])


@case("table_record_not_of_record")
def _t_not_record():
    return _table_refused("table_record_not_of_record", [write_session("t_notrec", edit=_mixed(
        lambda r: r.__setitem__("of_record", False)))])


@case("table_sha256_mismatch")
def _t_sha():
    sess = write_session("t_sha")
    real = EFT.list_records

    def then_change(sessions):
        found = real(sessions)
        p = found[0]["path"]
        p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")     # changed after it was listed
        return found
    with patched(EFT, "list_records", then_change):
        return _table_refused("table_sha256_mismatch", [sess])


@case("table_stage_missing_or_duplicate")
def _t_slots():
    missing = _table_refused("table_stage_missing", [write_session("t_miss", skip={("E5", "latency")})])
    duplicate = _table_refused("table_stage_duplicate", [write_session("t_dup", extra=[("E2", "static")])])
    return missing[0] and duplicate[0], f"missing {missing[1]}; duplicate {duplicate[1]}"


@case("table_artifact_not_pinned")
def _t_pins():
    pins = copy.deepcopy(PINS)
    pins["E6"]["x86_sha256"] = "0" * 64
    return _table_refused("table_artifact_not_pinned", [write_session("t_pin")], write_pins("pins_bad", pins))


@case("table_pins_invalid")
def _t_pins_invalid():
    pins = copy.deepcopy(PINS)
    del pins["E4"]["converted_sha256"]
    return _table_refused("table_pins_invalid", [write_session("t_pinv")], write_pins("pins_invalid", pins))


@case("table_two_sittings_same_type_accepted")
def _t_two():
    first = {(s, g) for s in ("teacher", "E1", "E2", "E3", "E4", "E7") for g in ("static", "latency", "memory")}
    rest = {(s, g) for s in EFF.EFFICIENCY_ANALYSIS_STAGES for g in ("static", "latency", "memory")} - first
    a = write_session("t_sit_a", slots=first)
    b = write_session("t_sit_b", slots=rest, edit=lambda st, gr, r: r["host"].update(label="pod-b", note="second"))
    code, _p, doc, _out = table([a, b])
    if code is not None:
        return False, f"refused: {code}"
    labels = sorted({lab for s in doc["sittings"] for lab in s["host_labels"]})
    return labels == sorted({HOST_LABEL, "pod-b"}) and len(doc["sittings"]) == 2, f"labels {labels}"


SITTING_A = {(s, g) for s in ("teacher", "E1", "E2", "E3", "E4", "E7") for g in ("static", "latency", "memory")}
SITTING_B = {(s, g) for s in EFF.EFFICIENCY_ANALYSIS_STAGES for g in ("static", "latency", "memory")} - SITTING_A


def table_message(sessions, pins=None) -> tuple:
    """(code, message, out): a refusal's code and text; (None, "", out) when the table was written."""
    out = fresh("table_msg_out")
    try:
        EFT.build([str(s) for s in sessions], str(pins or write_pins("pins")), str(out))
    except EFT.TableRefused as e:
        return e.code, str(e), out
    return None, "", out


@case("table_session_missing_or_empty_refused")
def _t_empty():
    full = write_session("t_full_sitting")
    never = fresh("t_never_sitting")                                 # never created
    empty = fresh("t_empty_sitting")
    empty.mkdir(parents=True)
    flat = fresh("t_flat_sitting")                                   # records at <session>/<stage>/, no group level
    (flat / "E1").mkdir(parents=True)
    (flat / "E1" / "efficiency_E1_flat.json").write_text(json.dumps(synthetic_record("E1", "static")),
                                                         encoding="utf-8")
    a = table_message([full, never])
    b = table_message([full, empty])
    c = table_message([flat])
    ok = (a[0] == b[0] == c[0] == "table_stage_missing" and "is not a directory" in a[1]
          and "holds no record" in b[1] and "holds no record" in c[1]
          and not any(os.path.lexists(x[2] / EFT.TABLE_NAME) for x in (a, b, c)) and not os.path.lexists(never))
    return ok, f"never created: {a[0]}; empty: {b[0]}; flat: {c[0]}"


@case("table_session_directory_forbidden_name_refused")
def _t_forbidden_dir():
    got = {}
    for tag, level in (("group", ("E1", "static")), ("stage", ("E2",))):
        a = write_session(f"t_dir_{tag}_a", slots=SITTING_A - {("E1", "static"), ("E2", "static"), ("E2", "latency"),
                                                                ("E2", "memory")})
        b = write_session(f"t_dir_{tag}_b", slots=SITTING_B | {("E1", "static"), ("E2", "static"),
                                                               ("E2", "latency"), ("E2", "memory")})
        link = a.joinpath(*level)
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(TMP / "nowhere" / f"plantseg_TeSt_{tag}", link, target_is_directory=True)   # never created
        code, _m, out = table_message([a, b])
        got[tag] = code == "path_forbidden_name" and not os.path.lexists(out / EFT.TABLE_NAME)
    return all(got.values()) and not os.path.lexists(TMP / "nowhere"), \
        f"a stage or group directory whose resolved path names the token is refused before it is listed: {got}"


@case("table_session_or_pins_forbidden_name_refused")
def _t_forbidden_args():
    s = write_session("t_args")
    a = table_message([s, TMP / "nowhere" / "plantseg_TEST_sitting"])[0]            # never created
    b = table_message([s], pins=TMP / "nowhere" / "plantseg_TEST_pins.json")[0]     # never created
    return a == b == "path_forbidden_name" and not os.path.lexists(TMP / "nowhere"), f"--session {a}; --pins {b}"


@case("table_malformed_pins_or_record_refused")
def _t_malformed():
    got = {}
    pins = copy.deepcopy(PINS)
    pins["E1"] = PINS["E1"]["checkpoint_sha256"]                    # a string, not an object
    sess = write_session("t_mal_ok")
    for tag, argv_pins, session in (("pins_stage_string", write_pins("pins_mal", pins), sess),) + tuple(
            (tag, None, write_session(f"t_mal_{tag}", edit=_mixed(edit))) for tag, edit in (
                ("thread_policy_string", lambda r: r.__setitem__("thread_policy", "1")),
                ("host_list", lambda r: r.__setitem__("host", [HOST_FP])),
                ("inter_op_number", lambda r: r["thread_policy"].__setitem__("inter_op", 1)),
                ("x86_copy_null", lambda r: r["provenance"]["binding"].__setitem__("x86_copy", None)),
                ("memory_string", lambda r: r.__setitem__("memory", "1024")))):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = EFT.main(["build", "--session", str(session), "--pins", str(argv_pins or write_pins(f"pins_{tag}")),
                           "--out-dir", str(fresh(f"t_mal_out_{tag}"))])
        want = "table_pins_invalid" if tag.startswith("pins") else "table_record_not_of_record"
        got[tag] = rc == EFT.EXIT_REFUSED and stdout.getvalue().startswith(f"RESULT: REFUSED [{want}]")
    return all(got.values()), f"a refusal and a RESULT line, never a traceback: {got}"


@case("table_too_many_sessions")
def _t_three():
    s = write_session("t_three")
    return _table_refused("table_too_many_sessions", [s, s, s])


@case("table_record_forbidden_name_refused")
def _t_forbidden_record():
    sess = write_session("t_forbid")
    link = sess / "E1" / "static" / "efficiency_E1_static_link.json"
    os.symlink(TMP / "nowhere" / "plantseg_TeSt_record.json", link)      # resolved form only; never created
    code, _p, _d, out = table([sess])
    return code == "path_forbidden_name" and not os.path.lexists(out / EFT.TABLE_NAME), \
        f"{code}: a session record whose resolved path names the token is refused before it is read"


@case("table_paths_refused")
def _t_paths():
    s = write_session("t_paths")
    probe = TMP / "nowhere" / "plantseg_TEST_table"                 # never created
    inside = REPO / "leff_b2_table_probe"                            # never created
    a = table([s], out=probe)[0]
    b = table([s], out=inside)[0]
    full = fresh("t_full")
    full.mkdir(parents=True)
    (full / "earlier.json").write_text("{}", encoding="utf-8")
    c = table([s], out=full)[0]
    ok = (a, b, c) == ("path_forbidden_name", "out_dir_inside_repository", "out_dir_not_empty") and \
        not os.path.lexists(probe.parent) and not os.path.lexists(inside)
    return ok, f"{a}, {b}, {c}"


# ---------------------------------------------------------------- 5. the host (LF31) and the repository
HOST_KEYS = {"machine", "vendor", "model_name", "isa_flags", "hypervisor", "logical_cpus", "physical_cores",
             "threads_per_core", "affinity_cpus", "cgroup_cpu_quota", "cgroup_cpu_quota_source", "usable_cpus",
             "mem_total_bytes", "kernel", "os_release", "dockerenv", "scaling_governor", "thread_env", "parallel_info",
             "hostname", "host_type_fingerprint"}


@case("host_facts_keys_and_fingerprint_stable")
def _host_keys():
    a, b = EFF.host_facts(), EFF.host_facts()
    ok = HOST_KEYS <= set(a) and a["host_type_fingerprint"] == b["host_type_fingerprint"] == \
        EFF.host_type_fingerprint(a)
    return ok, (f"{a['vendor']} {a['model_name']}; {a['logical_cpus']} logical, {a['physical_cores']} cores, "
                f"usable {a['usable_cpus']}; ISA {a['isa_flags'][:6]}")


@case("fingerprint_excludes_hostname")
def _hostname():
    a = EFF.host_facts()
    return EFF.host_type_fingerprint(a) == EFF.host_type_fingerprint({**a, "hostname": "another-pod"}), \
        "hostname recorded, never fingerprinted"


@case("host_type_fingerprint_ignores_kernel_and_memtotal")
def _kernel():
    a = EFF.host_facts()
    b = {**a, "kernel": "0.0.0-other", "mem_total_bytes": 1, "physical_cores": 99}
    c = {**a, "usable_cpus": (a["usable_cpus"] or 0) + 1}
    ok = EFF.host_type_fingerprint(a) == EFF.host_type_fingerprint(b) != EFF.host_type_fingerprint(c)
    return ok, "kernel, MemTotal and the topology recorded only; usable CPUs fingerprinted"


def cgroup_host_facts(tag, cgroup_lines, quota_files) -> dict:
    """host_facts over a synthetic /proc and /sys that hold only the process's cgroup membership (/proc/self/cgroup)
    and the given quota files under <sys>/fs/cgroup, with an affinity of 4 CPUs injected."""
    base = fresh(tag)
    proc, sys_root = base / "proc", base / "sys"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("".join(f"{line}\n" for line in cgroup_lines), encoding="ascii")
    for rel, text in quota_files.items():
        f = sys_root / "fs" / "cgroup" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text + "\n", encoding="ascii")
    with patched(EFF.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}):
        return EFF.host_facts(proc=str(proc), sys_root=str(sys_root), etc=str(base / "etc"), root=str(base))


@case("cgroup_quota_read_on_the_process_cgroup_path")
def _cgroup_path():
    v2 = cgroup_host_facts("cg_v2", ["0::/leff.slice/child/leaf"], {
        "leff.slice/child/leaf/cpu.max": "max 100000",              # the process's own cgroup sets none
        "leff.slice/child/cpu.max": "350000 100000",                # 3.5 CPUs
        "leff.slice/cpu.max": "250000 100000"})                     # 2.5 CPUs on an ancestor: the smallest binds
    v1 = cgroup_host_facts("cg_v1", ["2:cpuacct:/", "1:cpu,cpuacct:/docker/ctr", "0::/"], {
        "cpu,cpuacct/docker/ctr/cpu.cfs_quota_us": "150000", "cpu,cpuacct/docker/ctr/cpu.cfs_period_us": "100000"})
    unset = cgroup_host_facts("cg_unset", ["0::/"], {"cpu.max": "max 100000"})
    ok = (v2["cgroup_cpu_quota"] == 2.5 and v2["cgroup_cpu_quota_source"].endswith("fs/cgroup/leff.slice")
          and v2["usable_cpus"] == 2 and v1["cgroup_cpu_quota"] == 1.5 and v1["usable_cpus"] == 1
          and v1["cgroup_cpu_quota_source"].endswith("fs/cgroup/cpu,cpuacct/docker/ctr")
          and unset["cgroup_cpu_quota"] is None and unset["usable_cpus"] == 4)
    return ok, (f"v2 nested: {v2['cgroup_cpu_quota']} CPUs from {Path(v2['cgroup_cpu_quota_source'] or '').name} -> "
                f"usable {v2['usable_cpus']}; v1: {v1['cgroup_cpu_quota']} -> usable {v1['usable_cpus']}; unset -> "
                f"usable {unset['usable_cpus']} (affinity 4)")


BEFORE = None


@case("repo_state_unchanged")
def _repo():
    verdict = state_verdict(BEFORE, repo_state(REPO))
    return verdict == "unchanged", verdict


@case("no_forbidden_named_path_created")
def _no_forbidden():
    bad = [os.path.join(r, n) for r, ds, fs in os.walk(TMP) for n in ds + fs if FORBIDDEN in n.lower() or ":" in n]
    return not bad, f"{len(bad)} under {TMP}" + (f": {bad[:3]}" if bad else "")


# ---------------------------------------------------------------- the child processes (LF30)
def child_world(checkout: str) -> None:
    """A child's world of record: the parent's synthetic clean checkout as the governed root, the KD image's digest
    (this child's own environment only) and its fingerprint injected, so no run reaches a real checkout."""
    import scripts.preflight_distill as PD
    os.environ["PLANTSEG_IMAGE_DIGEST"] = PD.KD_IMAGE_DIGEST
    fp = fingerprint_of_record()
    PE.image_fingerprint = lambda: (0, fp, "")
    PE.GOVERNED_ROOT = Path(checkout)


def record_argv(ckpt, out) -> list:
    return ["--of-record", "--stage", "E1", "--checkpoint", ckpt, "--expected-checkpoint-sha256", sha256_path(ckpt),
            "--metrics", "params,size,flops", "--threads", "1", "--host-label", HOST_LABEL, "--out-dir", out]


def quiet_main(run_argv) -> tuple:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = PE.main([str(a) for a in run_argv])
    return rc, err.getvalue()


def child_main(name: str, argv) -> int:
    """The fresh processes of the inter-op cases (LF30):
      interop_start CKPT OUT                   main() first in a fresh process, off record
      interop_locked CKPT OUT CHECKOUT         a torch whose set_num_interop_threads raises (injected), then a run of
                                               record with every earlier gate passing
      interop_twice CKPT OUT1 OUT2 CHECKOUT    main() twice in one process: off record, then of record
      torch_twice                              torch.set_num_interop_threads twice and nothing else (torch itself)
      torch_after_work                         a scripted torch.jit.fork, then torch.set_num_interop_threads once"""
    if name == "torch_twice":
        torch.set_num_interop_threads(1)
        print(CHILD_TAG + json.dumps({"first": "returned"}), flush=True)
        torch.set_num_interop_threads(1)           # torch 2.1.0 ends the process here (SIGABRT)
        print(CHILD_TAG + json.dumps({"second": "returned"}), flush=True)
        return 0
    if name == "torch_after_work":
        torch.jit.script(_forked_neg)(torch.ones(4))         # the scripted fork runs on the inter-op pool
        print(CHILD_TAG + json.dumps({"work": "done"}), flush=True)
        torch.set_num_interop_threads(1)           # torch 2.1.0 ends the process here (SIGABRT)
        print(CHILD_TAG + json.dumps({"set": "returned"}), flush=True)
        return 0
    ckpt, out = argv[0], argv[1]
    doc: dict = {}
    if name == "interop_locked":
        def raises(_n):
            raise RuntimeError("Error: cannot set number of interop threads after parallel work has started or "
                               "set_num_interop_threads called (injected)")
        torch.set_num_interop_threads = raises
        child_world(argv[2])
        rc, err = quiet_main(record_argv(ckpt, out))
    elif name == "interop_twice":
        doc["first_rc"], _first_err = quiet_main(["--stage", "E1", "--checkpoint", ckpt, "--metrics", "params",
                                                  "--out-dir", out])
        if doc["first_rc"] == 0:
            doc["first_inter_op"] = json.loads(next(Path(out).iterdir()).read_text(encoding="utf-8"))[
                "thread_policy"]["inter_op"]
        child_world(argv[3])
        rc, err = quiet_main(record_argv(ckpt, argv[2]))
        out = argv[2]
    else:
        rc, err = quiet_main(["--stage", "E1", "--checkpoint", ckpt, "--metrics", "params", "--out-dir", out])
    doc.update({"rc": rc, "stderr": err})
    if rc == 0:
        doc["thread_policy"] = json.loads(next(Path(out).iterdir()).read_text(encoding="utf-8"))["thread_policy"]
    print(CHILD_TAG + json.dumps(doc, default=str), flush=True)
    return 0


def main(argv=None) -> int:
    global BEFORE, TMP, INTEROP
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated check names (the mutation runner's)")
    ap.add_argument("--child", nargs="+", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        return child_main(args.child[0], args.child[1:])
    INTEROP = EFF.set_interop_threads(EFF.INTEROP_THREADS_OF_RECORD)     # this process's one call, as main() makes it
    only = None if args.only is None else [n for n in args.only.split(",") if n]
    unknown = sorted(set(only or ()) - set(CASES))
    if unknown:
        print(f"unknown checks {unknown}")
        return 2
    TMP = safe_tmpdir("leff_b2_")
    BEFORE = repo_state(REPO)
    threads = torch.get_num_threads()
    print("=" * 78)
    print("LEFF B-2 smoke: synthetic only; no dataset, no checkpoint of record, no GPU, no install")
    print(f"torch {torch.__version__} | engines {list(torch.backends.quantized.supported_engines)} | temp {TMP}")
    print(f"inter-op at start: {INTEROP}")
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
