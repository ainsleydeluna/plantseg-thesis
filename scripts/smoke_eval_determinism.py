#!/usr/bin/env python3
"""L-EVAL-DET smoke (DL-17; docs/EVALUATION_CONTRACT.md section 10). CPU-only, data-free.

Sections:
  P  policy equivalence, in fresh child interpreters with an inline literal state reader
  G  scope truth table
  D  model-device forward hook and post-evaluation checks (D5: the hook leaves the metrics bitwise equal)
  W  warning capture
  A  artifact additivity of run.eval_runtime (fixture git repositories only)
  R  scripts/evaluate_model.run on CPU with a synthetic checkpoint and synthetic data (patched adapter;
     no data-root listing); the --device cuda paths run in children
  X  fill_uninitialized_memory recorded as null where torch lacks torch.utils.deterministic (X6)
  C  scripts/compare_eval_artifacts.py: constants, identity, band and validity exit codes (Q6)

Writes only under tempfile.mkdtemp() (removed in `finally`). Children run with -B. Never reads the git
state of the project repository: every request uses a temp fixture repository.

Run:  python -B scripts/smoke_eval_determinism.py      (exit 1 on any failure; SKIP is never a PASS)
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import traceback
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import numpy as np      # noqa: E402
import torch            # noqa: E402

import compare_eval_artifacts as CMP                                       # noqa: E402
import evaluate_model as CLI                                               # noqa: E402
from src.eval import (Condition, DatasetMeta, EvalBatch, ManifestEntry,   # noqa: E402
                      RunMeta, evaluate_model as core_evaluate)
from src.eval import adapters as ADP                                       # noqa: E402
from src.eval import eval_runtime as ER                                    # noqa: E402
from src.eval import model_loading as ML                                   # noqa: E402
from src.eval.artifacts import (ARTIFACT_FILES, EVAL_RUNTIME_VERSION,      # noqa: E402
                                MANIFEST_NAME, METRIC_PROTOCOL, SCHEMA_VERSION,
                                ArtifactRequestError, ArtifactWriteError,
                                build_config_payload, prepare_artifact_request,
                                validate_artifact_request, verify_artifact, write_artifact)

C = 116
CHECKS: list[tuple[str, str, str]] = []          # (name, PASS|FAIL, detail)

B6_ON = {"deterministic_algorithms": True, "deterministic_algorithms_warn_only": True,
         "cudnn_deterministic": True, "cudnn_benchmark": False,
         "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
B6_OFF = {"deterministic_algorithms": False, "deterministic_algorithms_warn_only": False,
          "cudnn_deterministic": False, "cudnn_benchmark": False, "CUBLAS_WORKSPACE_CONFIG": None}
RECORD_KEYS_LITERAL = {
    "eval_runtime_version", "model_device", "model_tensor_devices", "input_devices",
    "forward_batches", "batch_size", "num_workers", "determinism_policy_applied",
    "cuda_initialized_before_policy", "inherited_cublas_workspace_config", "determinism",
    "fill_uninitialized_memory", "cudnn_enabled", "tf32", "gpu_name", "gpu_capability",
    "cudnn_version", "torch_cuda", "torch_num_threads", "pillow", "checkpoint_iteration",
    "checkpoint_best_val_miou_all_class", "image_digest", "eval_warnings",
    "eval_warnings_truncated", "nondeterministic_alert_count"}
CONFIG_PAYLOAD_KEYS_PRE_S1 = {
    "schema_version", "metric_protocol", "stage", "model_role", "precision", "quant_backend",
    "random_init", "device", "dataset_name", "dataset_doi", "split", "condition",
    "preprocess_protocol", "expected_rows", "num_classes", "background_index", "ignore_index"}
ALERT = ("upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation, but you "
         "set 'torch.use_deterministic_algorithms(True, warn_only=True)'.")
SYNTH_CLASS_MAP = [{"id": 0, "name": "synthetic_background", "role": "background"},
                   *[{"id": c, "name": f"synthetic_disease_{c:03d}", "role": "disease"}
                     for c in range(1, C)]]


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, "PASS" if ok else "FAIL", " | ".join(str(detail).split("\n")).strip()))


# ------------------------------------------------------------------------------------------ children
CHILD_PRELUDE = '''
import json, os, sys, warnings
def five():
    import torch
    return {"deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_algorithms_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
def full():
    import torch
    d = five()
    try:
        import torch.utils.deterministic as det
        d["fill_uninitialized_memory"] = bool(det.fill_uninitialized_memory)
    except (ImportError, AttributeError):
        d["fill_uninitialized_memory"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
        d["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        d["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        d["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    return d
def emit(obj):
    print("RESULT_JSON:" + json.dumps(obj, sort_keys=True, default=str), flush=True)
'''


def run_child(work: Path, body: str) -> dict:
    """Run `body` in a fresh `python -B` child (CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED removed)."""
    path = work / f"child_{len(list(work.glob('child_*.py'))):02d}.py"
    path.write_text(CHILD_PRELUDE + textwrap.dedent(body), encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / "scripts")]),
               PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-B", str(path)], cwd=str(REPO), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=900)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")]
    if not lines:
        raise AssertionError(f"child produced no result (rc={proc.returncode}): "
                             f"{(proc.stdout + proc.stderr)[-600:]}")
    return json.loads(lines[-1][len("RESULT_JSON:"):])


# ------------------------------------------------------------------------------------------ fixtures
def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=evaldet-fixture", "-c", "user.email=evaldet-fixture@invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), capture_output=True, text=True, check=False,
        env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(repo.parent)))
    if proc.returncode != 0:
        raise AssertionError(f"fixture git {args} failed ({proc.returncode}): {proc.stderr[:200]}")
    return proc.stdout


def make_fixture_repo(root: Path, *, governed_dirty: bool = False) -> Path:
    """A throwaway repository (never the project). Clean, or with one untracked governed path."""
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "seed.txt").write_text("eval-det fixture\n", encoding="utf-8")
    _git(root, "add", "--", "seed.txt")
    _git(root, "commit", "-q", "-m", "fixture baseline")
    top = Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != root.resolve():
        raise AssertionError(f"fixture is not its own repository: {top}")
    if governed_dirty:
        (root / "scripts").mkdir()
        (root / "scripts" / "probe.txt").write_text("governed-dirty probe\n", encoding="utf-8")
    return root


def rm_tree(path: Path) -> None:
    for p in path.rglob("*"):
        try:
            os.chmod(p, stat.S_IWRITE)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def synth_manifest(indices) -> list[ManifestEntry]:
    return [ManifestEntry(int(i), f"syn_val_{int(i):05d}", f"syn_val_{int(i):05d}")
            for i in sorted(indices)]


def synth_sample(i: int, size: int = 64):
    g = torch.Generator().manual_seed(1000 + int(i))
    image = torch.rand(3, size, size, generator=g)
    target = torch.randint(0, C, (size, size), generator=g)
    target[:4, :4] = 255
    return image, target


class FakeEvalDataset(torch.utils.data.Dataset):
    """Stands in for adapters.PlantSegEvalDataset: synthetic tensors, no data root."""

    def __init__(self, split, source_indices=None):
        self.split = split
        self.source_indices = sorted(int(i) for i in source_indices)

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, i):
        src = self.source_indices[i]
        image, target = synth_sample(src)
        sid = f"syn_val_{src:05d}"
        return {"image": image, "target": target, "image_id": sid, "clean_image_id": sid,
                "manifest_index": src}


DATA_ROOT_LISTINGS: list[str] = []


def install_fakes():
    """Patch the adapter module attributes CLI.run imports at call time, and spy on listings of the
    configured data root. Returns an undo callable."""
    from configs.data import DATA
    root = str(Path(DATA["root"]))
    saved = (ADP.PlantSegEvalDataset, ADP.build_expected_manifest_for, ADP.list_split_stems,
             Path.iterdir, os.scandir)

    def _no_listing(*a, **k):
        raise AssertionError("list_split_stems must not be called (data-free smoke)")

    def _iterdir(self):
        if str(self).startswith(root):
            DATA_ROOT_LISTINGS.append(str(self))
        return saved[3](self)

    def _scandir(path="."):
        if str(path).startswith(root):
            DATA_ROOT_LISTINGS.append(str(path))
        return saved[4](path)

    ADP.PlantSegEvalDataset = FakeEvalDataset
    ADP.build_expected_manifest_for = lambda split, idx: synth_manifest(idx)
    ADP.list_split_stems = _no_listing
    Path.iterdir = _iterdir
    os.scandir = _scandir

    def undo():
        (ADP.PlantSegEvalDataset, ADP.build_expected_manifest_for, ADP.list_split_stems,
         Path.iterdir, os.scandir) = saved
    return undo


def make_checkpoint(path: Path) -> Path:
    model = ML.build_fp32_student()
    torch.save({"iter": 4000, "model_state_dict": model.state_dict(), "num_classes": C,
                "best_val_miou_all_class": 0.123}, path)
    return path


def cli_args(**kw):
    base = dict(stage="E1", model_role="student", precision="fp32", split="val", condition="clean",
                corruption_severity=None, out_dir="X", checkpoint=None, teacher_config=None,
                provenance=None, random_init=False, artifact_status="provisional", batch_size=2,
                max_samples=4, confirm_test_split=False, run_id="evaldet", device="cpu")
    base.update(kw)
    return type("A", (), base)()


def cpu_record(**over):
    fwd = ER.ModelDeviceForward(torch.device("cpu"))
    fwd.input_devices.add("cpu")
    rec = ER.build_eval_runtime_record(
        model=torch.nn.Conv2d(3, C, 1), model_device=torch.device("cpu"), fwd=fwd, batch_size=2,
        forward_batches=3, policy_applied=False, policy_info=None, ckpt_info=None,
        warn_summary={"eval_warnings": [], "eval_warnings_truncated": False,
                      "nondeterministic_alert_count": 0})
    rec.update(over)
    return rec


def synth_request(out_dir: Path, repo: Path, *, status="smoke", random_init=True, ck=None):
    manifest = synth_manifest(range(5))
    return prepare_artifact_request(
        out_dir=out_dir, artifact_status=status, run_id="evaldet_artifact",
        run=RunMeta(stage="E1", model_role="student", precision="fp32", quant_backend=None,
                    checkpoint_path=None if random_init else str(ck),
                    checkpoint_sha256=None if random_init else "0" * 64,
                    random_init=random_init, device="cpu"),
        dataset=DatasetMeta(name="SYNTHETIC eval-det fixture (NOT PlantSeg data)",
                            doi="10.5281/zenodo.17719108", split="val", condition=Condition("clean"),
                            preprocess_protocol="synthetic-evaldet/1.0.0", expected_rows=5),
        expected_manifest=manifest, class_map=SYNTH_CLASS_MAP, repo_root=repo)


def synth_batches(indices, bs=2):
    out = []
    idx = sorted(indices)
    for k in range(0, len(idx), bs):
        grp = idx[k:k + bs]
        imgs, tgts = zip(*(synth_sample(i, 8) for i in grp))
        ids = [f"syn_val_{i:05d}" for i in grp]
        out.append(EvalBatch(torch.stack(imgs), torch.stack(tgts), ids, list(ids), list(grp)))
    return out


def seeded_conv():
    torch.manual_seed(7)
    return torch.nn.Conv2d(3, C, 1).eval()


def core(model, batches, forward=None):
    return core_evaluate(model, batches, expected_manifest=synth_manifest(range(5)),
                         condition=Condition("clean"), num_classes=C, background_index=0,
                         ignore_index=255, forward=forward)


def parent_state():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {**ER.determinism_state(), **ER.tf32_state(),
                "cublas_env": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


# ------------------------------------------------------------------------------------------ P
def section_p(work: Path) -> dict:
    p1 = run_child(work, 'import torch\nemit({"five": five(), "full": full(), "init": torch.cuda.is_initialized()})\n')
    p2 = run_child(work, 'import torch\nfrom src.seeds import set_seed\nset_seed(42)\n'
                         'emit({"five": five(), "full": full(), "init": torch.cuda.is_initialized()})\n')
    p3 = run_child(work, 'import torch\nfrom src.eval import eval_runtime as ER\ninfo = ER.apply_eval_determinism()\n'
                         'emit({"five": five(), "full": full(), "init": torch.cuda.is_initialized(),\n'
                         '      "expected": ER.EXPECTED_POLICY, "info": info})\n')
    check("P1 bare child: five-state all off, CUBLAS unset, CUDA uninitialised",
          p1["five"] == B6_OFF and p1["init"] is False, json.dumps(p1["five"]))
    check("P2 set_seed(42) child: five-state == B6 (T,T,T,F,':4096:8')",
          p2["five"] == B6_ON, json.dumps(p2["five"]))
    check("P3 apply_eval_determinism child: full dict == set_seed's; five == EXPECTED_POLICY; CUDA uninitialised",
          p3["full"] == p2["full"] and p3["five"] == p3["expected"] == B6_ON and p3["init"] is False
          and p3["info"] == {"cuda_initialized_before_policy": False,
                             "inherited_cublas_workspace_config": None},
          f"fill_uninitialized_memory={p3['full'].get('fill_uninitialized_memory')!r}")
    diff = sorted(k for k in B6_ON if p1["five"][k] != p2["five"][k])
    check("P4 negative control: bare vs set_seed differ in exactly four named keys",
          diff == sorted(["deterministic_algorithms", "deterministic_algorithms_warn_only",
                          "cudnn_deterministic", "CUBLAS_WORKSPACE_CONFIG"]), str(diff))
    rng_body = '''
import hashlib, pickle, random
import numpy as np, torch
torch.manual_seed(123); np.random.seed(123); random.seed(123)
def dig():
    h = lambda b: hashlib.sha256(b).hexdigest()
    return {"torch": h(torch.get_rng_state().numpy().tobytes()), "numpy": h(pickle.dumps(np.random.get_state())),
            "python": h(pickle.dumps(random.getstate())), "phs": os.environ.get("PYTHONHASHSEED")}
before = dig()
{call}
emit({{"before": before, "after": dig()}})
'''
    ap = run_child(work, rng_body.replace("{call}", "from src.eval import eval_runtime as ER; ER.apply_eval_determinism()")
                   .replace("{{", "{").replace("}}", "}"))
    ss = run_child(work, rng_body.replace("{call}", "from src.seeds import set_seed; set_seed(42)")
                   .replace("{{", "{").replace("}}", "}"))
    check("P5 apply leaves torch/numpy/python RNG and PYTHONHASHSEED unchanged; set_seed changes all",
          ap["before"] == ap["after"]
          and all(ss["before"][k] != ss["after"][k] for k in ("torch", "numpy", "python"))
          and ss["after"]["phs"] == "42",
          f"apply_equal={ap['before'] == ap['after']} set_seed_phs={ss['after']['phs']}")
    p6 = run_child(work, '''
import torch
torch.cuda.is_initialized = lambda: True
from src.eval import eval_runtime as ER
try:
    ER.apply_eval_determinism(); code = None
except ER.EvalRuntimeError as e:
    code = e.code
emit({"code": code, "five": five()})
''')
    check("P6 CUDA already initialised -> cuda_initialized_before_policy; nothing applied",
          p6["code"] == "cuda_initialized_before_policy" and p6["five"] == B6_OFF, str(p6["code"]))
    p7 = run_child(work, '''
import torch
torch.cuda.is_initialized = lambda: os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
from src.eval import eval_runtime as ER
try:
    ER.apply_eval_determinism(); code = None
except ER.EvalRuntimeError as e:
    code = e.code
emit({"code": code})
''')
    check("P7 state-keyed fake (CUDA initialises once CUBLAS is set) -> cuda_initialized_during_policy",
          p7["code"] == "cuda_initialized_during_policy", str(p7["code"]))
    p8 = run_child(work, '''
import src.eval.eval_runtime
mods = sorted(m for m in sys.modules if m.split(".")[0] in ("mmseg", "mmcv", "mmengine")
              or m in ("src.data", "configs.data") or m.startswith("src.data."))
emit({"five": five(), "mods": mods})
''')
    check("P8 importing eval_runtime alone: no state change, no mm*/src.data/configs.data loaded",
          p8["five"] == B6_OFF and p8["mods"] == [], str(p8["mods"]))
    p9 = run_child(work, '''
import src.training.train_e1, src.training.train_distill, src.quant.runner
bad = [m for m in ("src.eval.eval_runtime", "src.eval.artifacts", "evaluate_model",
                   "scripts.evaluate_model") if m in sys.modules]
emit({"bad": bad})
''')
    check("P9 train_e1/train_distill/quant.runner imports load no eval_runtime, artifacts or CLI (I1/I5)",
          p9["bad"] == [], str(p9["bad"]))
    return p1


# ------------------------------------------------------------------------------------------ G, D
def section_g_d() -> None:
    table = [(("student", "fp32", "cuda"), True), (("student", "fp32", "cuda:0"), True),
             (("student", "fp32", "cpu"), False), (("teacher", "fp32", "cuda"), False),
             (("student", "int8_ptq", "cpu"), False), (("student", "int8_qat", "cpu"), False)]
    got = [(a, ER.determinism_required(*a)) for a, _ in table]
    check("G1 determinism_required truth table (FP32 student on CUDA only); CUDA uninitialised",
          all(g == e for (_, g), (_, e) in zip(got, table)) and not torch.cuda.is_initialized(),
          str(got))

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x):
            self.seen.append(x)
            return x

    x = torch.zeros(2, 3, 4, 4)
    spy, fwd = Spy(), ER.ModelDeviceForward(torch.device("meta"))
    fwd(spy, x)
    fwd(spy, x)
    check("D1 ModelDeviceForward(meta) moves inputs to the model device",
          len(spy.seen) == 2 and all(s.device.type == "meta" for s in spy.seen)
          and fwd.input_devices == {"meta"})
    spy2, fwd2 = Spy(), ER.ModelDeviceForward(torch.device("cpu"))
    fwd2(spy2, x)
    check("D2 on CPU the identical tensor object is passed through",
          spy2.seen[0] is x and fwd2.input_devices == {"cpu"})
    check("D3 tensor devices ([] for a parameterless module); teacher/INT8 resolve to CPU without CUDA init; "
          "cpu:0 normalises to cpu",
          ER.model_tensor_devices(torch.nn.Module()) == []
          and ER.model_tensor_devices(torch.nn.Conv2d(1, 1, 1)) == ["cpu"]
          and ER.resolve_model_device("cuda", cpu_only=True) == torch.device("cpu")
          and str(ER.resolve_model_device("cpu:0", cpu_only=False)) == "cpu"
          and not torch.cuda.is_initialized())

    def code_of(fn):
        try:
            fn()
            return None
        except ER.EvalRuntimeError as e:
            return e.code

    fm = ER.ModelDeviceForward(torch.device("meta"))
    fm.input_devices.add("meta")
    c1 = code_of(lambda: ER.check_post_eval(model=torch.nn.Conv2d(1, 1, 1),
                                            model_device=torch.device("meta"), fwd=fm,
                                            policy_applied=False))
    fc = ER.ModelDeviceForward(torch.device("meta"))
    fc.input_devices.add("cpu")
    c2 = code_of(lambda: ER.check_post_eval(model=torch.nn.Module(), model_device=torch.device("meta"),
                                            fwd=fc, policy_applied=False))
    c3 = code_of(lambda: ER.check_post_eval(model=torch.nn.Module(), model_device=torch.device("meta"),
                                            fwd=fm, policy_applied=False))
    check("D4 post-eval checks: model_device_mismatch, input_device_mismatch; a matching run passes",
          (c1, c2, c3) == ("model_device_mismatch", "input_device_mismatch", None), str((c1, c2, c3)))

    r0 = core(seeded_conv(), synth_batches(range(5)))
    r1 = core(seeded_conv(), synth_batches(range(5)), forward=ER.ModelDeviceForward(torch.device("cpu")))
    arrays = ("sparse_image_index", "sparse_class_id", "sparse_tp", "sparse_gt", "sparse_pred",
              "dataset_tp", "dataset_gt", "dataset_pred")
    same = (r0.dataset_level == r1.dataset_level and r0.per_class == r1.per_class
            and [r.as_dict() for r in r0.rows] == [r.as_dict() for r in r1.rows]
            and r0.forward_batches == r1.forward_batches
            and all(np.array_equal(getattr(r0, a), getattr(r1, a)) and getattr(r0, a).dtype == np.int64
                    for a in arrays))
    check("D5 core with forward=None vs ModelDeviceForward(cpu): metrics, rows and NPZ arrays bitwise equal (I3)",
          same, f"all_class_miou={r0.dataset_level.get('all_class_miou')!r}")


# ------------------------------------------------------------------------------------------ W
def section_w() -> None:
    def warn_fwd(m, x):
        warnings.warn(ALERT, UserWarning)
        return m(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = core(seeded_conv(), synth_batches(range(5)), forward=warn_fwd)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        res, summ = ER.evaluate_capturing_warnings(core, seeded_conv(), synth_batches(range(5)),
                                                   forward=warn_fwd)
    check("W1 the seed-42 alert on 3 batches: 1 unique entry, alert count 3, metrics unchanged",
          len(summ["eval_warnings"]) == 1 and summ["nondeterministic_alert_count"] == 3
          and summ["eval_warnings_truncated"] is False and res.dataset_level == plain.dataset_level,
          json.dumps(summ)[:200])

    def other():
        warnings.warn("an unrelated notice", UserWarning)
        return 1
    with contextlib.redirect_stderr(io.StringIO()):
        _, s2 = ER.evaluate_capturing_warnings(other)
    check("W2 an unrelated UserWarning is captured with alert count 0",
          s2["eval_warnings"] == ["UserWarning: an unrelated notice"]
          and s2["nondeterministic_alert_count"] == 0)
    _, s3 = ER.evaluate_capturing_warnings(lambda: 1)
    check("W3 no warnings -> ([], False, 0)",
          (s3["eval_warnings"], s3["eval_warnings_truncated"], s3["nondeterministic_alert_count"])
          == ([], False, 0))
    text = err.getvalue()
    check("W4 stderr carries the alert once, prefixed [eval-warning x3]",
          text.count("does not have a deterministic implementation") == 1
          and "[eval-warning x3] UserWarning:" in text, text[:160])

    def many():
        for i in range(60):
            warnings.warn(f"op{i:02d} does not have a deterministic implementation", UserWarning)
        return 1
    with contextlib.redirect_stderr(io.StringIO()):
        _, s5 = ER.evaluate_capturing_warnings(many)
    check("W5 60 unique warnings -> 50 kept, truncated, every emission counted",
          len(s5["eval_warnings"]) == 50 and s5["eval_warnings_truncated"] is True
          and s5["nondeterministic_alert_count"] == 60)

    def boom():
        warnings.warn(ALERT, UserWarning)
        raise RuntimeError("core failed")
    err6 = io.StringIO()
    raised = False
    with contextlib.redirect_stderr(err6):
        try:
            ER.evaluate_capturing_warnings(boom)
        except RuntimeError:
            raised = True
    check("W6 a failing evaluation still echoes its captured warnings to stderr, then re-raises",
          raised and "[eval-warning x1] UserWarning:" in err6.getvalue(), err6.getvalue()[:120])


# ------------------------------------------------------------------------------------------ A
def section_a(work: Path) -> None:
    repo = make_fixture_repo(work / "fx_clean_a")
    result = core(seeded_conv(), synth_batches(range(5)))

    out1 = work / "a1"
    req1 = synth_request(out1, repo)
    write_artifact(result, req1, validate_artifact_request(req1))
    s1 = verify_artifact(out1)
    check("A1 no eval_runtime -> key absent, verify passes, exactly four files",
          "eval_runtime" not in s1["run"] and len(list(out1.iterdir())) == 4)

    rec = cpu_record()
    out2 = work / "a2"
    req2 = synth_request(out2, repo)
    write_artifact(result, req2, validate_artifact_request(req2), eval_runtime=rec)
    s2 = verify_artifact(out2)
    check("A2 record round-trips; schema_version and config_sha256 unchanged",
          s2["run"]["eval_runtime"] == json.loads(json.dumps(rec))
          and s2["schema_version"] == "plantseg-eval/1.0.0" == SCHEMA_VERSION
          and s2["run"]["config_sha256"] == s1["run"]["config_sha256"])

    def refused(out: Path, rt) -> bool:
        req = synth_request(out, repo)
        prov = validate_artifact_request(req)
        try:
            write_artifact(result, req, prov, eval_runtime=rt)
            return False
        except ArtifactWriteError:
            return not out.exists() and not list(out.parent.glob(f".{out.name}.tmp-*"))

    check("A3 a non-dict record is refused; nothing created", refused(work / "a3", [1]))
    bad4 = cpu_record()
    bad4["tf32"] = dict(bad4["tf32"], float32_matmul_precision="")
    check("A4 a \"\" at depth 2 is refused; nothing created", refused(work / "a4", bad4))
    check("A5 a wrong eval_runtime_version is refused",
          refused(work / "a5", cpu_record(eval_runtime_version="plantseg-eval-runtime/0.9.0")))

    ck = work / "a6_ck.pt"
    ck.write_bytes(b"not read")
    out6 = work / "a6"
    req6 = synth_request(out6, repo, status="official", random_init=False, ck=ck)
    prov6 = validate_artifact_request(req6)
    try:
        write_artifact(result, req6, prov6)
        e6 = None
    except ArtifactWriteError as e:
        e6 = str(e)
    ok_refuse = (e6 is not None and "section 10" in e6 and not out6.exists()
                 and not list(out6.parent.glob(f".{out6.name}.tmp-*")))
    write_artifact(result, req6, prov6, eval_runtime=rec)
    s6 = verify_artifact(out6)
    check("A6 official without a record is refused (names section 10); with a record it is written",
          ok_refuse and s6["run"]["artifact_status"] == "official", str(e6)[:120])
    check("A7 config payload key set == the pre-S1 17 keys",
          set(build_config_payload(req1)) == CONFIG_PAYLOAD_KEYS_PRE_S1)


# ------------------------------------------------------------------------------------------ R
R_CHILD = '''
import torch
{patch}
sys.path.insert(0, os.environ["EVALDET_SCRIPTS"])
import smoke_eval_determinism as S
import evaluate_model as CLI
from pathlib import Path
undo = S.install_fakes()
CLI.REPO = Path(os.environ["EVALDET_REPO"])
ctr = CLI.Counters(dataset=[], model=[])
args = S.cli_args(checkpoint=os.environ["EVALDET_CK"], out_dir=os.environ["EVALDET_OUT"],
                  device="cuda", artifact_status=os.environ["EVALDET_STATUS"])
err = code = rt = None
tb_funcs = []
try:
    CLI.run(args, counters=ctr)
    from src.eval.artifacts import verify_artifact
    rt = verify_artifact(Path(os.environ["EVALDET_OUT"]))["run"]["eval_runtime"]
except Exception as e:
    import traceback
    err = type(e).__name__ + ": " + str(e)[:200]
    code = getattr(e, "code", None)
    tb_funcs = [f.name for f in traceback.extract_tb(e.__traceback__)]
emit({{"err": err, "code": code, "five": five(), "init": torch.cuda.is_initialized(),
      "dataset": ctr.dataset, "model": ctr.model, "out_exists": Path(os.environ["EVALDET_OUT"]).exists(),
      "listings": S.DATA_ROOT_LISTINGS, "tb_funcs": tb_funcs,
      "rt": rt and {{k: rt[k] for k in ("model_device", "input_devices", "determinism_policy_applied")}}}})
'''


def r_child(work: Path, *, repo: Path, ck: Path, out: Path, status="provisional", patch="") -> dict:
    os.environ.update(EVALDET_SCRIPTS=str(REPO / "scripts"), EVALDET_REPO=str(repo),
                      EVALDET_CK=str(ck), EVALDET_OUT=str(out), EVALDET_STATUS=status)
    try:
        return run_child(work, R_CHILD.format(patch=patch))
    finally:
        for k in ("EVALDET_SCRIPTS", "EVALDET_REPO", "EVALDET_CK", "EVALDET_OUT", "EVALDET_STATUS"):
            os.environ.pop(k, None)


def section_r(work: Path, p1: dict) -> None:
    from src.training import train_e1
    repo = make_fixture_repo(work / "fx_clean_r")
    ck = make_checkpoint(work / "synthetic_ck.pt")
    before = parent_state()
    undo = install_fakes()
    saved_repo = CLI.REPO
    saved_digest = os.environ.get("PLANTSEG_IMAGE_DIGEST")
    try:
        CLI.REPO = repo
        os.environ["PLANTSEG_IMAGE_DIGEST"] = "sha256:test"
        out1 = work / "r1"
        with contextlib.redirect_stderr(io.StringIO()):
            CLI.run(cli_args(checkpoint=str(ck), out_dir=str(out1), run_id="r1"))
        s1 = verify_artifact(out1)
        rt = s1["run"]["eval_runtime"]
        check("R1 CLI.run on CPU writes four files and verify_artifact passes",
              sorted(p.name for p in out1.iterdir()) == sorted([*ARTIFACT_FILES, MANIFEST_NAME])
              and DATA_ROOT_LISTINGS == [], f"data-root listings={DATA_ROOT_LISTINGS}")
        check("R2 run.eval_runtime has exactly the 26 literal keys",
              set(rt) == RECORD_KEYS_LITERAL and len(RECORD_KEYS_LITERAL) == 26,
              str(sorted(set(rt) ^ RECORD_KEYS_LITERAL)))
        check("R3 CPU device fields; GPU fields null; policy not applied",
              rt["model_device"] == "cpu" and rt["model_tensor_devices"] == ["cpu"]
              and rt["input_devices"] == ["cpu"] and rt["gpu_name"] is None
              and rt["gpu_capability"] is None and rt["cudnn_version"] is None
              and rt["determinism_policy_applied"] is False
              and rt["cuda_initialized_before_policy"] is None
              and rt["eval_runtime_version"] == EVAL_RUNTIME_VERSION)
        check("R4 batch 2, workers 0, forward_batches 2, checkpoint iteration 4000, best 0.123",
              (rt["batch_size"], rt["num_workers"], rt["forward_batches"], rt["checkpoint_iteration"],
               rt["checkpoint_best_val_miou_all_class"]) == (2, 0, 2, 4000, 0.123), json.dumps(rt)[:200])

        os.environ["PLANTSEG_IMAGE_DIGEST"] = "  "
        out5 = work / "r5"
        with contextlib.redirect_stderr(io.StringIO()):
            CLI.run(cli_args(checkpoint=str(ck), out_dir=str(out5), run_id="r5"))
        rt5 = verify_artifact(out5)["run"]["eval_runtime"]
        digest_eq = []
        for v in (None, "", "  ", "sha256:ab"):
            if v is None:
                os.environ.pop("PLANTSEG_IMAGE_DIGEST", None)
            else:
                os.environ["PLANTSEG_IMAGE_DIGEST"] = v
            digest_eq.append(ER.image_digest() == train_e1._image_digest())

        def has_empty(o):
            if isinstance(o, str):
                return o == ""
            if isinstance(o, dict):
                return any(has_empty(v) for v in o.values())
            if isinstance(o, list):
                return any(has_empty(v) for v in o)
            return False
        check("R5 image digest recorded verbatim / blank -> null; == train_e1._image_digest; no \"\"",
              rt["image_digest"] == "sha256:test" and rt5["image_digest"] is None and all(digest_eq)
              and not has_empty(rt) and not has_empty(rt5), str(digest_eq))
        check("R6 parent determinism, TF32 and CUBLAS state unchanged by the CPU runs",
              parent_state() == before)

        real_load = ML.load_student_checkpoint

        def wrong_sha(path, **kw):
            m, info = real_load(path, **kw)
            return m, type(info)(path=info.path, sha256="f" * 64, num_classes=info.num_classes,
                                 iteration=info.iteration,
                                 best_val_miou_all_class=info.best_val_miou_all_class)
        ML.load_student_checkpoint = wrong_sha
        out7 = work / "r7"
        try:
            CLI.run(cli_args(checkpoint=str(ck), out_dir=str(out7), run_id="r7"))
            code7 = None
        except ER.EvalRuntimeError as e:
            code7 = e.code
        finally:
            ML.load_student_checkpoint = real_load
        check("R7 checkpoint bytes changed after validation -> refused, no out-dir",
              code7 == "checkpoint_changed_after_validation" and not out7.exists(), str(code7))
    finally:
        CLI.REPO = saved_repo
        undo()
        if saved_digest is None:
            os.environ.pop("PLANTSEG_IMAGE_DIGEST", None)
        else:
            os.environ["PLANTSEG_IMAGE_DIGEST"] = saved_digest

    has_cuda = torch.cuda.is_available()
    out8 = work / "r8"
    r8 = r_child(work, repo=repo, ck=ck, out=out8)
    if has_cuda:
        rt8 = r8.get("rt") or {}
        ok8 = (r8["err"] is None and r8["out_exists"] and r8["five"] == B6_ON
               and str(rt8.get("model_device", "")).startswith("cuda:")
               and rt8.get("input_devices") == [rt8.get("model_device")]
               and rt8.get("determinism_policy_applied") is True)
    else:
        ok8 = (r8["err"] is not None and r8["five"] == B6_ON and r8["init"] is False
               and r8["model"] == [["model", False]] and not r8["out_exists"]
               and "load_student_checkpoint" in r8["tb_funcs"] and r8["listings"] == [])
    check("R8 --device cuda child: policy applied before the load (CPU stack: fails at the load, "
          "CUDA uninitialised, no out-dir)", ok8, f"cuda={has_cuda} err={str(r8['err'])[:90]}")
    r9 = r_child(work, repo=repo, ck=ck, out=work / "r9",
                 patch="torch.cuda.is_initialized = lambda: True")
    check("R9 CUDA already initialised -> refused before any construction",
          r9["code"] == "cuda_initialized_before_policy" and r9["dataset"] == [] and r9["model"] == []
          and not r9["out_exists"], str(r9["code"]))
    dirty = make_fixture_repo(work / "fx_dirty_r", governed_dirty=True)
    r10 = r_child(work, repo=dirty, ck=ck, out=work / "r10", status="official")
    check("R10 refused official request (governed-dirty) applies nothing",
          r10["err"] is not None and r10["err"].startswith("ArtifactRequestError")
          and r10["five"] == p1["five"] == B6_OFF and r10["dataset"] == [], str(r10["err"])[:90])


# ------------------------------------------------------------------------------------------ X
def section_x() -> None:
    saved = sys.modules.get("torch.utils.deterministic", "__absent__")
    sys.modules["torch.utils.deterministic"] = None          # import now raises ImportError
    try:
        simulated = ER.fill_uninitialized_memory_state()
        rec = cpu_record()
    finally:
        if saved == "__absent__":
            del sys.modules["torch.utils.deterministic"]
        else:
            sys.modules["torch.utils.deterministic"] = saved
    check("X1 torch.utils.deterministic absent -> fill_uninitialized_memory null, key set still 26",
          simulated is None and rec["fill_uninitialized_memory"] is None
          and set(rec) == RECORD_KEYS_LITERAL)
    present = importlib.util.find_spec("torch.utils.deterministic") is not None
    live = ER.fill_uninitialized_memory_state()
    check("X2 live torch: null exactly when torch.utils.deterministic is absent",
          (live is None) == (not present) and (live is None or isinstance(live, bool)),
          f"torch {torch.__version__}: module {'present' if present else 'ABSENT (null path)'}, value {live!r}")
    fwd = ER.ModelDeviceForward(torch.device("cpu"))
    fwd.input_devices.add("cpu")
    info = type("I", (), {"iteration": 5, "best_val_miou_all_class": float("inf")})()
    rec_inf = ER.build_eval_runtime_record(
        model=torch.nn.Conv2d(3, C, 1), model_device=torch.device("cpu"), fwd=fwd, batch_size=2,
        forward_batches=1, policy_applied=False, policy_info=None, ckpt_info=info,
        warn_summary={"eval_warnings": [], "eval_warnings_truncated": False, "nondeterministic_alert_count": 0})
    check("X3 a non-finite stored best is recorded as null (section 5.2), so the record stays writable",
          rec_inf["checkpoint_best_val_miou_all_class"] is None and rec_inf["checkpoint_iteration"] == 5)


# ------------------------------------------------------------------------------------------ C
def dl17_summary(**over) -> dict:
    gt = [1000] * C
    gt[0] = CMP.DL17_GT_SUPPORT - 1000 * (C - 1)
    rt = {"eval_runtime_version": EVAL_RUNTIME_VERSION, "determinism_policy_applied": True,
          "model_device": "cuda:0", "model_tensor_devices": ["cuda:0"], "input_devices": ["cuda:0"],
          "batch_size": 16, "num_workers": 0, "forward_batches": 53,
          "checkpoint_iteration": CMP.DL17_CHECKPOINT_ITER,
          "checkpoint_best_val_miou_all_class": CMP.DL17_REFERENCE_MIOU,
          "image_digest": CMP.DL17_IMAGE_DIGEST, "gpu_name": "NVIDIA A40"}
    s = {"schema_version": SCHEMA_VERSION, "metric_protocol": METRIC_PROTOCOL,
         "run": {"run_id": "dl17_run1", "timestamp_utc": "2026-10-01T00:00:00Z",
                 "artifact_status": "provisional", "checkpoint_sha256": CMP.DL17_CHECKPOINT_SHA256,
                 "eval_runtime": rt},
         "dataset": {"split": "val", "expected_rows": 846, "actual_rows": 846},
         "dataset_level": {"all_class_miou": CMP.DL17_REFERENCE_MIOU},
         "per_class": {"gt_support": gt}}
    for dotted, value in over.items():
        node = s
        keys = dotted.split("__")
        for k in keys[:-1]:
            node = node[k]
        if value is DELETE:
            node.pop(keys[-1], None)
        else:
            node[keys[-1]] = value
    return s


DELETE = object()


def write_fixture_artifact(out: Path, summary: dict, *, rows: str | None = None, npz_tp=None) -> Path:
    import hashlib
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8", newline="\n")
    (out / "per_image.jsonl").write_text(
        rows if rows is not None else '{"image_id": "a", "miou": 0.5}\n{"image_id": "b", "miou": 0.25}\n',
        encoding="utf-8", newline="\n")
    tp = np.array([1, 2, 3], dtype=np.int64) if npz_tp is None else npz_tp
    np.savez_compressed(out / "sufficient_stats.npz", tp=tp, gt=np.array([4, 5, 6], dtype=np.int64),
                        manifest_ids=np.array(["a", "b"], dtype=np.str_))
    lines = "".join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n" for n in ARTIFACT_FILES)
    (out / MANIFEST_NAME).write_text(lines, encoding="utf-8", newline="\n")
    return out


def cmp_rc(a: Path, b: Path, *extra: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            rc = CMP.main([*extra, str(a), str(b)])
        except SystemExit as e:
            rc = e.code
    return rc, buf.getvalue()


def section_c(work: Path) -> None:
    check("C1 comparator constants (Q6)",
          CMP.DL17_REFERENCE_MIOU == 0.36314016580581665 and CMP.DL17_CHECKPOINT_ITER == 80000
          and CMP.DL17_CHECKPOINT_SHA256
          == "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03"
          and CMP.DL17_IMAGE_DIGEST
          == "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
          and CMP.DL17_GT_SUPPORT == 159279104 and CMP.DL17_GPU_NAME_SUBSTRING == "A40"
          and (CMP.EXIT_PASS, CMP.EXIT_FAIL, CMP.EXIT_VALIDITY) == (0, 1, 2))
    root = work / "cmp"
    a = write_fixture_artifact(root / "a", dl17_summary())
    b = write_fixture_artifact(root / "b", dl17_summary())
    rc, out = cmp_rc(a, b)
    check("C2 identical valid pair -> 0 (full mode)", rc == 0 and "RESULT: PASS" in out, out[-120:])
    rc, out = cmp_rc(a, b, "--identity-only")
    check("C3 identical pair -> 0 (--identity-only), both all_class_miou values printed",
          rc == 0 and out.count(repr(CMP.DL17_REFERENCE_MIOU)) >= 2, out[:160])
    b2 = write_fixture_artifact(root / "b2", dl17_summary(run__run_id="dl17_run2",
                                                          run__timestamp_utc="2026-10-01T01:02:03Z"))
    rc, out = cmp_rc(a, b2)
    check("C4 a pair differing only in run_id and timestamp_utc -> 0", rc == 0, out[-120:])
    b3 = write_fixture_artifact(root / "b3", dl17_summary(),
                                rows='{"image_id": "a", "miou": 0.5}\n{"image_id": "b", "miou": 0.35}\n')
    rc, out = cmp_rc(a, b3)
    check("C5 one per_image.jsonl byte changed -> 1", rc == 1 and "per_image.jsonl" in out, out[-160:])
    b4 = write_fixture_artifact(root / "b4", dl17_summary(), npz_tp=np.array([1, 2, 4], dtype=np.int64))
    rc, out = cmp_rc(a, b4)
    check("C6 one NPZ value changed -> 1", rc == 1 and "NPZ tp" in out, out[-160:])
    above = CMP.DL17_REFERENCE_MIOU + 1.01e-4
    a5 = write_fixture_artifact(root / "a5", dl17_summary(dataset_level__all_class_miou=above))
    b5 = write_fixture_artifact(root / "b5", dl17_summary(dataset_level__all_class_miou=above))
    rc, out = cmp_rc(a5, b5)
    check("C7 |delta| just above 1e-4 -> 1", rc == 1 and "outside the DL-17 band" in out, out[-160:])
    inband = CMP.DL17_REFERENCE_MIOU - 6.49e-5
    a6 = write_fixture_artifact(root / "a6", dl17_summary(dataset_level__all_class_miou=inband))
    b6 = write_fixture_artifact(root / "b6", dl17_summary(dataset_level__all_class_miou=inband))
    rc, out = cmp_rc(a6, b6)
    check("C8 |delta| in (1e-6, 1e-4] -> 0 and recorded", rc == 0 and "RECORDED:" in out, out[-160:])

    gt_off = dl17_summary()["per_class"]["gt_support"]
    gt_off[1] += 1
    violations = [
        ("eval_runtime missing", {"run__eval_runtime": DELETE}),
        ("determinism_policy_applied false", {"run__eval_runtime__determinism_policy_applied": False}),
        ("model_device cpu", {"run__eval_runtime__model_device": "cpu",
                              "run__eval_runtime__input_devices": ["cpu"]}),
        ("input_devices != model_device", {"run__eval_runtime__input_devices": ["cpu"]}),
        ("batch_size 2", {"run__eval_runtime__batch_size": 2}),
        ("num_workers 2", {"run__eval_runtime__num_workers": 2}),
        ("forward_batches 52", {"run__eval_runtime__forward_batches": 52}),
        ("actual_rows 845", {"dataset__actual_rows": 845}),
        ("checkpoint_iteration 79999", {"run__eval_runtime__checkpoint_iteration": 79999}),
        ("stored best differs", {"run__eval_runtime__checkpoint_best_val_miou_all_class": 0.36}),
        ("image digest differs", {"run__eval_runtime__image_digest": "sha256:" + "0" * 64}),
        ("sum(gt_support) off by one", {"per_class__gt_support": gt_off}),
        ("checkpoint_sha256 differs", {"run__checkpoint_sha256": "0" * 64}),
        ("gpu_name not A40", {"run__eval_runtime__gpu_name": "NVIDIA RTX A6000"}),
    ]
    for i, (label, over) in enumerate(violations):
        va = write_fixture_artifact(root / f"va{i}", dl17_summary(**over))
        vb = write_fixture_artifact(root / f"vb{i}", dl17_summary(**over))
        rc, out = cmp_rc(va, vb)
        check(f"C9.{i + 1:02d} validity violated ({label}) -> 2",
              rc == 2 and "VALIDITY" in out, out[-140:])
    rc, _ = cmp_rc(a, b, "--no-such-flag")
    rc_help, _ = cmp_rc(a, b, "--help")
    rc_same, _ = cmp_rc(a, a)
    check("C10 a malformed command line, --help, or the same directory twice exit 3, never 0 or 2",
          rc == 3 and rc_help == 3 and rc_same == 3, f"rc={rc}/{rc_help}/{rc_same}")
    bad = write_fixture_artifact(root / "tampered", dl17_summary())
    (bad / "per_image.jsonl").write_text("tampered\n", encoding="utf-8", newline="\n")
    rc, out = cmp_rc(a, bad)
    check("C11 an artifact failing verify_artifact fails identity -> 1",
          rc == 1 and "verify_artifact(B) failed" in out, out[-140:])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "compare_eval_artifacts.py"),
                           str(a), str(b)], cwd=str(REPO), env=env, capture_output=True, text=True,
                          timeout=300)
    proc2 = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "compare_eval_artifacts.py"),
                            str(root / "va1"), str(root / "vb1")], cwd=str(REPO), env=env,
                           capture_output=True, text=True, timeout=300)
    check("C12 process exit codes: valid identical pair 0, validity violation 2",
          proc.returncode == 0 and proc2.returncode == 2, f"{proc.returncode}/{proc2.returncode}")
    b13 = write_fixture_artifact(root / "b13", dl17_summary(run__artifact_status="smoke"))
    rc, out = cmp_rc(a, b13)
    check("C13 a summary.json difference outside run_id/timestamp_utc -> 1",
          rc == 1 and "summary.json differs" in out, out[-140:])
    rc, out = cmp_rc(root / "va0", root / "vb0", "--identity-only")
    check("C14 --identity-only ignores the validity rules (an identical but invalid pair -> 0)",
          rc == 0 and "RESULT: IDENTICAL" in out, out[-120:])
    rc, out = cmp_rc(a, b3, "--identity-only")
    check("C15 --identity-only on a non-identical pair -> 1", rc == 1 and "NOT IDENTICAL" in out, out[-120:])
    b16 = write_fixture_artifact(root / "b16", dl17_summary(run__eval_runtime__batch_size=2),
                                 rows='{"image_id": "a", "miou": 0.5}\n{"image_id": "b", "miou": 0.35}\n')
    rc, out = cmp_rc(a, b16)
    check("C16 validity is judged before identity (B invalid AND different -> 2)",
          rc == 2 and "VALIDITY" in out and "IDENTITY" not in out, out[-140:])
    a17 = write_fixture_artifact(root / "a17", dl17_summary(dataset_level__all_class_miou="n/a"))
    b17 = write_fixture_artifact(root / "b17", dl17_summary(dataset_level__all_class_miou="n/a"))
    rc, out = cmp_rc(a17, b17)
    check("C17 an unexpected error is RESULT: ERROR with exit 3, never PASS/FAIL/VALIDITY",
          rc == 3 and "RESULT: ERROR" in out, out[-140:])


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="smoke_eval_determinism_"))
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  workdir={work}")
    try:
        p1 = None
        try:
            p1 = section_p(work)
        except Exception as e:                            # noqa: BLE001 -- the section fails, visibly
            check("P section completed", False,
                  f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-400:]}")
        for label, fn in (("G/D", section_g_d), ("W", section_w),
                          ("A", lambda: section_a(work)),
                          ("R", lambda: section_r(work, p1 or {"five": B6_OFF})),
                          ("X", section_x), ("C", lambda: section_c(work))):
            try:
                fn()
            except Exception as e:                        # noqa: BLE001 -- the section fails, visibly
                check(f"{label} section completed", False,
                      f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-400:]}")
    finally:
        rm_tree(work)
    width = max(len(n) for n, _, _ in CHECKS)
    for name, verdict, detail in CHECKS:
        print(f"  [{verdict}] {name:<{width}}  {detail}")
    n_ok = sum(1 for _, v, _ in CHECKS if v == "PASS")
    print(f"SUMMARY  {n_ok}/{len(CHECKS)} checks passed")
    ok_all = n_ok == len(CHECKS)
    print(f"RESULT: {'EVAL-DET OK' if ok_all else 'EVAL-DET FAILED'} ({n_ok}/{len(CHECKS)})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
