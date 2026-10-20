#!/usr/bin/env python3
"""Lane L-AM3, B-2: E6-KD's trainer (src/quant/qat_kd.py, scripts/run_e6kd.py) and its check-run-meta
(scripts/e6kd_eval.py) on synthetic data.

The real student runs E6-KD through run_qat_kd in mode "smoke" on Q2's 32 synthetic 64x64 TRAIN images (batch 16,
drop_last: 2 steps per epoch, so the smoke's ramp is 2 steps) and 8 VAL images. The teacher is the harness's stub
SegNeXt (scripts/invariance_harness.py: 8x8 logits and a 4x4 Stage-3 feature at 64x64, through the real M4-KD
stream), wrapped as the thesis teacher is (SegNeXtTeacherAdapter in FrozenTeacher).

The launch goes through main() on synthetic fixtures: E3 seed 42's parent run (its best checkpoint, best.json, run_meta
and run_end rows, and projection.pt written by src/distill/cwd_projection.py save_projection), the λ and α selections
and the trigger records committed in a temp git repository, clip_selection.json and a teacher file. The trainer then
runs in mode "smoke" on the CPU in place of the CUDA run. The cloud has no GPU, no data and no teacher of record, so
the gates after the CUDA gate are reached through the smokes' stubs (LB34; pass_patches): torch.cuda reports an
uninitialised device; PLANTSEG_IMAGE_DIGEST is the KD image's; the fingerprint child returns the fingerprint of record;
the data root passes the TRAIN/VAL isolation; the code identity is clean at --expect-head; the teacher of record's
sha256 is the fixture file's and its identity check passes; the teacher is built by the stub builder. d4 and the
determinism checks are bitwise claims on the CPU only (LB33). No dataset, no checkpoint, no download; every file goes
to a temp directory outside the repository.

Sections: d2, clip, kd, d3, d4, records, aborts, gates, runs, routing, sl1, profile. --only runs named cases
(scripts/smoke_e6kd_mutations.py). No check name, case name, file or folder holds the token SL-1 forbids; the name
checks use paths that are never created.

    python -B scripts/smoke_e6kd_runner.py [--sections S,...] [--only CASE,...]

Ends with one RESULT line; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import safe_tmpdir  # noqa: E402  (imports no repository code)

TMP = safe_tmpdir("smoke_e6kd_runner_")                       # a temp path SL-1's name check and VAL's both pass
os.environ["PLANTSEG_DATA_ROOT"] = str(TMP / "absent_data_root")   # before configs.data is first imported
os.environ.pop("PLANTSEG_IMAGE_DIGEST", None)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import scripts.e6kd_eval as EV  # noqa: E402
import src.data.isolation as ISO  # noqa: E402
import src.quant.runner as RUNNER  # noqa: E402
from scripts.invariance_harness import build_stub_segnext  # noqa: E402
from src.distill import teacher as TT  # noqa: E402
from src.distill.cwd_projection import PROJECTION_FILE, PROJECTION_GROUP, save_projection, split_optimizer_state  # noqa: E402
from src.distill.segnext_teacher import SegNeXtTeacherAdapter, TeacherStateDictMismatch  # noqa: E402
from src.eval import artifacts as ART  # noqa: E402
from src.eval import stage_artifacts as SA  # noqa: E402
from src.models.student import build_student  # noqa: E402
from src.quant import e6kd_trigger as TRG  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant import qat_kd as KD  # noqa: E402
from src.quant.checkpoint import CWD_PROJECTION_KEY, CWD_PROJECTION_PREFIXES  # noqa: E402
from src.quant.qconfig import QuantBackendUnavailable  # noqa: E402
from src.quant.stages import QUANT_STAGES, resolve_quant_stage  # noqa: E402
from src.training import losses as L  # noqa: E402
from src.training import train_distill as TD  # noqa: E402

NC = 116
SEED = 42
STAGE = resolve_quant_stage(KD.STAGE_KEY)
E3W = {"logit_kd": 0.5, "cwd_feat": 25.0, "cwd_logit": 3.0}        # E3 seed 42's of record: λ 0.5, α 25, β 3
W = {"logit_kd": 0.25, "cwd_feat": 12.5, "cwd_logit": 1.5}          # AM-3: exactly 0.5x
ZERO = {t: 0.0 for t in KD.TERMS}
GRIDS = {"logit_kd_grid": "smoke", "cwd_feat_grid": "smoke", "cwd_logit_grid": "smoke"}
E3REC = {"e3_weights": dict(E3W), "grids": dict(GRIDS)}
SOURCE = {"sha256": "0" * 64, "path": None, "bytes": None}
PD = KD.preflight_distill()
KD_DIGEST = PD.KD_IMAGE_DIGEST
HEAD = "a" * 40
PARENT_ID = "e3_s42_lambda0.5_alpha25_80k_a1"
CKPT_NAME = "e3_student_best_iter76000.pt"
POD_TEACHER = "/workspace/teacher/iter_24000.pth"            # the pod's path in the profile rows; never created here
NEVER_OBSERVED = sorted(f"features.{i}.skip_add.activation_post_process" for i in (2, 4, 7, 11, 13))
ROW_KEYS = ("epoch", "step", "epoch_step", "lr", "loss", "ce", "dice", "grad_norm", "clipped", "bn_frozen",
            "observers_enabled", "batch_sha256", "state_nonfinite")
VAL_KEYS = ("epoch", "step", "all_class_miou", "disease_only_miou_PROVISIONAL", "val_batches", "val_total_px")
VOLATILE = ("wall_clock", "iter_seconds", "val_seconds")
FLAG_KEYS = ("state_finite", "state_nonfinite", "nonfinite_since_step")
SECTIONS = ("d2", "clip", "kd", "d3", "d4", "records", "aborts", "gates", "runs", "routing", "sl1", "profile")
# the run_meta row of an E6-KD launch: Q2's QAT keys without cwd_projection_loaded, E6-KD's keys, and main()'s
EXPECTED_RUN_META_KEYS = sorted([
    "event", "wall_clock", "schema", "lane", "mode", "stage", "source_stage", "seed", "run_id", "out_dir", "parent",
    "source_checkpoint_sha256", "source_checkpoint", "source_checkpoint_bytes", "clip_norm", "clip_source", "epochs",
    "batch_size", "drop_last", "gradient_accumulation", "ema", "distillation", "steps_per_epoch", "total_steps",
    "train_images", "val_images", "loader", "num_workers", "val_batches_cap", "persistent_workers", "scheduler",
    "optimizer", "lr", "momentum", "weight_decay", "nesterov", "dampening", "bn_freeze_epoch", "obs_freeze_epoch",
    "bn_frozen_from_epoch", "obs_frozen_from_epoch", "bn_freeze_after_step", "obs_freeze_after_step", "objective",
    "class_weights_sha256", "ignore_index", "engine", "qconfig", "observers", "observer_census", "fake_quant_modules",
    "never_observed_modules", "observer_freeze_scope", "fused_convbn", "unfused_batchnorm", "tf32", "determinism",
    "cuda_initialized_at_seed", "device", "cuda_available", "gpu_name", "torch", "torchvision", "numpy", "python",
    "image_digest", "host", "git_head", "code_clean_at_head", "code_files_sha256",
    # E6-KD (run_qat_kd)
    "e6kd", "weight_factor", "descriptive", "holm_family", "terms", "e3_weights", "weights", "T_logit", "T_cwd",
    "cwd_C", "logit_kd_semantics", "grids", "ramp_iters", "ramp_rule", "projection_source", "projection_params",
    "projection_fake_quant_modules", "projection_in_student", "teacher_provenance", "teacher_nmf", "teacher_mock",
    "abort_rule",
    # main()
    "clip_selection", "u4_pilot", "e6_selections", "data_root", "isolation_counts", "expect_head",
    "expect_source_sha256", "trigger", "trigger_sha256", "trigger_binding", "teacher_identity", "kd_image"])
results: list[tuple[str, bool, str]] = []
CASES: list[tuple[str, str, object]] = []


# ---------------------------------------------------------------- the smoke's own helpers
def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def case(section: str):
    def deco(fn):
        CASES.append((fn.__name__, section, fn))
        return fn
    return deco


def quiet(*_a, **_k) -> None:
    return None


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p: Path, doc) -> None:
    p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def code_of(fn, *a, **kw) -> str | None:
    """fn's refusal or STOP code; None when it returns; "raised:<type>" for any other exception, so a missing
    refusal fails its own named check instead of ending the section."""
    try:
        fn(*a, **kw)
    except (Q.QATRefused, Q.QATStop) as e:
        return e.code
    except Exception as e:                                       # noqa: BLE001 -- reported in the check's detail
        return f"raised:{type(e).__name__}"
    return None


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def patches(table: dict):
    with contextlib.ExitStack() as stack:
        for (obj, name), value in table.items():
            stack.enter_context(patched(obj, name, value))
        yield


@contextlib.contextmanager
def env_set(values: dict):
    before = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def probe_recorder(path: Path):
    """os.stat, os.lstat, os.listdir, os.scandir and Path.iterdir record every call on `path` (or below it) while the
    block runs, and are restored after it: a name SL-1 forbids is refused before it is probed or listed."""
    root = os.path.abspath(path)
    seen: list[str] = []
    real = (os.stat, os.lstat, os.listdir, os.scandir, Path.iterdir)

    def hit(p) -> None:
        try:
            a = os.path.abspath(os.fsdecode(p))
        except TypeError:                                         # a file descriptor
            return
        if a == root or a.startswith(root + os.sep):
            seen.append(a)

    def stat(p, *a, **kw):
        hit(p)
        return real[0](p, *a, **kw)

    def lstat(p, *a, **kw):
        hit(p)
        return real[1](p, *a, **kw)

    def listdir(p=".", *a, **kw):
        hit(p)
        return real[2](p, *a, **kw)

    def scandir(p=".", *a, **kw):
        hit(p)
        return real[3](p, *a, **kw)

    def iterdir(self):
        hit(self)
        return real[4](self)
    os.stat, os.lstat, os.listdir, os.scandir, Path.iterdir = stat, lstat, listdir, scandir, iterdir
    try:
        yield seen
    finally:
        os.stat, os.lstat, os.listdir, os.scandir, Path.iterdir = real


def git(repo: Path, *args) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
           "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def argv_set(argv: list[str], flag: str, value: str) -> list[str]:
    """argv with flag's value replaced (the flag appended when absent)."""
    out = list(argv)
    if flag in out:
        out[out.index(flag) + 1] = value
        return out
    return out + [flag, value]


def argv_without(argv: list[str], *flags: str) -> list[str]:
    """argv without each named flag; a store_true flag alone, any other with its value."""
    switches = ("--real-run", "--confirm-real-run")
    out, i = [], 0
    while i < len(argv):
        if argv[i] in flags:
            i += 1 if argv[i] in switches else 2
            continue
        out.append(argv[i])
        i += 1
    return out


def rows_at(out: Path) -> list[dict]:
    tel = Path(out) / Q.TELEMETRY_NAME
    if not tel.is_file():
        return []
    return [json.loads(ln) for ln in tel.read_text(encoding="utf-8").splitlines() if ln.strip()]


def rows_of(r: dict, event: str) -> list[dict]:
    return [x for x in r["rows"] if x.get("event") == event]


def param_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensors_equal(a: dict, b: dict) -> bool:
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def checkpoint(out: Path, epoch: int) -> dict:
    return torch.load(Path(out) / Q.EPOCH_DIR / f"e{epoch:02d}.pt", map_location="cpu", weights_only=True)


# ---------------------------------------------------------------- fixtures
def make_loaders(seed: int = SEED, *, batch: int = 16, n_train: int = 32):
    """Q2's synthetic loaders (scripts/smoke_qat_runner.py make_loaders): 32 TRAIN images of 64x64 (batch 16,
    drop_last: 2 steps per epoch) and 8 VAL images, the TRAIN order from a generator seeded `seed`."""
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(32, 3, 64, 64, generator=g)
    y = torch.randint(0, NC, (32, 64, 64), generator=g)
    y[:, :4, :] = 255
    vx = torch.randn(8, 3, 64, 64, generator=g)
    vy = torch.randint(0, NC, (8, 64, 64), generator=g)
    train = DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=batch, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(vx, vy), batch_size=16, shuffle=False)
    return train, val


def student():
    torch.manual_seed(0)
    return build_student(pretrained=False)


def stub_segnext_builder(_path):
    """The smokes' teacher builder (load_frozen_teacher's `builder`): the harness's stub SegNeXt with its isolated
    NMF module, adapted as the thesis teacher is. The loader hashes the file; this builder never reads it."""
    return SegNeXtTeacherAdapter(build_stub_segnext())


def stub_teacher():
    return TT.FrozenTeacher(stub_segnext_builder(None))


def projection_fixture(seed: int = 7) -> nn.Conv2d:
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.utils.skip_init(nn.Conv2d, 160, 320, 1, bias=False)
    with torch.no_grad():
        p.weight.copy_(torch.randn(320, 160, 1, 1, generator=g) * 0.05)
    return p


def write_parent(d: Path, *, ckpt_over: dict | None = None, meta_over: dict | None = None,
                 proj_over: dict | None = None, proj_raw: bytes | None = None,
                 no_projection: bool = False) -> tuple[str, str | None]:
    """E3 seed 42's parent run as train_distill keeps it: the best checkpoint (projection-free student, stage E3,
    iter 76000), best.json, the run_meta row with E3's terms and weights, a run_end record, and projection.pt written by
    save_projection after one real optimizer step (so it holds a momentum buffer). Returns (the checkpoint's sha256,
    projection.pt's sha256 or None)."""
    d.mkdir(parents=True)
    torch.manual_seed(0)
    net = build_student(pretrained=False)
    torch.save({"stage": "E3", "iter": 76000, "num_classes": NC, "model_state_dict": net.state_dict(),
                **(ckpt_over or {})}, d / CKPT_NAME)
    write_json(d / "best.json", {"best_ckpt": f"/workspace/e3_ckpts/{PARENT_ID}/{CKPT_NAME}",
                                 "best_val_miou_all_class": 0.36334073543548584})
    meta = {"event": "run_meta", "stage": "E3", "mode": "real", "seed": 42, "lambda_logit": 0.5, "alpha_cwd": 25.0,
            "beta_cwd": TD.BETA_CWD_LOGIT, "terms": {t: True for t in KD.TERMS}, "T_logit": TD.T_LOGIT,
            "T_cwd": TD.T_CWD, "cwd_C": TD.CWD_C_FEAT, "logit_kd_semantics": KD.LOGIT_KD_SEMANTICS,
            "ramp_iters": 335, **GRIDS, **(meta_over or {})}
    (d / "e3_run_meta.jsonl").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    (d / "e3_telemetry.jsonl").write_text(json.dumps({"event": "run_end", "iter": 80000, "checks_passed": True})
                                          + "\n", encoding="utf-8")
    if no_projection:
        return sha_file(d / CKPT_NAME), None
    p = d / PROJECTION_FILE
    if proj_raw is not None:
        p.write_bytes(proj_raw)
    else:
        proj = projection_fixture(11)
        opt = torch.optim.SGD([{"params": list(net.parameters()), "name": "student"},
                               {"params": list(proj.parameters()), "name": PROJECTION_GROUP}], lr=0.01, momentum=0.9)
        for prm in list(net.parameters())[:1] + list(proj.parameters()):
            prm.grad = torch.ones_like(prm)
        opt.step()
        _rest, part = split_optimizer_state(opt.state_dict())
        save_projection(p, proj, part, stage="E3", it=76000, checkpoint=CKPT_NAME)
        if proj_over:
            doc = torch.load(p, map_location="cpu", weights_only=True)
            doc.update(proj_over)
            torch.save(doc, p)
    return sha_file(d / CKPT_NAME), sha_file(p)


def trigger_doc(g: dict, *, ck_sha: str | None = None, e3_val: float = 0.36334073543548584,
                e6_val: float = 0.33082112669944763, **over) -> dict:
    """A trigger record in e6kd_trigger.json's form, its decision following from its two values (B-1's
    drop_decision), bound to the fixture's E3 run, E6's clip of record and the selections; `over` replaces
    record keys (its "e3" and "e6" are the record's blocks, the two values are e3_val and e6_val)."""
    dec = TRG.drop_decision(e3_val, e6_val)
    doc = {"format": TRG.TRIGGER_FORMAT, "smoke_inputs": False, "seed": 42, "e3_val": e3_val, "e6_val": e6_val,
           "drop_pp": dec["drop_pp"], "triggered": dec["triggered"],
           "e3": {"run_id": PARENT_ID, "checkpoint_sha256": ck_sha or g["ck_sha"]},
           "e6": {"clip_norm": 5.0, "clip_selection_sha256": g["csha"]},
           "selections": {"lambda": {"sha256": g["lsha"]}, "alpha": {"sha256": g["asha"]}}}
    doc.update(over)
    return doc


def commit_trigger(g: dict, tag: str, doc: dict | None = None, *, raw: bytes | None = None,
                   commit: bool = True) -> tuple[str, str]:
    rel = f"trig/{tag}.json"
    data = raw if raw is not None else (json.dumps(doc if doc is not None else trigger_doc(g), indent=1)
                                        + "\n").encode("utf-8")
    (g["repo"] / rel).write_bytes(data)
    if commit:
        git(g["repo"], "add", "--", rel)
        git(g["repo"], "commit", "-q", "-m", tag)
    return rel, hashlib.sha256(data).hexdigest()


@functools.lru_cache(maxsize=None)
def gfx() -> dict:
    """The launch fixtures (the module docstring) and base argv that passes every gate under pass_patches."""
    g: dict = {}
    pdir = TMP / "parents" / "of_record" / PARENT_ID
    g["parent_dir"] = pdir
    g["ck_sha"], g["psha"] = write_parent(pdir)
    repo = TMP / "selrepo"
    (repo / "sel").mkdir(parents=True)
    (repo / "trig").mkdir()
    write_json(repo / "sel" / "lambda.json", {"format": Q.LAMBDA_SELECTION_FORMAT,
                                              "winner": {"lambda": 0.5, "run_id": "e3_s42_lambda0.5_80k_a1"}})
    write_json(repo / "sel" / "alpha.json", {"format": Q.ALPHA_SELECTION_FORMAT,
                                             "winner": {"alpha": 25.0, "run_id": PARENT_ID,
                                                        "ckpt_sha256": g["ck_sha"]}})
    git(repo, "init", "-q")
    git(repo, "add", "--", "sel/lambda.json", "sel/alpha.json")
    git(repo, "commit", "-q", "-m", "selections")
    g["repo"] = repo
    g["lsha"], g["asha"] = sha_file(repo / "sel" / "lambda.json"), sha_file(repo / "sel" / "alpha.json")
    clip = TMP / "clip_selection.json"
    write_json(clip, {"format": Q.CLIP_SELECTION_FORMAT, "winner": {"clip_norm": 5.0}})
    g["clip"], g["csha"] = clip, sha_file(clip)
    g["trigger_rel"], g["tsha"] = commit_trigger(g, "e6kd_trigger")
    tdir = TMP / "teacher"
    tdir.mkdir()
    tfile = tdir / "iter_24000.pth"
    tfile.write_bytes(b"the smokes' teacher file: hashed by the loader, never read (the stub builder builds)\n")
    g["teacher_file"], g["teacher_sha"] = tfile, sha_file(tfile)
    g["out_new"] = str(TMP / "e6kd_out_new")
    g["argv"] = ["--real-run", "--confirm-real-run", "--expect-head", HEAD, "--seed", "42", "--num-workers", "12",
                 "--grad-clip-norm", "5.0", "--clip-selection", str(clip), "--clip-selection-sha256", g["csha"],
                 "--lambda-selection", "sel/lambda.json", "--lambda-selection-sha256", g["lsha"],
                 "--alpha-selection", "sel/alpha.json", "--alpha-selection-sha256", g["asha"],
                 "--trigger", g["trigger_rel"], "--trigger-sha256", g["tsha"],
                 "--source-run-dir", str(pdir), "--expect-source-sha256", g["ck_sha"],
                 "--projection-sha256", g["psha"], "--teacher-ckpt", str(tfile),
                 "--teacher-ckpt-sha256", g["teacher_sha"], "--out-dir", g["out_new"], "--device", "cuda"]
    return g


def variant(tag: str, **kw) -> tuple[list[str], dict]:
    """A parent run named as the fixture's (so its run_id binds), in a folder of its own, written with `kw`
    (write_parent's), and a trigger record committed for its checkpoint; returns the base argv bound to them."""
    g = gfx()
    d = TMP / "parents" / tag / PARENT_ID
    ck_sha, psha = write_parent(d, **kw)
    rel, tsha = commit_trigger(g, f"for_{tag}", trigger_doc(g, ck_sha=ck_sha))
    argv = argv_set(argv_set(argv_set(argv_set(g["argv"], "--source-run-dir", str(d)), "--expect-source-sha256",
                                      ck_sha), "--trigger", rel), "--trigger-sha256", tsha)
    if psha is not None:
        argv = argv_set(argv, "--projection-sha256", psha)
    return argv, {"dir": d, "ck_sha": ck_sha, "psha": psha}


def fingerprint_of_record() -> dict:
    return {**PD.FINGERPRINT_OF_RECORD, "imported": {"numpy": ["numpy"], "PIL": ["Pillow"], "torch": ["torch"]},
            "import mmseg.apis": True, "import ftfy": True}


def pass_patches(g: dict) -> dict:
    """The smokes' stubs that let a CPU cloud host reach gates 9-16 (LB34; F-15), named here and in the report."""
    clean = {"git_head": HEAD, "code_clean_at_head": True, "code_files_sha256": {}}
    return {(torch.cuda, "is_available"): (lambda: True), (torch.cuda, "is_initialized"): (lambda: False),
            (PD, "_child_json"): (lambda argv, tag, timeout, env=None: (0, fingerprint_of_record(), "")),
            (ISO, "assert_trainval_only_root"): (lambda root, expected=None: {"counts": {"train": 5367, "val": 846}}),
            (KD, "code_identity"): (lambda: dict(clean)),
            (PD, "TEACHER_SHA256"): g["teacher_sha"],
            (PD, "teacher_identity_problems"): (lambda rec, ckpt: [])}


def gate(argv: list[str] | None = None, over: dict | None = None, *, env: dict | None = None,
         builder=stub_segnext_builder) -> str | None:
    """real_run_gates on argv (the base argv by default) under pass_patches with `over` laid on them: its refusal
    code, None when every gate passes, "usage" when argparse refuses the flags."""
    g = gfx()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            args = KD.build_parser(STAGE).parse_args(g["argv"] if argv is None else argv)
    except SystemExit:
        return "usage"
    with patches({**pass_patches(g), **(over or {})}), env_set({"PLANTSEG_IMAGE_DIGEST": KD_DIGEST, **(env or {})}):
        return code_of(KD.real_run_gates, args, STAGE, repo_root=g["repo"], teacher_builder=builder)


def real_gates(argv: list[str] | None = None) -> dict:
    """real_run_gates' result on argv under pass_patches (every gate passing)."""
    g = gfx()
    args = KD.build_parser(STAGE).parse_args(g["argv"] if argv is None else argv)
    with patches(pass_patches(g)), env_set({"PLANTSEG_IMAGE_DIGEST": KD_DIGEST}):
        return KD.real_run_gates(args, STAGE, repo_root=g["repo"], teacher_builder=stub_segnext_builder)


# ---------------------------------------------------------------- the trainer, observed
class Recorder(KD.E6KDHooks):
    """The trainer's own loop, observed through its hooks: digests after each step, both groups' lr, the outputs of
    the two fake-quant modules the taps read, the student's input, the VAL bracket; spy_teacher adds every teacher
    call and every stream start."""

    def __init__(self, projection: nn.Module | None = None):
        self.projection = projection
        self.params, self.states, self.lrs, self.student_inputs = [], [], [], []
        self.in_val = False
        self.fq: dict = {}
        self.teacher_calls: list[dict] = []
        self.begin_calls = 0
        self.projection_at_end: dict = {}
        self.start: dict = {}

    def on_start(self, prepared, optimizer, scheduler, run_meta):
        self.prepared, self.optimizer, self.run_meta = prepared, optimizer, run_meta
        self.start = {"optimizer_state_entries": len(optimizer.state),
                      "lrs": [g["lr"] for g in optimizer.param_groups],
                      "projection": None if self.projection is None else self.projection.weight.detach().clone()}
        mods = dict(prepared.named_modules())
        for key, name in (("c5", "features.15.skip_add.activation_post_process"),
                          ("head", "head.ff_add.activation_post_process")):
            mods[name].register_forward_hook(lambda _m, _i, out, key=key: self.fq.__setitem__(key, out))
        prepared.register_forward_pre_hook(
            lambda _m, inp: None if self.in_val else self.student_inputs.append(inp[0].data_ptr()))

    def after_step(self, step, prepared, row):
        self.params.append(param_digest(prepared))
        self.states.append(Q.state_digest(prepared))
        self.lrs.append([g["lr"] for g in self.optimizer.param_groups])

    def before_val(self, epoch, prepared):
        self.in_val = True

    def after_val(self, epoch, prepared):
        self.in_val = False

    def after_epoch(self, epoch, prepared, checkpoint_path, row):
        if self.projection is not None:
            self.projection_at_end[epoch] = Q.state_digest(self.projection)


def spy_teacher(teacher, rec: Recorder):
    """Every call of the frozen teacher (inside VAL or not, the caller's CPU RNG state around it, eval mode, the input
    tensor) and every start of its NMF stream, recorded on `rec`."""
    real_forward, real_begin = teacher.forward, teacher.begin_nmf_stream

    def forward(x, *a, **kw):
        before = torch.get_rng_state()
        out = real_forward(x, *a, **kw)
        rec.teacher_calls.append({"in_val": rec.in_val, "rng_unchanged": bool(torch.equal(before, torch.get_rng_state())),
                                  "eval_mode": not teacher.teacher.training, "ptr": x.data_ptr(), "out": out,
                                  "kwargs": sorted(kw)})
        return out

    def begin(policy, seed=42):
        rec.begin_calls += 1
        return real_begin(policy, seed)
    teacher.forward = forward
    teacher.begin_nmf_stream = begin
    return teacher


@contextlib.contextmanager
def losses_spy(rec: Recorder | None = None):
    """train_distill.distillation_losses, as run_qat_kd calls it: its arguments, its result, the projection's weight
    at the call and whether c5 and head_logits are the very outputs of the fake-quant modules the taps read."""
    real = TD.distillation_losses
    calls: list[dict] = []

    def spy(**kw):
        total, parts = real(**kw)
        calls.append({**kw, "total": total.detach().clone(), "parts": dict(parts), "parts_order": list(parts),
                      "proj_weight": None if kw.get("projection") is None else kw["projection"].weight.detach().clone(),
                      "c5_is_tap": rec is not None and kw.get("c5") is rec.fq.get("c5"),
                      "head_is_tap": rec is not None and kw.get("head_logits") is rec.fq.get("head")})
        return total, parts
    with patched(TD, "distillation_losses", spy):
        yield calls


def run_kd(name: str, *, weights=None, clip: float = 1.0, inject=None, rec: Recorder | None = None, max_steps=None,
           teacher=None, projection=None, model=None, loaders=None, extra_meta=None, max_val_batches=None,
           mode: str = "smoke", device: str = "cpu", e3=None, clip_source: str = "smoke") -> dict:
    """run_qat_kd called directly (mode "smoke" by default) into TMP/runs/<name>."""
    out = TMP / "runs" / name
    teacher = teacher if teacher is not None else stub_teacher()
    projection = projection if projection is not None else projection_fixture()
    if rec is not None:
        rec.projection = projection
        spy_teacher(teacher, rec)
    r = {"out": out, "summary": None, "error": None, "rows": [], "rec": rec, "teacher": teacher,
         "projection": projection}
    try:
        r["summary"] = KD.run_qat_kd(
            mode=mode, model=model if model is not None else student(), teacher=teacher, projection=projection,
            weights=dict(W if weights is None else weights), e3=E3REC if e3 is None else e3, source_meta=SOURCE,
            out_dir=out, seed=SEED, clip_norm=clip, clip_source=clip_source, device=device, num_workers=0,
            loaders=make_loaders() if loaders is None else loaders, extra_meta=extra_meta, hooks=rec, inject=inject,
            max_steps=max_steps, max_val_batches=max_val_batches, log=quiet)
    except Exception as e:                                       # noqa: BLE001 -- the checks read it
        r["error"] = e
    r["rows"] = rows_at(out)
    return r


def run_e6(name: str, *, clip: float, rec: Recorder | None = None, max_steps=None) -> dict:
    """E6 (src/quant/qat.py run_qat, mode "smoke") on the same student, loaders and seed."""
    out = TMP / "runs" / name
    r = {"out": out, "summary": None, "error": None, "rows": [], "rec": rec}
    try:
        r["summary"] = Q.run_qat(stage="e6", mode="smoke", model=student(), source_meta=SOURCE, out_dir=out, seed=SEED,
                                 clip_norm=clip, clip_source="smoke", device="cpu", num_workers=0,
                                 loaders=make_loaders(), hooks=rec, max_steps=max_steps, log=quiet)
    except Exception as e:                                       # noqa: BLE001 -- the checks read it
        r["error"] = e
    r["rows"] = rows_at(out)
    return r


def main_run(name: str, *, max_steps=None, inject=None) -> dict:
    """scripts/run_e6kd.py's main() on the fixtures: real_run_gates (under pass_patches), the source loaded by
    Q.load_source, then the trainer in mode "smoke" on the CPU with Q2's loaders in place of the CUDA run (LB34),
    observed by a Recorder and the spies. Returns its exit code, its RESULT line, the run and what main() passed."""
    g = gfx()
    out = TMP / "runs" / name
    argv = argv_set(g["argv"], "--out-dir", str(out))
    gates = real_gates(argv)
    rec = Recorder(projection=gates["projection"])
    spy_teacher(gates["teacher"], rec)
    teacher_before = param_digest(gates["teacher"])
    real_run = KD.run_qat_kd
    passed: dict = {}

    def smoke_run(**kw):
        passed.update(kw)
        kw.update(mode="smoke", device="cpu", num_workers=0, loaders=make_loaders(), hooks=rec, inject=inject,
                  max_steps=max_steps, log=quiet)
        return real_run(**kw)
    buf = io.StringIO()
    with contextlib.ExitStack() as stack:
        calls = stack.enter_context(losses_spy(rec))
        stack.enter_context(patched(KD, "real_run_gates", lambda args, stage, **k: gates))
        stack.enter_context(patched(KD, "run_qat_kd", smoke_run))
        stack.enter_context(contextlib.redirect_stdout(buf))
        rc = KD.main(argv, teacher_builder=stub_segnext_builder)
    res = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    return {"rc": rc, "result": res[-1] if res else buf.getvalue()[-300:], "out": out, "rows": rows_at(out),
            "rec": rec, "calls": calls, "gates": gates, "passed": passed, "teacher_before": teacher_before}


@functools.lru_cache(maxsize=None)
def ms() -> dict:
    """The short launch: main() on the fixtures, 4 steps (2 epochs: the 2-step ramp, two VALs and two epoch ends)."""
    return main_run("main_short", max_steps=4)


@functools.lru_cache(maxsize=None)
def mfull() -> dict:
    """The complete launch: main() on the fixtures, 15 epochs."""
    return main_run("main_full")


@functools.lru_cache(maxsize=None)
def e6_short() -> dict:
    return run_e6("e6_short/run", clip=5.0, max_steps=4)


def meta_of(r: dict) -> dict:
    return r["rows"][0] if r["rows"] and r["rows"][0].get("event") == "run_meta" else {}


def train_rows(r: dict) -> list[dict]:
    return sorted(rows_of(r, "train"), key=lambda x: x["step"])


# ---------------------------------------------------------------- d2: the recipe, the weights and the projection
@case("d2")
def d2_weights_exactly_half():
    w = KD.e6kd_weights(E3W)
    gates = real_gates()
    ok = (w == W and all(type(v) is float for v in w.values()) and all(2.0 * w[t] == E3W[t] for t in KD.TERMS)
          and gates["weights"] == W and gates["e3"]["e3_weights"] == E3W and KD.WEIGHT_FACTOR == 0.5)
    check("d2_weights_exactly_half", ok, f"{w} from E3's {E3W}; the gates' {gates['weights']}")


@case("d2")
def d2_T_unchanged():
    m, e3 = meta_of(ms()), real_gates()["e3"]
    ok = (m.get("T_logit") == 4.0 and m.get("T_cwd") == 4.0 and m.get("cwd_C") == 320 and TD.T_LOGIT == 4.0
          and TD.T_CWD == 4.0 and TD.CWD_C_FEAT == 320 and e3["T_logit"] == 4.0 and e3["T_cwd"] == 4.0
          and e3["cwd_C"] == 320 and m.get("logit_kd_semantics") == KD.LOGIT_KD_SEMANTICS)
    check("d2_T_unchanged", ok, f"run_meta T_logit {m.get('T_logit')} T_cwd {m.get('T_cwd')} C {m.get('cwd_C')}")


@case("d2")
def d2_terms_all_on():
    r = ms()
    m, rows = meta_of(r), train_rows(r)
    stage_ok = all(c["stage"] is KD.E3_STAGE and all(c["stage"][t] for t in KD.TERMS) for c in r["calls"])
    ok = (m.get("terms") == {t: True for t in KD.TERMS} and len(rows) == 4 and stage_ok
          and all(isinstance(x.get(t), float) and math.isfinite(x[t]) for x in rows for t in KD.TERMS))
    check("d2_terms_all_on", ok, f"terms {m.get('terms')}; {len(rows)} rows; E3's stage in every call: {stage_ok}")


RECIPE_KEYS = ("epochs", "batch_size", "drop_last", "gradient_accumulation", "ema", "steps_per_epoch", "total_steps",
               "scheduler", "lr", "momentum", "weight_decay", "nesterov", "dampening", "bn_freeze_epoch",
               "obs_freeze_epoch", "bn_frozen_from_epoch", "obs_frozen_from_epoch", "bn_freeze_after_step",
               "obs_freeze_after_step", "class_weights_sha256", "ignore_index", "engine", "qconfig", "observers",
               "observer_census", "fake_quant_modules", "never_observed_modules", "observer_freeze_scope",
               "fused_convbn", "unfused_batchnorm", "seed", "clip_norm")
STUDENT_GROUP_KEYS = ("class", "params_in_group", "model_params", "group_holds_every_model_param", "lr", "initial_lr",
                      "momentum", "weight_decay", "dampening", "nesterov")


@case("d2")
def d2_e6_recipe_pins():
    mk, m6 = meta_of(ms()), meta_of(e6_short())
    diff = [k for k in RECIPE_KEYS if mk.get(k) != m6.get(k)]
    ok_ = mk.get("optimizer") or {}
    o6 = m6.get("optimizer") or {}
    gdiff = [k for k in STUDENT_GROUP_KEYS if ok_.get(k) != o6.get(k)]
    check("d2_e6_recipe_pins", mk and m6 and not diff and not gdiff and m6.get("distillation") is False
          and mk.get("distillation") is True, f"differ: {diff + gdiff}")


@case("d2")
def d2_three_term_ramps():
    seen = []
    real = TD.build_term_ramps

    def spy(stage, ramp_iters):
        out = real(stage, ramp_iters)
        seen.append((stage, ramp_iters, out))
        return out
    with patched(TD, "build_term_ramps", spy):
        r = run_kd("d2_ramps/run", max_steps=1)
    ok = (r["error"] is None and len(seen) == 1 and seen[0][0] is KD.E3_STAGE and seen[0][1] == 2
          and list(seen[0][2]) == list(KD.TERMS)
          and all(isinstance(v, TD.TermRamp) and v.ramp_iters == 2 for v in seen[0][2].values()))
    check("d2_three_term_ramps", ok, f"{[(s[1], list(s[2])) for s in seen]}; {r['error']!r}")


@case("d2")
def d2_ramp_335_unit():
    m = meta_of(ms())
    unit = (TD.distill_ramp(1, 335) == 0.0 and TD.distill_ramp(2, 335) == 1.0 / 334.0
            and TD.distill_ramp(335, 335) == 1.0 and TD.distill_ramp(336, 335) == 1.0 and Q.REAL_STEPS_PER_EPOCH == 335)
    ok = (unit and m.get("ramp_iters") == m.get("steps_per_epoch") == 2 and m.get("ramp_rule") == KD.RAMP_RULE
          and real_gates()["e3"]["ramp_iters"] == 335)
    check("d2_ramp_335_unit", ok, f"unit {unit}; smoke ramp_iters {m.get('ramp_iters')} / steps per epoch "
                                  f"{m.get('steps_per_epoch')}")


@case("d2")
def d2_smoke_ramp_two_steps():
    r = ms()
    rows = train_rows(r)
    want = [0.0, 1.0, 1.0, 1.0]
    ok = ([x.get("ramp") for x in rows] == want and len(r["calls"]) == 4
          and all(c["ramp"] == {t: v for t in KD.TERMS} for c, v in zip(r["calls"], want)))
    check("d2_smoke_ramp_two_steps", ok, f"rows {[x.get('ramp') for x in rows]}")


@case("d2")
def d2_distill_zero_at_step1():
    r = ms()
    s1 = next((x for x in r["rows"] if x.get("event") == "step1_checks"), {})
    rows = train_rows(r)
    names = ("distill_ramp_starts_at_zero", "distill_zero_at_step1", "sup_added_unscaled", "sup_positive_finite")
    ok = (s1.get("passed") is True and all(s1.get("checks", {}).get(k) is True for k in names) and rows
          and rows[0]["loss"] == rows[0]["sup"] and r["calls"] and float(r["calls"][0]["total"]) == 0.0)
    check("d2_distill_zero_at_step1", ok, f"step1_checks {s1.get('checks')}")


def f32(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float32)


@case("d2")
def d2_sup_unramped():
    r = ms()
    rows, calls = train_rows(r), r["calls"]
    bad = []
    for x, c in zip(rows, calls):
        sup = f32(x["ce"]) + f32(x["dice"])
        if float(sup) != x["sup"] or float(sup + c["total"]) != x["loss"]:
            bad.append(x["step"])
    check("d2_sup_unramped", len(rows) == len(calls) == 4 and not bad and rows[0]["sup"] > 0.0,
          f"steps where sup != ce + dice or loss != sup + distill: {bad}")


@case("d2")
def d2_two_optimizer_groups():
    r = ms()
    o = meta_of(r).get("optimizer") or {}
    opt = r["rec"].optimizer
    proj = r["gates"]["projection"]
    ok = (o.get("class") == "SGD" and o.get("param_groups") == 2
          and [x.get("name") for x in o.get("groups") or []] == ["student", PROJECTION_GROUP]
          and o.get("group_holds_every_model_param") is True and o.get("projection_group_holds_the_projection") is True
          and len(opt.param_groups) == 2 and opt.param_groups[1]["params"] == [proj.weight]
          and not KD.kd_optimizer_error(o))
    check("d2_two_optimizer_groups", ok, f"groups {o.get('groups')}")


@case("d2")
def d2_projection_group_values():
    o = meta_of(ms()).get("optimizer") or {}
    groups = o.get("groups") or [{}, {}]
    want = {"lr": 3e-4, "initial_lr": 3e-4, "momentum": 0.9, "weight_decay": 1e-4, "dampening": 0.0, "nesterov": False}
    ok = (len(groups) == 2 and all(groups[1].get(k) == v for k, v in want.items())
          and all(groups[0].get(k) == groups[1].get(k) for k in want)
          and (Q.LEARNING_RATE, Q.MOMENTUM, Q.WEIGHT_DECAY) == (3e-4, 0.9, 1e-4))
    check("d2_projection_group_values", ok, f"{groups}")


@case("d2")
def d2_projection_group_fresh():
    r = ms()
    o = meta_of(r).get("optimizer") or {}
    ok = o.get("projection_state_loaded") is False and r["rec"].start.get("optimizer_state_entries") == 0
    check("d2_projection_group_fresh", ok, f"projection_state_loaded {o.get('projection_state_loaded')}; optimizer "
                                           f"state entries at the start {r['rec'].start.get('optimizer_state_entries')}")


@case("d2")
def d2_cosine_on_both_groups():
    r = ms()
    lrs = r["rec"].lrs
    rows6 = train_rows(e6_short())
    ok = (len(lrs) == 4 and all(a == b for a, b in lrs) and [x["lr"] for x in train_rows(r)] == [x["lr"] for x in rows6]
          and r["rec"].start.get("lrs") == [Q.LEARNING_RATE, Q.LEARNING_RATE])
    check("d2_cosine_on_both_groups", ok, f"lr pairs after each step {lrs}")


@case("d2")
def d2_projection_weights_bitwise_from_file():
    r, g = ms(), gfx()
    doc = torch.load(g["parent_dir"] / PROJECTION_FILE, map_location="cpu", weights_only=True)
    w0 = r["rec"].start.get("projection")
    ok = w0 is not None and torch.equal(w0, doc["projection_state_dict"]["weight"]) and w0.dtype == torch.float32
    check("d2_projection_weights_bitwise_from_file", ok, "the run's starting projection against projection.pt's")


@case("d2")
def d2_projection_not_fake_quantized():
    r = ms()
    m, proj = meta_of(r), r["gates"]["projection"]
    prepared = r["rec"].prepared
    # by identity and by its 320x160 shape: the student's own LR-ASPP head has modules named high_proj
    inside = ([n for n, mod in prepared.named_modules() if mod is proj]
              + [n for n, p in prepared.named_parameters() if tuple(p.shape) == (320, 160, 1, 1)])
    ok = (m.get("projection_fake_quant_modules") == 0 and m.get("projection_in_student") is False
          and m.get("projection_params") == 51200 and isinstance(proj, nn.Conv2d)
          and getattr(proj, "qconfig", None) is None and proj.weight.dtype == torch.float32
          and not [n for n, mod in proj.named_modules() if isinstance(mod, Q._fq_base())]
          and not inside)
    check("d2_projection_not_fake_quantized", ok, f"projection fake-quant modules {m.get('projection_fake_quant_modules')}"
                                                  f"; in the prepared student: {inside[:3]}")


@case("d2")
def d2_projection_never_in_checkpoint():
    r = ms()
    bad = []
    for e in (1, 2):
        ck = checkpoint(r["out"], e)
        sd = ck["model_state_dict"]
        # the checkpoint contract's own test (src/quant/checkpoint.py) and the projection's shape; the student's
        # LR-ASPP head keys head.high_proj.* are its own
        bad += [k for k in sd if any(k == pre or k.startswith(pre + ".") for pre in CWD_PROJECTION_PREFIXES)]
        bad += [k for k, v in sd.items() if torch.is_tensor(v) and tuple(v.shape) == (320, 160, 1, 1)]
        bad += [k for k in ck if k == CWD_PROJECTION_KEY or "proj" in k.lower() or "cwd" in k.lower()]
    check("d2_projection_never_in_checkpoint", not bad and (r["out"] / Q.EPOCH_DIR / "e02.pt").is_file(), f"{bad[:5]}")


@case("d2")
def d2_projection_no_global_rng_draw():
    g = gfx()
    parent = Q.resolve_parent(STAGE, g["parent_dir"], g["ck_sha"], 42)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    proj, rec = KD.projection_of_record(parent, g["psha"])
    after = torch.get_rng_state()
    check("d2_projection_no_global_rng_draw", torch.equal(before, after) and rec["paired"] is True,
          "the global CPU RNG state is unchanged across projection_of_record")


@case("d2")
def d2_projection_trains():
    r, g = ms(), gfx()
    ends = rows_of(r, "epoch_end")
    from_file = Q.state_digest(KD.projection_of_record(Q.resolve_parent(STAGE, g["parent_dir"], g["ck_sha"], 42),
                                                       g["psha"])[0])
    shas = [x.get("projection_sha256") for x in ends]
    check("d2_projection_trains", len(shas) == 2 and len(set(shas)) == 2 and from_file not in shas,
          f"epoch_end projection_sha256 {[s[:12] if s else s for s in shas]}")


# ---------------------------------------------------------------- clip: two levels over student and projection
def grads_fixture(seed: int, proj_zero: bool) -> tuple[list, list]:
    g = torch.Generator().manual_seed(seed)
    s = [nn.Parameter(torch.randn(*shape, generator=g)) for shape in ((8, 3, 3, 3), (8,), (16, 8, 1, 1))]
    p = [nn.Parameter(torch.randn(320, 160, 1, 1, generator=g))]
    for x in s:
        x.grad = torch.randn(x.shape, generator=g) * 3.0
    for x in p:
        x.grad = torch.zeros_like(x) if proj_zero else torch.randn(x.shape, generator=g) * 0.5
    return s, p


def clones(params: list) -> list:
    out = []
    for x in params:
        y = nn.Parameter(x.detach().clone())
        y.grad = x.grad.detach().clone()
        out.append(y)
    return out


@case("clip")
def clip_two_level_equals_torch_when_projection_zero():
    s, p = grads_fixture(1, True)
    s2 = clones(s)
    total, n_s, n_p = KD.grad_norms(s, p)
    ref = torch.nn.utils.clip_grad_norm_(s2, max_norm=1.0)
    KD.apply_clip(s, p, total, 1.0)
    ok = (torch.equal(total, ref) and torch.equal(n_s, ref) and float(n_p) == 0.0
          and all(torch.equal(a.grad, b.grad) for a, b in zip(s, s2)) and float(ref) > 1.0)
    check("clip_two_level_equals_torch_when_projection_zero", ok, f"total {float(total)!r} torch {float(ref)!r}")


@case("clip")
def clip_includes_projection():
    s, p = grads_fixture(2, False)
    total, n_s, n_p = KD.grad_norms(s, p)
    hyp = math.hypot(float(n_s), float(n_p))
    unit = float(n_p) > 0.0 and abs(float(total) - hyp) <= 1e-6 * hyp and float(total) > float(n_s)
    rows = train_rows(ms())
    run_ok = (rows and rows[0]["grad_norm_projection"] == 0.0 and rows[0]["grad_norm"] == rows[0]["grad_norm_student"]
              and all(x["grad_norm_projection"] > 0.0
                      and abs(x["grad_norm"] - math.hypot(x["grad_norm_student"], x["grad_norm_projection"]))
                      <= 1e-6 * x["grad_norm"] for x in rows[1:]))
    check("clip_includes_projection", unit and run_ok, f"unit total {float(total)} vs hypot {hyp}; run rows "
                                                       f"{[(x['grad_norm_student'], x['grad_norm_projection']) for x in rows]}")


@case("clip")
def clip_structure_spy():
    calls, clip_calls = [], []
    real_norm, real_clip = torch._foreach_norm, torch.nn.utils.clip_grad_norm_

    def norm_spy(tensors, *a, **kw):
        calls.append(len(tensors))
        return real_norm(tensors, *a, **kw)

    def clip_spy(*a, **kw):
        clip_calls.append(1)
        return real_clip(*a, **kw)
    rec = Recorder()
    with patched(torch, "_foreach_norm", norm_spy), patched(torch.nn.utils, "clip_grad_norm_", clip_spy):
        r = run_kd("clip_spy/run", rec=rec, max_steps=2)
    # per step: one group norm over the student's gradients, then one over the projection's single gradient
    ok = (r["error"] is None and len(calls) == 4 and calls[0] == calls[2] > 1 and calls[1] == calls[3] == 1
          and not clip_calls)
    check("clip_structure_spy", ok, f"_foreach_norm list lengths {calls}; clip_grad_norm_ calls {len(clip_calls)}; "
                                    f"{r['error']!r}")


@case("clip")
def clip_coefficient_applied_to_both_groups():
    s, p = grads_fixture(3, False)
    s0, p0 = [x.grad.clone() for x in s], [x.grad.clone() for x in p]
    total, _n_s, _n_p = KD.grad_norms(s, p)
    coef = torch.clamp(0.5 / (total + 1e-6), max=1.0)
    KD.apply_clip(s, p, total, 0.5)
    scaled = all(torch.equal(x.grad, g0 * coef) for x, g0 in zip(s + p, s0 + p0))
    s, p = grads_fixture(3, False)
    total, _n_s, _n_p = KD.grad_norms(s, p)
    KD.apply_clip(s, p, total, 1e9)
    kept = all(torch.equal(x.grad, g0) for x, g0 in zip(s + p, s0 + p0))
    check("clip_coefficient_applied_to_both_groups", scaled and kept and float(coef) < 1.0,
          f"coefficient {float(coef)}; scaled {scaled}; unclipped kept {kept}")


# ---------------------------------------------------------------- kd: the three terms on the fake-quant taps
@case("kd")
def kd_inputs_are_fake_quantized_taps():
    calls = ms()["calls"]
    ok = (len(calls) == 4 and all(c["c5_is_tap"] and c["head_is_tap"] for c in calls)
          and all(tuple(c["c5"].shape) == (16, 160, 4, 4) and tuple(c["head_logits"].shape) == (16, NC, 8, 8)
                  and tuple(c["logits"].shape) == (16, NC, 64, 64) for c in calls))
    check("kd_inputs_are_fake_quantized_taps", ok,
          f"{len(calls)} calls; taps {[(c['c5_is_tap'], c['head_is_tap']) for c in calls]}")


def direct_terms(c: dict) -> dict:
    """The three terms by direct src/training/losses.py calls on the call's own tensors, at the values of record (T 4;
    the feature map's C 320, the logit map's own 116 channels)."""
    with torch.no_grad():
        t = c["teacher_out"]
        head, c5, mask = c["head_logits"], c["c5"], c["mask"]
        valid8 = L.downsample_validity(mask, tuple(head.shape[-2:]))
        valid16 = L.downsample_validity(mask, tuple(c5.shape[-2:]))
        return {"logit_kd": L.logit_kd_kl(head, t.logits, valid8, T=4.0),
                "cwd_feat": L.cwd_channelwise_kl(F.conv2d(c5, c["proj_weight"]), t.feat_s16, valid16, T=4.0,
                                                 channels_norm=320),
                "cwd_logit": L.cwd_channelwise_kl(head, t.logits, valid8, T=4.0)}


@case("kd")
def kd_terms_equal_direct_loss_calls():
    calls = ms()["calls"]
    bad = []
    for i, c in enumerate(calls, start=1):
        d = direct_terms(c)
        r = 0.0 if i == 1 else 1.0
        if (c["parts"] != {k: float(v) for k, v in d.items()} or c["lambda_logit"] != W["logit_kd"]
                or c["alpha"] != W["cwd_feat"] or c["beta"] != W["cwd_logit"] or c["ramp"] != {t: r for t in KD.TERMS}
                or c.get("targets") is not None):
            bad.append(i)
    check("kd_terms_equal_direct_loss_calls", len(calls) == 4 and not bad, f"steps that differ: {bad}")


@case("kd")
def kd_scaling_T2_and_C():
    calls, m = ms()["calls"], meta_of(ms())
    if len(calls) < 2:
        check("kd_scaling_T2_and_C", False, f"{len(calls)} calls")
        return
    c = calls[1]
    with torch.no_grad():
        t, head, mask = c["teacher_out"], c["head_logits"], c["mask"]
        valid = L.downsample_validity(mask, tuple(head.shape[-2:])).float()
        s_logp = torch.log_softmax(head / 4.0, dim=1)
        t_logp = torch.log_softmax(t.logits / 4.0, dim=1)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=1) * 16.0
        manual = float((kl * valid).sum() / valid.sum().clamp_min(1.0))
        s_feat = F.conv2d(c["c5"], c["proj_weight"])
        v16 = L.downsample_validity(mask, tuple(s_feat.shape[-2:]))
        c320 = float(L.cwd_channelwise_kl(s_feat, t.feat_s16, v16, T=4.0, channels_norm=320))
        c116 = float(L.cwd_channelwise_kl(s_feat, t.feat_s16, v16, T=4.0, channels_norm=116))
    lk = c["parts"]["logit_kd"]
    ok = (abs(manual - lk) <= 1e-5 * max(1.0, abs(lk)) and c["parts"]["cwd_feat"] == c320 and c320 != c116
          and abs(c116 * 116.0 / 320.0 - c320) <= 1e-5 * max(1.0, abs(c320))
          and (m.get("T_logit"), m.get("T_cwd"), m.get("cwd_C")) == (4.0, 4.0, 320))
    check("kd_scaling_T2_and_C", ok, f"logit KD {lk} vs T^2-scaled KL {manual}; CWD feature {c['parts']['cwd_feat']} "
                                     f"vs C=320 {c320} (C=116 would give {c116})")


@case("kd")
def kd_addition_order():
    calls = ms()["calls"]
    bad = []
    for i, c in enumerate(calls, start=1):
        d = direct_terms(c)
        r = 0.0 if i == 1 else 1.0
        total = d["logit_kd"].new_zeros(())
        total = total + r * W["logit_kd"] * d["logit_kd"]
        total = total + r * W["cwd_feat"] * d["cwd_feat"]
        total = total + r * W["cwd_logit"] * d["cwd_logit"]
        if c["parts_order"] != list(KD.TERMS) or not torch.equal(total, c["total"]):
            bad.append(i)
    check("kd_addition_order", len(calls) == 4 and not bad and list(KD.TERMS) == ["logit_kd", "cwd_feat", "cwd_logit"],
          f"steps whose total is not ((0 + logit_kd) + cwd_feat) + cwd_logit, bitwise: {bad}")


@case("kd")
def kd_teacher_on_wrong_grid_refused():
    t = TT.FrozenTeacher(TT.MockTeacher(NC, logits_stride=4))       # 16x16 logits against the head's 8x8
    r = run_kd("kd_grid/run", teacher=t, max_steps=2)
    last = r["rows"][-1] if r["rows"] else {}
    ok = (isinstance(r["error"], RuntimeError) and "resampled" in str(r["error"]) and not rows_of(r, "train")
          and last.get("event") == "run_abort" and last.get("rule") == KD.EXCEPTION_RULE)
    check("kd_teacher_on_wrong_grid_refused", ok, f"{r['error']!r}; last row {last.get('event')} {last.get('rule')}")


# ---------------------------------------------------------------- d3: the teacher in the loop
@case("d3")
def d3_stream_begun_once():
    r = ms()
    nmf = meta_of(r).get("teacher_nmf") or {}
    ok = (r["rec"].begin_calls == 1 and nmf.get("policy") == "M4-KD" and nmf.get("seed") == 42 and nmf.get("draws") == 0
          and len(rows_of(r, "epoch_end")) == 2)
    check("d3_stream_begun_once", ok, f"{r['rec'].begin_calls} stream start(s) over 2 epochs; {nmf}")


@case("d3")
def d3_one_teacher_forward_per_step():
    r = ms()
    train_calls = [c for c in r["rec"].teacher_calls if not c["in_val"]]
    check("d3_one_teacher_forward_per_step", len(train_calls) == len(train_rows(r)) == 4,
          f"{len(train_calls)} teacher calls for {len(train_rows(r))} steps")


@case("d3")
def d3_draws_equal_steps():
    ends = rows_of(ms(), "epoch_end")
    check("d3_draws_equal_steps", [x.get("teacher_nmf_draws") for x in ends] == [2, 4] == [x["step"] for x in ends],
          f"draws at the epoch ends {[x.get('teacher_nmf_draws') for x in ends]}")


@case("d3")
def d3_no_teacher_forward_in_val():
    r = ms()
    in_val = [c for c in r["rec"].teacher_calls if c["in_val"]]
    check("d3_no_teacher_forward_in_val", not in_val and len(rows_of(r, "val")) == 2,
          f"{len(in_val)} teacher calls inside VAL")


@case("d3")
def d3_no_reset_across_epochs():
    r = ms()
    draws = [x.get("teacher_nmf_draws") for x in rows_of(r, "epoch_end")]
    check("d3_no_reset_across_epochs", r["rec"].begin_calls == 1 and draws == sorted(draws) and len(set(draws)) == 2
          and draws[-1] == len([c for c in r["rec"].teacher_calls if not c["in_val"]]), f"draws {draws}")


@case("d3")
def d3_caller_rng_unchanged():
    calls = ms()["rec"].teacher_calls
    check("d3_caller_rng_unchanged", calls and all(c["rng_unchanged"] for c in calls),
          f"{sum(1 for c in calls if not c['rng_unchanged'])} of {len(calls)} calls moved the caller's CPU RNG")


@case("d3")
def d3_teacher_frozen():
    r = ms()
    s1 = next((x for x in r["rows"] if x.get("event") == "step1_checks"), {}).get("checks", {})
    t = r["gates"]["teacher"]
    ok = (s1.get("teacher_params_frozen") is True and s1.get("teacher_stayed_frozen") is True
          and s1.get("optimizer_excludes_teacher") is True and all(c["eval_mode"] for c in r["rec"].teacher_calls)
          and param_digest(t) == r["teacher_before"] and not any(p.requires_grad for p in t.parameters()))
    check("d3_teacher_frozen", ok, f"step-1 checks {[(k, s1.get(k)) for k in ('teacher_params_frozen', 'teacher_stayed_frozen')]}")


@case("d3")
def d3_teacher_fed_student_input():
    r = ms()
    s1 = next((x for x in r["rows"] if x.get("event") == "step1_checks"), {}).get("checks", {})
    t_ptrs = [c["ptr"] for c in r["rec"].teacher_calls if not c["in_val"]]
    ok = (s1.get("teacher_same_augmented_input") is True and t_ptrs == r["rec"].student_inputs and len(t_ptrs) == 4
          and all(c["kwargs"] == [] for c in r["rec"].teacher_calls))
    check("d3_teacher_fed_student_input", ok, f"student inputs {len(r['rec'].student_inputs)}, teacher inputs "
                                              f"{len(t_ptrs)}; equal {t_ptrs == r['rec'].student_inputs}")


# ---------------------------------------------------------------- d4: zero weights equal E6, on the CPU (LB33)
@functools.lru_cache(maxsize=None)
def d4_pair(clip: float, steps) -> tuple[dict, dict]:
    """E6 (run_qat) and E6-KD at weights 0, same student, loaders, seed and clip; both run directories are named
    "run" (so the checkpoints' run_id agrees)."""
    tag = f"d4_clip{clip}_{steps}".replace(".", "p")
    e6 = run_e6(f"{tag}_e6/run", clip=clip, rec=Recorder(), max_steps=steps)
    kd = run_kd(f"{tag}_kd/run", weights=ZERO, clip=clip, rec=Recorder(), max_steps=steps)
    return e6, kd


@functools.lru_cache(maxsize=None)
def d4_real() -> dict:
    """E6-KD at its real weights, clip 0.1, 4 steps."""
    return run_kd("d4_real/run", clip=0.1, rec=Recorder(), max_steps=4)


def same_rows(a: dict, b: dict, keys=ROW_KEYS) -> list:
    ra, rb = train_rows(a), train_rows(b)
    if len(ra) != len(rb) or not ra:
        return [("lengths", len(ra), len(rb))]
    return [(x["step"], k) for x, y in zip(ra, rb) for k in keys if x.get(k) != y.get(k)
            or type(x.get(k)) is not type(y.get(k))]


@case("d4")
def d4_four_steps_finite():
    r = d4_real()
    rows = train_rows(r)
    ok = (r["error"] is None and len(rows) == 4 and not rows_of(r, "run_abort")
          and all(math.isfinite(x["loss"]) and math.isfinite(x["grad_norm"]) for x in rows))
    check("d4_four_steps_finite", ok, f"{len(rows)} rows; {r['error']!r}")


@case("d4")
def d4_kd_columns_present():
    rows = train_rows(d4_real())
    keys = (*KD.TERMS, "ramp", "sup", "grad_norm_student", "grad_norm_projection")
    ok = (len(rows) == 4 and all(isinstance(x.get(k), float) for x in rows for k in keys)
          and all(x["grad_norm_projection"] > 0.0 for x in rows[1:]))
    check("d4_kd_columns_present", ok, f"{[sorted(k for k in keys if not isinstance(x.get(k), float)) for x in rows]}")


@case("d4")
def d4_zero_weights_equal_e6_rows_clip_0_1():
    e6, kd = d4_pair(0.1, None)
    bad = same_rows(e6, kd)
    check("d4_zero_weights_equal_e6_rows_clip_0_1", e6["error"] is None and kd["error"] is None and not bad
          and len(train_rows(kd)) == 30, f"{bad[:5]}; {e6['error']!r} {kd['error']!r}")


@case("d4")
def d4_zero_weights_equal_e6_states_clip_0_1():
    e6, kd = d4_pair(0.1, None)
    ok = (len(kd["rec"].states) == 30 and e6["rec"].states == kd["rec"].states
          and e6["rec"].params == kd["rec"].params)
    first = next((i + 1 for i, (a, b) in enumerate(zip(e6["rec"].states, kd["rec"].states)) if a != b), None)
    check("d4_zero_weights_equal_e6_states_clip_0_1", ok, f"first differing step {first}")


@case("d4")
def d4_zero_weights_equal_e6_val_clip_0_1():
    e6, kd = d4_pair(0.1, None)
    v6, vk = rows_of(e6, "val"), rows_of(kd, "val")
    bad = [(x["epoch"], k) for x, y in zip(v6, vk) for k in VAL_KEYS if x.get(k) != y.get(k)]
    check("d4_zero_weights_equal_e6_val_clip_0_1", len(v6) == len(vk) == 15 and not bad, f"{bad[:5]}")


@case("d4")
def d4_zero_weights_equal_e6_checkpoints_clip_0_1():
    e6, kd = d4_pair(0.1, None)
    e6_ends, kd_ends = rows_of(e6, "epoch_end"), rows_of(kd, "epoch_end")
    bad = []
    for e in range(1, 16):
        a, b = checkpoint(e6["out"], e), checkpoint(kd["out"], e)
        if not tensors_equal(a["model_state_dict"], b["model_state_dict"]):
            bad.append((e, "model_state_dict"))
        bad += [(e, k) for k in a if k not in ("model_state_dict", "stage") and a[k] != b.get(k)]
        if (a["stage"], b["stage"]) != ("E6", "E6-KD") or sorted(a) != sorted(b):
            bad.append((e, "stage or keys"))
    shas = [x["state_sha256"] for x in e6_ends] == [x["state_sha256"] for x in kd_ends]
    check("d4_zero_weights_equal_e6_checkpoints_clip_0_1", len(kd_ends) == 15 and shas and not bad, f"{bad[:5]}")


@case("d4")
def d4_zero_weights_equal_e6_rows_clip_5():
    e6, kd = d4_pair(5.0, 4)
    bad = same_rows(e6, kd)
    check("d4_zero_weights_equal_e6_rows_clip_5", len(train_rows(kd)) == 4 and not bad, f"{bad[:5]}")


@case("d4")
def d4_zero_weights_equal_e6_states_clip_5():
    e6, kd = d4_pair(5.0, 4)
    check("d4_zero_weights_equal_e6_states_clip_5", len(kd["rec"].states) == 4 and e6["rec"].states == kd["rec"].states
          and e6["rec"].params == kd["rec"].params, "4 steps")


@case("d4")
def d4_step1_equal_with_real_weights():
    e6, _kd = d4_pair(0.1, None)
    r = d4_real()
    a, b = train_rows(e6)[:1], train_rows(r)[:1]
    ok = (a and b and all(a[0].get(k) == b[0].get(k) for k in ROW_KEYS) and r["rec"].states[:1] == e6["rec"].states[:1]
          and r["rec"].params[:1] == e6["rec"].params[:1])
    check("d4_step1_equal_with_real_weights", ok, "the ramp is 0 at step 1: the step is E6's")


@case("d4")
def d4_real_weights_differ_from_step2():
    e6, _kd = d4_pair(0.1, None)
    r = d4_real()
    a, b = train_rows(e6)[1:4], train_rows(r)[1:4]
    ok = (len(b) == 3 and all(x["loss"] != y["loss"] for x, y in zip(a, b))
          and all(s6 != sk for s6, sk in zip(e6["rec"].params[1:4], r["rec"].params[1:4])))
    check("d4_real_weights_differ_from_step2", ok, f"losses E6 {[x['loss'] for x in a]} E6-KD {[x['loss'] for x in b]}")


# ---------------------------------------------------------------- records
@case("records")
def rec_telemetry_name_and_order():
    r = ms()
    events = [x.get("event") for x in r["rows"]]
    want = ["run_meta", "train", "step1_checks", "train", "val", "epoch_end", "train", "train", "val", "epoch_end"]
    check("rec_telemetry_name_and_order", (r["out"] / "qat_telemetry.jsonl").is_file() and Q.TELEMETRY_NAME ==
          "qat_telemetry.jsonl" and events == want and meta_of(r).get("schema") == KD.TELEMETRY_SCHEMA, f"{events}")


@case("records")
def rec_read_run_record_complete():
    r = mfull()
    rec = A.read_run_record(r["out"])
    ok = (rec["complete"] is True and sorted(rec["ends"]) == list(range(1, 16)) and rec["run_meta"]["stage"] == "E6-KD"
          and r["rc"] == 0 and r["result"] == "RESULT: E6-KD COMPLETE (E6-KD, seed 42, clip 5.0, 15/15 epochs)")
    check("rec_read_run_record_complete", ok, f"exit {r['rc']} {r['result']!r}; {rec['incomplete_reasons']}")


@case("records")
def rec_freeze_cross_check():
    r = mfull()
    rows, m = train_rows(r), meta_of(r)
    ok = (len(rows) == 30 and m.get("bn_freeze_after_step") == 20 and m.get("obs_freeze_after_step") == 24
          and all(x["bn_frozen"] is (x["step"] > 20) for x in rows)
          and all(x["observers_enabled"] is (x["step"] <= 24) for x in rows)
          and all(e["bn_frozen"] is (e["epoch"] > 10) and e["observers_enabled"] is (e["epoch"] <= 12)
                  for e in rows_of(r, "epoch_end")))
    check("rec_freeze_cross_check", ok, f"BN frozen from step {next((x['step'] for x in rows if x['bn_frozen']), None)}"
                                        f", observers off from {next((x['step'] for x in rows if not x['observers_enabled']), None)}")


@case("records")
def rec_state_flag_in_every_checkpoint():
    r = mfull()
    bad = []
    for e in rows_of(r, "epoch_end"):
        ck = checkpoint(r["out"], e["epoch"])
        if any(k not in ck or ck[k] != e.get(k) or type(ck[k]) is not type(e.get(k)) for k in FLAG_KEYS):
            bad.append(e["epoch"])
        if ck.get("state_finite") is not True or ck.get("stage") != "E6-KD" or ck.get("source_stage") != "E3":
            bad.append(("flag", e["epoch"]))
    check("rec_state_flag_in_every_checkpoint", len(rows_of(r, "epoch_end")) == 15 and not bad, f"{bad[:5]}")


@case("records")
def rec_epoch_end_projection_sha256():
    r = ms()
    ends = rows_of(r, "epoch_end")
    ok = (len(ends) == 2 and all(e.get("projection_sha256") == r["rec"].projection_at_end.get(e["epoch"]) for e in ends)
          and all(isinstance(e.get("projection_sha256"), str) and len(e["projection_sha256"]) == 64 for e in ends)
          and not list((r["out"]).glob("*projection*")) and not list((r["out"] / Q.EPOCH_DIR).glob("*projection*")))
    check("rec_epoch_end_projection_sha256", ok, f"{[e.get('projection_sha256', '')[:12] for e in ends]}")


@case("records")
def rec_run_meta_kd_keys_exact():
    m, m6 = meta_of(ms()), meta_of(e6_short())
    keys = sorted(m)
    q2_missing = sorted(set(m6) - set(m) - {"cwd_projection_loaded"})
    ok = (keys == EXPECTED_RUN_META_KEYS and not q2_missing and "cwd_projection_loaded" not in m
          and m.get("e6kd") is True and m.get("weight_factor") == 0.5 and m.get("distillation") is True
          and m.get("trigger_sha256") == gfx()["tsha"] and m.get("stage") == "E6-KD")
    check("rec_run_meta_kd_keys_exact", ok, f"extra {sorted(set(keys) - set(EXPECTED_RUN_META_KEYS))}, missing "
                                            f"{sorted(set(EXPECTED_RUN_META_KEYS) - set(keys))}, Q2's missing {q2_missing}")


@case("records")
def rec_descriptive():
    m = meta_of(ms())
    ok = (m.get("descriptive") is True and m.get("holm_family") is False and "E6-KD" in SA.DESCRIPTIVE_ONLY_STAGES
          and "E6-KD" in SA.PROJECTION_FREE_STAGES and "E6-KD" in ART.STAGES)
    check("rec_descriptive", ok, f"descriptive {m.get('descriptive')}, holm_family {m.get('holm_family')}")


@case("records")
def rec_projection_source():
    r, g = ms(), gfx()
    src = meta_of(r).get("projection_source") or {}
    want = {"path": str((g["parent_dir"] / PROJECTION_FILE).resolve()), "sha256": g["psha"], "format": "cwd_projection/1",
            "stage": "E3", "iter": 76000, "checkpoint": CKPT_NAME, "paired": True, "optimizer_state_loaded": False,
            "in_channels": 160, "out_channels": 320, "params": 51200}
    ok = all(src.get(k) == v for k, v in want.items()) and src == r["gates"]["projection_record"]
    check("rec_projection_source", ok, f"{ {k: src.get(k) for k in want if src.get(k) != want[k]} }")


@case("records")
def rec_trigger():
    r, g = ms(), gfx()
    m = meta_of(r)
    t = m.get("trigger") or {}
    ok = (t == {"path": g["trigger_rel"], "sha256": g["tsha"], "e3_val": 0.36334073543548584,
                "e6_val": 0.33082112669944763, "drop_pp": TRG.drop_decision(0.36334073543548584,
                                                                            0.33082112669944763)["drop_pp"],
                "triggered": True}
          and m.get("trigger_sha256") == g["tsha"]
          and m.get("trigger_binding") == {"trigger": g["trigger_rel"], "trigger_sha256": g["tsha"],
                                           "e3_checkpoint_sha256": g["ck_sha"], "clip_norm": 5.0})
    check("rec_trigger", ok, f"{t}; binding {m.get('trigger_binding')}")


@case("records")
def rec_teacher_provenance():
    r, g = ms(), gfx()
    m = meta_of(r)
    prov = m.get("teacher_provenance") or {}
    ok = (prov == r["gates"]["teacher"].provenance.as_dict() and prov.get("builder") == "stub_segnext_builder"
          and prov.get("ckpt_sha256") == prov.get("expected_sha256") == g["teacher_sha"]
          and prov.get("ckpt_path") == str(g["teacher_file"].resolve()) and m.get("teacher_mock") is False
          and m.get("teacher_identity") == r["gates"]["teacher_record"]
          and (m.get("teacher_nmf") or {}).get("policy") == "M4-KD")
    check("rec_teacher_provenance", ok, f"{prov.get('builder')} {str(prov.get('ckpt_sha256'))[:12]}")


@case("records")
def rec_result_line():
    r = ms()
    s = {"stage": "E6-KD", "seed": 42, "clip_norm": 5.0, "epochs_completed": 15, "complete": True,
         "nonfinite_since_step": 7}
    ok = (r["rc"] == 3 and r["result"] == "RESULT: E6-KD INCOMPLETE (E6-KD, seed 42, clip 5.0, 2/15 epochs)"
          and KD.result_line(s) == "RESULT: E6-KD COMPLETE, STATE NON-FINITE first found at step 7 (E6-KD, seed 42, "
                                   "clip 5.0, 15/15 epochs)"
          and KD.result_line({**s, "nonfinite_since_step": None}) == "RESULT: E6-KD COMPLETE (E6-KD, seed 42, clip 5.0, "
                                                                     "15/15 epochs)")
    check("rec_result_line", ok, f"exit {r['rc']} {r['result']!r}")


@case("records")
def rec_determinism_two_runs_identical():
    a = run_kd("det_a/run", rec=Recorder(), max_steps=4)
    b = run_kd("det_b/run", rec=Recorder(), max_steps=4)

    def strip(rows):
        return [{k: v for k, v in x.items() if k not in VOLATILE and k not in ("out_dir",)} for x in rows]
    ck_same = all(sha_file(a["out"] / Q.EPOCH_DIR / f"e{e:02d}.pt") == sha_file(b["out"] / Q.EPOCH_DIR / f"e{e:02d}.pt")
                  for e in (1, 2))
    ok = (a["error"] is None and b["error"] is None and strip(a["rows"]) == strip(b["rows"]) and ck_same
          and a["rec"].states == b["rec"].states)
    check("rec_determinism_two_runs_identical", ok, f"rows equal {strip(a['rows']) == strip(b['rows'])}; checkpoint "
                                                    f"bytes equal {ck_same}")


# ---------------------------------------------------------------- aborts (DL-04 (a); LB16, LB17)
@functools.lru_cache(maxsize=None)
def abort_run(tag: str) -> dict:
    """input25 is step 25, the first step of epoch 13, after the observer freeze (Q.OBS_FREEZE_EPOCH 12, 2 steps per
    epoch); at input3 the observers still record, and torch's observer asserts on the NaN range inside the forward."""
    inject = {"loss3": {"loss_nan": [3]}, "grad3": {"grad_inf": [3]}, "loss1": {"loss_nan": [1]},
              "check1": {"check_fail": [1]}, "input3": {"input_nan": [3]}, "input25": {"input_nan": [25]},
              "teacher3": {"teacher_nan": [3]}, "raise3": {"raise_at": [3]}, "state3": {"state_nan": [3]}}[tag]
    return run_kd(f"abort_{tag}/run", rec=Recorder(), inject=inject, max_steps=26 if tag == "input25" else 6)


@functools.lru_cache(maxsize=None)
def abort_main(tag: str) -> dict:
    inject = {"loss3": {"loss_nan": [3]}, "raise3": {"raise_at": [3]}}[tag]
    return main_run(f"abort_main_{tag}", inject=inject, max_steps=6)


def last_abort(r: dict) -> dict:
    last = r["rows"][-1] if r["rows"] else {}
    return last if last.get("event") == "run_abort" else {}


@case("aborts")
def abort_nonfinite_loss_before_backward():
    r = abort_run("loss3")
    a, rows, rec = last_abort(r), train_rows(r), r["rec"]
    ok = (isinstance(r["error"], KD.E6KDAborted) and [x["step"] for x in rows] == [1, 2, 3]
          and rows[-1].get("loss") is None and (rows[-1].get("nonfinite") or {}).get("loss") == "nan"
          and rows[-1].get("grad_norm") is None and a.get("rule") == KD.GENERIC_RULE and a.get("step") == 3
          and a.get("nonfinite_quantity") == "loss" and (a.get("detail") or {}).get("loss") == "nan"
          and (a.get("detail") or {}).get("grad_norm") is None
          and all(p.grad is None for p in rec.prepared.parameters())
          and param_digest(rec.prepared) == rec.params[1])
    check("abort_nonfinite_loss_before_backward", ok, f"{a.get('rule')} step {a.get('step')} "
                                                      f"{a.get('nonfinite_quantity')}; {r['error']!r}")


@case("aborts")
def abort_nonfinite_grad_before_step():
    r = abort_run("grad3")
    a, rows, rec = last_abort(r), train_rows(r), r["rec"]
    ok = (isinstance(r["error"], KD.E6KDAborted) and [x["step"] for x in rows] == [1, 2, 3]
          and math.isfinite(rows[-1]["loss"]) and rows[-1].get("grad_norm") is None
          and (rows[-1].get("nonfinite") or {}).get("grad_norm") == "inf" and a.get("rule") == KD.GENERIC_RULE
          and a.get("nonfinite_quantity") == "grad_norm" and (a.get("detail") or {}).get("grad_norm") == "inf"
          and param_digest(rec.prepared) == rec.params[1])
    check("abort_nonfinite_grad_before_step", ok, f"{a.get('rule')} {a.get('nonfinite_quantity')} "
                                                  f"{(a.get('detail') or {}).get('grad_norm')}; {r['error']!r}")


@case("aborts")
def abort_step1_rule():
    r1, rc = abort_run("loss1"), abort_run("check1")
    a1, ac = last_abort(r1), last_abort(rc)
    s1 = next((x for x in rc["rows"] if x.get("event") == "step1_checks"), {})
    ok = (a1.get("rule") == KD.STEP1_RULE == "step1_checks" and a1.get("cause") == "checks_failed"
          and a1.get("step") == 1 and a1.get("nonfinite_quantity") == "loss"
          and ac.get("rule") == "step1_checks" and ac.get("cause") == "checks_failed"
          and ac.get("failed_checks") == ["injected_check"] and ac.get("nonfinite_quantity") is None
          and s1.get("passed") is False and isinstance(rc["error"], KD.E6KDAborted))
    check("abort_step1_rule", ok, f"loss at step 1: {a1.get('rule')}/{a1.get('cause')}; failed check: "
                                  f"{ac.get('rule')}/{ac.get('cause')} {ac.get('failed_checks')}")


@case("aborts")
def abort_cause_input_nonfinite():
    a = last_abort(abort_run("input25"))
    check("abort_cause_input_nonfinite", a.get("cause") == "input_nonfinite" and a.get("input_finite") is False
          and a.get("rule") == KD.GENERIC_RULE and a.get("step") == 25 and a.get("epoch") == 13
          and a.get("nonfinite_quantity") == "loss", f"{a.get('cause')} input {a.get('input_finite')} step {a.get('step')}")


@case("aborts")
def abort_input_while_observing_is_exception():
    r = abort_run("input3")
    a = last_abort(r)
    check("abort_input_while_observing_is_exception", isinstance(r["error"], AssertionError)
          and a.get("rule") == KD.EXCEPTION_RULE and a.get("cause") == "exception"
          and a.get("exception_type") == "AssertionError" and a.get("input_finite") is False and a.get("step") == 3,
          f"{a.get('rule')} {a.get('exception_type')} input {a.get('input_finite')}; {r['error']!r}"[:300])


@case("aborts")
def abort_cause_teacher_nonfinite():
    a = last_abort(abort_run("teacher3"))
    check("abort_cause_teacher_nonfinite", a.get("cause") == "teacher_nonfinite" and a.get("input_finite") is True
          and a.get("teacher_finite") is False and a.get("rule") == KD.GENERIC_RULE,
          f"{a.get('cause')} input {a.get('input_finite')} teacher {a.get('teacher_finite')}")


@case("aborts")
def abort_cause_student_nonfinite():
    got = [(last_abort(abort_run(t)).get("cause"), last_abort(abort_run(t)).get("input_finite"),
            last_abort(abort_run(t)).get("teacher_finite")) for t in ("loss3", "grad3")]
    check("abort_cause_student_nonfinite", got == [("student_nonfinite", True, True)] * 2, f"{got}")


@case("aborts")
def abort_exception_writes_run_abort():
    r = abort_run("raise3")
    a = last_abort(r)
    ok = (isinstance(r["error"], RuntimeError) and not isinstance(r["error"], KD.E6KDAborted)
          and a.get("rule") == KD.EXCEPTION_RULE and a.get("cause") == "exception"
          and a.get("exception_type") == "RuntimeError" and "injected exception" in str(a.get("message")))
    check("abort_exception_writes_run_abort", ok, f"{a.get('rule')} {a.get('exception_type')} {r['error']!r}")


@case("aborts")
def abort_exception_record_survives_failed_readings():
    """DL-24 T1/AC1: after a device fault the finiteness readings themselves raise (a sticky CUDA error); the run_abort
    record is still written, with those readings null."""
    def device_fault(_t):
        raise RuntimeError("CUDA error: device-side assert triggered (smoke)")
    with patched(KD, "_finite", device_fault):
        r = run_kd("abort_fault/run", max_steps=2)
    a = last_abort(r)
    ok = (isinstance(r["error"], RuntimeError) and "device-side assert" in str(r["error"])
          and a.get("rule") == KD.EXCEPTION_RULE and a.get("cause") == "exception"
          and a.get("exception_type") == "RuntimeError" and a.get("input_finite") is None
          and a.get("teacher_finite") is None and a.get("params_finite") == {"student": None, "projection": None})
    check("abort_exception_record_survives_failed_readings", ok,
          f"last row {a.get('event')} {a.get('rule')} input {a.get('input_finite')} params {a.get('params_finite')}; "
          f"{r['error']!r}"[:300])


@case("aborts")
def abort_generic_reading_failure_is_exception():
    """DL-24 T1, the guard's bound: a generic abort's readings stay unguarded (abort_cause would read a missing reading
    as non-finite), so a reading that fails there ends the run on the exception path, whose record is written."""
    real = KD._finite

    def device_fault(t):
        if t.dim() == 0:                                    # step 1's scalar checks (loss, sup) read as before
            return real(t)
        raise RuntimeError("CUDA error: device-side assert triggered (smoke)")
    with patched(KD, "_finite", device_fault):
        r = run_kd("abort_generic_fault/run", inject={"loss_nan": [3]}, max_steps=4)
    a = last_abort(r)
    ok = (isinstance(r["error"], RuntimeError) and not isinstance(r["error"], KD.E6KDAborted)
          and "device-side assert" in str(r["error"]) and a.get("rule") == KD.EXCEPTION_RULE
          and a.get("cause") == "exception" and a.get("step") == 3 and a.get("input_finite") is None
          and a.get("params_finite") == {"student": None, "projection": None}
          and [x["step"] for x in train_rows(r)] == [1, 2, 3])
    check("abort_generic_reading_failure_is_exception", ok,
          f"last row {a.get('event')} {a.get('rule')}/{a.get('cause')} step {a.get('step')}; {r['error']!r}"[:300])


ABORT_FIELDS = ("event", "wall_clock", "epoch", "step", "rule", "cause", "nonfinite_quantity", "detail", "input_finite",
                "teacher_finite", "params_finite", "state_finite", "state_failing", "state_nonfinite",
                "nonfinite_since_step", "abort_kind")


@case("aborts")
def abort_record_fields():
    bad = []
    for t in ("loss3", "grad3", "loss1", "check1", "input25", "input3", "teacher3", "raise3"):
        a = last_abort(abort_run(t))
        if not a or any(k not in a for k in ABORT_FIELDS) or a.get("abort_kind") != KD.ABORT_KIND \
                or sorted(a.get("detail") or {}) != ["grad_norm", "grad_norm_projection", "grad_norm_student", "loss"] \
                or sorted(a.get("params_finite") or {}) != ["projection", "student"]:
            bad.append(t)
    check("abort_record_fields", not bad, f"records lacking the fields: {bad}")


@case("aborts")
def abort_exit_3_and_result_line():
    m, x = abort_main("loss3"), abort_main("raise3")
    ok = (m["rc"] == 3 and m["result"].startswith("RESULT: ABORTED rule=generic_abort (DL-04 (a); DL-51) step=3 "
                                                  "cause=student_nonfinite -- loss;")
          and x["rc"] == 3 and x["result"] == "RESULT: ABORTED [exception] -- RuntimeError: injected exception at step 3 "
                                              "(smoke)")
    check("abort_exit_3_and_result_line", ok, f"{m['rc']} {m['result'][:100]!r}; {x['rc']} {x['result'][:80]!r}")


@case("aborts")
def abort_run_is_incomplete():
    rec = A.read_run_record(abort_run("loss3")["out"])
    check("abort_run_is_incomplete", rec["complete"] is False
          and any("run_abort" in reason for reason in rec["incomplete_reasons"]), f"{rec['incomplete_reasons']}")


@case("aborts")
def abort_no_am7b_monitor():
    r = run_kd("abort_no_monitor/run", weights={t: 1000.0 for t in KD.TERMS}, max_steps=6)
    rows = train_rows(r)
    ok = (r["error"] is None and len(rows) == 6 and not rows_of(r, "run_abort") and rows[1]["loss"] > 10.0 * rows[0]["loss"]
          and not any("AM-7" in json.dumps(x) for x in r["rows"][1:]))
    check("abort_no_am7b_monitor", ok, f"losses {[round(x['loss'], 3) for x in rows]}; {r['error']!r}")


@case("aborts")
def abort_never_student_divergence():
    texts = []
    for t in ("loss3", "grad3", "loss1", "check1", "input25", "input3", "teacher3", "raise3", "state3"):
        r = abort_run(t)
        texts += [json.dumps(x) for x in r["rows"]]
    for t in ("loss3", "raise3"):
        m = abort_main(t)
        texts += [json.dumps(x) for x in m["rows"]] + [m["result"]]
    rules = [json.loads(x).get("rule") for x in texts if x.startswith("{") and '"run_abort"' in x]
    src = [(REPO / p).read_text(encoding="utf-8") for p in ("src/quant/qat_kd.py", "scripts/e6kd_eval.py")]
    token = "student_" + "divergence"
    ok = (rules and not any(token in x for x in texts) and all(r in (KD.STEP1_RULE, KD.GENERIC_RULE, KD.EXCEPTION_RULE)
                                                              for r in rules)
          and not any("AM-7(a)" in x or "AM-7 (a)" in x for x in texts)
          and not any(token in s for s in src))
    check("abort_never_student_divergence", ok, f"{len(texts)} records scanned; rules {sorted(set(map(str, rules)))}")


@case("aborts")
def abort_nonfinite_state_continues():
    r = abort_run("state3")
    ends, rows = rows_of(r, "epoch_end"), train_rows(r)
    flags_ok = all(checkpoint(r["out"], e["epoch"])[k] == e[k] for e in ends for k in FLAG_KEYS)
    ok = (r["error"] is None and not rows_of(r, "run_abort") and len(rows) == 6 and len(ends) == 3
          and ends[0]["state_finite"] is True and ends[0]["state_nonfinite"] is False
          and all(e["state_finite"] is False and e["nonfinite_since_step"] == 4 and e["state_nonfinite"] is True
                  for e in ends[1:])
          and all(x["state_nonfinite"] is (x["step"] > 4) for x in rows) and flags_ok
          and r["summary"]["nonfinite_since_step"] == 4 and rows_of(r, "state_nonfinite"))
    check("abort_nonfinite_state_continues", ok, f"epoch ends {[(e['epoch'], e['state_finite'], e['nonfinite_since_step']) for e in ends]}; "
                                                 f"{r['error']!r}")


# ---------------------------------------------------------------- gates: real_run_gates in order, each refusal alone
REAL_ISO = ISO.assert_trainval_only_root
REAL_IDENTITY = PD.teacher_identity_problems


def base_argv() -> list[str]:
    return list(gfx()["argv"])


def no_backend():
    raise QuantBackendUnavailable("qnnpack is not available here (smoke)")


def raising(exc: BaseException):
    def builder(_path):
        raise exc
    return builder


def with_trigger(rel: str, sha: str) -> list[str]:
    return argv_set(argv_set(base_argv(), "--trigger", rel), "--trigger-sha256", sha)


def trig(tag: str, **over) -> dict:
    g = gfx()
    return {"argv": with_trigger(*commit_trigger(g, tag, trigger_doc(g, **over)))}


def trig_untracked() -> dict:
    return {"argv": with_trigger(*commit_trigger(gfx(), "untracked", commit=False))}


def trig_changed() -> dict:
    g = gfx()
    rel, _sha = commit_trigger(g, "changed")
    data = (g["repo"] / rel).read_bytes() + b" \n"
    (g["repo"] / rel).write_bytes(data)
    return {"argv": with_trigger(rel, hashlib.sha256(data).hexdigest())}


def parent_other_name() -> dict:
    d = TMP / "parents" / "other_name" / "e3_s42_lambda0.5_alpha25_80k_a2"
    ck_sha, psha = write_parent(d)
    return {"argv": argv_set(argv_set(argv_set(base_argv(), "--source-run-dir", str(d)), "--expect-source-sha256",
                                      ck_sha), "--projection-sha256", psha)}


def run_meta_changed() -> dict:
    argv, _v = variant("meta_changed")
    real = Q.resolve_parent

    def resolve(*a, **kw):
        p = real(*a, **kw)
        f = Path(p["run_dir"]) / p["run_meta_file"]
        f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return p
    return {"argv": argv, "over": {(Q, "resolve_parent"): resolve}}


def parent_with_lines(tag: str, lines: list[str]) -> tuple[Path, str]:
    """A parent run whose run_meta file holds `lines` ahead of its run_meta row; returns (its folder, the checkpoint's
    sha256)."""
    d = TMP / "parents" / tag / PARENT_ID
    ck_sha, _psha = write_parent(d)
    f = d / "e3_run_meta.jsonl"
    f.write_text("".join(ln + "\n" for ln in lines) + f.read_text(encoding="utf-8"), encoding="utf-8")
    return d, ck_sha


def out_used() -> dict:
    d = TMP / "e6kd_out_used"
    d.mkdir(exist_ok=True)
    (d / "row.txt").write_text("a previous run\n", encoding="utf-8")
    return {"argv": argv_set(base_argv(), "--out-dir", str(d))}


def teacher_other_bytes() -> dict:
    f = TMP / "teacher_other" / "iter_24000.pth"
    f.parent.mkdir(exist_ok=True)
    f.write_bytes(b"another file under the teacher's name (smoke)\n")
    return {"argv": argv_set(base_argv(), "--teacher-ckpt", str(f))}


def teacher_symlink(dir_name: str, target: Path, name: str) -> str:
    """A symlink TMP/<dir_name>/<name> to `target` (made once)."""
    d = TMP / dir_name
    d.mkdir(exist_ok=True)
    link = d / name
    if not link.is_symlink():
        os.symlink(target, link)
    return str(link)


def teacher_copy(dir_name: str, name: str) -> Path:
    """A byte copy of the fixture's teacher file under another name (made once)."""
    d = TMP / dir_name
    d.mkdir(exist_ok=True)
    p = d / name
    if not p.is_file():
        shutil.copyfile(gfx()["teacher_file"], p)
    return p


def fingerprint_child(rc: int, doc):
    return lambda argv, tag, timeout, env=None: (rc, doc, "" if rc == 0 else "ModuleNotFoundError: mmseg (smoke)")


GATE_CASES = [
    # 1-3: the flags, configs/quant.py, the head's form, seed 42 only, the workers
    ("gate_refuses_without_real_run", "real_run_flag", lambda: {"argv": argv_without(base_argv(), "--real-run")}),
    ("gate_refuses_without_confirm", "confirm_real_run_flag",
     lambda: {"argv": argv_without(base_argv(), "--confirm-real-run")}),
    ("gate_refuses_config_pins", "config_pins",
     lambda: {"over": {(Q, "config_pins_error"): lambda: "configs/quant.py qat.epochs = 14, AM-4a: 15 (smoke)"}}),
    ("gate_refuses_head_format", "expect_head_format", lambda: {"argv": argv_set(base_argv(), "--expect-head", "abc")}),
    ("gate_refuses_seed_43", "seed_not_42", lambda: {"argv": argv_set(base_argv(), "--seed", "43")}),
    ("gate_refuses_seed_44", "seed_not_42", lambda: {"argv": argv_set(base_argv(), "--seed", "44")}),
    ("gate_refuses_num_workers_0", "num_workers", lambda: {"argv": argv_set(base_argv(), "--num-workers", "0")}),
    ("gate_refuses_u4_pilot_flag", "usage", lambda: {"argv": base_argv() + ["--u4-pilot"]}),
    # 4-5: the clip of record and the λ/α selections (src/quant/qat.py, reused)
    ("gate_refuses_clip_not_the_winner", "clip_selection_winner_mismatch",
     lambda: {"argv": argv_set(base_argv(), "--grad-clip-norm", "1.0")}),
    ("gate_refuses_clip_selection_sha256_mismatch", "clip_selection_sha256_mismatch",
     lambda: {"argv": argv_set(base_argv(), "--clip-selection-sha256", "f" * 64)}),
    ("gate_refuses_clip_selection_missing", "clip_selection_required",
     lambda: {"argv": argv_without(base_argv(), "--clip-selection", "--clip-selection-sha256")}),
    ("gate_refuses_lambda_selection_missing", "lambda_selection_missing",
     lambda: {"argv": argv_without(base_argv(), "--lambda-selection", "--lambda-selection-sha256")}),
    ("gate_refuses_alpha_selection_sha256_mismatch", "selection_sha256_mismatch",
     lambda: {"argv": argv_set(base_argv(), "--alpha-selection-sha256", "f" * 64)}),
    # 6: the committed trigger record (src/quant/e6kd_trigger.py, B-1)
    ("gate_refuses_trigger_missing", "trigger_missing",
     lambda: {"argv": argv_without(base_argv(), "--trigger", "--trigger-sha256")}),
    ("gate_refuses_trigger_not_repo_relative", "trigger_path_not_repo_relative",
     lambda: {"argv": argv_set(base_argv(), "--trigger", str(gfx()["repo"] / gfx()["trigger_rel"]))}),
    ("gate_refuses_trigger_untracked", "trigger_not_tracked", trig_untracked),
    ("gate_refuses_trigger_changed_since_head", "trigger_changed_since_head", trig_changed),
    ("gate_refuses_trigger_sha256_mismatch", "trigger_sha256_mismatch",
     lambda: {"argv": argv_set(base_argv(), "--trigger-sha256", "f" * 64)}),
    ("gate_refuses_trigger_format", "trigger_format", lambda: trig("wrong_format", format="e6kd_trigger/0")),
    ("gate_refuses_trigger_false", "trigger_not_triggered", lambda: trig("not_triggered", e3_val=0.36, e6_val=0.355)),
    ("gate_refuses_trigger_smoke_inputs", "trigger_smoke_inputs", lambda: trig("smoke_inputs", smoke_inputs=True)),
    # 7: a fresh out dir outside the repository
    ("gate_refuses_out_dir_in_repo", "out_dir", lambda: {"argv": argv_set(base_argv(), "--out-dir", str(REPO / "e6kd_out"))}),
    ("gate_refuses_out_dir_not_fresh", "out_dir", out_used),
    # 8: CUDA, TF32, QNNPACK
    ("gate_refuses_cpu", "cuda_required", lambda: {"over": {(torch.cuda, "is_available"): lambda: False}}),
    ("gate_refuses_device_flag_cpu", "cuda_required", lambda: {"argv": argv_set(base_argv(), "--device", "cpu")}),
    ("gate_refuses_cuda_initialised", "cuda_initialized_before_seed",
     lambda: {"over": {(torch.cuda, "is_initialized"): lambda: True}}),
    ("gate_refuses_tf32_not_default", "tf32_not_default",
     lambda: {"over": {(Q, "tf32_error"): lambda *a: "NVIDIA_TF32_OVERRIDE is '1' (smoke)"}}),
    ("gate_refuses_backend_unavailable", "backend_unavailable", lambda: {"over": {(KD, "select_qnnpack_backend"): no_backend}}),
    # 9: the KD image of record (DL-87) and its fingerprint (PL-26)
    ("gate_refuses_image_not_kd", "image_not_kd", lambda: {"env": {"PLANTSEG_IMAGE_DIGEST": Q.IMAGE_DIGEST_OF_RECORD}}),
    ("gate_refuses_image_digest_unset", "image_not_kd", lambda: {"env": {"PLANTSEG_IMAGE_DIGEST": None}}),
    ("gate_refuses_fingerprint_mismatch", "fingerprint_mismatch",
     lambda: {"over": {(PD, "_child_json"): fingerprint_child(0, {**fingerprint_of_record(), "torch": "2.1.0+cpu"})}}),
    ("gate_refuses_fingerprint_child_failed", "fingerprint_mismatch",
     lambda: {"over": {(PD, "_child_json"): fingerprint_child(1, None)}}),
    # 10-11: the data root; HEAD and the code paths
    ("gate_refuses_data_root_not_trainval_only", "data_root_not_trainval_only",
     lambda: {"over": {(ISO, "assert_trainval_only_root"): REAL_ISO}}),
    ("gate_refuses_head_mismatch", "expect_head_mismatch",
     lambda: {"over": {(KD, "code_identity"): lambda: {"git_head": "b" * 40, "code_clean_at_head": True,
                                                       "code_files_sha256": {}}}}),
    ("gate_refuses_code_not_clean", "code_not_clean",
     lambda: {"over": {(KD, "code_identity"): lambda: {"git_head": HEAD, "code_clean_at_head": False,
                                                       "code_files_sha256": {}}}}),
    # 12: the parent (src/quant/qat.py resolve_parent, e6_parent_binding)
    ("gate_refuses_parent_sha256_mismatch", "parent_sha256_mismatch",
     lambda: {"argv": argv_set(base_argv(), "--expect-source-sha256", "f" * 64)}),
    ("gate_refuses_parent_not_alpha_winner", "parent_run_id_mismatch", parent_other_name),
    # 13: the trigger's bindings (B-1)
    ("gate_refuses_trigger_e3_mismatch", "trigger_e3_mismatch", lambda: trig("e3_other", ck_sha="0" * 64)),
    ("gate_refuses_trigger_clip_mismatch", "trigger_clip_mismatch",
     lambda: trig("clip_other", e6={"clip_norm": 1.0, "clip_selection_sha256": gfx()["csha"]})),
    ("gate_refuses_trigger_selection_mismatch", "trigger_selection_mismatch",
     lambda: trig("selection_other", selections={"lambda": {"sha256": "1" * 64}, "alpha": {"sha256": gfx()["asha"]}})),
    # 14: E3's terms and weights (e3_record)
    ("gate_refuses_parent_run_meta_changed", "parent_run_meta_changed", run_meta_changed),
    ("gate_refuses_parent_terms", "parent_terms",
     lambda: {"argv": variant("terms_off", meta_over={"terms": {"logit_kd": True, "cwd_feat": True,
                                                                "cwd_logit": False}})[0]}),
    ("gate_refuses_parent_weights_mismatch", "parent_weights",
     lambda: {"argv": variant("beta_two", meta_over={"beta_cwd": 2.0})[0]}),
    ("gate_refuses_parent_T_logit", "parent_weights", lambda: {"argv": variant("t_two", meta_over={"T_logit": 2})[0]}),
    # 15: the projection of record (projection_of_record; DL-51, LB14)
    ("gate_refuses_projection_sha256_format", "projection_sha256_format",
     lambda: {"argv": argv_set(base_argv(), "--projection-sha256", "abc")}),
    ("gate_refuses_projection_missing", "projection_missing", lambda: {"argv": variant("no_projection", no_projection=True)[0]}),
    ("gate_refuses_projection_sha256_mismatch", "projection_sha256_mismatch",
     lambda: {"argv": argv_set(base_argv(), "--projection-sha256", "f" * 64)}),
    ("gate_refuses_projection_unreadable", "projection_unreadable",
     lambda: {"argv": variant("garbled", proj_raw=b"not a torch file (smoke)\n")[0]}),
    ("gate_refuses_projection_format", "projection_format",
     lambda: {"argv": variant("p_format", proj_over={"format": "cwd_projection/0"})[0]}),
    ("gate_refuses_projection_stage", "projection_stage", lambda: {"argv": variant("p_stage", proj_over={"stage": "E2"})[0]}),
    ("gate_refuses_projection_unpaired_checkpoint", "projection_unpaired_checkpoint",
     lambda: {"argv": variant("p_ckpt", proj_over={"checkpoint": "e3_student_best_iter72000.pt"})[0]}),
    ("gate_refuses_parent_checkpoint_invalid", "parent_checkpoint_invalid",
     lambda: {"argv": variant("ck_stage", ckpt_over={"stage": "E2"})[0]}),
    ("gate_refuses_projection_unpaired_iter", "projection_unpaired_iter",
     lambda: {"argv": variant("p_iter", proj_over={"iter": 72000})[0]}),
    ("gate_refuses_projection_shape", "projection_shape",
     lambda: {"argv": variant("p_shape", proj_over={"training_only": False})[0]}),
    # 16: the teacher of record (teacher_of_record; DL-88)
    ("gate_refuses_teacher_sha256_format", "teacher_sha256_format",
     lambda: {"argv": argv_set(base_argv(), "--teacher-ckpt-sha256", "abc")}),
    ("gate_refuses_teacher_not_of_record", "teacher_not_of_record",
     lambda: {"argv": argv_set(base_argv(), "--teacher-ckpt-sha256", "e" * 64)}),
    ("gate_refuses_teacher_missing", "teacher_missing",
     lambda: {"argv": argv_set(base_argv(), "--teacher-ckpt", str(TMP / "teacher_absent" / "iter_24000.pth"))}),
    # E-31's form (DL-88; DL-24 G2), each condition alone: outside the repository, the given name, the resolved name
    ("gate_refuses_teacher_ckpt_in_repo", "teacher_ckpt_path",
     lambda: {"argv": argv_set(base_argv(), "--teacher-ckpt", str(REPO / "nowhere_e6kd" / "iter_24000.pth"))}),
    ("gate_refuses_teacher_ckpt_given_name", "teacher_ckpt_path",
     lambda: {"argv": argv_set(base_argv(), "--teacher-ckpt",
                               teacher_symlink("teacher_alias", gfx()["teacher_file"], "other_name.pth"))}),
    ("gate_refuses_teacher_ckpt_resolved_name", "teacher_ckpt_path",
     lambda: {"argv": argv_set(base_argv(), "--teacher-ckpt",
                               teacher_symlink("teacher_link", teacher_copy("teacher_copy", "copy_of_record.pth"),
                                               "iter_24000.pth"))}),
    ("gate_refuses_teacher_sha256_mismatch", "teacher_sha256_mismatch", teacher_other_bytes),
    ("gate_refuses_teacher_state_dict_mismatch", "teacher_state_dict_mismatch",
     lambda: {"builder": raising(TeacherStateDictMismatch(missing=["decode_head.conv_seg.weight"], source="(smoke)"))}),
    ("gate_refuses_teacher_stack_missing", "teacher_stack_missing",
     lambda: {"builder": raising(TT.TeacherStackMissing("mmseg is not importable (smoke)"))}),
    ("gate_refuses_teacher_identity", "teacher_identity",
     lambda: {"over": {(PD, "teacher_identity_problems"): REAL_IDENTITY}}),
]


def register_gate_case(name: str, want: str, make) -> None:
    def fn():
        spec = make()
        got = gate(spec.get("argv"), spec.get("over"), env=spec.get("env"),
                   builder=spec.get("builder") or stub_segnext_builder)
        check(name, got == want and not Path(gfx()["out_new"]).exists(), f"[{got}]")
    CASES.append((name, "gates", fn))


for _name, _want, _make in GATE_CASES:
    register_gate_case(_name, _want, _make)


@case("gates")
def gate_refuses_teacher_ckpt_relative():
    """E-31's absolute form alone (DL-24 G2): a relative path that resolves to the fixture's own teacher file."""
    with contextlib.chdir(TMP):
        got = gate(argv_set(base_argv(), "--teacher-ckpt", "teacher/iter_24000.pth"))
    check("gate_refuses_teacher_ckpt_relative", got == "teacher_ckpt_path" and not Path(gfx()["out_new"]).exists(),
          f"[{got}]")


@case("gates")
def gate_e3_record_reads_the_run_meta_row():
    """DL-24 T3: gate 14 reads the run_meta row resolve_parent read (Q._run_meta_rows: exactly one). Ahead of it the
    file holds a row of another event that names "run_meta", with E3's terms and other weights, or a torn line that
    names it; the weights read are the run_meta row's either way."""
    g = gfx()
    want = KD.e3_record(Q.resolve_parent(STAGE, g["parent_dir"], g["ck_sha"], 42))["e3_weights"]
    row = json.loads((g["parent_dir"] / "e3_run_meta.jsonl").read_text(encoding="utf-8"))
    decoy = json.dumps({**row, "event": "note", "about": "run_meta", "lambda_logit": 9.0, "alpha_cwd": 99.0})
    got = {}
    for tag, line in (("meta_decoy", decoy), ("meta_torn", '{"event": "run_meta", "torn')):
        d, ck_sha = parent_with_lines(tag, [line])
        got[tag] = KD.e3_record(Q.resolve_parent(STAGE, d, ck_sha, 42))["e3_weights"]
    check("gate_e3_record_reads_the_run_meta_row", got == {"meta_decoy": want, "meta_torn": want},
          f"{got}; want {want}")


@case("gates")
def gates_pass_with_valid_inputs():
    g = gfx()
    got = gate()
    res = real_gates()
    ok = (got is None and res["weights"] == W and res["e3"]["e3_weights"] == E3W
          and res["clip"]["clip_source"] == "clip_selection" and res["trigger"]["sha256"] == g["tsha"]
          and res["trigger_binding"]["e3_checkpoint_sha256"] == g["ck_sha"]
          and res["projection_record"]["sha256"] == g["psha"] and isinstance(res["teacher"], TT.FrozenTeacher)
          and res["teacher"].provenance.ckpt_sha256 == g["teacher_sha"] and res["image_digest"] == KD_DIGEST
          and res["e6_binding"]["alpha_selection"]["sha256"] == g["asha"] and res["parent"]["run_id"] == PARENT_ID
          and not Path(g["out_new"]).exists())
    check("gates_pass_with_valid_inputs", ok, f"[{got}]")


@case("gates")
def gates_refusal_through_main_exit_2():
    g = gfx()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = KD.main(argv_set(base_argv(), "--seed", "43"), teacher_builder=stub_segnext_builder)
    res = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    check("gates_refusal_through_main_exit_2", rc == 2 and res and res[-1].startswith("RESULT: REFUSED [seed_not_42]")
          and res[-1].endswith("Nothing was written.") and not Path(g["out_new"]).exists(), f"{rc} {res[-1:]}")


@case("gates")
def gates_error_before_launch_exit_4():
    g = gfx()
    gates = real_gates()

    def boom(stage, parent):
        raise OSError("the source checkpoint cannot be read (smoke)")
    buf = io.StringIO()
    with patched(KD, "real_run_gates", lambda args, stage, **k: gates), patched(Q, "load_source", boom), \
            contextlib.redirect_stdout(buf):
        rc = KD.main(base_argv(), teacher_builder=stub_segnext_builder)
    res = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    check("gates_error_before_launch_exit_4", rc == 4 and res and res[-1].startswith("RESULT: ERROR [OSError]")
          and not Path(g["out_new"]).exists(), f"{rc} {res[-1:]}")


# ---------------------------------------------------------------- runs: run_qat_kd's own refusals
def run_args(tag: str, **kw) -> dict:
    """run_qat_kd's arguments in mode "smoke" (2 steps, so a removed refusal ends quickly), into a never-written dir."""
    base = dict(mode="smoke", model=student(), teacher=stub_teacher(), projection=projection_fixture(),
                weights=dict(W), e3=E3REC, source_meta=SOURCE, out_dir=TMP / "unwritten" / tag, seed=SEED,
                clip_norm=1.0, clip_source="smoke", device="cpu", num_workers=0, loaders=make_loaders(), max_steps=2,
                log=quiet)
    base.update(kw)
    return base


def real_args(tag: str, **kw) -> dict:
    """Mode "real" on a CUDA device name: the CPU host never moves a tensor before these refusals."""
    return run_args(tag, **{"mode": "real", "clip_source": "clip_selection", "loaders": None, "max_steps": None,
                            "device": "cuda", **kw})


def smoke_loaders_as_built(batch: int = 16):
    return lambda seed, num_workers: make_loaders(seed, batch=batch)


def unfrozen_teacher():
    t = stub_teacher()
    next(t.teacher.parameters()).requires_grad_(True)
    return t


def projection_with_qconfig():
    p = projection_fixture()
    p.qconfig = "a qconfig (smoke)"
    return p


class ValChangesState(KD.E6KDHooks):
    def before_val(self, epoch, prepared):
        with torch.no_grad():
            next(prepared.parameters()).add_(1.0)


def wrong_lr_optimizer(prepared, projection):
    return torch.optim.SGD([{"params": list(prepared.parameters()), "name": "student"},
                            {"params": list(projection.parameters()), "name": PROJECTION_GROUP}],
                           lr=1e-3, momentum=Q.MOMENTUM, weight_decay=Q.WEIGHT_DECAY)


SMOKE_PATCHES = {(Q, "REAL_STEPS_PER_EPOCH"): 2, (Q, "REAL_T_MAX"): 30}
RUN_CASES = [
    ("run_refuses_stage_not_qat_kd", "stage_not_qat_kd", lambda: (run_args("r01"),
                                                                 {(KD, "resolve_quant_stage"): lambda k: resolve_quant_stage("e6")})),
    ("run_refuses_mode", "mode_invalid", lambda: (run_args("r02", mode="dry"), {})),
    ("run_refuses_loaders_in_real", "real_run_overrides", lambda: (real_args("r03", loaders=make_loaders(), device="cpu"), {})),
    ("run_refuses_hooks_in_real", "real_run_overrides", lambda: (real_args("r04", hooks=KD.E6KDHooks(), device="cpu"), {})),
    ("run_refuses_injection_in_real", "real_run_overrides",
     lambda: (real_args("r05", inject={"loss_nan": [3]}, device="cpu"), {})),
    ("run_refuses_step_cap_in_real", "real_run_overrides", lambda: (real_args("r06", max_steps=2, device="cpu"), {})),
    ("run_refuses_val_cap_in_real", "real_run_overrides", lambda: (real_args("r07", max_val_batches=1, device="cpu"), {})),
    # DL-24 T2: a cap of 0 is a cap too
    ("run_refuses_zero_step_cap_in_real", "real_run_overrides",
     lambda: (real_args("r27", max_steps=0, device="cpu"), {})),
    ("run_refuses_zero_val_cap_in_real", "real_run_overrides",
     lambda: (real_args("r28", max_val_batches=0, device="cpu"), {})),
    ("run_refuses_clip_source_in_real", "clip_source", lambda: (real_args("r08", clip_source="smoke"), {})),
    ("run_refuses_config_pins", "config_pins",
     lambda: (run_args("r09"), {(Q, "config_pins_error"): lambda: "configs/quant.py qat.epochs = 14 (smoke)"})),
    ("run_refuses_bad_clip", "grad_clip_norm_invalid", lambda: (run_args("r10", clip_norm=0.0), {})),
    ("run_refuses_weights_invalid", "weights_invalid",
     lambda: (run_args("r11", weights={"logit_kd": 0.25, "cwd_feat": 12.5}), {})),
    ("run_refuses_weights_not_half_in_real", "weights_not_half", lambda: (real_args("r12", weights=dict(E3W)), {})),
    ("run_refuses_teacher_not_frozen", "teacher_not_frozen", lambda: (run_args("r13", teacher=unfrozen_teacher()), {})),
    ("run_refuses_projection_not_of_record", "projection_not_of_record",
     lambda: (run_args("r14", projection=projection_with_qconfig()), {})),
    ("run_refuses_cpu_in_real", "cuda_required", lambda: (real_args("r15", device="cpu"), {})),
    ("run_refuses_cuda_initialised_in_real", "cuda_initialized_before_seed",
     lambda: (real_args("r16"), {(torch.cuda, "is_initialized"): lambda: True})),
    ("run_refuses_nonempty_out_dir", "out_dir_not_empty", lambda: (run_args("r17", out_dir=used_dir()), {})),
    ("run_refuses_steps_per_epoch_in_real", "steps_per_epoch",
     lambda: (real_args("r18"), {(Q, "build_qat_loaders"): smoke_loaders_as_built()})),
    ("run_refuses_loader_of_record_in_real", "loader_of_record",
     lambda: (real_args("r19"), {(Q, "build_qat_loaders"): smoke_loaders_as_built(8), (Q, "REAL_STEPS_PER_EPOCH"): 4,
                                 (Q, "REAL_T_MAX"): 60})),
    ("run_refuses_empty_loader", "empty_loader", lambda: (run_args("r20", loaders=make_loaders(n_train=8)), {})),
    ("run_refuses_backend_unavailable", "backend_unavailable",
     lambda: (run_args("r21"), {(KD, "select_qnnpack_backend"): no_backend})),
    ("run_refuses_unfused_batchnorm_in_real", "unfused_batchnorm",
     lambda: (real_args("r22"), {(Q, "build_qat_loaders"): smoke_loaders_as_built(), **SMOKE_PATCHES,
                                 (KD, "unfused_batchnorm"): lambda m: ["features.0.1 (smoke)"]})),
    ("run_refuses_teacher_nmf_missing_in_real", "teacher_nmf_missing",
     lambda: (real_args("r23", teacher=TT.FrozenTeacher(TT.MockTeacher(NC))),
              {(Q, "build_qat_loaders"): smoke_loaders_as_built(), **SMOKE_PATCHES})),
    ("run_stops_optimizer_of_record", "optimizer_of_record",
     lambda: (run_args("r24"), {(KD, "kd_optimizer"): wrong_lr_optimizer})),
]


def used_dir() -> Path:
    d = TMP / "used_out"
    d.mkdir(exist_ok=True)
    (d / "row.txt").write_text("a previous run\n", encoding="utf-8")
    return d


def register_run_case(name: str, want: str, make) -> None:
    def fn():
        kw, over = make()
        with patches(over):
            got = code_of(KD.run_qat_kd, **kw)
        out = Path(kw["out_dir"])
        unwritten = not out.exists() if out.name != "used_out" else sorted(p.name for p in out.iterdir()) == ["row.txt"]
        check(name, got == want and unwritten, f"[{got}]; nothing written: {unwritten}")
    CASES.append((name, "runs", fn))


for _name, _want, _make in RUN_CASES:
    register_run_case(_name, _want, _make)


@case("runs")
def run_refuses_unknown_injection():
    kw = run_args("r25", inject={"nan_at": [1]})
    got = code_of(KD.run_qat_kd, **kw)
    check("run_refuses_unknown_injection", got == "raised:ValueError" and not Path(kw["out_dir"]).exists(), f"[{got}]")


@case("runs")
def run_stops_val_bracket_changed_state():
    kw = run_args("r26", hooks=ValChangesState())
    got = code_of(KD.run_qat_kd, **kw)
    rows = rows_at(kw["out_dir"])
    last = rows[-1] if rows else {}
    check("run_stops_val_bracket_changed_state", got == "val_bracket_changed_state" and last.get("event") == "run_stop"
          and last.get("code") == "val_bracket_changed_state", f"[{got}]; last row {last.get('event')}")


@case("runs")
def run_stop_through_main_exit_1():
    g = gfx()
    gates = real_gates(argv_set(base_argv(), "--out-dir", str(TMP / "runs" / "stop_main")))

    def stopping(**kw):
        raise Q.QATStop("optimizer_of_record", "the optimizer differs from the recipe of record (smoke)")
    buf = io.StringIO()
    with patched(KD, "real_run_gates", lambda args, stage, **k: gates), patched(KD, "run_qat_kd", stopping), \
            contextlib.redirect_stdout(buf):
        rc = KD.main(argv_set(base_argv(), "--out-dir", str(TMP / "runs" / "stop_main")),
                     teacher_builder=stub_segnext_builder)
    res = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    check("run_stop_through_main_exit_1", rc == 1 and res and res[-1].startswith("RESULT: STOP [optimizer_of_record]"),
          f"{rc} {res[-1:]}; {g['tsha'][:8]}")


# ---------------------------------------------------------------- routing
E5_AT_O = {"name": "E5", "method": "qat", "source_stage": "E1", "objective": "INT8 QAT from the E1 FP32 checkpoint"}
E6_AT_O = {"name": "E6", "method": "qat", "source_stage": "E3",
           "objective": "INT8 QAT from the E3 projection-free student (identical config to E5)"}


@case("routing")
def runner_routes_e6kd_to_qat_kd():
    import scripts.run_e6kd as LAUNCHER
    seen = []

    def spy(argv, stage_key="?", **kw):
        seen.append((list(argv), stage_key))
        return 0
    with patched(KD, "main", spy):
        rc = RUNNER.main(["--real-run"], "e6-kd")
        rc2 = LAUNCHER.main(["--seed", "42"])
    ok = rc == 0 and rc2 == 0 and seen == [(["--real-run"], "e6-kd"), (["--seed", "42"], "e6-kd")] \
        and LAUNCHER.STAGE == "e6-kd"
    check("runner_routes_e6kd_to_qat_kd", ok, f"{seen}")


@case("routing")
def run_qat_refuses_e6kd():
    out = TMP / "unwritten" / "q01"
    got = code_of(Q.run_qat, stage="e6-kd", mode="smoke", model=student(), source_meta=SOURCE, out_dir=out, seed=SEED,
                  clip_norm=1.0, clip_source="smoke", device="cpu", num_workers=0, loaders=make_loaders(), max_steps=2,
                  log=quiet)
    check("run_qat_refuses_e6kd", got == "stage_not_qat" and not out.exists(), f"[{got}]")


@case("routing")
def ptq_path_never_reached():
    calls = []

    def spy(argv, stage_key="?", **kw):
        return 0

    def old_path(parser):
        calls.append(1)
        raise SystemExit(9)
    with patched(KD, "main", spy), patched(RUNNER, "add_common_args", old_path):
        rc = RUNNER.main([], "e6-kd")
        try:
            RUNNER.main([], "e7")
            e7 = "returned"
        except SystemExit:
            e7 = "old path"
    check("ptq_path_never_reached", rc == 0 and calls == [1] and e7 == "old path",
          f"e6-kd exit {rc}; the old runner's path reached {len(calls)} time(s), by e7: {e7}")


@case("routing")
def e5_e6_stage_dicts_unchanged():
    keys = list(QUANT_STAGES)
    ok = (QUANT_STAGES["e5"] == E5_AT_O and QUANT_STAGES["e6"] == E6_AT_O
          and keys == ["e4", "e5", "e6", "e7", "e6-kd", "e6-e2", "e7-e2", "e6-r1", "e6-r2"])
    check("e5_e6_stage_dicts_unchanged", ok, f"{keys}")


@case("routing")
def e6kd_stage_entry():
    e = QUANT_STAGES.get("e6-kd") or {}
    ok = (sorted(e) == ["method", "name", "objective", "source_stage"] and e.get("name") == "E6-KD"
          and e.get("method") == "qat_kd" and e.get("source_stage") == "E3"
          and resolve_quant_stage("E6-KD")["key"] == "e6-kd" and STAGE["method"] == KD.METHOD
          and SA.STAGE_ARTIFACTS.get("E6-KD") == {"kind": "int8_artifact", "precision": "int8_qat",
                                                  "source_stage": "E3", "method": "qat"}
          and ART.STAGES[-1] == "E6-KD" and SA.PROJECTION_FREE_STAGES[-1] == "E6-KD"
          and SA.DESCRIPTIVE_ONLY_STAGES[-1] == "E6-KD")
    check("e6kd_stage_entry", ok, f"{e}")


def child_json(code: str, tag: str) -> tuple[int, object, str]:
    p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=600, cwd=str(REPO))
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(tag)]
    return p.returncode, (json.loads(lines[-1][len(tag):]) if lines else None), p.stderr[-300:]


@case("routing")
def runner_imports_qat_kd_only_in_its_route():
    code = ("import json, sys; sys.path.insert(0, %r); import src.quant.runner, src.quant.stages, src.quant.qat_select; "
            "print('MODULES:' + json.dumps(sorted(sys.modules)))" % str(REPO))
    rc, mods, err = child_json(code, "MODULES:")
    banned = ("src.quant.qat_kd", "src.training.train_distill", "src.distill.teacher")
    loaded = sorted(set(mods or []) & set(banned))
    check("runner_imports_qat_kd_only_in_its_route", rc == 0 and mods and not loaded, f"loaded {loaded} {err}")


@case("routing")
def qat_kd_imports_no_teacher_stack():
    code = ("import json, sys\nsys.path.insert(0, %r)\nseen = []\n"
            "class Rec:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('mmseg', 'mmcv', 'mmengine', 'mmpretrain', 'mmdet'):\n"
            "            seen.append(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Rec())\n"
            "import src.quant.qat_kd, scripts.e6kd_eval, scripts.run_e6kd\n"
            "print('SEEN:' + json.dumps(seen))\n" % str(REPO))
    rc, seen, err = child_json(code, "SEEN:")
    check("qat_kd_imports_no_teacher_stack", rc == 0 and seen == [], f"{seen} {err}")


@case("routing")
def code_paths_cover_the_launch_imports():
    """DL-24 G1 (LB26): every repository module a launch process loads is in E6KD_CODE_PATHS, which gate 11 requires
    clean and run_meta hashes. A child imports qat_kd, code_identity's src/quant/ptq.py and what run_qat_kd and the
    gates import when they run; the teacher config's custom_imports module needs mmcv, so its own repository imports
    at load are read from its source and imported."""
    code = ("import ast, json, os, sys\nsys.path.insert(0, %r)\nos.environ['PLANTSEG_DATA_ROOT'] = %r\n"
            "import importlib\nimport src.quant.qat_kd as KD\nKD.preflight_distill()\n"
            "for m in ('src.quant.ptq', 'src.seeds', 'src.training.losses', 'src.training.train_e1', "
            "'src.quant.qconfig', 'src.quant.checkpoint', 'src.quant.prepare', 'src.data.isolation', 'configs.data'):\n"
            "    importlib.import_module(m)\n"
            "tc = ast.parse(open(os.path.join(%r, 'src/training/teacher_components.py'), encoding='utf-8').read())\n"
            "for n in tc.body:\n"
            "    if isinstance(n, ast.ImportFrom) and n.module and n.module.split('.')[0] in ('src', 'configs'):\n"
            "        importlib.import_module(n.module)\n"
            "root = os.path.realpath(%r)\n"
            "mods = [m for m in list(sys.modules.values()) if getattr(m, '__file__', None)]\n"
            "files = {os.path.realpath(m.__file__) for m in mods}\n"
            "loaded = sorted(os.path.relpath(f, root) for f in files if f.startswith(root + os.sep))\n"
            "print('LOADED:' + json.dumps(loaded))\n") % (str(REPO), str(TMP / "absent_data_root"), str(REPO),
                                                           str(REPO))
    rc, loaded, err = child_json(code, "LOADED:")
    missing = [p for p in (loaded or []) if p not in KD.E6KD_CODE_PATHS]
    check("code_paths_cover_the_launch_imports", rc == 0 and bool(loaded) and not missing,
          f"{len(loaded or [])} repository modules loaded; not in E6KD_CODE_PATHS: {missing} {err}")


# ---------------------------------------------------------------- sl1: SL-1's name check before any probe
def sl1(name: str, want: str, bad: Path, argv: list[str]) -> None:
    with probe_recorder(bad) as seen:
        got = gate(argv)
    check(name, got == want and not seen and not bad.exists(), f"[{got}]; probed {seen[:2]}")


@case("sl1")
def sl1_forbidden_source_dir_refused_before_any_probe():
    bad = TMP / "nowhere_latest_e3"                     # never created; its name holds the token SL-1 forbids
    sl1("sl1_forbidden_source_dir_refused_before_any_probe", "parent_dir_test_path", bad,
        argv_set(base_argv(), "--source-run-dir", str(bad)))


@case("sl1")
def sl1_forbidden_teacher_refused_before_any_probe():
    bad = TMP / "nowhere_latest_teacher"
    sl1("sl1_forbidden_teacher_refused_before_any_probe", "teacher_forbidden_path", bad,
        argv_set(base_argv(), "--teacher-ckpt", str(bad / "iter_24000.pth")))


@case("sl1")
def sl1_forbidden_trigger_refused_before_any_probe():
    g = gfx()
    sl1("sl1_forbidden_trigger_refused_before_any_probe", "trigger_forbidden_path", g["repo"] / "trig" / "latest.json",
        argv_set(base_argv(), "--trigger", "trig/latest.json"))


@case("sl1")
def sl1_forbidden_out_dir_refused_before_any_probe():
    bad = TMP / "nowhere_latest_out"
    sl1("sl1_forbidden_out_dir_refused_before_any_probe", "out_dir", bad, argv_set(base_argv(), "--out-dir", str(bad)))


@case("sl1")
def sl1_forbidden_clip_selection_refused_before_any_probe():
    bad = TMP / "nowhere_latest_clip.json"
    sl1("sl1_forbidden_clip_selection_refused_before_any_probe", "clip_selection_test_path", bad,
        argv_set(base_argv(), "--clip-selection", str(bad)))


@case("sl1")
def sl1_forbidden_paths_refused_by_check_run_meta():
    good_run = TMP / "profile" / "sl1_run"
    good_run.mkdir(parents=True)
    out = {}
    for what, bad, argv in (("run_dir", TMP / "nowhere_latest_run", profile_argv(TMP / "nowhere_latest_run")),
                            ("teacher_ckpt", TMP / "nowhere_latest_t",
                             profile_argv(good_run, **{"--teacher-ckpt": str(TMP / "nowhere_latest_t" / "iter_24000.pth")}))):
        with probe_recorder(bad) as seen:
            rc, res, _ = ev_main(argv)
        out[what] = (rc, res[:80], len(seen))
    ok = (out["run_dir"][0] == 2 and "[run_dir_forbidden_path]" in out["run_dir"][1] and out["run_dir"][2] == 0
          and out["teacher_ckpt"][0] == 2 and "[teacher_forbidden_path]" in out["teacher_ckpt"][1]
          and out["teacher_ckpt"][2] == 0)
    check("sl1_forbidden_paths_refused_by_check_run_meta", ok, f"{out}")


@case("sl1")
def sl1_forbidden_expanded_teacher_refused_before_any_probe():
    """DL-24 G3: '~' expands to a home whose name holds the token SL-1 forbids (never created); the name check reads
    the expanded form too, before any probe."""
    home = TMP / "nowhere_latest_home"
    with probe_recorder(home) as seen, env_set({"HOME": str(home)}), patches(pass_patches(gfx())):
        got = code_of(KD.teacher_of_record, "~/iter_24000.pth", gfx()["teacher_sha"], builder=stub_segnext_builder)
    check("sl1_forbidden_expanded_teacher_refused_before_any_probe",
          got == "teacher_forbidden_path" and not seen and not home.exists(), f"[{got}]; probed {seen[:2]}")


@case("sl1")
def sl1_forbidden_expanded_paths_refused_by_check_run_meta():
    """DL-24 G3, check-run-meta's side: the expanded --teacher-ckpt is checked before the run's record is read."""
    home = TMP / "nowhere_latest_home"
    good_run = TMP / "profile" / "sl1_run_home"
    good_run.mkdir(parents=True, exist_ok=True)
    with probe_recorder(home) as seen, env_set({"HOME": str(home)}):
        rc, res, _ = ev_main(profile_argv(good_run, **{"--teacher-ckpt": "~/iter_24000.pth"}))
    check("sl1_forbidden_expanded_paths_refused_by_check_run_meta",
          rc == 2 and "[teacher_forbidden_path]" in res and not seen and not home.exists(), f"{rc} {res[:90]}")


# ---------------------------------------------------------------- profile: scripts/e6kd_eval.py check-run-meta
REAL_PROFILE = EV.e6kd_profile_mismatches


def set_dotted(d: dict, key: str, value) -> None:
    """Set an existing dotted key (a missing one raises: a renamed trainer key fails the profile smoke)."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d[part]
    if parts[-1] not in d:
        raise KeyError(key)
    d[parts[-1]] = value


def provenance_of_record() -> dict:
    return {"builder": PD.TEACHER_BUILDER, "ckpt_path": str(Path(POD_TEACHER).expanduser().resolve()),
            "ckpt_sha256": PD.TEACHER_SHA256, "ckpt_bytes": PD.TEACHER_BYTES, "expected_sha256": PD.TEACHER_SHA256,
            "config_path": str(TD.DEFAULT_TEACHER_CONFIG), **copy.deepcopy(PD.TEACHER_VALUES)}


LAUNCH = {"mode": "real", "num_workers": 12, "persistent_workers": True, "steps_per_epoch": 335, "total_steps": 5025,
          "scheduler.T_max": 5025, "bn_freeze_after_step": 3350, "obs_freeze_after_step": 4020, "val_batches_cap": None,
          "torch": "2.1.0+cu121", "cuda_initialized_at_seed": False, "image_digest": KD_DIGEST, "git_head": HEAD,
          "code_clean_at_head": True, "ramp_iters": 335, "teacher_mock": False}


@functools.lru_cache(maxsize=None)
def kd_row_json() -> str:
    """A real launch's run_meta row: the row main() and the trainer write in the short launch, with the launch values
    of record set over it (Q2's launched_row form)."""
    row = copy.deepcopy(meta_of(ms()))
    for k, v in {**LAUNCH, "teacher_provenance": provenance_of_record()}.items():
        set_dotted(row, k, v)
    return json.dumps(row)


def kd_row() -> dict:
    return json.loads(kd_row_json())


def profile_argv(run_dir, **over) -> list[str]:
    g = gfx()
    flags = {"--run-dir": str(run_dir), "--seed": "42", "--grad-clip-norm": "5.0", "--expect-head": HEAD,
             "--expect-source-sha256": g["ck_sha"], "--clip-selection": str(g["clip"]),
             "--clip-selection-sha256": g["csha"], "--lambda-selection": "sel/lambda.json",
             "--lambda-selection-sha256": g["lsha"], "--alpha-selection": "sel/alpha.json",
             "--alpha-selection-sha256": g["asha"], "--trigger": g["trigger_rel"], "--trigger-sha256": g["tsha"],
             "--projection-sha256": g["psha"], "--teacher-ckpt": POD_TEACHER,
             "--teacher-ckpt-sha256": PD.TEACHER_SHA256, **over}
    return ["check-run-meta"] + [x for kv in flags.items() for x in kv]


def mismatches(row: dict, **over) -> dict:
    args = EV.build_parser().parse_args(profile_argv(TMP, **over))
    return REAL_PROFILE(row, args, repo_root=gfx()["repo"])


def ev_main(argv: list[str], row=None, *, first_line: str | None = None) -> tuple[int, str, str]:
    """scripts/e6kd_eval.py main() on a run dir holding `row` (or `first_line`); the profile reads the selections and
    the trigger from the fixture's repository."""
    if row is not None or first_line is not None:
        d = Path(argv[argv.index("--run-dir") + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / Q.TELEMETRY_NAME).write_text((first_line if first_line is not None else json.dumps(row)) + "\n",
                                          encoding="utf-8")
    buf = io.StringIO()
    with patched(EV, "e6kd_profile_mismatches", lambda meta, args, **k: REAL_PROFILE(meta, args, repo_root=gfx()["repo"])), \
            contextlib.redirect_stdout(buf):
        rc = EV.main(argv)
    res = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    return rc, (res[-1] if res else ""), buf.getvalue()


@case("profile")
def profile_e6kd_passes():
    bad = mismatches(kd_row())
    check("profile_e6kd_passes", bad == {}, f"{sorted(bad)} {[(k, bad[k]) for k in sorted(bad)][:2]}")


@case("profile")
def profile_passes_through_main_exit_0():
    rc, res, out = ev_main(profile_argv(TMP / "profile" / "ok"), kd_row())
    check("profile_passes_through_main_exit_0", rc == 0 and res == "RESULT: CHECK-RUN-META PASS", f"{rc} {res} {out[-200:]}")


def set_row(**edits) -> dict:
    row = kd_row()
    for k, v in edits.items():
        set_dotted(row, k.replace("__", "."), v)
    return row


PROFILE_STOPS = [
    ("profile_stops_on_student_image", lambda: mismatches(set_row(image_digest=Q.IMAGE_DIGEST_OF_RECORD)), "image_digest"),
    ("profile_stops_on_weights", lambda: mismatches(set_row(weights=dict(E3W))), "weights"),
    ("profile_stops_on_T_logit", lambda: mismatches(set_row(T_logit=2)), "T_logit"),
    ("profile_stops_on_teacher_provenance",
     lambda: mismatches(set_row(teacher_provenance={**provenance_of_record(), "builder": "another_builder"})),
     "teacher_provenance"),
    ("profile_stops_on_teacher_sha256_flag",
     lambda: mismatches(kd_row(), **{"--teacher-ckpt-sha256": "e" * 64}), "teacher_provenance"),
    ("profile_stops_on_never_observed_four", lambda: mismatches(set_row(never_observed_modules=NEVER_OBSERVED[:4])),
     "never_observed_modules"),
    ("profile_stops_on_one_optimizer_group", lambda: mismatches(set_row(optimizer__param_groups=1)), "optimizer"),
    ("profile_stops_on_trigger_binding",
     lambda: mismatches(set_row(trigger_binding={**kd_row()["trigger_binding"], "clip_norm": 1.0})), "trigger_binding"),
    ("profile_stops_on_projection_source",
     lambda: mismatches(set_row(projection_source={**kd_row()["projection_source"], "sha256": "f" * 64})),
     "projection_source"),
    ("profile_stops_on_teacher_nmf_draws",
     lambda: mismatches(set_row(teacher_nmf={**kd_row()["teacher_nmf"], "draws": 3})), "teacher_nmf"),
    ("profile_stops_on_cwd_projection_loaded", lambda: mismatches({**kd_row(), "cwd_projection_loaded": False}),
     "cwd_projection_loaded"),
    ("profile_stops_on_parent_not_alpha_winner",
     lambda: mismatches(set_row(parent={**kd_row()["parent"], "run_id": "e3_s42_lambda0.5_alpha25_80k_a2"})),
     "e6kd_binding"),
    ("profile_stops_on_lambda_selection_sha256",
     lambda: mismatches(set_row(e6_selections={**kd_row()["e6_selections"],
                                               "lambda_selection": {"sha256": "0" * 64}})),
     "e6_selections.lambda_selection"),
    ("profile_stops_on_e3_weights", lambda: mismatches(set_row(e3_weights={**E3W, "cwd_feat": 50.0})), "e3_weights"),
    ("profile_stops_on_ramp_iters", lambda: mismatches(set_row(ramp_iters=334)), "ramp_iters"),
]


def register_profile_stop(name: str, make, key: str) -> None:
    def fn():
        bad = make()
        check(name, key in bad, f"{sorted(bad)}")
    CASES.append((name, "profile", fn))


for _name, _make, _key in PROFILE_STOPS:
    register_profile_stop(_name, _make, _key)


@case("profile")
def profile_e3_weights_by_value():
    row = set_row(e3_weights={"logit_kd": 0.5, "cwd_feat": 25, "cwd_logit": 3.0})
    bad = mismatches(row)
    check("profile_e3_weights_by_value", "e3_weights" not in bad and "weights" not in bad, f"{sorted(bad)}")


@case("profile")
def profile_stops_on_e6_row():
    row = copy.deepcopy(meta_of(e6_short()))
    for k, v in {**LAUNCH, "parent": kd_row()["parent"]}.items():
        if k in ("ramp_iters", "teacher_mock"):
            continue
        set_dotted(row, k, v)
    bad = mismatches(row)
    check("profile_stops_on_e6_row", {"e6kd", "schema", "weights", "teacher_provenance"} <= set(bad), f"{sorted(bad)}")


@case("profile")
def profile_run_without_telemetry_exit_3():
    d = TMP / "profile" / "no_telemetry"
    d.mkdir(parents=True)
    rc, res, _ = ev_main(profile_argv(d))
    check("profile_run_without_telemetry_exit_3", rc == 3 and "[telemetry_missing]" in res, res)


@case("profile")
def profile_first_row_not_json():
    rc, res, _ = ev_main(profile_argv(TMP / "profile" / "not_json"), first_line='{"event": "run_meta", "lr": NaN}')
    check("profile_first_row_not_json", rc == 2 and "[run_meta_rows]" in res, res)


@case("profile")
def profile_first_row_not_run_meta():
    rc, res, _ = ev_main(profile_argv(TMP / "profile" / "not_meta"), first_line=json.dumps({"event": "train", "step": 1}))
    check("profile_first_row_not_run_meta", rc == 2 and "[run_meta_rows]" in res, res)


@case("profile")
def profile_mismatch_stops_exit_1():
    rc, res, out = ev_main(profile_argv(TMP / "profile" / "weights_off"), set_row(weights=dict(E3W)))
    check("profile_mismatch_stops_exit_1", rc == 1 and "[run_meta_profile]" in res and "weights" in out, res)


@case("profile")
def profile_takes_no_stage_flag():
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            EV.build_parser().parse_args(profile_argv(TMP) + ["--stage", "E6-KD"])
        got = "accepted"
    except SystemExit:
        got = "usage"
    check("profile_takes_no_stage_flag", got == "usage", got)


# ---------------------------------------------------------------- main
def forbidden_names_created() -> list[str]:
    """Every path under this smoke's own temp directory whose name holds the token SL-1 forbids, or ':'."""
    token = "".join(("t", "e", "s", "t"))
    bad = []
    for d, dirs, files in os.walk(TMP):
        for n in dirs + files:
            if token in n.lower() or ":" in n:
                bad.append(os.path.join(d, n))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--only", default=None, help="comma-separated case names (scripts/smoke_e6kd_mutations.py)")
    args = ap.parse_args()
    want = set(args.sections.split(","))
    names = [n for n, _, _ in CASES]
    token = "".join(("t", "e", "s", "t"))
    if want - set(SECTIONS) or len(set(names)) != len(names) or any(token in n.lower() for n in names):
        print(f"RESULT: ERROR unknown sections {sorted(want - set(SECTIONS))}, or a duplicate or forbidden case name")
        return 4
    if args.only:
        only = args.only.split(",")
        if set(only) - set(names):
            print(f"RESULT: ERROR unknown cases {sorted(set(only) - set(names))}")
            return 4
        todo = [c for c in CASES if c[0] in only]
    else:
        todo = [c for c in CASES if c[1] in want]
    t_all = time.time()
    print("=" * 78)
    print("E6-KD RUNNER SMOKE (lane L-AM3, B-2) -- the trainer, the launch gates and check-run-meta, synthetic")
    print(f"{len(todo)} cases | sections {sorted({c[1] for c in todo})} | temp {TMP}")
    print("=" * 78)
    for name, _section, fn in todo:
        before = len(results)
        t0 = time.time()
        try:
            fn()
        except Exception as e:                               # noqa: BLE001 -- the case fails, the smoke goes on
            check(name, False, f"the case raised {type(e).__name__}: {str(e)[:300]}")
        if len(results) == before:
            check(name, False, "the case reported no check")
        print(f"[case] {name} ({time.time() - t0:.1f}s)", flush=True)
    bad = forbidden_names_created()
    check("no_forbidden_named_path_created", not bad, f"{len(bad)} path(s), e.g. {bad[:2]}")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        detail = detail.replace("\n", " | ")                  # one line per check: no detail line reads as a RESULT
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n[time] {time.time() - t_all:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
