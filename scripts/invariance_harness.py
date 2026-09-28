#!/usr/bin/env python3
"""Invariance harness for the distillation trainer (docs/lane_specs/part2.md, cross-lane rules).

Runs `src.training.train_distill.run` for N steps on a synthetic TRAIN/VAL set (16 + 4 random
512x512 RGB images with 116-class block masks), batch 2, num_workers 0, CPU, seed 42 (run() applies
it through set_seed, with the determinism settings on), and writes ONE JSON PER STEP:

    the telemetry row, verbatim           sha256 of the student / projection state_dict
    sha256 of the student / projection    sha256 of that step's teacher logits and Stage-3 map
      gradients the step applied          teacher forward count; NMF `_build_bases` count and
    sha256 of the CPU / numpy / python      stream position (M4-KD)
      RNG state after the step            the validation result, on validation steps

The file has two roles:
  * a library imported by scripts/smoke_invariance_distill.py: synthetic data, `git archive`
    exports, worker launch, step-JSON comparison;
  * the worker, run as `python scripts/invariance_harness.py worker ...` in a fresh process whose
    sys.path starts at the code root under test: a `git archive` export of a commit, or the working
    tree. The worker never edits the code under test. It only rebinds module attributes (`_jsonl`,
    `build_student`, `build_cwd_projection`, `validate`, `save_distill_checkpoint`) and adds a forward
    hook to the teacher, so the pre-lane (73fd4d7) and the lane code run through the identical
    driver. A cross-commit run passes when every step JSON is byte-identical, except fields the lane
    declares new.

Teachers. `stub`: a fixed-weight tiny MSCAN-shaped network with the teacher of record's output
shapes ([B,116,64,64] logits, [B,320,32,32] Stage 3). Its ham head draws its NMF bases through the
attached NMFStream in `_build_bases`, and it is wrapped in the code-under-test's own
SegNeXtTeacherAdapter and FrozenTeacher, so the M4-KD stream path is the real one. `real`:
load_frozen_teacher(--teacher-ckpt, --teacher-config) from the code under test; local only (teacher
image and the checkpoint of record).

Exports use `git archive <commit> -- <explicit paths>` (docs/lane_specs/errata.md E-5): src, configs
and the class-weight file the trainer reads. Nothing under docs/ is exported, read or listed.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HARNESS = Path(__file__).resolve()
REPO = HARNESS.parents[1]

EXPORT_PATHS = ("src", "configs", "reports/e1_class_weights.json")
EXPORT_EXCLUDE = ":(exclude)docs/reference/reference.pdf"     # errata E-5: the hook's exclude suffix
N_TRAIN, N_VAL, SIZE = 16, 4, 512
NUM_CLASSES = 116
DATA_SEED = 20260928          # synthetic images and masks
STUB_SEED = 4242              # stub-teacher weights (a private generator; the global RNG is untouched)
SEED = 42
BATCH = 2
STEP_GLOB = "step_*.json"
SUMMARY = "run_summary.json"


# ------------------------------------------------------------------------------------------ hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dumps(obj) -> str:
    """The one serialisation of a step record: insertion-ordered keys, shortest-repr floats."""
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), allow_nan=True) + "\n"


# ------------------------------------------------------------------------------ synthetic data
def make_synthetic_dataset(root: Path, n_train: int = N_TRAIN, n_val: int = N_VAL, size: int = SIZE,
                           seed: int = DATA_SEED) -> dict:
    """TRAIN/VAL only (no TEST surface), in the loader's layout: images/<split>/*.jpg and
    annotations/<split>/<stem>.png. Masks are 116-class blocks (8x8 cells, 40% background)."""
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(seed)
    files = {}
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir, ann_dir = root / "images" / split, root / "annotations" / split
        img_dir.mkdir(parents=True, exist_ok=False)
        ann_dir.mkdir(parents=True, exist_ok=False)
        for i in range(n):
            stem = f"k1_{split}_{i:02d}"
            img = rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
            cells = rng.randint(0, NUM_CLASSES, size=(8, 8))
            cells[rng.rand(8, 8) < 0.4] = 0
            mask = np.kron(cells, np.ones((size // 8, size // 8), dtype=np.int64)).astype(np.uint8)
            Image.fromarray(img, "RGB").save(img_dir / f"{stem}.jpg", quality=95)
            Image.fromarray(mask, "L").save(ann_dir / f"{stem}.png")
            for p in (img_dir / f"{stem}.jpg", ann_dir / f"{stem}.png"):
                files[p.relative_to(root).as_posix()] = sha256_file(p)
    return {"root": str(root), "n_train": n_train, "n_val": n_val, "size": size, "seed": seed,
            "files": files, "manifest_sha256": sha256_bytes(dumps(files).encode())}


def dataset_layout(root: Path) -> list[str]:
    """The directories present under the synthetic root (used to prove no TEST surface exists)."""
    return sorted(p.relative_to(root).as_posix() for p in root.glob("*/*") if p.is_dir())


# ------------------------------------------------------------------------------------- exports
def resolve_commit(repo: Path, commit: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def export_commit(repo: Path, commit: str, dest: Path, paths=EXPORT_PATHS) -> dict:
    """`git archive <commit> -- <explicit paths> <exclude suffix>` extracted into `dest` (errata E-5).

    Every member must lie under one of `paths`; anything else, and anything under docs/, aborts."""
    full = resolve_commit(repo, commit)
    tar_bytes = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", full, "--", *paths,
                                EXPORT_EXCLUDE], capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        members = tf.getmembers()
        for m in members:
            name = m.name.rstrip("/")
            inside = any(name == p or name.startswith(p.rstrip("/") + "/") for p in paths)
            parent_dir = m.isdir() and any(p.startswith(name + "/") for p in paths)
            if name.startswith("docs") or not (inside or parent_dir):
                raise RuntimeError(f"export of {full}: unexpected member {m.name!r}")
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(dest, **kwargs)
    return {"commit": full, "paths": list(paths), "n_files": sum(1 for m in members if m.isfile()),
            "tar_sha256": sha256_bytes(tar_bytes)}


# --------------------------------------------------------------------------------- worker launch
def run_worker(*, code_root: Path, data_root: Path, out_dir: Path, stage: str, steps: int,
               val_interval: int = 4, teacher: str = "stub", teacher_ckpt=None, teacher_config=None,
               alpha: float | None = None, grad_clip_norm: float | None = None, threads: int = 4,
               python: str = sys.executable, timeout: int = 7200) -> dict:
    """Run one worker in a fresh process and return its exit code and run summary."""
    cmd = [python, str(HARNESS), "worker", "--code-root", str(code_root), "--data-root", str(data_root),
           "--out", str(out_dir), "--stage", stage, "--steps", str(steps),
           "--val-interval", str(val_interval), "--teacher", teacher, "--threads", str(threads)]
    if alpha is not None:
        cmd += ["--alpha", repr(float(alpha))]
    if grad_clip_norm is not None:
        cmd += ["--grad-clip-norm", repr(float(grad_clip_norm))]
    if teacher_ckpt:
        cmd += ["--teacher-ckpt", str(teacher_ckpt)]
    if teacher_config:
        cmd += ["--teacher-config", str(teacher_config)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads), "PLANTSEG_DATA_ROOT": str(data_root)})
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout,
                          cwd=str(out_dir.parent))
    log = out_dir.parent / f"{out_dir.name}.worker.log"
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    summary_path = out_dir / SUMMARY
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {"returncode": proc.returncode, "summary": summary, "log": str(log)}


# ---------------------------------------------------------------------------------- comparison
def step_files(d: Path) -> list[Path]:
    return sorted(d.glob(STEP_GLOB))


def first_difference(a, b, path: str = ""):
    """The first differing leaf between two decoded JSON values, as (path, a, b); None if equal."""
    if type(a) is not type(b):
        return (path or "<root>", a, b)
    if isinstance(a, dict):
        if list(a) != list(b):
            return (f"{path}<keys>", list(a), list(b))
        for k in a:
            d = first_difference(a[k], b[k], f"{path}.{k}" if path else k)
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return (f"{path}<len>", len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, float) and isinstance(b, float):
        same = (a == b and math.copysign(1.0, a) == math.copysign(1.0, b)) or (a != a and b != b)
        return None if same else (path, a, b)
    return None if a == b else (path, a, b)


def _drop_field(obj: dict, dotted: str) -> None:
    head, _, rest = dotted.partition(".")
    if head in obj:
        if rest and isinstance(obj[head], dict):
            _drop_field(obj[head], rest)
        elif not rest:
            del obj[head]


def compare_step_dirs(old_dir: Path, new_dir: Path, declared_new=()) -> dict:
    """Byte comparison of two runs' step JSONs. A declared-new field is removed from the NEW side
    only, so a field that also exists on the old side still has to match."""
    a, b = step_files(old_dir), step_files(new_dir)
    out = {"old": str(old_dir), "new": str(new_dir), "n_old": len(a), "n_new": len(b),
           "same_names": [p.name for p in a] == [p.name for p in b],
           "declared_new": list(declared_new), "mismatches": []}
    for pa, pb in zip(a, b):
        ba, bb = pa.read_bytes(), pb.read_bytes()
        if declared_new:
            jb = json.loads(bb)
            for field in declared_new:
                _drop_field(jb, field)
            bb = dumps(jb).encode()
        if ba != bb:
            diff = first_difference(json.loads(ba), json.loads(bb))
            out["mismatches"].append({"file": pa.name, "first_difference": diff})
    out["identical"] = bool(a) and out["same_names"] and not out["mismatches"]
    return out


# ======================================================================================= worker
def _torch_helpers():
    import numpy as np
    import random
    import torch

    def sha_tensors(items) -> str:
        h = hashlib.sha256()
        for name, t in items:
            h.update(str(name).encode() + b"\0")
            if t is None:
                h.update(b"<none>\0")
                continue
            t = t.detach().cpu().contiguous()
            h.update(f"{t.dtype}|{tuple(t.shape)}\0".encode())
            h.update(t.numpy().tobytes())
        return h.hexdigest()

    def rng_hashes() -> dict:
        st = np.random.get_state()
        return {"torch_cpu": sha256_bytes(torch.get_rng_state().numpy().tobytes()),
                "numpy": sha256_bytes(st[1].tobytes() + repr((st[0], st[2], st[3], st[4])).encode()),
                "python": sha256_bytes(repr(random.getstate()).encode())}

    def bitwise_equal(x, y) -> bool:
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            return (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype == y.dtype
                    and tuple(x.shape) == tuple(y.shape)
                    and x.detach().cpu().contiguous().numpy().tobytes()
                    == y.detach().cpu().contiguous().numpy().tobytes())
        if isinstance(x, dict) and isinstance(y, dict):
            return list(x) == list(y) and all(bitwise_equal(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return len(x) == len(y) and all(bitwise_equal(p, q) for p, q in zip(x, y))
        return type(x) is type(y) and x == y

    return torch, sha_tensors, rng_hashes, bitwise_equal


def build_stub_segnext():
    """The fixed-weight stub teacher model (SegNeXt surface: extract_feat + decode_head)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class StubMSCAN(nn.Module):
        """Stage features at strides 4/8/16/32 with MSCAN-B widths 64/128/320/512."""

        def __init__(self):
            super().__init__()
            self.s1 = nn.Conv2d(3, 64, 1, bias=False)
            self.s2 = nn.Conv2d(64, 128, 1, bias=False)
            self.s3 = nn.Conv2d(128, 320, 1, bias=False)
            self.s4 = nn.Conv2d(320, 512, 1, bias=False)

        def forward(self, x):
            f1 = torch.tanh(self.s1(F.avg_pool2d(x, 4)))
            f2 = torch.tanh(self.s2(F.avg_pool2d(f1, 2)))
            f3 = torch.tanh(self.s3(F.avg_pool2d(f2, 2)))
            f4 = torch.tanh(self.s4(F.avg_pool2d(f3, 2)))
            return (f1, f2, f3, f4)

    class StubIsolatedNMF2D(nn.Module):
        """Stand-in for IsolatedNMF2D: `_build_bases` draws torch.rand((B*S, D, R)) through the
        attached NMFStream (M4-KD), then a few multiplicative NMF updates use the bases."""
        SUPPORTS_NMF_STREAM = True

        def __init__(self, S: int = 1, R: int = 8, steps: int = 2):
            super().__init__()
            self.S, self.R, self.steps = S, R, steps
            self.nmf_stream = None
            self.build_bases_calls = 0

        def _build_bases(self, B, S, D, R):
            self.build_bases_calls += 1
            if self.nmf_stream is None:
                raise RuntimeError("stub NMF: no stream attached (the harness always runs M4-KD)")
            return self.nmf_stream.draw(lambda: F.normalize(torch.rand((B * S, D, R)), dim=1))

        def forward(self, x):
            B, C, H, W = x.shape
            D, N = C // self.S, H * W
            x = x.reshape(B * self.S, D, N)
            bases = self._build_bases(B, self.S, D, self.R)
            coef = F.softmax(torch.bmm(x.transpose(1, 2), bases), dim=-1)
            for _ in range(self.steps):
                num = torch.bmm(x.transpose(1, 2), bases)
                den = coef.bmm(bases.transpose(1, 2).bmm(bases))
                coef = coef * num / (den + 1e-6)
                num = torch.bmm(x, coef)
                den = bases.bmm(coef.transpose(1, 2).bmm(coef))
                bases = bases * num / (den + 1e-6)
            return torch.bmm(bases, coef.transpose(1, 2)).reshape(B, C, H, W)

    class StubHamHead(nn.Module):
        """LightHamHead shape: stages 2-4 resized to stride 8, squeezed, NMF, 116-class classifier."""

        def __init__(self):
            super().__init__()
            self.num_classes = NUM_CLASSES
            self.squeeze = nn.Conv2d(128 + 320 + 512, 32, 1, bias=False)
            self.ham = StubIsolatedNMF2D()
            self.conv_seg = nn.Conv2d(32, NUM_CLASSES, 1)

        def forward(self, feats):
            inputs = feats[1:]
            size = inputs[0].shape[-2:]
            x = torch.cat([inputs[0]] + [F.interpolate(t, size=size, mode="bilinear",
                                                       align_corners=False) for t in inputs[1:]], 1)
            return self.conv_seg(self.ham(F.relu(self.squeeze(x))))

    class StubSegNeXt(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = StubMSCAN()
            self.decode_head = StubHamHead()

        def extract_feat(self, x):
            return self.backbone(x)

    model = StubSegNeXt()
    g = torch.Generator().manual_seed(STUB_SEED)
    with torch.no_grad():
        for _, p in model.named_parameters():
            scale = 2.0 / math.sqrt(p[0].numel()) if p.dim() > 1 else 0.1
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return model


class _Recorder:
    """Collects one record per training step through rebound module attributes."""

    def __init__(self, td, cp, teacher, helpers):
        self.torch, self.sha_tensors, self.rng_hashes, self.bitwise_equal = helpers
        self.td, self.cp, self.teacher = td, cp, teacher
        self.student = None
        self.projection = None
        self.projection_builds = 0
        self.optimizer = None
        self.steps: list[dict] = []
        self.t_count = 0
        self.t_logits = self.t_stage3 = None
        self.t_shapes = None
        self.draws_prev = 0
        self.snapshot = None
        self.originals = {}

    # --- rebinding ---------------------------------------------------------------------------
    def install(self) -> None:
        td, cp = self.td, self.cp
        self.originals = {"td._jsonl": td._jsonl, "td.build_student": td.build_student,
                          "td.validate": td.validate,
                          "td.save_distill_checkpoint": td.save_distill_checkpoint,
                          "cp.build_cwd_projection": cp.build_cwd_projection}
        if hasattr(td, "build_cwd_projection"):
            self.originals["td.build_cwd_projection"] = td.build_cwd_projection
        o = self.originals

        def jsonl(path, rec):
            o["td._jsonl"](path, rec)
            if rec.get("event") == "train":
                self.on_train(rec)

        def build_student(*args, **kwargs):
            self.student = o["td.build_student"](*args, **kwargs)
            return self.student

        def build_cwd_projection(*args, **kwargs):
            self.projection = o["cp.build_cwd_projection"](*args, **kwargs)
            self.projection_builds += 1
            return self.projection

        def validate(*args, **kwargs):
            out = o["td.validate"](*args, **kwargs)
            if self.steps:
                all_miou, disease_miou, cm, nvb = out
                self.steps[-1]["val"] = {"all_class_miou": all_miou, "disease_only_miou": disease_miou,
                                         "cm_sha256": self.sha_tensors([("cm", cm)]), "val_batches": nvb}
            return out

        save_sig = inspect.signature(o["td.save_distill_checkpoint"])

        def save_distill_checkpoint(*args, **kwargs):
            path = o["td.save_distill_checkpoint"](*args, **kwargs)
            bound = save_sig.bind(*args, **kwargs)
            proj, opt = bound.arguments.get("projection"), bound.arguments.get("optimizer")
            self.optimizer = opt
            self.snapshot = {
                "iter": bound.arguments.get("it"), "checkpoint": str(path),
                "projection_state": None if proj is None else
                {k: v.detach().clone() for k, v in proj.state_dict().items()},
                "optimizer_state": copy.deepcopy(opt.state_dict())}
            return path

        td._jsonl = jsonl
        td.build_student = build_student
        td.validate = validate
        td.save_distill_checkpoint = save_distill_checkpoint
        cp.build_cwd_projection = build_cwd_projection
        if "td.build_cwd_projection" in o:
            td.build_cwd_projection = build_cwd_projection
        self.teacher.register_forward_hook(self.on_teacher)

    def uninstall(self) -> None:
        for key, fn in self.originals.items():
            mod, name = key.split(".", 1)
            setattr(self.td if mod == "td" else self.cp, name, fn)

    # --- hooks -------------------------------------------------------------------------------
    def on_teacher(self, _module, _inputs, output) -> None:
        self.t_count += 1
        self.t_logits = self.sha_tensors([("logits", output.logits)])
        self.t_stage3 = self.sha_tensors([("feat_s16", output.feat_s16)])
        self.t_shapes = [list(output.logits.shape),
                         None if output.feat_s16 is None else list(output.feat_s16.shape)]

    def on_train(self, row: dict) -> None:
        nmf = self.teacher.nmf_stream_state()
        draws = None if nmf is None else int(nmf["draws"])
        rec = {"step": int(row["iter"]),
               "telemetry": json.loads(json.dumps(row, ensure_ascii=False)),
               "student_sha256": self.sha_tensors(self.student.state_dict().items()),
               "student_grad_sha256": self.sha_tensors((n, p.grad) for n, p
                                                       in self.student.named_parameters())}
        if self.projection is not None:
            rec["projection_sha256"] = self.sha_tensors(self.projection.state_dict().items())
            rec["projection_grad_sha256"] = self.sha_tensors((n, p.grad) for n, p
                                                             in self.projection.named_parameters())
        rec.update({
            "teacher_forwards": self.t_count,
            "teacher_logits_sha256": self.t_logits, "teacher_stage3_sha256": self.t_stage3,
            "teacher_shapes": self.t_shapes,
            "build_bases_calls": None if draws is None else draws - self.draws_prev,
            "nmf": None if nmf is None else {k: nmf[k] for k in ("policy", "seed", "draws",
                                                                 "state_sha256")},
            "rng_after_step": self.rng_hashes()})
        self.steps.append(rec)
        self.t_count = 0
        self.draws_prev = draws or 0


def _inspect_run(torch, td, ckpt_dir: Path, stage_key: str, rec: _Recorder, out: dict) -> None:
    """Run-level facts for the lane checks (d8, run_meta, best.json, optimizer groups)."""
    from src.distill.export import find_projection_keys

    meta_path = ckpt_dir / f"{stage_key}_run_meta.jsonl"
    out["run_meta"] = ([json.loads(ln) for ln in meta_path.read_text(encoding="utf-8").splitlines()
                        if ln.strip()] if meta_path.exists() else [])
    tel_path = ckpt_dir / f"{stage_key}_telemetry.jsonl"
    rows = ([json.loads(ln) for ln in tel_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if tel_path.exists() else [])
    out["telemetry_rows"] = len(rows)
    out["telemetry_keys"] = sorted({tuple(r) for r in rows if r.get("event") == "train"})
    out["run_end"] = rows[-1] if rows and rows[-1].get("event") == "run_end" else None
    out["telemetry_finite"] = all(math.isfinite(v) for r in rows for v in r.values()
                                  if isinstance(v, float))
    out["ckpt_files"] = sorted(p.name for p in ckpt_dir.iterdir())
    ckpts = sorted(ckpt_dir.glob(f"{stage_key}_student_best_iter*.pt"),
                   key=lambda p: int(p.stem.rsplit("iter", 1)[1]))
    out["checkpoints"] = []
    for p in ckpts:
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        groups = ck.get("optimizer_state_dict", {}).get("param_groups", [])
        out["checkpoints"].append({
            "file": p.name, "keys": sorted(ck), "stage": ck.get("stage"),
            "projection_keys_in_student": find_projection_keys(ck["model_state_dict"]),
            "cwd_projection_state_dict_present": ck.get("cwd_projection_state_dict") is not None,
            "optimizer_groups": [g.get("name") for g in groups],
            "optimizer_state_entries": len(ck.get("optimizer_state_dict", {}).get("state", {})),
            "terms": ck.get("terms")})
    proj_file = ckpt_dir / "projection.pt"
    if proj_file.exists():
        pj = torch.load(str(proj_file), map_location="cpu", weights_only=False)
        sd = pj.get("projection_state_dict", {})
        out["projection_pt"] = {
            "keys": sorted(pj), "checkpoint": pj.get("checkpoint"), "iter": pj.get("iter"),
            "stage": pj.get("stage"), "state_keys": sorted(sd),
            "numel": int(sum(v.numel() for v in sd.values())),
            "optimizer_groups": [g.get("name") for g in pj.get("optimizer_state", {})
                                 .get("param_groups", [])]}
    else:
        out["projection_pt"] = None
    best = ckpt_dir / "best.json"
    out["best_json"] = json.loads(best.read_text(encoding="utf-8")) if best.exists() else None
    if rec.optimizer is not None:
        out["optimizer_groups"] = [{"name": g.get("name"), "tensors": len(g["params"]),
                                    "numel": int(sum(p.numel() for p in g["params"]))}
                                   for g in rec.optimizer.param_groups]
    out["student_params"] = int(sum(p.numel() for p in rec.student.parameters()))
    out["projection_builds"] = rec.projection_builds
    out["projection_numel"] = (None if rec.projection is None else
                               int(sum(p.numel() for p in rec.projection.parameters())))
    stub_nmf = [m for m in rec.teacher.modules() if hasattr(m, "build_bases_calls")]
    out["stub_build_bases_calls"] = sum(m.build_bases_calls for m in stub_nmf) if stub_nmf else None


def _restore_check(torch, td, cp, stage, rec: _Recorder, bitwise_equal) -> dict:
    """projection.pt is the only home of the projection: restore it with its optimizer group into
    freshly built objects and require bit-for-bit equality with the live state at the last save."""
    if not hasattr(cp, "restore_projection"):
        return {"ran": False, "reason": "code under test has no restore_projection (pre-lane)"}
    snap = rec.snapshot
    if snap is None or snap["projection_state"] is None:
        return {"ran": False, "reason": "no projection was saved (the stage has no cwd_feat term)"}
    ckpt = Path(snap["checkpoint"])
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    torch.manual_seed(987654321)            # fresh objects start from unrelated weights
    student = rec.originals["td.build_student"](pretrained=False)
    projection = cp.build_cwd_projection_for(stage)
    optimizer, _ = td.build_optimizer(student, projection)
    student.load_state_dict(ck["model_state_dict"], strict=True)
    info = cp.restore_projection(ckpt.parent / cp.PROJECTION_FILE, projection, optimizer,
                                 student_optimizer_state=ck["optimizer_state_dict"],
                                 checkpoint=ckpt)
    live_opt = snap["optimizer_state"]
    restored_opt = optimizer.state_dict()
    return {"ran": True, "iter": snap["iter"], "restore_info": info,
            "projection_weights_bitwise": bitwise_equal(
                {k: v for k, v in projection.state_dict().items()}, snap["projection_state"]),
            "optimizer_state_bitwise": bitwise_equal(restored_opt, live_opt),
            "projection_group_restored": [g.get("name") for g in restored_opt["param_groups"]]}


def _e6e7_loader_check(ckpt_dir: Path, stage_key: str) -> dict:
    """E6/E7 take E3's checkpoint through src/quant/checkpoint.py; nothing may need stripping."""
    from src.quant.checkpoint import SourceCheckpointInvalid, load_student_from_e3

    ckpts = sorted(ckpt_dir.glob(f"{stage_key}_student_best_iter*.pt"),
                   key=lambda p: int(p.stem.rsplit("iter", 1)[1]))
    if not ckpts:
        return {"ran": False}
    try:
        load_student_from_e3(ckpts[-1])
        return {"ran": True, "ok": True, "checkpoint": ckpts[-1].name}
    except SourceCheckpointInvalid as e:
        return {"ran": True, "ok": False, "checkpoint": ckpts[-1].name, "error": str(e)[:300]}


def worker(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="invariance_harness.py worker")
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--val-interval", type=int, default=4)
    ap.add_argument("--teacher", choices=("stub", "real"), default="stub")
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--teacher-config", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--grad-clip-norm", type=float, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)

    code_root = Path(a.code_root).resolve()
    data_root = Path(a.data_root).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    # The code root goes first; no other checkout of this repository may shadow it.
    sys.path[:] = [str(code_root)] + [p for p in sys.path if p and Path(p).resolve() != code_root
                                      and not (Path(p) / "src" / "training" / "train_distill.py").exists()]
    os.environ["PLANTSEG_DATA_ROOT"] = str(data_root)

    torch, sha_tensors, rng_hashes, bitwise_equal = _torch_helpers()
    torch.set_num_threads(a.threads)

    import configs.data as cdata
    counts = {s: sum(1 for p in (data_root / "images" / s).iterdir() if p.suffix == ".jpg")
              for s in ("train", "val")}
    if cdata.DATA["root"] != str(data_root):
        raise SystemExit(f"configs.data resolved root {cdata.DATA['root']!r}, expected {data_root}")
    cdata.SPLIT_SIZES.update(counts)       # the loader enforces the locked counts; this set is 16 + 4

    import src.distill.cwd_projection as cp
    import src.training.train_distill as td
    from src.distill.segnext_teacher import SegNeXtTeacherAdapter
    from src.distill.teacher import FrozenTeacher

    for name in ("src.training.train_distill", "src.training.train_e1", "src.distill.cwd_projection",
                 "src.distill.teacher", "src.distill.segnext_teacher", "src.distill.nmf_stream",
                 "src.data.dataset", "configs.data", "configs.distill", "src.training.losses"):
        f = Path(sys.modules[name].__file__).resolve()
        if code_root not in f.parents:
            raise SystemExit(f"{name} was imported from {f}, outside the code root {code_root}")

    if a.teacher == "stub":
        teacher = FrozenTeacher(SegNeXtTeacherAdapter(build_stub_segnext()))
    else:
        teacher = td.load_frozen_teacher(a.teacher_ckpt, config_path=a.teacher_config)

    stage = td.resolve_stage(a.stage)
    run_params = inspect.signature(td.run).parameters
    has_logit_kd = bool(stage.get("logit_kd", True))
    kwargs = dict(stage=stage, mode="dry", device="cpu", pretrained=False, teacher=teacher,
                  lambda_logit=1.0 if has_logit_kd else None, batch_size=BATCH, max_iters=a.steps,
                  val_interval=a.val_interval, max_val_batches=None, num_workers=0,
                  ckpt_dir_arg=None, grad_clip_norm=a.grad_clip_norm, log_every=1, seed=SEED)
    if a.alpha is not None:
        if "alpha" not in run_params:
            raise SystemExit("--alpha was given, but the code under test's run() takes no alpha")
        kwargs["alpha"] = a.alpha
    uses_alpha = bool(stage.get("cwd_feat", stage.get("cwd", False)))
    alpha_tag = "" if not uses_alpha else f"_alpha{(a.alpha if a.alpha is not None else 50):g}"
    ckpt_dir = out / f"ckpt_{stage['key']}{alpha_tag}"
    kwargs["ckpt_dir_arg"] = str(ckpt_dir)

    rec = _Recorder(td, cp, teacher, (torch, sha_tensors, rng_hashes, bitwise_equal))
    rec.install()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = td.run(**kwargs)
    rec.uninstall()
    trainer_out = buf.getvalue()
    (out / "trainer_stdout.txt").write_text(trainer_out, encoding="utf-8")

    for r in rec.steps:
        (out / f"step_{r['step']:04d}.json").write_text(dumps(r), encoding="utf-8")

    summary = {"code_root": str(code_root), "stage": stage["key"], "stage_name": stage["name"],
               "steps": a.steps, "alpha_arg": a.alpha, "grad_clip_norm": a.grad_clip_norm,
               "teacher": a.teacher, "threads": a.threads,
               "torch": torch.__version__, "trainer_rc": rc,
               "trainer_result": next((ln.strip() for ln in reversed(trainer_out.splitlines())
                                       if ln.strip().startswith("RESULT")), None),
               "train_distill_sha256": sha256_file(Path(td.__file__)),
               "n_step_records": len(rec.steps),
               "data_layout": dataset_layout(data_root), "split_counts": counts}
    _inspect_run(torch, td, ckpt_dir, stage["key"], rec, summary)
    summary["restore"] = _restore_check(torch, td, cp, stage, rec, bitwise_equal)
    summary["e6e7_loader"] = _e6e7_loader_check(ckpt_dir, stage["key"])
    (out / SUMMARY).write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.exit(worker(sys.argv[2:]))
    sys.exit("usage: invariance_harness.py worker ... (the smoke is scripts/smoke_invariance_distill.py)")
