#!/usr/bin/env python3
"""TRAIN/VAL-only E1 gate for B66 pods (DL-21; B65 R6; B66-prep S2).

The data root is touched first by `data_isolation`, where isolation.assert_trainval_only_root plus a
TRAIN/VAL stem-pairing check take the place of verify_env.py's dataset loop: TEST is existence-checked
on three exact paths and never listed or opened. verify_env.py and preflight_e1.py are NEVER executed:
preflight_e1 is loaded from <checkout>/scripts/preflight_e1.py for four stage functions and two constants,
and its main() is never called. This gate does not modify sys.path; it runs with PYTHONPATH set to the
checkout (stage `arguments` refuses anything else) and verifies where every repo module came from.

Usage:
  python -B scripts/preflight_e1_trainval.py gate --seed S --ckpt-dir D --expect-head SHA40
         --record PATH [--profile e1_80k]          (a real gate run; --record is required)
  python -B scripts/preflight_e1_trainval.py gate --rehearsal --seed S --ckpt-dir D --expect-head SHA40
         [--record PATH] [--profile e1_80k]        (off-pod rehearsal; never a GO)
  python -B scripts/preflight_e1_trainval.py check-run-meta --ckpt-dir D --seed S --expect-head SHA40
         [--profile e1_80k]
Exit: 0 GO / PASS · 1 NO-GO / FAIL (a rehearsal ALWAYS exits 1) · 2 usage.

A rehearsal (`--rehearsal`) runs every stage it can off-pod. It SKIPs only work that needs the pod
(CUDA, the pre-staged ImageNet backbone, the partial clone's sparse configuration, a safety-floor commit
absent from a history-less scratch repository); it never waives a FAIL and never prints a launch block.
The only writes are a temporary dry-run directory (removed) and `--record` (outside the repository).
"""
import sys

sys.dont_write_bytecode = True                  # X8: before the first non-stdlib import

import os  # noqa: E402

BYTECODE_ENV_AT_ENTRY = os.environ.get("PYTHONDONTWRITEBYTECODE")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"      # children inherit it (PF1._run copies os.environ)

import argparse          # noqa: E402
import hashlib           # noqa: E402
import importlib.util    # noqa: E402
import json              # noqa: E402
import platform          # noqa: E402
import re                # noqa: E402
import shlex             # noqa: E402
import shutil            # noqa: E402
import subprocess        # noqa: E402
import tempfile          # noqa: E402
import time              # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# X8: preflight_e1 is loaded from the checkout by file location (never through sys.path), and only
# its definitions execute: its top level is stdlib imports, constants and defs; main() is guarded.
_PF1_SPEC = importlib.util.spec_from_file_location("preflight_e1", str(REPO / "scripts" / "preflight_e1.py"))
PF1 = importlib.util.module_from_spec(_PF1_SPEC)
_PF1_SPEC.loader.exec_module(PF1)

# The four preflight_e1 stage functions reused as the SAME objects (Q2; AGENTS.md relies on R5's).
REUSED_PF1 = {
    "class_weights": PF1.stage_class_weights,
    "smoke_loss": PF1.stage_loss,
    "smoke_dataloader": PF1.stage_dataloader,
    "seed_sequence_R5": PF1.stage_stochasticity,
}

# ---------------------------------------------------------------------------------------- constants
TRAINVAL_COUNTS = {"train": 5367, "val": 846}               # configs/data.py SPLIT_SIZES [counted]
OFFICIAL_SEEDS = (42, 43, 44)
E1_IMAGE_DIGEST = "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"   # DL-21
IMAGENET_FILE = "mobilenet_v3_large-5c1a4163.pth"
IMAGENET_SHA256 = "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"          # seed 42
IMAGENET_BYTES = 22_132_113
IMAGENET_ALIAS = "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2"
CLASS_WEIGHTS_REL = "reports/e1_class_weights.json"
CLASS_WEIGHTS_SHA256 = "fd78ba13d0bc06c0e806a5ab0b211f0044b448a494915b6135753c9701892dcf"   # 5325e32 blob
MAX_SM = (9, 0)
E1_SAFETY_FLOOR = "885523a"
SPARSE_PATTERNS = ("/*", "!/docs/reference/")
GOVERNED_PATHSPECS = ("src", "configs", "scripts", "requirements*",
                      "docs/EVALUATION_CONTRACT.md", "docs/IMPLEMENTATION_CONTRACT.md")
STATUS_PATHSPECS = GOVERNED_PATHSPECS + (CLASS_WEIGHTS_REL,)           # stage 3 (X5)
UNCHANGED_PATHSPECS = STATUS_PATHSPECS + ("reports",)                  # stage 14 (X5 + any reports write)
E1_NUM_WORKERS = 12
LOG_EVERY = 50
PROFILES = {
    "e1_80k": {"extra_args": (), "max_iters": 80000, "poly_horizon": 80000},
    # AM-16 item 3 / DL-27 longer-schedule control (L-AM16-ITERS, E1 part): --iterations sets both
    # the poly horizon and the run length; the gate's dry run exercises the same horizon.
    "e1_160k": {"extra_args": ("--iterations", "160000"), "dry_args": ("--iterations", "160000"),
                "max_iters": 160000, "poly_horizon": 160000},
}
RUN_META_KEYS = (
    "event", "wall_clock", "mode", "seed", "git_head", "git_head_source", "image_digest", "torch",
    "numpy", "device", "cuda_available", "gpu_name", "num_workers", "batch_size", "max_iters",
    "val_interval", "max_val_batches", "ckpt_interval", "resumed_from", "num_classes", "ignore_index",
    "learning_rate", "momentum", "weight_decay", "lr_power", "poly_horizon", "grad_clip_norm",
    "used_pretrained", "params")
RUN_META_EXPECT = {                              # the MEASURED seed-42 run_meta row (29 keys)
    "event": "run_meta", "mode": "real", "git_head_source": "git_checkout",
    "image_digest": E1_IMAGE_DIGEST, "torch": "2.1.0+cu121", "numpy": "1.26.4", "device": "cuda",
    "cuda_available": True, "num_workers": 12, "batch_size": 16, "val_interval": 4000,
    "max_val_batches": None, "ckpt_interval": 2000, "resumed_from": None, "num_classes": 116,
    "ignore_index": 255, "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0001,
    "lr_power": 0.9, "grad_clip_norm": None, "used_pretrained": True, "params": 2933688,
}
RUN_META_EXEMPT = ("seed", "git_head", "wall_clock", "gpu_name", "max_iters", "poly_horizon")
EXPECTED_GPU_NAME = "NVIDIA A40"
STAGE_NAMES = ("arguments", "data_isolation", "repo_state", "module_provenance", "class_weights",
               "smoke_loss", "image", "cuda", "imagenet_backbone", "smoke_loader_seed",
               "smoke_dataloader", "seed_sequence_R5", "dry_run", "repo_unchanged")


class GateError(RuntimeError):
    """A named gate failure; the code makes every refusal independently testable."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


# ---------------------------------------------------------------------------------------- helpers
def _git(args, timeout=120):
    """git in the checkout: argv list, no shell."""
    return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, timeout=timeout)


def _child(argv, timeout=3600):
    """A child interpreter (-B), cwd = checkout, inheriting PYTHONDONTWRITEBYTECODE=1."""
    return subprocess.run([sys.executable, "-B", *[str(a) for a in argv]], cwd=str(REPO),
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=timeout)


def _under(path: Path, root: Path) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_porcelain_paths(raw: bytes) -> list:
    """Paths from `git status --porcelain=v1 -z` (a rename carries its origin as the next record)."""
    parts = [p for p in raw.split(b"\x00") if p]
    paths, i = [], 0
    while i < len(parts):
        rec = parts[i].decode("utf-8", "replace")
        if len(rec) < 4:
            i += 1
            continue
        paths.append(rec[3:])
        i += 2 if ("R" in rec[:2] or "C" in rec[:2]) else 1
    return paths


def scoped_dirty(porcelain: bytes) -> list:
    """Dirty or untracked paths under the governed prefixes or the class-weight artifact."""
    from src.eval.artifacts import GOVERNED_PREFIXES
    return sorted(p for p in parse_porcelain_paths(porcelain)
                  if p == CLASS_WEIGHTS_REL or any(p.startswith(g) for g in GOVERNED_PREFIXES))


def status_argv(pathspecs=STATUS_PATHSPECS, ignored=False) -> list:
    argv = ["-c", "core.quotepath=false", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    if ignored:
        argv.append("--ignored=matching")
    return argv + ["--", *pathspecs]


def check_backbone(path, sha=None, size=None):
    """None when the file is the seed-42 backbone; otherwise the failure code."""
    sha = IMAGENET_SHA256 if sha is None else sha            # read at call time, never frozen as defaults
    size = IMAGENET_BYTES if size is None else size
    path = Path(path)
    if not path.is_file():
        return "imagenet_not_cached"
    if path.stat().st_size != size:
        return "imagenet_size_mismatch"
    if sha256_file(path) != sha:
        return "imagenet_sha_mismatch"
    return None


def pairing_check(root: Path, split: str) -> dict:
    """Every image stem (.jpg/.jpeg, case-insensitive) has exactly <stem>.png, and vice versa.

    Mirrors PlantSegDataset's pairing (dataset.py:43-60) by name only; no file is opened.
    """
    img_dir, mask_dir = root / "images" / split, root / "annotations" / split
    imgs = [e.name for e in os.scandir(img_dir)
            if e.is_file() and os.path.splitext(e.name)[1].lower() in (".jpg", ".jpeg")]
    masks = [e.name for e in os.scandir(mask_dir)
             if e.is_file() and os.path.splitext(e.name)[1] == ".png"]
    img_stems = [os.path.splitext(n)[0] for n in imgs]
    mask_stems = {os.path.splitext(n)[0] for n in masks}
    dup = sorted({s for s in img_stems if img_stems.count(s) > 1}) if len(set(img_stems)) != len(img_stems) else []
    no_mask = sorted(n for n in imgs if os.path.splitext(n)[0] not in mask_stems)
    no_img = sorted(n for n in masks if os.path.splitext(n)[0] not in set(img_stems))
    if dup or no_mask or no_img:
        raise GateError("pairing_stem_mismatch",
                        f"split {split}: images without <stem>.png {no_mask[:5]}, masks without an "
                        f"image {no_img[:5]}, duplicate image stems {dup[:5]} -- DL-21: the staged "
                        "TRAIN/VAL root must pair every image with its mask by stem")
    return {"pairs": len(img_stems)}


def module_provenance(extra=()) -> list:
    """Every loaded src.*/configs.* module (and preflight_e1) must come from this checkout."""
    foreign = []
    for name, mod in list(sys.modules.items()):
        if name in ("src", "configs") or name.startswith(("src.", "configs.")):
            f = getattr(mod, "__file__", None)
            if f and not _under(Path(f), REPO):
                foreign.append(f"{name}={f}")
    for label, f in (("preflight_e1", PF1.__file__), *extra):
        if not f or not _under(Path(f), REPO):
            foreign.append(f"{label}={f}")
    return foreign


def _ok(detail, **record):
    return "PASS", detail, record


def _skip(detail, **record):
    return "SKIPPED", detail, record


def _pf1(name):
    ok, detail, _secs = REUSED_PF1[name](False)
    return ("PASS" if ok else "FAIL"), detail, {}


# ---------------------------------------------------------------------------------------- stages
def stage_arguments(ctx):
    raw = os.environ.get("PLANTSEG_DATA_ROOT", "").strip()
    if not raw:
        raise GateError("data_root_unset", "PLANTSEG_DATA_ROOT must name the staged TRAIN/VAL root")
    if not os.path.isabs(raw):
        raise GateError("data_root_relative", f"PLANTSEG_DATA_ROOT must be absolute: {raw!r}")
    root = Path(raw)
    pp = os.environ.get("PYTHONPATH", "").strip()
    entries = [e for e in pp.split(os.pathsep) if e]
    if len(entries) != 1 or Path(entries[0]).resolve() != REPO:
        raise GateError("pythonpath_not_checkout",
                        f"PYTHONPATH must be exactly the checkout {REPO} (got {pp!r}); the image's baked "
                        "PYTHONPATH points at a stale source copy")
    ck = ctx["ckpt_dir"]
    if not os.path.isabs(ck):
        raise GateError("ckpt_dir_relative", f"--ckpt-dir must be absolute: {ck!r}")
    ck = Path(ck)
    if _under(ck, REPO):
        raise GateError("ckpt_dir_inside_repo", f"--ckpt-dir {ck} is inside the repository")
    if _under(ck, root):
        raise GateError("ckpt_dir_inside_data_root", f"--ckpt-dir {ck} is inside the data root")
    if ck.exists() and (not ck.is_dir() or any(ck.iterdir())):
        raise GateError("ckpt_dir_not_fresh", f"--ckpt-dir {ck} exists and is not an empty directory")
    rec = ctx["record"]
    if rec is not None:
        rp = Path(rec)
        if not rp.is_absolute():
            raise GateError("record_relative", f"--record must be absolute: {rec!r}")
        if _under(rp, REPO):
            raise GateError("record_inside_repo", f"--record {rp} is inside the repository")
        if _under(rp, ck):
            raise GateError("record_inside_ckpt_dir", f"--record {rp} is inside --ckpt-dir")
        if _under(rp, root):
            raise GateError("record_inside_data_root", f"--record {rp} is inside the data root")
        if rp.exists():
            raise GateError("record_exists", f"--record {rp} already exists")
        if not rp.parent.is_dir():
            raise GateError("record_parent_missing", f"--record's directory {rp.parent} does not exist")
    ctx["root"] = root
    return _ok(f"root={root} ckpt_dir={ck} (absent or empty) PYTHONPATH=checkout",
               data_root=str(root), ckpt_dir=str(ck))


def stage_data_isolation(ctx):
    import src.data.isolation as iso_mod
    from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
    if not _under(Path(iso_mod.__file__), REPO):
        raise GateError("import_origin_foreign", f"src.data.isolation loaded from {iso_mod.__file__}")
    root = ctx["root"]
    try:
        iso = assert_trainval_only_root(root, dict(TRAINVAL_COUNTS))
    except TrainValIsolationError as e:
        raise GateError(e.code, f"{e} -- DL-21: B66 E1 roots are staged TRAIN/VAL-only; the shared "
                                "M11 message names teacher/E2/E3") from e
    pairing = {split: pairing_check(root, split)["pairs"] for split in ("train", "val")}
    ctx["isolation"] = {**iso, "pairing": pairing}
    return _ok(f"TEST surfaces absent (exact-path lexists); counts {iso['counts']}; pairs {pairing}",
               isolation=ctx["isolation"])


def stage_repo_state(ctx):
    exp = ctx["expect_head"]
    if not re.fullmatch(r"[0-9a-f]{40}", exp or ""):
        raise GateError("expect_commit_malformed", f"--expect-head must be 40 lowercase hex: {exp!r}")
    r = _git(["rev-parse", "HEAD"])
    head = r.stdout.decode().strip()
    if r.returncode != 0 or head != exp:
        raise GateError("head_mismatch", f"HEAD {head or '?'} != --expect-head {exp}")
    floor = _git(["cat-file", "-e", f"{E1_SAFETY_FLOOR}^{{commit}}"])
    count = _git(["rev-list", "--count", "HEAD"])
    history_less = count.returncode == 0 and count.stdout.decode().strip() == "1"
    if floor.returncode != 0 and ctx["rehearsal"] and history_less:
        floor_detail = (f"safety floor {E1_SAFETY_FLOOR}: SKIPPED (rehearsal; a history-less scratch "
                        "repository cannot contain it)")
    else:
        anc = _git(["merge-base", "--is-ancestor", E1_SAFETY_FLOOR, "HEAD"])
        if anc.returncode != 0:
            raise GateError("safety_floor_missing", f"{E1_SAFETY_FLOOR} is not an ancestor of HEAD")
        floor_detail = f"safety floor {E1_SAFETY_FLOOR} is an ancestor"
    st = _git(status_argv())
    if st.returncode != 0:
        raise GateError("governed_state_unprovable",
                        f"scoped git status failed ({st.returncode}): {st.stderr.decode()[:200]}")
    dirty = scoped_dirty(st.stdout)
    if dirty:
        raise GateError("governed_paths_dirty", f"governed paths dirty or untracked: {dirty[:10]}")
    if os.environ.get("PLANTSEG_GIT_COMMIT", "").strip():
        raise GateError("git_commit_env_set",
                        "PLANTSEG_GIT_COMMIT is set (the image bakes f77d05d7, and train_e1 prefers it "
                        "over the checkout); unset it")
    if ctx["rehearsal"]:
        sparse_detail = "sparse checkout: SKIPPED (rehearsal)"
    else:
        sc = _git(["config", "--get", "core.sparseCheckout"]).stdout.decode().strip()
        cone = _git(["config", "--get", "core.sparseCheckoutCone"]).stdout.decode().strip()
        gp = _git(["rev-parse", "--git-path", "info/sparse-checkout"]).stdout.decode().strip()
        pfile = (REPO / gp) if gp and not os.path.isabs(gp) else Path(gp or "/nonexistent")
        pats = tuple(ln.strip() for ln in pfile.read_text(encoding="utf-8").splitlines()
                     if ln.strip()) if pfile.is_file() else ()
        if sc != "true" or cone != "false" or pats != SPARSE_PATTERNS:
            raise GateError("sparse_checkout_mismatch",
                            f"core.sparseCheckout={sc!r} core.sparseCheckoutCone={cone!r} patterns={pats} "
                            f"(DL-19 partial clone requires true/false/{SPARSE_PATTERNS})")
        sparse_detail = f"sparse checkout {SPARSE_PATTERNS} (non-cone)"
    snap = _git(status_argv(UNCHANGED_PATHSPECS, ignored=True))
    ctx["snapshot"] = snap.stdout if snap.returncode == 0 else None
    origin = _git(["config", "--get", "remote.origin.url"]).stdout.decode().strip() or None
    return _ok(f"HEAD={head}; {floor_detail}; scoped status clean; PLANTSEG_GIT_COMMIT unset; "
               f"{sparse_detail}", head=head, remote_origin_url=origin)


def stage_module_provenance(ctx):
    import configs.data  # noqa: F401 -- the modules later in-process stages use, loaded now
    import src.data.isolation  # noqa: F401
    import src.eval.artifacts  # noqa: F401
    import src.models.student  # noqa: F401
    foreign = module_provenance()
    if foreign:
        raise GateError("import_origin_foreign", f"modules loaded from outside {REPO}: {foreign[:5]}")
    return _ok(f"all src.*/configs.* modules and preflight_e1 resolve under {REPO}",
               preflight_e1_file=PF1.__file__)


def stage_class_weights(ctx):
    status, detail, rec = _pf1("class_weights")
    if status != "PASS":
        return status, detail, rec
    sha = sha256_file(REPO / CLASS_WEIGHTS_REL)
    if sha != CLASS_WEIGHTS_SHA256:
        raise GateError("class_weights_sha_mismatch",
                        f"sha256({CLASS_WEIGHTS_REL}) = {sha} != {CLASS_WEIGHTS_SHA256} (5325e32 blob)")
    return _ok(f"{detail}; sha256 {sha[:16]}… == 5325e32 blob", class_weights_sha256=sha)


def stage_smoke_loss(ctx):
    return _pf1("smoke_loss")


def stage_image(ctx):
    digest = os.environ.get("PLANTSEG_IMAGE_DIGEST", "").strip()
    if not digest:
        raise GateError("image_digest_missing", "PLANTSEG_IMAGE_DIGEST is not set (DL-21)")
    if digest != E1_IMAGE_DIGEST:
        raise GateError("image_digest_mismatch", f"PLANTSEG_IMAGE_DIGEST {digest} != DL-21 {E1_IMAGE_DIGEST}")
    mode = "image" if ctx["rehearsal"] else "gpu"
    r = _child([REPO / "scripts" / "preflight_environment.py", "--mode", mode], timeout=900)
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("RESULT:")), "")
    if r.returncode != 0 or not line.startswith("RESULT: PASS"):
        fails = [ln.strip() for ln in r.stdout.splitlines() if ln.rstrip().endswith("FAIL") or ": FAIL" in ln]
        raise GateError("environment_preflight_failed",
                        f"preflight_environment --mode {mode}: rc={r.returncode} {line!r} {fails[:5]}")
    return _ok(f"digest == DL-21; preflight_environment --mode {mode}: {line}", image_digest=digest,
               environment_mode=mode, environment_result=line)


def stage_cuda(ctx):
    if ctx["rehearsal"]:
        return _skip("SKIPPED (rehearsal: no CUDA probe off-pod)")
    r = _child([Path(__file__).resolve(), "_cuda-probe"], timeout=300)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("CUDA_PROBE_JSON:")]
    if r.returncode != 0 or not lines:
        raise GateError("cuda_probe_failed", f"rc={r.returncode} {(r.stdout + r.stderr)[-400:]}")
    rec = json.loads(lines[-1][len("CUDA_PROBE_JSON:"):])
    if not rec.get("ok"):
        raise GateError(rec.get("code") or "cuda_probe_failed", rec.get("detail", ""))
    ctx["cuda"] = rec
    return _ok(f"{rec['gpu_name']} sm_{rec['capability']} alloc+kernel+sync OK, student forward "
               f"{rec['forward_shape']} finite", cuda=rec)


def cuda_probe() -> int:
    """Child process only: never runs in the gate's own interpreter."""
    rec = {"ok": False, "cpu_count": os.cpu_count(), "runpod_cpu_count": os.environ.get("RUNPOD_CPU_COUNT")}
    try:
        shm = shutil.disk_usage("/dev/shm") if os.path.isdir("/dev/shm") else None
        rec["dev_shm_bytes"] = shm.total if shm else None
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            rec.update(code="cuda_unavailable", detail="torch.cuda.is_available() is False")
        else:
            caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
            bad = [c for c in caps if tuple(c) > MAX_SM]
            if bad:
                rec.update(code="compute_capability_unsupported", detail=f"{caps} > sm_{MAX_SM}")
            else:
                x = torch.ones(1 << 20, device="cuda")
                total = float((x * 2).sum())
                torch.cuda.synchronize()
                from src.models.student import build_student
                model = build_student(pretrained=False).cuda().eval()
                with torch.no_grad():
                    out = model(torch.zeros(1, 3, 512, 512, device="cuda"))
                torch.cuda.synchronize()
                shape_ok = tuple(out.shape[:2]) == (1, 116)
                finite = bool(torch.isfinite(out).all())
                rec.update(gpu_name=torch.cuda.get_device_name(0),
                           capability=f"{caps[0][0]}{caps[0][1]}",
                           total_memory=torch.cuda.get_device_properties(0).total_memory,
                           forward_shape=list(out.shape), alloc_sum=total)
                if total != 2.0 * (1 << 20) or not shape_ok or not finite:
                    rec.update(code="cuda_kernel_check_failed",
                               detail=f"sum={total} shape={list(out.shape)} finite={finite}")
                else:
                    rec["ok"] = True
    except Exception as e:                                # noqa: BLE001 -- reported, not raised
        rec.update(code="cuda_probe_error", detail=f"{type(e).__name__}: {e}")
    print("CUDA_PROBE_JSON:" + json.dumps(rec, sort_keys=True, default=str), flush=True)
    return 0


def stage_imagenet_backbone(ctx):
    import torch
    raw_hub = torch.hub.get_dir()
    # Absolute, symlinks NOT followed: the launch block pins TORCH_HOME=<hub>/.. and torch then reads
    # $TORCH_HOME/hub, which reproduces this directory only if it is literally named "hub".
    hub = Path(os.path.abspath(raw_hub))
    if hub.name != "hub":
        raise GateError("hub_dir_nonstandard",
                        f"torch.hub.get_dir() = {raw_hub!r} is not <TORCH_HOME>/hub; the launch cannot pin it")
    path = hub / "checkpoints" / IMAGENET_FILE
    code = check_backbone(path)
    if code == "imagenet_not_cached" and ctx["rehearsal"]:
        return _skip(f"SKIPPED (rehearsal: {path} not pre-staged)", hub_dir=str(hub), hub_dir_raw=raw_hub)
    if code:
        raise GateError(code, f"{path}: expected {IMAGENET_BYTES} B, sha256 {IMAGENET_SHA256} (Q13: "
                              "pre-stage the backbone by a verified fetch or transfer; no in-run download)")
    from src.models.student import build_student
    model = build_student(pretrained=IMAGENET_ALIAS)
    if getattr(model, "used_pretrained", False) is not True:
        raise GateError("imagenet_not_loaded", "build_student(IMAGENET1K_V2) did not report used_pretrained")
    ctx["hub_dir"] = str(hub)
    return _ok(f"{path} sha256 == seed-42 capture; cached build_student(IMAGENET1K_V2) loaded",
               hub_dir=str(hub), hub_dir_raw=raw_hub, backbone_sha256=IMAGENET_SHA256)


def stage_smoke_loader_seed(ctx):
    r = _child([REPO / "scripts" / "smoke_loader_seed.py"], timeout=900)
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("RESULT:")), "")
    if r.returncode != 0 or not line.startswith("RESULT: PASS"):
        raise GateError("smoke_loader_seed_failed", f"rc={r.returncode} {line!r}")
    return _ok(line)


def stage_smoke_dataloader(ctx):
    return _pf1("smoke_dataloader")


def stage_seed_sequence(ctx):
    return _pf1("seed_sequence_R5")


def dry_run_argv(ctx, tmp: Path) -> list:
    return [REPO / "src" / "training" / "train_e1.py", "--dry-run", "--seed", str(ctx["seed"]),
            "--ckpt-dir", str(tmp), *PROFILES[ctx["profile"]].get("dry_args", ())]


def stage_dry_run(ctx):
    tmp = Path(tempfile.mkdtemp(prefix="e1tv_dryrun_")).resolve()
    try:
        if _under(tmp, REPO) or _under(tmp, ctx["root"]):
            raise GateError("dry_run_tmp_misplaced", f"{tmp} is inside the repository or the data root")
        r = _child(dry_run_argv(ctx, tmp), timeout=3600)
        line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("RESULT:")), "")
        m = re.search(r"\((\d+)/(\d+) checks exercised, (\d+) skipped\)", line)
        ck = re.search(r"^\[ckpt\] dir=(.+?) \(verified OUTSIDE repo\)$", r.stdout, re.M)
        ok = (r.returncode == 0 and line.startswith("RESULT: PASS") and m is not None
              and int(m.group(2)) == PF1.HARD_CHECKS_TOTAL and int(m.group(1)) == int(m.group(2))
              and int(m.group(3)) == 0 and ck is not None and Path(ck.group(1)).resolve() == tmp)
        if not ok:
            raise GateError("dry_run_failed",
                            f"rc={r.returncode} {line!r} ckpt_dir={ck.group(1) if ck else None} "
                            f"(expected {tmp}) {r.stderr[-300:]}")
        return _ok(f"{line} into {tmp} (removed)", dry_run=line)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stage_repo_unchanged(ctx):
    snap = _git(status_argv(UNCHANGED_PATHSPECS, ignored=True))
    if ctx.get("snapshot") is None or snap.returncode != 0:
        raise GateError("repo_state_unreadable", "the scoped repository snapshot could not be read")
    if snap.stdout != ctx["snapshot"]:
        raise GateError("repo_changed",
                        f"the scoped status changed during the gate: {parse_porcelain_paths(snap.stdout)[:10]}")
    foreign = module_provenance()
    if foreign:
        raise GateError("import_origin_foreign", f"modules loaded from outside {REPO}: {foreign[:5]}")
    return _ok("scoped status (governed + reports, incl. ignored) unchanged; module provenance re-checked")


STAGES = [
    ("arguments", stage_arguments),
    ("data_isolation", stage_data_isolation),
    ("repo_state", stage_repo_state),
    ("module_provenance", stage_module_provenance),
    ("class_weights", stage_class_weights),
    ("smoke_loss", stage_smoke_loss),
    ("image", stage_image),
    ("cuda", stage_cuda),
    ("imagenet_backbone", stage_imagenet_backbone),
    ("smoke_loader_seed", stage_smoke_loader_seed),
    ("smoke_dataloader", stage_smoke_dataloader),
    ("seed_sequence_R5", stage_seed_sequence),
    ("dry_run", stage_dry_run),
    ("repo_unchanged", stage_repo_unchanged),
]


# ---------------------------------------------------------------------------------------- verdict
def build_launch_block(ctx) -> str:
    q = shlex.quote
    d, root = q(str(ctx["ckpt_dir"])), q(str(ctx["root"]))
    extra = " ".join(q(a) for a in PROFILES[ctx["profile"]]["extra_args"])
    env = [f"PYTHONPATH={q(str(REPO))}", f"PLANTSEG_DATA_ROOT={root}",
           f"PLANTSEG_IMAGE_DIGEST={q(E1_IMAGE_DIGEST)}"]
    if ctx.get("hub_dir"):
        env.append(f"TORCH_HOME={q(str(Path(ctx['hub_dir']).parent))}")
    cmd = (f"env -u PLANTSEG_GIT_COMMIT {' '.join(env)} nohup {q(sys.executable)} -B "
           f"src/training/train_e1.py --real-run --confirm-real-run --init imagenet --ckpt-dir {d} "
           f"--seed {ctx['seed']} --num-workers {E1_NUM_WORKERS} --log-every {LOG_EVERY}"
           + (f" {extra}" if extra else "") + f" > {d}/e1_stdout.log 2>&1 &")
    return "\n".join([
        "set -eu",
        f"cd {q(str(REPO))}",
        f"[ -z \"$(ls -A {d} 2>/dev/null)\" ] || {{ printf 'STOP: %s is not empty\\n' {d}; exit 1; }}",
        f"mkdir -p {d}",
        cmd,
        f"echo $! > {d}/e1.pid",
    ])


def _write_record(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_gate(args) -> int:
    if not args.rehearsal and args.record is None:
        print("usage error: a real gate run requires --record PATH (or pass --rehearsal)", file=sys.stderr)
        return 2
    ctx = {"seed": args.seed, "ckpt_dir": args.ckpt_dir, "expect_head": args.expect_head,
           "record": args.record, "rehearsal": args.rehearsal, "profile": args.profile}
    print("=" * 78)
    print("E1 TRAIN/VAL-ONLY GATE (DL-21) -- verify_env.py and preflight_e1.py are not run")
    print(f"repo={REPO}")
    print(f"python={sys.executable} {platform.python_version()} | {platform.platform()}")
    print(f"PLANTSEG_DATA_ROOT={os.environ.get('PLANTSEG_DATA_ROOT')!r} | profile={args.profile} | "
          f"seed={args.seed} | PYTHONDONTWRITEBYTECODE at entry={BYTECODE_ENV_AT_ENTRY!r}")
    print(f"cpu_count={os.cpu_count()} RUNPOD_CPU_COUNT={os.environ.get('RUNPOD_CPU_COUNT')!r}")
    if args.rehearsal:
        print("MODE: --rehearsal -- exercises the gate off-pod; NEVER a launch authorization.")
    print("=" * 78, flush=True)

    results, first = [], None
    for name, fn in STAGES:
        t0 = time.time()
        try:
            status, detail, rec = fn(ctx)
            code = None
        except GateError as e:
            status, detail, rec, code = "FAIL", str(e), {}, e.code
        except Exception as e:                            # noqa: BLE001 -- a crash is a FAIL, never a GO
            status, detail, rec, code = "FAIL", f"[stage_error] {type(e).__name__}: {e}", {}, "stage_error"
        results.append({"stage": name, "status": status, "detail": detail, "code": code,
                        "secs": round(time.time() - t0, 1), "record": rec})
        print(f"\n>>> {name}: {status}  {detail}", flush=True)
        if status == "FAIL":
            first = (name, code)
            break

    print("\n" + "=" * 78)
    print("GATE SUMMARY")
    print("=" * 78)
    print(f"  {'stage':<26}{'result':<10}{'secs':>7}  detail")
    for r in results:
        print(f"  {r['stage']:<26}{r['status']:<10}{r['secs']:>7.1f}  {r['detail'][:160]}")
    for name, _fn in STAGES[len(results):]:
        print(f"  {name:<26}{'SKIPPED':<10}{'-':>7}  not reached")

    go = (not args.rehearsal and len(results) == len(STAGES)
          and all(r["status"] == "PASS" for r in results))
    # The record path is only trusted once `arguments` has vetted it (outside repo/ckpt/root, absent).
    record_ok = args.record is not None and results and results[0]["status"] == "PASS"
    print()
    if go:
        block = build_launch_block(ctx)
        try:                                   # the evidence record is written BEFORE GO is printed
            _write_record(Path(args.record), {"verdict": "GO", "seed": args.seed, "profile": args.profile,
                                              "repo": str(REPO), "stages": results, "launch_block": block})
        except OSError as e:
            print(f"VERDICT: NO-GO -- the evidence record could not be written ({type(e).__name__}: {e})")
            return 1
        print(f"record written: {args.record}")
        print("VERDICT: GO")
        print("Save the block below as <evidence>/launch_<seed>_<n>.sh and run it with bash from THIS shell:")
        print("----- launch block -----")
        print(block)
        print("----- end launch block -----")
        return 0
    if args.rehearsal:
        if first is None:
            print("VERDICT: NO-GO (rehearsal) -- not a launch authorization")
        else:
            print(f"VERDICT: NO-GO (rehearsal) -- first failing stage: {first[0]} [{first[1]}]")
        print("(launch block withheld)")
        if record_ok:
            _write_record(Path(args.record), {"verdict": "NO-GO (rehearsal)", "seed": args.seed,
                                              "profile": args.profile, "repo": str(REPO), "stages": results})
        return 1
    if first is None:
        print("VERDICT: NO-GO -- a stage did not PASS (SKIPPED is never a GO)")
    else:
        print(f"VERDICT: NO-GO -- first failing stage: {first[0]} [{first[1]}]")
    if record_ok:
        _write_record(Path(args.record), {"verdict": "NO-GO", "seed": args.seed, "profile": args.profile,
                                          "repo": str(REPO), "stages": results})
    return 1


def check_run_meta(args) -> int:
    path = Path(args.ckpt_dir) / "e1_telemetry.jsonl"
    if not path.is_file():
        print(f"MISMATCH: telemetry file not found: {path}")
        print("RESULT: FAIL -- run_meta vs the seed-42 recipe (no telemetry)")
        return 1
    rows, problems, warns = [], [], []
    with open(path, encoding="utf-8") as f:
        for n, ln in enumerate(f, 1):
            if '"run_meta"' not in ln:
                continue                       # train/val rows may still be appending; never parsed
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError as e:
                problems.append(f"telemetry line {n} mentions run_meta but is not valid JSON ({e})")
                continue
            if obj.get("event") == "run_meta":
                rows.append(obj)
    if len(rows) != 1:
        problems.append(f"run_meta rows: {len(rows)} (exactly 1 required; >1 means an appended relaunch)")
    if rows:
        row = rows[0]
        keys = tuple(row)
        if set(keys) != set(RUN_META_KEYS) or len(keys) != len(RUN_META_KEYS):
            problems.append(f"run_meta keys: missing {sorted(set(RUN_META_KEYS) - set(keys))}, "
                            f"extra {sorted(set(keys) - set(RUN_META_KEYS))}")
        for k, v in RUN_META_EXPECT.items():
            if k in row and (row[k] != v or type(row[k]) is not type(v)):
                problems.append(f"{k}: {row[k]!r} != {v!r}")
        prof = PROFILES[args.profile]
        for k, v in (("seed", args.seed), ("git_head", args.expect_head),
                     ("max_iters", prof["max_iters"]), ("poly_horizon", prof["poly_horizon"])):
            if row.get(k) != v:
                problems.append(f"{k}: {row.get(k)!r} != {v!r}")
        if row.get("gpu_name") != EXPECTED_GPU_NAME:
            warns.append(f"gpu_name {row.get('gpu_name')!r} != {EXPECTED_GPU_NAME!r} (recorded, not gated)")
    for w in warns:
        print(f"WARN: {w}")
    for p in problems:
        print(f"MISMATCH: {p}")
    ok = not problems
    print(f"RESULT: {'PASS' if ok else 'FAIL'} -- run_meta vs the seed-42 recipe "
          f"(exempt: {', '.join(RUN_META_EXEMPT)}; profile {args.profile})")
    return 0 if ok else 1


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(2)


def build_parser():
    p = _Parser(description="TRAIN/VAL-only E1 gate (DL-21).")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    g = sub.add_parser("gate")
    g.add_argument("--seed", type=int, required=True, choices=OFFICIAL_SEEDS)
    g.add_argument("--ckpt-dir", required=True)
    g.add_argument("--expect-head", required=True)
    g.add_argument("--profile", default="e1_80k", choices=sorted(PROFILES))
    g.add_argument("--record", default=None,
                   help="evidence JSON (absolute, outside the repo, the ckpt-dir and the data root); "
                        "required for a GO")
    g.add_argument("--rehearsal", action="store_true",
                   help="exercise the gate off-pod; never a GO, never waives a FAIL")
    c = sub.add_parser("check-run-meta")
    c.add_argument("--seed", type=int, required=True, choices=OFFICIAL_SEEDS)
    c.add_argument("--ckpt-dir", required=True)
    c.add_argument("--expect-head", required=True)
    c.add_argument("--profile", default="e1_80k", choices=sorted(PROFILES))
    sub.add_parser("_cuda-probe")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "_cuda-probe":
        return cuda_probe()
    if args.cmd == "check-run-meta":
        return check_run_meta(args)
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
