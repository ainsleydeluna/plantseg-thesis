#!/usr/bin/env python3
"""Smoke for the teacher arms (lane L-TEACHER-ARMS, commit C4): the R1/R2 configurations, the R2 dataset wrapper and
paste transform, the teacher-arm launcher, the R2 table builder, the arm teacher record and the go rule (ARMS plan
§4 item 1, the DL-24 audit's F32 cases and the Phase B-2 GO's fixed list: 70 + 17 cases and mutations + 11
mutations = 98 checks).

CPU only, with mmengine 0.10.7, mmsegmentation 1.2.2 and mmcv 2.1.0; without them it prints "RESULT: NOT RUN
(teacher stack missing ...)" and exits 2, never a PASS. Everything is synthetic and lives in a scratch work dir
outside the repository:
  * a 5,367/846 TRAIN/VAL root of 32x24 images: classes 1..19 with c images each (rare, n_c = 1..19), 82 classes
    with 54 and 14 with 53, seven images without a disease pixel; one class-3 image carries an EXIF rotation;
  * its strata file, and the R2 table the real builder writes from them (the strata file's own 11-check validator
    reads the repository's class-weight file, so this smoke replaces build_r2_table.strata_check with one that
    accepts the synthetic strata file);
  * a scratch git repository: P_T (HEAD) and records commits holding the decision-log rows, the strata file and
    the schedule; a second one whose HEAD holds the go rule's rows;
  * tiny EMA models and checkpoints, synthetic mmengine logs and DIAG D2 outputs.
The launcher's environment gates that are pod facts (the image's inherited cuBLAS value, the governed paths of this
checkout) are replaced in-process, and the init checkpoint's sha256 is accepted only inside the frozen
check_init_checkpoint call (smoke_teacher_launch's pattern); those gates are smoke_teacher_launch's. No folder or
file this smoke creates has "test" in its name: the "test" refusals are shown on never-created path strings,
refused before any filesystem call. Mutations are monkeypatches, each restored, and each must make its named
check FAIL.

    python -B scripts/smoke_teacher_arms.py [--work-dir DIR]
Exit 0 only when every check passes ("RESULT: PASS (98/98)"); 1 otherwise; 2 when it cannot run.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SMOKE = sys.modules[__name__]
STACK = {"mmengine": "0.10.7", "mmseg": "1.2.2", "mmcv": "2.1.0"}


def teacher_stack_problem() -> str | None:
    import importlib
    bad = []
    for mod, want in STACK.items():
        try:
            got = importlib.import_module(mod).__version__
        except Exception as e:  # noqa: BLE001
            bad.append(f"{mod}: {type(e).__name__}")
            continue
        if got != want:
            bad.append(f"{mod} {got} != {want}")
    return "; ".join(bad) or None


# ------------------------------------------------------------------------------------------- the registry
CASES, ORDER, MUTATIONS = {}, [], []


def case(fn):
    CASES[fn.__name__] = fn
    ORDER.append(fn.__name__)
    return fn


@contextlib.contextmanager
def patched(*triples):
    """(target, attribute, value) monkeypatches, restored on exit."""
    saved = [(t, a, getattr(t, a)) for t, a, _ in triples]
    try:
        for t, a, v in triples:
            setattr(t, a, v)
        yield
    finally:
        for t, a, v in reversed(saved):
            setattr(t, a, v)


def mutation(name, check, make):
    MUTATIONS.append((name, check, make))


L = SimpleNamespace()          # the libraries, imported once the teacher stack is known to load
W = None                       # the work dir
FX = {}                        # cached fixtures


def libs():
    import numpy as np
    import torch
    from PIL import Image
    import scripts.arm_go_rule as agr
    import scripts.arm_teacher_record as atr
    import scripts.build_r2_table as brt
    import scripts.launch_teacher_arm as lta
    import scripts.launch_teacher_finetune as ltf
    import scripts.preflight_arms as pa
    from src.distill import segnext_teacher as stc
    from src.training import teacher_arm_components as tac
    L.__dict__.update(np=np, torch=torch, Image=Image, agr=agr, atr=atr, brt=brt, lta=lta, ltf=ltf, pa=pa, stc=stc,
                      tac=tac)


# ------------------------------------------------------------------------------------------- fixtures
PLAN_N = {**{c: c for c in range(1, 20)}, **{c: 54 for c in range(20, 102)}, **{c: 53 for c in range(102, 116)}}
NO_DISEASE = 7
N_TRAIN, N_VAL = 5367, 846
STEMS = [f"p{i:05d}" for i in range(N_TRAIN)]
ROTATED = "p00003"
STRATA_REL = "reports/strata/train_strata_v1.json"
ENV_VARS = ("PLANTSEG_DATA_ROOT", "SEGNEXT_ADE20K_CKPT", "TEACHER_WORK_DIR", "PLANTSEG_R2_TABLE")
ADE_NAME = "segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth"


def class_plan() -> dict:
    assign = []
    for c in range(1, 116):
        assign += [c] * PLAN_N[c]
    assign += [None] * NO_DISEASE
    assert len(assign) == N_TRAIN
    return dict(zip(STEMS, assign))


def canon(doc) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else Path(b).read_bytes()).hexdigest()


def write_pair(img_dir: Path, ann_dir: Path, stem: str, img, mask, exif_orientation=None):
    ex = None
    if exif_orientation is not None:
        ex = L.Image.Exif()
        ex[0x0112] = exif_orientation
    kw = {"quality": 92}
    if ex is not None:
        kw["exif"] = ex
    L.Image.fromarray(img).save(img_dir / f"{stem}.jpg", **kw)
    L.Image.fromarray(mask).save(ann_dir / f"{stem}.png")


def make_root(root: Path, plan: dict, rotated=()):
    np = L.np
    for d in ("images/train", "annotations/train", "images/val", "annotations/val"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(sorted(plan)):
        img = np.random.RandomState(i).randint(0, 256, (24, 32, 3)).astype(np.uint8)
        m = np.zeros((24, 32), dtype=np.uint8)
        c = plan[stem]
        if isinstance(c, int):
            y, x = 2 + i % 5, 3 + (i * 7) % 11
            m[y:y + 9, x:x + 13] = c
        elif isinstance(c, tuple):                        # a two-label mask (the builder's STOP)
            m[2:8, 2:10], m[12:20, 14:26] = c
        write_pair(root / "images/train", root / "annotations/train", stem, img, m,
                   6 if stem in rotated else None)
    for i in range(N_VAL if len(plan) == N_TRAIN else 0):
        (root / "images/val" / f"v{i:04d}.jpg").write_bytes(b"")
        (root / "annotations/val" / f"v{i:04d}.png").write_bytes(b"")


def strata_doc(plan: dict) -> dict:
    """The fields the builder and the launcher read: n_images, split_list_sha256, per_class images."""
    counts = {c: 0 for c in range(1, 116)}
    for c in plan.values():
        for k in (c if isinstance(c, tuple) else (c,)):
            if k is not None:
                counts[k] += 1
    stems = sorted(plan)
    return {"schema": "smoke-synthetic-strata", "n_images": len(stems),
            "split_list_sha256": hashlib.sha256("".join(f"{s}\n" for s in stems).encode()).hexdigest(),
            "per_class": [{"class_id": c, "images": n} for c, n in counts.items()]}


def synthetic_strata_check(path):
    return True, "synthetic strata file (smoke): build_train_strata.check_document reads the real class weights"


def build_table(root, strata_path, out, *, extra=()):
    with patched((L.brt, "strata_check", synthetic_strata_check)):
        return run_main(L.brt.main, ["--data-root", str(root), "--strata", str(strata_path), "--out-dir", str(out),
                                     *extra])


def run_main(fn, argv):
    """(rc, last stderr line, stdout) of a CLI main, with os.environ restored."""
    out, err = io.StringIO(), io.StringIO()
    saved = dict(os.environ)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fn(argv)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    e = err.getvalue().strip()
    return rc, (e.splitlines()[-1] if e else ""), out.getvalue()


def git(repo: Path, *args) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    r = subprocess.run(["git", "-c", "user.name=smoke", "-c", "user.email=smoke@example.invalid", "-c",
                        "commit.gpgsign=false", "-C", str(repo), *args], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr}")
    return r.stdout.strip()


def commit_files(repo: Path, files: dict, msg: str) -> str:
    for rel, data in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        git(repo, "add", "--", rel)
    git(repo, "commit", "-q", "-m", msg)
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", f"fx_{head[:12]}", head)
    return head


def dl_text(rows) -> str:
    return "\n".join(["# Decision log (smoke)", "", "| ID | Status | Decision |", "|---|---|---|"]
                     + [f"| {i} | {s} | {t} |" for i, s, t in rows]) + "\n"


def fx(name):
    if name not in FX:
        FX[name] = globals()[f"_fx_{name}"]()
    return FX[name]


def _fx_root():
    root = W / "root"
    plan = class_plan()
    make_root(root, plan, rotated=(ROTATED,))
    strata = W / "strata" / "train_strata_v1.json"
    strata.parent.mkdir(parents=True)
    strata.write_bytes(canon(strata_doc(plan)))
    rc, _, out = build_table(root, strata, W / "r2_table")
    assert rc == 0, out
    table = W / "r2_table" / "r2_table_v1.json"
    return SimpleNamespace(root=root, plan=plan, strata=strata, strata_b=strata.read_bytes(), table=table,
                           table_b=table.read_bytes(), table_sha=sha(table), lookups=L.tac.validate_table(
                               json.loads(table.read_text(encoding="utf-8"))))


def _fx_init():
    p = W / "init" / ADE_NAME
    p.parent.mkdir(parents=True)
    p.write_bytes(b"synthetic ADE20K init checkpoint (smoke)\n")
    return SimpleNamespace(path=p, sha=sha(p))


def _fx_scalars():
    """An mmengine-like scalars.json: training rows every 50 iterations (time 0.50..0.54), VAL rows."""
    p = W / "reference" / "scalars.json"
    p.parent.mkdir(parents=True)
    rows = [{"lr": 1e-4, "data_time": 0.01, "loss": 1.0, "time": 0.50 + (k % 5) / 100, "iter": 50 * (k + 1),
             "memory": 1, "step": 50 * (k + 1)} for k in range(80)]
    rows.append({"mIoU_full": 0.3, "mIoU": 30.0, "data_time": 0.0, "time": 9.0, "step": 4000})
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return SimpleNamespace(path=p, sha=sha(p), median=0.52)


def _fx_r1record():
    """The R1 record R2's launch reads (complete), and a diverged and an incomplete one."""
    out = {}
    for status in ("complete", "diverged", "incomplete"):
        p = W / "records" / f"arm_teacher_record_R1_{status}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(canon({"format": "arm_teacher_record/1", "arm": "R1", "status": status,
                             "work_dir": {"name": "teacher-r1_s42_40k_a1", "attempt": 1},
                             "logs": {"scalars_sha256": fx("scalars").sha}}))
        out[status] = SimpleNamespace(path=p, sha=sha(p))
    return SimpleNamespace(**out)


def base_rows(p_t, **override):
    r = fx("root")
    lta = L.lta
    rows = {"DL-900": ("RECORDED", "schedule T (smoke)"),
            "DL-901": ("RECORDED (smoke)", "AM-18 written approval, as reported (smoke)"),
            "DL-902": ("RECORDED", "AM-20 item 3: open (smoke token row)"),
            "DL-903": ("RECORDED", f"R1 configuration: sha256 {sha(REPO / lta.ARM_CONFIGS['R1'])}; registered "
                                   f"difference sha256 {lta.reg_sha256('R1')}; P_T = {p_t}"),
            "DL-904": ("RECORDED", f"R2 configuration and table: sha256 {sha(REPO / lta.ARM_CONFIGS['R2'])}; "
                                   f"registered difference sha256 {lta.reg_sha256('R2')}; P_T = {p_t}; "
                                   f"r2_table_v1.json sha256 {r.table_sha}"),
            "DL-905": ("RECORDED", f"R1 record sha256 {fx('r1record').complete.sha}")}
    rows.update(override)
    return [(k, *v) for k, v in rows.items() if v is not None]


def _fx_repo():
    repo = W / "gitroot"
    repo.mkdir()
    git(repo, "init", "-q")
    p_t = commit_files(repo, {"README.md": "P_T (smoke)\n"}, "P_T")
    sched = json.dumps({"format": "kd_schedule/1", "in_force": "T", "decision_log_entry": "DL-900"})
    fxr = SimpleNamespace(repo=repo, p_t=p_t, sched=sched)
    FX["repo"] = fxr
    fxr.h = records_commit(base_rows(p_t))
    fxr.h2 = records_commit(base_rows(p_t), {"reports/arms/am8a_r1_a1.md": "AM-8a report of a1 (smoke)\n"})
    git(repo, "checkout", "-q", "--orphan", "side")
    fxr.side = commit_files(repo, {"docs/DECISION_LOG.md": dl_text(base_rows(p_t)),
                                   STRATA_REL: fx("root").strata_b, "configs/kd_schedule.json": sched}, "side")
    git(repo, "checkout", "-q", "--detach", p_t)
    return fxr


def records_commit(rows, extra=None) -> str:
    fxr = FX["repo"]
    git(fxr.repo, "checkout", "-q", "--detach", fxr.p_t)
    files = {"docs/DECISION_LOG.md": dl_text(rows), STRATA_REL: fx("root").strata_b,
             "configs/kd_schedule.json": fxr.sched, **(extra or {})}
    h = commit_files(fxr.repo, files, "records")
    git(fxr.repo, "checkout", "-q", "--detach", fxr.p_t)
    return h


@contextlib.contextmanager
def launcher_ctx():
    """The launcher against the scratch repository, with the pod-fact gates replaced (see the docstring)."""
    lta, ltf = L.lta, L.ltf
    orig_cic = ltf.check_init_checkpoint
    init_sha = fx("init").sha

    def cic(value):
        saved = ltf.EXPECTED_ADE20K_SHA256
        ltf.EXPECTED_ADE20K_SHA256 = init_sha
        try:
            return orig_cic(value)
        finally:
            ltf.EXPECTED_ADE20K_SHA256 = saved

    env = dict(os.environ)
    for k in ENV_VARS:
        os.environ.pop(k, None)
    try:
        with patched((lta, "GIT_ROOT", fx("repo").repo), (ltf, "check_init_checkpoint", cic),
                     (ltf, "check_inherited_cublas", lambda: None),
                     (ltf, "check_governed_clean", lambda porcelain=None: {"governed_paths_clean": True,
                                                                           "smoke_stub": True})):
            yield
    finally:
        os.environ.clear()
        os.environ.update(env)


_RUNS = [0]


def launch_argv(arm, *, work=None, records=None, extra=(), **flags):
    r, fxr = fx("root"), fx("repo")
    _RUNS[0] += 1
    work = work or (W / "runs" / f"n{_RUNS[0]}" / f"teacher-{arm.lower()}_s42_40k_a1")
    a = {"--arm": arm, "--config": str(REPO / L.lta.ARM_CONFIGS[arm]),
         "--config-sha256": sha(REPO / L.lta.ARM_CONFIGS[arm]), "--config-dl": "DL-903" if arm == "R1" else "DL-904",
         "--data-root": str(r.root), "--init-ckpt": str(fx("init").path), "--work-dir": str(work),
         "--code-pin": fxr.p_t, "--records-commit": records or fxr.h, "--approval-dl": "DL-901",
         "--recheck-dl": "DL-902", "--r2-config-dl": "DL-904", "--r2-table": str(r.table),
         "--r2-table-sha256": r.table_sha, "--r2-table-dl": "DL-904"}
    if arm == "R2":
        a.update({"--r1-record": str(fx("r1record").complete.path), "--r1-record-sha256": fx("r1record").complete.sha,
                  "--r1-record-dl": "DL-905"})
    for k, v in flags.items():
        key = "--" + k.replace("_", "-")
        if v is None:
            a.pop(key, None)
        else:
            a[key] = v
    argv = ["--real-run", "--confirm-real-run", "--create-work-dir", "--skip-cuda-probe"]
    for k, v in a.items():
        argv += [k, str(v)]
    return argv + list(extra)


def launch(arm, **kw):
    with launcher_ctx():
        return run_main(L.lta.main, launch_argv(arm, **kw))


def provenance_of(out: str) -> dict:
    return json.loads(out[out.index("\n{") + 1:out.rindex("\n}") + 2])


@contextlib.contextmanager
def cfg_env(table=None):
    r = fx("root")
    saved = dict(os.environ)
    os.environ.update(PLANTSEG_DATA_ROOT=str(r.root), SEGNEXT_ADE20K_CKPT=str(fx("init").path),
                      TEACHER_WORK_DIR=str(W / "runs" / "cfg" / "teacher-r1_s42_40k_a1"),
                      PLANTSEG_R2_TABLE=str(table or r.table))
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def load_cfg(name, table=None):
    """The record ("record"), R1 or R2 configuration, loaded as the launcher loads it (a mutation target)."""
    from mmengine.config import Config
    path = L.lta.RECORD_CONFIG if name == "record" else REPO / L.lta.ARM_CONFIGS[name]
    with cfg_env(table):
        return Config.fromfile(str(path))


def arm_diff(arm, table=None) -> list:
    """diff_cfg(record, arm) as the launcher computes it: with the variables still set, so host-dependent values
    appear by their names."""
    rec, cfg = load_cfg("record", table), load_cfg(arm, table)
    with cfg_env(table):
        return L.lta.diff_cfg(rec, cfg)


def r1_ema_dict() -> dict:
    """R1's fourth hook, as configured (a mutation target)."""
    return dict(load_cfg("R1").custom_hooks[3])


def build_r2_dataset():
    from mmengine.registry import DATASETS, DefaultScope
    cfg = load_cfg("R2")
    with DefaultScope.overwrite_default_scope("mmseg"):
        return DATASETS.build(copy.deepcopy(cfg.train_dataloader.dataset))


def info_of(ds_inner, stem):
    for i in range(len(ds_inner)):
        d = ds_inner.get_data_info(i)
        if Path(d["img_path"]).stem == stem:
            return d
    raise KeyError(stem)


def r2_inner():
    if "inner" not in FX:
        FX["inner"] = build_r2_dataset().dataset
    return FX["inner"]


def record_reference(info, seed):
    """The transform's output with the paste draws first, then train_preprocess on the same stream."""
    np, tac = L.np, L.tac
    from configs.augment import AUGMENT
    from src.data.transforms import train_preprocess
    from src.training.teacher_components import _pack_canvas
    np.random.seed(seed)
    rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
    with L.Image.open(info["img_path"]) as im, L.Image.open(info["seg_map_path"]) as mk:
        if info["rare_class"] is None:
            img, mask = train_preprocess(im, mk, rng, AUGMENT)
        else:
            ip, mp, _ = tac.copy_paste(im, mk, int(info["rare_class"]), list(info["paste_sources"]),
                                       str(Path(info["img_path"]).parent), str(Path(info["seg_map_path"]).parent), rng)
            img, mask = train_preprocess(ip, mp, rng, AUGMENT)
    return _pack_canvas(dict(info), img, mask)


def apply_transform(info, seed, transform=None):
    L.np.random.seed(seed)
    t = transform or L.tac.RareClassCopyPasteTrainTransform()
    return t(copy.deepcopy(info))


def same_output(a, b) -> bool:
    return L.np.array_equal(a["img"], b["img"]) and L.np.array_equal(a["gt_seg_map"], b["gt_seg_map"])


class Script:
    """A scripted RandomState for the paste: rand(), uniform() and randint() return what the fixture names."""

    def __init__(self, rand=0.1, u=1.0, ints=()):
        self._rand, self._u, self.ints, self.calls = rand, u, list(ints), []

    def rand(self):
        self.calls.append("rand")
        return self._rand

    def uniform(self, a, b):
        self.calls.append("uniform")
        return self._u

    def randint(self, lo, hi=None):
        lo, hi = (0, lo) if hi is None else (lo, hi)
        v = self.ints.pop(0)
        assert lo <= v < hi, f"scripted {v} outside [{lo}, {hi})"
        self.calls.append(("randint", lo, hi))
        return v


def _fx_paste():
    """Source files for the paste unit checks (class 9): s_rect, a 4x5 block in a 40x40 S; s_diag, two pixels at
    (10,10) and (12,12) of a 64x64 S (its resized 1x1 patch holds no c pixel); t_rot, a 32x24 T with EXIF 6."""
    np = L.np
    d = W / "paste"
    img_dir, ann_dir = d / "images" / "train", d / "annotations" / "train"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    m = np.zeros((40, 40), dtype=np.uint8)
    m[10:14, 20:25] = 9
    write_pair(img_dir, ann_dir, "s_rect", np.random.RandomState(1).randint(0, 256, (40, 40, 3)).astype(np.uint8), m)
    m = np.zeros((64, 64), dtype=np.uint8)
    m[10, 10] = m[12, 12] = 9
    write_pair(img_dir, ann_dir, "s_diag", np.random.RandomState(2).randint(0, 256, (64, 64, 3)).astype(np.uint8), m)
    m = np.zeros((24, 32), dtype=np.uint8)
    m[0:6, 0:8] = 9
    write_pair(img_dir, ann_dir, "t_rot", np.random.RandomState(3).randint(0, 256, (24, 32, 3)).astype(np.uint8), m,
               exif_orientation=6)
    return SimpleNamespace(img_dir=img_dir, ann_dir=ann_dir)


def t_pair(mask, seed=4):
    np = L.np
    h, w = mask.shape
    return (L.Image.fromarray(np.random.RandomState(seed).randint(0, 256, (h, w, 3)).astype(np.uint8)),
            L.Image.fromarray(mask.astype(np.uint8)))


def paste(im, mk, rng, pool=("s_rect",), c=9):
    p = fx("paste")
    return L.tac.copy_paste(im, mk, c, list(pool), str(p.img_dir), str(p.ann_dir), rng)


# ------------------------------------------------------------------ the run fixtures (arm_teacher_record)
def make_run(name, *, last_iter=40000, vals=10, nan_iter=None, val_nan_at=None, final=True, selection=True,
             ckpts=True, arm="R1", no_log=False) -> Path:
    torch = L.torch
    work = W / "runs_rec" / name / f"teacher-{arm.lower()}_s42_40k_a1"
    (work / "20261011_000000" / "vis_data").mkdir(parents=True)
    cfg_sha = sha(REPO / L.lta.ARM_CONFIGS[arm])
    (work / "teacher_arm_launch_provenance.json").write_bytes(canon({
        "format": "teacher_arm_launch_provenance/1", "arm": arm, "work_dir": str(work),
        "config": {"sha256": cfg_sha}, "rows": {"config": {"dl": "DL-903"}, "r2_table": {"dl": "DL-904"}},
        "registered_difference": {"sha256": L.lta.reg_sha256(arm)}, "modules": {"launcher": {"sha256": "a" * 64}},
        "checkout_git_head": "b" * 40, "records_commit": "c" * 40, "r2_table": {"sha256": "d" * 64}}))
    (work / "teacher_launch_provenance.json").write_bytes(canon({"image_digest_declared": "sha256:" + "e" * 64}))
    if no_log:                                       # stopped before the first log row (iteration 50)
        return work
    rows = []
    for it in range(50, last_iter + 1, 50):
        loss = float("nan") if nan_iter is not None and it >= nan_iter else 1.0 / it
        rows.append({"lr": 1e-4, "data_time": 0.01, "loss": loss, "decode.loss_ce": loss, "time": 0.5, "iter": it,
                     "step": it})
        if it % 4000 == 0 and it // 4000 <= vals:
            score = float("nan") if val_nan_at == it else 0.3 + it / 1e6
            rows.append({"mIoU_full": score, "mIoU_disease_full": score, "mIoU": 30.0, "val_images": 846, "time": 1.0,
                         "step": it})
    (work / "20261011_000000" / "vis_data" / "scalars.json").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    recs = []
    for it in range(4000, min(last_iter, vals * 4000) + 1, 4000):
        ck = work / f"iter_{it}.pth"
        if ckpts:
            g = torch.Generator().manual_seed(it)
            ema = {"backbone.w": torch.randn(3, generator=g, dtype=torch.float64),
                   "decode_head.w": torch.randn(2, generator=g, dtype=torch.float64)}
            raw = {f"module.{k}": v + 1.0 for k, v in ema.items()}
            torch.save({"state_dict": ema, "ema_state_dict": {**raw, "steps": torch.tensor(it)}, "meta": {}}, ck)
        score = 0.3 + it / 1e6
        recs.append({"iteration": it, "mIoU_full": score, "mIoU_disease_full": score, "checkpoint_path": str(ck),
                     "checkpoint_sha256": sha(ck) if ck.is_file() else None})
    (work / "teacher_selection_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs),
                                                         encoding="utf-8")
    if selection and recs:
        best = max(recs, key=lambda r: r["mIoU_full"])
        (work / "teacher_selection.json").write_bytes(canon({
            "status": "final" if final else "running", "selected_iteration": best["iteration"],
            "selected_mIoU_full": best["mIoU_full"], "checkpoint_path": best["checkpoint_path"],
            "checkpoint_sha256": best["checkpoint_sha256"]}))
    return work


def record_argv(work, out, arm="R1"):
    return ["record", "--arm", arm, "--work-dir", str(work), "--config", str(REPO / L.lta.ARM_CONFIGS[arm]),
            "--config-sha256", sha(REPO / L.lta.ARM_CONFIGS[arm]), "--eval-config", str(L.lta.RECORD_CONFIG),
            "--out", str(out), "--generated-utc", "2026-10-11T00:00:00Z"]


def _fx_complete_record():
    work = make_run("complete")
    out = W / "records" / "arm_teacher_record_R1_complete_run.json"
    rc, _, text = run_main(L.atr.main, record_argv(work, out))
    assert rc == 0, text
    return SimpleNamespace(work=work, path=out, sha=sha(out), doc=json.loads(out.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ the go rule's D2 outputs
def d2_doc(role, *, purpose=None, arm_id=None, miou=0.39, ece=0.05, ckpt_sha=None, head="f" * 40, host="laptop",
           start="2026-10-12T02:00:00Z", end="2026-10-12T02:30:00Z", status="written", gate_passed=True,
           of_record_sha=None) -> dict:
    doc = {"lane": "L-TEACHER-DIAG", "script": "scripts/teacher_d2_calibration.py", "artifact_status": "smoke",
           "mode": "stub", "role": role, "arm_id": arm_id, "purpose": purpose, "code": {"head": head},
           "start_utc": start, "status": status,
           "teacher": {"role": role, "checkpoint": {"sha256_verified": ckpt_sha}},
           "pass": {"all_class_miou": miou}, "calibration": {"ece": ece},
           "environment": {"hostname": f"container-{role}-{purpose or arm_id}", "cpu_model": host,
                           "start_utc": start, "end_utc": end}}
    if role == "record":
        doc["gates"] = {"passed": gate_passed, "value": miou, "delta": abs(miou - 0.38576993346214294)}
    if purpose == "control":
        doc["inputs"] = {"of_record_output": {"sha256": of_record_sha}}
        doc["control"] = {"ece_b": ece, "all_class_miou_b": miou}
    return doc


def item1b_bytes() -> bytes:
    return canon(d2_doc("record", purpose="item1", miou=0.38576993346214294, ece=0.061, start="2026-10-01T00:00:00Z",
                        end="2026-10-01T01:00:00Z"))


def go_files(tag, *, val_x=0.39, ece_x=0.05, ece_b=0.06, control_miou=0.38576993346214294, control_status="written",
             gate_passed=True, arm_ckpt=None, host_arm="laptop", control_head="f" * 40):
    d = W / "go" / tag
    d.mkdir(parents=True)
    rec = fx("complete_record")
    item1b = d / "teacher_d2_20261001T000000Z.json"
    item1b.write_bytes(item1b_bytes())
    control = d / "teacher_d2_control_20261012T010000Z.json"
    control.write_bytes(canon(d2_doc("record", purpose="control", miou=control_miou, ece=ece_b, head=control_head,
                                     start="2026-10-12T01:00:00Z", end="2026-10-12T01:30:00Z", status=control_status,
                                     gate_passed=gate_passed, of_record_sha=sha(item1b))))
    arm = d / "teacher_d2_arm-R1_20261012T020000Z.json"
    arm.write_bytes(canon(d2_doc("arm", arm_id="R1", miou=val_x, ece=ece_x, host=host_arm,
                                 ckpt_sha=arm_ckpt or rec.doc["selection"]["checkpoint"]["sha256"])))
    return SimpleNamespace(dir=d, item1b=item1b, control=control, arm=arm, rec=rec)


def go_argv(g, out, extra=()):
    return ["--arm", "R1", "--teacher-record", str(g.rec.path), "--teacher-record-sha256", g.rec.sha,
            "--arm-output", str(g.arm), "--control-output", str(g.control), "--item1b-output", str(g.item1b),
            "--out", str(out), "--generated-utc", "2026-10-12T03:00:00Z", *extra]


def _fx_dlrepo():
    """A clone whose HEAD holds the go rule's rows: a fault of the item 1(b) host and a void session of the record."""
    repo = W / "dlroot"
    repo.mkdir()
    git(repo, "init", "-q")
    rec = fx("complete_record")
    commit_files(repo, {"docs/DECISION_LOG.md": dl_text([
        ("DL-950", "RECORDED", f"fault of the host of the item 1(b) output {sha(item1b_bytes())} (smoke)"),
        ("DL-951", "RECORDED", f"void session of the R1 teacher record {rec.sha} (smoke)")])}, "rows")
    return repo


# =========================================================================================== the checks
# ---- R1 configuration
@case
def cm_r1_diff_exact():
    got = arm_diff("R1")
    dpr = [e for e in got if e[0] == "model.backbone.drop_path_rate"]
    ok = (L.lta.canonical_bytes(got) == L.lta.canonical_bytes(L.lta.reg("R1"))
          and [e[0] for e in got] == ["custom_hooks", "model.backbone.drop_path_rate"]
          and dpr == [("model.backbone.drop_path_rate", 0.1, 0.2)])
    return ok, f"leaves {[e[0] for e in got]}; {dpr}"


@case
def cm_r1_ema_hook_dict():
    from mmengine.registry import HOOKS
    hooks = load_cfg("R1").custom_hooks
    ema = dict(hooks[3])
    want = {"type": "EMAHook", "ema_type": "ExponentialMovingAverage", "momentum": 0.0002, "update_buffers": True,
            "interval": 1, "begin_iter": 0, "strict_load": False}
    h = HOOKS.build(copy.deepcopy(ema))
    ok = (ema == want and [x["type"] for x in hooks[:3]] == ["TeacherInitCompatibilityHook", "TeacherNMFEvalStreamHook",
                                                            "TeacherSelectionRecordHook"]
          and type(h).__name__ == "EMAHook" and h.priority == "NORMAL" and h.strict_load is False and h.begin_iter == 0
          and h.ema_cfg == {"type": "ExponentialMovingAverage", "momentum": 0.0002, "update_buffers": True,
                            "interval": 1})
    return ok, f"{ema}; priority {h.priority}"


@case
def cm_r1_record_locks_hold():
    rec = L.ltf.check_locked_config(load_cfg("record"))
    r1 = L.ltf.check_locked_config(load_cfg("R1"))
    ok = rec == [] and len(r1) == 1 and r1[0].startswith("custom_hooks=")
    return ok, f"record {rec}; R1 (the frozen launcher refuses a fourth hook) {r1}"


def rows_gate(arm, records, **flags):
    args = L.lta.parse_args(launch_argv(arm, records=records, **flags))
    try:
        L.lta.gate_rows(args, L.lta.ss.HeadSource(fx("repo").repo), fx("repo").p_t)
        return None
    except L.lta.PreflightError as e:
        return e.code


@case
def cm_config_row_required():
    p_t = fx("repo").p_t
    no_row = records_commit(base_rows(p_t, **{"DL-903": None}))
    no_reg = records_commit(base_rows(p_t, **{"DL-903": ("RECORDED", f"R1 config sha256 "
                                                                       f"{sha(REPO / L.lta.ARM_CONFIGS['R1'])}; "
                                                                       f"P_T = {p_t}")}))
    no_pin = records_commit(base_rows(p_t, **{"DL-903": ("RECORDED", f"R1 config sha256 "
                                                                       f"{sha(REPO / L.lta.ARM_CONFIGS['R1'])}; "
                                                                       f"diff {L.lta.reg_sha256('R1')}")}))
    got = {k: rows_gate("R1", h) for k, h in (("no_row", no_row), ("no_reg", no_reg), ("no_pin", no_pin),
                                              ("ok", fx("repo").h))}
    return got == {"no_row": "config_row_invalid", "no_reg": "config_row_invalid", "no_pin": "config_row_invalid",
                   "ok": None}, got


# ---- R2 configuration
@case
def cm_r2_diff_exact():
    got = arm_diff("R2")
    ok = (L.lta.canonical_bytes(got) == L.lta.canonical_bytes(L.lta.reg("R2"))
          and [e[0] for e in got] == ["custom_hooks", "custom_imports.imports", "model.backbone.drop_path_rate",
                                      "train_dataloader.dataset"])
    ds = [e for e in got if e[0] == "train_dataloader.dataset"][0][2]
    ok = ok and ds["type"] == "ClassBalancedDataset" and ds["oversample_thr"] == 20 / 5367 \
        and ds["dataset"]["type"] == "RareClassTrainDataset"
    return ok, f"leaves {[e[0] for e in got]}"


@case
def cm_r2_val_test_unchanged():
    rec, r2 = load_cfg("record").to_dict(), load_cfg("R2").to_dict()
    keys = ("val_dataloader", "val_evaluator", "val_cfg", "test_dataloader", "test_evaluator", "test_cfg")
    same = {k: L.lta._form(rec.get(k)) == L.lta._form(r2.get(k)) for k in keys}
    tr = {k: L.lta._form(rec["train_dataloader"][k]) == L.lta._form(r2["train_dataloader"][k])
          for k in ("batch_size", "sampler", "num_workers", "persistent_workers")}
    ok = all(same.values()) and all(tr.values()) and all(r2[k] is None for k in keys[3:])
    return ok, f"VAL/TEST equal {same}; train loader keys equal {tr}"


# ---- EMA (mmengine 0.10.7's EMAHook as R1 configures it)
def tiny_model():
    torch = L.torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 2, 1)
            self.bn = torch.nn.BatchNorm2d(2)

    torch.manual_seed(0)
    return Tiny().double()


def ema_setup():
    from mmengine.hooks import EMAHook
    d = r1_ema_dict()
    d.pop("type")
    m = tiny_model()
    hook = EMAHook(**d)
    runner = SimpleNamespace(model=m, iter=0, epoch=0)
    hook.before_run(runner)
    return m, hook, runner


def set_all(m, value):
    torch = L.torch
    with torch.no_grad():
        for t in list(m.parameters()) + [m.bn.running_mean, m.bn.running_var]:
            t.fill_(value)


@case
def cm_ema_first_update_copies():
    torch = L.torch
    m, hook, runner = ema_setup()
    set_all(m, 3.0)
    hook.after_train_iter(runner, 0)
    ema = hook.ema_model.module.state_dict()
    ok = all(torch.equal(ema[k], v) for k, v in m.state_dict().items()) and int(hook.ema_model.steps) == 1
    return ok, f"steps {int(hook.ema_model.steps)}"


@case
def cm_ema_decay_closed_form():
    m, hook, runner = ema_setup()
    got = []
    for t in range(1, 6):
        set_all(m, 1.0 if t == 1 else 0.0)
        runner.iter = t - 1
        hook.after_train_iter(runner, t - 1)
        got.append(float(hook.ema_model.module.conv.weight.flatten()[0]))
    want = [0.9998 ** (t - 1) for t in range(1, 6)]
    err = max(abs(a - b) for a, b in zip(got, want))
    return err <= 1e-12, f"max |EMA - 0.9998^(t-1)| over t = 1..5 = {err:.3e}"


@case
def cm_ema_bn_buffers_averaged():
    m, hook, runner = ema_setup()
    set_all(m, 1.0)
    hook.after_train_iter(runner, 0)
    set_all(m, 5.0)
    runner.iter = 1
    hook.after_train_iter(runner, 1)
    got = float(hook.ema_model.module.bn.running_mean[0])
    want = 1.0 + 0.0002 * (5.0 - 1.0)
    return abs(got - want) <= 1e-12, f"EMA running_mean {got!r}, expected {want!r} (averaged, not copied)"


@case
def cm_ema_val_scores_ema_weights():
    torch = L.torch
    m, hook, runner = ema_setup()
    set_all(m, 1.0)
    hook.after_train_iter(runner, 0)
    set_all(m, 5.0)
    runner.iter = 1
    hook.after_train_iter(runner, 1)
    ema = {k: v.clone() for k, v in hook.ema_model.module.state_dict().items()}
    raw = {k: v.clone() for k, v in m.state_dict().items()}
    hook.before_val_epoch(runner)
    during = all(torch.equal(m.state_dict()[k], ema[k]) for k in raw if m.state_dict()[k].is_floating_point())
    hook.after_val_epoch(runner)
    after = all(torch.equal(m.state_dict()[k], raw[k]) for k in raw)
    return during and after, f"VAL sees the EMA weights {during}; restored after {after}"


@case
def cm_ckpt_state_dict_holds_ema():
    torch = L.torch
    m, hook, runner = ema_setup()
    set_all(m, 1.0)
    hook.after_train_iter(runner, 0)
    set_all(m, 5.0)
    runner.iter = 1
    hook.after_train_iter(runner, 1)
    ema = {k: v.clone() for k, v in hook.ema_model.module.state_dict().items()}
    raw = {k: v.clone() for k, v in m.state_dict().items()}
    ck = {"state_dict": dict(m.state_dict()), "meta": {}}
    hook.before_save_checkpoint(runner, ck)
    sd, es = ck["state_dict"], ck["ema_state_dict"]
    ok = (all(torch.equal(sd[k], ema[k]) for k in raw if raw[k].is_floating_point())
          and all(torch.equal(es[f"module.{k}"], raw[k]) for k in raw if raw[k].is_floating_point())
          and int(es["steps"]) == 2)
    return ok, f"state_dict = EMA; ema_state_dict = raw under module.* plus steps {int(es['steps'])}"


KD_LOAD_MODULES = ("src/distill/segnext_teacher.py", "src/distill/teacher.py", "src/eval/teacher_diag.py",
                   "src/training/train_distill.py", "src/training/teacher_components.py",
                   "src/training/teacher_arm_components.py")


@case
def cm_kd_loader_reads_state_dict():
    torch = L.torch
    ema = {"backbone.w": torch.tensor([1.0, 2.0]), "decode_head.w": torch.tensor([3.0])}
    raw = {"module.backbone.w": torch.tensor([9.0, 9.0]), "module.decode_head.w": torch.tensor([9.0])}
    p = W / "kd_ckpt" / "ema_checkpoint.pth"
    p.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": ema, "ema_state_dict": {**raw, "steps": torch.tensor(2)}, "meta": {}}, p)
    sd = L.stc.load_teacher_state_dict(p)
    loads_ema = sorted(sd) == sorted(ema) and all(torch.equal(sd[k], ema[k]) for k in ema)
    readers = [rel for rel in KD_LOAD_MODULES if "ema_state_dict" in (REPO / rel).read_text(encoding="utf-8")]
    return loads_ema and not readers, f"loader returns state_dict (the EMA weights): {loads_ema}; modules naming " \
                                      f"ema_state_dict: {readers}"


# ---- repeat rule
@case
def cm_repeat_table_exact():
    tac = L.tac
    got = [tac.m_of(n) for n in range(1, 26)]
    want = [5, 4, 3, 3] + [2] * 15 + [1] * 6
    flo = [math.ceil(max(1.0, math.sqrt((20 / 5367) / (n / 5367)))) for n in range(1, 26)]
    return got == want == flo and [L.brt.float_m(n, 5367) for n in range(1, 26)] == want, f"m_c for n_c 1..25: {got}"


@case
def cm_class_balanced_matches_table():
    r = fx("root")
    rc = L.lta.repeat_check(load_cfg("R2"), r.lookups, r.table_sha)
    m = {c: r.lookups["rare"][c] for c in range(1, 20)}
    ok = (rc["per_index_equal"] and rc["length"] == r.lookups["repeated"] and m == {c: L.tac.m_of(c) for c in m}
          and all(r.lookups["n_c"][c] >= 20 for c in range(20, 116)))
    return ok, f"{rc}; rare m_c {m}"


@case
def cm_no_disease_image_appears_once():
    ds = build_r2_dataset()
    inner = ds.dataset
    counts = [0] * len(inner)
    for i in ds.repeat_indices:
        counts[i] += 1
    none = [i for i in range(len(inner)) if fx("root").plan[Path(inner.get_data_info(i)["img_path"]).stem] is None]
    cats = [inner.get_cat_ids(i) for i in none]
    ok = len(none) == NO_DISEASE and all(c == [] for c in cats) and all(counts[i] == 1 for i in none) \
        and all(0 not in inner.get_cat_ids(i) for i in range(len(inner)))
    return ok, f"{len(none)} images without a disease pixel: cat ids {cats[:2]}, repeats {[counts[i] for i in none]}"


@case
def cm_pass_is_permutation():
    from mmengine.dataset import InfiniteSampler
    ds = build_r2_dataset()
    it = iter(InfiniteSampler(ds, shuffle=True, seed=42))
    n = len(ds)
    first, second = [next(it) for _ in range(n)], [next(it) for _ in range(n)]
    ok = sorted(first) == list(range(n)) == sorted(second) and first != second
    return ok, f"two passes of {n} are permutations of the repeated list"


# ---- paste
@case
def cm_paste_draw_order():
    inner = r2_inner()
    info = info_of(inner, "p00010")                 # class 5: four sources
    hits = 0
    for seed in range(40):
        rng = L.np.random.RandomState(seed)
        with L.Image.open(info["img_path"]) as im, L.Image.open(info["seg_map_path"]) as mk:
            _, _, inf = L.tac.copy_paste(im, mk, 5, list(info["paste_sources"]), str(Path(info["img_path"]).parent),
                                         str(Path(info["seg_map_path"]).parent), rng)
        rep = L.np.random.RandomState(seed)
        if (rep.rand() < 0.5) != inf["decision"]:
            return False, f"seed {seed}: the decision is not the first draw"
        if inf["decision"]:
            s = info["paste_sources"][rep.randint(len(info["paste_sources"]))]
            u = float(rep.uniform(0.75, 1.25))
            if (s, u) != (inf["source"], inf["u"]):
                return False, f"seed {seed}: source and u are not the next draws"
            for _ in range(inf["tries"]):
                rep.randint(0, 24 - inf["size"][0] + 1)
                rep.randint(0, 32 - inf["size"][1] + 1)
            hits += inf["accepted"]
        if rep.get_state()[1].tolist() != rng.get_state()[1].tolist() or rep.get_state()[2] != rng.get_state()[2]:
            return False, f"seed {seed}: the stream position after the paste differs from the replay"
    same = all(same_output(apply_transform(info, s), record_reference(info, s)) for s in range(12))
    return hits > 5 and same, f"{hits} pastes replayed draw by draw; M2/M3 follow on the same stream: {same}"


@case
def cm_non_rare_bit_identical_to_record_transform():
    from src.training.teacher_components import ThesisTeacherTrainTransform
    inner = r2_inner()
    n = 0
    for stem in ("p00190", "p00500", STEMS[-1]):
        info = info_of(inner, stem)
        for seed in (0, 1, 7):
            a = apply_transform(info, seed, ThesisTeacherTrainTransform())
            sa = L.np.random.get_state()[1][:8].tolist()
            b = apply_transform(info, seed)
            sb = L.np.random.get_state()[1][:8].tolist()
            n += same_output(a, b) and sa == sb
    return n == 9, f"{n}/9 identical outputs and global streams"


@case
def cm_rare_no_paste_decision_consumes_one_draw():
    from configs.augment import AUGMENT
    from src.data.transforms import train_preprocess
    np = L.np
    inner = r2_inner()
    n = 0
    for stem in ("p00003", "p00010"):
        info = info_of(inner, stem)
        for seed in range(25):
            np.random.seed(seed)
            rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
            if rng.rand() < 0.5:
                continue
            out = apply_transform(info, seed)
            with L.Image.open(info["img_path"]) as im, L.Image.open(info["seg_map_path"]) as mk:
                img, mask = train_preprocess(im, mk, rng, AUGMENT)
            if not (np.array_equal(out["img"], np.ascontiguousarray(img[..., ::-1]))
                    and np.array_equal(out["gt_seg_map"], mask.astype(np.uint8))):
                return False, f"{stem} seed {seed}"
            n += 1
    return n >= 10, f"{n} no-paste samples equal the record path after one draw"


@case
def cm_single_image_class_nothing_pasted():
    from configs.augment import AUGMENT
    from src.data.transforms import train_preprocess
    np = L.np
    info = info_of(r2_inner(), "p00000")            # class 1: one TRAIN image
    t = L.tac.RareClassCopyPasteTrainTransform()
    ok = info["rare_class"] == 1 and info["paste_sources"] == []
    for seed in range(12):
        out = apply_transform(info, seed, t)
        np.random.seed(seed)
        rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
        rng.rand()
        with L.Image.open(info["img_path"]) as im, L.Image.open(info["seg_map_path"]) as mk:
            img, mask = train_preprocess(im, mk, rng, AUGMENT)
        ok = ok and np.array_equal(out["img"], np.ascontiguousarray(img[..., ::-1])) \
            and np.array_equal(out["gt_seg_map"], mask.astype(np.uint8))
    return ok and t.counts["empty_pool"] > 0 and t.counts["pasted"] == 0, f"counts {t.counts}"


@case
def cm_patch_resize_rule():
    ps = L.tac.patch_size
    got = {"5x5 at f=1/2": ps(5, 5, (10, 10), (20, 20), 1.0), "7x3 at f=1/2": ps(7, 3, (10, 10), (20, 20), 1.0),
           "1x1 at f=0.075": ps(1, 1, (10, 10), (100, 100), 0.75), "10x10 at f=2.5": ps(10, 10, (40, 40), (20, 20), 1.25)}
    want = {"5x5 at f=1/2": (2, 2), "7x3 at f=1/2": (4, 2), "1x1 at f=0.075": (1, 1), "10x10 at f=2.5": (25, 25)}
    return got == want, f"{got} (2.5 -> 2 and 3.5 -> 4 and 1.5 -> 2: round half to even; at least 1)"


# Fixtures on which a float fit-down differs from the exact one (found by a search over small sizes): a 19 x 38
# patch in a 16 x 21 T has g = 21/38 and a height of exactly 10.5, which round() takes to 10, while the float
# 19 * (21 / 38) = 10.500000000000002 rounds to 11; a 98 x 147 patch in a 1 x 2 T has a width of exactly 1.5 (-> 2),
# while the float 147 * (1 / 98) = 1.4999999999999998 rounds to 1.
FIT_DOWN_CASES = [((19, 38, (16, 21), (21, 21), 1.0), (10, 21)), ((98, 147, (1, 2), (2, 2), 1.0), (1, 2))]


@case
def cm_patch_fit_down_rule():
    ps = L.tac.patch_size
    a = ps(30, 10, (20, 40), (30, 30), 1.0)          # f = 4/3: 40 x 13 > T: g = 1/2: 20 x 6.5 -> 20 x 6
    b = ps(10, 50, (30, 20), (25, 25), 1.0)          # f = 6/5: 12 x 60 > T: g = 1/3: 4 x 20
    ok = a == (20, 6) and b == (4, 20)
    extra = [(args, want, ps(*args)) for args, want in FIT_DOWN_CASES]
    ok = ok and bool(extra) and all(w == g for _, w, g in extra)
    return ok, f"{a}, {b}; exact on the float-distinguishing fixtures (a float fit-down gives (11, 21) and (1, 1)): " \
               f"{[g for _, _, g in extra]}"


def overlap_fixture():
    np = L.np
    m = np.zeros((40, 40), dtype=np.uint8)
    m[0, 0] = m[0, 1] = 9                           # placement 1 at (0, 0): 2 of 20 patch pixels (10%)
    m[20, 20] = 9                                    # placement 2 at (20, 20): 1 of 20 (5%)
    return t_pair(m)


@case
def cm_overlap_5pct_rule():
    im, mk = overlap_fixture()
    rng = Script(ints=[0, 0, 0, 20, 20, 5, 5])
    _, mask, info = paste(im, mk, rng)
    ok = info["tries"] == 2 and info["at"] == [20, 20] and info["pasted_pixels"] == 20 and rng.ints == [5, 5]
    return ok, f"tries {info['tries']} at {info.get('at')}: 10% rejected, 5% accepted"


@case
def cm_ten_tries_then_nothing():
    np = L.np
    im, mk = t_pair(np.full((40, 40), 9, dtype=np.uint8))
    rng = Script(ints=[0] + [1, 1] * 12)
    img, mask, info = paste(im, mk, rng)
    ok = info["tries"] == 10 and not info["accepted"] and img is im and mask is mk and len(rng.ints) == 4
    return ok, f"tries {info['tries']}, accepted {info['accepted']}, scripted ints left {len(rng.ints)}"


@case
def cm_overwrite_no_blend_label_c():
    np = L.np
    im, mk = t_pair(np.zeros((40, 40), dtype=np.uint8))
    img, mask, info = paste(im, mk, Script(ints=[0, 5, 7]))
    p = fx("paste")
    with L.Image.open(p.img_dir / "s_rect.jpg") as s_im, L.Image.open(p.ann_dir / "s_rect.png") as s_mk:
        s_img, s_mask, _ = L.tac.paste_frame(s_im, s_mk)
        s_img.load()
        s_lab = np.array(s_mask)
    p_rgb, patch = L.tac.make_patch(s_img, s_lab, 9, (40, 40), 1.0)
    out_rgb, out_lab, src = np.asarray(img), np.asarray(mask), np.asarray(im)
    y, x = 5, 7
    hp, wp = patch.shape
    region_rgb, region_lab = out_rgb[y:y + hp, x:x + wp], out_lab[y:y + hp, x:x + wp]
    outside = np.ones((40, 40), dtype=bool)
    outside[y:y + hp, x:x + wp][patch] = False
    ok = (info["accepted"] and np.array_equal(region_rgb[patch], p_rgb[patch]) and (region_lab[patch] == 9).all()
          and np.array_equal(out_rgb[outside], src[outside]) and (out_lab[outside] == 0).all())
    return ok, f"{int(patch.sum())} pasted pixels equal the patch exactly; label 9; the rest unchanged"


@case
def cm_one_class_per_image_kept():
    inner = r2_inner()
    seen = set()
    for stem, c in (("p00010", 5), ("p00003", 3), ("p00150", 17)):
        info = info_of(inner, stem)
        for seed in range(15):
            out = apply_transform(info, seed)
            labels = {int(v) for v in L.np.unique(out["gt_seg_map"])} - {0, 255}
            if labels - {c}:
                return False, f"{stem} seed {seed}: labels {labels}"
            seen.add(stem)
    return len(seen) == 3, "every pasted sample keeps its one disease class"


# ---- table
def mini_root(tag, two_label=None):
    plan = {f"m{i:03d}": (1 + i % 4) for i in range(40)}
    plan.update({f"m{i:03d}": None for i in range(36, 40)})
    if two_label:
        plan[two_label] = (3, 5)
    root = W / "mini" / tag
    make_root(root, plan)
    return root, plan


def mini_strata(tag, plan, bump=None):
    doc = strata_doc(plan)
    if bump:
        doc["per_class"][bump - 1]["images"] += 1
    p = W / "mini" / f"{tag}_strata.json"
    p.write_bytes(canon(doc))
    return p


@case
def cm_table_builder_counts_vs_strata():
    root, plan = mini_root("counts")
    good = build_table(root, mini_strata("counts_ok", plan), W / "mini" / "counts_ok_out",
                       extra=["--expected-images", "40", "--generated-utc", "2026-10-11T00:00:00Z"])
    bad = build_table(root, mini_strata("counts_bad", plan, bump=2), W / "mini" / "counts_bad_out",
                      extra=["--expected-images", "40", "--generated-utc", "2026-10-11T00:00:00Z"])
    stop_p = W / "mini" / "counts_bad_out" / "r2_table_STOP.json"
    stop = json.loads(stop_p.read_text(encoding="utf-8")) if stop_p.is_file() else {}
    ok = (good[0] == 0 and (W / "mini" / "counts_ok_out" / "r2_table_v1.json").is_file() and bad[0] == 1
          and [d["class_id"] for d in stop.get("differing_classes", [])] == [2]
          and not (W / "mini" / "counts_bad_out" / "r2_table_v1.json").exists())
    return ok, f"matching strata: exit {good[0]}; class 2 off by one: exit {bad[0]}, STOP lists " \
               f"{[d['class_id'] for d in stop.get('differing_classes', [])]}"


@case
def cm_launcher_asserts_repeat_counts():
    r = fx("root")
    ok_run = L.lta.repeat_check(load_cfg("R2"), r.lookups, r.table_sha)
    orig = L.tac.RareClassTrainDataset.get_cat_ids

    def drop_class_2(self, idx):
        ids = orig(self, idx)
        return [] if ids == [2] else ids
    with patched((L.tac.RareClassTrainDataset, "get_cat_ids", drop_class_2)):
        try:
            L.lta.repeat_check(load_cfg("R2"), r.lookups, r.table_sha)
            code = None
        except L.lta.PreflightError as e:
            code = e.code
    return ok_run["per_index_equal"] and code == "r2_repeat_counts", f"{ok_run}; a dataset repeating class 2 " \
                                                                      f"once: [{code}]"


@case
def cm_launcher_refuses_stop_table():
    stop = W / "stop" / "r2_table_STOP.json"
    stop.parent.mkdir(exist_ok=True)
    stop.write_bytes(canon({"format": L.tac.STOP_FORMAT, "two_label_images": ["p00001"]}))
    args = L.lta.parse_args(launch_argv("R1", r2_table=str(stop), r2_table_sha256=sha(stop)))
    try:
        L.lta.gate_table(args, L.lta.ss.HeadSource(fx("repo").repo), None)
        code = None
    except L.lta.PreflightError as e:
        code = e.code
    return code == "r2_table_stop", f"[{code}]"


# ---- launch order
@case
def cm_r1_requires_r2_pinned():
    p_t = fx("repo").p_t
    no_r2 = records_commit(base_rows(p_t, **{"DL-904": None}))
    no_tab = records_commit(base_rows(p_t, **{"DL-904": ("RECORDED", f"R2 config sha256 "
                                                                       f"{sha(REPO / L.lta.ARM_CONFIGS['R2'])}; "
                                                                       f"diff {L.lta.reg_sha256('R2')}; P_T = {p_t}")}))
    got = {"no_r2_config_row": rows_gate("R1", no_r2), "no_table_sha": rows_gate("R1", no_tab, r2_table_dl="DL-904"),
           "pinned": rows_gate("R1", fx("repo").h)}
    return got == {"no_r2_config_row": "r2_not_pinned", "no_table_sha": "r2_not_pinned", "pinned": None}, got


@case
def cm_r2_requires_r1_complete():
    got = {}
    for status in ("incomplete", "complete"):
        rec = getattr(fx("r1record"), status)
        args = L.lta.parse_args(launch_argv("R2", r1_record=str(rec.path), r1_record_sha256=rec.sha))
        try:
            got[status] = L.lta.gate_r1_record(args)["status"]
        except L.lta.PreflightError as e:
            got[status] = e.code
    return got == {"incomplete": "r1_not_complete", "complete": "complete"}, got


@case
def cm_approval_gate():
    p_t = fx("repo").p_t
    variants = {"no_row": None, "decided": ("DECIDED", "AM-18 written approval (smoke)"),
                "no_marker": ("RECORDED", "the adviser replied (smoke)")}
    got = {k: rows_gate("R1", records_commit(base_rows(p_t, **{"DL-901": v}))) for k, v in variants.items()}
    got["recorded"] = rows_gate("R1", fx("repo").h)
    return got == {"no_row": "approval_missing", "decided": "approval_missing", "no_marker": "approval_missing",
                   "recorded": None}, got


# ---- record
@case
def cm_record_divergence_detection():
    work = make_run("diverged", nan_iter=150)
    rc_p, _, out_p = run_main(L.atr.main, ["progress", "--work-dir", str(work)])
    out = W / "records" / "arm_teacher_record_R1_diverged_run.json"
    rc_r, _, _ = run_main(L.atr.main, record_argv(work, out))
    doc = json.loads(out.read_text(encoding="utf-8"))
    work_v = make_run("val_nan", val_nan_at=8000)
    rc_v, _, out_v = run_main(L.atr.main, ["progress", "--work-dir", str(work_v)])
    f = doc["first_nonfinite"] or {}
    ok = (rc_p == 1 and "at iteration 150" in out_p and rc_r == 1 and doc["status"] == "diverged"
          and f.get("iteration") == 150 and f.get("source") == "train" and rc_v == 1 and "at iteration 8000" in out_v)
    return ok, f"progress exit {rc_p}; record {doc['status']} {f}; VAL NaN: exit {rc_v}"


@case
def cm_record_complete():
    rec = fx("complete_record")
    d = rec.doc
    sel = d["selection"]["checkpoint"]
    p = rec.work / sel["name"]
    ok = (d["status"] == "complete" and sel["name"] == "iter_40000.pth" and sel["sha256"] == sha(p)
          and sel["bytes"] == p.stat().st_size and [c["iteration"] for c in d["interval_checkpoints"]]
          == list(range(4000, 40001, 4000)) and d["ema_checks"]["steps_equal_iteration"]
          and d["ema_checks"]["state_dict_differs_from_raw"] and d["work_dir"]["attempt"] == 1
          and d["logs"]["scalars_sha256"] == sha(rec.work / "20261011_000000" / "vis_data" / "scalars.json")
          and d["eval_config"]["sha256"] == sha(L.lta.RECORD_CONFIG))
    return ok, f"status {d['status']}; selected {sel['name']}; ema {d['ema_checks']}"


@case
def cm_record_status_incomplete():
    work = make_run("stopped", last_iter=12000, vals=3, final=False)
    out = W / "records" / "arm_teacher_record_R1_stopped_run.json"
    rc, _, _ = run_main(L.atr.main, record_argv(work, out))
    doc = json.loads(out.read_text(encoding="utf-8"))
    early = make_run("stopped_early", no_log=True)
    out_e = W / "records" / "arm_teacher_record_R1_stopped_early.json"
    rc_e, _, _ = run_main(L.atr.main, record_argv(early, out_e))
    doc_e = json.loads(out_e.read_text(encoding="utf-8")) if out_e.is_file() else {}
    ok = (rc == 1 and doc["status"] == "incomplete" and doc["first_nonfinite"] is None and rc_e == 1
          and doc_e.get("status") == "incomplete" and doc_e.get("progress", {}).get("last_logged_iteration") == 0)
    return ok, f"stopped at 12,000: exit {rc}, {doc['status']}; before the first log row: exit {rc_e}, " \
               f"{doc_e.get('status')}"


# ---- go rule
@case
def cm_go_rule_conditions():
    d = L.agr.decide
    gate, floor = 0.38576993346214294, 0.37576993346214294
    got = {"=gate, ECE equal": d(gate, 0.1, 0.1)[0], "just over gate": d(math.nextafter(gate, 1), 0.2, 0.1)[0],
           "=floor, ECE lower": d(floor, 0.09, 0.1)[0], "just under floor": d(math.nextafter(floor, 0), 0.09, 0.1)[0],
           "=floor, ECE equal": d(floor, 0.1, 0.1)[0]}
    want = {"=gate, ECE equal": "no_go", "just over gate": "go", "=floor, ECE lower": "go",
            "just under floor": "no_go", "=floor, ECE equal": "no_go"}
    g = go_files("conditions", val_x=gate, ece_x=0.07, ece_b=0.06)
    rc, _, out = run_main(L.agr.main, go_argv(g, g.dir / "go.json"))
    doc = json.loads((g.dir / "go.json").read_text(encoding="utf-8"))
    ok = (got == want and L.agr.VAL_GATE == gate and L.agr.VAL_FLOOR == floor and rc == 0
          and doc["verdict"] == "no_go" and doc["values"]["val_x"] == gate)
    return ok, f"{got}; through the CLI at VAL_X = gate: {doc['verdict']}"


@case
def cm_go_rule_control_reproduction():
    g = go_files("repro")
    rc_ok, _, _ = run_main(L.agr.main, go_argv(g, g.dir / "ok.json"))
    v = go_files("void", control_miou=0.38576993346214294 + 2e-5, control_status="not reproduced", gate_passed=False)
    rc_v, _, out_v = run_main(L.agr.main, go_argv(v, v.dir / "void.json"))
    with patched((L.agr, "GIT_ROOT", fx("dlrepo"))):
        rc_n, _, _ = run_main(L.agr.main, go_argv(v, v.dir / "ne.json", ["--void-session-dl", "DL-951"]))
    ne = json.loads((v.dir / "ne.json").read_text(encoding="utf-8")) if (v.dir / "ne.json").is_file() else {}
    o = go_files("otherhost", host_arm="another laptop")
    rc_h, _, _ = run_main(L.agr.main, go_argv(o, o.dir / "host.json"))
    with patched((L.agr, "GIT_ROOT", fx("dlrepo"))):
        rc_f, _, _ = run_main(L.agr.main, go_argv(o, o.dir / "fault.json", ["--host-fault-dl", "DL-950"]))
    late = go_files("late", control_head="e" * 40)
    rc_c, _, _ = run_main(L.agr.main, go_argv(late, late.dir / "commit.json"))
    fault = json.loads((o.dir / "fault.json").read_text(encoding="utf-8")) if (o.dir / "fault.json").is_file() else {}
    ok = (rc_ok == 0 and rc_v == 1 and "void" in out_v and not (v.dir / "void.json").exists() and rc_n == 1
          and ne.get("verdict") == "not_evaluable" and rc_h == 2 and rc_f == 0
          and (fault.get("session") or {}).get("host_fault", {}).get("dl") == "DL-950" and rc_c == 2)
    return ok, f"reproduced: exit {rc_ok}; 2e-5 off: exit {rc_v}, nothing written; recurrence: {ne.get('verdict')}; " \
               f"another host: exit {rc_h}, with a documented fault: exit {rc_f}; another commit: exit {rc_c}"


@case
def cm_go_rule_binds_m12_checkpoint():
    g = go_files("binds", arm_ckpt="0" * 64)
    rc, _, out = run_main(L.agr.main, go_argv(g, g.dir / "go.json"))
    return rc == 2 and "M12" in out and not (g.dir / "go.json").exists(), f"exit {rc}"


# ---- refusals
def never_created(p: Path) -> bool:
    return not os.path.lexists(p)


@case
def rf_test_path_launcher():
    bad = W / "plantseg_test_root"
    rc, err, _ = launch("R1", data_root=str(bad))
    return rc == 2 and "[test_path]" in err and never_created(bad), err[:80]


@case
def rf_test_path_builder():
    bad = W / "plantseg_test_root"
    rc, _, out = run_main(L.brt.main, ["--data-root", str(bad), "--strata", str(fx("root").strata), "--out-dir",
                                       str(W / "tbl_never")])
    return rc == 2 and "REFUSED" in out and never_created(bad) and never_created(W / "tbl_never"), out.strip()[:80]


@case
def rf_test_path_record():
    bad = W / "teacher_test_run"
    rc, _, out = run_main(L.atr.main, ["progress", "--work-dir", str(bad)])
    return rc == 2 and "REFUSED" in out and never_created(bad), out.strip()[:80]


@case
def rf_test_path_go_rule():
    g = go_files("tp")
    bad = g.dir / "go_test.json"
    rc, _, out = run_main(L.agr.main, go_argv(g, bad))
    return rc == 2 and "REFUSED" in out and never_created(bad), out.strip()[:80]


@case
def rf_existing_output_builder():
    out = W / "existing_builder_out"
    out.mkdir()
    rc, _, text = build_table(fx("root").root, fx("root").strata, out)
    return rc == 2 and "exists" in text and not any(out.iterdir()), text.strip()[:80]


@case
def rf_existing_output_record():
    out = W / "records" / "existing_record.json"
    out.write_text("{}\n", encoding="utf-8")
    rc, _, text = run_main(L.atr.main, record_argv(fx("complete_record").work, out))
    return rc == 2 and out.read_text(encoding="utf-8") == "{}\n", text.strip()[:80]


@case
def rf_existing_output_go_rule():
    g = go_files("exists")
    out = g.dir / "go.json"
    out.write_text("{}\n", encoding="utf-8")
    rc, _, text = run_main(L.agr.main, go_argv(g, out))
    return rc == 2 and out.read_text(encoding="utf-8") == "{}\n", text.strip()[:80]


@case
def rf_missing_input_strata():
    rc, _, text = build_table(fx("root").root, W / "strata" / "absent.json", W / "tbl_absent")
    return rc == 2 and never_created(W / "tbl_absent"), text.strip()[:80]


@case
def rf_missing_input_selection():
    work = make_run("no_selection", selection=False)
    out = W / "records" / "no_selection.json"
    rc, _, text = run_main(L.atr.main, record_argv(work, out))
    return rc == 2 and "teacher_selection.json" in text and never_created(out), text.strip()[:90]


@case
def rf_missing_input_control_output():
    g = go_files("nocontrol")
    argv = go_argv(g, g.dir / "go.json")
    argv[argv.index("--control-output") + 1] = str(g.dir / "absent_control.json")
    rc, _, text = run_main(L.agr.main, argv)
    return rc == 2 and never_created(g.dir / "go.json"), text.strip()[:80]


@case
def rf_wrong_checkpoint_pin_init_sha():
    other = W / "init_other" / ADE_NAME
    other.parent.mkdir()
    other.write_bytes(b"another file\n")
    rc, err, _ = launch("R1", init_ckpt=str(other))
    return rc == 2 and "[init_ckpt_sha_mismatch]" in err, err[:80]


@case
def rf_wrong_checkpoint_pin_config_sha():
    rc, err, _ = launch("R1", config_sha256="0" * 64)
    return rc == 2 and "[config_sha_mismatch]" in err, err[:80]


@case
def rf_wrong_checkpoint_pin_r1_record_sha():
    rec = fx("r1record").complete
    changed = W / "records" / "arm_teacher_record_R1_changed.json"
    changed.write_bytes(rec.path.read_bytes().replace(b'"complete"', b'"complete" '))
    rc, err, _ = launch("R2", r1_record=str(changed))
    return rc == 2 and "[r1_record_sha256_mismatch]" in err, err[:80]


@case
def rf_r2_after_diverged_r1():
    div = fx("r1record").diverged
    h = records_commit(base_rows(fx("repo").p_t, **{"DL-905": ("RECORDED", f"R1 record sha256 {div.sha}")}))
    rc, err, _ = launch("R2", records=h, r1_record=str(div.path), r1_record_sha256=div.sha)
    return rc == 2 and "[r1_diverged]" in err, err[:80]


@case
def rf_work_dir_in_repo():
    inside = REPO / "teacher-r1_s42_40k_a1"
    argv = [a for a in launch_argv("R1", work=inside) if a != "--create-work-dir"]
    with launcher_ctx():
        rc, err, _ = run_main(L.lta.main, argv)
    return rc == 2 and "[work_dir_inside_repo]" in err and never_created(inside), err[:80]


# ---- determinism
@case
def dt_table_twice_identical_bytes():
    r = fx("root")
    rc, _, _ = build_table(r.root, r.strata, W / "r2_table_again")
    again = W / "r2_table_again" / "r2_table_v1.json"
    with patched((L.brt, "strata_check", synthetic_strata_check)):
        rc_c, _, out_c = run_main(L.brt.main, ["--check", str(r.table), "--data-root", str(r.root), "--strata",
                                               str(r.strata)])
    return rc == 0 and again.read_bytes() == r.table_b and rc_c == 0 and "IDENTICAL" in out_c, f"sha {r.table_sha}"


@case
def dt_paste_twice_identical():
    info = info_of(r2_inner(), "p00010")
    pairs = [(apply_transform(info, s), apply_transform(info, s)) for s in range(6)]
    return all(same_output(a, b) for a, b in pairs), "six seeds, two runs each, byte-identical"


@case
def dt_record_twice_identical():
    rec = fx("complete_record")
    out = W / "records" / "arm_teacher_record_R1_complete_again.json"
    rc, _, _ = run_main(L.atr.main, record_argv(rec.work, out))
    return rc == 0 and out.read_bytes() == rec.path.read_bytes(), f"sha {rec.sha}"


@case
def dt_go_rule_twice_identical():
    g = go_files("twice")
    a, b = g.dir / "a.json", g.dir / "b.json"
    rc_a, _, _ = run_main(L.agr.main, go_argv(g, a))
    rc_b, _, _ = run_main(L.agr.main, go_argv(g, b))
    return rc_a == rc_b == 0 and a.read_bytes() == b.read_bytes(), f"sha {sha(a)}"


# ---- the GO's cases
@case
def cm_launcher_env_resolution():
    lta, ltf = L.lta, L.ltf
    with launcher_ctx():
        args = lta.parse_args(launch_argv("R2"))
        prov = ltf.preflight(args, record_gpu_names=False)
        got = {}
        lta.gate_environment(args, prov)
        got["set"] = {k: os.environ.get(k) for k in ENV_VARS}
        cfg = load_cfg_noenv(lta.ARM_CONFIGS["R2"])
        got["resolves"] = lta.arm_config_problems(cfg, prov, args) == []
        os.environ["PLANTSEG_DATA_ROOT"] = prov["plantseg_root"]
        lta.gate_environment(args, prov)
        got["same_inherited_ok"] = True
        os.environ["PLANTSEG_DATA_ROOT"] = str(W)
        try:
            lta.gate_environment(args, prov)
            got["conflict"] = None
        except lta.PreflightError as e:
            got["conflict"] = e.code
    want_set = {"PLANTSEG_DATA_ROOT": prov["plantseg_root"], "SEGNEXT_ADE20K_CKPT": prov["ade20k_init_checkpoint"]["path"],
                "TEACHER_WORK_DIR": prov["work_dir"], "PLANTSEG_R2_TABLE": str(fx("root").table.resolve())}
    ok = got["set"] == want_set and got["resolves"] and got["conflict"] == "env_conflict"
    return ok, f"conflict [{got['conflict']}]; config resolves to the gated values {got['resolves']}"


def load_cfg_noenv(rel):
    from mmengine.config import Config
    return Config.fromfile(str(REPO / rel))


@case
def rf_launch_config_mismatch():
    lta, ltf = L.lta, L.ltf
    extra = W / "cfg" / "r1_unregistered.py"
    extra.parent.mkdir(exist_ok=True)
    extra.write_text(f"_base_ = ['{REPO / lta.ARM_CONFIGS['R1']}']\ntrain_cfg = dict(val_interval=2000)\n",
                     encoding="utf-8")
    got = {}
    with launcher_ctx():
        args = lta.parse_args(launch_argv("R1"))
        prov = ltf.preflight(args, record_gpu_names=False)
        lta.gate_environment(args, prov)
        try:
            lta.gate_configuration(args, prov)
            got["registered"] = None
        except lta.PreflightError as e:
            got["registered"] = e.code
        try:
            lta.gate_configuration(args, dict(prov, config_path=str(extra)))
            got["unregistered"] = None
        except lta.PreflightError as e:
            got["unregistered"] = e.code
    return got == {"registered": None, "unregistered": "launch_config_mismatch"}, got


@case
def rf_teacher_launch_without_recheck():
    p_t = fx("repo").p_t
    no_tok = records_commit(base_rows(p_t, **{"DL-902": ("RECORDED", "the Tier F re-check (no token) (smoke)")}))
    got = {"no_token": rows_gate("R1", no_tok), "token": rows_gate("R1", fx("repo").h)}
    rc, err, _ = launch("R1", records=no_tok)
    return got == {"no_token": "approval_recheck_missing", "token": None} and rc == 2, f"{got}; {err[:60]}"


@case
def cm_empty_patch_draws_one_placement():
    np = L.np
    im, mk = t_pair(np.zeros((16, 16), dtype=np.uint8))
    rng = Script(u=0.75, ints=[0, 3, 4, 9, 9])
    img, mask, info = paste(im, mk, rng, pool=("s_diag",))
    ok = (info["size"] == [1, 1] and info["tries"] == 1 and info["accepted"] and info["pasted_pixels"] == 0
          and img is im and mask is mk and rng.ints == [9, 9])
    return ok, f"size {info['size']}, tries {info['tries']}, pasted {info['pasted_pixels']}, ints left {rng.ints}"


@case
def cm_paste_exif_rotated_frame():
    np = L.np
    p = fx("paste")
    with L.Image.open(p.img_dir / "t_rot.jpg") as im, L.Image.open(p.ann_dir / "t_rot.png") as mk:
        stored = im.size
        img, mask, info = paste(im, mk, Script(ints=[0, 28, 19]))
        out_size, mask_size = img.size, mask.size
        lab = np.asarray(mask)
    t = L.tac.RareClassCopyPasteTrainTransform()
    t._count({"img_path": "t_rot.jpg"}, info)
    ok = (stored == (32, 24) and info["frame_resized_t"] and out_size == mask_size == (24, 32) and info["accepted"]
          and (lab[28:31, 19:23] == 9).any() and t.counts["pasted_with_resized_frame"] == 1)
    return ok, f"stored {stored}, frame {out_size}, mask {mask_size}, counts {t.counts['pasted_with_resized_frame']}"


@case
def rf_table_two_labels_stop():
    root, plan = mini_root("two", two_label="m005")
    rc, _, _ = build_table(root, mini_strata("two", plan), W / "mini" / "two_out",
                           extra=["--expected-images", "40", "--generated-utc", "2026-10-11T00:00:00Z"])
    stop_p = W / "mini" / "two_out" / "r2_table_STOP.json"
    stop = json.loads(stop_p.read_text(encoding="utf-8")) if stop_p.is_file() else {}
    ok = rc == 1 and stop.get("two_label_images") == ["m005"] and never_created(W / "mini" / "two_out" /
                                                                                 "r2_table_v1.json")
    return ok, f"exit {rc}; two-label images {stop.get('two_label_images')}"


@case
def cm_r2_data_info_fields():
    inner = r2_inner()
    rare, non, none = info_of(inner, "p00010"), info_of(inner, "p00500"), info_of(inner, STEMS[-1])
    want_src = [s for s in fx("root").lookups["sources"][5] if s != "p00010"]

    def no_table(*a, **k):
        raise AssertionError("the transform read the table")
    saved = os.environ.pop("PLANTSEG_R2_TABLE", None)
    try:
        with patched((L.tac, "read_table", no_table)):
            outs = [apply_transform(info, 3) for info in (rare, non, none)]
    finally:
        if saved is not None:
            os.environ["PLANTSEG_R2_TABLE"] = saved
    ok = (rare["rare_class"] == 5 and rare["paste_sources"] == want_src and non["rare_class"] is None
          and non["paste_sources"] == [] and none["rare_class"] is None and len(outs) == 3)
    return ok, f"rare: {rare['rare_class']}, {len(rare['paste_sources'])} sources; the transform ran without the " \
               "table or its variable"


@case
def cm_canonical_diff_table_path_by_variable():
    other = W / "elsewhere" / "r2_table_v1.json"
    a = L.lta.canonical_bytes(arm_diff("R2"))
    b = L.lta.canonical_bytes(arm_diff("R2", table=other))
    reg = L.lta.canonical_bytes(L.lta.reg("R2"))
    return a == b == reg and b'"$PLANTSEG_R2_TABLE"' in a, f"two hosts' table paths give one diff, sha {sha(a)}"


@case
def rf_r2_table_file_sha_mismatch():
    r = fx("root")
    copy_p = W / "tablecopy" / "r2_table_v1.json"
    copy_p.parent.mkdir()
    copy_p.write_bytes(r.table_b + b"\n")
    rc, err, _ = launch("R2", r2_table=str(copy_p))
    return rc == 2 and "[r2_table_sha256_mismatch]" in err, err[:80]


def big_log():
    p = W / "reference" / "slow_scalars.json"
    if not p.exists():
        p.write_text("".join(json.dumps({"time": 100.0, "iter": 50 * (k + 1)}) + "\n" for k in range(3)),
                     encoding="utf-8")
    return p


@case
def cm_teacher_arith_reference_log():
    s = fx("scalars")
    rc, err, out = launch("R1", reference_log=str(s.path), reference_log_sha256=s.sha)
    ar = provenance_of(out)["arithmetic"] if rc == 0 else {}
    ref = (ar.get("references") or [{}])[0]
    slow = big_log()
    rc_s, err_s, _ = launch("R1", reference_log=str(slow), reference_log_sha256=sha(slow))
    ok = (rc == 0 and ar.get("applied_s_per_iter") == s.median and ref.get("rows") == 80 and ar.get("cut") is False
          and rc_s == 2 and "[arm_cut_by_arithmetic]" in err_s)
    return ok, f"median {ar.get('applied_s_per_iter')} over {ref.get('rows')} training rows: not cut; 100 s/iter: " \
               f"{err_s[:40]}"


@case
def cm_teacher_arith_no_measurement_not_cut():
    rc, err, out = launch("R1")
    return rc == 0 and "no measurement: not cut" in out, f"exit {rc} {err[:60]}"


@case
def cm_rehearsal_covers_gates_6_to_12():
    s = fx("scalars")
    work = W / "runs" / "rehearsal" / "teacher-r2_s42_40k_a1"
    rc, err, out = launch("R2", work=work, reference_log=str(s.path), reference_log_sha256=s.sha)
    doc = provenance_of(out) if rc == 0 else {}
    p_t = fx("repo").p_t
    rc_rows, err_rows, _ = launch("R2", records=records_commit(base_rows(p_t, **{"DL-905": None})))
    stop = W / "stop" / "r2_table_STOP.json"
    rc_stop, err_stop, _ = launch("R1", r2_table=str(stop), r2_table_sha256=sha(stop),
                                  records=records_commit(base_rows(p_t, **{"DL-904": (
                                      "RECORDED", f"R2 config sha256 {sha(REPO / L.lta.ARM_CONFIGS['R2'])}; diff "
                                      f"{L.lta.reg_sha256('R2')}; P_T = {p_t}; table {sha(stop)}")})))
    ok = (rc == 0 and "PREFLIGHT" in out and sorted(doc.get("rows", {})) == ["approval", "config", "r1_record",
                                                                              "r2_config", "r2_table", "recheck"]
          and doc["registered_difference"]["sha256"] == L.lta.reg_sha256("R2")
          and doc["r2_table"]["repeat_check"]["per_index_equal"] and doc["r1_record"]["status"] == "complete"
          and doc["arithmetic"]["measured"] and doc["launch"] is False and not (work / "teacher_arm_launch_provenance.json").exists()
          and rc_rows == 2 and "[r1_record_row_invalid]" in err_rows and rc_stop == 2 and "[r2_table_stop]" in err_stop)
    return ok, f"rehearsal exit {rc}; without R1's row: {err_rows[:45]}; a STOP table: {err_stop[:40]}"


@case
def cm_rows_read_from_records_commit():
    fxr = fx("repo")
    head_has_log = L.lta.ss.HeadSource(fxr.repo).show("HEAD", "docs/DECISION_LOG.md") is not None
    args = L.lta.parse_args(launch_argv("R1"))
    out = L.lta.gate_rows(args, L.lta.ss.HeadSource(fxr.repo), fxr.p_t)
    return not head_has_log and sorted(out["rows"]) == ["approval", "config", "r2_config", "r2_table", "recheck"], \
        f"P_T holds no decision log; every row read at H {fxr.h[:12]}"


@case
def rf_pin_not_ancestor_of_records_commit():
    rc, err, _ = launch("R1", records=fx("repo").side)
    return rc == 2 and "[pin_not_ancestor_of_records_commit]" in err, err[:80]


def prev_record(status, attempt=1):
    p = W / "records" / f"prev_{status}.json"
    p.write_bytes(canon({"format": "arm_teacher_record/1", "arm": "R1", "status": status,
                         "work_dir": {"name": f"teacher-r1_s42_40k_a{attempt}", "attempt": attempt}}))
    return p


def repeat_launch(status, *, report="reports/arms/am8a_r1_a1.md", records=None):
    p = prev_record(status)
    _RUNS[0] += 1
    work = W / "runs" / f"r{_RUNS[0]}" / "teacher-r1_s42_40k_a2"
    extra = ["--previous-record", str(p), "--previous-record-sha256", sha(p)]
    if report is not None:
        extra += ["--am8a-report", report]
    return launch("R1", work=work, records=records or fx("repo").h2, extra=extra)


@case
def rf_teacher_arm_rerun_after_divergence():
    rc, err, _ = repeat_launch("diverged")
    return rc == 2 and "[arm_rerun_after_divergence]" in err, err[:80]


@case
def rf_teacher_arm_repeat_without_report():
    rc_a, err_a, _ = repeat_launch("incomplete", report=None)
    rc_b, err_b, _ = repeat_launch("incomplete", records=fx("repo").h)
    ok = rc_a == rc_b == 2 and "[repeat_report_missing]" in err_a and "[repeat_report_missing]" in err_b
    return ok, f"no flag: {err_a[:40]}; not at H: {err_b[:40]}"


@case
def cm_teacher_arm_repeat_after_incomplete():
    rc, err, out = repeat_launch("incomplete")
    doc = provenance_of(out) if rc == 0 else {}
    prev = doc.get("previous_attempt") or {}
    ok = rc == 0 and doc.get("attempt") == 2 and (prev.get("previous_record") or {}).get("status") == "incomplete" \
        and (prev.get("am8a_report") or {}).get("path") == "reports/arms/am8a_r1_a1.md"
    return ok, f"exit {rc}; attempt {doc.get('attempt')}; {err[:60]}"


# =========================================================================================== mutations
def m_round_not_ceil():
    return patched((L.tac, "m_of", lambda n: None if n < 1 else max(1, round(math.sqrt(20 / n)))))


def m_background_as_class():
    orig = L.tac.RareClassTrainDataset.get_cat_ids
    return patched((L.tac.RareClassTrainDataset, "get_cat_ids", lambda self, i: orig(self, i) or [0]))


def m_paste_after_m2m3():
    from configs.augment import AUGMENT
    from src.data.transforms import train_preprocess
    from src.training.teacher_components import _pack_canvas
    tac = L.tac

    def transform(self, results):
        rng = L.np.random.RandomState(int(L.np.random.randint(0, 2 ** 31 - 1)))
        with L.Image.open(results["img_path"]) as im, L.Image.open(results["seg_map_path"]) as mk:
            img, mask = train_preprocess(im, mk, rng, AUGMENT)
        if results["rare_class"] is not None:
            ip, mp, _ = tac.copy_paste(L.Image.fromarray(img), L.Image.fromarray(mask.astype("uint8")),
                                       int(results["rare_class"]), list(results["paste_sources"]),
                                       str(Path(results["img_path"]).parent), str(Path(results["seg_map_path"]).parent),
                                       rng)
            img, mask = L.np.asarray(ip.convert("RGB")), L.np.asarray(mp).astype("int64")
        return _pack_canvas(results, img, mask)
    return patched((tac.RareClassCopyPasteTrainTransform, "transform", transform))


def m_alpha_blend():
    def overwrite(t_rgb, t_lab, p_rgb, patch, at, c):
        y, x = at
        hp, wp = patch.shape
        region = t_rgb[y:y + hp, x:x + wp]
        region[patch] = ((region[patch].astype("uint16") + p_rgb[patch]) // 2).astype("uint8")
        t_lab[y:y + hp, x:x + wp][patch] = c
    return patched((L.tac, "overwrite", overwrite))


def m_overlap_10():
    return patched((L.tac, "OVERLAP_DEN", 10))


def ema_variant(**over):
    orig = r1_ema_dict
    return patched((SMOKE, "r1_ema_dict", lambda: {**orig(), **over}))


def m_drop_path_01():
    orig = load_cfg

    def load(name, table=None):
        cfg = orig(name, table)
        if name == "R1":
            cfg.model.backbone.drop_path_rate = 0.1
        return cfg
    return patched((SMOKE, "load_cfg", load))


def m_builder_skips_strata():
    return patched((L.brt, "strata_differences", lambda *a, **k: []))


def m_val_pipeline_changed():
    orig = load_cfg

    def load(name, table=None):
        cfg = orig(name, table)
        if name == "R2":
            cfg.val_dataloader.dataset.pipeline = [dict(type="ThesisTeacherEvalTransform"), dict(type="Resize"),
                                                   dict(type="PackSegInputs")]
        return cfg
    return patched((SMOKE, "load_cfg", load))


def m_ge_not_gt():
    agr = L.agr

    def decide(val_x, ece_x, ece_b):
        c1 = val_x >= agr.VAL_GATE
        c2 = val_x >= agr.VAL_FLOOR and ece_x < ece_b
        return ("go" if c1 or c2 else "no_go"), {}
    return patched((agr, "decide", decide))


def m_eleven_tries():
    return patched((L.tac, "MAX_TRIES", 11))


def m_raw_weights_loaded():
    torch = L.torch

    def load(path, map_location="cpu"):
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        return {k[len("module."):]: v for k, v in payload["ema_state_dict"].items() if k.startswith("module.")}
    return patched((L.stc, "load_teacher_state_dict", load))


def m_inherited_env_accepted():
    lta = L.lta

    def gate_environment(args, prov):
        values = {lta.ltf.DATA_ROOT_ENV: prov["plantseg_root"],
                  lta.ltf.INIT_CKPT_ENV: prov["ade20k_init_checkpoint"]["path"], lta.ltf.WORK_DIR_ENV: prov["work_dir"]}
        if args.arm == "R2":
            values["PLANTSEG_R2_TABLE"] = str(Path(args.r2_table).resolve())
        for k, v in values.items():
            os.environ.setdefault(k, v)
        return {k: {"value": os.environ[k]} for k in values}
    return patched((lta, "gate_environment", gate_environment))


def m_frozen_load_launch_config():
    lta, ltf = L.lta, L.ltf
    return patched((lta, "gate_configuration", lambda args, prov: (ltf.load_launch_config(prov), {})))


def m_token_ignored():
    return patched((L.pa, "RECHECK_TOKEN", ""))


def m_empty_patch_special_case():
    orig = L.tac.place
    return patched((L.tac, "place", lambda t_lab, patch, rng: (0, (0, 0)) if not patch.any() else orig(t_lab, patch,
                                                                                                         rng)))


def m_exif_frame_ignored():
    return patched((L.tac, "paste_frame", lambda im, mk: (im.convert("RGB"), mk, False)))


def m_two_label_tolerated():
    orig = L.brt.mask_classes

    def mask_classes(pairs):
        by_stem, _ = orig(pairs)
        return by_stem, []
    return patched((L.brt, "mask_classes", mask_classes))


def m_rows_at_head():
    orig = L.lta.gate_rows

    def gate_rows(args, src, head):
        a = copy.copy(args)
        a.records_commit = "HEAD"
        return orig(a, src, head)
    return patched((L.lta, "gate_rows", gate_rows))


def m_table_path_by_value():
    keys = {k: v for k, v in L.lta.ENV_KEYS.items() if k != "table_path"}
    return patched((L.lta, "ENV_KEYS", keys))


def m_half_up():
    def patch_size(h, w, t_hw, s_hw, u):
        (H_T, W_T), (H_S, W_S) = t_hw, s_hw
        f = Fraction(max(H_T, W_T), max(H_S, W_S)) * Fraction(u)
        hp, wp = max(1, math.floor(h * f + Fraction(1, 2))), max(1, math.floor(w * f + Fraction(1, 2)))
        if hp > H_T or wp > W_T:
            g = min(Fraction(H_T, hp), Fraction(W_T, wp))
            hp, wp = min(H_T, max(1, math.floor(hp * g + Fraction(1, 2)))), min(W_T, max(1, math.floor(wp * g + Fraction(1, 2))))
        return hp, wp
    return patched((L.tac, "patch_size", patch_size))


def m_rerun_after_divergence():
    orig = L.lta.gate_attempt

    def gate_attempt(args, src, n):
        try:
            return orig(args, src, n)
        except L.lta.PreflightError as e:
            if e.code != "arm_rerun_after_divergence":
                raise
            return {"previous_record": {"status": "diverged"}}
    return patched((L.lta, "gate_attempt", gate_attempt))


def m_refuse_without_measurement():
    orig = L.pa.cut_arithmetic

    def cut(ss, iterations, figures, schedule_in_force, now):
        ar = orig(ss, iterations, figures, schedule_in_force, now)
        return dict(ar, cut=True, text="no measurement: cut") if not figures else ar
    return patched((L.pa, "cut_arithmetic", cut))


for _name, _check, _make in (
        ("round instead of ceil", "cm_repeat_table_exact", m_round_not_ceil),
        ("background as a class", "cm_no_disease_image_appears_once", m_background_as_class),
        ("paste after M2/M3", "cm_paste_draw_order", m_paste_after_m2m3),
        ("alpha blend", "cm_overwrite_no_blend_label_c", m_alpha_blend),
        ("overlap 10%", "cm_overlap_5pct_rule", m_overlap_10),
        ("momentum 0.9998", "cm_ema_decay_closed_form", lambda: ema_variant(momentum=0.9998)),
        ("update_buffers False", "cm_ema_bn_buffers_averaged", lambda: ema_variant(update_buffers=False)),
        ("drop_path 0.1", "cm_r1_diff_exact", m_drop_path_01),
        ("builder skips the strata check", "cm_table_builder_counts_vs_strata", m_builder_skips_strata),
        ("VAL pipeline changed", "cm_r2_val_test_unchanged", m_val_pipeline_changed),
        (">= instead of > in the go rule", "cm_go_rule_conditions", m_ge_not_gt),
        ("eleven tries", "cm_ten_tries_then_nothing", m_eleven_tries),
        ("raw weights loaded", "cm_kd_loader_reads_state_dict", m_raw_weights_loaded),
        ("an inherited env value accepted", "cm_launcher_env_resolution", m_inherited_env_accepted),
        ("the frozen load_launch_config used on the arm config", "rf_launch_config_mismatch",
         m_frozen_load_launch_config),
        ("the token ignored", "rf_teacher_launch_without_recheck", m_token_ignored),
        ("an empty-patch special case", "cm_empty_patch_draws_one_placement", m_empty_patch_special_case),
        ("the EXIF frame ignored", "cm_paste_exif_rotated_frame", m_exif_frame_ignored),
        ("a two-label image tolerated", "rf_table_two_labels_stop", m_two_label_tolerated),
        ("rows read at HEAD", "cm_rows_read_from_records_commit", m_rows_at_head),
        ("table_path compared by value", "cm_canonical_diff_table_path_by_variable", m_table_path_by_value),
        ("half-up rounding", "cm_patch_resize_rule", m_half_up),
        ("a rerun after divergence allowed", "rf_teacher_arm_rerun_after_divergence", m_rerun_after_divergence),
        ("a refusal when there is no measurement", "cm_teacher_arith_no_measurement_not_cut",
         m_refuse_without_measurement)):
    mutation(_name, _check, _make)


# =========================================================================================== the runner
def run_case(name):
    t0 = time.time()
    try:
        ok, detail = CASES[name]()
    except Exception as e:  # noqa: BLE001 -- a crash is a FAIL
        ok, detail = False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"
    return bool(ok), str(detail), time.time() - t0


def main(argv=None) -> int:
    global W
    p = argparse.ArgumentParser(description="Teacher-arm smoke (L-TEACHER-ARMS C4).")
    p.add_argument("--work-dir", default=None)
    args = p.parse_args(argv)
    problem = teacher_stack_problem()
    if problem:
        print(f"RESULT: NOT RUN (teacher stack missing: {problem})")
        return 2
    if args.work_dir is None:
        base, n = Path(tempfile.gettempdir()), 1
        while (base / f"teacher_arms_smoke_{os.getpid()}_{n}").exists():
            n += 1
        args.work_dir = str(base / f"teacher_arms_smoke_{os.getpid()}_{n}")
    if "test" in args.work_dir.lower():
        print(f"REFUSED: --work-dir {args.work_dir!r} contains 'test' (SL-1)")
        print("RESULT: REFUSED (test_path)")
        return 2
    W = Path(args.work_dir).resolve()
    if W == REPO or REPO in W.parents or (W.exists() and any(W.iterdir())):
        print(f"REFUSED: --work-dir {W} must be a fresh directory outside the repository")
        return 2
    W.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(W / "scratch")
    (W / "scratch").mkdir()
    libs()
    print("=" * 100)
    print("TEACHER ARMS SMOKE (L-TEACHER-ARMS C4) -- CPU, synthetic roots, a scratch git repository, tiny models")
    print(f"work dir: {W}")
    print("=" * 100, flush=True)
    t0 = time.time()
    fx("root")
    print(f"fixtures: root, strata and table in {time.time() - t0:.1f}s", flush=True)
    results = []
    for name in ORDER:
        ok, detail, secs = run_case(name)
        results.append((name, ok))
        print(f"  {name:<52}: {'PASS' if ok else 'FAIL'} ({secs:.1f}s) {detail[:400]}", flush=True)
    print("-" * 100)
    for mname, check, make in MUTATIONS:
        t1 = time.time()
        try:
            with make():
                ok, detail, _ = run_case(check)
            killed = not ok
        except Exception as e:  # noqa: BLE001
            killed, detail = False, f"the mutation could not be applied: {type(e).__name__}: {e}"
        results.append((f"mut[{mname}]", killed))
        print(f"  mutation {mname!r:<58} -> {check}: {'FAIL (killed)' if killed else 'PASS (not killed)'} "
              f"({time.time() - t1:.1f}s)", flush=True)
    n_ok = sum(ok for _, ok in results)
    print("=" * 100)
    print(f"checks {len(ORDER)}, mutations {len(MUTATIONS)}, total {len(results)}")
    bad = [n for n, ok in results if not ok]
    if bad:
        print(f"FAILED: {bad}")
    print(f"RESULT: {'PASS' if n_ok == len(results) else 'FAIL'} ({n_ok}/{len(results)})")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
