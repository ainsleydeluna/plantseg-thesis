#!/usr/bin/env python3
"""S4 — the corrupted-condition evaluator (L-CORR B-3). Torch side; synthetic; no PlantSeg data.

Fixture: a TRAIN/VAL-only data root (5,367 TRAIN and 846 VAL names, counted only; the six VAL images
that `--max-images 6` selects are real special images with real masks: EXIF 6 and 8, modes L and
CMYK, a portrait, one below the 32-px floor, one without disease ground truth); a capped smoke cache
built by scripts/build_corruption_cache.py in a subprocess, so brightness never runs in this torch
process; a provisional copy verified by scripts/verify_corruption_cache.py in a subprocess;
manifest-only forged copies (re-sealed) for the official and mismatch gates; a seeded student
checkpoint; an E4 INT8 artifact and its provenance on QNNPACK; a teacher checkpoint and an injected
builder whose head holds one isolated NMF module; two throwaway git repositories, one clean and
one with an untracked governed path, for the runner's official gates. No file or folder this smoke
creates has "test" in its name (the work folder is named before it is created). The refusals of a
TEST-named path run on never-created paths; the split, name and policy gates refuse before any path
is read, while the official cache gates read the forged synthetic manifests under the work folder
(never TEST data). src.eval.adapters.list_split_stems is patched for the whole smoke to fail if any
split but val were listed, and PLANTSEG_IMAGE_DIGEST is unset for every runner case that does not
set it.

  W  the wrapper and the cell (src/eval/corrupted.py)
  E  runs of scripts/evaluate_corruptions.py, in-process (fog s3 on the 3-image subset 0, 282, 564)
  X  LE1: scoring never generates (a call tracer)
  T  the teacher (M4-V; LE5, LE11)
  ST stages, sources and identity (LE2, LE15); the artifact schema's descriptive stages (LE23);
     the official INT8 gate, int8_official_error, under R (LE22)
  G  the section 7 guards (G4 dropped: there is no random-init path, LE12) and the out-dir names
     (TL section 2 over every component, as given and resolved: N1, N1b, N1c)
  P  LE2's printed line
  C  the cache gates (LE3)
  R  the grid runner scripts/run_corruption_grid.py (LE2, LE3, LE10; IF section 1: physical
     cores, the INT8 interim rule, the teacher slot; the DL-24 start gates, lock and STOPs)
  M  mutations: every new refusal gate disabled one code at a time, and the plan's eight; each is
     killed by its named case. A case that raised is no kill (its checks never ran); a refusal case
     that sees an outcome other than its code has run its checks.

Run:  python -B scripts/smoke_corrupted_eval.py
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import itertools
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

import scripts.evaluate_corruptions as EC  # noqa: E402
import scripts.evaluate_model as EM  # noqa: E402
import scripts.run_corruption_grid as RG  # noqa: E402
import src.corruption_cache as cc  # noqa: E402
import src.eval.adapters as AD  # noqa: E402
import src.eval.corrupted as CO  # noqa: E402
import src.eval.eval_runtime as ER  # noqa: E402
import src.eval.stage_artifacts as SA  # noqa: E402
from configs.data import DATA  # noqa: E402
from src import corruption_roles  # noqa: E402
from src.data.transforms import IMAGENET_MEAN_8BIT, SIZE, core_preprocess, finalize  # noqa: E402
from src.eval.artifacts import ARTIFACT_SCHEMA_VERSION, verify_artifact  # noqa: E402
from src.eval.artifacts import STAGES as ARTIFACT_STAGES  # noqa: E402
from src.eval.eval_runtime import RECORD_KEYS  # noqa: E402
from src.eval.metrics import per_image_miou  # noqa: E402
from src.models.student import build_student  # noqa: E402

NC = 116
CASES: dict = {}
S: dict = {}
results: list = []
_SEQ = itertools.count()
TEACHER_CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
#: The six --max-images 6 images: stored (w, h), JPEG mode, EXIF orientation.
SPECIALS = {0: ((60, 400), "RGB", 6),       # EXIF 6: displayed 400 x 60 -> 512 x 77
            141: ((400, 50), "L", None),    # grayscale JPEG -> 512 x 64
            282: ((300, 40), "CMYK", None),  # CMYK JPEG -> 512 x 68; no disease ground truth (AM-5)
            423: ((500, 29), "RGB", None),  # 29.7 -> 30: below the 32-px floor (LC7)
            564: ((333, 500), "RGB", None),  # portrait: 341 x 512
            705: ((64, 512), "RGB", 8)}     # EXIF 8: displayed 512 x 64
ZERO_DISEASE = 282
SUBSET = [0, 282, 564]                      # deterministic_subset(846, 3)
VERIFICATION_SCHEMA = EC.VERIFICATION_SCHEMA


# ---------------------------------------------------------------- harness
def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def run(name) -> tuple[bool, str]:
    try:
        ok, detail = CASES[name]()
        return bool(ok), str(detail)
    except Exception as e:                                   # noqa: BLE001
        return False, f"raised {type(e).__name__}: {e}"


@contextlib.contextmanager
def patched(obj, attr, value):
    saved = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, saved)


@contextlib.contextmanager
def env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def mutant(label, targets, *patches):
    """Killed only when every named case fails by its own checks: a case that raised is no kill."""
    with contextlib.ExitStack() as stack:
        for obj, attr, value in patches:
            stack.enter_context(patched(obj, attr, value))
        outcomes = {t: run(t) for t in targets}
    killed = all(not ok and not d.startswith("raised ") for ok, d in outcomes.values())
    why = "; ".join(f"{t}: {d[:90]}" for t, (_, d) in outcomes.items())
    print(f"MUTANT {label} {'killed' if killed else 'SURVIVED'} by {', '.join(targets)} [{why}]")
    results.append((f"{label} killed by {', '.join(targets)}", killed, why))


def skip(mod, helper: str, code: str):
    """A patch that disables one refusal code at its raising helper (the gate removed)."""
    real = getattr(mod, helper)

    def gate(c, *a, **k):
        if c == code:
            return None
        return real(c, *a, **k)
    return (mod, helper, gate)


def outcome(fn, *a, **kw) -> tuple[str, object]:
    """('ok', value), or (the refusal code, else the exception type name, exception)."""
    try:
        return "ok", fn(*a, **kw)
    except Exception as e:                                   # noqa: BLE001 -- recorded
        return (getattr(e, "code", None) or type(e).__name__), e


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fresh(tag: str) -> Path:
    """A never-yet-created path under the work folder (no 'test' in it)."""
    return S["work"] / "out" / f"{tag}_{next(_SEQ)}"


def never(tag: str) -> Path:
    return S["work"] / "never_created" / f"{tag}_{next(_SEQ)}"


# ---------------------------------------------------------------- fixtures
def make_val_root(root: Path) -> None:
    rs = np.random.RandomState(20261017)
    for split, n in (("train", 5367), ("val", 846)):
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i in range(n):
            if split == "val" and i in SPECIALS:
                continue
            (root / "images" / split / f"syn_{split}_{i:05d}.jpg").touch()
            (root / "annotations" / split / f"syn_{split}_{i:05d}.png").touch()
    for i, ((w, h), mode, orient) in SPECIALS.items():
        im = Image.fromarray((rs.rand(h, w, 3) * 255).astype(np.uint8))
        if mode != "RGB":
            im = im.convert(mode)
        kw = {"quality": 95}
        if orient is not None:
            exif = Image.Exif()
            exif[0x0112] = orient
            kw["exif"] = exif
        im.save(root / "images" / "val" / f"syn_val_{i:05d}.jpg", format="JPEG", **kw)
        dw, dh = (h, w) if orient in (6, 8) else (w, h)       # the displayed size
        mask = np.zeros((dh, dw), np.uint8)
        if i != ZERO_DISEASE:
            mask[dh // 4: max(dh // 4 + 1, 3 * dh // 4), dw // 5: max(dw // 5 + 1, 4 * dw // 5)] = \
                1 + i % 115
        Image.fromarray(mask).save(root / "annotations" / "val" / f"syn_val_{i:05d}.png")


def child_env(**extra) -> dict:
    e = {k: v for k, v in os.environ.items()
         if k not in ("PLANTSEG_DATA_ROOT", "PLANTSEG_IMAGE_DIGEST")}
    e.update(PYTHONDONTWRITEBYTECODE="1", **extra)
    return e


def build_smoke_cache(root: Path) -> dict:
    p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "build_corruption_cache.py"),
                        "--real-run", "--confirm-real-run", "--split", "val", "--cache-status",
                        "smoke", "--data-root", str(S["val"]), "--cache-dir", str(root),
                        "--max-images", "6"], capture_output=True, text=True, cwd=str(REPO),
                       env=child_env(), timeout=900)
    if p.returncode != 0:
        raise AssertionError(f"cache build failed ({p.returncode}): {p.stderr[-400:]}")
    return json.loads((root / cc.MANIFEST_NAME).read_text(encoding="utf-8"))


def reseal(m: dict) -> dict:
    m["canonical_manifest_sha256"] = cc.canonical_manifest_sha256(m)
    return m


def write_manifest(root: Path, m: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / cc.MANIFEST_NAME).write_bytes(cc.manifest_bytes(m))


def verify_cache(root: Path) -> dict:
    """scripts/verify_corruption_cache.py in a subprocess; its four labelled lines and the record."""
    p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "verify_corruption_cache.py"),
                        "--cache-dir", str(root), "--data-root", str(S["val"]), "--split", "val"],
                       capture_output=True, text=True, cwd=str(REPO),
                       env=child_env(PLANTSEG_IMAGE_DIGEST=cc.STUDENT_IMAGE_DIGEST), timeout=900)
    if p.returncode != 0:
        raise AssertionError(f"verification failed ({p.returncode}): {p.stderr[-400:]}")
    values = dict(ln.split(": ", 1) for ln in p.stdout.strip().splitlines())
    records = sorted((root / "verification").iterdir())
    return {"values": values, "record": records[-1]}


def forged(tag: str, mutate, *, base=None) -> tuple[Path, dict]:
    """A manifest-only copy of a cache manifest, mutated and re-sealed (its items are not copied:
    every gate it serves refuses before an item is read)."""
    m = json.loads(json.dumps(base if base is not None else S["m_smoke"]))
    mutate(m)
    reseal(m)
    root = S["work"] / "forged" / f"{tag}_{next(_SEQ)}"
    write_manifest(root, m)
    return root, m


def forged_items(tag: str, mutate) -> tuple[Path, dict]:
    """A full copy of the smoke cache (items included), its manifest mutated and re-sealed: past a
    removed gate a run would read real items and finish."""
    root = S["work"] / "forged" / f"{tag}_{next(_SEQ)}"
    shutil.copytree(S["c_smoke"], root)
    m = json.loads(json.dumps(S["m_smoke"]))
    mutate(m)
    write_manifest(root, reseal(m))
    return root, m


def as_tday(m: dict, **over) -> None:
    """The forged official TEST-like manifest: split test (data only), official, 1,561 sources."""
    m["split"] = "test"
    m["dataset"]["split"] = "test"
    m["pin"]["parameters"]["split"] = "test"
    m["status"] = "official"
    m["source_image_count"] = 1561
    m.update(over)


def record_for(root: Path, m: dict, **over) -> Path:
    rec = {"schema": VERIFICATION_SCHEMA, "canonical_manifest_sha256": m["canonical_manifest_sha256"],
           "inferential_cells_sha256": m["inferential_cells_sha256"],
           "manifest_file_sha256": sha256_file(root / cc.MANIFEST_NAME), "split": m["split"],
           "status": m["status"], "pin_sha256": (m.get("pin") or {}).get("pin_sha256"),
           "counts": {"sources_verified": m["source_image_count"],
                      "items_verified": len(m["entries"]), "cells": len(m["cells"])}}
    rec.update(over)
    path = root.parent / f"{root.name}_record_{next(_SEQ)}.json"
    path.write_bytes(cc.canonical_json_bytes(rec) + b"\n")
    return path


def save_student(path: Path, *, stage=None, seed=0) -> Path:
    torch.manual_seed(seed)
    payload = {"iter": 7, "model_state_dict": build_student(pretrained=False).state_dict(),
               "num_classes": NC, "best_val_miou_all_class": 0.25}
    if stage is not None:
        payload["stage"] = stage
    torch.save(payload, path)
    return path


def make_int8(work: Path) -> tuple[Path, Path]:
    from src.quant.prepare import convert_model, prepare_ptq
    torch.backends.quantized.engine = "qnnpack"
    torch.manual_seed(42)
    converted = convert_model(prepare_ptq(build_student(pretrained=False), select_backend=False))
    art = work / "e4_int8.pt"
    torch.save({"stage": "E4", "quantization": "ptq", "num_classes": NC,
                "model": converted.state_dict()}, art)
    prov = work / "e4_provenance.json"
    prov.write_text(json.dumps({"stage": "E4", "source_stage": "E1", "quantization_method": "ptq",
                                "random_init": False, "num_classes": NC, "backend": "qnnpack",
                                "converted_artifact": str(art),
                                "converted_artifact_sha256": sha256_file(art)}, indent=2),
                    encoding="utf-8")
    return art, prov


class StubMSCAN(nn.Module):
    def __init__(self, dims=(64, 128, 320, 512), strides=(4, 8, 16, 32)):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(3, d, 1, bias=False) for d in dims])
        self.strides = strides

    def forward(self, x):
        return tuple(p(torch.nn.functional.adaptive_avg_pool2d(
            x, (max(x.shape[-2] // s, 1), max(x.shape[-1] // s, 1))))
            for p, s in zip(self.projs, self.strides))


class StubIsolatedNMF(nn.Module):
    """Stand-in for IsolatedNMF2D: `draws` torch.rand basis draws per forward, through the stream."""
    SUPPORTS_NMF_STREAM = True

    def __init__(self, draws: int = 1):
        super().__init__()
        self.nmf_stream = None
        self.draws = draws

    def forward(self, x):
        bases = None
        for _ in range(self.draws):
            def draw():
                return torch.rand(x.shape[0], x.shape[1], 1, 1)
            bases = draw() if self.nmf_stream is None else self.nmf_stream.draw(draw)
        return x * bases


class StubHead(nn.Module):
    def __init__(self, draws: int):
        super().__init__()
        self.num_classes = NC
        self.squeeze = nn.Conv2d(128, 512, 1, bias=False)
        self.nmf = StubIsolatedNMF(draws)
        self.conv_seg = nn.Conv2d(512, NC, 1)

    def forward(self, feats):
        return self.conv_seg(self.nmf(self.squeeze(feats[1])))


class StubSegNeXt(nn.Module):
    def __init__(self, draws: int):
        super().__init__()
        torch.manual_seed(3)
        self.backbone = StubMSCAN()
        self.decode_head = StubHead(draws)

    def extract_feat(self, x):
        return self.backbone(x)


def teacher_builder(built: list, draws: int = 1):
    def build(_path):
        from src.distill.segnext_teacher import SegNeXtTeacherAdapter
        adapter = SegNeXtTeacherAdapter(StubSegNeXt(draws))
        built.append(adapter)
        return adapter
    return build


def make_teacher_ckpt(path: Path) -> Path:
    torch.manual_seed(5)
    torch.save({"meta": {"mmseg_version": "1.2.2"},
                "state_dict": {"backbone.projs.0.weight": torch.randn(4, 3, 1, 1),
                               "decode_head.conv_seg.weight": torch.randn(NC, 8, 1, 1),
                               "decode_head.conv_seg.bias": torch.randn(NC)}}, path)
    return path


def setup(work: Path) -> None:
    S["work"] = work
    S["val"] = work / "val_root"
    make_val_root(S["val"])
    S["saved_root"] = DATA["root"]
    DATA["root"] = str(S["val"])
    S["real_list_stems"] = AD.list_split_stems

    def val_only_listing(split):
        if split != "val":
            raise AssertionError(f"a {split!r} listing was reached")
        return S["real_list_stems"](split)
    AD.list_split_stems = val_only_listing
    caches = work / "caches"
    caches.mkdir()
    S["c_smoke"] = caches / "smoke6"
    S["m_smoke"] = build_smoke_cache(S["c_smoke"])
    S["c_prov"] = caches / "prov6"
    shutil.copytree(S["c_smoke"], S["c_prov"])
    m = json.loads((S["c_prov"] / cc.MANIFEST_NAME).read_text(encoding="utf-8"))
    m["status"] = "provisional"
    write_manifest(S["c_prov"], reseal(m))
    S["m_prov"] = m
    S["v_prov"] = verify_cache(S["c_prov"])
    S["ckpt"] = save_student(work / "e1.pt")
    S["ckpt_sha"] = sha256_file(S["ckpt"])
    S["ckpt_e3"] = save_student(work / "e3_nostage.pt", seed=1)
    S["int8_art"], S["int8_prov"] = make_int8(work)
    S["tk"] = make_teacher_ckpt(work / "teacher.pth")
    S["tk_sha"] = sha256_file(S["tk"])
    S["head"] = RG.git_head()
    S["inputs"] = work / "inputs"
    S["inputs"].mkdir()
    shutil.copy2(S["ckpt"], S["inputs"] / "e1.pt")
    shutil.copy2(S["tk"], S["inputs"] / "t.pth")
    shutil.copy2(S["int8_prov"], S["inputs"] / "e4.json")
    shutil.copy2(S["int8_prov"], S["inputs"] / "e7.json")
    S["repo_clean"] = make_fixture_repo(work / "fixture_repo_clean")
    S["repo_dirty"] = make_fixture_repo(work / "fixture_repo_dirty", governed_dirty=True)
    S["stems"] = {i: f"syn_val_{i:05d}" for i in range(846)}
    # the clean passes the corrupted ones are compared with, built once before any mutant runs
    S["clean_e1"] = em_clean(work / "out" / "clean_e1")
    built: list = []
    S["clean_t"] = em_clean(work / "out" / "clean_teacher", teacher=teacher_builder(built))
    S["clean_t_draws"] = built[-1].nmf_stream.describe()["draws"]


# ---------------------------------------------------------------- evaluator helpers
def ec_argv(out, *, stage="E1", corruption="fog", severity=3, cache=None, split="val",
            status="smoke", max_samples=3, batch=1, **extra) -> list[str]:
    argv = ["--stage", stage, "--corruption", corruption, "--severity", str(severity),
            "--out-dir", str(out), "--corruption-cache", str(cache or S["c_smoke"]),
            "--split", split, "--artifact-status", status, "--batch-size", str(batch)]
    if max_samples is not None:
        argv += ["--max-samples", str(max_samples)]
    if stage in ("E1", "E2", "E3") and "checkpoint" not in extra and "provenance" not in extra:
        extra["checkpoint"] = S["ckpt"]
    if stage == "E4" and "provenance" not in extra and "checkpoint" not in extra:
        extra["provenance"] = S["int8_prov"]
    if stage == "teacher":
        extra.setdefault("checkpoint", S["tk"])
        extra.setdefault("teacher_config", TEACHER_CONFIG)
        extra.setdefault("teacher_ckpt_sha256", S["tk_sha"])
    for k, v in extra.items():
        if v is True:
            argv.append(f"--{k.replace('_', '-')}")
        elif v not in (None, False):
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def ec_args(out, **kw):
    return EC.build_parser().parse_args(ec_argv(out, **kw))


def ec_run(args, stdout=None, **kw):
    """EC.run in this process; its printed lines go to `stdout` (a list) when one is given."""
    threads = torch.get_num_threads()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return EC.run(args, **kw)
    finally:
        torch.set_num_threads(threads)
        if stdout is not None:
            stdout.append(buf.getvalue())


def ec_outcome(args, **kw):
    return outcome(ec_run, args, **kw)


def official_argv(out, cache, **extra) -> list:
    return dict(split="test", status="official", max_samples=None, confirm_test_split=True,
                expected_checkpoint_sha256=S["ckpt_sha"], threads=1, cache=cache, **extra)


def rows_of(out) -> list[dict]:
    return [json.loads(ln) for ln in (Path(out) / "per_image.jsonl").read_text("utf-8").splitlines()]


def summary_of(out) -> dict:
    return json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))


def em_clean(out, *, teacher=None) -> Path:
    if teacher is None:
        argv = ["--stage", "E1", "--split", "val", "--checkpoint", str(S["ckpt"]),
                "--artifact-status", "smoke", "--max-samples", "3", "--batch-size", "1",
                "--out-dir", str(out)]
    else:
        argv = ["--stage", "teacher", "--model-role", "teacher", "--split", "val", "--checkpoint",
                str(S["tk"]), "--teacher-config", str(TEACHER_CONFIG), "--teacher-ckpt-sha256",
                S["tk_sha"], "--artifact-status", "smoke", "--max-samples", "3", "--batch-size",
                "1", "--out-dir", str(out)]
    with contextlib.redirect_stdout(io.StringIO()):
        return EM.run(EM.build_parser().parse_args(argv), teacher_builder=teacher)


class StubCell:
    """A cell whose source and item the case supplies; records every load."""

    def __init__(self, source, load):
        self._source, self._load, self.loads = source, load, []

    def source(self, stem):
        return self._source(stem)

    def load(self, stem):
        self.loads.append(stem)
        return self._load(stem)


def clean_window(i: int):
    """The clean canvas, mask canvas and resized window of special image i (real core_preprocess)."""
    img = S["val"] / "images" / "val" / f"syn_val_{i:05d}.jpg"
    msk = S["val"] / "annotations" / "val" / f"syn_val_{i:05d}.png"
    with Image.open(img) as im, Image.open(msk) as mk:
        canvas, mcanvas = core_preprocess(im, mk)
    rec = next(s for s in S["m_smoke"]["sources"] if s["manifest_index"] == i)
    top, left, h, w = CO.window(rec["resized_shape"])
    return canvas, mcanvas, rec, canvas[top:top + h, left:left + w].copy()


def real_cell(corruption="fog", severity=3, cache=None, manifest=None):
    root = cache or S["c_smoke"]
    m = manifest or S["m_smoke"]
    return CO.CacheCell(Path(root).resolve(), m, corruption, severity,
                        load=EC.cached_image_loader(root, m, corruption, severity))


# ---------------------------------------------------------------- W. the wrapper and the cell
@case("W1")
def w1():
    """Identity: a cell whose item is the clean window gives (image, target) equal to
    finalize(core_preprocess) bitwise, for every special image."""
    bad = []
    for i in SPECIALS:
        canvas, mcanvas, rec, win = clean_window(i)
        cell = StubCell(lambda stem, rec=rec: rec, lambda stem, win=win: win.copy())
        ds = CO.PlantSegCorruptedEvalDataset("val", [i], cell)
        got = ds[0]
        want_img, want_tgt = finalize(canvas, mcanvas)
        if not (torch.equal(got["image"], want_img) and torch.equal(got["target"], want_tgt)):
            bad.append(i)
    return not bad, f"bitwise unequal at {bad}" if bad else "6 special images bitwise equal"


@case("W2")
def w2():
    """The target is the clean path's: equal to PlantSegEvalDataset's for every special image."""
    clean = AD.PlantSegEvalDataset("val", list(SPECIALS))
    ds = CO.PlantSegCorruptedEvalDataset("val", list(SPECIALS), real_cell())
    bad = [i for k, i in enumerate(sorted(SPECIALS))
           if not torch.equal(ds[k]["target"], clean[k]["target"])]
    return not bad, f"targets differ at {bad}" if bad else "6 targets equal"


@case("W3")
def w3():
    """The padding bytes are the clean canvas's and the window holds the cached item."""
    bad = []
    cell = real_cell()
    for i in SPECIALS:
        canvas, _, rec, _ = clean_window(i)
        stem = S["stems"][i]
        out = CO.corrupted_canvas(stem, i, canvas, cell)
        top, left, h, w = CO.window(rec["resized_shape"])
        outside = np.ones((SIZE, SIZE), bool)
        outside[top:top + h, left:left + w] = False
        item = cell.load(stem)
        if not (np.array_equal(out[outside], canvas[outside])
                and np.array_equal(out[top:top + h, left:left + w], item)):
            bad.append(i)
    return not bad, f"canvas wrong at {bad}" if bad else "6 canvases"


@case("W4")
def w4():
    """An item that is not the resized clean shape -> corrupted_shape_mismatch: one row short, and
    an (h, w, 1) item that numpy would broadcast into the window silently."""
    canvas, mcanvas, rec, win = clean_window(141)
    codes = []
    for item in (win[:-1].copy(), win[..., :1].copy()):
        cell = StubCell(lambda s: rec, lambda s, item=item: item)
        codes.append(outcome(CO.corrupted_canvas, S["stems"][141], 141, canvas, cell)[0])
    return codes == ["corrupted_shape_mismatch"] * 2, str(codes)


@case("W5")
def w5():
    """A source digest that is not the clean window -> source_pixels_mismatch; the item is never
    read."""
    canvas, mcanvas, rec, win = clean_window(141)
    forged_rec = {**rec, "resized_pixel_sha256": "0" * 64}
    cell = StubCell(lambda s: forged_rec, lambda s: win.copy())
    code, _ = outcome(CO.corrupted_canvas, S["stems"][141], 141, canvas, cell)
    return code == "source_pixels_mismatch" and cell.loads == [], f"{code}, loads={cell.loads}"


@case("W6")
def w6():
    """Identity fields: image_id == clean_image_id == the stem, manifest_index the listing's."""
    ds = CO.PlantSegCorruptedEvalDataset("val", SUBSET, real_cell())
    got = [(ds[k]["image_id"], ds[k]["clean_image_id"], ds[k]["manifest_index"])
           for k in range(len(SUBSET))]
    want = [(S["stems"][i], S["stems"][i], i) for i in SUBSET]
    return got == want, str(got)


@case("W7")
def w7():
    """An absent cell -> cache_cell_absent, at the cache gate and at the cell."""
    root, m = forged("w7", lambda m: m.update(cells=[c for c in m["cells"]
                                                     if c["cell_id"] != "fog_s3"]))
    code_gate, _ = outcome(EC.resolve_corruption_cache, root, "fog", 3, split="val",
                           official=False)
    code_cell, _ = outcome(CO.CacheCell(root, m, "fog", 3).entry)
    return (code_gate == "cache_cell_absent" and code_cell == "cache_cell_absent"), \
        f"gate {code_gate}, cell {code_cell}"


@case("W8")
def w8():
    """A tampered item is refused at scoring and never regenerated."""
    root = S["work"] / "caches" / f"w8_{next(_SEQ)}"
    shutil.copytree(S["c_smoke"], root)
    e = next(x for x in S["m_smoke"]["entries"]
             if x["image_id"] == S["stems"][0] and x["corruption_id"] == "fog" and x["severity"] == 3)
    p = root / e["cache_relative_path"]
    np.save(p, (np.load(p) + 3).astype(np.uint8), allow_pickle=False)
    with tracer() as calls:
        ds = CO.PlantSegCorruptedEvalDataset("val", [0], real_cell(cache=root))
        code, _ = outcome(lambda: ds[0])
    return code == "cache_file_hash_mismatch" and calls["generation"] == 0, \
        f"{code}, generation calls {calls['generation']}"


@case("W9")
def w9():
    """A cached source whose manifest_index is not the listing's -> cache_source_index_mismatch."""
    canvas, _, rec, win = clean_window(141)
    cell = StubCell(lambda s: {**rec, "manifest_index": 142}, lambda s: win.copy())
    code, _ = outcome(CO.corrupted_canvas, S["stems"][141], 141, canvas, cell)
    return code == "cache_source_index_mismatch" and cell.loads == [], f"{code}, {cell.loads}"


@case("W10")
def w10():
    """A recorded window that is not the image's -> canvas_window_mismatch: a window two rows short
    whose digest matches (image pixels outside it), and a window larger than the canvas."""
    canvas, _, rec, win = clean_window(0)
    h, w = rec["resized_shape"]
    top, left, _, _ = CO.window([h - 2, w])
    short = canvas[top:top + h - 2, left:left + w]
    rec_short = {**rec, "resized_shape": [h - 2, w],
                 "resized_pixel_sha256": hashlib.sha256(np.ascontiguousarray(short).tobytes())
                 .hexdigest()}
    cell = StubCell(lambda s: rec_short, lambda s: short.copy())
    code_short, _ = outcome(CO.corrupted_canvas, S["stems"][0], 0, canvas, cell)
    cell_big = StubCell(lambda s: {**rec, "resized_shape": [SIZE + 2, w]}, lambda s: win)
    code_big, _ = outcome(CO.corrupted_canvas, S["stems"][0], 0, canvas, cell_big)
    return code_short == code_big == "canvas_window_mismatch", f"{code_short}, {code_big}"


@case("W11")
def w11():
    """A stem the cache does not record -> cache_source_absent."""
    code, _ = outcome(real_cell().source, "syn_val_99999")
    return code == "cache_source_absent", code


@case("W12")
def w12():
    """A record write that fails (fsync raises) leaves no record and no temporary file behind;
    the error propagates (EV-5)."""
    path = fresh("w12").with_name(f"w12_{next(_SEQ)}{CO.RECORD_SUFFIX}")
    path.parent.mkdir(parents=True, exist_ok=True)

    def no_fsync(fd):
        raise OSError(28, "No space left on device (simulated)")
    with patched(os, "fsync", no_fsync):
        code, _ = outcome(CO.write_cell_record, path, {"schema": CO.CELL_RECORD_SCHEMA})
    left = [n for n in os.listdir(path.parent) if n.startswith(f".{path.name}.tmp-")]
    return code == "OSError" and not path.exists() and left == [], f"{code}, leftovers {left}"


# ---------------------------------------------------------------- tracer (LE1)
@contextlib.contextmanager
def tracer():
    """Counts every generation call (the vendored corruptions, the cache generators) and every
    verified cache read during the block."""
    import src.eval.robustness as RB
    import src.vendor.imagecorruptions as V
    import src.vendor.imagecorruptions.corruptions as VC
    calls = {"generation": 0, "reads": 0}

    def count(kind, fn):
        def wrapper(*a, **k):
            calls[kind] += 1
            return fn(*a, **k)
        return wrapper
    targets = [(V, "apply_corruption"), (cc, "apply_corruption"), (cc, "generate"),
               (cc, "corrupt_source"), (cc, "write_entry"), (cc, "build_cache"),
               (RB, "real_corruption_transform")]
    targets += [(VC, name) for name in corruption_roles.CANONICAL_IDS if hasattr(VC, name)]
    with contextlib.ExitStack() as stack:
        for obj, attr in targets:
            stack.enter_context(patched(obj, attr, count("generation", getattr(obj, attr))))
        stack.enter_context(patched(cc, "load_cached", count("reads", cc.load_cached)))
        yield calls


@case("X1")
def x1():
    """LE1: scoring never generates. A full E1 cell calls none of the vendored corruptions or the
    cache generators, and reads each of its rows from the cache once."""
    with tracer() as calls:
        code, out = ec_outcome(ec_args(fresh("x1")))
    return (code == "ok" and calls["generation"] == 0 and calls["reads"] == len(SUBSET)), \
        f"{code}; {calls}"


# ---------------------------------------------------------------- E. CLI runs
@case("E1")
def e1():
    """A smoke-status run (val, --max-samples 3, fog s3) writes a verifying artifact and its
    record."""
    code, out = ec_outcome(ec_args(fresh("e1")))
    if code != "ok":
        return False, code
    summ = verify_artifact(out)
    rec = CO.read_cell_record(CO.record_path(out))
    CO.verify_cell_record(out, rec)
    return (summ["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
            and summ["dataset"]["condition"] == {"type": "corruption", "name": "fog", "severity": 3}
            and summ["dataset"]["preprocess_protocol"] == "core_preprocess/1.0.0"
            and "protocol" not in summ and summ["dataset"]["actual_rows"] == 3), \
        f"{out.name}: {summ['dataset']['condition']}"


@case("E2")
def e2():
    """Bare stems and the condition in every row (DL-74)."""
    code, out = ec_outcome(ec_args(fresh("e2")))
    if code != "ok":
        return False, code
    rows = rows_of(out)
    ok = [(r["image_id"], r["clean_image_id"], r["manifest_index"]) for r in rows] == \
        [(S["stems"][i], S["stems"][i], i) for i in SUBSET]
    cond = all(r["condition"] == {"type": "corruption", "name": "fog", "severity": 3} for r in rows)
    return ok and cond, str([r["image_id"] for r in rows])


@case("E3")
def e3():
    """The split digest equals the clean runner's on the same subset, and the bare-stem digest of
    the listing (EVAL section 5.3)."""
    from src.eval.artifacts import hash_split_manifest
    from src.eval.evaluate import ManifestEntry
    code, out = ec_outcome(ec_args(fresh("e3")))
    if code != "ok":
        return False, code
    a, b = summary_of(out)["dataset"], summary_of(S["clean_e1"])["dataset"]
    bare = hash_split_manifest([ManifestEntry(i, S["stems"][i], S["stems"][i]) for i in SUBSET])
    return a["split_manifest_sha256"] == b["split_manifest_sha256"] == bare, \
        f"{a['split_manifest_sha256'][:16]} vs {b['split_manifest_sha256'][:16]}"


@case("E4")
def e4():
    """The AM-5 flags and gt_disease_classes equal the clean artifact's (ground truth decides)."""
    code, out = ec_outcome(ec_args(fresh("e4")))
    if code != "ok":
        return False, code
    clean = S["clean_e1"]
    pick = ("am5_excluded", "gt_disease_classes", "n_eligible_disease_only")
    a = [{k: r[k] for k in pick} for r in rows_of(out)]
    b = [{k: r[k] for k in pick} for r in rows_of(clean)]
    am5 = summary_of(out)["am5"] == summary_of(clean)["am5"]
    return a == b and am5 and summary_of(out)["am5"]["excluded_count"] == 1, str(a)


@case("E5")
def e5():
    """Per-image values of two images equal a manual recomputation: the clean canvas, the cached
    item in its window, finalize, the model, src.eval.metrics.per_image_miou."""
    code, out = ec_outcome(ec_args(fresh("e5")))
    if code != "ok":
        return False, code
    from src.eval.model_loading import load_student_checkpoint
    model = load_student_checkpoint(S["ckpt"])[0]
    rows = {r["manifest_index"]: r for r in rows_of(out)}
    cell = real_cell()
    bad = []
    for i in (0, 564):
        canvas, mcanvas, rec, _ = clean_window(i)
        img, tgt = finalize(CO.corrupted_canvas(S["stems"][i], i, canvas, cell), mcanvas)
        with torch.no_grad():
            pred = model(img[None]).argmax(1)
        allc = per_image_miou(pred, tgt[None], NC, class_indices=None, ignore_index=255)[0]
        dis = per_image_miou(pred, tgt[None], NC, class_indices=torch.arange(1, NC),
                             ignore_index=255)[0]
        if not (rows[i]["all_class_miou"] == float(allc)
                and rows[i]["disease_only_miou"] == float(dis)):
            bad.append((i, rows[i]["all_class_miou"], float(allc)))
    return not bad, str(bad) if bad else "2 images recomputed exactly"


@case("E6")
def e6():
    """run.eval_runtime holds the 26 keys, in order, and workers 0."""
    code, out = ec_outcome(ec_args(fresh("e6")))
    if code != "ok":
        return False, code
    rt = summary_of(out)["run"]["eval_runtime"]
    return tuple(rt) == RECORD_KEYS and len(rt) == 26 and rt["num_workers"] == 0, \
        f"{len(rt)} keys, workers {rt.get('num_workers')}"


@case("E7")
def e7():
    """fog s5 is scored (descriptive_only); gaussian_noise s5 is refused (severity_excluded)."""
    code, out = ec_outcome(ec_args(fresh("e7"), severity=5))
    if code != "ok":
        return False, code
    rec = CO.read_cell_record(CO.record_path(out))
    code_gn, _ = ec_outcome(ec_args(fresh("e7_gn"), corruption="gaussian_noise", severity=5))
    return (rec["role"] == "descriptive_only" and rec["condition"]["severity"] == 5
            and code_gn == "severity_excluded"), f"role {rec['role']}, gaussian_noise s5: {code_gn}"


@case("E8")
def e8():
    """The cell record (TL section 7, F-17): its keys, the four hashes, the cache checksum, the cell
    digest, the role, the counts, the commit and started_utc before the artifact's time."""
    code, out = ec_outcome(ec_args(fresh("e8")))
    if code != "ok":
        return False, code
    rec = CO.read_cell_record(CO.record_path(out))
    summ = summary_of(out)
    cell = next(c for c in S["m_smoke"]["cells"] if c["cell_id"] == "fog_s3")
    four = {n: sha256_file(Path(out) / n) for n in CO.ARTIFACT_FOUR_FILES}
    checks = {
        "keys": tuple(rec) == tuple(sorted(CO.CELL_RECORD_KEYS)) or set(rec) == set(CO.CELL_RECORD_KEYS),
        "four": rec["artifact_sha256"] == four and len(four) == 4,
        "checksum": rec["cache"]["canonical_manifest_sha256"]
                    == S["m_smoke"]["canonical_manifest_sha256"],
        "cell": rec["cache"]["cell_sha256"] == cell["cell_sha256"],
        "role": rec["role"] == "inferential",
        "counts": rec["items_verified"] == rec["source_digests_verified"] == 3,
        "commit": rec["evaluator_commit"] == summ["run"]["repo_commit"],
        "started": rec["started_utc"][:19] <= summ["run"]["timestamp_utc"][:19]
                   <= rec["written_utc"][:19],
        "split": rec["cache"]["split"] == "val" and rec["cache"]["status"] == "smoke",
        "cache": (rec["cache"]["inferential_cells_sha256"]
                  == S["m_smoke"]["inferential_cells_sha256"]
                  and rec["cache"]["source_image_count"] == S["m_smoke"]["source_image_count"]),
        "identity": (rec["run_id"] == summ["run"]["run_id"] and rec["stage"] == "E1"
                     and rec["condition"] == {"type": "corruption", "name": "fog", "severity": 3}
                     and rec["artifact_dir"] == Path(out).name
                     and rec["evaluator_script"] == "scripts/evaluate_corruptions.py"),
    }
    return all(checks.values()), str({k: v for k, v in checks.items() if not v})


@case("E9")
def e9():
    """A second run into an existing out-dir -> out_dir_exists, before any input is read."""
    out = fresh("e9")
    code1, _ = ec_outcome(ec_args(out))
    code2, _ = ec_outcome(ec_args(out, severity=2))
    return code1 == "ok" and code2 == "out_dir_exists", f"{code1}, then {code2}"


@case("E10")
def e10():
    """config_sha256 differs between two cells only through the condition."""
    a_code, a = ec_outcome(ec_args(fresh("e10a"), severity=1))
    b_code, b = ec_outcome(ec_args(fresh("e10b"), severity=2))
    if (a_code, b_code) != ("ok", "ok"):
        return False, f"{a_code} {b_code}"
    from src.eval.artifacts import METRIC_PROTOCOL, SCHEMA_VERSION, canonical_json_bytes
    sums = [summary_of(a), summary_of(b)]

    def payload(s):
        r, d = s["run"], s["dataset"]
        return {"schema_version": SCHEMA_VERSION, "metric_protocol": METRIC_PROTOCOL,
                "stage": r["stage"], "model_role": r["model_role"], "precision": r["precision"],
                "quant_backend": r["quant_backend"], "random_init": r["random_init"],
                "device": r["env"]["device"], "dataset_name": d["name"], "dataset_doi": d["doi"],
                "split": d["split"], "condition": d["condition"],
                "preprocess_protocol": d["preprocess_protocol"], "expected_rows": d["expected_rows"],
                "num_classes": d["num_classes"], "background_index": d["background_index"],
                "ignore_index": d["ignore_index"]}
    pa, pb = payload(sums[0]), payload(sums[1])
    recompute = all(hashlib.sha256(canonical_json_bytes(p)).hexdigest() == s["run"]["config_sha256"]
                    for p, s in zip((pa, pb), sums))
    diff = sorted(k for k in pa if pa[k] != pb[k])
    return recompute and diff == ["condition"] and \
        sums[0]["run"]["config_sha256"] != sums[1]["run"]["config_sha256"], f"differ in {diff}"


@case("E11")
def e11():
    """One stage's clean artifact and its 15 inferential cells load under NONOFFICIAL_SMOKE, and
    the frozen assemble_miou_c gives n = the clean n."""
    from src.stats.corruption_protocol import official_corruption_grid
    from src.stats.ingest import Policy, load_run
    from src.stats.robustness import assemble_miou_c
    pol = Policy.NONOFFICIAL_SMOKE
    clean = load_run(S["clean_e1"], pol)
    runs = []
    for cid in corruption_roles.INFERENTIAL_CELL_IDS:
        c, s = corruption_roles.parse_cell_id(cid)
        code, out = ec_outcome(ec_args(fresh(f"e11_{cid}"), corruption=c, severity=s))
        if code != "ok":
            return False, f"{cid}: {code}"
        runs.append(load_run(out, pol, expect_condition=(c, s)))
    mc = assemble_miou_c(runs, official_corruption_grid(), policy=pol, stage="E1", clean=clean)
    return (len(mc.values) == clean.n_included == 2 and np.isfinite(mc.values).all()), \
        f"n = {len(mc.values)}, clean n = {clean.n_included}"


@case("E12")
def e12():
    """INT8 E4 on QNNPACK: a cell scored from --provenance with both identity flags; the artifact
    records the converted artifact's sha256 and the backend."""
    code, out = ec_outcome(ec_args(fresh("e12"), stage="E4",
                                   expected_checkpoint_sha256=sha256_file(S["int8_prov"]),
                                   expected_converted_sha256=sha256_file(S["int8_art"])))
    if code != "ok":
        return False, code
    run_ = summary_of(out)["run"]
    rec = CO.read_cell_record(CO.record_path(out))
    return (run_["precision"] == "int8_ptq" and run_["quant_backend"] == "qnnpack"
            and run_["checkpoint_sha256"] == sha256_file(S["int8_art"])
            and rec["stage"] == "E4"), f"{run_['precision']} {run_['quant_backend']}"


# ---------------------------------------------------------------- T. the teacher
@case("T1")
def t1():
    """The teacher stage requires --teacher-ckpt-sha256."""
    code, _ = outcome(EC.validate_args, ec_args(fresh("t1"), stage="teacher",
                                                teacher_ckpt_sha256=None))
    return code == "teacher_ckpt_sha256_required", code


@case("T2")
def t2():
    """A malformed --teacher-ckpt-sha256 -> teacher_ckpt_sha256_format."""
    code, _ = outcome(EC.validate_args, ec_args(fresh("t2"), stage="teacher",
                                                teacher_ckpt_sha256=S["tk_sha"].upper()))
    return code == "teacher_ckpt_sha256_format", code


@case("T3")
def t3():
    """--teacher-ckpt-sha256 on a student -> teacher_ckpt_sha256_not_teacher."""
    code, _ = outcome(EC.validate_args, ec_args(fresh("t3"), teacher_ckpt_sha256=S["tk_sha"]))
    return code == "teacher_ckpt_sha256_not_teacher", code


@case("T4")
def t4():
    """A teacher hash mismatch is refused before the parse (the parser is never called)."""
    import src.distill.segnext_teacher as SN
    calls = []
    real = SN.load_teacher_state_dict
    with patched(SN, "load_teacher_state_dict", lambda p: calls.append(p) or real(p)):
        code, _ = ec_outcome(ec_args(fresh("t4"), stage="teacher", teacher_ckpt_sha256="1" * 64))
    return code == "teacher_hash_mismatch" and calls == [], f"{code}, parses {len(calls)}"


@case("T5")
def t5():
    """The teacher through an injected builder: batch 1, CPU, M4-V NMF draws == rows == the clean
    pass's (LE11), the stream's policy and seed recorded in the cell's log line, and the artifact
    and its record verify."""
    built_e, printed = [], []
    code, out = ec_outcome(ec_args(fresh("t5"), stage="teacher"),
                           teacher_builder=teacher_builder(built_e), stdout=printed)
    if code != "ok":
        return False, code
    verify_artifact(out)
    CO.verify_cell_record(out, CO.read_cell_record(CO.record_path(out)))
    rt = summary_of(out)["run"]["eval_runtime"]
    d_clean = S["clean_t_draws"]
    stream = built_e[-1].nmf_stream.describe()
    d_corr = stream["draws"]
    line = f"NMF stream M4-V: 3 draws for 3 rows (seed 42, state {stream['state_sha256'][:16]})"
    return (rt["batch_size"] == 1 and rt["model_device"] == "cpu" and d_corr == d_clean == 3
            and (stream["policy"], stream["seed"]) == ("M4-V", 42) and line in printed[-1]
            and summary_of(S["clean_t"])["run"]["eval_runtime"]["batch_size"] == 1), \
        f"draws corrupted {d_corr}, clean {d_clean}, rows 3; log line {printed[-1][:90]!r}"


@case("T6a")
def t6a():
    """--expected-checkpoint-sha256 and --teacher-ckpt-sha256 differ -> teacher_disagrees."""
    return le2_case(outcome(EC.validate_args, ec_args(fresh("t6a"), stage="teacher",
                                                      expected_checkpoint_sha256="2" * 64)),
                    "expected_checkpoint_sha256_teacher_disagrees", e="2" * 64, t=S["tk_sha"])


@case("T6b")
def t6b():
    """The teacher's checkpoint not the expected one (H == t, both wrong) ->
    expected_checkpoint_sha256_mismatch before any parse."""
    import src.distill.segnext_teacher as SN
    calls = []
    real = SN.load_teacher_state_dict
    with patched(SN, "load_teacher_state_dict", lambda p: calls.append(p) or real(p)):
        got = ec_outcome(ec_args(fresh("t6b"), stage="teacher", teacher_ckpt_sha256="3" * 64,
                                 expected_checkpoint_sha256="3" * 64))
    ok, detail = le2_case(got, "expected_checkpoint_sha256_mismatch", e="3" * 64,
                          actual=S["tk_sha"])
    return ok and calls == [], f"{detail}, parses {len(calls)}"


@case("T7")
def t7():
    """A teacher that draws its NMF bases twice per image -> nmf_draws_mismatch; nothing written."""
    out = fresh("t7")
    code, _ = ec_outcome(ec_args(out, stage="teacher"), teacher_builder=teacher_builder([], 2))
    return (code == "nmf_draws_mismatch" and not out.exists()
            and not CO.record_path(out).exists()), code


@case("T8")
def t8():
    """A teacher whose NMF stream is not M4-V with seed 42 (the loader's seed patched to 43) ->
    nmf_stream_mismatch; nothing written. The evaluator's own constant is the loader's seed."""
    import src.distill.nmf_stream as NS
    same = (EC.M4V_POLICY, EC.M4V_SEED) == ("M4-V", NS.M4_NMF_SEED)
    out = fresh("t8")
    with patched(NS, "M4_NMF_SEED", 43):
        code, _ = ec_outcome(ec_args(out, stage="teacher"), teacher_builder=teacher_builder([]))
    return (same and code == "nmf_stream_mismatch" and not out.exists()
            and not CO.record_path(out).exists()), f"{code}; constants agree {same}"


# ---------------------------------------------------------------- ST. stages, sources, identity
#: LE23: the clause that makes each artifact-schema stage outside STAGES descriptive (the case's own
#: copy, never read from the module under check); A, F and G are NOT_IN_ROBUSTNESS's arms.
LE23_CLAUSES = {"E6-E2": "AM-23", "E7-E2": "AM-23", "E6-R1": "AM-23", "E6-R2": "AM-23",
                "E6-KD": "AM-3"}


@case("ST1")
def st1():
    """A, F and G -> stage_not_in_robustness; the artifact schema's other stages outside STAGES,
    E6-E2, E7-E2, E6-R1, E6-R2 (AM-23) and E6-KD (AM-3), -> stage_not_in_robustness naming their
    clause and TL section 9, these five being every such stage (LE23); an unknown stage ->
    stage_unknown; before any file is read."""
    codes = [outcome(EC.validate_args, ec_args(never("st1"), stage=s, checkpoint=never("ck")))[0]
             for s in ("A", "F", "G")]
    bad = []
    for s, clause in LE23_CLAUSES.items():
        code, e = outcome(EC.validate_args, ec_args(never("st1"), stage=s, provenance=never("pv")))
        if not (code == "stage_not_in_robustness" and f"({clause}:" in str(e)
                and "TL interface section 9" in str(e)):
            bad.append((s, code))
    every = set(ARTIFACT_STAGES) - set(EC.STAGES) == set(EC.NOT_IN_ROBUSTNESS) | set(LE23_CLAUSES)
    unknown, _ = outcome(EC.validate_args, ec_args(never("st1"), stage="E9", checkpoint="x"))
    # 'TEACHER' is a stage-artifact key outside STAGES: without the gate the checks would go on
    upper, _ = outcome(EC.validate_args, ec_args(never("st1"), stage="TEACHER", checkpoint="x"))
    return (codes == ["stage_not_in_robustness"] * 3 and not bad and every
            and unknown == upper == "stage_unknown"), \
        f"{codes} {unknown} {upper}; descriptive rows failing {bad}; every such stage {every}"


@case("ST2")
def st2():
    """INT8 without --provenance -> refused, with evaluate_model.py's text."""
    code, e = outcome(EC.validate_args, ec_args(never("st2"), stage="E4", provenance=None))
    return code == "CorruptionCliError" and "requires --provenance" in str(e), str(e)[:80]


@case("ST3")
def st3():
    """INT8 on cuda -> refused (CPU only)."""
    code, e = outcome(EC.validate_args, ec_args(never("st3"), stage="E4", device="cuda"))
    return code == "CorruptionCliError" and "CPU-only" in str(e), str(e)[:80]


@case("ST4")
def st4():
    """An E3 run handed a checkpoint that records no stage -> stage_undeclared (DL-52)."""
    code, _ = ec_outcome(ec_args(fresh("st4"), stage="E3", checkpoint=S["ckpt_e3"]))
    return code == "stage_undeclared", code


@case("ST5")
def st5():
    """--protocol upstream -> protocol_canvas_only (G11)."""
    code, _ = outcome(EC.validate_args, ec_args(never("st5"), protocol="upstream"))
    return code == "protocol_canvas_only", code


PARITY = [
    ("test split unconfirmed", ["--stage", "E1", "--checkpoint", "c", "--split", "test",
                                "--artifact-status", "provisional"], {}),
    ("test split smoke", ["--stage", "E1", "--checkpoint", "c", "--split", "test",
                          "--confirm-test-split", "--artifact-status", "smoke"], {}),
    ("test split capped", ["--stage", "E1", "--checkpoint", "c", "--split", "test",
                           "--confirm-test-split", "--artifact-status", "provisional",
                           "--max-samples", "5"], {}),
    ("INT8 on cuda", ["--stage", "E4", "--provenance", "p", "--device", "cuda"],
     {"em": ["--precision", "int8_ptq"]}),
    ("INT8 two sources", ["--stage", "E4", "--provenance", "p", "--checkpoint", "c"],
     {"em": ["--precision", "int8_ptq"]}),
    ("INT8 without provenance", ["--stage", "E4"], {"em": ["--precision", "int8_ptq"]}),
    ("FP32 with provenance", ["--stage", "E1", "--checkpoint", "c", "--provenance", "p"], {}),
    ("teacher without checkpoint", ["--stage", "teacher", "--teacher-config", "cfg",
                                    "--batch-size", "1"], {"em": ["--model-role", "teacher"]}),
    ("teacher without config", ["--stage", "teacher", "--checkpoint", "t", "--batch-size", "1"],
     {"em": ["--model-role", "teacher"]}),
    ("teacher batch 2", ["--stage", "teacher", "--checkpoint", "t", "--teacher-config", "cfg",
                         "--batch-size", "2"], {"em": ["--model-role", "teacher"]}),
    ("teacher sha missing", ["--stage", "teacher", "--checkpoint", "t", "--teacher-config", "cfg",
                             "--batch-size", "1"], {"em": ["--model-role", "teacher"]}),
    ("teacher sha malformed", ["--stage", "teacher", "--checkpoint", "t", "--teacher-config",
                               "cfg", "--batch-size", "1", "--teacher-ckpt-sha256", "ABC"],
     {"em": ["--model-role", "teacher"]}),
    ("student with teacher sha", ["--stage", "E1", "--checkpoint", "c",
                                  "--teacher-ckpt-sha256", "a" * 64], {}),
    ("batch size 0", ["--stage", "E1", "--checkpoint", "c", "--batch-size", "0"], {}),
    ("max samples 0", ["--stage", "E1", "--checkpoint", "c", "--max-samples", "0"], {}),
]


@case("ST6")
def st6():
    """Model-source parity: every row of an argument table is refused with the same text by
    evaluate_model.validate_cli_args and evaluate_corruptions.validate_args."""
    bad = []
    for label, common, extra in PARITY:
        em = EM.build_parser().parse_args(common + extra.get("em", []) + ["--out-dir", "o"])
        ec = EC.build_parser().parse_args(common + ["--corruption", "fog", "--severity", "2",
                                                    "--out-dir", "o", "--corruption-cache", "c"])
        _, a = outcome(EM.validate_cli_args, em)
        _, b = outcome(EC.validate_args, ec)
        if not (isinstance(a, Exception) and isinstance(b, Exception) and str(a) == str(b)):
            bad.append((label, str(a)[:60], str(b)[:60]))
    return not bad, str(bad[:2]) if bad else f"{len(PARITY)} rows refused identically"


#: LE2's codes and messages, verbatim from the ruling: every LE2 case also checks its text (SM-11).
LE2_TEMPLATES = {
    "expected_checkpoint_sha256_required":
        "--split test with --artifact-status official requires --expected-checkpoint-sha256, the "
        "sha256 of the file this run is handed (--checkpoint, or --provenance for an INT8 stage)",
    "expected_checkpoint_sha256_format": "--expected-checkpoint-sha256 {sha256_format_error(value)}",
    "expected_checkpoint_sha256_teacher_disagrees":
        "--expected-checkpoint-sha256 {e} and --teacher-ckpt-sha256 {t} differ; for the teacher "
        "both name --checkpoint and must be equal",
    "expected_checkpoint_sha256_mismatch":
        "{path} has sha256 {actual}, expected {e}; the file was not parsed",
    "source_changed_during_validation":
        "{path} had sha256 {e} before validation and {actual} after it",
    "expected_converted_sha256_required":
        "--split test with --artifact-status official requires --expected-converted-sha256 for an "
        "INT8 stage, the sha256 of the converted artifact its provenance names",
    "expected_converted_sha256_format": "--expected-converted-sha256 {sha256_format_error(value)}",
    "expected_converted_sha256_not_int8":
        "--expected-converted-sha256 applies to an INT8 stage handed --provenance; this run "
        "(precision={p!r}, random_init={r}) takes none",
    "expected_converted_sha256_mismatch":
        "the converted artifact {path} has sha256 {actual}, expected {e2}; it was not loaded",
    "converted_changed_before_load":
        "the converted artifact {path} had sha256 {e2} at validation and {actual} before the "
        "load; it was not loaded",
    "threads_invalid": "--threads must be a positive integer, got {t!r}",
    "threads_required":
        "--split test with --artifact-status official requires --threads, the torch thread count "
        "of the execution context of record",
    "threads_not_applied": "torch.set_num_threads({t}) left torch.get_num_threads() at {n}",
    "threads_drifted":
        "run.eval_runtime.torch_num_threads is {n}, --threads is {t}; no artifact was written",
}
_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)(\(value\))?(!r)?\}")


def le2_text_ok(exc, code: str, /, **known) -> bool:
    """str(exc) is exactly `[code] message` on LE2's template; a placeholder not in `known` matches
    any non-empty text."""
    tpl, parts, pos = LE2_TEMPLATES[code], [re.escape(f"[{code}] ")], 0
    for m in _PLACEHOLDER.finditer(tpl):
        parts.append(re.escape(tpl[pos:m.start()]))
        name = m.group(1)
        if name in known:
            parts.append(re.escape(repr(known[name]) if m.group(3) else str(known[name])))
        else:
            parts.append(".+?")
        pos = m.end()
    parts.append(re.escape(tpl[pos:]))
    return re.fullmatch("".join(parts), str(exc)) is not None


def le2_case(got, code: str, /, **known) -> tuple[bool, str]:
    c, exc = got
    return c == code and le2_text_ok(exc, code, **known), f"{c}: {str(exc)[:120]}"


@case("ST7a")
def st7a():
    """Under R (--split test, official) without --expected-checkpoint-sha256 -> ..._required."""
    kw = {**official_argv(None, never("st7a_cache")), "expected_checkpoint_sha256": None}
    return le2_case(outcome(EC.validate_args, ec_args(never("st7a"), **kw)),
                    "expected_checkpoint_sha256_required")


@case("ST7b")
def st7b():
    """A malformed --expected-checkpoint-sha256 -> expected_checkpoint_sha256_format."""
    return le2_case(outcome(EC.validate_args, ec_args(never("st7b"),
                                                      expected_checkpoint_sha256="xyz")),
                    "expected_checkpoint_sha256_format",
                    sha256_format_error=EC.sha256_format_error("xyz"))


@case("ST7c")
def st7c():
    """The handed checkpoint is not the expected file -> expected_checkpoint_sha256_mismatch,
    before any torch.load of it."""
    loads = []
    real = torch.load
    with patched(torch, "load", lambda p, *a, **k: loads.append(str(p)) or real(p, *a, **k)):
        got = ec_outcome(ec_args(fresh("st7c"), expected_checkpoint_sha256="4" * 64))
    ok, detail = le2_case(got, "expected_checkpoint_sha256_mismatch", e="4" * 64,
                          actual=S["ckpt_sha"])
    return ok and str(S["ckpt"]) not in loads, f"{detail}, loads {loads[:2]}"


@case("ST7d")
def st7d():
    """The handed checkpoint changes during validation -> source_changed_during_validation."""
    victim = S["work"] / f"st7d_{next(_SEQ)}.pt"
    shutil.copy2(S["ckpt"], victim)
    real = EC.resolve_evaluation_source

    def tamper(*a, **k):
        out = real(*a, **k)
        with open(victim, "ab") as f:
            f.write(b"\0")
        return out
    with patched(EC, "resolve_evaluation_source", tamper):
        got = ec_outcome(ec_args(fresh("st7d"), checkpoint=victim))
    return le2_case(got, "source_changed_during_validation", e=S["ckpt_sha"],
                    actual=sha256_file(victim))


@case("ST7e")
def st7e():
    """Under R, an INT8 stage without --expected-converted-sha256 -> ..._required."""
    args = ec_args(never("st7e"), **{**official_argv(None, never("c")), "stage": "E4",
                                     "expected_checkpoint_sha256": "5" * 64})
    return le2_case(outcome(EC.validate_args, args), "expected_converted_sha256_required")


@case("ST7f")
def st7f():
    """A malformed --expected-converted-sha256 -> expected_converted_sha256_format."""
    return le2_case(outcome(EC.validate_args, ec_args(never("st7f"), stage="E4",
                                                      expected_converted_sha256="Z" * 64)),
                    "expected_converted_sha256_format",
                    sha256_format_error=EC.sha256_format_error("Z" * 64))


@case("ST7g")
def st7g():
    """--expected-converted-sha256 on an FP32 stage -> expected_converted_sha256_not_int8."""
    return le2_case(outcome(EC.validate_args, ec_args(never("st7g"),
                                                      expected_converted_sha256="6" * 64)),
                    "expected_converted_sha256_not_int8", p="fp32", r=False)


@case("ST7h")
def st7h():
    """The converted artifact is not the expected one -> expected_converted_sha256_mismatch; it is
    never loaded."""
    import src.eval.model_loading as ML
    loads = []
    real = ML.load_int8_student
    with patched(ML, "load_int8_student", lambda *a, **k: loads.append(1) or real(*a, **k)):
        got = ec_outcome(ec_args(fresh("st7h"), stage="E4", expected_converted_sha256="7" * 64))
    ok, detail = le2_case(got, "expected_converted_sha256_mismatch", e2="7" * 64,
                          actual=sha256_file(S["int8_art"]))
    return ok and loads == [], f"{detail}, {loads}"


@case("ST7i")
def st7i():
    """The converted artifact changes between validation and the load ->
    converted_changed_before_load; it is never loaded."""
    import src.eval.model_loading as ML
    work = S["work"] / f"st7i_{next(_SEQ)}"
    work.mkdir()
    art = work / "e4_int8.pt"
    shutil.copy2(S["int8_art"], art)
    prov = work / "e4_provenance.json"
    doc = json.loads(S["int8_prov"].read_text(encoding="utf-8"))
    doc["converted_artifact"] = str(art)
    prov.write_text(json.dumps(doc), encoding="utf-8")
    loads = []
    real_load, real_cell = ML.load_int8_student, EC.PlantSegCorruptedEvalDataset

    def dataset_then_tamper(*a, **k):
        with open(art, "ab") as f:
            f.write(b"\0")
        return real_cell(*a, **k)
    with patched(ML, "load_int8_student", lambda *a, **k: loads.append(1) or real_load(*a, **k)), \
            patched(EC, "PlantSegCorruptedEvalDataset", dataset_then_tamper):
        got = ec_outcome(ec_args(fresh("st7i"), stage="E4", provenance=prov))
    ok, detail = le2_case(got, "converted_changed_before_load", e2=sha256_file(S["int8_art"]),
                          actual=sha256_file(art))
    return ok and loads == [], f"{detail}, {loads}"


@case("ST7j")
def st7j():
    """--threads 0 and --threads x -> threads_invalid."""
    oks = [le2_case(outcome(EC.validate_args, ec_args(never("st7j"), threads=t)),
                    "threads_invalid", t=t) for t in ("0", "x")]
    return all(ok for ok, _ in oks), str([d for _, d in oks])


@case("ST7k")
def st7k():
    """Under R without --threads -> threads_required."""
    kw = {**official_argv(None, never("st7k_cache")), "threads": None}
    return le2_case(outcome(EC.validate_args, ec_args(never("st7k"), **kw)), "threads_required")


@case("ST7l")
def st7l():
    """torch.set_num_threads does not take -> threads_not_applied, before any model is built."""
    other = 1 if torch.get_num_threads() != 1 else 2
    held = torch.get_num_threads()
    with patched(torch, "set_num_threads", lambda n: None):
        got = ec_outcome(ec_args(fresh("st7l"), threads=other))
    return le2_case(got, "threads_not_applied", t=other, n=held)


@case("ST7m")
def st7m():
    """run.eval_runtime.torch_num_threads differs from --threads -> threads_drifted; nothing is
    written."""
    out = fresh("st7m")
    real = ER.build_eval_runtime_record

    def drifted(**kw):
        rec = real(**kw)
        rec["torch_num_threads"] = rec["torch_num_threads"] + 1
        return rec
    with patched(ER, "build_eval_runtime_record", drifted):
        got = ec_outcome(ec_args(out, threads=1))
    ok, detail = le2_case(got, "threads_drifted", n=2, t=1)
    return ok and not out.exists() and not CO.record_path(out).exists(), detail


@case("ST7n")
def st7n():
    """The FP32 checkpoint changes between validation and its load ->
    checkpoint_changed_after_validation; nothing is written."""
    victim = S["work"] / f"st7n_{next(_SEQ)}.pt"
    shutil.copy2(S["ckpt"], victim)
    out = fresh("st7n")
    real_cell = EC.PlantSegCorruptedEvalDataset

    def dataset_then_swap(*a, **k):
        shutil.copy2(S["ckpt_e3"], victim)          # another valid checkpoint at the same path
        return real_cell(*a, **k)
    with patched(EC, "PlantSegCorruptedEvalDataset", dataset_then_swap):
        code, e = ec_outcome(ec_args(out, checkpoint=victim))
    return (code == "checkpoint_changed_after_validation" and not out.exists()
            and not CO.record_path(out).exists()), f"{code}: {str(e)[:90]}"


@case("ST8a")
def st8a():
    """--precision qat_fakequant_off -> qat_fakequant_off_clean_only (LE15)."""
    code, _ = outcome(EC.validate_args, ec_args(never("st8a"), stage="E5",
                                                precision="qat_fakequant_off", provenance="p"))
    return code == "qat_fakequant_off_clean_only", code


@case("ST8b")
def st8b():
    """A precision that is not the stage's -> precision_stage_mismatch."""
    code, _ = outcome(EC.validate_args, ec_args(never("st8b"), stage="E4", precision="fp32",
                                                provenance="p"))
    return code == "precision_stage_mismatch", code


@case("ST9")
def st9():
    """An FP32 stage without --checkpoint -> checkpoint_required (no random-init path)."""
    code, _ = outcome(EC.validate_args, ec_args(never("st9"), checkpoint=None, provenance=None,
                                                stage="E2"))
    return code == "checkpoint_required", code


@case("ST10")
def st10():
    """A malformed --expected-cache-sha256 -> expected_cache_sha256_format."""
    code, _ = outcome(EC.validate_args, ec_args(never("st10"), expected_cache_sha256="12ab"))
    return code == "expected_cache_sha256_format", code


# ---------------------------------------------------------------- ST11. the INT8 gate (LE22)
#: The function the official INT8 path calls, held before any mutant patches it: the cases' own
#: expectation.
REAL_INT8_OFFICIAL_ERROR = SA.int8_official_error


class CacheGateReached(Exception):
    """Raised by the stand-in for resolve_corruption_cache: the run passed every model-source gate
    and reached the cache step, which reads nothing."""
    code = "cache_gate_reached"


def ptq_ts_fixture(tag: str) -> tuple[Path, Path]:
    """An E4 run folder as scripts/smoke_ptq_record.py's case_s8_e2e leaves it: a TorchScript
    artifact and its run provenance (artifact_format torchscript), whose calibration block claims
    the committed list of record (configs/calibration/ptq_calibration_seed42.json's checksum, seed
    42, role am10_list_of_record, list_artifact_status registered), and no e4_ptq_record.json. The
    artifact is never loaded: the official gate refuses first."""
    from src.quant.ptq import artifact_names
    names = artifact_names("e4")
    run_dir = S["work"] / f"{tag}_{next(_SEQ)}"
    run_dir.mkdir(parents=True)
    ts = run_dir / names["torchscript"]
    torch.jit.save(torch.jit.script(nn.Identity()), str(ts))
    listed = json.loads((REPO / "configs" / "calibration" / "ptq_calibration_seed42.json")
                        .read_text(encoding="utf-8"))
    prov = run_dir / names["run_meta"]
    prov.write_text(json.dumps({
        "stage": "E4", "source_stage": "E1", "quantization_method": "ptq", "random_init": False,
        "num_classes": NC, "backend": "qnnpack", "artifact_format": "torchscript",
        "converted_artifact": ts.name, "converted_artifact_sha256": sha256_file(ts),
        "calibration": {"checksum_sha256": listed["checksum_sha256"], "seed": 42,
                        "role": "am10_list_of_record", "list_artifact_status": "registered"}},
        indent=2), encoding="utf-8")
    return ts, prov


def qat_fixture(tag: str) -> tuple[Path, Path]:
    """A QAT E5 (from E1, QNNPACK) provenance naming a converted artifact."""
    run_dir = S["work"] / f"{tag}_{next(_SEQ)}"
    run_dir.mkdir(parents=True)
    art = run_dir / "e5_int8.pt"
    shutil.copy2(S["int8_art"], art)
    prov = run_dir / "e5_provenance.json"
    prov.write_text(json.dumps({"stage": "E5", "source_stage": "E1", "quantization_method": "qat",
                                "random_init": False, "num_classes": NC, "backend": "qnnpack",
                                "converted_artifact": str(art),
                                "converted_artifact_sha256": sha256_file(art)}, indent=2),
                    encoding="utf-8")
    return art, prov


def le22_run(tag: str, stage: str, prov: Path, art: Path):
    """One cell under R (--split test --confirm-test-split --artifact-status official) for an INT8
    stage, every earlier gate passing (H, H2, T, a fresh out-dir); the cache step is a stand-in that
    reads nothing. Returns ((code, exception), stand-in calls, nothing built or written)."""
    out = fresh(tag)
    reached = []

    def stand_in(*a, **k):
        reached.append(1)
        raise CacheGateReached()
    args = ec_args(out, **{**official_argv(None, never(f"{tag}_cache")), "stage": stage,
                           "provenance": prov, "expected_checkpoint_sha256": sha256_file(prov),
                           "expected_converted_sha256": sha256_file(art)})
    counters = EM.Counters(dataset=[], model=[])
    with patched(EC, "resolve_corruption_cache", stand_in):
        got = ec_outcome(args, counters=counters)
    clean = (not out.exists() and not CO.record_path(out).exists()
             and counters.dataset == [] and counters.model == [])
    return got, reached, clean


@case("ST11a")
def st11a():
    """LE22: an official E4 as case_s8_e2e leaves it: int8_official_calibration_error passes, and
    the run is refused under R, before any cache or dataset is read, with int8_official_error's
    [ptq_record_missing] text unchanged."""
    art, prov = ptq_ts_fixture("st11a")
    resolved = SA.resolve_evaluation_source("E4", provenance=prov)
    calibration = SA.int8_official_calibration_error(resolved, repo_root=EC.REPO)
    want = REAL_INT8_OFFICIAL_ERROR(resolved, repo_root=EC.REPO)
    (code, e), reached, clean = le22_run("st11a", "E4", prov, art)
    return (calibration is None and want is not None and "[ptq_record_missing]" in want
            and code == "CorruptionCliError" and str(e) == want and not reached and clean), \
        f"{code}: {str(e)[:110]}; calibration gate {calibration!r}; cache reached {len(reached)}"


@case("ST11b")
def st11b():
    """LE22: the src/quant/runner.py state_dict E4 under R -> int8_official_error's
    [legacy_ptq_artifact] text unchanged, before any cache or dataset is read."""
    resolved = SA.resolve_evaluation_source("E4", provenance=S["int8_prov"])
    want = REAL_INT8_OFFICIAL_ERROR(resolved, repo_root=EC.REPO)
    (code, e), reached, clean = le22_run("st11b", "E4", S["int8_prov"], S["int8_art"])
    return (want is not None and "[legacy_ptq_artifact]" in want and code == "CorruptionCliError"
            and str(e) == want and not reached and clean), \
        f"{code}: {str(e)[:110]}; cache reached {len(reached)}"


@case("ST11c")
def st11c():
    """LE22: a QAT E5 under R passes the official INT8 gate: the run reaches the cache step (the
    stand-in), with nothing built or written."""
    art, prov = qat_fixture("st11c")
    resolved = SA.resolve_evaluation_source("E5", provenance=prov)
    gate = REAL_INT8_OFFICIAL_ERROR(resolved, repo_root=EC.REPO)
    (code, e), reached, clean = le22_run("st11c", "E5", prov, art)
    return (gate is None and code == "cache_gate_reached" and reached == [1] and clean), \
        f"gate {gate!r}; {code}; cache reached {len(reached)}"


# ---------------------------------------------------------------- G. section 7 guards and names
@case("G1")
def g1():
    """--split test without --confirm-test-split is refused before any path is read."""
    out, cache = never("g1"), never("g1_cache")
    code, e = outcome(EC.validate_args, ec_args(out, split="test", status="provisional",
                                                max_samples=None, cache=cache))
    return (code == "CorruptionCliError" and "--confirm-test-split" in str(e)
            and not out.parent.exists()), str(e)[:60]


@case("G2")
def g2():
    """--split test refuses artifact_status smoke."""
    code, e = outcome(EC.validate_args, ec_args(never("g2"), split="test", status="smoke",
                                                max_samples=None, confirm_test_split=True))
    return code == "CorruptionCliError" and "smoke" in str(e), str(e)[:60]


@case("G3")
def g3():
    """--split test refuses --max-samples, whatever the status."""
    code, e = outcome(EC.validate_args, ec_args(never("g3"), split="test", status="provisional",
                                                max_samples=5, confirm_test_split=True))
    return code == "CorruptionCliError" and "sample cap" in str(e), str(e)[:60]


@case("G5a")
def g5a():
    """Official on val -> refused (official requires split test)."""
    code, e = outcome(EC.validate_args, ec_args(never("g5a"), status="official", max_samples=None))
    return code == "CorruptionCliError" and "requires split='test'" in str(e), str(e)[:70]


def official_case(name, mutate, code_wanted, *, record=False, expected=True, doc=""):
    @case(name)
    def _c():
        root, m = forged(name.lower(), mutate)
        extra = {}
        if expected:
            extra["expected_cache_sha256"] = m["canonical_manifest_sha256"]
        if record:
            extra["verification_record"] = record_for(root, m, **(record if isinstance(record, dict)
                                                                  else {}))
        args = ec_args(fresh(name.lower()), **official_argv(None, root), **extra)
        code, _ = ec_outcome(args)
        return code == code_wanted, code
    _c.__doc__ = doc
    return _c


official_case("G5b", lambda m: as_tday(m, source_image_count=6), "cache_incomplete_for_official",
              doc="An official run on a cache of 6 sources -> cache_incomplete_for_official.")
official_case("G5c", lambda m: as_tday(m, status="provisional"), "cache_status_insufficient",
              doc="An official run on a provisional cache -> cache_status_insufficient.")
official_case("G5d", lambda m: (as_tday(m), m.update(pin=None)), "cache_pin_missing",
              doc="An official run on a cache without its pin -> cache_pin_missing.")
official_case("G5e", as_tday, "expected_cache_sha256_required", expected=False,
              doc="An official run without --expected-cache-sha256 -> ..._required.")
official_case("G5g", as_tday, "verification_record_required",
              doc="An official run without --verification-record -> ..._required.")
official_case("G5h", as_tday, "verification_record_mismatch",
              record={"counts": {"sources_verified": 1561, "items_verified": 1, "cells": 24}},
              doc="An official run with a record of another cache -> ..._mismatch.")


@case("G5f")
def g5f():
    """An official run whose --expected-cache-sha256 is not the cache's -> cache_checksum_mismatch;
    and an unreadable verification record -> verification_record_mismatch."""
    root, m = forged("g5f", as_tday)
    code, _ = ec_outcome(ec_args(fresh("g5f"), **official_argv(None, root),
                                 expected_cache_sha256="8" * 64))
    junk = root.parent / f"g5f_junk_{next(_SEQ)}.json"
    junk.write_text("{not json", encoding="utf-8")
    code2, _ = ec_outcome(ec_args(fresh("g5f2"), **official_argv(None, root),
                                  expected_cache_sha256=m["canonical_manifest_sha256"],
                                  verification_record=junk))
    return code == "cache_checksum_mismatch" and code2 == "verification_record_mismatch", \
        f"{code}, {code2}"


@case("G6a")
def g6a():
    """A provisional run on a smoke cache -> cache_status_insufficient."""
    code, _ = ec_outcome(ec_args(fresh("g6a"), status="provisional",
                                 expected_cache_sha256=S["m_smoke"]["canonical_manifest_sha256"]))
    return code == "cache_status_insufficient", code


@case("G6b")
def g6b():
    """A provisional run without --expected-cache-sha256 -> expected_cache_sha256_required."""
    code, _ = ec_outcome(ec_args(fresh("g6b"), status="provisional", cache=S["c_prov"]))
    return code == "expected_cache_sha256_required", code


@case("G6c")
def g6c():
    """A provisional run without --verification-record -> verification_record_required."""
    code, _ = ec_outcome(ec_args(fresh("g6c"), status="provisional", cache=S["c_prov"],
                                 expected_cache_sha256=S["m_prov"]["canonical_manifest_sha256"]))
    return code == "verification_record_required", code


@case("G6d")
def g6d():
    """A provisional run with every gate passing: the record names the verifier's record by its
    sha256, which equals the verifier's printed value."""
    code, out = ec_outcome(ec_args(fresh("g6d"), status="provisional", cache=S["c_prov"],
                                   expected_cache_sha256=S["m_prov"]["canonical_manifest_sha256"],
                                   verification_record=S["v_prov"]["record"]))
    if code != "ok":
        return False, code
    rec = CO.read_cell_record(CO.record_path(out))
    want = S["v_prov"]["values"]["verification_record_sha256"]
    return (rec["cache"]["verification_record_sha256"] == want == sha256_file(S["v_prov"]["record"])
            and summary_of(out)["run"]["artifact_status"] == "provisional"), \
        rec["cache"]["verification_record_sha256"][:16]


@case("G7")
def g7():
    """A stray cell record without its out-dir -> record_exists, before any input is read; and the
    record writer refuses an existing record."""
    out = fresh("g7")
    out.parent.mkdir(parents=True, exist_ok=True)
    CO.record_path(out).write_text("{}", encoding="utf-8")
    code, _ = ec_outcome(ec_args(out))
    path = fresh("g7_rec")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    code2, _ = outcome(CO.write_cell_record, path, {"schema": CO.CELL_RECORD_SCHEMA})
    return code == code2 == "record_exists" and not out.exists(), \
        f"{code}, {code2}, out-dir created: {out.exists()}"


@case("G8")
def g8():
    """An out-dir inside the repository -> out_dir_inside_repo; nothing is created there."""
    out = REPO / f"never_created_lcorr_g8_{next(_SEQ)}"
    code, _ = ec_outcome(ec_args(out, cache=never("g8_cache")))
    return code == "out_dir_inside_repo" and not out.exists(), code


@case("G9")
def g9():
    """A val run on a cache whose pin names split test -> cache_split_mismatch (the split is read
    from the pin's parameters, LE3)."""
    def mutate(m):
        m["pin"]["parameters"]["split"] = "test"
    root, _ = forged_items("g9", mutate)
    code, _ = ec_outcome(ec_args(fresh("g9"), cache=root))
    return code == "cache_split_mismatch", code


@case("G10")
def g10():
    """A cache generated under another protocol file -> protocol_changed_since_generation."""
    root, _ = forged_items("g10", lambda m: m["protocol"].update(sha256="9" * 64))
    code, _ = ec_outcome(ec_args(fresh("g10"), cache=root))
    return code == "protocol_changed_since_generation", code


@case("G5i")
def g5i():
    """A provisional VAL run whose verification record is another cache's ->
    verification_record_mismatch (past that gate the run would score the provisional cache)."""
    other = record_for(S["c_smoke"], S["m_smoke"])
    code, _ = ec_outcome(ec_args(fresh("g5i"), status="provisional", cache=S["c_prov"],
                                 expected_cache_sha256=S["m_prov"]["canonical_manifest_sha256"],
                                 verification_record=other))
    return code == "verification_record_mismatch", code


@case("N1")
def n1():
    """An out-dir holding 'test' in any case, or ':' in its name -> out_dir_name_invalid; nothing
    is created."""
    a = S["work"] / "never_created" / f"corr_TeSt_{next(_SEQ)}"
    b = S["work"] / "never_created" / f"corr:{next(_SEQ)}"
    codes = [ec_outcome(ec_args(p, cache=never("n1_cache")))[0] for p in (a, b)]
    return codes == ["out_dir_name_invalid"] * 2 and not (S["work"] / "never_created").exists(), \
        str(codes)


@case("N1b")
def n1b():
    """':' in a PARENT component of the out-dir (write_artifact would create it) ->
    out_dir_name_invalid (TL section 2 covers every component); nothing is created."""
    p = S["work"] / "never_created" / f"run:{next(_SEQ)}" / "corr_e1v_fog_s3"
    code, _ = ec_outcome(ec_args(p, cache=never("n1b_cache")))
    return code == "out_dir_name_invalid" and not (S["work"] / "never_created").exists(), code


def dangling_link(tag: str) -> Path:
    """A symlink named without 'test' whose target text holds ':' and does not exist: only the
    resolved path shows the forbidden character, and nothing is created at the target."""
    link = S["work"] / f"{tag}_{next(_SEQ)}"
    os.symlink(f"/nonexistent_lcorr_{secrets.token_hex(4)}/run:{next(_SEQ)}", link)
    return link


@case("N1c")
def n1c():
    """An out-dir reached through a symlink whose target holds ':' -> out_dir_name_invalid (the
    resolved path is checked too)."""
    p = dangling_link("n1c_link") / "corr_e1v_fog_s3"
    code, _ = ec_outcome(ec_args(p, cache=never("n1c_cache")))
    return code == "out_dir_name_invalid", code


@case("P1")
def p1():
    """LE2's printing: a coded refusal is printed on stderr exactly as `[code] message`; an
    uncoded one keeps its exception type."""
    lines = []
    for argv in (ec_argv(never("p1"), stage="A", checkpoint="x"),
                 ec_argv(never("p1"), batch=0)):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = EC.main(argv)
        lines.append((rc, err.getvalue().strip()))
    coded = lines[0][1].startswith("[stage_not_in_robustness] ") and lines[0][0] == 2
    uncoded = lines[1][1] == "CorruptionCliError: --batch-size must be >= 1" and lines[1][0] == 2
    return coded and uncoded, str(lines)


# ---------------------------------------------------------------- C. the cache gates
@case("C1")
def c1():
    """The split listing and the cache's sources differ -> cache_sources_mismatch (the expected
    manifest comes from the listing, never from the cache)."""
    def rename(m):
        old = S["stems"][0]
        for s in m["sources"]:
            if s["image_id"] == old:
                s["image_id"] = "zz_renamed"
        for e in m["entries"]:
            if e["image_id"] == old:
                e["image_id"] = "zz_renamed"
    root, _ = forged("c1", rename)
    S["c1_root"] = root
    code, _ = ec_outcome(ec_args(fresh("c1"), cache=root))
    return code == "cache_sources_mismatch", code


@case("C3")
def c3():
    """--expected-cache-sha256 not the cache's (a smoke run) -> cache_checksum_mismatch."""
    code, _ = ec_outcome(ec_args(fresh("c3"), expected_cache_sha256="a" * 64))
    return code == "cache_checksum_mismatch", code


@case("C4")
def c4():
    """A cache root that does not exist -> cache_root_missing."""
    code, _ = ec_outcome(ec_args(fresh("c4"), cache=never("c4_cache")))
    return code == "cache_root_missing", code


@case("C5")
def c5():
    """Items and source digests not verified once per row -> verification_count_mismatch, nothing
    written; and the record builder refuses mismatched counts."""
    out = fresh("c5")
    real = CO.PlantSegCorruptedEvalDataset.__getitem__

    def uncounted(self, i):
        sample = real(self, i)
        self.items_verified -= 1
        return sample
    with patched(CO.PlantSegCorruptedEvalDataset, "__getitem__", uncounted):
        code, _ = ec_outcome(ec_args(out))
    # an existing artifact directory: without the count gate the builder would return a record
    code2, _ = outcome(CO.build_cell_record, out_dir=S["clean_e1"], run_id="r", stage="E1",
                       condition={"type": "corruption", "name": "fog", "severity": 3},
                       role="inferential", manifest=S["m_smoke"], cell_sha256="0" * 64,
                       verification_record_sha256=None, rows=3, items_verified=2,
                       source_digests_verified=3, evaluator_commit="0" * 40, started_utc="s",
                       split="val")
    return (code == code2 == "verification_count_mismatch" and not out.exists()), \
        f"{code}, {code2}"


# ---------------------------------------------------------------- R. the grid runner
def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=s4-fixture", "-c", "user.email=s4-fixture@invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), capture_output=True, text=True, check=False,
        env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(repo.parent)))
    if proc.returncode != 0:
        raise AssertionError(f"fixture git {args} failed ({proc.returncode}): {proc.stderr[:200]}")
    return proc.stdout


def make_fixture_repo(root: Path, *, governed_dirty: bool = False) -> Path:
    """A throwaway repository (never the project): clean, or with one untracked governed path."""
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "seed.txt").write_text("s4 fixture\n", encoding="utf-8")
    _git(root, "add", "--", "seed.txt")
    _git(root, "commit", "-q", "-m", "fixture baseline")
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root.resolve():
        raise AssertionError("the fixture is not its own repository")
    if governed_dirty:
        (root / "scripts").mkdir()
        (root / "scripts" / "probe.txt").write_text("governed-dirty probe\n", encoding="utf-8")
    return root


def inventory(rows, *, threads=1, image_digest=None, commit=None, **top) -> dict:
    doc = {"schema": RG.INVENTORY_SCHEMA, "evaluator_commit": commit or S["head"],
           "execution_context": {"model_device": "cpu", "input_devices": ["cpu"],
                                 "determinism_policy_applied": False, "batch_size": 1,
                                 "num_workers": 0, "torch_num_threads": threads,
                                 "image_digest": image_digest},
           "manifest_sha256": "b" * 64, "models": rows}
    doc.update(top)
    return doc


def model_row(file_id, stage, cells, *, seed=42, path="e1.pt", sha=None, precision=None,
              converted=None, role=None) -> dict:
    precision = precision or RG.STAGE_PRECISION.get(stage, "fp32")
    kind = "provenance" if precision in RG.INT8 else "checkpoint"
    return {"file_id": file_id, "stage": stage, "model_role": role or (
            "teacher" if stage == "teacher" else "student"), "precision": precision,
            "seed": None if stage == "teacher" else seed,
            "source": {"kind": kind, "path": path, "sha256": sha or S["ckpt_sha"]},
            "converted_artifact_sha256": converted if precision in RG.INT8 else None,
            "cells": list(cells)}


def write_inventory(doc) -> Path:
    p = S["work"] / f"inventory_{next(_SEQ)}.json"
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return p


def grid_argv(inv, out_root, *, split="val", plan_only=False, confirm=None, values=None,
              **over) -> list:
    v = dict(S["v_prov"]["values"])
    v.update(values or {})
    argv = ["--inventory", str(inv), "--split", split, "--artifact-status", "provisional",
            "--corruption-cache", str(over.pop("cache", S["c_prov"])),
            "--verification-record", str(over.pop("record", S["v_prov"]["record"])),
            "--expected-cache-sha256", v["canonical_manifest_sha256"],
            "--expected-inferential-cells-sha256", v["inferential_cells_sha256"],
            "--expected-manifest-file-sha256", v["manifest_file_sha256"],
            "--expected-verification-record-sha256", v["verification_record_sha256"],
            "--input-root", str(S["inputs"]), "--out-root", str(out_root), "--max-samples", "3"]
    if confirm or (confirm is None and split == "test"):
        argv.append("--confirm-test-split")
    if plan_only:
        argv.append("--plan-only")
    for k, val in over.items():
        argv += [f"--{k.replace('_', '-')}", str(val)]
    return argv


class FakeProc:
    def __init__(self, rc):
        self.returncode, self.pid = rc, None

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


def inprocess_launcher(calls: list, fail=None):
    """Runs evaluate_corruptions.main in this process; `fail(argv, calls)` makes a launch write
    nothing and exit 1 (or call a writer that leaves a partial output)."""
    def launch(argv, log_path):
        calls.append(list(argv))
        if fail is not None:
            rc = fail(argv, calls)
            if rc is not None:
                return FakeProc(rc)
        threads = torch.get_num_threads()
        try:
            with open(log_path, "w", encoding="utf-8") as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                rc = EC.main(argv[3:])
        finally:
            torch.set_num_threads(threads)
        return FakeProc(rc)
    return launch


def grid(argv, launcher=None, env_kv=None) -> tuple[int, str, str]:
    """RG.main in this process, with PLANTSEG_IMAGE_DIGEST unset unless the case sets it (the
    smoke must not depend on the variable a scoring container carries)."""
    err, out = io.StringIO(), io.StringIO()
    with env(**{"PLANTSEG_IMAGE_DIGEST": None, **(env_kv or {})}), \
            contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        rc = RG.main(argv, launcher=launcher)
    return rc, err.getvalue(), out.getvalue()


def grid_outcome(argv, launcher=None, env_kv=None) -> tuple:
    """(rc, stderr, stdout), or ('raised', the exception) when the runner itself raised: a case
    that sees anything but its expected refusal has run its checks."""
    try:
        return grid(argv, launcher, env_kv)
    except Exception as e:                                   # noqa: BLE001 -- recorded
        return ("raised", f"{type(e).__name__}: {e}", "")


def log_events(out_root: Path) -> list:
    logs = sorted(out_root.glob("grid_run_*.jsonl"))
    return [json.loads(ln) for p in logs for ln in p.read_text(encoding="utf-8").splitlines()]


def grid_code(err: str) -> str | None:
    import re
    m = re.search(r"\[([a-z0-9_]+)\]", err)
    return m.group(1) if m else None


@case("R1")
def r1():
    """The TEST-session order: E1 and E6 at seed 42 first, then E2, E3, E4, E5, E7, then E1 and
    E6 at seeds 43 and 44; per model the 15 inferential cells, then the 9 descriptive; the
    teacher's lane the same."""
    full, inf = list(corruption_roles.CELL_IDS), list(corruption_roles.INFERENTIAL_CELL_IDS)
    rows = [model_row("tch", "teacher", reversed(full), path="t.pth")]
    for st in ("E7", "E5", "E4", "E3", "E2", "E6", "E1"):
        rows.append(model_row(f"{st.lower()}s42", st, reversed(full),
                              converted=("c" * 64 if RG.STAGE_PRECISION[st] in RG.INT8 else None)))
    for seed in (44, 43):
        rows += [model_row(f"e6s{seed}", "E6", inf, seed=seed, converted="c" * 64),
                 model_row(f"e1s{seed}", "E1", inf, seed=seed)]
    doc = inventory(rows)
    RG.check_scopes(doc)
    teacher, students = RG.plan(doc, "test")
    models = []
    for j in students:
        if not models or models[-1] != (j["stage"], j["seed"]):
            models.append((j["stage"], j["seed"]))
    roles_ok = all([j["cell"] for j in students if (j["stage"], j["seed"]) == k] ==
                   (inf + [c for c in full if c not in inf] if k[1] == 42 else inf)
                   for k in models)
    t_ok = [j["cell"] for j in teacher] == inf + [c for c in full if c not in inf]
    names = all(j["out_name"] == f"tday_corr_{j['file_id']}_{j['cell']}" for j in teacher + students)
    return (models == list(RG.STUDENT_ORDER) and roles_ok and t_ok and names), str(models[:4])


def e1_inventory(cells=("fog_s1", "fog_s3"), **kw):
    return write_inventory(inventory([model_row("e1v", "E1", cells)], **kw))


@case("R2")
def r2():
    """Done cells are skipped: a second run over the same out-root launches nothing and logs a
    skip for each cell (R2)."""
    out_root = fresh("r2")
    inv = e1_inventory()
    calls1, calls2 = [], []
    rc1, err1, _ = grid(grid_argv(inv, out_root), inprocess_launcher(calls1))
    rc2, err2, _ = grid(grid_argv(inv, out_root), inprocess_launcher(calls2))
    skips = [e for e in log_events(out_root) if e["kind"] == "skip"]
    return (rc1 == 0 and rc2 == 0 and len(calls1) == 2 and calls2 == [] and len(skips) == 2), \
        f"rc {rc1}/{rc2}, launches {len(calls1)}/{len(calls2)}, skips {len(skips)} {err1[-80:]}"


@case("R3a")
def r3a():
    """An artifact directory without its record -> artifact_without_record, nothing started."""
    out_root = fresh("r3a")
    (out_root / "val_corr_e1v_fog_s1").mkdir(parents=True)
    calls = []
    rc, err, _ = grid(grid_argv(e1_inventory(), out_root), inprocess_launcher(calls))
    return rc == 2 and grid_code(err) == "artifact_without_record" and calls == [], err[-90:]


@case("R3b")
def r3b():
    """A record without its artifact directory -> record_without_artifact, nothing started."""
    out_root = fresh("r3b")
    out_root.mkdir(parents=True)
    (out_root / "val_corr_e1v_fog_s1.corruption.json").write_text("{}", encoding="utf-8")
    calls = []
    rc, err, _ = grid(grid_argv(e1_inventory(), out_root), inprocess_launcher(calls))
    return rc == 2 and grid_code(err) == "record_without_artifact" and calls == [], err[-90:]


@case("R3c")
def r3c():
    """A done cell whose artifact no longer matches its record -> cell_does_not_verify."""
    out_root = fresh("r3c")
    inv = e1_inventory(cells=("fog_s1",))
    rc1, _, _ = grid(grid_argv(inv, out_root), inprocess_launcher([]))
    rec = out_root / "val_corr_e1v_fog_s1.corruption.json"
    doc = json.loads(rec.read_text(encoding="utf-8"))
    doc["artifact_sha256"]["summary.json"] = "0" * 64
    rec.write_text(json.dumps(doc), encoding="utf-8")
    calls = []
    rc, err, _ = grid(grid_argv(inv, out_root), inprocess_launcher(calls))
    return rc1 == 0 and rc == 2 and grid_code(err) == "cell_does_not_verify" and calls == [], \
        err[-90:]


@case("R4")
def r4():
    """One process per cell: two cells, two real evaluate_corruptions.py processes with distinct
    pids, each argv naming one cell; both artifacts and records verify."""
    out_root = fresh("r4")
    inv = e1_inventory(cells=("fog_s1", "gaussian_noise_s4"))
    with env(PLANTSEG_DATA_ROOT=str(S["val"]), PYTHONDONTWRITEBYTECODE="1"):
        rc, err, _ = grid(grid_argv(inv, out_root))
    starts = [e for e in log_events(out_root) if e["kind"] == "start"]
    row = model_row("e1v", "E1", ["fog_s1", "gaussian_noise_s4"])
    ok = all(RG.cell_state(out_root, {"out_name": e["out_dir"], "stage": "E1", "row": row,
                                      "corruption": e["cell"].rsplit("_s", 1)[0],
                                      "severity": int(e["cell"].rsplit("_s", 1)[1])},
                           S["v_prov"]["values"],
                           {"head": S["head"], "status": "provisional"}) == "done"
             for e in starts)
    return (rc == 0 and len(starts) == 2 and len({e["pid"] for e in starts}) == 2
            and {e["cell"] for e in starts} == {"fog_s1", "gaussian_noise_s4"} and ok), \
        f"rc {rc}, starts {[(e['cell'], e['pid']) for e in starts]} {err[-120:]}"


@case("R5")
def r5():
    """An orphan temporary directory of a cell -> orphan_temporary_directory, named; nothing
    started."""
    out_root = fresh("r5")
    orphan = out_root / ".val_corr_e1v_fog_s1.tmp-0123456789ab"
    orphan.mkdir(parents=True)
    calls = []
    rc, err, _ = grid(grid_argv(e1_inventory(), out_root), inprocess_launcher(calls))
    return (rc == 2 and grid_code(err) == "orphan_temporary_directory" and orphan.name in err
            and calls == [] and orphan.is_dir()), err[-120:]


@case("R6a")
def r6a():
    """A cell that wrote no artifact is re-run once into the same out-dir, and the re-run is
    logged; the grid completes."""
    out_root = fresh("r6a")
    calls = []
    state = {"failed": False}

    def first_fails(argv, calls):
        if argv[argv.index("--severity") + 1] == "1" and not state["failed"]:
            state["failed"] = True
            return 1
        return None
    rc, err, _ = grid(grid_argv(e1_inventory(), out_root), inprocess_launcher(calls, first_fails))
    reruns = [e for e in log_events(out_root) if e["kind"] == "rerun"]
    return (rc == 0 and len(calls) == 3 and len(reruns) == 1
            and reruns[0]["out_dir"] == "val_corr_e1v_fog_s1"), \
        f"rc {rc}, launches {len(calls)}, reruns {[r['out_dir'] for r in reruns]} {err[-80:]}"


@case("R6b")
def r6b():
    """A cell that fails twice -> STOP cell_failed_twice (exit 3)."""
    out_root = fresh("r6b")
    calls = []

    def always_fails(argv, calls):
        return 1 if argv[argv.index("--severity") + 1] == "1" else None
    rc, err, _ = grid(grid_argv(e1_inventory(cells=("fog_s1",)), out_root),
                      inprocess_launcher(calls, always_fails))
    stops = [e for e in log_events(out_root) if e["kind"] == "stop"]
    return (rc == 3 and len(calls) == 2 and stops and stops[0]["code"] == "cell_failed_twice"), \
        f"rc {rc}, launches {len(calls)}, {stops[:1]}"


@case("R6c")
def r6c():
    """A failed cell that left a partial output -> STOP cell_left_partial_output (exit 3)."""
    out_root = fresh("r6c")
    calls = []

    def partial(argv, calls):
        Path(argv[argv.index("--out-dir") + 1]).mkdir(parents=True)
        return 1
    rc, err, _ = grid(grid_argv(e1_inventory(cells=("fog_s1",)), out_root),
                      inprocess_launcher(calls, partial))
    stops = [e for e in log_events(out_root) if e["kind"] == "stop"]
    return (rc == 3 and len(calls) == 1 and stops
            and stops[0]["code"] == "cell_left_partial_output"), f"rc {rc}, {stops[:1]}"


def gate_case(name, code_wanted, doc, *, inv=None, split="val", confirm=None, out_root=None,
              env_kv=None, values=None, **over):
    @case(name)
    def _c():
        root = out_root() if callable(out_root) else (out_root or never(name.lower()))
        inv_path = write_inventory(inv()) if callable(inv) else (inv or e1_inventory())
        rc, err, _ = grid_outcome(grid_argv(inv_path, root, split=split, plan_only=True,
                                            confirm=confirm, values=values, **over),
                                  env_kv=env_kv)
        return rc == 2 and grid_code(err) == code_wanted, f"rc {rc}: {err[-100:]}"
    _c.__doc__ = doc
    return _c


def bad_inventories():
    rows = [model_row("e1v", "E1", ["fog_s1"])]
    return [inventory(rows, schema="tday-inventory/0"),
            inventory([model_row("e1_TeSt", "E1", ["fog_s1"])]),
            inventory(rows + [model_row("e1v", "E1", ["fog_s2"])]),
            inventory([model_row("e5v", "E5", ["fog_s1"], precision="qat_fakequant_off",
                                 converted=None)]),
            inventory([model_row("e1v", "E1", ["fog_s1"], path="../e1.pt")]),
            inventory([model_row("e1v\n", "E1", ["fog_s1"])])]


@case("R7c")
def r7c():
    """Malformed exports -> inventory_invalid: another schema, a file_id holding 'test', two rows of
    one model, a qat_fakequant_off row with cells, a source path leaving the input root, a file_id
    ending in a newline (the patterns are fullmatched)."""
    codes = []
    for doc in bad_inventories():
        rc, err, _ = grid_outcome(grid_argv(write_inventory(doc), never("r7c"), plan_only=True))
        codes.append(grid_code(err) if rc == 2 else f"rc {rc}")
    return codes == ["inventory_invalid"] * 6, str(codes)


gate_case("R7a", "test_split_unconfirmed", "--split test without --confirm-test-split.",
          split="test", confirm=False)
gate_case("R7b", "out_name_invalid", "An out-root holding 'TeSt' (never created).",
          out_root=lambda: S["work"] / "never_created" / f"grid_TeSt_{next(_SEQ)}")
gate_case("R7d", "inventory_scope_invalid",
          "Split test: E1 at seed 42 with only its 15 inferential cells (TL section 9).",
          inv=lambda: inventory([model_row("e1s42", "E1", corruption_roles.INFERENTIAL_CELL_IDS)]),
          split="test")
gate_case("R7d2", "inventory_scope_invalid",
          "Split test: E1 at seed 43 without E6 at seed 43 (the pair rule).",
          inv=lambda: inventory([model_row("e1s43", "E1", corruption_roles.INFERENTIAL_CELL_IDS,
                                           seed=43)]), split="test")
gate_case("R7e", "teacher_config_required", "A teacher row without --teacher-config.",
          inv=lambda: inventory([model_row("tch", "teacher", ["fog_s1"], path="t.pth")]))
gate_case("R7f", "evaluator_commit_mismatch", "An export of another evaluator commit.",
          inv=lambda: inventory([model_row("e1v", "E1", ["fog_s1"])], commit="0" * 40))
gate_case("R7g", "execution_context_invalid", "An execution context with T = 0.",
          inv=lambda: inventory([model_row("e1v", "E1", ["fog_s1"])], threads=0))
gate_case("R7h", "image_digest_mismatch", "PLANTSEG_IMAGE_DIGEST not the export's image.",
          inv=lambda: inventory([model_row("e1v", "E1", ["fog_s1"])],
                                image_digest="sha256:" + "c" * 64))
gate_case("R7i", "processes_threads_exceed_cpus", "P x T above the CPU count.",
          processes=(os.cpu_count() or 1) + 1)
gate_case("R7j", "recorded_value_format", "A recorded value that is not 64 hex characters.",
          values={"inferential_cells_sha256": "XYZ"})
gate_case("R7l", "out_root_inside_repo", "An out-root inside the repository (never created).",
          out_root=lambda: REPO / f"never_created_lcorr_grid_{next(_SEQ)}")
gate_case("R7b2", "out_name_invalid", "':' in a parent component of the out-root (never created).",
          out_root=lambda: S["work"] / "never_created" / f"grid:{next(_SEQ)}" / "root")
gate_case("R7b3", "out_name_invalid",
          "An out-root through a symlink whose target holds ':' (the resolved path is checked).",
          out_root=lambda: dangling_link("r7b3_link") / "root")
gate_case("R7n", "source_missing", "A source path inside the input root that is not a file.",
          inv=lambda: inventory([model_row("e1v", "E1", ["fog_s1"], path="missing.pt")]))
gate_case("R7p", "cell_argv_refused",
          "A grid every cell of which the evaluator would refuse (--max-samples 0), refused at "
          "start by the evaluator's own argument checks.", max_samples=0)


@case("R7e2")
def r7e2():
    """A teacher row whose --teacher-config is not a file -> teacher_config_required."""
    inv = write_inventory(inventory([model_row("tch", "teacher", ["fog_s1"], path="t.pth")]))
    rc, err, _ = grid_outcome(grid_argv(inv, never("r7e2"), plan_only=True,
                                        teacher_config=never("r7e2_cfg")))
    return rc == 2 and grid_code(err) == "teacher_config_required", f"rc {rc}: {err[-100:]}"


@case("R7h2")
def r7h2():
    """An official TEST grid whose export names no image (image_digest null, the variable unset)
    -> image_digest_invalid (IF section 1)."""
    inv = write_inventory(inventory([model_row("e1s42", "E1", corruption_roles.CELL_IDS)]))
    with patched(RG, "GOVERNED_REPO", S["repo_clean"]):
        rc, err, _ = grid_outcome(grid_argv(inv, never("r7h2"), split="test", plan_only=True,
                                            artifact_status="official"))
    return rc == 2 and grid_code(err) == "image_digest_invalid", f"rc {rc}: {err[-100:]}"


def fake_sysfs(root: Path, cores) -> Path:
    for cpu, core in enumerate(cores):
        topo = root / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        (topo / "physical_package_id").write_text("0\n", encoding="utf-8")
        (topo / "core_id").write_text(f"{core}\n", encoding="utf-8")
    return root


@case("R7i2")
def r7i2():
    """P x T is bounded by PHYSICAL cores (IF section 1): four CPUs on two cores refuse T = 3,
    which the logical count would allow; /proc/cpuinfo serves when sysfs has no topology."""
    sysfs = fake_sysfs(S["work"] / f"sysfs_{next(_SEQ)}", [0, 0, 1, 1])
    cpuinfo = S["work"] / f"cpuinfo_{next(_SEQ)}"
    cpuinfo.write_text("".join(f"processor\t: {i}\nphysical id\t: 0\ncore id\t\t: {i // 2}\n\n"
                               for i in range(4)), encoding="utf-8")
    from_info = RG.physical_cores({0, 1, 2, 3}, sysfs=never("no_sysfs"), cpuinfo=cpuinfo)
    inv = write_inventory(inventory([model_row("e1v", "E1", ["fog_s1"])], threads=3))
    with patched(RG, "SYSFS_CPU", sysfs), patched(RG, "host_cpus", lambda: {0, 1, 2, 3}):
        rc, err, _ = grid_outcome(grid_argv(inv, never("r7i2"), plan_only=True))
    return (from_info == 2 and rc == 2 and grid_code(err) == "processes_threads_exceed_cpus"), \
        f"cpuinfo {from_info}; rc {rc}: {err[-100:]}"


@case("R7m")
def r7m():
    """No topology to read (no sysfs, no /proc/cpuinfo) -> physical_cores_unknown."""
    with patched(RG, "SYSFS_CPU", never("sysfs")), patched(RG, "PROC_CPUINFO", never("cpuinfo")):
        rc, err, _ = grid_outcome(grid_argv(e1_inventory(), never("r7m"), plan_only=True))
    return rc == 2 and grid_code(err) == "physical_cores_unknown", f"rc {rc}: {err[-100:]}"


@case("R7o")
def r7o():
    """A cache that cannot serve the grid (a provisional grid on a smoke cache whose four recorded
    values are its own) -> cache_scope_mismatch at start, naming the evaluator's code."""
    rec = record_for(S["c_smoke"], S["m_smoke"])
    values = {"canonical_manifest_sha256": S["m_smoke"]["canonical_manifest_sha256"],
              "inferential_cells_sha256": S["m_smoke"]["inferential_cells_sha256"],
              "manifest_file_sha256": sha256_file(S["c_smoke"] / cc.MANIFEST_NAME),
              "verification_record_sha256": sha256_file(rec)}
    rc, err, _ = grid_outcome(grid_argv(e1_inventory(), never("r7o"), plan_only=True,
                                        values=values, cache=S["c_smoke"], record=rec))
    return (rc == 2 and grid_code(err) == "cache_scope_mismatch"
            and "[cache_status_insufficient]" in err), f"rc {rc}: {err[-120:]}"


@case("R7q")
def r7q():
    """An official grid in a checkout whose governed paths are dirty -> governed_paths_dirty at
    start, before any process parses a checkpoint."""
    with patched(RG, "GOVERNED_REPO", S["repo_dirty"]):
        rc, err, _ = grid_outcome(grid_argv(e1_inventory(), never("r7q"), plan_only=True,
                                            artifact_status="official"))
    return (rc == 2 and grid_code(err) == "governed_paths_dirty"
            and "scripts/probe.txt" in err), f"rc {rc}: {err[-100:]}"


@case("R7k")
def r7k():
    """Each of the four recorded values not the cache's -> recorded_values_mismatch."""
    codes = []
    for key in ("canonical_manifest_sha256", "inferential_cells_sha256", "manifest_file_sha256",
                "verification_record_sha256"):
        rc, err, _ = grid(grid_argv(e1_inventory(), never("r7k"), plan_only=True,
                                    values={key: "d" * 64}))
        codes.append(grid_code(err) if rc == 2 else f"rc {rc}")
    return codes == ["recorded_values_mismatch"] * 4, str(codes)


@case("R8")
def r8():
    """--plan-only runs every start gate, prints the ordered cells and creates nothing."""
    out_root = never("r8")
    rc, err, out = grid(grid_argv(e1_inventory(cells=("fog_s5", "fog_s1", "gaussian_noise_s4")),
                                  out_root, plan_only=True))
    plan = [json.loads(ln) for ln in out.splitlines()]
    return (rc == 0 and [p["cell"] for p in plan] == ["fog_s1", "gaussian_noise_s4", "fog_s5"]
            and not out_root.exists()), f"rc {rc} {[p['cell'] for p in plan]} {err[-60:]}"


@case("R3d")
def r3d():
    """A done cell whose four files verify but whose record names another cache checksum ->
    cell_does_not_verify (the record-identity comparison, past the file check)."""
    out_root = fresh("r3d")
    inv = e1_inventory(cells=("fog_s1",))
    rc1, _, _ = grid(grid_argv(inv, out_root), inprocess_launcher([]))
    rec = out_root / "val_corr_e1v_fog_s1.corruption.json"
    doc = json.loads(rec.read_text(encoding="utf-8"))
    doc["cache"]["canonical_manifest_sha256"] = "e" * 64
    rec.write_bytes(CO.record_bytes(doc))
    calls = []
    rc, err, _ = grid_outcome(grid_argv(inv, out_root), inprocess_launcher(calls))
    return (rc1 == 0 and rc == 2 and grid_code(err) == "cell_does_not_verify"
            and "canonical_manifest_sha256" in err and calls == []), f"rc {rc}: {err[-110:]}"


@case("R3e")
def r3e():
    """A done cell counts only for this export row's identity and this checkout's commit: a record
    of another evaluator commit, or an export whose source is another file ->
    cell_does_not_verify, naming the field."""
    out_root = fresh("r3e")
    inv = e1_inventory(cells=("fog_s1",))
    rc1, _, _ = grid(grid_argv(inv, out_root), inprocess_launcher([]))
    rec = out_root / "val_corr_e1v_fog_s1.corruption.json"
    original = rec.read_bytes()
    doc = json.loads(original)
    doc["evaluator_commit"] = "0" * 40
    rec.write_bytes(CO.record_bytes(doc))
    rc_a, err_a, _ = grid_outcome(grid_argv(inv, out_root), inprocess_launcher([]))
    rec.write_bytes(original)
    other = write_inventory(inventory([model_row("e1v", "E1", ["fog_s1"], sha="9" * 64)]))
    rc_b, err_b, _ = grid_outcome(grid_argv(other, out_root), inprocess_launcher([]))
    ok = (rc1 == 0 and rc_a == 2 and grid_code(err_a) == "cell_does_not_verify"
          and "evaluator_commit" in err_a and rc_b == 2
          and grid_code(err_b) == "cell_does_not_verify" and "checkpoint_sha256" in err_b)
    return ok, f"{err_a[-70:]} | {err_b[-70:]}"


@case("R5b")
def r5b():
    """An orphan of an out-dir outside the current plan is refused and named too."""
    out_root = fresh("r5b")
    orphan = out_root / ".val_corr_dropped_fog_s1.tmp-0123456789ab"
    orphan.mkdir(parents=True)
    calls = []
    rc, err, _ = grid_outcome(grid_argv(e1_inventory(), out_root), inprocess_launcher(calls))
    return (rc == 2 and grid_code(err) == "orphan_temporary_directory" and orphan.name in err
            and calls == [] and orphan.is_dir()), f"rc {rc}: {err[-100:]}"


def prior_log(out_root: Path, events) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"kind": "header", "utc": "2026-10-16T00:00:00.000000Z"})]
    lines += [json.dumps({"utc": "2026-10-16T00:00:01.000000Z", **e}) for e in events]
    (out_root / "grid_run_20261016T000000000000Z.jsonl").write_text("\n".join(lines) + "\n",
                                                                    encoding="utf-8")


@case("R6d")
def r6d():
    """A cell with one failure logged by an earlier run gets one more attempt (LE10 across
    restarts): failing again is a STOP cell_failed_twice after a single launch."""
    out_root = fresh("r6d")
    prior_log(out_root, [{"kind": "rerun", "out_dir": "val_corr_e1v_fog_s1", "rc": 1}])
    calls = []
    rc, err, _ = grid_outcome(grid_argv(e1_inventory(cells=("fog_s1",)), out_root),
                              inprocess_launcher(calls, lambda argv, calls: 1))
    stops = [e for e in log_events(out_root) if e["kind"] == "stop"]
    return (rc == 3 and len(calls) == 1 and bool(stops)
            and stops[-1]["code"] == "cell_failed_twice"), f"rc {rc}, launches {len(calls)}"


@case("R6e")
def r6e():
    """A cell with two failures logged by earlier runs -> refused at start (cell_failed_twice);
    nothing starts."""
    out_root = fresh("r6e")
    prior_log(out_root, [{"kind": "rerun", "out_dir": "val_corr_e1v_fog_s1", "rc": 1},
                         {"kind": "stop", "code": "cell_failed_twice",
                          "out_dir": "val_corr_e1v_fog_s1"}])
    calls = []
    rc, err, _ = grid_outcome(grid_argv(e1_inventory(cells=("fog_s1",)), out_root),
                              inprocess_launcher(calls))
    return rc == 2 and grid_code(err) == "cell_failed_twice" and calls == [], \
        f"rc {rc}: {err[-100:]}"


class TickProc:
    """A fake process that finishes on its second poll; `active` holds what is running."""

    def __init__(self, active: dict, key):
        self.active, self.key, self.polls, self.returncode, self.pid = active, key, 0, None, None

    def poll(self):
        self.polls += 1
        if self.polls >= 2 and self.returncode is None:
            self.returncode = 0
            self.active.pop(self.key, None)
        return self.returncode

    def wait(self):
        self.returncode = 0 if self.returncode is None else self.returncode
        self.active.pop(self.key, None)
        return self.returncode


@case("R9")
def r9():
    """Scheduling at P = 3 with the teacher, two INT8 rows (E4, E7) and E1: a teacher cell starts
    in the first wave, the teacher never runs twice while student cells wait, no two INT8 cells
    run at once (IF section 1's interim rule), and every cell starts once."""
    e4_sha, e7_sha = sha256_file(S["inputs"] / "e4.json"), sha256_file(S["inputs"] / "e7.json")
    rows = [model_row("tch", "teacher", ["fog_s1", "fog_s3"], path="t.pth", sha=S["tk_sha"]),
            model_row("e4s42", "E4", ["fog_s1", "fog_s3"], path="e4.json", sha=e4_sha,
                      converted="c" * 64),
            model_row("e7s42", "E7", ["fog_s1"], path="e7.json", sha=e7_sha, converted="c" * 64),
            model_row("e1s42", "E1", ["fog_s1", "fog_s3"])]
    inv = write_inventory(inventory(rows))
    active, starts, snaps = {}, [], []

    def launch(argv, log_path):
        stage = argv[argv.index("--stage") + 1]
        prec = argv[argv.index("--precision") + 1]
        key = argv[argv.index("--out-dir") + 1]
        active[key] = (stage, prec)
        starts.append(stage)
        snaps.append({"int8": sum(1 for _, p in active.values() if p in RG.INT8),
                      "teacher": sum(1 for s, _ in active.values() if s == "teacher"),
                      "students_left": 5 - sum(1 for s in starts if s != "teacher")})
        return TickProc(active, key)
    args = RG.build_parser().parse_args(grid_argv(inv, fresh("r9"), processes=3,
                                                  teacher_config=TEACHER_CONFIG))
    err = io.StringIO()
    with patched(RG, "finished_ok", lambda *a, **k: True), env(PLANTSEG_IMAGE_DIGEST=None), \
            contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        code, rc = outcome(RG.run_grid, args, launcher=launch, poll_seconds=0)
    ok = (code == "ok" and rc == 0 and len(starts) == 7 and starts[0] == "teacher"
          and max(s["int8"] for s in snaps) == 1
          and all(s["teacher"] <= 1 for s in snaps if s["students_left"] > 0))
    return ok, f"{code} rc {rc}; order {starts}; int8 max {max((s['int8'] for s in snaps), default=0)}"


@case("R10")
def r10():
    """The runner's argv per row kind (LE2): FP32 -> --checkpoint with H = source.sha256 and no
    H2; INT8 -> --provenance with H = source.sha256 and H2 = converted_artifact_sha256; the
    teacher -> --teacher-config and --teacher-ckpt-sha256 = H = source.sha256; every cell ->
    --threads T."""
    rows = [model_row("e1s42", "E1", ["fog_s1"], sha="1" * 64),
            model_row("e4s42", "E4", ["fog_s1"], path="e4.json", sha="2" * 64, converted="3" * 64),
            model_row("tch", "teacher", ["fog_s1"], path="t.pth", sha="4" * 64)]
    doc = inventory(rows, threads=3)
    teacher, students = RG.plan(doc, "val")
    args = RG.build_parser().parse_args(grid_argv(write_inventory(doc), never("r10"),
                                                  teacher_config=TEACHER_CONFIG))
    ctx = {"threads": 3, "batch_size": 1, "device": "cpu"}

    def flags(argv):
        return {a: (argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--")
                    else True) for i, a in enumerate(argv) if a.startswith("--")}
    got = {j["file_id"]: flags(RG.cell_argv(j, args, ctx, S["inputs"], never("r10_out")))
           for j in teacher + students}
    e1, e4, t = got.get("e1s42", {}), got.get("e4s42", {}), got.get("tch", {})
    ok = (e1.get("--expected-checkpoint-sha256") == "1" * 64 and "--checkpoint" in e1
          and "--expected-converted-sha256" not in e1 and e1.get("--threads") == "3"
          and e4.get("--expected-checkpoint-sha256") == "2" * 64 and "--provenance" in e4
          and e4.get("--expected-converted-sha256") == "3" * 64 and e4.get("--threads") == "3"
          and t.get("--teacher-ckpt-sha256") == t.get("--expected-checkpoint-sha256") == "4" * 64
          and "--teacher-config" in t and t.get("--threads") == "3")
    return ok, str({k: sorted(v) for k, v in got.items()})[:220]


@case("R11")
def r11():
    """Relative --corruption-cache and --verification-record from another working directory: the
    start gates and every cell's argv name the same absolute files."""
    inv = e1_inventory()
    rel_cache = os.path.relpath(S["c_prov"], S["work"])
    rel_rec = os.path.relpath(S["v_prov"]["record"], S["work"])
    args = RG.build_parser().parse_args(grid_argv(inv, never("r11"), cache=rel_cache,
                                                  record=rel_rec))
    with contextlib.chdir(S["work"]), env(PLANTSEG_IMAGE_DIGEST=None), \
            contextlib.redirect_stdout(io.StringIO()):
        code, ready = outcome(RG.prepare, args)
    if code != "ok":
        return False, f"prepare: {code}: {str(ready)[:90]}"
    cell = RG.cell_argv(ready["students"][0], args, ready["context"], ready["input_root"],
                        ready["out_root"])
    cache = cell[cell.index("--corruption-cache") + 1]
    rec = cell[cell.index("--verification-record") + 1]
    return (cache == str(S["c_prov"].resolve())
            and rec == str(Path(S["v_prov"]["record"]).resolve())), f"{cache} | {rec}"


@case("R13")
def r13():
    """A second runner on an out-root whose lock is held -> out_root_locked; nothing starts."""
    out_root = fresh("r13")
    out_root.mkdir(parents=True)
    fd = os.open(out_root / RG.LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    calls = []
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        rc, err, _ = grid_outcome(grid_argv(e1_inventory(cells=("fog_s1",)), out_root),
                                  inprocess_launcher(calls))
    finally:
        os.close(fd)
    return rc == 2 and grid_code(err) == "out_root_locked" and calls == [], \
        f"rc {rc}: {err[-100:]}"


@case("R14")
def r14():
    """HEAD moves during the session -> STOP head_changed before the next start (exit 3)."""
    out_root = fresh("r14")
    real = RG.git_head
    seen = []

    def moving():
        seen.append(1)
        return "f" * 40 if len(seen) == 2 else real()
    calls = []
    with patched(RG, "git_head", moving):
        rc, err, _ = grid_outcome(grid_argv(e1_inventory(cells=("fog_s1",)), out_root),
                                  inprocess_launcher(calls))
    stops = [e for e in log_events(out_root) if e["kind"] == "stop"] if out_root.exists() else []
    return (rc == 3 and calls == [] and bool(stops) and stops[-1]["code"] == "head_changed"), \
        f"rc {rc}, launches {len(calls)}: {err[-80:]}"


@case("R15")
def r15():
    """An error inside the run loop (a launch that raises) -> STOP runner_error, recorded, exit 3."""
    out_root = fresh("r15")

    def broken(argv, log_path):
        raise OSError(12, "Cannot allocate memory (simulated fork failure)")
    rc, err, _ = grid_outcome(grid_argv(e1_inventory(cells=("fog_s1",)), out_root), broken)
    stops = [e for e in log_events(out_root) if e["kind"] == "stop"] if out_root.exists() else []
    return (rc == 3 and bool(stops) and stops[-1]["code"] == "runner_error"), \
        f"rc {rc}: {err[-100:]}"


# ---------------------------------------------------------------- mutations
def mutations() -> None:
    sk = lambda code: skip(EC, "_refuse", code)        # noqa: E731
    co = lambda code: skip(CO, "_fail", code)           # noqa: E731
    rg = lambda code: skip(RG, "_refuse", code)         # noqa: E731
    # the plan's eight
    real_getitem = CO.PlantSegCorruptedEvalDataset.__getitem__
    real_expected = AD.build_expected_manifest_for

    def suffixed_getitem(self, i):
        sample = real_getitem(self, i)
        sample["image_id"] = f"{sample['image_id']}__fog_s3"
        return sample

    def suffixed_expected(split, idx):
        from src.eval.evaluate import ManifestEntry
        return [ManifestEntry(e.manifest_index, f"{e.image_id}__fog_s3", e.clean_image_id)
                for e in real_expected(split, idx)]
    mutant("M1 suffixed image_id", ["E2", "E3", "W6"],
           (CO.PlantSegCorruptedEvalDataset, "__getitem__", suffixed_getitem),
           (AD, "build_expected_manifest_for", suffixed_expected),
           (EC, "check_cache_sources", lambda *a, **k: None))   # past C1: E2, E3 decide
    real_canvas = CO.corrupted_canvas

    def zero_padding(stem, idx, clean, cell):
        out = real_canvas(stem, idx, clean, cell)
        top, left, h, w = CO.window(cell.source(stem)["resized_shape"])
        z = np.zeros_like(out)
        z[top:top + h, left:left + w] = out[top:top + h, left:left + w]
        return z
    mutant("M2 zero padding", ["W1", "W3"], (CO, "corrupted_canvas", zero_padding))
    real_finalize = CO.finalize

    def unnormalised(img, msk):
        t, m = real_finalize(img, msk)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0, m
    mutant("M3 normalisation skipped", ["W1"], (CO, "finalize", unnormalised))
    real_core = CO.core_preprocess

    def wrong_target(im, mk):
        canvas, mcanvas = real_core(im, mk)
        return canvas, np.where(mcanvas == 255, 255, 0).astype(np.int64)
    mutant("M4 target from the wrong path", ["W2", "E4"], (CO, "core_preprocess", wrong_target))
    mutant("M5 no source-digest check", ["W5"], co("source_pixels_mismatch"))
    mutant("M6 teacher sha optional", ["T1"], sk("teacher_ckpt_sha256_required"))
    real_cell_cls = EC.CacheCell

    def wrong_cell(root, manifest, corruption, severity, load=None):
        s = 2 if int(severity) == 3 else int(severity)
        return real_cell_cls(root, manifest, corruption, s,
                             load=EC.cached_image_loader(root, manifest, corruption, s))
    mutant("M7 the wrong cell loaded", ["E8"], (EC, "CacheCell", wrong_cell))

    def from_cache(split, idx):
        from src.eval.evaluate import ManifestEntry
        m = json.loads((S["c1_root"] / cc.MANIFEST_NAME).read_text(encoding="utf-8"))
        by = {s["manifest_index"]: s["image_id"] for s in m["sources"]}
        return [ManifestEntry(i, by[i], by[i]) for i in sorted(idx)]
    mutant("M8 expected manifest taken from the cache", ["C1"],
           (AD, "build_expected_manifest_for", from_cache))
    # LE1: the wrapper regenerates instead of reading the cache
    import src.vendor.imagecorruptions as V

    def regenerating(stem, idx, clean, cell):
        rec = cell.source(stem)
        top, left, h, w = CO.window(rec["resized_shape"])
        item = V.apply_corruption(np.ascontiguousarray(clean[top:top + h, left:left + w]), "fog",
                                  3, image_id=stem)[0]
        out = clean.copy()
        out[top:top + h, left:left + w] = item
        return out
    mutant("M-X1 the wrapper regenerates (LE1)", ["X1"], (CO, "corrupted_canvas", regenerating))
    # every new refusal code of evaluate_corruptions.py, disabled one at a time
    for code, targets in (
            ("stage_not_in_robustness", ["ST1"]), ("stage_unknown", ["ST1"]),
            ("qat_fakequant_off_clean_only", ["ST8a"]), ("precision_stage_mismatch", ["ST8b"]),
            ("protocol_canvas_only", ["ST5"]), ("checkpoint_required", ["ST9"]),
            ("expected_cache_sha256_format", ["ST10"]),
            ("teacher_ckpt_sha256_format", ["T2"]), ("teacher_ckpt_sha256_not_teacher", ["T3"]),
            ("expected_checkpoint_sha256_required", ["ST7a"]),
            ("expected_checkpoint_sha256_format", ["ST7b"]),
            ("expected_checkpoint_sha256_teacher_disagrees", ["T6a"]),
            ("expected_checkpoint_sha256_mismatch", ["ST7c", "T6b"]),
            ("source_changed_during_validation", ["ST7d"]),
            ("expected_converted_sha256_required", ["ST7e"]),
            ("expected_converted_sha256_format", ["ST7f"]),
            ("expected_converted_sha256_not_int8", ["ST7g"]),
            ("expected_converted_sha256_mismatch", ["ST7h"]),
            ("converted_changed_before_load", ["ST7i"]),
            ("threads_invalid", ["ST7j"]), ("threads_required", ["ST7k"]),
            ("threads_not_applied", ["ST7l"]), ("threads_drifted", ["ST7m"]),
            ("checkpoint_changed_after_validation", ["ST7n"]),
            ("out_dir_name_invalid", ["N1", "N1b", "N1c"]), ("out_dir_inside_repo", ["G8"]),
            ("out_dir_exists", ["E9"]), ("record_exists", ["G7"]),
            ("cache_root_missing", ["C4"]), ("cache_split_mismatch", ["G9"]),
            ("cache_cell_absent", ["W7"]), ("cache_incomplete_for_official", ["G5b"]),
            ("cache_status_insufficient", ["G5c", "G6a"]), ("cache_pin_missing", ["G5d"]),
            ("expected_cache_sha256_required", ["G5e", "G6b"]),
            ("cache_checksum_mismatch", ["G5f", "C3"]),
            ("protocol_changed_since_generation", ["G10"]),
            ("verification_record_required", ["G5g", "G6c"]),
            ("verification_record_mismatch", ["G5h", "G5i"]),
            ("cache_sources_mismatch", ["C1"]), ("verification_count_mismatch", ["C5"]),
            ("nmf_draws_mismatch", ["T7"]), ("nmf_stream_mismatch", ["T8"])):
        mutant(f"M-{code}", targets, sk(code))
    # DL-24: the printed line, the created-path rule, the record writer's cleanup, E8's fields
    mutant("M-P1 refusals printed after their type", ["P1"],
           (EC, "refusal_line", lambda e: f"{type(e).__name__}: {e}"))

    def basename_only(path):
        raw = str(path)
        return ("holds 'test'" if "test" in raw.lower() else
                CO.path_text_problem(os.path.basename(os.path.abspath(raw))))
    mutant("M-created path checked in its last component only", ["N1b", "R7b2"],
           (CO, "created_path_problem", basename_only), (EC, "created_path_problem", basename_only))

    def unresolved(path):
        return next((p for p in (CO.path_text_problem(t) for t in (str(path),
                                                                    os.path.abspath(str(path))))
                     if p), None)
    mutant("M-created path never resolved", ["N1c", "R7b3"],
           (CO, "created_path_problem", unresolved), (EC, "created_path_problem", unresolved))
    mutant("M-record writer leaves its temporary file", ["W12"],
           (CO, "_discard_tmp", lambda tmp: None))
    real_build = CO.build_cell_record

    def wrong_digest(**kw):
        rec = real_build(**kw)
        rec["cache"]["inferential_cells_sha256"] = "0" * 64
        return rec
    mutant("M-record carries another inferential digest", ["E8"],
           (EC, "build_cell_record", wrong_digest))
    # LE22, LE23
    mutant("M-the old official INT8 gate, calibration only (LE22)", ["ST11a"],
           (SA, "int8_official_error", SA.int8_official_calibration_error))
    mutant("M-descriptive stages left to stage_unknown (LE23)", ["ST1"],
           (EC, "ARTIFACT_STAGES", EC.STAGES + EC.NOT_IN_ROBUSTNESS))
    # every new refusal code of src/eval/corrupted.py
    for code, targets in (("cache_cell_absent", ["W7"]), ("cache_source_absent", ["W11"]),
                          ("cache_source_index_mismatch", ["W9"]),
                          ("canvas_window_mismatch", ["W10"]),
                          ("corrupted_shape_mismatch", ["W4"]),
                          ("verification_count_mismatch", ["C5"]), ("record_exists", ["G7"])):
        mutant(f"M-{code} (corrupted.py)", targets, co(code))
    # the grid runner's start gates and STOPs
    for code, targets in (("test_split_unconfirmed", ["R7a"]),
                          ("out_name_invalid", ["R7b", "R7b2", "R7b3"]),
                          ("inventory_invalid", ["R7c"]),
                          ("inventory_scope_invalid", ["R7d", "R7d2"]),
                          ("teacher_config_required", ["R7e", "R7e2"]),
                          ("source_missing", ["R7n"]),
                          ("evaluator_commit_mismatch", ["R7f"]),
                          ("governed_paths_dirty", ["R7q"]),
                          ("execution_context_invalid", ["R7g"]),
                          ("image_digest_invalid", ["R7h2"]),
                          ("image_digest_mismatch", ["R7h"]),
                          ("physical_cores_unknown", ["R7m"]),
                          ("processes_threads_exceed_cpus", ["R7i", "R7i2"]),
                          ("recorded_value_format", ["R7j"]),
                          ("recorded_values_mismatch", ["R7k"]),
                          ("cache_scope_mismatch", ["R7o"]),
                          ("out_root_inside_repo", ["R7l"]),
                          ("cell_argv_refused", ["R7p"]),
                          ("orphan_temporary_directory", ["R5", "R5b"]),
                          ("artifact_without_record", ["R3a"]),
                          ("record_without_artifact", ["R3b"]),
                          ("cell_does_not_verify", ["R3c", "R3d"]),
                          ("cell_failed_twice", ["R6e"]),
                          ("out_root_locked", ["R13"])):
        mutant(f"M-{code} (runner)", targets, rg(code))
    for code, targets in (("cell_failed_twice", ["R6b"]), ("cell_left_partial_output", ["R6c"]),
                          ("head_changed", ["R14"]), ("runner_error", ["R15"])):
        mutant(f"M-{code} (runner STOP)", targets, skip(RG, "_stop", code))
    # DL-24: the runner's semantics
    mutant("M-P x T against logical CPUs", ["R7i2"],
           (RG, "physical_cores", lambda cpus, sysfs=None, cpuinfo=None: len(cpus)))
    mutant("M-done cells compared on files only", ["R3d", "R3e"],
           (RG, "done_problems", lambda *a, **k: []))
    mutant("M-inputs never resolved", ["R11"], (RG, "resolve_inputs", lambda args: None))
    mutant("M-earlier failures ignored", ["R6d", "R6e"],
           (RG, "prior_failures", lambda out_root: RG.collections.Counter()))
    mutant("M-INT8 cells run together", ["R9"], (RG, "INT8_ONE_AT_A_TIME", False))
    mutant("M-teacher queued after the students", ["R9"],
           (RG, "pick_job", lambda t, s, r, p: s.popleft() if s else (t.popleft() if t else None)))
    mutant("M-orphans of planned cells only", ["R5b"],
           (RG, "start_orphans",
            lambda out_root, jobs: RG.orphans(out_root, [j["out_name"] for j in jobs])))
    mutant("M-file ids matched with a trailing newline", ["R7c"],
           (RG, "FILE_ID", re.compile(r"[A-Za-z0-9._-]+\n?")))

    def drop_flag(flag):
        real = RG.cell_argv

        def dropped(*a, **k):
            argv = real(*a, **k)
            if flag in argv:
                i = argv.index(flag)
                del argv[i:i + 2]
            return argv
        return (RG, "cell_argv", dropped)
    for flag in ("--expected-checkpoint-sha256", "--expected-converted-sha256", "--threads",
                 "--teacher-ckpt-sha256"):
        mutant(f"M-cell argv without {flag}", ["R10"], drop_flag(flag))


LABELS = {
    "W1": "W1 identity: the clean window as the item gives finalize(core_preprocess) bitwise",
    "W2": "W2 the target is the clean path's (PlantSegEvalDataset)",
    "W3": "W3 the padding is the clean canvas's; the window holds the cached item",
    "W4": "W4 an item of another shape, or (h, w, 1) -> corrupted_shape_mismatch",
    "W5": "W5 a source digest that is not the clean window -> source_pixels_mismatch, no read",
    "W6": "W6 image_id == clean_image_id == the stem; manifest_index the listing's",
    "W7": "W7 an absent cell -> cache_cell_absent (the cache gate and the cell)",
    "W8": "W8 a tampered item -> cache_file_hash_mismatch, never regenerated",
    "W9": "W9 a cached source at another manifest_index -> cache_source_index_mismatch",
    "W10": "W10 a recorded window not the image's -> canvas_window_mismatch (two forms)",
    "W11": "W11 a stem the cache does not record -> cache_source_absent",
    "X1": "X1 LE1: scoring calls no corruption or generator; one cache read per row",
    "E1": "E1 a smoke run (val, 3 rows, fog s3) writes a verifying artifact and record",
    "E2": "E2 bare stems and the condition in every row (DL-74)",
    "E3": "E3 the split digest equals the clean runner's on the same subset",
    "E4": "E4 the AM-5 flags and gt_disease_classes equal the clean artifact's",
    "E5": "E5 per-image values of 2 images equal a manual recomputation",
    "E6": "E6 run.eval_runtime holds the 26 keys and workers 0",
    "E7": "E7 fog s5 scored (descriptive_only); gaussian_noise s5 -> severity_excluded",
    "E8": "E8 the cell record: keys, four hashes, checksum, cell and inferential digests, source count, role, counts, times, run, stage, condition, script",
    "E9": "E9 a second run into an existing out-dir -> out_dir_exists",
    "E10": "E10 config_sha256 differs between two cells only through the condition",
    "E11": "E11 clean + 15 cells load (NONOFFICIAL_SMOKE); assemble_miou_c n = clean n",
    "E12": "E12 INT8 E4 on QNNPACK with both identity flags: precision, backend, artifact sha",
    "T1": "T1 the teacher without --teacher-ckpt-sha256 -> teacher_ckpt_sha256_required",
    "T2": "T2 a malformed --teacher-ckpt-sha256 -> teacher_ckpt_sha256_format",
    "T3": "T3 --teacher-ckpt-sha256 on a student -> teacher_ckpt_sha256_not_teacher",
    "T4": "T4 a teacher hash mismatch -> teacher_hash_mismatch, never parsed",
    "T5": "T5 the teacher (injected builder): batch 1, CPU, NMF draws = rows = the clean pass's, the stream M4-V seed 42 in the log line",
    "T6a": "T6a H and --teacher-ckpt-sha256 differ -> expected_checkpoint_sha256_teacher_disagrees",
    "T6b": "T6b the teacher's file not H (H == t) -> expected_checkpoint_sha256_mismatch, unparsed",
    "T7": "T7 two NMF draws per image -> nmf_draws_mismatch, nothing written",
    "ST1": "ST1 A, F, G and E6-E2, E7-E2, E6-R1, E6-R2 (AM-23), E6-KD (AM-3) -> stage_not_in_robustness (LE23); E9 and TEACHER -> stage_unknown",
    "ST2": "ST2 INT8 without --provenance -> refused (evaluate_model.py's text)",
    "ST3": "ST3 INT8 on cuda -> refused (CPU only)",
    "ST4": "ST4 E3 handed a checkpoint without its stage -> stage_undeclared",
    "ST5": "ST5 --protocol upstream -> protocol_canvas_only (G11)",
    "ST6": "ST6 parity: 15 argument rows refused with identical text by both runners",
    "ST7a": "ST7a under R without --expected-checkpoint-sha256 -> ..._required",
    "ST7b": "ST7b a malformed --expected-checkpoint-sha256 -> ..._format",
    "ST7c": "ST7c the checkpoint not H -> expected_checkpoint_sha256_mismatch, never loaded",
    "ST7d": "ST7d the checkpoint changes during validation -> source_changed_during_validation",
    "ST7e": "ST7e under R, INT8 without --expected-converted-sha256 -> ..._required",
    "ST7f": "ST7f a malformed --expected-converted-sha256 -> ..._format",
    "ST7g": "ST7g --expected-converted-sha256 on FP32 -> expected_converted_sha256_not_int8",
    "ST7h": "ST7h the converted artifact not H2 -> expected_converted_sha256_mismatch, unloaded",
    "ST7i": "ST7i the converted artifact changes before the load -> converted_changed_before_load",
    "ST7j": "ST7j --threads 0 and x -> threads_invalid",
    "ST7k": "ST7k under R without --threads -> threads_required",
    "ST7l": "ST7l torch.set_num_threads does not take -> threads_not_applied",
    "ST7m": "ST7m eval_runtime threads differ from T -> threads_drifted, nothing written",
    "ST8a": "ST8a --precision qat_fakequant_off -> qat_fakequant_off_clean_only (LE15)",
    "ST8b": "ST8b a precision not the stage's -> precision_stage_mismatch",
    "ST9": "ST9 FP32 without --checkpoint -> checkpoint_required",
    "ST10": "ST10 a malformed --expected-cache-sha256 -> expected_cache_sha256_format",
    "ST11a": "ST11a under R, an E4 TorchScript artifact without its PTQ record -> [ptq_record_missing], before any cache read (LE22)",
    "ST11b": "ST11b under R, the state_dict E4 -> [legacy_ptq_artifact], before any cache read (LE22)",
    "ST11c": "ST11c under R, a QAT E5 passes the official INT8 gate and reaches the cache step (LE22)",
    "G1": "G1 --split test without --confirm-test-split, before any path is read",
    "G2": "G2 --split test refuses smoke",
    "G3": "G3 --split test refuses --max-samples",
    "G5a": "G5a official on val -> refused",
    "G5b": "G5b official, 6 cached sources -> cache_incomplete_for_official",
    "G5c": "G5c official on a provisional cache -> cache_status_insufficient",
    "G5d": "G5d official on a cache without its pin -> cache_pin_missing",
    "G5e": "G5e official without --expected-cache-sha256 -> expected_cache_sha256_required",
    "G5f": "G5f official, another checksum -> cache_checksum_mismatch; junk record -> mismatch",
    "G5g": "G5g official without --verification-record -> verification_record_required",
    "G5h": "G5h official with another cache's record -> verification_record_mismatch",
    "G6a": "G6a provisional on a smoke cache -> cache_status_insufficient",
    "G6b": "G6b provisional without --expected-cache-sha256 -> ..._required",
    "G6c": "G6c provisional without --verification-record -> ..._required",
    "G6d": "G6d provisional with every gate: the record names the verifier's record sha256",
    "G7": "G7 a stray record -> record_exists (the runner and the record writer)",
    "G8": "G8 an out-dir inside the repository -> out_dir_inside_repo, nothing created",
    "G9": "G9 the pin's split not the requested split -> cache_split_mismatch (LE3; items present)",
    "G10": "G10 another protocol file -> protocol_changed_since_generation (items present)",
    "N1": "N1 an out-dir holding 'TeSt' or ':' -> out_dir_name_invalid, nothing created",
    "C1": "C1 the listing and the cache's sources differ -> cache_sources_mismatch",
    "C3": "C3 --expected-cache-sha256 not the cache's -> cache_checksum_mismatch",
    "C4": "C4 a missing cache root -> cache_root_missing",
    "C5": "C5 reads not counted once per row -> verification_count_mismatch (both sites; an existing artifact dir)",
    "R1": "R1 the TEST-session order: E1, E6, E2, E3, E4, E5, E7, then seeds 43/44; inferential first",
    "R2": "R2 a second run skips the done cells and launches nothing",
    "R3a": "R3a an artifact without its record -> artifact_without_record",
    "R3b": "R3b a record without its artifact -> record_without_artifact",
    "R3c": "R3c a done cell that no longer verifies -> cell_does_not_verify",
    "R4": "R4 one real process per cell, distinct pids; both cells verify",
    "R5": "R5 an orphan temporary directory -> orphan_temporary_directory, named, kept",
    "R6a": "R6a a cell that wrote nothing is re-run once, logged; the grid completes",
    "R6b": "R6b a cell that fails twice -> STOP cell_failed_twice (exit 3)",
    "R6c": "R6c a failed cell with a partial output -> STOP cell_left_partial_output",
    "R7a": "R7a --split test without --confirm-test-split -> test_split_unconfirmed",
    "R7b": "R7b an out-root holding 'TeSt' -> out_name_invalid",
    "R7c": "R7c six malformed exports (one a file_id ending in a newline) -> inventory_invalid",
    "R7d": "R7d split test, E1 s42 with 15 cells -> inventory_scope_invalid",
    "R7d2": "R7d2 split test, E1 s43 without E6 s43 -> inventory_scope_invalid",
    "R7e": "R7e a teacher row without --teacher-config -> teacher_config_required",
    "R7f": "R7f another evaluator commit -> evaluator_commit_mismatch",
    "R7g": "R7g T = 0 -> execution_context_invalid",
    "R7h": "R7h PLANTSEG_IMAGE_DIGEST not the export's -> image_digest_mismatch",
    "R7i": "R7i P x T above the physical cores -> processes_threads_exceed_cpus",
    "R7j": "R7j a malformed recorded value -> recorded_value_format",
    "R7k": "R7k each recorded value not the cache's -> recorded_values_mismatch",
    "R7l": "R7l an out-root inside the repository -> out_root_inside_repo",
    "R8": "R8 --plan-only: every gate, the ordered cells, nothing created",
    "W12": "W12 a failed record write leaves no record and no temporary file (EV-5)",
    "T8": "T8 a teacher stream not M4-V seed 42 -> nmf_stream_mismatch, nothing written (LE11)",
    "ST7n": "ST7n the FP32 checkpoint changes before its load -> checkpoint_changed_after_validation",
    "G5i": "G5i a provisional VAL run with another cache's record -> verification_record_mismatch",
    "N1b": "N1b ':' in a parent component of the out-dir -> out_dir_name_invalid, nothing created",
    "N1c": "N1c an out-dir through a symlink to a ':' target -> out_dir_name_invalid (resolved)",
    "P1": "P1 a coded refusal prints exactly [code] message; an uncoded one keeps its type (LE2)",
    "R3d": "R3d a done cell whose record names another cache -> cell_does_not_verify",
    "R3e": "R3e a done cell of another commit or another source sha -> cell_does_not_verify",
    "R5b": "R5b an orphan of an unplanned out-dir -> orphan_temporary_directory, named, kept",
    "R6d": "R6d one earlier logged failure: one more attempt, then STOP cell_failed_twice",
    "R6e": "R6e two earlier logged failures -> cell_failed_twice at start, nothing started",
    "R7b2": "R7b2 ':' in a parent component of the out-root -> out_name_invalid",
    "R7b3": "R7b3 an out-root through a symlink to a ':' target -> out_name_invalid (resolved)",
    "R7e2": "R7e2 a --teacher-config that is not a file -> teacher_config_required",
    "R7h2": "R7h2 an official TEST grid with no image digest -> image_digest_invalid",
    "R7i2": "R7i2 SMT: P x T counted against physical cores (sysfs, /proc/cpuinfo)",
    "R7m": "R7m no CPU topology to read -> physical_cores_unknown",
    "R7n": "R7n a source that is not a file -> source_missing",
    "R7o": "R7o a cache that cannot serve the grid -> cache_scope_mismatch at start",
    "R7p": "R7p an argv every cell would refuse -> cell_argv_refused at start",
    "R7q": "R7q an official grid with dirty governed paths -> governed_paths_dirty",
    "R9": "R9 scheduling at P = 3: teacher slot, no two INT8 cells at once, each cell once",
    "R10": "R10 the argv per row kind: H, H2, T and the teacher sha (LE2)",
    "R11": "R11 relative cache and record paths: the gates and the cells name the same files",
    "R13": "R13 a held out-root lock -> out_root_locked, nothing started",
    "R14": "R14 HEAD moves during the session -> STOP head_changed",
    "R15": "R15 an error in the run loop -> STOP runner_error, recorded",
}


def make_work() -> Path:
    """The work folder, named before it is created: the temporary root is checked for 'test'
    first, and the name's random part is hex, which cannot spell it."""
    base = Path(tempfile.gettempdir())
    if "test" in str(base).lower():
        raise AssertionError(f"the temporary root {base} holds 'test'; set TMPDIR elsewhere")
    work = base / f"corrupted_eval_{secrets.token_hex(6)}"
    work.mkdir(mode=0o700)
    return work


def main() -> int:
    print("=" * 100)
    print("S4 CORRUPTED-CONDITION EVALUATOR SMOKE — synthetic; no PlantSeg data; no GPU")
    print("=" * 100)
    work = make_work()
    threads = torch.get_num_threads()
    try:
        setup(work)
        missing = sorted(set(CASES) - set(LABELS))
        if missing:
            raise AssertionError(f"cases without labels: {missing}")
        for name in CASES:
            ok, detail = run(name)
            results.append((LABELS[name], ok, detail))
            torch.set_num_threads(threads)
        mutations()
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        results.append(("FATAL", False, traceback.format_exc(limit=3)))
    finally:
        if "saved_root" in S:
            DATA["root"] = S["saved_root"]
        if "real_list_stems" in S:
            AD.list_split_stems = S["real_list_stems"]
        torch.set_num_threads(threads)
        shutil.rmtree(work, ignore_errors=True)
    print()
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if detail and not ok:
            print(f"         {detail[:240]}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
