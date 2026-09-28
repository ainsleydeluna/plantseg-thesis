#!/usr/bin/env python3
"""L-AM13 teacher smoke: the teacher half of d3 and the teacher evaluation path under the upstream
protocol (docs/lane_specs/part1.md lane 2 (d); EVALUATION_CONTRACT section 11). CPU-only, synthetic
data, a RANDOM-INIT SegNeXt-B built from the thesis teacher config (no teacher checkpoint). Needs the
MMSeg stack, so it runs in the teacher image; without the stack it exits 2 and passes nothing.

Sections:
  T3  geometry-only proof, teacher: a random 512x512 input yields bitwise-equal pre-crop logits under
      both protocols (no rescale, no pad in either path), with the M4-V NMF stream seeded 42 for each
      pass, through the tensor and the data path; the stream consumes the same draws; a non-square
      image runs at its upstream shape and returns logits at its original shape
  TE  scripts/evaluate_model.run for the teacher (M4-V, batch size 1) under both protocols on a
      synthetic PlantSeg-layout VAL root, from a temp file holding the same random-init weights:
      upstream rows and shapes, scoring at the original resolution, the 512x512 image scored
      identically under both protocols, the canvas artifact without protocol fields

Writes only under tempfile.mkdtemp() (removed in `finally`); the project repository's git state is
never read.

Run:  python -B scripts/smoke_am13_teacher.py      (exit 1 on any failure, 2 without the MMSeg stack)
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import numpy as np                    # noqa: E402
import torch                          # noqa: E402
from PIL import Image                 # noqa: E402

C = 116
CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
CHECKS: list[tuple[str, bool, str]] = []
# TE: the indices deterministic_subset(846, 4) selects, with the shapes placed there
TE_SHAPES = {0: (300, 450), 211: (512, 512), 422: (90, 120), 633: (257, 511)}
TE_ZERO_DISEASE = 422


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), " | ".join(str(detail).split("\n")).strip()))


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.name=am13-fixture", "-c", "user.email=am13-fixture@invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), capture_output=True, text=True, check=False,
        env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(repo.parent)))
    if proc.returncode != 0:
        raise AssertionError(f"fixture git {args} failed ({proc.returncode}): {proc.stderr[:200]}")
    return proc.stdout


def make_fixture_repo(root: Path) -> Path:
    """A throwaway clean repository (never the project) for artifact provenance."""
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "seed.txt").write_text("am13 teacher fixture\n", encoding="utf-8")
    _git(root, "add", "--", "seed.txt")
    _git(root, "commit", "-q", "-m", "fixture baseline")
    return root


def rm_tree(path: Path) -> None:
    for p in path.rglob("*"):
        try:
            os.chmod(p, stat.S_IWRITE)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def make_val_root(root: Path) -> None:
    """846 VAL pairs (the locked count), tiny except at the TE indices."""
    img_dir, ann_dir = root / "images" / "val", root / "annotations" / "val"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    rs = np.random.RandomState(20260929)
    for i in range(846):
        h, w = TE_SHAPES.get(i, (8, 10))
        mask = np.zeros((h, w), np.uint8)
        if i != TE_ZERO_DISEASE:
            mask[h // 4: max(h // 4 + 1, 3 * h // 4), w // 5: max(w // 5 + 1, 4 * w // 5)] = \
                1 + i % 115
        Image.fromarray((rs.rand(h, w, 3) * 255).astype(np.uint8)).save(
            img_dir / f"syn_val_{i:05d}.jpg", quality=95)
        Image.fromarray(mask).save(ann_dir / f"syn_val_{i:05d}.png")


def build_random_teacher():
    """Random-init SegNeXt-B from the thesis config, wrapped exactly as load_teacher_model wraps it."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    cfg = Config.fromfile(str(CONFIG))
    init_default_scope("mmseg")
    import src.training.teacher_components  # noqa: F401  (registers IsolatedNMFLightHamHead)
    from mmseg.registry import MODELS

    from src.distill.segnext_teacher import SegNeXtTeacherAdapter
    from src.distill.teacher import FrozenTeacher
    from src.eval.model_loading import TeacherEvalModel
    torch.manual_seed(0)
    segmentor = MODELS.build(cfg.model).eval()
    frozen = FrozenTeacher(SegNeXtTeacherAdapter(segmentor))
    return TeacherEvalModel(frozen).eval(), frozen, segmentor, cfg


# ------------------------------------------------------------------------------------------ T3
def section_t3(teacher, frozen) -> None:
    from src.data.transforms import core_preprocess, finalize
    from src.eval import EvalBatch
    from src.eval import protocols as P

    pre_crop = []

    def record(m, t):
        pre_crop.append(m(t))
        return pre_crop[-1]

    def canvas_pass(x):
        info = frozen.begin_nmf_stream("M4-V", 42)
        with torch.no_grad():
            out = teacher(x)
        return out, info, frozen.nmf_stream_state()

    def upstream_pass(batch):
        info = frozen.begin_nmf_stream("M4-V", 42)
        with torch.no_grad():
            out = P.UpstreamForward(record)(teacher, batch)
        return out, info, frozen.nmf_stream_state()

    torch.manual_seed(11)
    x = torch.randn(1, 3, 512, 512)
    shapes = P.upstream_shapes(512, 512)
    lc, info_c, end_c = canvas_pass(x)
    batch = EvalBatch(P.pad_normalized(x[0])[None], torch.zeros(1, 512, 512, dtype=torch.long),
                      ["t3"], ["t3"], [0], sample_meta=[shapes])
    lu, info_u, end_u = upstream_pass(batch)
    check("T3 M4-V stream attached and seeded 42 for both passes",
          info_c is not None and info_c["seed"] == 42 and info_c["policy"] == "M4-V"
          and info_c == info_u, str(info_c))
    check("T3 teacher, random tensor: pre-crop logits bitwise equal under both protocols",
          torch.equal(lc, pre_crop[0]))
    check("T3 teacher, random tensor: predictions equal; the NMF stream consumed the same draws",
          torch.equal(lc.argmax(1), lu.argmax(1)) and end_c == end_u,
          f"final logits bitwise equal: {torch.equal(lc, lu)}; stream {end_c} vs {end_u}")
    rs = np.random.RandomState(3)
    img = (rs.rand(512, 512, 3) * 255).astype(np.uint8)
    mask = rs.randint(0, C, (512, 512)).astype(np.uint8)
    ci, cm = core_preprocess(Image.fromarray(img), Image.fromarray(mask))
    x_canvas, _ = finalize(ci, cm)
    sample = P.prepare_upstream_sample(img, mask.astype(np.int64))
    pre_crop.clear()
    lc2, _, _ = canvas_pass(x_canvas[None])
    lu2, _, _ = upstream_pass(EvalBatch(sample.image[None], sample.target[None], ["t3b"], ["t3b"],
                                        [0], sample_meta=[sample.shapes]))
    check("T3 teacher, random image: identical inputs, bitwise-equal pre-crop logits",
          torch.equal(x_canvas, sample.image) and torch.equal(lc2, pre_crop[0])
          and torch.equal(lc2.argmax(1), lu2.argmax(1)))
    wide = P.prepare_upstream_sample((rs.rand(300, 450, 3) * 255).astype(np.uint8),
                                     np.zeros((300, 450), np.int64))
    pre_crop.clear()
    out, _, _ = upstream_pass(EvalBatch(wide.image[None], wide.target[None], ["t3c"], ["t3c"], [0],
                                        sample_meta=[wide.shapes]))
    check("T3 teacher on a 300x450 image: input 512x768, logits back at 300x450",
          tuple(pre_crop[0].shape) == (1, C, 512, 768) and tuple(out.shape) == (1, C, 300, 450),
          f"pre-crop {tuple(pre_crop[0].shape)} output {tuple(out.shape)}")


# ------------------------------------------------------------------------------------------ TE
def section_te(work: Path, segmentor, cfg) -> None:
    import evaluate_model as CLI
    from configs.data import DATA
    from src.eval import protocols as P
    from src.eval.artifacts import verify_artifact

    root = work / "plantseg_val_root"
    make_val_root(root)
    repo = make_fixture_repo(work / "fixture_repo")
    ckpt = work / "random_init_teacher.pth"
    torch.save({"meta": {"dataset_meta": {"classes": list(cfg.PLANTSEG_CLASSES)}},
                "state_dict": segmentor.state_dict()}, ckpt)

    def args(out: Path, protocol: str):
        return type("A", (), dict(
            stage="teacher", model_role="teacher", precision="fp32", split="val", condition="clean",
            corruption_severity=None, out_dir=str(out), checkpoint=str(ckpt),
            teacher_config=str(CONFIG), provenance=None, random_init=False,
            artifact_status="smoke", batch_size=1, max_samples=4, confirm_test_split=False,
            run_id=f"am13_teacher_{protocol}", device="cpu", protocol=protocol))()

    saved_root, saved_repo = DATA["root"], CLI.REPO
    DATA["root"], CLI.REPO = str(root), repo
    try:
        up = CLI.run(args(work / "te_up", "upstream"))
        cv = CLI.run(args(work / "te_canvas", "canvas"))
    finally:
        DATA["root"], CLI.REPO = saved_root, saved_repo
    s_up, s_cv = verify_artifact(up), verify_artifact(cv)
    rows_up = [json.loads(ln) for ln in (up / "per_image.jsonl").read_text("utf-8").splitlines()]
    rows_cv = [json.loads(ln) for ln in (cv / "per_image.jsonl").read_text("utf-8").splitlines()]
    want = [list(TE_SHAPES[k]) for k in sorted(TE_SHAPES)]
    check("TE teacher upstream run: 4 rows at the original shapes, shape fields last",
          [r["ori_shape"] for r in rows_up] == want
          and all(list(r)[-3:] == list(P.SHAPE_FIELDS) for r in rows_up))
    with np.load(up / "sufficient_stats.npz") as z:
        pred = np.bincount(z["image_index"], weights=z["pred"], minlength=len(rows_up))
        gt = np.bincount(z["image_index"], weights=z["gt"], minlength=len(rows_up))
    check("TE teacher predictions scored at the original resolution (sum pred == sum gt == H*W)",
          all(int(pred[k]) == int(gt[k]) == h * w for k, (h, w) in enumerate(map(tuple, want))))
    rt = s_up["run"]["eval_runtime"]
    check("TE teacher upstream summary: protocol block, M4-V batch size 1, CPU, one forward per image",
          s_up.get("protocol") == P.upstream_block() and rt["batch_size"] == 1
          and rt["forward_batches"] == 4 and rt["model_device"] == "cpu"
          and s_up["run"]["model_role"] == "teacher")
    k512 = sorted(TE_SHAPES).index(211)
    check("TE teacher: the 512x512 image scored identically under both protocols (same M4-V draws)",
          {k: v for k, v in rows_up[k512].items() if k not in P.SHAPE_FIELDS} == rows_cv[k512])
    check("TE teacher canvas run: no protocol block, rows without shape fields",
          "protocol" not in s_cv and all(not set(P.SHAPE_FIELDS) & set(r) for r in rows_cv))


def main() -> int:
    try:
        teacher, frozen, segmentor, cfg = build_random_teacher()
    except ImportError as e:
        print(f"ENVIRONMENT: teacher stack not importable ({e}); run in the teacher image.")
        print("RESULT: L-AM13 TEACHER NOT RUN (0/0)")
        return 2
    work = Path(tempfile.mkdtemp(prefix="am13_teacher_"))
    try:
        for name, fn in (("T3", lambda: section_t3(teacher, frozen)),
                         ("TE", lambda: section_te(work, segmentor, cfg))):
            try:
                fn()
            except Exception as e:                           # noqa: BLE001
                check(f"{name} section completed", False,
                      f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    finally:
        rm_tree(work)
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail and not ok:
            print(f"         {detail}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'L-AM13 TEACHER OK' if allok else 'L-AM13 TEACHER FAIL'} ({good}/{len(CHECKS)})")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
