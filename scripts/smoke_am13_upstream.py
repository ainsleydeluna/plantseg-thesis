#!/usr/bin/env python3
"""L-AM13 smoke: upstream-protocol scoring (docs/lane_specs/part1.md lane 2 (d); EVALUATION_CONTRACT
section 11). CPU-only, synthetic data only; the TEST split is never listed or opened.

Sections (d5, the real VAL runs, is local: scripts/am13_real_checks.py):
  D1  rescale: the spec's four shapes; a sweep against mmcv `rescale_size` when importable (else the
      formula) and mmcv `imrescale`'s output shapes; half-up rounding; bottom/right padding to a
      multiple of 32 against mmseg `stack_batch` when importable
  D2  two-step logit resize: on a 2-class map where logit-then-argmax and argmax-then-nearest
      disagree at a boundary pixel, the implementation returns the logit answer; the crop drops the
      padding; MMSeg `predict_by_feat` -> `postprocess_result` cross-check when importable
  D3  geometry-only proof, student: a random 512x512 input yields bitwise-equal pre-crop logits under
      both protocols (no rescale, no pad in either path), through the data path and the core
  D4  student forward at (1,3,512,672), (1,3,512,2048), (1,3,416,512): stride-8 head logits and
      input-size output, a failure being a STOP (a model change is out of scope); the converted
      INT8 student at the same shapes (AM-16 item 4 scores every student)
  D6  CLI: --split test under --protocol upstream is refused without --confirm-test-split, before any
      dataset, model or data-root listing; the other test guards; batch size 1
  A   artifact layout 1.2.0: summary.protocol, per-image shape fields, core and writer refusals, the
      original-resolution data variant, readers tolerate the new fields
  E   scripts/evaluate_model.run end to end on a synthetic PlantSeg-layout VAL root: upstream rows and
      shapes, predictions scored at the original resolution, two runs byte-identical, the 512x512
      image scored identically under both protocols, canvas rows and summary without protocol
      fields; the local checker scripts/am13_real_checks.py (preflight and d5) on this root and runs

The teacher half of D3 needs the MMSeg stack: scripts/smoke_am13_teacher.py (teacher image).
Writes only under tempfile.mkdtemp() (removed in `finally`); the project repository's git state is
never read (every artifact request uses a temp fixture repository).

Run:  python -B scripts/smoke_am13_upstream.py      (exit 1 on any failure)
"""
from __future__ import annotations

import contextlib
import io
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
import torch.nn.functional as F       # noqa: E402
from PIL import Image                 # noqa: E402

import evaluate_model as CLI                                              # noqa: E402
from src.data.transforms import core_preprocess, finalize                # noqa: E402
from src.eval import (Condition, DatasetMeta, EvalBatch,                  # noqa: E402
                      EvaluationIntegrityError, ManifestEntry, RunMeta,
                      evaluate_model as core_evaluate)
from src.eval import adapters as ADP                                      # noqa: E402
from src.eval import protocols as P                                       # noqa: E402
from src.eval.artifacts import (ARTIFACT_SCHEMA_VERSION, ArtifactRequestError,  # noqa: E402
                                ArtifactWriteError, build_config_payload,
                                prepare_artifact_request, validate_artifact_request,
                                validate_summary, verify_artifact, write_artifact)
from src.eval.model_loading import build_fp32_student                    # noqa: E402

C = 116
CHECKS: list[tuple[str, bool, str]] = []
NOTES: list[str] = []
SPEC_D1 = {(1009, 1200): (512, 609), (256, 256): (512, 512), (512, 2048): (512, 2048),
           (300, 1500): (410, 2048)}
D4_SHAPES = ((1, 3, 512, 672), (1, 3, 512, 2048), (1, 3, 416, 512))
# E: the indices deterministic_subset(846, 6) selects, with the shapes placed there
E_SHAPES = {0: (300, 450), 141: (640, 380), 282: (512, 512), 423: (1025, 1024), 564: (90, 120),
            705: (257, 511)}
E_ZERO_DISEASE = 564
SYNTH_CLASS_MAP = [{"class_id": 0, "name": "", "role": "background_or_non_disease"},
                   *[{"class_id": c, "name": f"synthetic_disease_{c:03d}", "role": "disease"}
                     for c in range(1, C)]]


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), " | ".join(str(detail).split("\n")).strip()))


def expect(exc, fn, *a, **kw):
    """Return the exception `fn` raises; fail loudly on a different exception or none."""
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as e:                                   # noqa: BLE001
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}") from e
    raise AssertionError(f"expected {exc.__name__}, none raised")


# ------------------------------------------------------------------------------------------ fixtures
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
    (root / "seed.txt").write_text("am13 fixture\n", encoding="utf-8")
    _git(root, "add", "--", "seed.txt")
    _git(root, "commit", "-q", "-m", "fixture baseline")
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root.resolve():
        raise AssertionError("fixture is not its own repository")
    return root


def rm_tree(path: Path) -> None:
    for p in path.rglob("*"):
        try:
            os.chmod(p, stat.S_IWRITE)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def make_val_root(root: Path) -> None:
    """846 VAL pairs (the locked count), tiny except at the E indices; masks 0 + one disease."""
    img_dir, ann_dir = root / "images" / "val", root / "annotations" / "val"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    rs = np.random.RandomState(20260928)
    for i in range(846):
        h, w = E_SHAPES.get(i, (8, 10))
        mask = np.zeros((h, w), np.uint8)
        if i != E_ZERO_DISEASE:
            mask[h // 4: max(h // 4 + 1, 3 * h // 4), w // 5: max(w // 5 + 1, 4 * w // 5)] = \
                1 + i % 115
        Image.fromarray((rs.rand(h, w, 3) * 255).astype(np.uint8)).save(
            img_dir / f"syn_val_{i:05d}.jpg", quality=95)
        Image.fromarray(mask).save(ann_dir / f"syn_val_{i:05d}.png")


def seeded_student():
    torch.manual_seed(0)
    return build_fp32_student()


class ConvModel(torch.nn.Module):
    """Tiny seeded model returning 116-class logits at the input size (like both evaluator models)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.conv = torch.nn.Conv2d(3, C, 3, padding=1)

    def forward(self, x):
        return F.interpolate(self.conv(F.avg_pool2d(x, 8)), size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


def upstream_batch(sample: P.UpstreamSample, image_id: str, index: int) -> EvalBatch:
    return EvalBatch(sample.image[None], sample.target[None], [image_id], [image_id], [index],
                     sample_meta=[sample.shapes])


# ------------------------------------------------------------------------------------------ D1
def section_d1() -> None:
    for (h, w), want in SPEC_D1.items():
        got = P.upstream_rescaled_shape(h, w)
        check(f"D1 rescale ({h},{w}) -> {want} (spec value)", got == want, f"got {got}")
    try:
        from mmcv import imrescale, rescale_size
        ref = "mmcv.rescale_size"

        def reference(h, w):
            nw, nh = rescale_size((w, h), P.UPSTREAM_SCALE)
            return nh, nw
    except ImportError:
        imrescale = None
        ref = "the formula int(side * s + 0.5)"

        def reference(h, w):
            s = min(2048 / max(h, w), 512 / min(h, w))
            return int(h * s + 0.5), int(w * s + 0.5)
    rs = np.random.RandomState(13)
    sizes = [(h, w) for h in range(1, 120, 7) for w in range(1, 120, 11)]
    sizes += [(int(a), int(b)) for a, b in rs.randint(1, 6000, (4000, 2))]
    sizes += [*SPEC_D1, (1025, 1024), (1024, 1025), (3, 1), (1, 3), (4096, 1024), (8192, 16)]
    bad = [s for s in sizes if P.upstream_rescaled_shape(*s) != reference(*s)]
    check(f"D1 sweep: {len(sizes)} shapes equal {ref}", not bad, f"mismatches {bad[:5]}")
    check("D1 half-up rounding: (1025,1024) -> (513,512), as mmcv (round-half-even would give 512)",
          P.upstream_rescaled_shape(1025, 1024) == (513, 512) == reference(1025, 1024)
          and round(1025 * 0.5) == 512, str(P.upstream_rescaled_shape(1025, 1024)))
    if imrescale is not None:
        shapes = [*SPEC_D1, (1025, 1024), (37, 91), (2000, 3000), (640, 380)]
        got = [imrescale(np.zeros((h, w, 3), np.uint8), P.UPSTREAM_SCALE).shape[:2] for h, w in shapes]
        want = [P.upstream_rescaled_shape(h, w) for h, w in shapes]
        check("D1 mmcv.imrescale output shapes equal the rescaled shapes", got == want,
              f"{got} vs {want}")
    else:
        NOTES.append("D1: mmcv not importable; the sweep used the formula, which restates the "
                     "implementation (only the spec values and the half-up case are independent)")
    check("D1 padded shape: next multiple of 32 (512x609 -> 512x640; 410x2048 -> 416x2048)",
          P.padded_shape(512, 609) == (512, 640) and P.padded_shape(410, 2048) == (416, 2048)
          and P.padded_shape(512, 512) == (512, 512) and P.padded_shape(513, 512) == (544, 512))
    t = torch.randn(3, 410, 609)
    padded = P.pad_normalized(t)
    check("D1 padding is bottom/right with 0.0 on the normalised tensor",
          tuple(padded.shape) == (3, 416, 640) and torch.equal(padded[:, :410, :609], t)
          and bool((padded[:, 410:, :] == 0).all()) and bool((padded[:, :, 609:] == 0).all()))
    try:
        from mmseg.utils import stack_batch
    except Exception:                                        # noqa: BLE001
        stack_batch = None
    if stack_batch is not None:
        ok = []
        for h, w in ((410, 609), (512, 512), (513, 512), (257, 1018)):
            x = torch.randn(3, h, w)
            ref_t, meta = stack_batch([x], size_divisor=32, pad_val=0)
            mine = P.pad_normalized(x)
            ok.append(torch.equal(ref_t[0], mine)
                      and tuple(meta[0]["pad_shape"]) == tuple(mine.shape[-2:]))
        check("D1 padding equals mmseg stack_batch(size_divisor=32, pad_val=0), bitwise", all(ok),
              f"{sum(ok)}/{len(ok)}")
    else:
        NOTES.append("D1: mmseg not importable; stack_batch cross-check not run")


# ------------------------------------------------------------------------------------------ D2
def section_d2() -> None:
    # Pixel A: class 0 (1.0 vs 0.0); pixel B: class 1 (0.0 vs 4.0). Upsampled 1x2 -> 1x4 bilinear, the
    # second output pixel is 0.75*A + 0.25*B = (0.75, 1.0) -> class 1; nearest keeps A there -> class 0.
    lo = torch.tensor([[[[1.0, 0.0]], [[0.0, 4.0]]]])
    shapes = {"padded_shape": [1, 2], "rescaled_shape": [1, 2], "ori_shape": [1, 4]}
    logit_first = P.logits_to_original(lo, shapes).argmax(1)[0]
    argmax_first = F.interpolate(lo.argmax(1, keepdim=True).float(), size=(1, 4),
                                 mode="nearest").long()[:, 0][0]
    check("D2 logit-then-argmax and argmax-then-nearest disagree at a boundary pixel",
          not torch.equal(logit_first, argmax_first), f"{logit_first.tolist()} vs {argmax_first.tolist()}")
    check("D2 the implementation returns the logit-upsampled answer [0, 1, 1, 1]",
          logit_first.tolist() == [[0, 1, 1, 1]], str(logit_first.tolist()))
    # The same content padded bottom/right with a class-0 value large enough to flip the answer if it
    # were not cropped before the second resize.
    padded = torch.zeros(1, 2, 2, 3)
    padded[..., :1, :2] = lo
    padded[:, 0, 1:, :] = 100.0
    padded[:, 0, :, 2:] = 100.0
    pshapes = {"padded_shape": [2, 3], "rescaled_shape": [1, 2], "ori_shape": [1, 4]}
    cropped = P.logits_to_original(padded, pshapes)
    uncropped = F.interpolate(padded, size=(1, 4), mode="bilinear", align_corners=False)
    check("D2 crop removes the padding before the resize to the original shape",
          torch.equal(cropped, P.logits_to_original(lo, shapes))
          and not torch.equal(cropped.argmax(1), uncropped.argmax(1)),
          f"cropped {cropped.argmax(1).tolist()} uncropped {uncropped.argmax(1).tolist()}")
    # Low-resolution logits (a head at stride 8): the explicit two-step reference.
    torch.manual_seed(5)
    low = torch.randn(1, 5, 8, 12)
    s2 = {"padded_shape": [64, 96], "rescaled_shape": [61, 90], "ori_shape": [37, 55]}
    ref = F.interpolate(F.interpolate(low, size=(64, 96), mode="bilinear", align_corners=False)
                        [:, :, :61, :90], size=(37, 55), mode="bilinear", align_corners=False)
    mine = P.logits_to_original(low, s2)
    one_step = F.interpolate(low, size=(37, 55), mode="bilinear", align_corners=False)
    check("D2 two-step resize (to padded, crop, to original) equals the explicit reference, bitwise",
          torch.equal(mine, ref))
    check("D2 one-step resize differs (why the two steps are pinned)",
          not torch.equal(mine.argmax(1), one_step.argmax(1)),
          f"{int((mine.argmax(1) != one_step.argmax(1)).sum())} pixels differ")
    try:
        from types import SimpleNamespace

        from mmseg.models.decode_heads.decode_head import BaseDecodeHead
        from mmseg.models.segmentors.base import BaseSegmentor
        from mmseg.structures import SegDataSample
    except Exception:                                        # noqa: BLE001
        NOTES.append("D2: mmseg not importable; predict_by_feat/postprocess_result cross-check not run")
        return
    fake = SimpleNamespace(align_corners=False)
    ok = []
    for lg, sh in ((low, s2), (lo, shapes), (padded, pshapes)):
        (ph, pw), (rh, rw), (oh, ow) = sh["padded_shape"], sh["rescaled_shape"], sh["ori_shape"]
        meta = {"img_shape": (rh, rw), "pad_shape": (ph, pw), "ori_shape": (oh, ow),
                "img_padding_size": [0, pw - rw, 0, ph - rh]}
        seg = BaseDecodeHead.predict_by_feat(fake, lg, [meta])
        ds = SegDataSample(metainfo=meta)
        out = BaseSegmentor.postprocess_result(fake, seg, [ds])[0]
        mine = P.logits_to_original(lg, sh)
        ok.append(torch.equal(out.seg_logits.data, mine[0])
                  and torch.equal(out.pred_sem_seg.data, mine.argmax(1)))
    check("D2 equals MMSeg predict_by_feat -> postprocess_result (logits and prediction), bitwise",
          all(ok), f"{sum(ok)}/{len(ok)}")


# ------------------------------------------------------------------------------------------ D3
def section_d3_student() -> None:
    model = seeded_student()
    # (a) tensor level: a random normalised 512x512 input
    torch.manual_seed(11)
    x = torch.randn(1, 3, 512, 512)
    shapes = P.upstream_shapes(512, 512)
    check("D3 512x512: rescaled and padded shapes are 512x512 (no rescale, no pad)",
          shapes == {"ori_shape": [512, 512], "rescaled_shape": [512, 512],
                     "padded_shape": [512, 512]}, str(shapes))
    pre_crop = []

    def record(m, t):
        pre_crop.append(m(t))
        return pre_crop[-1]
    with torch.no_grad():
        canvas_logits = model(x)
        batch = EvalBatch(P.pad_normalized(x[0])[None], torch.zeros(1, 512, 512, dtype=torch.long),
                          ["d3"], ["d3"], [0], sample_meta=[shapes])
        up_logits = P.UpstreamForward(record)(model, batch)
    check("D3 student, random tensor: pre-crop logits bitwise equal under both protocols",
          torch.equal(canvas_logits, pre_crop[0]))
    check("D3 student, random tensor: final logits and predictions equal",
          torch.equal(canvas_logits.argmax(1), up_logits.argmax(1)),
          f"final logits bitwise equal: {torch.equal(canvas_logits, up_logits)}")
    NOTES.append(f"D3: student logits after the same-size resize to ori_shape bitwise equal: "
                 f"{torch.equal(canvas_logits, up_logits)}")
    # (b) data level: a random uint8 image through core_preprocess and the upstream preparation
    rs = np.random.RandomState(3)
    img = (rs.rand(512, 512, 3) * 255).astype(np.uint8)
    mask = rs.randint(0, C, (512, 512)).astype(np.uint8)
    ci, cm = core_preprocess(Image.fromarray(img), Image.fromarray(mask))
    x_canvas, t_canvas = finalize(ci, cm)
    sample = P.prepare_upstream_sample(img, mask.astype(np.int64))
    check("D3 data path: input tensors bitwise equal (core_preprocess vs upstream)",
          torch.equal(x_canvas, sample.image))
    check("D3 data path: targets bitwise equal", torch.equal(t_canvas, sample.target))
    pre_crop.clear()
    with torch.no_grad():
        lc = model(x_canvas[None])
        lu = P.UpstreamForward(record)(model, upstream_batch(sample, "d3b", 0))
    check("D3 student, random image: pre-crop logits bitwise equal under both protocols",
          torch.equal(lc, pre_crop[0]))
    check("D3 student, random image: predictions equal", torch.equal(lc.argmax(1), lu.argmax(1)))
    # (c) through the evaluation core: canvas forward vs upstream batch forward, two 512x512 images
    samples = []
    for k in range(2):
        im = (rs.rand(512, 512, 3) * 255).astype(np.uint8)
        mk = np.zeros((512, 512), np.int64)
        mk[100:300, 50:400] = 7 + k
        samples.append(P.prepare_upstream_sample(im, mk))
    manifest = [ManifestEntry(k, f"d3_{k}", f"d3_{k}") for k in range(2)]
    canvas_batches = [EvalBatch(s.image[None], s.target[None], [f"d3_{k}"], [f"d3_{k}"], [k])
                      for k, s in enumerate(samples)]
    up_batches = [upstream_batch(s, f"d3_{k}", k) for k, s in enumerate(samples)]
    kw = dict(expected_manifest=manifest, condition=Condition("clean"), num_classes=C,
              background_index=0, ignore_index=255)
    rc = core_evaluate(model, canvas_batches, **kw)
    ru = core_evaluate(model, up_batches, batch_forward=P.UpstreamForward(), **kw)
    rows_equal = all({k: v for k, v in b.as_dict().items() if k not in P.SHAPE_FIELDS} == a.as_dict()
                     for a, b in zip(rc.rows, ru.rows))
    npz_equal = all(np.array_equal(getattr(rc, f), getattr(ru, f)) for f in (
        "sparse_image_index", "sparse_class_id", "sparse_tp", "sparse_gt", "sparse_pred",
        "dataset_tp", "dataset_gt", "dataset_pred"))
    check("D3 student through the core: rows (minus shape fields), sufficient stats and "
          "dataset-level values identical", rows_equal and npz_equal
          and rc.dataset_level == ru.dataset_level)


# ------------------------------------------------------------------------------------------ D4
def section_d4() -> None:
    model = seeded_student()
    fixed = [type(m).__name__ for m in model.modules()
             if isinstance(m, (torch.nn.AvgPool2d, torch.nn.MaxPool2d))]
    check("D4 the student has no fixed-kernel pooling module (the head pools adaptively)",
          not fixed, str(fixed))
    seen = []
    hook = model.head.register_forward_hook(lambda m, i, o: seen.append(tuple(o.shape)))
    try:
        for shp in D4_SHAPES:
            try:
                with torch.no_grad():
                    out = model(torch.randn(*shp))
                head = seen[-1]
                ok = head == (1, C, shp[2] // 8, shp[3] // 8) and tuple(out.shape) == (1, C, *shp[2:])
                detail = f"head {head}, output {tuple(out.shape)}"
            except Exception as e:                           # noqa: BLE001
                ok, detail = False, f"STOP: forward raised {type(e).__name__}: {e}"
            check(f"D4 student forward {shp}: stride-8 head logits, input-size output", ok, detail)
    finally:
        hook.remove()
    # AM-16 item 4 scores every student, so the converted INT8 form must take these shapes too.
    from src.quant.prepare import calibrate, convert_model, prepare_ptq
    engine = torch.backends.quantized.engine
    try:
        prepared = prepare_ptq(model)                            # selects the qnnpack engine
        calibrate(prepared, [torch.randn(1, 3, 64, 64) for _ in range(2)])
        converted = convert_model(prepared)
        for shp in D4_SHAPES:
            try:
                with torch.no_grad():
                    out = converted(torch.randn(*shp))
                ok, detail = tuple(out.shape) == (1, C, *shp[2:]), f"output {tuple(out.shape)}"
            except Exception as e:                           # noqa: BLE001
                ok, detail = False, f"forward raised {type(e).__name__}: {e}"
            check(f"D4 converted INT8 ({torch.backends.quantized.engine}) student forward {shp}: "
                  "input-size output", ok, detail)
    finally:
        torch.backends.quantized.engine = engine


# ------------------------------------------------------------------------------------------ D6
def section_d6(work: Path) -> None:
    parse = CLI.build_parser().parse_args
    touched: list[str] = []
    saved = (ADP.list_split_stems, ADP.build_expected_manifest_for, ADP.PlantSegEvalDataset,
             ADP.PlantSegUpstreamEvalDataset)

    def spy(name):
        def _f(*a, **k):
            touched.append(name)
            raise AssertionError(f"{name} must not run for a refused TEST request")
        return _f
    ADP.list_split_stems = spy("list_split_stems")
    ADP.build_expected_manifest_for = spy("build_expected_manifest_for")
    ADP.PlantSegEvalDataset = spy("PlantSegEvalDataset")
    ADP.PlantSegUpstreamEvalDataset = spy("PlantSegUpstreamEvalDataset")
    try:
        base = ["--protocol", "upstream", "--split", "test", "--batch-size", "1",
                "--checkpoint", str(work / "never_read.pt"), "--artifact-status", "provisional",
                "--out-dir", str(work / "d6_out")]
        counters = CLI.Counters(dataset=[], model=[])
        e = expect(CLI.CliError, CLI.run, parse(base), counters=counters)
        check("D6 --protocol upstream --split test without --confirm-test-split is refused",
              "--confirm-test-split" in str(e), str(e))
        check("D6 refused before any dataset, model or data-root listing",
              counters.dataset == [] and counters.model == [] and touched == []
              and not (work / "d6_out").exists(), f"touched={touched}")
        check("D6 main() exits 1 on the same request", CLI.main(base) == 1 and touched == [])
        cases = {
            "--random-init": (["--random-init", "--artifact-status", "smoke"], "forbids --random-init"),
            "--max-samples": (["--max-samples", "4"], "forbids any sample cap"),
            "artifact_status=smoke": (["--artifact-status", "smoke"], "refuses artifact_status=smoke"),
        }
        for label, (extra, needle) in cases.items():
            argv = [a for a in base if a != "--checkpoint" and a != str(work / "never_read.pt")] \
                if label == "--random-init" else list(base)
            e = expect(CLI.CliError, CLI.validate_cli_args, parse(argv + ["--confirm-test-split"] + extra))
            check(f"D6 upstream TEST with --confirm-test-split and {label} is still refused",
                  needle in str(e), str(e))
        argv = [a for a in base if a not in ("--checkpoint", str(work / "never_read.pt"))]
        e = expect(CLI.CliError, CLI.validate_cli_args, parse(argv + ["--confirm-test-split"]))
        check("D6 upstream TEST without a checkpoint is refused", "exactly one of" in str(e)
              or "requires --checkpoint" in str(e), str(e))
        check("D6 no guard touched the dataset", touched == [], str(touched))
    finally:
        (ADP.list_split_stems, ADP.build_expected_manifest_for, ADP.PlantSegEvalDataset,
         ADP.PlantSegUpstreamEvalDataset) = saved
    val = ["--split", "val", "--random-init", "--out-dir", str(work / "x")]
    e = expect(CLI.CliError, CLI.validate_cli_args, parse(val + ["--protocol", "upstream",
                                                                "--batch-size", "2"]))
    check("D6 upstream with --batch-size 2 is refused (batch size 1)", "batch size 1" in str(e), str(e))
    CLI.validate_cli_args(parse(val + ["--protocol", "upstream", "--batch-size", "1"]))
    check("D6 upstream on VAL with --batch-size 1 passes the argument guards", True)
    with contextlib.redirect_stderr(io.StringIO()):          # argparse prints its usage there
        e = expect(SystemExit, parse, val + ["--protocol", "bogus"])
    check("D6 an unknown --protocol value is rejected by the parser", e.code == 2)
    legacy = type("A", (), dict(stage="E1", model_role="student", precision="fp32", split="val",
                                condition="clean", random_init=True, checkpoint=None,
                                artifact_status="smoke", batch_size=2, max_samples=None,
                                confirm_test_split=False, device="cpu"))()
    CLI.validate_cli_args(legacy)
    check("D6 an args object without `protocol` (older callers) is the canvas protocol", True)
    legacy.protocol = "bogus"
    e = expect(CLI.CliError, CLI.validate_cli_args, legacy)
    check("D6 validate_cli_args refuses an unknown protocol", "not supported" in str(e), str(e))


# ------------------------------------------------------------------------------------------ A
def synth_request(out_dir: Path, repo: Path, protocol_id: str, n: int):
    manifest = [ManifestEntry(k, f"a_{k:02d}", f"a_{k:02d}") for k in range(n)]
    return prepare_artifact_request(
        out_dir=out_dir, artifact_status="smoke", run_id=f"am13_{out_dir.name}",
        run=RunMeta(stage="E1", model_role="student", precision="fp32", quant_backend=None,
                    checkpoint_path=None, checkpoint_sha256=None, random_init=True, device="cpu"),
        dataset=DatasetMeta(name="SYNTHETIC am13 fixture (NOT PlantSeg data)",
                            doi="10.5281/zenodo.17719108", split="val", condition=Condition("clean"),
                            preprocess_protocol=protocol_id, expected_rows=n),
        expected_manifest=manifest, class_map=SYNTH_CLASS_MAP, repo_root=repo), manifest


def section_a(work: Path) -> None:
    repo = make_fixture_repo(work / "fixture_repo")
    rs = np.random.RandomState(21)
    sizes = [(300, 450), (37, 91), (512, 512), (1025, 1024)]
    samples = []
    for k, (h, w) in enumerate(sizes):
        mk = np.zeros((h, w), np.int64)
        if k != 1:
            mk[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = 3 + k
        samples.append(P.prepare_upstream_sample((rs.rand(h, w, 3) * 255).astype(np.uint8), mk))
    model = ConvModel().eval()
    req, manifest = synth_request(work / "art_up", repo, P.UPSTREAM_PROTOCOL_ID, len(sizes))
    batches = [upstream_batch(s, f"a_{k:02d}", k) for k, s in enumerate(samples)]
    result = core_evaluate(model, batches, expected_manifest=manifest, condition=Condition("clean"),
                           num_classes=C, background_index=0, ignore_index=255,
                           batch_forward=P.UpstreamForward())
    prov = validate_artifact_request(req)
    out = write_artifact(result, req, prov)
    summary = verify_artifact(out)
    rows = [json.loads(ln) for ln in (out / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
    check("A upstream artifact: layout 1.2.0 and summary.protocol == the contract block",
          summary.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
          == "plantseg-eval-artifact/1.2.0" and summary.get("protocol") == P.upstream_block()
          and f"{summary['protocol']['name']}/{summary['protocol']['version']}"
          == summary["dataset"]["preprocess_protocol"] == "upstream/1.0.0")
    check("A the protocol block carries exactly the spec keys plus the class-count note",
          list(summary["protocol"]) == ["name", "version", "resize", "size_divisor",
                                        "pad_value_normalized", "logit_upsample", "align_corners",
                                        "scored_resolution", "note"]
          and summary["protocol"]["resize"] == {"short": 512, "long_max": 2048, "interp": "bilinear"}
          and "NOT DETERMINABLE" in summary["protocol"]["note"])
    check("A per-image rows end with ori_shape, rescaled_shape, padded_shape after am5_excluded",
          all(list(r)[-4:] == ["am5_excluded", *P.SHAPE_FIELDS] for r in rows)
          and [r["ori_shape"] for r in rows] == [list(s) for s in sizes])
    with np.load(out / "sufficient_stats.npz") as z:
        per_img_pred = np.bincount(z["image_index"], weights=z["pred"], minlength=len(rows))
        per_img_gt = np.bincount(z["image_index"], weights=z["gt"], minlength=len(rows))
    check("A every image is scored at its original resolution (sum pred == sum gt == H*W)",
          all(int(per_img_pred[k]) == int(per_img_gt[k]) == h * w for k, (h, w) in enumerate(sizes)))
    from src.stats.ingest import Policy, load_run
    run = load_run(out, Policy.NONOFFICIAL_SMOKE)
    check("A the statistics reader loads a 1.2.0 upstream artifact (extra fields tolerated)",
          len(run.records) == len(sizes) and run.am5.source == "artifact"
          and run.identity.preprocess_protocol == "upstream/1.0.0")
    payload_canvas = build_config_payload(synth_request(work / "c", repo, P.CANVAS_PROTOCOL_ID, 4)[0])
    payload_up = build_config_payload(req)
    diff = sorted(k for k in payload_up if payload_up[k] != payload_canvas.get(k))
    check("A config payload keys unchanged; canvas and upstream differ only in preprocess_protocol",
          set(payload_up) == set(payload_canvas) and diff == ["preprocess_protocol"], str(diff))
    # writer refusals
    req_c, _ = synth_request(work / "art_canvas_bad", repo, P.CANVAS_PROTOCOL_ID, len(sizes))
    e = expect(ArtifactWriteError, write_artifact, result, req_c, validate_artifact_request(req_c))
    check("A writer refuses shape fields on a non-upstream artifact", "only the upstream" in str(e)
          and not (work / "art_canvas_bad").exists(), str(e))
    canvas_batches = [EvalBatch(s.image[None], s.target[None], [f"a_{k:02d}"], [f"a_{k:02d}"], [k])
                      for k, s in enumerate(samples)]
    plain = core_evaluate(model, [b for b in canvas_batches if b.images.shape[-2:] == b.targets.shape[-2:]],
                          expected_manifest=[m for m in manifest if m.image_id == "a_02"],
                          condition=Condition("clean"), num_classes=C, background_index=0,
                          ignore_index=255)
    req_u1, _ = synth_request(work / "art_up_bad", repo, P.UPSTREAM_PROTOCOL_ID, 1)
    req_u1 = prepare_artifact_request(
        out_dir=req_u1.out_dir, artifact_status="smoke", run_id="am13_bad", run=req_u1.run,
        dataset=req_u1.dataset, expected_manifest=[ManifestEntry(2, "a_02", "a_02")],
        class_map=SYNTH_CLASS_MAP, repo_root=repo)
    e = expect(ArtifactWriteError, write_artifact, plain, req_u1, validate_artifact_request(req_u1))
    check("A writer refuses an upstream artifact whose rows lack the shape fields",
          "must end with" in str(e), str(e))
    tampered = list(samples[0].shapes["rescaled_shape"])
    tampered[1] += 1
    bad_rows = [type(r)(**{**r.__dict__, "protocol_fields": tuple(
        (k, tampered if k == "rescaled_shape" else v) for k, v in r.protocol_fields)})
        if r.image_id == "a_00" else r for r in result.rows]
    result_bad = type(result)(**{**result.__dict__, "rows": bad_rows})
    req_t, _ = synth_request(work / "art_up_tampered", repo, P.UPSTREAM_PROTOCOL_ID, len(sizes))
    e = expect(ArtifactWriteError, write_artifact, result_bad, req_t, validate_artifact_request(req_t))
    check("A writer refuses shape fields that are not the upstream geometry of ori_shape",
          "not the upstream geometry" in str(e), str(e))
    s_ok = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    e = expect(ArtifactWriteError, validate_summary, {**s_ok, "protocol": {**s_ok["protocol"],
                                                                           "size_divisor": 16}},
               req, result)
    check("A validate_summary refuses a protocol block that differs from the contract", "does not "
          "match" in str(e), str(e))
    # rows must hold exactly the contract keys (+ the shape fields for upstream): a batch forward that
    # skips the record check cannot smuggle another per-image key into either kind of artifact
    loose = [EvalBatch(b.images, b.targets, b.image_ids, b.clean_image_ids, b.manifest_indices,
                       sample_meta=[{**b.sample_meta[0], "foo": 1}]) for b in batches]
    result_loose = core_evaluate(
        model, loose, expected_manifest=manifest, condition=Condition("clean"), num_classes=C,
        background_index=0, ignore_index=255,
        batch_forward=lambda m, b: P.logits_to_original(m(b.images), b.sample_meta[0]))
    req_l, _ = synth_request(work / "art_up_extra", repo, P.UPSTREAM_PROTOCOL_ID, len(sizes))
    e = expect(ArtifactWriteError, write_artifact, result_loose, req_l, validate_artifact_request(req_l))
    check("A writer refuses an upstream row carrying a key beyond the shape fields",
          "must end with" in str(e) and "foo" in str(e), str(e))
    canvas_extra = [EvalBatch(samples[2].image[None], samples[2].target[None], ["a_02"], ["a_02"], [2],
                              sample_meta=[{"foo": 1}])]
    result_cx = core_evaluate(model, canvas_extra, expected_manifest=[ManifestEntry(2, "a_02", "a_02")],
                              condition=Condition("clean"), num_classes=C, background_index=0,
                              ignore_index=255)
    req_cx = prepare_artifact_request(
        out_dir=work / "art_canvas_extra", artifact_status="smoke", run_id="am13_cx", run=req.run,
        dataset=DatasetMeta(name="SYNTHETIC am13 fixture (NOT PlantSeg data)",
                            doi="10.5281/zenodo.17719108", split="val", condition=Condition("clean"),
                            preprocess_protocol=P.CANVAS_PROTOCOL_ID, expected_rows=1),
        expected_manifest=[ManifestEntry(2, "a_02", "a_02")], class_map=SYNTH_CLASS_MAP, repo_root=repo)
    e = expect(ArtifactWriteError, write_artifact, result_cx, req_cx, validate_artifact_request(req_cx))
    check("A writer refuses a canvas row carrying any key beyond the contract keys",
          "only the upstream protocol adds per-image fields" in str(e) and "foo" in str(e), str(e))
    req_v, _ = synth_request(work / "art_up_v9", repo, "upstream/9.9.9", len(sizes))
    e = expect(ArtifactRequestError, validate_artifact_request, req_v)
    check("A the pre-inference gate refuses an upstream protocol version it does not implement",
          "does not implement" in str(e), str(e))
    wrapper = ADP.PlantSegUpstreamEvalDataset.__new__(ADP.PlantSegUpstreamEvalDataset)
    wrapper.source_indices = [0]
    wrapper.base = type("Base", (), {
        "pairs": [(Path("syn_thin.jpg"), Path("syn_thin.png"))],
        "__getitem__": lambda self, i: (np.zeros((1, 10379, 3), np.uint8), np.zeros((1, 10379), np.int64))})()
    e = expect(ADP.AdapterError, wrapper.__getitem__, 0)
    check("A a sample the geometry cannot rescale (1x10379 -> a zero-height side) is refused by name",
          str(e).startswith("syn_thin:"), str(e))
    # core refusals
    e = expect(EvaluationIntegrityError, core_evaluate, model, batches[:1],
               expected_manifest=manifest[:1], condition=Condition("clean"), num_classes=C,
               background_index=0, ignore_index=255, forward=lambda m, x: m(x),
               batch_forward=P.UpstreamForward())
    check("A the core refuses forward and batch_forward together", "at most one" in str(e))
    clash = EvalBatch(samples[2].image[None], samples[2].target[None], ["a_02"], ["a_02"], [2],
                      sample_meta=[{"image_id": "x"}])
    e = expect(EvaluationIntegrityError, core_evaluate, model, [clash],
               expected_manifest=[ManifestEntry(2, "a_02", "a_02")], condition=Condition("clean"),
               num_classes=C, background_index=0, ignore_index=255)
    check("A the core refuses per-image fields that would overwrite contract fields",
          "overwrite" in str(e), str(e))
    e = expect(ValueError, EvalBatch, samples[2].image[None], samples[2].target[None], ["a"], ["a"],
               [0], sample_meta=[{}, {}])
    check("A EvalBatch refuses a sample_meta of the wrong length", "sample_meta" in str(e))
    two = EvalBatch(torch.cat([samples[2].image[None]] * 2), torch.cat([samples[2].target[None]] * 2),
                    ["p", "q"], ["p", "q"], [0, 1], sample_meta=[samples[2].shapes] * 2)
    e = expect(EvaluationIntegrityError, P.UpstreamForward(), model, two)
    check("A the upstream forward refuses a batch of two", "1 image per batch" in str(e), str(e))
    wrong_target = EvalBatch(samples[0].image[None], samples[2].target[None], ["p"], ["p"], [0],
                             sample_meta=[samples[0].shapes])
    e = expect(EvaluationIntegrityError, P.UpstreamForward(), model, wrong_target)
    check("A the upstream forward refuses a target that is not at ori_shape", "ori_shape" in str(e))
    e = expect(ADP.AdapterError, ADP.upstream_collate, [{"image": samples[2].image}] * 2)
    check("A the upstream collate refuses a batch of two", "batch size 1" in str(e), str(e))
    e = expect(P.ProtocolError, P.prepare_upstream_sample, np.zeros((40, 60, 3), np.uint8),
               np.zeros((60, 40), np.int64))
    check("A the protocol refuses an image and mask of different shapes", "never resizes" in str(e))
    # the original-resolution data variant: EXIF transpose on the image only, never on the mask
    from src.data.original_resolution import PlantSegOriginalResolutionDataset as ORD
    img = (rs.rand(40, 60, 3) * 255).astype(np.uint8)
    exif = Image.Exif()
    exif[0x0112] = 6                                          # orientation: rotate 90 on display
    Image.fromarray(img).save(work / "exif.jpg", quality=95, exif=exif)
    Image.fromarray(np.ones((40, 60), np.uint8)).save(work / "exif_raw.png")
    Image.fromarray(np.ones((60, 40), np.uint8)).save(work / "exif_display.png")
    ds = ORD.__new__(ORD)
    ds.pairs = [(work / "exif.jpg", work / "exif_display.png"), (work / "exif.jpg", work / "exif_raw.png")]
    image, mask = ds[0]
    check("A data variant: EXIF-transposed RGB image, raw int64 mask, original resolution",
          image.shape == (60, 40, 3) and image.dtype == np.uint8 and mask.dtype == np.int64
          and mask.shape == (60, 40))
    e = expect(ValueError, ds.__getitem__, 1)
    check("A data variant: an image/mask shape mismatch after EXIF transpose is refused, naming it",
          "exif" in str(e) and "never resizes" in str(e), str(e))
    e = expect(ValueError, ORD, "train")
    check("A data variant: evaluation splits only", "split must be one of" in str(e))
    check("A adapters: the canvas protocol id is unchanged (core_preprocess/1.0.0)",
          ADP.PREPROCESS_PROTOCOL == P.CANVAS_PROTOCOL_ID == "core_preprocess/1.0.0"
          and ADP.UPSTREAM_PREPROCESS_PROTOCOL == P.UPSTREAM_PROTOCOL_ID
          and P.summary_block(P.CANVAS_PROTOCOL_ID) is None)


# ------------------------------------------------------------------------------------------ E
def cli_args(out_dir: Path, protocol: str, ckpt: Path, **over):
    base = dict(stage="E1", model_role="student", precision="fp32", split="val", condition="clean",
                corruption_severity=None, out_dir=str(out_dir), checkpoint=str(ckpt),
                teacher_config=None, provenance=None, random_init=False, artifact_status="smoke",
                batch_size=1 if protocol == "upstream" else 2, max_samples=6,
                confirm_test_split=False, run_id=f"am13_{out_dir.name}", device="cpu",
                protocol=protocol)
    base.update(over)
    return type("A", (), base)()


def section_e(work: Path) -> None:
    from configs.data import DATA
    root = work / "plantseg_val_root"
    make_val_root(root)
    repo = make_fixture_repo(work / "fixture_repo_e")
    ckpt = work / "student.pt"
    torch.save({"iter": 4000, "model_state_dict": seeded_student().state_dict(), "num_classes": C,
                "best_val_miou_all_class": 0.1}, ckpt)
    import am13_real_checks as RC
    saved_root, saved_repo = DATA["root"], CLI.REPO
    DATA["root"], CLI.REPO = str(root), repo
    try:
        up1 = CLI.run(cli_args(work / "e_up1", "upstream", ckpt))
        up2 = CLI.run(cli_args(work / "e_up2", "upstream", ckpt))
        cv = CLI.run(cli_args(work / "e_canvas", "canvas", ckpt))
        with contextlib.redirect_stdout(io.StringIO()) as pf_ok:
            code_pf_ok = RC.preflight()
        # an image stored 8x10 whose EXIF orientation transposes it (10x8 on display), mask 8x10
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.fromarray(np.zeros((8, 10, 3), np.uint8)).save(
            root / "images" / "val" / "syn_val_00001.jpg", quality=95, exif=exif)
        with contextlib.redirect_stdout(io.StringIO()) as pf_bad:
            code_pf_bad = RC.preflight()
    finally:
        DATA["root"], CLI.REPO = saved_root, saved_repo
    check("E am13_real_checks preflight: 846 VAL pairs, no mismatch, largest image and its logits "
          "memory reported", code_pf_ok == RC.EXIT_PASS and "VAL pairs: 846" in pf_ok.getvalue()
          and "largest image: syn_val_00423 1025x1024" in pf_ok.getvalue(), pf_ok.getvalue()[-400:])
    check("E am13_real_checks preflight fails an EXIF-transposed image whose mask keeps the raw shape",
          code_pf_bad == RC.EXIT_FAIL and "syn_val_00001: image 10x8 after EXIF, mask 8x10"
          in pf_bad.getvalue(), pf_bad.getvalue()[-400:])
    s_up = verify_artifact(up1)
    s_cv = verify_artifact(cv)
    rows_up = [json.loads(ln) for ln in (up1 / "per_image.jsonl").read_text("utf-8").splitlines()]
    rows_cv = [json.loads(ln) for ln in (cv / "per_image.jsonl").read_text("utf-8").splitlines()]
    want = [list(E_SHAPES[k]) for k in sorted(E_SHAPES)]
    check("E upstream CLI run: 6 rows, ori_shape == each image's original (h, w)",
          [r["ori_shape"] for r in rows_up] == want, str([r["ori_shape"] for r in rows_up]))
    check("E upstream rows: rescaled/padded shapes are the upstream geometry (e.g. 300x450 -> "
          "512x768)", all(P.validate_shapes({k: r[k] for k in P.SHAPE_FIELDS}) for r in rows_up)
          and rows_up[0]["rescaled_shape"] == [512, 768] and rows_up[3]["rescaled_shape"] == [513, 512]
          and rows_up[3]["padded_shape"] == [544, 512])
    with np.load(up1 / "sufficient_stats.npz") as z:
        pred = np.bincount(z["image_index"], weights=z["pred"], minlength=len(rows_up))
        gt = np.bincount(z["image_index"], weights=z["gt"], minlength=len(rows_up))
    check("E predictions scored at the original resolution: per image sum pred == sum gt == H*W",
          all(int(pred[k]) == int(gt[k]) == h * w for k, (h, w) in enumerate(map(tuple, want))))
    check("E upstream summary: protocol block, preprocess_protocol upstream/1.0.0, batch size 1, "
          "one forward per image", s_up.get("protocol") == P.upstream_block()
          and s_up["dataset"]["preprocess_protocol"] == "upstream/1.0.0"
          and s_up["run"]["eval_runtime"]["batch_size"] == 1
          and s_up["run"]["eval_runtime"]["forward_batches"] == 6 and s_up["am5"]["excluded_count"] == 1)
    same_rows = (up1 / "per_image.jsonl").read_bytes() == (up2 / "per_image.jsonl").read_bytes()
    with np.load(up1 / "sufficient_stats.npz") as za, np.load(up2 / "sufficient_stats.npz") as zb:
        same_npz = sorted(za.files) == sorted(zb.files) and all(
            za[k].dtype == zb[k].dtype and za[k].tobytes() == zb[k].tobytes() for k in za.files)
    check("E two upstream runs: per_image.jsonl byte-identical, npz arrays bitwise equal",
          same_rows and same_npz)
    check("E canvas CLI run: layout 1.2.0, no protocol block, rows without shape fields",
          s_cv.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION and "protocol" not in s_cv
          and s_cv["dataset"]["preprocess_protocol"] == "core_preprocess/1.0.0"
          and all(list(r)[-1] == "am5_excluded" and not set(P.SHAPE_FIELDS) & set(r)
                  for r in rows_cv))
    k512 = sorted(E_SHAPES).index(282)
    check("E the 512x512 image is scored identically under both protocols (geometry only)",
          {k: v for k, v in rows_up[k512].items() if k not in P.SHAPE_FIELDS} == rows_cv[k512])
    check("E canvas and upstream runs score the same manifest",
          s_up["dataset"]["split_manifest_sha256"] == s_cv["dataset"]["split_manifest_sha256"]
          and s_up["run"]["config_sha256"] != s_cv["run"]["config_sha256"])
    # the local d5 checker, on these synthetic runs (row count lowered from 846 to 6 for the fixture)
    saved_rows = RC.VAL_ROWS
    RC.VAL_ROWS = len(E_SHAPES)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out_ok:
            code_ok = RC.d5(up1, up2, canvas_value=0.25,
                            expect_checkpoint_sha256=s_up["run"]["checkpoint_sha256"])
        with contextlib.redirect_stdout(io.StringIO()) as out_bad:
            code_bad = RC.d5(up1, cv, canvas_value=None)
        with contextlib.redirect_stdout(io.StringIO()) as out_model:
            code_model = RC.d5(up1, up2, canvas_value=None, expect_checkpoint_sha256="0" * 64)
    finally:
        RC.VAL_ROWS = saved_rows
    check("E am13_real_checks d5 passes two identical upstream runs of the expected model and reports "
          "the protocol effect", code_ok == RC.EXIT_PASS and "D5 PASS" in out_ok.getvalue()
          and "protocol effect (upstream - canvas)" in out_ok.getvalue(), out_ok.getvalue()[-400:])
    check("E am13_real_checks d5 fails an upstream/canvas pair", code_bad == RC.EXIT_FAIL
          and "not an upstream/1.0.0 artifact" in out_bad.getvalue(), out_bad.getvalue()[-400:])
    check("E am13_real_checks d5 fails runs of another model than --expect-checkpoint-sha256",
          code_model == RC.EXIT_FAIL and "is not the expected model" in out_model.getvalue(),
          out_model.getvalue()[-400:])


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="am13_smoke_"))
    try:
        for name, fn in (("D1", section_d1), ("D2", section_d2), ("D3", section_d3_student),
                         ("D4", section_d4), ("D6", lambda: section_d6(work)),
                         ("A", lambda: section_a(work)), ("E", lambda: section_e(work))):
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
    for n in NOTES:
        print(f"  NOTE: {n}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'L-AM13 OK' if allok else 'L-AM13 FAIL'} ({good}/{len(CHECKS)})")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
