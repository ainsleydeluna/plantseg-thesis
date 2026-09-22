#!/usr/bin/env python3
"""B62 — teacher data pipeline (M2 augmentation, M3 scale/canvas, M11 surfaces). Teacher image, CPU.

Synthetic images only (a temp dir outside the repository); the PlantSeg dataset is never read.
What is proven:
  * M2/M3 train: the teacher's MMSeg transform IS the E1-E3 recipe (`train_preprocess` +
    `configs/augment.AUGMENT`), bit-for-bit under the same RNG, and each locked behaviour holds:
    long side = round(512*r), r ~ U[0.75, 2.0]; rotation before the crop with ImageNet-mean / 255
    fills; 512 crop/pad with cat_max_ratio 0.95; independent H/V flips; image-only joint hue/sat;
    no excluded photometric operation.
  * M3 evaluation: the teacher's validation canvas IS `core_preprocess` (long side 512, bilinear
    image / nearest mask, symmetric pad, ImageNet-mean / 255 fills), the MMSeg-normalised tensor
    equals the thesis evaluator's `finalize()`, and prediction and ground truth are both 512x512.
  * M11: no active TEST surface in the merged config.
Exit codes: 0 pass · 1 a check fails · 2 the teacher stack is not importable.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
TMP = Path(tempfile.mkdtemp(prefix="smoke_teacher_pipeline_"))
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


class ScriptedRNG:
    """A RandomState stand-in with scripted draws, to force one augmentation branch at a time."""

    def __init__(self, rand=(), uniform=(), randint=()):
        self._rand, self._uniform, self._randint = list(rand), list(uniform), list(randint)

    def rand(self):
        return self._rand.pop(0)

    def uniform(self, lo, hi):
        return self._uniform.pop(0)

    def randint(self, lo, hi):
        return self._randint.pop(0) if self._randint else lo


def main() -> int:
    try:
        import numpy as np
        import torch
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from PIL import Image
    except ImportError as e:
        print(f"ENVIRONMENT: teacher stack not importable ({e}); run in the plantseg-teacher image.")
        return 2
    os.chdir(REPO)
    cfg = Config.fromfile(str(CONFIG))
    init_default_scope("mmseg")
    import src.training.teacher_components as tc
    from configs.augment import AUGMENT
    from mmseg.registry import MODELS, TRANSFORMS
    from src.data import transforms as T

    rs = np.random.RandomState(0)
    samples = {}
    for name, (h, w) in {"landscape": (300, 450), "portrait": (640, 380), "square": (512, 512),
                         "tiny": (90, 120), "large": (1200, 900)}.items():
        img = (rs.rand(h, w, 3) * 255).astype(np.uint8)
        mask = rs.randint(0, 116, (h, w)).astype(np.uint8)
        mask[: h // 3] = 0
        Image.fromarray(img).save(TMP / f"{name}.jpg", quality=95)
        Image.fromarray(mask, mode="L").save(TMP / f"{name}.png")
        samples[name] = dict(img_path=str(TMP / f"{name}.jpg"), seg_map_path=str(TMP / f"{name}.png"))

    def open_pair(res):
        return Image.open(res["img_path"]), Image.open(res["seg_map_path"])

    # ---- M2/M3 train: identity with the E1-E3 implementation ----
    same = []
    for name, res in samples.items():
        for seed in range(4):
            np.random.seed(seed)
            out = tc.ThesisTeacherTrainTransform()(dict(res))
            np.random.seed(seed)
            rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
            im, mk = open_pair(res)
            with im, mk:
                ri, rm = T.train_preprocess(im, mk, rng, AUGMENT)
            same.append(np.array_equal(out["img"][..., ::-1], ri)
                        and np.array_equal(out["gt_seg_map"], rm.astype(np.uint8))
                        and out["img"].shape == (512, 512, 3) and out["ori_shape"] == (512, 512))
    check("m2_train_transform_is_e1_recipe_bitwise", all(same), f"{sum(same)}/{len(same)} (5 shapes x 4 seeds)")

    # ---- M3 train scale: long side = round(512 * r), r ~ U[0.75, 2.0], before the crop ----
    rrc = AUGMENT["random_resized_crop"]
    ok = []
    for r in (0.75, 1.0, 1.37, 2.0):
        im, mk = open_pair(samples["portrait"])
        with im, mk:
            ri, _ = T._resize_long_side(im.convert("RGB"), mk, max(1, int(round(512 * r))))
        ok.append(max(ri.size) == int(round(512 * r)))
    check("m3_train_long_side_is_round_512r", all(ok) and tuple(rrc["scale_range"]) == (0.75, 2.0))

    # ---- M2 rotation: before the crop, ImageNet-mean image fill, 255 mask fill ----
    base = np.full((200, 300, 3), 200, np.uint8)
    base_mask = np.full((200, 300), 7, np.int64)
    ri, rm = T._apply_rotation(base, base_mask, ScriptedRNG(rand=[0.0], uniform=[10.0]), AUGMENT)
    check("m2_rotation_fills", tuple(ri[0, 0]) == T.IMAGENET_MEAN_8BIT and rm[0, 0] == 255
          and set(np.unique(rm)) <= {7, 255}, f"corner {tuple(ri[0, 0])} / mask {rm[0, 0]}")
    ri, rm = T._apply_rotation(base, base_mask, ScriptedRNG(rand=[0.9]), AUGMENT)
    check("m2_rotation_p_0_5_skip", np.array_equal(ri, base) and np.array_equal(rm, base_mask))
    check("m2_rotation_before_crop", T.train_preprocess.__code__.co_names.index("_apply_rotation")
          < T.train_preprocess.__code__.co_names.index("_random_crop_pad_512"))

    # ---- M2 crop/pad 512 with cat_max_ratio 0.95 ----
    dominant = np.zeros((600, 600), np.int64)
    dominant[:5, :5] = 3                                  # 99.99% one class -> every crop rejected
    img600 = np.zeros((600, 600, 3), np.uint8)
    _, cm, info = T._random_crop_pad_512(img600, dominant, np.random.RandomState(0), rrc["cat_max_ratio"])
    check("m2_cat_max_ratio_0_95_enforced", rrc["cat_max_ratio"] == 0.95 and info["fallback"] is True
          and info["attempts"] == 10 and cm.shape == (512, 512), str(info))
    small = np.full((100, 200, 3), 50, np.uint8)
    ci, cm, _ = T._random_crop_pad_512(small, np.ones((100, 200), np.int64), np.random.RandomState(1), 0.95)
    check("m2_crop_pads_with_mean_and_255", ci.shape == (512, 512, 3)
          and tuple(ci[0, 0]) == T.IMAGENET_MEAN_8BIT and cm[0, 0] == 255)

    # ---- M2 independent flips ----
    arr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    marr = np.arange(4).reshape(2, 2)
    hi, _ = T._apply_flips(arr, marr, ScriptedRNG(rand=[0.0, 0.9]), AUGMENT)
    vi, _ = T._apply_flips(arr, marr, ScriptedRNG(rand=[0.9, 0.0]), AUGMENT)
    bi, _ = T._apply_flips(arr, marr, ScriptedRNG(rand=[0.0, 0.0]), AUGMENT)
    check("m2_independent_h_and_v_flips",
          np.array_equal(hi, arr[:, ::-1]) and np.array_equal(vi, arr[::-1])
          and np.array_equal(bi, arr[::-1, ::-1])
          and AUGMENT["horizontal_flip_p"] == AUGMENT["vertical_flip_p"] == 0.5)

    # ---- M2 image-only joint hue/saturation, p 0.5 ----
    colour = (rs.rand(32, 32, 3) * 255).astype(np.uint8)
    out = T._apply_photometric(colour, ScriptedRNG(rand=[0.0], uniform=[0.015, 1.2]), AUGMENT)
    skip = T._apply_photometric(colour, ScriptedRNG(rand=[0.9]), AUGMENT)
    check("m2_joint_hue_sat_image_only", not np.array_equal(out, colour) and np.array_equal(skip, colour)
          and AUGMENT["photometric"] == {"hue": 0.015, "saturation_factor": (0.8, 1.2), "p": 0.5})
    src_text = Path(T.__file__).read_text(encoding="utf-8")
    forbidden = [tok for tok in ("ImageEnhance", "GaussianBlur", "ImageFilter", "quality=",
                                 "random_noise", "brightness(", "contrast(") if tok in src_text]
    check("m2_no_excluded_photometric_ops", forbidden == []
          and set(AUGMENT["excluded"]) == {"brightness", "contrast", "blur", "noise", "jpeg"}, str(forbidden))
    types = [t["type"] for t in cfg.train_dataloader.dataset.pipeline]
    check("m2_no_photometric_distortion", types == ["ThesisTeacherTrainTransform", "PackSegInputs"], str(types))

    # ---- M3 evaluation canvas == the thesis evaluator's canvas, through MMSeg ----
    torch.manual_seed(0)
    model = MODELS.build(cfg.model)
    model.eval()
    pack = TRANSFORMS.build(dict(type="PackSegInputs"))
    canvas_ok, tensor_err, pred_ok = [], [], []
    for name, res in samples.items():
        packed = pack(tc.ThesisTeacherEvalTransform()(dict(res)))
        im, mk = open_pair(res)
        with im, mk:
            ci, cm = T.core_preprocess(im, mk)
        s_t, m_t = T.finalize(ci, cm)
        data = model.data_preprocessor(dict(inputs=[packed["inputs"]], data_samples=[packed["data_samples"]]),
                                       False)
        canvas_ok.append(torch.equal(packed["data_samples"].gt_sem_seg.data[0].long(), m_t)
                         and np.array_equal(packed["inputs"].numpy().transpose(1, 2, 0)[..., ::-1], ci))
        tensor_err.append(float((data["inputs"][0] - s_t).abs().max()))
        with torch.no_grad():
            out = model.val_step(dict(inputs=[packed["inputs"]], data_samples=[packed["data_samples"]]))
        pred_ok.append(tuple(out[0].pred_sem_seg.data.shape) == (1, 512, 512)
                       and tuple(out[0].gt_sem_seg.data.shape) == (1, 512, 512)
                       and tuple(out[0].metainfo["ori_shape"]) == (512, 512))
    check("m3_val_target_is_thesis_canvas_exactly", all(canvas_ok), f"{sum(canvas_ok)}/{len(canvas_ok)} shapes")
    check("m3_normalised_input_equals_finalize", max(tensor_err) <= 1e-5, f"max |d| {max(tensor_err):.2e}")
    check("m3_prediction_and_gt_on_512_canvas", all(pred_ok))
    im, mk = open_pair(samples["landscape"])
    with im, mk:
        ci, cm = T.core_preprocess(im, mk)
    content_rows = np.where((cm != 255).any(axis=1))[0]
    top, bottom = int(content_rows[0]), int(511 - content_rows[-1])
    check("m3_symmetric_pad_long_side_512",
          (cm != 255).any(axis=0).all() and content_rows.size == 341 and top == (512 - 341) // 2
          and bottom == 512 - 341 - top and tuple(ci[0, 0]) == T.IMAGENET_MEAN_8BIT,
          f"450x300 -> 512x341, pad top {top} / bottom {bottom}")
    labels_in = set(np.unique(np.asarray(Image.open(samples["landscape"]["seg_map_path"]))))
    check("m3_nearest_mask_no_new_labels", set(np.unique(cm)) <= labels_in | {255})

    # ---- M11 ----
    check("m11_no_active_test_surface",
          cfg.test_dataloader is None and cfg.test_evaluator is None and cfg.test_cfg is None
          and cfg.test_pipeline is None)

    print("\n[CHECKS]")
    for name, ok_, detail in results:
        print(f"  {name:44}: {'PASS' if ok_ else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok_, _ in results if ok_)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
