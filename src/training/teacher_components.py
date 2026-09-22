"""B62 — MMSegmentation components for the thesis SegNeXt-B teacher (M2, M3, M4, M12, init provenance).

Loaded by the teacher config through `custom_imports`, so it runs only where MMSegmentation is
installed. It registers:

* `ThesisTeacherTrainTransform` (M2 + M3 train) and `ThesisTeacherEvalTransform` (M3 evaluation).
  Both reuse the student's own `src/data/transforms.py` with `configs/augment.AUGMENT` rather than a
  second implementation, so teacher/student parity holds by construction. They emit the 512x512
  thesis canvas (image padded with the 8-bit ImageNet mean, mask padded with 255) and set
  `ori_shape = img_shape = (512, 512)`, so predictions and ground truth are scored on that canvas.
* `IsolatedNMFLightHamHead` / `IsolatedNMF2D` (M4). The upstream algorithm is unchanged
  (`rand_init=True`). With no stream attached the upstream basis draw runs on the global CPU stream
  (M4-T). With a stream attached, only the basis draw runs inside the private-state window
  (M4-V / M4-KD); a stream attached in training mode fails closed.
* `ThesisConfusionMIoUMetric` (M12): the full-precision, same-pass, union-present all-class VAL mIoU,
  computed with `src/eval/metrics.py` — the reducer E1's checkpoint selection used.
* Hooks: `TeacherNMFEvalStreamHook` (M4-V), `TeacherSelectionRecordHook` (M12 record and
  cross-check) and `TeacherInitCompatibilityHook` (the 150 -> 116 classifier-only load rule).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Imported-code attestation (same pattern as teacher_runner.py): hash the bytes this import resolved to.
_MODULE_PATH = Path(__file__).resolve()
COMPONENTS_PROVENANCE = {"path": str(_MODULE_PATH),
                         "sha256": hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()}

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv.transforms import BaseTransform  # noqa: E402
from mmengine.evaluator import BaseMetric  # noqa: E402
from mmengine.hooks import Hook  # noqa: E402
from mmengine.model import is_model_wrapper  # noqa: E402
from mmengine.runner import BaseLoop  # noqa: E402
from mmseg.models.decode_heads.ham_head import NMF2D, LightHamHead  # noqa: E402
from mmseg.registry import HOOKS, METRICS, MODELS, TRANSFORMS  # noqa: E402
from PIL import Image  # noqa: E402

from configs.augment import AUGMENT  # noqa: E402
from src.data.transforms import SIZE, core_preprocess, train_preprocess  # noqa: E402
from src.distill.nmf_stream import (M4_NMF_SEED, NMFStream, NMFStreamError,  # noqa: E402
                                    attach_nmf_stream)
from src.eval.artifacts import hash_split_manifest  # noqa: E402
from src.eval.evaluate import ManifestEntry  # noqa: E402
from src.eval.metrics import confusion_matrix, miou_from_confusion  # noqa: E402

SELECTION_KEY = "mIoU_full"
DISEASE_KEY = "mIoU_disease_full"
DISPLAY_KEY = "mIoU"
CLASSIFIER_KEYS = ("decode_head.conv_seg.weight", "decode_head.conv_seg.bias")
RECORDS_FILE = "teacher_selection_records.jsonl"
SELECTION_FILE = "teacher_selection.json"


def _unwrap(model):
    return model.module if is_model_wrapper(model) else model


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def reused_module_hashes() -> dict:
    """SHA-256 of every module whose bytes define the teacher's data, NMF and metric behaviour."""
    import configs.augment
    import src.data.isolation
    import src.data.transforms
    import src.distill.nmf_stream
    import src.eval.metrics
    out = {"teacher_components": COMPONENTS_PROVENANCE["sha256"]}
    for name, mod in (("src/data/transforms.py", src.data.transforms),
                      ("configs/augment.py", configs.augment),
                      ("src/eval/metrics.py", src.eval.metrics),
                      ("src/distill/nmf_stream.py", src.distill.nmf_stream),
                      ("src/data/isolation.py", src.data.isolation)):
        out[name] = _sha256_file(Path(mod.__file__))
    return out


# ------------------------------------------------------------------------------------------------
# M2 + M3 — transforms (thin wrappers over the student's preprocessing)
# ------------------------------------------------------------------------------------------------
def _pack_canvas(results: dict, img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    if img_rgb.shape[:2] != (SIZE, SIZE) or mask.shape != (SIZE, SIZE):
        raise ValueError(f"thesis canvas must be {SIZE}x{SIZE}; got image {img_rgb.shape}, mask {mask.shape}")
    if int(mask.min()) < 0 or int(mask.max()) > 255:
        raise ValueError(f"mask labels outside 0..255: [{int(mask.min())}, {int(mask.max())}]")
    # BGR, as LoadImageFromFile would give: SegDataPreProcessor(bgr_to_rgb=True) flips it back to RGB.
    results["img"] = np.ascontiguousarray(img_rgb[..., ::-1])
    results["gt_seg_map"] = mask.astype(np.uint8)
    results["seg_fields"] = ["gt_seg_map"]
    results["img_shape"] = (SIZE, SIZE)
    results["ori_shape"] = (SIZE, SIZE)      # predictions stay on the canvas: no resize back
    results["pad_shape"] = (SIZE, SIZE)
    results["reduce_zero_label"] = False
    return results


@TRANSFORMS.register_module()
class ThesisTeacherTrainTransform(BaseTransform):
    """M2 + M3 train: `train_preprocess(img, mask, rng, AUGMENT)`, the E1-E3 recipe, unchanged.

    long side = round(512*r), r ~ U[0.75, 2.0] on the unpadded image -> rotation +/-10 deg (p 0.5,
    before the crop; ImageNet-mean / 255 fills) -> 512 crop/pad with cat_max_ratio 0.95 -> independent
    H and V flips (p 0.5 each) -> image-only hue +/-0.015 and saturation [0.8, 1.2] (jointly p 0.5).
    The per-sample RNG is derived exactly as the student dataset derives it; draws need not match E1.
    """

    def transform(self, results: dict) -> dict:
        rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))
        with Image.open(results["img_path"]) as im, Image.open(results["seg_map_path"]) as mk:
            img, mask = train_preprocess(im, mk, rng, AUGMENT)
        return _pack_canvas(results, img, mask)


@TRANSFORMS.register_module()
class ThesisTeacherEvalTransform(BaseTransform):
    """M3 evaluation: `core_preprocess` — long side 512 (bilinear image, nearest mask), symmetric pad
    to 512x512 (image 8-bit ImageNet mean, mask 255). The same canvas the thesis evaluator scores."""

    def transform(self, results: dict) -> dict:
        with Image.open(results["img_path"]) as im, Image.open(results["seg_map_path"]) as mk:
            img, mask = core_preprocess(im, mk)
        return _pack_canvas(results, img, mask)


# ------------------------------------------------------------------------------------------------
# M4 — isolated NMF basis draw
# ------------------------------------------------------------------------------------------------
class IsolatedNMF2D(NMF2D):
    """`NMF2D` whose random-basis draw honours an attached `NMFStream` (M4-V / M4-KD).

    No stream: `super()._build_bases` runs unchanged on the global CPU stream (M4-T). With a stream:
    exactly that upstream call runs inside the private-state window, and nothing else does.
    """

    SUPPORTS_NMF_STREAM = True

    def __init__(self, args=dict()):
        super().__init__(args)
        self.nmf_stream = None

    def _build_bases(self, B, S, D, R, device=None):
        stream = self.nmf_stream
        if stream is None:
            return super()._build_bases(B, S, D, R, device=device)
        if self.training:
            raise NMFStreamError("an M4-V/M4-KD NMF stream is attached while the teacher is in training "
                                 "mode; teacher training keeps the upstream draw (M4-T).")
        return stream.draw(lambda: super(IsolatedNMF2D, self)._build_bases(B, S, D, R, device=device))


@MODELS.register_module()
class IsolatedNMFLightHamHead(LightHamHead):
    """`LightHamHead` whose Hamburger uses `IsolatedNMF2D`. Parameters and state-dict keys are identical
    to the stock head (NMF2D holds no parameters or buffers when `rand_init=True`)."""

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super().__init__(ham_channels=ham_channels, ham_kwargs=ham_kwargs, **kwargs)
        if self.hamburger.ham.rand_init is not True:
            raise ValueError("M4 keeps the upstream NMF algorithm: ham_kwargs.rand_init must be True")
        self.hamburger.ham = IsolatedNMF2D(ham_kwargs)


# ------------------------------------------------------------------------------------------------
# M12 — same-pass full-precision metric
# ------------------------------------------------------------------------------------------------
@METRICS.register_module()
class ThesisConfusionMIoUMetric(BaseMetric):
    """Dataset-level, union-present all-class VAL mIoU in full precision (M12), plus disease-only.

    One accumulated confusion matrix (`src.eval.metrics.confusion_matrix`, int64 counts on CPU) and one
    reduction (`miou_from_confusion`) — the same functions E1's validation used. `mIoU_full` is the
    selection value; `mIoU` (percent, two decimals) is for display only and never selects anything.
    """

    default_prefix = None

    def __init__(self, num_classes: int = 116, ignore_index: int = 255, background_index: int = 0,
                 collect_device: str = "cpu", prefix: str | None = None, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.background_index = background_index

    def process(self, data_batch, data_samples) -> None:
        for sample in data_samples:
            pred = sample["pred_sem_seg"]["data"].squeeze(0).cpu()
            gt = sample["gt_sem_seg"]["data"].squeeze(0).cpu()
            if tuple(pred.shape) != (SIZE, SIZE) or tuple(gt.shape) != (SIZE, SIZE):
                raise ValueError(f"M3/M12: prediction {tuple(pred.shape)} and ground truth {tuple(gt.shape)} "
                                 f"must both be the {SIZE}x{SIZE} thesis canvas")
            self.results.append(confusion_matrix(pred, gt, self.num_classes, ignore_index=self.ignore_index))

    def compute_metrics(self, results: list) -> dict:
        cm = torch.stack(results).sum(0)
        disease = torch.tensor([c for c in range(self.num_classes) if c != self.background_index])
        full = miou_from_confusion(cm)
        return {SELECTION_KEY: full, DISEASE_KEY: miou_from_confusion(cm, disease),
                DISPLAY_KEY: round(full * 100, 2), "val_images": len(results)}


# ------------------------------------------------------------------------------------------------
# Hooks
# ------------------------------------------------------------------------------------------------
def val_manifest(dataset) -> tuple[list[str], str]:
    """Image stems in PASS order, and the ordered-manifest identity of that pass.

    The identity is the evaluator's own `src.eval.artifacts.hash_split_manifest` over one
    `(manifest_index = pass position, image_id = stem, clean_image_id = stem)` row per image — the
    definition the thesis evaluator writes as `split_manifest_sha256`, so R3 compares like with like.
    It encodes order, not just membership: under M4-V an image's NMF basis depends on its position.
    """
    stems = [Path(dataset.get_data_info(i)["img_path"]).stem for i in range(len(dataset))]
    rows = [ManifestEntry(manifest_index=i, image_id=s, clean_image_id=s) for i, s in enumerate(stems)]
    return stems, hash_split_manifest(rows)


@HOOKS.register_module()
class TeacherNMFEvalStreamHook(Hook):
    """M4-V for every complete validation pass: caller CPU RNG saved, a fresh private NMF stream seeded
    42, batch size 1, frozen sequential order, one basis per image, caller CPU RNG restored exactly.

    The save happens in `before_val_epoch`, before the val DataLoader creates its iterator (which draws a
    base seed from the default generator on first use), so the restore undoes every global draw of the
    pass: validation never changes the training RNG stream.
    """

    priority = "VERY_HIGH"

    def __init__(self, seed: int = M4_NMF_SEED):
        self.seed = int(seed)
        self._caller = None
        self._stream = None
        self._n_images = None
        self.passes: list[dict] = []

    def before_val_epoch(self, runner) -> None:
        loader = runner.val_loop.dataloader
        if loader.batch_size != 1:
            raise RuntimeError(f"M4-V requires val batch_size 1, got {loader.batch_size}")
        n = len(loader.dataset)
        if getattr(loader.sampler, "shuffle", None) is not False or list(iter(loader.sampler)) != list(range(n)):
            raise RuntimeError("M4-V requires a frozen sequential val order (DefaultSampler, shuffle=False)")
        _, manifest_sha = val_manifest(loader.dataset)
        self._caller = torch.get_rng_state()
        self._stream = NMFStream(self.seed, "M4-V")
        attach_nmf_stream(_unwrap(runner.model), self._stream)
        self._n_images = n
        runner.message_hub.update_info("m4v/seed", self.seed)
        runner.message_hub.update_info("m4v/manifest_sha256", manifest_sha)
        runner.message_hub.update_info("m4v/n_images", n)

    def after_val_epoch(self, runner, metrics=None) -> None:
        attach_nmf_stream(_unwrap(runner.model), None)
        torch.set_rng_state(self._caller)
        # The training iteration of this pass, read WITHOUT `runner.iter`: in a val-only runner that property
        # builds the train loop lazily, whose DataLoader iterator draws a base seed from the global CPU RNG.
        loop = getattr(runner, "_train_loop", None)
        iteration = loop.iter if isinstance(loop, BaseLoop) else None
        record = {"policy": "M4-V", "iteration": iteration, "seed": self.seed, "draws": self._stream.draws,
                  "n_images": self._n_images, "stream_state_sha256": self._stream.state_sha256(),
                  "manifest_sha256": runner.message_hub.get_info("m4v/manifest_sha256"),
                  "caller_cpu_rng_restored": bool(torch.equal(torch.get_rng_state(), self._caller))}
        self.passes.append(record)
        runner.message_hub.update_info("m4v/last_pass", record)
        runner.logger.info(f"M4-V pass: {json.dumps(record, sort_keys=True)}")
        if record["draws"] != self._n_images:
            raise RuntimeError(f"M4-V: {record['draws']} NMF draws for {self._n_images} images; batch size 1 "
                               "must give exactly one basis draw per image")


def select_checkpoint(records: list[dict], key: str = SELECTION_KEY) -> dict:
    """Highest full-precision value; an exact tie keeps the earliest iteration (strict >)."""
    best = None
    for rec in sorted(records, key=lambda r: r["iteration"]):
        if best is None or rec[key] > best[key]:
            best = rec
    return best


def require_uniform_val_manifest(records: list[dict]) -> str:
    """Every selection record names the SAME non-empty ordered VAL manifest; return it. Fail closed on a
    missing, null or empty hash, or on any validation that scored a different manifest or order."""
    hashes = [r.get("val_manifest_sha256") for r in records]
    if not records or any(not isinstance(h, str) or not h for h in hashes):
        raise RuntimeError(f"M12: a validation record lacks its ordered VAL manifest hash: "
                           f"{[(r.get('iteration'), h) for r, h in zip(records, hashes)]}")
    if len(set(hashes)) != 1:
        raise RuntimeError(f"M12: validations scored different VAL manifests/orders: "
                           f"{[(r.get('iteration'), h) for r, h in zip(records, hashes)]}")
    return hashes[0]


def verify_selected_checkpoint(selection: dict) -> None:
    """Fail closed unless the selected checkpoint still exists with its recorded SHA-256."""
    path = Path(selection["checkpoint_path"])
    if not path.is_file():
        raise RuntimeError(f"M12: the selected checkpoint no longer exists: {path}")
    sha = _sha256_file(path)
    if sha != selection["checkpoint_sha256"]:
        raise RuntimeError(f"M12: the selected checkpoint {path} changed: sha256 {sha} != recorded "
                           f"{selection['checkpoint_sha256']}")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


@HOOKS.register_module()
class TeacherSelectionRecordHook(Hook):
    """M12: record every training-time validation and maintain the selection, cross-checked against
    CheckpointHook's own best (fail closed on any disagreement)."""

    priority = "LOWEST"

    def __init__(self, key: str = SELECTION_KEY, disease_key: str = DISEASE_KEY):
        self.key = key
        self.disease_key = disease_key
        self.records: list[dict] = []
        self._active = False

    def before_train(self, runner) -> None:
        work = Path(runner.work_dir)
        for name in (RECORDS_FILE, SELECTION_FILE):
            if (work / name).exists():
                raise RuntimeError(f"M12: {work / name} already exists; every run needs a fresh work dir")
        self._interval = runner.train_loop.val_interval
        self._max_iters = runner.train_loop.max_iters
        self._active = True

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not self._active or not isinstance(getattr(runner, "_train_loop", None), BaseLoop):
            return                                        # a val-only call (e.g. runner.val()) is not selection
        it = runner.iter
        ckpt = Path(runner.work_dir) / f"iter_{it}.pth"
        if not ckpt.is_file():
            raise RuntimeError(f"M12: validation at iteration {it} has no checkpoint {ckpt}; the checkpoint "
                               "interval must equal the validation interval")
        # The ordered VAL manifest of THIS pass: the M4-V hook's record for this iteration (VERY_HIGH, so it
        # has already run in this after_val_epoch). No record for this iteration -> no selection record.
        m4v = runner.message_hub.get_info("m4v/last_pass")
        if not isinstance(m4v, dict) or m4v.get("iteration") != it:
            raise RuntimeError(f"M12: validation at iteration {it} has no M4-V pass record, so its ordered VAL "
                               "manifest is unknown; TeacherNMFEvalStreamHook must run every validation")
        rec = {"iteration": it, self.key: float(metrics[self.key]),
               self.disease_key: float(metrics[self.disease_key]),
               "display_mIoU_percent": metrics.get(DISPLAY_KEY), "val_images": metrics.get("val_images"),
               "nmf_seed": m4v.get("seed"),
               "val_manifest_sha256": m4v.get("manifest_sha256"),
               "checkpoint_path": str(ckpt), "checkpoint_sha256": _sha256_file(ckpt),
               "checkpoint_bytes": ckpt.stat().st_size}
        require_uniform_val_manifest(self.records + [rec])     # non-empty, and the same at every validation
        self.records.append(rec)
        with open(Path(runner.work_dir) / RECORDS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        selected = select_checkpoint(self.records, self.key)
        best_score = runner.message_hub.get_info("best_score")
        best_ckpt = runner.message_hub.get_info("best_ckpt")
        expected_best = f"best_{self.key}_iter_{selected['iteration']}.pth"
        if best_score != selected[self.key] or best_ckpt is None or Path(best_ckpt).name != expected_best:
            raise RuntimeError(f"M12: CheckpointHook best ({best_score!r}, {best_ckpt!r}) disagrees with the "
                               f"thesis selection (iteration {selected['iteration']}, {selected[self.key]!r})")
        _atomic_write_json(Path(runner.work_dir) / SELECTION_FILE, self._selection(selected, "running"))

    def after_train(self, runner) -> None:
        if not self._active:
            return
        expected = list(range(self._interval, self._max_iters + 1, self._interval))
        got = [r["iteration"] for r in self.records]
        if got != expected:
            raise RuntimeError(f"M12: validations at {got}, expected {expected}")
        require_uniform_val_manifest(self.records)
        selected = select_checkpoint(self.records, self.key)
        verify_selected_checkpoint(selected)
        _atomic_write_json(Path(runner.work_dir) / SELECTION_FILE, self._selection(selected, "final"))
        runner.logger.info(f"M12 selection (final): iteration {selected['iteration']} "
                           f"{self.key}={selected[self.key]!r} sha256={selected['checkpoint_sha256']}")

    def _selection(self, selected: dict, status: str) -> dict:
        return {"status": status, "selected_iteration": selected["iteration"],
                "selected_mIoU_full": selected[self.key],
                "selected_mIoU_disease_full": selected[self.disease_key],
                "checkpoint_path": selected["checkpoint_path"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "val_manifest_sha256": selected["val_manifest_sha256"],
                "validations": [r["iteration"] for r in self.records],
                "policy": {"metric": "dataset-level union-present all-class VAL mIoU, full precision "
                                     "(src/eval/metrics.miou_from_confusion)",
                           "rule": "highest; exact tie -> earliest iteration; no tolerance band",
                           "nmf": "M4-V seed 42 per full pass, batch 1, frozen order",
                           "val_manifest": "src.eval.artifacts.hash_split_manifest over the ordered "
                                           "(pass position, stem, stem) rows; identical at every validation",
                           "test_split": "never used"}}


@HOOKS.register_module()
class TeacherInitCompatibilityHook(Hook):
    """The B61 150 -> 116 rule, enforced at load time: only the classifier may be class-count
    incompatible. Any other missing, unexpected or shape-mismatched key fails closed."""

    priority = "VERY_HIGH"

    def after_load_checkpoint(self, runner, checkpoint) -> None:
        state = checkpoint.get("state_dict", checkpoint)
        state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
        model_sd = _unwrap(runner.model).state_dict()
        missing = sorted(set(model_sd) - set(state))
        unexpected = sorted(set(state) - set(model_sd))
        mismatch = sorted(k for k in set(model_sd) & set(state)
                          if tuple(model_sd[k].shape) != tuple(state[k].shape))
        resumed = bool(getattr(runner, "_resume", False))
        ok = not missing and not unexpected and (
            (mismatch == sorted(CLASSIFIER_KEYS)
             and state["decode_head.conv_seg.weight"].shape[0] != model_sd["decode_head.conv_seg.weight"].shape[0])
            or (resumed and not mismatch))
        record = {"matched_same_shape": len(set(model_sd) & set(state)) - len(mismatch),
                  "classifier_mismatch": mismatch, "missing": missing, "unexpected": unexpected,
                  "source_classes": int(state["decode_head.conv_seg.weight"].shape[0])
                  if "decode_head.conv_seg.weight" in state else None,
                  "model_classes": int(model_sd["decode_head.conv_seg.weight"].shape[0]),
                  "resume": resumed, "pass": ok}
        runner.message_hub.update_info("init_compatibility", record)
        runner.logger.info(f"Init compatibility (150->116 classifier-only rule): {json.dumps(record)}")
        if not ok:
            raise RuntimeError(f"init checkpoint violates the classifier-only rule: {record}")
