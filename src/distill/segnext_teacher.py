"""Concrete SegNeXt-B / MSCAN-B teacher adapter for E2/E3 distillation.

All MMSegmentation-specific code lives HERE, not in `teacher.py`, so `FrozenTeacher` stays
framework-agnostic and the normal E1/E2/E3 paths (and every synthetic test) import without ever
touching mmcv/mmseg. The mmseg import happens lazily inside `build_segnext_teacher`.

VERIFIED FACTS this adapter relies on (docs/teacher_prep_runbook.md §2-§3, corroborated by
IMPLEMENTATION_CONTRACT.md B1/B3):
  * stock init config      `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512`  (ADE20K, 150 classes)
  * fine-tune derived cfg  `segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512`  (116 classes)
  * MSCAN-B `embed_dims = [64, 128, 320, 512]`, `depths = [3, 3, 12, 3]`
    -> the backbone emits four stage features at strides 4 / 8 / 16 / 32
    -> **Stage-3 = 320 channels at stride 16** — the CWD `C_t` tap required by contract B3
  * decode head = LightHamHead, `channels = ham_channels = 512`
  * pinned stack MMSegmentation 1.2.2 + mmcv 2.1.0 (NOT part of requirements-e1.txt)

STAGE-3 RESOLUTION (contract B4 discipline): internal MSCAN/MMSeg attribute names vary across
configs and versions, so nothing here hooks a guessed attribute string. The adapter takes the
backbone's own tuple of stage outputs and selects the element that satisfies BOTH the channel count
(320) and the spatial stride (16) relative to the input, requiring exactly one match and failing
loudly otherwise. Architecture semantics are verified at runtime; names are not assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .teacher import TeacherStackMissing

# --- verified architecture constants (docs/teacher_prep_runbook.md §2-§3) ---
STOCK_INIT_CONFIG_STEM = "segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512"
PLANTSEG_CONFIG_STEM = "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512"
MSCAN_B_EMBED_DIMS = (64, 128, 320, 512)
STAGE3_CHANNELS = 320
STAGE3_STRIDE = 16
TEACHER_NUM_CLASSES = 116


class TeacherCheckpointInvalid(ValueError):
    """Raised when a teacher checkpoint is unreadable, malformed, or not a SegNeXt checkpoint."""


class TeacherArchitectureMismatch(RuntimeError):
    """Raised when the built teacher does not expose the expected Stage-3 / class-space semantics."""


# ------------------------------------------------------------------ checkpoint handling
def load_teacher_state_dict(ckpt_path: str | Path, map_location: str = "cpu") -> dict:
    """Read a teacher checkpoint and return a clean `state_dict`.

    Accepts the three formats realistically produced by MMSeg / the B1 fine-tune:
      * a raw `state_dict` mapping name -> tensor,
      * `{"state_dict": {...}}`,
      * a full checkpoint dict carrying metadata, e.g. `{"meta": {...}, "state_dict": {...}}`.

    `module.` prefixes (distributed training) are stripped. Anything else — a non-dict payload, an
    empty mapping, a mapping without tensors, or a state_dict that is not a SegNeXt/MSCAN encoder-
    decoder — raises `TeacherCheckpointInvalid`. Unrelated or random checkpoints are never silently
    accepted (contract: no silent teacher substitution).
    """
    path = Path(ckpt_path)
    try:
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
    except Exception as e:  # noqa: BLE001
        raise TeacherCheckpointInvalid(
            f"could not read teacher checkpoint {path}: {type(e).__name__}: {e}") from e

    if not isinstance(payload, dict):
        raise TeacherCheckpointInvalid(
            f"teacher checkpoint {path} holds {type(payload).__name__}, expected a dict "
            "(raw state_dict, {'state_dict': ...}, or a checkpoint dict with metadata)")

    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise TeacherCheckpointInvalid(f"teacher checkpoint {path} contains no usable state_dict")

    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    if not any(torch.is_tensor(v) for v in state.values()):
        raise TeacherCheckpointInvalid(
            f"teacher checkpoint {path} state_dict holds no tensors — not a model checkpoint")

    has_backbone = any(k.startswith("backbone.") for k in state)
    has_head = any(k.startswith("decode_head.") for k in state)
    if not (has_backbone and has_head):
        raise TeacherCheckpointInvalid(
            f"teacher checkpoint {path} does not look like a SegNeXt encoder-decoder "
            f"(backbone.* keys: {has_backbone}, decode_head.* keys: {has_head}). Refusing to load an "
            "unrelated checkpoint as the distillation teacher.")
    return state


def checkpoint_metadata(ckpt_path: str | Path, map_location: str = "cpu") -> dict:
    """Return the checkpoint's `meta` block if present (provenance aid); {} when absent."""
    payload = torch.load(str(Path(ckpt_path)), map_location=map_location, weights_only=False)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    return meta if isinstance(meta, dict) else {}


# ------------------------------------------------------------------ Stage-3 resolution
def select_stride16_feature(feats: Sequence[torch.Tensor], input_hw: tuple[int, int],
                            expected_ch: int = STAGE3_CHANNELS,
                            expected_stride: int = STAGE3_STRIDE) -> torch.Tensor:
    """Pick the MSCAN Stage-3 feature from the backbone's stage outputs, by SEMANTICS not by name.

    Requires exactly one tensor with `expected_ch` channels whose spatial size equals the input size
    divided by `expected_stride`. Zero or multiple matches raise `TeacherArchitectureMismatch` with a
    description of what the backbone actually produced — the adapter never falls back to "the
    nearest-looking layer".
    """
    if not isinstance(feats, (list, tuple)) or not feats:
        raise TeacherArchitectureMismatch(
            f"teacher backbone returned {type(feats).__name__}, expected a non-empty sequence of "
            "stage feature maps")
    ih, iw = input_hw
    want = (ih // expected_stride, iw // expected_stride)
    described, matches = [], []
    for i, f in enumerate(feats):
        if not torch.is_tensor(f) or f.dim() != 4:
            described.append(f"[{i}] {type(f).__name__}")
            continue
        c, (h, w) = f.shape[1], f.shape[-2:]
        stride = ih // h if h else 0
        described.append(f"[{i}] C={c} {h}x{w} stride~{stride}")
        if c == expected_ch and (h, w) == want:
            matches.append(f)
    if len(matches) != 1:
        raise TeacherArchitectureMismatch(
            f"expected exactly one stride-{expected_stride} feature with {expected_ch} channels "
            f"(MSCAN-B Stage-3, embed_dims={list(MSCAN_B_EMBED_DIMS)}); found {len(matches)}. "
            f"Backbone produced: {'; '.join(described)}. Input {ih}x{iw} -> wanted {want[0]}x{want[1]}.")
    return matches[0]


# ------------------------------------------------------------------ adapter
class SegNeXtTeacherAdapter(nn.Module):
    """Expose an MMSeg SegNeXt encoder-decoder as `{"logits", "feat_s16"}` for `FrozenTeacher`.

    `model` must provide the standard MMSeg segmentor surface: `extract_feat(x)` (or `backbone(x)`)
    returning the tuple of stage features, and `decode_head` whose `forward(feats)` returns seg
    logits. That surface is small enough to stub in tests, which is exactly how this class is
    verified without mmseg installed.
    """

    def __init__(self, model: nn.Module, *, num_classes: int = TEACHER_NUM_CLASSES,
                 stage3_channels: int = STAGE3_CHANNELS, stage3_stride: int = STAGE3_STRIDE):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.stage3_channels = stage3_channels
        self.stage3_stride = stage3_stride
        if not (hasattr(model, "extract_feat") or hasattr(model, "backbone")):
            raise TeacherArchitectureMismatch(
                "teacher model exposes neither extract_feat() nor backbone(); cannot reach the "
                "MSCAN stage features required for the CWD Stage-3 tap")
        if not hasattr(model, "decode_head"):
            raise TeacherArchitectureMismatch("teacher model has no decode_head")
        self._verify_class_space()

    def _verify_class_space(self) -> None:
        head = self.model.decode_head
        conv_seg = getattr(head, "conv_seg", None)
        declared = getattr(head, "num_classes", None)
        found = None
        if conv_seg is not None and hasattr(conv_seg, "out_channels"):
            found = int(conv_seg.out_channels)
        elif declared is not None:
            found = int(declared)
        if found is not None and found != self.num_classes:
            raise TeacherArchitectureMismatch(
                f"teacher decode head produces {found} classes, expected {self.num_classes} "
                "(PlantSeg: background 0 + diseases 1..115). The stock ADE20K head has 150 and must "
                "be re-headed to 116 at load — see docs/teacher_prep_runbook.md §5.")

    def _stage_features(self, x: torch.Tensor):
        if hasattr(self.model, "extract_feat"):
            return self.model.extract_feat(x)
        return self.model.backbone(x)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self._stage_features(x)
        feat_s16 = select_stride16_feature(feats, tuple(x.shape[-2:]),
                                           self.stage3_channels, self.stage3_stride)
        logits = self.model.decode_head.forward(feats)
        if torch.is_tensor(logits) and logits.dim() == 4 and logits.shape[1] != self.num_classes:
            raise TeacherArchitectureMismatch(
                f"teacher logits have {logits.shape[1]} channels, expected {self.num_classes}")
        return {"logits": logits, "feat_s16": feat_s16}


# ------------------------------------------------------------------ builders
def build_segnext_teacher(ckpt_path: str | Path, config_path: str | None = None,
                          model_factory: Callable[[str | None, str], nn.Module] | None = None,
                          num_classes: int = TEACHER_NUM_CLASSES) -> nn.Module:
    """Build the frozen-ready SegNeXt-B teacher from an explicit checkpoint.

    `model_factory(config_path, ckpt_path) -> nn.Module` is injectable so the adapter's
    framework-independent behaviour is testable without mmseg. When omitted, MMSegmentation is
    required and `TeacherStackMissing` is raised if it is not installed.
    """
    path = Path(ckpt_path)
    load_teacher_state_dict(path)          # validate the checkpoint BEFORE constructing anything
    if model_factory is None:
        model_factory = _mmseg_model_factory
    model = model_factory(config_path, str(path))
    return SegNeXtTeacherAdapter(model, num_classes=num_classes)


def _mmseg_model_factory(config_path: str | None, ckpt_path: str) -> nn.Module:
    """Construct the SegNeXt segmentor through MMSegmentation's own API."""
    try:
        from mmseg.apis import init_model
    except Exception as e:  # noqa: BLE001
        raise TeacherStackMissing(
            "building the SegNeXt-B teacher requires MMSegmentation 1.2.2 + mmcv 2.1.0, which are "
            "deliberately excluded from the E1/E2/E3 student stack (requirements-e1.txt). Install "
            "them in the teacher environment described in docs/teacher_prep_runbook.md §4 (note the "
            "mmseg MMCV_MAX compat edit in §4.5), or pass an explicit model_factory. "
            f"(import failed: {type(e).__name__}: {e})") from e
    if not config_path:
        raise TeacherCheckpointInvalid(
            "a teacher config path is required to build the SegNeXt segmentor — pass "
            f"--teacher-config pointing at the derived '{PLANTSEG_CONFIG_STEM}.py' "
            "(docs/teacher_prep_runbook.md §3)")
    return init_model(str(config_path), str(ckpt_path), device="cpu")


def segnext_builder(config_path: str | None = None,
                    model_factory: Callable[[str | None, str], nn.Module] | None = None,
                    num_classes: int = TEACHER_NUM_CLASSES) -> Callable[[Path], nn.Module]:
    """Return the `builder` callable consumed by `load_frozen_teacher(..., builder=...)`."""

    def _build(path: Path) -> nn.Module:
        return build_segnext_teacher(path, config_path=config_path,
                                     model_factory=model_factory, num_classes=num_classes)

    _build.__name__ = "segnext_mscan_b_builder"
    return _build
