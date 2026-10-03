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

STRICT LOAD (R6, L-CKPT-GUARD; IMPLEMENTATION_CONTRACT B1): the segmentor is built from the config
WITHOUT weights (`init_model(config, None)`), its `architecture_signature` is taken, and the
checkpoint's state must then match `model.state_dict()` exactly under this module's own comparison:
no missing, unexpected, shape-mismatched, dtype-mismatched or non-tensor entry, and no key that the
`module.` strip collapses. Only then does `load_state_dict(strict=True)` copy the weights. torch's
strict flag alone is not the check: it accepts a state without `num_batches_tracked` and `copy_` casts
dtypes. mmengine's non-strict `load_checkpoint` is not used. The build record carries the provenance
fields `TeacherProvenance` records (src/distill/teacher.py).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .nmf_stream import M4_NMF_SEED, NMFStream, NMFStreamError, attach_nmf_stream, isolated_nmf_modules
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


class TeacherStateDictMismatch(TeacherCheckpointInvalid):
    """R6: the checkpoint's state is refused by the strict teacher check: it does not match the built
    teacher exactly, or the `module.` strip collapses keys (raised before any build). One code.

    The five classes are exact counts, and their key lists are fields: `missing` (in the model, not in
    the checkpoint), `unexpected` (the reverse), `shape`, `dtype` and `non_tensor`. `duplicate` lists
    the keys that two checkpoint keys collapse onto when the `module.` prefix is stripped. The message
    names at most MAX_NAMES keys per class.
    """

    code = "teacher_state_dict_mismatch"
    CLASSES = ("missing", "unexpected", "shape", "dtype", "non_tensor")
    MAX_NAMES = 20

    def __init__(self, *, missing=(), unexpected=(), shape=(), dtype=(), non_tensor=(), duplicate=(),
                 source: str = ""):
        self.missing, self.unexpected, self.shape = list(missing), list(unexpected), list(shape)
        self.dtype, self.non_tensor, self.duplicate = list(dtype), list(non_tensor), list(duplicate)
        counts = " ".join(f"{c}={len(getattr(self, c))}" for c in (*self.CLASSES, "duplicate"))
        lines = [f"[{self.code}] teacher checkpoint {source} is refused by the strict teacher check: "
                 f"{counts}"]
        for c in (*self.CLASSES, "duplicate"):
            names = getattr(self, c)
            if names:
                more = (f" (+{len(names) - self.MAX_NAMES} more)" if len(names) > self.MAX_NAMES
                        else "")
                lines.append(f"  {c}: {names[:self.MAX_NAMES]}{more}")
        super().__init__("\n".join(lines))

    @property
    def counts(self) -> dict:
        return {c: len(getattr(self, c)) for c in (*self.CLASSES, "duplicate")}


class TeacherArchitectureMismatch(RuntimeError):
    """Raised when the built teacher does not expose the expected Stage-3 / class-space semantics."""


# ------------------------------------------------------------------ checkpoint handling
def load_teacher_state_dict(ckpt_path: str | Path, map_location: str = "cpu") -> dict:
    """Read a teacher checkpoint and return a clean `state_dict`.

    Accepts the three formats realistically produced by MMSeg / the B1 fine-tune:
      * a raw `state_dict` mapping name -> tensor,
      * `{"state_dict": {...}}`,
      * a full checkpoint dict carrying metadata, e.g. `{"meta": {...}, "state_dict": {...}}`.

    `module.` prefixes (distributed training) are stripped; a strip that collapses two keys onto one
    is refused (`TeacherStateDictMismatch`, `duplicate`). Anything else — a non-dict payload, an
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

    stripped = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    if len(stripped) != len(state):
        seen: dict[str, int] = {}
        for k in state:
            name = k[len("module."):] if k.startswith("module.") else k
            seen[name] = seen.get(name, 0) + 1
        raise TeacherStateDictMismatch(duplicate=sorted(k for k, n in seen.items() if n > 1),
                                       source=str(path))
    state = stripped
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


# ------------------------------------------------------------------ strict load (R6)
def architecture_signature(model: nn.Module) -> str:
    """SHA-256 of the canonical JSON list of [name, shape, dtype] over `model.state_dict()`, sorted by
    name (compact separators, ASCII). A tensor census of the built segmentor, taken before any weight
    is loaded; it identifies the parameter and buffer layout, not the model's behaviour."""
    entries = [[name, [int(d) for d in t.shape], str(t.dtype)]
               for name, t in sorted(model.state_dict().items())]
    blob = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def strict_load_teacher_state(model: nn.Module, state: dict, *, source: str = "") -> dict:
    """Load `state` into `model` only if it matches `model.state_dict()` exactly (R6).

    The comparison is this function's own: every model key present, no other key, every value a
    tensor of the model's shape and dtype. Any non-zero count raises `TeacherStateDictMismatch`. Only
    then `load_state_dict(strict=True)` copies the weights. Returns the five counts and the duplicate
    count (all 0; a duplicate is refused earlier, by load_teacher_state_dict) and the number of
    entries."""
    target = model.state_dict()
    missing = sorted(k for k in target if k not in state)
    unexpected = sorted(k for k in state if k not in target)
    non_tensor = sorted(k for k, v in state.items() if not torch.is_tensor(v))
    shape, dtype = [], []
    for k in sorted(set(state) & set(target)):
        v = state[k]
        if not torch.is_tensor(v):
            continue
        if tuple(v.shape) != tuple(target[k].shape):
            shape.append(k)
        if v.dtype != target[k].dtype:
            dtype.append(k)
    if missing or unexpected or shape or dtype or non_tensor:
        raise TeacherStateDictMismatch(missing=missing, unexpected=unexpected, shape=shape,
                                       dtype=dtype, non_tensor=non_tensor, source=source)
    model.load_state_dict(state, strict=True)
    return {"missing": 0, "unexpected": 0, "shape": 0, "dtype": 0, "non_tensor": 0, "duplicate": 0,
            "entries": len(target)}


def model_cfg_sha256(model: nn.Module) -> str:
    """SHA-256 of the canonical JSON (sorted keys, compact separators, ASCII) of the loaded instance's
    resolved `cfg.model`: the config file merged with its `_base_` from the installed mmseg, so it
    pins the NMF settings the file itself does not hold. A value json cannot encode raises."""
    blob = json.dumps(model.cfg.model.to_dict(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _build_record(model: nn.Module, *, mmseg_built: bool, config_path, signature: str,
                  strict_load: dict) -> dict:
    """The provenance fields `build_segnext_teacher` contributes (TeacherProvenance, DL-50). The config
    fields exist only for a model the mmseg factory built; the components hashes only when the decode
    head is teacher_components' IsolatedNMFLightHamHead. Every value is a plain type."""
    record = {"config_path": None, "config_sha256": None, "ham_kwargs": None,
              "architecture_signature": signature, "teacher_components_sha256": None,
              "reused_module_hashes": None, "model_cfg_sha256": None}
    if mmseg_built:
        resolved = Path(config_path).resolve()
        ham = model.cfg.model.decode_head.ham_kwargs.to_dict()
        json.dumps(ham)                               # plain values only, or this raises
        record.update(config_path=str(resolved), config_sha256=_sha256_file(resolved),
                      ham_kwargs=ham, model_cfg_sha256=model_cfg_sha256(model))
    tc = sys.modules.get("src.training.teacher_components")
    head_cls = getattr(tc, "IsolatedNMFLightHamHead", None) if tc is not None else None
    if head_cls is not None and type(getattr(model, "decode_head", None)) is head_cls:
        record.update(teacher_components_sha256=tc.COMPONENTS_PROVENANCE["sha256"],
                      reused_module_hashes=dict(tc.reused_module_hashes()))
    record["strict_load"] = dict(strict_load)
    return record


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
        # M4 (B61 §4): a stock NMF2D would draw its bases from the caller's global CPU stream. Only the
        # thesis config's IsolatedNMFLightHamHead can honour M4-V / M4-KD, so a stock one fails closed.
        stock_nmf = [m for m in model.modules() if type(m).__name__ == "NMF2D"]
        if stock_nmf:
            raise TeacherArchitectureMismatch(
                "the teacher uses the stock NMF2D, which cannot isolate its random-basis draw (M4). "
                "Build it from the thesis teacher config (decode head IsolatedNMFLightHamHead).")
        self._n_isolated_nmf = len(isolated_nmf_modules(model))
        self.nmf_stream = None

    def begin_nmf_stream(self, policy: str, seed: int = M4_NMF_SEED) -> dict | None:
        """Attach a fresh private NMF stream seeded `seed`.

        M4-V: call once at the start of each complete evaluation pass (batch size 1, frozen order).
        M4-KD: call once at the start of the E2/E3 run; the stream then advances across teacher calls
        and is never reset per batch. Only the NMF basis draw consumes it; the caller's CPU RNG and
        every CUDA generator are untouched.

        Returns None for a model with no NMF at all (a synthetic stub): there is nothing to isolate.
        A stock NMF2D never reaches here (refused at construction); real-run callers require non-None.
        """
        if self._n_isolated_nmf == 0:
            return None
        stream = NMFStream(seed, policy)
        attach_nmf_stream(self.model, stream)            # exactly one isolated module, or fail closed
        self.nmf_stream = stream
        return stream.describe()

    def end_nmf_stream(self) -> dict | None:
        """Detach the stream (end of an evaluation pass). Returns its final description."""
        if self.nmf_stream is None:
            return None
        attach_nmf_stream(self.model, None)
        stream, self.nmf_stream = self.nmf_stream, None
        return stream.describe()

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
        if self._n_isolated_nmf and self.nmf_stream is None:
            raise NMFStreamError(
                "the frozen SegNeXt teacher has no NMF stream attached; call begin_nmf_stream('M4-KD') "
                "once per E2/E3 run, or begin_nmf_stream('M4-V') per evaluation pass (M4, B61 §4)")
        # Stage-3 extraction runs outside any RNG window; only the NMF basis draw inside
        # decode_head.forward consumes the attached private stream.
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
    """Build the frozen-ready SegNeXt-B teacher from an explicit checkpoint (R6 order).

    Parse and validate the checkpoint, then the factory builds the segmentor without weights, then
    its `architecture_signature`, then the strict load, then the adapter, which carries the build
    record (`build_record`). One model instance; nothing reads the config file before the factory
    returns. `model_factory(config_path, ckpt_path) -> nn.Module` is injectable so the adapter's
    framework-independent behaviour is testable without mmseg; whatever it returns is loaded
    strictly. When omitted, MMSegmentation is required and `TeacherStackMissing` is raised if it is
    not installed.
    """
    path = Path(ckpt_path)
    state = load_teacher_state_dict(path)  # validate the checkpoint BEFORE constructing anything
    mmseg_built = model_factory is None
    factory = _mmseg_model_factory if mmseg_built else model_factory
    model = factory(config_path, str(path))
    signature = architecture_signature(model)
    report = strict_load_teacher_state(model, state, source=str(path))
    adapter = SegNeXtTeacherAdapter(model, num_classes=num_classes)
    adapter.build_record = _build_record(model, mmseg_built=mmseg_built, config_path=config_path,
                                         signature=signature, strict_load=report)
    return adapter


def _mmseg_model_factory(config_path: str | None, ckpt_path: str) -> nn.Module:
    """Construct the SegNeXt segmentor through MMSegmentation's own API, WITHOUT weights.

    `init_model(config, None)`: `build_segnext_teacher` loads the checkpoint strictly afterwards, so
    mmengine's non-strict `load_checkpoint` never runs. `ckpt_path` keeps the factory signature."""
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
    return init_model(str(config_path), None, device="cpu")


def segnext_builder(config_path: str | None = None,
                    model_factory: Callable[[str | None, str], nn.Module] | None = None,
                    num_classes: int = TEACHER_NUM_CLASSES) -> Callable[[Path], nn.Module]:
    """Return the `builder` callable consumed by `load_frozen_teacher(..., builder=...)`."""

    def _build(path: Path) -> nn.Module:
        return build_segnext_teacher(path, config_path=config_path,
                                     model_factory=model_factory, num_classes=num_classes)

    _build.__name__ = "segnext_mscan_b_builder"
    return _build
