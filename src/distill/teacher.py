"""Frozen SegNeXt-B / MSCAN-B teacher adapter for E2 (Logit KD) and E3 (Logit KD + CWD).

Contract (docs/IMPLEMENTATION_CONTRACT.md B1/B3):
  * teacher = SegNeXt-B / MSCAN-B, fine-tuned on PlantSeg; descriptive upper-bound reference only.
  * during student distillation the teacher is FROZEN: eval mode, no parameter updates, no gradient
    flow back into it.
  * E3 additionally needs the teacher's stride-16 MSCAN-B Stage-3 feature (C = 320).

SAFETY MODEL — the three failure modes this module makes impossible:
  1. **No silent random teacher.** A real run must pass an existing checkpoint path; a missing or
     unreadable path raises `TeacherCheckpointMissing`. The only weight-free teacher is
     `MockTeacher`, which must be constructed explicitly by name and is refused by the real-run path.
  2. **No silent download.** Nothing here fetches anything from the network.
  3. **No silent substitution.** The builder is explicit; provenance (path, sha256, builder name) is
     recorded on the instance so a run manifest can capture it later. With `expected_sha256` the
     checkpoint is hashed and compared BEFORE anything reads it or builds a model (R6), and the
     default builder loads it strictly (`segnext_teacher.strict_load_teacher_state`).

Teacher CONSTRUCTION from a SegNeXt checkpoint needs MMSegmentation 1.2.2 + mmcv 2.1.0, which are
deliberately absent from the E1/E2/E3 student stack (`requirements-e1.txt`). The mmseg builder is
therefore resolved lazily and raises a clear, actionable error when that stack is not installed —
teacher preparation is a separate workflow (docs/teacher_prep_runbook.md).
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

TEACHER_STRIDE16_CH = 320   # LOCKED: contract B3 — MSCAN-B Stage-3 stride-16 channel count


class TeacherCheckpointMissing(FileNotFoundError):
    """Raised when the required fine-tuned teacher checkpoint is absent or unreadable."""


class TeacherStackMissing(RuntimeError):
    """Raised when MMSegmentation/mmcv are needed to build the teacher but are not installed."""


class TeacherChecksumFormatError(ValueError):
    """An expected teacher SHA-256 that is not 64 lowercase hexadecimal characters ("" included)."""

    code = "teacher_ckpt_sha256_format"


class TeacherChecksumMismatch(ValueError):
    """The teacher checkpoint's SHA-256 is not the expected one; nothing was read or built (R6)."""

    code = "teacher_ckpt_sha256_mismatch"

    def __init__(self, expected: str, actual: str, path=None):
        self.expected, self.actual = expected, actual
        super().__init__(f"[{self.code}] teacher checkpoint {path} has sha256 {actual}, expected "
                         f"{expected}; it was neither parsed nor built")


SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def sha256_format_error(value) -> str | None:
    """None when `value` is 64 lowercase hexadecimal characters, else the refusal text."""
    if isinstance(value, str) and SHA256_HEX.fullmatch(value):
        return None
    return (f"{value!r} is not a SHA-256: 64 lowercase hexadecimal characters are required "
            "(PowerShell's Get-FileHash prints uppercase; lower-case it)")


@dataclass(frozen=True)
class TeacherProvenance:
    """The teacher's identity (DL-50). Twelve keys, in this order; every value is a plain type. The
    eight after `ckpt_bytes` are null when not applicable: `expected_sha256` when none was supplied,
    the build-record fields for a builder that bypasses `segnext_teacher.build_segnext_teacher`."""
    builder: str
    ckpt_path: str | None
    ckpt_sha256: str | None
    ckpt_bytes: int | None
    expected_sha256: str | None = None
    config_path: str | None = None
    config_sha256: str | None = None
    ham_kwargs: dict | None = None
    architecture_signature: str | None = None
    teacher_components_sha256: str | None = None
    reused_module_hashes: dict | None = None
    model_cfg_sha256: str | None = None

    def as_dict(self) -> dict:
        return {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}


# The provenance fields a builder contributes through its module's `build_record`.
BUILD_RECORD_KEYS = ("config_path", "config_sha256", "ham_kwargs", "architecture_signature",
                     "teacher_components_sha256", "reused_module_hashes", "model_cfg_sha256")


@dataclass(frozen=True)
class TeacherOutput:
    """Teacher tensors already aligned to the student's spatial grids (detached)."""
    logits: torch.Tensor          # [B, num_classes, H, W]
    feat_s16: torch.Tensor | None  # [B, 320, H/16, W/16] — None when the teacher exposes no feature


class FrozenTeacher(nn.Module):
    """Wraps a teacher module and guarantees it stays frozen and gradient-free.

    `module` must return either a logits tensor, or a `(logits, feat_s16)` pair, or a mapping with
    keys `logits` / `feat_s16`. `feature_getter`, when given, is used instead to pull the stride-16
    feature out of a framework-specific output.
    """

    def __init__(self, module: nn.Module, *,
                 feature_getter: Callable[[object], torch.Tensor] | None = None,
                 provenance: TeacherProvenance | None = None,
                 expected_feat_ch: int = TEACHER_STRIDE16_CH):
        super().__init__()
        self.teacher = module
        self._feature_getter = feature_getter
        self.provenance = provenance
        self.expected_feat_ch = expected_feat_ch
        # Freeze: eval mode + no grads. Re-applied by train() below so a parent .train() cannot
        # accidentally put the teacher back into training mode.
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenTeacher":
        """Ignore training-mode propagation: the teacher is ALWAYS in eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Teacher parameters that would still receive updates — must always be empty.

        Standard `nn.Module` semantics are kept deliberately: `parameters()` is NOT overridden, so
        the teacher's tensors stay present and introspectable (state_dict, device moves, parameter
        counts, debugging). Exclusion from training is enforced where it belongs — every parameter
        carries `requires_grad=False`, the forward runs under `no_grad` and returns detached
        tensors, and the optimizer is built explicitly from the student (+ E3 projection) and is
        asserted to share no parameter with this module.
        """
        return [p for p in self.teacher.parameters() if p.requires_grad]

    def parameter_ids(self) -> set[int]:
        """Identity of every teacher parameter, for the optimizer-disjointness assertion."""
        return {id(p) for p in self.teacher.parameters()}

    def begin_nmf_stream(self, policy: str, seed: int = 42) -> dict | None:
        """Start the teacher's private NMF stream (M4-KD once per E2/E3 run; M4-V per evaluation pass).

        Returns the stream description, or None when the wrapped module has no NMF to isolate (the
        explicit MockTeacher). A real run must require a non-None result.
        """
        begin = getattr(self.teacher, "begin_nmf_stream", None)
        return None if begin is None else begin(policy, seed)

    def nmf_stream_state(self) -> dict | None:
        stream = getattr(self.teacher, "nmf_stream", None)
        return None if stream is None else stream.describe()

    @staticmethod
    def _unpack(out) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(out, torch.Tensor):
            return out, None
        if isinstance(out, dict):
            return out["logits"], out.get("feat_s16")
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out[0], out[1]
        raise TypeError(
            f"teacher module returned {type(out).__name__}; expected a logits Tensor, a "
            "(logits, feat_s16) pair, or a mapping with 'logits'/'feat_s16'")

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, logits_size: tuple[int, int] | None = None,
                feat_size: tuple[int, int] | None = None) -> TeacherOutput:
        """Run the frozen teacher and align its outputs to the student's grids.

        `logits_size` / `feat_size` are the student's spatial sizes; the teacher's maps are
        bilinearly resampled onto them when they differ. Everything is detached: the returned
        tensors carry no graph, so no gradient can reach the teacher.
        """
        out = self.teacher(x)
        logits, feat = self._unpack(out)
        if self._feature_getter is not None:
            feat = self._feature_getter(out)

        if logits_size is not None and tuple(logits.shape[-2:]) != tuple(logits_size):
            logits = F.interpolate(logits, size=logits_size, mode="bilinear", align_corners=False)
        if feat is not None:
            if feat.shape[1] != self.expected_feat_ch:
                raise ValueError(
                    f"teacher stride-16 feature has {feat.shape[1]} channels, expected "
                    f"{self.expected_feat_ch} (contract B3: MSCAN-B Stage-3 C=320)")
            if feat_size is not None and tuple(feat.shape[-2:]) != tuple(feat_size):
                feat = F.interpolate(feat, size=feat_size, mode="bilinear", align_corners=False)
        return TeacherOutput(logits=logits.detach(), feat_s16=None if feat is None else feat.detach())


class MockTeacher(nn.Module):
    """EXPLICIT, weight-free teacher for synthetic smoke tests ONLY.

    Never reachable from the real-run path: `train_distill.py` refuses `--real-run` unless a real
    checkpoint is supplied, and this class must be named explicitly to be constructed.
    """

    is_mock = True

    def __init__(self, num_classes: int, feat_ch: int = TEACHER_STRIDE16_CH,
                 logits_stride: int = 8, feat_stride: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.feat_ch = feat_ch
        self.logits_stride = logits_stride
        self.feat_stride = feat_stride
        self.stem = nn.Conv2d(3, feat_ch, kernel_size=1, bias=False)
        self.cls = nn.Conv2d(feat_ch, num_classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        f = self.stem(F.adaptive_avg_pool2d(x, (max(h // self.feat_stride, 1),
                                                max(w // self.feat_stride, 1))))
        logits = self.cls(F.adaptive_avg_pool2d(f, (max(h // self.logits_stride, 1),
                                                    max(w // self.logits_stride, 1))))
        return {"logits": logits, "feat_s16": f}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def require_teacher_checkpoint(ckpt_path: str | None) -> Path:
    """Validate the teacher checkpoint path. Fails LOUD — never returns a fallback."""
    if not ckpt_path:
        raise TeacherCheckpointMissing(
            "no teacher checkpoint given. E2/E3 require the fine-tuned SegNeXt-B/MSCAN-B teacher; "
            "pass --teacher-ckpt <path>. There is no random/default teacher fallback "
            "(see docs/teacher_prep_runbook.md).")
    p = Path(ckpt_path).expanduser()
    if not p.exists():
        raise TeacherCheckpointMissing(
            f"teacher checkpoint not found: {p}. Prepare the fine-tuned SegNeXt-B teacher first "
            "(docs/teacher_prep_runbook.md); distillation will not start without it.")
    if not p.is_file():
        raise TeacherCheckpointMissing(f"teacher checkpoint path is not a file: {p}")
    return p.resolve()


def build_mmseg_teacher(ckpt_path: Path, config_path: str | None = None) -> nn.Module:
    """Build the SegNeXt-B / MSCAN-B teacher from an MMSegmentation checkpoint.

    Delegates to `segnext_teacher.build_segnext_teacher`, which owns every MMSeg-specific detail.
    The import is LAZY so this module — and therefore the whole E1/E2/E3 path — never pulls
    mmcv/mmseg on a normal run. Raises `TeacherStackMissing` when the teacher stack is unavailable.
    """
    from .segnext_teacher import build_segnext_teacher  # local import: keeps teacher.py mmseg-free
    return build_segnext_teacher(ckpt_path, config_path=config_path)


def load_frozen_teacher(ckpt_path: str | None, *,
                        builder: Callable[[Path], nn.Module] | None = None,
                        config_path: str | None = None,
                        record_sha256: bool = True,
                        expected_sha256: str | None = None) -> FrozenTeacher:
    """Load the frozen teacher from an explicit checkpoint path.

    Fails loud when the checkpoint is missing. When `expected_sha256` is not None it must be 64
    lowercase hex characters (`TeacherChecksumFormatError`, "" included); the file is then hashed once
    and compared BEFORE any `torch.load` or builder call (`TeacherChecksumMismatch`), and that hash is
    the recorded `ckpt_sha256` whatever `record_sha256` says. The default builder is
    `segnext_teacher.segnext_builder(config_path)` (recorded as `segnext_mscan_b_builder`), which
    builds without weights and loads the checkpoint strictly; `builder` lets tests inject another.
    """
    path = require_teacher_checkpoint(ckpt_path)
    if expected_sha256 is not None:
        error = sha256_format_error(expected_sha256)
        if error is not None:
            raise TeacherChecksumFormatError(f"[{TeacherChecksumFormatError.code}] expected teacher "
                                             f"sha256 {error}")
    digest = _sha256(path) if (record_sha256 or expected_sha256 is not None) else None
    if expected_sha256 is not None and digest != expected_sha256:
        raise TeacherChecksumMismatch(expected_sha256, digest, path)
    if builder is None:
        from .segnext_teacher import segnext_builder  # local import: keeps teacher.py mmseg-free
        make = segnext_builder(config_path)
    else:
        make = builder
    module = make(path)
    record = getattr(module, "build_record", None) or {}
    prov = TeacherProvenance(
        builder=getattr(make, "__name__", type(make).__name__),
        ckpt_path=str(path),
        ckpt_sha256=digest,
        ckpt_bytes=path.stat().st_size,
        expected_sha256=expected_sha256,
        **{k: copy.deepcopy(record.get(k)) for k in BUILD_RECORD_KEYS},
    )
    return FrozenTeacher(module, provenance=prov)
