"""Shared entry for the four quantization stages, and the guards the PTQ and QAT cores reuse.

    E4 : E1 FP32 checkpoint -> PTQ        E6 : E3 projection-free student -> QAT
    E5 : E1 FP32 checkpoint -> QAT        E7 : E3 projection-free student -> PTQ

`scripts/run_e4.py` … `run_e7.py` are thin wrappers that PIN their stage: the stage is not a CLI
flag, so `run_e4.py` can never be turned into E7.

E4 and E7 (ruling Q1-R2, L-CKPT-GUARD Q-part): `main` refuses every E4 or E7 run once its arguments are
parsed, whatever the flags (`old_runner_real_run`, exit 2), before anything is read, loaded or written. The
PTQ of record is made by `scripts/run_ptq.py`: src/quant/ptq.py's core (L-AM10) on the committed calibration
list of record, bound to its E1 or E3 parent by src/quant/ptq_record.py, which writes
`<stage>_ptq_record.json`. This module has no PTQ path of its own.

E5/E6 QAT is src/quant/qat.py's (AM-4/AM-4a, lanes L-AM4 + L-AM1q): `main` hands those two stages to
`src.quant.qat.main` before any PTQ argument is parsed; scripts/run_e5.py and run_e6.py reach it through
`main`. E6-KD (method "qat_kd"; AM-3, lane L-AM3 B-2) goes to `src.quant.qat_kd.main` the same way, imported inside
its route only; scripts/run_e6kd.py reaches it through `main`.

SAFETY MODEL — each guard below validates hard: random initialization is impossible, every stage loads a
validated source checkpoint, and nothing here downloads, substitutes a model, or falls back silently. Their
users: check_output_dir — src/quant/ptq.py, ptq_record.py and qat.py, scripts/ptq_calibration_range.py and
create_ptq_calibration_index.py; check_source — ptq.py and qat.py; CalibrationDataset (the TRAIN calibration
images) — ptq.py; build_run_provenance — ptq.py and qat_artifacts.py. check_backend and check_calibration (the
legacy E4/E7 calibration index) are kept for scripts/smoke_quant_runners.py only: the PTQ of record selects
QNNPACK in src/quant/ptq.py and pins its list through ptq.load_calibration_list (--expect-list-checksum).
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.quant import QUANT                                     # noqa: E402
from src.training.train_e1 import _assert_outside_repo              # noqa: E402  (reuse, unmodified)
from .calibration import CalibrationIndexError                      # noqa: E402
from .checkpoint import SourceCheckpointInvalid                     # noqa: E402
from .qconfig import QUANT_BACKEND, QuantBackendUnavailable, select_qnnpack_backend  # noqa: E402
from .stages import (load_source_for_stage, require_shared_calibration_index,  # noqa: E402
                     resolve_quant_stage)

NUM_CLASSES = 116
PTQ = QUANT["ptq"]


class QuantRunError(RuntimeError):
    """A named pre-flight/run failure; `code` keeps each guard independently testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ common gates
def check_output_dir(value: str | None, *, create: bool = False) -> Path:
    """Output/work dir must resolve OUTSIDE the git checkout — reuses E1's guard, unmodified."""
    if not value:
        raise QuantRunError("output_dir_unset",
                            "--out-dir is required; quantized artifacts are never written inside "
                            "the repository")
    try:
        resolved = _assert_outside_repo(Path(value))
    except RuntimeError as e:
        raise QuantRunError("output_dir_inside_repo",
                            f"output directory must be outside the repository: {e}") from e
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def check_source(stage: dict, ckpt_path: str | None):
    """Load + validate the source checkpoint through the stage's strict contract."""
    if not ckpt_path:
        raise QuantRunError("source_unset",
                            f"--source-ckpt is required: {stage['name']} starts from the official "
                            f"{stage['source_stage']} FP32 checkpoint. There is no random-init path.")
    p = Path(ckpt_path)
    if not p.is_file():
        raise QuantRunError("source_missing", f"source checkpoint not found: {p}")
    try:
        model, ckpt = load_source_for_stage(stage["key"], p)
    except SourceCheckpointInvalid as e:
        raise QuantRunError("source_invalid",
                            f"{stage['name']} rejected its source checkpoint: {e}") from e
    if getattr(model, "num_classes", NUM_CLASSES) != NUM_CLASSES:
        raise QuantRunError("num_classes_mismatch",
                            f"student has {model.num_classes} classes, expected {NUM_CLASSES}")
    meta = {"path": str(p.resolve()), "sha256": _sha256(p), "bytes": p.stat().st_size,
            "declared_stage": ckpt.get("stage")}
    return model, ckpt, meta


def check_backend() -> str:
    try:
        return select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        raise QuantRunError("backend_unavailable", str(e)) from e


def check_calibration(stage: dict, index_path: str | None, expected_checksum: str | None) -> dict:
    """PTQ stages must consume the shared E4/E7 artifact; E7 must pin E4's exact checksum."""
    if not index_path:
        raise QuantRunError("calibration_unset",
                            f"--calibration-index is required for {stage['name']}: the "
                            f"{PTQ['calibration_num_images']}-image seed-"
                            f"{PTQ['calibration_seed']} TRAIN subset is a frozen artifact shared by "
                            "E4 and E7, never resampled per run.")
    if stage["name"] == "E7" and not expected_checksum:
        raise QuantRunError(
            "calibration_checksum_unset",
            "E7 must pin the EXACT artifact E4 used: pass --calibration-sha256 <E4 checksum>. "
            "Matching seed and count is not sufficient.")
    try:
        return require_shared_calibration_index(index_path, expected_checksum=expected_checksum)
    except CalibrationIndexError as e:
        raise QuantRunError("calibration_invalid", str(e)) from e


# ------------------------------------------------------------------ calibration data
class CalibrationDataset(torch.utils.data.Dataset):
    """TRAIN images under CLEAN/core preprocessing — explicitly NO training augmentation.

    Reuses `src.data.transforms` unmodified. Only the frozen calibration identifiers are read; the
    validation and test splits are never touched.
    """

    def __init__(self, root: Path, ids: list[str]):
        from src.data.transforms import core_preprocess, finalize   # imported, not modified
        self._core, self._finalize = core_preprocess, finalize
        self.root = Path(root)
        img_dir, mask_dir = self.root / "images" / "train", self.root / "annotations" / "train"
        self.pairs = []
        missing = []
        for stem in ids:
            img = next((img_dir / f"{stem}{s}" for s in (".jpg", ".jpeg")
                        if (img_dir / f"{stem}{s}").exists()), None)
            mask = mask_dir / f"{stem}.png"
            if img is None or not mask.exists():
                missing.append(stem)
            else:
                self.pairs.append((img, mask))
        if missing:
            raise QuantRunError("calibration_ids_missing",
                                f"{len(missing)} calibration identifiers are absent from the TRAIN "
                                f"split, e.g. {missing[:5]}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i):
        from PIL import Image
        img_path, mask_path = self.pairs[i]
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            img_np, mask_np = self._core(im, mk)
        return self._finalize(img_np, mask_np)


# ------------------------------------------------------------------ provenance
def build_run_provenance(*, stage: dict, source_meta: dict, method: str, qconfig_summary: dict,
                         calibration: dict | None, training: dict | None,
                         converted_path: Path | None, coverage: dict | None,
                         forward: dict | None, backend: str) -> dict:
    import torchvision
    prov = {
        "stage": stage["name"],
        "source_stage": stage["source_stage"],
        "source_checkpoint": source_meta["path"],
        "source_checkpoint_sha256": source_meta["sha256"],
        "source_checkpoint_bytes": source_meta["bytes"],
        "random_init": False,
        "num_classes": NUM_CLASSES,
        "quantization_method": method,
        "backend": backend,
        "qconfig": qconfig_summary,
        "calibration": calibration,
        "training": training,
        "converted_artifact": None if converted_path is None else str(converted_path),
        "converted_artifact_sha256": (None if converted_path is None or not converted_path.exists()
                                      else _sha256(converted_path)),
        "operator_coverage": coverage,
        "converted_forward": forward,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if stage["source_stage"] in ("E2", "E3"):
        prov["cwd_projection_loaded"] = False
    return prov


# ------------------------------------------------------------------ CLI
def add_common_args(p):
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--source-ckpt", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    # PTQ
    p.add_argument("--calibration-index", default=None)
    p.add_argument("--calibration-sha256", default=None)
    return p


def main(argv, stage_key: str) -> int:
    import argparse
    stage = resolve_quant_stage(stage_key)
    if stage["method"] == "qat":
        from .qat import main as qat_main
        return qat_main(argv, stage_key)
    if stage["method"] == "qat_kd":                    # E6-KD (lane L-AM3, B-2): imported in its route only (LB25)
        from .qat_kd import main as qat_kd_main
        return qat_kd_main(argv, stage_key)
    parser = add_common_args(argparse.ArgumentParser(
        description=f"{stage['name']}: {stage['objective']} (stage is PINNED, not selectable)"))
    parser.parse_args(argv)
    # Ruling Q1-R2 (L-CKPT-GUARD Q-part): no E4 or E7 of record comes from this entry point, whatever the flags.
    print(f"RESULT: REFUSED [old_runner_real_run] -- scripts/run_{stage['key']}.py makes no {stage['name']} of "
          "record (ruling Q1-R2, L-CKPT-GUARD Q-part): E4 and E7 are made by scripts/run_ptq.py, which binds the "
          f"run to its E1 or E3 parent and writes {stage['key']}_ptq_record.json. Nothing was read, loaded or "
          "written.")
    return 2
