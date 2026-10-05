"""Shared execution core for the four quantization stages. Validates hard, then runs.

    E4 : E1 FP32 checkpoint -> PTQ        E6 : E3 projection-free student -> QAT
    E5 : E1 FP32 checkpoint -> QAT        E7 : E3 projection-free student -> PTQ

`scripts/run_e4.py` … `run_e7.py` are thin wrappers that PIN their stage: the stage is not a CLI
flag, so `run_e4.py` can never be turned into E7.

SAFETY MODEL — a real run requires `--real-run` AND `--confirm-real-run`, and every cheap
precondition is proven before any expensive work: source provenance, structural validity, 116
classes, an out-of-repo output directory (via E1's own guard), the QNNPACK backend, and — for
PTQ — the exact shared calibration artifact. Random initialization is impossible: every stage loads
a validated source checkpoint. Nothing here downloads, substitutes a model, or falls back silently.

E5/E6 QAT is src/quant/qat.py's (AM-4/AM-4a, lanes L-AM4 + L-AM1q): `main` hands those two stages to
`src.quant.qat.main` before any PTQ argument is parsed; scripts/run_e5.py and run_e6.py reach it through
`main`.
"""

from __future__ import annotations

import hashlib
import json
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
from .prepare import (CALIBRATION_BATCH_SIZE, calibrate, convert_model,  # noqa: E402
                      quantization_coverage, try_converted_forward)
from .qconfig import (QUANT_BACKEND, QuantBackendUnavailable, describe_qconfig, ptq_qconfig,  # noqa: E402
                      qat_qconfig, select_qnnpack_backend)
from .stages import (load_source_for_stage, prepare_for_stage, require_shared_calibration_index,  # noqa: E402
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


def build_calibration_loader(root: Path, ids: list[str], batch_size: int, num_workers: int = 0):
    ds = CalibrationDataset(root, ids)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, drop_last=False)


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
    if stage["source_stage"] == "E3":
        prov["cwd_projection_loaded"] = False
    return prov


def _qconfig_summary(method: str) -> dict:
    d = describe_qconfig(ptq_qconfig() if method == "ptq" else qat_qconfig())
    flat = {k: {kk: str(vv) for kk, vv in v.items()} for k, v in d.items()}
    fingerprint = hashlib.sha256(
        json.dumps(flat, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {"activation": flat["activation"], "weight": flat["weight"], "fingerprint": fingerprint}


# ------------------------------------------------------------------ PTQ execution
def run_ptq(stage: dict, args, model, source_meta: dict, out_dir: Path, backend: str) -> dict:
    index = check_calibration(stage, args.calibration_index, args.calibration_sha256)
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is None or not data_root.is_dir():
        raise QuantRunError("data_root_missing",
                            "--data-root must resolve to the PlantSeg root so the frozen "
                            "calibration identifiers can be read (TRAIN only)")
    prepared = prepare_for_stage(stage["key"], model, select_backend=False)
    # AM-10: one image per mini-batch; main() has already refused any other --batch-size.
    loader = build_calibration_loader(data_root, index["selected_ids"], CALIBRATION_BATCH_SIZE)
    n = calibrate(prepared, (img for img, _ in loader))
    converted = convert_model(prepared)
    coverage = quantization_coverage(converted)
    fwd = try_converted_forward(converted, torch.randn(1, 3, 512, 512))
    if not fwd["ok"]:
        raise QuantRunError("converted_forward_failed",
                            f"converted {stage['name']} model does not run: {fwd['error']}")
    path = out_dir / f"{stage['key']}_int8_student.pt"
    torch.save({"stage": stage["name"], "quantization": "ptq", "num_classes": NUM_CLASSES,
                "model": converted.state_dict()}, path)
    return build_run_provenance(
        stage=stage, source_meta=source_meta, method="ptq",
        qconfig_summary=_qconfig_summary("ptq"),
        calibration={"index_path": str(Path(args.calibration_index).resolve()),
                     "checksum_sha256": index["checksum_sha256"], "count": index["count"],
                     "seed": index["seed"], "split": index["split"],
                     "shared_by": index["shared_by"], "batches_seen": n,
                     "preprocessing": PTQ["calibration_preprocessing"], "augmentation": False},
        training=None, converted_path=path, coverage=coverage, forward=fwd, backend=backend)


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
    parser = add_common_args(argparse.ArgumentParser(
        description=f"{stage['name']}: {stage['objective']} (stage is PINNED, not selectable)"))
    args = parser.parse_args(argv)

    if not args.real_run:
        print(f"REFUSING: {stage['name']} requires --real-run --confirm-real-run. Nothing was "
              "read, loaded or written.", file=sys.stderr)
        return 2
    if not args.confirm_real_run:
        print(f"REFUSING: --real-run requires --confirm-real-run for {stage['name']}.",
              file=sys.stderr)
        return 2
    if stage["method"] == "ptq" and args.batch_size not in (None, CALIBRATION_BATCH_SIZE):
        print(f"REFUSING: {stage['name']} calibrates one image per mini-batch (AM-10); "
              f"--batch-size {args.batch_size} is refused. Nothing was read, loaded or written.",
              file=sys.stderr)
        return 2
    try:
        out_dir = check_output_dir(args.out_dir, create=True)
        model, _, source_meta = check_source(stage, args.source_ckpt)
        backend = check_backend()
        prov = run_ptq(stage, args, model, source_meta, out_dir, backend)
    except QuantRunError as e:
        print(f"{stage['name']} PREFLIGHT/RUN FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2

    (out_dir / f"{stage['key']}_run_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps(prov, indent=2))
    print(f"\n{stage['name']} COMPLETE — artifact and provenance written to {out_dir}")
    return 0
