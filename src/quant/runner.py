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

UNRESOLVED EXPERIMENT VALUES ARE HARD-GATED, NEVER INVENTED. `configs/quant.py` locks the QAT
optimizer family (SGD, momentum 0.9, lr 3e-4, cosine, ~15 epochs, no weight EMA, CE+Dice supervised,
best-val-mIoU selection) but leaves the global-norm `max_norm` unspecified, the BN-stat freeze as a
**range** ("65-70%"), the observer freeze as a **relation** ("shortly after BN freeze"), and neither
the QAT batch size nor its weight decay appears in B4 at all. A real E5/E6 launch must supply each
of those explicitly; the runner refuses and lists every missing one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
from .prepare import (BN_FREEZE_PCT_RANGE, bn_freeze_iteration, calibrate, convert_model,  # noqa: E402
                      disable_observers, freeze_bn_stats, qat_grad_clip_gate_error,
                      quantization_coverage, try_converted_forward)
from .qconfig import (QUANT_BACKEND, QuantBackendUnavailable, describe_qconfig, ptq_qconfig,  # noqa: E402
                      qat_qconfig, select_qnnpack_backend)
from .stages import (load_source_for_stage, prepare_for_stage, require_shared_calibration_index,  # noqa: E402
                     resolve_quant_stage)

NUM_CLASSES = 116
QAT = QUANT["qat"]
PTQ = QUANT["ptq"]
DRY_EPOCHS = QAT["epochs_approx"]      # synthetic/structural use ONLY — never a real-run default


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


def unresolved_qat_values(args) -> list[str]:
    """Every locked-but-unvalued QAT quantity a real E5/E6 launch must supply. Never guessed."""
    missing: list[str] = []
    err = qat_grad_clip_gate_error(args.grad_clip_norm)
    if err:
        missing.append(f"--grad-clip-norm ({err.splitlines()[0]})")
    if args.batch_size is None or args.batch_size <= 0:
        missing.append("--batch-size (B4's QAT table fixes no batch size; E1's 16 is the "
                       "E1/E2/E3 recipe, not the QAT one)")
    # Weight decay: B4's QAT table locks optimizer/momentum/lr/schedule and has NO weight-decay row
    # (the 0.01 and 1e-4 in the contract belong to B1 teacher AdamW and B2 student SGD). Genuinely
    # unspecified -> an explicit real-run decision. Any finite value >= 0 is acceptable, including 0.
    if (args.weight_decay is None or not math.isfinite(args.weight_decay)
            or args.weight_decay < 0):
        missing.append("--weight-decay, any finite value >= 0 (B4's QAT table locks SGD, momentum, "
                       "lr and cosine but has no weight-decay row at all)")
    # Epoch budget: the contract says "~15 epochs" and the config field is literally named
    # `epochs_approx`. Approximate guidance is NOT an exact locked value, so a real run must state
    # the budget it is actually committing to (supplying 15 makes 15 the recorded decision).
    if args.epochs is None or args.epochs <= 0:
        missing.append(f"--epochs <positive integer> (contract says '~{QAT['epochs_approx']}' and "
                       "the config field is `epochs_approx`; an approximation is not a locked value)")
    # Early stopping: the metric IS pinned (`early_stop: 'val_miou'`) but no patience is governed
    # anywhere, so the mechanism is implemented and the value is required at launch.
    if args.early_stop_patience is None or args.early_stop_patience <= 0:
        missing.append("--early-stop-patience <positive integer> (contract pins early stopping on "
                       f"'{QAT['early_stop']}' but fixes no patience)")
    lo, hi = BN_FREEZE_PCT_RANGE
    if args.bn_freeze_pct is None or not (lo <= args.bn_freeze_pct <= hi):
        missing.append(f"--bn-freeze-pct inside the locked range [{lo}, {hi}] "
                       f"(contract fixes the range '{QAT['bn_freeze_pct']}%', not a value)")
    if args.observer_freeze_pct is None:
        missing.append(f"--observer-freeze-pct (contract says only '{QAT['observer_freeze']}')")
    elif args.bn_freeze_pct is not None and args.observer_freeze_pct < args.bn_freeze_pct:
        missing.append("--observer-freeze-pct must be >= --bn-freeze-pct "
                       f"('{QAT['observer_freeze']}')")
    elif not (0.0 < args.observer_freeze_pct <= 1.0):
        missing.append("--observer-freeze-pct must lie in (0, 1]")
    return missing


# ------------------------------------------------------------------ early stopping
class EarlyStopper:
    """Early stopping on all-class validation mIoU (contract B4: "early-stop on val mIoU").

    Improvement uses the SAME strict comparison as best-checkpoint selection, `miou > best`. No
    minimum delta is governed anywhere in the contract, so none is invented: a plateau counts as a
    non-improvement. The counter advances only at validation points and resets on any improvement.
    Kept separate from the training loop so its semantics can be proven against a synthetic score
    sequence, and so E5 and E6 provably share one implementation.
    """

    def __init__(self, patience: int):
        if not isinstance(patience, int) or patience <= 0:
            raise QuantRunError("early_stop_patience_invalid",
                                f"early-stopping patience must be a positive integer, got {patience!r}")
        self.patience = patience
        self.best = float("-inf")
        self.best_step: int | None = None
        self.num_bad = 0
        self.triggered = False

    def update(self, miou: float, step: int) -> bool:
        """Record a validation result. Returns True when it improved on the best so far."""
        if miou > self.best:
            self.best, self.best_step, self.num_bad = miou, step, 0
            return True
        self.num_bad += 1
        if self.num_bad >= self.patience:
            self.triggered = True
        return False

    @property
    def should_stop(self) -> bool:
        return self.triggered


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


# ------------------------------------------------------------------ PTQ / QAT execution
def run_ptq(stage: dict, args, model, source_meta: dict, out_dir: Path, backend: str) -> dict:
    index = check_calibration(stage, args.calibration_index, args.calibration_sha256)
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is None or not data_root.is_dir():
        raise QuantRunError("data_root_missing",
                            "--data-root must resolve to the PlantSeg root so the frozen "
                            "calibration identifiers can be read (TRAIN only)")
    prepared = prepare_for_stage(stage["key"], model, select_backend=False)
    loader = build_calibration_loader(data_root, index["selected_ids"], args.batch_size or 1)
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


def run_qat(stage: dict, args, model, source_meta: dict, out_dir: Path, backend: str) -> dict:
    from src.data import build_dataloader
    from src.training.losses import CombinedCEDiceLoss
    from src.training.train_e1 import cycle, load_ce_weights, validate

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    prepared = prepare_for_stage(stage["key"], model, select_backend=False).to(device)
    prepared.train()

    train_loader = build_dataloader("train", args.batch_size, num_workers=args.num_workers)
    val_loader = build_dataloader("val", args.batch_size, num_workers=args.num_workers)
    # The TEST split is never constructed here.
    iters_per_epoch = len(train_loader)
    total_iters = iters_per_epoch * args.epochs
    bn_freeze_at = bn_freeze_iteration(total_iters, args.bn_freeze_pct)
    obs_freeze_at = max(bn_freeze_at, int(round(total_iters * args.observer_freeze_pct)))

    weights = load_ce_weights().to(device)
    criterion = CombinedCEDiceLoss(weight=weights).to(device)     # supervised-only; no distillation
    optimizer = torch.optim.SGD(prepared.parameters(), lr=QAT["learning_rate"],
                                momentum=QAT["momentum"], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    stopper = EarlyStopper(args.early_stop_patience)
    best_state, it, completed_epochs = None, 0, 0
    for img, mask in cycle(train_loader):
        it += 1
        if it > total_iters:
            break
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(prepared(img), mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prepared.parameters(), args.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        if it == bn_freeze_at:
            freeze_bn_stats(prepared)
        if it == obs_freeze_at:
            disable_observers(prepared)
        if it % iters_per_epoch == 0 or it == total_iters:
            completed_epochs = math.ceil(it / iters_per_epoch)
            all_miou, _, _, _ = validate(prepared, val_loader, device, NUM_CLASSES, None)
            if stopper.update(all_miou, it):      # selection = best all-class val mIoU
                best_state = copy.deepcopy(prepared.state_dict())
            prepared.train()
            if stopper.should_stop:               # patience exhausted -> stop early, keep the best
                break

    if best_state is None:
        raise QuantRunError("qat_no_checkpoint", "QAT produced no validated checkpoint")
    best_miou, best_iter = stopper.best, stopper.best_step
    qat_path = out_dir / f"{stage['key']}_qat_state.pt"
    torch.save({"stage": stage["name"], "quantization": "qat-train-state", "iter": best_iter,
                "best_val_miou_all_class": best_miou, "model_state_dict": best_state}, qat_path)

    prepared.load_state_dict(best_state)
    converted = convert_model(prepared.cpu())
    coverage = quantization_coverage(converted)
    fwd = try_converted_forward(converted, torch.randn(1, 3, 512, 512))
    if not fwd["ok"]:
        raise QuantRunError("converted_forward_failed",
                            f"converted {stage['name']} model does not run: {fwd['error']}")
    path = out_dir / f"{stage['key']}_int8_student.pt"
    torch.save({"stage": stage["name"], "quantization": "qat", "num_classes": NUM_CLASSES,
                "model": converted.state_dict()}, path)
    return build_run_provenance(
        stage=stage, source_meta=source_meta, method="qat",
        qconfig_summary=_qconfig_summary("qat"), calibration=None,
        training={"optimizer": f"SGD lr={QAT['learning_rate']} momentum={QAT['momentum']} "
                               f"weight_decay={args.weight_decay}",
                  "weight_decay": args.weight_decay,
                  "schedule": QAT["lr_schedule"],
                  "requested_max_epochs": args.epochs,
                  "completed_epochs": completed_epochs, "completed_iters": it,
                  "early_stop_metric": QAT["early_stop"],
                  "early_stop_patience": args.early_stop_patience,
                  "early_stop_triggered": stopper.triggered,
                  "early_stop_min_delta": None,   # none governed; strict improvement comparison
                  "best_epoch": (None if best_iter is None
                                 else math.ceil(best_iter / iters_per_epoch)),
                  "batch_size": args.batch_size, "iters_per_epoch": iters_per_epoch,
                  "total_iters": total_iters, "grad_clip_norm": args.grad_clip_norm,
                  "bn_freeze_pct": args.bn_freeze_pct, "bn_freeze_iter": bn_freeze_at,
                  "observer_freeze_pct": args.observer_freeze_pct,
                  "observer_freeze_iter": obs_freeze_at,
                  "weight_ema": QAT["weight_ema"], "distillation": QAT["distillation_during_qat"],
                  "objective": "CE+Dice (supervised only)",
                  "selection": "best all-class validation mIoU", "best_iter": best_iter,
                  "best_val_miou_all_class": best_miou,
                  "qat_state_artifact": str(qat_path), "test_split_used": False},
        converted_path=path, coverage=coverage, forward=fwd, backend=backend)


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
    # QAT — every one of these is an unresolved experiment value, never defaulted for a REAL run.
    # `DRY_EPOCHS` exists only so synthetic/structural exercises of the loop have a budget.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--bn-freeze-pct", type=float, default=None)
    p.add_argument("--observer-freeze-pct", type=float, default=None)
    return p


def main(argv, stage_key: str) -> int:
    import argparse
    stage = resolve_quant_stage(stage_key)
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
    if stage["method"] == "qat":
        missing = unresolved_qat_values(args)
        if missing:
            print(f"REFUSING to start the real {stage['name']} QAT run — these values are locked "
                  "by the contract only as families/ranges and must be supplied explicitly:",
                  file=sys.stderr)
            for m in missing:
                print(f"  - {m}", file=sys.stderr)
            return 2
    try:
        out_dir = check_output_dir(args.out_dir, create=True)
        model, _, source_meta = check_source(stage, args.source_ckpt)
        backend = check_backend()
        prov = (run_ptq if stage["method"] == "ptq" else run_qat)(
            stage, args, model, source_meta, out_dir, backend)
    except QuantRunError as e:
        print(f"{stage['name']} PREFLIGHT/RUN FAILED [{e.code}]: {e}", file=sys.stderr)
        return 2

    (out_dir / f"{stage['key']}_run_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps(prov, indent=2))
    print(f"\n{stage['name']} COMPLETE — artifact and provenance written to {out_dir}")
    return 0
