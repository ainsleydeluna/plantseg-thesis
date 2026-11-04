"""In-process VAL scoring through the evaluator of record's library (lane L-AM24: B1's passes, KL-3's parity).

scripts/evaluate_model.py loads only the registered precisions, so a site-matched fake-quant graph (B1) cannot go
through it. This module drives the same library calls the evaluator's run() makes, unchanged: the canvas protocol
adapter over the VAL split (src.eval.adapters build_protocol_adapter / build_expected_manifest_for /
build_eval_loader, EVAL_NUM_WORKERS), src.eval.evaluate_model with the clean condition, 116 classes, background 0 and
ignore 255, and the evaluator's device-forward hook (src.eval.eval_runtime ModelDeviceForward). The value is the
union-present all-class mIoU of `dataset_level`, the value of every VAL record of the study. No artifact is written
by the evaluator's writer here; the caller writes exploratory records. The split is VAL, always: TEST is refused by
name before anything is built. A sample cap is for smokes and timing only and is recorded.

`score_batches` takes already-built EvalBatch objects (the smokes' synthetic batches) and the same manifest checks.
"""
from __future__ import annotations

import time

from . import guard as G

NUM_CLASSES, BACKGROUND, IGNORE = 116, 0, 255
PROTOCOL = "canvas"


def _summary(result, seconds: float, extra: dict | None = None) -> dict:
    lvl = dict(result.dataset_level)
    return {"all_class_miou": lvl.get("all_class_miou"), "all_class_miou_n_eligible": lvl.get("all_class_miou_n_eligible"),
            "disease_only_miou_PROVISIONAL": lvl.get("disease_only_miou"), "rows": len(result.rows),
            "forward_batches": result.forward_batches, "seconds": seconds, **(extra or {})}


def score_batches(model, batches, manifest, *, forward=None) -> tuple[dict, object]:
    """Score `model` over identity-carrying batches with the evaluator core. Returns (summary, EvalResult)."""
    from src.eval import Condition, evaluate_model
    t0 = time.time()
    result = evaluate_model(model, batches, expected_manifest=manifest, condition=Condition("clean", None, None),
                            num_classes=NUM_CLASSES, background_index=BACKGROUND, ignore_index=IGNORE,
                            forward=forward)
    return _summary(result, time.time() - t0), result


def score_val(model, *, device: str = "cpu", split: str = "val", max_samples: int | None = None,
              batch_size: int = 1, forward=None) -> tuple[dict, object]:
    """Score `model` on the VAL split under the canvas protocol. The caller has run the guard (src/explore/guard.py) on the data root.
    `forward(model, images)` replaces the evaluator's device-forward hook (KL-3's agreement pass)."""
    G.refuse_split(split, allowed=G.VAL_ONLY)
    from src.eval.adapters import (build_eval_loader, build_expected_manifest_for, build_protocol_adapter,
                                   deterministic_subset, protocol_forward_kwargs)
    from src.eval.eval_runtime import EVAL_NUM_WORKERS, ModelDeviceForward, resolve_model_device
    n_total = G.VAL_IMAGES
    idx = deterministic_subset(n_total, max_samples) if max_samples is not None else list(range(n_total))
    manifest = build_expected_manifest_for("val", idx)
    adapter = build_protocol_adapter(PROTOCOL, "val", idx)
    model_device = resolve_model_device(device, cpu_only=str(device) == "cpu")
    fwd = ModelDeviceForward(model_device) if forward is None else forward
    loader = build_eval_loader(adapter, batch_size, num_workers=EVAL_NUM_WORKERS)
    from src.eval import Condition, evaluate_model
    t0 = time.time()
    result = evaluate_model(model, loader, expected_manifest=manifest, condition=Condition("clean", None, None),
                            num_classes=NUM_CLASSES, background_index=BACKGROUND, ignore_index=IGNORE,
                            **protocol_forward_kwargs(PROTOCOL, fwd))
    return _summary(result, time.time() - t0, {"split": "val", "protocol": PROTOCOL, "samples": len(idx),
                                               "max_samples": max_samples, "device": str(device),
                                               "batch_size": batch_size}), result


def synthetic_batches(n: int, *, size=(64, 64), seed: int = 0, num_classes: int = NUM_CLASSES):
    """Smokes: `n` one-image EvalBatch objects with random images and two-class masks, and their manifest."""
    import torch
    from src.eval.evaluate import EvalBatch, ManifestEntry
    g = torch.Generator().manual_seed(seed)
    batches, manifest = [], []
    for i in range(n):
        img = torch.randn(1, 3, *size, generator=g)
        cls = int(torch.randint(1, num_classes, (1,), generator=g))
        mask = torch.zeros(1, *size, dtype=torch.long)
        mask[:, size[0] // 4: 3 * size[0] // 4, size[1] // 4: 3 * size[1] // 4] = cls
        sid = f"synthetic_{i:04d}"
        batches.append(EvalBatch(images=img, targets=mask, image_ids=[sid], clean_image_ids=[sid],
                                 manifest_indices=[i]))
        manifest.append(ManifestEntry(manifest_index=i, image_id=sid, clean_image_id=sid))
    return batches, manifest
