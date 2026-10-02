"""Pixel-level calibration of the teacher on the evaluation canvas (AM-18 item 1(b); lane L-TEACHER-DIAG).

`src/eval/metrics.py` is frozen and untouched: this module only reads logits and targets.

ECE, exactly as AM-18 item 1(b) writes it, per image from the tensor the evaluation core argmaxes:

    pred        = logits.argmax(1)                          (the scored prediction, first max on ties)
    valid       = target != 255
    confidence  = the float64 softmax probability (T = 1) at pred, from those same logits
    bins        = searchsorted(np.arange(16) / 15.0, confidence, side="left")    bin m holds
                  (m-1)/15 < confidence <= m/15; any bin outside 1..15 is a STOP
    per image and bin: n_m and C_m (correct) as ints, S_m (the confidence sum) by math.fsum;
    across images: ints add, confidence sums by math.fsum of the per-image sums
    ECE         = math.fsum((n_m / n) * |C_m / n_m - S_m / n_m|) over the non-empty bins

The same for the disease subset (target in 1..115). Beside it, pooled over valid pixels the same way:
the mean max-probability and the mean entropy (nats, from log_softmax) at T = 1 and T = 4, and the share
of pixels whose max-probability at T = 1 is strictly above 0.99.

`CalibratingBatchForward` is the `evaluate_model(batch_forward=...)` hook of the D2 VAL pass: it calls the
evaluator's own forward, accumulates from the returned tensor and hands the same tensor back to the core.
It keeps no reference to the tensor, asserts the tensor was not modified in place, and records any
warning its own accumulation raises separately, so the run's `eval_runtime.eval_warnings` is what
scripts/evaluate_model.py would write.

Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import math
import warnings
import weakref

import numpy as np
import torch

NUM_BINS = 15
ECE_EDGES = np.arange(NUM_BINS + 1) / float(NUM_BINS)     # float64 m/15, m = 0..15
TEMPERATURES = (1.0, 4.0)
HIGH_CONFIDENCE = 0.99
IGNORE_INDEX = 255


class CalibrationStop(RuntimeError):
    """A calibration invariant failed (a bin outside 1..15, a tensor modified in place). Always fatal."""


def bin_index(conf: np.ndarray) -> np.ndarray:
    """Bin m of each confidence, (m-1)/15 < c <= m/15. STOP unless every bin is in 1..15."""
    b = np.searchsorted(ECE_EDGES, conf, side="left")
    if b.size and (int(b.min()) < 1 or int(b.max()) > NUM_BINS):
        raise CalibrationStop(f"a confidence fell outside (0, 1]: bins {sorted(set(b.tolist()) - set(range(1, 16)))}")
    return b


def ece_from_bins(n_m, c_m, s_m) -> float | None:
    """AM-18's formula as written: fsum over non-empty bins of (n_m/n) * |C_m/n_m - S_m/n_m|."""
    n = sum(int(n_m[m]) for m in range(1, NUM_BINS + 1))
    if n == 0:
        return None
    return math.fsum((int(n_m[m]) / n) * abs(int(c_m[m]) / int(n_m[m]) - s_m[m] / int(n_m[m]))
                     for m in range(1, NUM_BINS + 1) if int(n_m[m]) > 0)


class _BinTotals:
    """n and C per bin as ints; S per bin as the per-image fsums, reduced by fsum at the end."""

    def __init__(self):
        self.n = [0] * (NUM_BINS + 1)
        self.c = [0] * (NUM_BINS + 1)
        self.s_parts: list[list[float]] = [[] for _ in range(NUM_BINS + 1)]

    def add(self, bins: np.ndarray, conf: np.ndarray, correct: np.ndarray) -> None:
        if bins.size == 0:
            return
        n_img = np.bincount(bins, minlength=NUM_BINS + 1)
        c_img = np.bincount(bins[correct], minlength=NUM_BINS + 1)
        order = np.argsort(bins, kind="stable")
        sorted_conf = conf[order]
        ends = np.cumsum(n_img)
        start = 0
        for m in range(NUM_BINS + 1):
            stop = int(ends[m])
            if m >= 1 and stop > start:
                self.n[m] += int(n_img[m])
                self.c[m] += int(c_img[m])
                self.s_parts[m].append(math.fsum(sorted_conf[start:stop].tolist()))
            start = stop

    def result(self) -> dict:
        s = [math.fsum(p) for p in self.s_parts]
        return {"ece": ece_from_bins(self.n, self.c, s),
                "n": sum(self.n[1:]), "correct": sum(self.c[1:]),
                "bins": [{"m": m, "lo": float(ECE_EDGES[m - 1]), "hi": float(ECE_EDGES[m]),
                          "n": self.n[m], "correct": self.c[m], "conf_sum": s[m]}
                         for m in range(1, NUM_BINS + 1)]}


class CalibrationAccumulator:
    """Pooled pixel-level calibration over the valid pixels of every image passed to `update`."""

    def __init__(self, num_classes: int = 116, ignore_index: int = IGNORE_INDEX,
                 disease_classes=range(1, 116)):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        dc = sorted(int(c) for c in disease_classes)
        self.disease_lo, self.disease_hi = (dc[0], dc[-1]) if dc else (1, 0)
        if dc != list(range(self.disease_lo, self.disease_hi + 1)):
            raise ValueError("disease_classes must be one contiguous range")
        self.all = _BinTotals()
        self.disease = _BinTotals()
        self.parts = {k: [] for k in ("max_prob_t1", "entropy_t1", "max_prob_t4", "entropy_t4")}
        self.n_gt_high_t1 = 0
        self.images = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.dim() != 4 or logits.shape[1] != self.num_classes:
            raise CalibrationStop(f"logits must be [B, {self.num_classes}, H, W], got {tuple(logits.shape)}")
        if tuple(target.shape) != (logits.shape[0], *logits.shape[-2:]):
            raise CalibrationStop(f"target {tuple(target.shape)} does not match logits {tuple(logits.shape)}")
        with torch.no_grad():
            pred_all = logits.argmax(1)                         # the prediction the core scores
            for b in range(logits.shape[0]):
                t = target[b].reshape(-1).cpu()
                idx = torch.nonzero(t != self.ignore_index, as_tuple=False).reshape(-1)
                self.images += 1
                if idx.numel() == 0:
                    continue
                tv = t[idx]
                pv = pred_all[b].reshape(-1).cpu()[idx]
                zv = logits[b].reshape(self.num_classes, -1).cpu()[:, idx].to(torch.float64)
                lsm1 = torch.log_softmax(zv, dim=0)
                conf = lsm1.gather(0, pv.reshape(1, -1)).reshape(-1).exp().numpy()
                lsm4 = torch.log_softmax(zv / TEMPERATURES[1], dim=0)
                max1 = lsm1.max(dim=0).values.exp().numpy()
                ent1 = (-(lsm1.exp() * lsm1).sum(dim=0)).numpy()
                max4 = lsm4.max(dim=0).values.exp().numpy()
                ent4 = (-(lsm4.exp() * lsm4).sum(dim=0)).numpy()
                del lsm1, lsm4, zv
                correct = (pv == tv).numpy()
                bins = bin_index(conf)
                self.all.add(bins, conf, correct)
                tvn = tv.numpy()
                dmask = (tvn >= self.disease_lo) & (tvn <= self.disease_hi)
                self.disease.add(bins[dmask], conf[dmask], correct[dmask])
                for key, arr in (("max_prob_t1", max1), ("entropy_t1", ent1),
                                 ("max_prob_t4", max4), ("entropy_t4", ent4)):
                    self.parts[key].append(math.fsum(arr.tolist()))
                self.n_gt_high_t1 += int(np.count_nonzero(max1 > HIGH_CONFIDENCE))

    def result(self) -> dict:
        a, d = self.all.result(), self.disease.result()
        n = a["n"]

        def mean(key):
            return (math.fsum(self.parts[key]) / n) if n else None

        return {
            "n": n, "n_disease": d["n"], "images": self.images,
            "correct": a["correct"], "correct_disease": d["correct"],
            "ece": a["ece"], "ece_disease": d["ece"],
            "mean_max_prob_t1": mean("max_prob_t1"), "mean_entropy_t1": mean("entropy_t1"),
            "share_max_prob_gt_0_99_t1": (self.n_gt_high_t1 / n) if n else None,
            "n_max_prob_gt_0_99_t1": self.n_gt_high_t1,
            "mean_max_prob_t4": mean("max_prob_t4"), "mean_entropy_t4": mean("entropy_t4"),
            "bins": a["bins"], "bins_disease": d["bins"],
            "definition": ("ECE = fsum over non-empty bins of (n_m/n)*|C_m/n_m - S_m/n_m|; 15 bins "
                           "(m-1)/15 < conf <= m/15 (searchsorted side='left'); conf = float64 softmax "
                           "(T=1) at logits.argmax(1); valid = target != 255; disease = target in 1..115"),
        }


class CalibratingBatchForward:
    """`evaluate_model(batch_forward=...)`: the evaluator's forward, accumulated, returned unchanged."""

    def __init__(self, fwd, accumulator: CalibrationAccumulator):
        self.fwd = fwd
        self.acc = accumulator
        self.calls = 0
        self.accumulation_warnings: list[str] = []
        self.last_logits_ref = None

    def __call__(self, model, batch):
        logits = self.fwd(model, batch.images)
        version = logits._version
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.acc.update(logits, batch.targets)
        self.accumulation_warnings.extend(f"{w.category.__name__}: {str(w.message).splitlines()[0][:300]}"
                                          for w in caught)
        if logits._version != version:
            raise CalibrationStop("the evaluator's logits were modified in place during accumulation")
        self.last_logits_ref = weakref.ref(logits)
        self.calls += 1
        return logits


__all__ = ["NUM_BINS", "ECE_EDGES", "TEMPERATURES", "HIGH_CONFIDENCE", "CalibrationStop", "bin_index",
           "ece_from_bins", "CalibrationAccumulator", "CalibratingBatchForward"]
