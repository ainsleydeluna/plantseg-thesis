#!/usr/bin/env python3
"""Smoke: pixel-level calibration (src/eval/calibration.py) and the D2 VAL pass (lane L-TEACHER-DIAG).

Acceptance (d) of the lane prompt as amended by P16, P17 and P32(d): zero ECE for a perfectly calibrated
case; a hand-computed two-bin case; bin edges at exactly m/15 (m = 1..14), 1.0 in bin 15, a bin outside
1..15 stops; pixels labelled 255 are ignored; the disease subset; the float32-logit vector of P32(d)
(0.2799999955678827, a float32 softmax would give 0.47999998927116394); the hook scores the tensor the
core argmaxes (sum correct == sum TP), keeps no reference to it and refuses an in-place change.
Synthetic inputs only; no PlantSeg data, no checkpoint of record, no GPU.

    python -B scripts/smoke_teacher_calibration.py
"""
from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.eval.calibration import (ECE_EDGES, CalibratingBatchForward, CalibrationAccumulator,  # noqa: E402
                                  CalibrationStop, bin_index, ece_from_bins)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def acc_for(logits_rows, targets, num_classes):
    """One image of len(rows) pixels laid out as [1, C, 1, n]."""
    z = torch.tensor(np.asarray(logits_rows, dtype=np.float64).T[None, :, None, :])
    t = torch.tensor(np.asarray(targets, dtype=np.int64)[None, None, :])
    acc = CalibrationAccumulator(num_classes=num_classes, disease_classes=range(1, num_classes))
    acc.update(z, t)
    return acc.result()


def unit_cases() -> None:
    # d1: perfectly calibrated -- logits (0, 0): confidence exactly 0.5 at class 0, half the pixels right
    r = acc_for([[0.0, 0.0]] * 4, [0, 0, 1, 1], 2)
    check("d1 perfectly calibrated set: ECE == 0 exactly", r["ece"] == 0.0, f"ece={r['ece']!r}")
    check("d1 ece_from_bins on one bin with C/n == S/n gives 0",
          ece_from_bins([0] * 8 + [4] + [0] * 7, [0] * 8 + [2] + [0] * 7, [0.0] * 8 + [2.0] + [0.0] * 7) == 0.0)

    # d2: two bins by hand -- C = 4; conf 0.9 (bin 14) one right one wrong; conf 0.3 (bin 5) both right
    a9, a3 = math.log(27.0), math.log(9.0 / 7.0)          # e^a / (e^a + 3) = 0.9 and 0.3
    r = acc_for([[a9, 0, 0, 0], [a9, 0, 0, 0], [a3, 0, 0, 0], [a3, 0, 0, 0]], [0, 1, 0, 0], 4)
    want = 0.5 * abs(0.5 - 0.9) + 0.5 * abs(1.0 - 0.3)
    check("d2 two-bin hand case: 0.5*|0.5-0.9| + 0.5*|1-0.3| = 0.55",
          abs(r["ece"] - want) <= 1e-12 and [b["n"] for b in r["bins"]][4] == 2 and r["bins"][13]["n"] == 2,
          f"ece={r['ece']!r}")

    # d3: edges
    ok = all(int(bin_index(np.array([m / 15.0]))[0]) == m for m in range(1, 15))
    check("d3 conf == m/15 lands in bin m (m = 1..14)", ok)
    ok = all(int(bin_index(np.array([np.nextafter(m / 15.0, 2.0)]))[0]) == m + 1 for m in range(1, 15))
    check("d3 the next float above m/15 lands in bin m+1 (m = 1..14)", ok)
    check("d3 conf == 1.0 lands in bin 15", int(bin_index(np.array([1.0]))[0]) == 15)
    check("d3 edges are np.arange(16)/15.0", np.array_equal(ECE_EDGES, np.arange(16) / 15.0))
    check("d3 conf 0.0 (bin 0) stops", raises(lambda: bin_index(np.array([0.0])), CalibrationStop))
    check("d3 conf above 1.0 (bin 16) stops", raises(lambda: bin_index(np.array([np.nextafter(1.0, 2.0)])),
                                                      CalibrationStop))
    check("d3 a NaN confidence stops", raises(lambda: bin_index(np.array([np.nan])), CalibrationStop))

    # d4: 255 ignored
    rows = [[a9, 0, 0, 0], [a3, 0, 1.0, 0], [0.2, 0.1, 0, 0]]
    base = acc_for(rows, [0, 2, 1], 4)
    more = acc_for(rows + [[50.0, -3, 2, 0], [0, 0, 0, 9.0]], [0, 2, 1, 255, 255], 4)
    same = {k: v for k, v in base.items() if k != "images"} == {k: v for k, v in more.items() if k != "images"}
    check("d4 pixels labelled 255 change nothing", same and more["n"] == 3)

    # d6: disease subset -- background pixels leave the disease ECE unchanged
    r = acc_for([[a9, 0, 0, 0], [a9, 0, 0, 0], [0, 0, a3, 0], [0, 0, a3, 0]], [0, 0, 2, 1], 4)
    want_d = 1.0 * abs(0.5 - 0.3)
    check("d6 disease-pixel ECE over target in 1..C-1 only",
          r["n_disease"] == 2 and abs(r["ece_disease"] - want_d) <= 1e-12, f"ece_disease={r['ece_disease']!r}")

    # d7: P32(d) float32 vector
    z32 = torch.tensor([[1.3862942457199097, 0.0], [1.3862942457199097, 0.0],
                        [1.1526795625686646, 0.0], [1.1526795625686646, 0.0]], dtype=torch.float32)
    acc = CalibrationAccumulator(num_classes=2, disease_classes=range(1, 2))
    acc.update(z32.T[None, :, None, :].contiguous(), torch.tensor([[[0, 0, 1, 1]]]))
    r = acc.result()
    check("d7 float32 logits: ECE == 0.2799999955678827 (float64 softmax)", r["ece"] == 0.2799999955678827,
          f"ece={r['ece']!r}")
    pred = z32.argmax(1).numpy()
    c32 = torch.softmax(z32, 1).double().numpy()[np.arange(4), pred]
    b32 = bin_index(c32)
    n_m, c_m, s_m = [0] * 16, [0] * 16, [0.0] * 16
    for i in range(4):
        n_m[b32[i]] += 1
        c_m[b32[i]] += int(pred[i] == [0, 0, 1, 1][i])
        s_m[b32[i]] += c32[i]
    check("d7 a float32 softmax would give 0.47999998927116394 (the vector separates them)",
          ece_from_bins(n_m, c_m, s_m) == 0.47999998927116394, f"{ece_from_bins(n_m, c_m, s_m)!r}")

    # means vs an independent numpy reference
    rng = np.random.default_rng(5)
    z = rng.normal(0, 3, size=(1, 6, 5, 7)).astype(np.float32)
    t = rng.integers(0, 6, size=(1, 5, 7))
    t[0, 0, :3] = 255
    acc = CalibrationAccumulator(num_classes=6, disease_classes=range(1, 6))
    acc.update(torch.tensor(z), torch.tensor(t))
    r = acc.result()
    zz = z[0].reshape(6, -1).astype(np.float64)[:, t.reshape(-1) != 255]
    ref = {}
    for tag, T in (("t1", 1.0), ("t4", 4.0)):
        p = np.exp(zz / T - zz.max(0) / T)
        p = p / p.sum(0)
        ref[f"mean_max_prob_{tag}"] = p.max(0).mean()
        ref[f"mean_entropy_{tag}"] = (-(p * np.log(p)).sum(0)).mean()
    ok = all(abs(r[k] - v) <= 1e-12 for k, v in ref.items())
    check("means: max-probability and entropy at T=1 and T=4 match numpy to 1e-12", ok,
          str({k: (r[k], v) for k, v in ref.items()}))
    pmax = np.exp(zz - zz.max(0)) / np.exp(zz - zz.max(0)).sum(0)
    check("share max-prob > 0.99 at T=1 (strict) matches numpy",
          r["n_max_prob_gt_0_99_t1"] == int((pmax.max(0) > 0.99).sum()))


def hook_cases() -> None:
    from src.eval.evaluate import Condition, EvalBatch, ManifestEntry, evaluate_model

    rng = np.random.default_rng(11)
    n_img, C = 5, 116
    ids = [f"syn_{i:03d}" for i in range(n_img)]
    targets = torch.tensor(rng.integers(0, C, size=(n_img, 6, 8)))
    targets[0, 0, :4] = 255
    logits_all = torch.tensor(rng.normal(0, 2, size=(n_img, C, 6, 8)).astype(np.float32))

    class Fwd:
        def __call__(self, model, images):
            return images.clone()                        # the "model" output is a fresh tensor

    acc = CalibrationAccumulator()
    hook = CalibratingBatchForward(Fwd(), acc)
    batches = [EvalBatch(images=logits_all[i:i + 1], targets=targets[i:i + 1], image_ids=[ids[i]],
                         clean_image_ids=[ids[i]], manifest_indices=[i]) for i in range(n_img)]
    manifest = [ManifestEntry(i, ids[i], ids[i]) for i in range(n_img)]
    res = evaluate_model(None, batches, expected_manifest=manifest, condition=Condition(), num_classes=C,
                         background_index=0, ignore_index=255, batch_forward=hook)
    r = acc.result()
    check("d5 sum correct == sum TP of the core (same tensor, same argmax)",
          r["correct"] == int(res.dataset_tp.sum()), f"{r['correct']} vs {int(res.dataset_tp.sum())}")
    check("d5 n == sum GT of the core (valid pixels)", r["n"] == int(res.dataset_gt.sum()))
    gc.collect()
    check("d5 the hook keeps no tensor: its weakref is dead after the core's del",
          hook.last_logits_ref is not None and hook.last_logits_ref() is None)
    check("d5 one hook call per batch", hook.calls == n_img)

    class Mutating(CalibrationAccumulator):
        def update(self, logits, target):
            logits.add_(0.0)                             # an in-place op bumps _version
            super().update(logits, target)

    bad = CalibratingBatchForward(Fwd(), Mutating())
    check("d5 an in-place change of the core's tensor stops",
          raises(lambda: bad(None, batches[0]), CalibrationStop))

    class Warny(CalibrationAccumulator):
        def update(self, logits, target):
            import warnings
            warnings.warn("synthetic accumulation warning")
            super().update(logits, target)

    import warnings
    w_hook = CalibratingBatchForward(Fwd(), Warny())
    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        w_hook(None, batches[0])
    check("P17 accumulation warnings are kept apart from the evaluator's capture",
          not outer and len(w_hook.accumulation_warnings) == 1, f"outer={len(outer)}")


def main() -> int:
    for fn in (unit_cases, hook_cases):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- a crash is a failed case, never a pass
            check(f"{fn.__name__} raised", False, f"{type(e).__name__}: {e}")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    good = sum(ok for _, ok, _ in RESULTS)
    print(f"RESULT: {'PASS' if good == len(RESULTS) else 'FAIL'} ({good}/{len(RESULTS)})")
    return 0 if good == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
