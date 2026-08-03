#!/usr/bin/env python3
"""Metric-contract tests against the REAL production path (docs/EVALUATION_CONTRACT.md).

Lifecycle: authored in A1a as the RED phase (16 PASS / 5 EXPECTED_RED, proving the gap), then
re-run UNCHANGED in A1b after the D5 production correction, where all 21 must classify PASS.
A1b flipped exactly five declarations RED->PASS (A1, A2, A3, A4, D1), strengthened A3/A4 from
existence-only to behavioural assertions, and moved E1's literals from the pre-fix GT-present
values (0.75 / 0.50) to the contract's union-present values (0.50 / 0.25). No assertion was
weakened, no tolerance loosened, no case removed, and C4 is untouched.

ANTI-SELF-REFERENCE RULE (binding):
  This file contains NO union-present reducer, NO mAcc implementation, NO Dice
  implementation, and no copy of any production metric formula under another name.
  Every expected value is a LITERAL hand-computed constant; the arithmetic for each is
  written out in reports/a1a_metric_contract_red.md. Fixtures are built with ordinary
  tensor construction and plain counting only (counting pixels is accounting, not a
  metric). Where a required production function does not exist, its absence is probed
  through the real module and reported -- never substituted locally.

Run:  set PYTHONIOENCODING=utf-8 && python -B scripts/smoke_metrics_contract.py
Exit: 0 iff every case matched its declared state (EXPECTED_RED counts as a match).
      Non-zero on any UNEXPECTED_GREEN / UNEXPECTED_FAILURE.

CPU only. No training, no checkpoints, no dataset access, no installs, no downloads.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

# --- REAL production imports (the whole point of A1a) ---------------------------------
import src.eval.metrics as prod_metrics                                    # noqa: E402
from src.eval.metrics import (confusion_matrix, disease_only_miou,         # noqa: E402
                              all_class_miou, miou_from_confusion, per_image_miou)
from src.training.train_e1 import validate as prod_validate                # noqa: E402

C = 4                 # tiny synthetic class space: 0 = background, 1..3 = "diseases"
DISEASE_IDX = torch.tensor([1, 2, 3])   # disease-only index set for the tiny space
TOL = 1e-9            # for values that are exactly representable (1.0, 0.5, 0.75)

# Production reduces in FLOAT32 (`_miou_from_cm` does `torch.diag(cm).float()`), so any
# expected value that is not exactly representable in binary32 carries ~1e-8 rounding.
# float32 eps = 1.19e-7, so a 1e-9 tolerance is physically impossible to satisfy for
# values like 7/12 or 2/3. F32_TOL is the correct precision floor -- it still separates
# every candidate convention by ~5 orders of magnitude (e.g. |7/12 - 2/3| = 8.3e-2).
# See case C4, which pins this precision regime down explicitly.
F32_TOL = 1e-6


class ProductionAbsent(Exception):
    """Raised when a contract-required production attribute does not exist."""


# --------------------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------------------
CASES: list[tuple] = []


def case(cid: str, declared: str, reason_kind: str, title: str):
    """declared: 'PASS' | 'RED'.  reason_kind: 'assertion' | 'absent' | '' (for PASS)."""
    def deco(fn):
        CASES.append((cid, declared, reason_kind, title, fn))
        return fn
    return deco


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------------------
# fixtures -- plain construction only
# --------------------------------------------------------------------------------------
def flat(target: list[int], pred: list[int]):
    return torch.tensor(pred), torch.tensor(target)


# --------------------------------------------------------------------------------------
# A. Dataset-level contract divergence
# --------------------------------------------------------------------------------------
@case("A1", "PASS", "",
      "hand-computed GT-present vs union-present dataset mIoU divergence")
def a1():
    # target [0,0,1,1] / pred [0,0,1,2]
    #   c0 GT=2 PR=2 TP=2 -> IoU 1.0     c1 GT=2 PR=1 TP=1 -> IoU 0.5
    #   c2 GT=0 PR=1 TP=0 -> IoU 0.0 (FP-only)   c3 absent from both -> undefined
    # GT-present  {0,1}   = (1.0+0.5)/2       = 0.75   <- current production
    # union-present {0,1,2} = (1.0+0.5+0.0)/3 = 0.50   <- contract §3.1
    pred, target = flat([0, 0, 1, 1], [0, 0, 1, 2])
    cm = confusion_matrix(pred, target, C)
    observed = miou_from_confusion(cm)
    expected_union_present = 0.5          # literal, hand-derived (report §A1)
    current_gt_present = 0.75             # literal, hand-derived (report §A1)
    assert approx(observed, expected_union_present, F32_TOL), (
        f"contract union-present mIoU={expected_union_present} but production returned "
        f"{observed} (== hand-computed GT-present value {current_gt_present})")
    return {"asserted": expected_union_present, "observed": observed}


@case("A2", "PASS", "",
      "FP-only class scores IoU 0 and is INCLUDED in dataset mIoU")
def a2():
    # target [0,0,0,1,1] / pred [0,0,2,1,1]
    #   c0 GT=3 PR=2 TP=2 -> IoU 2/3     c1 GT=2 PR=2 TP=2 -> IoU 1.0
    #   c2 GT=0 PR=1 TP=0 -> IoU 0.0 (FP-only, drawn from a background pixel)
    # GT-present   {0,1}   = (2/3 + 1)/2     = 5/6
    # union-present{0,1,2} = (2/3 + 1 + 0)/3 = 5/9
    pred, target = flat([0, 0, 0, 1, 1], [0, 0, 2, 1, 1])
    cm = confusion_matrix(pred, target, C)
    observed = miou_from_confusion(cm)
    expected_union_present = 5 / 9        # literal fraction, hand-derived (report §A2)
    current_gt_present = 5 / 6            # literal fraction, hand-derived (report §A2)
    assert approx(observed, expected_union_present, F32_TOL), (
        f"contract requires the FP-only class c2 to contribute IoU 0 -> mIoU=5/9="
        f"{expected_union_present:.7f}; production excluded it and returned {observed} "
        f"(== hand-computed GT-present 5/6={current_gt_present:.7f})")
    return {"asserted": expected_union_present, "observed": observed}


@case("A3", "PASS", "", "dataset-level mAcc exists AND excludes the FP-only class (GT-present)")
def a3():
    candidates = ["macc_from_confusion", "macc", "mean_accuracy",
                  "accuracy_from_confusion", "class_accuracy"]
    found = [n for n in candidates if hasattr(prod_metrics, n)]
    if not found:
        raise ProductionAbsent(
            "src.eval.metrics defines no mAcc reducer; probed "
            f"{candidates!r}; module exports "
            f"{[n for n in dir(prod_metrics) if not n.startswith('__')]!r}")
    # A1b strengthening (approved): existence alone cannot show the GT-present rule.
    # Same CM as A1: TP=[2,1,0,0] GT=[2,2,0,0] PR=[2,1,1,0].
    #   Acc_0 = 2/2 = 1.0   Acc_1 = 1/2 = 0.5   c2 has GT=0 -> EXCLUDED   c3 GT=0 -> excluded
    # GT-present {0,1} -> (1.0 + 0.5)/2 = 0.75
    # If the FP-only class c2 were wrongly included: (1.0 + 0.5 + 0.0)/3 = 0.5
    pred, target = flat([0, 0, 1, 1], [0, 0, 1, 2])
    cm = confusion_matrix(pred, target, C)
    observed = getattr(prod_metrics, found[0])(cm)
    expected = 0.75                       # literal, hand-derived (report section A3)
    assert approx(observed, expected, F32_TOL), (
        f"mAcc must be GT-present -> 0.75 (FP-only class c2 excluded); got {observed} "
        "(0.5 would mean c2 was wrongly counted)")
    return {"asserted": expected, "observed": f"{found[0]}(cm)={observed}"}


@case("A4", "PASS", "", "dataset-level macro Dice exists AND uses union-present eligibility")
def a4():
    candidates = ["dice_from_confusion", "macro_dice", "mdice",
                  "dice_from_cm", "macro_dice_from_confusion"]
    found = [n for n in candidates if hasattr(prod_metrics, n)]
    if not found:
        raise ProductionAbsent(
            "src.eval.metrics defines no macro-Dice reducer; probed "
            f"{candidates!r} (note: CombinedCEDiceLoss in src.training.losses is a "
            "LOSS, not a metric reducer, and is out of scope)")
    # A1b strengthening (approved). Same CM as A1:
    #   Dice_0 = 2*2/(2+2) = 1.0   Dice_1 = 2*1/(2+1) = 2/3   Dice_2 = 2*0/(0+1) = 0.0
    #   c3: GT+PR = 0 -> excluded
    # union-present {0,1,2} -> (1.0 + 2/3 + 0.0)/3 = 5/9
    # If it wrongly used the GT-present rule {0,1}: (1.0 + 2/3)/2 = 5/6
    pred, target = flat([0, 0, 1, 1], [0, 0, 1, 2])
    cm = confusion_matrix(pred, target, C)
    observed = getattr(prod_metrics, found[0])(cm)
    expected = 5 / 9                      # literal fraction, hand-derived (report section A4)
    assert approx(observed, expected, F32_TOL), (
        f"macro Dice must be union-present -> 5/9={expected:.7f} (c2 counted as 0.0); "
        f"got {observed} (5/6=0.8333333 would mean the GT-present rule was used)")
    return {"asserted": expected, "observed": f"{found[0]}(cm)={observed}"}


@case("A5", "PASS", "", "class absent from BOTH GT and prediction is omitted")
def a5():
    # target [0,0,1,1] / pred [0,0,1,1] -> perfect; c2,c3 absent from both.
    # Both conventions agree here: mean(1.0, 1.0) = 1.0.
    # If absent-from-both were wrongly counted as 0, the answer would be 0.5.
    pred, target = flat([0, 0, 1, 1], [0, 0, 1, 1])
    cm = confusion_matrix(pred, target, C)
    observed = miou_from_confusion(cm)
    expected = 1.0                        # literal, hand-derived (report §A5)
    assert approx(observed, expected, TOL), f"expected {expected}, got {observed}"
    return {"asserted": expected, "observed": observed}


# --------------------------------------------------------------------------------------
# B. Ignore, class range, background
# --------------------------------------------------------------------------------------
@case("B1", "PASS", "", "ignore-255 invariance: predictions under ignored GT are dropped")
def b1():
    target = torch.tensor([0, 0, 255, 255])
    pred_a = torch.tensor([0, 0, 1, 1])
    pred_b = torch.tensor([0, 0, 2, 3])   # differs ONLY beneath ignored target pixels
    cm_a = confusion_matrix(pred_a, target, C)
    cm_b = confusion_matrix(pred_b, target, C)
    same_cm = torch.equal(cm_a, cm_b)
    miou_a, miou_b = miou_from_confusion(cm_a), miou_from_confusion(cm_b)
    assert same_cm, f"confusion matrices differ:\n{cm_a.tolist()}\n{cm_b.tolist()}"
    assert approx(miou_a, miou_b, TOL), f"mIoU differs: {miou_a} vs {miou_b}"
    assert int(cm_a.sum()) == 2, f"expected 2 valid pixels, got {int(cm_a.sum())}"
    return {"asserted": "identical CM and mIoU; 2 valid px",
            "observed": f"CM equal={same_cm}, mIoU={miou_a}, valid_px={int(cm_a.sum())}"}


@case("B2", "PASS", "", "background IS included in all-class metrics")
def b2():
    # target [0,0,1,1] / pred [0,1,1,1]
    #   c0 GT=2 PR=1 TP=1 -> IoU 1/2      c1 GT=2 PR=3 TP=2 -> IoU 2/3
    # all-class GT-present = (1/2 + 2/3)/2 = 7/12
    pred, target = flat([0, 0, 1, 1], [0, 1, 1, 1])
    observed = all_class_miou(pred, target, C)
    expected = 7 / 12                     # literal fraction, hand-derived (report §B2)
    assert approx(observed, expected, F32_TOL), f"expected 7/12={expected}, got {observed}"
    return {"asserted": expected, "observed": observed}


@case("B3", "PASS", "", "disease-only excludes EXACTLY class 0 (no off-by-one)")
def b3():
    # Same fixture as B2. disease-only GT-present = {c1} only -> 2/3.
    # If c0 were wrongly included -> 7/12. If c1 were wrongly excluded -> NaN.
    pred, target = flat([0, 0, 1, 1], [0, 1, 1, 1])
    observed = disease_only_miou(pred, target, C, background_index=0)
    all_class = all_class_miou(pred, target, C)
    expected = 2 / 3                      # literal fraction, hand-derived (report §B3)
    assert observed == observed, "disease-only returned NaN -> class 1 wrongly excluded"
    assert approx(observed, expected, F32_TOL), f"expected 2/3={expected}, got {observed}"
    assert not approx(observed, all_class, F32_TOL), (
        f"disease-only ({observed}) equals all-class ({all_class}) -> class 0 not excluded")
    return {"asserted": expected, "observed": observed}


@case("B4", "PASS", "", "per-image disease-only == IoU of the single GT disease class")
def b4():
    # 1 image, 2x2. target [[0,0],[1,1]] / pred [[0,1],[1,1]]  (same counts as B2)
    # single GT disease c1: GT=2 PR=3 TP=2 -> IoU 2/3
    target = torch.tensor([[[0, 0], [1, 1]]])
    pred = torch.tensor([[[0, 1], [1, 1]]])
    observed = per_image_miou(pred, target, C, class_indices=DISEASE_IDX)
    expected = 2 / 3                      # literal fraction, hand-derived (report §B4)
    assert len(observed) == 1, f"expected 1 row, got {len(observed)}"
    assert approx(observed[0], expected, F32_TOL), f"expected 2/3={expected}, got {observed[0]}"
    return {"asserted": [expected], "observed": observed}


@case("B5", "PASS", "", "per-image handles MULTIPLE GT disease classes (no hidden assumption)")
def b5():
    # 1 image, 2x2. target [[0,1],[2,2]] / pred [[0,1],[2,1]]  -> TWO GT diseases
    #   c0 GT=1 PR=1 TP=1 -> IoU 1.0
    #   c1 GT=1 PR=2 TP=1 -> IoU 1/2
    #   c2 GT=2 PR=1 TP=1 -> IoU 1/2
    # disease-only GT-present {1,2} = (1/2 + 1/2)/2 = 0.5
    target = torch.tensor([[[0, 1], [2, 2]]])
    pred = torch.tensor([[[0, 1], [2, 1]]])
    observed = per_image_miou(pred, target, C, class_indices=DISEASE_IDX)
    expected = 0.5                        # literal, hand-derived (report §B5)
    assert len(observed) == 1, f"expected 1 row, got {len(observed)}"
    assert approx(observed[0], expected, 1e-9), (
        f"expected 0.5 (mean over TWO GT diseases), got {observed[0]} -- a value of "
        "0.5 must come from averaging both, not from a one-disease shortcut")
    return {"asserted": [expected], "observed": observed}


@case("B6", "PASS", "", "per-image all-class == hand-computed mean over GT-present classes")
def b6():
    # Same fixture as B4: c0 IoU 1/2, c1 IoU 2/3 -> (1/2 + 2/3)/2 = 7/12
    target = torch.tensor([[[0, 0], [1, 1]]])
    pred = torch.tensor([[[0, 1], [1, 1]]])
    observed = per_image_miou(pred, target, C, class_indices=None)
    expected = 7 / 12                     # literal fraction, hand-derived (report §B6)
    assert approx(observed[0], expected, F32_TOL), f"expected 7/12={expected}, got {observed[0]}"
    return {"asserted": [expected], "observed": observed}


# --------------------------------------------------------------------------------------
# C. Accumulation and sufficient statistics
# --------------------------------------------------------------------------------------
def _c_fixture():
    """4 synthetic 2x2 images with an ignore region; plain construction."""
    target = torch.tensor([
        [[0, 0], [1, 1]],
        [[0, 1], [2, 2]],
        [[0, 0], [0, 255]],
        [[1, 1], [255, 255]],
    ])
    pred = torch.tensor([
        [[0, 1], [1, 1]],
        [[0, 1], [2, 1]],
        [[0, 2], [0, 3]],
        [[1, 2], [0, 0]],
    ])
    return pred, target


@case("C1", "PASS", "", "single-batch vs multi-batch confusion-matrix equivalence")
def c1():
    pred, target = _c_fixture()
    whole = confusion_matrix(pred, target, C)
    halves = (confusion_matrix(pred[:2], target[:2], C)
              + confusion_matrix(pred[2:], target[2:], C))
    singles = torch.zeros(C, C, dtype=torch.long)
    for i in range(pred.shape[0]):
        singles += confusion_matrix(pred[i], target[i], C)
    ok = torch.equal(whole, halves) and torch.equal(whole, singles)
    assert ok, (f"CMs differ:\n1x4={whole.tolist()}\n2x2={halves.tolist()}\n"
                f"4x1={singles.tolist()}")
    return {"asserted": "1x4 == 2x2 == 4x1", "observed": f"equal={ok}, CM={whole.tolist()}"}


@case("C2", "PASS", "", "per-image (TP, GT, PR) counts sum to the dataset CM totals")
def c2():
    pred, target = _c_fixture()
    # Plain counting on the fixture -- accounting, NOT a metric formula.
    tp = [0] * C
    gt = [0] * C
    pr = [0] * C
    for i in range(pred.shape[0]):
        t, p = target[i].reshape(-1), pred[i].reshape(-1)
        valid = t != 255
        t, p = t[valid], p[valid]
        for c in range(C):
            tp[c] += int(((t == c) & (p == c)).sum())
            gt[c] += int((t == c).sum())
            pr[c] += int((p == c).sum())
    cm = confusion_matrix(pred, target, C)
    cm_tp = torch.diag(cm).tolist()
    cm_gt = cm.sum(1).tolist()
    cm_pr = cm.sum(0).tolist()
    assert tp == cm_tp, f"TP mismatch: fixture {tp} vs CM diag {cm_tp}"
    assert gt == cm_gt, f"GT mismatch: fixture {gt} vs CM row sums {cm_gt}"
    assert pr == cm_pr, f"PR mismatch: fixture {pr} vs CM col sums {cm_pr}"
    return {"asserted": f"tp={tp} gt={gt} pr={pr}",
            "observed": f"tp={cm_tp} gt={cm_gt} pr={cm_pr}"}


@case("C3", "PASS", "", "confusion matrix is int64 with headroom past ~409M pixels")
def c3():
    pred, target = _c_fixture()
    cm = confusion_matrix(pred, target, C)
    full_scale = 1561 * 512 * 512          # 409,206,784 -- the real test-split pixel count
    headroom = torch.tensor([full_scale + 1], dtype=torch.int64)
    int64_max = torch.iinfo(torch.int64).max
    assert cm.dtype == torch.int64, f"CM dtype is {cm.dtype}, contract requires int64"
    assert int(headroom.item()) == full_scale + 1, "int64 round-trip failed at full scale"
    assert int64_max > full_scale * 1000, "insufficient int64 headroom"
    return {"asserted": "int64 and headroom > 409,206,784",
            "observed": f"dtype={cm.dtype}, full_scale={full_scale:,}, int64_max={int64_max:,}"}


@case("C4", "PASS", "", "reduction precision is float32 (discovered during A1a run 1)")
def c4():
    # Counts are exact int64, but `_miou_from_cm` casts to float32 before dividing, so
    # every non-binary-representable metric carries ~1e-8 rounding. Recorded because
    # (a) it invalidated a 1e-9 test tolerance in run 1, and (b) the artifact contract
    # §5.2 requires full-precision float serialisation -- writing 17 significant digits
    # of a float32 value stores ~10 digits of noise. Flagged for A1b/A2, not fixed here.
    pred, target = flat([0, 0, 1, 1], [0, 1, 1, 1])
    cm = confusion_matrix(pred, target, C)
    _, iou_vec = prod_metrics._miou_from_cm(cm)
    observed = miou_from_confusion(cm)
    exact_f64 = 7 / 12                      # literal, hand-derived (report §B2)
    drift = abs(observed - exact_f64)
    f32_eps = torch.finfo(torch.float32).eps
    assert cm.dtype == torch.int64, f"counts should be exact int64, got {cm.dtype}"
    assert iou_vec.dtype == torch.float32, (
        f"expected the documented float32 reduction, got {iou_vec.dtype}")
    assert drift < f32_eps, f"drift {drift} exceeds float32 eps {f32_eps}"
    assert drift > 0.0, "expected non-zero float32 rounding for 7/12"
    return {"asserted": "int64 counts, float32 reduction, 0 < drift < f32 eps",
            "observed": (f"iou dtype={iou_vec.dtype}, drift={drift:.3e}, "
                         f"f32_eps={f32_eps:.3e}")}


# --------------------------------------------------------------------------------------
# D. Integrity failures (expectations declared from the read-only preflight probe)
# --------------------------------------------------------------------------------------
@case("D1", "PASS", "", "prediction label >= num_classes must fail loudly")
def d1():
    # A1a recorded that confusion_matrix(pred=[4], target=[0], num_classes=4) SILENTLY
    # returned a CM with cm[1][0] == 1 -- k = target*C + pred = 4 aliased onto row 1,
    # fabricating a ground-truth-class-1 pixel. A1b added the label guard, so this must
    # now raise. Contract: EVALUATION_CONTRACT section 5.3 `invalid_pred_labels` == 0.
    pred, target = flat([0], [4])          # pred == num_classes == 4 -> invalid
    raised = None
    cm = None
    try:
        cm = confusion_matrix(pred, target, C)
    except Exception as e:                 # noqa: BLE001 -- classified, not swallowed
        raised = e
    assert raised is not None, (
        "contract requires an explicit failure for an out-of-range prediction label "
        f"(pred={C} with num_classes={C}); production returned a silently corrupted "
        f"confusion matrix instead: {cm.tolist()} -- note the phantom entry at cm[1][0], "
        "which fabricates a GT-class-1 pixel that does not exist")
    return {"asserted": "explicit exception", "observed": f"{type(raised).__name__}"}


@case("D2", "PASS", "", "negative prediction label fails loudly")
def d2():
    pred, target = flat([0], [-1])
    try:
        cm = confusion_matrix(pred, target, C)
    except Exception as e:                 # noqa: BLE001
        return {"asserted": "explicit exception",
                "observed": f"{type(e).__name__}: {str(e)[:70]}"}
    raise AssertionError(f"expected an exception, got CM {cm.tolist()}")


@case("D3", "PASS", "", "prediction/target shape mismatch fails loudly")
def d3():
    pred = torch.zeros(8, dtype=torch.long)
    target = torch.zeros(4, dtype=torch.long)
    try:
        cm = confusion_matrix(pred, target, C)
    except Exception as e:                 # noqa: BLE001
        return {"asserted": "explicit exception",
                "observed": f"{type(e).__name__}: {str(e)[:70]}"}
    raise AssertionError(f"expected an exception, got CM {cm.tolist()}")


@case("D4", "PASS", "", "zero valid pixels -> explicit non-finite undefined signal")
def d4():
    target = torch.full((4,), 255)
    pred = torch.zeros(4, dtype=torch.long)
    cm = confusion_matrix(pred, target, C)
    observed = miou_from_confusion(cm)
    assert int(cm.sum()) == 0, f"expected an all-zero CM, got sum {int(cm.sum())}"
    assert observed != observed, (                     # NaN is the only value != itself
        f"contract requires an explicit undefined signal; got the finite value {observed}, "
        "which is indistinguishable from a genuine score of 0.0")
    return {"asserted": "cm.sum()==0 and mIoU is NaN", "observed": f"cm.sum()=0, mIoU={observed}"}


@case("D5", "PASS", "", "class-count mismatch fails loudly")
def d5():
    # (a) a ground-truth label >= num_classes
    try:
        confusion_matrix(torch.tensor([0]), torch.tensor([7]), C)
        raise AssertionError("expected an exception for GT label 7 with num_classes=4")
    except AssertionError:
        raise
    except Exception as e_a:               # noqa: BLE001
        first = f"{type(e_a).__name__}"
    # (b) class_indices referencing a class outside the CM
    try:
        miou_from_confusion(torch.zeros(C, C, dtype=torch.long), torch.tensor([9]))
        raise AssertionError("expected an exception for class_indices=[9] on a 4x4 CM")
    except AssertionError:
        raise
    except Exception as e_b:               # noqa: BLE001
        second = f"{type(e_b).__name__}"
    return {"asserted": "both variants raise", "observed": f"GT-label={first}, indices={second}"}


# --------------------------------------------------------------------------------------
# E. E1 validation parity
# --------------------------------------------------------------------------------------
class _FixedPredModel(torch.nn.Module):
    """Deterministic dummy segmentation model: emits one-hot logits for a fixed map.

    Not a metric implementation -- it only produces model OUTPUT so the real production
    `validate` can be exercised end to end without a checkpoint or any dataset.
    """

    def __init__(self, pred_map: torch.Tensor, num_classes: int):
        super().__init__()
        self.register_buffer("pred_map", pred_map)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        logits = torch.zeros(b, self.num_classes, h, w)
        return logits.scatter(1, self.pred_map.unsqueeze(1), 1.0)


@case("E1", "PASS", "", "train_e1.validate returns the union-present contract values")
def e1():
    # 1 image, 2x2.  GT [[0,0],[1,1]]  pred [[0,0],[1,2]]   (same CM as A1)
    #   c0 GT=2 PR=2 TP=2 -> 1.0 | c1 GT=2 PR=1 TP=1 -> 0.5 | c2 GT=0 PR=1 -> 0.0
    # all-class     GT-present {0,1}   = 0.75   union-present {0,1,2}  = 0.50
    # disease-only  GT-present {1}     = 0.50   union-present {1,2}    = 0.25
    mask = torch.tensor([[[0, 0], [1, 1]]])
    pred_map = torch.tensor([[[0, 0], [1, 2]]])
    img = torch.zeros(1, 3, 2, 2)
    model = _FixedPredModel(pred_map, C)
    model.train()                                   # validate() must restore this
    loader = [(img, mask)]                          # synthetic in-memory loader

    all_miou, disease_miou, cm, n_batches = prod_validate(
        model, loader, torch.device("cpu"), C, None)

    ref_cm = confusion_matrix(pred_map, mask, C)
    exp_all_union = 0.50                    # literal, hand-derived (report section E1)
    exp_dis_union = 0.25                    # literal, hand-derived (report section E1)
    pre_a1b_all_gt_present = 0.75           # what A1a recorded before the D5 fix
    pre_a1b_dis_gt_present = 0.50           # what A1a recorded before the D5 fix

    assert n_batches == 1, f"expected 1 batch, got {n_batches}"
    assert torch.equal(cm, ref_cm), (
        f"validate CM {cm.tolist()} != direct CM {ref_cm.tolist()}")
    assert approx(all_miou, exp_all_union, F32_TOL), (
        f"expected union-present all-class {exp_all_union}, got {all_miou} "
        f"({pre_a1b_all_gt_present} would mean validate still uses the GT-present rule)")
    assert approx(disease_miou, exp_dis_union, F32_TOL), (
        f"expected union-present disease-only {exp_dis_union}, got {disease_miou} "
        f"({pre_a1b_dis_gt_present} would mean validate still uses the GT-present rule)")
    assert model.training, "validate() did not restore train() mode"
    return {
        "asserted": f"all={exp_all_union}, disease={exp_dis_union} (union-present)",
        "observed": (f"all={all_miou}, disease={disease_miou}; "
                     f"was {pre_a1b_all_gt_present}/{pre_a1b_dis_gt_present} in A1a "
                     "-- validate inherited the fix with NO edit to train_e1.py"),
    }


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------
def main() -> int:
    print("=" * 100)
    print("Metric-contract tests against the REAL production path (A1a RED -> A1b GREEN)")
    print(f"torch {torch.__version__} | python {sys.version.split()[0]} | CPU only")
    print(f"production modules: {prod_metrics.__file__}")
    print("=" * 100)

    results = []
    for cid, declared, reason_kind, title, fn in CASES:
        try:
            out = fn()
            cls = "PASS" if declared == "PASS" else "UNEXPECTED_GREEN"
            asserted, observed = out.get("asserted"), out.get("observed")
            detail = ""
        except ProductionAbsent as e:
            cls = "EXPECTED_RED" if (declared == "RED" and reason_kind == "absent") \
                else "UNEXPECTED_FAILURE"
            asserted, observed, detail = "production attribute exists", "ABSENT", str(e)
        except AssertionError as e:
            cls = "EXPECTED_RED" if (declared == "RED" and reason_kind == "assertion") \
                else "UNEXPECTED_FAILURE"
            asserted, observed, detail = "see reason", "assertion failed", str(e)
        except Exception as e:                       # noqa: BLE001 -- always recorded
            cls = "UNEXPECTED_FAILURE"
            asserted, observed = "see reason", f"{type(e).__name__}"
            detail = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

        results.append((cid, declared, cls, asserted, observed, detail, title))
        flag = {"PASS": "  ok", "EXPECTED_RED": " RED", "UNEXPECTED_GREEN": "!GRN",
                "UNEXPECTED_FAILURE": "!ERR"}[cls]
        print(f"\n[{flag}] {cid:3s} declared={declared:4s} -> {cls}")
        print(f"       {title}")
        print(f"       asserted : {asserted}")
        print(f"       observed : {observed}")
        if detail:
            print(f"       reason   : {detail.splitlines()[0][:180]}")
            for extra in detail.splitlines()[1:4]:
                print(f"                  {extra[:180]}")

    counts = {k: sum(1 for r in results if r[2] == k)
              for k in ("PASS", "EXPECTED_RED", "UNEXPECTED_GREEN", "UNEXPECTED_FAILURE")}
    print("\n" + "=" * 100)
    print(f"SUMMARY  PASS={counts['PASS']}  EXPECTED_RED={counts['EXPECTED_RED']}  "
          f"UNEXPECTED_GREEN={counts['UNEXPECTED_GREEN']}  "
          f"UNEXPECTED_FAILURE={counts['UNEXPECTED_FAILURE']}  (total {len(results)})")
    clean = counts["UNEXPECTED_GREEN"] == 0 and counts["UNEXPECTED_FAILURE"] == 0
    print(f"RESULT: {'OK -- every case matched its declared state' if clean else 'BLOCKED -- UNEXPECTED RESULT'}")
    if counts["EXPECTED_RED"]:
        print("Expected-RED cases are a deliverable, NOT production passes (A1a phase).")
    else:
        print("All cases declared PASS and passed -- the contract is satisfied by production (A1b phase).")
    print("=" * 100)
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
