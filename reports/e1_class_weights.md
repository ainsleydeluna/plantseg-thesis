# E1 Cross-Entropy Class Weights (B18a)

**RESULT: ✅ PASS** — length-116 CE class weights computed from the **raw training masks**.
**This is preparation for the real E1 training run — NOT training.**

_Generated: 2026-06-28 by `scripts`-style scratch compute (Anaconda Python, numpy + Pillow), CPU,
read-only over `annotations/train/*.png`. Artifact: `reports/e1_class_weights.json`._

---

## What this is
The real E1 supervised loss is **class-weighted CE + Dice**. B17 used `weight=None` (acceptable for a
wiring smoke only). This step computes the **thesis-aligned CE class weights** so the future E1 loop can
load them. The weighting method matches `src/training/losses.compute_class_weights` and the contract (B5)
**exactly**: `w_c = sqrt(total_nonignore_pixels / (num_classes · pixels_in_class))`, **median-normalized**,
absent classes → 1.0, **no cap**.

## Basis decision (documented)
Computed over the **raw annotation PNGs at native resolution**, *not* the resized/padded dataloader masks.
Rationale: the raw masks are the **canonical source label distribution**; resizing-to-512 + symmetric
padding can slightly alter per-class pixel frequencies and inject preprocessing effects (the 255 pad), so
the raw basis is the standard, reproducible choice. (255 is excluded either way.)

## Results (your 12 report items)
1. **Train mask count used:** **5,367** (all decoded; 0 unreadable; modes `['L']`, ndim `[2]`).
2. **Label range found:** non-ignore **0–115** (contiguous).
3. **255 excluded:** ✅ yes — and **255 is absent from the raw masks** (`contains_255_in_raw = False`); no out-of-range values (`>115, ≠255`) found.
4. **Pixel-count summary:** total non-ignore = **4,582,656,621**.
   - **background (class 0):** 3,703,512,045 px (**80.82%**) → weight **0.036954** (smallest, as expected).
   - **diseases (1–115):** 879,144,576 px (**19.18%**).
5. **Zero-count train classes:** **NONE** — all 116 classes (bg + 115 diseases) appear in train. (Consistent with B16: test lacks class 41, val lacks class 68, but **train has every class** — so no absent-class fallback was needed.)
6. **Weighting formula:** √ inverse-frequency, **median-normalized**, absent → 1.0, **no cap** — matches `compute_class_weights` / contract B5.
7. **Cap used:** **none** (per contract/config "no fixed cap").
8. **Final weight stats (all 116):** min **0.036954** (class 0), median **1.000000**, mean **1.315423**, max **9.591535** (class 42).
9. **All 116 finite:** ✅ yes.
10. **JSON artifact path:** **`reports/e1_class_weights.json`** (length-116 `weights` + `pixel_counts` + provenance/formula/stats).
11. **How the E1 loop should load/use it:**
    ```python
    import json, torch
    w = torch.tensor(json.load(open("reports/e1_class_weights.json"))["weights"], dtype=torch.float32)
    # CombinedCEDiceLoss(weight=w, ignore_index=255)  /  WeightedCrossEntropyLoss(weight=w, ignore_index=255)
    ```
    The array is **index-aligned to class id 0–115** (index 0 = background).
12. **Preparation, not training:** no model was trained, no loop implemented; only label frequencies were
    counted and weights computed.

### Weight extremes (insight)
| Lowest weight (most frequent) | px | | Highest weight (rarest) | px |
|---|---|---|---|---|
| class **0** (background) 0.0370 | 3,703,512,045 | | class **42** 9.5915 | 54,975 |
| class 90 0.2796 | 64,715,686 | | class 43 4.3533 | 266,869 |
| class 47 0.3833 | 34,431,625 | | class 57 4.3050 | 272,891 |
| class 6 0.3871 | 33,745,956 | | class 17 2.7414 | 672,980 |
| class 87 0.3878 | 33,626,118 | | class 71 2.6379 | 726,790 |

## Important notes (as required)
- **Background class 0 is weighted as a normal CE class** (weight 0.036954) — it is **not** excluded and
  **not** treated specially (`reduce_zero_label = False`).
- **Disease-only *metrics* may exclude background (class 0) later** (open question #2), but the **CE class
  *weights* include class 0** — these are two different things (loss weighting vs metric reporting).
- **255 is excluded** from the weight computation (and is absent from raw masks).
- **No cap** is applied.
- The **future E1 training loop should load `reports/e1_class_weights.json`** and pass the length-116
  tensor as the CE `weight` (see item 11).

## Provenance / reproducibility
Dataset root `C:\Users\admin\plantseg_data\plantseg`; split `train`; basis raw `.png`; `num_classes=116`;
`ignore_index=255`; method matches `src/training/losses.compute_class_weights`. Re-running the count over
the same raw masks reproduces these weights deterministically (pure pixel counts, no randomness).

_Preparation artifact only. `docs/reference/reference.pdf` untouched (deferred). No code/config/dataset
changed; nothing committed._
