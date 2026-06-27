# Conflicts Log — older sources vs. Chapter 3 (current authority)

**Rule:** `ch3.pdf` is the single source of truth for method/protocol; `Wei (2026)` is authoritative
for dataset facts; `context.md` is the reconciliation source-of-truth. Where an older/secondary file
(`ch1.pdf`, `ch2.pdf`, `reference.pdf`) disagrees with `ch3.pdf`, **ch3 wins ("current-wins")**. The
`num_classes` item (#1) was a genuine cross-authority conflict held OPEN pending evidence; it is now
**RESOLVED empirically (2026-06-27)** — see below — toward **116**, with a residual background/disease-only
`NEED_TO_CONFIRM`.

Source tags as in [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md).

---

## 1. ✅ RESOLVED (2026-06-27) — `num_classes` = 116 (was: Wei 0–114 vs ch3 116)

| Source | Claim | Empirical verdict |
|---|---|---|
| `[Wei]` | Indices **0–114** represent the 115 plant-disease combinations. | ✔ true **of the COCO `category_id`s** (verified `category_id` ∈ 0–114). |
| `[ch3]` | **`num_classes = 116`** (background + 115 diseases); self-labelled provisional. | ✔ true **of the mask pixel values** (verified 0–115). |
| `[ch1]` | "115 different plant diseases on 34 species." | ✔ 115 diseases (mask values 1–115). |
| `[ctx]` | "`num_classes` is NOT settled. NEVER hardcode it." | Now settled by measurement. |

**Resolution (no silent winner — decided by full-dataset `np.unique()`):** the two claims were a **frame
mismatch**, not a contradiction. Rasterized masks use **0–115**: **0 = background**, **1–115 = diseases**
(`mask value = category_id + 1`, **99.85%** per-image consistent over 7,774 masks; 12 off-by-one exceptions
logged). All-class **`num_classes = 116`**; Wei's 0–114 are the underlying `category_id`s. Evidence:
[reports/dataset_report.md](../reports/dataset_report.md).

**Residual (still open):** which index the *disease-only* metric excludes — empirically **0 (background)** —
is not explicitly named in the dataset files, so it stays `NEED_TO_CONFIRM`. → [open_questions.md](open_questions.md) #2.

---

## 2. RPD framed as "primary" robustness metric (ch1) vs. mIoU-C primary (ch3)

| Source | Claim |
|---|---|
| `[ch1]` | "RPD is used as the **primary** metric of robustness performance degradation under corruption." |
| `[ch3]` (wins) | **mIoU-C** is the primary robustness summary; the single inferential robustness test is **E1 vs E6 on per-image mIoU-C**; **RPD is descriptive only** and is never a test target; rCD also descriptive only. |

**Current-wins:** ch3 — mIoU-C primary, RPD/rCD descriptive.

---

## 3. LR-ASPP low-level projection — intermediate 256-ch layer on *both* branches (ch2) vs. high branch only (ch3)

| Source | Claim |
|---|---|
| `[ch2]` | Head task-adapted "by expanding **both** its low-level and high-level classification projections with an intermediate **256-channel** layer before the final output." |
| `[ch3]` (wins) | Only the **high-level** C5 branch carries the 1×1 Conv-BN-ReLU **256-ch** stage; the **low-level OS8 skip** is projected **directly** by a 1×1 conv to `num_classes` (no intermediate 256-ch layer specified on the low branch). |

**Current-wins:** ch3 head topology in [IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §(e).
*(Note: ch1/ch2 also use the label "Lite-ASPP-style" interchangeably with "LR-ASPP"; ch3 uses "LR-ASPP". Terminological, not structural.)*

---

## 4. "ICCC / imagecorruptions package" framing (ch1) vs. vendored functions (ch3)

| Source | Claim |
|---|---|
| `[ch1]` | Robustness "achieved through the Image Corruptions Common Corruption (ICCC)" using the **`imagecorruptions` package**. |
| `[ch3]` (wins) | Corruptions are **vendored functions** from the Hendrycks 2019 reference implementation (deprecated NumPy aliases patched), **not** the installed package; `[ctx]` pins this as vendored **imagecorruptions 1.1.2** with `np.float_`→`np.float64`. |

**Current-wins:** ch3 (vendored, patched). No "ICCC" acronym in ch3.

---

## Consistency checks — values that AGREE across files (recorded so they are not re-litigated)

- **5 corruptions** (motion blur, Gaussian noise, JPEG, brightness, fog) at **severity 1–3**: agree across ch1/ch2/ch3.
- **Global ReLU6 substitution rejected** / native Hard-Swish retained: agree ch2 §B + ch3 §D.
- **Teacher = SegNeXt-B/MSCAN-B**, chosen over SegNeXt-L (44.52%) and ConvNeXt-L (46.24%) for
  capacity/channel-compatibility: agree ch2 + ch3 + Wei numbers.
- **7,774 images / 115 / 34 hosts**, 70/10/20 split, JPEG+PNG, DOI 10.5281/zenodo.17719108,
  CC BY-NC 4.0: agree Wei + ch1 + ch3.
- **42.05% mIoU** teacher benchmark reference: agree ch1 + ch2 + ch3 + Wei Table 3.
- **Seed 42**, 512×512, batch 16, ignore_index 255: agree across files.

> No other numeric divergences (learning rates, iteration counts, temperatures, α/β weights, batch size,
> calibration size, augmentation probabilities) were found between the secondary files and ch3 — the
> secondary chapters defer the concrete config numbers to Chapter 3 rather than restating different values.
