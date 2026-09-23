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

## 5–13. Conflicts registered by B59 (2026-09-21) — recorded, not silently reconciled

Source: [reports/b59_pre_runpod_reconciliation.md](../reports/b59_pre_runpod_reconciliation.md). Class
labels: MANUSCRIPT ERROR · IMPLEMENTATION DEVIATION · DOCUMENTATION GAP · SOURCE-DERIVED IMPLEMENTATION
DETAIL · METHODOLOGY DECISION OPEN · VERIFIED MEASURED BEHAVIOUR.

| # | Conflict | Sources | Class | Standing |
|---|---|---|---|---|
| 5 | **Teacher protocol attribution.** ch3 p.102 calls AdamW 6e-5 / 40k / head lr_mult 10 "the Wei et al. (2026) PlantSeg repository configuration that produced the published 42.05%", and frames the teacher as *protocol matching*. | Wei paper: SGD lr 0.001, momentum 0.9, wd 0.0005, CE, batch 16. Public PlantSeg repo (`1a3dd4d9`): no MSCAN-B PlantSeg config; its T/L SegNeXt configs use an ImageNet-pretrained MSCAN with AdamW 6e-5. | MANUSCRIPT ERROR | Repo: **thesis-derived SegNeXt-B teacher configuration**; Wei 42.05/56.30/~28M contextual only. ADE20K init kept (ch3's explicit choice). ~~Acceptance band = METHODOLOGY DECISION OPEN~~ **[UPDATED 2026-09-23 — B64 C5]** Resolved: acceptance = R3 on VAL (strictly > 0.36314016580581665); the 42.05% comparison is descriptive, at TEST evaluation only, on the upstream-protocol score — AM-13, `8084671` (M5, B60). |
| 6 | **Augmentation order.** ch3 §E.2.b says augmentations are "applied after padding". | E1 (`src/data/transforms.py:166-186`) scales the *unpadded* image, then pads inside the random crop. Scale, rotation-before-crop, crop, cat_max_ratio, flips and photometric semantics match. | IMPLEMENTATION DEVIATION (non-material) + MANUSCRIPT ERROR (wording) | E1 kept; manuscript to describe the behaviour, not the order-of-padding phrase |
| 7 | **Albumentations named, NumPy/PIL used.** | ch3 Table 3.4 vs `requirements-e1.txt`, `configs/augment.py` | IMPLEMENTATION DEVIATION (library only) | Contract relabelled; manuscript to use behaviour-specific wording |
| 8 | **Per-image mIoU eligibility.** ch1 "Definition of Terms" excludes classes "with zero union between prediction and ground truth" (union-present). ch3 §F.2 excludes classes with zero ground-truth pixels (GT-present). | ch1 vs ch3 | MANUSCRIPT ERROR | ch3 (GT-present) governs; EVALUATION_CONTRACT §3.2 unchanged |
| 9 | **Holm family.** ch1 speaks of "ten planned pairwise comparisons"; ch3 Table 3.6's **E3 vs E6** row lists a Holm-corrected Wilcoxon. ch3 §F.1.d enumerates exactly the frozen eight. | ch1, ch3 §B, Table 3.6, §F.1.d | MANUSCRIPT ERROR | Eight-member family unchanged (STATISTICAL_ANALYSIS_CONTRACT §0.1, clarified) |
| 10 | **Pratt z formula.** ch3 p.143's μ = n(n+1)/4 and "preserving their contribution to the variance" vs pinned SciPy 1.11.4's Cureton adjustment, which removes the zero-rank block (`_morestats.py:4129-4133`). | ch3 vs SciPy | MANUSCRIPT ERROR | Frozen `scipy.stats.wilcoxon` call governs; manuscript formula to be corrected |
| 11 | **ch3 class-count prose.** "indices 0 through 114", LR-ASPP projections "to the 115 output channels", and "averages over all 115 verified classes, consisting of the background class and 115 disease classes". | ch3 vs empirical 0–115 | MANUSCRIPT ERROR | 116 outputs unchanged (conflict #1) |
| 12 | **ch3 internal contradictions left open.** (a) clipping in the distillation stages vs an identical E1/E2/E3 recipe; (b) QAT best-val selection vs "final post-quantization checkpoint, no validation selection"; (c) multi-seed "optional" vs "three seeds planned"; (d) E6-KD triggered by a clean-TEST drop, which is TEST-informed. | ch3 | MANUSCRIPT ERROR + METHODOLOGY DECISION OPEN | ~~Registered as M1, M9, M8, M7 in `open_questions.md`; nothing chosen~~ **[UPDATED 2026-09-23 — B64 C5]** Resolved in `8084671`: (a) AM-7, (b) AM-4, (c) AM-1, (d) AM-3 (docs/PREREGISTRATION_AMENDMENTS.md). |
| 13 | **PREREGISTRATION items that a future amendment must address.** PREREGISTRATION.md is **not edited** (anchor `569cfbb`; its evidentiary value depends on staying unchanged). Items: §10 U3/U4 (clipping pilots, "no clipping" excluded) → M1; §6 (E6-KD clean-TEST trigger) → M7; §10 U8 (RTX 4090 working assumption) → stale by measurement (B48; B52); §10 U9 (NMF control) → M4, now with CPU measurement. | PREREGISTRATION vs later evidence | DOCUMENTATION GAP (amendment record) | Any amendment is recorded as a new, dated document or register entry, never by rewriting the preregistration. **[UPDATED 2026-09-23 — B64 C5]** Resolved: recorded in docs/PREREGISTRATION_AMENDMENTS.md (`8084671`) — U3 withdrawn (AM-7), §6 superseded (AM-3), U4 unchanged, U8 stale, U9 superseded by M4 (B61). |

## 14. Teacher protocol locks — B60 (2026-09-22) resolves #5's open items by decision, not by silent reconciliation

Source: [reports/b60_teacher_methodology_lock.md](../reports/b60_teacher_methodology_lock.md).

| # | Conflict | Sources | Class | Standing |
|---|---|---|---|---|
| 14a | **Teacher acceptance.** ch3 p.102: "recover 42.05% within ±1.5–2.0 pp — protocol matching". | ch3 vs Wei paper (SGD) vs public PlantSeg code (AdamW; validation on TEST) | MANUSCRIPT ERROR → **LOCKED (M5)** | Readiness rule R1–R4 replaces the band. R3 is a controlled same-VAL operational floor (> E1 VAL 0.36314016580581665). Wei values contextual only. Historical records keep the old wording and are superseded |
| 14b | **Teacher augmentation.** ch3 is silent for the teacher, while the runtime config uses the upstream family pipeline with `PhotoMetricDistortion`. | ch3 §E.2.c (student recipe + contamination rule) vs MMSeg generic ADE20K default vs Guo (flip/scale/crop only) | METHODOLOGY DECISION → **LOCKED (M2)** | Semantic parity with the E1–E3 recipe; no brightness, contrast, blur, noise or JPEG. ~~The runtime is non-conformant until B60 step H~~ **[UPDATED 2026-09-23 — B64 C5]** Resolved: implemented in B62 (`3c43f89`) and frozen (runbook §4a, `a51a092`). |
| 14c | **Teacher scale.** Upstream short-side scaling vs the thesis long-side evaluation. | MMSeg / PlantSeg configs vs ch3 pp.128–129 ("common spatial footing") | METHODOLOGY DECISION → **LOCKED (M3)** | Long side 512·r, r ~ U[0.75, 2.0]; evaluation long side 512 + pad |
| 14d | **Validation interval.** The runtime config's "10,000 — public schedule_40k.py" is PlantSeg's file, which validated on TEST. MMSeg 1.2.2's own `schedule_40k.py` uses 4,000. | runtime config comment vs upstream sources | DOCUMENTATION GAP (mis-cited source) → **LOCKED (M5)** | Every 4,000 iterations, VAL only |
| 14e | **TEST during teacher preflight.** `check_splits` counts TEST filenames. | TEST policy vs launcher | METHODOLOGY DECISION → **LOCKED (M11)** | TRAIN/VAL-only configured data root with a fail-closed TEST-absence assertion; applies to the teacher, E2 and E3 |

~~**Still open:** M4 (NMF/Hamburger control) and M12 (operational checkpoint selection).~~
**[UPDATED 2026-09-22 — B61]** M4 and M12 are now LOCKED (§15). The manuscript amendments are listed in
B60 §11 and are not yet made.

## 15. Teacher NMF, checkpoint-selection and CE-normalisation locks — B61 (2026-09-22)

Source: [reports/b61_teacher_nmf_checkpoint_selection_lock.md](../reports/b61_teacher_nmf_checkpoint_selection_lock.md).

| # | Conflict | Sources | Class | Standing |
|---|---|---|---|---|
| 15a | **NMF evaluation randomness.** ch3 is silent. Upstream `rand_init=True` makes every forward draw fresh bases from the CPU generator; on the real checkpoint 20/20 forwards differ (26.0% mean pixel change). The MMSeg README advises a test-time seed; the Hamburger ablation favours random init. | ch3 (silent) vs pinned `ham_head.py` vs Hamburger App. F vs MMSeg README | METHODOLOGY DECISION → **LOCKED (M4)** | `rand_init=True` kept; M4-T upstream; M4-V pass-level seed-42 dedicated stream with caller-RNG restore, frozen order, batch 1; M4-KD private stream seeded once. `rand_init=False`, fixed basis, K-draw averaging and image-keyed bases rejected |
| 15b | **Operational checkpoint selection.** B60 recorded the best-VAL intent only; `save_best='mIoU'` depended on RNG state, and MMSeg's `IoUMetric` rounds its summary to two decimals. | B60 §2.3 vs pinned `iou_metric.py:136` / `checkpoint_hook.py:123` | METHODOLOGY DECISION → **LOCKED (M12)** | 10 validations (4k…40k) under M4-V; numerically highest full-precision all-class mIoU; exact tie → earliest; no tolerance band; persisted selection record; R3 under M4-V |
| 15c | **Padding in the teacher loss.** ch3 p.128: the 255 padding is excluded "from all loss calculations". MMSeg CE default `avg_non_ignore=False` gives ignored pixels zero loss but keeps them in the mean denominator; E1 averages over valid pixels. | ch3 p.128 vs pinned `cross_entropy_loss.py` vs `src/training/losses.py:53` | RUNTIME/METHODOLOGY CONFLICT → **LOCKED (M13, new)** | teacher CE `avg_non_ignore=True`, `ignore_index=255`; implemented before the official teacher run. Thesis-derived correction, not a claim about Wei or upstream |
| 15d | **Checkpoint filename suffix.** `b6f6c70c` is not the SHA-256 prefix of the official bytes (`647a0cda…`). | OpenMMLab naming convention vs measured bytes and Last-Modified 2023-02-24 | DOCUMENTATION NOTE (upstream re-upload, INFERRED) | Not a blocker. Measured SHA-256 recorded (trust-on-first-use, server MD5 match); the suffix is not used as verification |

---

## Consistency checks — values that AGREE across files (recorded so they are not re-litigated)

- **5 corruptions** (motion blur, Gaussian noise, JPEG, brightness, fog) at **severity 1–3**: agree across ch1/ch2/ch3.
- **Global ReLU6 substitution rejected** / native Hard-Swish retained: agree ch2 §B + ch3 §D.
- **Teacher = SegNeXt-B/MSCAN-B**, chosen over SegNeXt-L (44.52%) and ConvNeXt-L (46.24%) for
  capacity/channel-compatibility: agree ch2 + ch3 + Wei numbers.
- **7,774 images / 115 / 34 hosts**, 70/10/20 split, JPEG+PNG, DOI 10.5281/zenodo.17719108,
  CC BY-NC 4.0: agree Wei + ch1 + ch3.
- **42.05% mIoU** teacher benchmark reference: agree ch1 + ch2 + ch3 + Wei Table 3 — as a **contextual
  published value** only. The thesis teacher recipe is not the protocol that produced it (#5).
- **Seed 42**, 512×512, batch 16, ignore_index 255: agree across files.

> No other numeric divergences (learning rates, iteration counts, temperatures, α/β weights, batch size,
> calibration size, augmentation probabilities) were found between the secondary files and ch3 — the
> secondary chapters defer the concrete config numbers to Chapter 3 rather than restating different values.
