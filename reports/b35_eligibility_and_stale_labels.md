# B35 — Metric-eligibility reconciliation + stale-label sweep

**Type:** read-only audit. **Status:** COMPLETE. **Date:** 2026-09-05.
**Predecessor:** [`reports/b34_config_reconciliation.md`](b34_config_reconciliation.md).

**No file was edited, created, staged, committed, or pushed by this task except this report.**
`docs/reference/reference.pdf` was not opened, read, hashed, copied, or staged. No manuscript file was
touched — none is in scope. No training, no downloads, no GPU. Every claim is quoted from a file read
during this task.

**Headline:** both of the brief's premises are **refuted**. Part A is a level-confusion; Part B asks
for a field that already exists. One genuine finding survives (Part B, §2.3) and one process defect is
newly characterised (Part C, §3.4).

---

## 1. Part A — eligibility rule: is it a divergence?

### VERDICT

> # **ALIGNED**

Contract, code, and ch3 state the same rule **at each level**. The two levels deliberately differ from
each other, and that difference is specified in three places.

### A1. The rule as implemented — the two levels use DIFFERENT rules, by design

**Dataset-level macro-mIoU — UNION-present.** [`src/eval/metrics.py:128-130`](../src/eval/metrics.py#L128):

```python
    tp, gt, pr, un = _cm_parts(cm)
    eligible = _restrict(un > 0, class_indices)
    return _macro(tp / un.clamp_min(1e-9), eligible)
```

**Per-image mIoU — GT-present.** [`src/eval/metrics.py:74-83`](../src/eval/metrics.py#L74):

```python
    diag = torch.diag(cm).float()
    row = cm.sum(1).float()                 # GT pixels per class
    col = cm.sum(0).float()                 # predicted pixels per class
    iou = diag / (row + col - diag).clamp_min(1e-9)
    present = row > 0                        # classes present in ground truth
```

`per_image_miou` delegates to exactly this function
([`:177-184`](../src/eval/metrics.py#L177)). **The two do not use the same rule — and that is not a
defect.** The module says so itself, [`metrics.py:3-4`](../src/eval/metrics.py#L3):

> "Semantics are frozen by docs/EVALUATION_CONTRACT.md sections 3.1/3.2 (decisions D3/D3b in
> docs/open_questions.md). **THREE ELIGIBILITY RULES COEXIST BY DESIGN -- do not "harmonise" them**"

and marks the per-image path explicitly at [`:16-18`](../src/eval/metrics.py#L16):

> "per-image mIoU -> GT-present : preregistered by ch3 section (f) "per-image absent-class exclusion".
> `_miou_from_cm` / `per_image_miou` are the per-image path and are **DELIBERATELY** left on the
> GT-present rule."

`miou_from_confusion`'s own docstring repeats the boundary, [`:125`](../src/eval/metrics.py#L125):
*"NOT the per-image rule -- per-image metrics stay GT-present via `_miou_from_cm`."*

### A2. The rule as specified

**`EVALUATION_CONTRACT.md` §3.1 — dataset level** ([`:160-163`](../docs/EVALUATION_CONTRACT.md#L160)):

| **all-class mIoU** | `{0..115}` | **union-present** | `TP_c / UN_c` | **HEADLINE / official** |

**`EVALUATION_CONTRACT.md` §3.2 — per image** ([`:191-196`](../docs/EVALUATION_CONTRACT.md#L191)):

| **per-image disease-only mIoU** | `{1..115}` | **GT-present** | **PRIMARY INFERENTIAL UNIT** |
| **per-image all-class mIoU** | `{0..115}` | **GT-present** | secondary / retained |

With the provenance note at [`:199-201`](../docs/EVALUATION_CONTRACT.md#L199):

> "**GT-present per-image is preregistered, not chosen here.** `IMPLEMENTATION_CONTRACT.md` §(f)
> specifies *"per-image absent-class exclusion"* and `context.md:118` repeats *"Absent classes excluded
> per-image"*."

**`IMPLEMENTATION_CONTRACT.md`** ([`:426-428`](../docs/IMPLEMENTATION_CONTRACT.md#L426)):

> "**Primary inferential unit:** per-image **disease-only mIoU** (115 disease classes = mask values
> **1–115**; … per-image absent-class exclusion; 255 always excluded)."

and [`:421-422`](../docs/IMPLEMENTATION_CONTRACT.md#L421): *"class-eligibility rules (union-present for
dataset-level IoU/Dice, GT-present for mAcc and for both per-image vectors)"*.

Both contracts state the same two-level split the code implements.

### A3. ch3 — verbatim, and NOT provisional

**`docs/reference/ch3.pdf` is a real text PDF, not an OCR artifact.** It is 77 pages and yields an
embedded text layer directly; [`docs/updated_ch3_sync_audit.md:9-12`](../docs/updated_ch3_sync_audit.md#L9)
independently records extraction *"via `pdftotext -layout`"* at **19,630 words**, byte-identical to git
HEAD, SHA-256 `8a558e82…`. **The comparison below is therefore authoritative, not PROVISIONAL.** (This
file is *not* the protected `reference.pdf`, which was not touched.)

The absent-class wording occurs three times. **Every occurrence is scoped to per-image:**

> **p57:** "Per-image mIoU is the macro-average of class IoU values for classes with non-zero
> ground-truth pixels in that image (absent-class exclusion rule); classes with zero ground-truth
> pixels in a given image [are excluded from that image's per-image IoU average]"

> **p64:** "For each image, disease classes with zero ground-truth pixels in that image are excluded
> from that image's average"

> **p66:** "The same absent-class exclusion rule applied to clean mIoU is enforced: classes with zero
> ground-truth pixels in a given corrupted test image are excluded from that image's per-image IoU
> average."

p57 also gives the dataset-level construction — *"images per class, then taking a macro-average of IoU
per class"* — but **states no eligibility rule for it**. ch3 is *silent* on dataset-level eligibility,
not contradictory. The contract fills that gap from the official benchmark rather than by choice,
[`EVALUATION_CONTRACT.md:114-119`](../docs/EVALUATION_CONTRACT.md#L114):

> "### 2.3 `nan_to_num` is NOT configured — this is the decisive finding … **The official PlantSeg
> [configuration does not set it]** … reduction is genuinely NaN-aware (`np.nanmean`)."

### A4. The single line of evidence that decides it

> ch3 p57 — *"classes with **non-zero ground-truth pixels in that image**"*
> ≡ [`metrics.py:78`](../src/eval/metrics.py#L78) — `present = row > 0`, where `row = cm.sum(1)` is
> **GT pixels per class**.

Per-image: ch3, both contracts, and the code all specify GT-present. Nothing diverges.

### The brief's premise, stated plainly

The brief asserted that ch3 says *"the opposite"* of `metrics.py:129-131`. **It does not.** The premise
compares ch3's **per-image** sentence against the code's **dataset-level** reducer. These are two
different metrics with two deliberately different rules, and the code, both contracts, and
`open_questions.md` D3/D3b all say so. **This is a level-confusion, not a divergence.** It is recorded
as REFUTED rather than smoothed into a wording difference — and equally, it is not inflated into a
defect.

**A5 does not trigger.** The verdict is ALIGNED, not DIVERGENT-DOC and not UNDETERMINED, so no DOCX
reconciliation is required *on the eligibility question*. (A separate, unrelated manuscript item is
raised in §2.3 below.)

---

## 2. Part B — per-image eligible-class count as a recorded field

### 2.1 B1 and B3 — REFUTED. The fields already exist

[`src/eval/evaluate.py:145-158`](../src/eval/evaluate.py#L145):

```python
@dataclass(frozen=True)
class PerImageRow:
    """Exactly the fields frozen by contract section 5.4 -- no per-image classwise arrays."""
    image_id: str
    clean_image_id: str
    manifest_index: int
    condition: Condition
    all_class_miou: float | None
    all_class_miou_status: str
    disease_only_miou: float | None
    disease_only_miou_status: str
    n_eligible_all_class: int
    n_eligible_disease_only: int
    gt_disease_classes: list[int]
```

Serialized at [`:170-172`](../src/eval/evaluate.py#L170), populated at
[`:368-370`](../src/eval/evaluate.py#L368), and frozen in
[`EVALUATION_CONTRACT.md §5.4`](../docs/EVALUATION_CONTRACT.md#L386). Dataset-level counts exist too —
`all_class_miou_n_eligible`, `all_class_dice_n_eligible`, `all_class_macc_n_eligible`,
`disease_only_miou_n_eligible` ([`:397-403`](../src/eval/evaluate.py#L397)).

**B3's recommendation is therefore moot: there is nothing to add.**

### 2.2 B1's statistical concern does not apply at the per-image level

B1 argued that under union-present the count is model-dependent per image, so `d_i = mIoU_A(i) −
mIoU_B(i)` would average over different-sized class sets. **At the per-image level this is false.**
[`evaluate.py:355`](../src/eval/evaluate.py#L355) and [`:368`](../src/eval/evaluate.py#L368):

```python
                gt_present = np.nonzero(gt_i > 0)[0]
                ...
                    n_eligible_all_class=int(gt_present.size),
```

The count derives from **ground truth alone**. Ground truth is identical for both models on the same
image, so `n_eligible_*` is **model-INDEPENDENT per image**, and the paired difference is taken over
the *same* class set for A and B. **A Wilcoxon result cannot be denominator-driven at this level.**

The *"model-dependent"* language the brief cites —
[`EVALUATION_CONTRACT.md:188-190`](../docs/EVALUATION_CONTRACT.md#L188) — is about the **dataset-level**
union-present count, and the contract already requires it be recorded per run. **The same
level-confusion as Part A.**

**B4:** moot — nothing is added, so nothing changes. For completeness: had a field been added, it would
have been **purely additive telemetry**. No reducer in `metrics.py` reads `PerImageRow`.

### 2.3 The one real finding — ch3 p50 vs the repo's per-image artifact

ch3 p50 specifies a per-image artifact that the repo does not write. Each deviation was checked
**individually** against the contracts rather than assumed unreconciled:

| ch3 p50 deviation | repo behaviour | reconciled? |
|---|---|---|
| **CSV** | `per_image.jsonl` (JSON Lines) | ✅ **YES** — [`EVALUATION_CONTRACT.md:280`](../docs/EVALUATION_CONTRACT.md#L280) §5.1 *"Layout (chosen: hybrid; Options "one monolithic JSON" and **"summary + CSV only" are rejected**)"*, with rationale at [`:292-296`](../docs/EVALUATION_CONTRACT.md#L292): ~23k rows/stage at corruption scale, deterministic diffs, exact int64. |
| **1,554 test images** | 1,561 | ✅ **YES** — [`:35-42`](../docs/EVALUATION_CONTRACT.md#L35) *"### The `1,554` value is NOT a count — never use it"* … *"**The authoritative test count is 1,561.** Any document, prompt, or code path stating ≈1,554 is stale."* |
| **per-image `Dice` and `mAcc` columns** | not written | ❌ **NO** |
| `stage` as a per-image column | in `RunMeta` / directory, not per row | ❌ no (minor normalization) |

**ch3 contradicts itself on the third item.** p50 lists `Dice` and `mAcc` as per-image columns; p65
states the opposite:

> **p65:** "Reported only at the dataset level, per-image Dice is not computed because it would be
> redundant with per-image mIoU under the monotonic Dice = 2·IoU/(1+IoU) relationship."

The repo follows **p65**, consistent with
[`IMPLEMENTATION_CONTRACT.md:438-439`](../docs/IMPLEMENTATION_CONTRACT.md#L438):

> "- **Dice:** dataset-level **secondary, descriptive only** — no separate test (monotonic with IoU).
> - **mAcc:** descriptive only (benchmark against Wei)."

**But no document in the repository quotes or addresses p50's column list.** Searching the contracts for
the column sentence, `{image_id`, and `outputs/<stage>` returns only the repo's own `per_image.jsonl`
schema — never a reconciliation against p50. The *substance* is settled; the *manuscript sentence* is
not. **This is a manuscript-vs-repo gap for the DOCX.** A ready-to-apply, unapplied fix is in §4.1.

---

## 3. Part C — repo-wide stale-label sweep

### 3.1 Method: 366 raw hits → ~40 distinct questions

**Raw marker counts** across `docs/` + `reports/` (`.md` only): `NEED_TO_CONFIRM` 148 · `NTC-` 115 ·
`PROVISIONAL` 38 · `PARTIAL` 36 · `Open`/`OPEN` 29 · `TODO` 0. **Total 366.**

**Exclusion rule — stated so the arithmetic closes:**

| excluded | why | effect |
|---|---|---|
| `.pyc` binaries | compiled copies of source lines already counted | 4 |
| File-header narration | prose *describing* the convention, not an assignable status | 4 (per B34) |
| **`PARTIAL` as a runtime verdict token** | overwhelmingly `verify_env`'s PASS/PARTIAL/FAIL result, e.g. `b31c_closeout.md` *"PARTIAL (accepted under `--dev-rehearsal` ONLY)"*. **Not a status label.** Counting these would badly inflate the total. | ~34 of 36 |
| `Open`/`OPEN` in headings and meta-references | `# Open Questions`, `## (g) Open / NEED_TO_CONFIRM list`, *"held OPEN … it is now RESOLVED"* | ~14 of 29 |
| `b34_config_reconciliation.md` + this report | self-reference: they discuss markers rather than carry them | ~12 |

**Deduplication — by question text, not by identifier.** 87 `NTC-` table rows across 16 files → 51
distinct texts → **15 underlying topics**.

> ### ⚠ `NTC-N` numbering is **per-report LOCAL**, not global.
> `NTC-1` is *"`mask = category_id + 1` mapping"* at
> [`annotations_test_audit.md:103`](annotations_test_audit.md) but *"`reduce_zero_label` / background
> convention"* at [`dataset_code_expectation_audit.md:161`](dataset_code_expectation_audit.md).
> **Deduplicating by identifier would have merged unrelated questions.** Clustering was done on
> question text instead. This is not an incidental detail — see §3.4.

**Resulting distinct-question total: ~40** = 15 dataset-audit topics + 10 `configs/` keys (carried from
B34) + 9 `open_questions.md` items + ~6 from `conflicts.md` / PROVISIONAL topics.

### 3.2 The two buckets

Applying a single "STALE-LABEL" bucket to `reports/*.md` would be wrong, so it is split:

| bucket | definition | disposition |
|---|---|---|
| **STALE-LABEL (live doc)** | In a file that claims to describe **current** state: `docs/open_questions.md`, `docs/conflicts.md`, `configs/`, the contracts. | **Fixable.** Proposals in §4. |
| **AGED-SNAPSHOT (historical report)** | In a dated `reports/*.md` audit. **Accurate at its date.** | **MUST NOT be edited.** Editing would falsify the audit trail. |
| **LIVE** | Genuinely unresolved. | Named with what closes it. |
| **DEFERRED** | Correctly open; belongs to a post-E1 stage. | Leave. |

### 3.3 The table

**AGED-SNAPSHOT — the B16 dataset-audit chain (15 topics, 87 rows, 16 files). DO NOT EDIT.**

These are **sequenced carry-forwards**, not labels that outlived a resolution: each names the later
audit that will close it, and that audit exists. The chain terminates in
[`reports/dataset_audit_summary.md`](dataset_audit_summary.md) — *"# PlantSeg Dataset Audit —
Consolidated Closeout (B16-15)"*, generated **2026-06-28**, verdict *"**PASS (dataset validated
end-to-end)**"*, all three splits 5,367 / 846 / 1,561 with **0 silent drops**.

| # | topic | representative file:line | bucket | resolving artifact + date | E1-blocking |
|---|---|---|---|---|---|
| 1 | `mask = category_id + 1` mapping | `annotation_test_audit.md:128` | AGED-SNAPSHOT | `dataset_report.md` + `conflicts.md` §1 — empirical, **2026-06-27** (99.85%) | N |
| 2 | `reduce_zero_label` / background naming | `annotation_test_audit.md:133` (*"Open since B16-0."*) | AGED-SNAPSHOT | `open_questions.md:20` A0-FIX, **2026-07-26** | N |
| 3 | JSON `width`/`height` vs EXIF dims | `annotation_test_audit.md:132` | AGED-SNAPSHOT | `dataset_audit_summary.md` §0 — cosmetic metadata, no pipeline impact, **2026-06-28** | N |
| 4 | annotation id space 1…7,916 vs 7,774 | `annotation_train_audit.md:143` | AGED-SNAPSHOT | as above — cosmetic, **2026-06-28** | N |
| 5 | orphan images FLAG-A/E | `annotation_test_audit.md:131` | AGED-SNAPSHOT | as above, **2026-06-28** | N |
| 6 | FLAG-F / 0-annotation image+mask | `annotation_test_audit.md:130` | AGED-SNAPSHOT | `annotations_test_audit.md:23` — FLAG-F mask verified, **2026-06-28** | N |
| 7 | per-mask label range 0–115 / 255-absent | `annotations_test_audit.md:105` | AGED-SNAPSHOT | `test_mask_value_audit.md` + `trainval_mask_value_audit.md` (decoded), **2026-06-28** | N |
| 8 | mask inventory & image↔mask pairing | `images_test_audit.md:102` | AGED-SNAPSHOT | `annotations_test_audit.md:21,43,60` — exactly 1,561, 0 orphans, **2026-06-28** | N |
| 9 | `file_name` ↔ actual files | `annotation_test_audit.md:129` | AGED-SNAPSHOT | `dataset_audit_summary.md` §1 — stem sets identical, **2026-06-28** | N |
| 10 | 12 Metadata↔COCO exceptions | `metadata_csv_audit.md:116` | **already RESOLVED in place** | marked *"**RESOLVED** — yes, exactly 12/12"* | N |
| 11 | absent-class eval support (cls 41/68) | `metadata_csv_audit.md:118`, `trainval_mask_value_audit.md:99`, `test_mask_value_audit.md:109` | AGED-SNAPSHOT | `EVALUATION_CONTRACT.md` §2.3/§3.1 + `metrics.py:129-131`, **2026-07-26** | N |
| 12 | "split not yet audited" progress markers | `train_split_consistency_audit.md:91` | AGED-SNAPSHOT | the val/test audits themselves exist, **2026-06-28** | N |
| 13 | `dataset_report` `mask_value_encoding` inferred | `annotations_train_audit.md:84` | AGED-SNAPSHOT | superseded by topic 1, **2026-06-27** | N |
| 14 | total image files = 7,774 | `annotation_train_audit.md:133` | AGED-SNAPSHOT | `dataset_audit_summary.md` §1, **2026-06-28** | N |
| 15 | 100% exact (train+val+test annotated) | `trainval_mask_value_audit.md:77` | AGED-SNAPSHOT | as above, **2026-06-28** | N |

**STALE-LABEL (live docs) — fixable**

| # | marker | file:line | bucket | resolving artifact + date | E1-blocking |
|---|---|---|---|---|---|
| 16 | `real_class_weights: NEED_TO_CONFIRM` | `configs/loss.py:14` | **STALE-LABEL** | `reports/e1_class_weights.json` (B18a, `4eb5644`), **2026-06-29** | **N** — no code reads it (B34 §2) |
| 17 | `pretrained: NEED_TO_CONFIRM` | `configs/model.py:19` | **STALE-LABEL** (dead) | `configs/e1_student.py:8`; `MODEL["pretrained"]` has zero readers | **N** |
| 18 | comment *"still open — see open_questions #2"* | `configs/loss.py:16` | **STALE-LABEL** (comment only; value `False` is correct) | `open_questions.md:20` **RESOLVED 2026-07-26** | N |
| 19 | comment *"remains open; see open_questions #2"* | `configs/data.py:37` | **STALE-LABEL** (comment only) | as above, **2026-07-26** | N |
| 20 | **statsmodels** *"no version stated"* | `docs/open_questions.md:731` | **STALE-LABEL** | `requirements.lock:63` → `statsmodels==0.14.6`, **2026-08-16** | N |

**LIVE / DEFERRED — `docs/open_questions.md` §"NEED_TO_CONFIRM"**

| # | item | file:line | bucket | what closes it | E1-blocking |
|---|---|---|---|---|---|
| 21 | #3 `λ_logit` | `open_questions.md:716` | DEFERRED | E2 validation sweep `{0.25,0.5,1,2,4}` | N |
| 22 | #4 E6-KD reduced weights | `:722` | DEFERRED | conditional on E3→E6 drop > 1.0 pp | N |
| 23 | #5 **Albumentations** "pinned in the manifest" | `:729` | **LIVE (manuscript)** | not a version lookup: `requirements-e1.txt:10-11` records Albumentations/OpenCV as **excluded** from E1 — a manuscript-vs-repo claim | N |
| 24 | #6 additional seed values (3-seed E1/E3) | `:734` | **LIVE** | a scope/budget decision before multi-seed runs. See §5.3. | **N** — E1 launches at seed 42; this is budget scope, not a correctness gate |
| 25 | #7 RunPod hardware specifics | `:739` | LIVE | logged at run time | N |
| 26 | #8 citation details | `:744` | LIVE | manuscript verification | N |
| 27 | #9 all experimental result numbers | `:750` | LIVE by design | the runs themselves | N |
| 28 | D22 pooled robustness estimand | `:550` | DEFERRED (`⏸️`, explicitly *"not a limitation"*) | post-E1 | N |
| 29–38 | the 10 `configs/` keys | per B34 §2 | 2 STALE-LABEL (#16,#17 above) · 8 DEFERRED | per B34 | N |

### Counts

| bucket | count | E1-blocking |
|---|---|---|
| **AGED-SNAPSHOT** (historical; must not edit) | **15 topics** (87 rows, 16 files) | 0 |
| **STALE-LABEL** (live doc; fixable) | **5** | 0 |
| **LIVE** | **5** (#23, #24, #25, #26, #27) | 0 |
| **DEFERRED** | **10** (#21, #22, #28 + 8 `configs/` keys, minus the 2 STALE) | 0 |

> ### **E1-BLOCKING: 0**
>
> **C3, applied as specified.** The B34 standard holds throughout: *a stale record that no code reads
> is a documentation defect, not run-blocking.* Every STALE-LABEL item was checked for readers; none
> has any. The count is **not** inflated: item #24 (three-seed E1) is the only LIVE item that touches
> E1 at all, and it is a **budget/scope decision**, not a correctness gate — E1 launches correctly at
> seed 42 today.

### 3.4 C4 + C5 — the process defect, and its mechanical cause

B34 found three stale labels; this sweep finds **five** STALE-LABEL items in live documents, plus 15
AGED-SNAPSHOT topics that read as open to anyone who greps.

**The naive reading — "more labels need flipping" — is wrong and would make things worse.** The B16
audits were *correct at their date*: `annotation_test_audit.md:133` says *"Open since B16-0"* on
2026-06-28, and the question closed 2026-07-26. **Editing a historical audit to say "resolved" would
falsify the audit trail** and destroy the provenance the reports exist to provide.

> ### The defect is the absence of a single authoritative **live status index**, not insufficient
> label-flipping.

**C5 — the mechanical cause is the identifier scheme, not the labels.** `NTC-N` numbering is
**per-report local** (§3.1): `NTC-1` denotes different questions in different files. This is not a
separate observation — **it is the reason no live index can exist today.** Without a stable global id:

1. No artifact can reference a question **across** reports — `NTC-3` is meaningless without naming the
   file it lives in.
2. A later resolution cannot **point back at** every occurrence it supersedes.
3. Status can therefore only be recovered by **re-reading and re-clustering question text** — exactly
   the 87-row → 15-topic exercise this task performed by hand, which no future reader will repeat.

Any recommended index must therefore assign **stable global ids** and map each to the per-report local
`NTC-N` occurrences it supersedes. **Specified in §4.4. Not created in this pass.**

---

## 4. Proposed minimal edits — **UNAPPLIED**, grouped by tier

Nothing here has been applied. Tiers are separated so nothing is bundled.

### 4.1 Manuscript-only tier (DOCX) — B8

**Applies to the current DOCX. No manuscript file is in scope for this task; no manuscript file was
opened or edited. Recommendation only.**

**The self-contradiction, adjacent, as extracted from `docs/reference/ch3.pdf`:**

> **p50:** "Per-image mIoU is recorded for all 1,554 test images and all stages (E1–E7 + teacher),
> stored as a CSV with columns {image_id, stage, mIoU, Dice, mAcc}."

> **p65:** "Reported only at the dataset level, per-image Dice is not computed because it would be
> redundant with per-image mIoU under the monotonic Dice = 2·IoU/(1+IoU) relationship."

p50 lists per-image `Dice` and `mAcc` columns; p65 states per-image Dice is not computed. The repo
follows p65.

**FIND** (the column-list sentence from p50):

```
stored as a CSV with columns {image_id, stage, mIoU, Dice, mAcc}
```

**REPLACE** (passive voice, no first person, minimal change):

```
stored as JSON Lines (one record per image) with the fields image_id, clean_image_id,
manifest_index, condition, all_class_miou, disease_only_miou, their two status fields,
n_eligible_all_class, n_eligible_disease_only, and gt_disease_classes. Dice and mean
accuracy are reported at the dataset level only; per-image Dice is not computed, being
redundant with per-image mIoU under the monotonic Dice = 2·IoU/(1+IoU) relationship.
```

Field list re-derived from [`src/eval/evaluate.py:145-158`](../src/eval/evaluate.py#L145) (not copied
from any report); the Dice justification is p65's own reasoning.

> #### ⚠ The FIND string is from **PDF extraction** and MUST be verified against the current DOCX
> before use.
>
> Known extraction hazards that would silently break a find-and-replace:
> - **curly vs straight apostrophes** (`'` vs `'`) and quotation marks
> - **brace variants** — `{ }` may be typographic, or the list may be rendered without braces
> - **merged or split words** across the extractor's line breaks; the extracted text carries a newline
>   between *"CSV with columns {image_id, stage, mIoU, Dice, mAcc}. These"* and the next line
> - **en-dash vs hyphen** in `E1–E7`
> - the surrounding sentence also contains **`1,554`**, which is separately stale
>   ([`EVALUATION_CONTRACT.md:42`](../docs/EVALUATION_CONTRACT.md#L42) — authoritative count is
>   **1,561**). Correcting both in one edit is reasonable but is a **second** change, not part of this
>   one.

### 4.2 Code tier

**None.** Part A is ALIGNED and Part B's fields already exist. **No code change is recommended by this
audit.** `src/eval/metrics.py` and `src/eval/evaluate.py` are correct as written and must not be
"harmonised" ([`metrics.py:4`](../src/eval/metrics.py#L4)).

### 4.3 Telemetry tier

**None.** B3 is moot (§2.1). Carried forward unchanged from B34 §3 Tier 3 (gate hardening) — **that
work belongs to B34b and is not re-proposed here.**

### 4.4 Documentation tier — live-status index (C5). Specification only; NOT created

Five STALE-LABEL items (§3.3) may be corrected in place — they are in live documents. The two
`configs/` items are already specified in [B34 §3](b34_config_reconciliation.md); items #18–#20 are
comment/pointer text only, change no value, and are listed here for completeness.

**Recommended new artifact — `docs/status_index.md`.** A **new** file that points **at** the historical
reports. It must **never** trigger a retroactive edit to any `reports/*.md`.

**Id scheme:**

```
PSQ-###     PlantSeg Question. Three-digit, zero-padded, assigned monotonically,
            NEVER reused and NEVER renumbered — a retired id stays retired.
```

Each entry carries:

| field | content |
|---|---|
| `id` | `PSQ-###` |
| `question` | canonical one-line text (the cluster label, e.g. *"mask = category_id + 1 mapping"*) |
| `status` | `OPEN` · `RESOLVED` · `DEFERRED` · `LIVE-BY-DESIGN` |
| `resolved_by` | artifact path + **date** (empty if not resolved) |
| `stage` | `E1` · `E2`…`E7` · `teacher` · `manuscript` · `n/a` |
| `supersedes` | list of `<report path>#NTC-N` local occurrences this id subsumes |

`supersedes` is the field that repairs the local-numbering defect: it lets a reader go from a global id
to every historical row, and from any historical row back to current status, **without editing the
historical row**. Seeded from §3.3, it would carry ~40 entries with the 87 `NTC-` rows mapped into
`supersedes` lists.

**Not created in this pass, as instructed.**

---

## 5. What could NOT be determined from the repository

Stated as unknown rather than inferred.

1. **Whether the current DOCX still contains the p50 sentence in the extracted form.** The FIND string
   in §4.1 comes from `ch3.pdf`. `updated_ch3_sync_audit.md:7-9` records the PDF as byte-identical to
   git HEAD, but **no DOCX exists in the repository** (`find . -iname "*.docx"` returns nothing), so
   the manuscript's current wording is **unverifiable from here**. §4.1 is a recommendation contingent
   on that verification.

2. **Whether the p50/p65 contradiction is known to the author.** No file records it. Unknown whether it
   is an oversight or a superseded draft sentence.

3. **Whether topics 3–5 (EXIF dims, id space, orphan images) were formally closed or merely
   downgraded.** `dataset_audit_summary.md` §0 classes them as *"cosmetic dataset-metadata issues (no
   pipeline impact)"* and the final verdict is PASS — but no artifact marks them individually
   RESOLVED. They are bucketed AGED-SNAPSHOT on the strength of the consolidated PASS. **A stricter
   reading would call them LIVE-but-cosmetic.** Flagged rather than decided.

4. **Whether three-seed E1 (#24) is in budget.** `pre_e1_launch_audit.md:3849` raises it as the largest
   planning gap found (3× GPU budget; checkpoint names carry no seed). Whether the project intends to
   run it is a **project decision not recoverable from the files**. It is bucketed LIVE and *not*
   E1-blocking because E1 runs correctly at seed 42 — but that is a statement about the code, not about
   the thesis's requirements.

5. **The exact count of excluded `PARTIAL` / `Open` markers.** §3.1 gives approximate exclusion counts
   (`~34 of 36`, `~14 of 29`). Each was sampled and classified by inspection, not exhaustively
   enumerated line-by-line; the residual is small and does not change any bucket. Stated as approximate
   rather than presented as exact.

---

## Appendix — commands run (all read-only)

```
git status -sb ; git status --porcelain
grep -rIn --include=*.md -E "NEED_TO_CONFIRM|NTC-|PROVISIONAL|PARTIAL|\b(Open|OPEN)\b" docs/ reports/
grep -n "n_eligible|eligible" src/eval/*.py docs/EVALUATION_CONTRACT.md
grep -rn "per-image Dice|per-image mAcc|1,554|CSV" docs/EVALUATION_CONTRACT.md
grep -in "albumentations|statsmodels" requirements.lock requirements-e1.txt
find . -iname "*ch3*" -not -path "./.git/*" ; find . -iname "*.docx" -not -path "./.git/*"
```

ch3 text was extracted read-only with PyMuPDF into the session scratchpad
(`…/scratchpad/ch3_pages.txt`), **outside the repository**. `docs/reference/ch3.pdf` was not modified.
`docs/reference/reference.pdf` was never opened. NTC-row clustering used a read-only Python pass over
`reports/*.md` and `docs/*.md`.
