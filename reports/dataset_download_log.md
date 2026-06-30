# Dataset Download / Location / Provenance Log — PlantSeg (B20d)

**VERDICT: ✅ PASS.** Lightweight provenance/location log for the local PlantSeg dataset. **This is a
location/provenance/download log — NOT a fresh dataset audit.** Counts and pixel-level facts are reused
from already-committed reports; the only new action here is a **depth-1** existence/layout confirmation
(no recursion, no image/mask decode, no recount).

_Generated: 2026-06-30 · Branch: `master` · HEAD: `67f8737` · Dataset root: `C:\Users\admin\plantseg_data\plantseg`._

---

## 1. Download / provenance status
- **Acquisition:** **Manual, user-provided.** Nothing was downloaded by any task (no automated fetch,
  no URL/SHA capture exists). This is recorded as the provenance of record unless a future step adds a
  verified download log. (Consistent with `reports/dataset_location_log.md`.)
- **Dataset identity / provenance (from `configs/data.py`):** DOI **`10.5281/zenodo.17719108`**,
  license **CC BY-NC 4.0**, source PlantSeg (Wei et al., 2026).

## 2. Location (external to the repo)
- **Dataset root:** `C:\Users\admin\plantseg_data\plantseg` — **exists ✅** (depth-1 confirmed 2026-06-30).
- **Repo root:** `C:\Users\admin\plantseg-thesis`. The dataset lives in a **sibling** folder, **OUTSIDE
  the repository** → git cannot track it; `.gitignore` adds a secondary defense (verified in
  `reports/dataset_location_log.md`). Raw data is **not staged/committed**.

## 3. Top-level layout (depth-1 confirmation, no recursion)
| Entry | Type |
|---|---|
| `images/` | DIR |
| `annotations/` | DIR |
| `annotation_train.json` · `annotation_val.json` · `annotation_test.json` | FILE |
| `Metadata.csv` | FILE |

Matches the official PlantSeg layout.

## 4. Split folders used by the repo
The loader (`src/data/dataset.py`) pairs by **folder + stem** — `images/<split>/<stem>.jpg` ↔
`annotations/<split>/<stem>.png` — and **does not** parse the split JSONs/CSV for membership. All six
split directories confirmed present (depth-1, names only):

| Split | image folder | mask folder |
|---|---|---|
| train | `images/train` ✅ | `annotations/train` ✅ |
| val | `images/val` ✅ | `annotations/val` ✅ |
| test | `images/test` ✅ | `annotations/test` ✅ |

## 5. Counts (reused from `reports/dataset_audit_summary.md` — not recounted)
| Split | Complete image–mask pairs | Silent drops |
|---|---|---|
| train | **5,367** | 0 |
| val | **846** | 0 |
| test | **1,561** | 0 |
| **Total** | **7,774** | **0** |

## 6. Mask convention (reused from committed audits — not re-derived)
- **background = raw mask label 0**
- **disease labels = raw mask labels 1–115**
- **`num_classes = 116`**
- **255 = preprocessing pad/ignore ONLY** — absent from the raw released masks (0/7,774 masks contain
  255; introduced at resize-pad + rotation fill).
- **effective `reduce_zero_label = False`** — masks already encode the final 116-class space (bg=0 kept,
  no remap). *Nuance:* the `configs/data.py` literal is still `"NEED_TO_CONFIRM"`, a stale placeholder for
  the formal naming convention; the **proven pipeline behavior is no label remapping** (`reports/dataloader_smoke.md`,
  `reports/dataset_audit_summary.md`).
- `mask = category_id + 1` — proven 100% vs split-local COCO across all annotated images.

## 7. Were any dataset files modified?
**No.** Read-only throughout: a depth-1 directory listing only; no file moved/renamed/deleted/edited, no
image/mask decode, no recount, no re-audit. The dataset is external and untracked.

## 8. Source reports / inputs used
- `reports/dataset_location_log.md` — manual-download status, path, `.gitignore` safety.
- `reports/dataset_audit_summary.md` (B16-15) — counts, proven mask facts, known non-blocking issues.
- `reports/dataloader_smoke.md` (B20b) — live confirmation labels ⊆ {0..115,255}, effective `reduce_zero_label=False`.
- `reports/dataset_report.json` — original full read-only scan (base evidence).
- `configs/data.py` — root, `num_classes`, ignore index, background index, DOI/license.

## 9. Known / unresolved (non-blocking)
- **Download provenance is manual** (no URL/SHA/date captured); accepted as-is.
- **Absent-class support:** class 41 has no **test** mask support; class 68 has no **val** support (both in
  train) → eval must NaN/skip per-class IoU for those.
- **12 cosmetic `Metadata.csv` `Index`** errors (0.15%); masks/COCO authoritative — no pipeline impact.
- **Disease-only mIoU convention** (exclude bg 0) — reporting choice (open_questions #2).
- **Teacher-init weights** — env-gated (teacher/E2–E3 phase), not a dataset matter.

## 10. Strict-G0 / project status
- This log is the explicit artifact that **closes the *optional* strict-G0 "dataset download/location
  log" item** (complementing the still-valid `reports/dataset_location_log.md`). _(The closeout/status
  docs are not edited in this step.)_
- **GitHub remote/push remains DEFERRED** (no remote URL provided).
- **Real E1 training remains gated** by the GPU/CUDA stack (pinned `torch 2.1.0+cu121`) and the ImageNet
  cache/download — not runnable on this CPU-only machine (`reports/platform_verify.md`).

## Recommended next step
Commit B20d (this single file), explicit-path staging. Then the remaining open paths are unchanged:
**(a)** provide a GitHub remote URL + visibility → push `master`; **(b)** provision a GPU target for the
real E1 run; or **(c)** continue local docs/pre-E2/E3 planning. Optionally, a later doc step can mark this
optional item DONE in `reports/strict_G0_closeout_update.md` / `reports/week1_G0_status.md`.

---

## Provenance / guardrails honored
Only `reports/dataset_download_log.md` was written. Depth-1 existence/layout check only — no recursion,
no decode, no recount, no re-audit; no dataset files modified. No training/download/GPU. `reference.pdf`,
`context.md`, teacher-init, code/config, and the location/closeout/status docs untouched; nothing staged
or committed; Claude memory and `~/.claude` untouched.
