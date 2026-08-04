# A2b-0 — Authoritative PlantSeg class map: provenance and validation

**RESULT: ✅ A2b-0 COMPLETE — AUTHORITATIVE CLASS MAP VENDORED**
`configs/plantseg_class_map.json` · offline validator **21/21 PASS**, exit 0

> The ordered 116-entry class map comes **verbatim from the official PlantSeg source at a pinned
> commit**. `Metadata.csv["Index"]`, majority vote, annotation/mask frequency, alphabetical order,
> the thesis text, and manual retyping were **all rejected as sources**. The local COCO/Metadata
> evidence was used **only** as a secondary audit and had **no authority over the mapping**.

_Generated 2026-07-26 · CPU only · no dataset image, mask, or checkpoint accessed._

---

## 1. Preflight and repository state

`master` · HEAD `32d52f4` · **0 ahead / 0 behind**. A1b **21/21**, A2a **63/63** before starting.
Confirmed no authoritative ordered class-name list existed anywhere in `configs/`, `src/`, or
`reports/` — only scattered prose mentions of individual disease names inside B16 audit reports.
This artifact closes that gap, which was the A2b blocker.

## 2. Exact official source

| Field | Value |
|---|---|
| Repository | `tqwei05/PlantSeg` — https://github.com/tqwei05/PlantSeg |
| Source path | `mmseg/datasets/plantseg115.py` |
| Source object | `PlantSeg115Dataset.METAINFO["classes"]` |
| **Resolved commit SHA** | **`1a3dd4d9224bcc97a5850af7dd1c423abc24eae0`** |
| Upstream commit date | `2025-03-25T05:31:07Z` |
| **Immutable source URL** | `https://raw.githubusercontent.com/tqwei05/PlantSeg/1a3dd4d9224bcc97a5850af7dd1c423abc24eae0/mmseg/datasets/plantseg115.py` |
| Source size | 7,468 bytes |

`main` was resolved **once** via `GET https://api.github.com/repos/tqwei05/PlantSeg/commits/main`;
the mutable branch URL is recorded only as *how* the SHA was resolved and is never the provenance
pin. All subsequent references use the commit-pinned URL.

**Network actions performed (the full extent of the authorised exception):** one GitHub API GET to
resolve the SHA, one raw GET of the pinned file, and one repeat GET of the **identical pinned URL**
to exercise the generator's offline-reproduction mode (byte-identical, same SHA-256). No clone, no
dataset, no checkpoints, no unrelated files, no forks or mirrors, no installs.

**`WebFetch` was deliberately not used** — it renders pages to markdown through a model, so a
SHA-256 taken from it would be meaningless. Byte-exact retrieval used stdlib `urllib.request` with
an explicit User-Agent, `Accept: application/vnd.github+json` on the API call, a 30 s timeout, and
HTTP-status validation.

## 3. Extraction method

`python-ast-static/1.0.0` — standard-library `ast` only. **No `eval`, no `exec`, no dynamic import,
no regex extraction, no manual copy/paste.** The upstream module is never imported or executed.

The extractor requires: exactly one `ClassDef PlantSeg115Dataset` · exactly one `METAINFO`
assignment · exactly one literal `classes` entry (handling both `dict(...)` call form and `{...}`
literal form) · an `ast.Tuple` whose every element is a literal `str` · exactly 116 entries. Any
dynamic expression, conflicting candidate tuple, or count mismatch **fails loudly** with no partial
artifact written. The upstream `palette=[...][:116]` subscript is ignored, never evaluated.

**Byte handling:** received bytes hashed → written to a temp file → reopened → independently
re-hashed → both hashes required equal → AST extraction from the saved file → temp directory
removed in a `finally` block. **No upstream Python source remains in the repository or worktree.**

## 4. Class-map validation — 21/21

| Check | Result |
|---|---|
| Strict JSON parse (non-finite rejected) | ✅ 12,297 bytes |
| Schema identifier `plantseg-class-map/1.0.0` | ✅ |
| Provenance fields present + typed | ✅ 10/10, none missing or mistyped |
| Official repository / source object | ✅ `tqwei05/PlantSeg`, `METAINFO["classes"]` |
| Commit SHA full 40-char lowercase hex | ✅ |
| Immutable URL contains the SHA, not a branch | ✅ |
| `raw_source_sha256` is 64-hex | ✅ |
| Class space `116 / 0 / 255` | ✅ |
| Exactly 116 ordered entries | ✅ |
| `class_id` sequence exactly `0..115`, in order, none missing or extra | ✅ |
| **Class 0 name is exactly `""`** | ✅ — verified by explicit `is not None` + `len()==0`, never a truthiness test |
| Class 0 role `background_or_non_disease` | ✅ |
| Classes 1–115 role `disease` | ✅ 115/115 |
| Classes 1–115 names non-empty | ✅ |
| Disease-name uniqueness | ✅ 0 duplicates |
| **No palette vendored** | ✅ |
| Entries carry exactly `{class_id, name, role}` | ✅ |
| Canonical semantic hash reproduced independently | ✅ |
| Local vendored-file hash **not** embedded | ✅ (avoids recursive self-hash) |
| Validator imports no network module | ✅ |

**Regeneration idempotence:** re-running the generator in offline `--from-file` mode against the
same pinned bytes produced an **identical semantic payload** and an identical semantic SHA-256.
The local *file* hash differed only because `retrieved_utc` differs — which is exactly why the
three hashes are kept as separate concepts.

## 5. Class 0 treatment

The official source name for class 0 is the **empty string `""`**, and it is preserved verbatim.
It was **not** renamed to `"background"`. The role field carries the semantics
(`background_or_non_disease`), and any display label such as "background" must be derived at
presentation time — never substituted into the provenance artifact.

This is also an implementation trap that was guarded explicitly: `""` is falsy in Python, so every
validation uses identity and length tests rather than truthiness.

## 6. Secondary local cross-check — official source decides

Read-only audit using **`Metadata.csv` + `annotation_train.json` + `annotation_val.json` only**.
**The test split was never opened.** No dataset, annotation, metadata, or audit file was modified.

Recall from B16-14 that `Metadata.csv["Index"]` is **wrong for 12 of 7,774 rows**, which
contaminates exactly six `category_id`s when names are joined locally. Mask value = `category_id + 1`.

### The six contaminated IDs

| cid | mask | **Official (authority)** | Local majority | Minority observed | Status |
|---|---|---|---|---|---|
| 21 | 22 | `blueberry rust` | blueberry rust (258) | blueberry mummy berry ×1 | **AGREE** |
| 24 | 25 | `broccoli downy mildew` | broccoli downy mildew (211) | broccoli alternaria leaf spot ×45 | **AGREE** |
| 48 | 49 | `cucumber angular leaf spot` | cucumber angular leaf spot (1096) | cucumber powdery mildew ×5 | **AGREE** |
| 53 | 54 | `eggplant phytophthora blight` | eggplant phytophthora blight (34) | eggplant phomopsis fruit rot ×1 | **AGREE** |
| 57 | 58 | `ginger sheath blight` | ginger sheath blight (277) | ginger leaf spot ×13 | **AGREE** |
| 81 | 82 | `rice blast` | rice blast (550) | bean mosaic virus ×2 | **AGREE** |

(Counts are annotation polygons, not images — one image contributes many annotations.)

### Uncontaminated entries

**109 / 109 agree** with the official source; **0 disagreements**.

### Overall

Majority vote would have reproduced the official mapping for **115/115** category_ids. That is
reassuring corroboration — **and explicitly not the basis of this artifact.** Had they disagreed,
the official source would still have won and the disagreement would have been reported, never
repaired. This is precisely why the derivation was rejected as a provenance source in the A2b
preflight: agreement was not knowable in advance.

## 7. Hash results — three distinct concepts

| Concept | Value | What it hashes |
|---|---|---|
| **Raw upstream-source SHA-256** | `baffa8860cfba380674dedc3eef2ac9d0bf5b29d5041be4c784a3fa898e5940c` | the exact bytes returned from the immutable raw URL |
| **Local vendored-file SHA-256** | `56fc18f513153bdd8e4206c224dda8abe349a092edb093fd0fcdfe731d57376d` | the final raw bytes of `configs/plantseg_class_map.json` — **reported here, not embedded** |
| **Canonical semantic class-map SHA-256** | `d14182423b6f176f940cada979adb364701fe186be091655ce38a80ce791a729` | canonical JSON of the ordered `entries` list only |

The semantic hash uses UTF-8, sorted object keys, compact separators, `allow_nan=False`, list order
preserved — byte-for-byte the same rule as
`src.eval.artifacts.canonical_json_bytes(list(class_map))`, so A2b's `class_map_sha256` will equal
`d14182…a729` when it passes `entries` straight through. The generator and the offline validator
computed it **independently** and agreed.

## 8. Confirmations

- ✅ **Official source, not majority vote, determined the mapping.**
- ✅ **No PlantSeg image, mask, validation file, or test file was opened.** The cross-check read
  only `Metadata.csv`, `annotation_train.json`, and `annotation_val.json`.
- ✅ No dataset download, checkpoint download, package install, training, inference, or GPU use.
- ✅ No temporary upstream Python source remains.
- ✅ The generator performs **no network access on import** and refuses to run without an explicit
  mode flag (verified: bare invocation exits 2 with no network attempted).

## 9. Files created

| File | Role |
|---|---|
| `configs/plantseg_class_map.json` | the vendored authoritative map + provenance |
| `scripts/vendor_plantseg_class_map.py` | maintenance-only generator; `--resolve-main` (network) / `--from-file` (offline) |
| `scripts/smoke_plantseg_class_map.py` | stdlib-only, fully offline validator |
| `reports/a2b0_plantseg_class_map_provenance.md` | this report |

The generator is a **maintenance utility only**: never imported by evaluator runtime code, never
run automatically during evaluation, no network on import, and no modification of the class map
unless invoked explicitly.

The 116 names are not reproduced here — they live in the JSON, which is the single source.

## 10. Regression results

| Suite | Required | Observed |
|---|---|---|
| `scripts/smoke_plantseg_class_map.py` | all PASS | **21/21 PASS**, exit 0 |
| `scripts/smoke_metrics_contract.py` (A1b) | 21/21 | **21/21 PASS**, exit 0 |
| `scripts/smoke_eval_contract.py` (A2a) | 63/63 | **63/63 PASS**, exit 0 |

Neither prior smoke script was modified.

---

_Only the four approved files were created. `src/**`, existing `configs/*.py`, `docs/**`, prior
reports, prior smoke scripts, dataset and annotation files, `CLAUDE.md`, `AGENTS.md`, and
`docs/reference/**` are untouched. Nothing staged or committed._
