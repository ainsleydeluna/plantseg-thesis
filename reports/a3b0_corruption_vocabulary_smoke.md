# A3b-0 — Canonical corruption vocabulary frozen; official mIoU-C blocker cleared

**RESULT: ✅ A3b-0 COMPLETE — CORRUPTION VOCABULARY FROZEN** · `31/31 checks passed`, exit code **0**

> Freezes **vocabulary and severity roles only**. The corruption **implementation** — its bytes,
> parameters, the `np.float_`→`np.float64` patch, seed policy and cache protocol — is **not**
> vendored here. Corruption cache generation and corrupted evaluation remain **unimplemented**.

_Generated 2026-07-28 · offline · repository-local only._

---

## 1. The A3a blocker

A3a found that Chapter III, `IMPLEMENTATION_CONTRACT.md` §(f) and `STATISTICAL_ANALYSIS_CONTRACT.md`
§2 all named the five corruptions in **prose only**, while `EVALUATION_CONTRACT.md` typed
`condition.name` as an unconstrained `"string | null"`. Nothing disambiguated `motion_blur` vs
`motion-blur` vs `motion blur`, or `brightness` vs `brightness_variation`.

`src/stats/robustness.py` therefore refused to invent slugs and required an injected
`CorruptionGrid`. Correct — but it left **official mIoU-C blocked**.

## 2. Source and identifier decision

Chapter III specifies vendored functions from the `imagecorruptions` reference implementation, so
its **function names are adopted as the machine identifiers**. Machine identifier and display label
are kept strictly distinct:

| machine identifier (`condition.name`) | display label | reference function |
|---|---|---|
| `motion_blur` | motion blur | `motion_blur` |
| `gaussian_noise` | Gaussian noise | `gaussian_noise` |
| `jpeg_compression` | JPEG compression | `jpeg_compression` |
| **`brightness`** | brightness variation | `brightness` |
| `fog` | fog | `fog` |

Order follows Chapter III's presentation order and is the frozen `corruption_order`.

### Why `brightness`, not `brightness_variation`

The reference function is `brightness`. *"brightness variation"* is explanatory thesis prose that
distinguishes the corruption from the excluded **brightness augmentation**. Recording the prose as
an identifier would make evaluation artifacts un-joinable with the generator that produced them,
for a purely cosmetic gain. The display label is preserved in the protocol file, so nothing is
lost in reporting.

**Rejected outright:** `brightness_variation`, `motion-blur`, `Motion_Blur`, spaces, capitalisation
variants, filename-derived aliases, and any alias map. No official corruption artifact exists yet,
so **no migration compatibility is owed**.

> ⚠️ **Verification limit, recorded honestly.** `imagecorruptions` is **neither installed nor
> pinned in `requirements.lock`**, and A3b-0 permits no installs or downloads, so these five
> function names could **not** be checked against real reference bytes. They are frozen on the
> explicit `[project-decision]` grounded in ch3's choice of that implementation. **The future
> corruption-scaffold task must verify the vendored module actually exposes exactly these five
> names** and reconcile the missing `requirements.lock` pin.

## 3. Machine-readable protocol

`configs/corruption_protocol.json` — `schema_version = plantseg-corruptions/1.0.0`,
`identifier_basis = imagecorruptions_reference_function_name`, `condition_type = corruption`.

Contains `corruption_order` (5 IDs), the five `{id, display_name, reference_function}` entries, and
the three severity groups. **Severity roles:** `inferential_severities = [1,2,3]` ·
`descriptive_only_severities = [4]` · `excluded_severities = [5]`, mutually disjoint.

Deliberately absent: alias map, implementation source, random seed, corruption parameters.

Canonical vocabulary SHA-256 (ordered id/display/function triples):
`7aa33cc22845723e96e75a2a59027b210d812c30a5497ff61d2a3ac98828aa7b`

## 4. Stats-side loader

`src/stats/corruption_protocol.py` — `load_corruption_protocol(path) -> CorruptionProtocol` and
`build_official_corruption_grid(protocol) -> CorruptionGrid`, plus a convenience
`official_corruption_grid()`.

- **No file access on import** — verified by instrumenting `builtins.open` across
  `import src.stats`: zero protocol reads.
- Strict JSON only when called; returns immutable data (`MappingProxyType` for display names).
- Validates schema version, identifier basis, condition type, exact ordered ID sequence, entry
  count, uniqueness, lowercase snake_case, non-empty display names, `reference_function == id`,
  `corruptions[]` order matching `corruption_order`, ID-set equality, the three severity groups,
  their disjointness, agreement with `robustness.INFERENTIAL_SEVERITIES`, and absence of an alias
  map.
- Builds the grid by **delegating to `CorruptionGrid`'s own validation** rather than duplicating it.

**`src/stats/robustness.py` was NOT modified.** It still requires an injected grid and holds no
vocabulary; the loader is the single bridge, so the identifiers are asserted in exactly one place.
Verified by the smoke: no canonical ID appears in `robustness.py` outside its explanatory docstring.

## 5. Smoke results — 31/31

| Area | Result |
|---|---|
| strict JSON, schema version, five-ID order, uniqueness, snake_case | ✅ |
| exact display labels, `reference_function == id` for all five | ✅ |
| severities `[1,2,3]` / `[4]` / `[5]`, disjoint, agreeing with the mIoU-C rule | ✅ |
| immutable protocol + vocabulary hash | ✅ |
| official `CorruptionGrid` built from the protocol; **exactly 15 inferential cells** | ✅ |
| `brightness` accepted as the official ID with display "brightness variation" | ✅ |
| **`brightness_variation` rejected** · **`motion-blur` rejected** · **`Motion_Blur` rejected** | ✅ |
| reordered protocol · missing corruption · unexpected corruption (`snow`) rejected | ✅ |
| altered display/function pairing rejected · altered severity group rejected · alias map rejected | ✅ |
| severity 4 not inferential · severity 5 excluded entirely | ✅ |
| explicit synthetic grid still usable **only** through the injected nonofficial path | ✅ |
| `import src.stats` performs no protocol file read | ✅ |
| the real protocol JSON was never modified (rejection tests used temporary copies) | ✅ |

## 6. Regressions

| Suite | Required | Observed |
|---|---|---|
| A1b metric contract | 21/21 | **21/21**, exit 0 |
| A2a evaluator core | 63/63 | **63/63**, exit 0 |
| A2b-0 class map | 21/21 | **21/21**, exit 0 |
| A2b real validation path | 53/53 | **53/53**, exit 0 |
| A3a paired inference | 45/45 | **45/45**, exit 0 |
| A3b-0 corruption protocol | all PASS | **31/31**, exit 0 |

No prior smoke script was modified. A3a keeps its explicit synthetic `CorruptionGrid` fixture —
that historical test was **not** rewritten to use the official grid, since no production regression
required it and the synthetic path is exactly what it is meant to exercise.

## 7. Status

- ✅ **Official mIoU-C vocabulary ambiguity is CLOSED.** Official corruption analysis now has one
  machine-readable source of truth shared by generation, evaluation metadata, mIoU-C assembly,
  artifact naming, future cache manifests and statistics.
- ❌ **Corruption cache generation and corrupted evaluation are NOT implemented.** No corrupted
  image has been produced, no cache exists, and no implementation bytes are vendored.
- ⚠️ **`imagecorruptions` is not installed or pinned** — the five function names are frozen by
  decision, not by inspection of real bytes. Verification is owed by the corruption-scaffold task.
- No evaluation metric rule and no statistical formula changed.

---

_Created: `configs/corruption_protocol.json`, `src/stats/corruption_protocol.py`,
`scripts/smoke_corruption_protocol.py`, this report. Modified: `src/stats/__init__.py` (exports
only) and four governing documents. `src/stats/robustness.py` and `src/eval/**` untouched. Nothing
staged or committed._
