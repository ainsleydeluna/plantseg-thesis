# Lane LREP — TEST reporting: lane report

## Part 1 (B-1 to B-3)

Session LREP-B (cloud), under the orchestrator's GO of Fri 16 Oct 2026 (Manila) for B-0 to B-3 of the
Phase A plan (lrep_phase_a_plan_20261015.md, 62,237 B, sha256 c4482a83…) as rulings LRB1–LRB22 amend
it. IF§11 v2 = tl_interface_lrep_20261016.md (8,607 B, 7bb8697a…). Labels: MEASURED, DOCUMENTED,
REPOSITORY-PROVEN, INFERRED. Every file:line below was re-read at the commit it names.

Plan citations: the session holds a copy of the plan (58,953 B, sha256 d1318a21…), not the plan of
record (62,237 B, c4482a83…, not in this container). The DL-24 agents cited the copy's lines; this
report cites sections, not plan lines.

### 1. Base, commits and the scoped diff

- O = 2b3b55651e8092fd7fff6e786b73d971096deb17 (origin/master; lane Q2's merge). lane/lqp and
  lane/l-corr are not merged in O. lane/lrep was created at O (MEASURED); no merge.
- Commits on lane/lrep (no trailers; each by explicit path with `git commit --only`):

| | commit | subject |
|---|---|---|
| C1 | 1088988b4dcb0a14ae5d6b710793459aba93913e | LREP C1: TEST-day input layer and synthetic fixtures (B-1) |
| C2 | a6ea6e0f1bbe9eddffd228571047904e704ae6df | LREP C2: L-AM8 seed stability and the per-seed tables (B-1) |
| C3 | 341b4624f26d8105c25ef923d9611d216c684649 | LREP C3: rare-class strata TEST tables (B-2) |
| C4 | 69a079aee1c54ef0798ebc08369c77b8e98bd0fc | LREP C4: default candidates on TEST (B-2) |
| C5 | 8d3aca0abe557101d727847f3eb3f3f0ef21760f | LREP C5: L-AM6 GT-present on TEST for every model of record (B-3) |
| C6 | d8487dd542c81130241d76fd45889f64faecfa32 | LREP C6: the extension report and its command line (B-3) |
| C7 | d6cb71ad4d98263941c36f2f4c278adcf3636ba9 | LREP C7: the mutation harness (B-1 to B-3; LRB8) |
| F1 | ed3631f8dabb6a48a7e158dd95cad060bdecbcfe | LREP F1: DL-24 fixes in the input layer and the fixture world |
| F2 | 14da8ea0ff1ade19b8b8c3f4888caed776a7f7bd | LREP F2: DL-24 fixes in seed stability and its smoke |
| F3 | 29bdd6e37404eb04ea893d9b7f4a18bb8692d866 | LREP F3: DL-24 fixes in the strata, default-candidate and AM-6 tables |
| F4 | f5e73e0a606bbab643398db0c4477f2a6c09c12d | LREP F4: DL-24 fixes in the extension report and its command line |

  F4 is the last code commit; this report is the next commit and changes only this file.

- `git diff --stat <O> -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference'
  ':(exclude,icase)*test*'` at F4 (MEASURED 26 Oct 10:10 UTC; the report commit adds this file):

```
 scripts/lrep_fixtures.py            |  585 ++++++++++++++++
 scripts/report_extensions.py        |   94 +++
 scripts/smoke_am6_tables.py         |  350 +++++++++
 scripts/smoke_default_candidates.py |  345 +++++++++
 scripts/smoke_lrep_mutations.py     |  493 +++++++++++++
 scripts/smoke_report_extensions.py  |  548 +++++++++++++++
 scripts/smoke_seed_stability.py     | 1325 +++++++++++++++++++++++++++++++++++
 scripts/smoke_strata_tables.py      |  452 ++++++++++++
 src/stats/am6_tables.py             |  173 +++++
 src/stats/default_candidates.py     |  241 +++++++
 src/stats/heldout_inputs.py         | 1126 +++++++++++++++++++++++++++++
 src/stats/report_extensions.py      |  380 ++++++++++
 src/stats/seed_stability.py         |  569 +++++++++++++++
 src/stats/strata_tables.py          |  235 +++++++
 14 files changed, 6916 insertions(+)
```

- B-1 to B-3 add files only, all of them on the GO's list; no path contains "test" or ":". `git diff
  <O>` is empty for src/stats/report.py, rehearsal.py, driver.py, artifact.py, ingest.py, bootstrap.py,
  eligibility.py, perclass.py, scripts/eligibility_variants.py and src/eval/metrics.py (MEASURED at F4,
  26 Oct 10:10 UTC).
- Pushes: two, both of lane/lrep only: after C2 (tip a6ea6e0f…), and STEP 9's, which pushes this
  report's commit.

### 2. The venv against the lock

$SCR/lrepvenv, /usr/bin/python3.11 -m venv (Python 3.11.17), installed as STEP 3 and ruling
LREPB-ENV-1 (Sat 17 Oct) say. `pip list --format=freeze` (MEASURED 26 Oct): 32 packages; 28 equal
requirements.lock's pins; torch 2.1.0+cpu and torchvision 0.16.0+cpu differ from the lock's +cu121 in
build only; pip 24.0 and setuptools 79.0.1 are not in the lock. `pip check` (17 Oct): no broken
requirements.
PLANTSEG_DATA_ROOT unset; PYTHONDONTWRITEBYTECODE=1 and `python -B` for every run.

At the lock's pins (28): certifi 2026.6.17, charset-normalizer 3.4.7, filelock 3.14.0, fsspec 2026.4.0,
idna 3.18, ImageIO 2.37.3, Jinja2 3.1.6, lazy-loader 0.5, MarkupSafe 3.0.3, mpmath 1.3.0, networkx
3.6.1, numpy 1.26.4, packaging 24.2, pandas 3.0.3, patsy 1.0.2, pillow 12.3.0, python-dateutil
2.9.0.post0, pytz 2023.4, requests 2.28.2, scikit-image 0.23.2, scipy 1.11.4, six 1.17.0, statsmodels
0.14.6, sympy 1.14.0, tifffile 2026.3.3, typing_extensions 4.15.0, tzdata 2026.2, urllib3 1.26.20.
Build only: torch 2.1.0+cpu, torchvision 0.16.0+cpu (lock +cu121). Not in the lock: pip 24.0,
setuptools 79.0.1.

### 3. Baseline and final RESULT lines

Each smoke ran alone with the venv's python and `-B`: STEP 4 at O (17 Oct, MEASURED), STEP 7 at F4
f5e73e0 (26 Oct, MEASURED; logs in $SCR/baseline and $SCR/final). The four EXEMPT-1 smokes ran with
TMPDIR=$SCR, and smoke_mde with TMPDIR=$SCR (it aborts without TMPDIR). No other smoke, Python or git
process ran during any STEP 7 run; beside the long background runs only log peeks and text reads ran
(cat, tail, grep, sed, awk, wc). STEP 7 ran 09:48–10:35 UTC for the STEP 4 smokes; the session's
worker then restarted, and after the orchestrator's "Try again" the state was re-checked (12:18 UTC:
HEAD f5e73e0, refs/heads/lane/lrep, scoped status empty, no Python process, venv intact) before the
lane smokes ran (12:18–12:21 UTC).

STEP 4's lines are as the lane state recorded them; STEP 7's are as printed.

| smoke | GO's expected | STEP 4 at O | STEP 7 at F4 |
|---|---|---|---|
| smoke_frozen_blobs | 10/10 | FROZEN BLOBS OK (10/10) | RESULT: FROZEN BLOBS OK (10/10) |
| smoke_stats_driver | 55/55 | STATS DRIVER CORE OK (55/55) | RESULT: STATS DRIVER CORE OK (55/55) |
| smoke_stats_bootstrap | 155/155 | SUMMARY 155/155; A3b SMOKE OK | SUMMARY 155/155 checks passed; RESULT: A3b SMOKE OK |
| smoke_stats_paired | 45/45 | 45/45; A3a OK | SUMMARY 45/45 checks passed; RESULT: A3a OK -- PAIRED INFERENCE CORE PROVEN |
| smoke_eligibility_variants | — | L-AM17-GTPRESENT OK (34/34) | SUMMARY 34/34 checks passed; RESULT: L-AM17-GTPRESENT OK (34/34) |
| smoke_teacher_d4 | — | PASS (32/32) | RESULT: PASS (32/32) |
| smoke_eval_contract | 63/63 | 63/63; A2a OK | SUMMARY 63/63 checks passed; RESULT: A2a OK -- CORE CONTRACT PROVEN |
| smoke_metrics_contract | — | OK; PASS=21 (total 21) | RESULT: OK -- every case matched its declared state; PASS=21 (total 21) |
| smoke_mde | — | PASS 129/129 (mutations killed 35/35), 620 s | PASS 129/129 (mutations killed 35/35); 648 s |
| smoke_am5_exclusion (EXEMPT-1) | 50/50 | L-AM5 OK (50/50) | RESULT: L-AM5 OK (50/50) |
| smoke_stats_rehearsal (EXEMPT-1) | 41/41 | STATS REHEARSAL OK (41/41) | RESULT: STATS REHEARSAL OK (41/41) |
| smoke_stats_report (EXEMPT-1) | 63/63 | STATS REPORT OK (63/63) | RESULT: STATS REPORT OK (63/63) |
| smoke_stats_official_fullsize (EXEMPT-1) | 37/37 | STATS OFFICIAL FULL-SIZE OK (37/37) (re-run; slip 4) | RESULT: STATS OFFICIAL FULL-SIZE OK (37/37; pinned stack, (a) run); 10:10–10:35 UTC, scoped status empty before and after |
| smoke_stats_integrity | — | not run (fixture rule) | not run |
| smoke_perclass_gap_table | — | not run (fixture rule) | not run |
| smoke_gap_bootstrap_val | — | not run (fixture rule) | not run |
| smoke_train_strata | — | not run (fixture rule) | not run |

STEP 4 reproduced every expected count. Every STEP 4 count is unchanged at STEP 7. (The
eligibility log also holds a "RESULT: ERROR -- see above" line: it is the output of the CLI that its
case E4-9 runs and expects to refuse, printed before the smoke's own SUMMARY.)

**The new smokes** (STEP 7 at F4; the plan's §7.1 targets with LRB19 L-09's corrections):

| smoke | §7.1 target | STEP 7 RESULT |
|---|---|---|
| smoke_seed_stability | 66 → 67 (L-09) | RESULT: L-AM8 SEED STABILITY OK (109/109) |
| smoke_strata_tables | 32 | RESULT: STRATA TABLES OK (37/37) |
| smoke_default_candidates | 22 | RESULT: DEFAULT CANDIDATES OK (24/24) |
| smoke_am6_tables | 26 | RESULT: L-AM6 TABLES OK (24/24) |
| smoke_report_extensions | 24 → 23 (L-09) | RESULT: REPORT EXTENSIONS OK (41/41) |
| smoke_lrep_mutations | 40 (20 mutations) | RESULT: LREP MUTATIONS OK (213/213; mutations killed 110/110; refusal codes covered 41/41) |
| total | 210 | 448 |

Every change from §7.1, with its cases (LRB21):
- **smoke_seed_stability 67 → 109.** F 14 → 13: the three hand totals are one case (F1) and g, d one
  case (F6) (−3); added F5 (mC equals report.dataset_miou_c bitwise) and F13 (the seed-42 robustness
  effect equals report's contrast). S 20 → 21: S21, the D-6 reason names its entry's DL rows (round 1,
  R2r1-10). L 9 → 11: L10 not made and L11 a QAT divergence by epoch (LRB5, IF§11.2). W, new, 10:
  W1 the complete world's ten labelled rows (LRB13), W2 robustness at seeds 42, 43, 44 (LRB12), W3
  ignored slots (LRB4), W4, W5, W8 cut-run curves and comparators (LRB6), W6 E3 diverged, W7 E2's
  divergence in the sentence (LRB15), W9 paired cut runs (DL-24 R2-03), W10 no shared point (R2-02).
  R 18 → 46 (+28): R-2's one case became one per code (R2b–R2e) with the IF§11.5–11.6 codes (R2a
  stub_reader_official, R2f binding_unbound, R2g freeze_mismatch) and R2h, R2i, R2j (R1-04, round 1,
  R1-08) (+9); R3c unplanned_slot (LRB4) (+1); R4c, R4d (round 1, R1-14) (+2); R7b, R7c (LRB3's
  fields) (+2); R10b e1_checkpoint_record, its own code (+1); R12b–R12e (LRB5; R1-05, R2-01) (+4);
  R13b (round 1) (+1); R14b (R1-05) (+1); R16, R17, R17b (LRB2) (+3); R18, R18b (LRB7; R1-06) (+2);
  R19 path_unsafe (IF§2) (+1); R20 robustness_cells_missing (LRB11) (+1). X 4 → 6: X5 the path rule
  (IF§2) and X6 no slot key in the markdown (round 1, R2r1-14). D unchanged.
- **smoke_strata_tables 32 → 37.** Dropped: "a non-canvas artifact" and "an INT8 model with no model":
  the part reads only the 37 seed-42 inputs of the verified statistics artifact (D-9), never a binding
  record, and every seed-42 run of record has a model (R-13); R-8's case is seed R8. The
  plan's "a seed-43 artifact" is R5: with the 37 as inputs a wrong seed shows as an identity that is
  not the family slot's (seed42_mismatch). Added: E2 (IoU from summary.per_class exactly), E3
  (not-evaluable classes are the world's zero-gt classes), E4 (the strata file's record, terciles
  38/38/39, 28 rare), E5 (the committed DL-63 file is of record), E11 (the appendix; DL-24 R3-05), R2
  (strata_integrity's case) and R6 (summary.per_class against the npz; R3-02).
- **smoke_default_candidates 22 → 24.** Added: R3 (default_candidate_missing's case) and V14 (the
  candidate's GT-present fields follow the GT-present rule; DL-24 R4-01).
- **smoke_am6_tables 26 → 24.** Moved to B-5 with the AM-18 item 8(c) subset (LRB17): the subset sha
  and the subset id refusals, the four subset values and "eligibility changes recorded" (−7). Dropped:
  "a smoke-status artifact in the official profile" (no official statistics artifact can exist while
  the binding is unbound; the row rule is R-9's, seed R9). Added: R2 (R-10 over the rows read), R4
  (artifact_missing; LRB7's binding-driven list), R5 (DL-99's E1 record; R1-13), T3 (21 rows, the
  37's seven and the binding's fourteen; LRB7), T4 (int8_qat rows only, never fq_off; LRB7), T6 (a
  run without a model is listed with its label; LRB5).
- **smoke_report_extensions 23 → 41.** Exit 0 is checked inside W1 (−1). "The rehearsal documents are
  unchanged" is replaced by N2 (no LREP file imports rehearsal.py or assigns a frozen module's
  attribute; R4-08), with `git diff <O>` empty for rehearsal.py. The "four parts" are six: C5 fqoff
  (LRB7), C6 mde (LRB18). Added: Q1–Q4 (the fq_off table; LRB7), Q5–Q7 (an fq_off row without its
  record, of another seed, the 37's identities; R1r1-09, R3-01), M1 (the MDE label and shares; LRB18),
  W4 (the markdown renders the JSON), P4 (R3-08), P5, P6 (R3-09), F4 (R3-15), H8 (R3-07), and R1–R3
  (the cases of parts_invalid, mde_entry_invalid and usage_invalid).
- **smoke_lrep_mutations 40 → 213.** The harness counts one K line per killing case (unmutated pass,
  102), one line per mutation (mutated fail, 110) and X1 (every code covered). 19 of the plan's 20
  mutations are S01–S03, S05–S13, S18 and S24–S29; "the subset left without recomputed eligibility"
  moved to B-5 (LRB17). Added: C01–C41 (one per refusal code; GO STEP 5), R01–R12 (raise sites the
  DL-24 fixes added), S04 (§1.2's fsum), S14–S17 and S19–S23 (LRB4–LRB7, LRB11, D-6) and S30–S57
  (the DL-24 fixes).

### 4. Rulings LRB1–LRB22: where each is met (file:line at the final commit)

Lines are at F4 (f5e73e0); the report commit changes only this file, so they hold at the final commit.
H = src/stats/heldout_inputs.py, SS = src/stats/seed_stability.py, ST = src/stats/strata_tables.py,
DC = src/stats/default_candidates.py, AM6 = src/stats/am6_tables.py, RX = src/stats/report_extensions.py,
LF = scripts/lrep_fixtures.py.

- **LRB1.** B-1 to B-3 code against IF§11 v2 with fixtures and a stub reader: the stub
  `read_inventory` H:307-347 (its marker H:350), refused under Policy.OFFICIAL before anything is read
  (H:402-405, `stub_reader_official`). B-4 (the swap and a conformance smoke) is not in this part. The
  records in §9 read "IMPLEMENTED before D (AM-22 v6 item 1(a) condition 4)".
- **LRB2.** `read_telemetry` H:1061-1110: one read, the sha256 compared before parsing (H:1082-1088,
  `telemetry_sha256_mismatch`), only `event == "val"` rows kept (H:1101). The attempt of record is
  picked by the row's training_run_id (SS:388-396). TMAN supplies no score.
- **LRB3.** One record per (row, scope, cell): `binding_record` H:783-808 (a second record is
  `binding_invalid`, H:789-791; the fields H:794-804). `load_bound` H:846-899 refuses (R-7,
  `artifact_record_mismatch`) a mismatch in the four files (H:867-870), the protocol (H:873-876), and
  the checkpoint, the commit and the split manifest, and the row's identity (H:888-897).
- **LRB4.** `record_slots` H:464-490: only `family` (42) and `per_seed` (43/44) slots are cells
  (RECORD_ROLES H:70); every other slot is skipped and never refused (H:469-471); `unplanned_slot`
  fires only for a family/per_seed slot outside the 21 (H:477-481).
- **LRB5.** Kinds and roles verbatim (H:51-54); IF§11.2's labels H:88-91 and `_label` H:524-537;
  `resolve_status` H:599-652 (a QAT selection reads its QAT run's kind, H:616-630; a PTQ model with
  kind null its parent's cut or divergence, H:631-645); R-5 `superseded` H:550-552; R-12 through the
  QAT run in `check_parentage` H:664-692 (H:674-679, H:682-684). R-10's distinct run_id is the
  summary `run.run_id` (H:949-952); training_run_id is never required distinct (it only picks the
  telemetry attempt, SS:390-391, and ties an fq_off row to its int8_qat row, RX:89).
- **LRB6.** `cut_curves` SS:399-449: a cut KD or E1 run's comparators are its planned-cell partners at
  that seed, matched by iteration (SS:429-431); a cut QAT run's is its stage's seed-42 run, matched by
  epoch (SS:426-428); the cut entry's comparator is always shown and labelled (SS:435-448); a derived
  artifact has no curve (SS:412-414, SS:475-477); no interpolation (SS:465-468, SS:480-488).
- **LRB7.** (model_role, precision) = (student, the row's precision) for fq_off rows, else
  driver.STAGE_PRECISION (H:882-887). am6 reads its stage list from the binding's `artifacts`
  (AM6:109-110). The fq_off table: RX:84-139 (the row tied to its slot's int8_qat row, RX:84-93; E6-KD,
  RX:103-104; "not available: no fq_off artifact", RX:59 and RX:124-126); quant_descriptive slots are
  not read (RX:101-104). Confirmed at C1 with one fixture (MEASURED 17 Oct, a scratch script):
  driver.load_input passed runs/heldout/e5_s43_fqoff (qat_fakequant_off, student, E5) unchanged, so
  no own reader was needed; the check is kept as report Q2 (scripts/smoke_report_extensions.py:269-275,
  case :494). **Deviation:** the fq_off fixtures are not written as plain files; they go through the
  writer as int8_qat, then `run.precision` is rewritten and MANIFEST.sha256 rehashed with
  stats_fixtures.edit_summary (LF:24-26, LF:211). The writer never receives `qat_fakequant_off`; the
  result is the same artifact layout. The plan's §4 inventory gains E6-KD's TEST score and AM-23's
  quantities (records, §9).
- **LRB8.** Every mutation names its killer (scripts/smoke_lrep_mutations.py:158-421, the
  scripts/smoke_teacher_diag_mutations.py pattern); the killer must pass unmutated (K lines, :464-469)
  and fail mutated (:470-478), and X1 requires every code covered (:456-463). §5 lists them.
- **LRB9.** E-66 is dropped (§9). The non-finite label cites AM-21 item 2(b) (H:89); DL-a records D-6.
- **LRB10.** `gt_present_row` and `am6_rows` (AM6:41-92) take any artifact's statistics; the VAL
  invocation is not part of this lane's code (AM6:24-25).
- **LRB11.** `robustness_cells_missing` fires only when E1 and E6 both have a model at that seed and a
  cell is absent (SS:243-250); otherwise the robustness cell carries the pair's status (SS:311-316 with
  `_cell` SS:269-273).
- **LRB12.** ROBUSTNESS_SEEDS = (42, 43, 44) (SS:58-59); the robustness row at every seed
  (SS:309-316); mC SS:82-89.
- **LRB13.** Ten rows in driver.COMPARISON_TABLE order, labelled family, descriptive or
  non-inferiority (SS:60-62, SS:309-316).
- **LRB14.** m(S,k) from the float64 pooled totals (SS:77-79); DL-a states it (§9).
- **LRB15.** D-6: the labels (H:88-91, H:524-537) and the reason that names the deciding entry's DL
  rows (SS:154-165, SS:334-336). D-7: the fixed reason order and "a missing run never gives stable"
  (SS:180-207).
- **LRB16.** D-8: the zero-IoU list for the four direct conversions only (ST:45-46, ST:166-171). D-9:
  seed 42 only (ST:180, ST:185). D-10: descriptive values only, no p-value or interval (DC:18-23,
  DC:136-153, DC:61-62).
- **LRB17.** B-5 is STEP 10's note ($SCR/lrep_b5_note.md); no B-5 code here. The PA:1297 erratum is in
  §9.
- **LRB18.** A separate document (RX:1-6; scripts/report_extensions.py); `git diff <O>` is empty for
  report.py. The MDE label and each pair's share_ties and n_zero: `build_mde` RX:157-182, `render_mde`
  RX:185-197 (report M1).
- **LRB19.** L-09: the corrected counts are §3's starting targets. L-12: H:1011 ("an identity and
  pairing check"). L-13: R-15 is an assertion (SS:93-97; seed R15,
  scripts/smoke_seed_stability.py:881-886). L-14: X1 allows only engineering_shifts and hodges_lehmann
  (scripts/smoke_seed_stability.py:1051, :1069; DC:139). L-15, L-16,
  L-22: corrections to the plan's §5 and Annex R, taken as ruled; no code. L-17: the non-finite label
  (H:89) at seeds 43/44; at seed 42 R-1 fires first (H:991-1006), and R-13 (H:762-766). L-18: "planned
  cells" (H:65-67). L-19: §10. L-24: no action.
- **LRB20.** LF:28-31: rel_name is never used, split "val", no created path contains "test"
  (`make_root` LF:87); every lane smoke checks it (seed X4, strata E10, defaults D5, am6 D4, report
  H7). The four EXEMPT-1 smokes ran only in STEP 4 and STEP 7, each with TMPDIR=$SCR.
- **LRB21.** §3 names the final counts and the cases behind each change.
- **LRB22.** B-5 (STEP 10(c)). B-1 to B-3 never read a quant_descriptive slot: record_slots skips it
  (H:469-471) and the fq_off table reads family/per_seed E5/E6 and E6-KD only (RX:101-104); the
  fixture carries one such slot to prove it (LF:489).

### 5. Refusal codes and mutations

Every refusal is an `LrepRefusal` with a code of CODES (H:106-165, 41 codes), raised at a site that
reads `raise refusal("<code>", …)` or `raise H.refusal("<code>", …)`; the CLI exits 2 on any of them
and writes nothing. strict_json_text's three raises carry their caller's code (H:203-218). Each code
has a case that reaches it with every earlier gate passing, and a C mutation that a named case kills;
X1 checks that the C lines cover all 41 codes. The first table gives every raise site at F4 (the line
of the `refusal(` call, read from the source by AST; abbreviations as in §4, CLI =
scripts/report_extensions.py). The second is the mutation list (scripts/smoke_lrep_mutations.py
MUTATIONS): each mutation rewrites one function from its own source (an exact, unique text
replacement), and its killing case must pass unmutated and fail mutated. C: one per code, its raise
site disabled (`raise refusal(` → `if False: raise refusal(`); R: further raise sites the DL-24 fixes
added; S: semantic mutations. "mut:" cases are the harness's own in-process cases. At STEP 7 every
mutation was killed by its named case (110/110).

| code | rule (CODES) | raise sites |
|---|---|---|
| `stats_artifact_unverified` | R-1: the statistics artifact does not verify with its 37 inputs | H:1004 (load_stats_artifact) |
| `stub_reader_official` | IF§11.6: the stub reader is refused under Policy.OFFICIAL | H:404 (open_inventory) |
| `inventory_unreadable` | R-2: the export cannot be read as strict JSON | H:317 (read_inventory); H:324 (read_inventory) |
| `inventory_sha256_mismatch` | R-2: the export's sha256 is not the expected one | H:320 (read_inventory) |
| `inventory_invalid` | R-2: the export lacks, repeats or mistypes a field LREP reads | H:328 (read_inventory); H:378 (entry); H:411 (open_inventory); H:415 (open_inventory); H:702 (model_row); H:725 (model_row) |
| `binding_invalid` | R-2: the binding file is missing, differs from binding.sha256 or is malformed, or a bound export carries no binding.artifacts | H:345 (read_inventory); H:425 (open_inventory); H:429 (open_inventory); H:433 (open_inventory); H:790 (binding_record); H:806 (binding_record); H:817 (cell_condition) |
| `binding_unbound` | IF§11.5: under Policy.OFFICIAL the binding must be bound | H:417 (open_inventory) |
| `freeze_mismatch` | IF§11.5: freeze.manifest_sha256 is not manifest_sha256 (or no freeze, OFFICIAL) | H:422 (open_inventory) |
| `planned_slot_missing` | R-3: a planned (stage, seed) has no family/per_seed slot | H:488 (record_slots) |
| `planned_slot_duplicate` | R-3: two family/per_seed slots claim one planned (stage, seed) | H:485 (record_slots) |
| `unplanned_slot` | R-3/LRB4: a family/per_seed slot outside the 21 planned cells | H:480 (record_slots) |
| `status_invalid` | R-4: two entries for one planned slot, a kind outside the stage's kinds, or a missing or malformed record field | H:473 (record_slots); H:554 (_checked_kind); H:557 (_checked_kind); H:595 (_check_unit); H:612 (resolve_status); H:629 (resolve_status); H:644 (resolve_status); H:651 (resolve_status); SS:443 (cut_curves) |
| `superseded` | R-5: kind superseded is refused, never mapped | H:551 (_checked_kind) |
| `parent_rule` | R-12: a parentage or not-run inconsistency (AM-19 item 4(a); LRB5) | H:565 (_parent_of); H:588 (_not_run_status); H:620 (resolve_status); H:635 (resolve_status); H:677 (check_parentage); H:683 (check_parentage); H:687 (check_parentage); H:690 (check_parentage) |
| `seed42_not_complete` | R-13: a seed-42 run of record without a model | H:766 (check_seed42) |
| `seed_inconsistent` | R-14: a slot's model row or parent at another stage or seed, or its fq_off row not the model of its int8_qat row (LRB5) | H:582 (_not_run_status); H:638 (resolve_status); H:745 (check_rows); RX:91 (fqoff_row) |
| `artifact_missing` | R-6: a model of record without its model row or bound artifact | H:700 (model_row); H:787 (binding_record) |
| `artifact_for_missing_run` | R-6: a bound clean artifact for a planned run without a model | H:757 (check_missing_runs) |
| `artifact_record_mismatch` | R-7/LRB3: an artifact differs from its binding record or model row | H:869 (load_bound); H:874 (load_bound); H:896 (load_bound) |
| `not_canvas` | R-8: a binding record whose protocol is not canvas (an artifact whose protocol differs from its record is R-7; LRB3) | H:858 (load_bound) |
| `artifact_unreadable` | R-9: an artifact the driver refuses (files, manifest, the row rules) | H:866 (load_bound); H:880 (load_bound) |
| `precision_mismatch` | LRB7: (model_role, precision) of an artifact or a model row is not (student, the row's precision) | H:748 (check_rows); H:885 (load_bound) |
| `path_unsafe` | IF§2: a path that is not plain relative POSIX, or contains 'test' | H:237 (safe_relative_path) |
| `cross_artifact_mismatch` | R-10: manifest, class map, metric code, commit, AM-5, GT, run_id or checkpoint disagreement | DC:143 (pair_values); H:961 (cross_check) |
| `e1_checkpoint_record` | R-10: an E1 checkpoint differs from DL-99's record for its seed | H:970 (check_e1_record) |
| `seed42_mismatch` | R-11: a seed-42 value or identity differs from the statistics artifact's | DC:171 (record_row); H:1026 (check_seed42_identity); SS:296 (seed42_cross_checks) |
| `telemetry_sha256_mismatch` | LRB2: a telemetry file's sha256 differs from the export's | H:1087 (read_telemetry) |
| `telemetry_invalid` | LRB2: a telemetry item or file whose VAL rows cannot be read | H:1049 (_val_point); H:1055 (_val_point); H:1072 (read_telemetry); H:1078 (read_telemetry); H:1084 (read_telemetry); H:1093 (read_telemetry); H:1100 (read_telemetry); H:1105 (read_telemetry); SS:393 (_of_record_attempt) |
| `robustness_cells_missing` | LRB11: an E1/E6 pair with both models lacks one of its 15 cells | SS:248 (load_seed_inputs) |
| `strata_invalid` | §2.1: the strata file is unreadable or not one stratum per disease class | ST:67 (load_strata_checked) |
| `strata_not_of_record` | §2.1: official profile, strata file not committed at its DL-63 sha256 | ST:76 (load_strata_checked) |
| `strata_gt_mismatch` | §2.1: the seven models' TEST gt_support differ | ST:120 (model_checks) |
| `strata_integrity` | §2.1: a model's per-class block does not reproduce its summary bitwise | ST:87 (_per_class); ST:115 (model_checks); ST:144 (pair_table) |
| `default_candidate_missing` | §2.2: a default candidate has no slot, model or artifact | DC:99 (default_slots); DC:116 (default_slots) |
| `default_candidate_invalid` | §2.2: another candidate, or a default that is not complete | DC:87 (default_slots); DC:95 (default_slots); DC:114 (default_slots) |
| `selection_record_mismatch` | §2.2: a selected run's checkpoint is not DL-109's/DL-121's | DC:74 (check_selection_record) |
| `gt_present_unfaithful` | §3.3: R1 or R3 fails, or a seed-42 row differs from report.py's | AM6:90 (am6_rows); AM6:143 (build_am6_tables) |
| `out_dir_invalid` | D-14: the out-dir exists, or lies inside the repository or an input | RX:316 (check_out_dir); RX:319 (check_out_dir); RX:322 (check_out_dir); RX:324 (check_out_dir); RX:328 (check_out_dir); RX:337 (write_report); RX:347 (write_report) |
| `parts_invalid` | §3.3: --parts names an unknown or repeated part | RX:69 (parse_parts) |
| `mde_entry_invalid` | LRB18: the MDE entry cannot be read | RX:162 (build_mde); RX:168 (build_mde); RX:173 (build_mde) |
| `usage_invalid` | a bad command line (--repo-root without --synthetic-inputs, a missing input) | RX:246 (build_report); RX:361 (run); RX:363 (run); RX:372 (run) |

| id | mutation | target | killed by |
|---|---|---|---|
| C01 | stats_artifact_unverified not raised | H.load_stats_artifact | seed:R1 |
| C02 | stub_reader_official not raised | H.open_inventory | seed:R2a |
| C03 | inventory_unreadable not raised (unreadable file) | H.read_inventory | seed:R2b |
| C04 | inventory_sha256_mismatch not raised | H.read_inventory | seed:R2c |
| C05 | inventory_invalid not raised (reader and open_inventory) | H.read_inventory, H.open_inventory | seed:R2d |
| C06 | binding_invalid not raised (binding file) | H.read_inventory | seed:R2e |
| C07 | binding_unbound not raised | H.open_inventory | seed:R2f |
| C08 | freeze_mismatch not raised | H.open_inventory | seed:R2g |
| C09 | planned_slot_missing not raised | H.record_slots | seed:R3a |
| C10 | planned_slot_duplicate not raised | H.record_slots | seed:R3b |
| C11 | unplanned_slot not raised | H.record_slots | seed:R3c |
| C12 | status_invalid not raised (a missing record field) | H._checked_kind | seed:R4b |
| C13 | superseded not raised | H._checked_kind | seed:R5 |
| C14 | parent_rule not raised (not run while the parent is complete) | H._not_run_status | seed:R12 |
| C15 | seed42_not_complete not raised | H.check_seed42 | seed:R13 |
| C16 | seed_inconsistent not raised | H.check_rows | seed:R14 |
| C17 | artifact_missing not raised (no record) | H.binding_record | seed:R6a |
| C18 | artifact_for_missing_run not raised | H.check_missing_runs | seed:R6b |
| C19 | artifact_record_mismatch not raised (files) | H.load_bound | seed:R7a |
| C20 | not_canvas not raised (record protocol) | H.load_bound | seed:R8 |
| C21 | artifact_unreadable not raised (files) | H.load_bound | seed:R9 |
| C22 | precision_mismatch not raised | H.load_bound | seed:R18 |
| C23 | path_unsafe not raised | H.safe_relative_path | seed:R19 |
| C24 | cross_artifact_mismatch not raised | H.cross_check | seed:R10a |
| C25 | e1_checkpoint_record not raised | H.check_e1_record | seed:R10b |
| C26 | seed42_mismatch not raised (identity of the 37) | H.check_seed42_identity | seed:R11 |
| C27 | telemetry_sha256_mismatch not raised | H.read_telemetry | seed:R16 |
| C28 | telemetry_invalid not raised (no attempt of the run of record) | SS._of_record_attempt | seed:R17 |
| C29 | robustness_cells_missing not raised | SS.load_seed_inputs | seed:R20 |
| C30 | strata_invalid not raised | ST.load_strata_checked | strata:R4 |
| C31 | strata_not_of_record not raised | ST.load_strata_checked | strata:R3 |
| C32 | strata_gt_mismatch not raised | ST.model_checks | strata:R1 |
| C33 | strata_integrity not raised | ST.model_checks | strata:R2 |
| C34 | default_candidate_missing not raised (no slot) | DC.default_slots | defaults:R3 |
| C35 | default_candidate_invalid not raised (another candidate) | DC.default_slots | defaults:R1 |
| C36 | selection_record_mismatch not raised | DC.check_selection_record | defaults:R4 |
| C37 | gt_present_unfaithful not raised | AM6.am6_rows | am6:R1 |
| C38 | out_dir_invalid not raised (inside an input) | RX.check_out_dir | mut:out_dir |
| C39 | parts_invalid not raised | RX.parse_parts | report:R1 |
| C40 | mde_entry_invalid not raised (unreadable) | RX.build_mde | mut:mde |
| C41 | usage_invalid not raised (--repo-root without --synthetic-inputs) | RX.run | mut:usage |
| R01 | binding_invalid not raised (a bound export without binding.artifacts) | H.open_inventory | seed:R2h |
| R02 | status_invalid not raised (two entries for one planned slot) | H.record_slots | seed:R4c |
| R03 | status_invalid not raised (a divergence in the wrong unit) | H._check_unit | seed:R4d |
| R04 | parent_rule not raised (a selection whose parent is not its QAT run) | H.resolve_status | seed:R12c |
| R05 | parent_rule not raised (a PTQ model with kind null whose parent is another stage) | H.resolve_status | seed:R12e |
| R06 | seed_inconsistent not raised (a PTQ model with kind null whose parent is at another seed) | H.resolve_status | seed:R14b |
| R07 | parent_rule not raised (a descendant of a cut run that is not 'not run') | H.check_parentage | seed:R12d |
| R08 | parent_rule not raised (a model that descends from the wrong run) | H.check_parentage | seed:R12b |
| R09 | precision_mismatch not raised (a model row whose model_role is not student) | H.check_rows | seed:R18b |
| R10 | out_dir_invalid not raised by the out-dir check (a parent that does not exist) | RX.check_out_dir | report:P5 |
| R11 | seed_inconsistent not raised (an fq_off row that is not its int8_qat row's model) | RX.fqoff_row | report:Q6 |
| R12 | out_dir_invalid not raised by write_report (a parent that does not exist) | RX.write_report | report:P6 |
| S01 | condition 1: g > 0 becomes g >= 0 | SS.condition_1 | seed:S2 |
| S02 | condition 2: mean > max SD becomes >= | SS.condition_2 | seed:S7 |
| S03 | the SDs become population SDs (pstdev) | SS.mean_and_sds | seed:F9 |
| S04 | the mean gain becomes a naive sum | SS.mean_and_sds | seed:F7 |
| S05 | condition 3 compares with 2.0 (pp) instead of 0.020 | SS.condition_3 | seed:F11 |
| S06 | condition 3: d < 0.020 becomes d <= 0.020 | SS.condition_3 | seed:S9 |
| S07 | the drop's sign flipped | SS.build_tables | seed:F6 |
| S08 | a missing seed counted as passing (condition 1) | SS.condition_1 | seed:S5 |
| S09 | the E3-divergence override removed | SS.verdict | seed:S14 |
| S10 | an E2 divergence changes the verdict | SS.verdict | seed:S17 |
| S11 | the reason order changed (condition 1 first of all) | SS.verdict | seed:S16 |
| S12 | condition 2 computed over the seeds present (two), its guards removed | SS.condition_2, SS.mean_and_sds | seed:S8 |
| S13 | the missing-cell labels in the wrong order (candidate first) | SS._cell | seed:L7 |
| S14 | comparator values interpolated (a missing point filled) | SS._comparator | seed:W4 |
| S15 | the cut entry's comparator not flagged | SS.cut_curves | seed:W4 |
| S16 | a cut QAT run compared with its own seed, not seed 42 | SS.cut_curves | seed:W5 |
| S17 | the robustness refusal fires whatever the pair's status (LRB11) | SS.load_seed_inputs | seed:W5 |
| S18 | the seed-42 cross-check disabled | SS.seed42_cross_checks | mut:seed42_tamper |
| S19 | a QAT selection reads its own kind, not its QAT run's (LRB5) | H.resolve_status | seed:L6 |
| S20 | a PTQ model with kind null ignores a diverged parent | H.resolve_status | seed:L5 |
| S21 | slots without a family/per_seed role are no longer skipped (LRB4) | H.record_slots | seed:W1 |
| S22 | fq_off rows checked against the stage precision (LRB7) | H.load_bound | report:Q3 |
| S23 | the non-finite label replaced (D-6) | H._label | seed:L6 |
| S24 | the binding's sha256 comparison skipped | H.read_inventory | seed:R2e |
| S25 | strata delta direction swapped | ST.pair_table | strata:H1 |
| S26 | the zero-IoU list includes classes already at 0 | ST.pair_table | strata:H17 |
| S27 | the zero-IoU list includes E1 -> E6 (D-8) | ST.pair_table | strata:H16 |
| S28 | not-evaluable classes summarised as 0 | ST.pair_table | strata:H10 |
| S29 | GT-present float64 values computed with the union rule | AM6.gt_present_row | am6:F4 |
| S30 | a model row's entry_kind no longer checked against its slot's entry (IF§11.1) | H.model_row | seed:R2i |
| S31 | a telemetry item's path no longer required before it is read (LRB2) | H.read_telemetry | seed:R17b |
| S32 | the strict parse accepts a repeated JSON key | H.strict_json_text | seed:R2j |
| S33 | R-12 checked before R-13 (a seed-42 run of record without a model named parent_rule) | H.planned_cells | seed:R13b |
| S34 | a comparator without a model never shows the VAL rows it wrote (AM-19 item 3(c)) | SS._comparator | seed:W9 |
| S35 | the markdown drops the no-shared-point statement beside empty values (LRB6) | SS.render_markdown | seed:W10 |
| S36 | a missing run's reason drops its entry's decision-log rows (D-6) | SS.Missing.text | seed:S21 |
| S37 | the markdown prints the slot key of each model (IF§2) | SS.render_markdown | seed:X6 |
| S38 | the JSON's limitation_divergences left empty (plan §5 a16) | SS.build_tables | seed:W6 |
| S39 | a table cell's effect is baseline - candidate | SS._cell | seed:F2 |
| S40 | mC pools each corruption's totals over its severities instead of averaging the cells | SS.miou_c | seed:F4 |
| S41 | condition 2 compares the mean gain with the smaller SD | SS.condition_2 | seed:F10 |
| S42 | strata: summary.per_class no longer checked against the npz (plan §2.1) | ST.model_checks | strata:R6 |
| S43 | strata: the markdown appendix loses its per-class rows (plan §2.4) | ST.render_markdown | strata:E11 |
| S44 | strata: the pairs in reverse order | ST.build_strata_tables | strata:H20 |
| S45 | am6: the GT-present float32 values computed with the union rule | AM6.gt_present_row | am6:F4 |
| S46 | am6: the GT-present class counts taken from the union rule | AM6.gt_present_row | am6:F4 |
| S47 | am6: DL-99's E1 checkpoint record not checked | AM6.build_am6_tables | am6:R5 |
| S48 | defaults: the candidate's GT-present all-class value from the union rule | DC.candidate_values | defaults:V14 |
| S49 | defaults: the candidate's GT-present disease-only value from the union rule | DC.candidate_values | defaults:V14 |
| S50 | defaults: a coinciding row has 13 cells under the 12-column header | DC.render_markdown | defaults:V11 |
| S51 | report: the --strata file's directory no longer guarded against the out-dir (D-14) | RX.run | report:P4 |
| S52 | report: the --mde-entry's directory no longer guarded against the out-dir (D-14) | RX.run | report:P4 |
| S53 | report: an fq_off row without its binding record reads 'not available' (LRB3) | RX.build_fqoff | report:Q5 |
| S54 | report: --parts fqoff no longer checks the 37's identities (R-11) | RX.build_fqoff | report:Q7 |
| S55 | report: the banner no longer names a split that is not TEST | RX.banner | report:F4 |
| S56 | report: the frozen code's provenance no longer gates the profile (O3) | RX.official_unmet | report:H8 |
| S57 | seed: the Conditions table loses its banner (D-13) | SS.render_markdown | report:F1 |

### 6. The IF§11 v2 fields each loader reads, and the stub reader

**The stub reader** (IF§11.6): `read_inventory(path, *, expect_sha256=None) -> dict` (H:307), the
signature of `src.tman.validate.read_inventory`, marked `read_inventory.lrep_stub = True` (H:350).
It reads the export's bytes once (sha256 against `expect_sha256`), parses strict JSON (a repeated key
or a non-finite constant refused), checks the top level, and for a bound export reads the binding file
at `binding.path` resolved against the export's directory (an assumption B-4 settles), compares its
sha256 with `binding.sha256`, requires schema `tday_binding/1` with an `artifacts` list, and returns
the records under `binding.artifacts`. `open_inventory` (H:396-441) refuses it under Policy.OFFICIAL
(H:402-405).

**Fields, by loader** (H = src/stats/heldout_inputs.py):
- `_top_level_problem` (H:266-304): `schema` = "tday-inventory/1"; `evaluator_commit` (40 hex);
  `manifest_sha256` (64 hex); `execution_context` (an object); `models` and `entries` (lists of
  objects); `binding` {`state` bound | unbound, `path`, `sha256`}; `freeze` (null or
  {`manifest_sha256`}); `dedup` (null or an object; not read further here); `models[].slot`,
  `models[].precision`; `entries[].slot`, `entries[].role` (a list of strings).
- `open_inventory` (H:396-441): `binding.state` (bound under OFFICIAL), `freeze.manifest_sha256` =
  `manifest_sha256` (a freeze required under OFFICIAL); `binding.artifacts[]` {`slot`, `precision`,
  `scope`, `cell`}; indexes entries by slot, rows by (slot, precision), records by (slot, precision,
  scope, cell).
- `record_slots` (H:464-490): `entries[].role` (family, per_seed), `.stage`, `.seed`.
- `_checked_kind`, `_kind_fields_problem`, `_label`, `_record_of` (H:493-558): `entries[].kind`; cut:
  `cut_date` (YYYY-MM-DD), `rule`, `comparator` (null or a slot); diverged: exactly one of
  `iteration` | `epoch`, `rule`; aborted: `rule`, `cause`; not_run and not_made: `reason`; and
  `dl_rows`, kept for D-6's reason.
- `resolve_status`, `_parent_of`, `_not_run_status`, `check_parentage` (H:561-692):
  `entries[].catalogue.parent`, and the parent's `stage`, `seed` and `kind`.
- `model_row` (H:695-726): `models[]` `file_id` (IF§2: no "__", no "test"), `stage`, `model_role`,
  `seed`, `source` {`kind`, `sha256`}, `converted_artifact_sha256` (INT8 rows only), `cells` (a list),
  `entry_kind` (a scored kind, and the slot's entry kind), `role` (the slot's role set),
  `training_run_id`. `check_rows` (H:737-749) reads `stage`, `seed` and `model_role` of each model of
  record's stage-precision row.
- `binding_record` (H:783-808): `binding.artifacts[]` `protocol`, `repo_relative_path`,
  `files_sha256` (the four files), `checkpoint_sha256`, `repo_commit`, `split_manifest_sha256`.
- `load_bound` (H:846-899): those record fields against the artifact; the row's identity
  (`row_identity` H:729-734: `converted_artifact_sha256` for INT8, else `source.sha256`).
- `read_telemetry` (H:1061-1110): `entries[].telemetry[]` {`run_id`, `path`, `sha256`}; in the file,
  rows with `event` "val": `iter` | `epoch`, `all_class_miou`, `nonfinite.all_class_miou`.
- `cross_check` (H:919-961): `evaluator_commit`.
- `check_seed42_identity` (H:1009-1026): the family slots' stage-precision rows and their clean
  records' `files_sha256`.
- seed_stability: `models[].training_run_id` picks the attempt of record (SS:388-396); a cut entry's
  `comparator` (SS:435-448).
- default_candidates: `entries[]` with role default_candidate (`stage`, `seed`, `kind`), their fp32
  rows' `source.sha256`, and the family E2/E3 rows' `source.sha256` (DC:65-119).
- am6_tables: `binding.artifacts[]` with scope clean and cell null (`slot`, `precision`)
  (AM6:109-110).
- report_extensions: the `models[]` of stage E6-KD; each fq_off row's `file_id`, `stage`, `seed`,
  `model_role` and `training_run_id` against its int8_qat row's (RX:84-93); the fq_off record (scope
  fq_off); `binding.state`, `binding.sha256`, `manifest_sha256`, `freeze` and `evaluator_commit`,
  recorded (RX:271-275).

Not read in B-1 to B-3: `dedup` beyond its type (B-5), `execution_context` beyond its type,
`models[].cells` beyond its type, `models[].source.path`, and the records no table reads (upstream,
seed 42's corrupted cells of E2–E5 and E7, corrupted_sev45, quant_descriptive), which are ignored.

### 7. The DL-24 workflow

One read-only workflow, `wf_3c1eb175-c6f`, over `git diff <O>` (scoped) and the rulings, after C7.
- **Agents.** Five reviewers, each followed by one adversarial verifier: R1 the input layer
  (heldout_inputs, the fixtures), R2 seed stability, R3 the strata, default-candidate, AM-6 and report
  code, R4 the smokes, the mutation harness and the fixtures, R5 numerical (it recomputed §1.2's
  formulas and the C1–C3 states on hand-made vectors). Agents read by exact path, ran git with the
  exclude form, did not web fetch, and did not import src.stats from ad hoc Python (every src.stats
  import reaches src/stats/tests.py); only the six lane smokes ran.
- **Runs.** Round 1 (17 Oct, 07:38–08:18 UTC): R1 and R2 reviewed; the other five agents stopped on
  the weekly usage limit (a pause, resumed). The same run was resumed on 26 Oct after the orchestrator's
  "Try again": 10 agents, 0 errors, 3,867,327 subagent tokens, 728 tool uses, 7,250 s.
- **Before and after (MEASURED).** HEAD d6cb71ad…, `refs/heads/lane/lrep`, the scoped status empty,
  and `sha256sum -c` of the 14 files under review 14/14 OK, both before (17 Oct 07:3x; again 26 Oct
  06:37) and after (26 Oct 08:50 UTC). The workflow changed nothing.
- **Result.** 66 findings (R1 14, R2 9, R3 16, R4 21, R5 6): 50 confirmed, 16 refuted.

**Confirmed (50), with where each was resolved** (case; mutation):
- R1-02 telemetry hashed and parsed from two reads: F1, one read (H:1082).
- R1-04 a bound export without binding.artifacts: F1 (seed R2h; R01).
- R1-05 a selection's parent not its QAT run; a PTQ null parent of another stage or seed: F1 (seed
  R12c, R12e, R14b; R04, R05, R06).
- R1-06 a non-student model row named seed_inconsistent: F1, precision_mismatch (seed R18b; R09).
- R1-07 fixture fidelity (fq_off ghost scope, upstream/corrupted/sev45 ghosts, 24 cells, IF§1 keys):
  F1.
- R1-08 the docstring's raise rule vs strict_json_text: F1; a repeated JSON key (seed R2j; S32).
- R1-10, R4-21 the fq_off fixture method: stated in §4 (LRB7) and here; no code change.
- R1-13 am6 without DL-99's E1 check: F3 (am6 R5; S47).
- R1-14 a divergence in the wrong unit: F1 (seed R4d; R03).
- R2-01 a descendant of a cut run that is not "not run"; a model from the wrong run: F1 (seed R12d,
  R12b; R07, R08).
- R2-02 the no-shared-point statement lost beside empty values: F2 (seed W10; S35).
- R2-03 a comparator without a model loses its own VAL rows: F2 (seed W9; S34).
- R2-05, R4-12, R5-04 F12's count (12 → 14 bitwise checks): F2.
- R2-06, R4-05 S12 killed by a KeyError; S14's anchor: F2 (S12 rewritten, killed by seed S8).
- R2-07, R3-06, R4-04, R4-16 X1's scope and its allowed imports (L-14): F2 (seed X1).
- R3-01 fq_off seed-42 identity and row identity: F4 (report Q6, Q7; R11, S54). R3-01's suggested
  source.sha256 comparison is not used: IF§8 lets an INT8 row's `source` be its provenance file, so
  the identity is file_id (IF§8 uniqueness) with stage, seed, model_role and training_run_id
  (IF§11.1).
- R3-02 summary.per_class not checked against the npz: F3 (strata R6; S42).
- R3-03 the Conditions table without the banner: F2; R3-04, R4-02 F1 counted sections: F4 (report
  F1 per table; S57).
- R3-05 the strata markdown without per-class rows: F3 (strata E11; S43).
- R3-07 the frozen code not in the profile's code condition: F4 (report H8; S56).
- R3-08 the strata and MDE directories not guarded: F4 (report P4; S51, S52).
- R3-09 a missing parent created by mkdir: F4 (report P5, P6; R10, R12).
- R3-10 a coinciding default row with 13 cells: F3 (defaults V11; S50).
- R3-11, R4-13, R4-14 case texts: F3.
- R3-12 SELECTION_RECORD's docstring: F3.
- R3-14 GT-present coverage beyond the 21 (E6-KD, the defaults' float32 values, §3.2's gap list):
  open item 3.
- R3-15 the inputs' split not recorded: F4 (report F4; S55).
- R3-16 the CLI's exit-code text: F4.
- R4-01 am6 F4 checked four of twelve numbers: F3 (am6 F4; S45, S46; defaults V14; S48, S49).
- R4-03 expected values read from the module under test: F3 and F4 (strata H20; S44; report Q3, H1).
- R4-06, R5-01, R5-02 F2/F3 recomputed outside the module: F2 (S39).
- R4-07 X2, strata E8 and am6 D3 on exact float64 values: F2, F3.
- R4-08 N2 missed three import forms, and replaces plan §7.1's "the rehearsal documents are unchanged
  (1)": F4 (report N2). That §7.1 item is replaced by N1 and N2 (§3), and `git diff <O>` is empty for
  rehearsal.py.
- R4-10 F10's vector: F2 (S41).
- R4-11 M1's markdown check: F4 (report M1).
- R4-15 F3's name and docstring: F4 (report F3).
- R5-03 F4 did not separate the nested mean: F2 (seed F4; S40).

**Round-1 findings the complete run did not re-raise** (verified by this session, labelled so):
R1r1-01 fq_off row identity (F4); R1r1-07 and R2r1-09 two entries for one slot (F1; seed R4c; R02);
R1r1-08 a row's entry_kind and role (F1; seed R2i; S30); R1r1-09 an fq_off row without its record
(F4; report Q5; S53); R1r1-12 not_canvas's text, R1r1-13 R-13 before R-12 (seed R13b; S33), R1r1-16 a
telemetry item's path (seed R17b; S31) (F1); R1r1-18 fixture fidelity (F1; its item 4, k = 0 AM-5
images, kept: a k = 2 world is refused by the frozen driver under NONOFFICIAL_SMOKE, "non-finite value
in the difference vector"); R2r1-08, R2r1-10, R2r1-12, R2r1-14 wording, DL rows, the limitation key
and slot keys in the markdown (F2; seed S21, W6, X6; S36, S38, S37). Not changed: R1r1-10 (planned
cells are validated for every --parts selection; by design, D-3 and LRB4: the plan of cells is one
unit) and R2r1-13 (the E2 clause separator keeps plan §1.4's format).

**Refuted (16), each named with the verifier's reason:**
- R1-01, R4-09: no case reached check_parentage's raises. Refuted: GO STEP 5's rule is per refusal
  code, and parent_rule has its case and killed mutation; R12b/R12d were added anyway (F1).
- R1-03, R1-09: telemetry and artifact paths are not checked for symlink containment. Refuted: no
  clause requires it; bytes are bound to TMAN's sha256 and driver.load_input's containment still runs;
  the code fails closed.
- R1-11: the stub wrapped under OFFICIAL. Refuted: the wrapper is needed for seed R2f; the CLI passes
  no reader.
- R1-12: a null freeze under OFFICIAL has no case. Refuted: LREP's stricter addition; the IF§11.5 rule
  is covered (R2g; C08).
- R2-04: a listed telemetry file with no VAL row gives an empty attempt rather than "wrote no VAL
  rows". Refuted: the clauses govern cut runs that wrote VAL rows; nothing printed is false.
- R2-08: ROBUSTNESS_SEEDS is only echoed. Refuted: the constant has D-11's value, the loading follows
  LRB11 and LRB12, and it matters only if the code is edited.
- R2-09: a non-finite comparator VAL row prints like an absent one. Refuted: LRB6 does not speak to
  tags; null is the module's non-finite encoding; no false number, label or verdict.
- R3-13: a default MDE entry that cannot be chosen gives None. Refuted: the same exactly-one rule as
  report.require_official_mde; the official profile refuses (RX:160-162) and the nonofficial one names
  `mde_entry_not_committed`.
- R4-17: P2 could leave a directory in the tree under a regression of check_out_dir. Refuted: nothing
  is written at HEAD; P2 uses a never-created path; C38 kills that regression in process.
- R4-18: a mutant of read_inventory loses the lrep_stub marker. Refuted: it is LRB8's harness pattern,
  and no killer runs under Policy.OFFICIAL.
- R4-19: the RESULT's code count was the number of C mutations. Refuted: X1 enforces the coverage; the
  label was changed to the covered count anyway (F1).
- R4-20: raise sites without a base case. Refuted: the rule is per code (GO STEP 5 and ACCEPTANCE),
  and every code has a reached case and a killed mutation.
- R5-05: a NaN reads "fails" in C1/C3. Refuted: unreachable, since m from pooled integer totals is
  always finite and a non-finite run has no value (D-6).
- R5-06: exact-arithmetic ties decided by the last float64 bit. Refuted: LRB14 and D-2 fix float64
  with no tolerance; the DL-a text proposed in §9 states it.

### 8. The SL-1 log, slips and open items

`tail -n 80 ~/.claude/sl1_guard.log` (MEASURED 26 Oct 12:21 UTC; the guard is in log mode) holds 16
lines, each logged and none denied. Commands are cut as the log cuts them.

| time (UTC) | tool | class | what | source |
|---|---|---|---|---|
| 2026-10-15T12:27:36Z | Bash | SL1-B4 | `for p in '…'` docs line lookups, patterns with "TEST" | Phase A session (Annex R) |
| 2026-10-15T12:27:42Z | Bash | SL1-B4 | the same | Phase A session |
| 2026-10-15T12:27:50Z | Bash | SL1-B4 | the same, over PA, STATS, EVAL, DL and ERR | Phase A session |
| 2026-10-17T07:21:36Z | Bash | SL1-B1 | `for s in smoke_strata_tables smoke_default_candidates smoke_am6_tables; do …` | this session, slip 5 |
| 2026-10-17T07:22:29Z | Bash | SL1-B1 | `for s in smoke_am6_tables smoke_report_extensions; do …` | this session, slip 5 |
| 2026-10-17T07:24:16Z | Bash | SL1-B1 | `for s in smoke_am6_tables smoke_lrep_mutations; do …` | this session, slip 5 |
| 2026-10-17T07:27:11Z | Bash | SL1-B1 | `for s in smoke_strata_tables smoke_default_candidates; do …` | this session, slip 5 |
| 2026-10-17T08:09:55Z | Grep | SL1-G1 | scripts/ with a glob of the seven lane scripts | DL-24 round 1, reviewer R1 |
| 2026-10-26T07:22:32Z | Grep | SL1-G1 | scripts/ with a glob of the eight lane scripts | DL-24 resumed run (agent) |
| 2026-10-26T07:24:45Z | Bash | SL1-B2 | `git ls-files -- 'reports/derived/mde_entry_*.json' reports/strata/train_strata_v1.json` | DL-24 resumed run (agent; the R3-13 verifier) |
| 2026-10-26T07:27:32Z | Grep | SL1-G1 | src/stats with glob {report.py,val_artifacts.py,artifact.py} | DL-24 resumed run (agent) |
| 2026-10-26T07:30:11Z | Grep | SL1-G1 | src/stats with the GO's glob `!*[Tt][Ee][Ss][Tt]*` | DL-24 resumed run (agent) |
| 2026-10-26T07:32:26Z | Grep | SL1-G1 | src/stats with a glob of the six lane modules | DL-24 resumed run (agent) |
| 2026-10-26T07:38:17Z | Grep | SL1-G1 | scripts/ with a glob of the six lane smokes | DL-24 resumed run (agent) |
| 2026-10-26T10:00:57Z | Grep | SL1-G1 | scripts/ with glob {smoke_seed_stability.py,smoke_report_extensions.py} | this session, slip 8 |

The 26 Oct 07:22–07:38 lines fall in the resumed workflow's window (06:4x–08:4x UTC), when this
session ran no search, so they are attributed to its agents (INFERRED). No line is a read of a
test-named path, TEST data or docs/reference/: the B4 lines are patterns, and the G1 and B2 lines
searched or listed named lane files, src/stats or reports paths.

**Slips (this session's own; none opened a test-named path, TEST data or docs/reference/):**
1. STEP 4: a Bash grep on reports/derived/mde_entry_20261009T034825Z.json, not the Grep tool on the
   named file.
2. STEP 4: `git ls-files -- src scripts <X> | grep -n -E 'eligibility|stats/|tman|eval/'` printed
   three scripts/ paths ("Never list scripts/"). Not repeated.
3. STEP 4: a Glob "*" over $SCR listed the venv's site-packages paths, some containing "test"; none
   was opened.
4. STEP 4/5: the C1/C2 files were written into the tree while STEP 4's last baseline smoke ran, so
   smoke_stats_official_fullsize failed A6 (36/37: code_not_clean_at_head). The files were moved to
   $SCR/parked (not deleted), the smoke re-ran alone on a clean tree at O: 37/37. Nothing was written
   into the tree during STEP 7.
5. 17 Oct (STEP 5): four dev re-runs of the lane smokes in a Bash `for s in …` loop (SL1-B1).
6. 26 Oct (F1): one more such loop; the guard did not log it.
7. 26 Oct: Bash `sed -n` reads of src/stats/driver.py and src/stats/noninferiority.py (F1–F4), and of
   src/stats/bootstrap.py twice (STEP 7, between runs), by exact path (code, not docs; no rule names a
   form for code, but the Read tool was the expected one). In the same pause one docs lookup piped
   `sed -n '158p' docs/DECISION_LOG.md` into `grep -o -E`, a variant of the GO's docs form.
8. 26 Oct ~10:01 UTC (STEP 7, while smoke_mde ran): one Grep-tool search with path scripts/ and the
   glob `{smoke_seed_stability.py,smoke_report_extensions.py}`, not on named files with the GO's glob
   (SL1-G1 class). It matched only those two files.
9. 26 Oct (F4): one docs lookup used `grep -n -i` on docs/lane_specs/errata.md (one file, by exact
   path) instead of the GO's `grep -n '<pattern>'` form.

**Tooling conflicts, each reported once:**
- The stop hook (~/.claude/stop-hook-git-check.sh) asked twice to push unpushed commits on lane/lrep.
  Not complied with: the GO allows exactly two pushes (after C2, and STEP 9).
- The harness's attribution reminder (Co-Authored-By and Claude-Session trailers) and its designated
  branch claude/admiring-wozniak-knhx5w: ignored per the GO; no commit has a trailer, and only
  lane/lrep was touched.
- L-PROT refused one `mv` whose source paths used a shell variable ("moves a directory that holds
  docs/reference/reference.pdf"); nothing moved; repeated with literal paths.

**Open items (5):**
1. B-4: swap the stub for TMAN's reader and add the conformance smoke after TMAN's merge, before D.
   Two assumptions are checked then: the stub resolves `binding.path` against the export's directory
   (H:10), and the evaluator scores E6-KD as stage E6 (EVALUATOR_STAGE, H:76-78).
2. Four §7.3 smokes did not run under the fixture rule, at STEP 4 or STEP 7, because each creates a
   test-named path (REPOSITORY-PROVEN): smoke_stats_integrity (:956 → :150 → stats_fixtures.py:255,
   runs/test/…), smoke_perclass_gap_table (:414 "e1_test"), smoke_gap_bootstrap_val (:634 "e_test")
   and smoke_train_strata (:287, :289, :292). None imports an LREP file and B-1 to B-3 edit no
   existing file. Whether EXEMPT-1 extends to them is the orchestrator's call.
3. DL-24 R3-14: GT-present coverage beyond the 21 planned cells (E6-KD's GT-present row, the default
   candidates' float32 values, the plan's §3.2 gap list) is not in B-3.
4. Hardening the refuted findings name, for B-4 or later: a null-freeze case under OFFICIAL (R1-12),
   the "wrote no VAL rows" note for an empty telemetry file (R2-04), a non-finite tag on comparator
   rows (R2-09), the stub marker on mutants (R4-18), per-site cases (R4-20).
5. C4 cannot run its own smoke alone: smoke_default_candidates imports am6_tables, which arrives in C5
   (the commits were not rewritten). Every final run is at F4.

### 9. Records proposed (the plan's §8, as LRB1, LRB7, LRB9, LRB14, LRB17 and LRB19 amend it)

The lane writes none of these.
- **Plan line** (DL's current plan, DL:34): "… L-AM6 and L-AM8 (lane LREP, with the AM-17 item 8
  strata TEST tables and the AM-17 item 6 default-candidate table) IMPLEMENTED before D (AM-22 v6
  item 1(a) condition 4) …" (LRB1: not "before TEST").
- **IMPLEMENTED entries** (B-1 to B-3 at the final commit of lane/lrep; B-4, the stub swap and a
  conformance smoke, follows TMAN's merge, before D):
  - DL-13 Status: "L-AM8 IMPLEMENTED <sha> (lane/lrep; merged <sha>)".
  - DL-12 Status: "L-AM6 IMPLEMENTED <sha>".
  - DL-29 Status and Where: "AM-17 item 8 TEST tables and item 6 default-candidate table IMPLEMENTED
    <sha> (LREP); the TEST inference of L-AM17-DEFAULTS stays with the TEST-manifest lane".
  - DL-72 Where: add src/stats/report_extensions.py.
- **New DL rows (DECIDED, before D):**
  - DL-a: the LREP reading decisions D-2, D-3, D-6 to D-11 and D-17, as LRB4, LRB6, LRB11 and
    LRB14 to LRB16 rule them. It records D-6 citing AM-21 item 2(b) (LRB9), and states that m(S,k)
    comes from float64 pooled totals (LRB14) and that the conditions compare those float64 values with
    no tolerance, so a tie in exact arithmetic is decided by the last bit (DL-24 R5-06).
  - DL-b: the TMAN↔LREP interface T1–T6 (IF§11 v2), and that LREP owns AM-18 item 8(c)'s subset
    recomputation and AM-18's arm tables as B-5 (LRB17), and AM-23's descriptive rows (LRB22).
- **Errata:**
  - E-65: report.py's three extension points stay unpopulated in report.md; the extension report
    populates them (K1; LRB18).
  - E-66: dropped (LRB9).
  - E-67: stats_fixtures.rel_name names folders by split, and lane smokes never put "test" into a path
    (DL-41; LRB20).
  - The erratum to PA:1297 (AM-18's Code line) that records the B-5 split, with the orchestrator's
    records (LRB17).
- **The plan's §4 inventory** adds the E6-KD TEST score (if AM-3 triggers) and AM-23's quantities
  (conditional on its commit) (LRB7).
- **Chapter 4 sentences (drafts):** the plan's five stand. One is added for LRB7: "AM-4's fake-quant-off
  TEST score of each selected QAT model of record is reported beside its int8_qat value, descriptively
  (TL-12); a model without an fq_off artifact reads 'not available: no fq_off artifact'."

### 10. Phase A form deviations (LRB19 L-19; accepted as recorded, listed once)

From the plan's Annex R:
- On the two permitted reports files (reports/strata/train_strata_v1.json and
  reports/derived/mde_entry_20261009T034825Z.json) the Phase A session also ran `ls -l` and
  `sha256sum` (both hashes equal DL-63's and DL-99's), and one grep on the MDE entry, through Bash
  rather than the Read tool.
- One `git ls-files -- src scripts configs | grep -i -c test` ran without the test exclude. It printed
  only the count, 2, and opened nothing.
- Some docs lookups used `grep -n -F` or `-o` instead of `-E`, still one file per command.
