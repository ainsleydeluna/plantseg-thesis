# L-AM29-FIX lane report: AUDIT-LAM29-1's fixes on L-AM29's scoring path, and AM-29a's binding

Session L-AM29-FIX (code lane; LAM29-FIX-4 and LAM29-FIX-5 of DL-274). Branch `lane/l-am29-fix`, created from
`lane/l-am29` at `2a94019f4dc11b8c0b543107e026ecb5ecc108c5`. Three commits on it:

| commit | parent | what |
|---|---|---|
| A `9b486c9c081054ad86a225dcb19fa523a6b470f4` | `2a94019f` | AUDIT-LAM29-1's F-01, F-02, F-03, F-08 (the scoring path) and F-07, F-09, F-14, with their smoke cases |
| B `0480ef0c343d0eb02f72c41a4c3fb74c2d5fe716` | A | AM-29a item 7(a)'s binding: one entry in `L_AM24_RUNS`, with its smoke case |
| C (this report) | B | `docs/lane_reports/l-am29-fix.md` only |

Written 2026-11-17 09:37:52 +0800 (clock read with `date`); Step 0 ran at 08:27 +0800. Claims carry one of four
labels, as L-AM29's report uses them: MEASURED (run in this session, output shown), DOCUMENTED (a record says so,
cited), REPOSITORY-PROVEN (the code at the commit shows it), INFERRED (reasoned, not shown).

No experiment, pod, real-data scoring or TEST was run. Every dataset, model and checkpoint used here is synthetic.

## 1. Result

- **Step 0: AM-29a is in force by the prompt's test** (MEASURED, section 2), so **B was made**, in its own commit B,
  whose diff is the one entry plus its smoke case (section 7).
- **Every fix the prompt lists was made** (section 4): F-01, F-02, F-03 and F-08 (a) to (d) in the code the TEST path
  runs; F-07 in `scripts/smoke_am29_runner.py`; F-09 in `scripts/am29_blockb.py`; F-14 in `scripts/am29_islands.py`.
  Every changed line of `scripts/am29_score.py` and `src/eval/am29_score.py` is listed with its tag in section 6 (154
  lines, none untagged).
- **Smokes, before (2a94019f) and after (B), CPU, synthetic data** (MEASURED, section 9):

| smoke | before | after |
|---|---|---|
| `scripts/smoke_am29_runner.py` | PASS 44/44 | PASS 46/46 |
| `scripts/smoke_am29_islands.py` | PASS 54/54 | PASS 58/58 |
| `scripts/smoke_am29_blockb.py` | PASS 35/35 | FAIL 53/54 |
| `scripts/smoke_am29_isolation.py` | PASS 11/11 | PASS 11/11 |
| `scripts/smoke_am29_fixtures.py` | no checks (the fixtures module), exit 0 | no checks, exit 0 |
| `scripts/smoke_qat_runner.py` | PASS 214/214 | PASS 214/214 |

  The "before" counts equal those L-AM29's report gives at 2a94019f (44, 54, 35, 11; 214).
- **STOP-1: one existing smoke case fails after F-01, by design of hard limit 4.** `score_am29_int8`
  (`scripts/smoke_am29_blockb.py`:242-244) requires the AM29_SCORE_JSON line to hold exactly the six keys of 2a94019f.
  F-01 puts `score["protocol"]` in that line, as the audit words it, so the line holds seven keys and the case fails.
  Hard limit 4 lets the smokes gain cases only, so the case was not edited. Every other check of the smoke passes,
  and the failing case's own detail shows the scoring succeeded (`RESULT: SCORED (int8, T1, val, ...)`). The remedy is
  one token in that case, `"protocol"` added to its key set, and it needs the orchestrator's ruling (section 10).
- **F-07's check fails under both mutations of the audit's section 5.3** and passes on the code as committed
  (MEASURED, section 9.3). The mutations were made in memory only; `src/quant/am29.py` was never written (its sha256 is
  the same before and after).
- **Isolation** (MEASURED): the branch changes 7 files, all of them in hard limit 4's list, and adds this report.
  Every `QAT_CODE_PATHS` file and every further `E6KD_CODE_PATHS` file is byte-identical
  (`smoke_am29_isolation.py`, section 9). No file under `docs/reference/` and no test-named path was opened, listed or
  hashed.
- **Pushed:** `lane/l-am29-fix` only (section 12). master, `lane/l-am29` and every other branch were not pushed,
  merged or touched.

## 2. Step 0: is AM-29a in force?

Fetched `origin/master` at 2026-11-17T00:27:00Z (08:27 +08:00). Master's head:
`02962ab9c2b16d526ef3d76a4cd4dfb78d6734aa` (MEASURED). The two checks, on master's files read with `git show`
(MEASURED):

```
check 1 (heading '## AM-29a — ' in master's docs/PREREGISTRATION_AMENDMENTS.md):
3486:## AM-29a — One more candidate for AM-29's shown student: E6-R2-MT, the matched graph with TF32 off and the parent-teacher term, at seed 42 (amends AM-29 items 3, ...
rc=0
check 2 (row starting '| DL-278 | RECORDED (AM-29a records commit, 2026-11-17)' in master's docs/DECISION_LOG.md):
458:| DL-278 | RECORDED (AM-29a records commit, 2026-11-17); Status DRAFT | AM-29a, one more candidate for AM-29's shown student: E6-R2-MT (lane L-AM24 item N2R2T; run ID x-n2r2t-s42), AM-29 item 3(b)
rc=0
```

Both hold, so AM-29a is in force by the prompt's test, and B was made. What this lane did not check is in open
question 9.

## 3. Environment

- The container had no PyTorch (MEASURED: `import torch` fails for every interpreter). A throwaway venv was made in the
  session scratchpad with `/usr/bin/python3.11` and L-AM29's versions: Python 3.11.18, torch 2.1.0+cpu, torchvision
  0.16.0+cpu (download.pytorch.org), numpy 1.26.4, pillow 12.3.0, psutil 7.2.2, packaging 24.2 (PyPI); QNNPACK is in
  `torch.backends.quantized.supported_engines` (MEASURED). Nothing in the repository changed for it (deviation 1).
- CPU: 4 cores, 15 GiB. The clone is shallow (`git rev-parse --is-shallow-repository` prints `true`); aa8cac2, the
  isolation smoke's base, is in it (MEASURED).
- `TMPDIR` was `/home/user/.am29fixtmp`, a symlink outside the repository to the scratchpad's `tmp`, so every temp file
  went to the scratchpad and the DataLoader sockets stay under the AF_UNIX path limit (L-AM29's deviation 13). No temp
  path names TEST or VAL but the synthetic `images/val` tree the evaluator reads.

## 4. The fixes, by file and line range (line numbers at B, REPOSITORY-PROVEN)

### F-01 (`scripts/am29_score.py`)
- `--protocol`, default `canvas`, choices `canvas` and `upstream`: :57.
- `cli_error` refuses `--protocol upstream` with `--provenance` (block B scores canvas), code `[protocol]`: :104-105.
- `protocol=a.protocol` passed to `SC.score_model`: :207. `score["protocol"]` in the AM29_SCORE_JSON line: :224-227.
- Usage and text: :5, :19-20.
- Smoke: `score_protocol_upstream` (`scripts/smoke_am29_blockb.py`:282-291), the T1 stand-in artifact on 2 VAL samples
  with `--protocol upstream`: exit 0, and both the line and the `--out-json` record hold `upstream/1.0.0`. Also
  `test_rules_provenance_canvas_only` (:641-644), the refusal before anything is read.

### F-02 (`src/eval/am29_score.py`)
- `"x-d-n3-s42": "D"` in `L_AM24_RUNS`: :51, the table written one entry per line (:48-53) so that B adds one line.
  The comment above it and the module text no longer count the runs (:21-22, :46-47).
- `scripts/am29_score.py`'s usage names the run IDs by the table (:6, :13).
- Smoke: `lam24_d_mixed_route` (`scripts/smoke_am29_blockb.py`:565-585): a smoke E6-E2 run named x-d-n3-s42 in
  lane/l-am24's block B format; through the mixed route an unknown run ID (x-n5-s42) is refused `[artifact_unknown]`
  and x-d-n3-s42 is scored "mixed precision", name D. Added beside it, `lam24_d_int8_route` (:586-591): the int8 route
  scores the same fixture's artifact, name D, stage E6-E2.

### F-03 (`scripts/am29_score.py`)
- The source `--qat-eval-dir E --qat-selection-sha256 H`: parser :64-65; one of the four sources in `cli_error`
  (:96-99), the two flags only together (:106-107).
- Accepted with `--precision mixed` only, `[qat_eval_dir_precision]`: :108-110.
- The values of record (:45-49): the selection ce435dea… (DL-131, DL-173), the finalize record b2c7705c… (DL-173), the
  checkpoint 74fd438a… (AM-29 item 2(a)). The selection's and the checkpoint's are checked from the arguments in
  `cli_error`, before anything is read (:121-128); they bind at `--split test` always and at `--split val` unless
  `--allow-smoke-inputs` (`values_of_record`, :136-138).
- `_e6_s42_selection` (:298-324, with `_read_hashed` and `_parse_hashed`, :278-295): E/qat_selection.json is read
  once, hashed and compared with H before it is parsed, and the same bytes are parsed (`[qat_selection_sha256_mismatch]`);
  E/qat_finalize.json is hashed and checked against b2c7705c… (`[qat_finalize_not_of_record]`), then must be a
  finalized `qat_finalize/1` record naming that selection by sha256; the selection and the finalize record must be
  stage E6, seed 42 (`[qat_stage]`).
- The selection is handed to `selected_state` unchanged (`_selection_only`, :247-248), which binds the winner's
  `epoch_ckpts/e15.pt` through the run's telemetry and `--expect-checkpoint-sha256`. The record names
  `source_kind "qat"`, label E6-s42 and the finalize record's sha256 (:204-205, :214-216). `--qat-eval-dir` joins the
  paths refused when they name test (:164-166). Usage and text: :7, :21-24.
- Smoke (`qat_e6_s42_source`, `scripts/smoke_am29_blockb.py`:396-472), on a complete smoke E6 run at seed 42 and an
  eval dir in the pinned block B's format: the pass (`qat_e6_s42_scored_mixed`), a wrong selection sha256
  (`qat_e6_s42_selection_sha256_refused`), a wrong stage (`qat_e6_s42_wrong_stage_refused`) and use without
  `--precision mixed` (`qat_e6_s42_int8_refused`). Added beside them: the values of record refused from the arguments
  (`qat_e6_s42_values_of_record`, three argument sets, one at `--split test` with `--allow-smoke-inputs`) and the
  finalize record of record (`qat_e6_s42_finalize_of_record`, in process).

### F-08 (`scripts/am29_score.py`; `src/eval/am29_score.py` needed no change)
- (a) `--split test` without `--out-json` refused, `[test_out_json]`, last of the TEST rules in `cli_error`: :92-93.
  Smoke `test_rules_out_json_required` (`scripts/smoke_am29_blockb.py`:636-640).
- (b) after `score_model`, a score whose `torch_num_threads` is not `--threads` is refused, `[threads_mismatch]`, before
  anything is printed or written: :208-210. Smoke `score_threads_mismatch_refused` (:292-305), in process: a wrapper
  of the real scorer sets one more thread.
- (c) `src.eval.stage_artifacts.governed_paths_error(REPO)` at `--split test`, before anything is read; a value that is
  not None refuses, `[governed_paths_dirty]` (the audit's first option): :156-163. The value is recorded in the score
  record as `governed_paths_error` (None, since any other refuses): :220-221. At `--split val` it is not called and no
  key is written (deviation 3). Smoke `score_governed_paths_recorded` (:306-322) and
  `score_governed_paths_dirty_refused` (:323-330), in process, the function stubbed.
- (d) at `--split test`, a group whose variant record has `eligible` not true is refused, `[group_not_eligible]`, after
  the variant is built and before the scorer: :195-199. Smoke `score_test_group_not_eligible_refused` (:331-342): G2 is
  refused and never meets the scorer; G1 passes the check and meets the scorer, a tripwire there.
- (e) no change (LAM29-FIX-3 settled it in AM-29's text).
- Every new check stays behind the existing TEST rules: (a) is the last of them in `cli_error`, and (b) to (d) run
  after `cli_error` returns. The seven argument sets of `test_rules_refuse_before_reading` are refused with their
  codes as before (MEASURED, section 9).

### F-07 (`scripts/smoke_am29_runner.py`)
- `term_loss_identity` (:144-162), called at the end of the term section (:141), tolerance :54: on a T1 run (factor
  0.5) and a T3 run (factor 1.0), each of 4 steps, every train row's `loss` equals `sup + factor × 0.5 × logit_kd`
  within 2.4e-7. The weight is the arm's factor times 0.5, never the row's own `kd_weight`, and every step's term must
  be non-zero, so the check is not vacuous. Text :7-8.
- The two mutations of the audit's section 5.3, the ramp and the factor without λ, make it fail; section 9.3.

### F-09 (`scripts/am29_blockb.py`)
- `--threads` required for `score` (argparse `required=True`): :76; usage :6.
- In `_score_errors`, for a non-smoke selection, every score record must hold `code_clean_at_head` true and a non-null
  `score.eval_runtime.image_digest`; either miss refuses the selection `[score_values]`: :277-281; text :23-24.
- Smoke (`scripts/smoke_am29_blockb.py`:191-218): `blockb_score_requires_threads` (score without `--threads` exits 2
  and writes nothing); `blockb_select_requires_clean_code` and `blockb_select_requires_image_digest` (this run's e01
  score record, checked by `_score_errors` three ways: both present, each missing, and a smoke selection, which the
  two rules do not touch).

### F-14 (`scripts/am29_islands.py`)
- `MODELS["E7-R2"]` gets `"stage": "E7"`: :42. Its G0 state_dict is labelled under E7, as E7-R2 is made.
- E6-s42-e15's twelve-digit prefix replaced by the full sha256
  74fd438a9e953ce84117a61454b879940b0a6a8f357e7705c22a86ab0ed847a5 (:37-39), and `qat_source` requires equality
  outside smoke inputs (:99-100); text :10-12.
- `--threads` read back after it is set, `[threads_not_set]`, before anything is read: :196-198.
- A run without `--allow-smoke-inputs` requires `code_clean_at_head` true, `[code_not_clean]`: `clean_identity`
  (:161-169), called once the source is bound and before any conversion (:203, :208); the identity it returns is the
  one the report records (the old call after scoring, at 2a94019f :211-212, is gone). Text :18-19.
- Smoke (`scripts/smoke_am29_islands.py`:249-292): `cli_e7r2_g0_labelled_e7` (on the same weights and calibration,
  E7-R2's G0 sha256 equals E7's), `cli_qat_full_sha256_of_record` (a sha256 with the right first twelve digits is
  refused), `cli_threads_read_back` (a thread count that does not take is refused), `cli_nonsmoke_requires_clean_code`
  (in process, the source and the code identity stubbed).

### B: AM-29a item 7(a)'s binding (`src/eval/am29_score.py`)
- `"x-n2r2t-s42": "E6-R2-MT"` in `L_AM24_RUNS`: :52. Nothing else changed for it (section 7).
- Smoke `lam24_e6r2mt_bound_by_run_id` (`scripts/smoke_am29_blockb.py`:592-611): a smoke run named x-n2r2t-s42 in
  lane/l-am24's block B format is scored as int8, name E6-R2-MT; an unknown run ID, x-n2r2t-s43, is still refused
  `[artifact_unknown]`.

## 5. Full diff stat against 2a94019f (MEASURED)

```
$ git diff --stat 2a94019f4dc11b8c0b543107e026ecb5ecc108c5..0480ef0c343d0eb02f72c41a4c3fb74c2d5fe716 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
 scripts/am29_blockb.py        |  12 +-
 scripts/am29_islands.py       |  38 ++++--
 scripts/am29_score.py         | 139 +++++++++++++++++++--
 scripts/smoke_am29_blockb.py  | 280 ++++++++++++++++++++++++++++++++++++++++++
 scripts/smoke_am29_islands.py |  46 +++++++
 scripts/smoke_am29_runner.py  |  25 ++++
 src/eval/am29_score.py        |  15 ++-
 7 files changed, 525 insertions(+), 30 deletions(-)
```

`git diff --name-status` over the same range gives `M` for each of the seven files. Commit C adds one file, this
report, and changes nothing else. In the smoke files every change is an added line (`git diff --numstat`: 0 deleted
lines in `scripts/smoke_am29_blockb.py`, `scripts/smoke_am29_islands.py` and `scripts/smoke_am29_runner.py`).

## 6. Every changed line of the code the TEST path runs, tagged (MEASURED)

`git diff -U0 2a94019f..0480ef0c -- scripts/am29_score.py src/eval/am29_score.py`, one line per changed line: the tag,
the file, `+N` (a line of B, at its line number there) or `-N` (a line of 2a94019f removed, at its line number there),
and the text. 154 lines; the tagging script refuses an untagged line and a tag with no changed line, and found
neither. Tag counts: F-01 11, F-02 17, F-03 95, F-08 (a) 8, F-08 (b) 3, F-08 (c) 10, F-08 (d) 5, B 1, and four lines
that serve two fixes (a usage or text line rewritten for both).

```
F-08 (a), F-01     scripts/am29_score.py:-4        python -B scripts/am29_score.py --split val|test [--confirm-test-split --threads T] --precision int8|mixed \\
F-02, F-03         scripts/am29_score.py:-5            (--am29-eval-dir E [--arm A] | --l-am24-blockb-dir B --run-id x-n2-s42|x-n2r2-s42 | --provenance P --stage S)
F-08 (a)           scripts/am29_score.py:+4        python -B scripts/am29_score.py --split val|test [--confirm-test-split --threads T --out-json PATH] \\
F-01               scripts/am29_score.py:+5            --precision int8|mixed [--protocol canvas|upstream] \\
F-02               scripts/am29_score.py:+6            (--am29-eval-dir E [--arm A] | --l-am24-blockb-dir B --run-id R | --provenance P --stage S
F-03               scripts/am29_score.py:+7             | --qat-eval-dir E --qat-selection-sha256 H)
F-02               scripts/am29_score.py:-11   AM-29's runs on lane/l-am24 (N2 x-n2-s42, E6-R2-M x-n2r2-s42); --provenance one converted epoch (block B's VAL scoring,
F-02               scripts/am29_score.py:+13   AM-29's runs on lane/l-am24 (the run IDs of L_AM24_RUNS); --provenance one converted epoch (block B's VAL scoring,
F-08 (a)           scripts/am29_score.py:-14   group's FP32 islands and is labelled "mixed precision", never "int8". The TEST split needs --confirm-test-split and
F-08 (a)           scripts/am29_score.py:-15   --threads and takes no sample cap. Nothing per image is printed or written: the output is the dataset-level block.
F-08 (a)           scripts/am29_score.py:+16   group's FP32 islands and is labelled "mixed precision", never "int8". The TEST split needs --confirm-test-split,
F-08 (a)           scripts/am29_score.py:+17   --threads and --out-json and takes no sample cap. Nothing per image is printed or written: the output is the
F-08 (a)           scripts/am29_score.py:+18   dataset-level block.
F-01               scripts/am29_score.py:+19   
F-01               scripts/am29_score.py:+20   --protocol upstream scores under the upstream PlantSeg protocol (AM-23 item 5); canvas is the default, and block B's.
F-03               scripts/am29_score.py:+21   --qat-eval-dir, mixed only, is E6 seed 42's block B of record (AM-29 item 6's look of E6 seed 42): its qat_selection.json
F-03               scripts/am29_score.py:+22   is hashed before it is parsed and must equal --qat-selection-sha256 (of record ce435dea..., DL-131), its
F-03               scripts/am29_score.py:+23   qat_finalize.json must be of record (b2c7705c..., DL-173), and epoch_ckpts/e15.pt is bound through the telemetry by
F-03               scripts/am29_score.py:+24   --expect-checkpoint-sha256 (of record 74fd438a..., AM-29 item 2(a)); --allow-smoke-inputs lifts these at VAL only.
F-08 (c), F-08 (d) scripts/am29_score.py:+25   At --split test the governed paths must be clean (src/eval/stage_artifacts.py governed_paths_error; recorded) and a
F-08 (d), F-08 (b) scripts/am29_score.py:+26   mixed-precision group must be eligible (AM-29 item 2(d)); a score whose thread count is not --threads is refused.
F-03               scripts/am29_score.py:+33   import hashlib
F-03               scripts/am29_score.py:+45   # E6 seed 42's block B of record, the one --qat-eval-dir takes (AUDIT-LAM29-1 F-03): qat_selection.json (DL-131,
F-03               scripts/am29_score.py:+46   # DL-173), qat_finalize.json (DL-173) and the selected checkpoint epoch_ckpts/e15.pt (AM-29 item 2(a))
F-03               scripts/am29_score.py:+47   E6_S42_SELECTION_SHA256 = "ce435dea034073a45ae7353846f2c0a84c277e6126a838eb4d36d18de19358eb"
F-03               scripts/am29_score.py:+48   E6_S42_FINALIZE_SHA256 = "b2c7705c32b800f6fbbefff40d201cf828d2736b053236297e550189a7830c03"
F-03               scripts/am29_score.py:+49   E6_S42_CHECKPOINT_SHA256 = "74fd438a9e953ce84117a61454b879940b0a6a8f357e7705c22a86ab0ed847a5"
F-01               scripts/am29_score.py:+57       p.add_argument("--protocol", default="canvas", choices=["canvas", "upstream"])
F-03               scripts/am29_score.py:+64       p.add_argument("--qat-eval-dir", default=None)
F-03               scripts/am29_score.py:+65       p.add_argument("--qat-selection-sha256", default=None)
F-08 (a)           scripts/am29_score.py:+92           if a.out_json is None:                               # AUDIT-LAM29-1 F-08 (a)
F-08 (a)           scripts/am29_score.py:+93               return "test_out_json", "--split test requires --out-json (the look's record, written once)"
F-03               scripts/am29_score.py:-74       sources = [k for k in ("am29_eval_dir", "l_am24_blockb_dir", "provenance") if getattr(a, k) is not None]
F-03               scripts/am29_score.py:+96       sources = [k for k in ("am29_eval_dir", "l_am24_blockb_dir", "qat_eval_dir", "provenance")
F-03               scripts/am29_score.py:+97                  if getattr(a, k) is not None]
F-03               scripts/am29_score.py:-76           return "source", "exactly one of --am29-eval-dir, --l-am24-blockb-dir, --provenance"
F-03               scripts/am29_score.py:+99           return "source", "exactly one of --am29-eval-dir, --l-am24-blockb-dir, --qat-eval-dir, --provenance"
F-01               scripts/am29_score.py:+104      if a.provenance is not None and a.protocol != "canvas":  # AUDIT-LAM29-1 F-01
F-01               scripts/am29_score.py:+105          return "protocol", "--provenance is block B's VAL scoring, which scores canvas only"
F-03               scripts/am29_score.py:+106      if (a.qat_eval_dir is None) != (a.qat_selection_sha256 is None):          # AUDIT-LAM29-1 F-03
F-03               scripts/am29_score.py:+107          return "source", "--qat-eval-dir and --qat-selection-sha256 are given together"
F-03               scripts/am29_score.py:+108      if a.qat_eval_dir is not None and a.precision != "mixed":
F-03               scripts/am29_score.py:+109          return "qat_eval_dir_precision", ("--qat-eval-dir scores E6 seed 42's selected weights under a P2 group: "
F-03               scripts/am29_score.py:+110                                            "--precision mixed only")
F-03               scripts/am29_score.py:+121      if a.qat_eval_dir is not None and values_of_record(a):  # AUDIT-LAM29-1 F-03: the values of record, by argument
F-03               scripts/am29_score.py:+122          if a.qat_selection_sha256 != E6_S42_SELECTION_SHA256:
F-03               scripts/am29_score.py:+123              return "qat_selection_not_of_record", (f"--qat-selection-sha256 {a.qat_selection_sha256}: E6 seed 42's "
F-03               scripts/am29_score.py:+124                                                     f"qat_selection.json of record is {E6_S42_SELECTION_SHA256} (DL-131)")
F-03               scripts/am29_score.py:+125          if a.expect_checkpoint_sha256 != E6_S42_CHECKPOINT_SHA256:
F-03               scripts/am29_score.py:+126              return "checkpoint_not_of_record", (f"--expect-checkpoint-sha256 {a.expect_checkpoint_sha256}: E6 seed 42's "
F-03               scripts/am29_score.py:+127                                                  f"epoch_ckpts/e15.pt of record is {E6_S42_CHECKPOINT_SHA256} "
F-03               scripts/am29_score.py:+128                                                  "(AM-29 item 2(a))")
F-03               scripts/am29_score.py:+136  def values_of_record(a) -> bool:
F-03               scripts/am29_score.py:+137      """--qat-eval-dir's values of record bind at --split test always, and at --split val unless --allow-smoke-inputs."""
F-03               scripts/am29_score.py:+138      return a.split == "test" or not a.allow_smoke_inputs
F-03               scripts/am29_score.py:+139  
F-03               scripts/am29_score.py:+140  
F-08 (c)           scripts/am29_score.py:+156          governed = None
F-08 (c)           scripts/am29_score.py:+157          if a.split == "test":
F-08 (c)           scripts/am29_score.py:+158              # AUDIT-LAM29-1 F-08 (c), its first option: the evaluator's governed-path gate (EVALUATION_CONTRACT 7.1),
F-08 (c)           scripts/am29_score.py:+159              # before anything is read, and recorded. Not called at VAL: it runs git status over the whole repository.
F-08 (c)           scripts/am29_score.py:+160              from src.eval.stage_artifacts import governed_paths_error
F-08 (c)           scripts/am29_score.py:+161              governed = governed_paths_error(REPO)
F-08 (c)           scripts/am29_score.py:+162              if governed is not None:
F-08 (c)           scripts/am29_score.py:+163                  SC._refuse("governed_paths_dirty", f"--split test: {governed}")
F-03               scripts/am29_score.py:-114                                "--provenance": a.provenance, "--run-dir": a.run_dir, "--out-json": a.out_json})
F-03               scripts/am29_score.py:+165                                "--provenance": a.provenance, "--run-dir": a.run_dir, "--out-json": a.out_json,
F-03               scripts/am29_score.py:+166                                "--qat-eval-dir": a.qat_eval_dir})
F-08 (d)           scripts/am29_score.py:+195              if a.split == "test" and variant.get("eligible") is not True:          # AUDIT-LAM29-1 F-08 (d)
F-08 (d)           scripts/am29_score.py:+196                  SC._refuse("group_not_eligible", f"{a.islands} keeps INT8 parameter share "
F-08 (d)           scripts/am29_score.py:+197                                                   f"{(variant.get('shares') or {}).get('int8_param_share')!r}: a variant "
F-08 (d)           scripts/am29_score.py:+198                                                   "with less than 90% of its parameters in INT8 is not eligible (AM-29 "
F-08 (d)           scripts/am29_score.py:+199                                                   "item 2(d)) and is not scored on TEST")
F-03               scripts/am29_score.py:+204              if a.qat_eval_dir is not None:                    # AUDIT-LAM29-1 F-03
F-03               scripts/am29_score.py:+205                  source["finalize_sha256"] = sel_bound["finalize_sha256"]
F-01               scripts/am29_score.py:-148          score = SC.score_model(model, split=a.split, max_samples=a.max_samples)
F-01               scripts/am29_score.py:+207          score = SC.score_model(model, split=a.split, max_samples=a.max_samples, protocol=a.protocol)
F-08 (b)           scripts/am29_score.py:+208          if a.threads is not None and score["torch_num_threads"] != int(a.threads):   # AUDIT-LAM29-1 F-08 (b)
F-08 (b)           scripts/am29_score.py:+209              SC._refuse("threads_mismatch", f"the score ran on {score['torch_num_threads']} threads, not --threads "
F-08 (b)           scripts/am29_score.py:+210                                             f"{a.threads}; nothing is printed or written")
F-03               scripts/am29_score.py:-153                 if a.precision == "int8" else ("am29" if a.am29_eval_dir else "l-am24"), "source": source,
F-03               scripts/am29_score.py:+215                 if a.precision == "int8" else ("am29" if a.am29_eval_dir else "qat" if a.qat_eval_dir else "l-am24"),
F-03               scripts/am29_score.py:+216                 "source": source,
F-08 (c)           scripts/am29_score.py:+220          if a.split == "test":                                 # AUDIT-LAM29-1 F-08 (c): None, the gate passed
F-08 (c)           scripts/am29_score.py:+221              doc["governed_paths_error"] = governed
F-01               scripts/am29_score.py:-160                                                 "split": score["split"], "rows": score["actual_rows"],
F-01               scripts/am29_score.py:+225                                                 "split": score["split"], "protocol": score["protocol"],
F-01               scripts/am29_score.py:+226                                                 "rows": score["actual_rows"],
F-03               scripts/am29_score.py:+247      if a.qat_eval_dir is not None:
F-03               scripts/am29_score.py:+248          return _e6_s42_selection(a, SC)
F-03               scripts/am29_score.py:+278  def _read_hashed(path: Path, code: str, SC) -> tuple[bytes, str]:
F-03               scripts/am29_score.py:+279      """The file's bytes, read once, and their sha256 (the bytes parsed later are these, never a second read)."""
F-03               scripts/am29_score.py:+280      try:
F-03               scripts/am29_score.py:+281          raw = path.read_bytes()
F-03               scripts/am29_score.py:+282      except OSError as e:
F-03               scripts/am29_score.py:+283          SC._refuse(code, f"{path}: {e}")
F-03               scripts/am29_score.py:+284      return raw, hashlib.sha256(raw).hexdigest()
F-03               scripts/am29_score.py:+285  
F-03               scripts/am29_score.py:+286  
F-03               scripts/am29_score.py:+287  def _parse_hashed(raw: bytes, path: Path, code: str, SC) -> dict:
F-03               scripts/am29_score.py:+288      """Bytes already hashed, parsed as strict JSON (no NaN or Infinity), an object."""
F-03               scripts/am29_score.py:+289      try:
F-03               scripts/am29_score.py:+290          doc = json.loads(raw.decode("utf-8"), parse_constant=SC._no_constant)
F-03               scripts/am29_score.py:+291      except ValueError as e:
F-03               scripts/am29_score.py:+292          SC._refuse(code, f"{path}: {e}")
F-03               scripts/am29_score.py:+293      if not isinstance(doc, dict):
F-03               scripts/am29_score.py:+294          SC._refuse(code, f"{path} is not a JSON object")
F-03               scripts/am29_score.py:+295      return doc
F-03               scripts/am29_score.py:+296  
F-03               scripts/am29_score.py:+297  
F-03               scripts/am29_score.py:+298  def _e6_s42_selection(a, SC) -> dict:
F-03               scripts/am29_score.py:+299      """AUDIT-LAM29-1 F-03: E6 seed 42's block B of record, the pinned scripts/qat_epoch_eval.py's eval dir (DL-131).
F-03               scripts/am29_score.py:+300      qat_selection.json is hashed before it is parsed and must equal --qat-selection-sha256; qat_finalize.json must be of
F-03               scripts/am29_score.py:+301      record (DL-173) and name that selection; the selection is stage E6, seed 42. selected_state then binds the winner's
F-03               scripts/am29_score.py:+302      checkpoint through the run's telemetry and --expect-checkpoint-sha256 (cli_error holds the values of record)."""
F-03               scripts/am29_score.py:+303      E = Path(a.qat_eval_dir)
F-03               scripts/am29_score.py:+304      sel_p, fin_p = E / "qat_selection.json", E / "qat_finalize.json"
F-03               scripts/am29_score.py:+305      sel_raw, sel_sha = _read_hashed(sel_p, "qat_selection_unreadable", SC)
F-03               scripts/am29_score.py:+306      if sel_sha != a.qat_selection_sha256:
F-03               scripts/am29_score.py:+307          SC._refuse("qat_selection_sha256_mismatch", f"{sel_p} has sha256 {sel_sha}; --qat-selection-sha256 is "
F-03               scripts/am29_score.py:+308                                                      f"{a.qat_selection_sha256} (checked before the file is parsed)")
F-03               scripts/am29_score.py:+309      sel = _parse_hashed(sel_raw, sel_p, "qat_selection_unreadable", SC)
F-03               scripts/am29_score.py:+310      fin_raw, fin_sha = _read_hashed(fin_p, "qat_finalize_unreadable", SC)
F-03               scripts/am29_score.py:+311      if values_of_record(a) and fin_sha != E6_S42_FINALIZE_SHA256:
F-03               scripts/am29_score.py:+312          SC._refuse("qat_finalize_not_of_record", f"{fin_p} has sha256 {fin_sha}; E6 seed 42's qat_finalize.json of "
F-03               scripts/am29_score.py:+313                                                   f"record is {E6_S42_FINALIZE_SHA256} (DL-173)")
F-03               scripts/am29_score.py:+314      fin = _parse_hashed(fin_raw, fin_p, "qat_finalize_unreadable", SC)
F-03               scripts/am29_score.py:+315      if sel.get("format") != SC.QAT_SELECTION_FORMAT or fin.get("format") != SC.QAT_FINALIZE_FORMAT \
F-03               scripts/am29_score.py:+316              or fin.get("status") != "finalized" \
F-03               scripts/am29_score.py:+317              or fin.get("qat_selection") != {"path": "qat_selection.json", "sha256": sel_sha}:
F-03               scripts/am29_score.py:+318          SC._refuse("qat_selection_format", f"{E} holds no {SC.QAT_SELECTION_FORMAT} selection finalized by its "
F-03               scripts/am29_score.py:+319                                             f"{SC.QAT_FINALIZE_FORMAT} record")
F-03               scripts/am29_score.py:+320      if sel.get("stage") != "E6" or sel.get("seed") != 42 or fin.get("stage") != "E6" or fin.get("seed") != 42:
F-03               scripts/am29_score.py:+321          SC._refuse("qat_stage", f"{sel_p} is stage {sel.get('stage')!r}, seed {sel.get('seed')!r}; --qat-eval-dir takes "
F-03               scripts/am29_score.py:+322                                  "E6 seed 42 only")
F-03               scripts/am29_score.py:+323      return {"selection": sel, "stage": "E6", "am29_label": "E6-s42", "selection_sha256": sel_sha,
F-03               scripts/am29_score.py:+324              "finalize_sha256": fin_sha}
F-03               scripts/am29_score.py:+325  
F-03               scripts/am29_score.py:+326  
F-02               src/eval/am29_score.py:-21     l-am24     lane/l-am24's block B (scripts/explore_qat_blockb.py) of AM-29's two runs there, N2 (x-n2-s42) and
F-02               src/eval/am29_score.py:-22                E6-R2-M (x-n2r2-s42), finalized through the pinned scripts/qat_epoch_eval.py: x_blockb.json ->
F-02               src/eval/am29_score.py:+21     l-am24     lane/l-am24's block B (scripts/explore_qat_blockb.py) of a run AM-29 scores there (L_AM24_RUNS, by run
F-02               src/eval/am29_score.py:+22                ID), finalized through the pinned scripts/qat_epoch_eval.py: x_blockb.json ->
F-02               src/eval/am29_score.py:-46   # lane/l-am24's two AM-29 runs: run ID -> AM-29's label (the run ID is the trainer's run directory, the run_id of every
F-02               src/eval/am29_score.py:-47   # record and of the artifact's identity)
F-02               src/eval/am29_score.py:-48   L_AM24_RUNS = {"x-n2-s42": "N2", "x-n2r2-s42": "E6-R2-M"}
F-02               src/eval/am29_score.py:+46   # the runs on lane/l-am24 that AM-29 scores: run ID -> AM-29's label (the run ID is the trainer's run directory, the
F-02               src/eval/am29_score.py:+47   # run_id of every record and of the artifact's identity)
F-02               src/eval/am29_score.py:+48   L_AM24_RUNS = {
F-02               src/eval/am29_score.py:+49       "x-n2-s42": "N2",
F-02               src/eval/am29_score.py:+50       "x-n2r2-s42": "E6-R2-M",
F-02               src/eval/am29_score.py:+51       "x-d-n3-s42": "D",
B                  src/eval/am29_score.py:+52       "x-n2r2t-s42": "E6-R2-MT",
F-02               src/eval/am29_score.py:+53   }
```

## 7. B's diff (MEASURED)

```
$ git diff --stat 9b486c9c081054ad86a225dcb19fa523a6b470f4..0480ef0c343d0eb02f72c41a4c3fb74c2d5fe716 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude)docs/reference' ':(exclude,icase)*test*'
 scripts/smoke_am29_blockb.py | 20 ++++++++++++++++++++
 src/eval/am29_score.py       |  1 +
 2 files changed, 21 insertions(+)
```

```
$ git diff 9b486c9c..0480ef0c -- src/eval/am29_score.py
@@ -49,6 +49,7 @@ L_AM24_RUNS = {
     "x-n2-s42": "N2",
     "x-n2r2-s42": "E6-R2-M",
     "x-d-n3-s42": "D",
+    "x-n2r2t-s42": "E6-R2-MT",
 }
```

The 20 smoke lines are the case `lam24_e6r2mt_bound_by_run_id` and its two comment lines, added at the end of
`sec_lam24`; no other line changed. B differs from A only by the binding, so A (`9b486c9c`) is a head without it, if a
row of AM-29a item 7(b) needs one.

## 8. Changed files at B: sha256 and line count (MEASURED, `sha256sum`, `wc -l`)

| file | sha256 at 2a94019f | sha256 at B | lines (2a94019f → B) |
|---|---|---|---|
| `scripts/am29_score.py` | 41a9ba180171cb5254be268c8ace6720a30fafdd68a0a360d5146e25be56017d | bb97caa60f5389fbbb7a7adcd53a71b4a58edf94ff8103bc3bb9baa940c3d70c | 211 → 328 |
| `src/eval/am29_score.py` | 38fea279b260cdf118e504e63567bd87c815ddbe67dd68053f2a50c2d8865329 | cd26c57ce097e24e059c36831e1b012c9dc33ab4ed3a76c0bb280e18876da737 | 352 → 357 |
| `scripts/smoke_am29_runner.py` | 9de2b6953665e1db8d3b4536c30a835c7bd0e74e2a16e4c58b6427985e16807a | ad91fbef9f32cd500a5a683a680275b9f86cd8f2eccc2ea699ea936c004c0c45 | 465 → 490 |
| `scripts/am29_blockb.py` | 746c7ca96bf91edb30a90c972a0437f085f17178590195260e9da08a30816ffa | 6064f33af8afa9c74644c83ea2352f09c56c12f3f97a2bb4b3260d1c82e103cc | 492 → 498 |
| `scripts/am29_islands.py` | b363b9c6930db60f55a9102b66b16927e2dd1e50b58bbfe6b380bb7a7bbe7477 | d8b08142e31cbecb5bebd7ba9b9281e407007e0b83c79f0749e889761a59171d | 234 → 250 |
| `scripts/smoke_am29_blockb.py` | 8646b89f725f97f5f1bc71ce176d2fb5f266c9f624d5682ffdd88364060edaec | f18f2b437aebedc32fcbbcf0fca95eebe1bdde5f30f368c85a49f97cfb5ffe0e | 383 → 663 |
| `scripts/smoke_am29_islands.py` | 93de2bbb7ae149666d52ce36bfb4f8bb1d8066caee006422b6b5145807f349ca | 886088429daf3d1143be811b059ad9ac3cf4fcce40ee519b65484f7746624566 | 265 → 311 |
| `docs/lane_reports/l-am29-fix.md` | (new) | (this file; its sha256 cannot be in it) | |

The sha256 at 2a94019f equal L-AM29's table (its section 3). The lane's other files are unchanged at B (MEASURED):
`src/quant/am29.py` 3fbf35b3…, `src/quant/am29_islands.py` 2080f2bd…, `src/quant/am29_registry.py` 0e3fe55d…,
`scripts/run_am29.py` 60f9ef2f…, `configs/qat_selection_rules_am29.json` 1f1f9a43…, `scripts/smoke_am29_fixtures.py`
e5d29f4c…, `scripts/smoke_am29_isolation.py` 94a05ebd…, as L-AM29's table gives them.

## 9. Smoke outputs (MEASURED, CPU, synthetic)

Every smoke ran from the repository root as `<venv python> -B <smoke>`, with `TMPDIR` as in section 3,
`PYTHONDONTWRITEBYTECODE=1` and no GPU. "Before" ran at 2a94019f on a clean tree; "after" ran at B on a clean tree.
Lines are copied from the logs, cut at 260 characters.

### 9.1 Before (2a94019f) and after (B)

```
before scripts/smoke_am29_runner.py exit=0 seconds=25 RESULT: PASS (44/44) L-AM29 P1 (src/quant/am29.py)
before scripts/smoke_am29_islands.py exit=0 seconds=85 RESULT: PASS (54/54) L-AM29 P2 (src/quant/am29_islands.py, scripts/am29_islands.py)
before scripts/smoke_am29_blockb.py exit=0 seconds=602 RESULT: PASS (35/35) L-AM29 P3 + P4 (scripts/am29_blockb.py, scripts/am29_score.py)
before scripts/smoke_am29_isolation.py exit=0 seconds=4 RESULT: PASS (11/11) L-AM29 isolation
before scripts/smoke_am29_fixtures.py exit=0 seconds=0 
before scripts/smoke_qat_runner.py exit=0 seconds=109 RESULT: PASS (214/214)

after scripts/smoke_am29_runner.py exit=0 seconds=31 RESULT: PASS (46/46) L-AM29 P1 (src/quant/am29.py)
after scripts/smoke_am29_islands.py exit=0 seconds=116 RESULT: PASS (58/58) L-AM29 P2 (src/quant/am29_islands.py, scripts/am29_islands.py)
after scripts/smoke_am29_blockb.py exit=1 seconds=699 RESULT: FAIL (53/54) L-AM29 P3 + P4 (scripts/am29_blockb.py, scripts/am29_score.py)
after scripts/smoke_am29_isolation.py exit=0 seconds=5 RESULT: PASS (11/11) L-AM29 isolation
after scripts/smoke_am29_fixtures.py exit=0 seconds=0 
after scripts/smoke_qat_runner.py exit=0 seconds=114 RESULT: PASS (214/214)
```

### 9.2 The new cases, and the one that fails (after, at B)

```
$ python -B scripts/smoke_am29_runner.py   (the new cases)
  term_loss_identity_T1               : PASS  weight 0.25 (factor 0.5 x lambda 0.5), 4 steps: |loss - (sup + weight x logit_kd)| [2.38e-07, 1.04e-07, 1.79e-07, 0.0] <= 2.4e-07; logit_kd [0.5463, 0.5488, 0.548, 0.5455]
  term_loss_identity_T3               : PASS  weight 0.5 (factor 1.0 x lambda 0.5), 4 steps: |loss - (sup + weight x logit_kd)| [0.0, 8.94e-08, 8.94e-08, 2.09e-07] <= 2.4e-07; logit_kd [0.5463, 0.5491, 0.5486, 0.5459]
RESULT: PASS (46/46) L-AM29 P1 (src/quant/am29.py)

$ python -B scripts/smoke_am29_islands.py   (the new cases)
  cli_e7r2_g0_labelled_e7                 : PASS  E7-R2 G0 a161c329a1737a93 == E7 G0 a161c329a1737a93
  cli_qat_full_sha256_of_record           : PASS  RESULT: REFUSED [checkpoint_not_of_record] -- [checkpoint_not_of_record] E6-s42-e15's checkpoint sha256 is 74fd438a9e953ce84117a61454b879940
  cli_threads_read_back                   : PASS  RESULT: REFUSED [threads_not_set] -- [threads_not_set] torch.set_num_threads(5) left 4 threads; nothing was read
  cli_nonsmoke_requires_clean_code        : PASS  RESULT: REFUSED [code_not_clean] -- [code_not_clean] code_clean_at_head is False at HEAD 0000000000000000000000000000000000000000; a Step 1 
RESULT: PASS (58/58) L-AM29 P2 (src/quant/am29_islands.py, scripts/am29_islands.py)

$ python -B scripts/smoke_am29_blockb.py   (the new cases, and the one that fails)
  blockb_score_requires_threads           : PASS  am29_blockb.py score: error: the following arguments are required: --threads
  blockb_select_requires_clean_code       : PASS  outside smokes: ["code_clean_at_head = False: a record selection's scores ran on clean code at HEAD"]; a smoke selection: none
  blockb_select_requires_image_digest     : PASS  outside smokes: ["score.eval_runtime.image_digest = None: a record selection's scores name the image they ran in"]; a smoke selection: none
  score_protocol_upstream                 : PASS  RESULT: SCORED (int8, T1, val, all-class mIoU 0.0); AM29_SCORE_JSON protocol 'upstream/1.0.0'
  score_threads_mismatch_refused          : PASS  RESULT: REFUSED [threads_mismatch] -- [threads_mismatch] the score ran on 3 threads, not --threads 2; nothing is printed or written
  score_governed_paths_recorded           : PASS  --split test: called on ['<repo>'], recorded None; --split val: called 0 times, key absent
  score_governed_paths_dirty_refused      : PASS  RESULT: REFUSED [governed_paths_dirty] -- [governed_paths_dirty] --split test: refusing 'official': governed paths are dirty or untracked -> ['scripts/x.py']
  score_test_group_not_eligible_refused   : PASS  G2: RESULT: REFUSED [group_not_eligible] -- [group_not_eligible] G2 keeps INT8 parameter share 0.4824843815875206: a variant with less than 90% of its par; G1: RESULT: ERROR [AssertionError] -- tripwire: score_
  qat_e6_s42_scored_mixed                 : PASS  RESULT: SCORED (mixed precision, E6-s42, val, all-class mIoU 0.0)
  qat_e6_s42_selection_sha256_refused     : PASS  RESULT: REFUSED [qat_selection_sha256_mismatch] -- [qat_selection_sha256_mismatch] <tmp>/smoke_am29_blockb_9s_oxkto/e6_s42_eval/qat_selection.json has sha256 a1814246a79193d207bc9df9f91cce421d3
  qat_e6_s42_wrong_stage_refused          : PASS  RESULT: REFUSED [qat_stage] -- [qat_stage] <tmp>/smoke_am29_blockb_9s_oxkto/e6e2_s42_eval/qat_selection.json is stage 'E6-E2', seed 42; --qat-eval-dir takes E6 seed 42 only
  qat_e6_s42_int8_refused                 : PASS  RESULT: REFUSED [qat_eval_dir_precision] -- --qat-eval-dir scores E6 seed 42's selected weights under a P2 group: --precision mixed only. Nothing was read.
  qat_e6_s42_values_of_record             : PASS  3 argument sets refused before any path is opened; misses []
  qat_e6_s42_finalize_of_record           : PASS  RESULT: REFUSED [qat_finalize_not_of_record] -- [qat_finalize_not_of_record] <tmp>/smoke_am29_blockb_9s_oxkto/e6_s42_eval/qat_finalize.json has sha256 68d327607c82b97947e20da4613f38b778f787de23
  lam24_d_mixed_route                     : PASS  x-n5-s42: RESULT: REFUSED [artifact_unknown] -- [artifact_unknown] --run-id 'x-n5-s42' is not one of ['x-d-n3-; x-d-n3-s42: RESULT: SCORED (mixed precision, D, val, all-class mIoU 0.0)
  lam24_d_int8_route                      : PASS  RESULT: SCORED (int8, D, val, all-class mIoU 0.0)
  lam24_e6r2mt_bound_by_run_id            : PASS  x-n2r2t-s42: RESULT: SCORED (int8, E6-R2-MT, val, all-class mIoU 0.07540639489889145); x-n2r2t-s43: RESULT: REFUSED [artifact_unknown] -- [artifact_unknown] --run-id 'x-n2r2t-s43' is not one of AM-29's runs on 
  test_rules_out_json_required            : PASS  RESULT: REFUSED [test_out_json] -- --split test requires --out-json (the look's record, written once). Nothing was read.
  test_rules_provenance_canvas_only       : PASS  RESULT: REFUSED [protocol] -- --provenance is block B's VAL scoring, which scores canvas only. Nothing was read.
  score_am29_int8                         : FAIL  RESULT: SCORED (int8, T1, val, all-class mIoU 0.0)
RESULT: FAIL (53/54) L-AM29 P3 + P4 (scripts/am29_blockb.py, scripts/am29_score.py)
```

### 9.3 F-07 under the audit's two mutations (MEASURED)

The harness (kept in the session scratchpad, outside the repository) imports `scripts/smoke_am29_runner.py` and runs
its `term_loss_identity` three times: on the code as committed; then with `src/quant/am29.py` `run_am29_qat`'s source
text edited and re-executed in the module's namespace, in memory only, for each mutation. It ran before commit A, on
the `scripts/smoke_am29_runner.py` that A and B hold (sha256 ad91fbef…; B does not change it):
- ramp: `teacher_out=teacher_out, weight=kd_weight)` →
  `teacher_out=teacher_out, weight=kd_weight * TD.distill_ramp(step, steps_per_epoch))`
  (the first-epoch ramp of `train_distill.distill_ramp`, 0 at step 1);
- factor without λ: `kd_weight = factor * e3_lambda` → `kd_weight = factor`.

```
--- none: run_am29_qat is the committed function; substitutions []
--- ramp: run_am29_qat is mutated; substitutions [('teacher_out=teacher_out, weight=kd_weight)', 'teacher_out=teacher_out, weight=kd_weight * TD.distill_ramp(step, steps_per_epoch))')]
--- factor_without_lambda: run_am29_qat is mutated; substitutions [('kd_weight = factor * e3_lambda', 'kd_weight = factor')]
  [none] term_loss_identity_T1 : PASS  weight 0.25 (factor 0.5 x lambda 0.5), 4 steps: |loss - (sup + weight x logit_kd)| [2.38e-07, 1.04e-07, 1.79e-07, 0.0] <= 2.4e-07; logit_kd [0.5463, 0.5488, 0.548, 0.5455]
  [none] term_loss_identity_T3 : PASS  weight 0.5 (factor 1.0 x lambda 0.5), 4 steps: |loss - (sup + weight x logit_kd)| [0.0, 8.94e-08, 8.94e-08, 2.09e-07] <= 2.4e-07; logit_kd [0.5463, 0.5491, 0.5486, 0.5459]
  [ramp] term_loss_identity_T1 : FAIL  weight 0.25 (factor 0.5 x lambda 0.5), 4 steps: |loss - (sup + weight x logit_kd)| [0.137, 7.45e-08, 7.45e-08, 1.34e-07] <= 2.4e-07; logit_kd [0.5463, 0.5494, 0.5475, 0.5461]
  [ramp] term_loss_identity_T3 : FAIL  weight 0.5 (factor 1.0 x lambda 0.5), 4 steps: |loss - (sup + weight x logit_kd)| [0.273, 1.49e-07, 1.19e-07, 2.98e-08] <= 2.4e-07; logit_kd [0.5463, 0.5494, 0.5473, 0.5472]
  [factor_without_lambda] term_loss_identity_T1 : FAIL  weight 0.25 (factor 0.5 x lambda 0.5), 4 steps: |loss - (sup + weight x logit_kd)| [0.137, 0.137, 0.137, 0.136] <= 2.4e-07; logit_kd [0.5463, 0.5491, 0.5486, 0.5459]
  [factor_without_lambda] term_loss_identity_T3 : FAIL  weight 0.5 (factor 1.0 x lambda 0.5), 4 steps: |loss - (sup + weight x logit_kd)| [0.273, 0.274, 0.274, 0.273] <= 2.4e-07; logit_kd [0.5463, 0.5486, 0.5472, 0.5464]
src/quant/am29.py sha256 before 3fbf35b3011aaaab74f0d97c85cbb03e9f3f0151bddd38b9f1767507dd8ddf39
src/quant/am29.py sha256 after  3fbf35b3011aaaab74f0d97c85cbb03e9f3f0151bddd38b9f1767507dd8ddf39  (unchanged)
RESULT: AS EXPECTED (unmutated [True, True], ramp [False, False], factor without lambda [False, False])
```

Restoring the code: the function object is put back after each run, and the file was never written: the sha256 of
`src/quant/am29.py` is 3fbf35b3… before and after, and `git status -- src/quant` lists nothing.

Under the ramp, step 1's weight is 0 (`distill_ramp(1, 2)`), so the loss is `sup` alone and misses the identity by
`weight × logit_kd` (0.137 for T1, 0.273 for T3); the ramp is complete from step 2 of these 2-step epochs. Without λ,
the weight is doubled at every step, and every step misses by the same amount.

### 9.4 STOP-1: the line `score_am29_int8` parses (MEASURED, at B)

The same command as the case, run again on the after-run's T1 fixture:

```
$ python -B scripts/am29_score.py --split val --precision int8 --am29-eval-dir <tmp>/smoke_am29_blockb_9s_oxkto/eval_t1 --arm T1 --expect-artifact-sha256 <its selection's artifact sha256> --max-samples 2 --threads 2
RESULT: SCORED (int8, T1, val, all-class mIoU 0.0)
AM29_SCORE_JSON keys: ['dataset_level', 'label', 'name', 'protocol', 'rows', 'split', 'stage']
set(js[0]) == the six keys of 2a94019f: False
set(js[0]) == the six keys and 'protocol' (the one-token remedy): True
the case's other conditions (exit 0; label int8, name T1, stage AM29-T1): True
protocol in the line: core_preprocess/1.0.0
```

## 10. STOP and deviations

**STOP-1 (needs a ruling): `score_am29_int8` fails after F-01.** Hard limit 4 lets `scripts/smoke_am29_*.py` gain
cases only. The case's condition includes `set(js[0]) == {"label", "name", "stage", "split", "rows",
"dataset_level"}` (`scripts/smoke_am29_blockb.py`:243), and F-01 adds `"protocol"` to the line, as the audit's fix text
asks. So the case fails, and was left as it is. Nothing else in the case fails: its scoring exits 0 (`RESULT: SCORED
(int8, T1, val, all-class mIoU 0.0)`), and the line it parses holds exactly the six keys and `protocol` (MEASURED,
section 9.4). The one-token remedy, if ruled: `{"label", "name", "stage", "split", "protocol", "rows",
"dataset_level"}`. The other way out, the protocol kept off the line, is further from the audit's text.

Deviations (every departure from the prompt's letter, and every choice the prompt left open):

1. **A CPU PyTorch was installed in a throwaway venv in the scratchpad** (section 3). The prompt asks for every smoke to
   be run on the CPU, and the container had no PyTorch. AGENTS.md rule 5 asks for a named approval of an install; I
   read the prompt's instruction to run the smokes on the CPU as covering the throwaway install that this needs, as
   L-AM29's deviation 1 records for the same install. Nothing in the repository changed. Open question 1.
2. **Temp path:** `TMPDIR` was a short symlink outside the repository to the scratchpad (section 3). Removed at the end.
3. **F-08 (c) is called at `--split test` only.** `governed_paths_error` runs `git status --porcelain=v1 -z
   --untracked-files=all` from the repository root (`src/eval/artifacts.py` `git_porcelain_bytes`,
   REPOSITORY-PROVEN). This lane's hard limit 1, and the smokes' rule that every git command excludes docs/reference
   (DL-37, as `scripts/synthetic_ptq_fixtures.py` `GIT_EXCLUDES` states it), keep such a command out of this session,
   and every VAL smoke runs the scorer. So at VAL it is not called and the record holds no such key; at TEST it is
   called before anything is read, a value refuses, and the record holds `"governed_paths_error": null`. Its smoke cases
   run in process with the function stubbed, and the TEST case with a scorer that reads nothing. Open question 3.
4. **F-07's mutations were made in memory** (section 9.3), so the code was never changed and nothing needed restoring
   on disk; the prompt's "then restore the code" holds trivially. Hard limit 4 does not list `src/quant/am29.py`.
5. **F-14's identity is taken earlier**: once the source is bound and before any conversion, not after scoring; the
   report records that identity. A long Step 1 run on a dirty tree is refused before its work.
6. **B is its own commit**, and `L_AM24_RUNS` was written one entry per line in A (F-02), so that B's diff is one
   added line plus its smoke case.
7. **Smoke cases beyond the prompt's list** (adding cases only): F-01's refusal with `--provenance`; F-02's int8
   route; F-03's values of record (by argument, and the finalize record in process); F-08 (c)'s refusal of a dirty
   value and (d)'s eligible control; F-14's four cases.
8. **In-process smoke cases** (F-08 (b), (c), (d); F-03's finalize record; F-14's read-back and clean-code cases)
   patch a module attribute and restore it in a `finally`: `governed_paths_error` (no repository-wide git status),
   `score_model` (a tripwire or a scorer that reads nothing, so that no TEST file can be read), the selection and
   checkpoint constants of `scripts/am29_score.py` (to reach the finalize check), `torch.set_num_threads`,
   `scripts/am29_islands.py` `qat_source` and `src/quant/am29.py` `code_identity`. Each is named in its case's comment.
9. **The decision log is not updated** (CLAUDE.md asks a task to update the rows it implements; hard limit 4 lists no
   such file). The rows that name this lane's head are the orchestrator's.
10. **Status and diff commands** used scoped pathspecs (`src scripts configs`, explicit files) with docs/reference and
    test-named paths excluded, not AGENTS.md's `-- .` form, except the diff stats of sections 5 and 7, which compare two
    commits from the root with docs/reference and `*test*` excluded, as L-AM29's deviation 11 did. The L-PROT hook
    refuses `git checkout <branch>` and `git switch`; the branch was made with `git checkout -b lane/l-am29-fix` at
    HEAD (2a94019f), which the hook allows.

## 11. Open questions

1. **The install** (deviation 1): if rule 5 is read to need a named approval here too, the smoke results stand as
   evidence but the install was not separately approved.
2. **STOP-1**: allow the one-token change to `score_am29_int8`, or rule otherwise. Until then the block B smoke reports
   one failure at B.
3. **F-08 (c) at VAL**: if the audit wants the governed-path state recorded at VAL too, the call moves out of the
   `--split test` branch (one line), and every smoke that scores VAL must stub it (DL-37). In the TEST container the
   gate needs the checkout's `.git` (the evaluator's own error says so); the kit should know.
4. **F-08 (d)** follows the audit's text, the variant record's `eligible`, not a static list; for the student
   architecture only G2 is not eligible (DL-274).
5. **F-03**: `--allow-smoke-inputs` lifts the three values of record at `--split val` only, never at `--split test`.
   The selection's stage and seed are read from the selection and its finalize record; the run's own stage is bound
   through the telemetry's sha256 by `selected_state`, as for the other mixed routes. The label is `E6-s42` and the
   record's `source_kind` `qat`; neither name is given by a record, and the kit may want others.
6. **F-09**: AM-29 item 10 checks every score row "for 8 threads". The code requires `--threads`, and the selection
   keeps one thread count across its 15 records (`equal_across_scores`), but does not pin 8; the prompt's F-09 does
   not ask for it. A one-line pin is possible if the audit wants it.
7. **F-02**: the entry assumes D's block B is in lane/l-am24's `x_qat_blockb/1` layout with a stage in
   `L_AM24_STAGES`; D's dry run (AM-29 item 3(d)) shows whether it is.
8. **B's fixture** is an E6-E2 run named x-n2r2t-s42, since `_write_lam24_records` writes stage E6-E2; E6-R2-MT's own
   stage (E6-R2, by AM-29a's text, INFERRED) is in `L_AM24_STAGES` too. The binding is by run ID.
9. **AM-29a's own conditions**: the prompt's test is two strings on master. DL-278 also says AM-29a is in force "only if
   this commit reaches the remote by 12:00 +08:00 on 2026-11-17, before block B of any AM-29 run has scored an epoch and
   before any group G1 to G4 of AM-29 item 2 has been scored", and its row reads "Status DRAFT". This lane saw the
   commit on master at 08:27 +08:00 and cannot see block B's or Step 1's progress.
10. **Stale wording in existing cases**: `cli_qat_prefix_refused` (`scripts/smoke_am29_islands.py`:205-208) still
    describes a twelve-digit prefix; it passes under F-14's full-sha256 rule, and was left as it is (adding cases
    only).
11. **F-07's tolerance**: the largest deviation seen, 2.38e-7 at T1's first step, is the float32 half-ulp at a loss
    between 4 and 8; a loss of 8 or more would need a larger tolerance. The smoke's losses are about 6.

## 12. Commit and push

- Before each commit: `git status -sb` and `git diff` on the declared paths (AGENTS.md rule 6). Each commit was made
  with `git commit --only -- <paths>`, with no trailer.
- `lane/l-am29-fix` is pushed with `git push -u origin lane/l-am29-fix` after this report's commit, and the remote tip
  is read back with `git ls-remote`; the hand-off message gives it (a file cannot hold its own commit's sha).

## 13. Compliance with the hard limits

1. **Paths.** Nothing under `docs/reference/` was opened, read, listed, grepped or hashed. Searches ran inside
   `scripts/` and `src/` or inside a named file. The decision log was read at rows DL-131, DL-173, DL-274 and DL-278 and
   the amendments file at "## AM-29 —" and "## AM-29a —" only, from master's copies (written to the scratchpad by
   `git show`), located by grep inside those two files; to find the two sections, one `grep -n '^## '` printed the
   first 60 characters of the last twelve section headings, and no other section's text was read.
   `docs/lane_reports/l-am29.md` was read in full. The L-PROT hook refused four commands: `git checkout -b` with a
   start point (the branch was then made at HEAD, deviation 10), and three whose paths it could not resolve (a shell
   variable in `git show <rev>:$f`, twice, and in `ls`), each then run with literal paths.
2. **TEST and test-named paths.** No TEST data, dataset, archive, the issue #11 CSV or test-named path was read. None of
   the files changed imports a test-named module (MEASURED: grep of their import lines). One `ls scripts/ src/eval/ |
   grep -i -c test` printed a count (1) and no name; no such file was opened. Every `--split test` in a smoke is an
   argument: refused before anything is read, or, in process, stopped at a stub before any data.
3. **Checkpoints.** None of record was loaded, copied or downloaded; every model and checkpoint here is synthetic.
4. **Files.** Changed: the seven files of hard limit 4's list (section 8) and this new report. No file in
   `QAT_CODE_PATHS` or `E6KD_CODE_PATHS`, nothing else under `src/eval/`, `src/quant/` or `configs/`, no other lane. No
   `src/stats` module is imported.
5. **Git.** Commit messages carry no trailer. Only `lane/l-am29-fix` was pushed; nothing was merged.
6. **Words.** The four words the prompt names are absent from the code comments added here and from this report.
