# L-TEACHER-DIAG P41 follow-up — report (lane/teacher-diag-p41)

Authority: the GO of Sun 2026-10-04 22:20 (P41 follow-up, DG-3 R3), the STOP before step 4 (accepted), and
GO-2 of 23:31 (SCOPE extended by h1–h4, SCOPE d ruled "a tested function", one install approved).
Labels: MEASURED (run in this cloud session unless another runner is named), REPOSITORY-PROVEN,
DOCUMENTED, INFERRED. Nothing in this lane ran on real weights; the cloud has no dataset, checkpoint or
GPU.

## 1. Tip, commits, diff and PB-1a

- **Base:** 60c1417cdc5462103837f6ff86915108dc979ad5 (tree 316a8b64…), master at step 1 (MEASURED,
  `git rev-parse origin/master`). Master later moved to 6067e662b1d5f4036b38cef7179f8cd94b56e50e
  (stats-official); per GO-2 the base stands and nothing was rebased. No CODE_FILES path and none of
  this lane's files differ between 60c1417 and 6067e66 (MEASURED, `git diff --name-only`).
- **Tip:** the commit that adds this file; its parent is 328a426d0382add9f16e67ca352d6408aba9ae4d.
  Every smoke in §3 ran at 328a426; the tip differs from it by this file only.
- **Commits** (no trailers):

  | # | Commit | Content |
  |---|---|---|
  | 1/7 | ae1f553 | h1–h3: stub fixtures for K-part's strict load and the R6 flag; h2's case |
  | 2/7 | 0d691c9 | SCOPE a (both load lines pass the verified sha) and h4 (the trainer's pair exits 2) |
  | 3/7 | ef731b4 | SCOPE b (P8: the 12 names in order; empty containers refused in real mode) |
  | 4/7 | c7d4706 | SCOPE c (drift pin re-pinned at 60c1417; D2's mirror notes K-part's R6) |
  | 5/7 | 328a426 | SCOPE d (one tested reproduction gate for D2 VAL and D1's VAL part) |
  | 7/7 | this commit | the report (6/7 was held for harness fixes; the harness passed without any) |

- **Diff** (`git diff --stat 60c1417cdc5462103837f6ff86915108dc979ad5 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude,icase)*test*'`,
  MEASURED at the tip; DIAG-owned files and this report only):

  ```
   docs/lane_reports/teacher-diag-p41.md   | 447 ++++++++++++++++++++++++++++++++
   scripts/score_teacher_train.py          |   4 +-
   scripts/smoke_score_teacher_train.py    |   7 +-
   scripts/smoke_teacher_calibration.py    |  70 ++++-
   scripts/smoke_teacher_d1.py             | 301 +++++++++++++++++----
   scripts/smoke_teacher_diag_drift.py     |   5 +-
   scripts/smoke_teacher_diag_mutations.py | 105 ++++++--
   scripts/smoke_teacher_real_arch.py      |  12 +-
   scripts/teacher_d1_nmf_sensitivity.py   |  16 +-
   scripts/teacher_d2_calibration.py       |  17 +-
   scripts/teacher_diag_fixtures.py        |  20 +-
   src/eval/teacher_diag.py                | 108 ++++++--
   12 files changed, 980 insertions(+), 132 deletions(-)
  ```

  Untouched (MEASURED, same diff): docs/DECISION_LOG.md, docs/PREREGISTRATION_AMENDMENTS.md,
  docs/lane_specs/, requirements*, AGENTS.md, CLAUDE.md, .claude/, scripts/evaluate_model.py,
  src/distill/, src/training/, src/eval/metrics.py, src/eval/model_loading.py, src/eval/stage_artifacts.py.
- **PB-1a outputs** (MEASURED):
  - pre-checks: `git status --porcelain=v1 -- . ':(exclude)docs/reference/reference.pdf'` printed nothing;
    `git ls-remote --heads origin lane/teacher-diag-p41` printed nothing;
  - the chain exited 0;
  - post-checks: `git rev-parse HEAD` → 60c1417cdc5462103837f6ff86915108dc979ad5; `git symbolic-ref HEAD` →
    refs/heads/lane/teacher-diag-p41; status, `diff --stat` and `diff --cached --stat` against 60c1417
    printed nothing; the protected PDF's status probe printed nothing.

## 2. Install (GO-2 ruling 3)

- `pip install --no-deps torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu` ran once
  and installed torchvision 0.16.0+cpu (MEASURED).
- `import torchvision` then fails: `ModuleNotFoundError: No module named 'requests'` (MEASURED).
  torchvision 0.16 imports `requests` in `torchvision/datasets/utils.py`, and pip records
  `Requires: numpy, pillow, requests, torch`. `--no-deps` left it out, and installing it is a second
  install that GO-2 did not approve, so it was not installed (§8 question 1).
- Venv otherwise as reported for fc5f914: Python 3.11.15, torch 2.1.0+cpu, numpy 1.26.4, scipy 1.11.4,
  mmengine 0.10.7, mmcv 2.1.0, mmseg 1.2.2, statsmodels 0.14.6, pandas 3.0.3 (MEASURED).

## 3. Baseline (60c1417, before any edit) against the tip

All MEASURED in the venv; logs kept in the session scratchpad. The baseline was run twice for the three
K-part smokes: before the install and after it.

| Smoke | fc5f914 | Baseline at 60c1417 | Tip (328a426 + this file) |
|---|---|---|---|
| smoke_teacher_diag_drift | 8 | RESULT: FAIL (7/8) — P34 got d1c809d4… (the pin was b3313391…) | RESULT: PASS (8/8) |
| smoke_teacher_d1 | 137 | RESULT: FAIL (35/37) — `seam_cases` raised TeacherStateDictMismatch (missing=4, shape=2); `d1_cases` raised CliError teacher_ckpt_sha256_required | RESULT: PASS (145/145) |
| smoke_teacher_calibration | 48 | RESULT: FAIL (24/25) — `d2_cases` raised CliError teacher_ckpt_sha256_required | RESULT: PASS (52/52) |
| smoke_score_teacher_train | 13 | RESULT: FAIL (0/1) — TeacherStateDictMismatch | RESULT: PASS (13/13) |
| smoke_teacher_d3 | 12 | RESULT: PASS (12/12) | RESULT: PASS (12/12) |
| smoke_teacher_d4 | 32 | RESULT: PASS (32/32) | RESULT: PASS (32/32) |
| smoke_teacher_real_arch | 6 | RESULT: PASS (6/6) | RESULT: PASS (7/7) |
| smoke_teacher_diag_mutations | 173, 89/89 killed | RESULT: FAIL (0/1) — the harness raised CliError teacher_ckpt_sha256_required in its setup | RESULT: PASS (199/199), mutations killed: 105/105 |
| smoke_frozen_blobs | 10/10 | RESULT: FROZEN BLOBS OK (10/10) | RESULT: FROZEN BLOBS OK (10/10) |
| smoke_teacher (K-part 56/56) | — | no RESULT line: torchvision missing; after the install: `requests` missing | no RESULT line: `requests` missing (not run) |
| smoke_eval_teacher (K-part 60/60) | — | no RESULT line: torchvision missing; after the install: `requests` missing | no RESULT line: `requests` missing (not run) |
| smoke_teacher_strict_load (K-part 42/42) | — | RESULT: FAIL (39/42), both times; the 3 failures are the student build (torchvision, then `requests`) | RESULT: FAIL (39/42), the same 3 (`requests`) |

- Every DIAG count at the tip is at least its fc5f914 count.
- The three K-part smokes are K-part-owned. Their different counts are reported, not fixed (GO-2 ruling 3).
  Their failures are import failures of the student stack, before any K-part assertion runs. The K-part
  counts of record remain Ice's local gate (MEASURED by Ice).

## 4. Mutation table — new rows and rows changed in place

Appended after M89, so M01–M89 keep their numbers. Every row is killed at the tip (MEASURED).

| # | Mutation | Owner | Killed by |
|---|---|---|---|
| M90 | h1: the stub checkpoint written as zeros, not the factory's state | teacher_diag_fixtures.write_stub_ckpt | h1 the stub checkpoint holds exactly the factory's state_dict; the loaded stub equals the factory's state |
| M91 | h3: the reference builder omits --teacher-ckpt-sha256 | teacher_diag_fixtures.reference_artifact | h3 the reference builder passes --teacher-ckpt-sha256 (built and recorded; a wrong sha refused) |
| M92 | a: the kd load line drops expected_sha256 | teacher_diag.load_teacher | a both load lines pass the verified sha (expected_sha256 recorded in the kd and evaluator form) |
| M93 | a: the evaluator load line drops expected_sha256 | teacher_diag.load_teacher | a both load lines pass the verified sha (…) |
| M94 | h4: TeacherChecksumMismatch dropped from the refusal classes | teacher_diag.exit_code_for | h4 a checkpoint changed after validation is refused with exit 2 (TeacherChecksumMismatch) |
| M95 | h4: TeacherStateDictMismatch dropped from the refusal classes | teacher_diag.exit_code_for | h4 a checkpoint the strict load refuses is refused with exit 2 (TeacherStateDictMismatch) |
| M96 | P8: the order check weakened to a set comparison | teacher_diag.require_provenance_names | P8 names and order on stand-ins: 11 fields, the 12 reordered or one renamed refused; the merged 12 pass |
| M97 | P8: an empty container no longer counts as empty | teacher_diag.provenance_value_empty | P8 an empty container is refused in real mode after the load and in the record (DG-5 ruling 1) |
| M98 | P8: the after-load names check removed | teacher_diag.after_load_checks | P8 a real run refuses None, "", {} or [] and a missing field after the load, naming them |
| M99 | P8: the record's keys check removed | teacher_diag.teacher_record | P8 the written record's keys and dataclasses.fields must be the K-part fields in order |
| M100 | P8: the record site's dataclass check removed | teacher_diag.teacher_record | P8 the written record's keys and dataclasses.fields must be the K-part fields in order |
| M101 | d: the real branch inverted (the artifact in real mode, R3 in stub mode) | teacher_diag.reproduction_gate | d the real branch's reference is R3 itself, not the --val-reference artifact (finding 27) |
| M102 | d: a changed digit of R3 in the gate (…214294 → …215294) | teacher_diag.reproduction_gate | d R3_VAL_MIOU is the literal 0.38576993346214294 and PAIRING's value; tolerance 1e-5 |
| M103 | d: a changed tolerance (2e-5) | teacher_diag.reproduction_gate | d real mode: 9e-6 from R3 passes and 1.1e-5 is refused, on both sides |
| M104 | d: a None value passes | teacher_diag.reproduction_gate | d None and NaN are refused |
| M105 | d: a NaN value passes (a negated comparison) | teacher_diag.reproduction_gate | d None and NaN are refused |

Rows changed in place (the same mutation, a new anchor or killer):

| # | Mutation | Change |
|---|---|---|
| M32 | P2 parameter guard removed | killer is now h2's case: its own big_factory checkpoint, and the refusal names the parameter count |
| M33 | P8 provenance names check removed (before any file is read) | anchor `if real:`; killer is the names-and-order case |
| M41 | P18 the reproduction gate removed | anchor `passed = gate["passed"]` in D2's `_run` |
| M62 | P8 empty-field check removed | anchor `empty = [n for n in fields if provenance_value_empty(...)]`; killer is the None/""/{}/[] case |
| M63 | P8 record blank check removed | anchor `bad = [n for n in TEACHER_PROVENANCE_FIELDS if provenance_value_empty(...)]`; killer is the empty-container case |

The harness counts: 93 killers (83 before), 105 mutations (89 before), 1 coverage check; 199/199
(MEASURED).

## 5. SCOPE a–d and h1–h4 (file:line at the tip)

**a. Both load lines pass the verified sha** (0d691c9).
- src/eval/teacher_diag.py:467 `sha = inputs.sha256`, the --teacher-ckpt-sha256 that
  `verify_teacher_inputs` checked against the file at :418.
- src/eval/teacher_diag.py:474 `load_frozen_teacher(str(inputs.ckpt), config_path=str(inputs.config),
  expected_sha256=sha, **kw)`.
- src/eval/teacher_diag.py:478 `load_teacher_model(resolved, config_path=str(inputs.config),
  expected_sha256=sha, **kw)`.
- Stub calls keep `builder=segnext_builder(model_factory=...)` through `kw`.
- Case: scripts/smoke_teacher_d1.py:548 `case_expected_sha` (check at :718). Mutations M92 and M93.

**b. P8** (ef731b4).
- src/eval/teacher_diag.py:72 `TEACHER_PROVENANCE_FIELDS`, the 12 merged names in order; :75
  `EXPECTED_PROVENANCE_FIELDS = len(...)`.
- :339 `require_provenance_names`: same names, same order, or Refused naming the missing and extra.
- :349 `provenance_value_empty`: None, "", {} and [] are empty. It is applied in real mode only.
- Where it applies:
  - :356–361, before any file is read: `dataclasses.fields(TeacherProvenance)`;
  - :537–542, after the load: the loaded object's fields and values;
  - :558–560, at the record site: `dataclasses.fields(TeacherProvenance)`, then the record's keys in
    order, then the values.
- Cases in scripts/smoke_teacher_d1.py, each checked at :890–898:
  - :406 `case_p8_constant`: the constant equals this smoke's literal list and the merged dataclass;
  - :413 `case_p8_names`: 11 fields, a reordered or a renamed field;
  - :439 `case_p8_nonempty`: None, "", {} and [], plus an 11-field object;
  - :456 `case_p8_empty_container`: DG-5 ruling 1, after the load and in the record; stub mode
    unchecked;
  - :477 `case_p8_record`: a record lacking a key, a reordered record, a reordered dataclass.
- The real-run gate order case, smoke_teacher_d1.py:1268: at K-part a well-formed real run passes P8
  and is refused by the commit binding; an 11-field dataclass is refused by P8 first.
- Real path: scripts/smoke_teacher_real_arch.py:118. With no builder, mmseg 1.2.2, the thesis
  config and random weights, both load forms pass the real-mode after-load and record checks, and
  the record holds the 12 fields in order, none None, "", {} or [], with expected_sha256 equal to the
  verified sha (MEASURED). The STOP condition of SCOPE b ("a field legitimately null on the real
  path") did not occur. For the checkpoint of record this is INFERRED: only the checkpoint fields
  depend on the file.
- Mutations: M33, M62, M63 (in place) and M96–M100.

**c. P34 and D2's mirror** (c7d4706).
- scripts/smoke_teacher_diag_drift.py:32 pins `d1c809d4ad588d6efffd7fd98a1240fa22ade18d729ec6f187963125de97de2a`,
  `sha256(inspect.getsource(scripts.evaluate_model.run))` at 60c1417 (MEASURED; it was b3313391… at
  afd2d33).
- scripts/evaluate_model.py is not edited. The differences the mirror takes over:

  | K-part hunk (scripts/evaluate_model.py) | Behaviour in run() | D2's mirror |
  |---|---|---|
  | `@@ -251` (now :273–280) | `validate_teacher_artifact(args.checkpoint, expected_sha256=args.teacher_ckpt_sha256)`: the checkpoint's identity, hash first (R6), before its structure and before any dataset | src/eval/teacher_diag.py:469 `validate_teacher_artifact(inputs.ckpt, expected_sha256=inputs.sha256)`, in the seam since fc5f914. The file's own hash is also compared with the flag before any data is read (:418, P5) |
  | `@@ -308` (now :331–335) | `load_teacher_model(..., expected_sha256=args.teacher_ckpt_sha256)`: K-part's loader hashes the file again before it parses it, and the default builder loads strictly | src/eval/teacher_diag.py:478 (SCOPE a). The KD form :474 follows src/training/train_distill.py:1300–1302 |
  | `validate_cli_args` (:210–225; called by run() at :243, outside the pinned source) | the teacher stage requires --teacher-ckpt-sha256, 64 lowercase hex (`sha256_format_error`) | `check_teacher_flags`, src/eval/teacher_diag.py:305: required, `^[0-9a-f]{64}$` fullmatch, before any file is read. The same rule, in the seam since fc5f914. "Refused on other stages" does not apply: D2 is the teacher stage only |
  | (follows from `@@ -308`, not a run() line) | a checksum or strict-load mismatch raises; train_distill.py:1303 refuses that pair with exit 2 | h4, src/eval/teacher_diag.py:943–944 |

- D2's docstring (scripts/teacher_d2_calibration.py:10–13) records the pin and the R6 inheritance.

**d. Record-role reproduction gate** (328a426; GO-2 ruling 2).
- src/eval/teacher_diag.py:777 `reproduction_gate(value, ref_miou, *, stub, rule)`.
  - :785: the reference is R3_VAL_MIOU in real mode and the --val-reference value in stub mode.
  - :787: `passed` needs |value − reference| ≤ 1e-5. None fails, and NaN fails because `NaN <= 1e-5`
    is false.
- Callers, with behaviour and the written gate fields unchanged (the calibration smoke's output checks
  pass, MEASURED):
  - scripts/teacher_d2_calibration.py:213–216;
  - scripts/teacher_d1_nmf_sensitivity.py:391–393.
- check_val_reference still requires the artifact's own mIoU within 1e-5 of R3 (teacher_diag.py:764).
- Cases in scripts/smoke_teacher_calibration.py (`gate_cases`, :309):
  - :274: the real branch returns R3 even when the artifact is 2e-5 away; the artifact's own value is
    refused;
  - :287: R3 equals the literal 0.38576993346214294 and PAIRING's value, with tolerance 1e-5;
  - :296: 9e-6 passes and 1.1e-5 is refused, on both sides;
  - :303: None and NaN are refused.
- Mutations M101–M105, and M41's anchor.

**h1.** scripts/teacher_diag_fixtures.py:179–190. `write_stub_ckpt(..., factory=None)` writes
`(factory or stub_factory)().state_dict()` exactly (:186). The padded checkpoint keeps its extra tensor
and is refused by stat before any load (the existing P2 case). Case: smoke_teacher_d1.py:507
`case_stub_ckpt_exact` (check at :715). Mutation M90.

**h2.** smoke_teacher_d1.py:592 `case_big_model` (check at :847); harness killer at
smoke_teacher_diag_mutations.py:396.
- It writes a big_factory checkpoint to the case's temp folder (MEASURED 8,010,562 bytes, under 16 MiB).
- It requires the refusal text `a model with <n> parameters`, where n = 2,002,036 > 10**6 (MEASURED).
- M32 is killed.

**h3.** teacher_diag_fixtures.py:294–302: `reference_artifact(out_dir, ckpt, sha256, ...)` passes
`--teacher-ckpt-sha256` at :302. smoke_teacher_calibration.py:243 does the same in `_reference`.
- Case: smoke_teacher_d1.py:524 `case_reference_sha` (check at :1274). The evaluator builds the
  reference and records the sha; a wrong sha is refused (`teacher_hash_mismatch`).
- Mutation M91.

**h4.** src/eval/teacher_diag.py:943–944 add `TeacherChecksumMismatch` (src/distill/teacher.py) and
`TeacherStateDictMismatch` (src/distill/segnext_teacher.py) to `_REFUSAL_CLASSES`. `TeacherChecksumFormatError`
stays unexpected (exit 4).
- Cases:
  - smoke_teacher_d1.py:561 `case_h4_checksum`: a file rewritten right after validate_teacher_artifact
    passed gives exit 2;
  - :581 `case_h4_state_dict`: big_factory's state into the default stub gives exit 2;
  - the check at :844: the format error stays `None` (exit 4).
- Mutations M94 and M95.

## 6. Deferred local commands (AM-18's order)

None runs before this lane is merged and P is logged in DL-<s>. PowerShell on Ice's machine.

**Rules for every command:**
- The pinned image, `--network none`, `python -B`, working directory /work/repo, which is the clone at
  P.
- Every input mount is `--mount type=bind,source=<host path>,target=<container path>,readonly`.
- The one output mount is `--mount type=bind,source=<host out>,target=/out`.
- Never `-v`: on this machine PowerShell `-v` binds did not apply, and the container saw the image's
  own files (MEASURED by Ice, K-PART erratum 8).
- Unlike `-v`, `--mount` refuses a host source that does not exist, so `OUTM` below creates each output
  folder first.
- Replace every `<…>` before running.

**The in-container check.** One in-container check runs before each data or clone command, in the same
mounts as that command:
- `/work/repo` HEAD prints P;
- with `data`, `/data/images/train` holds 5,367 files and `/data/images/val` 846 (counts only, never
  names);
- every other input path named exists.

Any difference prints a STOP line and exits 1, and that is a STOP. The Python one-liner holds no `"`,
`$` or backtick, so PowerShell passes it unchanged (INFERRED; its logic MEASURED on Linux with the
container paths substituted).

```powershell
$IMG   = "ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b"
$P     = "<P>"                  # 40 hex: this lane's merged tip, logged in DL-<s>
$CLONE = "<CLONE_AT_P>"         # P38's dedicated sparse clone at P, never edited
$DS    = "<TRAINVAL_DS>"        # holds images\train, images\val, annotations\train, annotations\val
$CKD   = "<CKPT_DIR>"           # holds the iteration-24000 checkpoint of record
$OUT   = "<OUT>"                # outside the clone and outside the working checkout
$AM17B = "C:\Users\admin\plantseg_runs\data_20261002\am17b"
$T_RESCORE  = "$AM17B\teacher_val_cpu"      # run teacher_val_cpu_am17b2c; all-class mIoU 0.38576993346214294
$E1_RESCORE = "$AM17B\e1_s42_val_cpu"       # run e1_s42_val_cpu_am17b2c
$STRATA   = "reports/strata/train_strata_v1.json"
$GAP      = "reports/derived/gap_val_20261003T014324Z.json"
$PERCLASS = "reports/derived/perclass_gap_val_20261003T014403Z.json"
$CFG      = "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
$CKPT_FILE = "/ckpt/<ITER_24000_CKPT>.pth"
$TEACHER_SHA = "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e"

function RO([string]$src, [string]$dst) { @("--mount", "type=bind,source=$src,target=$dst,readonly") }
function OUTM([string]$src) {
    New-Item -ItemType Directory -Force -Path $src | Out-Null
    @("--mount", "type=bind,source=$src,target=/out")
}
$RUN  = @("run","--rm","--network","none",
          "-e","PLANTSEG_IMAGE_DIGEST=sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b",
          "-e","PLANTSEG_DATA_ROOT=/data","-e","PYTHONPATH=/work/repo","-e","GIT_OPTIONAL_LOCKS=0",
          "-e","GIT_CONFIG_COUNT=1","-e","GIT_CONFIG_KEY_0=safe.directory","-e","GIT_CONFIG_VALUE_0=/work/repo",
          "-e","EXPECT_P=$P") + (RO $CLONE "/work/repo") + @("-w","/work/repo")
$DATA = (RO "$DS\images\train" "/data/images/train") + (RO "$DS\images\val" "/data/images/val") +
        (RO "$DS\annotations\train" "/data/annotations/train") + (RO "$DS\annotations\val" "/data/annotations/val")
$CKPT = RO $CKD "/ckpt"
$CK   = @("--teacher-ckpt",$CKPT_FILE,"--teacher-ckpt-sha256",$TEACHER_SHA,"--teacher-config",$CFG,
          "--teacher-role","record")
$PIN  = @("--script-commit",$P,"--script-commit-dl-id","DL-<s>")
$CHECK = "import os,subprocess,sys;a=sys.argv[1:];d=a[:1]==['data'];a=a[1:] if d else a;h=subprocess.run(['git','-C','/work/repo','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip();r=[('HEAD',h,os.environ.get('EXPECT_P'))];c=lambda p:sum(1 for e in os.scandir(p) if e.is_file()) if os.path.isdir(p) else 'missing';r+=[(p,c(p),n) for p,n in (('/data/images/train',5367),('/data/images/val',846))] if d else [];r+=[(p,os.path.exists(p),True) for p in a];[print(k,v,'OK' if v==w else 'STOP') for k,v,w in r];sys.exit(0 if all(v==w for k,v,w in r) else 1)"
```

**Clone (P38)**, on the host:
```
git clone --no-checkout https://github.com/ainsleydeluna/plantseg-thesis.git <CLONE_AT_P>
git -C <CLONE_AT_P> sparse-checkout set --no-cone "/*" "!/docs/reference/"
git -C <CLONE_AT_P> checkout --detach <P>
```
Omit `--no-cone` if git rejects it.

**0. Command 0 (P36, R4).** Needs DL-<s> (P logged). No weights.
```powershell
docker @RUN $IMG python -B -c $CHECK "/work/repo/$CFG" "/work/repo/$STRATA" "/work/repo/$GAP" "/work/repo/$PERCLASS"
docker @RUN $IMG git --version
docker @RUN $IMG git -C /work/repo rev-parse HEAD                                    # must print $P
docker @RUN $IMG git -C /work/repo status --porcelain=v1 --untracked-files=all       # must print nothing
docker @RUN $IMG python -B -c "import src.stats.eligibility, src.stats.val_artifacts"
docker @RUN $IMG python -B -c "from src.eval.teacher_diag import code_provenance as c; d = c(); print(d['head'], d['code_digest'], d['code_files_clean_at_head'])"
foreach ($s in "smoke_teacher_calibration","smoke_teacher_d1","smoke_score_teacher_train","smoke_teacher_d3","smoke_teacher_d4","smoke_teacher_diag_drift","smoke_teacher_real_arch","smoke_teacher_diag_mutations","smoke_frozen_blobs") { docker @RUN $IMG python -B scripts/$s.py }
docker @RUN @DATA $IMG python -B -c $CHECK data
docker @RUN @DATA $IMG python -B scripts/hash_split_files.py --data-root /data --names-check   # R4: prints two counts only
```
- Any failure, or any names-check count above 0, is a STOP before any execution of record.
- The code-digest line prints the digest DL-<s> records (§7).
- Command 0 also decides the `safe.directory` form. If the image's git ignores the environment form,
  the fallback is the data session's recipe.

**1. D4 hasher.** Needs DL-<a> (the audit row), DL-<s> and a passed command 0.
```powershell
$M = OUTM "$OUT\d4_hash"
docker @RUN @DATA @M $IMG python -B -c $CHECK data "/work/repo/$STRATA" /out
docker @RUN @DATA @M $IMG python -B scripts/hash_split_files.py --data-root /data --strata $STRATA --out /out/hashes @PIN
```
Afterwards, log both printed sha256s as DL-<h>.

**2. D4 reducer.** Needs the gap output (on master through c689634) and DL-<h>.
```powershell
$DUP_SHA = "<sha256 of val_train_duplicates.json, from DL-<h>>"
$IN = (RO $T_RESCORE "/in/teacher") + (RO $E1_RESCORE "/in/e1") + (RO "$OUT\d4_hash" "/in/d4")
$M  = OUTM "$OUT\d4_dedup"
docker @RUN @IN @M $IMG python -B -c $CHECK /in/teacher/summary.json /in/e1/summary.json /in/d4/hashes/val_train_duplicates.json "/work/repo/$GAP" /out
docker @RUN @IN @M $IMG python -B scripts/dedup_val_scores.py --teacher /in/teacher --e1 /in/e1 --gap-output $GAP --duplicates /in/d4/hashes/val_train_duplicates.json --duplicates-sha256 $DUP_SHA --out-dir /out @PIN
```
The two re-score folders are the ones the gap output names in `inputs.teacher.dir_name` and
`inputs.e1.dir_name`. The reducer's own pairing checks (P23) refuse any other pair.

**3. D2 VAL, item 1(b).** Needs DL-<a>, DL-<s> and R5's re-score row.
```powershell
$IN = $CKPT + (RO $T_RESCORE "/in/ref")
$M  = OUTM "$OUT\d2_val"
docker @RUN @DATA @IN @M $IMG python -B -c $CHECK data $CKPT_FILE /in/ref/summary.json "/work/repo/$CFG" /out
docker @RUN @DATA @IN @M $IMG python -B scripts/teacher_d2_calibration.py @CK --purpose item1 --val-reference /in/ref --artifact-dir /out/artifact --out-dir /out @PIN
```

**4. D2 TRAIN.** Needs DL-<a>, DL-<s> and the strata row (the file is on master through c689634).
```powershell
$M = OUTM "$OUT\d2_train"
docker @RUN @DATA @CKPT @M $IMG python -B -c $CHECK data $CKPT_FILE "/work/repo/$CFG" "/work/repo/$STRATA" /out
docker @RUN @DATA @CKPT @M $IMG python -B scripts/score_teacher_train.py @CK --strata $STRATA --artifact-dir /out/artifact --out-dir /out @PIN
```

**5. D3.** Needs the TRAIN JSON's sha256 logged (DL-<t>). S3's per-class table is on master through
c689634.
```powershell
$TRAIN_JSON = "teacher_train_scores_<UTC>.json"          # the name step 4 printed
$IN = RO "$OUT\d2_train" "/in/train"
$M  = OUTM "$OUT\d3"
docker @RUN @IN @M $IMG python -B -c $CHECK "/in/train/$TRAIN_JSON" "/work/repo/$PERCLASS" "/work/repo/$STRATA" /out
docker @RUN @IN @M $IMG python -B scripts/teacher_d3_perclass_strata.py --train-scores "/in/train/$TRAIN_JSON" --perclass-val $PERCLASS --strata $STRATA --out-dir /out @PIN
```

**Only after the correction state is logged as DL-<c>:**
```powershell
$CORR = "<CORRECTION_STATE>"                              # available, declined or no_approval, as logged in DL-<c>
$CORR_DL = "DL-<c>"
```

**6. D1 crops.** Also needs the D2 JSON's sha256 logged.
```powershell
$D2_JSON = "teacher_d2_<UTC>.json"                        # the name step 3 printed
$IN = $CKPT + (RO "$OUT\d2_val" "/in/d2")
$M  = OUTM "$OUT\d1_crops"
docker @RUN @DATA @IN @M $IMG python -B -c $CHECK data $CKPT_FILE "/in/d2/$D2_JSON" "/work/repo/$CFG" "/work/repo/$STRATA" /out
docker @RUN @DATA @IN @M $IMG python -B scripts/teacher_d1_nmf_sensitivity.py --part crops @CK --correction-state $CORR --correction-dl-id $CORR_DL --strata $STRATA --d2-val-output "/in/d2/$D2_JSON" --out-dir /out @PIN
```

**7. D1 VAL part.**
```powershell
$IN = $CKPT + (RO $T_RESCORE "/in/ref")
$M  = OUTM "$OUT\d1_val"
docker @RUN @DATA @IN @M $IMG python -B -c $CHECK data $CKPT_FILE /in/ref/summary.json "/work/repo/$CFG" /out
docker @RUN @DATA @IN @M $IMG python -B scripts/teacher_d1_nmf_sensitivity.py --part val @CK --correction-state $CORR --correction-dl-id $CORR_DL --val-reference /in/ref --out-dir /out @PIN
```

**Runbook notes**
1. dedup_val_scores.py's exit 1 with status "sign disagreement" is AM-18 item 6(b)'s halt (AM-PC-1 Q1),
   never a STOP to repeat.
2. An execution stopped before its first complete output may be repeated with identical inputs and the
   same commit once the stop and its cause are entered in the decision log (AM-18 repeats (i) as amended
   at its commit, AM-PC-1 Q4).
3. The ETA printed after 8 images is information only. No execution of record is stopped for runtime.
   Long passes run overnight with the laptop on its charger and sleep and hibernate set to Never
   (DG-5 ruling 2).
4. Each output of record is written outside the repository (`$OUT`). Its sha256 is entered in the
   decision log, then a write session copies it byte for byte into reports/derived/ (P39).

## 7. Proposed records (not written)

**Script-commit row (DL-<s>).** It records:
- **P:** this lane's merged tip, the 40-hex commit Ice's merge of lane/teacher-diag-p41 creates on
  master.
- **The scripts and modules:**
  - scripts/teacher_d1_nmf_sensitivity.py, scripts/teacher_d2_calibration.py,
    scripts/score_teacher_train.py, scripts/teacher_d3_perclass_strata.py,
    scripts/hash_split_files.py and scripts/dedup_val_scores.py;
  - src/eval/teacher_diag.py, src/eval/calibration.py and src/eval/nmf_sensitivity.py.
- **The code digest command 0 prints:** the expected value is
  `db4f4a48ffbb86ccecd210fdf95fedc48a35b1522c16cfd2e4cb4c19b0f662df`.
  - MEASURED at 328a426: `code_provenance()` over the 42 CODE_FILES, clean at HEAD.
  - INFERRED for P: no CODE_FILES path differs between 60c1417 and master 6067e66 (MEASURED), and this
    report commit adds no code.
  - The row records the value command 0 prints at P. A different value is a question to raise, not a fix.
- **The image digest:**
  `ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b`.
- **The clone recipe:** the three host lines under "Clone (P38)" in §6, with a sparse checkout that
  never materialises docs/reference/.

**Note for the audit row (DL-<a>), if Ice wants it:**
- the P41 follow-up ran under the GO of 2026-10-04 22:20, its STOP before step 4, and GO-2 of 23:31;
- h1–h4 and SCOPE a–g are done;
- the one approved install (torchvision 0.16.0+cpu) ran; `requests` was not installed.

## 8. Open questions

1. **The K-part smokes in this venv.** They need `requests`, which torchvision 0.16 imports and
   `--no-deps` skipped. Its lock pins:
   - requests==2.28.2 (requirements.lock:58);
   - urllib3==1.26.20 (:73);
   - idna==3.18 (:17);
   - charset-normalizer==3.4.7 (:6);
   - certifi==2026.6.17 (:4).

   None is in the venv (MEASURED). One approval would cover
   `pip install --no-deps requests==2.28.2 urllib3==1.26.20 idna==3.18 charset-normalizer==3.4.7 certifi==2026.6.17`.
   Then the three smokes could be re-run and reported without any edit. Otherwise Ice's local K-part gate
   remains their record.
2. **The AM-17b folders.** §6 mounts `teacher_val_cpu` and `e1_s42_val_cpu` as the artifact folders
   themselves. The evaluator writes the artifact into `--out-dir` (src/eval/artifacts.py:573–582), so
   this is INFERRED. If each folder instead holds a run subfolder, the source gains
   `\teacher_val_cpu_am17b2c` or `\e1_s42_val_cpu_am17b2c`. The check's `/in/ref/summary.json`,
   `/in/teacher/summary.json` and `/in/e1/summary.json` catch a wrong level as a STOP.
3. **PowerShell.** The `RO`/`OUTM` array splatting and the `$CHECK` one-liner were not run on Windows
   (INFERRED). The first check of command 0 shows whether both survive PowerShell's argument passing;
   a failure there is a STOP before any data is mounted.
4. **Placeholders beyond the kept list.** `<CORRECTION_STATE>`, the `<UTC>` names that steps 3 and 4
   print, and the duplicates sha256 from DL-<h> remain placeholders, because their values exist only
   after earlier steps.

P41 pushed to lane/teacher-diag-p41; Ice merges it, then P is logged in DL-<s>.
