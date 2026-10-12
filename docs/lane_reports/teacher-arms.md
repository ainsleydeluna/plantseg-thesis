# Lane report — L-TEACHER-ARMS · teacher arms C2–C5 (`lane/teacher-arms`)

Session ARMS-T "teacher arms C2–C5 (cloud)", Phase B-2, Sun 11 Oct 2026 (UTC); Opus 5.5 at max effort (the GO's line 2).
Tags as in the earlier lane reports: MEASURED (run here), REPOSITORY-PROVEN (file:line at the lane's last code
commit, C5 `1c7077e`), DOCUMENTED (the GO, the plan, the audit or a record), INFERRED.

## Summary

- **Built (C2–C4).**
  - The KD-arm launch gate for E3-X and the E7-X wrapper: `scripts/preflight_arms.py`, `scripts/run_ptq_arm.py`.
  - R1's and R2's configurations, and R2's dataset wrapper and paste transform:
    `configs/teacher/arms/r1_bema.py`, `configs/teacher/arms/r2_bema_rare.py`,
    `src/training/teacher_arm_components.py`.
  - The teacher-arm launcher: `scripts/launch_teacher_arm.py`.
  - R2's table builder, the arm teacher record and AM-18 item 5's go rule: `scripts/build_r2_table.py`,
    `scripts/arm_teacher_record.py`, `scripts/arm_go_rule.py`.
  - Two smokes: `scripts/smoke_teacher_arms.py` (98 checks) and `scripts/smoke_arms_kd.py` (61 checks).
- **Reviewed and fixed (STEP 7, C5).** One read-only DL-24 verification workflow ran over C2–C4. It found 18
  confirmed items (16 distinct defects), all fixed in C5, and refuted 4 findings (section (e)).
- **Accepted (STEP 8, MEASURED, on the committed C5).**
  - smoke_teacher_arms 98/98, smoke_arms_kd 61/61, smoke_arms_invariance 16/16, and the cross-commit invariance
    smoke 14/14.
  - Every STEP 3 smoke at its baseline exit code and count.
  - Every mutation killed by its named check, each after an unmutated control passed. The float fit-down mutant is
    killed too, so it is not an equivalent mutant.
- **Not done here.** No record was written (section (h) gives the rows). Nothing ran on a GPU or on the real data,
  and no part of the runbook (section (g)) was run (R11).

## (a) Commits (MEASURED; `git log`, `git diff --numstat`)

All on `lane/teacher-arms`. Each was committed with `git commit --only -F <message> -- <paths>`, after an
explicit-path `git add`, a scoped `git status` and `git diff --cached --stat`. No commit carries a trailer: `git log
--format=%B ceafd12..HEAD | git interpret-trailers --parse` prints nothing.

| Commit | Parent | Subject | Paths (status, +/−) | Total |
|---|---|---|---|---|
| C1 `ceafd1296ce373e8959a41af12586e2e7ebbfb0a` (the earlier session) | `8487551` | L-TEACHER-ARMS C1: default-off --arm seam in the KD trainer, arm registry, R1/R2 registry entries, invariance smoke | configs/kd_arms/e3_r1.json (A, +16), configs/kd_arms/e3_r2.json (A, +16), scripts/smoke_arms_invariance.py (A, +707), src/training/kd_arms.py (A, +306), src/training/train_distill.py (M, +92/−21) | +1,137/−21 |
| C2 `4bf4513bcba2c74bf15c1bb686982f466a3955d9` (13:43:58Z) | C1 | L-TEACHER-ARMS C2: arm launch gate (preflight_arms), E7 arm wrapper, smoke_arms_kd | scripts/preflight_arms.py (A, +1,415), scripts/run_ptq_arm.py (A, +285), scripts/smoke_arms_kd.py (A, +1,577) | +3,277/−0 |
| C3 `4e32ab67b84e12b274f5c99c8977b977dcb6ba85` (14:41:46Z) | C2 | L-TEACHER-ARMS C3: R1/R2 arm configs, R2 dataset wrapper and paste transform, teacher-arm launcher | configs/teacher/arms/r1_bema.py (A, +27), configs/teacher/arms/r2_bema_rare.py (A, +32), scripts/launch_teacher_arm.py (A, +701), src/training/teacher_arm_components.py (A, +367) | +1,127/−0 |
| C4 `44684a74a29cc408899585ed73a7e2c670612b95` (14:41:56Z) | C3 | L-TEACHER-ARMS C4: R2 table builder, arm teacher record, go-rule evaluator, smoke_teacher_arms | scripts/arm_go_rule.py (A, +248), scripts/arm_teacher_record.py (A, +289), scripts/build_r2_table.py (A, +234), scripts/smoke_teacher_arms.py (A, +2,007) | +2,778/−0 |
| C5 `1c7077e4d8702565597b2225fa10d894d9cb0464` (16:29:24Z) | C4 | L-TEACHER-ARMS C5: review fixes | scripts/arm_go_rule.py (M, +83/−24), scripts/arm_teacher_record.py (M, +30/−8), scripts/launch_teacher_arm.py (M, +21/−4), scripts/smoke_arms_kd.py (M, +75/−18), scripts/smoke_teacher_arms.py (M, +194/−77) | +403/−131 |
| C6 (this report; its sha is in the session's final report) | C5 | L-TEACHER-ARMS C6: lane report | docs/lane_reports/teacher-arms.md (A) | — |

`git diff --name-only 8487551 HEAD -- . <X>` lists exactly the 17 declared paths: C1's five, the eleven the plan
assigns to C2–C4, and this report.

## (b) STEPs 0–3: outputs and the pip freeze (MEASURED)

### STEP 0 (12:32:36Z)

The attachments' sha256 and sizes match the GO:
- the plan: `da3c0c4cec675e68303488c234f0dc1a957956e031b9530c44b60bd053fb11bf` (84,260 B);
- the audit: `2a1e8eadfee3e93032f53cc09af95469be38bc668bc65af695f7c6b62ba75f32` (31,566 B);
- C1's report: `a996ef821499463041a8a257458a72aea500201638145b6dca8f400654c2d61f` (14,962 B).

The GO file itself is `93255ff0cb1b17f09f4a397d739ccf6b34a30a0a2cee6291c70299eab89be27a` (25,994 B); the GO names
no sha256 for itself.

### STEP 1 (12:33:49Z–12:34:06Z), one command per call

Before the chain, HEAD named `refs/heads/claude/blissful-brown-vbzggj`: `git status -sb` printed
`## claude/blissful-brown-vbzggj`, and `git log --oneline -5` began `2e3c88a KD launch log: 4 lambda candidates
launched`. Auto mode did not block the chain.

| # | Command | Output |
|---|---|---|
| 1 | `git status --porcelain=v1 -- . <X>` | (nothing) |
| 2 | `git fetch origin lane/teacher-arms` | `* branch lane/teacher-arms -> FETCH_HEAD`; `* [new branch] lane/teacher-arms -> origin/lane/teacher-arms` |
| 3 | `git rev-parse FETCH_HEAD` | `ceafd1296ce373e8959a41af12586e2e7ebbfb0a` |
| 4 | `git ls-remote --heads origin lane/teacher-arms` | `ceafd1296ce373e8959a41af12586e2e7ebbfb0a refs/heads/lane/teacher-arms` |
| 5 | `git branch lane/teacher-arms ceafd12… && git symbolic-ref HEAD refs/heads/lane/teacher-arms && git checkout --no-overlay HEAD -- . <X>` | (nothing) |
| 6 | `git rev-parse HEAD` | `ceafd1296ce373e8959a41af12586e2e7ebbfb0a` |
| 7 | `git symbolic-ref HEAD` | `refs/heads/lane/teacher-arms` |
| 8 | `git status --porcelain=v1 -- . <X>` | (nothing) |
| 9 | `git diff --stat ceafd12… -- . <X>` | (nothing) |
| 10 | `git diff --cached --stat ceafd12… -- . <X>` | (nothing) |
| 11 | `git status --porcelain=v1 -- docs/reference/reference.pdf` | (nothing) |

### STEP 2 (12:36:08Z–12:39:46Z)

The host: Python 3.11.17 (`/usr/bin/python3.11`), 4 CPUs, 15 GB of memory, no GPU. The packages were named before
each install. The commands, in order:

```
/usr/bin/python3.11 -m venv /tmp/tbvenv
/tmp/tbvenv/bin/python -m pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
/tmp/tbvenv/bin/python -m pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
/tmp/tbvenv/bin/python -m pip install --no-cache-dir mmengine==0.10.7 mmsegmentation==1.2.2 opencv-python==4.8.1.78 addict==2.4.0 PyYAML==6.0.3 yapf==0.43.0 platformdirs==4.10.0 termcolor==3.3.0 rich==13.4.2 markdown-it-py==4.2.0 mdurl==0.1.2 Pygments==2.20.0 matplotlib==3.11.0 contourpy==1.3.3 cycler==0.12.1 fonttools==4.63.0 kiwisolver==1.5.0 packaging==24.2 pyparsing==3.3.2 python-dateutil==2.9.0.post0 six==1.17.0 prettytable==3.18.0 wcwidth==0.8.1 scipy==1.11.4 regex==2026.5.9 numpy==1.26.4 pillow==12.3.0 filelock==3.14.0 fsspec==2026.4.0 Jinja2==3.1.6 MarkupSafe==3.0.3 networkx==3.6.1 sympy==1.14.0 mpmath==1.3.0 typing_extensions==4.15.0 requests==2.28.2 certifi==2026.6.17 charset-normalizer==3.4.7 idna==3.18 urllib3==1.26.20
/tmp/tbvenv/bin/python -m pip freeze
mkdir -p /tmp/tbscratch
/tmp/tbvenv/bin/python -m pip install --no-cache-dir --no-deps ftfy==6.3.0      # asked and approved, see (i)
```

mmcv 2.1.0 installed from the wheel index, with no source build, so the teacher-stack smokes ran here. An import
probe then failed every `mmseg.*` import with `No module named 'ftfy'`:
- mmsegmentation 1.2.2 imports ftfy when `mmseg.utils` loads;
- ftfy is not in requirements.lock;
- its version of record is 6.3.0 (`scripts/preflight_distill.py:88`, `FINGERPRINT_OF_RECORD`).

The GO approves no other install, so the session asked, and Ice approved exactly the one command above (12:39:37Z).
The re-probe then imported mmseg.utils, mmseg.datasets, mmseg.models and mmseg.apis.

The freeze, 44 packages (STEP 2's freeze plus ftfy), rerun at 16:47Z and unchanged:

```
addict==2.4.0             idna==3.18                  networkx==3.6.1               rich==13.4.2
certifi==2026.6.17        Jinja2==3.1.6               numpy==1.26.4                 scipy==1.11.4
charset-normalizer==3.4.7 kiwisolver==1.5.0           opencv-python==4.8.1.78       six==1.17.0
contourpy==1.3.3          markdown-it-py==4.2.0       packaging==24.2               sympy==1.14.0
cycler==0.12.1            MarkupSafe==3.0.3           pillow==12.3.0                termcolor==3.3.0
filelock==3.14.0          matplotlib==3.11.0          platformdirs==4.10.0          torch==2.1.0+cpu
fonttools==4.63.0         mdurl==0.1.2                prettytable==3.18.0           torchvision==0.16.0+cpu
fsspec==2026.4.0          mmcv==2.1.0                 Pygments==2.20.0              typing_extensions==4.15.0
ftfy==6.3.0               mmengine==0.10.7            pyparsing==3.3.2              urllib3==1.26.20
                          mmsegmentation==1.2.2       python-dateutil==2.9.0.post0  wcwidth==0.8.1
                          mpmath==1.3.0               PyYAML==6.0.3                 yapf==0.43.0
                                                      regex==2026.5.9
                                                      requests==2.28.2
```

### STEP 3: the baseline at ceafd12 (12:41:08Z to about 12:59:40Z)

Section (c) lists every run with its command, exit code and time. Compared with C1's report (its STEP 3 table at
8487551 and its STEP 5 final run), two things differ, both caused by the environment. So this STEP's values are the
baseline (the GO's STEP 3 rule), and "pass" in every later acceptance means the same exit code and count as here.

1. `smoke_preflight_distill` exits 1 with 183/184 (C1: 184/184). The one failing check,
   `f_this_venvs_fingerprint_is_not_the_kd_images_and_its_imports_have_values_of_record`, reports
   `python: '3.11.17' != '3.11.16'`, `torch: '2.1.0+cpu' != '2.1.0+cu121'` and
   `torchvision: '0.16.0+cpu' != '0.16.0+cu121'`. The cause is this host's interpreter (STEP 2's
   `/usr/bin/python3.11` is 3.11.17) and the GO's approved CPU wheels.
2. The teacher-stack smokes are not in C1's table, because C1's environment had no mmcv. Here, with mmcv 2.1.0 and
   mmsegmentation 1.2.2, `smoke_teacher_launch` (79/79), `smoke_teacher_config` (108/108) and `smoke_teacher_nmf`
   (18/18) pass as plain runs. Two need variables their image declares:
   - `smoke_teacher_selection` exits 1 with no RESULT line: `FileNotFoundError: … '/usr/local/lib/python3.11/site-packages/mmseg/.mim/configs/segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py'`,
     the stock config's path in the KD image. With `PLANTSEG_MMSEG_STOCK_CONFIG` pointing at this venv's copy, it
     passes 54/54.
   - `smoke_teacher_runner` exits 2: `ENVIRONMENT: CUBLAS_WORKSPACE_CONFIG must be inherited as ':4096:8' before the
     interpreter starts (G8); got None. Run inside the plantseg-teacher image, which declares it.` With that variable
     set, it passes 48/48.

   Every acceptance runs both forms. The plain runs must keep exit 1 and exit 2 with the same output, and the
   documented-environment runs must pass.

Everything else matches. smoke_arms_invariance (16/16) and the cross-commit invariance smoke (14/14) match C1's final
run, and the other nine smokes of C1's table have C1's counts.

## (c) Every run (MEASURED)

### c1. The acceptance batches

**How each run was made.** Every run is `timeout -k 60 10800 env TMPDIR=/tmp/tbscratch PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=/home/user/plantseg-thesis /tmp/tbvenv/bin/python -B <command>`, from the repository root, one at a time.
A scratch runner (`/tmp/tbscratch/run_smokes.sh`, `run_docenv.sh`) logs each run's output and exit code. The two
documented-environment runs add `PLANTSEG_MMSEG_STOCK_CONFIG=…` or `CUBLAS_WORKSPACE_CONFIG=:4096:8` to the env.
While a batch ran, nothing else ran but read-only shell commands and scratch writes.

**The batches:**

| Batch | Tree | Window | Measured | Estimate stated before the start |
|---|---|---|---|---|
| STEP 3 | ceafd12 | 12:41:08Z–12:55:29Z, then the documented pair 12:56:37Z–12:59:37Z | 1,041 s | none |
| STEP 4 | C2 | 13:25:36Z–13:41:48Z | 972 s | about 1,103 s in all |
| STEP 5 | C2 + C3's draft, untracked | 13:58:12Z–14:14:50Z | 998 s | about 16.5 min (STEP 4's measured times) |
| STEPs 5–6 acceptance | the final C3 + C4, untracked, before their commits | 14:22:49Z–14:41:17Z | 1,108 s | about 17 min (STEP 5's measured times plus 31 s) |
| STEP 8 final | C5 `1c7077e`, committed | 16:29:48Z–16:47:03Z | 1,035 s | about 1,124 s, stated at 16:29Z |

Each cell below is: exit code · result · estimate/measured seconds. "—" means no estimate was stated before the
run, a slip recorded in (i).

| Run | Command (after the prefix) | STEP 3 · ceafd12 | STEP 4 · C2 | STEP 5 · C2 + C3 draft | STEPs 5–6 acceptance · C3 + C4 | STEP 8 final · C5 |
|---|---|---|---|---|---|---|
| teacher_arms | `scripts/smoke_teacher_arms.py` | — | — | — | 0 · PASS (98/98) · 31/32 | 0 · PASS (98/98) · 39/39 |
| arms_kd | `scripts/smoke_arms_kd.py` | — | 0 · PASS (61/61) · 45/42 | 0 · PASS (61/61) · 42/40 | 0 · PASS (61/61) · 40/43 | 0 · PASS (61/61) · 50/49 |
| frozen_blobs | `scripts/smoke_frozen_blobs.py` | 0 · FROZEN BLOBS OK (10/10) · —/0 | 0 · FROZEN BLOBS OK (10/10) · 1/0 | 0 · FROZEN BLOBS OK (10/10) · 0/0 | 0 · FROZEN BLOBS OK (10/10) · 0/0 | 0 · FROZEN BLOBS OK (10/10) · 1/0 |
| invariance_distill | `scripts/smoke_invariance_distill.py` | 0 · PASS (112/112) · —/162 | 0 · PASS (112/112) · 165/130 | 0 · PASS (112/112) · 130/130 | 0 · PASS (112/112) · 130/136 | 0 · PASS (112/112) · 136/128 |
| kd_step_selfcheck | `scripts/smoke_kd_step.py --selfcheck` | 0 · PASS (108/108) · —/30 | 0 · PASS (108/108) · 30/29 | 0 · PASS (108/108) · 29/26 | 0 · PASS (108/108) · 26/30 | 0 · PASS (108/108) · 30/28 |
| distill_switches | `scripts/smoke_distill_switches.py` | 0 · PASS (116/116) · —/2 | 0 · PASS (116/116) · 2/2 | 0 · PASS (116/116) · 2/2 | 0 · PASS (116/116) · 2/2 | 0 · PASS (116/116) · 2/2 |
| preflight_distill | `scripts/smoke_preflight_distill.py` | 1 · FAIL (183/184) · —/25 | 1 · FAIL (183/184) · 25/27 | 1 · FAIL (183/184) · 27/26 | 1 · FAIL (183/184) · 26/27 | 1 · FAIL (183/184) · 27/26 |
| teacher_launch | `scripts/smoke_teacher_launch.py` | 0 · PASS (79/79) · —/9 | 0 · PASS (79/79) · 10/4 | 0 · PASS (79/79) · 4/9 | 0 · PASS (79/79) · 9/8 | 0 · PASS (79/79) · 8/4 |
| teacher_config | `scripts/smoke_teacher_config.py` | 0 · PASS (108/108) · —/4 | 0 · PASS (108/108) · 5/4 | 0 · PASS (108/108) · 4/4 | 0 · PASS (108/108) · 4/4 | 0 · PASS (108/108) · 4/3 |
| distill | `scripts/smoke_distill.py` | 0 · PASS (95/95) · —/2 | 0 · PASS (95/95) · 2/2 | 0 · PASS (95/95) · 2/2 | 0 · PASS (95/95) · 2/2 | 0 · PASS (95/95) · 2/3 |
| eval_stage_artifacts | `scripts/smoke_eval_stage_artifacts.py` | 0 · PASS (66/66) · —/4 | 0 · PASS (66/66) · 5/4 | 0 · PASS (66/66) · 4/4 | 0 · PASS (66/66) · 4/4 | 0 · PASS (66/66) · 4/4 |
| select_sweeps | `scripts/smoke_select_sweeps.py` | 0 · PASS (499/499) · —/83 | 0 · PASS (499/499) · 85/80 | 0 · PASS (499/499) · 80/81 | 0 · PASS (499/499) · 81/90 | 0 · PASS (499/499) · 90/86 |
| teacher_selection | `scripts/smoke_teacher_selection.py` | 1 · no RESULT line · —/3 | 1 · no RESULT line · 3/3 | 1 · no RESULT line · 3/3 | 1 · no RESULT line · 3/3 | 1 · no RESULT line · 3/3 |
| teacher_runner | `scripts/smoke_teacher_runner.py` | 2 · no RESULT line · —/0 | 2 · no RESULT line · 1/0 | 2 · no RESULT line · 0/0 | 2 · no RESULT line · 0/0 | 2 · no RESULT line · 1/0 |
| teacher_nmf | `scripts/smoke_teacher_nmf.py` | 0 · PASS (18/18) · —/25 | 0 · PASS (18/18) · 25/25 | 0 · PASS (18/18) · 25/22 | 0 · PASS (18/18) · 22/25 | 0 · PASS (18/18) · 25/23 |
| loader_parity | `scripts/smoke_loader_parity.py` | 0 · PASS (7/7) · —/3 | 0 · PASS (7/7) · 3/2 | 0 · PASS (7/7) · 2/3 | 0 · PASS (7/7) · 3/3 | 0 · PASS (7/7) · 3/3 |
| distill_schedule | `scripts/smoke_distill_schedule.py` | 0 · PASS (32/32, 2 skipped) · —/31 | 0 · PASS (32/32, 2 skipped) · 31/30 | 0 · PASS (32/32, 2 skipped) · 30/29 | 0 · PASS (32/32, 2 skipped) · 29/32 | 0 · PASS (32/32, 2 skipped) · 32/27 |
| arms_invariance | `scripts/smoke_arms_invariance.py` | 0 · PASS (16/16) · —/296 | 0 · PASS (16/16) · 300/266 | 0 · PASS (16/16) · 266/259 | 0 · PASS (16/16) · 259/296 | 0 · PASS (16/16) · 296/264 |
| invariance_cross_commit | `scripts/smoke_invariance_distill.py --cross-commit 8487551ca67bd50ffbe911d1f3e93244841cbc9f WORKTREE` | 0 · PASS (14/14) · —/182 | 0 · PASS (14/14) · 185/166 | 0 · PASS (14/14) · 166/189 | 0 · PASS (14/14) · 189/189 | 0 · PASS (14/14) · 189/174 |
| teacher_selection_docenv | `PLANTSEG_MMSEG_STOCK_CONFIG=/tmp/tbvenv/lib/python3.11/site-packages/mmseg/.mim/configs/segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py … scripts/smoke_teacher_selection.py` | 0 · PASS (54/54) · —/177 | 0 · PASS (54/54) · 177/154 | 0 · PASS (54/54) · 154/166 | 0 · PASS (54/54) · 166/178 | 0 · PASS (54/54) · 178/166 |
| teacher_runner_docenv | `CUBLAS_WORKSPACE_CONFIG=:4096:8 … scripts/smoke_teacher_runner.py` | 0 · PASS (48/48) · —/3 | 0 · PASS (48/48) · 3/2 | 0 · PASS (48/48) · 2/3 | 0 · PASS (48/48) · 3/4 | 0 · PASS (48/48) · 4/3 |

The non-zero exits are the baseline's at every batch. The logs of the plain teacher_selection and teacher_runner
runs equal STEP 3's except for timestamps, and preflight_distill's single FAIL is the same fingerprint check.

### c2. Development runs, the C3 working checks and the mutation run

Same prefix and wrapper. Unless the table names a script under `/tmp/tbscratch`, the command is the smoke named.

| Run | Command | Start | Estimate | Measured | Exit | Result |
|---|---|---|---|---|---|---|
| dev_akd_1 | `scripts/smoke_arms_kd.py` | 13:16:27Z | about 180 s | 50 s | 1 | no RESULT: a raised SystemExit escaped `run_case`; now a SystemExit fails its check |
| dev_akd_2 | the same | 13:17:34Z | — | 41 s | 0 | PASS (61/61) |
| dev_akd_3 | the same | 13:18:32Z | — | 41 s | 0 | PASS (61/61) |
| dev_akd_4 | the same (the stopped line added to a check) | 13:20:39Z | — | 44 s | 0 | PASS (61/61) |
| c3_work_1 | `/tmp/tbscratch/c3/work_checks.py` (with `CUBLAS_WORKSPACE_CONFIG=:4096:8`) | 13:52:39Z | about 3 min | 15 s | 1 | FAIL (9/11): the init-checkpoint patch was global; now scoped, as smoke_teacher_launch does |
| c3_work_2 | the same | 13:53:14Z | — | 13 s | 0 | PASS (11/11) |
| c3_work_3 | the same | 13:54:08Z | — | not timed | 0 | PASS (11/11) |
| c3_work_4 | the same (no-paste path added) | 13:55:49Z | — | 15 s | 0 | PASS (12/12) |
| c3_work_5 | the same (two refusals added) | 13:57:08Z | — | 14 s | 0 | PASS (13/13) |
| c3_work_6 | the same | 13:57:45Z | — | 16 s | 0 | PASS (13/13) |
| c3_work_7 | the same, with the C4 drafts in place | 14:16:27Z | about 2 min, with dev_sta_1 | 16 s | 0 | PASS (13/13) |
| dev_sta_1 | `scripts/smoke_teacher_arms.py` | 14:16:47Z | (the same 2 min) | 31 s | 1 | FAIL (96/98): two diffs were computed outside the env context; fixed |
| dev_sta_2 | the same | 14:17:37Z | — | 33 s | 0 | PASS (98/98) |
| dev_sta_3 | the same (the no-log incomplete case added) | 14:21:49Z | — | 31 s | 0 | PASS (98/98) |
| c5_sta_1 | `scripts/smoke_teacher_arms.py` (C5 draft) | 16:22:37Z | about 50 s | 38 s | 1 | FAIL (95/98): three controls failed after a fixture crash; fixed in C5 |
| c5_sta_2 | the same | 16:24:33Z | about 40 s | 36 s | 0 | PASS (98/98) |
| c5_akd_1 | `scripts/smoke_arms_kd.py` (C5 draft) | 16:25:20Z | about 75 s | 54 s | 0 | PASS (61/61) |
| c5 final bytes | both arm smokes, on the bytes C5 commits | 16:27:30Z | about 40 s and 55 s | 39 s and 50 s | 0, 0 | PASS (98/98), PASS (61/61) |
| float fit-down mutant | `cd /tmp/tbscratch/step8 && … TMPDIR=/tmp/tbscratch/step8 … python -B float_fitdown_mutant.py` | 16:47:2xZ | about 15 s | 3 s | 0 | KILLED (section (d)) |

The C3 working checks, 13 by the end, were scratch checks run before the C3 commit (the GO's STEP 5): reg_show_diff,
diff_cfg_equals_reg, table_validator, dataset_repeat_counts, non_rare_bit_identical, rare_paste_labels,
paste_draw_order, exif_frame_counted, no_paste_is_record_path, rehearsal_r1_partial, rehearsal_r2_partial,
launcher_refusals and launcher_more_refusals. smoke_teacher_arms covers each of them since C4.

### c3. Runs by the review agents

During STEP 7, the review agents ran short probes and the two arm smokes in their own folders,
`/tmp/tbscratch/review/<rev|ver>-<key>/`. Those runs are part of the review, not of the acceptance. Their outputs are
in section (e).

## (d) The mutations and their killing checks (MEASURED, STEP 8's final runs)

Both smokes run their mutation tables in every run. Each mutation is an in-memory monkeypatch, restored after its
check, and never touches the committed tree. Since C5, each mutation's check first runs again without the mutation
and must pass (the control), so a kill cannot come from a re-run that fails anyway (review LAUNCHER-V1). The tables
are from STEP 8's logs. smoke_teacher_arms's 24 are the plan's 13 plus the GO's 11; smoke_arms_kd's 13 are the plan's
9 plus the GO's 4.

### smoke_teacher_arms (24 mutations)

| # | Mutation | Killed by | Control | Result | What the check saw under the mutation |
|---|---|---|---|---|---|
| 1 | round instead of ceil | cm_repeat_table_exact | PASS | killed | m_c for n_c 1..25: [4, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, …] |
| 2 | background as a class | cm_no_disease_image_appears_once | PASS | killed | 7 images without a disease pixel: cat ids [[0], [0]], repeats [1, 1, 1, 1, 1, 1, 1] |
| 3 | paste after M2/M3 | cm_paste_draw_order | PASS | killed | 19 pastes replayed draw by draw; M2/M3 follow on the same stream: False |
| 4 | alpha blend | cm_overwrite_no_blend_label_c | PASS | killed | the check fails; its detail is fixed text ("20 pasted pixels equal the patch exactly; label 9; the rest unchanged") and does not name the comparison that failed |
| 5 | overlap 10% | cm_overlap_5pct_rule | PASS | killed | tries 1 at [0, 0]: the first placement (2 of 20 patch pixels, 10%) accepted; the rule needs tries 2 at [20, 20] |
| 6 | momentum 0.9998 | cm_ema_decay_closed_form | PASS | killed | max \|EMA − 0.9998^(t−1)\| over t = 1..5 = 9.996e-01 |
| 7 | update_buffers False | cm_ema_bn_buffers_averaged | PASS | killed | EMA running_mean 5.0, expected 1.0008 (averaged, not copied) |
| 8 | drop_path 0.1 | cm_r1_diff_exact | PASS | killed | leaves ['custom_hooks']; drop_path_rate leaves [] (the rule needs [("model.backbone.drop_path_rate", 0.1, 0.2)]) |
| 9 | builder skips the strata check | cm_table_builder_counts_vs_strata | PASS | killed | class 2 off by one: exit 0, STOP lists [] |
| 10 | VAL pipeline changed | cm_r2_val_test_unchanged | PASS | killed | VAL/TEST equal: val_dataloader False |
| 11 | >= instead of > in the go rule | cm_go_rule_conditions | PASS | killed | "=gate, ECE equal" gives go; through the CLI at VAL_X = gate: go |
| 12 | eleven tries | cm_ten_tries_then_nothing | PASS | killed | tries 11, accepted False, scripted ints left 2 |
| 13 | raw weights loaded | cm_kd_loader_reads_state_dict | PASS | killed | loader returns state_dict (the EMA weights): False |
| 14 | an inherited env value accepted | cm_launcher_env_resolution | PASS | killed | conflict [None] |
| 15 | the frozen load_launch_config used on the arm config | rf_launch_config_mismatch | PASS | killed | the registered config is refused too: [launch_config_mismatch] |
| 16 | the token ignored | rf_teacher_launch_without_recheck | PASS | killed | no_token: None (not refused) |
| 17 | an empty-patch special case | cm_empty_patch_draws_one_placement | PASS | killed | tries 0, ints left [3, 4, 9, 9] |
| 18 | the EXIF frame ignored | cm_paste_exif_rotated_frame | PASS | killed | frame (32, 24), counts 0; rotated S: size [8, 10], S frame resized False |
| 19 | a two-label image tolerated | rf_table_two_labels_stop | PASS | killed | exit 1; two-label images [] |
| 20 | rows read at HEAD | cm_rows_read_from_records_commit | PASS | killed | PreflightError: "docs/DECISION_LOG.md is not in the records commit" (the fixture repository's HEAD holds no decision log) |
| 21 | table_path compared by value | cm_canonical_diff_table_path_by_variable | PASS | killed | one diff: False; equal to REG: False; table_path by variable: False |
| 22 | half-up rounding | cm_patch_resize_rule | PASS | killed | 5x5 at f = 1/2: (3, 3), not (2, 2) |
| 23 | a rerun after divergence allowed | rf_teacher_arm_rerun_after_divergence | PASS | killed | exit 0, nothing on stderr: no refusal |
| 24 | a refusal when there is no measurement | cm_teacher_arith_no_measurement_not_cut | PASS | killed | exit 2 PREFLIGHT FAILED [arm_cut_by_arithmetic]: no measurement: cu… (the detail keeps stderr's first 60 characters) |

### smoke_arms_kd (13 mutations)

| # | Mutation | Killed by | What the check saw under the mutation |
|---|---|---|---|
| 1 | mut_parent_true | cm_run_meta_arm_fields | parent_of_e4_e7 True on an arm run |
| 2 | mut_target_construction_missing | cm_run_meta_target_construction_k1_v1 | no target_construction |
| 3 | mut_arm_key_in_default_run_meta | cm_default_run_meta_has_no_arm_key | an arm key in a default run's run_meta |
| 4 | mut_stage_e3_in_arm_line | cm_arm_line_not_alpha_candidate | launch_log_format: … line 2: e3-r1_s42_lambda1_alpha50_80k_a1 attempt 1 != 2 (1 + the earlier launch lines with the same stage, seed, …) |
| 5 | mut_gate_skips_go_rule | cm_gate_go_rule_go_required | no_go, not_evaluable, another checkpoint and another arm all PASS |
| 6 | mut_gate_accepts_record_teacher | cm_gate_teacher_identity_arm_values | the teacher of record not refused |
| 7 | mut_approval_marker_ignored | cm_gate_approval_row_required | a row without the marker: PASS |
| 8 | mut_pin_rule_off | cm_gate_arms_pin_rule | another P_K and another arm line's pin: PASS |
| 9 | mut_median_replaced_by_mean | cm_arith_cut | the fitting case is cut (9.909…) |
| 10 | mut_divergence_repeat_allowed | rf_arm_repeat_after_divergence | diverged: entry 0, gate PASS; stopped: entry 0, gate PASS |
| 11 | mut_min_in_place_of_max | cm_arith_flip_reference_max | PASS with 1.0 applied (E3's figure, not the arm's 1.5) |
| 12 | mut_alpha_record_rejected | cm_alpha_cut_record_accepted | selection_inputs:alpha_cut_record_invalid |
| 13 | mut_record_teacher_arm_accepts_arm_teacher | rf_record_teacher_arm_with_arm_teacher | the arm teacher's file passes the arguments stage |

### The float fit-down mutant (the GO's SMOKE CASES: "If no fixture can tell float fit-down from exact fit-down …")

It is not an equivalent mutant. The scratch driver replaces `teacher_arm_components.patch_size` with a copy whose
fit-down factor is a float, `g = min(H_T / h', W_T / w')`, and reruns `cm_patch_fit_down_rule`:

| | Exact (the code) | Float (the mutant) |
|---|---|---|
| fixture `(19, 38, (16, 21), (21, 21), 1.0)`: a 19 × 38 patch in a 16 × 21 T, g = 21/38, height exactly 10.5 | (10, 21) | (11, 21) |
| fixture `(98, 147, (1, 2), (2, 2), 1.0)`: a width of exactly 1.5 | (1, 2) | (1, 1) |
| `cm_patch_fit_down_rule` | PASS (the control) | FAIL: killed |

## (e) The DL-24 review (STEP 7): confirmed and refuted findings

**The workflow (MEASURED).** One read-only verification workflow over `ceafd12..44684a7` (C2–C4), requested in
plain language and launched at 14:45:30Z as run `wf_4d7f2e61-162`. It finished at about 16:10Z: 85 minutes, 8
agents, no agent error, about 3.08 M subagent tokens and 497 tool calls. Four reviewers each covered one file group:
- kd-gate: C2's three files;
- r2-data: the two configs and teacher_arm_components.py;
- launcher: launch_teacher_arm.py and build_r2_table.py;
- records: arm_teacher_record.py, arm_go_rule.py, and smoke_teacher_arms.py as a whole.

Each reviewer was followed by an adversarial verifier. The verifier tried to refute every finding, then tried to
break the reviewer's three riskiest "checked and correct" claims, and reported any counterexample as a new finding.
Every prompt carried the GO's hard limits, and each agent had its own scratch folder. The snapshot was identical
before and after the workflow: HEAD `44684a7`, the scoped status (clean), and the sha256 of the 16 files then present
(C2–C4's eleven and C1's five; the report did not exist yet).

**Count.** The reviewers raised 19 findings: 15 were confirmed and 4 refuted. The verifiers added 3 new findings,
each demonstrated by a probe. That makes 18 confirmed items, or 16 distinct defects: r2data-1, LAUNCHER-5 and REC-9
are one defect. All 16 are fixed in C5 (`1c7077e`). None needed an OLD file or the A4 interface.

### e1. Confirmed, with their C5 fix and the check that now covers each

| Id (reviewer → verifier severity) | Finding | C5 fix | Checked by |
|---|---|---|---|
| REC-1 (major → major) | `--host-fault-dl` accepted any RECORDED row naming the item 1(b) output's sha256. DL-92 already does, so the host rule could be lifted with no documented fault | the fault row must carry the literal "AM-18 item 5 host fault" and that sha256 (`HOST_FAULT_MARKER`) | STA cm_go_rule_control_reproduction: the item 1(b) row named as a fault row is refused |
| REC-2 (major → major) | `--void-session-dl` accepted any row naming the teacher record's sha256 (the R1-record row R2's launch reads, or this session's own row), so a first void session could be written as not_evaluable | `--void-control-output` names the earlier void session's control output (record role, not reproduced, the same item 1(b) output and commit, ended before this control started, another file), and the row must carry "AM-18 item 5 void session", the record's sha256 and that output's sha256 | the same check: the record row, this session's own control and the row alone are refused; the recurrence is written with the earlier control's sha256 |
| LAUNCHER-5 (minor → **major**) = r2data-1 (minor → minor) = REC-9 (minor → minor) | the mutation "table_path compared by value" deleted ENV_KEYS["table_path"], so its check died in a KeyError in REG, a kill that cannot fail | the mutant skips the by-variable rule for table_path only; ENV_KEYS and REG stay intact | the mutation table: killed by the comparison (row 21 of (d)) |
| LAUNCHER-V1 (new, major) | the mutation pass re-ran checks that write to fixed output paths, so "builder skips the strata check", "a two-label image tolerated" and ">= instead of > in the go rule" were killed whatever the code did | fresh output paths per run; in both smokes, each mutation's check first runs again unmutated and must pass (the control) | every control passes in STEP 8 (section (d)) |
| KDG-2 (minor → minor) | pa's own repeat rules (repeat_report_missing, previous_attempt_unresolved) and its kd_dry_run were untested | SAK cm_gate_stage_list sub-cases | entry and gate give repeat_report_missing, previous_attempt_unresolved and previous_run_missing; the dry-run argv carries `--arm e3-r1` and X's sha256; an output without the `[arm]` line gives kd_dry_run_failed |
| KDG-3 (minor → minor) | the go-rule record's binding (checkpoint, arm) and the teacher record's status and arm were untested | SAK sub-cases | cm_gate_go_rule_go_required: another checkpoint or arm gives go_rule_record_invalid; rf_missing_input_teacher_record: diverged, incomplete or another arm gives teacher_record_invalid |
| r2data-2 (minor → minor) | R6's EXIF rule for S had no fixture: an S-only frame mutant passed every check | an EXIF-rotated S pasted into an unrotated T | STA cm_paste_exif_rotated_frame: size [10, 8], with the S frame resized (an unframed S gives [8, 10]) |
| LAUNCHER-3 (minor → minor) | an unreadable `--r2-table`, `--r1-record`, `--previous-record` or `--reference-log` crashed with a traceback (exit 1) | named refusals with exit 2: [r2_table_unreadable], [r1_record_unreadable], [previous_record_unreadable], [reference_unusable]; a folder as `--am8a-report` gives [repeat_report_missing] | STA rf_r2_table_file_sha_mismatch, rf_wrong_checkpoint_pin_r1_record_sha, cm_teacher_arith_reference_log, rf_teacher_arm_repeat_without_report |
| LAUNCHER-4 (minor → minor) | two checks ran the arithmetic at the wall clock and would fail from 2026-11-03T10:13:20Z | the smoke pins the launcher's clock at 2026-10-11T00:00Z | both checks |
| REC-3 (minor → minor) | with a fault row, the control and the arm could run on two hosts | the control and the arm always share a host; the fault row lifts only the item 1(b) host rule | cm_go_rule_control_reproduction: a split session is refused even with the fault row |
| REC-4 (minor → minor) | the session's commit was not bound to the D2 script's commit of record | the control and the arm must run at the item 1(b) output's commit (P) | the same check: a session at another commit is refused |
| REC-6 (minor → minor) | the go-rule record lacked its own commit | `writer.commit` | cm_go_rule_conditions |
| REC-7 (minor → minor) | the record's `registered_difference` had no JSON (plan (iv)) | REG[arm]'s canonical JSON, after its sha256 is checked against the provenance's | cm_record_complete |
| REC-8 (minor → minor) | `--generated-utc` was accepted for a launched run | refused when the provenance records a launch | cm_record_complete |
| VREC-1 (new, minor) | `--item1b-output` was not bound to the output of record, so a fresh item1 pass on the scoring host passed the host rule | `--item1b-dl` (required) names a RECORDED row with "item 1(b) D2 VAL execution of record" (DL-92's words) and the file's sha256 | the same check: another item1 output is refused. The real DL-92 carries the marker, the sha256 and RECORDED, and neither the fault nor the void-session marker (checked with grep -n) |
| VREC-2 (new, minor) | the record bound the provenance by the pod's absolute work-dir path, so the laptop copy that R11 records was refused | bound by the work dir's name and attempt; the record keeps `launched_path` | cm_record_complete: a moved copy records as complete |

**Found while verifying C5 (not review findings; fixed in C5).** The new control re-run exposed two kills that came
from a fixture, not from the check:
- "the token ignored" was killed by a git branch-name collision in the records-commit fixture. Its re-run fell in the
  same second, so the commit, and the branch named after it, were identical. The crash also left the scratch
  repository's HEAD on a records commit, so three later controls failed. Fix: `branch -f`, and HEAD restored in a
  `finally`.
- "the EXIF frame ignored" was killed by the scripted RNG's range assertion. Fix: the placement now fits both
  frames, so the check's comparisons decide.

### e2. Refuted

| Id | Claim | Why refuted (the verifier) |
|---|---|---|
| KDG-1 | mut_stage_e3_in_arm_line is killed by ss.LaunchLog's attempt numbering, not by sweep_attempts | The GO asks only that each mutation be killed by its named check. With stage E3, the arm line joins the α = 50 launch group, and LaunchLog refuses it as an α-sweep attempt: that is the collision the check guards against. With the attempt corrected, sweep_attempts refuses it too. Which equivalent guard fires first is a style preference. |
| LAUNCHER-1 | R2 can skip the arithmetic by leaving out `--reference-log` | GO R7 makes the flag optional for both arms ("With no reference given it prints 'no measurement: not cut' and does not refuse"), and audit F21 says the launcher must not refuse. The duty sits in the runbook: both pod blocks pass the log (g2, g3). |
| LAUNCHER-2 | attempt 1 is self-declared, so a re-run of the `_a1` block on a fresh volume bypasses R4(b) | R4(b) fixes the mechanism exactly (the attempt from the `_a<n>` suffix; the previous record and report for n > 1), and the code implements every clause. The teacher arms have no launch log. The runbook says never to reuse a work dir name (g2). |
| REC-5 | not_evaluable can be written only if X was scored after the void session's STOP | The plan's interface makes `--arm-output` required, and R11 orders the control and then the arm, with no stop between. The runbook says X is scored even when the control does not reproduce (g4). |

### e3. The agents' conduct

No agent reported a refused command. The REC-6 verifier disclosed that it read its own probe's output with a plain
`/tmp/tbvenv/bin/python -B -c`, without the probe prefix's timeout and environment wrapper. The read was read-only and
inside its own folder. The commands of the agents that the SL-1 guard logged (14:45:56Z–16:00:48Z) are listed in (i).

## (f) Rulings and audit findings, mapped to code and checks (REPOSITORY-PROVEN at C5)

Function names refer to the files at C5. SAK = smoke_arms_kd, STA = smoke_teacher_arms.

| Ruling / finding | Implemented in | Checked by |
|---|---|---|
| R1 (F4): the frozen `load_launch_config` is never run on an arm config | `launch_teacher_arm.gate_configuration`: `ltf.check_locked_config(record)`; `arm_config_problems` (randomness, resume, work_dir, load_from, both data roots, R2's table path); `diff_cfg(record, arm) == reg(arm)` by canonical bytes, else `[launch_config_mismatch]` | STA cm_r1_record_locks_hold, cm_r1_diff_exact, cm_r2_diff_exact, rf_launch_config_mismatch (+ mutation "the frozen load_launch_config used on the arm config") |
| R2 (F5): the four variables come from the flags; an inherited different value is refused | `gate_environment` (runs before any `Config.fromfile`); `provenance.resolved_paths` | STA cm_launcher_env_resolution (+ mutation "an inherited env value accepted") |
| R3 (F6): HEAD = P_T, `--records-commit H`, P_T an ancestor of H, rows read by `git show H:` | `gate_checkout` (`HeadSource.head`, `commit_exists`, `is_ancestor`); `gate_rows` (`src.show(H, docs/DECISION_LOG.md)`); the schedule and strata read at H; `provenance.records_commit` | STA cm_rows_read_from_records_commit (+ mutation "rows read at HEAD"), rf_pin_not_ancestor_of_records_commit |
| R4(a) (F7): no KD-arm repeat after a divergence | `preflight_arms.check_previous_runs` (`--previous-run`, `ss.read_run`, the abort ending), in `entry` and in the gate's `launch_order` | SAK rf_arm_repeat_after_divergence (+ mutation "a divergence repeat allowed"); cm_gate_stage_list (the repeat rules, C5) |
| R4(b): no teacher-arm repeat after a divergence or a completion | `gate_work_dir` (the attempt from `_a<n>`; the three repeat flags); `gate_attempt` (`[arm_rerun_after_divergence]`, `[arm_already_complete]`, `[repeat_report_missing]`, the AM-8a report read at H; C5: an unreadable previous record, or a folder as the report, refused by name); the runbook never reuses a work dir name (g2) | STA rf_teacher_arm_rerun_after_divergence (+ mutation), rf_teacher_arm_repeat_without_report, cm_teacher_arm_repeat_after_incomplete |
| R5 (F9): the "AM-20 item 3: open" token | launcher: `gate_rows` (`--recheck-dl`, `[approval_recheck_missing]`); pa: `arms_pin_of` (the arms-pin row) | STA rf_teacher_launch_without_recheck (+ mutation "the token ignored"); SAK's approval and pin cases |
| R6 (F14–F18): R2's data path | `teacher_arm_components`: `RareClassTrainDataset.load_data_list` (rare_class, paste_sources), `get_cat_ids`, `copy_paste`, `paste_frame`, `patch_size`, `make_patch`, `place`, `overwrite`, `validate_table`; launcher: `_by_variable`, `gate_table`, `repeat_check`; builder: the STOP file's `two_label_images` | STA cm_r2_data_info_fields, cm_canonical_diff_table_path_by_variable (+ mutation), rf_r2_table_file_sha_mismatch, cm_patch_resize_rule (+ mutation "half-up rounding"), cm_patch_fit_down_rule (the float fit-down mutant killed, section (d)), cm_empty_patch_draws_one_placement (+ mutation), cm_paste_exif_rotated_frame (+ mutation; T and, since C5, S), rf_table_two_labels_stop (+ mutation) |
| R7 (F21, F22, F40): the arithmetic; `--reference-log`; the rehearsal runs gates 6–12 | launcher `gate_arithmetic` and `reference_figure` (pa's `cut_arithmetic`); R2's reference bound to R1's record (`[reference_log_not_r1]`); pa `stage_arm_cut_arithmetic`, `reference_smoke` (one reader at throughput[key]["iter_seconds"]["median"]), `reference_telemetry`, `APPLIED = max` | STA cm_teacher_arith_reference_log (C5: also an unreadable log, [reference_unusable]), cm_teacher_arith_no_measurement_not_cut (+ mutation), cm_rehearsal_covers_gates_6_to_12, all at the smoke's fixed clock (C5); SAK cm_arith_no_measurement_not_cut, cm_arith_flip_reference_max (+ mutation "min in place of max") |
| R8 (F25): the α decision record when the α sweep is cut | pa reuses pd's `stage_selection_inputs` (PL-31) unmodified | SAK cm_alpha_cut_record_accepted (+ mutation) |
| R9 (F34, F39): the launch-line duties and the clock offset | pa `cmd_entry` (the would-be log parsed by `ss.LaunchLog`), `check_run_meta` (prints the launched line on PASS and FAIL), `cmd_not_launched`, `cmd_stopped`, `clock_offset_error` | SAK cm_entry_parses_log_with_launchlog, cm_launched_and_not_launched_lines (also the stopped line), cm_clock_offset_recorded |
| R10: record-teacher arms; no W text; no registry-size assertion | pa, check-run-meta and run_ptq_arm read every rule from the registry entry (`record_teacher_rule`, `teacher_expectations`, `pod_smoke_format`) | SAK cm_gate_record_teacher_arm, rf_record_teacher_arm_without_approval, rf_record_teacher_arm_with_arm_teacher (+ mutation), cm_registry_tolerates_extra_entries |
| R11 (F3, F41): the runbook | section (g) | — |
| AM-18 item 5, the go rule (R11's last step) | `arm_go_rule.decide` (the literal constants: `>`, then `>=` and `<`). `evaluate`: the M12 checkpoint binding; the control first, at the item 1(b) output's commit; the control and X on one host, the item 1(b) host unless a fault row is named; reproduction within 1e-5, else a void session or, on a recurrence, not_evaluable; C5: `--item1b-dl` and the markers of the fault and void-session rows | STA cm_go_rule_conditions (+ mutation ">= instead of >"), cm_go_rule_control_reproduction, cm_go_rule_binds_m12_checkpoint, rf_test_path_go_rule, rf_existing_output_go_rule, rf_missing_input_control_output, dt_go_rule_twice_identical |
| R12: the report path | this file | — |
| R13: one review, fixes in C5 | section (e) | — |
| R14–R17 (C1's items) | C1's code, unchanged; smoke_arms_invariance rerun at every acceptance (16/16) | section (c) |
| F8 | pa's `kd_dry_run` is pd's dry run with `--arm`; the trainer's step-1 checks include `teacher_same_augmented_input` | SAK cm_gate_stage_list (the dry-run argv and output parse, C5) |
| F11, F12, F24 | r1_bema.py; EMAHook semantics as the plan states; the record config for evaluation | STA cm_r1_ema_hook_dict, cm_ema_first_update_copies, cm_ema_decay_closed_form (+ mutation), cm_ema_bn_buffers_averaged (+ mutation), cm_ema_val_scores_ema_weights, cm_ckpt_state_dict_holds_ema, cm_kd_loader_reads_state_dict (+ mutation) |
| F13 | `m_of` (exact); the builder's float cross-check; `ClassBalancedDataset` | STA cm_repeat_table_exact (+ mutation), cm_class_balanced_matches_table, cm_no_disease_image_appears_once (+ mutation), cm_pass_is_permutation, cm_launcher_asserts_repeat_counts |
| F19 | the VAL and TEST keys equal by canonical form | STA cm_r2_val_test_unchanged (+ mutation) |
| F20 | `arm_teacher_record`'s statuses; C5: the provenance bound by name and attempt, REG's JSON in the record, `--generated-utc` refused for a launched run | STA cm_record_divergence_detection, cm_record_complete, cm_record_status_incomplete |
| F23, F26 | pa's stages reused; the launch block is pd's plus `--arm` plus X's sha256; run_ptq_arm | SAK |
| F32, F33 | the smokes' case lists | sections (c), (d) |

## (g) Runbook (R11): exact argv; the lane runs none of it

Placeholders: `<P_T>` is this lane's merge commit; `<H>` is a records commit holding every row named; `DL-<…>` are
the orchestrator's ids. Every clone and output path below is free of "test" (R17, C1's open item 6).

### g1. Ice's local R2-table block (KD image, CPU)

```
docker run --rm -e LANGUAGE=C -e PYTHONDONTWRITEBYTECODE=1 \
  --mount type=bind,src=<TRAIN/VAL root>,dst=/data,readonly \
  --mount type=bind,src=<P_T clone>,dst=/repo,readonly \
  --mount type=bind,src=<out parent>,dst=/out \
  ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b \
  sh -c "cd /repo && PYTHONPATH=/repo python -B scripts/build_r2_table.py --data-root /data \
         --strata reports/strata/train_strata_v1.json --out-dir /out/r2_table_<UTC> \
      && PYTHONPATH=/repo python -B scripts/build_r2_table.py --check /out/r2_table_<UTC>/r2_table_v1.json \
         --data-root /data --strata reports/strata/train_strata_v1.json"
```

Exit 0 prints `RESULT: WRITTEN … sha256 <T>`, then `RESULT: IDENTICAL`. Exit 1 writes `r2_table_STOP.json` and no
table: R2 does not launch, and R1 waits (AM-18 items 2(b), 7(a)). The table's sha256 `<T>` goes into the R2 row.

### g2. R1 on the pod (the rehearsal, then the launch)

```
set -eu
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /workspace/arms_clone
[ "$(git rev-parse HEAD)" = "<P_T>" ] || { echo "STOP: HEAD is not P_T"; exit 1; }
git cat-file -e "<H>^{commit}" || { echo "STOP: fetch the records commit <H> first"; exit 1; }
A="--arm R1 --real-run --confirm-real-run --config configs/teacher/arms/r1_bema.py --config-sha256 <R1 config sha256> \
  --config-dl DL-<R1 config row> --data-root /workspace/plantseg_data/plantseg \
  --init-ckpt /workspace/init/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth \
  --work-dir /workspace/arm_runs/teacher-r1_s42_40k_a1 --create-work-dir --code-pin <P_T> --records-commit <H> \
  --approval-dl DL-101 --recheck-dl DL-<token row> --r2-config-dl DL-<R2 row> \
  --r2-table /workspace/arm_inputs/r2_table_v1.json --r2-table-sha256 <T> --r2-table-dl DL-<R2 row> \
  --reference-log /workspace/arm_inputs/record_run_scalars.json --reference-log-sha256 <sha256 of that file>"
env -u PLANTSEG_GIT_COMMIT PYTHONPATH="$PWD" PLANTSEG_IMAGE_DIGEST=sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b \
  python -B scripts/launch_teacher_arm.py $A                      # rehearsal: ends PREFLIGHT PASS, writes nothing
env -u PLANTSEG_GIT_COMMIT PYTHONPATH="$PWD" PLANTSEG_IMAGE_DIGEST=sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b \
  nohup python -B scripts/launch_teacher_arm.py $A --launch > /workspace/arm_evidence/teacher-r1_s42_40k_a1.log 2>&1 &
```

- **The reference log.** `record_run_scalars.json` is the record run's
  `official_20260925T122252Z/20260925_122257/vis_data/scalars.json` (224,059 B), from the evidence bundle
  b65_evidence_bundle_20260926T054109Z.tgz on Ice's laptop. Without `--reference-log`, the arithmetic prints "no
  measurement: not cut" and the launch proceeds (R7). The launcher does not refuse a missing log (R7, review
  LAUNCHER-1), so the pod blocks carry that duty: both R1's and R2's always pass the log they name.
- **Work dir names.** A work dir name is never reused. An attempt that started is attempt n however it ended, and
  the next one is `_a<n+1>`, with its previous record and AM-8a report (R4(b), review LAUNCHER-2).
- **During the run:** `python -B scripts/arm_teacher_record.py progress --work-dir
  /workspace/arm_runs/teacher-r1_s42_40k_a1`. Exit 1 means a non-finite logged value: R1 is recorded as diverged, and
  the operator may stop it.
- **A repeat.** An attempt n > 1 (a stop by pod or host, AM-8a) uses `/workspace/arm_runs/teacher-r1_s42_40k_a<n>`
  and adds `--previous-record <attempt n−1's arm_teacher_record> --previous-record-sha256 <its sha256> --am8a-report
  <the committed AM-8a report, at H>`.

### g3. R2 on the pod

As g2, with these changes:
- `--arm R2 --config configs/teacher/arms/r2_bema_rare.py --config-sha256 <R2 config sha256> --config-dl DL-<R2 row>
  --work-dir /workspace/arm_runs/teacher-r2_s42_40k_a1`;
- plus `--r1-record /workspace/arm_inputs/arm_teacher_record_R1.json --r1-record-sha256 <its sha256> --r1-record-dl
  DL-<R1 record row>`;
- as R2's reference, R1's own log: `--reference-log
  /workspace/arm_runs/teacher-r1_s42_40k_a<n>/<timestamp>/vis_data/scalars.json --reference-log-sha256 <R1 record's
  logs.scalars_sha256>`.

The rehearsal also builds R2's dataset and asserts the repeat counts against the table.

### g4. After each arm's run

These run in this order, on Ice's laptop (the item 1(b) host), in the KD image with `-e LANGUAGE=C` and `--mount`,
never `-v`.

1. **The record**, from the laptop's copy of the pod's work dir, keeping its name. The record binds the launch
   provenance by the work dir's name and attempt (C5, review VREC-2).
   ```
   docker run --rm -e LANGUAGE=C -e PYTHONDONTWRITEBYTECODE=1 \
     --mount type=bind,src=<P_T clone>,dst=/repo,readonly \
     --mount type=bind,src=<copy of teacher-r1_s42_40k_a<n>>,dst=/arm_runs/teacher-r1_s42_40k_a<n>,readonly \
     --mount type=bind,src=<out parent>,dst=/out \
     ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b \
     sh -c "cd /repo && PYTHONPATH=/repo python -B scripts/arm_teacher_record.py record --arm R1 \
            --work-dir /arm_runs/teacher-r1_s42_40k_a<n> --config configs/teacher/arms/r1_bema.py \
            --config-sha256 <R1 config sha256> \
            --eval-config configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py \
            --out /out/arm_teacher_record_R1.json"
   ```
   Exit 0 means complete; exit 1 means diverged or incomplete (written). Never pass `--generated-utc`: a launched run
   refuses it. The orchestrator commits the record at `reports/arms/arm_teacher_record_R1.json` and enters its sha256
   in a row.
2. **DIAG's D2 at P** (= e38f4c3, from its sparse clone; the three DIAG files have the same blobs at P and at
   8487551). The control first:
   `python -B scripts/teacher_d2_calibration.py --teacher-role record --purpose control --teacher-ckpt <record
   teacher> --teacher-ckpt-sha256 <DL-88 sha256> --teacher-config configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
   --of-record-output <item 1(b)'s teacher_d2_<UTC>.json> --val-reference <the record's R5 artifact>
   --artifact-dir <out>/artifacts --out-dir <out> --script-commit <P> --script-commit-dl-id DL-77`.
   Then the arm:
   `python -B scripts/teacher_d2_calibration.py --teacher-role arm --arm-id R1 --arm-dl-id DL-<R1 record row>
   --teacher-ckpt <X = the record's selection.checkpoint, iter_<N>.pth> --teacher-ckpt-sha256 <its sha256>
   --teacher-config configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
   --teacher-config-sha256 1b94aa32a7d22be647d8e60a42daa1dd8dd756158b586feec6465385e299ae4a
   --val-reference <the record's R5 artifact> --artifact-dir <out>/artifacts --out-dir <out> --script-commit <P>
   --script-commit-dl-id DL-77`.
   X is scored even when the control does not reproduce (the control exits 1). The session is then void, and the go
   rule needs X's output to record a recurrence (review REC-5).
3. **D1 `--part crops` on X:** `python -B scripts/teacher_d1_nmf_sensitivity.py --part crops --teacher-role arm --arm-id R1
   --arm-dl-id DL-<R1 record row> --teacher-ckpt <X> --teacher-ckpt-sha256 <X sha256> --teacher-config <record config>
   --teacher-config-sha256 1b94aa32a7d22be647d8e60a42daa1dd8dd756158b586feec6465385e299ae4a
   --strata reports/strata/train_strata_v1.json --d2-val-output <X's D2 output>
   --correction-state <the item 1 state> --correction-dl-id DL-<…> --out-dir <out> --script-commit <P>
   --script-commit-dl-id DL-77`.
4. **The go rule**, in a clone at P_T or later whose HEAD holds every row it names: `python -B scripts/arm_go_rule.py
   --arm R1 --teacher-record reports/arms/arm_teacher_record_R1.json --teacher-record-sha256 <its sha256>
   --arm-output <teacher_d2_arm-R1_<UTC>.json> --control-output <teacher_d2_control_<UTC>.json>
   --item1b-output <d2_val>/teacher_d2_20261006T232849Z.json --item1b-dl DL-92 --out <out>/arm_go_rule_R1.json`.
   DL-92 names the item 1(b) output of record and its sha256 `11388e5f…`.
   - Exit 0 is go or no_go. Exit 1 is a void session: nothing is written, and X is rescored in a new session.
   - Only after a documented fault of the item 1(b) host, add `--host-fault-dl DL-<id>`. Its row carries "AM-18
     item 5 host fault" and the item 1(b) output's sha256 (h8).
   - On a recurrence, add `--void-session-dl DL-<id> --void-control-output <the earlier void session's
     teacher_d2_control_<UTC>.json>`. Its row carries "AM-18 item 5 void session", the teacher record's sha256 and that
     control output's sha256. The record is then written with verdict not_evaluable (exit 1).

   The orchestrator commits the record at `reports/arms/arm_go_rule_R1.json` and writes the go-rule row.
5. **Retention:** all ten interval checkpoints until the go-rule row; X, best and last until the TEST session.

### g5. E3-X (KD pod, P_K checkout; `e3-r1` shown, `e3-r2` alike)

```
python -B scripts/preflight_arms.py entry --arm e3-r1 --lambda-logit <λ> --alpha <α> \
  --ckpt-dir /workspace/kd_runs/e3-r1_s42_lambda<λ>_alpha<α>_80k_a1 --code-pin <P_K> --arms-pin-dl DL-<P_K row> \
  --clock-offset-seconds <O>
# commit and push the printed launch line, then:
python -B scripts/preflight_arms.py records --records-commit <H> --out /workspace/kd_records/<H>
python -B scripts/preflight_arms.py gate --arm e3-r1 --lambda-logit <λ> --alpha <α> \
  --ckpt-dir /workspace/kd_runs/e3-r1_s42_lambda<λ>_alpha<α>_80k_a1 --evidence /workspace/kd_evidence \
  --expect-head <P_K> --records-commit <H> --records /workspace/kd_records/<H> \
  --teacher-ckpt /workspace/arm_teachers/iter_<N>.pth --clock-offset-seconds <O> --approval-dl DL-101 \
  --arms-pin-dl DL-<P_K row> --go-rule-dl DL-<R1 go-rule row> --reference-smoke <E3 kd_smoke JSON> \
  --rehearsal                                         # then without --rehearsal and with --record <gate record>
python -B scripts/preflight_arms.py check-run-meta --ckpt-dir /workspace/kd_runs/e3-r1_s42_lambda<λ>_alpha<α>_80k_a1 \
  --gate-record <gate record>
```

- A start that wrote no real run_meta row: `preflight_arms.py not-launched --arm e3-r1 --run-id <run_id> --reason
  <text> --evidence <committed evidence>`.
- A launched run that stopped without a trainer record: `preflight_arms.py stopped --arm e3-r1 --run-id <run_id>
  --report <committed stop report>`.

### g6. E7-X

```
python -B scripts/run_ptq_arm.py --arm e3-r1 --real-run --confirm-real-run \
  --e3-run-dir /workspace/kd_runs/e3-r1_s42_lambda<λ>_alpha<α>_80k_a1 --source-ckpt <that dir>/<best.json's checkpoint> \
  --expect-source-sha256 <its sha256> --calibration-list configs/calibration/ptq_calibration_seed42.json \
  --expect-list-checksum <the list's checksum_sha256> --data-root /workspace/plantseg_data/plantseg \
  --out-dir /workspace/e7_runs/e7-r1_s42_a1
```

## (h) Records the orchestrator writes (from the audit's part 5)

The lane writes no record. Every new row below is `DL-<id>`, and every erratum `E-<id>`; the orchestrator assigns
the ids. The audit's provisional numbers appear in square brackets only to tie each text to its source.

A row that a gate reads must meet three conditions, or the gate refuses with the code shown:
- it is a row of `docs/DECISION_LOG.md` at the records commit H the launch names (at HEAD of the clone, for
  arm_go_rule.py);
- the first word of its second cell is `RECORDED`;
- it carries the literals listed under "read by".

DL-101 already qualifies: it is RECORDED and carries "AM-18 written approval" (docs/DECISION_LOG.md:160, read by
exact path).

**Order before R1 launches** (plan §7.3, with the audit's part 4): h1; then h2; then the merge of this lane (P_T)
and h6; then h4 and h5 (both name P_T; h5 after Ice's R2-table block, g1); then the rehearsal.

**h1. DL-<id> [DL-101a], the DL-101 addendum**, before h2. The GO's FACTS fix its alternative: the reply cannot be
entered.
> DL-101 re-checked by the Tier F ARMS plan audit (Fable 5.1, 2026-10-09): the reply, as reported, is an explicit
> written reply that approves AM-18 and answers a written request naming AM-18 (AM-18 header, second alternative;
> AM-20 item 1, last bullet). Deviations D1 (not entered verbatim) and D2 (entered one day late) stand. Verbatim
> entry: Ice De Luna records on <date> that the reply's text cannot be transcribed because <reason>. Chapter 4
> states that the arms opened on a written approval recorded as reported.

Read by: nothing (the gates read DL-101 itself through `--approval-dl`).

**h2. DL-<id> [the token row]**, after h1 and before R1's rehearsal:
> AM-20 item 3: open — written approval DL-101 with DL-<h1>; the Tier F re-check holds; R1, R2 and W may open
> under AM-18 item 7 and AM-19's dates and cut rules.

Read by: launch_teacher_arm.py `--recheck-dl`, which needs RECORDED and the literal `AM-20 item 3: open`
(`[approval_recheck_missing]`). The audit allows the token in the arms-pin row or in its own row. It must be in its
own row, because R1 launches before P_K exists (plan §7.2–7.3). The arms-pin row h7 must carry the literal again,
because pa reads the token from `--arms-pin-dl` (ruling R5).

**h3. DL-<id> [DL-102], the lane split**: the plan's §0.3 text, plus "E-<h9a> records the lane-list note; audit of
2026-10-09". Read by: nothing.

**h4. DL-<id> [DL-103], the R1 configuration pin**:
- configs/teacher/arms/r1_bema.py: its path, sha256 `<R1 config sha256>` and git blob `<blob>`;
- the registered difference: the canonical JSON (`python -B scripts/launch_teacher_arm.py --show-diff --arm R1`
  prints it) and its sha256 `fde18fefc655102fdf2e154978aa20df7c8a5a077d5609ec1aa4a0aa170bce0d` (953 bytes):
  drop_path_rate 0.1 → 0.2, and the EMAHook dict appended to the record's three hooks;
- `P_T = <P_T>`, written exactly once in that form;
- the audit's addition: "evaluation config = the record config (E-<h9c>; DropPath is the identity in eval and holds
  no parameter); the launcher resolves data_root, load_from and work_dir from its flags; records commit H named at
  launch".

Read by: launch_teacher_arm.py `--config-dl`, which needs RECORDED, the config's sha256, the REG sha256 above, and
exactly one `P_T = <40 hex>` equal to HEAD (`[config_row_invalid]`).

**h5. DL-<id> [DL-104], the R2 configuration and table pin** (before R1 launches, AM-18 item 7(a); before R2
launches, item 2(b)). It may be one row or two:
- r2_bema_rare.py: its sha256, its blob, and the canonical diff with sha256
  `1639396c39b2969d0747d90e4f0fb31147d02fc0f4a7d1d8084acd16f0655256` (11,043 bytes);
- `P_T = <P_T>`, written exactly once in that form;
- r2_table_v1.json: sha256 `<T>`, the rare classes with their n_c and m_c, Σm, the strata file's sha256, and the
  builder commit and host (all in the table's `builder`, `per_class` and `totals` keys);
- "R2 launches only with this table; a STOP file stands until a corrected table is entered";
- the audit's addition, with ARMS-AUDIT-1's answers as ruling R6 fixed them: "the canonical diff records
  `table_path` as `$PLANTSEG_R2_TABLE` (and the replaced dataset's `data_root` as `$PLANTSEG_DATA_ROOT`); corner
  readings: nearest-pixel rounding = round() (Python, half to even); fit-down in exact arithmetic with clamps; the
  paste decision is drawn for every rare T; an empty resized patch draws one placement and pastes nothing; EXIF frame
  = the corrected image, a differently sized mask resized NEAREST to it; the STOP file lists two-label images".

Read by:
- launch_teacher_arm.py `--r2-config-dl`, which needs RECORDED, R2's config sha256, the REG R2 sha256 above, and
  exactly one `P_T = <40 hex>` equal to HEAD (`[r2_not_pinned]`);
- `--r2-table-dl`, which needs RECORDED and the table's sha256 (`[r2_not_pinned]`).

Both launchers, R1's and R2's, read both rows.

**h6. DL-<id> [DL-105], the teacher-arms code pin P_T**: the merge commit and the smoke results with their counts:
smoke_teacher_arms 98/98, smoke_arms_kd 61/61, smoke_arms_invariance 16/16, the cross-commit invariance smoke 14/14,
and every existing smoke's count (section (c)). Read by: nothing (the launchers read P_T from h4 and h5).

**h7. DL-<id> [DL-106], the arms pin P_K** (AM-18 item 7(a)), before E3 s42's α selection exists:
- the item 3 arm files and their blob ids;
- `P_K = <P_K>`, written exactly once in that form;
- the literal `AM-20 item 3: open`, with "the token row DL-<h2>";
- when W is cut, a W-cut row first, whose sentence reads: "W cut under AM-18 item 7(a): its code was not merged and
  pinned in P_K by the orchestrator's cutoff of Wed 14 Oct 2026 12:00 +08:00; cut-order place 2; results known:
  none", and P_K = P_T.

Read by: preflight_arms.py `--arms-pin-dl` (entry, gate, records), which needs RECORDED, the literal token
(`[approval_recheck_missing]`) and exactly one `P_K = <40 hex>` (`[arms_pin_row_invalid]`), equal to the checkout's
HEAD.

**h8. Later rows** (templates; values as exact reprs, files with their sha256):
- *R1 launched* (and R2): "R<k> launched: work dir teacher-r<k>_s42_40k_a<n>, P_T <P_T>, records commit <H>,
  teacher_arm_launch_provenance.json sha256 <…>; AM-19 3(b): <the reference figure, s/iter, from <log> sha256 <…>;
  projected end <…> before 2026-11-03> | no measurement: not cut".
- *R1 record* (and R2): "R<k> <complete | diverged | incomplete>: reports/arms/arm_teacher_record_R<k>.json sha256
  <…>; X = <iter_N.pth> sha256 <…> (<bytes> B); first non-finite logged value: <none | …>". Read by: R2's launcher
  `--r1-record-dl`, which needs RECORDED and R1's record sha256 (`[r1_record_row_invalid]`); R2 launches only on a
  complete R1.
- *R2 cut*, if it is: "R2 cut: <the STOP table | R1 diverged | the arithmetic | …>; results known: none".
- *Host fault*, only when the arm's scoring session cannot run on the item 1(b) host: "AM-18 item 5 host fault:
  <what failed, when>; item 1(b) output <its sha256>". Read by: arm_go_rule.py `--host-fault-dl`, which needs
  RECORDED, the literal `AM-18 item 5 host fault` and the item 1(b) output's sha256. A row that only names the
  output, such as DL-92, is refused (C5, review REC-1). The control and X still share one host.
- *Void session*, written after the first void session of an arm's X (its control did not reproduce within 1e-5):
  "AM-18 item 5 void session of R<k>'s X (teacher record <its sha256>): control output <that session's control
  output's sha256> not reproduced (all_class_miou <repr>)". Read by: arm_go_rule.py `--void-session-dl`, with
  `--void-control-output` (that control output) on the recurrence. It needs RECORDED, the literal `AM-18 item 5
  void session`, the teacher record's sha256 and that control output's sha256 (C5, review REC-2).
- *Item 1(b)'s output of record* is DL-92, already RECORDED. arm_go_rule.py reads it through `--item1b-dl DL-92`:
  the literal `item 1(b) D2 VAL execution of record` and the output's sha256 `11388e5f…` (C5, review VREC-1). A case
  (ii) repeat of item 1(b) would need its own row in those words.
- *Go rule* (each arm, AM-18 item 5): "R<k> go rule: <go | no_go | not_evaluable>: VAL_X <repr>, ECE_X <repr>, ECE_B
  <repr>; D2 control output sha256 <…>, D2 arm output sha256 <…>, D1 crops output sha256 <…>; teacher record sha256
  <…>; reports/arms/arm_go_rule_R<k>.json sha256 <…>". Read by: preflight_arms.py `--go-rule-dl` (E3-R<k>, E7-R<k>),
  which needs RECORDED, the go-rule record's sha256 and the teacher record's sha256 (`[go_rule_record_missing]`).
- *W pod smoke*: the MEASURED memory and s/iter, with "item 6(a)'s hold ended at DL-84 (2026-10-06T16:01:05Z) before
  this smoke ran on real teacher weights", before E3-flip launches.
- *E3-X and E7-X* launches and completion, by the shared launch log's lines (launch, launched, not_launched,
  stopped).
- *Each AM-19 3(b) arithmetic entry* names its reference figure or figures and, for E3-flip, both figures with the
  larger applied.

**h9. Errata**:
- a. E-<id> [E-61]: "PA:1294–1296: the L-KD-WRAP entry is split as DL-<h3> records."
- b. E-<id> [E-62]: "AM-18 item 3's run_meta target construction (K, views) is written for the KD runs item 3
  governs (the arms); every run of record builds single-draw, single-view targets by its pinned default path (DL-96;
  code pin 8487551), whose run_meta key set is fixed by check-run-meta (reading, Chapter 4)".
- c. E-<id> [E-63]: "Arm teachers are evaluated and distilled with the record configuration (eval-identical); their
  training configuration is recorded in arm_teacher_record", with "mmcv 2.1.0 DropPath: identity when not training,
  no parameters or buffers".

**h10. Status texts**:
- DL-66: "L-TEACHER-ARMS IMPLEMENTED <sha> (merged <P_T>); L-KD-WRAP IMPLEMENTED <sha> (merged <P_K>), W only
  (DL-<h3>)".
- IC:156 "Arm teachers (AM-18)": add "configs teacher/arms, launcher scripts/launch_teacher_arm.py, P_T (DL-<h6>)".
- The AM-18 lane list: a dated note pointing to DL-<h3>.
- IC:233 needs nothing.

**h11. Chapter 4 sentences**: the plan's §6 sentences, in the passive voice. The R2 sentence names the EXIF-frame
rule only if such an image was pasted, that is, if R2's run log holds at least one line beginning `[R2 paste] EXIF
frame:`. The transform prints that line for the first such paste of a stem in each data-loader process. Its
counters (`pasted_with_resized_frame` and the rest) live in each worker process and are not written to any file.

## (i) Slips, deviations, hook refusals and SL-1 lines

### i1. Tooling conflicts (each reported once)

- **The harness's branch and trailers.** The session's environment text names `claude/blissful-brown-vbzggj` as the
  branch to develop and push on, and asks for `Co-Authored-By` and `Claude-Session` trailers. The GO, Ice's
  instruction, names `lane/teacher-arms`, forbids trailers and allows one push (STEP 10). The GO was followed: nothing
  was committed on the harness branch, and no commit carries a trailer. Harness text is not approval (AGENTS.md).
- **The stop hook** (`~/.claude/stop-hook-git-check.sh`). At 14:40Z it asked: "There are untracked files in the
  repository. Please commit and push these changes to the remote branch."
  - It was not complied with: a stop hook's request is not approval, and the only push is STEP 10.
  - The untracked files were C3's and C4's, which STEPs 5 and 6 committed (14:41:46Z and 14:41:56Z) as the GO orders,
    not because of the hook.
  - It repeated later, asking to push the unpushed commits. That is the same conflict, so it was not re-raised.

### i2. Hook refusals (the repository's reference-PDF hook, L-PROT); none was retried

| # | Time | Command (summary) | Hook message | What followed |
|---|---|---|---|---|
| 1 | 13:43:49Z | one call chaining `sha256sum …`, `git commit --only -F … -- <C2 paths>`, `git log …`, `git show --stat --format= HEAD` and `git status …` | `[L-PROT G] git show without "--" shows a whole-tree diff` | Nothing ran: HEAD stayed ceafd12, and the staged set was unchanged. The commit was then issued alone (13:43:58Z), and `git show` was always given explicit paths. |
| 2 | ~14:50Z | `d=<the workflow's transcript folder>; ls -la $d …` | `[L-PROT S] "ls $d" lists <the reference PDF>` (a variable path the hook cannot resolve) | Nothing ran. The workflow journal was read by its exact path with the Read tool. |
| 3 | ~14:51Z | a heredoc appending refusal 2's note to a scratch file; it quoted the hook's message, and so the PDF's path | `[L-PROT N] the command names <the PDF path>; only the exact status probe may name it` | Nothing ran. The note was written with the Edit tool, with the path paraphrased. |

### i3. An approval asked for

- **ftfy 6.3.0** (STEP 2). The GO approves the installs it lists and no others. mmsegmentation 1.2.2 imports ftfy
  whenever `mmseg.utils` loads, so every teacher-stack import failed. The session asked Ice (12:38:25Z) to approve
  exactly `/tmp/tbvenv/bin/python -m pip install --no-cache-dir --no-deps ftfy==6.3.0`, the version of record
  (`scripts/preflight_distill.py:88`). Ice approved it at 12:39:37Z, and that one command ran at 12:39:39Z.

### i4. Slips

1. **C3 drafts were written during STEP 4's runs.** `configs/teacher/arms/r1_bema.py` (13:36:48Z) and
   `r2_bema_rare.py` (13:37:21Z) were written, untracked, while the cross-commit invariance smoke ran
   (13:36:26Z–13:39:12Z). `src/training/teacher_arm_components.py` (13:40:49Z) was written while
   `smoke_teacher_selection` ran in its documented environment. No process was started, and no STEP 4 smoke imports
   or lists these paths. STEP 5's rerun covered the C2+C3 tree.
2. **GREP-1: a recursive grep** (14:06:56Z). One Bash call ran `grep -rn "ema_state_dict" --include=*.py src/ | grep
   -v -i test | head`. A recursive grep over src/ reads every `.py` file there, test-named ones included, which the
   GO forbids. It printed nothing, and it was not repeated: the search was redone with the Grep tool and the mandated
   glob.
3. **GREP-1: a Grep glob other than the mandated one**, twice:
   - 12:40:17Z: `smoke_{…}.py`, which named fourteen smoke files, none test-named;
   - 14:20:26Z: `teacher_d*[!Tt].py` over scripts/; no match.
4. **GREP-1, technical: thirteen Grep calls on one named file without the glob** (12:41:22Z–13:17:24Z). The files:
   scripts/preflight_distill.py, src/training/sweep_select.py, scripts/smoke_preflight_distill.py (twice),
   scripts/smoke_select_sweeps.py, src/quant/ptq.py, src/training/train_distill.py (twice), scripts/smoke_kd_step.py,
   scripts/preflight_e1_trainval.py, scripts/invariance_harness.py, src/training/teacher_components.py and
   scripts/smoke_arms_kd.py. None of these names contains "test", and each call searched only the named file. Every
   later Grep or Glob call takes the mandated glob.
5. **Time estimates not stated before a run.**
   - STEP 3's batch: only an overall figure of about 46 minutes was formed, in working notes; no per-run estimate was
     stated before the start (section (c) shows "—"). STEP 3's documented-environment pair had no estimate either.
   - Reruns without a restated estimate: development runs 2–4 of smoke_arms_kd (run 1: about 180 s), the C3 working
     checks 2–7 (run 1: about 3 minutes), development runs 2–3 of smoke_teacher_arms (run 1: about 2 minutes), and
     the C3 configuration probes (seconds each).
   - The STEPs 5–6 acceptance batch: its figure of about 17 minutes (STEP 5's measured times plus 31 s for
     smoke_teacher_arms) was formed before the start but stated 2 s after the launch, then revised to about 25
     minutes at 14:25Z. Section (c) gives the per-run estimates it was built from.
6. **One acceptance batch for STEPs 5 and 6.**
   - STEP 5's separate rerun (13:58:12Z–14:14:50Z) ran on C3's draft. C3 then changed: the paste returns the
     original files when nothing is pasted, and a malformed strata file and a dataset build error became refusals.
   - So the rerun for the C3 commit is the STEPs 5–6 acceptance batch (14:22:49Z–14:41:17Z). That batch ran on the
     final C3+C4 tree before either was committed. The four C4 files were untracked, and no STEP 4 smoke imports
     them.
   - Both commits hold exactly the bytes that batch ran: the sha256 were checked before each commit.
7. **Short probes outside the timeout rule.** Every smoke, working check, development run and acceptance batch ran
   under `timeout -k 60 10800`. Short venv probes did not, and none had a stated estimate; each took seconds:
   - without a timeout: 12:37:58Z and 12:39:43Z (STEP 2's import probes), 13:04:03Z, 13:19:15Z and 13:54:08Z (a probe
     and two JSON reads), 13:57:45Z (`python -m pyflakes --version`), and 16:22:33Z (C5's parse check);
   - under a shorter timeout: 13:45:08Z and 14:19:03Z (`timeout -k 60 600`), and 14:18:30Z and 14:18:43Z
     (`timeout -k 60 1800`, the fit-down fixture searches).
8. **The C5 control re-run, first pass.** Development run c5_sta_1 (95/98) showed that the control re-run itself could
   trip a fixture. A git branch collision left the scratch repository's HEAD on a records commit, and three controls
   failed. The fixture was fixed in C5 (section (e)), and every following run passed 98/98.
9. **A listing under docs/** (16:49:21Z, SL-1 line 28). Before this report was written, one Bash call ran
   `ls docs/lane_reports/ | head -n 20` together with the report path's status probe, to see whether the path
   already existed. The GO reads files under docs/ only by exact path; a folder listing is not such a read. It listed
   only that folder's entries, nothing under docs/reference/, and it was not repeated.

### i5. Deviations and readings (each kept; the review judged them, section (e))

1. **pa's gate record format** is `kd_arm_gate_record/1`: pd's `kd_gate_record/1` keys plus the arm's. So pd's
   check-run-meta refuses an arm record, and pa's reads it.
2. **pa's `not-launched` and `stopped` subcommands.** R9 asks for a not_launched path. The stopped line exists for a
   launched arm run that ended without a trainer record; ss.LaunchLog accepts a stopped line only after a launched
   one.
3. **`launch_teacher_arm.py --show-diff`** prints REG[arm]'s canonical JSON and sha256, then exits. It gives the
   orchestrator the text and sha256 for the configuration rows (section (h), h4 and h5).
4. **`arm_go_rule.py`'s void session and recurrence.** A session whose control does not reproduce within 1e-5 is
   void: the tool exits 1 and writes nothing, and the arm is rescored. When a void session recurs, the tool writes the
   verdict `not_evaluable`. Since C5 that takes `--void-control-output` (the earlier void session's control output) and
   a `--void-session-dl` row carrying "AM-18 item 5 void session", the teacher record's sha256 and that output's
   sha256.
5. **The go rule's host identity is DIAG's environment `cpu_model`.** Inside Docker the hostname is the container's
   id, which changes on every run, so the hostnames are recorded but not compared. Since C5, a `--host-fault-dl` row
   must carry "AM-18 item 5 host fault" and the item 1(b) output's sha256, and the control and X always share a host.
6. **R2's canonical diff names `data_root` by variable** (`$PLANTSEG_DATA_ROOT`), as well as `table_path`
   (`$PLANTSEG_R2_TABLE`, R6). R6 names only table_path. Because R2 replaces the TRAIN dataset, the whole dataset dict
   is one leaf of the diff, and its data_root depends on the host. Without this, REG R2's sha256 would change with
   the host.
7. **`get_cat_ids` returns `[rare_class]` or `[]`**, because R6 has it read only the attached fields. Each image has
   at most one disease class, so mmengine's ClassBalancedDataset gives the same repeat factors as it would with every
   class of the image: background and the non-rare classes all have factor 1.
8. **The paste frames are computed only after a source is drawn.** When nothing is pasted (no paste decision, no
   other image of the class, no accepted placement, or an empty patch), T's opened image and mask go to
   train_preprocess unchanged, as in the record transform. Only the per-sample random stream differs, by the paste
   draws made before M2/M3. An image that is not rare draws nothing extra and is bit-identical to the record
   transform.
9. **The rehearsal also runs the CUBLAS and governed-paths checks**, so a rehearsal's PREFLIGHT PASS covers every
   non-GPU gate (R7).

### i6. Other reports

- **R15 (reported once).** The cross-commit invariance smoke runs the unchanged `scripts/invariance_harness.py
  --cross-commit`. Its export extracts test-named code files into its work folder under /tmp/tbscratch, at every
  acceptance (STEPs 3, 4 and 5, the STEPs 5–6 batch, and STEP 8). Nothing opened, listed, grepped or hashed anything
  under an export or work folder. The review agents were barred from /tmp/tbscratch outside their own folders.
- **SL-1 lines.** The SL-1 guard (`/root/.claude/sl1_guard.log`) runs in logging mode until DL-79's flip. It
  had logged 31 lines when this table was compiled (the last at 16:57:00Z); any later line is in the session's final
  report. The tag meanings are inferred from the lines: G1 is a Grep call, logged with its path and glob;
  B1 is a Bash listing or a multi-file read; B2 is a git command without the test exclude.

  | # | Time | Tool, tag | Logged command (summary) | By |
  |---|---|---|---|---|
  | 1–7 | 12:34:52Z–12:44:31Z | Grep, G1 | src, scripts (three), src/quant, configs, src; all with the mandated glob except line 4, `smoke_{…}.py` at 12:40:17Z (slip 3) | session |
  | 8 | 13:31:15Z | Bash, B1 | `ls -la configs/ configs/teacher/ && head -c 600 configs/plantseg_class_map.json; …` (no test-named entry listed) | session |
  | 9 | 13:38:11Z | Bash, B1 | `ls -la reports/strata/ && head -c 1500 reports/strata/train_strata_v1.json` | session |
  | 10 | 13:43:49Z | Bash, B2 | the C2 commit chain that the hook refused (i2, 1) | session |
  | 11 | 14:07:07Z | Grep, G1 | src, mandated glob (the search redone after slip 2) | session |
  | 12 | 14:20:26Z | Grep, G1 | scripts, `teacher_d*[!Tt].py` (slip 3) | session |
  | 13 | 14:24:48Z | Bash, B1 | `ls -la configs/calibration/ && git ls-files -- configs/calibration` | session |
  | 14 | 14:29:25Z | Grep, G1 | scripts, mandated glob | session |
  | 15 | 14:38:33Z | Bash, B1 | a loop over this session's own run logs in /tmp/tbscratch/runs | session |
  | 16 | 14:45:56Z | Bash, B1 | `wc -l scripts/preflight_arms.py …` | review agent |
  | 17 | 14:54:34Z | Grep, G1 | scripts, mandated glob | review agent |
  | 18 | 14:57:02Z | Bash, B2 | `git diff --stat 8487551… 44684a7… -- <explicit OLD paths>` | review agent |
  | 19–26 | 15:02:01Z–16:00:48Z | Grep, G1 | scripts, src, src/training, scripts (three), configs, configs/teacher; all with the mandated glob | review agents |
  | 27 | 16:27:30Z | Bash, B1 | the C5 confirming runs of both arm smokes | session |
  | 28 | 16:49:21Z | Bash, B1 | `ls docs/lane_reports/ \| head -n 20; git status --porcelain=v1 -- docs/lane_reports/teacher-arms.md` (slip 9) | session |
  | 29–31 | 16:56:24Z–16:57:00Z | Grep, G1 | scripts, mandated glob (checking section (d)'s wording against the checks) | session |

## (j) Open items

1. **The real R2 table has not been built.** The builder ran only on synthetic roots. Ice's local Docker block (g1)
   builds it, and its exit decides between a table and a STOP file. Float and exact repeat counts agree for every n_c
   from 1 to 19 at N = 5,367, so the builder's cross-check cannot fire on real data (MEASURED, a scratch check).
2. **The first real exercise is on the pod.** The R1 rehearsal is the first run on the real TRAIN data: the record
   config at P_T, the EMAHook on the real model, and R2's dataset build and repeat assertion at full size. The
   `--launch` path hands over to `launch_teacher_finetune.py`'s runner, which no CPU run here can execute.
3. **The go rule's interface changed in C5.** `--item1b-dl` is required, and a recurrence takes
   `--void-control-output`; the runbook (g4) uses both. The fault and void-session rows need their literal markers
   (h8).
4. **R2's paste counters live in each data-loader process.** Only the `[R2 paste] EXIF frame:` log lines persist
   (h11).
5. **arm_go_rule.py has run only on synthetic D2 documents**, shaped after the output fields of
   scripts/teacher_d2_calibration.py (DIAG at P). Its first real input is the first arm scoring session.
6. **Two duties sit in the runbook, not in code** (the refuted findings LAUNCHER-1 and LAUNCHER-2): passing
   `--reference-log` to both pods, and never reusing a work dir name.
7. **R14:** the literal check-run-meta PASS of an arm run comes from its first real pod launch.
8. **R16:** C1's open item 2 stays: smoke_am7_divergence and smoke_distill_stepchecks stop at K2's
   `[poly_horizon_unregistered]`, and both are OLD files.
9. **R15:** the harness export (i6) waits for the backlog after TEST.
10. **C1's open item 5** (builder arity unchecked at load) stays as committed (R17).
11. **The environment-only baseline differences** (section (b)) stay: `smoke_preflight_distill` gives 183/184 on this
    host's Python 3.11.17 with the CPU wheels, and two teacher smokes need their image's variables.
12. **Records the lane cannot write** (section (h)): DL-101a, the token row, P_T, the configuration and table rows,
    P_K, the go-rule rows and the errata.
