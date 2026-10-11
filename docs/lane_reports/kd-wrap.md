# Lane report — L-KD-WRAP · AM-18 item 3(b)'s arm W, run stage E3-flip (`lane/kd-wrap`)

Session KDW · L-KD-WRAP, Phase B (cloud), Sun 11 Oct 2026 (UTC); Opus 5.5 at max effort (the GO's line 2).
Tags as in the earlier lane reports: MEASURED (run here), REPOSITORY-PROVEN (file:line at the lane's last code
commit), DOCUMENTED (the GO, its amendment A1, the plan, the audit or a record), INFERRED.

- **GO.** `kdwrap_phase_b_go_20261011.md`, 16,929 B, sha256
  `b4abbe263980c76e2bea8f7c198586605989b6e4e8fbfb12bc999fa0899a8e74`, with GO AMENDMENT A1 (orchestrator, Sun 11 Oct
  2026, 20:3x +08:00; from L-TEACHER-ARMS C1's report, sha256 `a996ef82…`) and the orchestrator's K5 reading ("less
  the first" is cm_flip_gate_record_teacher_only, which moves to pa's smoke; cm_flip_registry_entry stays).
- **Attachments** (STEP 0, MEASURED, both match the GO): the plan `arms_phase_a_plan_20261010.md`, 84,260 B, sha256
  `da3c0c4cec675e68303488c234f0dc1a957956e031b9530c44b60bd053fb11bf`; the Tier F audit
  `arms-plan-audit-fable-20261010.md`, 31,566 B, sha256
  `2a1e8eadfee3e93032f53cc09af95469be38bc668bc65af695f7c6b62ba75f32`.
- **Base.** `ceafd1296ce373e8959a41af12586e2e7ebbfb0a`, L-TEACHER-ARMS C1 (GO ruling K1; the deviation of form is in
  (j)).
- **Writes.** Only the five declared paths: `src/distill/flip_targets.py`, `configs/kd_arms/e3_flip.json`,
  `scripts/smoke_kd_flip.py`, `scripts/smoke_kd_flip_pod.py` and this report. No existing file changes; the lane runs
  no pod and writes no record.

## (a) Commits

All on `lane/kd-wrap`, each by `git commit --only -- <declared paths>` after a scoped `git status` and `git diff`;
no message carries a trailer (STEP 8's check, (c)).

| # | sha | subject | paths (+/−) | total |
|---|---|---|---|---|
| C1 | `d5726083b32c924f53a2d7f5295004ef18ab05a3` | L-KD-WRAP C1: FlipTargets (flip-averaged KD targets behind the arm seam) and its math checks | `src/distill/flip_targets.py` (+179 −0), `scripts/smoke_kd_flip.py` (+1120 −0) | +1299 −0 |
| C2 | `1e72f032c362af3ae86bfcb48756e7ea2b6abb8b` | L-KD-WRAP C2: E3-flip registry entry, integration checks, W pod smoke | `configs/kd_arms/e3_flip.json` (+19 −0), `scripts/smoke_kd_flip.py` (+249 −52), `scripts/smoke_kd_flip_pod.py` (+853 −0) | +1121 −52 |
| C3 | `0b0cc7e53659ee7e616f6060158f2e830c1dc5cd` | L-KD-WRAP C3: review fixes | `src/distill/flip_targets.py` (+14 −3), `scripts/smoke_kd_flip.py` (+3 −2), `scripts/smoke_kd_flip_pod.py` (+41 −9) | +58 −14 |
| C4 | this commit | L-KD-WRAP C4: lane report | `docs/lane_reports/kd-wrap.md` (new) | — |

A commit cannot name its own sha; C4's sha and the pushed tip are in the session's STEP 9 report and on the remote
ref.

Files at the last code commit (C3, `0b0cc7e53659ee7e616f6060158f2e830c1dc5cd`), REPOSITORY-PROVEN:

| path | lines | bytes | sha256 | git blob |
|---|---|---|---|---|
| `src/distill/flip_targets.py` | 190 | 12695 | `83344833a14df3dec267331d772f92c115bd2c7e151c08d28f621b39db042556` | `a08814dcde95d0cc792182e60fbcfe00b9e360bd` |
| `configs/kd_arms/e3_flip.json` | 19 | 349 | `f6198343b73a00b320ca84be605e41be25ccc53e2ca9b49a0c8828194a57a404` | `e7f29e8ee670efdf30a4208881fa7134559208b8` |
| `scripts/smoke_kd_flip.py` | 1318 | 74564 | `4a394fa7ba5244a867f45b47fabb497f89c95c2293ccbd41ea5fc57946c9851b` | `714a125511643b9c8a3eeefdf2ca8705061cdc2d` |
| `scripts/smoke_kd_flip_pod.py` | 885 | 52209 | `700256934f89bb1c666b9fe2f56c71fa61c9897ddbc4ffa33275d57808abd93c` | `d5d5ec29e2c0b5debc2f8214465b4d4e7d098269` |

## (b) STEPS 1–3 and the pip freeze (MEASURED)

### STEP 1 (12:24–12:31 UTC)

Before step 5, HEAD named `refs/heads/claude/sharp-hypatia-airrh2` at `2e3c88a` (= master, the GO's FACTS), and the
scoped status was empty.

| # | command | output |
|---|---|---|
| 1 | `git status --porcelain=v1 -- . <X>` | (nothing), exit 0 |
| 2 | `git ls-remote --heads origin lane/kd-wrap` | (nothing), exit 0 |
| 3 | `git fetch origin lane/teacher-arms` | `* branch lane/teacher-arms -> FETCH_HEAD` / `* [new branch] lane/teacher-arms -> origin/lane/teacher-arms`, exit 0 |
| 4 | `git rev-parse FETCH_HEAD` | `ceafd1296ce373e8959a41af12586e2e7ebbfb0a` (C1 itself, no later commit) |
| 5 | `git branch lane/kd-wrap ceafd1296ce373e8959a41af12586e2e7ebbfb0a && git symbolic-ref HEAD refs/heads/lane/kd-wrap && git checkout --no-overlay HEAD -- . <X>` | first attempt (12:25) refused by the auto-mode classifier, not run; re-run once on Ice's approval (12:30): nothing printed, exit 0 |
| 6 | `git rev-parse HEAD` | `ceafd1296ce373e8959a41af12586e2e7ebbfb0a` |
| 7 | `git symbolic-ref HEAD` | `refs/heads/lane/kd-wrap` |
| 8 | `git status --porcelain=v1 -- . <X>` | (nothing) |
| 9 | `git diff --stat ceafd1296ce373e8959a41af12586e2e7ebbfb0a -- . <X>` | (nothing) |
| 10 | `git diff --cached --stat ceafd1296ce373e8959a41af12586e2e7ebbfb0a -- . <X>` | (nothing) |
| 11 | `git status --porcelain=v1 -- docs/reference/reference.pdf` | (nothing) |
| state | `git rev-parse HEAD` / `git symbolic-ref HEAD` / `git branch --list 'lane/kd-wrap'` / scoped status | `ceafd12…` / `refs/heads/lane/kd-wrap` / `* lane/kd-wrap` / (nothing) |

### STEP 2 (12:34–12:36 UTC)

The packages were named before installing: torch and torchvision (the GO's command), numpy and pillow (the only
other third-party modules the KD sources and the STEP 3 smokes import at run time; the teacher stack is not
installed).

| command | exit | time |
|---|---|---|
| `/usr/bin/python3.11 -m venv /tmp/kwvenv` | 0 | 7 s |
| `/tmp/kwvenv/bin/python -m pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu` | 0 | 49 s (estimate not stated) |
| `/tmp/kwvenv/bin/python -m pip install --no-cache-dir numpy==1.26.4 pillow==12.3.0` (requirements.lock's pins) | 0 | 7 s (estimate ~20 s) |
| `/tmp/kwvenv/bin/python -m pip freeze` | 0 | — |
| `mkdir -p /tmp/kwscratch` | 0 | — |

The first install resolved numpy 2.4.6 and pillow 12.3.0 as torch's dependencies; the second re-pinned numpy to
1.26.4 (requirements.lock) and left pillow at its lock version. The freeze:

```
certifi==2022.12.7
charset-normalizer==2.1.1
filelock==3.32.3
fsspec==2026.7.0
idna==3.4
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==1.26.4
pillow==12.3.0
requests==2.28.1
sympy==1.14.0
torch==2.1.0+cpu
torchvision==0.16.0+cpu
typing_extensions==4.16.0
urllib3==1.26.13
```

Python 3.11.17 (`/usr/bin/python3.11`), torch 2.1.0+cpu, CUDA not available, 4 threads, `nproc` 4.

### STEP 3, baseline at `ceafd12` (12:36–13:15 UTC)

Each run alone, as
`timeout -k 60 10800 env TMPDIR=/tmp/kwscratch PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/plantseg-thesis
/tmp/kwvenv/bin/python -B scripts/<smoke>.py [args]`.
These counts define "pass" for every later acceptance.

| # | smoke | args | exit | estimate | measured | counts |
|---|---|---|---|---|---|---|
| 1 | `smoke_frozen_blobs.py` |  | 0 | ~1 s | 0.1 s | FROZEN BLOBS OK (10/10) |
| 2 | `smoke_invariance_distill.py` |  | 0 | — (not stated) | 296.1 s | PASS (112/112) |
| 3 | `smoke_invariance_distill.py` | `--cross-commit 8487551ca67bd50ffbe911d1f3e93244841cbc9f WORKTREE` | 0 | 3–6 min | 373.9 s | PASS (14/14) |
| 4 | `smoke_arms_invariance.py` |  | 0 | 10–14 min | 561.0 s | PASS (16/16) |
| 5 | `smoke_kd_step.py` | `--selfcheck` | 0 | 2–4 min | 57.0 s | PASS (108/108) |
| 6 | `smoke_distill_switches.py` |  | 0 | ~1 min | 4.2 s | PASS (116/116) |
| 7 | `smoke_preflight_distill.py` |  | 0 | 2–5 min | 48.7 s | PASS (184/184) |
| 8 | `smoke_distill.py` |  | 0 | ~1 min | 4.9 s | PASS (95/95) |
| 9 | `smoke_eval_stage_artifacts.py` |  | 0 | — (not stated) | 7.0 s | PASS (66/66) |
| 10 | `smoke_select_sweeps.py` |  | 0 | 1–3 min | 160.4 s | PASS (499/499) |
| 11 | `smoke_loader_parity.py` |  | 0 | — (not stated) | 5.0 s | PASS (7/7) |
| 12 | `smoke_distill_schedule.py` |  | 0 | — (not stated) | 60.3 s | PASS (32/32, 2 skipped) |

## (c) Every run

Every run used STEP 3's command form, under `timeout -k 60 10800`, one at a time; the acceptance suites ran through a
sequential runner (`/tmp/kwscratch/run_suite.sh`, scratch, the same command per run). Estimates were made before each
run and are INFERRED; times are wall-clock, MEASURED.

### Development runs (not acceptance)

| run | command | exit | estimate | measured | counts |
|---|---|---|---|---|---|
| kd_flip dev 1 (C1 tree) | `scripts/smoke_kd_flip.py` | 0 | 3–5 min | 120.0 s | PASS 28/28 |
| kd_flip dev 2 (C1 tree) | `scripts/smoke_kd_flip.py` | 0 | — (not stated) | 115.8 s | PASS 28/28 |
| pod selfcheck dev 1 (C2 tree) | `scripts/smoke_kd_flip_pod.py --selfcheck` | 1 | ~1 min | 24.3 s | FAIL 16/17 (my bug in the selfcheck's mutation case, (j) S2) |
| pod selfcheck dev 2 | `scripts/smoke_kd_flip_pod.py --selfcheck` | 0 | — (not stated) | 24.2 s | PASS 17/17 |
| kd_flip dev 3 (C2 tree) | `scripts/smoke_kd_flip.py` | 0 | ~2.5 min | 117.4 s | PASS 32/32 |
| pod selfcheck dev 3 (with the driver case) | `scripts/smoke_kd_flip_pod.py --selfcheck` | 0 | — (not stated) | 24.4 s | PASS 17/17 |
| pod selfcheck dev 4 (C3 tree, before its commit) | `scripts/smoke_kd_flip_pod.py --selfcheck` | 0 | ~25 s | 29.1 s | PASS 17/17 |
| kd_flip dev 4 (C3 tree, before its commit) | `scripts/smoke_kd_flip.py` | 0 | ~2 min | 112.3 s | PASS 32/32 |

### Acceptance at C1 (13:21–13:48 UTC, the C1 tree before its commit; estimate ~28 min, measured 26.6 min)

| smoke | args | exit | estimate | measured | counts | = STEP 3 |
|---|---|---|---|---|---|---|
| `smoke_kd_flip.py` |  | 0 | — | 119.0 s | PASS (28/28) | new |
| `smoke_frozen_blobs.py` |  | 0 | — | 0.1 s | FROZEN BLOBS OK (10/10) | yes |
| `smoke_invariance_distill.py` |  | 0 | — | 267.1 s | PASS (112/112) | yes |
| `smoke_invariance_distill.py` | `--cross-commit 8487551ca67bd50ffbe911d1f3e93244841cbc9f WORKTREE` | 0 | — | 335.1 s | PASS (14/14) | yes |
| `smoke_arms_invariance.py` |  | 0 | — | 534.3 s | PASS (16/16) | yes |
| `smoke_kd_step.py` | `--selfcheck` | 0 | — | 55.2 s | PASS (108/108) | yes |
| `smoke_distill_switches.py` |  | 0 | — | 3.9 s | PASS (116/116) | yes |
| `smoke_preflight_distill.py` |  | 0 | — | 45.5 s | PASS (184/184) | yes |
| `smoke_distill.py` |  | 0 | — | 5.4 s | PASS (95/95) | yes |
| `smoke_eval_stage_artifacts.py` |  | 0 | — | 7.1 s | PASS (66/66) | yes |
| `smoke_select_sweeps.py` |  | 0 | — | 157.5 s | PASS (499/499) | yes |
| `smoke_loader_parity.py` |  | 0 | — | 4.9 s | PASS (7/7) | yes |
| `smoke_distill_schedule.py` |  | 0 | — | 59.7 s | PASS (32/32, 2 skipped) | yes |

### Acceptance at C2 (13:55–14:21 UTC, the C2 tree before its commit; estimate ~29 min, measured 26.3 min)

| smoke | args | exit | estimate | measured | counts | = STEP 3 |
|---|---|---|---|---|---|---|
| `smoke_kd_flip.py` |  | 0 | — | 118.7 s | PASS (32/32) | new |
| `smoke_kd_flip_pod.py` | `--selfcheck` | 0 | — | 20.7 s | PASS (17/17) | new |
| `smoke_frozen_blobs.py` |  | 0 | — | 0.1 s | FROZEN BLOBS OK (10/10) | yes |
| `smoke_invariance_distill.py` |  | 0 | — | 252.0 s | PASS (112/112) | yes |
| `smoke_invariance_distill.py` | `--cross-commit 8487551ca67bd50ffbe911d1f3e93244841cbc9f WORKTREE` | 0 | — | 342.0 s | PASS (14/14) | yes |
| `smoke_arms_invariance.py` |  | 0 | — | 519.6 s | PASS (16/16) | yes |
| `smoke_kd_step.py` | `--selfcheck` | 0 | — | 51.2 s | PASS (108/108) | yes |
| `smoke_distill_switches.py` |  | 0 | — | 4.1 s | PASS (116/116) | yes |
| `smoke_preflight_distill.py` |  | 0 | — | 43.5 s | PASS (184/184) | yes |
| `smoke_distill.py` |  | 0 | — | 4.5 s | PASS (95/95) | yes |
| `smoke_eval_stage_artifacts.py` |  | 0 | — | 6.2 s | PASS (66/66) | yes |
| `smoke_select_sweeps.py` |  | 0 | — | 148.2 s | PASS (499/499) | yes |
| `smoke_loader_parity.py` |  | 0 | — | 4.6 s | PASS (7/7) | yes |
| `smoke_distill_schedule.py` |  | 0 | — | 59.4 s | PASS (32/32, 2 skipped) | yes |

### Final runs (STEP 7, alone, at C3 `0b0cc7e53659ee7e616f6060158f2e830c1dc5cd`, 15:51:16–16:13:19 UTC; 14 runs; estimate ~26.5 min in all, measured 22.0 min of run time)

| smoke | args | exit | estimate | measured | counts | = C2 / STEP 3 |
|---|---|---|---|---|---|---|
| `smoke_kd_flip.py` |  | 0 | ~2 min | 100.4 s | PASS (32/32) | yes |
| `smoke_kd_flip_pod.py` | `--selfcheck` | 0 | ~25 s | 19.2 s | PASS (17/17) | yes |
| `smoke_frozen_blobs.py` |  | 0 | ~1 s | 0.1 s | FROZEN BLOBS OK (10/10) | yes |
| `smoke_invariance_distill.py` |  | 0 | ~4.5 min | 208.5 s | PASS (112/112) | yes |
| `smoke_invariance_distill.py` | `--cross-commit 8487551ca67bd50ffbe911d1f3e93244841cbc9f WORKTREE` | 0 | ~6 min | 274.5 s | PASS (14/14) | yes |
| `smoke_arms_invariance.py` |  | 0 | ~9 min | 430.4 s | PASS (16/16) | yes |
| `smoke_kd_step.py` | `--selfcheck` | 0 | ~1 min | 47.8 s | PASS (108/108) | yes |
| `smoke_distill_switches.py` |  | 0 | ~5 s | 3.6 s | PASS (116/116) | yes |
| `smoke_preflight_distill.py` |  | 0 | ~45 s | 37.3 s | PASS (184/184) | yes |
| `smoke_distill.py` |  | 0 | ~5 s | 4.2 s | PASS (95/95) | yes |
| `smoke_eval_stage_artifacts.py` |  | 0 | ~7 s | 6.3 s | PASS (66/66) | yes |
| `smoke_select_sweeps.py` |  | 0 | ~2.5 min | 128.5 s | PASS (499/499) | yes |
| `smoke_loader_parity.py` |  | 0 | ~5 s | 4.5 s | PASS (7/7) | yes |
| `smoke_distill_schedule.py` |  | 0 | ~1 min | 52.9 s | PASS (32/32, 2 skipped) | yes |

### Checks after the commits

Before each commit (MEASURED): `git status --porcelain=v1 -- . <X>` listed only the commit's declared paths,
`git diff --stat -- . <X>` only their changes, and each message parsed to no trailer under `git interpret-trailers
--parse`; each commit then ran as `git add -- <paths>` and `git commit --only -F <message> -- <paths>`. After each
commit the scoped status was empty. The lint gate ((j) D6) was clean on every committed version of the three Python
files.

STEP 8's two checks run on the final tip, after this report is committed, so their output cannot be in it; the session's
STEP 9 report gives them:

- `git log --format=%B ceafd1296ce373e8959a41af12586e2e7ebbfb0a..HEAD | git interpret-trailers --parse` must print
  nothing;
- `git diff --name-only ceafd1296ce373e8959a41af12586e2e7ebbfb0a HEAD -- . <X>` must print exactly the five declared
  paths.

## (d) The mutations and their killing checks

smoke_kd_flip runs each mutation by monkeypatch, never on the committed tree: the eight function-level ones inside
the smoke's own process (a context manager patches `src.distill.flip_targets` and restores it),
the two run-level ones inside a trainer worker process (`instrument()` patches the worker's modules before its
`run()` call). A mutation's check PASSes when its named check passes on the unmutated code and fails under the
mutation. The pod selfcheck's one mutation patches the module's `DRAWS_PER_STEP` in process. Results from the final
smoke_kd_flip run (STEP 7):

| # | mutation (how) | killed by | parts that fail under it |
|---|---|---|---|
| 1 | `mut_average_logits`: the targets built from the averaged logits, both views = (z1 + z2)/2, so each term takes the softmax of averaged logits (`FlipViewTargets.__init__` patched) | `cm_never_logits` (PASS) | ['Logit KD averages probabilities', 'CWD averages probabilities'] |
| 2 | `mut_no_flip_back`: `flip_targets.flip_back` = identity: view 2 stays on the flipped grid | `cm_flip_back_alignment` (PASS) | ["view 2 is the aligned map, call 2's logits flipped back"] |
| 3 | `mut_shared_softmax`: the CWD term's per-channel spatial softmax replaced by the Logit-KD term's class softmax (`spatial_probs` patched) | `cm_per_term_softmax` (PASS) | ['CWD at T 2 is its reference', "CWD's softmax is over the cells of each channel"] |
| 4 | `mut_feature_from_flipped_view`: after build(), call 1's Stage 3 overwritten in place by call 2's, flipped back (`FlipTargets.build` wrapped) | `cm_feature_term_unflipped_stage3` (PASS) | ["build() leaves call 1's Stage 3 untouched"] |
| 5 | `mut_one_batched_call`: build() makes one forward on the crop and its flip concatenated, one draw for both views, in place of call 2 (`FlipTargets.build` replaced) | `cm_nmf_two_draws_per_step` (PASS) | ["view 1 is call 1's output (draw 1)", 'build() refuses a call 2 that takes 0 draws', 'build() refuses a call 2 that takes 2 draws'] |
| 6 | `mut_student_sees_flip`: run level, in the worker: a forward pre-hook feeds the student the flipped crop in training | `cm_student_one_forward_original_crop` (PASS) | ['step 1: one training forward, on the batch itself', 'step 2: one training forward, on the batch itself', 'step 3: one training forward, on the batch itself'] |
| 7 | `mut_t_1`: the Logit-KD target's class softmax at T 1 whatever T is passed (`class_probs` patched) | `cm_logit_kd_prob_average_T4` (PASS) | ["the value is the reference's", "the gradient is the reference's"] |
| 8 | `mut_flipped_mask`: both terms flip the validity mask before using it (`logit_kd_loss` and `cwd_logit_loss` wrapped) | `cm_validity_mask_original_orientation` (PASS) | ['both terms use the mask as given'] |
| 9 | `mut_call2_inside_probe`: run level, in the worker: at step 1 the trainer's teacher probe hook is re-registered around call 2 | `cm_step1_checks_pass` (PASS) | ["the trainer's 20 checks PASS", 'RESULT: PASS', "step 1: the probe's hook is on call 1 and gone by call 2"] |
| 10 | `mut_finite_always_true`: `FlipViewTargets.finite` returns True (A1.1) | `cm_abort_teacher_finite_reads_targets` (PASS) | ['with call 2 non-finite, finite is false', "the record's teacher_finite is false with those targets"] |

## (e) The review (DL-24, STEP 6)

One read-only verification workflow on `ceafd12..1e72f03` (STEP 6, K10), requested in plain language:
`kdwrap-dl24-review`, run `wf_bccf20f9-6d5`, 14:47–15:43 UTC (55.0 min), 8 agents, 393 tool calls. Four reviewers,
one dimension each (spec fidelity of flip_targets.py; check validity of smoke_kd_flip.py; the W pod smoke;
registry, integration and rules), each followed by an adversarial verifier told to refute every finding and to
default to refuted. Every agent had the GO, the plan, the audit and amendment A1 (inlined verbatim), the hard limits,
GREP-1 and A1.2's rule, and was read-only.

**Snapshot** (STEP 6, MEASURED). Before: HEAD `1e72f032c362af3ae86bfcb48756e7ea2b6abb8b`, an empty scoped status, and
the
four lane files' sha256 `1c229f0a…`, `f6198343…`, `fffeb059…`, `6cc6e39b…`. After: identical (a `diff` of the two
snapshots printed nothing).

Nine findings, all minor: five confirmed, four refuted. Two pairs describe one defect each and were judged
differently by their two verifiers (FT-1/KW-CV-1, RIR-3/pod-1); the resolution follows the reasons, below.

| id | dimension | finding | verdict | resolution |
|---|---|---|---|---|
| FT-1 | spec | A teacher -inf at a valid cell leaves W's CWD logit-map term finite where E3's cwd_channelwise_kl gives NaN (L-KD-HARDEN item 9), so AM-7 (a) does not stop W for it; `finite` is false but abort() is never reached by that alone | CONFIRMED: "a recorded deviation or a ruling, not an arithmetic change" (line 130 is plan (vi)'s literal xlogy form, which K3 binds) | C3: the flip_targets.py docstring states it (NaN and +inf still give a NaN loss, as in E3); (k) item 1 asks the orchestrator for a ruling |
| FT-2 | spec | The equal-view CWD bound "within 1e-6 relative" is stated unconditionally but holds only while the term is large | CONFIRMED, documentation only: float64 agrees to about 1e-12; the float32 gap is cancellation that grows as the term shrinks | C3: the flip_targets.py docstring and smoke_kd_flip's clause-map line state the bound as the fixture's; the check is unchanged |
| KW-CV-1 | smoke | FT-1's effect, plus "no smoke check uses a non-finite input" | REFUTED: the code follows the binding spec, no binding clause requires the stop, and cm_abort_teacher_finite_reads_targets does inject a NaN; "at most an open item for a ruling" | covered by FT-1's resolution |
| KW-CV-2 | smoke | rf_pod_smoke_test_path's "nothing made" does not look at the test-named parent folder in the `--teacher-ckpt` case | REFUTED: cmd_run refuses before any filesystem call and creates nothing at or above the teacher path; the check tests the named property directly and fails on a missing refusal | none |
| pod-1 | pod | K6's `draws_per_step` counts teacher forwards per step, not M4-KD draws | REFUTED: unreachable with this code (the real adapter takes one draw per forward; build() raises at step 1 when call 2 does not) | fixed with RIR-3 |
| pod-2 | pod | The resolved-path half of the SL-1 refusal is covered by no check | REFUTED: the code is correct and no binding text asks for that case; a coverage suggestion. Its verifier adds, unadjudicated, that removing the as-typed loop would fail no check either | none |
| RIR-1 | scope | The `--selfcheck` work folder was not refused inside the checkout, and its SL-1 check ran after `tempfile` had probed (written and unlinked a file in) TMPDIR | CONFIRMED, with two corrections to its evidence (the evidence JSON is never written; the .pth files are gitignored) | C3: teacher_diag's `refuse_test_path` and `require_outside_repo`, imported (plan 1.0); TMPDIR, TEMP and TMP refused before the probe |
| RIR-2 | scope | A scope slip: bytecode of both C1 files was written into the checkout (`src/distill/__pycache__/flip_targets.cpython-311.pyc`, `scripts/__pycache__/smoke_kd_flip.cpython-311.pyc`, 13:21:00 UTC) | CONFIRMED: the pattern of a `py_compile` (which writes even under `-B`); both files gitignored; the flip_targets one valid, the other stale; neither changes a result | my `python -B -m py_compile` syntax check at 13:20:59 ((j) S6). No code change; the files stay (the GO forbids deletes; (k) item 7); syntax checks since are `ast.parse` in memory |
| RIR-3 | scope | pod-1's defect, with a reachable record: call 1 taking two draws passes build() (which checks call 2 only), completes 25 steps and reads `[2]*25` while the stream advanced 75 | CONFIRMED | C3: `draws_per_step` sums each step's `draws_delta`; the selfcheck's three-draws case adds that record. The verdict logic was already right |

Refuted findings (STEP 7's list): KW-CV-1, KW-CV-2, pod-1, pod-2. pod-1's defect was fixed under RIR-3 because RIR-3's
record is reachable where pod-1's is not; KW-CV-1's effect is FT-1's and is resolved with it.

**C3** `0b0cc7e53659ee7e616f6060158f2e830c1dc5cd` "L-KD-WRAP C3: review fixes": `src/distill/flip_targets.py`
(docstring only, +14 −3),
`scripts/smoke_kd_flip.py` (docstring only, +3 −2), `scripts/smoke_kd_flip_pod.py` (+41 −9). The counts are unchanged:
smoke_kd_flip 32, the pod selfcheck 17.

## (f) AM-18 item 3(b) and the binding audit findings, mapped to code and checks

AM-18 item 3(b) (docs/PREREGISTRATION_AMENDMENTS.md:1055-1067) and item 3's header (:1042-1046), clause by clause.
"fl" names `src/distill/flip_targets.py`, "td" `src/training/train_distill.py` at ceafd12 (unchanged), "kf" the
checks of `scripts/smoke_kd_flip.py`, "pod" those of `scripts/smoke_kd_flip_pod.py`.

| Clause | Code | Checks |
|---|---|---|
| Item 3 header: "Each KD run's run_meta records its target construction (K and the number of views)" | `configs/kd_arms/e3_flip.json` (K 1, views 2); td's seam writes `target_construction` last in run_meta and in the payload | kf cm_run_meta_flip, cm_flip_registry_entry; pod run_meta row |
| Item 3 header: targets built "as the runs of record do (one NMF draw per step ...)", the arm differing from E3 "only in what its definition names" | fl `FlipTargets.K = 1` (one draw per view); td's E3 path otherwise; `e3_flip.json` stage e3, seed 42, horizon 80000, teacher "record" | kf cm_k_not_1_refused, cm_unflipped_call_is_e3_call, cm_run_meta_flip (the row is E3's but for the four arm fields) |
| 3(b): "E3-flip = E3's recipe of record in which, at every step:" | `e3_flip.json` (name e3-flip, run_stage E3-flip); td calls `build()` once per training step | kf cm_flip_registry_entry, cm_two_calls_per_step (every step) |
| "the teacher runs on the crop and then on its horizontal flip" | call 1 is td's own call (td:1153); call 2 is `teacher(flip_input(model_input))` in `FlipTargets.build()`, `flip_input` = `torch.flip(x, dims=[-1])` | kf cm_two_calls_per_step, cm_second_call_on_hflip, cm_unflipped_call_is_e3_call |
| "and the second set of logits is flipped back" | fl `flip_back(out2.logits)` = `torch.flip(..., dims=[-1])`, onto the crop's grid | kf cm_flip_back_alignment (+ mut_no_flip_back) |
| "(for this arm only, a departure from contract B2's single teacher input)" | on record at docs/IMPLEMENTATION_CONTRACT.md:233 ("Arm W adds a forward on the flipped crop"); no contract note added (GO ruling K8, audit F10) | none needed (a record, not code) |
| "the Logit-KD and CWD logit-map targets are the mean of the two views' distributions" | fl `FlipViewTargets.logit_kd_loss` (p = average of the two class softmaxes) and `cwd_logit_loss` (q = average of the two spatial softmaxes per channel); td passes `targets` to `distillation_losses` | kf cm_logit_kd_prob_average_T4, cm_cwd_logit_prob_average_T4_per_channel |
| "probabilities are averaged, each term in its own softmax at T = 4, never logits" | fl `class_probs` (over the classes) and `spatial_probs` (over the cells of each channel), each at the T td passes (T_logit = T_cwd = 4); `average` takes probabilities only | kf cm_per_term_softmax, cm_never_logits, cm_equal_views_match_single_view (its 1e-6 bound on the CWD term is its fixture's, review finding FT-2) (+ mut_shared_softmax, mut_average_logits, mut_t_1) |
| "the CWD feature-map term uses the unflipped view's Stage 3" | the TargetBuilder interface leaves the feature-map term to td (call 1's `teacher_out.feat_s16`); `build()` discards call 2's Stage 3 and never touches call 1's | kf cm_feature_term_unflipped_stage3 (+ mut_feature_from_flipped_view) |
| "the M4-KD stream advances once per head forward" | each call is one head forward and one draw; `build()` refuses a call 2 that does not take exactly one draw, or that changes the CPU RNG state | kf cm_nmf_two_draws_per_step, cm_cpu_rng_unchanged_by_call2 (+ mut_one_batched_call); pod nmf_draws (K7) |
| "the student sees only the original crop" | td's one student forward on `model_input`; `build()` never receives the student; the validity mask is the crop's | kf cm_student_one_forward_original_crop, cm_validity_mask_original_orientation, cm_step1_checks_pass (+ mut_student_sees_flip, mut_flipped_mask) |
| "E7-flip = PTQ of E3-flip, as in (a)" | `e3_flip.json` e7_stage "E7-flip"; the PTQ arm wrapper is L-TEACHER-ARMS's, generic over the registry | kf cm_flip_registry_entry |
| "The flipped view carries a second NMF draw, so E3-flip - E3 reads as flip plus a second draw" | step t takes draw 2t-1 (crop) and 2t (flip) where E3 takes draw t (audit F28) | kf cm_nmf_two_draws_per_step (positions), cm_unflipped_call_is_e3_call (step 1) |
| "Under item 6(a)'s correction each view's distribution is its own K = 8 average ..." | does not apply: DL-84 makes the correction unavailable, DL-96 adds no K8 arm; a builder with K != 1 refuses itself, and kd_arm/1 accepts K 1 only | kf cm_k_not_1_refused |
| A1.1: the AM-7 abort record's teacher_finite reads the targets | fl `FlipViewTargets.finite`; td:1095-1097 (`targets is None or bool(targets.finite)`) | kf cm_abort_teacher_finite_reads_targets (+ mut_finite_always_true), at function level: every `abort()` call in `run()` is under `mode == "real"` or the real-only AM-7 (b) monitor (td:1201, 1220, 1241, 1247, 1273), and main() refuses a real run on the CPU (td:1506-1512); (j) D4 |

Not a 3(b) clause, but where W and E3 part (review finding FT-1, (k) item 1): a NaN or +inf teacher logit gives a
NaN loss in both, as L-KD-HARDEN item 9 intends, while a lone teacher -inf at a valid cell gives E3's CWD
logit-map term NaN and W's a finite value (plan (vi)'s xlogy form; fl's docstring, "non-finite").

Binding audit findings:

| Finding | What it requires | Where it is met |
|---|---|---|
| F10 | cite IC:233 for contract B2's departure; no new contract note | this report, the 3(b) row above; no docs file but this report changes (GO ruling K8) |
| F28 | W matches 3(b) clause by clause; draws crop 2t-1, flip 2t | the table above; kf cm_nmf_two_draws_per_step checks the positions at every step |
| F29 | call 2 after the step-1 probe's hooks are removed | td registers the probe's hooks at td:1129-1131 and calls `build()` at td:1159, after their try/finally (td:1132-1156, unchanged); kf cm_step1_checks_pass sees the probe's hook on call 1 and none on call 2, and mut_call2_inside_probe fails `teacher_same_augmented_input` |
| F30 | the one-A40 fit is INFERRED; the W pod smoke on real weights is allowed (item 6(a)'s hold ended at DL-84) and is E3-flip's precondition | `scripts/smoke_kd_flip_pod.py run` measures memory against K7's thresholds; its JSON records `am18_item6a_hold` (DL-84, 2026-10-06T16:01:05Z, before the run); (h) and (i) below |
| F34 | launched/not_launched/stopped lines and the clock offset for arm launches | the launch gate pa (L-TEACHER-ARMS) serves every registry entry generically; this lane supplies the entry pa reads (run_stage E3-flip, cut_place 2, teacher record, go_rule false); kf cm_flip_registry_entry |
| F40 | E3-flip's AM-19 3(b) arithmetic takes the larger of E3's and the W pod smoke's s/iter | the W pod smoke's JSON gives W's figure in smoke_kd_step's own shape (K6, (g) below), so pa reads both with one reader; pod selfcheck s_stub_child_record_and_driver_json_end_to_end checks that shape end to end |

## (g) The K6 schema, verbatim

GO ruling K6, verbatim:

> K6. The pod-smoke output that pa reads (F40). smoke_kd_flip_pod.py `run` writes one JSON with "format":
> "kd_flip_pod_smoke/1" and a top-level "throughput" map in smoke_kd_step's own shape:
> throughput["e3-flip"]["iter_seconds"] is the statistics dict of smoke_kd_step's throughput() (median, mean, n, …)
> over steps 6–25.
>    - pa then reads E3's figure (smoke_kd_step's kd_smoke JSON, throughput["e3"]["iter_seconds"]["median"]) and W's
>      figure (throughput["e3-flip"]["iter_seconds"]["median"]) with one reader, and takes the larger.
>    - The JSON also records max_memory_reserved, teacher_forward_ms for each call, the draws per step, the teacher
>      identity, HEAD and the evidence folder.
>    - Write this schema verbatim into the lane report, so that the parallel lane's reader matches it.

As implemented (`scripts/smoke_kd_flip_pod.py`, `cmd_run`; REPOSITORY-PROVEN). One file,
`<evidence>/kd_flip_pod_smoke_<UTC>.json`, written once with `open(..., "x")` and `json.dumps(doc, indent=1)`; its
top-level keys in this order:

| key | value |
|---|---|
| `format` | `"kd_flip_pod_smoke/1"` |
| `arm` | `"e3-flip"` (the registry name) |
| `run_stage` | `"E3-flip"` |
| `head` | `git rev-parse HEAD` of the checkout (40 hex) |
| `expect_head` | `--expect-head` (must equal `head`, or the run is refused before any child) |
| `evidence` | the resolved `--evidence` folder |
| `run_folder` | `<evidence>/kd_flip_pod_smoke_<UTC>` (the child's record `stage_flip.json` and log `stage_flip.log`) |
| `started_utc` | `YYYY-MM-DDTHH:MM:SSZ`, when the child started |
| `am18_item6a_hold` | `{"ended_at": "DL-84, 2026-10-06T16:01:05Z", "ended_before_this_run": <bool>}` (audit F30) |
| `environment` | `{"image_digest", "kd_image_digest", "head", "expect_head", "governed_dirty", "plantseg_git_commit", "driver", "data_root"}` (smoke_kd_step's environment row) |
| `teacher` | the teacher identity record of smoke_kd_step's `load_teacher` (ckpt path, file name, sha256, bytes, provenance, parameter counts, eval mode, state entries) |
| `e3_reference` | `null`, or `{"path", "sha256", "format": "kd_smoke/1", "verdict", "head", "iter_seconds_median"}` read from the kd_smoke JSON's `throughput["e3"]["iter_seconds"]["median"]` |
| `stage` | the child's whole record (train rows, per-forward NMF records, memory, run_meta, hard checks, warnings, timings) |
| `warnings` | the child's warnings plus those in its log |
| `throughput` | `{"e3-flip": smoke_kd_step.throughput(stage)}`, i.e. `{"iter_seconds": {"median", "mean", "max", "n"}, "next_seconds": {…}, "teacher_forward_ms": {…}, "teacher_share": <float or null>, "hook_seconds": {…}}`, each statistics dict over steps 6–25 |
| `max_memory_reserved` | `torch.cuda.max_memory_reserved()` after the run, in bytes (int) |
| `teacher_forward_ms` | `{"call1": {"ms": [25 floats], "stats_steps_6_25": {"median", "mean", "max", "n"}}, "call2": {…}}`, CUDA-event times of each call |
| `draws_per_step` | `[2, 2, …]`: the M4-KD draws each step 1–25 took, the sum of its forwards' `draws_delta` (`null` for a step whose draws are unknown; C3, review finding RIR-3) |
| `torch` | the child's `torch.__version__` |
| `gpu_name` | `torch.cuda.get_device_name(0)` |
| `arithmetic_input` | `{"e3_flip_s_per_iter_median", "e3_s_per_iter_median", "larger"}` (AM-19 3(b)'s input, F40) |
| `rows` | one `{"check", "status", "detail"}` per criterion: children, environment, teacher_identity, step_1, steps_25, nmf_draws, peak_memory, throughput, run_meta, warnings, losses |
| `verdict` | `"PASS"`, `"FLAG"` or `"STOP"` |

The reader pa needs: `doc["format"] == "kd_flip_pod_smoke/1"` and
`doc["throughput"]["e3-flip"]["iter_seconds"]["median"]`,
the same path as E3's `doc["throughput"]["e3"]["iter_seconds"]["median"]` in a `kd_smoke/1` JSON. The selfcheck case
`s_stub_child_record_and_driver_json_end_to_end` writes this JSON from a stub child and reads it back by that path.
Exit codes as smoke_kd_step: 0 PASS, 1 FLAG or STOP, 2 refused.

## (h) Ice's W pod-smoke block (written, never run)

One A40 48 GB (Secure Cloud), the KD image of record
`ghcr.io/ainsleydeluna/plantseg-thesis@sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b`
(DL-87), a sparse clone at P_K that never fetches docs/reference/reference.pdf (DL-19's pattern), the TRAIN/VAL-only
root (M11) at `/workspace/plantseg_data/plantseg`, the teacher of record `iter_24000.pth` (DL-88, sha256
`8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e`), and the λ wave's kd_smoke JSON (the E3 pod smoke,
`scripts/smoke_kd_step.py run`) copied to the pod. It runs after P_K exists and before E3-flip launches (K7, F30).
Item 6(a)'s hold ended at DL-84 (2026-10-06T16:01:05Z), before this run on real teacher weights; the JSON records it.
The block runs in the KD runbook's shell (KD.1) like the E3 pod smoke; td's `set_seed` exports
`CUBLAS_WORKSPACE_CONFIG` itself (td:857).

```
set -u
cd /workspace/kdw_clone
[ "$(git rev-parse HEAD)" = "<P_K>" ] || { echo "STOP: HEAD is not P_K"; exit 1; }
echo "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e  /workspace/teacher/iter_24000.pth" | sha256sum -c - \
  || { echo "STOP: the teacher file is not the teacher of record"; exit 1; }
mkdir -p /workspace/kdw_evidence
rc=0
env -u PLANTSEG_GIT_COMMIT PYTHONPATH="$PWD" \
  PLANTSEG_IMAGE_DIGEST=sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b \
  PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg \
  python -B scripts/smoke_kd_flip_pod.py run --expect-head <P_K> \
    --teacher-ckpt /workspace/teacher/iter_24000.pth \
    --teacher-ckpt-sha256 8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e \
    --evidence /workspace/kdw_evidence \
    --e3-reference <path of the lambda wave's kd_smoke JSON> \
  > /workspace/kdw_evidence/smoke_kd_flip_pod_console.log 2>&1 || rc=$?
echo "rc=$rc"; tail -n 20 /workspace/kdw_evidence/smoke_kd_flip_pod_console.log
sha256sum /workspace/kdw_evidence/kd_flip_pod_smoke_*.json
```

- rc 0 is PASS; rc 1 is FLAG or STOP (the JSON is written either way; its `rows` name the criterion); rc 2 is a
  refusal before any child (nothing is written: correct the input and rerun).
- No path argument may contain "test" (SL-1; the smoke refuses it as typed and as it resolves). The evidence folder
  must be outside the clone. An output is never overwritten.
- Afterwards the JSON and its sha256 go to the orchestrator for the W pod smoke row and the AM-19 3(b) arithmetic
  ((i) rows 4 and 5).

## (i) The DL rows the orchestrator will write

The lane writes none of these (SCOPE). Ids are placeholders.

The plan's §6 proposes ids from DL-102 (DL-102 the lane split, DL-106 the arms pin P_K); the orchestrator assigns
them. Values in angle brackets are filled when known.

1. **DL-<id>, L-KD-WRAP pushed** (this push): "lane/kd-wrap = `<tip>`, based at
   `ceafd1296ce373e8959a41af12586e2e7ebbfb0a`
   (GO ruling K1; one deviation of form, no merge commit); commits C1 `d5726083…`, C2 `1e72f032…`, C3 `0b0cc7e5…`, C4
   `<sha>`; the five declared paths only; acceptance (MEASURED, CPU): smoke_kd_flip 32/32, smoke_kd_flip_pod
   `--selfcheck` 17/17 (the plan's 16 plus one declared case), smoke_arms_invariance 16/16 with e3_flip.json present,
   `--cross-commit 8487551 WORKTREE` 14/14, every STEP 3 smoke at its count; DL-24 review: 9 findings: 5 confirmed
   (FT-2, RIR-1 and RIR-3 fixed in C3; FT-1 stated in C3 and open for a ruling; RIR-2 recorded), 4 refuted; report
   docs/lane_reports/kd-wrap.md."
2. **DL-<id>, registry entry E3-flip** (part of the arms pin row, plan's DL-106): "configs/kd_arms/e3_flip.json, sha256
   `f6198343b73a00b320ca84be605e41be25ccc53e2ca9b49a0c8828194a57a404`, git blob
   `e7f29e8ee670efdf30a4208881fa7134559208b8`: kd_arm/1, e3-flip,
   E3-flip, E7-flip, stage e3, seed 42, horizon 80000, teacher record, go_rule false, K 1, views 2, targets
   src.distill.flip_targets FlipTargets, cut place 2 (GO ruling K4); src/distill/flip_targets.py, sha256
   `83344833a14df3dec267331d772f92c115bd2c7e151c08d28f621b39db042556`, git blob
   `a08814dcde95d0cc792182e60fbcfe00b9e360bd`."
3. **DL-<id>, arms pin P_K** (AM-18 item 7(a); the plan's DL-106), before E3 s42's α selection exists: "P_K =
   `<merge commit of lane/kd-wrap>`; the item 3 arm files and their blob ids (row 2 for W's, L-TEACHER-ARMS's for
   the rest); the token "AM-20 item 3: open"". If L-KD-WRAP has not merged by **Wed 14 Oct 2026 12:00 +08:00**, a
   W-cut row comes first: "W cut: code not merged by the cutoff; AM-18 7(b) place 2; results known: none", and P_K =
   P_T.
4. **DL-<id>, W pod smoke result** (before E3-flip launches; K7, F30): "scripts/smoke_kd_flip_pod.py run at P_K
   `<sha>`, one A40, image `sha256:cb413304…c32c2b`, the teacher of record `iter_24000.pth` sha256 `8c0e649a…`:
   verdict `<PASS|FLAG|STOP>`; max_memory_reserved `<B>` (PASS ≤ 40e9, FLAG ≤ 44e9); E3-flip median s/iter
   `<w>` over steps 6–25; teacher_forward_ms medians call 1 `<ms>`, call 2 `<ms>`; draws per step 2 at every step;
   E3 reference `<kd_smoke JSON path>` sha256 `<…>`, E3 median `<e>`; JSON `<path>` sha256 `<…>`. Item 6(a)'s hold
   ended at DL-84 (2026-10-06T16:01:05Z), before this run on real teacher weights." A FLAG names its ruling; a STOP
   holds E3-flip.
5. **DL-<id>, AM-19 3(b) arithmetic for E3-flip** (F40): "the larger of E3's measured median s/iter (`<e>`, the λ
   wave's kd_smoke JSON) and W's (`<w>`, row 4) is `<max>` s/iter; E3-flip's 80,000 iterations take `<max × 80000 /
   3600>` h of training time at that rate; `<the schedule consequence under AM-19 items 1(e), 2 and 3(b)>`."
6. **DL-66 status** (the plan's status text): "L-TEACHER-ARMS IMPLEMENTED `<sha>` (merged `<P_T>`); L-KD-WRAP
   IMPLEMENTED `<tip>` (merged `<P_K>`), W only (DL-<lane split id>)".
7. **Later rows** (the plan's §6 templates): E3-flip launched, complete or stopped (launched / not_launched / stopped
   lines with the clock offset, F34, written by pa's launch path); E7-flip (PTQ of E3-flip) likewise.

## (j) Slips, deviations, hook refusals and SL-1 lines

**Deviations**

- **D1 (K1), base.** lane/kd-wrap starts at `ceafd12` (L-TEACHER-ARMS C1), not at 8487551 plus a merge commit as
  ARMS-AUDIT-1 Q7 and the plan's §7.1 C2 word it. The L-PROT hook refuses `git merge` and `git commit-tree` in
  sessions (the orchestrator's MEASURED probe); a branch from C1 carries the same tree. No `git merge` or
  `git commit-tree` ran. The deviation is one of form only.
- **D2, smoke_kd_flip's count.** 32 checks (A1): the plan's 19 clause-map checks less cm_flip_gate_record_teacher_only
  (K5 as the orchestrator read it; it moves to smoke_arms_kd with rf_flip_gate_without_approval and
  rf_flip_with_arm_teacher), plus cm_abort_teacher_finite_reads_targets; the 2 pod-smoke refusals; 1 determinism
  check; the plan's 9 mutations plus "finite always true".
- **D3, the selfcheck's declared case (K5).** smoke_kd_flip_pod `--selfcheck` has 17 cases: the plan's 16 and
  `s_stub_child_record_and_driver_json_end_to_end`. Reason: the 16 test the evaluator on synthetic records and the
  refusals; none runs the `_stage_flip` child or the driver's JSON assembly, which the pod run would otherwise run
  for the first time. The added case runs the child on the CPU (the harness's stub teacher with its M4-KD stream,
  3 steps at batch 2, synthetic batches), then the driver end to end with the HEAD, environment and teacher-file
  checks stubbed; it reads the written JSON back by K6's path, checks its verdict, and checks that a second run into
  the same output is refused (existing_output). No plan case is removed.
- **D4, A1.1 at function level.** No non-real mode reaches abort(): its five call sites (td:1201, 1220, 1241, 1247,
  1273) are guarded by `mode == "real"` (td:1199, 1218, 1239, 1272) or sit in the AM-7 (b) branch, whose monitor
  exists only when `mode == "real"` (td:1071–1072). A real run is outside the lane: main() refuses one on the CPU
  (td:1506–1512) and needs the M11 TRAIN/VAL-only root (td:1501), and the lane runs no pod. So
  cm_abort_teacher_finite_reads_targets takes the record's teacher_finite expression from td's own source (ast,
  td:1095–1097), evaluates it with targets whose call 2 is non-finite (false) and with finite targets (true), checks
  that FlipTargets' `finite` is false for a non-finite call 2 with a finite call 1, and checks the guard of every
  abort() call; mut_finite_always_true is killed by it.
- **D5, the venv.** Python 3.11.17 (the GO's `/usr/bin/python3.11`), where scripts/preflight_distill.py's
  FINGERPRINT_OF_RECORD names 3.11.16 for the pod image; torch 2.1.0+cpu where the image has 2.1.0+cu121. Cloud
  smokes only; the pod run uses the KD image of record.
- **D6, the lint gate.** The repository has no linter configuration. Each lane file was checked with the
  preinstalled `ruff check --no-cache --select F,E9` (pyflakes and syntax; `--no-cache` writes nothing into the
  repository), a 120-character line limit (the repository's style) and `ast.parse`; all clean at C1, C2 and C3.
  A default `ruff check --no-cache` (no `--select`) at C2 printed 26 style findings (import order, unused `noqa`,
  `subprocess.run` without `check`, and the like) of the kinds the repository's existing scripts carry; they are not
  a gate.

**Hook and classifier refusals** (none retried in the refused form)

- **R1, R2 (STEP 1, 12:25 UTC).** The step 5 chain was refused by the auto-mode classifier ("[Irreversible Local
  Destruction]"), and so was my read-only state check after it (`git rev-parse HEAD`, `git symbolic-ref HEAD`,
  `git branch --list 'lane/kd-wrap'`, the scoped status). I stopped and reported. Ice approved, once and in the
  conversation, a re-run of the chain exactly as written; it ran at 12:30 with exit 0, and every post-check printed
  what the GO expects ((b)).
- **R3 (STEP 1, slip).** My first post-check 11 added a `cd` prefix and an `; echo` suffix to
  `git status --porcelain=v1 -- docs/reference/reference.pdf`. L-PROT refused it: "[L-PROT N] the command names
  docs/reference/reference.pdf; only the exact status probe may name it". I then ran the GO's exact command, which
  printed nothing.
- **R4 (STEP 5, slip).** After C2, `git show --stat --format='%H%n%s' HEAD …` without `--`. L-PROT refused it:
  "[L-PROT G] git show without `--` shows a whole-tree diff". I used path-scoped `git log` and `git diff --stat`
  naming the three C2 paths instead.
- **R5 (STEP 6, slip).** While checking the reviewers' greps, a loop running `ls "$d"` over four src folders. L-PROT
  refused it: "[L-PROT S] `ls $d` lists docs/reference/reference.pdf." (the guard cannot resolve a variable). I used
  literal `ls src/data | grep -ci test` and the like instead: no folder the reviewers grepped holds a test-named file.
- **R6 (STEP 6, slip).** A Bash heredoc writing this section's draft into /tmp/kwscratch quoted R3's message and so
  named the protected PDF's path. L-PROT refused it: "[L-PROT N] the command names docs/reference/reference.pdf; only
  the exact status probe may name it." I wrote the draft with the editor tool instead.
- **R7 (STEP 6, slip).** A progress check of the review workflow's own folder (under /root/.claude) ran
  `ls -la "$(dirname "$f")"`. L-PROT refused it: "[L-PROT S] `ls $(dirname $f)` lists docs/reference/reference.pdf."
  I repeated the check with the literal folder path.
- **R8 (STEP 7, slip).** My audit of the review agents' tool calls ran as inline Python whose regex held git verbs.
  L-PROT refused it: "[L-PROT S] inline interpreter code that runs git is not analysed; run git from the shell." The
  audit ran instead as a script file in /tmp/kwscratch with no git words in it; it runs no git.
- **The review's agents** met three L-PROT refusals, all in the scope reviewer, each rephrased into another form:
  "[L-PROT S] inline interpreter code that runs git is not analysed; run git from the shell.", "[L-PROT G] git
  hash-object works on the whole tree or object store and is not allowed by L-PROT." and "[L-PROT S] `ls
  $d/__pycache__` lists docs/reference/reference.pdf.". My audit of their 393 calls found no read of a test-named path,
  of anything under docs/reference/ or of anything under /tmp/kwscratch, and no write. Three Grep calls (the scope
  reviewer's) named one non-test file each (`scripts/smoke_kd_flip_pod.py`, `scripts/smoke_kd_flip.py`,
  `scripts/preflight_distill.py`) without GREP-1's glob; the scope reviewer also read `.claude/settings.json` and two
  hook scripts, which this GO does not forbid (it forbids changing `.claude/`).

**Other slips**

- **S1.** `ls docs/lane_reports/` once (STEP 6), to see that this report's path was free: a listing of a docs
  subfolder where the rule reads docs files by exact path only. Names only; no content was read and no name contains
  "test". `test -e` on the exact path would have done.
- **S2.** Pod selfcheck development run 1 failed 16/17: the one-draw mutation case built its synthetic records while
  `DRAWS_PER_STEP` was patched. Fixed (records built before patching) before any acceptance run.
- **S3.** Per-run time estimates were not stated for four STEP 3 runs (the default invariance smoke,
  smoke_eval_stage_artifacts, smoke_loader_parity, smoke_distill_schedule) or for the runs inside the C1 and C2
  suites (one total each). Every final run (STEP 7) has its own.
- **S4.** Draft-stage fixes, all before the commit that holds the file: a placeholder line in the first draft of
  `c_cwd_t4` (the file was rewritten before any run); two unused imports and long lines in the C2 drafts; a docstring
  that overstated what 8487551's keywords cover; the driver's warnings list aliasing the child record's list.
- **S5.** The session's worker process restarted once, after the C2 suite's last run had finished; the suite's
  results were intact in its log, and nothing was rerun or lost.
- **S6 (review finding RIR-2), a scope slip.** At 13:20:59 UTC, during C1's development, I checked the two C1 files'
  syntax with `/tmp/kwvenv/bin/python -B -m py_compile src/distill/flip_targets.py scripts/smoke_kd_flip.py`.
  py_compile writes bytecode whatever `-B` says, so it wrote `src/distill/__pycache__/flip_targets.cpython-311.pyc`
  and `scripts/__pycache__/smoke_kd_flip.cpython-311.pyc` into the checkout: outside the five declared paths and the
  scratch folder. Both are gitignored (`.gitignore:1`), so no status or name check shows them, and no gate reads them.
  The flip_targets one matched its source until C3's docstring edit (Python then recompiles in memory, writing
  nothing under `-B`); the smoke_kd_flip one was stale from the start and is never loaded (the smoke runs as
  `__main__`). Neither changed a result. They stay: the GO forbids deleting ((k) item 7). Every syntax check since
  is `ast.parse` in memory.

**SL-1 lines.** No SL-1 line appeared in any tool output. I did not open the SL-1 log (.claude/ is out of scope).
The smokes' own SL-1 refusals ran only on path strings that are never created (`rf_pod_smoke_test_path`;
`s_a_path_sl1_refuses_is_refused`).

**A1.2, the extraction (reported once).** The unchanged scripts/invariance_harness.py's `--cross-commit` export
extracts 8487551's test-named code files into the run's work folder, and smoke_arms_invariance's check 1 child does
the same into `<work>/child/export_old`. That happened in these runs' work folders under /tmp/kwscratch:
`k1_invariance_2060_1` (STEP 3), `k1_invariance_6951_1` (C1), `k1_invariance_12370_1` (C2), `k1_invariance_9854_1`
(final runs), and
`arms_smoke_2530_1` (STEP 3), `arms_smoke_7500_1` (C1), `arms_smoke_12793_1` (C2), `arms_smoke_10137_1` (final runs).
I took the names from
each smoke's own summary line in my logs; nothing under them was opened, listed, grepped or hashed, and nothing was
deleted. The review's agents were told the same rule.

**Harness instructions (reported once, in STEP 0).** The session's standing instruction to develop and push on
`claude/sharp-hypatia-airrh2` is not approval (AGENTS.md; the GO names lane/kd-wrap, and DL-37's lane rule), so
nothing was pushed there. The session's commit-attribution boilerplate asks for `Co-Authored-By`/`Claude-Session`
trailers; the GO says "No trailers", so no commit carries one.

## (k) Open items

1. **A ruling on FT-1** (review finding, confirmed). A teacher -inf at a valid cell gives W's CWD logit-map term a
   finite value where E3's cwd_channelwise_kl gives NaN, so AM-7 (a) stops E3 on that input and not W; NaN and +inf
   stop both. W's arithmetic is plan (vi)'s literal `xlogy(q̄, q̄) − q̄·s_logp` (GO ruling K3), so the lane changed
   only the docstring. The orchestrator decides whether W should propagate a non-finite teacher value as E3 does
   (for example a NaN loss whenever `targets.finite` is false), which would be a change to the arithmetic of record
   and needs a ruling before E3-flip launches. Exposure is narrow: a lone -inf with no +inf or NaN anywhere (INFERRED
   by the verifier: the teacher of record's conv_seg reads ReLU features, so an overflow normally also gives +inf or
   NaN).
2. **The W pod smoke** ((h)) is Ice's later block, after P_K exists and before E3-flip launches (K7, F30). The lane
   ran no pod: W's memory fit and s/iter on one A40 remain INFERRED (plan (viii): about E3's peak plus < 1 GB; below
   2 × E3's s/iter) until it runs. Its `--e3-reference` is the λ wave's kd_smoke JSON; its path on the pod is Ice's to
   fill.
3. **GO-2 after P_T** (K9): after L-TEACHER-ARMS merges into master, a short GO-2 checks the merged tree before Ice's
   merge block, including the parallel lane's smoke_arms_kd, which holds the three checks moved out of this lane (K5:
   cm_flip_gate_record_teacher_only, rf_flip_gate_without_approval, rf_flip_with_arm_teacher). This lane ran
   smoke_arms_invariance on its own tree (A1.3), not smoke_arms_kd, which is not on lane/teacher-arms at `ceafd12`.
4. **pa's reader** must read W's figure as (g) states: `kd_flip_pod_smoke/1`,
   `throughput["e3-flip"]["iter_seconds"]["median"]`, beside E3's `throughput["e3"]["iter_seconds"]["median"]` from the
   kd_smoke JSON, and take the larger (F40). The selfcheck's declared case reads the written JSON by that path.
5. **The W cutoff** is Wed 14 Oct 2026 12:00 +08:00 (K9, AM-18 item 7(a)): if lane/kd-wrap has not merged by then, W
   is cut and the W-cut row precedes P_K ((i) row 3).
6. **Counts against the GO's text**: smoke_kd_flip is 32 (A1; the GO said 30, the plan 33) and smoke_kd_flip_pod
   `--selfcheck` is 17 (the GO and the plan say 16; one declared case, (j) D3). GO-2's acceptance lines should use 32
   and 17.
7. **Files this session left in its container**, none in the commits: the gitignored bytecode of (j) S6
   (`src/distill/__pycache__/flip_targets.cpython-311.pyc`, `scripts/__pycache__/smoke_kd_flip.cpython-311.pyc`), and
   A1.2's test-named extractions in the cross-commit and arms-invariance work folders under /tmp/kwscratch, unopened.
   The GO forbids deleting, so they stay; the container is reclaimed with the session, and a human may remove them
   before then.
8. **A1.1 at run level** would need a run that reaches abort(), which only a real-mode run does ((j) D4). If the
   orchestrator wants it, a real-mode case (on a pod, or a ruling for a real-mode stub run) is new scope.
9. **Review leftovers, not adjudicated**: pod-2's verifier noted that removing the as-typed SL-1 loop in
   smoke_kd_flip_pod.py's `run` would fail no check (the resolved-path loop prints the same refusal), and plan 1.0
   asks for teacher_diag's `refuse_test_path` "imported, not copied", where `run` copies its two halves inline
   (equivalent today; the selfcheck now imports it). Neither was a finding; both are candidates for a later
   hardening commit if the orchestrator wants one.
