# Decision log — methodology layer ⇄ Claude Code
Source of truth for decisions made in the THESIS2 methodology chat. Read at session start. Changed
only through B-SYNC packets, or by the task that implements an entry (Status/Where columns only).

## Rules
1 Statuses: DECIDED → RECORDED (formal file named in Where) → IMPLEMENTED (commit SHA) →
  VERIFIED (evidence). DROPPED and SUPERSEDED rows are kept.
2 DECIDED and RECORDED entries bind planning. Until an entry is IMPLEMENTED, the governed file it
  names still describes the code. Never launch a run whose recipe conflicts with such an entry — STOP
  and report.
3 Every task reads this file first, reports conflicts, and updates Status/Where for the entries it
  implements, in the same commit.
4 Ch4's deviation table is generated from entries with Ch4 = yes, using their RECORDED dates and SHAs.
5 Formal pre-registration amendments live in docs/PREREGISTRATION_AMENDMENTS.md, dated before any
  affected result exists; docs/PREREGISTRATION.md stays frozen.
6 The known failures listed below are the only test failures a Phase B may carry; every other check
  must pass.
7 Later B-SYNC packets list only the entry IDs they add, change or drop.

## Current plan (as of aa3823f)
- Teacher (critical path): runtime frozen at 3c43f89 (a51a092) → pod session B65, teacher image only
  (DL-19) → web audit → typed GO → official teacher run → R3 on VAL.
- B66: DL-17 and N10 (E1 gradient probe), then E1 seeds 43/44; seed fix live (2dc1fde); AM-7 no-clip
  branch.
- Code lanes the amendments require: before the λ sweep, L-AM7; before E4–E6, L-AM1q, L-AM4 and
  L-AM10; before TEST, L-AM3, L-AM5 (with the KF-1 re-pin), L-AM6, L-AM8, L-AM13 and L-EVAL-DET (DL-17);
  L-PROT (DL-26) done (4da20cd).
- Quantization: DL-18 runs locally before E4 (B65 Phase B); DL-17 in B66; then E4 and E5 on E1 seed 42;
  per seed later.
- KD: after teacher acceptance → λ sweep (DL-06) → E3 ×3 (+ E2 43/44 if budget) → E6/E7 per seed.
- TEST: one unlock, after every stage is final, the harness is frozen and every lane above is
  IMPLEMENTED.
- Manuscript: DL-20.

## Entries
| ID | Status | Decision | Where | Ch4 |
|---|---|---|---|---|
| DL-01 | IMPLEMENTED 0d86f83 | Working branch = claude/keen-curie-u4a8ig; master is fast-forwarded to it at every authorized push. | AGENTS.md:10 | no |
| DL-02 | IMPLEMENTED 2dc1fde | Every student-training DataLoader generator seeds from --seed, not the constant SEED; seed 42 is unchanged, proven by scripts/smoke_loader_seed.py (13/13, golden captured before the fix). QAT seeding is lane L-AM1q. | src/data/dataset.py; src/training/train_e1.py; src/training/train_distill.py | yes |
| DL-03 | IMPLEMENTED 2dc1fde | Per-iteration total gradient-norm telemetry in the E1 and KD trainers, measured before any clipping; gradients unchanged. | same files as DL-02 | no |
| DL-04 | RECORDED 8084671 (AM-7) | One clipping rule for E1–E3. E1 seed 42 logged no gradient norm, so the no-clipping branch applies: E1, E2 and E3 run unclipped. Divergence = (a) a non-finite loss or gradient norm, or (b) after the ramp, the 100-iteration mean total loss above 5× its running minimum. A divergence in E2/E3 stops the stage; one rule is then adopted for all FP32 stages and they are rerun. The E2/E3 clip pilot (U3) is withdrawn. | AMENDMENTS AM-7; code lane L-AM7 (launcher gate and in-trainer abort for (b)) | yes |
| DL-05 | RECORDED 8084671 (AM-1) | Seeds 42 (primary), 43 and 44 for E1 and E3; E4/E7 recomputed per seed; E5/E6 once per seed, taking the FP32 parent's seed; E2 at 43/44 optional, budget permitting, with λ fixed from the seed-42 sweep and in addition to the five-point sweep; teacher once. The Holm family and the E3-vs-E6 non-inferiority check use seed-42 models; seeds 43/44 enter only DL-13. The teacher's NMF inference stream stays seeded at 42 in every KD run (M4-KD). | AMENDMENTS AM-1; QAT seeding lane L-AM1q | yes |
| DL-06 | RECORDED 8084671 (AM-2) | λ_logit sweep: grid {0.25, 0.5, 1, 2, 4} at seed 42, 80,000 iterations each; a candidate's value is its best-checkpoint VAL all-class mIoU (strict >, earliest tie); candidates within 0.5 pp of the best are tied → smallest λ; a boundary winner is reported and the grid is not extended; winner = E2 seed 42; λ reused unchanged in E3 and E2 43/44. | AMENDMENTS AM-2; configs/distill.py | yes |
| DL-07 | RECORDED 8084671 (AM-3) | E6-KD trigger: E3 → E6 drop in dataset-level VAL all-class mIoU (E6 = converted INT8) > 1.0 pp at seed 42. If triggered: seed 42 only; Logit-KD and CWD weights 0.5× their E3 values; T unchanged; descriptive, outside Holm. The clean-TEST trigger is withdrawn. | AMENDMENTS AM-3; STATS §9.3; code lane L-AM3 | yes |
| DL-08 | RECORDED 8084671 (AM-13) | Teacher acceptance = R3 on VAL (strictly > 0.36314016580581665). The 42.05% comparison happens only at final evaluation, descriptively, with the teacher also scored under the upstream protocol (aspect-ratio-preserving resize, scored at original resolution). | AMENDMENTS AM-13; code lane L-AM13 | yes |
| DL-09 | RECORDED 8084671 (AM-4) | QAT: 15 fixed epochs, no early stopping; BN stats frozen after epoch 10, observers after epoch 12; a checkpoint every epoch, each converted (QNNPACK) and scored on VAL on CPU after training; evaluated = best converted VAL all-class mIoU (ties → earlier); physical batch 16. Supplementary: the selected model is also scored with fake-quant disabled. The U4 QAT clip pilot stays; after L-AM4 it compares candidates on converted-model VAL mIoU. | AMENDMENTS AM-4; code lane L-AM4 | yes |
| DL-10 | RECORDED 8084671 (AM-10) | PTQ calibration: fixed registered qconfig; 128 TRAIN images, seed 42, one image per mini-batch; one list shared by E4 and E7; no VAL-based selection. | AMENDMENTS AM-10; code lane L-AM10 | yes |
| DL-11 | RECORDED 8084671 (AM-14); code VERIFIED in B64 | Inference family = the eight tests at ch3 f.139: one-tailed Wilcoxon (Pratt), paired t-test sensitivity, Holm step-down. E3 vs E6 only via the paired-BCa non-inferiority check (margin 2.0 pp; sensitivity 1.0–2.5 pp); E1 vs E3 descriptive. src/stats already builds exactly this family. | AMENDMENTS AM-14 (resolves conflicts.md #9); STATS §1 | yes |
| DL-12 | RECORDED 8084671 (AM-5, AM-6) | TEST scoring: images with zero disease pixels are excluded from per-image disease-only analyses, including per-image mIoU-C (count reported; kept in dataset-level metrics); dataset-level mIoU is union-present, plus a descriptive ground-truth-present sensitivity; per-image mIoU averages the disease classes present in ground truth; per-image mIoU-C = mean over the 15 corrupted variants. | AMENDMENTS AM-5/AM-6; EVAL §3; code lanes L-AM5, L-AM6 | yes |
| DL-13 | RECORDED 8084671 (AM-8) | Seed stability, reported at the single TEST evaluation: stable iff the seed-paired E1 → E3 gain is positive at all three seeds, its mean exceeds the larger across-seed SD of E1 and E3, and the E3 → E6 drop is < 2.0 pp at every seed; plus a per-seed effect table for every planned comparison. | AMENDMENTS AM-8; STATS §9.4; code lane L-AM8 | yes |
| DL-14 | RECORDED 8084671 (AM-11) and 74ee587 (AM-15) | Not invoked or not run: the DIST fallback and the "E2 as deliverable" switch (AM-15: E6 is always built from E3, and E3 is reported whatever its result against E2); the extended-schedule E2 and the α_CWD sweep (AM-11). | AMENDMENTS AM-11, AM-15 | yes |
| DL-15 | DROPPED | Not adopted: two-sided tests; ARM-native latency (the approved x86 fbgemm-copy path stays; AM-9 withdrawn); λ-grid shrink; seed-averaged Wilcoxon; the 'robustly supported' replication rule across all eight tests; the per-image clean-minus-corrupted descriptive comparison; Ch1 motivation rewrite; H₁d rewording. | AMENDMENTS AM-9 (withdrawn slot) | no |
| DL-16 | DECIDED | Reporting: the E1 vs E6 mIoU-C test is described as accuracy on corrupted images; robustness claims come only from RPD/rCD; Ch5 discusses Wei's 10.22% against E1's baseline. The code ID robustness_e1_e6 is kept for artifact compatibility. | Ch4/Ch5 drafting; STATS note (74ee587) | no |
| DL-17 | DECIDED | Before E4–E7, in the B66 pod (pinned student image, GPU) and before the seed-43/44 launches: the local E1 seed-42 checkpoint (sha256 cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03) is re-scored on VAL twice at batch 16. Pass: the two runs are bitwise identical, and the absolute difference from 0.36314016580581665 is at most 1e-4; a difference between 1e-6 and 1e-4 is recorded, not a failure. Eval-path determinism flags: lane L-EVAL-DET before TEST. | B66 | no |
| DL-18 | DECIDED | Before E4, run locally in the pinned student image (Docker, CPU; QNNPACK is a CPU engine). A new probe using the registered PTQ qconfig asserts every converted conv weight is per-channel and records, through a runtime hook, whether the LR-ASPP Sigmoid runs quantized or through a float fallback. The probe fails (non-zero exit) only on a conversion error, a conv weight that isn't per-channel, a quantized engine other than qnnpack, or a run that doesn't complete. A float fallback is not a failure; it resolves U6. If the Sigmoid runs quantized, its output scale and zero_point are recorded as U6's values; if it falls back, the fallback is recorded and reported in the E4–E7 quantization coverage and Ch4. The full-student QNNPACK smoke's exit-code bug is fixed in the same commit. | B65 Phase B (local) | no |
| DL-19 | DECIDED (freeze done: a51a092) | Teacher GO chain: runtime frozen at 3c43f89 → one pod session B65, teacher image only. Partial clone with sparse checkout that never fetches the protected file (step 0 verifies HEAD 3c43f89, file absent, status clean, nine freeze hashes). Re-canary = shortened production config at 3c43f89 with every production hook (harness rewritten for the 3c43f89 API, validated offline in the local teacher image). Batch-16 teacher VRAM (G21); throughput incl. a worker-count check; TRAIN/VAL-only preflight (verify_env.py and preflight_e1.py never run there); PYTHONPATH override as in B62; A40 48 GB on Secure Cloud unless evidence says otherwise → web audit → typed GO. | B65 | no |
| DL-20 | DECIDED | Manuscript: Ch1–3 stay the approved plan; only the errata in THESIS2_manuscript_fixes_v2.1 are applied (101 rows plus references). Methodology changes are documented in Ch4 from entries with Ch4 = yes. | web layer | no |
| DL-21 | DECIDED | Official runs: image pinned by digest (E1 seeds: sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf); pods use the DL-19 partial clone; no-resume rule (IMPLEMENTATION_CONTRACT, D29 closed in 0d86f83); checkpoint sha256 captured on the pod before download (B52 N9); B66 uses a TRAIN/VAL-only preflight path. | runbooks (B65, B66); IMPLEMENTATION_CONTRACT | no |
| DL-22 | RECORDED 8084671 (AM-12) | The augmentation library is the NumPy/PIL implementation; described as such in Ch4; code unchanged. | AMENDMENTS AM-12 | yes |
| DL-23 | DECIDED | Workflow: capsule-first; tiers T0/T1/T2; reports carry MEASURED / REPO-PROVEN / INFERRED / NOT DETERMINABLE labels; official runs, TEST unlock and pushes each need a separate typed GO. | this file | no |
| DL-24 | DECIDED | Model and effort: Opus 5.5 everywhere. Claude Code: high by default; medium for mechanical doc and sync edits; xhigh for pod/GPU sessions and official-run launches; max and ultracode are not used; workflows and subagent fan-outs only when a prompt asks for one. Web chat: High by default; Max for deep audits. | prompt headers; CLAUDE.md, ai_guardrails.md, fix_template.md (74ee587) | no |
| DL-25 | RECORDED 0d86f83 (D33) | KF-1: smoke_stats_bootstrap's contract-digest check pins the 9267ffe contract (stale since B59, 5ae4ec7). Known failure, not a regression: a Phase B treats that single check as KF-1, and every other check must pass. Re-pin once, in the commit that aligns src/stats with the amended contract (L-AM3 + L-AM5). | open_questions D33 | no |
| DL-26 | IMPLEMENTED 4da20cd | Protected file docs/reference/reference.pdf. Incidents on record: three B63 calls and one B64 call ran git grep over the docs pathspec, which reads the committed blob (no content was output). Enforced by the committed .claude/settings.json: Read/Edit deny rules and a fail-closed PreToolUse hook (rules N, G, C and P, plus R2b for non-git tools and Grep/Glob), with a data-free test of 130 cases. Commit messages never name the file (use -F). merge, rebase, pull, switch, cherry-pick, worktree add and stash pop/apply are denied to Claude Code; a human runs them outside it. Accepted gap: git_porcelain_bytes, verify_env.py and write_run_manifest.py keep whole-repo git status, so one worktree-provenance definition holds for the whole campaign (the teacher runtime is frozen at 3c43f89 with it); the call records the file's status but never outputs its content. | .claude/settings.json; .claude/hooks/ | no |

## Known failures
- KF-1 — smoke_stats_bootstrap, check "repository-baseline contract digest matches the checkpoint" (DL-25).

## Amendment map
All recorded in 8084671 (2026-09-23): AM-1 → DL-05; AM-2 → DL-06; AM-3 → DL-07; AM-4 → DL-09;
AM-5/AM-6 → DL-12; AM-7 → DL-04; AM-8 → DL-13; AM-9 → withdrawn (DL-15); AM-10 → DL-10;
AM-11 → DL-14; AM-12 → DL-22; AM-13 → DL-08; AM-14 → DL-11.
AM-15 → DL-14 (74ee587).

## History
- 2026-09-23 · CP-001 to CP-003 (consolidated initial load; CP-001 was never applied separately) · DL-01 to DL-26
- 2026-09-23 · CP-004 · DL-13, DL-14, DL-16, DL-21, DL-24, DL-26; current plan
- 2026-09-23 · CP-005 · DL-17, DL-18, DL-19, DL-21; current plan; amendment map
