# B62 — Unified teacher runtime reconciliation (M2, M3, M4, M5, M11, M12, M13, G10, G11, R3)

**Date:** 2026-09-22. **Type:** implementation record (governed runtime change). **Repository authority:**
branch `claude/keen-curie-u4a8ig`, HEAD = origin = `5ecdd843ab340e212ffa47c17732fb88cf1a68c9` (B61).

**Status:** the locked teacher methodology is **implemented in runtime code** and the final bytes passed
CPU validation on the host and in the pinned teacher image (§11). **At the final pre-commit gate, the 28 B62
paths were staged and awaiting commit, and no push had occurred.** The commit-based runtime freeze, the
mandatory CUDA re-canary, G2, the official TRAIN/VAL preflight and an explicit teacher-training GO remained
pending (§9). **OFFICIAL TEACHER = NO-GO.**

**Scope of this phase:**
- Runtime, config, smoke and status-documentation changes only.
- No training and no GPU; no RunPod; no downloads or installs.
- No TEST content is opened, read, listed, counted or hashed. The M11 helper performs only fail-closed
  existence checks on the three approved TEST surface paths. During one early development smoke those
  existence checks ran against the real data root; that smoke now uses a synthetic TRAIN/VAL root
  (§7.1 disclosure).
- No change to M1 or M6–M10.

**Evidence labels:** MEASURED · UPSTREAM-SOURCE · REPO-SOURCE · INFERRED · THESIS-DERIVED.

---

## 1. What was implemented, lock by lock

| Lock | Runtime realisation | Proven by |
|---|---|---|
| **M2** train augmentation | `ThesisTeacherTrainTransform` calls the E1 `src/data/transforms.train_preprocess` with `configs/augment.AUGMENT` on a per-sample `RandomState` drawn from the worker's numpy stream. The MMSeg pipeline is exactly `[ThesisTeacherTrainTransform, PackSegInputs]`: no `PhotoMetricDistortion`, no short-side `RandomResize`, no Albumentations | `smoke_teacher_pipeline` (bitwise equality with the E1 recipe, 5 shapes × 4 seeds; rotation fills and order; cat-max-ratio; crop padding; independent flips; joint hue/saturation; excluded operations absent) |
| **M3** VAL canvas | `ThesisTeacherEvalTransform` calls the E1 `core_preprocess`. This fixes the Resize-before-LoadAnnotations defect: the target is the thesis 512×512 canvas, with the long side scaled to 512, symmetric padding (image mean / label 255) and nearest-neighbour masks. There is no evaluation-time padding in the data preprocessor | `smoke_teacher_pipeline` (the MMSeg target equals the thesis canvas for 5 shapes; normalised input equals `finalize`; prediction and ground truth are both 512²) |
| **M4-T** | `IsolatedNMFLightHamHead` / `IsolatedNMF2D`. `rand_init=True` is kept; with no stream attached, the upstream algorithm draws from the global stream exactly as stock. Attaching a stream in training mode raises `NMFStreamError` | `smoke_teacher_nmf` (`m4t_*`) |
| **M4-V** | `TeacherNMFEvalStreamHook(seed=42)`, priority VERY_HIGH. It requires VAL `batch_size=1` and a sequential sampler; attaches a fresh private CPU stream (`torch.Generator().manual_seed(42)` state) at `before_val_epoch`; draws only inside `_build_bases`; restores the caller CPU RNG at `after_val_epoch`; and asserts one draw per image. The seed, the VAL manifest SHA-256 and the image count go to the message hub. No `torch.manual_seed`, no CUDA RNG change | `smoke_teacher_nmf` (first pass = repeat pass bitwise; caller RNG restored across the persistent DataLoader's base-seed draw; seed-43 and no-hook negative controls; order change detected; batch 2 fails closed) |
| **M4-KD** | `FrozenTeacher.begin_nmf_stream("M4-KD", 42)`, called once per E2/E3 run in `train_distill.run()`. It is caller-transparent and advances without per-batch reset. A real run requires the stream to attach | `smoke_teacher_nmf` (`m4kd_*`), `smoke_teacher` (`test_m4_nmf_stream`) |
| **M4-V, repository evaluator** | `load_teacher_model(..., config_path=...)` begins an M4-V stream; `evaluate_model.py --model-role teacher` requires `--teacher-config` and `--batch-size 1`. Student paths are unchanged | `smoke_eval_teacher` (5 new checks) |
| **M5** cadence | `val_interval=4000`; `CheckpointHook(interval=4000)`; 10 validations, 4k…40k | `smoke_teacher_config`, `smoke_teacher_selection` (`m12_official_cadence_4k_to_40k`) |
| **M11** isolation | `test_dataloader`, `test_evaluator` and `test_cfg` are all `None`, and the test/TTA pipelines are removed. The stdlib helper `src/data/isolation.py` runs `os.path.lexists` on exactly `images/test`, `annotations/test` and `annotation_test.json`, then counts TRAIN and VAL by name (5,367 / 846). It is applied in the launcher and at the start of the E2/E3 real-run branch | `smoke_teacher_launch` (each surface fails closed; instrumented `scandir`/`listdir`/`open` prove no TEST content is touched); `smoke_distill` (`gate_m11_*`, on a synthetic root) |
| **M12** selection | `ThesisConfusionMIoUMetric` reuses `src.eval.metrics` (int64 confusion, union-present, E1's float32 reduction) and returns `mIoU_full` (selection), `mIoU_disease_full`, `mIoU` (2-decimal display) and `val_images`. `CheckpointHook(save_best='mIoU_full', rule='greater', max_keep_ckpts=-1)`. `TeacherSelectionRecordHook` appends one JSONL record per validation, cross-checks CheckpointHook's `best_score` / `best_ckpt` (fails closed on disagreement), writes `teacher_selection.json` atomically, and at `after_train` asserts the full cadence plus the selected file's existence and SHA-256. Every record carries its pass's ordered VAL manifest hash (`val_manifest_sha256`); all must be non-empty and identical, and `teacher_selection.json` carries it (§11) | `smoke_teacher_selection` (real Runner runs A–E) |
| **M13** CE normalisation | `CrossEntropyLoss(avg_non_ignore=True)`, `ignore_index=255`, no `class_weight` | `smoke_teacher_selection` (`m13_*`: valid-pixel mean, zero loss and gradient on ignored pixels, finite all-ignore batch) |

### 1.1 Checkpoint provenance, G10, G11

- **Launcher** (`scripts/launch_teacher_finetune.py`). The init checkpoint must:
  - come from the explicit `SEGNEXT_ADE20K_CKPT` path;
  - exist and sit outside the repository;
  - match SHA-256 `647a0cda…40ef1` (`init_ckpt_sha_mismatch` otherwise).

  Provenance records the checkpoint identity (config name, filename and SHA-256), the stock config's
  identity (resolved by `find_spec`, without importing MMSeg), the locked policy, the M11 result and the
  governed-path state. `--launch` additionally runs the approved governed-clean gate on the rule-8 paths
  only; `docs/reference/reference.pdf` cannot fail it.
- **Runtime compatibility audit.** `TeacherInitCompatibilityHook` runs at `after_load_checkpoint`. Only
  `decode_head.conv_seg.{weight,bias}` may mismatch (150 → 116), and on resume nothing may. Anything else
  raises. It is proven on a stock-shaped checkpoint (852 matched) and on a violating one.
- **G10 closed.** `configs/teacher_finetune.py` and the contract's B1 row carry the MMSeg 1.x identity, the
  exact filename and the SHA-256; no 0.x alias remains in executable metadata.
- **G11 closed.** `scripts/test_teacher_init.py`:
  - every third-party import sits inside the guard, so the script exits 2 with guidance even without
    numpy;
  - the checkpoint comes from arguments or `SEGNEXT_ADE20K_CKPT`, never a `weights/` fallback;
  - in-repo checkpoints and SHA mismatches are refused;
  - `--expect-sha256 ANY` exists for synthetic fixtures;
  - the stock 150-class test logic is unchanged.

### 1.2 R3 script

`scripts/teacher_readiness_r3.py` enforces the order of R3:
1. It verifies that the checkpoint is the M12-selected one (`teacher_selection.json` status `final`, and
   the SHA-256 equal to the file's).
2. Before any evaluation, it resolves the ordered VAL manifest the evaluator is about to process and
   requires its hash to equal the selection's `val_manifest_sha256` (`val_manifest_missing` /
   `val_manifest_mismatch` otherwise; §11).
3. It runs the repository evaluator once (teacher role, VAL, batch 1, M4-V) with
   `artifact_status = "provisional"`, which the frozen evaluator schema admits.
4. It refuses a summary that scored another checkpoint, and one whose `dataset.split_manifest_sha256` is
   absent, empty or different (`artifact_manifest_missing` / `artifact_manifest_mismatch`).
5. It applies the strict `>` against E1's `0.36314016580581665`, with no margin.
6. It writes `teacher_r3_readiness.json`, role `same_val_operational_floor`, including the manifest
   identities, and refuses to overwrite it.

Exit codes are 0 PASS, 2 refused, 3 FAIL. A FAIL prints STOP (no retraining, no relaxation, no TEST).

### 1.3 Smoke version-provenance correction

The B60 launch smoke expected fixed MMSeg/mmcv versions in provenance, which cannot hold on a host without
the stack. That produced the recorded 54/55 "expected/not applicable" in the image. Version checks are now
**environment-correct**: each recorded version must equal what the running environment reports
(`importlib.metadata`), or `None` where the package is absent. Torch must be real.

---

## 2. MMEngine / MMSeg facts that fixed design choices (MEASURED unless labelled)

- `IterBasedTrainLoop` saves the interval checkpoint in `after_train_iter`, **before** validation at the
  same iteration. The selection hook can therefore hash `iter_{N}.pth` in `after_val_epoch`.
- MMEngine passes CheckpointHook the **un-namespaced** metric key. With `save_best='mIoU_full'` the best
  file is `best_mIoU_full_iter_{N}.pth` (observed), and `best_score` is the full-precision value.
- `ValLoop` calls `before_val_epoch` **before** creating the DataLoader iterator. A DataLoader without a
  generator draws its base seed from the default CPU generator at that point. The M4-V hook therefore
  saves the caller state first and restores it after the pass, and the restore is proven across that
  draw.
- `BaseDecodeHead.predict` calls `self.forward` directly, so forward hooks on `decode_head` never fire
  (the NMF smoke hooks `decode_head.conv_seg`).
- `BaseSegDataset` sorts by `img_path` and serialises `data_list`, so order changes are tested through the
  `indices` argument on a second runner.
- A validation-only Runner has no `file_backend` for `CheckpointHook(save_best=...)`. The NMF smoke sets
  `save_best=None` there; the selection smoke uses real training runs.

---

## 3. Files

**New (7):**
- `src/training/teacher_components.py`
- `src/distill/nmf_stream.py`
- `src/data/isolation.py`
- `scripts/teacher_readiness_r3.py`
- `scripts/smoke_teacher_pipeline.py`
- `scripts/smoke_teacher_nmf.py`
- `scripts/smoke_teacher_selection.py`

**Modified runtime and config (10):**
- the teacher config;
- `configs/teacher_finetune.py`;
- `scripts/launch_teacher_finetune.py`;
- `src/distill/{segnext_teacher,teacher,__init__}.py`;
- `src/training/train_distill.py`;
- `src/eval/model_loading.py`;
- `scripts/evaluate_model.py`;
- `scripts/test_teacher_init.py`.

**Modified smokes (5):** `smoke_teacher_config`, `smoke_teacher_launch`, `smoke_teacher`,
`smoke_eval_teacher`, `smoke_distill`.

**Status docs (5):**
- `docs/IMPLEMENTATION_CONTRACT.md`
- `docs/EVALUATION_CONTRACT.md`
- `docs/open_questions.md`
- `docs/teacher_prep_runbook.md`
- `docs/teacher_init_source.md`

Plus this report.

### 3.1 Runtime SHA-256, before → after (working tree)

| File | Before | After |
|---|---|---|
| `configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py` | `510b212b…` | `1b94aa32a7d22be647d8e60a42daa1dd8dd756158b586feec6465385e299ae4a` |
| `configs/teacher_finetune.py` | `ccae2eaf…` | `0293ffd9959f429f79c7898a46d921df8c2a2e50df4612997c6fe4c9f9712ced` |
| `scripts/launch_teacher_finetune.py` | `58575276…` | `47537c016bde8a0119c625a04b31fbf4227924ad7cb201062bffbad69e8419a1` |
| `src/training/teacher_components.py` | new | `1b46680c6d03982c84387cc3c3ea78eb14de7ec643ac3b55b1937ef7cbf63025` |
| `src/distill/nmf_stream.py` | new | `053e53cc441d58a2477affbd281331d7321dd2023e9f09c9f75010b9b36a64d4` |
| `src/data/isolation.py` | new | `74a063313cc04c1e327dce22b0faa4ae3a9334a01e6cf7c58eb1bee13c2e5014` |
| `src/distill/segnext_teacher.py` | `34d0a271…` | `55a4a41a874499860484ae27c7949cf3909910161430f85d9787ad15a37a87a5` |
| `src/distill/teacher.py` | `431deeb0…` | `3f30120a3605852675909df0d75811df5b3131f258b529bbc961af26b756f9a3` |
| `src/distill/__init__.py` | `8238a5ae…` | `26f0afbd33a906866ddc34a4c9680df0b6db6a7473907c945c5cb85c1a2fcdfe` |
| `src/training/train_distill.py` | `34cfc64c…` | `4be98bcd69c5594554fd63fe46867183aac10f4fce12d63f77f441814e0a3727` |
| `src/eval/model_loading.py` | `b13bbce8…` | `4bef56f6a592d24ab351e9a7a955e916a5364aec41e19b8a4e497a6b6c0cdd63` |
| `scripts/evaluate_model.py` | `343dfca9…` | `c4a46d7bbb4fe4359b953788c7d32427089ac6e1dcc5f69af31286931dfc2e2a` |
| `scripts/teacher_readiness_r3.py` | new | `a045989c60f5b3061c76f5f43a9d7b6f67b1e5a530c76cd08215ab6e3a2c7240` |
| `scripts/test_teacher_init.py` | `73cb3caf…` | `9bf6fd0c8c8f368352c90130b7967a0d079aa773200447378e3ef02ef1f42684` |

These are the **final-byte** working-tree values: the bytes validated in §11, where all 28 B62 paths were
hashed before and after the revalidation and were identical. `scripts/smoke_teacher_selection.py`, the
smoke that carries the §11 manifest negatives, is
`adcb41a5b8775bd5add1355b95a87e63177d16d66eb1a263654cbec3df9c6430`. The binding freeze is still taken from
the **commit** at §9 step 2 and supersedes the runbook §4a step-8 table.

### 3.2 Unchanged by design (byte-identical, MEASURED)

- `src/training/teacher_runner.py` (G18 seam): `72206af7e00939bd…`.
- E1:
  - `src/data/transforms.py` `df920496…`
  - `src/eval/metrics.py` `9898d6dc…`
  - `src/training/train_e1.py` `bae45a18…`
  - `src/data/dataset.py` `9f469441…`
  - `configs/augment.py` `8c450f30…`
  - `configs/data.py` `a670f91a…`
- Also untouched: requirements files, `PREREGISTRATION.md`, historical reports (B61 and earlier), chapter
  PDFs, `smoke_teacher_runner.py` and `smoke_environment.py`.
- No MMSeg package file was edited, and there is no global monkeypatch.

---

## 4. Validation (CPU only)

### 4.1 Host (anaconda Python, no MM stack)

**Final bytes** (§11; both logs postdate the last runtime edit, and no B62 file has changed since):

| Smoke | Result |
|---|---|
| `smoke_distill` | PASS 94/94 (M11 gates on a synthetic TRAIN/VAL root) |
| `smoke_teacher_launch` | PASS 79/79 (real external checkpoint accepted and SHA-verified via `SMOKE_ADE20K_CKPT`) |

**Initial B62 run** (before the §11 repair; not re-run on the final bytes). On the host the §11 repair
changed no executable code these smokes reach: `teacher_components.py` cannot load without MMSeg, R3 is
not imported, and the `src/data/isolation.py` change is a docstring.

| Smoke | Result |
|---|---|
| `smoke_environment` | PASS 71/71 |
| `smoke_teacher` | PASS 51/51 |
| `smoke_eval_teacher` | PASS 43/43 |
| `smoke_eval_contract` | A2a OK |
| `smoke_eval_stage_artifacts` | PASS 45/45 |
| `smoke_eval_int8` | PASS 35/35 |
| `smoke_eval_plantseg` | A2b OK (capped VAL path) |
| `smoke_cwd_projection` | PASS |
| `smoke_kd_resolution` | PASS 24/24 |
| `smoke_quant_e6_e7` | PASS 37/37 |
| `smoke_quant_runners` | PASS 77/77 |
| `smoke_realrun_decisions` | PASS 78/78 |
| `test_teacher_init` | exit 2 with guidance (the G11 guard, expected with no MMSeg) |

G11 negative checks: a numpy-shadowed interpreter gives exit 2, and missing MMSeg gives exit 2.

### 4.2 Pinned teacher image

The image is `plantseg-teacher:local` = `sha256:cb413304…c32c2b`. The runs used CPU, `--network none`,
and a read-only export of `src/`, `configs/` and `scripts/` that was byte-identical to the working tree.

**Final bytes** (§11; run one at a time in the foreground, each exit code recorded):

| Test | Result |
|---|---|
| `smoke_teacher_config` | PASS 108/108 (the checkpoint variable unset, as §7.2 requires) |
| `smoke_teacher_launch` | PASS 81/81 |
| `smoke_teacher_pipeline` | PASS 17/17 |
| `smoke_teacher_nmf` | PASS 18/18, including `m4v_caller_cpu_rng_restored_first_pass` (§11.2) |
| `smoke_teacher_selection` | PASS 54/54, including the manifest negatives (§11.1) |
| `test_teacher_init` | PASS on the real readiness checkpoint: SHA-256 verified; missing 0 / unexpected 0; `conv_seg` 150; 27,636,310 params |
| Real official 116-class checkpoint-load integration | PASS 29/29 (§11.3) |
| Compatibility negative controls | PASS 7/7 (§11.3) |

**Initial B62 run only** (not re-run on the final bytes): `smoke_teacher_runner` PASS 48/48, the unchanged
G18 regression (`teacher_runner.py` is byte-identical to HEAD).

G11 refusals in the image: no checkpoint, in-repo checkpoint, and wrong SHA-256 are all refused.

### 4.3 Hygiene

- `git diff --check` on every modified path: clean (re-run on the final bytes, §11).
- New files: no trailing whitespace or tabs; every file is LF and ends with a newline.
- `scripts/smoke_distill.py` is CRLF in the working tree. That state **predates B62**: the pre-change hash
  equals HEAD's blob with CRLF endings. `.gitattributes` (`eol=lf`) normalises it at staging, so its
  committed hash will differ from the working-tree value.

---

## 5. M12 selection behaviour, as run through a real MMEngine Runner

- **Rounded false tie (run A).** Scripted values 0.312341 and 0.312344 both display as 31.23. Selection
  takes iteration 4; a rounded comparison would have kept iteration 2 (negative control). CheckpointHook
  agrees: `best_score` is 0.312344 and the best file is `best_mIoU_full_iter_4.pth`.
- **Exact tie (run B).** 3 validations; the earliest of the tied iterations is kept. Altering the selected
  file, or deleting it, is caught.
- **Retention (run C).** With `max_keep_ckpts=1` the end-of-run assertion catches the lost selection. This
  proves why the config uses `-1`.
- **Bad init (run D).** A non-classifier shape mismatch fails closed at `after_load_checkpoint`.
- **No M4-V hook (run E).** Without `TeacherNMFEvalStreamHook` a validation has no ordered VAL manifest, so
  the selection hook refuses to write a record and no `teacher_selection.json` is produced (§11.1).

---

## 6. Performance ("best defensible result under the locked design")

No methodology-compatible improvement was left undone. Each lock is realised at full fidelity: the
full-precision selection value, every validated state retained, the E1 augmentation recipe bitwise, and
the thesis canvas for training and evaluation.

The following would plausibly change the teacher's score but each **requires a methodology decision**,
so none was applied:
- multi-draw NMF averaging (B61 P4);
- test-time augmentation;
- a longer schedule;
- class weighting in the teacher CE;
- a float64 metric reduction, which would also break exact comparability with E1's float32 reduction.

No post-hoc tuning was performed.

---

## 7. Disclosures

### 7.1 Real data-root existence checks during development (MEASURED)

In the first host run of the modified `smoke_distill`, the new M11 gate at the start of the E2/E3
real-run branch ran against the host's real `DATA["root"]`:
- **9** real-run invocations (the CPU refusal, 2 grad-clip gates, 5 invalid clip values and the valid clip
  value);
- each called `os.path.lexists` on the three exact TEST paths.

A TEST surface was present, so every invocation was refused before any counting. **Nothing under any
TEST path was opened, read, listed, counted or hashed.** The failure surfaced as a false pass: the
pre-existing gate checks returned 2 because of M11, not their own gates, and `gate_valid_clip_proceeds_past_gate`
failed (88/89).

**Fix:** the whole safety-gate section now runs on a synthetic staged TRAIN/VAL root:
- the helper's expected counts are patched to 2/2 and restored afterwards;
- every refusal is asserted on its message;
- dedicated `gate_m11_*` checks cover the three surfaces with instrumented `scandir`/`listdir`.

**Operational consequence:** the official teacher, E2 and E3 need a root staged with TRAIN and VAL only.
The gate will refuse a full-dataset root.

### 7.2 Harness error in the image battery

The battery exported `SEGNEXT_ADE20K_CKPT` for every smoke. `smoke_teacher_config` asserts that the
unset case resolves to a non-path sentinel, so it failed 107/108. The rerun with the variable unset
passed 108/108. This was a harness error, and no code changed.

### 7.3 Warnings (not suppressed; attributed)

- **MMEngine "prefix is not set in metric class …"**: from the thesis metric and its scripted test
  subclass. Stock `IoUMetric` warns the same way.
- **"Default `avg_non_ignore` is False"**: raised only where the **stock 150-class ADE20K** config is
  built: `test_teacher_init` and the selection smoke's stock-shaped init fixture. The thesis teacher loss
  sets `avg_non_ignore=True` and no longer triggers it.
- **Deprecations:** `build_loss`, `FileClient` / `HardDiskBackend`.
- **Visualiser notices:** `draw is False` and `SegLocalVisualizer` instance reuse, from the smoke
  Runners.
- **"`dataset_meta` … not saved" / "No suitable dataset found"**: from `mmseg.apis.init_model` loading the
  synthetic 116-class fixture checkpoint in `smoke_teacher_nmf`. The fixture carries no `dataset_meta`.
- **One `ResourceWarning`** from temporary-directory cleanup in `smoke_teacher_nmf`.

---

## 8. Invariants re-checked

- **E1 unchanged:** the E1 code path is byte-identical (§3.2).
- **Teacher config values held:** 116 classes, background 0, `reduce_zero_label=False`, `ignore_index=255`.
- **Teacher optimisation unchanged** (AdamW 6e-5, wd 0.01, head `lr_mult` 10; poly power 1.0 over 40k,
  warm-up 1,500), and so are BN, batch 16, seed 42 with determinism, and `load_from` (not resume).
- **Not altered:** M1 and M6–M10.
- **Checkpoint:** the checkpoint and all smoke work directories stay outside the repository.

---

## 9. Remaining blocker chain (supersedes B61 §9 steps 2–5)

1. Review this diff and stage it by explicit path (done at the final pre-commit gate: the 28 B62 paths were
   staged). Next after that pre-commit snapshot: create the separately approved B62 commit, then push it
   under a separate gate.
2. Freeze the runtime commit and the new hashes (runbook §4a step 8).
3. **G20-style CUDA re-canary: mandatory.** The launch and runner path changed (pipeline, custom imports,
   hooks, metric, head).
4. G2: batch-16 VRAM with the real teacher configuration.
5. GPU selection.
6. Official TRAIN/VAL-only preflight on a staged root.
7. Explicit GO for teacher fine-tuning.
8. After fine-tuning: R3 via `scripts/teacher_readiness_r3.py`, with STOP on FAIL.

**OFFICIAL TEACHER = NO-GO.**

---

## 10. Governance documents changed in this phase (status lines only)

- **`docs/IMPLEMENTATION_CONTRACT.md`:** the B1 base/init row (G10 identity), a B62 note after the
  non-conformance block, and the §(g) register lines.
- **`docs/EVALUATION_CONTRACT.md`:** the §7.2 / §7.3 status lines and the R3 and M11 bullets.
- **`docs/open_questions.md`:** the B62 register note, and M2–M5 and M11–M13 set to "runtime implemented
  (B62); freeze + re-canary pending". M1 and M6–M10 are byte-identical.
- **`docs/teacher_prep_runbook.md`:** G10 closed; the B62 note; the §4a step-8 table marked superseded;
  M11 / M4 / M2–M3 status; the §3 rows and open items.
- **`docs/teacher_init_source.md`:** the G10 and G11 closed notes.
- **This report.**

---

## 11. Final-byte blocker repair and revalidation

**Why.** The B62 final diff audit found one runtime blocker: R3 did not prove that its VAL evaluation used
the same ordered VAL manifest that M12 checkpoint selection scored. R3 read `split_manifest_sha256` from
`summary["run"]`, where the evaluator never writes it, so the readiness record would always have held
null. `teacher_selection.json` carried no manifest either, and the R3 smoke fixture mirrored the wrong
schema, so it passed anyway. The audit also found two over-broad "no TEST access" phrases and a count
typo. The phrases were in the `src/data/isolation.py` docstring and in this record's scope line, and both
now use the precise existence-check wording. The count is in §3.

Under M4-V an image's NMF basis depends on its **position** in the pass, so the same image set in a
different order is not the same evaluation. The repair therefore proves order, not only membership.

### 11.1 Manifest repair

- **One canonical identity.** The evaluator's own `src.eval.artifacts.hash_split_manifest` over one
  ordered `(manifest_index, image_id, clean_image_id)` row per image. The repository's backend-parity
  script already uses the same hasher as its manifest identity. The M4-V hook (`val_manifest`), the R3
  pre-evaluation gate and the evaluator artifact (`dataset.split_manifest_sha256`) all use it; nothing
  re-implements it.
- **Selection records.**
  - Every M12 record carries `val_manifest_sha256`, taken from the M4-V pass record of the same
    iteration.
  - A record whose manifest is absent, null or empty, or differs from the earlier records, is refused
    before it is written.
  - At `after_train`, all ten 4k…40k records must carry the same non-empty hash.
- **`teacher_selection.json`** stores the selected record's `val_manifest_sha256`.
- **R3 before inference.**
  - R3 resolves the ordered VAL manifest the evaluator is about to process, using the evaluator's own
    functions (`build_expected_manifest_for` over every VAL row).
  - It hashes that manifest with the same hasher and requires it to equal the selection's hash.
  - Codes: `val_manifest_missing` or `val_manifest_mismatch`; nothing is evaluated on refusal.
- **R3 after inference.**
  - R3 reads the evaluator artifact's hash from `summary["dataset"]["split_manifest_sha256"]`, the
    frozen schema location.
  - An absent, null or empty value is refused (`artifact_manifest_missing`).
  - Because both sides use the same hasher over the same rows, equality with the selection hash is
    required (`artifact_manifest_mismatch`).
  - The readiness record stores the selection, pre-evaluation and artifact hashes.
- **`smoke_teacher_selection` 54/54 on the final bytes.**
  - Every record is non-empty and identical, and `teacher_selection.json` carries the hash.
  - The MMSeg VAL pass order hashes identically to the thesis evaluator's own listing.
  - A ten-record check covers one different hash, and an absent, null or empty hash.
  - The selection hook's own path refuses a changed manifest, a stale M4-V record and an empty
    manifest.
  - A real run without the M4-V hook writes no selection (run E).
  - R3 checks:

| R3 case | Outcome |
|---|---|
| Matching ordered manifest | PASS, recorded |
| Same members, changed order (injected) | REFUSED before evaluation (`val_manifest_mismatch`, 0 evaluator calls) |
| Same members, changed order (a real MMSeg pass in reversed order) | REFUSED before evaluation |
| One member replaced (injected) | REFUSED before evaluation |
| One member replaced (a real renamed VAL file) | REFUSED before evaluation |
| Selection manifest absent, null or empty | REFUSED before evaluation (`val_manifest_missing`) |
| Artifact manifest absent, null, or only under `run` | REFUSED after evaluation (`artifact_manifest_missing`), no readiness record |
| Artifact manifest different | REFUSED after evaluation (`artifact_manifest_mismatch`), no readiness record |

### 11.2 M4-V lazy-loop defect found and fixed during the repair

**Defect.** The first repair version made the M4-V hook read `runner.iter` in `after_val_epoch`, after it
had restored the caller's CPU RNG. In a validation-only runner (`runner.val()`), MMEngine 0.10.7's
`Runner.iter` builds the train loop lazily (`runner.py` lines 540, 591). Building it creates the train
DataLoader iterator, which draws a base seed from the global CPU generator (`dataloader.py` line 601).
That draw fell outside the M4-V save/restore window. `smoke_teacher_nmf` caught it:
`m4v_caller_cpu_rng_restored_first_pass` FAIL, 17/18. It was reproducible and occurred on the first pass
only. An instrumented diagnostic located the draw in `build_train_loop`.

**Fix.** The hook reads the iteration only from an already-built `runner._train_loop`
(`isinstance(_train_loop, BaseLoop)`); otherwise the iteration is recorded as `None`. Nothing triggers
lazy loop construction. The selection hook's single remaining `runner.iter` read sits behind the same
built-loop guard.

**Final-byte proof** (`smoke_teacher_nmf` 18/18):

| Check | Result |
|---|---|
| First-pass caller CPU RNG restored | PASS |
| Repeat-pass caller CPU RNG restored | PASS |
| First and repeated M4-V passes bit-identical | PASS |
| No `torch.manual_seed`, no CUDA seeding | PASS |
| M4-T and M4-KD checks unchanged | PASS |

The diagnostic re-run on the final bytes also reports the first pass restored. It records 4 draws, all
either inside the window or on the private stream, and none from building the train loop (5 before the
fix). The M4 code contains no CUDA RNG call.

### 11.3 Real checkpoint integration (final bytes)

**Setup.** Pinned image, CPU, `--network none`. The real external ADE20K checkpoint was mounted read-only:
SHA-256 `647a0cda7678a35396689a4f8e9fddc33a088d8b539195d0dc97485ab8640ef1`, verified before and after, and
unchanged. The official 116-class B62 config went through the launcher's own gates and its own `launch()`,
then `TeacherRunner.from_cfg`, the attestation hook and `train()`. The run was stopped at `before_train`,
before iteration 0; no training iteration ran. A synthetic noise TRAIN/VAL root outside the repository
exists only because `Runner.train()` builds both dataloaders before it loads weights.

**Result: PASS 29/29.**
- Model: SegNeXt-B / MSCAN-B with an `IsolatedNMFLightHamHead` of 116 classes, plain BN, 27,618,868
  parameters.
- Order: `TeacherInitCompatibilityHook` ran exactly once, after the checkpoint was read and before the
  state dict was applied.
- Only `decode_head.conv_seg.weight` and `decode_head.conv_seg.bias` differ (150 → 116). The remaining 852
  of 854 tensors are shape-compatible and loaded bit-for-bit. Nothing is missing or unexpected, and
  MMEngine's own loader reports only the two classifier size mismatches.
- The 116-class classifier kept its fresh initialisation (Normal std 0.01, bias 0), not the 150-class
  values.
- A synthetic 512×512 CPU forward is finite, returns 116 classes, and draws once per forward from the
  private M4-V NMF stream through `IsolatedNMF2D`.

**Negative controls: PASS 7/7.** A tampered backbone tensor shape, or a missing backbone tensor, is
rejected by the compatibility hook before the state dict is applied, and training never starts. The
control shows that MMEngine's `strict=False` loader alone would only warn, which is why the hook is the
guard.

### 11.4 Validation discipline and bytes

- **Stopped battery discarded.** A first final-byte image battery was stopped mid-run when the session
  ended: docker lost its connection to the container, and the shell could no longer start processes. No
  test reached a result. That run was discarded.
- **Serial foreground re-run.** Every test was then run one at a time in the foreground, with its exit code
  recorded (§4).
- **Freeze.** All 28 B62 paths were hashed before the re-run and again after it, and were identical. §3.1
  lists the runtime values.
- **Non-blocking provenance observation.** The launch record's `reused_module_hashes()` does not list
  `src/eval/artifacts.py`, although the manifest identity now depends on it. The file is unchanged, and the
  commit plus the governed-clean launch gate pin it, so no runtime change was made; changing it now would
  invalidate the completed final-byte validation.

### 11.5 Status after this section

- **Done by the final pre-commit snapshot:** the B62 runtime implementation was complete locally, the final
  bytes had passed CPU validation, and the 28 B62 paths were staged.
- **Pending at that snapshot:** the separately approved B62 commit and, under its own gate, the push. After
  the commit, the hashes are frozen from it. The G20-style CUDA re-canary remained mandatory, and G2, the
  official TRAIN/VAL preflight and an explicit teacher-training GO remained pending.
- **OFFICIAL TEACHER = NO-GO.**
