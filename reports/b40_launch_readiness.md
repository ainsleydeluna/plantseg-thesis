# B40 — RunPod launch-readiness audit

**Type:** read-only audit. **Date:** 2026-09-06. **HEAD:** `c059e1b0cb6945d5f985d2f19132e33b422fed28`,
in sync with `origin/master`.

Nothing was edited, created, staged, or committed. No training, no downloads, no GPU. Every claim
below was re-derived from the current files or from raw command output; **no finding rests on
B34/B35/B36/B38's assertions**, and where a prior report and the files disagree the disagreement is
recorded as a finding (§1 #19, §5).

---

```
RUNPOD LAUNCH READINESS: 95% (19/20)
CODE PATH 7/7 · CONFIG+DATA 4/4 · GATE+ENV 5/5 · REPRODUCIBILITY 3/4

FAILING ITEMS:
  #19  Run provenance in output artifacts — the container has no .git, so _git_head()
       returns "UNKNOWN" and every run_meta row is unattributable; no image-digest
       field exists anywhere in src/ — src/training/train_e1.py:228-234,316 ·
       src/eval/artifacts.py:135-141,169-181 · .dockerignore:57

BLOCKED BY ENVIRONMENT (PASS, not repo work):
  #12  preflight_e1.py stages 2-5 — the pod supplies a CUDA GPU (sm<=9.0) and the
       mounted dataset; stage 1/5 passes here and stages 2-5 are unreachable locally
```

---

## 1. Score — 20 criteria

### CODE PATH — 7/7

#### 1. Real-run gate requires BOTH flags — **PASS**

Every branch exercised, not read. `src/training/train_e1.py:547-566`:

```
=== --real-run alone ===
REFUSING to start the real E1 run: --real-run requires --confirm-real-run.
exit=2
=== --confirm-real-run alone ===
ERROR: --confirm-real-run given without --real-run; nothing to confirm.
exit=2
=== both --dry-run and --real-run ===
ERROR: pass only one of --dry-run / --real-run.
exit=2
=== --real-run --confirm-real-run on a CPU box ===
REFUSING to start the real E1 run on CPU: CUDA is not available (torch.cuda.is_available()=False).
exit=2
```

The fourth is a second, independent gate (`:582-589`): a real run refuses CPU even with both flags.

#### 2. In-repo `--ckpt-dir` refused — **PASS**

Guard traced to `_assert_outside_repo` (`train_e1.py:84-89`): `path.resolve()`, then
`if path == REPO or REPO in path.parents: raise`. It is applied at **five** call sites, not one —
`resolve_ckpt_dir:95`, `save_checkpoint:136`, `save_last:156`, `write_best_pointer:176`,
`prune_checkpoints:190` — so no writer can bypass it. Exercised directly:

```
  REFUSED : C:\Users\admin\plantseg-thesis
  REFUSED : C:\Users\admin\plantseg-thesis\reports
  REFUSED : C:\Users\admin\plantseg-thesis\a\b
  REFUSED : .
  ALLOWED : C:\Users\admin\AppData\Local\Temp\x
  ALLOWED : C:\Users\admin
```

`.resolve()` means a symlink or `..` path cannot smuggle a write inside the repo.

#### 3. CE class weights fail-closed — **PASS**

`train_e1.py:55-64`, quoted in full:

```python
def load_ce_weights(path: Path = CLASS_WEIGHTS_JSON) -> torch.Tensor:
    with open(path, encoding="utf-8") as f:
        w = json.load(f)["weights"]
    t = torch.tensor(w, dtype=torch.float32)
    if t.numel() != NUM_CLASSES:
        raise ValueError(f"class-weights length {t.numel()} != num_classes {NUM_CLASSES}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("class-weights contain non-finite values")
    return t
```

Five raise paths (missing file, bad JSON, missing key, wrong length, non-finite). No `try`/`except`,
no default, **no fallback to `None`, ones, or a batch-derived vector**. Called unconditionally at
`:293` for both modes. `losses.compute_class_weights` (the batch-derived scaffold) has **zero call
sites** in `train_e1.py`.

#### 4. ImageNet init is the real-run default; no silent fallback — **PASS**

`train_e1.py:590-591`: `init = args.init or "imagenet"`, then
`pretrained = False if init == "none" else E1_STUDENT["init_weights"]`. `src/models/student.py:56-60`
raises `RuntimeError` with guidance if the weights cannot be obtained — **no silent random fallback**.
`--init none` is reachable but explicit, and is surfaced twice: printed at `:280-281` and recorded as
`used_pretrained` in the run_meta row (`:326`).

#### 5. Multi-scale RRC is TRAIN-only — **PASS**

`src/data/dataset.py:__getitem__` branches on `if self.split == "train":` → `train_preprocess(...)`,
`else` → `core_preprocess(...)`. `train_preprocess` (`transforms.py:166-186`) is the only RRC path;
`core_preprocess` (`:52-58`) is resize + pad + normalize with no augmentation. Proof of absence:

```
$ grep -rn "train_preprocess" src/ | grep -v "def train_preprocess"
src/data/dataset.py:29:from .transforms import core_preprocess, finalize, train_preprocess
src/data/dataset.py:92:                img_np, mask_np = train_preprocess(im, mk, rng, self.aug_params)
```

One call site, inside the `split == "train"` branch.

#### 6. Checkpoint + resume — **PASS**, with the omission enumerated

`save_last` (`:148-168`) persists: `iter`, `model_state_dict`, `optimizer_state_dict`,
`scheduler_state_dict`, `scheduler`, `best_val_miou_all_class`, `best_ckpt`, `num_classes`,
`rng_state` (python / numpy / torch / torch_cuda / **loader_generator**), `prev_lr`. Writes are atomic
(`_atomic_save:102-111` — `.tmp` in the same directory, then `os.replace`).

Resume (`:340-360`) restores model, optimizer, scheduler, `best_miou`, `best_ckpt`, all five RNG
streams, `prev_lr`, and `start_iter = ck["iter"] + 1`.

**What it omits: the position within the current epoch's sample permutation.** `cycle(train_loader)`
is re-created unconditionally at `:373`, and `DataLoader.__iter__` draws a fresh permutation, so a
resumed run begins the next permutation rather than continuing the interrupted one. The code says so
itself at `:356-360`. The criterion object is not checkpointed either, but carries no learned state
and is rebuilt deterministically from the JSON at `:293-294`.

#### 7. Loss = weighted CE + Dice, `ignore_index` in BOTH — **PASS**

`losses.py:86-90`: `CombinedCEDiceLoss.__init__` passes `ignore_index` to both terms; `forward`
returns `self.ce(...) + self.dice(...)` — equal weight.
CE (`:53`): `F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index)`.
Dice (`:65-70`): `valid = (target != self.ignore_index)`, then `probs = probs * valid` and
`onehot = onehot * valid` — ignore masked **before** intersection and union.

### CONFIG + DATA — 4/4

#### 8. `num_classes=116`, `ignore_index=255` consistent — **PASS**

Single source `configs/data.py:36,38`. Re-derived by importing and printing every value the E1 path
actually reads (table under #11): `num_classes 116`, `ignore_index 255`, `background_index 0`.
`NUM_CLASSES` is re-exported once (`src/data/__init__.py:3` ← `dataset.py:31`), so there is no second
definition to drift.

#### 9. Root resolves via `PLANTSEG_DATA_ROOT` — **PASS**, with a noted fallback

`configs/data.py:33`: `"root": os.environ.get("PLANTSEG_DATA_ROOT", DEFAULT_PLANTSEG_DATA_ROOT)`.
The image sets it: `Dockerfile:73` `ENV PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg`.

A hardcoded Windows default does exist at `configs/data.py:14`. It is a *fallback*, not the E1 path on
the pod, and it cannot cause silent misbehaviour — an unset variable on Linux yields a non-existent
path and `dataset.py:40-41` raises `FileNotFoundError` naming it (see §2 G). Scored PASS; the residual
is recorded rather than hidden.

#### 10. Split sizes asserted somewhere that runs — **PASS**

`src/data/dataset.py:66-75`, inside `PlantSegDataset.__init__`, therefore on **every** dataloader
construction including `train_e1.py:286-288`:

```python
if split not in SPLIT_SIZES:
    raise RuntimeError(... "refusing to load an unverified split")
expected = SPLIT_SIZES[split]
if len(self.pairs) != expected:
    raise RuntimeError(f"split={split} pair count MISMATCH: expected {expected}, actual ...")
```

Against `SPLIT_SIZES = {'train': 5367, 'val': 846, 'test': 1561}` (`configs/data.py:22`), plus an
import-time internal-consistency check that the three sum to 7,774 (`:25-28`). Missing masks also fail
loud rather than silently shrinking the split (`dataset.py:57-60`).

#### 11. No `NEED_TO_CONFIRM` on any value the E1 path reads — **PASS**, proven by tracing readers

Not by grepping the string. Every config subscript on the E1 import path was enumerated, then the
**values actually read** were printed:

```
config       key                    value actually read
DATA         background_index       0
DATA         ignore_index           255
DATA         num_classes            116
DATA         root                   'C:\\Users\\admin\\plantseg_data\\plantseg'
E1_STUDENT   batch_size             16
E1_STUDENT   init_weights           'torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2'
E1_STUDENT   iterations             80000
E1_STUDENT   learning_rate          0.01
E1_STUDENT   lr_power               0.9
E1_STUDENT   momentum               0.9
E1_STUDENT   val_interval           4000
E1_STUDENT   weight_decay           0.0001
MODEL        backbone_truncate      16
MODEL        dilated                True
MODEL        head_inter_channels    256
MODEL        high_tap_index         15
MODEL        low_tap_index          6

values consumed by the E1 path that equal NEED_TO_CONFIRM: NONE
```

`AUGMENT` was walked recursively — no `NEED_TO_CONFIRM` at any depth. **`configs/loss.py` is never
imported by the E1 path** (`train_e1.py` imports only `configs.data` and `configs.e1_student`;
`dataset.py` imports `configs.augment` + `configs.data`; `student.py` imports `configs.model`), so its
two placeholders are unreachable. `MODEL["pretrained"]` is a placeholder and is **not in the read
set**.

### GATE + ENVIRONMENT — 5/5

#### 12. `preflight_e1.py` NO-GOs correctly — **PASS** (stages 2–5 BLOCKED BY ENVIRONMENT)

Run locally, raw:

```
  stage                     result       secs  detail
  class_weights             PASS          5.9  116 weights verified; ratio 259.551929
  verify_env                FAIL         42.2  verdict=PARTIAL — needs PASS (CUDA GPU, sm<=9.0, dataset OK)
  smoke_dataloader          SKIPPED         -  not reached
  smoke_aug_stochasticity   SKIPPED         -  not reached
  train_e1 --dry-run        SKIPPED         -  not reached

VERDICT: NO-GO — first failing stage: verify_env.
=== EXIT: 1 ===
```

It NO-GOs, stops at the first failure, and exits non-zero — which is what the criterion asks. Stages
2–5 cannot pass on a CPU box with no dataset; the pod supplies both.

#### 13. Class-weight stage passes inside the published image, BY DIGEST — **PASS**

`docker run --rm ghcr.io/ainsleydeluna/plantseg-thesis@sha256:0572c116…cedbe6`:

```
  [PASS] T1a  len(weights) == 116
  [PASS] T1b  all finite; dtype=torch.float32
  [PASS] T1c  provenance fingerprint (sorted[57]+sorted[58])/2 == 1.000000000000  [NOT a correctness check]
  [PASS] T1d  argmin=0 argmax=42; w[42]/w[0] = 259.551929472253 == sqrt(3703512045/54975)
  [PASS] T1e  w[0] = 0.036954205368 (background, 80.82% of pixels)
  [PASS] index alignment: all 116 weights reproduce (max deviation 0.000e+00)
  class_weights             PASS          8.7  116 weights verified; ratio 259.551929
```

#### 14. `requirements-runpod.lock` hash-pinned and complete — **PASS**

```
pinned packages (== form)  : 78
packages with >=1 --hash   : 78
packages with NO hash      : none
total --hash lines         : 78
non-== (range) requirements: none
```

Enforced at `Dockerfile:53` — `pip install --require-hashes`, which refuses any unpinned requirement
or altered artifact.

#### 15. `build_and_push_image.sh` gates fire — **PASS**

Happy path, all gates evaluated:

```
  commit   : c059e1b0cb6945d5f985d2f19132e33b422fed28
  branch   : master (present on origin)
RESULT: DRY-RUN — every gate passed. Nothing was built or pushed.
exit=0
```

Failure mechanisms exercised directly (no repo writes were made to force them):

```
--- gate 3 (HEAD on origin): the exact ancestor test ---
  HEAD on origin/master  -> exit 0 (0 = yes, gate passes)
  HEAD on origin/master^ -> exit 1 (1 = no, gate would REFUSE)
```

Gate 2 runs `git status --porcelain -- <governed>`; on the clean set it returns nothing (`$DIRTY`
empty → pass), and the same command with a known-dirty path returns ` M docs/reference/reference.pdf`,
showing the non-empty branch that triggers `die`. Gate 4 is a string equality,
`[ "$LABEL" = "$GIT_COMMIT" ] || die` (`:113`). **Disclosure:** gates 2 and 4 were verified by
exercising their underlying mechanism and by code inspection, not by forcing the script to abort —
forcing them would have required writing to the repo, which this audit is forbidden to do.

#### 16. Authoritative digest recorded in `docs/` and matches the registry — **PASS**

```
docs/runpod_environment.md:82:      @sha256:0572c1166980d11ea0eef86aa7c2eb76b5ae6961ae406a77bbda9bcc66cedbe6
docs/runpod_environment.md:92:docker pull ghcr.io/ainsleydeluna/plantseg-thesis@sha256:0572c116...cedbe6

$ docker buildx imagetools inspect ghcr.io/ainsleydeluna/plantseg-thesis:official --format '{{.Manifest.Digest}}'
sha256:0572c1166980d11ea0eef86aa7c2eb76b5ae6961ae406a77bbda9bcc66cedbe6
```

Documented digest == registry digest. **The published image is labelled `28d1038` while HEAD is
`c059e1b`, but its contents are identical**: `git diff --stat 28d1038..c059e1b` over every path the
Dockerfile copies (`src configs scripts requirements* the three contract docs
reports/e1_class_weights.json Dockerfile .dockerignore`) returns **empty**. The only changes since are
`docs/runpod_environment.md`, `docs/teacher_prep_runbook.md` and `reports/b38_docker_image.md`, none of
which enters the image. The image is label-stale by two commits and **content-current**.

### REPRODUCIBILITY — 3/4

#### 17. Seeds pinned and actually set — **PASS**

`train_e1.py:267` — `set_seed(seed)` is the first statement of `run()`, before any model, dataloader,
or device work, and `run()` is the single entry point for both modes. `src/seeds.py:28-37` seeds
`random`, `numpy`, `torch`, `torch.cuda` (+`_all`), and sets cuDNN deterministic / non-benchmark and
`use_deterministic_algorithms(True, warn_only=True)`. Verified:

```
  seed42 draw 1 : (0.6394267984578837, 0.3745401188473625, 0.8822692632675171)
  seed42 draw 2 : (0.6394267984578837, 0.3745401188473625, 0.8822692632675171)
  identical     : True
  seed43 differs: True
  cudnn.deterministic = True | benchmark = False
  CUBLAS_WORKSPACE_CONFIG = :4096:8
```

*Minor, not scored:* `set_seed` sets `PYTHONHASHSEED` at `seeds.py:23`, after interpreter start, where
it has no effect on the running process. It does not affect numeric reproducibility here.

#### 18. Pre-registration committed before any result existed — **PASS**, re-derived from git history

Not cited from B37. From history alone:

```
--- commit that introduced docs/PREREGISTRATION.md ---
fcc8acc294bef56c3a80da6109d3becdeee4f91b 2026-09-05 18:51:07 +0800 B37: pre-registration ...

--- any result artifact ever added at ANY commit in history? ---
[exit 0 — empty above = no such file was ever added in any commit]

--- files matching those globs present in the tree AT the pre-registration commit ---
[exit 1 — empty = none present at that commit]
```

The second check is stronger than B37's: it searches `--all --diff-filter=A` across **every commit in
history**, not just the state at one commit. No `summary.json`, `per_image.jsonl`, `bootstrap.npz`,
`family.json`, `MANIFEST.sha256`, `*.pt`, `*.pth`, `*.npz`, `outputs/` or `results/` has ever been
added to this repository.

#### 19. Run provenance captured in output artifacts — **FAIL**

> ### This is the audit's substantive finding, and it contradicts a claim in the Dockerfile.

**Commit hash: recorded locally, `"UNKNOWN"` in the image where E1 actually runs.** `train_e1.py:316`
writes `"git_head": _git_head()` into the run_meta row. `_git_head()` (`:228-234`) shells out to
`git rev-parse HEAD` and **catches every exception, returning `"UNKNOWN"`**. `.dockerignore:57`
excludes `.git/**`. Inside the published image, by digest:

```
$ ls -a /workspace/plantseg-thesis
configs  docs  reports  requirements-e1.txt  requirements-runpod.in
requirements-runpod.lock  requirements.lock  scripts  src
.git ABSENT

_git_head() -> 'UNKNOWN'
GIT_COMMIT env -> None
```

`ARG GIT_COMMIT` (`Dockerfile:84`) is build-time only; it becomes a `LABEL`, never a runtime
environment variable. So **every `run_meta` row E1 writes on the pod will carry
`"git_head": "UNKNOWN"`** — the field exists, is populated, and is worthless.

**Image digest: no field exists anywhere.** Proof of absence:

```
$ grep -rniE "image_digest|imagedigest|repo_digest|container_image|OCI|image.revision" src/
src/vendor/imagecorruptions/LICENSE:151:  risks associated with Your exercise of permissions under this License.
```

One unrelated licence line. The `Provenance` dataclass (`src/eval/artifacts.py:169-181`) carries
`repo_commit`, `worktree_state_sha256`, `metric_impl_sha256`, `config_sha256`,
`split_manifest_sha256`, `class_map_sha256` — and **no image identifier**.
`docs/runpod_environment.md:66` instructs "Put the resulting image ID/digest in the run provenance of
every official artifact", but no code path accepts or writes one.

**And the official evaluation artifact path cannot be finalised inside the image at all:**

```
git_commit             -> ArtifactRequestError: git rev-parse HEAD failed
git_porcelain_bytes    -> ArtifactRequestError: git status failed (128): fatal: not a git repository
```

These are hard raises (`artifacts.py:135-141`, `:105-113`), not degradations. This directly
contradicts `Dockerfile:30-32`:

> `git -> LOAD-BEARING, not convenience: src/eval/artifacts.py records the commit and the
> governed-path porcelain in every official artifact, so official evaluation cannot be finalised
> without it.`

The `git` **binary** is installed; the `.git` **directory** it needs is excluded. Installing git
without the repository metadata satisfies the letter of that comment and none of its purpose.

**Impact.** E1 *training* still runs — `_git_head()` degrades rather than raising. But the run is
unattributable to a commit or an image, which is precisely what the B37 pre-registration and the B38
digest work exist to make possible. Evaluation artifacts, when that stage arrives, will raise.

#### 20. Governed paths clean, HEAD pushed, protected PDF unstaged — **PASS**

```
$ git status --porcelain src/ configs/ scripts/ requirements* docs/EVALUATION_CONTRACT.md docs/IMPLEMENTATION_CONTRACT.md
(no output)

$ git status --porcelain
 M docs/reference/reference.pdf

local  = c059e1b0cb6945d5f985d2f19132e33b422fed28
origin = c059e1b0cb6945d5f985d2f19132e33b422fed28
```

Governed set produced no output — clean. Local == origin. The PDF shows ` M`: index column a space,
therefore unstaged.

---

## 2. Failure-mode audit

### A. Pod evicted mid-run

**Can happen — likely, on Community Cloud.** `last.pt` is written every `ckpt_interval` iterations
(CLI default **2000**, `train_e1.py:531`) and at `max_iters` (`:467`), atomically (`:102-111`), so a
kill mid-write destroys the `.tmp` and leaves the previous good checkpoint intact.

**Survives:** model, optimizer (SGD momentum), scheduler position, `best_miou`, `best_ckpt`, all five
RNG streams, `prev_lr`. The telemetry JSONL is opened per write in append mode (`:205-209`), so no
buffered rows are lost and a resumed run continues the same file.

**Lost:** up to `ckpt_interval − 1` iterations of work, and the intra-epoch sample position (#6).
**Must be re-done:** re-launch with `--resume <ckpt-dir>/last.pt`; it continues from `iter + 1`.

**Unknown (§4):** whether `--ckpt-dir` survives the eviction at all. Checkpoints are written outside
the repo, on RunPod conventionally `/workspace/e1_ckpts`. Whether that path is a persistent network
volume or ephemeral container storage is a **pod provisioning choice this repository cannot
determine**. If it is ephemeral, everything above is moot and the run restarts from zero.

### B. Dataset present, different pixel histogram than the weights assume

**Can happen, and would not be caught by the launch gate.** `dataset.py:66-75` checks **cardinality**
(5,367/846/1,561), not content. `preflight_e1.py` stage 1 checks the weight artifact's **internal**
consistency — length, finiteness, argmin/argmax, `w[42]/w[0] == sqrt(N_0/N_42)`, and elementwise
reproduction from its own `pixel_counts` — all of which are properties of the JSON, not of the
mounted data.

**What the code does:** nothing. Training proceeds with weights derived from a different distribution.
The failure is silent and would only surface as anomalous class-wise IoU much later.

**The one check that closes this exists but is advisory and not wired in:**
`scripts/verify_class_weights_pod.py` recomputes the full train histogram and asserts exact equality
against `pixel_counts` and `total_nonignore_pixels`. It is deliberately **not** a `preflight_e1.py`
stage. If the operator does not run it, nothing else will.

### C. OOM at batch 16

**Can happen** (24 GB RTX 4090, 512², MobileNetV3 + LR-ASPP, 116-class logits).

**Documented policy, no mechanism.** `docs/IMPLEMENTATION_CONTRACT.md:590-594`:

> "16 is a **physical** batch. Gradient accumulation is deliberately **not** introduced… The GPU must
> accommodate the registered batch; if it genuinely cannot, that is an explicit experiment-design
> issue, not a silent change of batch semantics."

`grep -rn "accumulation|accum_steps|accumulate_grad" src/training/train_e1.py` → exit 1, no match:
**gradient accumulation is not implemented**, consistent with the contract.

**What the code does on OOM:** nothing catches `torch.cuda.OutOfMemoryError`; it propagates and the
process dies. **Does a workaround change the recorded config?** `--batch-size` is accepted
(`:522`, `:592`) with **no warning and no guard** when it differs from the registered 16. It *is*
recorded — `"batch_size"` in the run_meta row (`:319`) — so the deviation is visible in telemetry
afterwards, but nothing flags it at launch. An operator reacting to an OOM by lowering the batch would
produce a run that silently departs from the registered design and is detectable only by reading the
JSONL.

### D. NaN / inf loss

**Detected, and the run stops — on real runs only.** `train_e1.py:392-396`:

```python
if mode == "real" and not bool(torch.isfinite(loss)):
    raise RuntimeError(
        f"non-finite loss at iter {it}: loss={loss.item():.4f} ce={ce.item():.4f} "
        f"dice={dice.item():.4f}. Aborting the real E1 run to avoid poisoning the model "
        "or wasting compute.")
```

Checked **every iteration**, before `backward()`, so a non-finite loss never reaches the optimizer.
The message names the iteration and decomposes CE vs Dice. In dry-run mode the check is skipped
(`loss_finite` is still recorded as a hard check at `:399`).

### E. Validation mIoU with a class absent from val

**Handled by the union-present rule; no NaN reaches the headline unless the matrix is empty.** Class 68
has no val ground truth. Traced by exercising `miou_from_confusion`:

```
perfect prediction, only classes {0,5} present:
  all-class mIoU        = 1.0
  eligible classes      = 2 (union-present)

model predicts absent class 68:
  all-class mIoU        = 0.65625
  eligible classes      = 3 -> class 68 now counted with IoU 0

empty confusion matrix (nothing eligible):
  all-class mIoU        = nan
```

A class absent from **both** GT and prediction is omitted from the mean; a class absent from GT but
predicted contributes `IoU = 0` and **is** counted — so hallucinating class 68 lowers the score.
`NaN` appears only when nothing at all is eligible, which cannot occur on a real val pass
(background is present in 846/846 val masks). `train_e1.py:456` compares `all_miou > best_miou`;
a `NaN` would compare `False` and simply never update the best checkpoint rather than corrupting it.

### F. Disk fills during training

**Bounded by design, then fails loud.** Checkpoints and telemetry both live in `--ckpt-dir`, outside
the repo. Retention is capped: `prune_checkpoints` (`:185-202`) keeps the `keep_ckpts` newest best
checkpoints (default **3**, `:537`) and never deletes `last.pt`, `best.json`, or the current best. So
steady-state usage is roughly 3 best + 1 last (~24 MB each per the comment at `:188`) plus the JSONL
(one row per iteration; ~80k rows).

**If the disk fills anyway:** `torch.save` inside `_atomic_save` raises, and there is no `try`/`except`
around it — the run dies. Because the write goes to `<name>.tmp` and `os.replace` only runs on
success, **the previous good checkpoint is left intact**. A failed `_jsonl` append would likewise
raise. Fail-loud with the last resume point preserved is the correct outcome; there is no silent
degradation.

### G. `PLANTSEG_DATA_ROOT` unset

**Fails immediately with a good message.** `configs/data.py:33` falls back to the Windows default;
`dataset.py:39-41`:

```python
root = Path(DATA["root"])
if not root.exists():
    raise FileNotFoundError(f"dataset root does not exist: {root}")
```

On Linux the Windows path cannot exist, so the run aborts at the first `build_dataloader` call
(`train_e1.py:286`) — before the model trains a single iteration — naming the exact path it tried.
Inside the image the variable is always set (`Dockerfile:73`), so the realistic failure is a
**mounted-volume** problem, which produces the same error naming
`/workspace/plantseg_data/plantseg`. Message quality: good — it states the resolved path, which is the
one thing the operator needs.

There is no *silent* failure mode here: the code never proceeds with a partial or empty dataset
(`dataset.py:61-62` also raises if zero pairs are found, and `:57-60` if any mask is missing).

---

## 3. First-hour prediction

What the operator should see on a correctly provisioned pod. Iteration-indexed facts are derived from
code; wall-clock is not (see §4).

**Pre-flight.** `python scripts/preflight_e1.py` → `VERDICT: GO`, exit 0, with all five stages PASS:

```
  class_weights             PASS     (~3-9 s)
  verify_env                PASS     requires CUDA, sm<=9.0, dataset verified
  smoke_dataloader          PASS
  smoke_aug_stochasticity   PASS     R5 seed sequence must MATCH the B30 §9 baseline
  train_e1 --dry-run        PASS     6/6 checks exercised, 0 skipped
```

Stage 4 additionally compares the observed augmentation seed sequence against
`B30_SEED_BASELINE = [[1608637542, 1273642419], [1935803228, 787846414]]`
(`preflight_e1.py:65`). A **DEVIATION** verdict there is a hard NO-GO and means every seed-dependent
artifact under `reports/` describes a schedule the pod will not run.

**Launch banner.** `--real-run --confirm-real-run` should print
`[mode] REAL | torch 2.1.0+cu121 | device=cuda | cuda_available=True | num_classes=116`, then
`[budget] max_iters=80000 batch_size=16 val_interval=4000 …`, then
`[student] params=… used_pretrained=True`, `[data] train_index=5367 val_index=846`,
`[loss] CombinedCEDiceLoss(weight=len116, ignore_index=255) weight_on=cuda:0`,
`[opt] SGD lr=0.01 momentum=0.9 weight_decay=0.0001 | scheduler=PolynomialLR (total_iters=80000, power=0.9)`,
and `[ckpt] dir=… (verified OUTSIDE repo)`.

**First-iteration loss ≈ 5.0–6.5.** The head is randomly initialised regardless of ImageNet backbone
init, so initial per-pixel predictions are near-uniform over 116 classes: CE ≈ `ln(116) ≈ 4.75`, and
soft Dice with near-uniform probabilities gives ≈ 0.98–1.0, for a total ≈ **5.7**. Measured locally on
CPU with random init and `batch_size 2`: iter 1 `loss=5.9618`, iter 2 `loss=6.5878`; the B36 drill saw
5.6–6.6 across iterations 11–20. ImageNet init should if anything start slightly lower, and the value
is noisy at small batch. **A first loss near 4.75 with no Dice component, or below ~2, or above ~15,
does not match this architecture at initialisation.**

**Log cadence: every iteration.** `--log-every` defaults to **1** (`:539`), so `[iter n/80000] …`
prints 80,000 times, and `_jsonl` writes one `train` row per iteration regardless. Expect a very
chatty stdout; this is by design, not a fault.

**LR at iteration 1 ≈ `9.99988750e-03`** and monotonically non-increasing thereafter
(`lr = 0.01 * (1 - it/80000)^0.9`).

**First checkpoint: `last.pt` at iteration 2000** (`ckpt_interval` default 2000, `:467`), printed as
`[last 2000/80000] resume point -> …/last.pt`.

**First validation: iteration 4000** (`val_interval` from `E1_STUDENT["val_interval"] = 4000`), full
val set (846 images, `max_val_batches=None` on a real run), printed as
`[val 4000/80000] cm_batches=53 cm_total_px=… all_class_miou=…`. Because `best_miou` starts at
`-inf`, that first validation **always** writes a best checkpoint and `best.json`.

### Three observations that mean something is wrong

1. **`used_pretrained=False` in the `[student]` line.** The real-run default is ImageNet
   (`:590-591`); `False` means `--init none` was passed or the config was altered. The run would be a
   different experiment from the registered one, and nothing else will stop it.
2. **`[data] train_index=` anything other than 5367** — or the run aborting with
   `split=train pair count MISMATCH`. Either means the mounted dataset is not the audited one, and
   the CE class weights (computed over exactly those 5,367 masks) no longer describe it.
3. **`weight_on=cpu` in the `[loss]` line, or `device=cpu` in the mode banner.** The real run refuses
   CPU at `:582-589`, so seeing `cpu` at all means the guard was bypassed or `--device` was forced —
   an 80k-iteration CPU run would take weeks.

A fourth, cheaper tell: `git_head` in the first JSONL row. It **will** read `"UNKNOWN"` inside the
image (#19). That is currently expected, not a fault — but once #19 is fixed, `"UNKNOWN"` becomes a
signal that the run is not the image you think it is.

---

## 4. What this audit could not determine

**Verified locally, expected to hold on the pod:**

- The gate's stage ordering, refusal messages and exit codes (exercised locally, and stage 1 exercised
  inside the published image by digest).
- Every version pin, the class-weight artifact's contents, and the class-weight gate stage — all
  verified **inside the published image addressed by digest**, so they hold wherever that digest runs.
- The seeding protocol's CPU-visible half (`random`/`numpy`/`torch`). `torch.cuda.manual_seed_all` and
  the cuDNN determinism flags were **set** but cannot be observed taking effect without a GPU.

**Unverifiable until the pod exists:**

- **GPU-dependent preflight stages (2–5).** `verify_env` needs CUDA and the dataset; the compute-
  capability gate (`sm ≤ 9.0`) has never executed against a real device.
- **Whether batch 16 fits in 24 GB.** No measurement exists in this repository, and §2 C shows there
  is no fallback if it does not.
- **Throughput and wall-clock.** No recorded it/s or GPU-hour estimate exists anywhere in `docs/` or
  `reports/` (searched). Every §3 milestone is therefore stated in **iterations**, not minutes.
- **Whether the checkpoint directory survives eviction.** This is a RunPod volume-type choice, not a
  repository property (§2 A).
- **Whether the mounted dataset's pixel histogram matches the class weights.** Only
  `verify_class_weights_pod.py` can answer this, and it must be run on the pod (§2 B).
- **The R5 augmentation-seed comparison on the pinned stack.** `preflight_e1.py` stage 4 compares the
  observed sequence against the B30 §9 baseline; that comparison has never run on the pinned Linux
  stack with a GPU present.
- **Whether the GHCR package is publicly pullable.** Not checked — it requires an unauthenticated
  pull from outside this machine's credential state.

**Determined, and negative:** #19. No inference was required — the container was run by digest and
`_git_head()`, `git_commit()` and `git_porcelain_bytes()` were called inside it.

---

## 5. Disagreement with a prior report

B38 §5.5 recorded the `.dockerignore` exclusions as correct and complete, and B38 §8.2 recorded the
image as verified for launch. Both stand for what they checked. **Neither checked whether provenance
capture works inside the image**, and it does not (#19). The files win: `.git/**` is excluded, so the
Dockerfile's own load-bearing justification for installing `git` is not satisfied at runtime. This is
recorded as a finding, not as a correction to B38's scope.

---

## 6. Verdict

> ### Can E1 launch today on a correctly provisioned pod? **YES.**

All seven code-path criteria, all four config/data criteria, and all five gate/environment criteria
pass. The training loop is gated, fail-closed on the class weights, refuses CPU and in-repo
checkpoints, asserts split sizes on every dataloader build, consumes no unresolved config value,
detects non-finite loss and stops, and runs from a published image whose digest is recorded and
matches the registry.

**The one failure (#19) is not a launch blocker.** `_git_head()` degrades to `"UNKNOWN"` rather than
raising, so training proceeds. What is lost is attribution: every run_meta row on the pod will be
unattributable to a commit or an image, which undercuts the B37 pre-registration and the B38 digest
work rather than the training itself.

**Smallest change that flips #19** — two lines, no behaviour change:

```dockerfile
# Dockerfile, beside the existing ARG GIT_COMMIT
ENV PLANTSEG_GIT_COMMIT=${GIT_COMMIT}
```

```python
# train_e1.py:_git_head — prefer the baked commit when .git is absent
return os.environ.get("PLANTSEG_GIT_COMMIT") or <existing git rev-parse path>
```

The **image digest cannot be baked in** — a digest is the hash of the config that would have to
contain it, so it does not exist until after the build. It must be supplied at run time, e.g.
`docker run -e PLANTSEG_IMAGE_DIGEST=sha256:0572c116…` with one extra key in the run_meta row. That
is a runbook line plus one dictionary entry.

Neither change is required before launching. Both are cheaper now than after 80,000 iterations of
telemetry have been written with `"git_head": "UNKNOWN"`.

---

## 7. Note on the criteria

All 20 were scorable as written; none was substituted or invented. Two are worth a remark rather than
a change:

- **#12** conflates "exists and NO-GOs correctly" (fully verifiable locally, and it passes) with the
  GPU-dependent stages. Scored on what it asks; the environment-blocked part is called out separately.
- **#15** cannot be fully exercised under this audit's own read-only constraint, because forcing the
  dirty-tree and label-mismatch gates to abort requires writing to the repository. Scored PASS on the
  mechanism plus the happy path, with that limitation disclosed in the item rather than hidden.
