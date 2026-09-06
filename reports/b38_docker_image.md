# B38 — Phase 0: official Docker image

**Type:** build + verification. **Status:** BUILT, VERIFIED AND **PUSHED**.
**Dates:** built and verified 2026-09-05 · pushed to GHCR 2026-09-06.
**Image commit:** `28d1038a4b36caf02736b190dc3e032f181b7bb5` · **Task started from:** `fcc8acc`
(precondition verified: in sync with `origin/master`, governed-path porcelain returned no output).

> ## Summary
> A launch-blocking defect was found by verification and fixed: the CE class-weight artifact was not
> in the image, so **E1 could not have run**. The image now passes all six checks, and is on GHCR at
> digest **`sha256:0572c1166980d11ea0eef86aa7c2eb76b5ae6961ae406a77bbda9bcc66cedbe6`** (§8). A separate hazard
> was found on the registry: **an image already pushed to GHCR is 19 commits stale and must not be
> used** (§7). The missing build/push script — the root cause of that staleness — has been written.

---

## 1. What already existed — a verify-and-push task, not an authoring one

| artifact | state | provenance |
|---|---|---|
| `Dockerfile` | **already existed**, complete and well-documented | `e5a6349` (2026-08-17) |
| `.dockerignore` | **already existed**, deny-by-default allow-list | `e5a6349` |
| build/verify/digest procedure | **already existed**, `docs/runpod_environment.md` §1–§4 | — |
| prior local build evidence | **already existed**, `runpod_environment.md:136-148` | 2026-08-17 |
| **build/push script** | **did NOT exist** — written in this pass (§6) | — |

No new container definition was authored. The base is pinned by immutable digest
`python@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91`, and the Dockerfile
documents why not the historical PyTorch image: *"the upstream v2.1.0 Dockerfile defaults to
`ARG PYTHON_VERSION=3.8`… Using them would silently violate the registered interpreter."*

---

## 2. Version pins — no disagreement, and a correction to the install authority

> **Correction, recorded so the next reader does not repeat it.** The install authority is
> **`requirements-runpod.lock`** — the hash-pinned, 78-artifact file the Dockerfile installs from
> (`Dockerfile:53-56`, under `pip install --require-hashes`).
> **`requirements.lock` is the human-readable *exact-version registry***
> (`runpod_environment.md:7`); `requirements-runpod.in` is "derived verbatim from the registry minus
> Windows-only wheels". **`requirements-e1.txt` is student-only** (torch, torchvision, numpy, pillow)
> and installs nothing in the image. Reading pins from `requirements.lock` or `requirements-e1.txt`
> gives the right numbers by coincidence of agreement, not by authority.

All three agree on the six pins in scope, and all match the task specification. **No disagreement
found.** Verified elementwise inside the rebuilt image against the lock it was built from:

```
interpreter : 3.11.16 x86_64 Linux
torch       : 2.1.0+cu121 | torch.version.cuda = 12.1
pins parsed : 78

package          lock               installed          match
torch            2.1.0+cu121        2.1.0+cu121        True
torchvision      0.16.0+cu121       0.16.0+cu121       True
numpy            1.26.4             1.26.4             True
mmcv             2.1.0              2.1.0              True
mmsegmentation   1.2.2              1.2.2              True
mmengine         0.10.7             0.10.7             True
opencv-python    4.8.1.78           4.8.1.78           True
scipy            1.11.4             1.11.4             True
statsmodels      0.14.6             0.14.6             True
pillow           12.3.0             12.3.0             True
scikit-image     0.23.2             0.23.2             True

mismatches: 0 []
```

> **Method note.** The first run of this check printed `False` on all eleven rows. That was a **bug in
> the comparison script**, not a version mismatch: a regex with escaped backslashes did not survive
> shell→docker→python quoting, so **0** pins parsed and every expected value was `None`. Recorded
> because a reader who saw only that output would have concluded the image was wrong. A verification
> that fails *open* — reporting mismatch when it merely failed to read its reference — is as dangerous
> as one that fails closed silently.

---

## 3. The launch-blocking defect: found, then fixed

### What was wrong

`.dockerignore` excluded `reports/**`, so `reports/e1_class_weights.json` was **not in the image**.
Running the gate inside the first build:

```
STAGE 1/5 — CE class-weight artifact (B34b Tier 1)
  [FAIL] artifact unreadable
         artifact contained: FileNotFoundError: [Errno 2] No such file or directory: '/workspace/plantseg-thesis/reports/e1_class_weights.json'

VERDICT: NO-GO — first failing stage: class_weights.
```

This was **not merely a failing gate stage**. `train_e1.py:293` calls `load_ce_weights()`
unconditionally, and B34 §A7 established that loader is **fail-closed** — no fallback to `None`, ones,
or a batch-derived vector. **E1 would have aborted at loss construction on a provisioned GPU pod.**

The in-build `--mode image` gate passed **36/36** regardless, because it proves *software identity*,
not *runtime readiness*. This defect lived precisely in the gap between the two gates, and the B34b
stage caught what it was built to catch.

### The fix — two parts, both required

**`Dockerfile`** — no `COPY` previously brought `reports/` in, so the ignore change alone was inert:

```dockerfile
COPY reports/e1_class_weights.json ./reports/
```

**`.dockerignore`** — placed **below** `reports/**`, because the file is **last-match-wins**; a
re-include above the exclusion is silently overridden (the first attempt at this made exactly that
mistake and was reverted):

```
!reports/e1_class_weights.json
```

**Only that one file is re-included, never `reports/` wholesale.** Verified inside the image:

```
$ ls -l reports/
-rwxr-xr-x 1 root root 6224 Jun 29 09:49 e1_class_weights.json
--- reports/ contents count: 1
```

**Alternative considered and rejected: mount it at runtime.** `train_e1.py:49` resolves
`REPO / "reports" / "e1_class_weights.json"` — the path is not configurable, so a mount would have to
shadow a directory inside the image on every pod, and any pod wired wrong kills E1. Baking a 6 KB
git-tracked provenance artifact makes the image self-sufficient, which is the point of the image.
Copying it also creates `reports/`, which `scripts/verify_class_weights_pod.py` needs to write into.

---

## 4. The provenance label

`Dockerfile`, +2 functional lines, no behaviour change, declared **last** so the `ARG` does not
invalidate the ~8.6 GB pip layer on every commit:

```dockerfile
ARG GIT_COMMIT
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
```

Built with the **full 40-character hash**, matching the `148af2c9b30f…` precedent and removing
short-hash ambiguity.

---

## 5. Verification — all six checks, on the rebuilt image

### 5.1 In-build software gate — PASS 36/36

The final build layer runs `preflight_environment.py --mode image`, which is fail-closed
(`verdict = passed == len(results)`; `return 0 if verdict else 1`):

```
RESULT: PASS (36/36)
```

### 5.2 `scripts/preflight_e1.py` inside the container — class_weights **PASS**

```
STAGE 1/5 — CE class-weight artifact (B34b Tier 1)
  [PASS] T1a  len(weights) == 116
  [PASS] T1b  all finite; dtype=torch.float32
  [PASS] T1c  provenance fingerprint (sorted[57]+sorted[58])/2 == 1.000000000000  [NOT a correctness check]
  [PASS] T1d  argmin=0 argmax=42; w[42]/w[0] = 259.551929472253 == sqrt(3703512045/54975)
  [PASS] T1e  w[0] = 0.036954205368 (background, 80.82% of pixels)
  [PASS] index alignment: all 116 weights reproduce (max deviation 0.000e+00)
```

```
  stage                     result       secs  detail
  class_weights             PASS          2.8  116 weights verified; ratio 259.551929
  verify_env                FAIL         10.4  verdict=FAIL — needs PASS (CUDA GPU, sm<=9.0, dataset OK)
  smoke_dataloader          SKIPPED         -  not reached
  smoke_aug_stochasticity   SKIPPED         -  not reached
  train_e1 --dry-run        SKIPPED         -  not reached

VERDICT: NO-GO — first failing stage: verify_env.
```

**NO-GO on CUDA/dataset grounds, as expected on a CPU builder with no dataset mounted — and the
required stage PASSES before that stop.**

### 5.3 V1 — imports, and the `MMCV_MAX` question, settled

```
mmcv 2.1.0 | mmseg 1.2.2 | mmengine 0.10.7 | compiled ops OK
MMCV_MIN = '2.0.0rc4'  MMCV_MAX = '2.2.0'
```

> ### The patch is NOT needed.
> `mmseg 1.2.2` ships `MMCV_MAX = '2.2.0'` out of the box, which already admits the pinned
> `mmcv 2.1.0`. The install is **unmodified**; no patch is applied in the image and none is required
> at runtime. All imports succeed, including compiled ops.

**UNAPPLIED recommendation.** [`docs/teacher_prep_runbook.md:105-108`](../docs/teacher_prep_runbook.md#L105)
instructs a manual edit whose premise is false for this version:

> "mmseg 1.2.2's `mmseg/__init__.py` upper-bounds mmcv **below 2.1.0**, so it **rejects** the pinned
> mmcv **2.1.0** at import. **Fix:** … relax the upper bound `MMCV_MAX` from `'2.1.0'` → `'2.2.0'`."

The constant is **already** `'2.2.0'`. Recommend correcting that step to state no edit is required for
the pinned pair, keeping its read-the-constant-first instruction as the check that establishes it —
that instruction is what would have caught this, had the on-disk value ever been read. **Not applied**
(the runbook is out of scope for this pass).

### 5.4 V2 — the image self-identifies

```
$ docker image inspect plantseg-thesis:official --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
28d1038a4b36caf02736b190dc3e032f181b7bb5
expected: 28d1038a4b36caf02736b190dc3e032f181b7bb5
```

**Exact match** to the commit whose Dockerfile produced it.

### 5.5 Exclusions — dataset, checkpoints, reference.pdf

`.dockerignore` denies everything (`**`) then re-includes an allow-list; re-exclusions win, with the
single documented exception in §3. Excluded: **`docs/reference/**`** (the protected PDF), `data/**`,
`datasets/**`, `plantseg_data/**`, `weights/**`, `outputs/**`, `*.pt` `*.pth` `*.ckpt` `*.onnx`,
archives, secrets (`.env` `*.pem` `*.key` `id_rsa*` `.ssh/**` `.aws/**` `*credentials*` `*secret*`),
`__pycache__`, `.git/**`, `.github/**`, `reports/**`. **No dataset, no checkpoint, no reference.pdf,
no secrets** — `reports/` in the image contains exactly one file (§3).

---

## 6. `scripts/build_and_push_image.sh` — the missing artifact (S1)

Written in this pass; it did not exist. **It is the fix for the root cause in §7.**

| requirement | implementation |
|---|---|
| commit **derived**, never an argument | `GIT_COMMIT="$(git rev-parse HEAD)"` — full 40 chars |
| refuse if any **governed path** is dirty | bare `git status --porcelain -- "${GOVERNED[@]}"`, verdict read from actual output; `docs/reference/reference.pdf` is deliberately **not** in the governed set |
| refuse if **HEAD is not on origin** | `git merge-base --is-ancestor HEAD origin/$BRANCH` |
| tag with **both** full hash and `:official` | three `-t` flags including the local tag |
| print the **registry digest** after push | `docker image inspect … {{json .RepoDigests}}` |
| **never handle credentials** | on push failure, exits with an instruction to run `docker login ghcr.io` |

It additionally re-verifies, before pushing, that the revision label matches the commit and that
`preflight_e1.py` stage 1/5 PASSes inside the built image — so §3's defect cannot reach the registry
again.

Gates exercised (`--dry-run`, while the script was still untracked):

```
REFUSING: governed paths are dirty (listed above). The image would not correspond to any commit.
```

---

## 7. STANDALONE HAZARD — the stale image already on GHCR (S2)

> ### ⚠ `ghcr.io/ainsleydeluna/plantseg-thesis@sha256:5f5dba46b8668949dc166bfbe7acc6db0fc90beba746b425d5e057183adbf9d2`
> ### is ON THE REGISTRY, is 19 COMMITS BEHIND, and MUST NOT BE USED TO PROVISION.

| field | value |
|---|---|
| digest | `sha256:5f5dba46b8668949dc166bfbe7acc6db0fc90beba746b425d5e057183adbf9d2` |
| tags | `:148af2c` · `:runpod-preflight-2026-08-17` |
| revision label | `148af2c9b30ffc78e7e2f4d0b9ff55cd40948607` |
| pushed | 2026-08-18 (`RepoDigests` populated, which only occurs after push/pull) |
| behind HEAD | **19 commits** |

Found only by inspecting local Docker state — **the push is documented nowhere** in the repository.
`grep` over `scripts/` for `docker build|--label|docker push|ghcr` returns two unrelated docstring
matches; `docs/` and `reports/` contain no `ghcr` reference.

**What it is missing** (`git diff --stat 148af2c..HEAD -- src configs scripts requirements-runpod.lock Dockerfile`):

```
 configs/data.py                     |  18 +-
 configs/distill.py                  |  17 ++
 scripts/preflight_e1.py             | 348 ++++++++++++++++++++++++++++++++++++
 scripts/smoke_aug_stochasticity.py  | 176 ++++++++++++++++++
 scripts/smoke_distill.py            |  10 +-
 scripts/smoke_kd_resolution.py      | 174 ++++++++++++++++++
 scripts/smoke_losses_metrics.py     |   7 +-
 scripts/smoke_resume_identity.py    | 348 ++++++++++++++++++++++++++++++++++++
 scripts/verify_class_weights_pod.py | 263 +++++++++++++++++++++++++++
 scripts/verify_env.py               |  90 +++++++++-
 src/data/dataset.py                 |  48 ++++-
 src/data/transforms.py              |  24 ++-
 src/training/__init__.py            |   2 +
 src/training/losses.py              |  30 +++-
 src/training/train_distill.py       | 116 ++++++++++--
 src/training/train_e1.py            | 304 ++++++++++++++++++++++++++++---
 16 files changed, 1914 insertions(+), 61 deletions(-)
```

It **predates `scripts/preflight_e1.py` entirely** — the gate did not exist — along with the
B31/B31c train-loop hardening and the B31-5/A1 augmentation-RNG fix in `dataset.py`. A pod
provisioned from it would run E1 with none of them, and with no gate to say so.

**Root cause.** The image was built and pushed by hand; its revision label was applied on the docker
CLI, since the Dockerfile has never contained a `LABEL` (touched only by `e5a6349` and `a15506f`,
neither of which is `148af2c`). Nothing recorded the procedure, so the published image drifted while
still looking authoritative. §6 removes that manual step.

**UNAPPLIED recommendation.** Add to `docs/runpod_environment.md`, in the image-identity section:

```markdown
**Authoritative image.** Provision only from the digest recorded in `reports/b38_docker_image.md` §8.
Any earlier tag or digest of `ghcr.io/ainsleydeluna/plantseg-thesis` is **superseded** — in
particular `sha256:5f5dba46b866…` (tags `:148af2c`, `:runpod-preflight-2026-08-17`), which predates
`scripts/preflight_e1.py` and the B31 training-loop fixes and must not be used. Build and push only
via `scripts/build_and_push_image.sh`, which derives the commit and refuses a dirty or unpushed tree.
```

**Not applied** — `docs/runpod_environment.md` is outside this pass's scope.

---

## 8. Image identity — THE CITABLE ARTIFACT

> # ```
> # ghcr.io/ainsleydeluna/plantseg-thesis
> #   @sha256:0572c1166980d11ea0eef86aa7c2eb76b5ae6961ae406a77bbda9bcc66cedbe6
> # ```
>
> ### This digest is the citable identifier. Chapter 4 cites the DIGEST, never a tag.
>
> `:official` **will move** — it is a moving pointer that will be reassigned by the next official
> build, and `:28d1038a4b36…` could in principle be force-pushed. A digest is content-addressed and
> cannot change meaning. **Every official artifact's run provenance must record the digest above.**
> Provision pods by digest: `docker pull ghcr.io/ainsleydeluna/plantseg-thesis@sha256:0572c116…`

| field | value |
|---|---|
| **registry digest (cite this)** | `sha256:0572c1166980d11ea0eef86aa7c2eb76b5ae6961ae406a77bbda9bcc66cedbe6` |
| registry | `ghcr.io/ainsleydeluna/plantseg-thesis` |
| tags (both resolve to that digest) | `:28d1038a4b36caf02736b190dc3e032f181b7bb5` · `:official` |
| `org.opencontainers.image.revision` | `28d1038a4b36caf02736b190dc3e032f181b7bb5` |
| source commit | `28d1038` — the commit whose Dockerfile produced it, present on `origin/master` |
| media type | `application/vnd.oci.image.index.v1+json` (linux/amd64 + attestation) |
| linux/amd64 manifest | `sha256:c85e02e195463d0e46ea7f372e10b72b0a4449bc97cfb6ba72d4094c475bee9a` |
| base | `python@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91` |
| pushed | 2026-09-06, via `scripts/build_and_push_image.sh` |

### 8.1 Round-trip verification — the pushed image is the verified image

The push was performed by `scripts/build_and_push_image.sh`, which **rebuilt** before pushing. That
rebuild produced a different local image ID from the earlier direct build
(`46e0f459…` → `0572c116…`), so the verifications were re-run **against the pushed artifact addressed
by digest**, not against the earlier local build. Raw output:

```
=== running BY DIGEST: ghcr.io/ainsleydeluna/plantseg-thesis@sha256:0572c116...cedbe6 ===
interpreter : 3.11.16 x86_64 Linux
torch       : 2.1.0+cu121 | cuda 12.1
openmmlab   : mmcv 2.1.0 | mmseg 1.2.2 | mmengine 0.10.7 | compiled ops OK
MMCV_MAX    : '2.2.0'
version mismatches vs lock (78 pins): 0 []
class-weight artifact present: True | reports/ entries: 1
```

```
STAGE 1/5 — CE class-weight artifact (B34b Tier 1)
  [PASS] T1a  len(weights) == 116
  [PASS] T1b  all finite; dtype=torch.float32
  [PASS] T1c  provenance fingerprint (sorted[57]+sorted[58])/2 == 1.000000000000  [NOT a correctness check]
  [PASS] T1d  argmin=0 argmax=42; w[42]/w[0] = 259.551929472253 == sqrt(3703512045/54975)
  [PASS] T1e  w[0] = 0.036954205368 (background, 80.82% of pixels)
  [PASS] index alignment: all 116 weights reproduce (max deviation 0.000e+00)
  class_weights             PASS          2.8  116 weights verified; ratio 259.551929
  verify_env                FAIL          9.9  verdict=FAIL — needs PASS (CUDA GPU, sm<=9.0, dataset OK)
VERDICT: NO-GO — first failing stage: verify_env.
```

**Label survived the push round-trip**, read back from the registry (not locally) with
`docker buildx imagetools inspect`:

```
$ docker buildx imagetools inspect ... --format "{{json .Image}}"  (labels extracted)
  platform linux/amd64      revision = 28d1038a4b36caf02736b190dc3e032f181b7bb5
```

**Both tags resolve to the same index digest** at the registry:

```
  28d1038a4b36caf02736b190dc3e032f181b7bb5      sha256:0572c116...cedbe6
  official                                      sha256:0572c116...cedbe6
```

**Local ↔ registry identity.** After the push, the local image's `RepoDigests` carries the registry
digest, confirming the local artifact and the registry artifact are the same content:

```
Local ImageId (config digest): sha256:0572c1166980d11ea0eef86aa7c2eb76b5ae6961ae406a77bbda9bcc66cedbe6
RepoDigests:                   ["plantseg-thesis@sha256:0572c116...cedbe6",
                                "ghcr.io/ainsleydeluna/plantseg-thesis@sha256:0572c116...cedbe6"]
Revision label:                28d1038a4b36caf02736b190dc3e032f181b7bb5
```

> **Note on identifiers, so the next reader does not mistake a non-match for a defect.** A local image
> ID and a registry digest are *not* required to be equal in general — the ID is a config digest and
> the registry digest is a manifest/index digest. Here they coincide because the containerd image
> store records the index digest as the ID. The load-bearing checks are the three above: the label
> round-trip, both tags resolving to one digest, and `RepoDigests` linking local to registry.

### 8.2 Gates that ran before this push

The push went through `scripts/build_and_push_image.sh`; it was **not** a direct `docker push`. Its
gates executed and passed in order:

```
  commit   : 28d1038a4b36caf02736b190dc3e032f181b7bb5
  branch   : master (present on origin)
---- docker build ----
#10 [11/12] COPY reports/e1_class_weights.json ./reports/
---- verifying the revision label ----
  label in image : 28d1038a4b36caf02736b190dc3e032f181b7bb5
  expected       : 28d1038a4b36caf02736b190dc3e032f181b7bb5
---- verifying the E1 gate's class-weight stage inside the image ----
    class_weights             PASS          5.7  116 weights verified; ratio 259.551929
```

An earlier invocation of the same script **refused to push** with
`error from registry: unauthenticated`, exiting with its documented instruction to run
`docker login ghcr.io`. It never touched a credential. That refusal is recorded here as evidence the
guard works on the failure path, not only the happy path.

### 8.3 Ordering

The recipe (`Dockerfile`, `.dockerignore`, `scripts/build_and_push_image.sh`) was committed as
`28d1038` and pushed to `origin/master` **before** the image was built and pushed. So the image's
label names a commit that exists publicly and whose Dockerfile can actually produce it. Building first
and committing after would have labelled the image with a commit whose recipe could not have built
it — a provenance lie. This report is committed afterwards because it must contain the digest, which
does not exist until the push completes.

**Ordering.** The recipe (`Dockerfile`, `.dockerignore`, `scripts/build_and_push_image.sh`) was
committed as `28d1038` **before** the rebuild, so the image's label names a commit whose Dockerfile
can actually produce it. Building first and committing after would have labelled the image with a
commit whose recipe could not have built it — a provenance lie. This report is committed afterwards
because it must contain the verification output of the image built from that recipe.

---

## 9. What was NOT done

- **No credential was handled, printed, or stored at any point.** The operator ran
  `docker login ghcr.io`; the script only ever reacts to an authentication *failure* by printing an
  instruction. An earlier run refused the push for exactly that reason (§8.2).
- **The GHCR package's visibility was not changed** — those steps are for the operator to run.
- **The GPU preflight was not run** — it requires a real accelerator. `--mode image` proves software
  identity only and says so in its own output.
- **`docs/teacher_prep_runbook.md` was not edited** (§5.3 recommendation unapplied).
- **`docs/runpod_environment.md` was not edited** (§7 recommendation unapplied).
