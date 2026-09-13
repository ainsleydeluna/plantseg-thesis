# B55 — teacher acquisition: BLOCKED at step 1 by a missing dependency of the pinned stack

**Type:** blocker record + correction to [B54](b54_teacher_prerequisites.md). **Status:** the gated
sequence **stopped at step 1 of 6**. Nothing was downloaded, no VRAM number was obtained, and all
four `teacher_init_source.md` fields remain `NEED_TO_CONFIRM`.
**Date:** 2026-09-13. **Pod:** `17e08c4bbf2d` / `g3xbbjz06abf2p`, Secure A40, `CA-MTL-1`, terminated
the same session. Cost ≈ **$0.10**.
**Image:** `sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf` — the same
digest that ran E1 seed 42.

> ## The finding in one line
>
> **`mmsegmentation 1.2.2` cannot be imported in the official image**, because its dependency
> **`ftfy` is in no lock file, no `.in` file and not in the Dockerfile**. Every model class fails to
> register, so the teacher cannot be built, sized, or initialised.

---

## 1. What this changes

[B54](b54_teacher_prerequisites.md) concluded **one blocker**, having found the teacher stack
installed. That conclusion is **withdrawn**. There are two, and the second is worse than the one B54
dismissed — not a missing config file, but a missing dependency of the pinned stack itself.

| B54 said | Actually |
|---|---|
| §3 "the image carries the teacher stack" — **NOT A BLOCKER** | **BLOCKER.** Packages installed; `mmseg` not importable past its top level (§2) |
| §3.6 the second-image plan is **unnecessary** | **Necessary**, by the opposite route (§6) |
| §6 item 4 "dependencies — **DONE**, zero cost" | **OPEN**; needs a `requirements*` edit and an image build |
| §6 "no clone needed, the source is baked into the image" | **Wrong on this pod shape** — the volume shadows it (§4) |
| §2 the checkpoint is the one real blocker | **Still true**, still unresolved — the sequence never reached the download |

What B54 got right and this session confirms: the checkpoint must be fetched (B8 stands), the init
test is CPU-only and read-only, and **G2 gates provisioning**.

---

## 2. The blocker — `ftfy`

### 2.1 Where it stops, verbatim

`scripts/test_teacher_init.py` (step 1), `harness_smoke.log`:

```text
FAIL: MMSegmentation env not importable (ModuleNotFoundError: No module named 'ftfy').
      Run inside the pinned env: torch 2.1.0, mmcv 2.1.0, mmseg 1.2.2.
EXIT CODE: 2
```

**Exit 2 of the wrong kind.** The script returns 2 from two places — `:60` for an un-importable MM
environment, `:66` for a missing config/checkpoint. Step 1's pass condition was the *second*. Getting
the first is the failure the step existed to catch, and the sequence stopped there.

### 2.2 The import chain, verbatim — `ftfy_traceback.log`

```text
  File "/usr/local/lib/python3.11/site-packages/mmseg/apis/__init__.py", line 2, in <module>
    from .inference import inference_model, init_model, show_result_pyplot
  File "/usr/local/lib/python3.11/site-packages/mmseg/apis/inference.py", line 14, in <module>
    from mmseg.models import BaseSegmentor
  File "/usr/local/lib/python3.11/site-packages/mmseg/models/__init__.py", line 3, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/usr/local/lib/python3.11/site-packages/mmseg/models/backbones/__init__.py", line 2, in <module>
    from .beit import BEiT
  File "/usr/local/lib/python3.11/site-packages/mmseg/models/backbones/beit.py", line 19, in <module>
    from ..utils import PatchEmbed
  File "/usr/local/lib/python3.11/site-packages/mmseg/models/utils/__init__.py", line 2, in <module>
    from .basic_block import BasicBlock, Bottleneck
  File "/usr/local/lib/python3.11/site-packages/mmseg/models/utils/basic_block.py", line 10, in <module>
    from mmseg.utils import OptConfigType
  File "/usr/local/lib/python3.11/site-packages/mmseg/utils/__init__.py", line 18, in <module>
    from .tokenizer import tokenize
  File "/usr/local/lib/python3.11/site-packages/mmseg/utils/tokenizer.py", line 13, in <module>
    import ftfy
ModuleNotFoundError: No module named 'ftfy'
```

**Unconditional.** `mmseg/utils/__init__.py:18` pulls the CLIP tokenizer at module scope, and
`tokenizer.py:13` imports `ftfy` with no `try`/`except`. There is no degraded mode: any path reaching
`mmseg.utils` — which includes every backbone, since `basic_block.py` imports `OptConfigType` from
it — fails.

### 2.3 The measurement that proves the teacher cannot be built

`import_audit.log`:

```text
ftfy                 FAIL ModuleNotFoundError: No module named 'ftfy'
mmcv                 OK   2.1.0
mmengine             OK   0.10.7
mmseg                OK   1.2.2
mmseg.registry       OK
mmseg.structures     OK
mmseg.models         FAIL ModuleNotFoundError: No module named 'ftfy'
mmseg.apis           FAIL ModuleNotFoundError: No module named 'ftfy'

--- can the model registry actually build? ---
MODELS registry imported OK, entries: 0
```

**`entries: 0` is decisive.** The registry object imports cleanly and is **empty**, because
registration is a side effect of importing `mmseg.models`. `MODELS.build(cfg.model)` fails for any
architecture. So step 2 (VRAM probe), step 5 (init test) and step 6 (key audit) were all
unreachable. Steps 3–4 would have run, but downloading a checkpoint nothing can load was not worth
the minutes — which is what the gating order was for.

### 2.4 `ftfy` is absent from the specification, not merely from the image

`[MEASURED 2026-09-13]`, instruments stated per N17:

| Claim | Instrument | Result |
|---|---|---|
| Not installed on the pod | `pip freeze` (76 packages) + `pip show ftfy` | absent; `WARNING: Package(s) not found: ftfy` |
| Not in the runtime lock | `grep -ci '^ftfy' requirements-runpod.lock` — pattern contains the adjudicated string; a lock line is `ftfy==x.y.z` | **0** |
| Nowhere in the repository | `grep -rn -i 'ftfy'` across all files except `.git/` | **no matches** |

### 2.5 How the lock came to miss it — hypothesis tested, and refined

The working hypothesis was that `regex` (the tokenizer's *other* import, which **is** installed at
`2026.5.9`) arrived transitively while `ftfy` did not, implying the lock never captured mmseg's
dependency closure. **Tested and refined rather than confirmed.**

`regex` is listed **directly** at `requirements-runpod.in:56`. But so is nearly everything else:
`requirements-runpod.in` is not a pip-tools style file of direct requirements — it is a **flat,
fully-pinned list** including every transitive package (`addict`, `aliyun-python-sdk-core`, `cffi`,
`mdurl`, `wcwidth`…).

Compared against the pod's `pip freeze`:

```text
requirements-runpod.in entries : 75
pod pip freeze entries         : 76
in .in but NOT in freeze       : (none)
in freeze but NOT in .in       : triton==2.1.0
```

**`requirements-runpod.in` is a `pip freeze` snapshot of some environment**, differing from the pod's
by exactly one entry — `triton`, which pip pulls as a torch dependency and which the snapshot missed.

So the mechanism is not a failed resolution. **`ftfy` was absent from the source environment the
snapshot was taken from** — which means that environment never imported `mmseg.models` either. The
same layer-1 verification gap as §3, present at lock-generation time.

**This predicts further gaps in lazily-imported paths**, because nothing ever exercised them. It is
the mechanism most likely to bite again during the KD stages.

Two candidate explanations remain, and **neither is determinable from this container** (mmseg is not
installed here):

1. `mmsegmentation 1.2.2` declares `ftfy` as a runtime dependency, and the source environment was
   assembled in a way that skipped it (e.g. `--no-deps`, or a partially-populated cache).
2. `mmsegmentation` does **not** declare it — `ftfy` is an undeclared or extras-only dependency that
   `mmseg/utils/tokenizer.py` imports anyway, in which case *no* resolver would have caught it.

**One line settles it on the next pod:** `pip show mmsegmentation` and read the `Requires:` field.
That is G12's first diagnostic, because the two answers imply very different audits.

### 2.6 Nothing was installed to work around it

The image is hash-locked under `--require-hashes`; a missing package is a finding, not something to
patch with `pip`. Installing `ftfy` on the pod would also have put the running environment out of
correspondence with the attested digest, destroying the provenance chain the campaign rests on.

---

## 3. Why this was not caught — the four-layer verification gap

`verify_env` reported `mmseg : actual=1.2.2 pinned=1.2.2` and `PASS` on the E1 pod, and
[B54 §3.3](b54_teacher_prerequisites.md) cited that as runtime confirmation that the image "carries
the teacher stack". **That inference does not follow.**

`scripts/verify_env.py:61-73`:

```python
def import_version(module_name: str, dist_name: str | None = None):
    """Return version string, or None if the package is not installed (never raises)."""
    try:
        m = importlib.import_module(module_name)
    except Exception:
        return None
    v = getattr(m, "__version__", None)
```

It calls `importlib.import_module("mmseg")` — the **top-level package only**. `mmseg/__init__.py` is
thin and does not pull `mmseg.utils`, so it imports fine and exposes `__version__ = 1.2.2`. The check
is true and useless for the question asked of it.

### 3.1 The refinement to N17 — name the layer

"X is available" is not one claim. For an installed framework there are at least **four distinct
layers**, each requiring its own test:

| Layer | Claim | Test | Status on this image |
|---|---|---|---|
| 1 | the package is installed and its top level imports | `import mmseg` | **PASS** — what `verify_env` tests |
| 2 | its submodules import | `import mmseg.models` | **FAIL** (§2.3) |
| 3 | the registry is populated | `len(MODELS.module_dict) > 0` | **FAIL** — `entries: 0` |
| 4 | configs resolve | `Config.fromfile('mmseg::…')` | **not reached** |

**`verify_env` tested layer 1. B54 claimed layer 4.** Two layers of unexamined inference between
instrument and conclusion.

**Rule, added to N17:** *when adjudicating "X is available", name the capability being claimed and
test at that layer.* A pass at a lower layer is not evidence about a higher one.

The forecast — *version-present is not configs-present* — was directionally right and aimed **one
layer too high**. The break came at layer 2, not layer 4, and the guard written for layer 4 (the
probe's exit-3/exit-4 branch, Appendix A) never got the chance to fire.

### 3.2 Pattern instance 6 — a step change, not another tally

[B53 §5](b53_uvm_host_fault.md) records instances 3 and 5 as caught **in review, before
publication**. This one was different in kind:

- It was written into B54, committed as `8259469`, **pushed**, and acted on.
- It was caught by a **rented GPU pod**, not by reading.
- **It is the first instance in the ledger that cost money.**

Recorded as a **step change in consequence**, not as an increment. The earlier instances cost a
paragraph of rework; this one cost a provisioning cycle and a session's plan. What separates it from
3 and 5 is not the reasoning error — that is identical — but that nothing caught it before it was
spent against.

**N23 is promoted above the cosmetic queue items** on that basis: one line in step 0 checking
`len(MODELS.module_dict)` would have caught this before the pod was rented.

---

## 4. Second correction — the volume shadows the image's source tree

[B54 §6](b54_teacher_prerequisites.md) said no clone was needed because the source is baked into the
image. On this pod, `/workspace/plantseg-thesis` was **empty**:

```text
total 979
drwxrwxrwx 2 root root       1 Sep 13 13:58 .
drwxrwxrwx 3 root root 1001416 Sep 13 14:04 ..
ls: cannot access '/workspace/plantseg-thesis/src': No such file or directory
ls: cannot access '/workspace/plantseg-thesis/configs': No such file or directory
ls: cannot access '/workspace/plantseg-thesis/scripts': No such file or directory
```

`find / -maxdepth 3 -name seeds.py -path "*src*"` found no copy elsewhere. RunPod mounts a network
volume (`mfs#ca-mtl-1.runpod.net:9421`) at `/workspace`, **shadowing** whatever the image `COPY`'d
there. `PYTHONPATH=/workspace/plantseg-thesis` pointed at an empty directory.

**Same class of error as §3, different instrument:** reading `COPY src ./src` establishes what is in
the *image layer*, and nothing about what is *visible at runtime* once a volume is mounted over that
path. The site-packages are unaffected — they live in `/usr/local/lib/python3.11/site-packages`,
outside the mount — which is why §2's packages were present while the source tree was not.

**A clone is always required on this pod shape.** Done cleanly at the image's own commit:

```text
clone ok
detached
HEAD=f77d05d7b35187bf0da7e7b94a629549fe2e1c05
CHECKOUT OK — f77d05d7, the image commit
worktree (expect empty):          <- empty, i.e. clean
```

Written to `step0b_checkout.log`, deliberately **not** appended to `step0.log`, because
[B52 §2.2](b52_e1_seed42_completion.md) records how appending a checkout block into `step0.log` made
the E1 record ambiguous.

---

## 5. G2 — REMAINS UNRESOLVED

**No teacher VRAM number was obtained.** The probe was authored, syntax-checked, transferred and
hash-verified on the pod — and **never reached**, because `mmseg` could not import and the model
could not be built (§2.3).

Everything [B54 §4](b54_teacher_prerequisites.md) states therefore stands unchanged:

- **Teacher VRAM is not determinable.** The only figure in the repository remains
  `teacher_prep_runbook.md:166`'s **~31 GB** — unsourced, unmeasured, for **ADE20K's 150 classes**,
  and predating the determinism finding. **It must not be carried forward as evidence.**
- The `~37.7 GB` scaling in B54 §4.3 remains **`[ILLUSTRATIVE ONLY — DO NOT SIZE AGAINST THIS]`**.
- **Provisioning for the 40k fine-tune stays blocked.** No card can be chosen on evidence.

The probe (Appendix A) is ready to run the moment an image exists that can import `mmseg.models`. It
is **authored and syntax-checked but never executed**, so it is also untested against a real build —
that caveat travels with it.

---

## 6. The teacher image — and a determinism finding that changes G8

### 6.1 The second image is necessary after all, by the opposite route

[B54 §3.6](b54_teacher_prerequisites.md) dropped the second-image plan because nothing needed adding,
arguing that inaction was strictly better since it preserves `sha256:b80b645d…866aaf` for E1 seeds 2
and 3. **Something does need adding** — and B54's own reason for preferring inaction is exactly why
two images is now correct:

- A teacher image must add `ftfy` (and whatever else §2.5's audit uncovers), producing a **new
  digest**.
- E1 seed 42 is attested against `b80b645d…866aaf`, and seeds 2 and 3 must run against that digest.
- Therefore the teacher image is a **second** image, not a replacement.

### 6.2 Build FROM the E1 digest, not from base with an amended lock

The teacher image must be built as:

```dockerfile
FROM ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
# + the missing packages, pinned and hashed
```

**Not** from the same base with an amended lock. The reason is provenance, not convenience:

- Building **FROM** the attested digest makes the student stack **bit-identical by construction**.
  Every package E1 ran against is inherited, not re-resolved, so no resolver drift can silently
  change a version between the student and teacher environments.
- Re-running the lock from base would produce an environment that is *intended* to match and
  *verified* to match only as far as someone checks — the weaker guarantee, and the one this session
  just demonstrated the cost of.
- It lets ch3 state the relationship in one sentence: **"teacher image = student image + N
  packages"**, with N enumerable and auditable, rather than "two images built from the same lock,
  believed equivalent".

### 6.3 G8 cannot be ruled yet — the teacher may not be running the four measures at all

[B52 §11.3](b52_e1_seed42_completion.md) set the default for G5/G8/G9: *do not land mid-campaign
unless another change has already forced a rebuild, at which point it rides along at zero marginal
cost.* The teacher stage forces a rebuild. The question was whether G8 (`CUBLAS_WORKSPACE_CONFIG` as
a Dockerfile `ENV`) therefore rides along free.

**It does not, and the reason is a finding in its own right.**

`[MEASURED 2026-09-13 — instrument: `grep -rn 'set_seed' src/ scripts/ configs/`, full coverage of
every call site]`

| Stage | Seeded by | `CUBLAS_WORKSPACE_CONFIG` set by repo code? |
|---|---|---|
| E1 | `src/seeds.py` via `train_e1.py:301` | **Yes** — `src/seeds.py:26` |
| E2 / E3 (KD) | `src/seeds.py` via `train_distill.py:260` | **Yes** |
| Smoke tests | `src/seeds.py`, ~20 call sites | **Yes** |
| **Teacher (B1)** | **nothing in the repo calls `set_seed`** | **No** |

The teacher is seeded solely by MMEngine, through the config at
`configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:352`:

```python
randomness = dict(seed=42, deterministic=True)
```

**Consequence:** `CUBLAS_WORKSPACE_CONFIG` is set by **no repository code** on the teacher path. So
**ch3 §D's four-measure claim is verified for E1 and E2/E3 and unverified — possibly false — for the
teacher stage.**

**What is not determinable here:** whether MMEngine 0.10.7's `deterministic=True` sets
`CUBLAS_WORKSPACE_CONFIG` internally. `mmengine` is not installed in this container, so the claim
cannot be adjudicated — and per §3.1 it is not being guessed at. **One line settles it on the next
pod:** run the teacher config's randomness setup and echo the variable, exactly as `step0b`'s N4
block did for `set_seed`.

The two branches, both of which must be recorded before G8 is ruled on:

- **If MMEngine does NOT set it** → `CUBLAS_WORKSPACE_CONFIG` is unset for the entire teacher stage,
  ch3 §D's claim does not hold there, and **G8 stops being free — it becomes the only mechanism.**
  That converts G8 from a nice-to-have into a correctness fix.
- **If MMEngine DOES set it** → G8's default stays **don't**. Landing it in the teacher image only
  would create two images that differ in determinism configuration, which ch3 would then have to
  explain. **Zero marginal build cost is not zero cost.**

**G8 is therefore not ruled on here.** The determinism gap is raised regardless of where G8 lands,
and is queued as **G14**.

### 6.4 A related fact, recorded because it bears on sequencing

`scripts/launch_teacher_finetune.py` does not launch training. Its own docstring, `:18`:

> "The actual `runner.train()` invocation is future work and is explicitly not performed: the gate
> stops at ready-to-launch and reports."

So the teacher fine-tune has **no implemented runner invocation**. That is not a blocker this report
resolves, but it means the teacher stage needs code as well as an image, and it should be discovered
now rather than on a rented pod.

---

## 7. Next session — the sequence, and why it is G12 and not seeds 2/3

### 7.1 The gated sequence for the G12 validation pod

Written here so it is not reconstructed later. **Order is load-bearing.**

| # | Step | Why it comes where it does |
|---|---|---|
| 1 | **Step 0 gate** — UVM device-open, `cuInit`, N4 determinism echo, N5 captures | B53: a dead pod must be found before anything else |
| 2 | **Registry check (N23)** — `import mmseg.models; len(MODELS.module_dict)` | The check this session was missing. Seconds. If `entries: 0`, the image is still broken — **stop and terminate**, do not proceed |
| 3 | **VRAM probe (Appendix A)** — resolves G2 | The **first substantive action**, before download or anything else, because it gates card choice for the 40k run |
| 4 | `pip show mmsegmentation` → `Requires:` | Settles §2.5's two candidate mechanisms |
| 5 | MMEngine determinism echo | Settles §6.3's two branches, and therefore G8 |
| 6 | Harness smoke (`test_teacher_init.py`, expect exit 2 *file-not-found*) | Now able to reach its real pass condition |
| 7 | Download → hash → init test → 116-class key audit (Appendix B) | Steps 3–6 of the original sequence |

Steps 2 and 3 run **before** any download. If step 2 fails, nothing after it is reachable and the pod
should be terminated immediately — the same stopping rule that saved this session most of its cost.

### 7.2 Why not E1 seeds 2 and 3

An earlier draft of this report recommended seeds 2 and 3 as the next action, on the grounds that
they are unaffected by this blocker. **That recommendation is withdrawn — it contradicts the
contract.**

`docs/IMPLEMENTATION_CONTRACT.md:37-38`:

> "Single-seed vertical slice (**E1→E7 + full eval**) before any multi-seed scaling `[ch3 §D; ctx]`."

Seeds 2 and 3 **are** multi-seed scaling. The contract defers them until the vertical slice is
complete, and the slice has not started: this session established that the pipeline has never run
end to end, because the teacher stack does not import. E2/E3's design could still change as a
result.

The comparison is stark once stated:

| | E1 seeds 2+3 | G12 |
|---|---|---|
| Cost | ~46 GPU-hours, competing for the same rental | hours, **buildable without a GPU** |
| Unblocks | nothing | the teacher → E2, E3, and the rest of the slice |
| Contract status | deferred by `:37-38` until after the slice | on the critical path |
| Output | appendix-grade variance figures | a working pipeline |

**G12 first.** The facts in favour of the seeds remain true and are recorded in §7.3 — they are
simply not a reason to run them now.

### 7.3 What this blocker does not affect

- **E1 seed 42 is untouched.** The student stack imports no mmseg; `requirements-e1.txt` excludes it
  deliberately and `train_e1.py` never reaches it. The run, its checkpoint, and every number in B52
  stand.
- **E1 seeds 2 and 3 remain runnable** against `b80b645d…866aaf` exactly as attested, whenever the
  contract permits them.
- **E5/E6** (supervised-only student stages) do not need mmseg.
- **E2 and E3 are blocked**, because they need the teacher.

---

## 8. Pod record, artifacts and closures

### 8.1 ch3 field set — `step0.log`

```text
start_time_utc : 2026-09-13T14:04:08Z
pod_host       : 17e08c4bbf2d
RUNPOD_POD_ID  = g3xbbjz06abf2p
RUNPOD_DC_ID   = CA-MTL-1
RUNPOD_GPU_NAME = NVIDIA+A40
RUNPOD_CPU_COUNT = 9
RUNPOD_MEM_GB  = 50
RUNPOD_VOLUME_ID =
gpu            : NVIDIA A40, 46068 MiB, 8.6, 570.195.03
image_digest   : sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
rate           : $0.50/hr Secure A40 (operator-recorded)
overlay 25G (/) · mfs#ca-mtl-1.runpod.net:9421 965T (/workspace)

open /dev/nvidiactl OK · open /dev/nvidia-uvm OK · open /dev/nvidia-uvm-tools OK
cuInit rc 0 · is_available True
gpu            NVIDIA A40 44.43 GiB sm_86 · alloc OK
cpu_count 96 | RUNPOD_CPU_COUNT 9 | torch_threads 48
VERDICT: GPU USABLE — proceed
```

**The pod was healthy.** The UVM gate from [B53 §6](b53_uvm_host_fault.md) passed on a second Secure
pod in `CA-MTL-1` — a third Secure success against three Community failures, consistent with B53 §4
and still not a proof about Community in general.

Three confirmations worth recording:

1. **`44.43 GiB` is the torch figure, settled.** [B52 §3.1](b52_e1_seed42_completion.md) argued E1's
   `44.4` was a one-decimal rounding of the same `total_memory` value, distinct from nvidia-smi's
   `46068 MiB`. Printed at two decimals here it reads **44.43**, matching `grad_probe.log`'s P0
   header from the E1 pod exactly. The reconciliation holds.
2. **The CPU oversubscription reproduces.** `cpu_count 96` against `RUNPOD_CPU_COUNT 9`, with
   `torch_threads 48` — identical to E1 (B52 §3.3). **N6** is a standing configuration item, not a
   one-off.
3. **Rate drift, for N16.** `$0.50/hr` Secure A40 today against `$0.49/hr` for the same card on
   2026-09-11. Both operator-recorded. Recorded, not averaged.

### 8.2 N4 closed — determinism attested at runtime for the first time

`step0b_checkout.log`:

```text
CUBLAS_WORKSPACE_CONFIG before = [<unset>]
after set_seed: CUBLAS_WORKSPACE_CONFIG = :4096:8
deterministic  = True
warn_only      = True
cudnn.det      = True | cudnn.bench = False
```

All four ch3 §D measures captured at runtime, unset → `:4096:8` across the `set_seed` call.

**Scope, precisely:** this does **not** retroactively attest the E1 seed-42 run, which has no such
capture — a grep of every E1 artifact returned zero `cublas` hits — so
[B52 §6.6](b52_e1_seed42_completion.md)'s code-level wording stands unchanged. It closes **N4** going
forward, and it corroborates the mechanism on the identical image digest and commit.

**And note its limit against §6.3:** this attests the **`src/seeds.py` path**, which the teacher does
not use.

### 8.3 N5 and N9 closed

**N5** — `pip_freeze.txt` (76 packages) and `nvidia_smi.txt` captured. The first as-installed record
for this image, and the primary evidence in §2.4–2.5.

**N9** — hashes taken **pod-side before download** and verified after transfer. Every file matched:

| sha256 | file | bytes |
|---|---|---|
| `16a5f7e4f701d21b396e430af4fbe46ac7338a549a9e4f9b692cc0201bd68fa5` | `b55_teacher_blocked.tar.gz` | 3.4 kB |
| `917151cd6373d07bb4691a2d7152711648c594857afc22db28453e38dcbd83d9` | `step0.log` | 1191 |
| `cecf023566bf31616a96474b82e256265e60c017850c8cb9b92f6307356596dc` | `step0b_checkout.log` | 690 |
| `cd7ab2105181ca43bf42fa5c79b73304fce3ae7c1b3fa6e07a17f1cafc411800` | `pip_freeze.txt` | 1318 |
| `ab5a1800ea324548e847e26c8ff913939dccc39991e4d1e160c8174d7a3007cb` | `nvidia_smi.txt` | 1778 |
| `f98fed83790b41bcd515cacaa69af66bd779f1a199000797bf2a2b3f85bc6c2f` | `artifact_hashes.txt` | 324 |
| `0cf3b02df81308af85517e9bac127f3457b94cd6d556d8c2f81c20e5d4c6f9d9` | `harness_smoke.log` | 158 |
| `7c343722798b64cd1f9b75ed574064713a146693c47b2294e09fc9415cf7b26e` | `import_audit.log` | 449 |
| `6e785e8f0cd6001a045d40f216d6b1882f08686d01201faf3cf4fc5cda571224` | `ftfy_traceback.log` | 1408 |

`final_hashes.txt` is the manifest and has no pod-side counterpart — it cannot list its own hash.
Correct, not a gap.

**This is the first artifact bundle in the project with a pod-side hash verified after download** —
the procedure [B52 §8.3](b52_e1_seed42_completion.md) recorded as missing for E1's checkpoint, whose
laptop hash still stands alone.

The probes were transferred and hash-verified on the pod (`d0285c1e…` / `bf23d7b2…`) before use, and
the checkout's worktree was clean (`git status --porcelain` empty) — the `sys.dont_write_bytecode`
guard did its job.

---

## 9. Queue

**New, governed** — each requires a plan and an explicit go:

| # | Item |
|---|---|
| **G12** | Add `ftfy` (and audit the rest of the `mmsegmentation 1.2.2` dependency closure — §2.5) and build a **second, teacher-only image FROM the E1 digest** (§6.2). The E1 digest must not change while seeds 2/3 are outstanding. G8's inclusion depends on G14. |
| **G13** | `scripts/verify_env.py:61-73` tests layer 1 only. Extend the mmseg/mmcv check to import a **registration-forcing submodule** (`mmseg.models`) and assert `len(MODELS.module_dict) > 0`, so version-present is never again read as usable (§3.1). |
| **G14** | **Teacher determinism gap.** No repository code calls `set_seed` on the teacher path; it is seeded only by MMEngine's `randomness=dict(seed=42, deterministic=True)`. Establish whether MMEngine 0.10.7 sets `CUBLAS_WORKSPACE_CONFIG`; if not, ch3 §D's four-measure claim does not hold for the teacher stage and must be either fixed or amended (§6.3). **Blocks the G8 ruling.** |

**New, non-governed:**

| # | Item | Priority |
|---|---|---|
| **N23** | Add the registry check (`import mmseg.models`, `len(MODELS.module_dict)`) to step 0. One line; would have caught this before the pod was rented | **PROMOTED — ahead of the cosmetic queue items** (§3.2) |
| **N22** | Record in the runbook that a RunPod volume at `/workspace` **shadows** the image's source tree, so a clone at the image commit is always required (§4) | normal |
| **N24** | `scripts/launch_teacher_finetune.py` does not invoke `runner.train()` — the teacher stage needs code as well as an image (§6.4) | normal |

**Corrections to committed records:** B54 §1, §3, §3.6 and §6 are superseded by §1 of this report.
[B53 §5](b53_uvm_host_fault.md)'s pattern ledger gains **instance 6** (§3.2) — the first not
intercepted before publication, and the first to cost money.

---

## Appendix — the two probes, authored and never executed

Both scripts were written for this session, syntax-checked (`py_compile`), transferred to the pod and
hash-verified there. **Neither was executed**, because `mmseg.models` could not import. They are
embedded here in full because they otherwise exist only in an ephemeral container and a chat log, and
because §7.1 makes the VRAM probe the first substantive action of the next session.

**They live in `reports/` rather than `scripts/` deliberately:** `scripts/**` is on rule 8's path
list and therefore governed under **G11**. These are throwaway diagnostics, not repository tooling,
and committing them to `scripts/` would require a governed decision they do not warrant.

**Exit-code contract for Appendix A**, preserved because it encodes a distinction that matters:

| Exit | Meaning |
|---|---|
| **0** | Model built, probe ran — the VRAM number is in the output |
| **2** | CUDA unavailable — the probe cannot size a GPU it cannot see |
| **3** | `.mim/configs/segnext` **absent** and `Config.fromfile` failed → **B54 §3.4 is falsified**: the image carries the package but not the configs |
| **4** | `.mim/configs/segnext` **present** and `Config.fromfile` failed anyway → **a different fault**. B54 §3.4 is **NOT** falsified by this and must not be marked so; report the exception |

The 3/4 split exists because an earlier draft collapsed them, asserting a cause the probe had not
established — §3's error in miniature, caught in review that time.

### Appendix A — `teacher_vram_probe.py`

sha256 `d0285c1ebaf00f95e19066652e80da3e65459d0e4a3e3b1b372e7e6428add888` · 165 lines · 8128 bytes

```python
#!/usr/bin/env python3
"""B55 STEP 2 — teacher VRAM probe. Settles G2.

Random init (checkpoint=None): peak memory does not depend on weight VALUES, only on
shapes, dtypes and the graph. No download needed, so this runs BEFORE step 3.

Reports peak after forward, after backward, and after ONE AdamW step. The step is
included because AdamW allocates exp_avg + exp_avg_sq lazily on first step() -- 2x the
parameter memory -- and omitting it understates the real peak in the dangerous
direction. B48's OOM was at iteration 2, not iteration 1. Nothing is trained: random
weights, synthetic data, no checkpoint written, no dataloader, no dataset touched.

Determinism is configured via src.seeds.set_seed exactly as production, which is what
routes F.interpolate through torch._decomp and produced E1's 12.658 GiB transient.
"""

import os
import sys

# Write NO bytecode: importing src.seeds would otherwise create src/__pycache__ inside the
# repo working tree. .gitignore covers it, but a launch-time `git status` should be clean
# for reasons beyond git -- so the probe leaves the checkout byte-for-byte untouched.
sys.dont_write_bytecode = True

REPO = os.environ.get("PLANTSEG_REPO", "/workspace/plantseg-thesis")
CFG = os.path.join(REPO, "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py")
BATCH, CROP, NUM_CLASSES, IGNORE = 16, 512, 116, 255
GiB = 1024 ** 3

if REPO not in sys.path:
    sys.path.insert(0, REPO)


def gb(x):
    return x / GiB


def main() -> int:
    import torch
    from src.seeds import set_seed

    set_seed(42)  # production determinism protocol: cudnn.deterministic, warn_only, CUBLAS

    print("=" * 78)
    print("B55 STEP 2 — teacher VRAM probe (random init, one batch, no training)")
    print("=" * 78)
    print(f"torch                    {torch.__version__}")
    print(f"deterministic_enabled    {torch.are_deterministic_algorithms_enabled()}")
    print(f"warn_only                {torch.is_deterministic_algorithms_warn_only_enabled()}")
    print(f"CUBLAS_WORKSPACE_CONFIG  {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<unset>')}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable — this probe needs the GPU it is sizing.")
        return 2
    prop = torch.cuda.get_device_properties(0)
    total = prop.total_memory
    print(f"card                     {prop.name}  sm_{prop.major}{prop.minor}")
    print(f"total_memory             {gb(total):.3f} GiB  ({total // (1024*1024)} MiB)")

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS
    from mmseg.structures import SegDataSample
    from mmengine.structures import PixelData

    init_default_scope("mmseg")

    # RULING 1 — an mmseg:: resolution failure FALSIFIES B54 section 3.4, it is not merely a
    # step-2 blocker. verify_env checks package VERSIONS; version-present is not configs-present,
    # and the wheel can ship without mmseg/.mim/configs/. Establish coverage before trusting it.
    import mmseg
    mim_cfgs = os.path.join(os.path.dirname(mmseg.__file__), ".mim", "configs")
    segnext_dir = os.path.join(mim_cfgs, "segnext")
    print(f"mmseg package            {os.path.dirname(mmseg.__file__)}")
    print(f".mim/configs present     {os.path.isdir(mim_cfgs)}")
    print(f".mim/configs/segnext     {os.path.isdir(segnext_dir)}")
    try:
        cfg = Config.fromfile(CFG)
    except Exception as e:  # noqa: BLE001
        configs_present = os.path.isdir(segnext_dir)
        print("\n" + "!" * 78)
        if not configs_present:
            # Directories ABSENT and resolution failed -> the wheel did not ship the configs.
            print("STOP — B54 SECTION 3.4 IS FALSIFIED.")
            print(f"  Config.fromfile failed: {type(e).__name__}: {e}")
            print(f"  .mim/configs/segnext ABSENT at {segnext_dir}")
            print("  The image carries the mmseg PACKAGE but not the CONFIGS its _base_ needs.")
            print("  'the image carries the teacher stack' is wrong in a way the version check")
            print("  could not detect: verify_env compares versions, never config availability.")
            print("  Do not work around this. Report it.")
            print("!" * 78)
            return 3
        # Directories PRESENT and resolution still failed -> a DIFFERENT fault.
        print("STOP — config resolution failed, but NOT for the reason B54 section 3.4 covers.")
        print(f"  .mim/configs/segnext PRESENT at {segnext_dir}")
        print(f"  Config.fromfile failed anyway: {type(e).__name__}: {e}")
        print("  The configs ARE shipped, so B54 section 3.4 is NOT falsified by this and must")
        print("  NOT be marked so. This is a different fault -- scope resolution, a mmengine")
        print("  version interaction, a path or permissions problem. Report the exception.")
        print("!" * 78)
        import traceback
        traceback.print_exc()
        return 4
    print(f"config                   {CFG}")
    print(f"_base_ resolved          {cfg.filename}")
    dh = cfg.model["decode_head"]
    print(f"decode_head.num_classes  {dh['num_classes']}  (expect {NUM_CLASSES})")
    print(f"decode_head.ignore_index {dh['ignore_index']}  (expect {IGNORE})")
    assert dh["num_classes"] == NUM_CLASSES, dh["num_classes"]

    # Random init: build from config only. load_from / work_dir sentinels are never read here.
    model = MODELS.build(cfg.model).cuda().train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params                   {n_params:,} (~{n_params/1e6:.1f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)

    imgs = torch.randn(BATCH, 3, CROP, CROP, device="cuda")
    gts = torch.randint(0, NUM_CLASSES, (BATCH, CROP, CROP), device="cuda", dtype=torch.long)
    samples = []
    for i in range(BATCH):
        s = SegDataSample()
        s.gt_sem_seg = PixelData(data=gts[i].unsqueeze(0))
        s.set_metainfo(dict(img_shape=(CROP, CROP), ori_shape=(CROP, CROP), pad_shape=(CROP, CROP)))
        samples.append(s)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc, base_res = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    print(f"\nresident before step     alloc {gb(base_alloc):7.3f}  reserved {gb(base_res):7.3f}  GiB")

    def peak(tag):
        torch.cuda.synchronize()
        a, r = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        print(f"{tag:24s} alloc {gb(a):7.3f}  reserved {gb(r):7.3f}  GiB")
        return a, r

    losses = model.loss(imgs, samples)
    loss = sum(v for v in losses.values() if v.ndim == 0)
    fa, fr = peak("PEAK after forward")

    loss.backward()
    ba, br = peak("PEAK after backward")

    opt.step()          # materialises AdamW exp_avg + exp_avg_sq; random weights, synthetic data
    sa, sr = peak("PEAK after AdamW step")

    print("\n" + "=" * 78)
    print(f"G2 ANSWER — max_memory_allocated  {gb(ba):.3f} GiB (fwd+bwd)   {gb(sa):.3f} GiB (incl. AdamW state)")
    print(f"G2 ANSWER — max_memory_reserved   {gb(br):.3f} GiB (fwd+bwd)   {gb(sr):.3f} GiB (incl. AdamW state)")
    print(f"card total                        {gb(total):.3f} GiB")
    print(f"headroom on this card             {gb(total - sr):.3f} GiB (vs reserved incl. AdamW)")
    print(f"loss value (sanity, not a metric) {float(loss):.6f}")
    print(f"VERDICT: {'FITS' if sr < total else 'DOES NOT FIT'} on {prop.name}")
    print("-" * 78)
    print("CAVEAT, carried with the number: this is a TRAINING-STEP peak on a SYNTHETIC batch.")
    print("It includes no validation-pass headroom -- ValLoop runs at batch 1 but holds the")
    print("model plus whole-image inference buffers, and is not exercised here. Real images")
    print("may also differ from randn in cache behaviour. Size against this number WITH that")
    print("caveat attached; it is a floor for the training step, not a ceiling for the run.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Appendix B — `teacher_key_audit.py`

sha256 `bf23d7b2d08ea5da24d9dbbe5ddc3a7facb468aadfe620a8a6c6b1cf1ac02a84` · 122 lines · 5186 bytes

```python
#!/usr/bin/env python3
"""B55 STEP 6 — load the ADE20K .pth into the 116-CLASS teacher config and audit every key.

Step 5 only proves the checkpoint loads into an ADE20K-configured (150-class) model.
The teacher is 116-class. This reports, verbatim:
    missing     -- in the model, absent from the checkpoint
    unexpected  -- in the checkpoint, absent from the model
    mismatched  -- in both, DIFFERENT shape
and the resulting decode_head.conv_seg.out_channels.

Shapes are compared MANUALLY before loading, because load_state_dict(strict=False)
still RAISES RuntimeError on a size mismatch -- strict only governs missing/unexpected.
Pre-filtering is the only way to enumerate all three categories.

Randomly initialised keys = missing UNION mismatched. Those are what the 40k fine-tune
must recover to reach ~42.05%.

READ-ONLY: no training, no optimizer, no write. CPU only.
"""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True   # leave the repo checkout byte-for-byte untouched

REPO = os.environ.get("PLANTSEG_REPO", "/workspace/plantseg-thesis")
CFG = os.path.join(REPO, "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py")
WEIGHTS = Path(REPO) / "weights"
EXPECTED_NUM_CLASSES = 116

if REPO not in sys.path:
    sys.path.insert(0, REPO)


def find_ckpt():
    if len(sys.argv) >= 2:
        return Path(sys.argv[1])
    for pat in ("segnext_mscan-b*ade20k*.pth", "segnext_mscan-b*.pth", "*.pth"):
        hits = sorted(WEIGHTS.glob(pat))
        if hits:
            return hits[0]
    return None


def main() -> int:
    import torch  # noqa: F401  (imported for its side effects on mm* imports)
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import CheckpointLoader
    from mmseg.registry import MODELS

    ckpt_path = find_ckpt()
    if not (ckpt_path and ckpt_path.exists()):
        print(f"FAIL: no .pth found in {WEIGHTS}. Run step 3 first.")
        return 2

    init_default_scope("mmseg")
    cfg = Config.fromfile(CFG)
    print("=" * 78)
    print("B55 STEP 6 — ADE20K checkpoint -> 116-class teacher config, key audit")
    print("=" * 78)
    print(f"config      {CFG}")
    print(f"checkpoint  {ckpt_path}")
    print(f"cfg decode_head.num_classes = {cfg.model['decode_head']['num_classes']}")

    model = MODELS.build(cfg.model)          # random init, 116-class head
    msd = model.state_dict()

    raw = CheckpointLoader.load_checkpoint(str(ckpt_path), map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    mkeys, ckeys = set(msd), set(state)
    missing = sorted(mkeys - ckeys)
    unexpected = sorted(ckeys - mkeys)
    mismatched = sorted(k for k in (mkeys & ckeys) if tuple(msd[k].shape) != tuple(state[k].shape))

    print(f"\nmodel keys {len(mkeys)}   checkpoint keys {len(ckeys)}")
    print(f"\n--- MISSING ({len(missing)}) — in model, not in checkpoint ---")
    for k in missing:
        print(f"  {k}   model{tuple(msd[k].shape)}")
    print(f"\n--- UNEXPECTED ({len(unexpected)}) — in checkpoint, not in model ---")
    for k in unexpected:
        print(f"  {k}   ckpt{tuple(state[k].shape)}")
    print(f"\n--- SIZE MISMATCH ({len(mismatched)}) — in both, different shape ---")
    for k in mismatched:
        print(f"  {k}   ckpt{tuple(state[k].shape)} -> model{tuple(msd[k].shape)}")

    # Load with mismatched keys dropped; confirm nothing else fails.
    filtered = {k: v for k, v in state.items() if k not in set(mismatched)}
    res_missing, res_unexpected = model.load_state_dict(filtered, strict=False)
    print(f"\nafter filtered load: missing {len(res_missing)}  unexpected {len(res_unexpected)}")

    rand_init = sorted(set(missing) | set(mismatched))
    print(f"\n--- RANDOMLY INITIALISED ({len(rand_init)}) — what the 40k fine-tune must learn ---")
    for k in rand_init:
        print(f"  {k}   {tuple(msd[k].shape)}")
    n_rand = sum(msd[k].numel() for k in rand_init)
    n_tot = sum(v.numel() for v in msd.values())
    print(f"\nrandomly initialised params {n_rand:,} of {n_tot:,}  ({100*n_rand/n_tot:.4f}%)")

    out_ch = int(model.decode_head.conv_seg.out_channels)
    print(f"decode_head.conv_seg.out_channels = {out_ch} (expect {EXPECTED_NUM_CLASSES})")

    outside = [k for k in rand_init if not k.startswith("decode_head.conv_seg")]
    print("\n" + "=" * 78)
    if outside:
        print(f"STOP — {len(outside)} randomly-initialised key(s) OUTSIDE decode_head.conv_seg:")
        for k in outside:
            print(f"  {k}")
        print("The backbone does not transfer cleanly. The 40k fine-tune premise needs review.")
    else:
        print("CONFIRMED: only decode_head.conv_seg.{weight,bias} are randomly initialised.")
        print("Everything else transfers from the ADE20K checkpoint.")
    print(f"VERDICT: {'FAIL' if outside or out_ch != EXPECTED_NUM_CLASSES else 'PASS'}")
    print("=" * 78)
    return 1 if (outside or out_ch != EXPECTED_NUM_CLASSES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
```
