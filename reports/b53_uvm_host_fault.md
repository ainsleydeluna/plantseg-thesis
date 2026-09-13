# B53 — three dead pods: a host-side UVM fault, and how it was misdiagnosed once

**Type:** incident record + root-cause investigation. **Status:** cause **LOCALISED** to the host's
UVM device nodes by direct measurement on one pod; **workaround FOUND** (different tier/region) and
exercised — E1 seed 42 completed the same day on the replacement pod.
**Date:** 2026-09-11. **Written:** 2026-09-13.
**Companion:** [B52](b52_e1_seed42_completion.md) records the run these pods delayed.
**Source:** four pods rented and terminated on 2026-09-11. The pods are gone; every figure below
comes from terminal captures taken while they were alive.

---

## 1. What happened, and what was actually measured

Three consecutive pods came up with working `nvidia-smi`, a healthy-looking GPU, and CUDA that
would not initialise. The fourth — a deliberately relocated control — worked, and the campaign
moved to it.

The distinction that matters, and which an earlier draft of this record got wrong: **the EIO was
measured on one pod and inferred on two.** The device-open test did not exist yet when pods 1 and 2
were diagnosed — that is §5's subject. The table carries the distinction so a reader scanning it
cannot miss it.

| # | UTC | Host | Tier | Card | Driver | DC | `nvidia-smi` | `/dev/nvidiactl` | `/dev/nvidia-uvm` | `cuInit` | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ~04:26 | `ccd6a2d89c70` | Community *(recollected)* | L40S 46068 MiB sm_89 | 550.163.01 | `<unknown>` | OK | **not tested** | **not tested** | not captured | DEAD |
| 2 | 05:17 | `112d8e145589` | Community *(rate line)* | L40S 46068 MiB sm_89 | 550.163.01 | `<unknown>` | OK | **not tested** | **not tested** | **999** | DEAD |
| 3 | 05:30 | `bd234879f092` | Community *(recollected)* | RTX 4090 24564 MiB sm_89 | **595.80** | `<unknown>` | OK | **open OK** `[MEASURED]` | **EIO** `[MEASURED]` | **999** | DEAD |
| 4 | 05:42 | `f2f3edf4dabf` | **Secure** | RTX 4090 24564 MiB sm_89 | 580.159.04 | **EUR-IS-1** | OK | **open OK** | **open OK** | **0** | REGION OK |
| 5 | 11:59 | `1166491d9cf4` | **Secure** | A40 46068 MiB sm_86 | 570.195.03 | **CA-MTL-1** | OK | **open OK** | **open OK** | **0** | USABLE — ran E1 |

Reading the table honestly:

- **Pod 3 is the only direct measurement of the fault.** `[MEASURED]`
- **Pods 1 and 2 are consistent with the same cause and are not proof of it.** `[INFERRED]` They
  showed the same downstream symptom — `torch.cuda.is_available() == False` with the same
  `CUDA unknown error` warning, and `cuInit` 999 on pod 2 — but nobody opened their UVM nodes.
- **Datacenter is `<unknown>` for all three failures**, not reconstructed. RunPod's console does not
  retain the region for terminated pods, and `RUNPOD_DC_ID` was not captured until pod 4. Recording
  a guess here would have made the support ticket refutable.
- Tier: pod 2 carries its own operator-recorded `$0.79/hr Community` line; pods 1 and 3 are
  Community by operator recollection with **no captured tier field**. Marked accordingly rather
  than asserted.

### 1.1 Pod 3 — the measurement, verbatim

```text
image_digest: sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
NVIDIA GeForce RTX 4090, 24564 MiB, 8.9, 595.80
open /dev/nvidiactl OK
open /dev/nvidia-uvm FAILED OSError [Errno 5] Input/output error: '/dev/nvidia-uvm'
open /dev/nvidia-uvm-tools FAILED OSError [Errno 5] Input/output error: '/dev/nvidia-uvm-tools'
cuInit rc 999
torch.cuda.is_available False
VERDICT: FAIL — not card-specific
```

### 1.2 Pod 1 — the symptom, and the listing that misled

```text
=== device nodes — /dev/nvidia-uvm is the one that matters ===
crw-rw-rw- 1 nobody nogroup 195, 254 Jun 10 06:55 /dev/nvidia-modeset
crw-rw-rw- 1 nobody nogroup 509,   0 Jun 10 06:55 /dev/nvidia-uvm
crw-rw-rw- 1 nobody nogroup 509,   1 Jun 10 06:55 /dev/nvidia-uvm-tools
crw-rw-rw- 1 nobody nogroup 195,   2 Jun 10 06:55 /dev/nvidia2
crw-rw-rw- 1 nobody nogroup 195, 255 Jun 10 06:55 /dev/nvidiactl
=== nvidia-smi ===
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|   0  NVIDIA L40S                    On  |   00000000:12:00.0 Off |                    0 |
| N/A   35C    P8             34W /  350W |       1MiB /  46068MiB |      0%      Default |
=== driver ===
NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.163.01  Tue Apr  8 12:41:17 UTC 2025
=== torch ===
torch         2.1.0+cu121
is_available  False
device_count  1
alloc        FAILED: RuntimeError CUDA unknown error - this may be due to an incorrectly set up
environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the
available devices to be zero.
```

Every UVM node is **present**, with correct permissions and major/minor numbers. `nvidia-smi`
reports a healthy idle GPU. `device_count` even returns 1. The listing looks like a clean bill of
health and is not one — §5.

---

## 2. The diagnosis

**CUDA cannot initialise without the Unified Virtual Memory driver.** `cuInit` opens
`/dev/nvidia-uvm`; when that open returns `EIO`, `cuInit` fails, and the only thing the CUDA runtime
can report upward is `CUDA_ERROR_UNKNOWN` (999). PyTorch renders that as the generic
"CUDA unknown error… e.g. changing env variable `CUDA_VISIBLE_DEVICES` after program start" warning,
which points at a configuration mistake that had not been made:

```text
CUDA_VISIBLE_DEVICES=[<unset>]
NVIDIA_VISIBLE_DEVICES=[<unset>]
```

**`nvidia-smi` working is not evidence that CUDA works.** `nvidia-smi` talks to the GPU through
NVML and `/dev/nvidiactl`; neither path touches UVM. So the healthy-looking `nvidia-smi` output on
all three pods is entirely compatible with a GPU that no CUDA program can use. This is the single
most useful sentence in the record for anyone triaging the next one.

**`EIO` on a device node that exists and has correct permissions is a host-side fault.** It is the
kernel module refusing the open, not a container-visible misconfiguration. Nothing inside the
container — no package, no environment variable, no image change — can repair it. The correct
response is to terminate and redeploy, not to debug.

---

## 3. Hypotheses, and what killed each

| Hypothesis | Test | Outcome |
|---|---|---|
| A broken or outdated GPU driver | Compare failing drivers against working ones | **REFUTED.** Pod 3 failed on **595.80**, which is *newer* than both drivers that worked the same day — 580.159.04 (pod 4) and 570.195.03 (pod 5). A newer driver failing while older ones succeed rules out driver age as the discriminator. |
| A bad card class — L40S specifically | Try a different card class | **REFUTED.** Pods 1 and 2 were L40S; pod 3 was an RTX 4090 and failed identically. Two card classes, same fault. Pod 3's own verdict line said so: `FAIL — not card-specific`. |
| Our container image | Compare image digests across successes and failures | **EXONERATED.** Digest `sha256:b80b645d…866aaf` is byte-identical on all five pods above **and** on the pods that ran successfully on 2026-09-09 and 2026-09-10. The same bytes produced opposite outcomes, so the image cannot be the discriminator. |
| A container-visible misconfiguration | Read the visibility variables | **REFUTED.** Both `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` were unset, exactly as on the working pods. |
| Host pool / tier | §4 | **SUPPORTED, not proven** — see §4's stated limit. |

The digest-identity argument is the strongest single item for a support ticket, because it
preempts the first reply such tickets receive ("rebuild your container"). The image that failed is
the image that worked, to the byte, on three other days.

---

## 4. The tier/region control — and its cost

With the fault localised but its scope unknown, the choice was between "stop deploying and wait"
and "deploy again somewhere else". A fifteen-second control decided it for about a cent.

The control was originally specified as Community in a geographically distant region; that proved
impossible, because **Community does not expose region selection — that is a Secure-tier feature.**
It was therefore deployed on **Secure**, which made it a deliberate two-variable test, with both
answers useful: a pass would scope the fault to the Community pool and/or one region and permit
launching the same day; a failure would show the fault spanning tiers and regions, escalating the
ticket and ending the deploy attempts.

**Result: `REGION OK`.** Pod 4 (`f2f3edf4dabf`, Secure, `EUR-IS-1`, driver 580.159.04) opened all
three device nodes and returned `cuInit rc 0`.

Pod 5 then strengthened it without being designed to. The production pod landed in **`CA-MTL-1`**,
not the Iceland region the control validated, and worked. **Two different Secure regions worked on
the same day that Community failed three times.** That points the discriminator at **tier or host
pool rather than region** — which is the more useful finding, because it is actionable: pay for
Secure.

**Stated limit, because the evidence does not reach further:** three Community failures against two
Secure successes is a small sample with tier and host pool confounded, and the failing pods' regions
are `<unknown>`. This is not a demonstration that Community is unreliable in general. It is
sufficient to justify preferring Secure for the remaining campaign, and it feeds the cost decision
queued as B52 N16 — Secure's premium now has a measured reliability argument behind it, not only a
rate-card difference.

### 4.1 Cost

Four pods rented and terminated. Only pod 2 captured its own rate (`$0.79/hr Community`,
operator-recorded); the control ran at roughly `$0.74/hr` Secure. Rental start and stop times were
not recorded — only first and last observation — so these are **observation windows, not billed
durations**:

| Pod | Observed window | Approximate |
|---|---|---|
| 1 `ccd6a2d89c70` | 04:04 → 04:27 | ~25 min, plus unobserved setup and teardown |
| 2 `112d8e145589` | 05:17 → 05:23 | ~6 min |
| 3 `bd234879f092` | 05:30 → 05:32 | ~2 min |
| 4 `f2f3edf4dabf` (control) | 05:42, terminated immediately | ~1 min |

Aggregate is well under **$1**. **Credit was requested on all three failures — none ever became
usable.** The control is not part of the credit request: it worked, and it was worth its cent.

A support ticket was drafted leading with the `EIO` rather than the 999 — CUDA_ERROR_UNKNOWN is
what a thousand tickets a week say and routes to a template reply, whereas a device node returning
`EIO` is specific and actionable. It carries all pod IDs, cards, drivers, `<unknown>` datacenters,
the measured/inferred distinction, and the digest-identity argument. **The operator sends it; this
report does not.**

---

## 5. The methodological note — why this took an hour and two extra pods

**On pod 1 the UVM hypothesis was raised, and then discarded on bad evidence.** The device nodes
were **listed** (`ls -la /dev/nvidia*`), seen to be present with correct permissions, and the
hypothesis was written off as wrong. It was correct. **Existence is not usability.** One
`os.open()` would have settled it an hour and two pod rentals earlier.

That is one instance of a pattern this repository tracks. **The ledger below records what is
evidenced and marks what is not.** Two confirmed entries with three marked gaps beats five entries
where three are guesses — reconstructing a pattern ledger from recollection defeats its purpose,
and would itself be an instance of the pattern.

| # | Instance | Status |
|---|---|---|
| 1 | Referenced in a prior session | **NOT RECONSTRUCTABLE** from artifacts available here |
| 2 | Referenced in a prior session | **NOT RECONSTRUCTABLE** from artifacts available here |
| 3 | Pod 1's UVM hypothesis discarded because the nodes were **listed**, not opened (§1.2) | **EVIDENCED** — the `ls -la /dev/nvidia*` output and the later pod-3 `EIO` measurement are both in this record |
| 4 | *Candidate:* the first support-ticket draft stated the `EIO` was measured on all three pods, when it was measured on one and inferred on two | **INFERRED** from a conversational reference; the draft itself is not in the artifacts, so this is not confirmed |
| 5 | The temptation to read the absence of any further nondeterminism warning as exonerating the backbone forward, where the deduplication regime is unverified — [B52 §6.4](b52_e1_seed42_completion.md) | **EVIDENCED** |

Instances 1, 2 and 4 are gaps, not entries. **Queued non-governed as B52 N20:** recover them from
repository history at the commits where those corrections landed, rather than from recollection.

What the two evidenced instances share is the shape, and what distinguishes 3 from 5 is *when* it
was caught. Instance 3 cost an hour and two pod rentals because it was acted on. Instance 5 was
intercepted before publication. Candidate 4, if confirmed, would be the same interception: a
support engineer who checks an overstated claim and finds it overstated discounts the entire
report, so weaker-sounding and survivable beats strong and false.

**The rule this produced now lives in [B52 §11.4](b52_e1_seed42_completion.md) as queue item N17:**

> Before recording an absence, state which instrument was used and why its coverage includes the
> adjudicated string. If coverage cannot be established, record "not determinable from X" rather
> than "absent."

Instance 3 is that rule's device-node shape: `ls` establishes existence, and existence was never the
adjudicated property.

### 5.1 A second error in this investigation: a credential printed to the terminal

The control's identity block was written as `env | grep -i "^RUNPOD" | sort`, which printed
**`RUNPOD_API_KEY` in full** to the terminal and into the session record. That was an authoring
error in this investigation, not a RunPod fault.

- **Remediated immediately** by replacing the pattern with an explicit variable-name loop. The
  production pod's step 0 lists exactly seven variables by name — `RUNPOD_POD_ID`, `RUNPOD_DC_ID`,
  `RUNPOD_GPU_NAME`, `RUNPOD_GPU_COUNT`, `RUNPOD_CPU_COUNT`, `RUNPOD_MEM_GB`, `RUNPOD_VOLUME_ID` —
  and cannot print a secret it was not asked for.
- **Blast radius is bounded — that is not the same as resolved.** The operator's RunPod account
  API-keys page was empty, so the value was **pod-injected**, not an account-level credential. That
  bounds *what* it could reach. It does not make the exposure hypothetical: **the key was live for
  the pod's lifetime**, and the exposure window closed when the pod terminated — by termination,
  not by any action taken against the key.
- **The value is not reproduced in this report**, and must not be quoted from the session record
  into any artifact.

**This incident is the justification for queue item N8** ([B52 §11.2](b52_e1_seed42_completion.md)):
capture RunPod environment by explicit variable name, never `env | grep`. The two resolve to each
other — N8 exists because of §5.1, and §5.1 is closed by N8 — in the same way §6 records the
device-open gate as N7's justification.

---

## 6. What changed as a result

**The step-0 gate now opens the device nodes.** Before this incident, step 0 called
`torch.cuda.is_available()` and stopped there. That call answers *whether* CUDA is broken; it cannot
say *why*, and it cannot distinguish an unfixable host fault from something worth debugging. The
current gate — exercised on pods 4 and 5, and reproduced in [B52 §3](b52_e1_seed42_completion.md) —
opens `/dev/nvidiactl`, `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools`, calls `cuInit` directly
through `libcuda.so.1`, and only then checks torch:

```python
for node in ('/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'):
    try:
        fd = os.open(node, os.O_RDWR); os.close(fd); print('open', node, 'OK')
    except Exception as e:
        print('open', node, 'FAILED', type(e).__name__, e)
print('cuInit rc      ', ctypes.CDLL('libcuda.so.1').cuInit(0))
```

It costs one second and it renders this class of fault unmistakable at pod start, before any clone,
download, or pre-flight. **That is the best-justified item in the non-governed queue** — B52 **N7**,
which promotes it from an ad-hoc block into the runbook permanently, together with **N8**'s safe
environment capture.

**Operational rule, for the runbook:** a pod whose step 0 reports `EIO` on a UVM node is dead.
Terminate it. Do not clone, do not download, do not run pre-flight, do not debug — nothing inside
the container can fix a host-side fault, and every minute spent is billed.

**What this record does not settle:** why those hosts' UVM modules were faulted. That is RunPod's to
answer, and the ticket asks. Nothing in this repository depends on the answer — the workaround is
established and the campaign moved on the same day.
