# B46 — ImageNet backbone provenance: what is actually protected, and what is not

**Type:** provenance audit + documented capture procedure. **Status:** procedure recorded; the
capture itself is pending the first real E1 launch. **Date:** 2026-09-10.
**Implements:** [docs/runpod_environment.md](../docs/runpod_environment.md) §7.
**Does not implement:** verification against a pinned value — governed path, see §4.

---

## 1. Why this was opened

The registered stack pins 78 distributions by full `sha256` (`requirements-runpod.lock`), the base
image by immutable digest, and the container by digest (`sha256:b80b645d…866aaf`). Against that, the
MobileNetV3 ImageNet initialisation arrives over the network at first use and determines the
starting state of every E1 run. It is the only unpinned input of that kind.

## 2. What the load path actually does

| Step | Evidence |
|---|---|
| `--init imagenet` resolves the enum | `src/models/student.py:52-53` — `MobileNet_V3_Large_Weights.IMAGENET1K_V2` |
| Handed to the constructor | `src/models/student.py:55` — `mobilenet_v3_large(weights=weights, …)` |
| Download / cache load | torchvision `WeightsEnum.get_state_dict()` → `torch.hub.load_state_dict_from_url()` |
| Expected filename | `mobilenet_v3_large-5c1a4163.pth`, pinned at `scripts/verify_env.py:26` |
| URL confirmed offline, no download | [imagenet_init_wiring.md](imagenet_init_wiring.md) `:37` (B18d) |
| Cache location on the pod | `/root/.cache/torch/hub/checkpoints/`; `imagenet_cached: False` on a fresh pod ([b42_pod_gpu_validation.md](b42_pod_gpu_validation.md) §6) |

The `-5c1a4163` suffix is `torch.hub`'s hash-prefix convention: `HASH_REGEX = r'-([a-f0-9]*)\.'`
extracts it, and `download_url_to_file()` verifies the SHA256 *begins with* it when
`check_hash=True`.

**`[UNVERIFIED]` — whether torchvision 0.16.0 passes `check_hash=True`.** This was not determined.
torchvision is not installed on a docs-only checkout, and B18d resolved the URL without examining
the download path. It is not asserted either way here. Resolve it on the pod:

    python -c "import inspect, torchvision.models._api as a; print(inspect.getsource(a.WeightsEnum.get_state_dict))"

## 3. Correction — transport integrity, not a provenance pin

This gap has been characterised in earlier sessions, including by this agent, as the weights being
fetched "unverified." On inspection that overstates it, and the accurate framing is:

- **Eight hex characters is 32 bits**, in a stack that pins everything else by full SHA256.
- **The prefix is shipped by torchvision inside its own URL** — it is not a value this repository
  pinned.

So the mechanism confirms that the bytes match what the *installed torchvision* points at. It does
not confirm that they match what E1 was registered against: a different torchvision, or a changed
enum URL, would satisfy it against different weights. **Transport-integrity check, not a provenance
pin.** The gap is real but narrower than "unverified" implies, and it is a *pinning* gap rather than
an *integrity* one — which also changes what closing it requires (§4).

## 4. Verifying on later runs requires a governed-path change — described, not implemented

Any check that compares the cached file against a pinned value before training consumes it must live
in `src/**` or `scripts/**`. Both are governed paths (`AGENTS.md` rule 8), so this session described
the options and stopped.

**Smaller, and recommended — `scripts/verify_env.py`.** It already reports on this exact file at
`:181-188` and is stage 2/5 of `preflight_e1.py`, so a failure there already blocks the real run.
The change is a `sha256` comparison at that site against a new module constant beside
`MOBILENET_CKPT` at `:26`. It extends a gate that exists rather than adding one.

**Larger — `src/models/student.py:_build_backbone`.** Verifying at the load site cannot be bypassed
by skipping preflight, but it puts filesystem hashing inside the model constructor, on a path the
dry-run also traverses.

Neither is implemented. Either requires a session explicitly approved to edit governed paths.

## 5. The sequencing condition — the first seed trains before the pin exists

The hash cannot be pinned until a real run has downloaded the file. The order is therefore fixed:

1. The first real E1 launch populates the cache and runs the §7 capture.
2. A later governed-path session adds the comparison using the captured value.

**The first official seed is therefore trained before any pin exists.** This is recorded so the run
is known to have that property rather than assumed not to. It is a documented condition, not a
blocker: gating E1 on it would cost schedule for no gain, because the value being pinned can only
come from the run itself.

## 6. What this commit changes

| File | Change |
|---|---|
| `docs/runpod_environment.md` | New §7 — the capture procedure, what the URL prefix does and does not protect, and the governed-path boundary. |
| `docs/runpod_environment.md` | "Known residual gaps" — the "No GPU run has been performed" entry is marked superseded and closed, pointing at B42's `VERDICT: GO`. It had been false since 2026-09-09. |
| `reports/b46_imagenet_init_provenance.md` | This report. |

## Provenance / guardrails honored

- No training, download, install, or GPU use. The capture procedure is documented, not executed —
  there is no torch in this environment to execute it with.
- No governed path modified; the verification change is described in §4 and stopped there.
- `docs/reference/reference.pdf` untouched.
