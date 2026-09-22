"""Private CPU RNG stream for the SegNeXt LightHamHead NMF basis draw (M4, B61 §4).

Torch-only, so the frozen-teacher adapter and the evaluator can use it without MMSegmentation.

The pinned MMSeg 1.2.2 `NMF2D._build_bases` draws `torch.rand((B*S, D, R))` from the CPU default
generator on every forward. M4 keeps that algorithm (`rand_init=True`) and isolates its randomness:

  * M4-T  teacher training   — no stream attached; the upstream global-stream draw runs unchanged.
  * M4-V  every evaluation pass — a fresh stream seeded 42 at the start of the pass.
  * M4-KD frozen E2/E3 teacher — one stream seeded 42 at the start of the run, advancing across calls.

`NMFStream.draw(fn)` is the whole seam: it saves the caller's CPU RNG state, installs the private
state, runs `fn` (which must be exactly the upstream basis construction), captures the advanced
private state and restores the caller's state. Nothing else ever runs while the private state is
installed, so NMF is its only consumer.

Seeding goes through a CPU `torch.Generator`, never `torch.manual_seed`, which also reseeds every
CUDA generator. Only the CPU default generator is ever read or written here.
"""

from __future__ import annotations

import hashlib

import torch

M4_NMF_SEED = 42
POLICIES = ("M4-V", "M4-KD")


class NMFStreamError(RuntimeError):
    """A violation of the M4 stream rules (misuse, wrong policy, missing stream)."""


class NMFStream:
    """The private CPU RNG state consumed only by the NMF random-basis draw."""

    def __init__(self, seed: int = M4_NMF_SEED, policy: str = "M4-V"):
        if policy not in POLICIES:
            raise NMFStreamError(f"unknown NMF stream policy {policy!r}; expected one of {POLICIES}")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))          # a CPU generator: no CUDA generator is touched
        self._state = generator.get_state()
        self.seed = int(seed)
        self.policy = policy
        self.draws = 0

    def draw(self, fn):
        """Run `fn()` with the private state installed on the CPU default generator, then restore."""
        caller = torch.get_rng_state()
        torch.set_rng_state(self._state)
        try:
            out = fn()
        finally:
            self._state = torch.get_rng_state()   # capture the advanced private state
            torch.set_rng_state(caller)           # the caller's stream is never consumed
        self.draws += 1
        return out

    def state_sha256(self) -> str:
        return hashlib.sha256(self._state.numpy().tobytes()).hexdigest()

    def describe(self) -> dict:
        return {"policy": self.policy, "seed": self.seed, "draws": self.draws,
                "state_sha256": self.state_sha256(), "seeding": "torch.Generator(cpu).manual_seed"}


def isolated_nmf_modules(model: torch.nn.Module) -> list:
    """Modules that honour an attached NMF stream (duck-typed: no MMSeg import needed)."""
    return [m for m in model.modules() if getattr(type(m), "SUPPORTS_NMF_STREAM", False) is True]


def attach_nmf_stream(model: torch.nn.Module, stream: NMFStream | None) -> int:
    """Attach `stream` to the model's single isolated NMF module (None detaches). Returns the count.

    Exactly one isolated module is required. A model with none — the stock `LightHamHead` — cannot
    honour M4, so it fails closed instead of silently drawing from the global stream.
    """
    mods = isolated_nmf_modules(model)
    if len(mods) != 1:
        raise NMFStreamError(
            f"expected exactly one isolated NMF module (IsolatedNMF2D), found {len(mods)}. The teacher "
            "must be built from the thesis teacher config, whose decode head is "
            "IsolatedNMFLightHamHead; the stock LightHamHead cannot honour M4.")
    for m in mods:
        m.nmf_stream = stream
    return len(mods)
