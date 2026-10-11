"""AM-18 item 3(b)'s arm W, run stage E3-flip: flip-averaged KD targets behind the KD arm seam (lane L-KD-WRAP,
commit C1; ARMS plan decision (vi); Tier F audit F28 and F29).

W is E3's recipe of record with its Logit-KD and CWD logit-map targets replaced: "the teacher runs on the crop and
then on its horizontal flip, and the second set of logits is flipped back ... probabilities are averaged, each term
in its own softmax at T = 4, never logits; the CWD feature-map term uses the unflipped view's Stage 3; the M4-KD
stream advances once per head forward; the student sees only the original crop" (AM-18 item 3(b)). Its registry
entry, configs/kd_arms/e3_flip.json, names FlipTargets. The trainer reaches this module only through the
TargetBuilder interface that src/training/kd_arms.py states (frozen: audit ruling A4) and checks structurally.

  call 1   the trainer's own teacher call on the step's input, inside its step-1 input probe, unchanged: E3's call.
           Its logits are view 1; its Stage 3 stays the CWD feature-map term's input (the trainer's own term).
  call 2   teacher(torch.flip(model_input, dims=[-1])), made by build(), which the trainer calls after the probe's
           hooks are removed, so the probe compares the student's input with call 1 only. Its logits, flipped back
           along the width (an exact index permutation onto the crop's grid), are view 2; its Stage 3 is discarded.
  stream   each call is one head forward and one M4-KD draw (src/distill/nmf_stream.py): step t takes draw 2t-1
           for the crop and draw 2t for the flip, where E3 takes draw t. build() refuses a call 2 that does not
           advance the stream by exactly one draw or that changes the caller's CPU RNG state.
  targets  per term, the two views' probabilities are averaged in that term's own softmax at the T the trainer
           passes (T_logit = T_cwd = 4); logits are never averaged.
           Logit KD   p = (softmax_c(z1/T) + softmax_c(z2/T)) / 2 at each cell, in losses.logit_kd_kl's arithmetic
                      with p in place of its t_prob: KL summed over the classes, times T^2, mean over the valid
                      cells. With z2 == z1, p is t_prob bit for bit and the loss is E3's bit for bit.
           CWD logit  per channel, q = (q1 + q2) / 2, with q_v the softmax over the h*w cells of z_v/T and the
                      invalid cells at the dtype's most negative finite value (cwd_channelwise_kl's masking); the
                      loss sums xlogy(q, q) - q log_softmax(s/T) over the valid cells (plan (vi)'s form) and takes
                      cwd_channelwise_kl's normalisation (T^2 / C with C = 116, zero for a sample with no valid
                      cell, batch mean). With z2 == z1 it is E3's term in exact arithmetic (float64 agrees to about
                      1e-12 relative) but not bit for bit: W takes xlogy of a softmax where E3 uses a log_softmax
                      and its exp. In float32 the two round differently in the cancellation between the sums of
                      q log q and q log_softmax(s/T), a gap that grows as the term shrinks (about 1e-7 relative at
                      a term value of 0.5, 1e-5 at 5e-3, 1e-3 at 5e-5), of the order of E3's own float32 error
                      (review finding FT-2).
           The validity mask is the trainer's, on the crop's grid: view 2 is flipped back, the mask never is.
  non-finite  a NaN or +inf teacher logit in either view gives a NaN loss, as in E3 (both terms at a valid cell, the
           Logit-KD term at a masked one), so AM-7 (a) stops a real run. A teacher -inf at a valid cell does not
           (review finding FT-1, an open item for a ruling): that view's softmax gives the cell probability 0, so
           the CWD logit-map term stays finite where E3's cwd_channelwise_kl gives NaN (L-KD-HARDEN item 9), and
           W's run goes on. `finite` is false then, and the AM-7 abort record reads it whenever another rule stops
           the run.
  K        1 only. DL-84 makes item 6(a)'s K-draw correction unavailable and DL-96 adds no K8 arm, so a builder
           whose K is not 1 is refused.

The student forwards once per step, on the original crop, in the trainer: no builder receives the student.
src/training/losses.py is unchanged, and no run without an arm imports this module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

K = 1                       # NMF draws per view and step (DL-84, DL-96)
VIEWS = 2                   # the crop and its horizontal flip
WIDTH = -1                  # the width axis of a [B, C, H, W] map: a horizontal flip reverses it


class FlipTargetsError(RuntimeError):
    """A W invariant that does not hold: a builder whose K is not 1 or whose views are not 2, or a call 2 that did
    not advance the M4-KD stream by exactly one draw, changed the caller's CPU RNG state or returned logits on
    another grid than call 1's. The run stops there: its targets would not be W's."""


def flip_input(x: torch.Tensor) -> torch.Tensor:
    """Call 2's input: the step's input [B, 3, H, W] flipped horizontally."""
    return torch.flip(x, dims=[WIDTH])


def flip_back(logits: torch.Tensor) -> torch.Tensor:
    """Call 2's logits [B, C, h, w] flipped back along the width, onto the crop's grid."""
    return torch.flip(logits, dims=[WIDTH])


def class_probs(logits: torch.Tensor, T: float) -> torch.Tensor:
    """The Logit-KD term's softmax: over the classes at each cell, at temperature T (logit_kd_kl's t_prob)."""
    return F.softmax(logits / T, dim=1)


def spatial_probs(logits: torch.Tensor, m: torch.Tensor, T: float) -> torch.Tensor:
    """The CWD logit-map term's softmax: per channel over the h*w cells at temperature T, the cells where `m`
    ([B, 1, h*w], bool) is false at the dtype's most negative finite value, as cwd_channelwise_kl masks them."""
    b, c, h, w = logits.shape
    t = logits.reshape(b, c, h * w) / T
    t = t.masked_fill(~m, torch.finfo(t.dtype).min)
    return F.softmax(t, dim=-1)


def average(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """The two views' probabilities, averaged; (p + p) * 0.5 is p bit for bit."""
    return (p1 + p2) * 0.5


class FlipViewTargets:
    """One step's W targets (kd_arms' Targets): the two views' logits on the crop's grid.

    view1_logits is call 1's logits, the trainer's teacher_out.logits itself; view2_logits is call 2's, flipped back.
    Besides the interface (logit_kd_loss, cwd_logit_loss, finite) both are read-only diagnostics."""

    def __init__(self, view1_logits: torch.Tensor, view2_logits: torch.Tensor):
        if tuple(view1_logits.shape) != tuple(view2_logits.shape):
            raise FlipTargetsError(f"[flip_grid] the views' logits are on different grids: "
                                   f"{tuple(view1_logits.shape)} and {tuple(view2_logits.shape)}")
        self.view1_logits = view1_logits
        self.view2_logits = view2_logits

    @property
    def finite(self) -> bool:
        """True when both views' logits are finite: the AM-7 abort record's teacher_finite requires it."""
        return bool(torch.isfinite(self.view1_logits).all()) and bool(torch.isfinite(self.view2_logits).all())

    def _check(self, head_logits: torch.Tensor, valid: torch.Tensor) -> None:
        """The student's head map on the views' grid, and the validity mask [B, h, w] on that grid too (the shape
        check of logit_kd_kl and cwd_channelwise_kl: a mismatch fails loudly instead of broadcasting)."""
        if tuple(head_logits.shape) != tuple(self.view1_logits.shape):
            raise ValueError(f"head logits {tuple(head_logits.shape)} are not on the views' grid "
                             f"{tuple(self.view1_logits.shape)}")
        expected = (head_logits.shape[0],) + tuple(head_logits.shape[-2:])
        if tuple(valid.shape) != expected:
            raise ValueError(f"valid_mask shape {tuple(valid.shape)} does not match the logits grid {expected}")

    def logit_kd_loss(self, head_logits: torch.Tensor, valid: torch.Tensor, T: float) -> torch.Tensor:
        """The Logit-KD term: logit_kd_kl's arithmetic, the two views' averaged class probabilities as target."""
        self._check(head_logits, valid)
        v = valid.float()                                          # [B,h,w]
        s_logp = F.log_softmax(head_logits / T, dim=1)
        p_bar = average(class_probs(self.view1_logits, T), class_probs(self.view2_logits, T))
        kl = F.kl_div(s_logp, p_bar, reduction="none").sum(dim=1)  # [B,h,w]
        kl = kl * (T * T)                                          # T^2 absorbed into the term
        denom = v.sum().clamp_min(1.0)
        return (kl * v).sum() / denom

    def cwd_logit_loss(self, head_logits: torch.Tensor, valid: torch.Tensor, T: float) -> torch.Tensor:
        """The CWD logit-map term: per channel, the two views' averaged spatial probabilities as target, with
        cwd_channelwise_kl's masking and normalisation (C = the map's own 116 channels)."""
        self._check(head_logits, valid)
        b, c, h, w = head_logits.shape
        m = valid.reshape(b, 1, h * w)                             # [B,1,HW] bool
        s = head_logits.reshape(b, c, h * w) / T
        s = s.masked_fill(~m, torch.finfo(s.dtype).min)
        s_logp = F.log_softmax(s, dim=-1)
        q_bar = average(spatial_probs(self.view1_logits, m, T), spatial_probs(self.view2_logits, m, T))
        kl = torch.xlogy(q_bar, q_bar) - q_bar * s_logp            # [B,C,HW]; 0 log 0 is 0
        kl = torch.where(m, kl, kl.new_zeros(()))                  # zero the masked cells only
        has_valid = m.any(dim=-1).squeeze(1)                       # [B]
        per_sample = kl.sum(dim=(1, 2)) * (T * T) / c
        per_sample = per_sample * has_valid.to(per_sample.dtype)
        return per_sample.mean()


class FlipTargets:
    """Arm W's target builder, kd_arms' TargetBuilder: K = 1, VIEWS = 2, build(teacher, model_input, teacher_out).

    The trainer makes one per run, with no arguments, and calls build() once per training step after the step-1
    input probe's try/finally, with its FrozenTeacher, the step's input (the one augmented batch the student
    received) and the TeacherOutput of its own call on that input."""

    K = K
    VIEWS = VIEWS

    def __init__(self):
        cls = type(self)
        for attr, want in (("K", K), ("VIEWS", VIEWS)):
            got = getattr(cls, attr, None)
            if not (isinstance(got, int) and not isinstance(got, bool) and got == want):
                raise FlipTargetsError(f"[flip_{attr.lower()}] {cls.__name__}.{attr} is {got!r}, not {want}: W builds "
                                       "single-draw targets on two views (DL-84: the K-draw correction is "
                                       "unavailable; DL-96: no K8 arm)")

    def build(self, teacher, model_input: torch.Tensor, teacher_out) -> FlipViewTargets:
        """Call 2 on the horizontal flip of the step's input, its logits flipped back, and the step's targets. A
        teacher with no NMF stream (a dry run's explicit MockTeacher; a real run refuses one) has no draw to count."""
        before = teacher.nmf_stream_state()
        rng = torch.get_rng_state()
        out2 = teacher(flip_input(model_input))
        rng_unchanged = torch.equal(rng, torch.get_rng_state())
        after = teacher.nmf_stream_state()
        if not rng_unchanged:
            raise FlipTargetsError("[flip_cpu_rng] call 2 changed the caller's CPU RNG state: only the M4-KD stream "
                                   "may feed the teacher's NMF draw (src/distill/nmf_stream.py)")
        if (before is None) != (after is None):
            raise FlipTargetsError(f"[flip_nmf_draws] the teacher's NMF stream changed during call 2 (before "
                                   f"{before}, after {after})")
        if before is not None and not (after["draws"] == before["draws"] + 1 and after["policy"] == before["policy"]
                                       and after["seed"] == before["seed"]):
            raise FlipTargetsError(f"[flip_nmf_draws] call 2 took {after['draws'] - before['draws']} draws of the "
                                   f"{before['policy']} stream, not 1: the stream advances once per head forward")
        view2 = flip_back(out2.logits)
        if tuple(view2.shape) != tuple(teacher_out.logits.shape):
            raise FlipTargetsError(f"[flip_grid] call 2's logits {tuple(view2.shape)} are not on call 1's grid "
                                   f"{tuple(teacher_out.logits.shape)}")
        return FlipViewTargets(teacher_out.logits, view2)
