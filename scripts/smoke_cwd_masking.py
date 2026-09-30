#!/usr/bin/env python3
"""CWD masking smoke (lane L-KD-HARDEN item 9; A1 findings CWD-1, CWD-5). Synthetic CPU tensors only.

`src.training.losses.cwd_channelwise_kl` zeroes ONLY the masked positions (`torch.where`); a non-finite
value at a VALID position propagates, so the trainer's AM-7 (a) check sees it instead of a scrubbed
number. At 44c05dc the function scrubbed every non-finite value (`nan_to_num`), masked or not.

  V1  NaN at a valid position (student map, teacher map) -> NaN loss
  V2  student -inf at a valid position with a non-zero teacher probability there -> +inf loss (NaN
      where that probability is 0); teacher -inf -> NaN loss (both AM-7 (a) stops)
  U1  the unmasked path (valid_mask=None) is not scrubbed either: a NaN gives a NaN loss
  M1  Table 3.1 gate "CWD unchanged when padded values altered", on the production function: NaN,
      +1e4, -1e4, +inf or -inf written at masked positions of the student or the teacher map -> loss
      and student gradient bitwise unchanged
  E1  a sample with no valid location contributes exactly 0 (alone: loss 0.0 and zero gradient; next
      to a valid sample: the batch mean of 0 and the valid sample's value)
  R0  the embedded 44c05dc reference matches that commit's function body (its statements, compared
      without the docstring; SKIP, never PASS, when the commit or git is unavailable, e.g. in a
      `git archive` export)
  R1  magnitude 1e33 at valid positions -> finite, and bitwise equal to the 44c05dc implementation
  R2  random finite inputs (feature map C=320 with channels_norm=320, logit map C=116, masked and
      unmasked paths, several seeds) -> loss and student gradient bitwise equal to 44c05dc
  T1  Table 3.1 gate "CWD channel-wise softmax sums to 1 over valid locations", by equivalence on the
      production function: the loss equals a valid-only reference that compacts each sample to its
      valid locations before the spatial softmax (allclose)
  T2  additional: masked positions receive exactly zero gradient; the valid positions' gradient is
      non-zero in aggregate
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.training.losses import cwd_channelwise_kl  # noqa: E402

REF_COMMIT = "44c05dc"
results: list[tuple[str, bool | None, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    results.append((name, ok, detail))


def bits(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().numpy().tobytes()


# ---------------------------------------------------------------------- 44c05dc reference (verbatim)
def _cwd_44c05dc(student_map: torch.Tensor, teacher_map: torch.Tensor,
                 valid_mask: torch.Tensor | None = None, T: float = 4.0,
                 channels_norm: int | None = None) -> torch.Tensor:
    if student_map.shape != teacher_map.shape:
        raise ValueError(f"CWD shape mismatch: student {tuple(student_map.shape)} != teacher "
                         f"{tuple(teacher_map.shape)} (project/resample before calling)")
    b, c, h, w = student_map.shape
    s = student_map.reshape(b, c, h * w) / T
    t = teacher_map.reshape(b, c, h * w) / T

    if valid_mask is not None:
        if tuple(valid_mask.shape) != (b, h, w):
            raise ValueError(f"CWD validity mask {tuple(valid_mask.shape)} does not match map grid "
                             f"{(b, h, w)}")
        m = valid_mask.reshape(b, 1, h * w)                        # [B,1,HW] bool
        neg = torch.finfo(s.dtype).min
        s = s.masked_fill(~m, neg)
        t = t.masked_fill(~m, neg)
        has_valid = m.any(dim=-1).squeeze(1)                       # [B]
    else:
        m = None
        has_valid = torch.ones(b, dtype=torch.bool, device=student_map.device)

    s_logp = F.log_softmax(s, dim=-1)
    t_logp = F.log_softmax(t, dim=-1)
    kl = t_logp.exp() * (t_logp - s_logp)                          # [B,C,HW]
    if m is not None:
        kl = kl * m                                                # drop the all -inf columns
    kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)     # samples with no valid location

    cnorm = c if channels_norm is None else channels_norm
    per_sample = kl.sum(dim=(1, 2)) * (T * T) / cnorm              # [B]
    per_sample = per_sample * has_valid.to(per_sample.dtype)
    return per_sample.mean()


def _function_source(text: str, name: str) -> str | None:
    """The function's body statements as source text, without its docstring (if it has one)."""
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            stmts = list(node.body)
            if stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant) \
                    and isinstance(stmts[0].value.value, str):
                stmts = stmts[1:]
            return "\n".join(ast.get_source_segment(text, n) or "" for n in stmts)
    return None


def test_reference_provenance() -> None:
    try:
        old = subprocess.run(["git", "-C", str(REPO), "show", f"{REF_COMMIT}:src/training/losses.py"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        check("R0_reference_is_44c05dc_function", None, f"SKIP: git unavailable ({type(e).__name__})")
        return
    if old.returncode != 0:
        check("R0_reference_is_44c05dc_function", None, f"SKIP: {REF_COMMIT} not readable here")
        return
    want = _function_source(old.stdout, "cwd_channelwise_kl")
    got = _function_source(Path(__file__).read_text(encoding="utf-8"), "_cwd_44c05dc")
    check("R0_reference_is_44c05dc_function", want is not None and got is not None and want == got,
          "embedded copy vs git show 44c05dc:src/training/losses.py (function body)")


# --------------------------------------------------------------------------------------- inputs
def maps(seed: int, b: int = 2, c: int = 116, h: int = 8, w: int = 8, scale: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, c, h, w, generator=g) * scale
    t = torch.randn(b, c, h, w, generator=g) * scale
    m = torch.ones(b, h, w, dtype=torch.bool)
    m[0, :2, :] = False                                   # a padded band in sample 0
    m[1, :, -3:] = False                                  # a padded strip in sample 1
    return s, t, m


def loss_and_grad(fn, s, t, m, **kw):
    s = s.clone().requires_grad_(True)
    loss = fn(s, t, m, **kw)
    loss.backward()
    return loss.detach(), s.grad.detach()


# ---------------------------------------------------------------------------------- non-finite
def test_valid_nonfinite() -> None:
    s, t, m = maps(1)
    b, h, w = 1, 2, 3                                      # a VALID position of sample 1
    check("V0_probe_position_is_valid", bool(m[b, h, w]))
    for label, which, value, want in (("student_nan", "s", float("nan"), "nan"),
                                      ("teacher_nan", "t", float("nan"), "nan"),
                                      ("student_minus_inf", "s", float("-inf"), "+inf"),
                                      ("teacher_minus_inf", "t", float("-inf"), "nan")):
        s2, t2 = s.clone(), t.clone()
        (s2 if which == "s" else t2)[b, 5, h, w] = value
        loss = cwd_channelwise_kl(s2, t2, m, T=4.0)
        got = "nan" if torch.isnan(loss) else ("+inf" if loss == float("inf") else
                                               ("-inf" if loss == float("-inf") else "finite"))
        tag = "V1" if value != value else "V2"
        check(f"{tag}_{label}_at_valid_position_gives_{want}", got == want, f"loss={loss.item()!r}")
        old = _cwd_44c05dc(s2, t2, m, T=4.0)
        check(f"{tag}_{label}_was_scrubbed_at_44c05dc", bool(torch.isfinite(old)),
              f"44c05dc loss={old.item()!r} (documents the fixed behaviour)")
    s2 = s.clone()
    s2[0, 3, 0, 0] = float("nan")
    check("U1_unmasked_path_not_scrubbed", bool(torch.isnan(cwd_channelwise_kl(s2, t, None, T=4.0))))


def test_masked_perturbations() -> None:
    for c, norm in ((116, None), (320, 320)):
        s, t, m = maps(2, c=c)
        kw = {"T": 4.0, "channels_norm": norm}
        base_l, base_g = loss_and_grad(cwd_channelwise_kl, s, t, m, **kw)
        invalid = ~m.unsqueeze(1).expand_as(s)                     # [B,C,h,w]
        for label, value in (("nan", float("nan")), ("plus_1e4", 1e4), ("minus_1e4", -1e4),
                             ("plus_inf", float("inf")), ("minus_inf", float("-inf"))):
            for which in ("student", "teacher"):
                s2, t2 = s.clone(), t.clone()
                (s2 if which == "student" else t2)[invalid] = value
                l2, g2 = loss_and_grad(cwd_channelwise_kl, s2, t2, m, **kw)
                check(f"M1_C{c}_{which}_{label}_at_masked_positions_bitwise_unchanged",
                      bits(l2) == bits(base_l) and bits(g2) == bits(base_g),
                      f"loss {l2.item()!r} vs {base_l.item()!r}")


def test_empty_sample() -> None:
    s, t, m = maps(3)
    none = torch.zeros_like(m)
    loss, grad = loss_and_grad(cwd_channelwise_kl, s, t, none, T=4.0)
    check("E1_all_empty_batch_is_exactly_zero", loss.item() == 0.0 and not bool(torch.signbit(loss))
          and torch.equal(grad, torch.zeros_like(grad)), f"loss={loss.item()!r}")
    mixed = m.clone()
    mixed[0] = False
    both = cwd_channelwise_kl(s, t, mixed, T=4.0)
    alone = cwd_channelwise_kl(s[1:], t[1:], mixed[1:], T=4.0)
    check("E1_empty_sample_contributes_zero_to_the_mean", torch.allclose(both, alone / 2, rtol=1e-6,
                                                                        atol=0.0),
          f"batch={both.item()!r} valid_only/2={(alone / 2).item()!r}")


# ------------------------------------------------------------------------------ vs 44c05dc
def test_against_reference() -> None:
    s, t, m = maps(4, scale=1e33)
    new = cwd_channelwise_kl(s, t, m, T=4.0)
    old = _cwd_44c05dc(s, t, m, T=4.0)
    check("R1_magnitude_1e33_finite_and_equal_to_44c05dc", bool(torch.isfinite(new)) and bits(new) == bits(old),
          f"new={new.item()!r} old={old.item()!r}")
    n_ok, n = 0, 0
    for seed in range(6):
        for c, norm in ((116, None), (320, 320)):
            for masked in (True, False):
                s, t, m = maps(100 + seed, c=c)
                mm = m if masked else None
                ln, gn = loss_and_grad(cwd_channelwise_kl, s, t, mm, T=4.0, channels_norm=norm)
                lo, go = loss_and_grad(_cwd_44c05dc, s, t, mm, T=4.0, channels_norm=norm)
                n += 1
                n_ok += int(bits(ln) == bits(lo) and bits(gn) == bits(go))
    check("R2_finite_inputs_loss_and_grad_bitwise_equal_to_44c05dc", n_ok == n, f"{n_ok}/{n} cases")


# ------------------------------------------------------------------------------- Table 3.1
def valid_only_reference(s, t, m, T=4.0, channels_norm=None):
    """Compact each sample to its valid locations, then spatial softmax and KL over those only."""
    b, c, h, w = s.shape
    total = []
    for i in range(b):
        idx = m[i].reshape(-1).nonzero().squeeze(1)
        if idx.numel() == 0:
            total.append(s.new_zeros(()))
            continue
        sv = s[i].reshape(c, h * w)[:, idx] / T
        tv = t[i].reshape(c, h * w)[:, idx] / T
        tp = F.softmax(tv, dim=-1)
        kl = (tp * (F.log_softmax(tv, dim=-1) - F.log_softmax(sv, dim=-1))).sum()
        total.append(kl * (T * T) / (c if channels_norm is None else channels_norm))
    return torch.stack(total).mean()


def test_table31_gates() -> None:
    for c, norm in ((116, None), (320, 320)):
        s, t, m = maps(7, c=c)
        m[1, 0, 0] = False
        prod = cwd_channelwise_kl(s, t, m, T=4.0, channels_norm=norm)
        ref = valid_only_reference(s, t, m, T=4.0, channels_norm=norm)
        check(f"T1_C{c}_production_equals_valid_only_reference",
              torch.allclose(prod, ref, rtol=1e-5, atol=1e-7), f"prod={prod.item()!r} ref={ref.item()!r}")
        _, grad = loss_and_grad(cwd_channelwise_kl, s, t, m, T=4.0, channels_norm=norm)
        invalid = ~m.unsqueeze(1).expand_as(s)
        check(f"T2_C{c}_masked_positions_get_exactly_zero_gradient",
              torch.equal(grad[invalid], torch.zeros_like(grad[invalid]))
              and float(grad[~invalid].abs().sum()) > 0.0)


def main() -> int:
    print("=" * 78)
    print("CWD MASKING SMOKE (L-KD-HARDEN item 9) - synthetic CPU tensors only")
    print(f"torch {torch.__version__}")
    print("=" * 78)
    for fn in (test_reference_provenance, test_valid_nonfinite, test_masked_perturbations,
               test_empty_sample, test_against_reference, test_table31_gates):
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        state = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {name:66}: {state}{('  ' + detail) if detail and ok is not True else ''}")
    exercised = [ok for _, ok, _ in results if ok is not None]
    passed = sum(1 for ok in exercised if ok)
    skipped = len(results) - len(exercised)
    print(f"\nRESULT: {'PASS' if passed == len(exercised) and exercised else 'FAIL'} "
          f"({passed}/{len(exercised)}, {skipped} skipped)")
    return 0 if passed == len(exercised) and exercised else 1


if __name__ == "__main__":
    raise SystemExit(main())
