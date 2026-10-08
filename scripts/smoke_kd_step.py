#!/usr/bin/env python3
"""KD step smoke for the pod (lane 5, K2; plan P6; PL-33 to PL-38; PL-39 cases 37 to 41; PL-40 mutations 18, 19).

Usage:
  python -B scripts/smoke_kd_step.py run --expect-head <PIN> --teacher-ckpt <T> --teacher-ckpt-sha256 <sha>
         --old-root <export of 73fd4d7> --evidence <E>
      On the first pod of a wave (runbook KD.6), after KD.1 to KD.5, in KD.1's shell, from the checkout of the pin
      (PL-28: HEAD is the pin, so --expect-head takes the pin's full commit). Before any child runs it refuses
      (RESULT: REFUSED, exit 2, no STOP: correct and rerun) another --expect-head, an environment not as the
      environment row requires (the KD digest exported, PLANTSEG_GIT_COMMIT unset, the governed paths clean),
      teacher arguments other than the teacher of record (its sha256, the file name as typed and of the file it
      resolves to, a file; the children hash it) and the other wrong inputs. Each check runs in a fresh child
      process (E-17): `_stage S` (25 steps at batch 16, for E2, E3, A, F and G), `_repeat` (E3 again),
      `_cross --code-root <old> S` (E2 and E3, 20 steps, with 73fd4d7's code), `_ce_rate`, and the 11 refusal
      invocations of train_distill.py against a
      synthetic full-count TRAIN/VAL root of empty files with a throwaway teacher file (PL-36). The children call
      the real train_distill.run() in dry mode on CUDA with the real teacher and the real TRAIN/VAL data
      (pretrained=False, num_workers=12, val_interval=4000); no checkpoint is written (save_distill_checkpoint
      and write_best_pointer are stubbed) and TEST is never read. The five `_stage` children validate on the full
      VAL set at step 25 (max_val_batches=None): that is the row check-run-meta --allow-smoke accepts, and E2's
      is the full VAL pass; `_repeat` and `_cross` validate on one batch. Writes <E>/kd_smoke_<UTC>/ (the
      children's records and logs) and <E>/kd_smoke_<UTC>.json; prints one row per check and RESULT: PASS,
      FLAG (a ruling is needed) or STOP.
  python -B scripts/smoke_kd_step.py --selfcheck
      In the cloud, CPU only; never a pod PASS. The evaluator on synthetic results, each criterion in its PASS
      and STOP (or FLAG) form; the 11 refusal children on a synthetic full-count root with a throwaway teacher
      file and --device cuda; --old-root refused for another train_distill.py, and an evidence folder inside the
      checkout, LOG_EVERY, --expect-head, a path SL-1 refuses, the environment and the teacher arguments before any
      child; the driver end to end with its children stubbed
      (the gate's workers handed to each child that reads data, the trainer and refusal children; the warnings of
      a trainer child's log, of check-run-meta and of the refusal children listed); a stub-teacher `_stage` child
      (CPU, batch 2, 3 steps, synthetic batches). The driver's scoped status is stubbed, so the selfcheck runs no
      `git status` on the checkout (the trainer children only read HEAD).

Criteria (plan P6; PL-33 to PL-38): children; environment; determinism; TF32; teacher identity; CUDA order; data
root; shapes at batch 16; interpolation; the NMF stream; step 1; 25 steps; the E3 repeat; cross-commit; CE
nondeterminism; peak memory; throughput; the full VAL pass; schema; gate refusals; warnings; losses. The values of
record are imported from scripts/preflight_distill.py (PL-11).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEACHER_CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"

# --------------------------------------------------------------------------------------- thresholds
MEM_PASS_BYTES = 40 * 10**9            # PL-33: max_memory_reserved <= 40e9 B PASS
MEM_STOP_BYTES = 44 * 10**9            # above 44e9 B STOP; in between FLAG for a ruling
CROSS_REL_TOL = 1e-6                   # PL-34: |delta loss| <= 1e-6 relative per step
E3_S_PER_ITER_FLAG = 2.0               # PL-38
PL38_MESSAGE = ("recompute the launch plan under AM-19 items 1(e), 2 and 3(b) before any GO; the dates do not "
                "move")
OLD_TRAIN_DISTILL_SHA256 = "2f134189ef10a14d84278019b0eea656c5b3a9949e47cda05aa3a324efb8b5e2"   # 73fd4d7 (PL-34)
REFUSAL_LIMIT_S = 300                  # PL-36
ALLOWED_NONDET_WARNING = "nll_loss2d_forward_out_cuda_template"     # PL-35
OLD_CE_TWICE_CHECKS = ("supervised_never_ramped",)   # 73fd4d7's checks that evaluate the CUDA CE twice, bitwise
STAGE_KEYS = ("e2", "e3", "a", "f", "g")
STEPS, CROSS_STEPS, BATCH = 25, 20, 16
THROUGHPUT_STEPS = (6, 25)             # PL-37: steps 6 to 25
LOG_EVERY = 50                         # the launch block's value; cmd_run refuses another than the gate's
VAL_BATCHES, VAL_TOTAL_PX = 53, 159_279_104
TRAIN_COUNT, VAL_COUNT = 5367, 846
RAMP_DENOMINATOR = 334                 # ramp == (it - 1)/334: one epoch of 335 iterations at batch 16
SHAPES = {"teacher_logits": [16, 116, 64, 64], "teacher_feat": [16, 320, 32, 32], "head_logits": [16, 116, 64, 64],
          "c5": [16, 160, 32, 32]}
TERM_KEYS = {"e2": ["logit_kd"], "e3": ["logit_kd", "cwd_feat", "cwd_logit"], "a": ["cwd_feat", "cwd_logit"],
             "f": ["cwd_feat"], "g": ["cwd_logit"]}
HARD_CHECKS = ("logits_shape", "c5_channels", "head_logits_64x64", "c5_32x32", "teacher_logits_shape",
               "teacher_feat_shape", "loss_finite", "has_expected_terms", "optimizer_step", "teacher_stayed_frozen",
               "teacher_params_frozen", "optimizer_excludes_teacher", "teacher_same_augmented_input",
               "distill_ramp_starts_at_zero", "distill_zero_at_step1", "sup_added_unscaled", "sup_positive_finite",
               "grad_clip_applied", "val_cm_accumulated", "lr_non_increasing")
# PL-36: the 11 refusals (P6), each a valid real E2 launch with one change; the code the trainer prints
REFUSALS = ("ckpt_dir_required", "ckpt_dir_not_fresh", "grad_clip_am7", "lambda_grid", "seed", "batch_size",
            "max_iters_not_horizon", "max_val_batches", "num_workers", "init", "teacher_ckpt_sha256_mismatch")
WARN_TAG = "[smoke-warning] "
C10_WARNING_RE = re.compile(r"^\[W[^\]]*\] ")
PY_WARNING_RE = re.compile(r"\b[A-Za-z]*Warning: ")


def train_keys(stage: str) -> list:
    """E-12's train-row keys of a stage, in the trainer's order (write_train_row)."""
    keys = ["event", "iter", "loss", "sup", "ce", "dice", *TERM_KEYS[stage], "ramp", "lr", "grad_norm",
            "grad_norm_student"]
    if "cwd_feat" in TERM_KEYS[stage]:
        keys.append("grad_norm_projection")
    return keys + ["wall_clock", "iter_seconds", "samples_per_sec"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def stats(xs: list) -> dict:
    xs = [x for x in xs if finite(x)]
    return {"median": statistics.median(xs) if xs else None, "mean": statistics.fmean(xs) if xs else None,
            "max": max(xs) if xs else None, "n": len(xs)}


# ---------------------------------------------------------------------------------------- evaluator
def row(check: str, status: str, detail: str) -> dict:
    return {"check": check, "status": status, "detail": detail}


def children_of(doc: dict) -> dict:
    out = {f"stage_{s}": v for s, v in (doc.get("stages") or {}).items()}
    out["repeat_e3"] = doc.get("repeat") or {}
    out.update({f"cross_{s}": v for s, v in (doc.get("cross") or {}).items()})
    out["ce_rate"] = doc.get("ce_rate") or {}
    return out


def eval_children(doc: dict) -> dict:
    """Every child exited 0 and its trainer run printed RESULT: PASS. One exception, for the two cross children: 73fd4d7's
    trainer evaluates the weighted CE twice and requires bitwise equal results (OLD_CE_TWICE_CHECKS), which the
    CUDA CE forward (nll_loss2d_forward_out_cuda_template, PL-35) does not promise; a cross run that failed only
    those checks is reported (INFO) and its losses are compared under PL-34 as any other."""
    kids = children_of(doc)
    bad, info = {}, []
    for k, v in kids.items():
        crashed = v.get("child_rc") != 0 or v.get("error")
        failed = sorted(c for c, ok in (v.get("hard_checks") or {}).items() if not ok)
        if (k.startswith("cross_") and not crashed and v.get("run_rc") == 1 and failed
                and set(failed) <= set(OLD_CE_TWICE_CHECKS)):
            info.append(f"{k}: 73fd4d7's own check(s) {failed} failed (the CE evaluated twice is not bitwise equal "
                        f"under {ALLOWED_NONDET_WARNING})")
            continue
        if crashed or ("run_rc" in v and v.get("run_rc") != 0):
            bad[k] = {"child_rc": v.get("child_rc"), "run_rc": v.get("run_rc"), "error": str(v.get("error"))[:200],
                      "failed_checks": failed}
    missing = [s for s in STAGE_KEYS if s not in (doc.get("stages") or {})] + \
              [f"cross_{s}" for s in ("e2", "e3") if s not in (doc.get("cross") or {})]
    if missing:
        bad["missing"] = missing
    note = f"; INFO: {'; '.join(info)}" if info else ""
    return row("children", "STOP" if bad else "PASS", f"{bad}{note}" if bad else
               f"{len(kids)} children exited 0; every trainer run printed RESULT: PASS{note}")


def eval_environment(doc: dict) -> dict:
    env = doc.get("environment") or {}
    bad = [k for k, ok in (("image digest", env.get("image_digest") == env.get("kd_image_digest")),
                           ("HEAD", env.get("head") is not None and env.get("head") == env.get("expect_head")),
                           ("governed paths", env.get("governed_dirty") == []),
                           ("PLANTSEG_GIT_COMMIT", env.get("plantseg_git_commit") in (None, ""))) if not ok]
    return row("environment", "STOP" if bad else "PASS", f"not as required: {bad}" if bad else
               f"KD digest, HEAD {str(env.get('head'))[:12]} = the pin, governed paths clean, "
               "PLANTSEG_GIT_COMMIT unset")


def eval_determinism(doc: dict) -> dict:
    bad = {}
    for s, st in (doc.get("stages") or {}).items():
        d = st.get("determinism") or {}
        if not (d.get("deterministic_algorithms") is True and d.get("warn_only") is True
                and d.get("cudnn_deterministic") is True and d.get("cudnn_benchmark") is False
                and d.get("cublas_after_set_seed") == ":4096:8"):
            bad[s] = d
    return row("determinism", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               "deterministic algorithms (warn_only), cuDNN deterministic and not benchmark, "
               "CUBLAS_WORKSPACE_CONFIG=:4096:8 after set_seed (its value at process start is recorded)")


def eval_tf32(doc: dict) -> dict:
    bad = {s: st.get("tf32") for s, st in (doc.get("stages") or {}).items()
           if st.get("tf32") is None or st.get("tf32") != st.get("tf32_defaults")}
    return row("tf32", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               "TF32 defaults in every stage")


def eval_teacher(doc: dict, problems_fn) -> dict:
    sys.path.insert(0, str(REPO))
    import scripts.preflight_distill as pd     # the parent's import only: the values of record (PL-11)
    bad = {}
    for s, st in (doc.get("stages") or {}).items():
        t = st.get("teacher") or {}
        p = list(problems_fn(t, t.get("ckpt_path")))
        if t.get("file_name") != pd.TEACHER_FILE:
            p.append(f"file name {t.get('file_name')!r}")
        if p:
            bad[s] = p[:4]
    return row("teacher_identity", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               "the teacher of record (DL-88): sha256, bytes, provenance, parameters, none trainable, eval mode, "
               f"854 state entries and the file name {pd.TEACHER_FILE} (E-31)")


def eval_cuda_order(doc: dict) -> dict:
    bad = {}
    for s, st in (doc.get("stages") or {}).items():
        got = (st.get("cuda_initialized_after_teacher_load"), st.get("cuda_initialized_before_run"),
               st.get("cuda_initialized_after_first_forward"))
        if got != (False, False, True):
            bad[s] = got
    return row("cuda_order", "STOP" if bad or not doc.get("stages") else "PASS",
               f"(after the teacher load, before run(), after the first forward): {bad}" if bad else
               "CUDA uninitialised after the teacher load and before run(); initialised after the first forward")


def eval_data_root(doc: dict) -> dict:
    want = {"train": {"images": TRAIN_COUNT, "masks": TRAIN_COUNT}, "val": {"images": VAL_COUNT, "masks": VAL_COUNT}}
    bad = {s: st.get("data_root") for s, st in (doc.get("stages") or {}).items() if st.get("data_root") != want}
    return row("data_root", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               "assert_trainval_only_root: 5,367/846 pairs, no TEST surface")


def eval_shapes(doc: dict) -> dict:
    bad = {s: st.get("shapes") for s, st in (doc.get("stages") or {}).items() if st.get("shapes") != SHAPES}
    return row("shapes", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               f"batch 16: {SHAPES}")


def eval_interpolation(doc: dict) -> dict:
    bad = {s: st.get("interpolate_calls") for s, st in (doc.get("stages") or {}).items()
           if st.get("interpolate_calls") != 0}
    return row("interpolation", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               "no F.interpolate call whose caller frame is in teacher.py or distillation_losses")


def eval_nmf(doc: dict) -> dict:
    stages = doc.get("stages") or {}
    seqs, bad = {}, {}
    for s, st in stages.items():
        n = st.get("nmf") or {}
        calls = n.get("forwards") or []
        if not (len(calls) == STEPS and all(c.get("draws_delta") == 1 for c in calls)
                and n.get("final_draws") == STEPS and all(c.get("rng_unchanged") is True for c in calls)):
            bad[s] = {"forwards": len(calls), "final_draws": n.get("final_draws"),
                      "draws_delta": sorted({str(c.get("draws_delta")) for c in calls}),
                      "rng_unchanged": all(c.get("rng_unchanged") is True for c in calls)}
        seqs[s] = [c.get("state_sha256") for c in calls]
    if not seqs or len({json.dumps(v) for v in seqs.values()}) != 1:
        bad["state_sha256_sequences"] = "differ across the stages"
    return row("nmf_stream", "STOP" if bad else "PASS", f"{bad}" if bad else
               "one NMF basis draw per step, draws == steps, the same state_sha256 sequence in all five stages, "
               "CPU and CUDA RNG states bitwise unchanged across every teacher forward")


def eval_step1(doc: dict) -> dict:
    bad = {}
    for s, st in (doc.get("stages") or {}).items():
        r = (st.get("train_rows") or [{}])[0]
        hard = st.get("hard_checks") or {}
        failed = [k for k in HARD_CHECKS if hard.get(k) is not True]
        ok = (r.get("iter") == 1 and r.get("ramp") == 0.0 and finite(r.get("loss")) and r.get("loss") == r.get("sup")
              and not failed and all(finite(r.get(k)) for k in TERM_KEYS.get(s, []))
              and finite(r.get("grad_norm")) and finite(r.get("grad_norm_student"))
              and ("cwd_feat" not in TERM_KEYS.get(s, []) or r.get("grad_norm_projection") == 0.0))
        if not ok:
            bad[s] = {"ramp": r.get("ramp"), "loss": r.get("loss"), "sup": r.get("sup"), "failed_checks": failed,
                      "parts": {k: r.get(k) for k in TERM_KEYS.get(s, [])},
                      "grad_norm_projection": r.get("grad_norm_projection")}
    return row("step_1", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               "ramp 0.0 and distill 0.0 (distill_zero_at_step1), loss == sup bitwise, every hard check true, the "
               "unramped parts and the norms finite, grad_norm_projection 0.0 in E3, A and F")


def eval_steps(doc: dict) -> dict:
    bad = {}
    for s, st in (doc.get("stages") or {}).items():
        rows = st.get("train_rows") or []
        nonfinite = [r.get("iter") for r in rows
                     if "nonfinite" in r or not all(finite(v) for k, v in r.items() if k != "event")]
        ramp_bad = [r.get("iter") for r in rows if r.get("ramp") != (r.get("iter", 0) - 1) / RAMP_DENOMINATOR]
        if len(rows) != STEPS or [r.get("iter") for r in rows] != list(range(1, STEPS + 1)) or nonfinite or ramp_bad:
            bad[s] = {"rows": len(rows), "nonfinite": nonfinite[:5], "ramp": ramp_bad[:5]}
    return row("steps_25", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               "25 rows per stage, every value finite, ramp == (it-1)/334 bitwise")


def repeat_bitwise(doc: dict) -> bool | None:
    rep, e3 = doc.get("repeat") or {}, (doc.get("stages") or {}).get("e3") or {}
    a, b = [r.get("loss") for r in e3.get("train_rows") or []], [r.get("loss") for r in rep.get("train_rows") or []]
    if not a or len(a) != len(b) or not e3.get("student_sha256") or len(e3.get("student_sha256")) != len(a):
        return None
    return a == b and e3.get("student_sha256") == rep.get("student_sha256")


def eval_repeat(doc: dict) -> dict:
    bit = repeat_bitwise(doc)
    if bit is None:
        return row("repeat_e3", "STOP", "the E3 repeat has no comparable loss and student-state sequence")
    e3 = (doc.get("stages") or {}).get("e3") or {}
    rep = doc.get("repeat") or {}
    pairs = [(x.get("loss"), y.get("loss")) for x, y in zip(e3.get("train_rows"), rep.get("train_rows"))]
    if not all(finite(x) and finite(y) for x, y in pairs):
        return row("repeat_e3", "STOP", "a non-finite loss in the E3 stage or its repeat (the losses row names it)")
    deltas = [abs(x - y) for x, y in pairs]
    return row("repeat_e3", "INFO", "bitwise: the losses and the student state sha256 at every step" if bit else
               f"not bitwise: max |delta loss| {max(deltas):.3e} (informative; the cross-commit rule then allows "
               f"{CROSS_REL_TOL:g} relative)")


def eval_cross(doc: dict) -> dict:
    bit = repeat_bitwise(doc)
    bad = {}
    for s in ("e2", "e3"):
        new, old = ((doc.get("stages") or {}).get(s) or {}), ((doc.get("cross") or {}).get(s) or {})
        a = [r.get("loss") for r in (new.get("train_rows") or [])[:CROSS_STEPS]]
        b = [r.get("loss") for r in (old.get("train_rows") or [])[:CROSS_STEPS]]
        if len(a) != CROSS_STEPS or len(b) != CROSS_STEPS or not all(finite(x) for x in a + b):
            bad[s] = f"{len(a)} new and {len(b)} old finite steps"
            continue
        if old.get("train_distill_sha256") != OLD_TRAIN_DISTILL_SHA256:
            bad[s] = f"the old side ran train_distill.py {old.get('train_distill_sha256')}"
            continue
        if not new.get("first_batch_sha256") or new.get("first_batch_sha256") != old.get("first_batch_sha256"):
            bad[s] = "the first batch differs between the two commits"
            continue
        rel = [abs(x - y) / max(abs(x), abs(y), 1e-30) for x, y in zip(a, b)]
        if bit and a != b:
            bad[s] = f"the E3 repeat is bitwise and the commits differ (max relative {max(rel):.3e}; PL-34)"
        elif max(rel) > CROSS_REL_TOL:
            bad[s] = f"max relative |delta loss| {max(rel):.3e} > {CROSS_REL_TOL:g}"
    return row("cross_commit", "STOP" if bad else "PASS", f"{bad}" if bad else
               ("bitwise against 73fd4d7 over 20 steps (the E3 repeat is bitwise)" if bit else
                f"within {CROSS_REL_TOL:g} relative per step against 73fd4d7 over 20 steps"))


def eval_ce(doc: dict) -> dict:
    ce = doc.get("ce_rate") or {}
    return row("ce_nondeterminism", "INFO", f"{ce.get('distinct')} distinct values in {ce.get('trials')} trials "
                                            "(informative)")


def eval_memory(doc: dict) -> dict:
    worst, peaks = "PASS", {}
    for s, st in (doc.get("stages") or {}).items():
        r = (st.get("memory") or {}).get("max_memory_reserved")
        peaks[s] = r
        if not isinstance(r, int) or r > MEM_STOP_BYTES:
            worst = "STOP"
        elif r > MEM_PASS_BYTES and worst == "PASS":
            worst = "FLAG"
    if not peaks:
        worst = "STOP"
    return row("peak_memory", worst, f"max_memory_reserved bytes {peaks} (PASS <= {MEM_PASS_BYTES}, FLAG <= "
                                     f"{MEM_STOP_BYTES}, STOP above; PL-33)")


def throughput(st: dict) -> dict:
    lo, hi = THROUGHPUT_STEPS
    rows = [r for r in st.get("train_rows") or [] if lo <= r.get("iter", 0) <= hi]
    it_s = stats([r.get("iter_seconds") for r in rows])
    t_ms = stats((st.get("teacher_forward_ms") or [])[lo - 1:hi])
    share = (t_ms["median"] / 1000.0 / it_s["median"]) if finite(t_ms["median"]) and finite(it_s["median"]) \
        and it_s["median"] > 0 else None
    return {"iter_seconds": it_s, "next_seconds": stats((st.get("next_seconds") or [])[lo - 1:hi]),
            "teacher_forward_ms": t_ms, "teacher_share": share,
            "hook_seconds": stats((st.get("hook_seconds") or [])[lo - 1:hi])}


def eval_throughput(doc: dict) -> dict:
    tp = {s: throughput(st) for s, st in (doc.get("stages") or {}).items()}
    n_want = THROUGHPUT_STEPS[1] - THROUGHPUT_STEPS[0] + 1
    e3 = ((tp.get("e3") or {}).get("iter_seconds") or {}).get("median")
    if not finite(e3) or len(tp) != len(STAGE_KEYS) or any(v["iter_seconds"]["n"] != n_want for v in tp.values()):
        return row("throughput", "STOP", f"the trainer's iter_seconds of steps 6 to 25 are missing: {tp}")
    text = "median s/iter (steps 6-25): " + ", ".join(f"{s} {v['iter_seconds']['median']:.3f}" for s, v in tp.items())
    hook = ((tp.get("e3") or {}).get("hook_seconds") or {}).get("median")
    if finite(hook):        # the E3 stage copies the student state inside the timed step (the repeat's comparison)
        text += f"; E3's median includes {hook:.3f} s/iter of the smoke's own state copy (hook_seconds)"
    if e3 > E3_S_PER_ITER_FLAG:
        return row("throughput", "FLAG", f"E3 median {e3:.3f} s/iter > {E3_S_PER_ITER_FLAG}: {PL38_MESSAGE}. {text}")
    return row("throughput", "PASS", f"{text} (AM-19 item 3(b)'s number for each stage is that stage's median; "
                                     "PL-37)")


def eval_val(doc: dict) -> dict:
    v = ((doc.get("stages") or {}).get("e2") or {}).get("val_row") or {}
    ok = (v.get("iter") == STEPS and v.get("val_batches") == VAL_BATCHES and v.get("val_total_px") == VAL_TOTAL_PX
          and finite(v.get("all_class_miou")) and finite(v.get("val_seconds")))
    return row("val_full_pass", "PASS" if ok else "STOP",
               f"E2 student at step {v.get('iter')}: {v.get('val_batches')} batches, {v.get('val_total_px')} px, "
               f"all-class mIoU {v.get('all_class_miou')}, {v.get('val_seconds')} s"
               + ("" if ok else f" (want {VAL_BATCHES} batches and {VAL_TOTAL_PX} px)"))


def eval_schema(doc: dict) -> dict:
    bad = {}
    for s, st in (doc.get("stages") or {}).items():
        rows = st.get("train_rows") or []
        if not rows or any(list(r) != train_keys(s) for r in rows):
            bad[s] = (sorted({tuple(r) for r in rows} - {tuple(train_keys(s))})[:1] or "no rows")
        elif not st.get("val_row"):
            bad[s] = "no VAL row"
        elif st.get("check_run_meta_rc") != 0:
            bad[s] = f"check-run-meta --allow-smoke rc={st.get('check_run_meta_rc')}"
    return row("schema", "STOP" if bad or not doc.get("stages") else "PASS", f"{bad}" if bad else
               "train rows carry E-12's keys in order, VAL rows present, check-run-meta --allow-smoke PASS")


def eval_refusals(doc: dict) -> dict:
    got = {r.get("code"): r for r in doc.get("refusals") or []}
    bad = {}
    for code in REFUSALS:
        r = got.get(code)
        if r is None:
            bad[code] = "not run"
        elif r.get("overrun"):
            bad[code] = f"killed after {REFUSAL_LIMIT_S} s"
        elif r.get("rc") != 2:
            bad[code] = f"rc={r.get('rc')}"
        elif r.get("run_meta_files"):
            bad[code] = f"run_meta written: {r.get('run_meta_files')}"
        elif f"[{code}]" not in (r.get("stderr") or ""):
            bad[code] = "refused with another code"
    return row("gate_refusals", "STOP" if bad else "PASS", f"{bad}" if bad else
               f"the {len(REFUSALS)} refusals exit 2 with their codes and write no run_meta (PL-36)")


def nondeterminism_warning(w: dict) -> bool:
    return "deterministic" in f"{w.get('category', '')} {w.get('message', '')}".lower()


def eval_warnings(doc: dict) -> dict:
    ws = doc.get("warnings") or []
    stop = [w for w in ws if nondeterminism_warning(w) and ALLOWED_NONDET_WARNING not in (w.get("message") or "")]
    other = [w for w in ws if not nondeterminism_warning(w)]
    if stop:
        return row("warnings", "STOP", f"nondeterminism warning(s) other than {ALLOWED_NONDET_WARNING}: "
                                       f"{[w.get('message', '')[:120] for w in stop[:3]]}")
    if other:
        return row("warnings", "FLAG", f"{len(other)} other warning(s), all listed in the JSON: "
                                       f"{[w.get('message', '')[:120] for w in other[:3]]}")
    return row("warnings", "PASS", f"{len(ws)} warning(s), none but {ALLOWED_NONDET_WARNING}")


def eval_losses(doc: dict) -> dict:
    bad = []
    for name, st in children_of(doc).items():
        for r in st.get("train_rows") or []:
            keys = ["loss", *(k for k in ("sup", "ce", "dice", "logit_kd", "cwd_feat", "cwd_logit") if k in r)]
            if "nonfinite" in r or not all(finite(r.get(k)) for k in keys):
                bad.append(f"{name}@{r.get('iter')}")
    return row("losses", "STOP" if bad else "PASS", f"non-finite: {bad[:6]}" if bad else
               "every loss of every child finite")


def evaluate(doc: dict, teacher_problems) -> tuple:
    """Every criterion's row. An evaluator that raises on a malformed record gives its row a STOP with the error,
    so the verdict and the JSON are still written."""
    fns = (eval_children, eval_environment, eval_determinism, eval_tf32,
           lambda d: eval_teacher(d, teacher_problems), eval_cuda_order, eval_data_root, eval_shapes,
           eval_interpolation, eval_nmf, eval_step1, eval_steps, eval_repeat, eval_cross, eval_ce, eval_memory,
           eval_throughput, eval_val, eval_schema, eval_refusals, eval_warnings, eval_losses)
    rows = []
    for fn in fns:
        try:
            rows.append(fn(doc))
        except Exception as e:  # noqa: BLE001 - a record the evaluator cannot read is a STOP, never a traceback
            name = "teacher" if fn.__name__ == "<lambda>" else fn.__name__.removeprefix("eval_")
            rows.append(row(f"{name}_evaluator_error", "STOP", f"{type(e).__name__}: {e}"))
    statuses = {r["status"] for r in rows}
    return rows, "STOP" if "STOP" in statuses else "FLAG" if "FLAG" in statuses else "PASS"


# ----------------------------------------------------------------------------------------- children
def install_warning_log(rec: dict) -> None:
    """PL-35: every warning the child shows (Python's default filters, as a launch has them) is kept in its
    record and printed to stderr with a tag, so a DataLoader worker's warning reaches the child's log too."""
    import warnings
    seen = {}

    def show(message, category, filename, lineno, file=None, line=None):
        w = {"category": category.__name__, "message": str(message)[:400], "filename": str(filename),
             "lineno": lineno}
        key = json.dumps(w, sort_keys=True)
        if key in seen:
            seen[key]["count"] += 1
            return
        w["count"] = 1
        seen[key] = w
        rec["warnings"].append(w)
        print(WARN_TAG + json.dumps(w), file=sys.stderr, flush=True)
    warnings.showwarning = show


def load_teacher(td, ckpt: str, sha: str):
    """The real teacher through the code under test's loader, the file hashed first (73fd4d7's loader takes no
    expected_sha256: this keeps "hashed before it is read" for both sides); the identity record of P5 stage 9."""
    digest = sha256_file(ckpt)
    if digest != sha:
        raise RuntimeError(f"[teacher_ckpt_sha256_mismatch] {ckpt} hashes to {digest}, not {sha}")
    kw = {"config_path": str(TEACHER_CONFIG)}
    if "expected_sha256" in inspect.signature(td.load_frozen_teacher).parameters:
        kw["expected_sha256"] = sha
    teacher = td.load_frozen_teacher(ckpt, **kw)
    mod = teacher.teacher
    rec = {"ok": True, "sha256": digest, "bytes": os.path.getsize(ckpt), "ckpt_path": ckpt,
           "file_name": Path(ckpt).name, "provenance": teacher.provenance.as_dict(),
           "params": int(sum(p.numel() for p in mod.parameters())),
           "trainable": int(sum(p.numel() for p in mod.parameters() if p.requires_grad)),
           "training_modules": sorted(n for n, m in mod.named_modules() if m.training)[:10],
           "state_entries": len(mod.state_dict())}
    return teacher, rec


def child_stage(a, kind: str, rec: dict) -> None:
    """One instrumented dry run of train_distill.run() (`_stage`, `_repeat`, `_cross`), recorded into `rec`."""
    rec.update(kind=kind, stage=a.stage, cublas_at_start=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), warnings=[])
    install_warning_log(rec)
    code_root = Path(a.code_root).resolve() if a.code_root else REPO
    sys.path.insert(0, str(code_root))
    import torch
    import torch.nn.functional as F
    import src.training.train_distill as td
    td_path = Path(td.__file__).resolve()
    if td_path.parents[2] != code_root:
        raise RuntimeError(f"train_distill came from {td_path}, not from {code_root}")
    rec.update(code_root=str(code_root), train_distill_sha256=sha256_file(td_path), torch=torch.__version__)
    stage = td.resolve_stage(a.stage)
    has_logit_kd = bool(stage.get("logit_kd", True))      # 73fd4d7's stages are E2 and E3, both with Logit KD
    want_state = kind in ("stage", "repeat") and a.stage == "e3"
    cuda = str(a.device).startswith("cuda")
    ck = Path(a.out).with_suffix("") / f"{a.stage}_s42_smoke"
    if a.stub:                                      # selfcheck: a weight-free teacher and synthetic batches
        teacher = td.FrozenTeacher(td.MockTeacher(td.NUM_CLASSES))
        rec.update(teacher={"stub": True}, data_root=None)
    else:
        from src.data.isolation import assert_trainval_only_root
        rec["data_root"] = assert_trainval_only_root(Path(td.DATA["root"]))["counts"]
        teacher, rec["teacher"] = load_teacher(td, a.teacher_ckpt, a.teacher_ckpt_sha256)
    rec["cuda_initialized_after_teacher_load"] = torch.cuda.is_initialized()
    # set_seed: the determinism state it leaves (PL-37: CUBLAS_WORKSPACE_CONFIG at start and after set_seed)
    orig_set_seed = td.set_seed

    def set_seed(seed):
        out = orig_set_seed(seed)
        rec["determinism"] = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                              "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                              "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                              "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                              "cublas_after_set_seed": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        return out
    td.set_seed = set_seed
    holder = {}
    orig_build_student = td.build_student

    def build_student(*args, **kwargs):
        holder["student"] = orig_build_student(*args, **kwargs)
        return holder["student"]
    td.build_student = build_student
    if a.stub:
        def build_dataloader(split, batch_size, **_kw):
            g = torch.Generator().manual_seed(0 if split == "train" else 1)
            n = 3 if split == "train" else 1
            batches = [(torch.randn(batch_size, 3, 512, 512, generator=g),
                        torch.randint(0, td.NUM_CLASSES, (batch_size, 512, 512), generator=g)) for _ in range(n)]

            class _Loader(list):
                dataset = [0] * (n * batch_size)
            return _Loader(batches)
        td.build_dataloader = build_dataloader
    # the train iterator: the time blocked in next() per step and the first batch's sha256 (PL-34, PL-37)
    orig_cycle = td.cycle
    rec["next_seconds"] = []

    def cycle(loader):
        it = orig_cycle(loader)
        while True:
            t0 = time.perf_counter()
            batch = next(it)
            rec["next_seconds"].append(time.perf_counter() - t0)
            if "first_batch_sha256" not in rec:
                rec["first_batch_sha256"] = hashlib.sha256(b"".join(
                    t.detach().cpu().contiguous().numpy().tobytes() for t in batch)).hexdigest()
            yield batch
    td.cycle = cycle
    # the optimizer step: peak memory after each step (PL-33) and, in E3 and its repeat, a CPU copy of the
    # student's state (hashed after the run, outside the timed loop; PL-37). A bound method: the LR scheduler
    # wraps optimizer.step and needs __self__.
    rec["step_rows"], rec["hook_seconds"], snapshots = [], [], []
    if hasattr(td, "build_optimizer"):
        orig_build_optimizer = td.build_optimizer

        def build_optimizer(*args, **kwargs):
            opt, trainable = orig_build_optimizer(*args, **kwargs)
            orig_step = opt.step

            def step(self, *sa, **sk):
                out = orig_step(*sa, **sk)
                t0 = time.perf_counter()
                r = {"step": len(rec["step_rows"]) + 1}
                if cuda:
                    r["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
                    r["max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
                if want_state:
                    snapshots.append([(k, v.detach().to("cpu", copy=True))
                                      for k, v in holder["student"].state_dict().items()])
                rec["step_rows"].append(r)
                rec["hook_seconds"].append(time.perf_counter() - t0)
                return out
            opt.step = types.MethodType(step, opt)
            return opt, trainable
        td.build_optimizer = build_optimizer
    # distillation_losses: the maps' shapes at the first call (P6)
    orig_dl = td.distillation_losses

    def distillation_losses(*args, **kw):
        if "shapes" not in rec:
            tout = kw.get("teacher_out")

            def shape(t):
                return None if t is None else list(t.shape)
            rec["shapes"] = {"teacher_logits": shape(getattr(tout, "logits", None)),
                             "teacher_feat": shape(getattr(tout, "feat_s16", None)),
                             "head_logits": shape(kw.get("head_logits")), "c5": shape(kw.get("c5"))}
        return orig_dl(*args, **kw)
    td.distillation_losses = distillation_losses
    # F.interpolate whose caller frame is in teacher.py or distillation_losses (P6: none)
    rec["interpolate_calls"] = 0
    teacher_py = (td_path.parents[1] / "distill" / "teacher.py").as_posix()
    orig_interp = F.interpolate

    def interpolate(*args, **kwargs):
        caller = sys._getframe(1)
        if Path(caller.f_code.co_filename).resolve().as_posix() == teacher_py \
                or caller.f_code.co_name == "distillation_losses":
            rec["interpolate_calls"] += 1
        return orig_interp(*args, **kwargs)
    F.interpolate = interpolate
    # the teacher forward: the NMF draw, the RNG states, CUDA initialisation and its CUDA-event time
    nmf = {"forwards": []}
    rec["nmf"] = nmf
    events = []
    orig_forward = teacher.forward
    stream_state = getattr(teacher, "nmf_stream_state", lambda: None)

    def forward(x, **kw):
        on_cuda = cuda and torch.cuda.is_initialized()
        cpu0 = torch.get_rng_state()
        cuda0 = torch.cuda.get_rng_state_all() if on_cuda else []
        s0 = stream_state() or {}
        ev = None
        if on_cuda:
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
        out = orig_forward(x, **kw)
        if ev is not None:
            ev[1].record()
        events.append(ev)
        s1 = stream_state() or {}
        cuda1 = torch.cuda.get_rng_state_all() if on_cuda else []
        if "cuda_initialized_after_first_forward" not in rec:
            rec["cuda_initialized_after_first_forward"] = torch.cuda.is_initialized()
        nmf["forwards"].append({"draws_delta": (s1["draws"] - s0.get("draws", 0)) if "draws" in s1 else None,
                                "state_sha256": s1.get("state_sha256"),
                                "rng_unchanged": bool(torch.equal(cpu0, torch.get_rng_state()))
                                and len(cuda0) == len(cuda1) and all(torch.equal(p, q) for p, q in zip(cuda0, cuda1))})
        return out
    teacher.forward = forward
    # no checkpoint is written (P6)
    td.save_distill_checkpoint = lambda *args, **kwargs: str(ck / "not_written.pt")
    td.write_best_pointer = lambda *args, **kwargs: None
    rec["cuda_initialized_before_run"] = torch.cuda.is_initialized()
    rec["tf32_defaults"] = getattr(td, "TF32_DEFAULTS", None)
    kw = dict(stage=stage, mode="dry", device=a.device, pretrained=False, teacher=teacher,
              lambda_logit=1.0 if has_logit_kd else None, batch_size=a.batch, max_iters=a.steps,
              val_interval=4000, max_val_batches=a.max_val_batches, num_workers=a.num_workers,
              ckpt_dir_arg=str(ck), grad_clip_norm=None, log_every=LOG_EVERY, seed=42)
    accepted = inspect.signature(td.run).parameters
    kw = {k: v for k, v in kw.items() if k in accepted}
    rec["ckpt_dir"] = str(ck)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            rec["run_rc"] = td.run(**kw)
    finally:
        text = out.getvalue()
        rec["stdout_tail"] = text[-3000:]
    rec["hard_checks"] = {ln.split(":")[0].strip(): ln.rsplit(":", 1)[1].strip() == "PASS"
                          for ln in text.split("\n[CHECKS]\n", 1)[-1].splitlines()
                          if ln.startswith("  ") and ln.rstrip().endswith(("PASS", "FAIL"))}
    rec["tf32"] = td.tf32_state() if hasattr(td, "tf32_state") else None
    rec["cudnn"] = torch.backends.cudnn.version() if cuda else None
    if cuda:
        torch.cuda.synchronize()
        rec["memory"] = {"max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                         "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                         "at_step_1": {k: v for k, v in (rec["step_rows"] or [{}])[0].items() if k != "step"},
                         "total_memory": int(torch.cuda.get_device_properties(0).total_memory)}
        rec["gpu_name"] = torch.cuda.get_device_name(0)
    rec["teacher_forward_ms"] = [None if e is None else e[0].elapsed_time(e[1]) for e in events]
    nmf["final_draws"] = (stream_state() or {}).get("draws")
    if want_state:
        hashes = []
        for snap in snapshots:
            h = hashlib.sha256()
            for k, t in snap:
                h.update(k.encode("utf-8"))
                h.update(t.contiguous().numpy().tobytes())
            hashes.append(h.hexdigest())
        rec["student_sha256"] = hashes
    tel = ck / f"{a.stage}_telemetry.jsonl"
    rows = [json.loads(ln) for ln in tel.read_text(encoding="utf-8").splitlines() if ln.strip()] if tel.exists() else []
    rec["train_rows"] = [r for r in rows if r.get("event") == "train"]
    vals = [r for r in rows if r.get("event") == "val"]
    rec["val_row"] = None if not vals else {k: vals[-1].get(k) for k in ("iter", "all_class_miou", "val_batches",
                                                                         "val_total_px", "val_seconds")}
    rec["run_end"] = next((r for r in rows if r.get("event") == "run_end"), None)


def child_ce_rate(a, rec: dict) -> None:
    """CE nondeterminism on CUDA (informative, P6): one fixed batch, the weighted CE computed repeatedly."""
    rec["warnings"] = []
    install_warning_log(rec)
    sys.path.insert(0, str(REPO))
    import torch
    import src.training.train_distill as td
    td.set_seed(42)
    dev = a.device
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(BATCH, td.NUM_CLASSES, 512, 512, generator=g).to(dev)
    mask = torch.randint(0, td.NUM_CLASSES, (BATCH, 512, 512), generator=g)
    mask[:, :8, :] = td.IGNORE_INDEX                  # some padding, as the loaders produce
    mask = mask.to(dev)
    crit = td.CombinedCEDiceLoss(weight=td.load_ce_weights().to(dev), ignore_index=td.IGNORE_INDEX).to(dev)
    vals = [float(crit.ce(logits, mask)) for _ in range(a.trials)]
    rec.update(trials=a.trials, distinct=len(set(vals)), values=sorted(set(vals))[:8])


def emit(path: str, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------------------- the refusals
def synthetic_root(folder: Path) -> Path:
    """A full-count TRAIN/VAL root of empty files (PL-36): 5,367 and 846 image/mask pairs, no TEST surface."""
    root = folder / "plantseg_synthetic"
    for split, n in (("train", TRAIN_COUNT), ("val", VAL_COUNT)):
        for kind, suffix in (("images", ".jpg"), ("annotations", ".png")):
            d = root / kind / split
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"s{i:05d}{suffix}").touch()
    return root


def throwaway_teacher(folder: Path) -> Path:
    """A teacher file that is not the teacher of record: a refusal child that passed every gate would stop at
    the checksum, never train (PL-36)."""
    p = folder / "teacher" / "iter_24000.pth"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"a throwaway teacher file (smoke_kd_step refusal children)\n")
    return p


def refusal_argv(code: str, base: Path, teacher: str, sha: str, num_workers: int) -> tuple:
    """A real E2 launch that is valid except for the one change that `code` refuses; its --ckpt-dir. The base is
    the launch block's shape (PL-29: a records commit, the workers of record) plus --device cuda, which the launch
    block never passes but the CPU --selfcheck needs, so that each child is refused before any CUDA work."""
    ck = base / f"ck_{code}"
    args = {"--real-run": None, "--confirm-real-run": None, "--stage": "e2", "--seed": "42", "--lambda-logit": "1",
            "--init": "imagenet", "--num-workers": str(num_workers), "--log-every": str(LOG_EVERY),
            "--records-commit": "0" * 40, "--teacher-ckpt": teacher, "--teacher-ckpt-sha256": sha,
            "--ckpt-dir": str(ck), "--device": "cuda"}
    change = {"ckpt_dir_required": {"--ckpt-dir": False}, "ckpt_dir_not_fresh": {},
              "grad_clip_am7": {"--grad-clip-norm": "100"}, "lambda_grid": {"--lambda-logit": "3"},
              "seed": {"--seed": "45"}, "batch_size": {"--batch-size": "8"},
              "max_iters_not_horizon": {"--max-iters": "1000"}, "max_val_batches": {"--max-val-batches": "1"},
              "num_workers": {"--num-workers": "0"}, "init": {"--init": "none"},
              "teacher_ckpt_sha256_mismatch": {}}[code]
    for k, v in change.items():
        if v is False:
            args.pop(k)
        else:
            args[k] = v
    if code == "ckpt_dir_not_fresh":
        ck.mkdir(parents=True)
        (ck / "an_earlier_file.txt").write_text("not fresh\n", encoding="utf-8")
    argv = [str(REPO / "src" / "training" / "train_distill.py")]
    for k, v in args.items():
        argv += [k] if v is None else [k, v]
    return argv, (ck if "--ckpt-dir" in args else None)


def refusal_result(code: str, r) -> dict:
    """One refusal child's completed process as its record: the return code, the stderr tail and every warning of
    its stdout and stderr (PL-35)."""
    return {"rc": r.returncode, "stderr": (r.stderr or "")[-1500:],
            "warnings": warnings_in((r.stdout or "") + "\n" + (r.stderr or ""), f"refusal_{code}")}


def run_refusals(base: Path, data_root: Path, teacher: str, sha: str, *, num_workers: int, limit=REFUSAL_LIMIT_S,
                 codes=REFUSALS) -> list:
    """PL-36: each refusal in a child with the synthetic root, a 300-second limit, rc 2 and no run_meta. Every
    warning in a child's output is kept in its record (PL-35)."""
    out = []
    env = dict(os.environ, PLANTSEG_DATA_ROOT=str(data_root), PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    env.pop("PLANTSEG_GIT_COMMIT", None)
    for code in codes:
        argv, ck = refusal_argv(code, base, teacher, sha, num_workers)
        rec = {"code": code, "overrun": False}
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-B", *argv], cwd=str(REPO), env=env, capture_output=True, text=True,
                               timeout=limit)
            rec.update(refusal_result(code, r))
        except subprocess.TimeoutExpired:
            rec.update(rc=None, overrun=True, stderr="", warnings=[])
        rec["seconds"] = round(time.time() - t0, 1)
        rec["run_meta_files"] = (sorted(p.name for p in ck.iterdir() if p.name.endswith("_run_meta.jsonl"))
                                 if ck is not None and ck.is_dir() else [])
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------- the pod driver
def git_out(*args) -> str | None:
    r = subprocess.run(["git", "-C", str(REPO), "--no-optional-locks", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def run_child(argv: list, out_json: Path, log: Path, timeout: int) -> dict:
    with open(log, "w", encoding="utf-8") as fh:
        try:
            r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *argv, "--out", str(out_json)],
                               cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = None
    doc = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"error": "no record written"}
    doc["child_rc"] = rc
    return doc


def log_warnings(log: Path, source: str) -> list:
    """PL-35: the warnings in a child's log (warnings_in)."""
    return warnings_in(log.read_text(encoding="utf-8", errors="replace"), source)


def warnings_in(text: str, source: str) -> list:
    """PL-35: the warnings in a child's output: its tagged Python warnings (DataLoader workers' included), c10's
    [W ...] lines and any other '...Warning: ' line."""
    out = []
    for ln in text.splitlines():
        if ln.startswith(WARN_TAG):
            try:
                w = json.loads(ln[len(WARN_TAG):])
            except json.JSONDecodeError:
                w = {"category": "unparsed", "message": ln[len(WARN_TAG):]}
        elif C10_WARNING_RE.match(ln) or PY_WARNING_RE.search(ln):
            w = {"category": "log", "message": ln[:400]}
        else:
            continue
        w["source"] = source
        if w not in out:
            out.append(w)
    return out


def scoped_status(pd) -> subprocess.CompletedProcess:
    """The checkout's scoped status (E1's status_argv: the governed paths), read-only: `git -C <checkout>
    --no-optional-locks`, which never writes the index (PL-22; plan P3(a))."""
    return subprocess.run(["git", "-C", str(REPO), "--no-optional-locks", *pd.E1G.status_argv()], capture_output=True)


def gpu_driver() -> str:
    """The NVIDIA driver version, or why it is unavailable."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({type(e).__name__})"


def run_check_run_meta(cmd: list) -> subprocess.CompletedProcess:
    """One `preflight_distill.py check-run-meta --allow-smoke` child, from the checkout (PL-30)."""
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1"))


def teacher_problems(path: str, sha256: str, pd) -> list:
    """The teacher arguments against the values of record (DL-88), before any child: the sha256 given, the file
    name as typed and of the file it resolves to (E-31; the children load the resolved file, and eval_teacher judges
    its name), and a regular file; a path that cannot be read is refused too. The children hash the file, and
    eval_teacher judges what they found."""
    p = Path(path)
    bad = [] if sha256 == pd.TEACHER_SHA256 else [f"--teacher-ckpt-sha256 {sha256!r}"]
    if p.name != pd.TEACHER_FILE:
        bad.append(f"file name {p.name!r}")
    try:
        if not p.is_file():
            bad.append(f"no file at {path!r}")
        elif p.resolve().name != pd.TEACHER_FILE:
            bad.append(f"{path!r} resolves to {str(p.resolve())!r}, a file not named {pd.TEACHER_FILE}")
    except (OSError, RuntimeError) as e:
        bad.append(f"{path!r} cannot be read ({type(e).__name__})")
    return bad


def refused(code: str, text: str) -> int:
    """A refusal before any child runs (exit 2): no STOP; correct the argument, the shell or the checkout and run
    the smoke again."""
    print(f"REFUSED [{code}]: {text}")
    print(f"RESULT: REFUSED ({code}) -- no child has run; this is no STOP")
    return 2


def cmd_run(a) -> int:
    sys.path.insert(0, str(REPO))
    import scripts.preflight_distill as pd
    old = Path(a.old_root).resolve()
    old_td = old / "src" / "training" / "train_distill.py"
    if not old_td.is_file() or sha256_file(old_td) != OLD_TRAIN_DISTILL_SHA256:
        return refused("old_root", f"{old_td} does not hash to {OLD_TRAIN_DISTILL_SHA256} (73fd4d7; PL-34)")
    evidence = Path(a.evidence).resolve()
    if evidence == REPO or REPO in evidence.parents:
        return refused("evidence_in_repo", f"{evidence} is inside the checkout")
    if LOG_EVERY != pd.LOG_EVERY:                # PL-11: the values of record are the gate's
        return refused("values_of_record", f"LOG_EVERY {LOG_EVERY} != the gate's {pd.LOG_EVERY}")
    head = git_out("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", a.expect_head or "") or head != a.expect_head:
        return refused("expect_head", f"--expect-head {a.expect_head!r} is not HEAD "
                                      f"({'git rev-parse HEAD failed' if head is None else head}): run from the "
                                      "checkout of the pin and pass its full 40-hex commit (PL-28)")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = evidence / f"kd_smoke_{stamp}"
    scratch = Path(tempfile.gettempdir()).resolve() / f"kd_step_refusals_{os.getpid()}"
    for p in (base, scratch):
        if "test" in str(p).lower():
            return refused("sl1_path", f"{p} contains 'test' (SL-1)")
    st = scoped_status(pd)
    env_row = {"image_digest": os.environ.get("PLANTSEG_IMAGE_DIGEST"), "kd_image_digest": pd.KD_IMAGE_DIGEST,
               "head": head, "expect_head": a.expect_head,
               "governed_dirty": pd.E1G.scoped_dirty(st.stdout) if st.returncode == 0 else ["git status failed"],
               "plantseg_git_commit": os.environ.get("PLANTSEG_GIT_COMMIT"), "driver": gpu_driver(),
               "data_root": os.environ.get("PLANTSEG_DATA_ROOT")}
    env_check = eval_environment({"environment": env_row})  # judged now, before any child, and again at the end
    if env_check["status"] != "PASS":
        return refused("environment", f"{env_check['detail']} (PLANTSEG_IMAGE_DIGEST {env_row['image_digest']!r}, "
                                      f"PLANTSEG_GIT_COMMIT {env_row['plantseg_git_commit']!r}, governed "
                                      f"{env_row['governed_dirty'][:10]}): run it in KD.1's shell (the KD digest "
                                      "exported, PLANTSEG_GIT_COMMIT unset) on the clean checkout of the pin")
    t_bad = teacher_problems(a.teacher_ckpt, a.teacher_ckpt_sha256, pd)
    if t_bad:
        return refused("teacher", f"{t_bad}: --teacher-ckpt and --teacher-ckpt-sha256 name the teacher of record "
                                  f"(DL-88: {pd.TEACHER_FILE}, sha256 {pd.TEACHER_SHA256[:12]}...); the children "
                                  "hash the file")
    base.mkdir(parents=False)
    teacher = str(Path(a.teacher_ckpt).resolve())
    common = ["--teacher-ckpt", teacher, "--teacher-ckpt-sha256", a.teacher_ckpt_sha256,
              "--num-workers", str(pd.NUM_WORKERS)]                      # PL-11: the gate's value of record
    doc = {"format": "kd_smoke/1", "environment": env_row, "stages": {}, "cross": {}, "warnings": []}
    logs = {}
    for s in STAGE_KEYS:
        logs[f"stage_{s}"] = base / f"stage_{s}.log"
        doc["stages"][s] = run_child(["_stage", s, *common], base / f"stage_{s}.json", logs[f"stage_{s}"], 3600)
        cmd = [sys.executable, "-B", str(REPO / "scripts" / "preflight_distill.py"), "check-run-meta", "--allow-smoke",
               "--ckpt-dir", doc["stages"][s].get("ckpt_dir") or str(base / "no_ckpt_dir"), "--stage", s,
               "--seed", "42", "--profile", "distill_80k", "--expect-head", a.expect_head,
               *(["--lambda-logit", "1"] if s in ("e2", "e3") else []),
               *(["--alpha", "50"] if s in ("e3", "a", "f") else [])]
        r = run_check_run_meta(cmd)
        doc["stages"][s]["check_run_meta_rc"] = r.returncode
        (base / f"check_run_meta_{s}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        doc["warnings"] += [{"category": "check-run-meta", "message": ln[:400], "source": f"check_run_meta_{s}"}
                            for ln in r.stdout.splitlines() if ln.startswith("WARN:")]
        doc["warnings"] += warnings_in(r.stderr or "", f"check_run_meta_{s}")
    logs["repeat_e3"] = base / "repeat_e3.log"
    doc["repeat"] = run_child(["_repeat", *common], base / "repeat_e3.json", logs["repeat_e3"], 3600)
    for s in ("e2", "e3"):
        logs[f"cross_{s}"] = base / f"cross_{s}.log"
        doc["cross"][s] = run_child(["_cross", s, *common, "--code-root", str(old)], base / f"cross_{s}.json",
                                    logs[f"cross_{s}"], 3600)
    logs["ce_rate"] = base / "ce_rate.log"
    doc["ce_rate"] = run_child(["_ce_rate"], base / "ce_rate.json", logs["ce_rate"], 600)
    scratch.mkdir(parents=False)
    try:
        doc["refusals"] = run_refusals(scratch / "refusals", synthetic_root(scratch),
                                       str(throwaway_teacher(scratch)), a.teacher_ckpt_sha256,
                                       num_workers=pd.NUM_WORKERS)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for name, log in logs.items():
        doc["warnings"] += log_warnings(log, name)
    for r in doc["refusals"]:
        doc["warnings"] += r.get("warnings") or []
    e2 = doc["stages"].get("e2") or {}
    doc.update(torch=e2.get("torch"), cudnn=e2.get("cudnn"), gpu_name=e2.get("gpu_name"),
               throughput={s: throughput(v) for s, v in doc["stages"].items()})
    rows, verdict = evaluate(doc, pd.teacher_identity_problems)
    doc.update(rows=rows, verdict=verdict)
    out = evidence / f"kd_smoke_{stamp}.json"
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['status']:5} {r['check']:18} {r['detail'][:220]}")
    for r in rows:
        if r["status"] == "STOP":
            print(f"STOP: {r['check']}: {r['detail'][:400]}")
    print(f"written: {out}")
    print(f"RESULT: {verdict}")
    return 0 if verdict == "PASS" else 1


# -------------------------------------------------------------------------------------- the selfcheck
SELF_RESULTS: list = []


def scheck(name: str, ok, detail="") -> None:
    SELF_RESULTS.append((name, bool(ok), "" if ok else str(detail)[:500]))


def synthetic_doc() -> dict:
    """A child-result document that meets every criterion (the evaluator's PASS form)."""
    sys.path.insert(0, str(REPO))
    import scripts.preflight_distill as pd
    ckpt = "/workspace/teacher/iter_24000.pth"
    prov = {"builder": pd.TEACHER_BUILDER, "ckpt_path": ckpt, "ckpt_sha256": pd.TEACHER_SHA256,
            "ckpt_bytes": pd.TEACHER_BYTES, "expected_sha256": pd.TEACHER_SHA256, "config_path": str(TEACHER_CONFIG),
            **pd.TEACHER_VALUES}
    teacher = {"ok": True, "sha256": pd.TEACHER_SHA256, "bytes": pd.TEACHER_BYTES, "provenance": prov,
               "params": pd.TEACHER_PARAMS, "trainable": 0, "training_modules": [],
               "state_entries": pd.TEACHER_STATE_ENTRIES, "ckpt_path": ckpt, "file_name": pd.TEACHER_FILE}
    seq = [f"{i:064x}" for i in range(1, STEPS + 1)]

    def stage(s, *, steps=STEPS, kind="stage"):
        rows = []
        for it in range(1, steps + 1):
            loss = 5.0 + 0.001 * it                      # a CE + Dice sized loss: relative and absolute differ
            r = {"event": "train", "iter": it, "loss": loss, "sup": loss if it == 1 else loss - 0.01, "ce": 4.1,
                 "dice": 0.9}
            r.update({t: 0.37 for t in TERM_KEYS[s]})   # the unramped parts: not 0 at step 1
            r.update(ramp=(it - 1) / RAMP_DENOMINATOR, lr=0.01, grad_norm=1.5, grad_norm_student=1.5)
            if "cwd_feat" in TERM_KEYS[s]:
                r["grad_norm_projection"] = 0.0 if it == 1 else 0.2
            r.update(wall_clock=1.0 * it, iter_seconds=1.2, samples_per_sec=13.3)
            rows.append(r)
        return {"kind": kind, "stage": s, "child_rc": 0, "run_rc": 0,
                "determinism": {"deterministic_algorithms": True, "warn_only": True, "cudnn_deterministic": True,
                                "cudnn_benchmark": False, "cublas_after_set_seed": ":4096:8"},
                "tf32": {"x": 1}, "tf32_defaults": {"x": 1}, "teacher": teacher,
                "cuda_initialized_after_teacher_load": False, "cuda_initialized_before_run": False,
                "cuda_initialized_after_first_forward": True,
                "data_root": {"train": {"images": TRAIN_COUNT, "masks": TRAIN_COUNT},
                              "val": {"images": VAL_COUNT, "masks": VAL_COUNT}},
                "shapes": dict(SHAPES), "interpolate_calls": 0,
                "nmf": {"forwards": [{"draws_delta": 1, "state_sha256": seq[i], "rng_unchanged": True}
                                     for i in range(steps)], "final_draws": steps},
                "hard_checks": {k: True for k in HARD_CHECKS}, "train_rows": rows,
                "val_row": {"iter": steps, "all_class_miou": 0.012, "val_batches": VAL_BATCHES,
                            "val_total_px": VAL_TOTAL_PX, "val_seconds": 60.0},
                "memory": {"max_memory_reserved": 30 * 10**9}, "check_run_meta_rc": 0,
                "next_seconds": [0.01] * steps, "teacher_forward_ms": [300.0] * steps, "hook_seconds": [0.0] * steps,
                "first_batch_sha256": "f" * 64, "student_sha256": [f"{i:064x}" for i in range(steps)]
                if s == "e3" and kind != "cross" else None,
                "train_distill_sha256": OLD_TRAIN_DISTILL_SHA256 if kind == "cross" else "9" * 64}
    return {"environment": {"image_digest": pd.KD_IMAGE_DIGEST, "kd_image_digest": pd.KD_IMAGE_DIGEST,
                            "head": "c" * 40, "expect_head": "c" * 40, "governed_dirty": [], "plantseg_git_commit": None},
            "stages": {s: stage(s) for s in STAGE_KEYS}, "repeat": stage("e3", kind="repeat"),
            "cross": {s: stage(s, steps=CROSS_STEPS, kind="cross") for s in ("e2", "e3")},
            "ce_rate": {"trials": 50, "distinct": 1, "child_rc": 0},
            "refusals": [{"code": c, "rc": 2, "overrun": False, "run_meta_files": [], "stderr": f"REFUSING: [{c}] x"}
                         for c in REFUSALS],
            "warnings": [{"category": "UserWarning", "message": f"{ALLOWED_NONDET_WARNING} does not have a "
                                                               "deterministic implementation, but you set 'torch.use_"
                                                               "deterministic_algorithms(True, warn_only=True)'"}]}


def selfcheck_evaluator() -> None:
    sys.path.insert(0, str(REPO))
    import scripts.preflight_distill as pd

    def verdict_of(edit) -> tuple:
        doc = synthetic_doc()
        edit(doc)
        rows, verdict = evaluate(doc, pd.teacher_identity_problems)
        return verdict, {r["check"]: r["status"] for r in rows}, {r["check"]: r["detail"] for r in rows}

    v, st, _ = verdict_of(lambda d: None)
    scheck("s_good_results_pass", v == "PASS" and set(st.values()) <= {"PASS", "INFO"}, st)

    def case(name, check, edit, want="STOP"):
        v, st, _ = verdict_of(edit)
        scheck(name, st.get(check) == want and (want != "STOP" or v == "STOP"), f"{check}={st.get(check)} verdict={v}")

    def at(d, s):
        return d["stages"][s]

    def rows_of(d, s):
        return d["stages"][s]["train_rows"]
    case("s_children_a_crashed_child_stops", "children", lambda d: d["cross"]["e3"].update(child_rc=1, error="boom"))
    case("s_children_a_trainer_failing_its_checks_stops", "children", lambda d: at(d, "g").update(run_rc=1))
    # C5 workflow c5-1: a cross run of 73fd4d7 that failed only its CE-twice check (CUDA's nondeterministic CE
    # forward) is reported, not stopped; any other failed check of it stops
    case("s_cross_child_failing_only_the_old_ce_twice_check_passes", "children",
         lambda d: d["cross"]["e3"].update(run_rc=1, hard_checks={**d["cross"]["e3"]["hard_checks"],
                                                                   "supervised_never_ramped": False}), want="PASS")
    case("s_cross_child_failing_another_old_check_stops", "children",
         lambda d: d["cross"]["e3"].update(run_rc=1, hard_checks={**d["cross"]["e3"]["hard_checks"],
                                                                   "supervised_never_ramped": False,
                                                                   "teacher_feat_shape": False}))
    # round 4 (c5r2-2): the exception needs a failed check, and a child that did not crash
    case("s_cross_child_returning_1_with_no_failed_check_stops", "children",
         lambda d: d["cross"]["e3"].update(run_rc=1))
    # round 5 (c5r3-2): a child that wrote its whole record and then died (a signal, no error) or ran past its
    # limit (child_rc None) crashed: the exit code alone decides it
    case("s_crashed_cross_child_failing_only_the_ce_twice_check_stops", "children",
         lambda d: d["cross"]["e3"].update(child_rc=-11, run_rc=1,
                                           hard_checks={**d["cross"]["e3"]["hard_checks"],
                                                        "supervised_never_ramped": False}))
    case("s_children_a_child_timing_out_after_its_record_stops", "children",
         lambda d: at(d, "g").update(child_rc=None))
    # c5-9: each determinism flag, and PLANTSEG_GIT_COMMIT set
    for flag, value in (("deterministic_algorithms", False), ("warn_only", False), ("cudnn_deterministic", False)):
        case(f"s_determinism_{flag}_off_stops", "determinism",
             lambda d, flag=flag, value=value: at(d, "e2")["determinism"].update({flag: value}))
    case("s_environment_plantseg_git_commit_set_stops", "environment",
         lambda d: d["environment"].update(plantseg_git_commit="c" * 40))
    # c5-2: a non-finite E3 loss (the trainer writes it as null, with its nonfinite map) stops the repeat and the
    # losses rows and the evaluator still completes; an evaluator that cannot read a record gives its row a STOP
    try:
        v, st, det = verdict_of(lambda d: rows_of(d, "e3")[3].update(loss=None, nonfinite={"loss": "nan"}))
        ok, detail = (v == "STOP" and st.get("losses") == "STOP" and st.get("repeat_e3") == "STOP"
                      and "non-finite" in det.get("repeat_e3", "")), st
    except Exception as e:  # noqa: BLE001 - an escape is what this case catches
        ok, detail = False, f"raised {type(e).__name__}: {e}"
    scheck("s_a_null_e3_loss_stops_and_the_evaluator_completes", ok, detail)
    try:
        v, st, det = verdict_of(lambda d: rows_of(d, "a")[2].update(iter=None))
        ok, detail = v == "STOP" and any(k.endswith("_evaluator_error") and s == "STOP" for k, s in st.items()), st
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"raised {type(e).__name__}: {e}"
    scheck("s_an_evaluator_that_cannot_read_a_record_gives_a_stop_row", ok, detail)
    # c5-7 (PL-35: every warning is listed): a child's output, a refusal child's stderr included, gives its tagged
    # Python warning, its c10 line and any other '...Warning: ' line
    text = (WARN_TAG + json.dumps({"category": "UserWarning", "message": "a tagged warning"}) + "\n"
            "[W socket.cpp:436] a c10 warning\nno warning here\n"
            "/x/y.py:3: DeprecationWarning: an untagged warning\n")
    got_w = warnings_in(text, "refusal_seed")
    scheck("s_a_childs_output_gives_every_warning_with_its_source",
           [w.get("category") for w in got_w] == ["UserWarning", "log", "log"]
           and all(w.get("source") == "refusal_seed" for w in got_w), got_w)
    case("s_environment_other_digest_stops", "environment", lambda d: d["environment"].update(image_digest="sha256:0"))
    case("s_environment_head_not_the_pin_stops", "environment", lambda d: d["environment"].update(head="d" * 40))
    case("s_environment_governed_path_dirty_stops", "environment",
         lambda d: d["environment"].update(governed_dirty=["src/training/train_distill.py"]))
    case("s_determinism_benchmark_on_stops", "determinism", lambda d: at(d, "e3")["determinism"].update(cudnn_benchmark=True))
    case("s_determinism_cublas_unset_after_set_seed_stops", "determinism",
         lambda d: at(d, "g")["determinism"].update(cublas_after_set_seed=None))
    case("s_tf32_changed_stops", "tf32", lambda d: at(d, "a").update(tf32={"x": 2}))
    case("s_teacher_853_state_entries_stops", "teacher_identity", lambda d: at(d, "f")["teacher"].update(state_entries=853))
    case("s_teacher_other_file_name_stops", "teacher_identity", lambda d: at(d, "e2")["teacher"].update(file_name="x.pth"))
    case("s_teacher_other_sha256_stops", "teacher_identity", lambda d: at(d, "e3")["teacher"].update(sha256="0" * 64))
    case("s_cuda_initialised_before_run_stops", "cuda_order", lambda d: at(d, "e2").update(cuda_initialized_before_run=True))
    case("s_cuda_initialised_by_the_teacher_load_stops", "cuda_order",
         lambda d: at(d, "a").update(cuda_initialized_after_teacher_load=True))
    case("s_data_root_other_counts_stops", "data_root", lambda d: at(d, "e2")["data_root"]["val"].update(images=845))
    case("s_shapes_other_teacher_grid_stops", "shapes",
         lambda d: at(d, "e3")["shapes"].update(teacher_logits=[16, 116, 128, 128]))
    case("s_an_interpolate_call_stops", "interpolation", lambda d: at(d, "e3").update(interpolate_calls=1))
    case("s_two_nmf_draws_in_a_step_stops", "nmf_stream", lambda d: at(d, "e2")["nmf"]["forwards"][3].update(draws_delta=2))
    case("s_nmf_sequences_differing_across_stages_stops", "nmf_stream",
         lambda d: at(d, "g")["nmf"]["forwards"][7].update(state_sha256="0" * 64))
    case("s_rng_changed_by_a_teacher_forward_stops", "nmf_stream",
         lambda d: at(d, "a")["nmf"]["forwards"][0].update(rng_unchanged=False))
    case("s_step1_distill_not_zero_stops", "step_1", lambda d: at(d, "e2")["hard_checks"].update(distill_zero_at_step1=False))
    case("s_step1_loss_not_sup_bitwise_stops", "step_1",
         lambda d: rows_of(d, "e3")[0].update(loss=math.nextafter(rows_of(d, "e3")[0]["sup"], 10.0)))
    case("s_step1_ramp_not_zero_stops", "step_1", lambda d: rows_of(d, "g")[0].update(ramp=1e-12))
    case("s_step1_projection_norm_not_zero_stops", "step_1", lambda d: rows_of(d, "f")[0].update(grad_norm_projection=0.1))
    case("s_step1_failed_hard_check_stops", "step_1", lambda d: at(d, "g")["hard_checks"].update(teacher_feat_shape=False))
    case("s_ramp_one_ulp_off_stops", "steps_25",
         lambda d: rows_of(d, "e2")[9].update(ramp=math.nextafter(9 / RAMP_DENOMINATOR, 1.0)))
    case("s_24_steps_stop", "steps_25", lambda d: rows_of(d, "a").pop())
    case("s_a_null_value_stops", "steps_25", lambda d: rows_of(d, "e3")[11].update(grad_norm_student=None))
    case("s_repeat_missing_stops", "repeat_e3", lambda d: d["repeat"].update(train_rows=[]))
    v, st, det = verdict_of(lambda d: d["repeat"]["train_rows"][1].update(loss=d["repeat"]["train_rows"][1]["loss"] + 1e-9))
    scheck("s_repeat_not_bitwise_is_informative", st.get("repeat_e3") == "INFO" and "not bitwise" in det["repeat_e3"]
           and v == "PASS", (st, det.get("repeat_e3")))

    # PL-34 and PL-39 case 37: tolerance 1e-6 vs 2e-6 relative (losses near 5, so an absolute rule differs);
    # a bitwise repeat with a one-ulp cross-commit difference
    def cross_rel(rel, *, bitwise_repeat=False):
        def edit(d):
            x = d["stages"]["e2"]["train_rows"][4]["loss"]
            d["cross"]["e2"]["train_rows"][4]["loss"] = x * (1 + rel)
            if not bitwise_repeat:
                d["repeat"]["train_rows"][7]["loss"] += 1e-9
        return edit
    case("s_cross_commit_0_9e_6_relative_passes", "cross_commit", cross_rel(0.9e-6), want="PASS")
    case("s_cross_commit_2e_6_relative_stops", "cross_commit", cross_rel(2e-6))
    # C5 workflow c5-9: a relative difference equal to the tolerance itself passes (PL-34: at most 1e-6); the
    # tolerance is set, for this case only, to the very value the evaluator computes
    d = synthetic_doc()
    cross_rel(0.9e-6)(d)
    x, y = d["stages"]["e2"]["train_rows"][4]["loss"], d["cross"]["e2"]["train_rows"][4]["loss"]
    g = globals()
    saved_tol, g["CROSS_REL_TOL"] = g["CROSS_REL_TOL"], abs(x - y) / max(abs(x), abs(y), 1e-30)
    try:
        st_tol = {r["check"]: r["status"] for r in evaluate(d, pd.teacher_identity_problems)[0]}
    finally:
        g["CROSS_REL_TOL"] = saved_tol
    scheck("s_cross_commit_at_the_tolerance_itself_passes", st_tol.get("cross_commit") == "PASS", st_tol)
    v, st, _ = verdict_of(lambda d: d["cross"]["e3"]["train_rows"][2].update(
        loss=math.nextafter(d["stages"]["e3"]["train_rows"][2]["loss"], 10.0)))
    scheck("pl39_37_bitwise_repeat_with_a_one_ulp_cross_commit_difference_stops",
           st.get("cross_commit") == "STOP" and st.get("repeat_e3") == "INFO" and v == "STOP", st)
    v, st, _ = verdict_of(lambda d: (d["repeat"]["train_rows"][1].update(loss=d["repeat"]["train_rows"][1]["loss"] + 1e-9),
                                     d["cross"]["e3"]["train_rows"][2].update(
                                         loss=math.nextafter(d["stages"]["e3"]["train_rows"][2]["loss"], 10.0))))
    scheck("s_one_ulp_cross_commit_difference_without_a_bitwise_repeat_passes", st.get("cross_commit") == "PASS", st)
    case("s_cross_commit_first_batch_differing_stops", "cross_commit",
         lambda d: d["cross"]["e2"].update(first_batch_sha256="0" * 64))
    case("s_cross_commit_old_side_not_73fd4d7_stops", "cross_commit",
         lambda d: d["cross"]["e3"].update(train_distill_sha256="1" * 64))
    # PL-33, PL-39 case 38 and P8: 40/42/45 GB and one byte above each threshold, as literal byte counts (not from
    # the constants, so thresholds read in another unit fail here)
    for label, val, want in (("40e9", 40_000_000_000, "PASS"), ("40e9_plus_1_byte", 40_000_000_001, "FLAG"),
                             ("42e9", 42_000_000_000, "FLAG"), ("44e9", 44_000_000_000, "FLAG"),
                             ("44e9_plus_1_byte", 44_000_000_001, "STOP"), ("45e9", 45_000_000_000, "STOP")):
        v, st, _ = verdict_of(lambda d, val=val: at(d, "e3")["memory"].update(max_memory_reserved=val))
        scheck(f"pl39_38_memory_{label}_is_{want.lower()}", st.get("peak_memory") == want, st.get("peak_memory"))
    case("s_memory_not_recorded_stops", "peak_memory", lambda d: at(d, "g").update(memory={}))
    # PL-38: E3 above 2.0 s/iter
    v, st, det = verdict_of(lambda d: [r.update(iter_seconds=2.1) for r in rows_of(d, "e3")])
    scheck("s_e3_at_2_1_s_per_iter_flags_with_pl38s_message",
           st.get("throughput") == "FLAG" and PL38_MESSAGE in det["throughput"] and v == "FLAG", det.get("throughput"))
    case("s_e3_at_2_0_s_per_iter_passes", "throughput", lambda d: [r.update(iter_seconds=2.0) for r in rows_of(d, "e3")],
         want="PASS")
    case("s_throughput_without_iter_seconds_stops", "throughput", lambda d: [r.pop("iter_seconds") for r in rows_of(d, "a")])
    case("s_val_pass_52_batches_stops", "val_full_pass", lambda d: at(d, "e2")["val_row"].update(val_batches=52))
    case("s_val_pass_other_pixel_count_stops", "val_full_pass",
         lambda d: at(d, "e2")["val_row"].update(val_total_px=VAL_TOTAL_PX - 1))
    case("s_schema_missing_key_stops", "schema", lambda d: rows_of(d, "g")[3].pop("samples_per_sec"))
    case("s_schema_check_run_meta_failing_stops", "schema", lambda d: at(d, "f").update(check_run_meta_rc=1))
    case("s_schema_no_val_row_stops", "schema", lambda d: at(d, "a").update(val_row=None))
    # PL-36 and PL-39 case 40: a refusal child that returns 0, one that overruns, one that leaves a run_meta file
    for label, upd in (("returning_0", {"rc": 0}), ("overrunning_its_limit", {"rc": None, "overrun": True}),
                       ("leaving_a_run_meta_file", {"run_meta_files": ["e2_run_meta.jsonl"]})):
        case(f"pl39_40_refusal_child_{label}_stops", "gate_refusals", lambda d, upd=upd: d["refusals"][3].update(upd))
    case("s_refusal_child_refused_with_another_code_stops", "gate_refusals",
         lambda d: d["refusals"][5].update(stderr="REFUSING: [other] x"))
    case("s_refusal_child_missing_stops", "gate_refusals", lambda d: d["refusals"].pop())
    # PL-35 and PL-39 case 39
    case("pl39_39_a_warning_not_about_determinism_flags", "warnings",
         lambda d: d["warnings"].append({"category": "UserWarning", "message": "some deprecation"}), want="FLAG")
    case("pl39_39_another_nondeterminism_warning_stops", "warnings",
         lambda d: d["warnings"].append({"category": "UserWarning", "message": "upsample_bilinear2d_backward_out_cuda "
                                                                              "does not have a deterministic implementation"}))
    case("s_a_non_finite_loss_stops", "losses", lambda d: rows_of(d, "e2")[5].update(loss=math.inf))
    case("s_a_non_finite_cross_commit_loss_stops", "losses", lambda d: d["cross"]["e2"]["train_rows"][5].update(sup=math.nan))


def selfcheck_children(work: Path) -> None:
    sys.path.insert(0, str(REPO))
    import scripts.preflight_distill as pd
    root = synthetic_root(work)
    teacher = throwaway_teacher(work)
    recs = {r["code"]: r for r in run_refusals(work / "refusals", root, str(teacher), pd.TEACHER_SHA256,
                                               num_workers=pd.NUM_WORKERS)}
    for code in REFUSALS:
        r = recs.get(code) or {}
        scheck(f"s_refusal_{code}_exits_2_with_its_code_and_writes_no_run_meta",
               r.get("rc") == 2 and f"[{code}]" in (r.get("stderr") or "") and not r.get("run_meta_files")
               and not r.get("overrun"),
               f"rc={r.get('rc')} run_meta={r.get('run_meta_files')} {(r.get('stderr') or '')[-300:]}")
    scheck("s_refusal_children_records_keep_their_warnings",
           len(recs) == len(REFUSALS) and all(isinstance(r.get("warnings"), list) for r in recs.values()),
           {c: type(r.get("warnings")).__name__ for c, r in recs.items()})
    # PL-39 case 41: an --old-root with another train_distill.py is refused before anything runs
    old = work / "old_root" / "src" / "training"
    old.mkdir(parents=True)
    (old / "train_distill.py").write_text("# not 73fd4d7's file\n", encoding="utf-8")
    ev = work / "evidence"
    ev.mkdir()
    buf = io.StringIO()
    g = globals()
    saved = g["run_child"], g["run_refusals"]

    child_calls = []

    def no_child(*args, **_k):                  # past the refusals, no child runs on this CPU (C5 workflow c5-9)
        child_calls.append(args)
        raise RuntimeError("the driver went on to launch a child")
    g["run_child"] = g["run_refusals"] = no_child
    try:
        with contextlib.redirect_stdout(buf):
            rc = cmd_run(argparse.Namespace(old_root=str(work / "old_root"), evidence=str(ev), expect_head="c" * 40,
                                            teacher_ckpt=str(teacher), teacher_ckpt_sha256=pd.TEACHER_SHA256))
    except RuntimeError as e:
        rc = f"raised: {e}"
    finally:
        g["run_child"], g["run_refusals"] = saved
    scheck("pl39_41_old_root_with_another_train_distill_is_refused",
           rc == 2 and "REFUSED [old_root]" in buf.getvalue() and not any(ev.iterdir()), f"{rc} {buf.getvalue()[-300:]}")
    # C5 workflow c5-4 and c5-6, and rounds 4 and 5 (c5r2-*, c5r3-*): past the old-root check, a LOG_EVERY other
    # than the gate's, an --expect-head that is not HEAD, an environment not as required and teacher arguments other
    # than the teacher of record are refused (RESULT: REFUSED, exit 2) before any child runs: a wrong pin, shell or
    # teacher path would cost the wave (KD.6 makes a STOP the wave's). The checkout's status, the GPU probe and the
    # children are stubbed: the selfcheck runs no git command on the checkout.
    good_old = work / "old_root_ok" / "src" / "training" / "train_distill.py"
    good_old.parent.mkdir(parents=True)
    good_old.write_text("# stands for 73fd4d7's file (its sha256 is made the expected one)\n", encoding="utf-8")
    env_names = ("PLANTSEG_IMAGE_DIGEST", "PLANTSEG_GIT_COMMIT")

    def past_old_root(label: str, expect_head: str, *, head_value: str = "d" * 40, digest=pd.KD_IMAGE_DIGEST,
                      sha: str = pd.TEACHER_SHA256, teacher_path: str | None = None, git_commit: str | None = None,
                      evidence: Path | None = None, **over) -> tuple:
        ev_l = work / f"evidence_{label}" if evidence is None else evidence      # a given path is never created
        if evidence is None:
            ev_l.mkdir()
        names = ("run_child", "run_refusals", "OLD_TRAIN_DISTILL_SHA256", "LOG_EVERY", "git_out", "scoped_status",
                 "gpu_driver", *over)
        keep = {k: g[k] for k in dict.fromkeys(names)}
        env_keep = {k: os.environ.get(k) for k in env_names}
        real_git_out = keep["git_out"]
        child_calls.clear()
        stubs = {"run_child": no_child, "run_refusals": no_child, "OLD_TRAIN_DISTILL_SHA256": sha256_file(good_old),
                 "git_out": lambda *args: head_value if args == ("rev-parse", "HEAD") else real_git_out(*args),
                 "scoped_status": lambda _pd: subprocess.CompletedProcess([], 0, b"", b""),
                 "gpu_driver": lambda: "stub"}
        stubs.update(over)
        g.update(stubs)
        os.environ.pop("PLANTSEG_GIT_COMMIT", None)
        if git_commit is not None:
            os.environ["PLANTSEG_GIT_COMMIT"] = git_commit
        if digest is None:
            os.environ.pop("PLANTSEG_IMAGE_DIGEST", None)
        else:
            os.environ["PLANTSEG_IMAGE_DIGEST"] = digest
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                got = cmd_run(argparse.Namespace(old_root=str(work / "old_root_ok"), evidence=str(ev_l),
                                                 expect_head=expect_head, teacher_ckpt=teacher_path or str(teacher),
                                                 teacher_ckpt_sha256=sha))
        except Exception as e:  # noqa: BLE001 - the stub's escape, or any other, is the case's result
            got = f"raised: {e}"
        finally:
            g.update(keep)
            for k, v in env_keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return got, out.getvalue(), sorted(p.name for p in ev_l.iterdir()) if ev_l.is_dir() else None
    rc, out, files = past_old_root("log_every", "d" * 40, LOG_EVERY=pd.LOG_EVERY + 1)
    scheck("s_log_every_other_than_the_gates_is_refused_before_any_child",
           rc == 2 and "REFUSED [values_of_record]" in out and files == [], f"{rc} {out[-300:]} {files}")
    # at the C5 commit: the two path refusals before any child. An evidence folder inside the checkout (a path
    # there that does not exist, never created) and a scratch path that SL-1 refuses (the temporary root stubbed
    # to a name holding "test"; past the refusal the driver would create only the evidence folder's run folder)
    inside = REPO / "kd_smoke_evidence_inside_the_checkout"
    rc, out, files = past_old_root("in_repo", "d" * 40, evidence=inside)
    scheck("s_an_evidence_folder_inside_the_checkout_is_refused_before_any_child",
           rc == 2 and "REFUSED [evidence_in_repo]" in out and files is None and not child_calls,
           f"{rc} {out[-300:]} {files}")
    real_gettempdir = tempfile.gettempdir
    tempfile.gettempdir = lambda: str(work / "kd_tmp_test")
    try:
        rc, out, files = past_old_root("sl1_path", "d" * 40)
    finally:
        tempfile.gettempdir = real_gettempdir
    scheck("s_a_scratch_path_that_sl1_refuses_is_refused_before_any_child",
           rc == 2 and "REFUSED [sl1_path]" in out and files == [] and not child_calls,
           f"{rc} {out[-300:]} {files}")
    rc, out, files = past_old_root("head_other", "c" * 40)
    scheck("s_expect_head_other_than_head_is_refused_before_any_child",
           rc == 2 and "REFUSED [expect_head]" in out and "RESULT: REFUSED (expect_head)" in out and files == [],
           f"{rc} {out[-300:]} {files}")
    rc, out, files = past_old_root("head_upper", "D" * 40, head_value="D" * 40)
    scheck("s_expect_head_not_40_lowercase_hex_is_refused_even_when_it_is_head",
           rc == 2 and "REFUSED [expect_head]" in out and files == [], f"{rc} {out[-300:]} {files}")
    rc, out, files = past_old_root("environment", "d" * 40, digest=None)
    scheck("s_environment_not_as_required_is_refused_before_any_child",
           rc == 2 and "REFUSED [environment]" in out and "image digest" in out
           and "PLANTSEG_IMAGE_DIGEST None" in out and files == [], f"{rc} {out[-300:]} {files}")
    # round 6 (c5r4-3): a scoped status that git cannot read is named in the environment refusal, not read as dirt
    rc, out, files = past_old_root("status_failed", "d" * 40,
                                   scoped_status=lambda _pd: subprocess.CompletedProcess([], 128, b"", b""))
    scheck("s_a_status_git_cannot_read_is_named_in_the_environment_refusal",
           rc == 2 and "REFUSED [environment]" in out and "git status failed" in out and files == [],
           f"{rc} {out[-300:]} {files}")
    # round 7 (c5r5-4): PLANTSEG_GIT_COMMIT set in the shell is named in the environment refusal
    rc, out, files = past_old_root("git_commit_set", "d" * 40, git_commit="abc1234")
    scheck("s_plantseg_git_commit_set_is_named_in_the_environment_refusal",
           rc == 2 and "REFUSED [environment]" in out and "PLANTSEG_GIT_COMMIT 'abc1234'" in out and files == [],
           f"{rc} {out[-300:]} {files}")
    # round 5 (c5r3-5): a HEAD that git cannot read is named as such; (c5r3-6, and round 6: c5r4-1, c5r4-2) teacher
    # arguments other than the teacher of record are refused before any child, which would otherwise fail at the
    # teacher load or STOP at the teacher_identity row: another sha256, a file named otherwise, no file, and a path
    # named iter_24000.pth that resolves to a file named otherwise (the children load the resolved file)
    rc, out, files = past_old_root("head_none", "d" * 40, head_value=None)
    scheck("s_a_head_git_cannot_read_is_named_and_refused_before_any_child",
           rc == 2 and "REFUSED [expect_head]" in out and "git rev-parse HEAD failed" in out and files == [],
           f"{rc} {out[-300:]} {files}")
    rc, out, files = past_old_root("teacher", "d" * 40, sha="0" * 64)
    scheck("s_teacher_arguments_other_than_the_teacher_of_record_are_refused_before_any_child",
           rc == 2 and "REFUSED [teacher]" in out and f"--teacher-ckpt-sha256 '{'0' * 64}'" in out and files == []
           and not child_calls, f"{rc} {out[-300:]} {files}")
    named_other = work / "teacher_named_other" / "model.pth"
    named_other.parent.mkdir()
    named_other.write_bytes(b"a file named otherwise\n")
    link = work / "teacher_link" / pd.TEACHER_FILE
    link.parent.mkdir()
    link.symlink_to(named_other)
    # round 7 (c5r5-1, -2, -3): a folder named iter_24000.pth (a regular file is required, not any path), a path
    # whose stat fails (a component longer than NAME_MAX), and a path named otherwise that resolves to iter_24000.pth
    folder = work / "teacher_folder" / pd.TEACHER_FILE
    folder.mkdir(parents=True)
    alias = work / "teacher_alias" / "model.pth"
    alias.parent.mkdir()
    alias.symlink_to(teacher)
    for name, label, path, want in (
            ("s_a_teacher_file_named_otherwise_is_refused_before_any_child", "teacher_named_other", named_other,
             "file name 'model.pth'"),
            ("s_a_teacher_path_with_no_file_is_refused_before_any_child", "teacher_absent",
             work / "teacher_absent" / pd.TEACHER_FILE, "no file at"),
            ("s_a_teacher_path_resolving_to_a_file_named_otherwise_is_refused_before_any_child", "teacher_link",
             link, "resolves to"),
            ("s_a_teacher_path_that_is_a_folder_is_refused_before_any_child", "teacher_folder", folder,
             "no file at"),
            ("s_a_teacher_path_that_cannot_be_read_is_refused_before_any_child", "teacher_unreadable",
             work / ("x" * 256) / pd.TEACHER_FILE, "cannot be read (OSError)"),
            ("s_a_teacher_path_named_otherwise_resolving_to_iter_24000_is_refused_before_any_child", "teacher_alias",
             alias, "file name 'model.pth'")):
        rc, out, files = past_old_root(label, "d" * 40, teacher_path=str(path))
        scheck(name, rc == 2 and "REFUSED [teacher]" in out and want in out and files == [] and not child_calls,
               f"{rc} {out[-300:]} {files}")
    rc, out, files = past_old_root("head_ok", "d" * 40)
    first = list(child_calls[0][0]) if child_calls and child_calls[0] else []
    scheck("s_expect_head_equal_to_head_goes_on_to_the_children_with_the_gates_workers",
           rc == "raised: the driver went on to launch a child" and "REFUSED" not in out
           and first[:2] == ["_stage", "e2"] and "--num-workers" in first
           and first[first.index("--num-workers") + 1] == str(pd.NUM_WORKERS), f"{rc} {first} {out[-300:]}")
    # round 4 (c5r2-3) and round 5 (c5r3-1, c5r3-3): the driver end to end with its children stubbed: every child
    # that reads data (the 8 trainer children and the refusal children; _ce_rate reads none) gets the gate's
    # workers, and the warnings of a trainer child's log, of check-run-meta (its WARN lines and its stderr) and of
    # the refusal children reach the JSON it writes
    good = synthetic_doc()
    argvs = []
    ref_kw = {}

    def fake_child(argv, out_json, log, timeout):
        argvs.append(list(argv))
        log.write_text(WARN_TAG + json.dumps({"category": "UserWarning", "message": "a synthetic child warning"})
                       + "\n" if argv[:2] == ["_stage", "e3"] else "", encoding="utf-8")
        if argv[0] == "_stage":
            return json.loads(json.dumps(good["stages"][argv[1]]))
        if argv[0] == "_repeat":
            return json.loads(json.dumps(good["repeat"]))
        if argv[0] == "_cross":
            return json.loads(json.dumps(good["cross"][argv[1]]))
        return dict(good["ce_rate"])
    refusal_recs = [dict(r, warnings=[]) for r in good["refusals"]]
    refusal_recs[0]["warnings"] = [{"category": "log", "message": "SomeWarning: from a refusal child",
                                    "source": f"refusal_{refusal_recs[0]['code']}"}]
    rc, out, files = past_old_root(
        "driver", "d" * 40, run_child=fake_child, run_refusals=lambda *_a, **k: ref_kw.update(k) or refusal_recs,
        run_check_run_meta=lambda cmd: subprocess.CompletedProcess(
            cmd, 0, "WARN: a synthetic check-run-meta warning\n", "UserWarning: a synthetic stderr warning\n"),
        synthetic_root=lambda folder: folder, throwaway_teacher=lambda folder: folder / "iter_24000.pth")
    written = [f for f in files if f.endswith(".json")]
    doc = json.loads((work / "evidence_driver" / written[0]).read_text(encoding="utf-8")) if len(written) == 1 else {}
    warns = doc.get("warnings") or []
    trainer_argvs = [v for v in argvs if v[0] in ("_stage", "_repeat", "_cross")]
    scheck("s_driver_hands_the_gates_workers_to_each_child_that_reads_data_and_lists_every_warning",
           len(trainer_argvs) == 8 and all("--num-workers" in v and v[v.index("--num-workers") + 1]
                                           == str(pd.NUM_WORKERS) for v in trainer_argvs)
           and ref_kw.get("num_workers") == pd.NUM_WORKERS
           and any(w.get("source") == "stage_e3" and "synthetic child warning" in str(w.get("message"))
                   for w in warns)
           and any(w.get("category") == "check-run-meta" and w.get("source") == "check_run_meta_e2" for w in warns)
           and any("synthetic stderr warning" in str(w.get("message")) and w.get("source") == "check_run_meta_e2"
                   for w in warns)
           and any(w.get("source") == f"refusal_{refusal_recs[0]['code']}" for w in warns),
           f"rc={rc} files={files} argvs={len(trainer_argvs)} refusal_kw={ref_kw} warnings={warns[:6]}")
    # round 4 (c5r2-1): the checkout's status is read-only (-C <checkout>, --no-optional-locks)
    seen = []
    real_run = subprocess.run
    subprocess.run = lambda argv, **_kw: seen.append(list(argv)) or subprocess.CompletedProcess(argv, 0, b"", b"")
    try:
        scoped_status(pd)
    finally:
        subprocess.run = real_run
    scheck("s_scoped_status_is_read_only_on_the_checkout",
           len(seen) == 1 and seen[0][1:4] == ["-C", str(REPO), "--no-optional-locks"], seen)
    # round 4 (c5r2-3): a refusal child's record lists the warnings of its stdout and its stderr
    rr = refusal_result("seed", subprocess.CompletedProcess(
        [], 2, WARN_TAG + json.dumps({"category": "UserWarning", "message": "a tagged warning"}) + "\n",
        "[W 1007 x.cpp:1] a c10 warning\n"))
    scheck("s_a_refusal_childs_record_lists_the_warnings_of_its_stdout_and_stderr",
           rr["rc"] == 2 and sorted(w.get("category") for w in rr["warnings"]) == ["UserWarning", "log"]
           and all(w.get("source") == "refusal_seed" for w in rr["warnings"]), rr)
    # a stub-teacher _stage child: CPU, batch 2, 3 steps, synthetic batches (the schema of its record)
    out = work / "stub_e3.json"
    r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "_stage", "e3", "--stub", "--device", "cpu",
                        "--batch", "2", "--steps", "3", "--num-workers", "0", "--max-val-batches", "1",
                        "--out", str(out)], cwd=str(REPO), capture_output=True, text=True, timeout=1200,
                       env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1"))
    rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    keys = ("kind", "stage", "determinism", "tf32", "tf32_defaults", "cuda_initialized_after_teacher_load",
            "cuda_initialized_before_run", "cuda_initialized_after_first_forward", "shapes", "interpolate_calls",
            "nmf", "hard_checks", "train_rows", "val_row", "first_batch_sha256", "next_seconds", "hook_seconds",
            "teacher_forward_ms", "student_sha256", "run_rc", "warnings")
    scheck("s_stub_stage_child_writes_the_record_the_evaluator_reads",
           r.returncode == 0 and all(k in rec for k in keys) and rec.get("run_rc") == 0 and not rec.get("error")
           and len(rec.get("student_sha256") or []) == 3 and len(set(rec.get("student_sha256") or [])) == 3
           and rec["shapes"] == {"teacher_logits": [2, 116, 64, 64], "teacher_feat": [2, 320, 32, 32],
                                 "head_logits": [2, 116, 64, 64], "c5": [2, 160, 32, 32]}
           and rec.get("interpolate_calls") == 0 and len(rec["nmf"]["forwards"]) == 3
           and rec["hard_checks"] == {k: True for k in HARD_CHECKS} and rec["tf32"] == rec["tf32_defaults"]
           and rec["determinism"]["cublas_after_set_seed"] == ":4096:8" and len(rec["next_seconds"]) == 3,
           f"rc={r.returncode} missing={[k for k in keys if k not in rec]} error={str(rec.get('error'))[:300]} "
           f"{r.stdout[-300:]} {r.stderr[-600:]}")
    rows = rec.get("train_rows") or []
    scheck("s_stub_stage_child_train_rows_carry_e12s_keys_and_a_val_row",
           len(rows) == 3 and all(list(x) == train_keys("e3") for x in rows) and (rec.get("val_row") or {}).get("iter") == 3
           and rows[0]["ramp"] == 0.0 and rows[0]["loss"] == rows[0]["sup"], [list(x) for x in rows[:1]])


def cmd_selfcheck() -> int:
    print("=" * 78)
    print("KD STEP SMOKE --selfcheck (CPU; the evaluator, the refusals, a stub child; never a pod PASS)")
    print("=" * 78)
    work = Path(tempfile.gettempdir()).resolve() / f"kd_step_selfcheck_{os.getpid()}"
    if "test" in str(work).lower():
        print(f"REFUSED: {work} contains 'test' (SL-1)")
        return 2
    work.mkdir()
    try:
        for name, fn in (("evaluator", selfcheck_evaluator), ("children", lambda: selfcheck_children(work))):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - a crash is a failed check
                scheck(f"s_{name}_completed", False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-700:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("\n[CHECKS]")
    for name, ok, detail in SELF_RESULTS:
        print(f"  {name:74}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in SELF_RESULTS if ok)
    print(f"\nRESULT: {'PASS' if passed == len(SELF_RESULTS) else 'FAIL'} ({passed}/{len(SELF_RESULTS)})")
    return 0 if passed == len(SELF_RESULTS) else 1


# ------------------------------------------------------------------------------------------- the CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KD step smoke (lane 5, K2).", allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", allow_abbrev=False)
    for name in ("--expect-head", "--teacher-ckpt", "--teacher-ckpt-sha256", "--old-root", "--evidence"):
        r.add_argument(name, required=True)
    for name in ("_stage", "_repeat", "_cross"):
        s = sub.add_parser(name, allow_abbrev=False)
        if name != "_repeat":
            s.add_argument("stage", choices=STAGE_KEYS if name == "_stage" else ("e2", "e3"))
        s.add_argument("--out", required=True)
        s.add_argument("--teacher-ckpt", default=None)
        s.add_argument("--teacher-ckpt-sha256", default=None)
        s.add_argument("--code-root", default=None, required=name == "_cross")
        s.add_argument("--steps", type=int, default=CROSS_STEPS if name == "_cross" else STEPS)
        s.add_argument("--batch", type=int, default=BATCH)
        s.add_argument("--device", default="cuda")
        s.add_argument("--num-workers", type=int, required=True)    # the parent passes the gate's value
        s.add_argument("--max-val-batches", type=int, default=None if name == "_stage" else 1)
        if name == "_stage":
            s.add_argument("--stub", action="store_true")
    c = sub.add_parser("_ce_rate", allow_abbrev=False)
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cuda")
    c.add_argument("--trials", type=int, default=50)
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selfcheck"]:
        return cmd_selfcheck()
    a = build_parser().parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    rec: dict = {}
    try:
        if a.cmd == "_ce_rate":
            child_ce_rate(a, rec)
        else:
            if a.cmd == "_repeat":
                a.stage = "e3"
            a.stub = getattr(a, "stub", False)
            if a.stub == bool(a.teacher_ckpt and a.teacher_ckpt_sha256):
                print("usage error: a child takes --teacher-ckpt and --teacher-ckpt-sha256, or (_stage) --stub",
                      file=sys.stderr)
                return 2
            child_stage(a, a.cmd.lstrip("_"), rec)
    except Exception as e:  # noqa: BLE001 - the record says what failed; the evaluator stops on it
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
        print(rec["traceback"], file=sys.stderr)
    emit(a.out, rec)
    return 1 if rec.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
