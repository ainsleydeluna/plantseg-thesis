#!/usr/bin/env python3
"""W pod smoke: arm W, run stage E3-flip, measured on its pod before E3-flip launches (lane L-KD-WRAP, commit C2;
ARMS plan decision (viii) and section 4 item 5; Tier F audit F30 and F40; GO rulings K6 and K7).

Usage:
  python -B scripts/smoke_kd_flip_pod.py run --expect-head <P_K> --teacher-ckpt <T> --teacher-ckpt-sha256 <sha>
         --evidence <E> [--e3-reference <kd_smoke JSON>]
      On one A40 with the KD image of record, from the checkout of the arms pin P_K (HEAD is the pin, so
      --expect-head takes its full commit), with the teacher of record (W distils from it; item 6(a)'s hold ended at
      DL-84, 2026-10-06T16:01:05Z, before this smoke runs on real teacher weights: audit F30). --e3-reference is
      the lambda wave's kd_smoke JSON (scripts/smoke_kd_step.py run), whose throughput["e3"] holds E3's measured
      s/iter. Before any child runs it refuses (RESULT: REFUSED, exit 2, no STOP: correct and rerun), in this order:
      a path argument containing "test" (SL-1: as typed, before any filesystem call, then as it resolves), an evidence
      folder inside the checkout or not a folder, an output that already exists, another --expect-head, an environment
      not as the environment row requires (the KD digest exported, PLANTSEG_GIT_COMMIT unset, the governed paths
      clean), teacher arguments other than the teacher of record (smoke_kd_step's rule: its sha256, the file name as
      typed and of the file it resolves to, a regular file), a teacher file whose sha256 is not the one given (hashed
      here), and an E3 reference that is not a kd_smoke/1 JSON with a finite positive E3 median s/iter.
      One child then runs in a fresh process (E-17), `_stage_flip`: train_distill.run() in dry mode on CUDA with
      the registered arm e3-flip (kd_arms.load_arm, so the trainer makes FlipTargets and calls build() through the
      seam), the real teacher and the real TRAIN/VAL data, 25 steps at batch 16, the gate's 12 loader workers,
      val_interval 4000 and one VAL batch at step 25; no checkpoint is written and TEST is never read. Writes
      <E>/kd_flip_pod_smoke_<UTC>/ (the child's record and log) and <E>/kd_flip_pod_smoke_<UTC>.json; prints one row
      per criterion and RESULT: PASS, FLAG (a ruling is needed) or STOP.
  python -B scripts/smoke_kd_flip_pod.py --selfcheck
      In the cloud, CPU only; never a pod PASS. The plan's 16 cases: the evaluator on synthetic records, the
      refusals before any child, the evaluator's determinism and a one-draw rule mutation. One declared addition: a
      stub `_stage_flip` child (CPU, batch 2, 3 steps, synthetic batches, scripts/invariance_harness.py's stub teacher
      with its M4-KD stream) and the driver end to end on a PASS-form record, its HEAD, environment and teacher-file
      checks stubbed: the JSON it writes and its verdict.

Criteria (K7; plan (viii)): the child (exit 0, RESULT: PASS); the environment; the teacher of record's identity;
the step-1 checks; 25 finite steps; the M4-KD draws: exactly 2 per step, call 1 taking draw 2t-1 and call 2 draw
2t, each forward one draw with the CPU and CUDA RNG states unchanged, else STOP; peak memory: max_memory_reserved
<= 40e9 B PASS, <= 44e9 B FLAG, above STOP (smoke_kd_step's PL-33 thresholds); throughput: the median s/iter of
steps 6 to 25 above 2 x E3's measured median (the E3 reference) FLAGs, and with no E3 reference above the 4.0 s/iter
fallback (2 x PL-38's 2.0) FLAGs; run_meta's arm fields; warnings (smoke_kd_step's rule); losses finite.

The JSON (format kd_flip_pod_smoke/1) carries a top-level "throughput" map in smoke_kd_step's own shape:
throughput["e3-flip"] is smoke_kd_step.throughput() of the child's record, so throughput["e3-flip"]["iter_seconds"]
is the statistics dict (median, mean, max, n) of the trainer's iter_seconds over steps 6 to 25, and the launch gate
reads E3's figure (the kd_smoke JSON's throughput["e3"]["iter_seconds"]["median"]) and W's
(throughput["e3-flip"]["iter_seconds"]["median"]) with one reader and takes the larger for AM-19 item 3(b) (audit
F40). It also records max_memory_reserved, each call's teacher_forward_ms, the draws per step, the teacher's
identity, HEAD and the evidence folder.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The E3 pod smoke's evaluator parts and pod-side tools, reused unchanged (standard library only at import)
from scripts.smoke_kd_step import (E3_S_PER_ITER_FLAG, HARD_CHECKS, LOG_EVERY, MEM_PASS_BYTES,  # noqa: E402
                                   MEM_STOP_BYTES, RAMP_DENOMINATOR, THROUGHPUT_STEPS, eval_environment,
                                   eval_warnings, finite, git_out, gpu_driver, install_warning_log, load_teacher, row,
                                   scoped_status, sha256_file, stats, teacher_problems, throughput, train_keys,
                                   warnings_in)

FORMAT = "kd_flip_pod_smoke/1"
E3_REFERENCE_FORMAT = "kd_smoke/1"
ARM = "e3-flip"
RUN_STAGE = "E3-flip"
TARGET_CONSTRUCTION = {"K": 1, "views": 2}
STEPS, BATCH = 25, 16
DRAWS_PER_STEP = 2                     # K7: call 1 and call 2, one M4-KD draw each; any other count is a STOP
FLAG_FACTOR = 2.0                      # K7: s/iter above 2 x E3's measured median FLAGs
FALLBACK_S_PER_ITER = FLAG_FACTOR * E3_S_PER_ITER_FLAG     # 4.0: with no E3 reference
DL84_UTC = "2026-10-06T16:01:05Z"      # item 6(a)'s hold ended (audit F30)
STUB_STEPS, STUB_BATCH = 3, 2


# ---------------------------------------------------------------------------------------------- evaluator
def per_step_forwards(stage: dict) -> dict:
    """{step: [forward records]} from the child's teacher forwards."""
    out: dict = {}
    for f in (stage.get("nmf") or {}).get("forwards") or []:
        out.setdefault(f.get("step"), []).append(f)
    return out


def draws_per_step(stage: dict) -> list:
    """K6's draws per step: the M4-KD draws that each step's teacher forwards took, the sum of their draws_delta
    (None for a step with a forward whose draws are unknown), steps in order (review finding RIR-3: a forward count
    would read 2 where call 1 took two draws)."""
    by_step = per_step_forwards(stage)
    out = []
    for t in sorted(k for k in by_step if isinstance(k, int)):
        deltas = [f.get("draws_delta") for f in by_step[t]]
        out.append(sum(deltas) if all(isinstance(d, int) and not isinstance(d, bool) for d in deltas) else None)
    return out


def eval_children(doc: dict) -> dict:
    st = doc.get("stage") or {}
    ok = st.get("child_rc") == 0 and st.get("run_rc") == 0 and not st.get("error")
    return row("children", "PASS" if ok else "STOP",
               "the _stage_flip child exited 0 and its trainer run printed RESULT: PASS" if ok else
               f"child_rc={st.get('child_rc')} run_rc={st.get('run_rc')} error={str(st.get('error'))[:300]}")


def eval_teacher(doc: dict, problems_fn) -> dict:
    import scripts.preflight_distill as pd     # the values of record (PL-11)
    t = (doc.get("stage") or {}).get("teacher") or {}
    bad = list(problems_fn(t, t.get("ckpt_path")))
    if t.get("file_name") != pd.TEACHER_FILE:
        bad.append(f"file name {t.get('file_name')!r}")
    return row("teacher_identity", "STOP" if bad else "PASS", f"{bad[:4]}" if bad else
               f"the teacher of record (DL-88): sha256 {pd.TEACHER_SHA256[:12]}..., bytes, provenance, parameters, "
               f"none trainable, eval mode, 854 state entries, the file name {pd.TEACHER_FILE}")


def eval_step1(doc: dict) -> dict:
    st = doc.get("stage") or {}
    r = (st.get("train_rows") or [{}])[0]
    hard = st.get("hard_checks") or {}
    failed = [k for k in HARD_CHECKS if hard.get(k) is not True]
    ok = (r.get("iter") == 1 and r.get("ramp") == 0.0 and finite(r.get("loss")) and r.get("loss") == r.get("sup")
          and not failed and r.get("grad_norm_projection") == 0.0)
    return row("step_1", "PASS" if ok else "STOP",
               "ramp 0.0, loss == sup bitwise, the trainer's 20 checks true, grad_norm_projection 0.0" if ok else
               f"ramp={r.get('ramp')} loss={r.get('loss')} sup={r.get('sup')} failed_checks={failed}")


def eval_steps(doc: dict) -> dict:
    rows = (doc.get("stage") or {}).get("train_rows") or []
    nonfinite = [r.get("iter") for r in rows
                 if "nonfinite" in r or not all(finite(v) for k, v in r.items() if k != "event")]
    ramp_bad = [r.get("iter") for r in rows if r.get("ramp") != (r.get("iter", 0) - 1) / RAMP_DENOMINATOR]
    keys_bad = [r.get("iter") for r in rows if list(r) != train_keys("e3")]
    ok = [r.get("iter") for r in rows] == list(range(1, STEPS + 1)) and not (nonfinite or ramp_bad or keys_bad)
    return row("steps_25", "PASS" if ok else "STOP",
               f"{STEPS} rows with E-12's E3 keys, every value finite, ramp == (it-1)/{RAMP_DENOMINATOR}" if ok else
               f"rows={len(rows)} nonfinite={nonfinite[:5]} ramp={ramp_bad[:5]} keys={keys_bad[:5]}")


def eval_draws(doc: dict) -> dict:
    """K7: exactly DRAWS_PER_STEP M4-KD draws per step, call k of step t at position DRAWS_PER_STEP*(t-1) + k - 1,
    each forward one draw with the CPU and CUDA RNG states unchanged; else STOP."""
    st = doc.get("stage") or {}
    by_step = per_step_forwards(st)
    counts = {t: len(by_step.get(t, [])) for t in range(1, STEPS + 1)}
    order_bad, rng_bad, delta_bad = [], [], []
    for t in range(1, STEPS + 1):
        for k, f in enumerate(by_step.get(t, []), start=1):
            if f.get("draws_before") != DRAWS_PER_STEP * (t - 1) + k - 1:
                order_bad.append((t, k, f.get("draws_before")))
            if f.get("draws_delta") != 1:
                delta_bad.append((t, k, f.get("draws_delta")))
            if f.get("rng_unchanged") is not True:
                rng_bad.append((t, k))
    final = (st.get("nmf") or {}).get("final_draws")
    wrong = {t: n for t, n in counts.items() if n != DRAWS_PER_STEP}
    ok = not (wrong or order_bad or rng_bad or delta_bad) and final == DRAWS_PER_STEP * STEPS \
        and set(by_step) == set(range(1, STEPS + 1))
    return row("nmf_draws", "PASS" if ok else "STOP",
               f"{DRAWS_PER_STEP} draws per step (call 1 at 2t-1 on the crop, call 2 at 2t on the flip), one per "
               f"forward, final {final}, CPU and CUDA RNG states unchanged by every forward" if ok else
               f"steps with another count {dict(list(wrong.items())[:5])}; positions {order_bad[:4]}; draws per "
               f"forward {delta_bad[:4]}; RNG changed {rng_bad[:4]}; final {final}")


def eval_memory(doc: dict) -> dict:
    r = ((doc.get("stage") or {}).get("memory") or {}).get("max_memory_reserved")
    status = ("STOP" if not isinstance(r, int) or isinstance(r, bool) or r > MEM_STOP_BYTES
              else "FLAG" if r > MEM_PASS_BYTES else "PASS")
    return row("peak_memory", status, f"max_memory_reserved {r} B (PASS <= {MEM_PASS_BYTES}, FLAG <= {MEM_STOP_BYTES}, "
                                      "STOP above; K7, smoke_kd_step's PL-33 thresholds)")


def e3_median(doc: dict):
    ref = doc.get("e3_reference")
    return None if not ref else ref.get("iter_seconds_median")


def eval_throughput(doc: dict) -> dict:
    tp = throughput(doc.get("stage") or {})
    w = tp["iter_seconds"]["median"]
    n_want = THROUGHPUT_STEPS[1] - THROUGHPUT_STEPS[0] + 1
    if not finite(w) or tp["iter_seconds"]["n"] != n_want:
        return row("throughput", "STOP",
                   f"the trainer's iter_seconds of steps 6 to 25 are missing: {tp['iter_seconds']}")
    e3 = e3_median(doc)
    if e3 is None:
        limit, basis = FALLBACK_S_PER_ITER, f"no E3 reference: the {FALLBACK_S_PER_ITER} s/iter fallback"
    else:
        limit, basis = FLAG_FACTOR * e3, f"{FLAG_FACTOR:g} x E3's measured median {e3!r} s/iter"
    text = (f"E3-flip median {w:.3f} s/iter (steps 6-25; teacher {tp['teacher_share']} of it), limit {limit:.3f} "
            f"({basis})")
    if w > limit:
        return row("throughput", "FLAG", f"{text}: recompute the launch plan under AM-19 items 1(e), 2 and 3(b) "
                                         "before any GO; the dates do not move")
    return row("throughput", "PASS", text)


def eval_run_meta(doc: dict) -> dict:
    m = (doc.get("stage") or {}).get("run_meta") or {}
    want = {"arm": RUN_STAGE, "descriptive": True, "parent_of_e4_e7": False, "target_construction": TARGET_CONSTRUCTION}
    bad = {k: m.get(k, "<absent>") for k, v in want.items() if m.get(k, "<absent>") != v}
    if m and list(m)[-1] != "target_construction":
        bad["last key"] = list(m)[-1]
    return row("run_meta", "STOP" if bad or not m else "PASS", f"not as an E3-flip row: {bad}" if bad or not m else
               "arm E3-flip, descriptive, never a parent of E4-E7, target_construction {K 1, views 2} last")


def eval_losses(doc: dict) -> dict:
    bad = []
    for r in (doc.get("stage") or {}).get("train_rows") or []:
        keys = ["loss", *(k for k in ("sup", "ce", "dice", "logit_kd", "cwd_feat", "cwd_logit") if k in r)]
        if "nonfinite" in r or not all(finite(r.get(k)) for k in keys):
            bad.append(r.get("iter"))
    return row("losses", "STOP" if bad else "PASS", f"non-finite at {bad[:6]}" if bad else "every loss finite")


def evaluate(doc: dict, teacher_problems_fn) -> tuple:
    """Every criterion's row; an evaluator that raises gives its row a STOP with the error. The verdict: STOP if a row
    stops, else FLAG if a row flags, else PASS."""
    fns = (eval_children, eval_environment, lambda d: eval_teacher(d, teacher_problems_fn), eval_step1, eval_steps,
           eval_draws, eval_memory, eval_throughput, eval_run_meta, eval_warnings, eval_losses)
    rows = []
    for fn in fns:
        try:
            rows.append(fn(doc))
        except Exception as e:  # noqa: BLE001 - a record the evaluator cannot read is a STOP, never a traceback
            name = "teacher" if fn.__name__ == "<lambda>" else fn.__name__.removeprefix("eval_")
            rows.append(row(f"{name}_evaluator_error", "STOP", f"{type(e).__name__}: {e}"))
    statuses = {r["status"] for r in rows}
    return rows, "STOP" if "STOP" in statuses else "FLAG" if "FLAG" in statuses else "PASS"


# ----------------------------------------------------------------------------------------------- the child
def child_stage_flip(a, rec: dict) -> None:
    """One instrumented dry run of train_distill.run() with the registered arm e3-flip, recorded into `rec`."""
    rec.update(kind="stage_flip", arm=ARM, cublas_at_start=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), warnings=[])
    install_warning_log(rec)
    import torch
    import src.training.train_distill as td
    from src.training import kd_arms
    rec.update(train_distill_sha256=sha256_file(td.__file__), torch=torch.__version__)
    spec = kd_arms.load_arm(ARM)
    rec["arm_spec"] = {"path": Path(spec.path).name, "sha256": spec.sha256, "run_stage": spec.run_stage,
                       "target_construction": spec.target_construction,
                       "builder": None if spec.targets is None else ".".join(spec.targets)}
    stage = td.resolve_stage("e3")
    cuda = str(a.device).startswith("cuda")
    ck = Path(a.out).with_suffix("") / "e3_flip_s42_smoke"
    if a.stub:                                 # selfcheck: the harness's stub teacher (M4-KD stream) and batches
        from scripts.invariance_harness import build_stub_segnext
        from src.distill.segnext_teacher import SegNeXtTeacherAdapter
        teacher = td.FrozenTeacher(SegNeXtTeacherAdapter(build_stub_segnext()))
        rec.update(teacher={"stub": True}, data_root=None)

        def build_dataloader(split, batch_size, **_kw):
            g = torch.Generator().manual_seed(0 if split == "train" else 1)
            n = STUB_STEPS if split == "train" else 1
            batches = [(torch.randn(batch_size, 3, 512, 512, generator=g),
                        torch.randint(0, td.NUM_CLASSES, (batch_size, 512, 512), generator=g)) for _ in range(n)]

            class _Loader(list):
                dataset = [0] * (n * batch_size)
            return _Loader(batches)
        td.build_dataloader = build_dataloader
    else:
        from src.data.isolation import assert_trainval_only_root
        rec["data_root"] = assert_trainval_only_root(Path(td.DATA["root"]))["counts"]
        teacher, rec["teacher"] = load_teacher(td, a.teacher_ckpt, a.teacher_ckpt_sha256)
    rec["cuda_initialized_after_teacher_load"] = torch.cuda.is_initialized()
    state = {"step": 0}
    orig_cycle = td.cycle
    rec["next_seconds"] = []

    def cycle(loader):                         # the training step's index, and the time blocked in next()
        it = orig_cycle(loader)
        while True:
            t0 = time.perf_counter()
            batch = next(it)
            rec["next_seconds"].append(time.perf_counter() - t0)
            state["step"] += 1
            yield batch
    td.cycle = cycle
    rec["step_rows"], rec["hook_seconds"] = [], []
    orig_build_optimizer = td.build_optimizer

    def build_optimizer(*args, **kwargs):      # the peak memory after each optimizer step (PL-33)
        opt, trainable = orig_build_optimizer(*args, **kwargs)
        orig_step = opt.step

        def step(self, *sa, **sk):
            out = orig_step(*sa, **sk)
            t0 = time.perf_counter()
            r = {"step": len(rec["step_rows"]) + 1}
            if cuda:
                r["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
                r["max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
            rec["step_rows"].append(r)
            rec["hook_seconds"].append(time.perf_counter() - t0)
            return out
        opt.step = types.MethodType(step, opt)
        return opt, trainable
    td.build_optimizer = build_optimizer
    nmf = {"forwards": []}
    rec["nmf"] = nmf
    events = []
    orig_forward = teacher.forward

    def forward(x, **kw):                      # each teacher call: its draw, the RNG states, its CUDA-event time
        on_cuda = cuda and torch.cuda.is_initialized()
        cpu0 = torch.get_rng_state()
        cuda0 = torch.cuda.get_rng_state_all() if on_cuda else []
        s0 = teacher.nmf_stream_state() or {}
        ev = None
        if on_cuda:
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
        out = orig_forward(x, **kw)
        if ev is not None:
            ev[1].record()
        s1 = teacher.nmf_stream_state() or {}
        cuda1 = torch.cuda.get_rng_state_all() if on_cuda else []
        if "cuda_initialized_after_first_forward" not in rec:
            rec["cuda_initialized_after_first_forward"] = torch.cuda.is_initialized()
        k = sum(1 for f in nmf["forwards"] if f["step"] == state["step"]) + 1
        events.append((state["step"], k, ev))
        nmf["forwards"].append({"step": state["step"], "call": k, "draws_before": s0.get("draws"),
                                "draws_delta": (s1["draws"] - s0.get("draws", 0)) if "draws" in s1 else None,
                                "state_sha256": s1.get("state_sha256"),
                                "rng_unchanged": bool(torch.equal(cpu0, torch.get_rng_state()))
                                and len(cuda0) == len(cuda1) and all(torch.equal(p, q) for p, q in zip(cuda0, cuda1))})
        return out
    teacher.forward = forward
    td.save_distill_checkpoint = lambda *args, **kwargs: str(ck / "not_written.pt")     # no checkpoint
    td.write_best_pointer = lambda *args, **kwargs: None
    rec["cuda_initialized_before_run"] = torch.cuda.is_initialized()
    kw = dict(stage=stage, mode="dry", device=a.device, pretrained=False, teacher=teacher, lambda_logit=1.0,
              batch_size=a.batch, max_iters=a.steps, val_interval=4000, max_val_batches=1,
              num_workers=a.num_workers, ckpt_dir_arg=str(ck), grad_clip_norm=None, log_every=LOG_EVERY, seed=42,
              arm=spec)
    rec["ckpt_dir"] = str(ck)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            rec["run_rc"] = td.run(**kw)
    finally:
        rec["stdout_tail"] = out.getvalue()[-3000:]
    text = out.getvalue()
    rec["hard_checks"] = {ln.split(":")[0].strip(): ln.rsplit(":", 1)[1].strip() == "PASS"
                          for ln in text.split("\n[CHECKS]\n", 1)[-1].splitlines()
                          if ln.startswith("  ") and ln.rstrip().endswith(("PASS", "FAIL"))}
    if cuda:
        torch.cuda.synchronize()
        rec["memory"] = {"max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                         "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                         "total_memory": int(torch.cuda.get_device_properties(0).total_memory)}
        rec["gpu_name"] = torch.cuda.get_device_name(0)
    calls = {1: [], 2: []}
    for step, k, ev in events:
        if k in calls:
            calls[k].append(None if ev is None else ev[0].elapsed_time(ev[1]))
    rec["teacher_forward_ms_calls"] = {f"call{k}": v for k, v in calls.items()}
    rec["teacher_forward_ms"] = [None if a_ is None or b_ is None else a_ + b_          # per step: both calls
                                 for a_, b_ in zip(calls[1], calls[2])]
    nmf["final_draws"] = (teacher.nmf_stream_state() or {}).get("draws")
    tel = ck / "e3_telemetry.jsonl"
    rows = [json.loads(ln) for ln in tel.read_text(encoding="utf-8").splitlines() if ln.strip()] if tel.exists() else []
    rec["train_rows"] = [r for r in rows if r.get("event") == "train"]
    meta = ck / "e3_run_meta.jsonl"
    metas = ([json.loads(ln) for ln in meta.read_text(encoding="utf-8").splitlines() if ln.strip()]
             if meta.exists() else [])
    rec["run_meta"] = metas[0] if len(metas) == 1 else None


# ---------------------------------------------------------------------------------------------- the driver
def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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


def read_e3_reference(path: str) -> dict:
    """The lambda wave's kd_smoke JSON: E3's measured median s/iter (throughput["e3"]["iter_seconds"]["median"])."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"{path} is not a file")
    data = p.read_bytes()
    doc = json.loads(data.decode("utf-8"))
    if not isinstance(doc, dict) or doc.get("format") != E3_REFERENCE_FORMAT:
        raise ValueError(f"{path} is not a {E3_REFERENCE_FORMAT} JSON")
    med = (((doc.get("throughput") or {}).get("e3") or {}).get("iter_seconds") or {}).get("median")
    if not finite(med) or med <= 0:
        raise ValueError(f"{path}: throughput['e3']['iter_seconds']['median'] is {med!r}, not a positive number")
    return {"path": str(p.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "format": doc["format"],
            "verdict": doc.get("verdict"), "head": (doc.get("environment") or {}).get("head"),
            "iter_seconds_median": med}


def refused(code: str, text: str) -> int:
    print(f"REFUSED [{code}]: {text}")
    print(f"RESULT: REFUSED ({code}) -- no child has run; this is no STOP")
    return 2


def cmd_run(a) -> int:
    import scripts.preflight_distill as pd
    paths = ((a.evidence, "--evidence"), (a.teacher_ckpt, "--teacher-ckpt"), (a.e3_reference, "--e3-reference"))
    for value, what in paths:                  # SL-1: each as typed, before any filesystem call
        if value is not None and "test" in str(value).lower():
            return refused("sl1_path", f"{what} {value!r} contains 'test' (SL-1)")
    for value, what in paths:                  # and each as it resolves
        if value is not None and "test" in os.path.realpath(value).lower():
            return refused("sl1_path", f"{what} {value!r} resolves to a path containing 'test' (SL-1)")
    evidence = Path(a.evidence).resolve()
    if evidence == REPO or REPO in evidence.parents:
        return refused("evidence_in_repo", f"{evidence} is inside the checkout")
    if not evidence.is_dir():
        return refused("evidence", f"{evidence} is not a folder: create it first")
    stamp = utc_stamp()
    base, out = evidence / f"kd_flip_pod_smoke_{stamp}", evidence / f"kd_flip_pod_smoke_{stamp}.json"
    existing = [str(p) for p in (base, out) if os.path.lexists(p)]
    if existing:
        return refused("existing_output", f"{existing} already exist: an output is never overwritten")
    head = git_out("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", a.expect_head or "") or head != a.expect_head:
        return refused("expect_head", f"--expect-head {a.expect_head!r} is not HEAD "
                                      f"({'git rev-parse HEAD failed' if head is None else head}): run from the "
                                      "checkout of the arms pin P_K and pass its full 40-hex commit")
    st = scoped_status(pd)
    env_row = {"image_digest": os.environ.get("PLANTSEG_IMAGE_DIGEST"), "kd_image_digest": pd.KD_IMAGE_DIGEST,
               "head": head, "expect_head": a.expect_head,
               "governed_dirty": pd.E1G.scoped_dirty(st.stdout) if st.returncode == 0 else ["git status failed"],
               "plantseg_git_commit": os.environ.get("PLANTSEG_GIT_COMMIT"), "driver": gpu_driver(),
               "data_root": os.environ.get("PLANTSEG_DATA_ROOT")}
    env_check = eval_environment({"environment": env_row})      # judged now, before any child, and again at the end
    if env_check["status"] != "PASS":
        return refused("environment", f"{env_check['detail']} (PLANTSEG_IMAGE_DIGEST {env_row['image_digest']!r}, "
                                      f"PLANTSEG_GIT_COMMIT {env_row['plantseg_git_commit']!r}, governed "
                                      f"{env_row['governed_dirty'][:10]}): the KD digest exported, PLANTSEG_GIT_COMMIT "
                                      "unset, the clean checkout of the pin")
    t_bad = teacher_problems(a.teacher_ckpt, a.teacher_ckpt_sha256, pd)
    if t_bad:
        return refused("teacher", f"{t_bad}: --teacher-ckpt and --teacher-ckpt-sha256 name the teacher of record "
                                  f"(DL-88: {pd.TEACHER_FILE}, sha256 {pd.TEACHER_SHA256[:12]}...)")
    digest = sha256_file(a.teacher_ckpt)
    if digest != a.teacher_ckpt_sha256:
        return refused("teacher_sha256_mismatch", f"{a.teacher_ckpt} hashes to {digest}, not {a.teacher_ckpt_sha256}")
    reference = None
    if a.e3_reference is not None:
        try:
            reference = read_e3_reference(a.e3_reference)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return refused("e3_reference", f"{e}: --e3-reference is the lambda wave's {E3_REFERENCE_FORMAT} JSON")
    base.mkdir(parents=False)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log = base / "stage_flip.log"
    stage = run_child(["_stage_flip", "--teacher-ckpt", str(Path(a.teacher_ckpt).resolve()),
                       "--teacher-ckpt-sha256", a.teacher_ckpt_sha256, "--num-workers", str(pd.NUM_WORKERS)],
                      base / "stage_flip.json", log, 3600)
    warns = list(stage.get("warnings") or [])
    warns += [w for w in warnings_in(log.read_text(encoding="utf-8", errors="replace"), "stage_flip") if w not in warns]
    calls = stage.get("teacher_forward_ms_calls") or {}
    doc = {"format": FORMAT, "arm": ARM, "run_stage": RUN_STAGE, "head": head, "expect_head": a.expect_head,
           "evidence": str(evidence), "run_folder": str(base), "started_utc": started,
           "am18_item6a_hold": {"ended_at": f"DL-84, {DL84_UTC}", "ended_before_this_run": started > DL84_UTC},
           "environment": env_row, "teacher": stage.get("teacher"), "e3_reference": reference, "stage": stage,
           "warnings": warns,
           "throughput": {RUN_STAGE.lower(): throughput(stage)},
           "max_memory_reserved": (stage.get("memory") or {}).get("max_memory_reserved"),
           "teacher_forward_ms": {k: {"ms": v, "stats_steps_6_25": stats((v or [])[THROUGHPUT_STEPS[0] - 1:
                                                                                   THROUGHPUT_STEPS[1]])}
                                  for k, v in calls.items()},
           "draws_per_step": draws_per_step(stage),
           "torch": stage.get("torch"), "gpu_name": stage.get("gpu_name")}
    rows, verdict = evaluate(doc, pd.teacher_identity_problems)
    w_med = doc["throughput"][RUN_STAGE.lower()]["iter_seconds"]["median"]
    figures = [v for v in (w_med, e3_median(doc)) if finite(v)]
    doc["arithmetic_input"] = {"e3_flip_s_per_iter_median": w_med, "e3_s_per_iter_median": e3_median(doc),
                               "larger": max(figures) if finite(w_med) else None}
    doc.update(rows=rows, verdict=verdict)
    with open(out, "x", encoding="utf-8") as fh:             # exclusive: an output is never overwritten
        fh.write(json.dumps(doc, indent=1, default=str) + "\n")
    for r in rows:
        print(f"{r['status']:5} {r['check']:18} {r['detail'][:220]}")
    for r in rows:
        if r["status"] == "STOP":
            print(f"STOP: {r['check']}: {r['detail'][:400]}")
    print(f"written: {out}")
    print(f"RESULT: {verdict}")
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------------------------------------- selfcheck
SELF_RESULTS: list = []


def scheck(name: str, ok, detail="") -> None:
    SELF_RESULTS.append((name, bool(ok), "" if ok else str(detail)[:500]))


def synthetic_doc(steps: int = STEPS) -> dict:
    """A record that meets every criterion (the evaluator's PASS form): the teacher of record, `steps` rows at 1.2
    s/iter, two draws per step, 30e9 B reserved, an E3 reference of 1.0 s/iter."""
    import scripts.preflight_distill as pd
    ckpt = "/workspace/teacher/iter_24000.pth"
    prov = {"builder": pd.TEACHER_BUILDER, "ckpt_path": ckpt, "ckpt_sha256": pd.TEACHER_SHA256,
            "ckpt_bytes": pd.TEACHER_BYTES, "expected_sha256": pd.TEACHER_SHA256,
            "config_path": str(REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"),
            **pd.TEACHER_VALUES}
    teacher = {"ok": True, "sha256": pd.TEACHER_SHA256, "bytes": pd.TEACHER_BYTES, "provenance": prov,
               "params": pd.TEACHER_PARAMS, "trainable": 0, "training_modules": [],
               "state_entries": pd.TEACHER_STATE_ENTRIES, "ckpt_path": ckpt, "file_name": pd.TEACHER_FILE}
    rows = []
    for it in range(1, steps + 1):
        loss = 5.0 + 0.001 * it
        values = {"event": "train", "iter": it, "loss": loss, "sup": loss if it == 1 else loss - 0.01, "ce": 4.1,
                  "dice": 0.9, "logit_kd": 0.37, "cwd_feat": 0.37, "cwd_logit": 0.37,
                  "ramp": (it - 1) / RAMP_DENOMINATOR, "lr": 0.01, "grad_norm": 1.5, "grad_norm_student": 1.5,
                  "grad_norm_projection": 0.0 if it == 1 else 0.2, "wall_clock": 1.0 * it, "iter_seconds": 1.2,
                  "samples_per_sec": 13.3}
        rows.append({k: values[k] for k in train_keys("e3")})
    forwards = [{"step": t, "call": k, "draws_before": DRAWS_PER_STEP * (t - 1) + k - 1, "draws_delta": 1,
                 "state_sha256": f"{DRAWS_PER_STEP * (t - 1) + k:064x}", "rng_unchanged": True}
                for t in range(1, steps + 1) for k in range(1, DRAWS_PER_STEP + 1)]
    run_meta = {"stage": "e3", "seed": 42, "arm": RUN_STAGE, "descriptive": True, "parent_of_e4_e7": False,
                "selection_sha256": {"lambda_logit": None, "alpha_cwd": None},
                "selection_files": {"lambda_logit": None, "alpha_cwd": None}, "records_commit": None,
                "target_construction": dict(TARGET_CONSTRUCTION)}
    stage = {"kind": "stage_flip", "arm": ARM, "child_rc": 0, "run_rc": 0, "teacher": teacher,
             "hard_checks": {k: True for k in HARD_CHECKS}, "train_rows": rows,
             "nmf": {"forwards": forwards, "final_draws": DRAWS_PER_STEP * steps},
             "memory": {"max_memory_reserved": 30 * 10**9}, "next_seconds": [0.01] * steps,
             "hook_seconds": [0.0] * steps, "teacher_forward_ms": [600.0] * steps,
             "teacher_forward_ms_calls": {"call1": [300.0] * steps, "call2": [300.0] * steps}, "run_meta": run_meta}
    return {"format": FORMAT,
            "environment": {"image_digest": pd.KD_IMAGE_DIGEST, "kd_image_digest": pd.KD_IMAGE_DIGEST,
                            "head": "c" * 40, "expect_head": "c" * 40, "governed_dirty": [],
                            "plantseg_git_commit": None},
            "stage": stage, "e3_reference": {"iter_seconds_median": 1.0}, "warnings": []}


def selfcheck_evaluator() -> None:
    import scripts.preflight_distill as pd

    def verdict_of(edit, doc=None) -> tuple:
        doc = synthetic_doc() if doc is None else doc
        edit(doc)
        rows, verdict = evaluate(doc, pd.teacher_identity_problems)
        return verdict, {r["check"]: r["status"] for r in rows}, {r["check"]: r["detail"] for r in rows}

    # memory (4): one byte above each threshold, as literal byte counts (a threshold in another unit fails here)
    for label, value, want in (("40e9", 40_000_000_000, "PASS"), ("40e9_plus_1_byte", 40_000_000_001, "FLAG"),
                               ("44e9", 44_000_000_000, "FLAG"), ("44e9_plus_1_byte", 44_000_000_001, "STOP")):
        v, st, _ = verdict_of(lambda d, value=value: d["stage"]["memory"].update(max_memory_reserved=value))
        scheck(f"s_memory_{label}_is_{want.lower()}", st.get("peak_memory") == want and v == want, (st, v))

    # draws (2): one draw per step, and three
    def one_per_step(d):
        st = d["stage"]
        st["nmf"]["forwards"] = [dict(f, draws_before=f["step"] - 1) for f in st["nmf"]["forwards"] if f["call"] == 1]
        st["nmf"]["final_draws"] = STEPS

    def three_per_step(d):
        st = d["stage"]
        st["nmf"]["forwards"] = [{"step": t, "call": k, "draws_before": 3 * (t - 1) + k - 1, "draws_delta": 1,
                                  "state_sha256": "0" * 64, "rng_unchanged": True}
                                 for t in range(1, STEPS + 1) for k in (1, 2, 3)]
        st["nmf"]["final_draws"] = 3 * STEPS

    def three_in_two_forwards(d):          # call 1 takes two draws and call 2 one: two forwards, three draws
        st = d["stage"]
        st["nmf"]["forwards"] = [{"step": t, "call": k, "draws_before": 3 * (t - 1) + 2 * (k - 1),
                                  "draws_delta": 3 - k, "state_sha256": "0" * 64, "rng_unchanged": True}
                                 for t in range(1, STEPS + 1) for k in (1, 2)]
        st["nmf"]["final_draws"] = 3 * STEPS

    def stops(edit, draws: int) -> tuple:  # the draws check STOPs, and K6's draws_per_step reads the draws taken
        doc = synthetic_doc()
        v, st, det = verdict_of(edit, doc=doc)
        got = draws_per_step(doc["stage"])
        return st.get("nmf_draws") == "STOP" and v == "STOP" and got == [draws] * STEPS, (det.get("nmf_draws"), got[:3])
    ok, detail = stops(one_per_step, 1)
    scheck("s_draws_one_per_step_stops", ok, detail)
    (ok_a, detail_a), (ok_b, detail_b) = stops(three_per_step, 3), stops(three_in_two_forwards, 3)
    scheck("s_draws_three_per_step_stops", ok_a and ok_b, (detail_a, detail_b))

    # time (2): at 2 x E3's median, and above it
    def seconds(value):
        return lambda d: [r.update(iter_seconds=value) for r in d["stage"]["train_rows"]]
    v, st, det = verdict_of(seconds(2.0))
    scheck("s_s_per_iter_at_2_x_e3_passes", st.get("throughput") == "PASS" and v == "PASS", det.get("throughput"))
    v, st, det = verdict_of(seconds(math.nextafter(2.0, 3.0)))
    scheck("s_s_per_iter_above_2_x_e3_flags", st.get("throughput") == "FLAG" and v == "FLAG", det.get("throughput"))

    # no E3 reference: the 4.0 s/iter fallback, at it and above it
    def no_reference(value):
        def edit(d):
            d["e3_reference"] = None
            seconds(value)(d)
        return edit
    v1, st1, det1 = verdict_of(no_reference(FALLBACK_S_PER_ITER))
    v2, st2, det2 = verdict_of(no_reference(math.nextafter(FALLBACK_S_PER_ITER, 5.0)))
    scheck("s_no_e3_reference_uses_the_4_s_per_iter_fallback_flag",
           FALLBACK_S_PER_ITER == 4.0 and st1.get("throughput") == "PASS" and st2.get("throughput") == "FLAG"
           and v2 == "FLAG" and "fallback" in det2.get("throughput", ""),
           (det1.get("throughput"), det2.get("throughput")))

    # determinism: the evaluator twice on the same record, and on a fresh copy, gives identical JSON
    doc = synthetic_doc()
    first = json.dumps(evaluate(doc, pd.teacher_identity_problems), sort_keys=True)
    again = json.dumps(evaluate(doc, pd.teacher_identity_problems), sort_keys=True)
    fresh = json.dumps(evaluate(synthetic_doc(), pd.teacher_identity_problems), sort_keys=True)
    scheck("s_evaluator_twice_gives_identical_json", first == again == fresh and json.loads(first)[1] == "PASS",
           (first[:200], again[:200]))

    # mutation: smoke_kd_step's one-draw rule in place of K7's two; the draws check must kill it. Both records are
    # built under K7's rule, before the mutation: the PASS form (two draws per step) and the one-draw case.
    two, one = synthetic_doc(), synthetic_doc()
    one_per_step(one)
    g = globals()
    saved = g["DRAWS_PER_STEP"]
    g["DRAWS_PER_STEP"] = 1
    try:
        base_status = verdict_of(lambda d: None, doc=two)[1].get("nmf_draws")
        one_status = verdict_of(lambda d: None, doc=one)[1].get("nmf_draws")
    finally:
        g["DRAWS_PER_STEP"] = saved
    scheck("s_one_draw_rule_mutation_is_killed_by_the_draws_check",
           base_status == "STOP" and one_status == "PASS",
           f"under the mutation: two draws {base_status}, one draw {one_status} (the draws check passes the one and "
           "stops the two)")


def selfcheck_refusals(work: Path) -> None:
    """The refusals before any child: the checkout's HEAD and status, the GPU probe and the child are stubbed, so no
    git command runs on the checkout and nothing launches."""
    import scripts.preflight_distill as pd
    teacher = work / "teacher" / pd.TEACHER_FILE
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"a throwaway file named as the teacher of record (smoke_kd_flip_pod selfcheck)\n")
    g = globals()
    child_calls = []

    def no_child(*args, **_k):
        child_calls.append(args)
        raise RuntimeError("the driver went on to launch a child")

    def attempt(label: str, *, expect_head="d" * 40, head="d" * 40, sha=pd.TEACHER_SHA256, teacher_path=None,
                evidence=None, e3_reference=None) -> tuple:
        ev = work / f"evidence_{label}" if evidence is None else evidence          # a given path is never created
        if evidence is None:
            ev.mkdir()
        names = ("run_child", "git_out", "scoped_status", "gpu_driver")
        keep = {k: g[k] for k in names}
        env_keep = {k: os.environ.get(k) for k in ("PLANTSEG_IMAGE_DIGEST", "PLANTSEG_GIT_COMMIT")}
        child_calls.clear()
        g.update(run_child=no_child, git_out=lambda *args: head if args == ("rev-parse", "HEAD") else None,
                 scoped_status=lambda _pd: subprocess.CompletedProcess([], 0, b"", b""), gpu_driver=lambda: "stub")
        os.environ["PLANTSEG_IMAGE_DIGEST"] = pd.KD_IMAGE_DIGEST
        os.environ.pop("PLANTSEG_GIT_COMMIT", None)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                rc = cmd_run(argparse.Namespace(expect_head=expect_head, teacher_ckpt=teacher_path or str(teacher),
                                                teacher_ckpt_sha256=sha, evidence=str(ev), e3_reference=e3_reference))
        except Exception as e:  # noqa: BLE001 - the stub's escape, or any other, is the case's result
            rc = f"raised: {e}"
        finally:
            g.update(keep)
            for k, v in env_keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return rc, out.getvalue(), sorted(p.name for p in ev.iterdir()) if ev.is_dir() else None

    rc, out, files = attempt("teacher_other", sha="0" * 64)
    scheck("s_a_teacher_other_than_the_record_is_refused",
           rc == 2 and "REFUSED [teacher]" in out and files == [] and not child_calls, f"{rc} {out[-300:]} {files}")
    rc, out, files = attempt("teacher_sha256")
    scheck("s_a_teacher_file_whose_sha256_differs_is_refused",
           rc == 2 and "REFUSED [teacher_sha256_mismatch]" in out and files == [] and not child_calls,
           f"{rc} {out[-300:]} {files}")
    inside = REPO / "kd_flip_pod_evidence_inside_the_checkout"
    rc, out, files = attempt("in_repo", evidence=inside)
    scheck("s_an_evidence_folder_inside_the_checkout_is_refused",
           rc == 2 and "REFUSED [evidence_in_repo]" in out and files is None and not inside.exists()
           and not child_calls, f"{rc} {out[-300:]} {files}")
    sl1 = [attempt("sl1_evidence", evidence=work / "kd_flip_pod_test_evidence"),
           attempt("sl1_reference", e3_reference=str(work / "kd_smoke_test.json")),
           attempt("sl1_teacher", teacher_path=str(work / "teacher_test" / pd.TEACHER_FILE))]
    scheck("s_a_path_sl1_refuses_is_refused",
           all(rc == 2 and "REFUSED [sl1_path]" in out for rc, out, _ in sl1)
           and not (work / "kd_flip_pod_test_evidence").exists() and not child_calls,
           [(rc, out[-160:]) for rc, out, _ in sl1])
    rc, out, files = attempt("head_other", expect_head="c" * 40)
    scheck("s_an_expect_head_other_than_head_is_refused",
           rc == 2 and "REFUSED [expect_head]" in out and files == [] and not child_calls, f"{rc} {out[-300:]} {files}")


def selfcheck_child(work: Path) -> None:
    """A stub `_stage_flip` child (CPU, batch 2, 3 steps, synthetic batches, the harness's stub teacher with its
    M4-KD stream): the record the evaluator reads. Then the driver end to end, its HEAD, environment and teacher-file
    checks stubbed and its child returning the PASS-form record: the kd_flip_pod_smoke/1 JSON it writes (GO ruling
    K6), exclusively, and its verdict."""
    import scripts.preflight_distill as pd
    out = work / "stub_stage_flip.json"
    r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "_stage_flip", "--stub", "--device",
                        "cpu", "--batch", str(STUB_BATCH), "--steps", str(STUB_STEPS), "--num-workers", "0",
                        "--out", str(out)], cwd=str(REPO), capture_output=True, text=True, timeout=1200,
                       env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1"))
    rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    g = globals()
    saved = g["STEPS"]
    g["STEPS"] = STUB_STEPS
    try:
        draws = eval_draws({"stage": rec})
        meta = eval_run_meta({"stage": rec})
    finally:
        g["STEPS"] = saved
    rows = rec.get("train_rows") or []
    child_ok = (r.returncode == 0 and rec.get("run_rc") == 0 and not rec.get("error") and draws["status"] == "PASS"
                and meta["status"] == "PASS" and rec.get("hard_checks") == {k: True for k in HARD_CHECKS}
                and len(rows) == STUB_STEPS and all(list(x) == train_keys("e3") for x in rows)
                and len(rec.get("next_seconds") or []) == STUB_STEPS
                and sorted(rec.get("teacher_forward_ms_calls") or {}) == ["call1", "call2"])
    ev = work / "evidence_driver"
    ev.mkdir()
    reference = work / "e3_reference_kd_smoke.json"
    e3_stats = {"median": 1.0, "mean": 1.0, "max": 1.0, "n": 20}
    reference.write_text(json.dumps({"format": E3_REFERENCE_FORMAT, "verdict": "PASS",
                                     "environment": {"head": "e" * 40},
                                     "throughput": {"e3": {"iter_seconds": e3_stats}}}), encoding="utf-8")
    teacher = work / "teacher_driver" / pd.TEACHER_FILE
    teacher.parent.mkdir()
    teacher.write_bytes(b"a throwaway file named as the teacher of record (smoke_kd_flip_pod selfcheck)\n")
    good, stamp = synthetic_doc()["stage"], "20261011T000000Z"

    def fake_child(argv, out_json, log, timeout):
        log.write_text("", encoding="utf-8")
        return {**json.loads(json.dumps(good)), "child_rc": 0}
    g = globals()
    names = ("run_child", "git_out", "scoped_status", "gpu_driver", "teacher_problems", "sha256_file", "utc_stamp")
    keep = {k: g[k] for k in names}
    env_keep = {k: os.environ.get(k) for k in ("PLANTSEG_IMAGE_DIGEST", "PLANTSEG_GIT_COMMIT")}
    g.update(run_child=fake_child, git_out=lambda *args: "d" * 40 if args == ("rev-parse", "HEAD") else None,
             scoped_status=lambda _pd: subprocess.CompletedProcess([], 0, b"", b""), gpu_driver=lambda: "stub",
             teacher_problems=lambda *_a: [], sha256_file=lambda _p: pd.TEACHER_SHA256, utc_stamp=lambda: stamp)
    os.environ["PLANTSEG_IMAGE_DIGEST"] = pd.KD_IMAGE_DIGEST
    os.environ.pop("PLANTSEG_GIT_COMMIT", None)
    args = argparse.Namespace(expect_head="d" * 40, teacher_ckpt=str(teacher), teacher_ckpt_sha256=pd.TEACHER_SHA256,
                              evidence=str(ev), e3_reference=str(reference))
    outs = [io.StringIO(), io.StringIO()]
    try:
        with contextlib.redirect_stdout(outs[0]):
            rc1 = cmd_run(args)
        with contextlib.redirect_stdout(outs[1]):         # the same stamp again: an existing output
            rc2 = cmd_run(args)
    finally:
        g.update(keep)
        for k, v in env_keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    written = ev / f"kd_flip_pod_smoke_{stamp}.json"
    doc = json.loads(written.read_text(encoding="utf-8")) if written.is_file() else {}
    it_s = ((doc.get("throughput") or {}).get("e3-flip") or {}).get("iter_seconds") or {}
    driver_ok = (rc1 == 0 and "RESULT: PASS" in outs[0].getvalue() and doc.get("format") == FORMAT
                 and doc.get("verdict") == "PASS" and set(it_s) == {"median", "mean", "max", "n"}
                 and it_s.get("median") == 1.2 and it_s.get("n") == 20
                 and doc.get("max_memory_reserved") == 30 * 10**9
                 and sorted(doc.get("teacher_forward_ms") or {}) == ["call1", "call2"]
                 and doc.get("draws_per_step") == [DRAWS_PER_STEP] * STEPS and doc.get("head") == "d" * 40
                 and doc.get("evidence") == str(ev.resolve()) and (doc.get("teacher") or {}).get("sha256")
                 == pd.TEACHER_SHA256 and (doc.get("e3_reference") or {}).get("iter_seconds_median") == 1.0
                 and (doc.get("arithmetic_input") or {}).get("larger") == 1.2
                 and (doc.get("am18_item6a_hold") or {}).get("ended_before_this_run") is True
                 and rc2 == 2 and "REFUSED [existing_output]" in outs[1].getvalue())
    scheck("s_stub_child_record_and_driver_json_end_to_end", child_ok and driver_ok,
           f"child: rc={r.returncode} error={str(rec.get('error'))[:300]} draws={draws} meta={meta} "
           f"{r.stdout[-200:]} {r.stderr[-400:]} | driver: rc={rc1}/{rc2} verdict={doc.get('verdict')} "
           f"iter_seconds={it_s} {outs[0].getvalue()[-300:]} {outs[1].getvalue()[-150:]}")


def cmd_selfcheck() -> int:
    print("=" * 78)
    print("W POD SMOKE --selfcheck (CPU; the evaluator, the refusals, a stub child and the driver; never a pod PASS)")
    print("=" * 78)
    # SL-1 and the out-root rule (plan 1.0; review finding RIR-1): the DIAG guards, imported
    from src.eval.teacher_diag import Refused, refuse_test_path, require_outside_repo
    try:
        for var in ("TMPDIR", "TEMP", "TMP"):          # before tempfile probes a candidate by writing to it
            if os.environ.get(var):
                refuse_test_path(os.environ[var], f"${var} (set it elsewhere)")
        work = Path(tempfile.gettempdir()) / f"kd_flip_pod_selfcheck_{os.getpid()}"
        refuse_test_path(work, "the selfcheck work dir (set TMPDIR elsewhere)")
        work = require_outside_repo(work, "the selfcheck work dir")
    except Refused as e:
        return refused("selfcheck_work_dir", str(e))
    work.mkdir()
    for name, fn in (("evaluator", selfcheck_evaluator), ("refusals", lambda: selfcheck_refusals(work)),
                     ("child", lambda: selfcheck_child(work))):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a crash is a failed check
            scheck(f"s_{name}_completed", False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-700:]}")
    print("\n[CHECKS]")
    for name, ok, detail in SELF_RESULTS:
        print(f"  {name:62}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in SELF_RESULTS if ok)
    print(f"\nwork dir (kept): {work}")
    print(f"RESULT: {'PASS' if passed == len(SELF_RESULTS) else 'FAIL'} ({passed}/{len(SELF_RESULTS)})")
    return 0 if passed == len(SELF_RESULTS) else 1


# --------------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="W pod smoke (lane L-KD-WRAP).", allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", allow_abbrev=False)
    for name in ("--expect-head", "--teacher-ckpt", "--teacher-ckpt-sha256", "--evidence"):
        r.add_argument(name, required=True)
    r.add_argument("--e3-reference", default=None)
    s = sub.add_parser("_stage_flip", allow_abbrev=False)
    s.add_argument("--out", required=True)
    s.add_argument("--teacher-ckpt", default=None)
    s.add_argument("--teacher-ckpt-sha256", default=None)
    s.add_argument("--steps", type=int, default=STEPS)
    s.add_argument("--batch", type=int, default=BATCH)
    s.add_argument("--device", default="cuda")
    s.add_argument("--num-workers", type=int, required=True)    # the parent passes the gate's value
    s.add_argument("--stub", action="store_true")
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selfcheck"]:
        return cmd_selfcheck()
    a = build_parser().parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.stub == bool(a.teacher_ckpt and a.teacher_ckpt_sha256):
        print("usage error: _stage_flip takes --teacher-ckpt and --teacher-ckpt-sha256, or --stub", file=sys.stderr)
        return 2
    rec: dict = {}
    try:
        child_stage_flip(a, rec)
    except Exception as e:  # noqa: BLE001 - the record says what failed; the evaluator stops on it
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
        print(rec["traceback"], file=sys.stderr)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    return 1 if rec.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
