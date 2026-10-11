#!/usr/bin/env python3
"""Smoke for AM-18 item 3(b)'s arm W, run stage E3-flip (lane L-KD-WRAP, commit C1; ARMS plan section 4 item 4 and
decision (vi); Tier F audit F28 and F29; GO rulings K2, K3 and K5, amendment A1).

CPU only, torch only: no dataset, checkpoint, GPU, download or teacher stack. Two kinds of evidence:
  function level  src/distill/flip_targets.py in this process, against float64 references written out here and
                  against src/training/losses.py. Its stub teacher is exactly flip-equivariant (logits and Stage 3
                  are cell-wise functions of the input's 8x8 and 16x16 block maxima, and a maximum ignores order)
                  and scales each class by 1 + 3r, r its forward's draw from a real M4-KD NMFStream, so each view is
                  computed here independently of the builder.
  run level       train_distill.run() dry runs (stage e3, CPU, batch 2, seed 42, 3 steps, a validation at step 3)
                  through scripts/invariance_harness.py's worker, unchanged: its synthetic 16 + 4 TRAIN/VAL set and
                  its stub teacher, a fixed-weight MSCAN-shaped network whose ham head draws once per forward from
                  the M4-KD stream, wrapped in the trainer's own SegNeXtTeacherAdapter and FrozenTeacher. The arm is
                  bound to run() (arm=<ArmSpec>), so the trainer makes the builder and calls build() through the seam.
                  The recorders only observe: each teacher call (its input, the stream's draws, the CPU RNG state,
                  the probe's hooks, its outputs), each student forward, each build() and the mask each loss gets.
At C1 no registry file exists: the arm is kd_arms.parse_arm() of a kd_arm/1 file outside the repository holding the
registry entry's values (GO ruling K4), which the trainer takes as it takes a registered arm.

Checks (28 at C1):
  cm_two_calls_per_step                     build() calls the teacher once; each training step makes two calls, the
                                            trainer's on the batch and then build()'s, both at batch 2
  cm_unflipped_call_is_e3_call              at step 1, call 1 is E3's call: the same input, logits, Stage 3 and
                                            stream position as an E3 run's, inside the probe; the step's supervised
                                            losses and feature term are E3's bit for bit
  cm_second_call_on_hflip                   call 2's input is torch.flip(input, dims=[-1]): the width reversed
  cm_flip_back_alignment                    view 2 is call 2's logits flipped back: the aligned map computed here,
                                            bit for bit, never call 2's raw logits
  cm_logit_kd_prob_average_T4               the Logit-KD term at T_logit = 4 equals the float64 reference with
                                            p = (softmax(z1/4) + softmax(z2/4)) / 2, value and gradient
  cm_cwd_logit_prob_average_T4_per_channel  the CWD logit-map term at T_cwd = 4 equals the float64 reference that
                                            averages each channel's distributions over the valid cells (T^2 / 116, a
                                            sample with no valid cell 0), value and gradient
  cm_per_term_softmax                       each term in its own softmax at its own T: Logit KD over the classes, CWD
                                            over the cells of each channel (T 4 and 2 told apart)
  cm_never_logits                           both terms are the probability averages, not the softmax of the averaged
                                            logits
  cm_equal_views_match_single_view          with two equal views the Logit-KD term and its gradient are logit_kd_kl's
                                            bit for bit and the CWD term is cwd_channelwise_kl's within 1e-6
                                            relative; a teacher whose draws do not move its outputs gives build() two
                                            equal views
  cm_feature_term_unflipped_stage3          build() leaves call 1's Stage 3 untouched; with W's targets the trainer's
                                            feature term is E3's bit for bit; in every run step it reads call 1's
                                            Stage 3, never call 2's
  cm_nmf_two_draws_per_step                 call 1 takes draw 2t-1 and call 2 draw 2t, one per forward; view 1 is
                                            call 1's output and view 2 call 2's; build() refuses a call 2 that takes
                                            0 or 2 draws
  cm_cpu_rng_unchanged_by_call2             no call changes the CPU RNG state; build() refuses a teacher that draws
                                            from it; the CPU, numpy and Python RNG states after each step are E3's
  cm_student_one_forward_original_crop      one training forward of the student per step, on the batch itself;
                                            build() never receives the student
  cm_step1_checks_pass                      the trainer's 20 checks PASS and RESULT: PASS; at step 1 the probe's hook
                                            sees call 1 and is gone by call 2 (F29)
  cm_k_not_1_refused                        FlipTargets is K 1 and VIEWS 2; a builder with another K refuses itself;
                                            kd_arm/1 refuses K 2 and 8; an arm with K 1 refuses a builder with K 8
  cm_validity_mask_original_orientation     both terms use the mask as given; in every run step both get the
                                            trainer's valid_os8 of the crop
  cm_abort_teacher_finite_reads_targets     A1.1, at function level: with call 1 finite and call 2 not, finite is
                                            false and train_distill.py's own abort-record expression, compiled from
                                            its source, gives teacher_finite false (true without targets or with
                                            finite ones). No non-real run reaches abort(): each abort() call in
                                            run() sits under `mode == "real"` or the real-only AM-7 (b) monitor
  dt_flip_two_runs_identical_bytes          two W runs: their step JSONs and records are byte-identical
  mut_<name> x10                            each mutation, by monkeypatch here or in a worker, fails its named check,
                                            which passes without it: average_logits (never_logits), no_flip_back
                                            (flip_back_alignment), shared_softmax (per_term_softmax),
                                            feature_from_flipped_view (feature_term_unflipped_stage3),
                                            one_batched_call (nmf_two_draws_per_step), student_sees_flip
                                            (student_one_forward_original_crop), t_1 (logit_kd_prob_average_T4),
                                            flipped_mask (validity_mask_original_orientation), call2_inside_probe
                                            (step1_checks_pass), finite_always_true (abort_teacher_finite_reads_targets)

Usage:  python -B scripts/smoke_kd_flip.py [--work-dir DIR] [--threads N]
  --work-dir  a new folder outside the repository (default: <tempdir>/kd_flip_smoke_<pid>_<n>); outputs are kept
The smoke re-invokes itself with `--internal flip_worker ...` for each trainer run; that is not part of its CLI.
Exit codes: 0 PASS; 1 a check failed; 2 refused, nothing written (a path containing "test", an existing --work-dir,
a --work-dir inside the repository).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only smoke: hide GPUs from every process it starts

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Standard-library-only modules: torch and src/ load in main() and in the workers.
from scripts.invariance_harness import compare_step_dirs, make_synthetic_dataset  # noqa: E402
from scripts.smoke_kd_step import HARD_CHECKS  # noqa: E402

# GO ruling K4: the registry entry's values, in e3_r1.json's key order
REGISTRY_ENTRY = {"format": "kd_arm/1", "name": "e3-flip", "run_stage": "E3-flip", "e7_stage": "E7-flip",
                  "stage": "e3", "seed": 42, "horizon": 80000, "teacher": "record", "arm_teacher": None,
                  "go_rule": False, "K": 1, "views": 2,
                  "targets": {"module": "src.distill.flip_targets", "builder": "FlipTargets"}, "cut_place": 2}
STEPS, VAL_INTERVAL, BATCH = 3, 3, 2           # the harness's batch; a validation at the last step
WORKER_TIMEOUT = 3600                          # seconds; below STEP 5's 10,800
STUB_SEED = 4243
NUM_CLASSES, FEAT_CH = 116, 320
REL_TOL = 1e-5                                 # float32 against the float64 references
EQUAL_VIEWS_REL_TOL = 1e-6                     # plan (vi): the CWD logit-map term with two equal views
DISCRIMINATE = 1e-3                            # a variant a check must tell apart differs by more, relatively
IDENTITY_KEYS = ("wall_clock", "git_head", "git_head_source")
RUN_MUTATIONS = ("student_sees_flip", "call2_inside_probe")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail="") -> None:
    results.append((name, bool(ok), str(detail)))


def tsha(t) -> str:
    """sha256 over a tensor's dtype, shape and bytes."""
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


# =========================================================================================== run-level worker
def _child_paths(*values) -> None:
    """An internal child takes only paths the smoke built from its guarded roots; each is refused as a string
    (SL-1) before any use."""
    for v in values:
        if v is not None and "test" in str(v).lower():
            print("REFUSED: an internal path containing 'test'", file=sys.stderr)
            raise SystemExit(2)


def instrument(rec, *, arm_json, mutation: str, code_root: Path, record: dict) -> None:
    """Bind the arm to run() and add the recorders (and a run-level mutation) in the worker's process, once the
    harness has installed its own recorder and before its run() call."""
    import torch
    from src.training import kd_arms
    from src.training.losses import downsample_validity
    ft = importlib.import_module("src.distill.flip_targets")
    for mod in (kd_arms, ft):
        if code_root not in Path(mod.__file__).resolve().parents:
            raise SystemExit(f"{mod.__name__} was imported from {mod.__file__}, outside the code root {code_root}")
    td, teacher = rec.td, rec.teacher
    inner = teacher.teacher                     # the adapter, which the trainer's step-1 probe hooks
    state = {"step": 0, "batch": None, "in_build": False, "pending": None, "probe": None, "targets": None}
    record.update(arm=None, mutation=mutation, batches=[], calls=[], student=[], builds=[], losses=[], masks=[])

    if arm_json is not None:
        spec = kd_arms.parse_arm(arm_json)
        spec.builder_class()
        record["arm"] = {"run_stage": spec.run_stage, "target_construction": spec.target_construction}
        run = td.run

        @functools.wraps(run)
        def run_with_arm(*args, **kwargs):
            return run(*args, arm=spec, **kwargs)
        td.run = run_with_arm

    orig_cycle = td.cycle

    def cycle(loader):                          # one TRAIN batch per training step
        it = orig_cycle(loader)
        while True:
            img, mask = next(it)
            state["step"] += 1
            state["batch"] = img
            record["batches"].append({"step": state["step"], "img": tsha(img),
                                      "img_flip_w": tsha(torch.flip(img, dims=[-1])),
                                      "img_flip_h": tsha(torch.flip(img, dims=[-2])), "mask": tsha(mask)})
            yield img, mask
    td.cycle = cycle

    inner_build_student = td.build_student

    def build_student(*args, **kwargs):
        student = inner_build_student(*args, **kwargs)
        orig_forward = student.forward

        def forward(x, *fa, **fk):
            record["student"].append({"step": state["step"], "training": bool(student.training), "x": tsha(x),
                                      "x_is_batch": x is state["batch"]})
            return orig_forward(x, *fa, **fk)
        student.forward = forward
        if mutation == "student_sees_flip":     # the student fed the flipped crop
            student.register_forward_pre_hook(
                lambda m, inputs: (torch.flip(inputs[0], dims=[-1]),) if m.training else None)
        return student
    td.build_student = build_student

    def stream_draws():
        st = teacher.nmf_stream_state()
        return None if st is None else st["draws"]

    def pre(_module, inputs):
        x = inputs[0]
        state["pending"] = {"step": state["step"], "in_build": state["in_build"], "batch_size": int(x.shape[0]),
                            "x": tsha(x), "x_is_batch": x is state["batch"],
                            "probe_hooks": len(inner._forward_pre_hooks), "draws_before": stream_draws(),
                            "cpu_rng_before": tsha(torch.get_rng_state())}

    def post(_module, _inputs, output):
        call = state["pending"]
        call.update(draws_after=stream_draws(), cpu_rng_after=tsha(torch.get_rng_state()),
                    logits=tsha(output.logits), logits_flip_w=tsha(torch.flip(output.logits, dims=[-1])),
                    feat=tsha(output.feat_s16), feat_flip_w=tsha(torch.flip(output.feat_s16, dims=[-1])))
        record["calls"].append(call)
    teacher.register_forward_pre_hook(pre)
    teacher.register_forward_hook(post)

    if mutation == "call2_inside_probe":        # the trainer's step-1 teacher probe, kept to re-register it
        orig_probe = td.input_probe

        def input_probe(seen, name):
            hook = orig_probe(seen, name)
            if name == "teacher":
                state["probe"] = hook
            return hook
        td.input_probe = input_probe

    orig_build = ft.FlipTargets.build

    def build(self, tchr, model_input, teacher_out):
        entry = {"step": state["step"], "model_input_is_batch": model_input is state["batch"],
                 "teacher_out_logits": tsha(teacher_out.logits), "feat_before": tsha(teacher_out.feat_s16),
                 "cpu_rng_before": tsha(torch.get_rng_state()), "draws_before": stream_draws()}
        extra = None
        if mutation == "call2_inside_probe" and state["step"] == 1 and state["probe"] is not None:
            extra = tchr.teacher.register_forward_pre_hook(state["probe"])     # call 2 inside the probe
        state["in_build"] = True
        try:
            targets = orig_build(self, tchr, model_input, teacher_out)
        finally:
            state["in_build"] = False
            if extra is not None:
                extra.remove()
        entry.update(feat_after=tsha(teacher_out.feat_s16), cpu_rng_after=tsha(torch.get_rng_state()),
                     draws_after=stream_draws(), view1=tsha(targets.view1_logits), view2=tsha(targets.view2_logits),
                     view1_is_teacher_out_logits=targets.view1_logits is teacher_out.logits,
                     finite=bool(targets.finite))
        record["builds"].append(entry)
        state["targets"] = targets
        return targets
    ft.FlipTargets.build = build

    for term in ("logit_kd_loss", "cwd_logit_loss"):
        orig_term = getattr(ft.FlipViewTargets, term)

        def wrapped(self, head_logits, valid, T, _orig=orig_term, _term=term):
            record["masks"].append({"step": state["step"], "term": _term, "valid": tsha(valid), "T": T})
            return _orig(self, head_logits, valid, T)
        setattr(ft.FlipViewTargets, term, wrapped)

    orig_dl = td.distillation_losses

    def distillation_losses(*args, **kw):
        tgt, tout = kw.get("targets"), kw["teacher_out"]
        valid = downsample_validity(kw["mask"], kw["head_logits"].shape[-2:], ignore_index=td.IGNORE_INDEX)
        record["losses"].append({"step": state["step"], "has_targets": tgt is not None,
                                 "targets_are_built": tgt is not None and tgt is state["targets"],
                                 "feat": tsha(tout.feat_s16), "logits": tsha(tout.logits),
                                 "valid_os8": tsha(valid), "valid_os8_flip": tsha(torch.flip(valid, dims=[-1]))})
        return orig_dl(*args, **kw)
    td.distillation_losses = distillation_losses


def flip_worker(argv) -> int:
    """invariance_harness's worker, unchanged, with instrument()'s recorders added when it installs its own
    recorder, just before its run() call; the records and each checkpoint's arm fields go to <out>/flip_record.json."""
    ap = argparse.ArgumentParser(prog="smoke_kd_flip.py --internal flip_worker")
    ap.add_argument("--arm-json", default=None)
    ap.add_argument("--mutation", default="none", choices=("none", *RUN_MUTATIONS))
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--out", required=True)
    a, rest = ap.parse_known_args(argv)
    _child_paths(a.arm_json, a.code_root, a.out, *rest)
    code_root = Path(a.code_root).resolve()
    import scripts.invariance_harness as harness
    install = harness._Recorder.install
    record: dict = {}

    def install_with_flip(rec) -> None:
        install(rec)
        instrument(rec, arm_json=a.arm_json, mutation=a.mutation, code_root=code_root, record=record)
    harness._Recorder.install = install_with_flip
    rc = harness.worker(["--code-root", str(code_root), "--out", a.out, *rest])
    import torch
    record["payloads"] = []
    for p in sorted(Path(a.out).glob("ckpt_*/*_student_best_iter*.pt")):
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        record["payloads"].append({"file": p.name,
                                   **{k: ck.get(k, "<absent>") for k in ("arm", "target_construction")}})
    (Path(a.out) / "flip_record.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return rc


INTERNAL = {"flip_worker": flip_worker}


def run_flip_worker(*, work: Path, data_root: Path, label: str, threads: int, arm_json=None,
                    mutation: str = "none") -> dict:
    """One trainer run in a fresh process, in invariance_harness.run_worker's environment: its exit code, the
    harness's run summary and step JSONs, the recorders' record and the trainer's printed checks."""
    out = work / label
    args = ["--code-root", REPO, "--data-root", data_root, "--out", out, "--stage", "e3", "--steps", STEPS,
            "--val-interval", VAL_INTERVAL, "--threads", threads]
    if arm_json is not None:
        args += ["--arm-json", arm_json]
    if mutation != "none":
        args += ["--mutation", mutation]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads), "PLANTSEG_DATA_ROOT": str(data_root)})
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--internal", "flip_worker", *map(str, args)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT, cwd=str(work), env=env)
        rc, text = proc.returncode, proc.stdout + "\n--- stderr ---\n" + proc.stderr
    except subprocess.TimeoutExpired:
        rc, text = None, f"timed out after {WORKER_TIMEOUT} s"
    log = work / f"{label}.worker.log"
    log.write_text(text, encoding="utf-8")

    def read(name):
        p = out / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    stdout = out / "trainer_stdout.txt"
    trainer = stdout.read_text(encoding="utf-8") if stdout.is_file() else ""
    checks = ({ln.split(":")[0].strip(): ln.rsplit(":", 1)[1].strip() == "PASS"
               for ln in trainer.split("\n[CHECKS]\n", 1)[1].splitlines()
               if ln.startswith("  ") and ln.rstrip().endswith(("PASS", "FAIL"))}
              if "\n[CHECKS]\n" in trainer else {})
    return {"label": label, "returncode": rc, "seconds": round(time.time() - t0, 1), "log": str(log), "out": out,
            "summary": read("run_summary.json"), "record": read("flip_record.json"), "trainer_checks": checks,
            "steps": [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("step_*.json"))]}


# ===================================================================================== function-level tools
def cellwise(m, w):
    """sum over k of w[c, k] * m[:, k] at every cell, in one order and with elementwise operations only, so that a
    flipped input gives exactly the flipped map."""
    return (w[None, :, 0, None, None] * m[:, 0:1] + w[None, :, 1, None, None] * m[:, 1:2]) \
        + w[None, :, 2, None, None] * m[:, 2:3]


def nmf_draw():
    """What the function-level stub draws from its stream at each forward."""
    import torch
    return torch.rand(NUM_CLASSES + FEAT_CH)


def stub_teacher(*, scale: float = 3.0, draws: int = 1, global_rng: bool = False, nonfinite_forward=None):
    """A FrozenTeacher around the function-level stub, its M4-KD stream begun as the trainer begins it. Each forward
    takes `draws` draws from its stream (r: the last one, zeros without a draw) and returns the logits
    cellwise(maxpool8(x)) * (1 + scale r_c) and the Stage 3 cellwise(maxpool16(x)) * (1 + scale r_k). With global_rng
    it also draws from the caller's CPU RNG; forward number `nonfinite_forward` returns one NaN logit."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.distill.nmf_stream import NMFStream
    from src.distill.teacher import FrozenTeacher

    class FlipStubTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(STUB_SEED)
            self.w_logits = nn.Parameter(torch.randn(NUM_CLASSES, 3, generator=g))
            self.w_feat = nn.Parameter(torch.randn(FEAT_CH, 3, generator=g))
            self.nmf_stream = None
            self.forwards = 0

        def begin_nmf_stream(self, policy, seed=42):
            self.nmf_stream = NMFStream(seed, policy)
            return self.nmf_stream.describe()

        def forward(self, x):
            self.forwards += 1
            r = torch.zeros(NUM_CLASSES + FEAT_CH)
            for _ in range(draws):
                r = self.nmf_stream.draw(nmf_draw)
            if global_rng:
                torch.rand(1)
            logits = cellwise(F.max_pool2d(x, 8), self.w_logits) * (1.0 + scale * r[:NUM_CLASSES]).view(1, -1, 1, 1)
            feat = cellwise(F.max_pool2d(x, 16), self.w_feat) * (1.0 + scale * r[NUM_CLASSES:]).view(1, -1, 1, 1)
            if self.forwards == nonfinite_forward:
                logits = logits.clone()
                logits[0, 0, 0, 0] = float("nan")
            return {"logits": logits, "feat_s16": feat}

    teacher = FrozenTeacher(FlipStubTeacher())
    teacher.begin_nmf_stream("M4-KD", 42)
    return teacher


def flip_scenario(ft, *, scale: float = 3.0, **kw):
    """Call 1 as the trainer makes it, then build(): what call 2 received and returned, the CPU RNG state around
    build(), the stream's draws, and the two views computed here from its replayed draws 1 and 2."""
    import torch
    import torch.nn.functional as F
    from src.distill.nmf_stream import NMFStream
    teacher = stub_teacher(scale=scale, **kw)
    x = torch.randn(BATCH, 3, 64, 64, generator=torch.Generator().manual_seed(11))
    out1 = teacher(x)
    feat1 = out1.feat_s16.clone()
    calls = []
    hook = teacher.register_forward_hook(lambda _m, inputs, output: calls.append((inputs[0], output)))
    rng0 = torch.get_rng_state()
    targets = error = None
    try:
        targets = ft.FlipTargets().build(teacher, x, out1)
    except Exception as e:  # noqa: BLE001 - the scenario's result
        error = e
    finally:
        hook.remove()
    rng_unchanged = torch.equal(rng0, torch.get_rng_state())
    replay = NMFStream(42, "M4-KD")
    r1, r2 = replay.draw(nmf_draw), replay.draw(nmf_draw)
    base = cellwise(F.max_pool2d(x, 8), teacher.teacher.w_logits)
    return types.SimpleNamespace(
        teacher=teacher, x=x, out1=out1, feat1=feat1, calls=calls, targets=targets, error=error,
        rng_unchanged=rng_unchanged, draws=teacher.nmf_stream_state()["draws"],
        z1=base * (1.0 + scale * r1[:NUM_CLASSES]).view(1, -1, 1, 1),
        z2=base * (1.0 + scale * r2[:NUM_CLASSES]).view(1, -1, 1, 1))


def math_fixture():
    """Student head logits and two distinct views [3, 116, 8, 8], and a validity mask: sample 0 asymmetric (a flipped
    mask differs), sample 1 all valid, sample 2 without a valid cell."""
    import torch
    g = torch.Generator().manual_seed(21)
    shape = (3, NUM_CLASSES, 8, 8)
    s = torch.randn(shape, generator=g) * 2.0
    z1 = torch.randn(shape, generator=g) * 3.0
    z2 = torch.randn(shape, generator=g) * 3.0
    valid = torch.ones(3, 8, 8, dtype=torch.bool)
    valid[0, :, :3] = False
    valid[0, 5:, 6] = False
    valid[2] = False
    return s, z1, z2, valid


def ref_logit_kd(s, z1, z2, valid, T, *, average_logits=False):
    """Float64 reference of W's Logit-KD term (with average_logits: of the softmax of the averaged logits)."""
    import torch
    s, z1, z2 = s.double(), z1.double(), z2.double()
    p = (torch.softmax((z1 + z2) / 2 / T, dim=1) if average_logits
         else (torch.softmax(z1 / T, dim=1) + torch.softmax(z2 / T, dim=1)) / 2)
    kl = (torch.xlogy(p, p) - p * torch.log_softmax(s / T, dim=1)).sum(dim=1) * (T * T)
    v = valid.double()
    return (kl * v).sum() / v.sum().clamp_min(1.0)


def ref_cwd(s, z1, z2, valid, T, *, average_logits=False):
    """Float64 reference of W's CWD logit-map term: per sample and channel, softmaxes over the valid cells only."""
    import torch
    b, c = s.shape[:2]
    per = []
    for i in range(b):
        m = valid[i].reshape(-1)
        if not bool(m.any()):
            per.append(torch.zeros((), dtype=torch.float64))
            continue

        def cells(z):
            return z[i].double().reshape(c, -1)[:, m] / T
        q = (torch.softmax((cells(z1) + cells(z2)) / 2, dim=1) if average_logits
             else (torch.softmax(cells(z1), dim=1) + torch.softmax(cells(z2), dim=1)) / 2)
        per.append((torch.xlogy(q, q) - q * torch.log_softmax(cells(s), dim=1)).sum() * (T * T) / c)
    return torch.stack(per).mean()


def value_and_grad(fn, s):
    import torch
    s = s.detach().clone().requires_grad_(True)
    out = fn(s)
    (g,) = torch.autograd.grad(out, s)
    return out.detach(), g


def rel(a, b) -> float:
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def grad_rel(g, g_ref) -> float:
    return float((g.double() - g_ref.double()).abs().max()) / max(float(g_ref.double().abs().max()), 1e-30)


def bitwise(a, b) -> bool:
    import torch
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a, b))


def raises(fn, exc, text: str) -> bool:
    try:
        fn()
    except exc as e:
        return text in str(e)
    except Exception:  # noqa: BLE001 - another exception is not the refusal
        return False
    return False


def temperatures() -> tuple:
    """The trainer's T_logit and T_cwd (configs/distill.py, as train_distill.py reads them)."""
    from configs.distill import DISTILL
    return DISTILL["logit_kd"]["T_logit"], DISTILL["cwd"]["T_cwd"]


def write_arm_json(folder: Path, **over) -> Path:
    """A kd_arm/1 file named e3_flip.json outside the registry: the registry entry's values with `over` applied."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "e3_flip.json"
    path.write_text(json.dumps({**REGISTRY_ENTRY, **over}, indent=2) + "\n", encoding="utf-8")
    return path


def abort_record_expression():
    """train_distill.run()'s nested abort(): the expression its record's teacher_finite takes, compiled from the
    file's own source, and for each abort() call in run() its line and whether an enclosing `if` tests
    `mode == "real"` or `am7b is not None` (the AM-7 (b) monitor exists only in a real run)."""
    src = (REPO / "src" / "training" / "train_distill.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    abort = next(n for n in ast.walk(run) if isinstance(n, ast.FunctionDef) and n.name == "abort")
    assign = next(n for n in ast.walk(abort) if isinstance(n, ast.Assign)
                  and [getattr(t, "id", None) for t in n.targets] == ["teacher_finite"])
    code = compile(ast.Expression(body=assign.value), f"train_distill.py:{assign.lineno}", "eval")
    parents = {child: node for node in ast.walk(run) for child in ast.iter_child_nodes(node)}
    guards = []
    for call in (n for n in ast.walk(run) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "abort"):
        node, ok = call, False
        while node in parents and not ok:
            node = parents[node]
            if isinstance(node, ast.If):
                test = ast.get_source_segment(src, node.test) or ""
                ok = 'mode == "real"' in test or "am7b is not None" in test
        guards.append((call.lineno, ok))
    return code, ast.get_source_segment(src, assign.value), assign.lineno, sorted(guards)


# ============================================================================================== the checks
def run_steps(r: dict) -> list:
    return [b["step"] for b in (r.get("record") or {}).get("batches", [])]


def at(r: dict, kind: str, t: int) -> list:
    return [e for e in (r.get("record") or {}).get(kind, []) if e["step"] == t]


def ran(r: dict) -> bool:
    """The run exited 0, its trainer printed RESULT: PASS, and it recorded every step."""
    sm = r.get("summary") or {}
    return (r.get("returncode") == 0 and str(sm.get("trainer_result", "")).startswith("RESULT: PASS")
            and sm.get("n_step_records") == STEPS and run_steps(r) == list(range(1, STEPS + 1)))


def meta_row(r: dict):
    rows = (r.get("summary") or {}).get("run_meta") or []
    return rows[0] if len(rows) == 1 and isinstance(rows[0], dict) else None


def canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), allow_nan=True).encode("utf-8")


def without_identity(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in IDENTITY_KEYS}


def verdict(parts: dict, extra: str = "") -> tuple:
    bad = [k for k, ok in parts.items() if not ok]
    return not bad, (f"failed: {bad}" if bad else f"{len(parts)} parts hold") + (f"; {extra}" if extra else "")


def c_two_calls(ft, runs):
    sc = flip_scenario(ft)
    parts = {"build() calls the teacher once": sc.error is None and len(sc.calls) == 1}
    w = runs["flip"]
    parts["W run recorded every step"] = run_steps(w) == list(range(1, STEPS + 1))
    for t in run_steps(w):
        cs, b, bs = at(w, "calls", t), at(w, "batches", t)[0], at(w, "builds", t)
        parts[f"step {t}: the trainer's call on the batch, then build()'s, at batch {BATCH}"] = (
            len(cs) == 2 and cs[0]["x_is_batch"] and cs[0]["x"] == b["img"] and not cs[0]["in_build"]
            and cs[1]["in_build"] and cs[0]["batch_size"] == cs[1]["batch_size"] == BATCH and len(bs) == 1
            and bs[0]["teacher_out_logits"] == cs[0]["logits"] and bs[0]["model_input_is_batch"])
    return verdict(parts)


def c_unflipped_call(ft, runs):
    w, e3 = runs["flip"], runs["e3"]
    parts = {"the W and E3 runs finished": ran(w) and ran(e3),
             "E3 makes one call per step": bool(run_steps(e3)) and all(len(at(e3, "calls", t)) == 1
                                                                      for t in run_steps(e3))}
    wc, ec = at(w, "calls", 1), at(e3, "calls", 1)
    parts["step 1: call 1 is the trainer's, inside the probe, as E3's"] = (
        bool(wc) and bool(ec) and not wc[0]["in_build"] and wc[0]["x_is_batch"]
        and wc[0]["probe_hooks"] == ec[0]["probe_hooks"] == 1)
    parts["step 1: the same input, logits, Stage 3, stream position and CPU RNG as E3's call"] = (
        bool(wc) and bool(ec) and all(wc[0][k] == ec[0][k] for k in ("x", "logits", "feat", "draws_before",
                                                                     "draws_after", "cpu_rng_before")))
    ws = ((w.get("steps") or [{}])[0]).get("telemetry", {})
    es = ((e3.get("steps") or [{}])[0]).get("telemetry", {})
    parts["step 1: loss, sup, ce, dice and the feature term are E3's bit for bit"] = bool(ws) and all(
        k in ws and canonical(ws[k]) == canonical(es.get(k)) for k in ("loss", "sup", "ce", "dice", "cwd_feat"))
    return verdict(parts)


def c_second_call_hflip(ft, runs):
    import torch
    sc = flip_scenario(ft)
    x2 = sc.calls[0][0] if sc.calls else None
    parts = {"build()'s call gets torch.flip(x, dims=[-1])": (
        x2 is not None and bitwise(x2, torch.flip(sc.x, dims=[-1])) and not bitwise(x2, torch.flip(sc.x, dims=[-2]))
        and not bitwise(x2, sc.x))}
    w = runs["flip"]
    parts["W run recorded every step"] = run_steps(w) == list(range(1, STEPS + 1))
    for t in run_steps(w):
        cs, b = at(w, "calls", t), at(w, "batches", t)[0]
        parts[f"step {t}: call 2's input is the batch's width flip"] = (
            len(cs) == 2 and cs[1]["x"] == b["img_flip_w"] and cs[1]["x"] not in (b["img"], b["img_flip_h"]))
    return verdict(parts)


def c_flip_back(ft, runs):
    import torch
    sc = flip_scenario(ft)
    out2 = sc.calls[0][1] if sc.calls else None
    parts = {"view 2 is the aligned map, call 2's logits flipped back": (
        sc.error is None and out2 is not None and bitwise(sc.targets.view2_logits, sc.z2)
        and bitwise(sc.targets.view2_logits, torch.flip(out2.logits, dims=[-1]))
        and not bitwise(sc.targets.view2_logits, out2.logits))}
    w = runs["flip"]
    parts["W run recorded every step"] = run_steps(w) == list(range(1, STEPS + 1))
    for t in run_steps(w):
        cs, bs = at(w, "calls", t), at(w, "builds", t)
        parts[f"step {t}: view 2 is call 2's logits flipped back"] = (
            len(cs) == 2 and len(bs) == 1 and bs[0]["view2"] == cs[1]["logits_flip_w"]
            and bs[0]["view2"] != cs[1]["logits"])
    return verdict(parts)


def c_logit_kd_t4(ft, runs):
    s, z1, z2, valid = math_fixture()
    T, _ = temperatures()
    t = ft.FlipViewTargets(z1, z2)
    mine, g = value_and_grad(lambda x: t.logit_kd_loss(x, valid, T), s)
    ref, g_ref = value_and_grad(lambda x: ref_logit_kd(x, z1, z2, valid, T), s)
    t1 = ref_logit_kd(s, z1, z2, valid, 1)
    parts = {"T_logit is 4": T == 4, "the value is the reference's": rel(mine, ref) <= REL_TOL,
             "the gradient is the reference's": grad_rel(g, g_ref) <= REL_TOL,
             "the fixture tells T 4 from T 1": rel(t1, ref) > DISCRIMINATE}
    return verdict(parts, f"loss {float(mine)!r}, float64 reference {float(ref)!r} (rel {rel(mine, ref):.1e}), "
                          f"gradient rel {grad_rel(g, g_ref):.1e}; the reference at T 1: {float(t1)!r}")


def c_cwd_t4(ft, runs):
    import torch
    s, z1, z2, valid = math_fixture()
    _, T = temperatures()
    t = ft.FlipViewTargets(z1, z2)
    mine, g = value_and_grad(lambda x: t.cwd_logit_loss(x, valid, T), s)
    ref, g_ref = value_and_grad(lambda x: ref_cwd(x, z1, z2, valid, T), s)
    m = valid.reshape(valid.shape[0], 1, -1)
    q = ft.average(ft.spatial_probs(z1, m, T), ft.spatial_probs(z2, m, T))
    has = m.any(dim=-1).squeeze(1)
    sums = q[has].sum(dim=-1)
    t2 = ft.FlipViewTargets(z1[2:].contiguous(), z2[2:].contiguous())
    parts = {"T_cwd is 4": T == 4, "the value is the reference's": rel(mine, ref) <= REL_TOL,
             "the gradient is the reference's": grad_rel(g, g_ref) <= REL_TOL,
             "each channel's target sums to 1 over its valid cells and is 0 elsewhere": (
                 bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
                 and bool((q[has].masked_select(~m[has].expand_as(q[has])) == 0).all())),
             "a sample with no valid cell gives 0": float(t2.cwd_logit_loss(s[2:].contiguous(), valid[2:], T)) == 0.0}
    return verdict(parts, f"loss {float(mine)!r}, float64 reference {float(ref)!r} (rel {rel(mine, ref):.1e}), "
                          f"gradient rel {grad_rel(g, g_ref):.1e}")


def c_per_term_softmax(ft, runs):
    import torch
    s, z1, z2, valid = math_fixture()
    t = ft.FlipViewTargets(z1, z2)
    kd4, cwd2 = t.logit_kd_loss(s, valid, 4), t.cwd_logit_loss(s, valid, 2)
    r_kd4, r_cwd2 = ref_logit_kd(s, z1, z2, valid, 4), ref_cwd(s, z1, z2, valid, 2)
    m = valid.reshape(valid.shape[0], 1, -1)
    cp, sp = ft.class_probs(z1, 4), ft.spatial_probs(z1, m, 4)
    has = m.any(dim=-1).squeeze(1)
    parts = {"Logit KD at T 4 is its reference": rel(kd4, r_kd4) <= REL_TOL,
             "CWD at T 2 is its reference": rel(cwd2, r_cwd2) <= REL_TOL,
             "the fixture tells the CWD term's T apart": rel(r_cwd2, ref_cwd(s, z1, z2, valid, 4)) > DISCRIMINATE,
             "Logit KD's softmax is over the classes at each cell": (
                 tuple(cp.shape) == tuple(z1.shape) and bool(torch.allclose(cp.sum(dim=1), torch.ones(1), atol=1e-5))),
             "CWD's softmax is over the cells of each channel": (
                 tuple(sp.shape) == (z1.shape[0], z1.shape[1], z1.shape[2] * z1.shape[3])
                 and bool(torch.allclose(sp[has].sum(dim=-1), torch.ones(1), atol=1e-5)))}
    return verdict(parts, f"Logit KD rel {rel(kd4, r_kd4):.1e}, CWD rel {rel(cwd2, r_cwd2):.1e}")


def c_never_logits(ft, runs):
    s, z1, z2, valid = math_fixture()
    T_logit, T_cwd = temperatures()
    t = ft.FlipViewTargets(z1, z2)
    kd, cw = t.logit_kd_loss(s, valid, T_logit), t.cwd_logit_loss(s, valid, T_cwd)
    p_kd, l_kd = ref_logit_kd(s, z1, z2, valid, T_logit), ref_logit_kd(s, z1, z2, valid, T_logit,
                                                                        average_logits=True)
    p_cw, l_cw = ref_cwd(s, z1, z2, valid, T_cwd), ref_cwd(s, z1, z2, valid, T_cwd, average_logits=True)
    parts = {"Logit KD averages probabilities": rel(kd, p_kd) <= REL_TOL and rel(kd, l_kd) > DISCRIMINATE,
             "CWD averages probabilities": rel(cw, p_cw) <= REL_TOL and rel(cw, l_cw) > DISCRIMINATE}
    return verdict(parts, f"Logit KD {float(kd)!r} (probabilities {float(p_kd)!r}, logits {float(l_kd)!r}); CWD "
                          f"{float(cw)!r} (probabilities {float(p_cw)!r}, logits {float(l_cw)!r})")


def c_equal_views(ft, runs):
    import torch
    from src.training.losses import cwd_channelwise_kl, logit_kd_kl
    s, z1, _, valid = math_fixture()
    T_logit, T_cwd = temperatures()
    t = ft.FlipViewTargets(z1, z1.clone())
    kd, g_kd = value_and_grad(lambda x: t.logit_kd_loss(x, valid, T_logit), s)
    e3, g_e3 = value_and_grad(lambda x: logit_kd_kl(x, z1, valid, T=T_logit), s)
    cw, g_cw = value_and_grad(lambda x: t.cwd_logit_loss(x, valid, T_cwd), s)
    e3c, g_e3c = value_and_grad(lambda x: cwd_channelwise_kl(x, z1, valid, T=T_cwd), s)
    sc = flip_scenario(ft, scale=0.0)
    head = torch.randn(BATCH, NUM_CLASSES, 8, 8, generator=torch.Generator().manual_seed(5))
    mask8 = torch.ones(BATCH, 8, 8, dtype=torch.bool)
    mask8[1, :2] = False
    parts = {"Logit KD is logit_kd_kl's bit for bit": bitwise(kd, e3),
             "and so is its gradient": bitwise(g_kd, g_e3),
             "CWD is cwd_channelwise_kl's within 1e-6 relative": rel(cw, e3c) <= EQUAL_VIEWS_REL_TOL,
             "and its gradient within 1e-5": grad_rel(g_cw, g_e3c) <= REL_TOL,
             "build() gives two equal views when the draws move nothing": (
                 sc.error is None and bitwise(sc.targets.view1_logits, sc.targets.view2_logits)),
             "and its Logit-KD term is E3's bit for bit": sc.error is None and bitwise(
                 sc.targets.logit_kd_loss(head, mask8, T_logit), logit_kd_kl(head, sc.out1.logits, mask8, T=T_logit))}
    return verdict(parts, f"CWD rel {rel(cw, e3c):.1e}, gradient rel {grad_rel(g_cw, g_e3c):.1e}")


def c_feature_term(ft, runs):
    import torch
    import src.training.train_distill as td
    from src.training.losses import downsample_validity
    sc = flip_scenario(ft)
    parts = {"build() leaves call 1's Stage 3 untouched": sc.error is None and bitwise(sc.out1.feat_s16, sc.feat1)}
    if sc.error is None:
        g = torch.Generator().manual_seed(31)
        stage = td.resolve_stage("e3")
        head = torch.randn(BATCH, NUM_CLASSES, 8, 8, generator=g)
        logits = torch.randn(BATCH, NUM_CLASSES, 64, 64, generator=g)
        c5 = torch.randn(BATCH, 160, 4, 4, generator=g)
        mask = torch.randint(0, NUM_CLASSES, (BATCH, 64, 64), generator=g)
        mask[0, :, :16] = td.IGNORE_INDEX
        torch.manual_seed(0)
        projection = td.build_cwd_projection_for(stage)
        kw = dict(stage=stage, logits=logits, head_logits=head, c5=c5, mask=mask, teacher_out=sc.out1,
                  projection=projection, lambda_logit=1.0, ramp=1.0, alpha=50.0)
        _, e3 = td.distillation_losses(**kw)
        _, w = td.distillation_losses(**kw, targets=sc.targets)
        valid = downsample_validity(mask, head.shape[-2:], ignore_index=td.IGNORE_INDEX)
        T_logit, T_cwd = temperatures()
        parts["the trainer's feature term with W's targets is E3's bit for bit"] = w["cwd_feat"] == e3["cwd_feat"]
        parts["its two logit terms are the targets' own"] = (
            w["logit_kd"] == float(sc.targets.logit_kd_loss(head, valid, T_logit))
            and w["cwd_logit"] == float(sc.targets.cwd_logit_loss(head, valid, T_cwd))
            and w["logit_kd"] != e3["logit_kd"])
    w_run = runs["flip"]
    parts["W run recorded every step"] = run_steps(w_run) == list(range(1, STEPS + 1))
    for t in run_steps(w_run):
        cs, ls, bs = at(w_run, "calls", t), at(w_run, "losses", t), at(w_run, "builds", t)
        parts[f"step {t}: the feature term reads call 1's Stage 3"] = (
            len(cs) == 2 and len(ls) == 1 and len(bs) == 1 and ls[0]["feat"] == cs[0]["feat"]
            and ls[0]["feat"] not in (cs[1]["feat"], cs[1]["feat_flip_w"])
            and bs[0]["feat_before"] == bs[0]["feat_after"])
    return verdict(parts)


def c_nmf_draws(ft, runs):
    sc = flip_scenario(ft)
    parts = {"call 1 takes draw 1 and build()'s call draw 2": sc.error is None and sc.draws == 2,
             "view 1 is call 1's output (draw 1)": (sc.error is None and sc.targets.view1_logits is sc.out1.logits
                                                    and bitwise(sc.targets.view1_logits, sc.z1)),
             "view 2 is call 2's (draw 2)": (sc.error is None and bitwise(sc.targets.view2_logits, sc.z2)
                                             and not bitwise(sc.z1, sc.z2))}
    for d in (0, 2):
        bad = flip_scenario(ft, draws=d)
        parts[f"build() refuses a call 2 that takes {d} draws"] = (isinstance(bad.error, ft.FlipTargetsError)
                                                                   and "[flip_nmf_draws]" in str(bad.error))
    w = runs["flip"]
    parts["W run recorded every step"] = run_steps(w) == list(range(1, STEPS + 1))
    for t in run_steps(w):
        cs, bs = at(w, "calls", t), at(w, "builds", t)
        parts[f"step {t}: draws {2 * t - 1} (the crop) and {2 * t} (the flip), one per forward"] = (
            len(cs) == 2 and len(bs) == 1 and cs[0]["batch_size"] == cs[1]["batch_size"] == BATCH
            and (cs[0]["draws_before"], cs[0]["draws_after"], cs[1]["draws_before"], cs[1]["draws_after"])
            == (2 * t - 2, 2 * t - 1, 2 * t - 1, 2 * t)
            and bs[0]["view1"] == cs[0]["logits"] and bs[0]["view1_is_teacher_out_logits"]
            and bs[0]["view2"] == cs[1]["logits_flip_w"])
    steps = w.get("steps") or []
    parts["the harness counts two draws per step"] = len(steps) == STEPS and all(
        (st.get("nmf") or {}).get("draws") == 2 * st["step"] and st.get("build_bases_calls") == 2 for st in steps)
    return verdict(parts)


def c_cpu_rng(ft, runs):
    sc = flip_scenario(ft)
    bad = flip_scenario(ft, global_rng=True)
    parts = {"build() leaves the CPU RNG state unchanged": sc.error is None and sc.rng_unchanged,
             "build() refuses a teacher that draws from the CPU RNG": (isinstance(bad.error, ft.FlipTargetsError)
                                                                        and "[flip_cpu_rng]" in str(bad.error))}
    w, e3 = runs["flip"], runs["e3"]
    calls, builds = (w.get("record") or {}).get("calls", []), (w.get("record") or {}).get("builds", [])
    parts["the W and E3 runs finished"] = ran(w) and ran(e3)
    parts["no teacher call and no build() changes the CPU RNG state"] = (
        len(calls) == 2 * STEPS and all(c["cpu_rng_before"] == c["cpu_rng_after"] for c in calls)
        and len(builds) == STEPS and all(b["cpu_rng_before"] == b["cpu_rng_after"] for b in builds))
    ws, es = w.get("steps") or [], e3.get("steps") or []
    parts["the CPU, numpy and Python RNG states after each step are E3's"] = (
        len(ws) == len(es) == STEPS and all(a["rng_after_step"] == b["rng_after_step"] for a, b in zip(ws, es)))
    return verdict(parts)


def c_student(ft, runs):
    parts = {"build() never receives the student": list(inspect.signature(ft.FlipTargets.build).parameters)
             == ["self", "teacher", "model_input", "teacher_out"]}
    w = runs["flip"]
    parts["W run recorded every step"] = run_steps(w) == list(range(1, STEPS + 1))
    for t in run_steps(w):
        fw = [s for s in at(w, "student", t) if s["training"]]
        b = at(w, "batches", t)[0]
        parts[f"step {t}: one training forward, on the batch itself"] = (
            len(fw) == 1 and fw[0]["x_is_batch"] and fw[0]["x"] == b["img"] and fw[0]["x"] != b["img_flip_w"])
    return verdict(parts)


def c_step1(ft, runs):
    w = runs["flip"]
    checks = w.get("trainer_checks") or {}
    cs = at(w, "calls", 1)
    later = [c for c in (w.get("record") or {}).get("calls", []) if c["step"] > 1]
    parts = {"the trainer's 20 checks PASS": sorted(checks) == sorted(HARD_CHECKS) and all(checks.values()),
             "RESULT: PASS": (w.get("returncode") == 0
                              and str((w.get("summary") or {}).get("trainer_result", "")).startswith("RESULT: PASS")),
             "step 1: the probe's hook is on call 1 and gone by call 2": (
                 len(cs) == 2 and cs[0]["probe_hooks"] == 1 and cs[1]["probe_hooks"] == 0),
             "no probe hook after step 1": bool(later) and all(c["probe_hooks"] == 0 for c in later)}
    return verdict(parts, f"failing trainer checks: {sorted(k for k, ok in checks.items() if not ok)}")


def c_k_not_1(ft, runs, work: Path):
    from src.training import kd_arms
    parts = {"FlipTargets is K 1 and VIEWS 2": ft.FlipTargets.K == 1 and ft.FlipTargets.VIEWS == 2}
    for k in (8, 0, True):
        cls = type("FlipTargetsOtherK", (ft.FlipTargets,), {"K": k})
        parts[f"a builder with K {k!r} refuses itself"] = raises(cls, ft.FlipTargetsError, "[flip_k]")
    for k in (2, 8):
        p = write_arm_json(work / "arm_json" / f"k{k}", K=k)
        parts[f"kd_arm/1 refuses K {k}"] = raises(lambda p=p: kd_arms.parse_arm(p), kd_arms.ArmSpecError, f"K {k}")
    spec = kd_arms.parse_arm(write_arm_json(work / "arm_json" / "k1"))
    parts["an arm with K 1 resolves FlipTargets"] = (spec.K == 1 and spec.views == 2
                                                     and spec.builder_class() is ft.FlipTargets)
    saved = ft.FlipTargets.K
    ft.FlipTargets.K = 8
    try:
        parts["an arm with K 1 refuses a builder with K 8"] = raises(spec.builder_class, kd_arms.ArmSpecError,
                                                                    "FlipTargets.K is 8")
        parts["and FlipTargets() refuses K 8"] = raises(ft.FlipTargets, ft.FlipTargetsError, "[flip_k]")
    finally:
        ft.FlipTargets.K = saved
    return verdict(parts)


def c_mask(ft, runs):
    import torch
    s, z1, z2, valid = math_fixture()
    T_logit, T_cwd = temperatures()
    t = ft.FlipViewTargets(z1, z2)
    flipped = torch.flip(valid, dims=[-1])
    kd, cw = t.logit_kd_loss(s, valid, T_logit), t.cwd_logit_loss(s, valid, T_cwd)
    r_kd, r_cw = ref_logit_kd(s, z1, z2, valid, T_logit), ref_cwd(s, z1, z2, valid, T_cwd)
    parts = {"both terms use the mask as given": rel(kd, r_kd) <= REL_TOL and rel(cw, r_cw) <= REL_TOL,
             "the fixture tells a flipped mask apart": (
                 rel(r_kd, ref_logit_kd(s, z1, z2, flipped, T_logit)) > DISCRIMINATE
                 and rel(r_cw, ref_cwd(s, z1, z2, flipped, T_cwd)) > DISCRIMINATE)}
    w = runs["flip"]
    parts["W run recorded every step"] = run_steps(w) == list(range(1, STEPS + 1))
    for t_ in run_steps(w):
        ms, ls = at(w, "masks", t_), at(w, "losses", t_)
        parts[f"step {t_}: both terms get the crop's valid_os8"] = (
            len(ms) == 2 and len(ls) == 1 and {m["term"] for m in ms} == {"logit_kd_loss", "cwd_logit_loss"}
            and all(m["valid"] == ls[0]["valid_os8"] for m in ms) and ls[0]["has_targets"]
            and ls[0]["targets_are_built"])
    return verdict(parts)


def c_abort(ft, runs):
    import torch
    code, text, line, guards = abort_record_expression()
    bad, good = flip_scenario(ft, nonfinite_forward=2), flip_scenario(ft)

    def record_value(targets, tout):
        return eval(code, {"torch": torch}, {"teacher_out": tout, "targets": targets})  # noqa: S307 - the file's own
    parts = {"with call 2 non-finite, finite is false": bad.error is None and bad.targets.finite is False,
             "call 1 is finite": (bool(torch.isfinite(bad.out1.logits).all())
                                  and bool(torch.isfinite(bad.out1.feat_s16).all())),
             "the record's teacher_finite is false with those targets": (
                 bad.error is None and record_value(bad.targets, bad.out1) is False),
             "and true without targets": record_value(None, bad.out1) is True,
             "and true with finite targets": (good.error is None and good.targets.finite is True
                                              and record_value(good.targets, good.out1) is True),
             "the expression reads targets.finite": "targets.finite" in text,
             "every abort() call in run() is real-run only": bool(guards) and all(ok for _, ok in guards)}
    return verdict(parts, f"train_distill.py:{line} teacher_finite = {' '.join(text.split())}; abort() calls at "
                          f"lines {[ln for ln, _ in guards]}, each under `mode == \"real\"` or `am7b is not None`: "
                          f"{[ok for _, ok in guards]}")


def c_determinism(ft, runs):
    a, b = runs["flip"], runs["flip_repeat"]
    cmp = compare_step_dirs(a["out"], b["out"])
    ra, rb = a["out"] / "flip_record.json", b["out"] / "flip_record.json"
    ma, mb = meta_row(a), meta_row(b)
    parts = {"both runs finished": ran(a) and ran(b), "the step JSONs are byte-identical": cmp["identical"],
             "the records are byte-identical": ra.is_file() and rb.is_file() and ra.read_bytes() == rb.read_bytes(),
             "the run_meta rows are equal but for run identity": (
                 ma is not None and mb is not None
                 and canonical(without_identity(ma)) == canonical(without_identity(mb)))}
    return verdict(parts, f"{cmp['n_old']}/{cmp['n_new']} step JSONs, first mismatch {cmp['mismatches'][:1]}")


CHECKS = (("cm_two_calls_per_step", c_two_calls), ("cm_unflipped_call_is_e3_call", c_unflipped_call),
          ("cm_second_call_on_hflip", c_second_call_hflip), ("cm_flip_back_alignment", c_flip_back),
          ("cm_logit_kd_prob_average_T4", c_logit_kd_t4), ("cm_cwd_logit_prob_average_T4_per_channel", c_cwd_t4),
          ("cm_per_term_softmax", c_per_term_softmax), ("cm_never_logits", c_never_logits),
          ("cm_equal_views_match_single_view", c_equal_views), ("cm_feature_term_unflipped_stage3", c_feature_term),
          ("cm_nmf_two_draws_per_step", c_nmf_draws), ("cm_cpu_rng_unchanged_by_call2", c_cpu_rng),
          ("cm_student_one_forward_original_crop", c_student), ("cm_step1_checks_pass", c_step1),
          ("cm_k_not_1_refused", c_k_not_1), ("cm_validity_mask_original_orientation", c_mask),
          ("cm_abort_teacher_finite_reads_targets", c_abort), ("dt_flip_two_runs_identical_bytes", c_determinism))
# Each mutation and the check that must fail under it; "run" mutations act in a worker's trainer run
MUTATIONS = (("average_logits", "cm_never_logits", "function"),
             ("no_flip_back", "cm_flip_back_alignment", "function"),
             ("shared_softmax", "cm_per_term_softmax", "function"),
             ("feature_from_flipped_view", "cm_feature_term_unflipped_stage3", "function"),
             ("one_batched_call", "cm_nmf_two_draws_per_step", "function"),
             ("student_sees_flip", "cm_student_one_forward_original_crop", "run"),
             ("t_1", "cm_logit_kd_prob_average_T4", "function"),
             ("flipped_mask", "cm_validity_mask_original_orientation", "function"),
             ("call2_inside_probe", "cm_step1_checks_pass", "run"),
             ("finite_always_true", "cm_abort_teacher_finite_reads_targets", "function"))


@contextlib.contextmanager
def mutated(ft, name: str):
    """A function-level mutation of flip_targets, undone on exit."""
    import torch
    saved = []

    def patch(obj, attr, value):
        saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)
    if name == "average_logits":             # the targets from the averaged logits
        init = ft.FlipViewTargets.__init__

        def averaged(self, v1, v2, _init=init):
            z = (v1 + v2) * 0.5
            _init(self, z, z)
        patch(ft.FlipViewTargets, "__init__", averaged)
    elif name == "no_flip_back":
        patch(ft, "flip_back", lambda logits: logits)
    elif name == "shared_softmax":           # the CWD term takes the Logit-KD term's softmax over the classes
        class_probs = ft.class_probs
        patch(ft, "spatial_probs", lambda z, m, T: class_probs(z, T).reshape(z.shape[0], z.shape[1], -1))
    elif name == "feature_from_flipped_view":  # call 1's Stage 3 overwritten with call 2's, flipped back
        build = ft.FlipTargets.build

        def from_flipped(self, teacher, model_input, teacher_out, _build=build):
            outs = []
            hook = teacher.register_forward_hook(lambda _m, _i, o: outs.append(o))
            try:
                targets = _build(self, teacher, model_input, teacher_out)
            finally:
                hook.remove()
            teacher_out.feat_s16.copy_(ft.flip_back(outs[-1].feat_s16))
            return targets
        patch(ft.FlipTargets, "build", from_flipped)
    elif name == "one_batched_call":         # both views from one forward on the crop and its flip
        def batched(self, teacher, model_input, teacher_out):
            b = model_input.shape[0]
            both = teacher(torch.cat([model_input, ft.flip_input(model_input)]))
            return ft.FlipViewTargets(both.logits[:b], ft.flip_back(both.logits[b:]))
        patch(ft.FlipTargets, "build", batched)
    elif name == "t_1":                      # the Logit-KD targets' softmax at T 1
        class_probs = ft.class_probs
        patch(ft, "class_probs", lambda z, T: class_probs(z, 1))
    elif name == "flipped_mask":
        for term in ("logit_kd_loss", "cwd_logit_loss"):
            orig = getattr(ft.FlipViewTargets, term)
            patch(ft.FlipViewTargets, term,
                  lambda self, h, v, T, _o=orig: _o(self, h, torch.flip(v, dims=[-1]), T))
    elif name == "finite_always_true":
        patch(ft.FlipViewTargets, "finite", property(lambda self: True))
    else:
        raise ValueError(f"no function-level mutation {name!r}")
    try:
        yield
    finally:
        for obj, attr, value in reversed(saved):
            setattr(obj, attr, value)


def evaluate(fn, ft, runs, work: Path) -> tuple:
    """A check's (ok, detail); a check that raises fails with the error."""
    try:
        return fn(ft, runs, work) if fn is c_k_not_1 else fn(ft, runs)
    except Exception as e:  # noqa: BLE001 - a check that raises fails
        return False, f"raised {type(e).__name__}: {e}"


# ===================================================================================================== main
def refused(text: str) -> int:
    print(f"REFUSED: {text}")
    print("RESULT: REFUSED -- nothing ran, nothing written")
    return 2


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--internal"] and len(argv) > 1 and argv[1] in INTERNAL:
        return INTERNAL[argv[1]](argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    # SL-1 (plan 1.0): the DIAG guard, imported
    from src.eval.teacher_diag import Refused, refuse_test_path, require_outside_repo
    try:
        if a.work_dir is not None:
            refuse_test_path(a.work_dir, "--work-dir")
            work = require_outside_repo(a.work_dir, "--work-dir")
            if work.exists():
                return refused(f"--work-dir {work} exists; an output folder is never reused")
        else:
            for var in ("TMPDIR", "TEMP", "TMP"):      # before tempfile probes a candidate by writing to it
                if os.environ.get(var):
                    refuse_test_path(os.environ[var], f"${var} (set it elsewhere)")
            base = tempfile.gettempdir()
            n = 0
            while True:
                n += 1
                work = Path(base) / f"kd_flip_smoke_{os.getpid()}_{n}"
                refuse_test_path(work, "the default work dir (set TMPDIR elsewhere)")
                work = require_outside_repo(work, "the default work dir")
                if not work.exists():
                    break
    except Refused as e:
        return refused(str(e))
    work.mkdir(parents=True)
    t0 = time.time()
    print("=" * 78)
    print(f"KD FLIP SMOKE (L-KD-WRAP C1) -- CPU, function level and {STEPS}-step stub-teacher trainer runs")
    print(f"work dir: {work}")
    print("=" * 78, flush=True)
    import torch
    torch.set_num_threads(a.threads)
    ft = importlib.import_module("src.distill.flip_targets")
    arm_json = write_arm_json(work / "arm_json" / "registry_values")
    data_root = work / "data"
    make_synthetic_dataset(data_root)
    runs = {}
    for label, arm, mutation in (("flip", arm_json, "none"), ("flip_repeat", arm_json, "none"), ("e3", None, "none"),
                                 *((f"mut_{m}", arm_json, m) for m in RUN_MUTATIONS)):
        runs[label] = run_flip_worker(work=work, data_root=data_root, label=label, threads=a.threads,
                                      arm_json=arm, mutation=mutation)
        r = runs[label]
        print(f"[run] {label:28} rc={r['returncode']} {r['seconds']:7.1f} s "
              f"{(r.get('summary') or {}).get('trainer_result')}", flush=True)
    base = {}
    for name, fn in CHECKS:
        ok, detail = evaluate(fn, ft, runs, work)
        check(name, ok, detail)
        base[name] = ok
    fns = dict(CHECKS)
    for mutation, killer, kind in MUTATIONS:
        if kind == "run":
            ok, detail = evaluate(fns[killer], ft, {**runs, "flip": runs[f"mut_{mutation}"]}, work)
        else:
            with mutated(ft, mutation):
                ok, detail = evaluate(fns[killer], ft, runs, work)
        check(f"mut_{mutation}", base.get(killer) is True and not ok,
              f"{killer} passes without it: {base.get(killer)}; under it: {'passes' if ok else 'fails'} ({detail})")
    seconds = round(time.time() - t0, 1)
    (work / "kd_flip_summary.json").write_text(json.dumps({
        "seconds": seconds, "runs": {k: {"returncode": r["returncode"], "seconds": r["seconds"], "log": r["log"]}
                                     for k, r in runs.items()},
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results]}, indent=1), encoding="utf-8")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:42}: {'PASS' if ok else 'FAIL'}  {detail[:600]}".rstrip())
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nsummary: {work / 'kd_flip_summary.json'} ({seconds} s)")
    print(f"RESULT: {'PASS' if results and passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if results and passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
