#!/usr/bin/env python3
"""d1 (lane 6, L-AM4 + L-AM1q): the QAT trainer of record, src/quant/qat.py, on synthetic data.

The real student runs through run_qat in mode "smoke" on 32 synthetic 64x64 TRAIN images (batch 16,
drop_last: 2 steps per epoch, T_max 30, BN statistics frozen after step 20 and every observer after
step 24) and 8 VAL images. Three trainings go through the trainer's own step, VAL bracket and
checkpoint code: a clean one at max_norm 0.1 (below the fixture's gradient norm, so every step clips),
and two with a real NaN injected into the loss, once at step 10 (observers on) and once at step 27
(observers off). The launch gates are exercised one refusal at a time, G1's name checks by half and by
place, with nothing named "test" created. No dataset, no checkpoint, no download; every file goes to a
temp directory outside the repository.

    python -B scripts/smoke_qat_runner.py
    python -B scripts/smoke_qat_runner.py --device cuda
    python -B scripts/smoke_qat_runner.py --sections gates,isolation,profile

Without --device every section runs on the CPU. --device cuda runs the same three trainings on the GPU (P38;
block A, before pilot run 1) and ends with RESULT: ERROR, exit 4, where no CUDA device exists. The profile
section checks scripts/qat_epoch_eval.py check-run-meta against run_meta rows the trainer wrote (P14).

Ends with one RESULT line; exit 0 only when every check passes.
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
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import safe_tmpdir  # noqa: E402  (imports no repository code)

TMP = safe_tmpdir("smoke_qat_runner_")                       # a temp path naming neither TEST nor VAL
os.environ["PLANTSEG_DATA_ROOT"] = str(TMP / "absent_data_root")   # before configs.data is imported

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.distill.export import CWD_PROJECTION_KEY  # noqa: E402
from src.models.student import build_student  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant.prepare import qat_freeze_steps  # noqa: E402
from src.quant.qconfig import QuantBackendUnavailable  # noqa: E402
from src.quant.stages import resolve_quant_stage  # noqa: E402

NC = 116
SECTIONS = ("d1", "nan", "gates", "isolation", "profile")
SKIP_ADD_NEVER_OBSERVED = [f"features.{i}.skip_add.activation_post_process" for i in (11, 13, 2, 4, 7)]
EXPECTED_CENSUS = {"FixedQParamsObserver": 9, "MovingAverageMinMaxObserver": 110,
                   "MovingAveragePerChannelMinMaxObserver": 65}
results: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- fixtures
def make_loaders(seed: int = 42):
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(32, 3, 64, 64, generator=g)
    y = torch.randint(0, NC, (32, 64, 64), generator=g)
    y[:, :4, :] = 255
    vx = torch.randn(8, 3, 64, 64, generator=g)
    vy = torch.randint(0, NC, (8, 64, 64), generator=g)
    train = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(vx, vy), batch_size=16, shuffle=False)
    return train, val


def student():
    torch.manual_seed(0)
    return build_student(pretrained=False)


SOURCE = {"sha256": "0" * 64, "path": None}


def grad_norm(params) -> float:
    gs = [p.grad.detach() for p in params if p.grad is not None]
    if not gs:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t, 2.0) for t in gs]), 2.0))


class Recorder(Q.QATHooks):
    """Observes the trainer's own loop: the norm at optimizer.step, buffer digests, flags inside VAL."""

    def __init__(self):
        self.at_call, self.applied_lr, self.bn, self.obs, self.train_mode_at_val = [], [], [], [], {}
        self.val_flags, self.ckpt_before_val, self.after_val_digest, self.ends = {}, {}, {}, {}
        self._seen, self._handles = set(), []

    def on_start(self, prepared, optimizer, scheduler, run_meta):
        self.optimizer, self.scheduler, self.run_meta = optimizer, scheduler, run_meta
        self.out_dir = Path(run_meta["out_dir"])
        self.bn0, self.obs0 = Q.bn_buffer_digest(prepared), Q.observer_buffer_digest(prepared)
        self.model_params = list(prepared.parameters())
        orig = optimizer.step

        def wrapped(*a, **k):
            self.at_call.append(grad_norm(p for g in optimizer.param_groups for p in g["params"]))
            self.applied_lr.append(optimizer.param_groups[0]["lr"])        # the lr this step applies
            return orig(*a, **k)
        wrapped._with_counter = True              # the scheduler's own counter still runs inside `orig`
        optimizer.step = wrapped

    def after_step(self, step, prepared, row):
        self.bn.append(Q.bn_buffer_digest(prepared))
        self.obs.append(Q.observer_buffer_digest(prepared))

    def before_val(self, epoch, prepared):
        self.train_mode_at_val[epoch] = prepared.training
        self.ckpt_before_val[epoch] = (self.out_dir / Q.EPOCH_DIR / f"e{epoch:02d}.pt").is_file()
        self._seen = set()
        self._handles = [m.register_forward_pre_hook(
            lambda mod, inp: self._seen.add(int(mod.observer_enabled.item())))
            for _, m in Q.fake_quant_modules(prepared)]

    def after_val(self, epoch, prepared):
        for h in self._handles:
            h.remove()
        self.val_flags[epoch] = sorted(self._seen)
        self.after_val_digest[epoch] = Q.state_digest(prepared)

    def after_epoch(self, epoch, prepared, checkpoint_path, row):
        self.ends[epoch] = row


def strict_rows(path: Path) -> tuple[list[dict], bool]:
    """Telemetry rows, and whether every line is strict JSON (no NaN / Infinity tokens)."""
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    rows, strict = [], True
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line, parse_constant=bad))
        except ValueError:
            strict = False
            rows.append(json.loads(line))
    return rows, strict


def run_case(name: str, *, clip: float, device: str, nan_at=()) -> dict:
    out = TMP / name
    rec = Recorder()
    case = {"out": out, "rec": rec, "error": None, "summary": None, "rows": [], "strict": False}
    try:
        case["summary"] = Q.run_qat(stage="e5", mode="smoke", model=student(), source_meta=SOURCE, out_dir=out,
                                    seed=42, clip_norm=clip, clip_source="smoke", device=device, num_workers=0,
                                    loaders=make_loaders(42), hooks=rec, inject_nan_at_steps=nan_at,
                                    log=lambda *a: None)
    except Exception as e:                                       # noqa: BLE001 -- every check below then fails
        case["error"] = f"{type(e).__name__}: {e}"
    tel = out / Q.TELEMETRY_NAME
    if tel.is_file():
        case["rows"], case["strict"] = strict_rows(tel)
    return case


def rows_of(case: dict, event: str) -> list[dict]:
    return [r for r in case["rows"] if r.get("event") == event]


def closed_form(t: int, T: int) -> float:
    return 1.5e-4 * (1.0 + math.cos(math.pi * t / T))


def changes(seq: list[str], first: str) -> list[bool]:
    prev, out = first, []
    for d in seq:
        out.append(d != prev)
        prev = d
    return out


# ---------------------------------------------------------------- d1: the clean training
def test_d1(device: str) -> dict:
    c = run_case("d1_clean", clip=0.1, device=device)
    rec, meta = c["rec"], (rows_of(c, "run_meta") or [{}])[0]
    train = rows_of(c, "train")
    check("d1_run_completed", c["error"] is None and c["summary"] and c["summary"]["complete"], c["error"] or "")
    check("d1_steps_per_epoch_2", meta.get("steps_per_epoch") == 2, str(meta.get("steps_per_epoch")))
    check("d1_total_steps_30", meta.get("total_steps") == 30 and len(train) == 30
          and (meta.get("scheduler") or {}).get("T_max") == 30, f"{len(train)} train rows")
    lrs = [r.get("lr") for r in train]
    check("d1_lr_first_3e-4", bool(lrs) and lrs[0] == 3e-4, repr(lrs[:1]))
    check("d1_lr_last_closed_form",
          len(lrs) == 30 and abs(lrs[29] - closed_form(29, 30)) <= 1e-12 * closed_form(29, 30),
          f"{lrs[29] if len(lrs) == 30 else None!r} vs {closed_form(29, 30)!r}")
    check("d1_lr_all_steps_match_closed_form",
          len(lrs) == 30 and all(abs(lr - closed_form(t, 30)) <= 1e-12 * closed_form(t, 30)
                                 for t, lr in enumerate(lrs)),
          "1.5e-4 x (1 + cos(pi t / 30)), rel tol 1e-12")
    applied = rec.applied_lr
    check("d1_lr_applied_at_each_optimizer_step",
          len(applied) == 30 and applied[0] == 3e-4 and applied == lrs
          and all(abs(lr - closed_form(t, 30)) <= 1e-12 * closed_form(t, 30) for t, lr in enumerate(applied)),
          "the lr optimizer.step applies at step t+1 is the closed form at t: scheduler.step() follows it")
    ends_ = rows_of(c, "epoch_end")
    check("d1_row_flags_match_the_freeze_epochs",
          len(train) == 30 and len(ends_) == 15
          and all(r.get("bn_frozen") is (r["epoch"] > 10) and r.get("observers_enabled") is (r["epoch"] <= 12)
                  for r in train + ends_),
          "train and epoch_end rows: bn_frozen from epoch 11, observers off from epoch 13")
    opt = getattr(rec, "optimizer", None)
    rec_opt = meta.get("optimizer") or {}
    check("d1_optimizer_of_record",
          opt is not None and len(opt.param_groups) == 1
          and {id(p) for p in opt.param_groups[0]["params"]} == {id(p) for p in rec.model_params}
          and opt.param_groups[0]["momentum"] == 0.9 and opt.param_groups[0]["weight_decay"] == 1e-4
          and opt.param_groups[0]["dampening"] == 0 and opt.param_groups[0]["nesterov"] is False
          and opt.param_groups[0]["initial_lr"] == 3e-4 and type(opt).__name__ == "SGD"
          and rec_opt.get("param_groups") == 1 and rec_opt.get("group_holds_every_model_param") is True
          and meta.get("lr") == 3e-4 and meta.get("momentum") == opt.param_groups[0]["momentum"]
          and meta.get("weight_decay") == opt.param_groups[0]["weight_decay"]
          and meta.get("nesterov") is False and meta.get("dampening") == 0.0
          and meta.get("scheduler") == {"class": "CosineAnnealingLR", "T_max": 30, "eta_min": 0.0,
                                        "step": "once after every optimizer.step"},
          "one group with every parameter; run_meta read from the optimizer object")
    pre = [r.get("grad_norm") for r in train]
    at = rec.at_call
    check("d1_clip_applied_at_the_step",
          len(at) == 30 and len(pre) == 30 and all(isinstance(p, float) and p > 0.1 for p in pre)
          and all(a <= 0.1 * (1 + 1e-5) for a in at) and all(a >= 0.1 * (1 - 1e-3) for a in at)
          and all(r.get("clipped") is True for r in train),
          f"pre-clip min {min(pre) if pre and all(isinstance(p, float) for p in pre) else None}, "
          f"at optimizer.step max {max(at) if at else None}")
    bn_ch = changes(rec.bn, getattr(rec, "bn0", ""))
    obs_ch = changes(rec.obs, getattr(rec, "obs0", ""))
    check("d1_bn_unfrozen_through_step_20", len(bn_ch) == 30 and all(bn_ch[:20]), f"{sum(bn_ch[:20])}/20 changed")
    check("d1_bn_frozen_from_step_21", len(bn_ch) == 30 and not any(bn_ch[20:]), f"{sum(bn_ch[20:])}/10 changed")
    check("d1_observers_on_through_step_24", len(obs_ch) == 30 and all(obs_ch[:24]), f"{sum(obs_ch[:24])}/24 changed")
    check("d1_observers_off_from_step_25", len(obs_ch) == 30 and not any(obs_ch[24:]), f"{sum(obs_ch[24:])}/6 changed")
    check("d1_freeze_helper_335_gives_3350_4020", qat_freeze_steps(335) == (3350, 4020), str(qat_freeze_steps(335)))
    check("d1_freeze_helper_3_gives_30_36", qat_freeze_steps(3) == (30, 36), str(qat_freeze_steps(3)))
    check("d1_freeze_steps_in_run_meta",
          meta.get("bn_freeze_after_step") == 20 and meta.get("obs_freeze_after_step") == 24
          and meta.get("bn_frozen_from_epoch") == 11 and meta.get("obs_frozen_from_epoch") == 13)

    ends = rows_of(c, "epoch_end")
    ck_dir = c["out"] / Q.EPOCH_DIR
    files_ok = (len(ends) == 15 and all((ck_dir / f"e{e:02d}.pt").is_file() for e in range(1, 16))
                and all(sha_file(ck_dir / f"e{r['epoch']:02d}.pt") == r["checkpoint_sha256"] for r in ends))
    check("d1_fifteen_epoch_checkpoints", files_ok and sorted(p.name for p in ck_dir.iterdir())
          == [f"e{e:02d}.pt" for e in range(1, 16)], f"{len(ends)} epoch_end rows")
    flags = {}
    for e in range(1, 16):
        p = ck_dir / f"e{e:02d}.pt"
        if p.is_file():
            ck = torch.load(p, map_location="cpu", weights_only=True)
            flags[e] = Q.flag_summary(ck["model_state_dict"])
    check("d1_checkpoint_observer_flags",
          len(flags) == 15 and all(flags[e]["observer_enabled"] == ([1] if e <= 12 else [0]) for e in flags)
          and all(flags[e]["fake_quant_enabled"] == [1] for e in flags),
          "observer_enabled 1 in e01-e12, 0 in e13-e15; fake_quant_enabled 1 throughout")
    check("d1_checkpoint_written_before_val",
          len(rec.ckpt_before_val) == 15 and all(rec.ckpt_before_val.values()), "eNN.pt exists when VAL starts")
    vals = rows_of(c, "val")
    check("val_leaves_state_unchanged",
          len(vals) == 15 and all(r.get("state_unchanged") is True and r.get("observers_disabled_during_val") is True
                                  for r in vals)
          and len(rec.val_flags) == 15 and all(v == [0] for v in rec.val_flags.values())
          and all(rec.after_val_digest.get(r["epoch"]) == r["state_sha256"] for r in ends),
          "every FakeQuantize ran with observer_enabled 0 in VAL; the state after VAL equals the checkpoint's")
    order_ok = True
    for e in range(1, 16):
        idx = [i for i, r in enumerate(c["rows"])
               if r.get("epoch") == e and r.get("event") in ("train", "val", "epoch_end")]
        kinds = [c["rows"][i]["event"] for i in idx]
        order_ok &= kinds == ["train", "train", "val", "epoch_end"]
    check("d1_epoch_order_train_val_epoch_end", order_ok and len(ends) == 15)
    last = c["rows"][-1] if c["rows"] else {}
    check("last_epoch_record_is_completion",
          last.get("event") == "epoch_end" and last.get("epoch") == 15 and last.get("run_complete") is True
          and sum(1 for r in ends if r.get("run_complete")) == 1, str({k: last.get(k) for k in ("event", "epoch")}))
    want_keys = {"bn_frozen_from_epoch", "obs_frozen_from_epoch", "bn_freeze_after_step", "obs_freeze_after_step",
                 "scheduler", "nesterov", "dampening", "observer_census", "fake_quant_modules",
                 "never_observed_modules", "observer_freeze_scope", "tf32", "persistent_workers", "gpu_name",
                 "clip_source", "git_head", "code_clean_at_head", "code_files_sha256", "wall_clock", "objective",
                 "class_weights_sha256", "ignore_index", "seed", "stage", "mode", "clip_norm", "epochs",
                 "steps_per_epoch", "total_steps", "batch_size", "drop_last", "engine", "qconfig", "image_digest",
                 "torch", "device", "host", "source_checkpoint_sha256", "source_checkpoint_bytes"}
    check("run_meta_fields_exact",
          want_keys <= set(meta) and meta.get("observer_census") == EXPECTED_CENSUS
          and meta.get("fake_quant_modules") == 184
          and meta.get("never_observed_modules") == sorted(SKIP_ADD_NEVER_OBSERVED)
          and meta.get("observer_freeze_scope") == "all FakeQuantize modules, weight and activation"
          and meta.get("ignore_index") == 255 and meta.get("engine") == "qnnpack"
          and meta.get("class_weights_sha256") == sha_file(REPO / "reports" / "e1_class_weights.json")
          and meta.get("batch_size") == 16 and meta.get("drop_last") is True and meta.get("clip_source") == "smoke"
          and isinstance(meta.get("code_files_sha256"), dict) and "src/quant/qat.py" in meta["code_files_sha256"]
          and set(meta.get("tf32") or {}) == set(Q.TF32_DEFAULTS)
          and set(meta.get("host") or {}) == {"hostname", "pod_id", "cpu_model"},
          f"missing {sorted(want_keys - set(meta))}; census {meta.get('observer_census')}; "
          f"never observed {meta.get('never_observed_modules')}")
    check("every_row_has_wall_clock",
          bool(c["rows"]) and all(isinstance(r.get("wall_clock"), float) and r["wall_clock"] > 1.7e9
                                  for r in c["rows"]),
          f"{len(c['rows'])} rows")
    check("telemetry_strict_json", c["strict"] and bool(c["rows"]))
    check("no_unfused_bn", meta.get("unfused_batchnorm") == 0
          and meta.get("fused_convbn") == {"ConvBn2d": 34, "ConvBnReLU2d": 12}, str(meta.get("fused_convbn")))
    pat = re.compile(r"patience|EarlyStopper|early_stop")
    files = ["src/quant/runner.py", "src/quant/qat.py", "src/quant/qat_artifacts.py", "src/quant/qat_select.py",
             "configs/quant.py", "scripts/run_e5.py", "scripts/run_e6.py", "scripts/qat_epoch_eval.py",
             "scripts/select_qat_epoch.py", "scripts/select_clip.py"]
    hits = [f"{f}:{i}" for f in files if (REPO / f).is_file()
            for i, line in enumerate((REPO / f).read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
    check("d1_no_patience_path_grep", not hits, str(hits[:5]))
    if c["summary"]:
        check("d1_result_line",
              Q.result_line(c["summary"]) == "RESULT: QAT COMPLETE (E5, seed 42, clip 0.1, 15/15 epochs)",
              Q.result_line(c["summary"]))
    else:
        check("d1_result_line", False, c["error"] or "")
    return c


# ---------------------------------------------------------------- non-finite states (AM-19 item 3(a))
def test_nonfinite(device: str, clean: dict | None) -> None:
    clean_train = rows_of(clean, "train") if clean else []
    for k in (10, 27):
        c = run_case(f"d1_nan_step{k}", clip=1.0, device=device, nan_at=(k,))
        ends, vals = rows_of(c, "epoch_end"), rows_of(c, "val")
        steps = sorted([r for r in c["rows"] if r.get("event") in ("train", "step_error")], key=lambda r: r["step"])
        at_k = next((r for r in steps if r["step"] == k), {})
        ck_dir = c["out"] / Q.EPOCH_DIR
        tag = "observers on" if k <= 24 else "observers off"
        check(f"nonfinite_loss_recorded_run_reaches_epoch_15_step{k}",
              c["error"] is None and c["summary"] and c["summary"]["complete"]
              and c["summary"]["nonfinite_since_step"] == k and len(ends) == 15 and len(vals) == 15
              and ends[-1].get("run_complete") is True and not rows_of(c, "run_abort")
              and all((ck_dir / f"e{e:02d}.pt").is_file() for e in range(1, 16))
              and at_k.get("event") == "train" and at_k.get("loss") is None
              and (at_k.get("nonfinite") or {}).get("loss") == "nan",
              f"{tag}: {c['error'] or ''} since {c['summary'] and c['summary']['nonfinite_since_step']}")
        first_bad = (k - 1) // 2 + 1
        check(f"d1_nonfinite_rows_marked_step{k}",
              all(r.get("state_nonfinite") is True for r in steps if r["step"] >= k)
              and all(r.get("state_nonfinite") is False for r in steps if r["step"] < k)
              and all(r.get("state_finite") is False and r.get("nonfinite_since_step") == k
                      and r.get("state_nonfinite") is True for r in ends if r["epoch"] >= first_bad)
              and all(r.get("state_finite") is True and r.get("state_nonfinite") is False
                      for r in ends if r["epoch"] < first_bad),
              f"epoch_end state_finite false and state_nonfinite true from epoch {first_bad}")
        check(f"d1_schedule_and_batches_continue_step{k}",
              len(steps) == 30 and len(clean_train) == 30
              and [r["lr"] for r in steps] == [r["lr"] for r in clean_train]
              and [r["batch_sha256"] for r in steps] == [r["batch_sha256"] for r in clean_train],
              "same 30 lr values and batch fingerprints as the clean run")
        if c["summary"]:
            check(f"d1_result_line_nonfinite_step{k}",
                  Q.result_line(c["summary"]).startswith(
                      f"RESULT: QAT COMPLETE, STATE NON-FINITE first found at step {k} ("),
                  Q.result_line(c["summary"]))
        else:
            check(f"d1_result_line_nonfinite_step{k}", False, c["error"] or "")
        if k <= 24:
            errs = rows_of(c, "step_error")
            check("d1_step_errors_recorded_in_nonfinite_state",
                  bool(errs) and all(r["step"] > k and r.get("state_nonfinite") is True for r in errs),
                  f"{len(errs)} step_error rows (a raising forward once the state is non-finite)")
        shutil.rmtree(c["out"], ignore_errors=True)
    # a VAL pass that raises once the state is non-finite: recorded, the run goes on, in train mode (P12)
    import src.training.train_e1 as te1
    real_validate, calls = te1.validate, []

    def raising_once(model, *a, **k):
        calls.append(1)
        if len(calls) == 5:                                      # epoch 5's VAL: the NaN came at step 10
            model.eval()                                         # as validate() does before its loop
            raise RuntimeError("a VAL pass that raises in a non-finite state")
        return real_validate(model, *a, **k)
    with patched(te1, "validate", raising_once):
        c = run_case("d1_nan_step10_val_raises", clip=1.0, device=device, nan_at=(10,))
    vals, mode = rows_of(c, "val"), c["rec"].train_mode_at_val
    v5 = next((r for r in vals if r.get("epoch") == 5), {})
    check("d1_val_error_recorded_and_train_mode_restored",
          c["error"] is None and c["summary"] and c["summary"]["complete"] and len(vals) == 15
          and v5.get("all_class_miou") is None and "RuntimeError" in str(v5.get("error"))
          and v5.get("state_nonfinite") is True and not rows_of(c, "run_abort")
          and sorted(mode) == list(range(1, 16)) and all(mode.values()),
          f"{c['error'] or ''} epoch-5 VAL {v5.get('error')!r}; train mode at each VAL start "
          f"{[e for e, m in sorted(mode.items()) if not m]} off")
    shutil.rmtree(c["out"], ignore_errors=True)

    # AM-21 items 1(b) and 1(e): the predicate is read at every raise, and a raise ends the run only when the state
    # is then finite -- also after an earlier non-finite flag, which a test-only repair makes finite again
    class Raiser(Recorder):
        """A training forward that raises at the `raise_at`-th training step, or at the first one after
        `repair_epoch` when `arm`; at the end of `repair_epoch`, the state saved after epoch 1 and a fresh
        optimizer state (a repair the trainer itself never makes, AM-21 item 1(d))."""

        def __init__(self, *, raise_at: int | None = None, repair_epoch: int | None = None, arm: bool = False,
                     poison: bool = False):
            super().__init__()
            self.raise_at, self.repair_epoch, self.arm, self.poison = raise_at, repair_epoch, arm, poison
            self.forwards, self.saved, self.armed = 0, None, False

        def on_start(self, prepared, optimizer, scheduler, run_meta):
            super().on_start(prepared, optimizer, scheduler, run_meta)

            def pre(mod, inp):
                if not mod.training:
                    return
                self.forwards += 1
                if self.forwards == self.raise_at or self.armed:
                    self.armed = False
                    if self.poison:                              # the raise meets a state just made non-finite
                        with torch.no_grad():
                            next(mod.parameters()).fill_(float("nan"))
                    raise RuntimeError("an injected raise in a training forward")
            prepared.register_forward_pre_hook(pre)

        def after_epoch(self, epoch, prepared, checkpoint_path, row):
            super().after_epoch(epoch, prepared, checkpoint_path, row)
            if epoch == 1:                                       # with its _metadata: the observers' load needs it
                state = prepared.state_dict()
                self.saved = type(state)((k, v.detach().clone()) for k, v in state.items())
                self.saved._metadata = state._metadata
            if epoch == self.repair_epoch:
                prepared.load_state_dict(self.saved)
                self.optimizer.state.clear()
                self.armed = self.arm

    def raise_case(name: str, hooks: Recorder, nan_at=(), val_raise_call: int | None = None) -> dict:
        val_calls = []

        def validate_raising(model, *a, **k):
            val_calls.append(1)
            if len(val_calls) == val_raise_call:
                model.eval()
                raise RuntimeError("an injected raise in a VAL pass")
            return real_validate(model, *a, **k)
        out = TMP / name
        c = {"out": out, "error": None, "rows": [], "summary": None}
        with patched(te1, "validate", validate_raising):
            try:
                c["summary"] = Q.run_qat(stage="e5", mode="smoke", model=student(), source_meta=SOURCE, out_dir=out,
                                         seed=42, clip_norm=1.0, clip_source="smoke", device=device, num_workers=0,
                                         loaders=make_loaders(42), hooks=hooks, inject_nan_at_steps=nan_at,
                                         log=lambda *a: None)
            except Exception as e:                               # noqa: BLE001 -- the abort under test
                c["error"] = f"{type(e).__name__}: {e}"
        if (out / Q.TELEMETRY_NAME).is_file():
            c["rows"], _ = strict_rows(out / Q.TELEMETRY_NAME)
        shutil.rmtree(out, ignore_errors=True)
        return c

    def aborted_at(c: dict, *, step: int, epoch: int, flagged_at: int | None) -> bool:
        last = c["rows"][-1] if c["rows"] else {}
        marks = [r.get("nonfinite_since_step") for r in c["rows"] if r.get("event") == "state_nonfinite"]
        return (str(c["error"]).startswith("RuntimeError: an injected raise") and last.get("event") == "run_abort"
                and last.get("exception_type") == "RuntimeError" and last.get("step") == step
                and last.get("epoch") == epoch and marks == ([flagged_at] if flagged_at else []))

    def last_row(c: dict) -> str:
        last = c["rows"][-1] if c["rows"] else {}
        return f"{c['error']}; last row {({k: last.get(k) for k in ('event', 'epoch', 'step', 'state_nonfinite')})}"
    c = raise_case("d1_step_raises_finite", Raiser(raise_at=3))
    check("d1_step_raise_while_finite_aborts", aborted_at(c, step=3, epoch=2, flagged_at=None), last_row(c))
    c = raise_case("d1_val_raises_finite", Raiser(), val_raise_call=1)
    check("d1_val_raise_while_finite_aborts", aborted_at(c, step=2, epoch=1, flagged_at=None)
          and not rows_of(c, "val"), last_row(c))
    c = raise_case("d1_step_raises_repaired", Raiser(repair_epoch=6, arm=True), nan_at=(10,))
    check("d1_step_raise_after_repair_aborts", aborted_at(c, step=13, epoch=7, flagged_at=10), last_row(c))
    c = raise_case("d1_val_raises_repaired", Raiser(repair_epoch=6), nan_at=(10,), val_raise_call=7)
    check("d1_val_raise_after_repair_aborts", aborted_at(c, step=14, epoch=7, flagged_at=10)
          and [r["epoch"] for r in rows_of(c, "epoch_end")] == list(range(1, 7)), last_row(c))

    def completed_nonfinite(c: dict, *, first_found: int) -> tuple[bool, str]:
        marks = [r.get("nonfinite_since_step") for r in c["rows"] if r.get("event") == "state_nonfinite"]
        ends = rows_of(c, "epoch_end")
        line = Q.result_line(c["summary"]) if c["summary"] else str(c["error"])
        ok = (c["error"] is None and bool(c["summary"]) and c["summary"]["complete"] and marks == [first_found]
              and not rows_of(c, "run_abort") and len(ends) == 15
              and line.startswith(f"RESULT: QAT COMPLETE, STATE NON-FINITE first found at step {first_found} ("))
        return ok, f"{line}; marks {marks}"
    # a step that raises when the state has just become non-finite, before any flag: recorded, the step marked, the
    # run goes on (AM-21 item 1(c)); the parameter written NaN inside the raising forward is test-only
    c = raise_case("d1_step_raises_poisoned", Raiser(raise_at=3, poison=True))
    ok, detail = completed_nonfinite(c, first_found=3)
    at3 = next((r for r in c["rows"] if r.get("step") == 3 and r.get("event") in ("train", "step_error")), {})
    check("d1_step_raise_in_unflagged_nonfinite_state_recorded",
          ok and at3.get("event") == "step_error" and at3.get("state_nonfinite") is True, detail)

    class NeverObservedPoisoner(Recorder):
        """After step 9, a never-observed fake-quant's observer minimum becomes NaN (test-only): no forward reads it,
        so the loss and the gradient stay finite and nothing raises; only the epoch-end evaluation of the
        checkpoint's state (AM-21 item 1(b)) can find it."""

        def after_step(self, step, prepared, row):
            super().after_step(step, prepared, row)
            if step == 9:
                fq = dict(prepared.named_modules())[SKIP_ADD_NEVER_OBSERVED[0]]
                fq.activation_post_process.min_val.fill_(float("nan"))
    c = raise_case("d1_epoch_end_finds_alone", NeverObservedPoisoner())
    ok, detail = completed_nonfinite(c, first_found=10)
    ends = rows_of(c, "epoch_end")
    check("d1_epoch_end_alone_finds_a_nonfinite_state",
          ok and [r.get("state_finite") for r in ends] == [True] * 4 + [False] * 11
          and not rows_of(c, "step_error") and not any(r.get("nonfinite") for r in rows_of(c, "train")), detail)


# ---------------------------------------------------------------- launch gates, one refusal at a time
def code_of(fn, *a, **kw) -> str | None:
    """fn's refusal code; None when it returns; "raised:<type>" for any other exception, so a refusal that is
    missing fails its own named check instead of ending the section."""
    try:
        fn(*a, **kw)
    except Q.QATRefused as e:
        return e.code
    except Exception as e:                                      # noqa: BLE001 -- reported in the check's detail
        return f"raised:{type(e).__name__}"
    return None


def refusal_of(fn, *a, **kw) -> tuple[str | None, str]:
    """fn's refusal code and message; (None, "") when it returns; ("raised:<type>", message) for any other
    exception."""
    try:
        fn(*a, **kw)
    except Q.QATRefused as e:
        return e.code, str(e)
    except Exception as e:                                      # noqa: BLE001 -- reported in the check's detail
        return f"raised:{type(e).__name__}", str(e)
    return None, ""


@contextlib.contextmanager
def listing_recorder(path: Path):
    """os.listdir, os.scandir and Path.iterdir record every call on `path` (or below it) while the block runs,
    and are restored after it (G1: a path whose name holds "test" is refused before it is listed)."""
    root = os.path.abspath(path)
    seen: list[str] = []
    real = (os.listdir, os.scandir, Path.iterdir)

    def hit(p) -> None:
        try:
            a = os.path.abspath(os.fsdecode(p))
        except TypeError:                                       # a file descriptor
            return
        if a == root or a.startswith(root + os.sep):
            seen.append(a)

    def listdir(p=".", *a, **kw):
        hit(p)
        return real[0](p, *a, **kw)

    def scandir(p=".", *a, **kw):
        hit(p)
        return real[1](p, *a, **kw)

    def iterdir(self):
        hit(self)
        return real[2](self)
    os.listdir, os.scandir, Path.iterdir = listdir, scandir, iterdir
    try:
        yield seen
    finally:
        os.listdir, os.scandir, Path.iterdir = real


def write_e1_parent(d: Path, *, seed: int = 42, mode: str = "real", val_at_max: bool = True,
                    best: dict | None = None, extra_meta_rows: int = 0) -> str:
    d.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().clone() for k, v in student().state_dict().items()}
    ck = d / "e1_student_best_iter80000.pt"
    torch.save({"iter": 80000, "num_classes": NC, "model_state_dict": state}, ck)
    (d / "best.json").write_text(json.dumps(best if best is not None else
                                            {"best_ckpt": "/workspace/e1_ckpts/e1_student_best_iter80000.pt",
                                             "best_val_miou_all_class": 0.3}), encoding="utf-8")
    rows = [{"event": "run_meta", "mode": mode, "seed": seed, "max_iters": 80000}] * (1 + extra_meta_rows)
    rows += [{"event": "val", "iter": 80000 if val_at_max else 76000, "all_class_miou": 0.3}]
    (d / "e1_telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return sha_file(ck)


def write_e3_parent(d: Path, *, seed: int = 42, lam: float = 1.0, alpha: float = 50.0, run_end: bool = True,
                    abort: bool = False, projection_in_student: bool = False) -> str:
    d.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().clone() for k, v in student().state_dict().items()}
    if projection_in_student:
        state["cwd_projection.weight"] = torch.randn(320, 160, 1, 1)
    ck = d / "e3_student_best_iter80000.pt"
    torch.save({"stage": "E3", "num_classes": NC, "model_state_dict": state,
                CWD_PROJECTION_KEY: {"weight": torch.randn(320, 160, 1, 1)}}, ck)
    (d / "best.json").write_text(json.dumps({"best_ckpt": str(ck), "best_val_miou_all_class": 0.31}),
                                 encoding="utf-8")
    (d / "e3_run_meta.jsonl").write_text(json.dumps({"event": "run_meta", "stage": "E3", "mode": "real",
                                                     "seed": seed, "lambda_logit": lam, "alpha_cwd": alpha}) + "\n",
                                         encoding="utf-8")
    tel = []
    if abort:
        tel.append({"event": "run_abort", "iter": 100})
    if run_end:
        tel.append({"event": "run_end", "iter": 80000, "checks_passed": True})
    (d / "e3_telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tel), encoding="utf-8")
    return sha_file(ck)


def git(repo: Path, *args) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
           "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def without(argv: list[str], *flags: str) -> list[str]:
    """argv with each named flag and its value removed."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] in flags:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def test_gates() -> None:
    e5, e6 = resolve_quant_stage("e5"), resolve_quant_stage("e6")
    # ---- the parent run directory (AM-19 item 4(a))
    good = TMP / "e1_s42"
    sha = write_e1_parent(good)
    p = Q.resolve_parent(e5, good, sha, 42)
    check("parent_e1_accepted", p["checkpoint_sha256"] == sha and p["run_id"] == "e1_s42"
          and p["best_json_sha256"] == sha_file(good / "best.json")
          and p["telemetry_sha256"] == sha_file(good / "e1_telemetry.jsonl"))
    cases = [
        ("refuses_parent_dir_missing", (e5, TMP / "absent_parent", sha, 42), "parent_dir_missing"),
        ("refuses_parent_dir_test_path", (e5, TMP / "latest_parent", sha, 42), "parent_dir_test_path"),
        ("refuses_expect_source_sha256_format", (e5, good, "abc", 42), "expect_source_sha256_format"),
        ("refuses_parent_sha256_mismatch", (e5, good, "f" * 64, 42), "parent_sha256_mismatch"),
        ("refuses_parent_seed_mismatch", (e5, good, sha, 43), "parent_seed_mismatch"),
    ]
    d = TMP / "e1_nobest"
    write_e1_parent(d)
    (d / "best.json").unlink()
    cases.append(("refuses_parent_best_json_missing", (e5, d, sha_file(d / "e1_student_best_iter80000.pt"), 42),
                  "parent_best_json_missing"))
    d = TMP / "e1_badbest"
    s2 = write_e1_parent(d, best={"best_ckpt": "x.pt", "best_val_miou_all_class": 0.3, "extra": 1})
    cases.append(("refuses_parent_best_json_format", (e5, d, s2, 42), "parent_best_json_format"))
    d = TMP / "e1_nockpt"
    s2 = write_e1_parent(d, best={"best_ckpt": "/workspace/e1_ckpts/other.pt", "best_val_miou_all_class": 0.3})
    cases.append(("refuses_parent_checkpoint_missing", (e5, d, s2, 42), "parent_checkpoint_missing"))
    d = TMP / "e1_notel"
    s2 = write_e1_parent(d)
    (d / "e1_telemetry.jsonl").unlink()
    cases.append(("refuses_parent_records_missing", (e5, d, s2, 42), "parent_records_missing"))
    d = TMP / "e1_tworows"
    s2 = write_e1_parent(d, extra_meta_rows=1)
    cases.append(("refuses_parent_run_meta_rows", (e5, d, s2, 42), "parent_run_meta_rows"))
    d = TMP / "e1_dry"
    s2 = write_e1_parent(d, mode="dry")
    cases.append(("refuses_parent_mode_not_real", (e5, d, s2, 42), "parent_mode_not_real"))
    d = TMP / "e1_partial"
    s2 = write_e1_parent(d, val_at_max=False)
    cases.append(("refuses_parent_incomplete_e1", (e5, d, s2, 42), "parent_incomplete"))
    d = TMP / "e1_nomaxiters"
    s2 = write_e1_parent(d)
    (d / "e1_telemetry.jsonl").write_text(json.dumps({"event": "run_meta", "mode": "real", "seed": 42}) + "\n",
                                          encoding="utf-8")
    cases.append(("refuses_parent_run_meta_key", (e5, d, s2, 42), "parent_run_meta_key"))
    e3 = TMP / "e3_s42_a50"
    s3 = write_e3_parent(e3)
    cases.append(("refuses_e1_layout_for_e6", (e6, good, sha, 42), "parent_records_missing"))
    d = TMP / "e3_aborted"
    s4 = write_e3_parent(d, abort=True)
    cases.append(("refuses_parent_incomplete_e3_abort", (e6, d, s4, 42), "parent_incomplete"))
    d = TMP / "e3_noend"
    s4 = write_e3_parent(d, run_end=False)
    cases.append(("refuses_parent_incomplete_e3_no_run_end", (e6, d, s4, 42), "parent_incomplete"))
    for name, args, want in cases:
        got = code_of(Q.resolve_parent, *args)
        check(name, got == want, f"[{got}]")
    p3 = Q.resolve_parent(e6, e3, s3, 42)
    check("parent_e3_accepted", p3["stage"] == "E3" and p3["lambda_logit"] == 1.0 and p3["run_id"] == "e3_s42_a50")
    d = TMP / "e3_wrongstage"
    s5 = write_e3_parent(d)
    (d / "e3_run_meta.jsonl").write_text(json.dumps({"event": "run_meta", "stage": "E2", "mode": "real",
                                                     "seed": 42}) + "\n", encoding="utf-8")
    check("refuses_parent_stage_mismatch", code_of(Q.resolve_parent, e6, d, s5, 42) == "parent_stage_mismatch")

    # ---- the clip binding (U4)
    sel = TMP / "clip_selection.json"
    sel.write_text(json.dumps({"format": "qat_clip_selection/1", "winner": {"clip_norm": 5.0}}), encoding="utf-8")
    ssha = sha_file(sel)
    cb = Q.clip_binding
    check("clip_pilot_binding_accepted", cb("E5", 42, 1.0, u4_pilot=True, clip_selection=None,
                                            clip_selection_sha256=None)["clip_source"] == "u4_pilot")
    check("clip_selection_binding_accepted", cb("E6", 43, 5.0, u4_pilot=False, clip_selection=str(sel),
                                                clip_selection_sha256=ssha)["clip_selection"]["winner"] == 5.0)
    bad = TMP / "clip_selection_bad.json"
    bad.write_text(json.dumps({"format": "other/1", "winner": {"clip_norm": 5.0}}), encoding="utf-8")
    for name, args, kw, want in [
        ("refuses_clip_absent", ("E5", 42, None),
         dict(u4_pilot=True, clip_selection=None, clip_selection_sha256=None), "grad_clip_norm_invalid"),
        ("refuses_clip_not_candidate", ("E5", 42, 2.0),
         dict(u4_pilot=True, clip_selection=None, clip_selection_sha256=None), "grad_clip_norm_not_candidate"),
        ("refuses_u4_pilot_outside_e5_s42", ("E5", 43, 1.0),
         dict(u4_pilot=True, clip_selection=None, clip_selection_sha256=None), "u4_pilot_not_e5_s42"),
        ("refuses_u4_pilot_with_selection", ("E5", 42, 1.0),
         dict(u4_pilot=True, clip_selection=str(sel), clip_selection_sha256=ssha), "u4_pilot_with_clip_selection"),
        ("refuses_e5_s42_without_u4_pilot", ("E5", 42, 1.0),
         dict(u4_pilot=False, clip_selection=str(sel), clip_selection_sha256=ssha), "u4_pilot_required"),
        ("refuses_missing_clip_selection", ("E5", 44, 5.0),
         dict(u4_pilot=False, clip_selection=None, clip_selection_sha256=None), "clip_selection_required"),
        ("refuses_clip_selection_sha_format", ("E5", 44, 5.0),
         dict(u4_pilot=False, clip_selection=str(sel), clip_selection_sha256="abc"), "clip_selection_sha256_format"),
        ("refuses_clip_selection_file_missing", ("E5", 44, 5.0),
         dict(u4_pilot=False, clip_selection=str(TMP / "no.json"), clip_selection_sha256=ssha),
         "clip_selection_missing"),
        ("refuses_clip_selection_sha_mismatch", ("E5", 44, 5.0),
         dict(u4_pilot=False, clip_selection=str(sel), clip_selection_sha256="f" * 64),
         "clip_selection_sha256_mismatch"),
        ("refuses_clip_selection_format", ("E5", 44, 5.0),
         dict(u4_pilot=False, clip_selection=str(bad), clip_selection_sha256=sha_file(bad)), "clip_selection_format"),
        ("refuses_clip_not_the_winner", ("E5", 44, 1.0),
         dict(u4_pilot=False, clip_selection=str(sel), clip_selection_sha256=ssha), "clip_selection_winner_mismatch"),
    ]:
        got = code_of(cb, *args, **kw)
        check(name, got == want, f"[{got}]")

    # ---- the λ and α selections: tracked and unchanged at HEAD (a temp git repository)
    repo = TMP / "selrepo"
    (repo / "sel").mkdir(parents=True)
    lam_doc = {"format": "lambda_selection/1", "winner": {"lambda": 1.0, "run_id": "e3_s42_l1"}}
    alp_doc = {"format": "alpha_selection/1", "winner": {"alpha": 50.0, "run_id": "e3_s42_a50"}}
    (repo / "sel" / "lambda.json").write_text(json.dumps(lam_doc), encoding="utf-8")
    (repo / "sel" / "alpha.json").write_text(json.dumps(alp_doc), encoding="utf-8")
    (repo / "sel" / "wrongfmt.json").write_text(json.dumps({"format": "x/1", "winner": {}}), encoding="utf-8")
    (repo / "sel" / "changed.json").write_text(json.dumps(lam_doc), encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "sel/lambda.json", "sel/alpha.json", "sel/wrongfmt.json", "sel/changed.json")
    git(repo, "commit", "-q", "-m", "selections")
    (repo / "sel" / "changed.json").write_text(json.dumps({**lam_doc, "note": "edited"}), encoding="utf-8")
    (repo / "sel" / "untracked.json").write_text(json.dumps(lam_doc), encoding="utf-8")
    rts = Q.read_tracked_selection
    lsha, asha = sha_file(repo / "sel" / "lambda.json"), sha_file(repo / "sel" / "alpha.json")
    lam = rts("sel/lambda.json", lsha, "lambda_selection/1", "lambda", repo)
    alpha = rts("sel/alpha.json", asha, "alpha_selection/1", "alpha", repo)
    check("selections_accepted_when_tracked_and_unchanged", lam["winner"]["lambda"] == 1.0
          and alpha["winner"]["run_id"] == "e3_s42_a50")
    for name, args, want in [
        ("refuses_lambda_selection_missing", (None, lsha, "lambda_selection/1", "lambda", repo),
         "lambda_selection_missing"),
        ("refuses_alpha_selection_missing", (None, asha, "alpha_selection/1", "alpha", repo),
         "alpha_selection_missing"),
        ("refuses_selection_sha_format", ("sel/lambda.json", "abc", "lambda_selection/1", "lambda", repo),
         "selection_sha256_format"),
        ("refuses_selection_absolute_path", (str(repo / "sel" / "lambda.json"), lsha, "lambda_selection/1", "lambda",
                                             repo), "selection_path_not_repo_relative"),
        ("refuses_selection_test_path", ("sel/latest.json", lsha, "lambda_selection/1", "lambda", repo),
         "selection_test_path"),
        ("refuses_selection_missing_file", ("sel/none.json", lsha, "lambda_selection/1", "lambda", repo),
         "selection_missing_file"),
        ("refuses_selection_untracked", ("sel/untracked.json", lsha, "lambda_selection/1", "lambda", repo),
         "selection_not_tracked"),
        ("refuses_selection_changed_since_head",
         ("sel/changed.json", sha_file(repo / "sel" / "changed.json"), "lambda_selection/1", "lambda", repo),
         "selection_changed_since_head"),
        ("refuses_selection_sha_mismatch", ("sel/lambda.json", "f" * 64, "lambda_selection/1", "lambda", repo),
         "selection_sha256_mismatch"),
        ("refuses_selection_format", ("sel/wrongfmt.json", sha_file(repo / "sel" / "wrongfmt.json"),
                                      "lambda_selection/1", "lambda", repo), "selection_format"),
        ("refuses_alpha_file_as_lambda", ("sel/alpha.json", asha, "lambda_selection/1", "lambda", repo),
         "selection_format"),
    ]:
        got = code_of(rts, *args)
        check(name, got == want, f"[{got}]")
    # ---- E6's parent is the run the selections name
    check("e6_binding_accepted", Q.e6_parent_binding(p3, 42, lam, alpha)["alpha_selection"]["sha256"] == asha)
    for name, parent, seed, want in [
        ("refuses_e6_parent_not_e3", {**p3, "stage": "E1"}, 42, "parent_stage_mismatch"),
        ("refuses_e6_parent_seed", p3, 43, "parent_seed_mismatch"),
        ("refuses_e6_parent_lambda", {**p3, "lambda_logit": 0.5}, 42, "parent_lambda_mismatch"),
        ("refuses_e6_parent_alpha", {**p3, "alpha_cwd": 25.0}, 42, "parent_alpha_mismatch"),
        ("refuses_e6_s42_parent_not_alpha_winner", {**p3, "run_id": "e3_s42_a25"}, 42, "parent_run_id_mismatch"),
    ]:
        got = code_of(Q.e6_parent_binding, parent, seed, lam, alpha)
        check(name, got == want, f"[{got}]")
    p43 = {**p3, "seed": 43, "run_id": "e3_s43"}
    check("e6_s43_parent_need_not_be_the_alpha_run", code_of(Q.e6_parent_binding, p43, 43, lam, alpha) is None)

    # ---- real_run_gates in order (CUDA availability patched where a later gate is the subject)
    head = "0" * 40
    out_ok = str(TMP / "qat_out_new")

    def gate(argv, stage=e5, patches=None):
        args = Q.build_parser(stage).parse_args(argv)
        with contextlib.ExitStack() as stack:
            for (obj, name), value in (patches or {}).items():
                stack.enter_context(patched(obj, name, value))
            return code_of(Q.real_run_gates, args, stage, repo_root=repo)

    base = ["--real-run", "--confirm-real-run", "--expect-head", head, "--seed", "42", "--num-workers", "12",
            "--grad-clip-norm", "1.0", "--u4-pilot", "--source-run-dir", str(good), "--expect-source-sha256", sha,
            "--out-dir", out_ok]
    nonempty = TMP / "qat_out_used"
    nonempty.mkdir()
    (nonempty / "x.txt").write_text("x", encoding="utf-8")
    cuda = {(torch.cuda, "is_available"): (lambda: True), (torch.cuda, "is_initialized"): (lambda: False)}
    import src.data.isolation as iso
    fake_iso = (iso, "assert_trainval_only_root")
    ok_iso = lambda root, expected=None: {"counts": {"train": 5367, "val": 846}}       # noqa: E731
    ident = (Q, "code_identity")
    clean_at = lambda: {"git_head": head, "code_clean_at_head": True, "code_files_sha256": {}}   # noqa: E731
    dirty_at = lambda: {"git_head": head, "code_clean_at_head": False, "code_files_sha256": {}}  # noqa: E731
    other_at = lambda: {"git_head": "1" * 40, "code_clean_at_head": True, "code_files_sha256": {}}  # noqa: E731
    for name, argv, patches, want in [
        ("gate_refuses_without_real_run", base[1:], {}, "real_run_flag"),
        ("gate_refuses_without_confirm", [base[0]] + base[2:], {}, "confirm_real_run_flag"),
        ("gate_refuses_head_format", [a if a != head else "abc" for a in base], {}, "expect_head_format"),
        ("gate_refuses_seed", [a if a != "42" else "41" for a in base], {}, "seed"),
        ("gate_refuses_num_workers_0", [a if a != "12" else "0" for a in base], {}, "num_workers"),
        ("gate_refuses_out_dir_in_repo", [a if a != out_ok else str(REPO / "qat_out") for a in base], {}, "out_dir"),
        ("gate_refuses_out_dir_not_empty", [a if a != out_ok else str(nonempty) for a in base], {}, "out_dir"),
        ("gate_refuses_out_dir_test_path", [a if a != out_ok else str(TMP / "latest_out") for a in base], {},
         "out_dir"),
        ("gate_refuses_cpu", base + ["--device", "cpu"], cuda, "cuda_required"),
        ("gate_refuses_cuda_initialised", base, {**cuda, (torch.cuda, "is_initialized"): (lambda: True)},
         "cuda_initialized_before_seed"),
        ("gate_refuses_tf32_not_default", base, {**cuda, (Q, "tf32_state"): (lambda: {**Q.TF32_DEFAULTS,
                                                                                         "NVIDIA_TF32_OVERRIDE": "1"})},
         "tf32_not_default"),
        ("gate_refuses_data_root_not_trainval_only", base, cuda, "data_root_not_trainval_only"),
        ("gate_refuses_head_mismatch", base, {**cuda, fake_iso: ok_iso, ident: other_at}, "expect_head_mismatch"),
        ("gate_refuses_code_not_clean", base, {**cuda, fake_iso: ok_iso, ident: dirty_at}, "code_not_clean"),
        ("gate_refuses_parent", [a if a != sha else "f" * 64 for a in base], {**cuda, fake_iso: ok_iso,
                                                                               ident: clean_at},
         "parent_sha256_mismatch"),
        ("gate_refuses_selection_flags_for_e5", base + ["--alpha-selection", "sel/alpha.json"], {},
         "selection_not_applicable"),
    ]:
        got = gate(argv, patches=patches)
        check(name, got == want, f"[{got}]")
    with patched(Q, "config_pins_error", lambda: "configs/quant.py qat.epochs = 14, AM-4a: 15"):
        check("gate_refuses_config_pins", gate(base) == "config_pins")
    check("gates_pass_with_valid_inputs", gate(base, patches={**cuda, fake_iso: ok_iso, ident: clean_at}) is None)
    e6_base = ["--real-run", "--confirm-real-run", "--expect-head", head, "--seed", "42", "--num-workers", "12",
               "--grad-clip-norm", "5.0", "--clip-selection", str(sel), "--clip-selection-sha256", ssha,
               "--lambda-selection", "sel/lambda.json", "--lambda-selection-sha256", lsha,
               "--alpha-selection", "sel/alpha.json", "--alpha-selection-sha256", asha,
               "--source-run-dir", str(e3), "--expect-source-sha256", s3, "--out-dir", out_ok]
    allp = {**cuda, fake_iso: ok_iso, ident: clean_at}
    check("gates_pass_e6_with_bound_selections", gate(e6_base, e6, patches=allp) is None)
    check("gate_refuses_e6_without_alpha",
          gate(without(e6_base, "--alpha-selection", "--alpha-selection-sha256"), e6, patches=allp)
          == "alpha_selection_missing")
    d = TMP / "e3_l05"
    s6 = write_e3_parent(d, lam=0.5)
    argv6 = [a if a not in (str(e3), s3) else (str(d) if a == str(e3) else s6) for a in e6_base]
    check("gate_refuses_e6_parent_not_the_selected_run", gate(argv6, e6, patches=allp) == "parent_lambda_mismatch")
    # ---- run_qat's own refusals: nothing is written by any of them
    out = TMP / "never_written"
    for name, kw, want in [
        ("run_refuses_ptq_stage", dict(stage="e4"), "stage_not_qat"),
        ("run_refuses_mode", dict(mode="dry"), "mode_invalid"),
        ("run_refuses_test_hooks_in_real", dict(mode="real", hooks=Q.QATHooks(), clip_source="u4_pilot"),
         "real_run_test_hooks"),
        ("run_refuses_clip_source_in_real", dict(mode="real", loaders=None), "clip_source"),
        ("run_refuses_cpu_in_real", dict(mode="real", loaders=None, clip_source="u4_pilot"), "cuda_required"),
        ("run_refuses_bad_clip", dict(clip_norm=0.0), "grad_clip_norm_invalid"),
        ("run_refuses_nonempty_out_dir", dict(out_dir=nonempty), "out_dir_not_empty"),
    ]:
        args = dict(stage="e5", mode="smoke", model=student(), source_meta=SOURCE, out_dir=out, seed=42,
                    clip_norm=1.0, clip_source="smoke", device="cpu", num_workers=0, loaders=make_loaders(),
                    log=lambda *a: None)
        args.update(kw)
        got = code_of(Q.run_qat, **args)
        check(name, got == want and not out.exists(), f"[{got}]")
    with patched(Q, "config_pins_error", lambda: "configs/quant.py qat.epochs = 14, AM-4a: 15"):
        got = code_of(Q.run_qat, stage="e5", mode="smoke", model=student(), source_meta=SOURCE, out_dir=out, seed=42,
                      clip_norm=1.0, clip_source="smoke", device="cpu", num_workers=0, loaders=make_loaders(),
                      log=lambda *a: None)
        check("run_refuses_config_pins", got == "config_pins" and not out.exists(), f"[{got}]")
    # ---- the E3 source: projection-free only (AM-4a item 5)
    d = TMP / "e3_leak"
    s7 = write_e3_parent(d, projection_in_student=True)
    pl = Q.resolve_parent(e6, d, s7, 42)
    check("load_source_refuses_projection_in_e6_student", code_of(Q.load_source, e6, pl) == "source_invalid")
    model, meta = Q.load_source(e6, p3)
    check("load_source_e6_projection_free", meta["sha256"] == s3 and not any("cwd" in k for k in model.state_dict()))
    # ---- each refusal occurrence of P8-P10 with its own case (scripts/smoke_qat_mutations.py removes them)
    d = TMP / "e1_notjson"
    s8 = write_e1_parent(d)
    (d / "best.json").write_text("{not json", encoding="utf-8")
    check("refuses_parent_best_json_not_json", code_of(Q.resolve_parent, e5, d, s8, 42) == "parent_best_json_format")
    d = TMP / "e1_nomode"
    s8 = write_e1_parent(d)
    (d / "e1_telemetry.jsonl").write_text(json.dumps({"event": "run_meta", "seed": 42, "max_iters": 80000}) + "\n"
                                          + json.dumps({"event": "val", "iter": 80000}) + "\n", encoding="utf-8")
    check("refuses_parent_run_meta_without_mode", code_of(Q.resolve_parent, e5, d, s8, 42) == "parent_run_meta_key")
    d = TMP / "e1_otherstage"
    s8 = write_e1_parent(d)
    (d / "e1_telemetry.jsonl").write_text(json.dumps({"event": "run_meta", "mode": "real", "seed": 42, "stage": "E2",
                                                      "max_iters": 80000}) + "\n"
                                          + json.dumps({"event": "val", "iter": 80000}) + "\n", encoding="utf-8")
    check("refuses_e1_parent_of_another_stage", code_of(Q.resolve_parent, e5, d, s8, 42) == "parent_stage_mismatch")
    notjson = TMP / "clip_selection_notjson.json"
    notjson.write_text("{not json", encoding="utf-8")
    nocand = TMP / "clip_selection_nocandidate.json"
    nocand.write_text(json.dumps({"format": "qat_clip_selection/1", "winner": {"clip_norm": 2.0}}), encoding="utf-8")
    latest = TMP / "latest_clip_selection.json"
    got = code_of(cb, "E5", 44, 5.0, u4_pilot=False, clip_selection=str(latest), clip_selection_sha256=ssha)
    check("refuses_clip_selection_test_path", got == "clip_selection_test_path" and not latest.exists(), f"[{got}]")
    for name, f in (("refuses_clip_selection_not_json", notjson),
                    ("refuses_clip_selection_winner_not_a_candidate", nocand)):
        got = code_of(cb, "E5", 44, 5.0, u4_pilot=False, clip_selection=str(f), clip_selection_sha256=sha_file(f))
        check(name, got == "clip_selection_format", f"[{got}]")
    (repo / "sel" / "notjson.json").write_text("{not json", encoding="utf-8")
    git(repo, "add", "sel/notjson.json")
    git(repo, "commit", "-q", "-m", "a selection that is not JSON")
    got = code_of(rts, "sel/notjson.json", sha_file(repo / "sel" / "notjson.json"), "lambda_selection/1", "lambda",
                  repo)
    check("refuses_selection_not_json", got == "selection_format", f"[{got}]")

    def no_backend():
        raise QuantBackendUnavailable("qnnpack is not available here")
    check("gate_refuses_backend_unavailable",
          gate(base, patches={**cuda, (Q, "select_qnnpack_backend"): no_backend}) == "backend_unavailable")
    import src.quant.runner as runner_mod
    with patched(runner_mod, "check_source", lambda stage, path: (student(), {}, {"sha256": "0" * 64})):
        check("load_source_refuses_changed_source", code_of(Q.load_source, e6, p3) == "source_changed")
    leaky = student()
    leaky.register_buffer("cwd_projection_weight", torch.zeros(1))
    with patched(runner_mod, "check_source", lambda stage, path: (leaky, {}, {"sha256": p3["checkpoint_sha256"]})):
        check("load_source_refuses_projection_keys", code_of(Q.load_source, e6, p3) == "source_projection_keys")

    # ---- run_qat's later refusals and STOPs, reached on the CPU through patches; nothing is trained
    class FakeLoader:
        def __init__(self, n: int, batch_size: int, drop_last: bool):
            self.n, self.batch_size, self.drop_last = n, batch_size, drop_last
            self.dataset = range(n * batch_size)

        def __len__(self):
            return self.n

        def __iter__(self):
            return iter(())

    def real_attempt(name: str, want: str, patches: dict, **kw) -> None:
        o = TMP / f"run_real_{name}"
        args = dict(stage="e5", mode="real", model=student(), source_meta=SOURCE, out_dir=o, seed=42, clip_norm=1.0,
                    clip_source="u4_pilot", device="cuda", num_workers=1, log=lambda *a: None)
        args.update(kw)
        # on a CUDA pod d1 has initialised CUDA in this process: each case states the flag it needs
        with contextlib.ExitStack() as stack:
            for (obj, attr), value in {(torch.cuda, "is_initialized"): (lambda: False), **patches}.items():
                stack.enter_context(patched(obj, attr, value))
            got = code_of(Q.run_qat, **args)
        check(name, got == want and not o.exists(), f"[{got}]")

    of_record = lambda seed, nw: (FakeLoader(335, 16, True), FakeLoader(53, 16, False))   # noqa: E731
    real_attempt("run_refuses_cuda_initialised_in_real", "cuda_initialized_before_seed",
                 {(torch.cuda, "is_initialized"): (lambda: True)})
    real_attempt("run_refuses_steps_per_epoch_in_real", "steps_per_epoch",
                 {(Q, "build_qat_loaders"): (lambda seed, nw: (FakeLoader(2, 16, True), FakeLoader(1, 16, False)))})
    real_attempt("run_refuses_loader_not_of_record_in_real", "loader_of_record",
                 {(Q, "build_qat_loaders"): (lambda seed, nw: (FakeLoader(335, 8, True), FakeLoader(53, 8, False)))})
    real_attempt("run_refuses_unfused_bn_in_real", "unfused_batchnorm",
                 {(Q, "build_qat_loaders"): of_record, (Q, "unfused_batchnorm"): (lambda m: ["features.0.0.bn"])})
    o = TMP / "run_smoke_empty_loader"
    empty = DataLoader(TensorDataset(torch.zeros(0, 3, 64, 64), torch.zeros(0, 64, 64, dtype=torch.long)),
                       batch_size=16, drop_last=True)
    got = code_of(Q.run_qat, stage="e5", mode="smoke", model=student(), source_meta=SOURCE, out_dir=o, seed=42,
                  clip_norm=1.0, clip_source="smoke", device="cpu", num_workers=0, loaders=(empty, make_loaders()[1]),
                  log=lambda *a: None)
    check("run_refuses_empty_loader", got == "empty_loader" and not o.exists(), f"[{got}]")
    o = TMP / "run_smoke_no_backend"
    with patched(Q, "select_qnnpack_backend", no_backend):
        got = code_of(Q.run_qat, stage="e5", mode="smoke", model=student(), source_meta=SOURCE, out_dir=o, seed=42,
                      clip_norm=1.0, clip_source="smoke", device="cpu", num_workers=0, loaders=make_loaders(),
                      log=lambda *a: None)
    check("run_refuses_unavailable_backend", got == "backend_unavailable" and not o.exists(), f"[{got}]")

    def stop_of(**kw) -> str | None:
        try:
            Q.run_qat(stage="e5", mode="smoke", model=student(), source_meta=SOURCE, seed=42, clip_norm=1.0,
                      clip_source="smoke", device="cpu", num_workers=0, loaders=make_loaders(), log=lambda *a: None,
                      **kw)
        except Q.QATStop as e:
            return e.code
        except Exception as e:                                  # noqa: BLE001 -- the named check fails
            return f"raised:{type(e).__name__}"
        return None
    o = TMP / "run_smoke_optimizer"
    with patched(Q, "optimizer_record", lambda opt, model: {"class": "SGD", "param_groups": 1, "momentum": 0.0}):
        got = stop_of(out_dir=o)
    check("run_stops_on_an_optimizer_not_of_record", got == "optimizer_of_record"
          and not (o / Q.TELEMETRY_NAME).exists(), f"[{got}]")

    class Meddler(Q.QATHooks):
        def before_val(self, epoch, prepared):
            bn = next(m for m in prepared.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
            bn.running_mean.add_(1.0)
    o = TMP / "run_smoke_meddled_val"
    got = stop_of(out_dir=o, hooks=Meddler(), max_steps=2, max_val_batches=1)
    rows = [json.loads(x) for x in (o / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()] \
        if (o / Q.TELEMETRY_NAME).is_file() else []
    check("run_stops_when_val_changes_the_state", got == "val_bracket_changed_state"
          and rows and rows[-1].get("event") == "run_stop" and rows[-1].get("code") == "val_bracket_changed_state",
          f"[{got}]")
    # ---- a checkpoint name holding "test"; a clip selection given by its sha256 alone
    d = TMP / "e1_ckname"
    s9 = write_e1_parent(d, best={"best_ckpt": "/workspace/e1_ckpts/latest.pt", "best_val_miou_all_class": 0.3})
    got = code_of(Q.resolve_parent, e5, d, s9, 42)
    check("refuses_parent_checkpoint_test_name", got == "parent_checkpoint_test_path", f"[{got}]")
    got = code_of(Q.read_clip_selection, None, ssha)
    check("refuses_clip_selection_path_absent", got == "clip_selection_missing", f"[{got}]")

    # ---- G1 (F3): each half of names_test and each name check's place; nothing named "test" is created
    # (a) the given name: a never-created --source-run-dir named latest is refused by its name before any probe
    #     (resolve_parent's is_dir() first would report parent_dir_missing)
    g1_latest = TMP / "latest"
    got = gate([a if a != str(good) else str(g1_latest) for a in base], patches=allp)
    check("g1_given_name_refused_before_any_probe", got == "parent_dir_test_path" and not os.path.lexists(g1_latest),
          f"[{got}]")
    # (b) the resolved name: a real symlink named current_run whose target is a never-created path under a directory
    #     named test is refused once resolved, before is_dir() (the link dangles: parent_dir_missing otherwise)
    g1_link, g1_target = TMP / "current_run", TMP / "g1_absent" / "test" / "e1_s42"
    os.symlink(g1_target, g1_link)
    got = gate([a if a != str(good) else str(g1_link) for a in base], patches=allp)
    check("g1_resolved_name_refused_before_is_dir",
          got == "parent_dir_test_path" and "test" not in str(g1_link).lower() and os.path.islink(g1_link)
          and "test" in os.path.realpath(g1_link).lower() and not os.path.lexists(TMP / "g1_absent"), f"[{got}]")
    # a never-created --clip-selection named latest_selection.json is refused by its name before is_file()
    # (clip_selection_missing otherwise), and nothing lists it
    g1_sel = TMP / "latest_selection.json"
    argv43 = ["--real-run", "--confirm-real-run", "--expect-head", head, "--seed", "43", "--num-workers", "12",
              "--grad-clip-norm", "5.0", "--clip-selection", str(g1_sel), "--clip-selection-sha256", ssha,
              "--source-run-dir", str(good), "--expect-source-sha256", sha, "--out-dir", out_ok]
    with listing_recorder(g1_sel) as listed:
        got = gate(argv43, patches=allp)
    check("g1_clip_selection_name_refused_before_probe",
          got == "clip_selection_test_path" and not listed and not os.path.lexists(g1_sel), f"[{got}]")
    # the --out-dir order (report §11): a never-created --out-dir inside the repository named latest_qat_out is
    # refused by its name, before check_output_dir (which refuses any path in the repository) and before a listing
    g1_out = REPO / "latest_qat_out"
    with listing_recorder(g1_out) as listed:
        code, msg = refusal_of(Q.real_run_gates, Q.build_parser(e5).parse_args(
            [a if a != out_ok else str(g1_out) for a in base]), e5, repo_root=repo)
    check("g1_out_dir_name_refused_before_listing",
          code == "out_dir" and "contains 'test'" in msg and not listed and not os.path.lexists(g1_out),
          f"[{code}] {msg[:120]}")

    # ---- main(): a refused launch prints one RESULT line and writes nothing
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = Q.main(base, "e5")
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    check("main_refusal_exit_2_one_result_line", rc == 2 and len(lines) == 1 and "REFUSED [" in lines[0]
          and not Path(out_ok).exists(), lines[0] if lines else buf.getvalue()[-200:])
    # an exception before the run_meta row is an ERROR (exit 4); after it, the run ABORTED (exit 3)
    fake_gates = {"clip": {"clip_source": "u4_pilot", "u4_pilot": True, "clip_selection": None}, "parent": {},
                  "e6_binding": None, "data_root": "none", "isolation": {"counts": {}}, "identity": {}}

    def main_with(argv, run_fn) -> tuple[int, list[str]]:
        out_buf = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(out_buf))
            stack.enter_context(patched(Q, "real_run_gates", lambda args, stage, **kw: fake_gates))
            stack.enter_context(patched(Q, "load_source", lambda stage, parent: (student(), SOURCE)))
            stack.enter_context(patched(Q, "run_qat", run_fn))
            code = Q.main(argv, "e5")
        return code, [ln for ln in out_buf.getvalue().splitlines() if ln.startswith("RESULT:")]

    def fails_before_telemetry(**kw):
        raise RuntimeError("out of memory before the first row")

    def fails_after_telemetry(**kw):
        o_dir = Path(kw["out_dir"])
        o_dir.mkdir(parents=True)
        (o_dir / Q.TELEMETRY_NAME).write_text(json.dumps({"event": "run_meta"}) + "\n", encoding="utf-8")
        raise RuntimeError("a failure after the run_meta row")
    rc, lines = main_with([a if a != out_ok else str(TMP / "qat_out_early") for a in base], fails_before_telemetry)
    check("main_error_before_telemetry_exit_4", rc == 4 and len(lines) == 1
          and lines[0].startswith("RESULT: ERROR [RuntimeError]"), lines[0] if lines else "")
    rc, lines = main_with([a if a != out_ok else str(TMP / "qat_out_late") for a in base], fails_after_telemetry)
    check("main_abort_after_telemetry_exit_3", rc == 3 and len(lines) == 1
          and lines[0].startswith("RESULT: ABORTED [exception]"), lines[0] if lines else "")


# ---------------------------------------------------------------- import isolation
def test_isolation() -> None:
    code = ("import json, sys; sys.path.insert(0, %r); import src.quant.qat; "
            "print(json.dumps(sorted(sys.modules)))" % str(REPO))
    p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=600)
    mods = set(json.loads(p.stdout.strip().splitlines()[-1])) if p.returncode == 0 else set()
    banned = ("src.eval.eval_runtime", "src.eval.artifacts", "scripts.evaluate_model", "src.training.train_distill",
              "src.training.sweep_select", "scripts.preflight_e1_trainval")
    check("import_isolation", p.returncode == 0 and not (mods & set(banned)),
          f"loaded: {sorted(mods & set(banned))}" if p.returncode == 0 else p.stderr[-300:])
    in_proc = [m for m in banned if m in sys.modules]
    check("no_lane_k2_module_after_runs", not in_proc, str(in_proc))


# ---------------------------------------------------------------- check-run-meta: the QAT launch profile
def set_dotted(d: dict, key: str, value) -> None:
    """Set an existing dotted key (a missing one raises: a renamed trainer key fails the profile smoke)."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d[part]
    if parts[-1] not in d:
        raise KeyError(key)
    d[parts[-1]] = value


def launched_row(stage: str, *, extra: dict, over: dict) -> dict:
    """A real launch's run_meta row: the row the trainer writes in a 2-step smoke run, with the keys
    scripts/run_e5.py adds and the launch values of record set over it."""
    out = TMP / f"profile_src_{stage}_{len(list(TMP.glob('profile_src_*')))}"
    Q.run_qat(stage=stage, mode="smoke", model=student(), source_meta=SOURCE, out_dir=out, seed=42, clip_norm=1.0,
              clip_source="smoke", device="cpu", num_workers=0, loaders=make_loaders(42), extra_meta=extra,
              max_steps=2, max_val_batches=1, log=lambda *a: None)
    row = json.loads((out / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()[0])
    for k, v in over.items():
        set_dotted(row, k, v)
    return row


def test_profile() -> None:
    import scripts.qat_epoch_eval as QEE
    head, src = "a" * 40, "c" * 64
    launch = {"mode": "real", "seed": 42, "clip_norm": 1.0, "clip_source": "u4_pilot", "git_head": head,
              "code_clean_at_head": True, "source_checkpoint_sha256": src, "image_digest": Q.IMAGE_DIGEST_OF_RECORD,
              "num_workers": 12, "persistent_workers": True, "steps_per_epoch": 335, "total_steps": 5025,
              "scheduler.T_max": 5025, "bn_freeze_after_step": 3350, "obs_freeze_after_step": 4020,
              "val_batches_cap": None, "torch": "2.1.0+cu121", "cuda_initialized_at_seed": False,
              "parent": {"checkpoint_sha256": src, "seed": 42, "mode": "real", "stage": None}}
    e5_extra = {"clip_selection": None, "u4_pilot": True, "e6_selections": None}
    row = launched_row("e5", extra=e5_extra, over=launch)
    e5_args = ["--stage", "E5", "--seed", "42", "--grad-clip-norm", "1.0", "--u4-pilot", "--expect-head", head,
               "--expect-source-sha256", src]

    def cli(name: str, r: dict, argv: list[str]) -> tuple[int, str, str]:
        d = TMP / f"profile_{name}"
        d.mkdir()
        (d / Q.TELEMETRY_NAME).write_text(json.dumps(r) + "\n", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = QEE.main(["check-run-meta", "--run-dir", str(d), *argv])
        out = buf.getvalue()
        lines = [ln for ln in out.splitlines() if ln.startswith("RESULT:")]
        return rc, (lines[-1] if lines else out[-300:]), out

    rc, res, _ = cli("e5_ok", row, e5_args)
    check("profile_e5_s42_pilot_passes", rc == 0 and res == "RESULT: CHECK-RUN-META PASS", res)
    census = dict(QEE.EXPECTED_CENSUS)
    census["MovingAverageMinMaxObserver"] = 109
    for name, key, value in [
        ("num_workers", "num_workers", 8), ("steps_per_epoch", "steps_per_epoch", 336),
        ("t_max", "scheduler.T_max", 335), ("eta_min", "scheduler.eta_min", 1e-6),
        ("scheduler_class", "scheduler.class", "OneCycleLR"),
        ("bn_freeze_step", "bn_freeze_after_step", 3685), ("observer_freeze_step", "obs_freeze_after_step", 4355),
        ("lr", "lr", 1e-3), ("momentum", "momentum", 0.0), ("weight_decay", "weight_decay", 0.0),
        ("nesterov", "nesterov", True), ("batch_size", "batch_size", 8), ("drop_last", "drop_last", False),
        ("engine", "engine", "fbgemm"), ("observer_census", "observer_census", census),
        ("fake_quant_count", "fake_quant_modules", 183), ("unfused_batchnorm", "unfused_batchnorm", 1),
        ("tf32", "tf32", {**Q.TF32_DEFAULTS, "cuda_matmul_allow_tf32": True}), ("torch", "torch", "2.2.0+cu121"),
        ("image_digest", "image_digest", "sha256:" + "0" * 64), ("head", "git_head", "b" * 40),
        ("code_not_clean", "code_clean_at_head", False), ("source_sha", "source_checkpoint_sha256", "d" * 64),
        ("parent_sha", "parent", {**launch["parent"], "checkpoint_sha256": "d" * 64}),
        ("parent_seed", "parent", {**launch["parent"], "seed": 43}), ("mode", "mode", "smoke"),
        ("seed", "seed", 43), ("clip", "clip_norm", 5.0), ("dampening_as_int", "dampening", 0),
        ("cuda_initialised_at_seed", "cuda_initialized_at_seed", True), ("val_cap", "val_batches_cap", 4),
        ("stage", "stage", "E6"),
    ]:
        r = json.loads(json.dumps(row))
        set_dotted(r, key, value)
        rc, res, out = cli(f"e5_{name}", r, e5_args)
        shown = key.split(".")[0] if key in ("parent",) else key
        check(f"profile_stops_on_{name}", rc == 1 and "[run_meta_profile]" in res and shown in out, res[:160])
    rc, res, out = cli("e5_no_u4_flag", row, [a for a in e5_args if a != "--u4-pilot"])
    check("profile_pilot_checked_without_u4_pilot_stops", rc == 1 and "clip_source" in out, res[:160])
    # a non-pilot launch is bound to clip_selection.json (P10)
    clip_p = TMP / "profile_clip_selection.json"
    clip_p.write_text(json.dumps({"format": Q.CLIP_SELECTION_FORMAT, "winner": {"clip_norm": 5.0}}), encoding="utf-8")
    csha = sha_file(clip_p)
    bound = {"clip_selection": {"path": str(clip_p), "sha256": csha, "winner": 5.0}, "u4_pilot": False,
             "e6_selections": None}
    r43 = launched_row("e5", extra=bound, over={**launch, "seed": 43, "clip_norm": 5.0, "clip_source": "clip_selection",
                                                 "parent": {**launch["parent"], "seed": 43}})
    a43 = ["--stage", "E5", "--seed", "43", "--grad-clip-norm", "5.0", "--clip-selection", str(clip_p),
           "--clip-selection-sha256", csha, "--expect-head", head, "--expect-source-sha256", src]
    rc, res, _ = cli("e5_s43_ok", r43, a43)
    check("profile_e5_s43_bound_to_clip_selection_passes", rc == 0 and res == "RESULT: CHECK-RUN-META PASS", res)
    rc, res, out = cli("e5_s43_other_sha", r43, [a if a != csha else "e" * 64 for a in a43])
    check("profile_stops_on_other_clip_selection", rc == 1 and "clip_selection" in out, res[:160])
    # the bound file is read again at check time: its bytes changed after the launch, or no file given
    clip_q = TMP / "profile_clip_selection_changed.json"
    clip_q.write_text(clip_p.read_text(encoding="utf-8"), encoding="utf-8")
    qsha = sha_file(clip_q)
    r43q = json.loads(json.dumps(r43))
    r43q["clip_selection"] = {"path": str(clip_q), "sha256": qsha, "winner": 5.0}
    a43q = [str(clip_q) if a == str(clip_p) else (qsha if a == csha else a) for a in a43]
    clip_q.write_text(json.dumps({"format": Q.CLIP_SELECTION_FORMAT, "winner": {"clip_norm": 5.0}, "note": "edited"}),
                      encoding="utf-8")
    rc, res, out = cli("e5_s43_file_changed", r43q, a43q)
    check("profile_stops_on_clip_selection_file_changed", rc == 1 and "clip_selection_file" in out, res[:160])
    rc, res, out = cli("e5_s43_sha_without_file", r43, without(a43, "--clip-selection"))
    check("profile_stops_on_clip_selection_sha_without_file", rc == 1 and "clip_selection_file" in out, res[:160])
    rc, res, out = cli("e5_s43_u4", r43, a43 + ["--u4-pilot"])
    check("profile_stops_on_u4_pilot_for_s43", rc == 1 and "clip_source" in out, res[:160])
    r43u = json.loads(json.dumps(r43))
    r43u.update({"u4_pilot": True, "clip_source": "u4_pilot", "clip_selection": None})
    rc, res, out = cli("e5_s43_launched_as_pilot", r43u, a43)
    check("profile_stops_on_s43_launched_as_pilot", rc == 1 and "clip_source" in out, res[:160])
    rc, res, out = cli("e5_alpha_flag", row, e5_args + ["--alpha-selection", "sel/alpha.json"])
    check("profile_stops_on_e6_selections_for_e5", rc == 1 and "e6_selections" in out, res[:160])

    # E6 (O2): the λ and α selections, tracked and unchanged at HEAD, and the parent they name
    repo = TMP / "profile_selrepo"
    (repo / "sel").mkdir(parents=True)
    lam_doc = {"format": "lambda_selection/1", "winner": {"lambda": 1.0, "run_id": "e3_s42_l1"}}
    alp_doc = {"format": "alpha_selection/1", "winner": {"alpha": 50.0, "run_id": "e3_s42_a50"}}
    (repo / "sel" / "lambda.json").write_text(json.dumps(lam_doc), encoding="utf-8")
    (repo / "sel" / "alpha.json").write_text(json.dumps(alp_doc), encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "sel/lambda.json", "sel/alpha.json")
    git(repo, "commit", "-q", "-m", "selections")
    lsha, asha = sha_file(repo / "sel" / "lambda.json"), sha_file(repo / "sel" / "alpha.json")
    p3 = {"checkpoint_sha256": src, "seed": 42, "mode": "real", "stage": "E3", "lambda_logit": 1.0,
          "alpha_cwd": 50.0, "run_id": "e3_s42_a50"}
    e6_bound = {"clip_selection": {"path": str(clip_p), "sha256": csha, "winner": 5.0}, "u4_pilot": False,
                "e6_selections": {"lambda_selection": {"path": "sel/lambda.json", "sha256": lsha},
                                  "alpha_selection": {"path": "sel/alpha.json", "sha256": asha}}}
    r6 = launched_row("e6", extra=e6_bound, over={**launch, "clip_norm": 5.0, "clip_source": "clip_selection",
                                                   "parent": p3})
    a6 = QEE.build_parser().parse_args(
        ["check-run-meta", "--run-dir", str(TMP), "--stage", "E6", "--seed", "42", "--grad-clip-norm", "5.0",
         "--clip-selection", str(clip_p), "--clip-selection-sha256", csha, "--expect-head", head,
         "--expect-source-sha256", src, "--lambda-selection", "sel/lambda.json", "--lambda-selection-sha256", lsha,
         "--alpha-selection", "sel/alpha.json", "--alpha-selection-sha256", asha])
    bad = QEE.profile_mismatches(r6, a6, repo_root=repo)
    check("profile_e6_bound_to_its_selections_passes", bad == {} and r6.get("cwd_projection_loaded") is False,
          str(sorted(bad)))
    for name, edit, key in [
        ("e6_parent_lambda", {"parent": {**p3, "lambda_logit": 0.5}}, "e6_binding"),
        ("e6_parent_alpha", {"parent": {**p3, "alpha_cwd": 25.0}}, "e6_binding"),
        ("e6_s42_parent_not_alpha_winner", {"parent": {**p3, "run_id": "e3_s42_a25"}}, "e6_binding"),
        ("e6_lambda_selection_sha", {"e6_selections": {**e6_bound["e6_selections"],
                                                       "lambda_selection": {"sha256": "0" * 64}}},
         "e6_selections.lambda_selection"),
        ("e6_projection_loaded", {"cwd_projection_loaded": True}, "cwd_projection_loaded"),
    ]:
        r = {**json.loads(json.dumps(r6)), **edit}
        bad = QEE.profile_mismatches(r, a6, repo_root=repo)
        check(f"profile_stops_on_{name}", key in bad, str(sorted(bad)))
    # the record itself: no telemetry yet (exit 3), a first line that is not strict JSON, or not the run_meta row
    none = TMP / "profile_no_telemetry"
    none.mkdir()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = QEE.main(["check-run-meta", "--run-dir", str(none), *e5_args])
    res = ([ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")] or [""])[-1]
    check("profile_run_without_telemetry_exit_3", rc == 3 and "[telemetry_missing]" in res, res)
    for name, first in (("profile_first_row_not_json", '{"event": "run_meta", "lr": NaN}'),
                        ("profile_first_row_not_run_meta", json.dumps({"event": "train", "step": 1}))):
        d = TMP / f"profile_{name}"
        d.mkdir()
        (d / Q.TELEMETRY_NAME).write_text(first + "\n", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = QEE.main(["check-run-meta", "--run-dir", str(d), *e5_args])
        res = ([ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")] or [""])[-1]
        check(name, rc == 2 and "[run_meta_rows]" in res, res)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    args = ap.parse_args()
    want = [s for s in args.sections.split(",") if s]
    unknown = [s for s in want if s not in SECTIONS]
    if unknown:
        print(f"RESULT: ERROR unknown sections {unknown}")
        return 4
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print(f"RESULT: ERROR --device {args.device} but this host has no CUDA device (the CPU run omits --device)")
        return 4
    print("=" * 78)
    print("QAT RUNNER SMOKE (d1) — synthetic; the real student through src/quant/qat.py, no dataset")
    print(f"torch {torch.__version__} | device {args.device} | sections {want} | temp {TMP}")
    print("=" * 78)
    clean = None
    if "d1" in want or "nan" in want:
        print("\n--- d1: clean run (max_norm 0.1) ---")
        clean = test_d1(args.device)
    if "nan" in want:
        print("\n--- d1: NaN in the loss at step 10 and at step 27 ---")
        test_nonfinite(args.device, clean)
    if clean is not None:
        shutil.rmtree(clean["out"], ignore_errors=True)
    if "gates" in want:
        print("\n--- launch gates ---")
        test_gates()
    if "isolation" in want:
        print("\n--- import isolation ---")
        test_isolation()
    if "profile" in want:
        print("\n--- check-run-meta: the QAT launch profile ---")
        test_profile()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
