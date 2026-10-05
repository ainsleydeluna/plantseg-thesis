#!/usr/bin/env python3
"""d1 (lane 6, L-AM4 + L-AM1q): the QAT trainer of record, src/quant/qat.py, on synthetic data.

The real student runs through run_qat in mode "smoke" on 32 synthetic 64x64 TRAIN images (batch 16,
drop_last: 2 steps per epoch, T_max 30, BN statistics frozen after step 20 and every observer after
step 24) and 8 VAL images. Three trainings go through the trainer's own step, VAL bracket and
checkpoint code: a clean one at max_norm 0.1 (below the fixture's gradient norm, so every step clips),
and two with a real NaN injected into the loss, once at step 10 (observers on) and once at step 27
(observers off). The launch gates are exercised one refusal at a time. No dataset, no checkpoint, no
download; every file goes to a temp directory outside the repository.

    python -B scripts/smoke_qat_runner.py                 # CPU
    python -B scripts/smoke_qat_runner.py --device cuda   # the same trainings on the GPU (before run 1)
    python -B scripts/smoke_qat_runner.py --sections gates,isolation

Ends with one RESULT line; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from src.quant.stages import resolve_quant_stage  # noqa: E402

NC = 116
SECTIONS = ("d1", "nan", "gates", "isolation")
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
        self.at_call, self.bn, self.obs = [], [], []
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
            return orig(*a, **k)
        wrapped._with_counter = True              # the scheduler's own counter still runs inside `orig`
        optimizer.step = wrapped

    def after_step(self, step, prepared, row):
        self.bn.append(Q.bn_buffer_digest(prepared))
        self.obs.append(Q.observer_buffer_digest(prepared))

    def before_val(self, epoch, prepared):
        self.ckpt_before_val[epoch] = (self.out_dir / Q.EPOCH_DIR / f"e{epoch:02d}.pt").is_file()
        self._seen = set()
        self._handles = [m.register_forward_pre_hook(
            lambda mod, inp: self._seen.add(int(mod.observer_enabled.item()))) for _, m in Q.fake_quant_modules(prepared)]

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
    check("d1_lr_last_closed_form", len(lrs) == 30 and abs(lrs[29] - closed_form(29, 30)) <= 1e-12 * closed_form(29, 30),
          f"{lrs[29] if len(lrs) == 30 else None!r} vs {closed_form(29, 30)!r}")
    check("d1_lr_all_steps_match_closed_form",
          len(lrs) == 30 and all(abs(lr - closed_form(t, 30)) <= 1e-12 * closed_form(t, 30) for t, lr in enumerate(lrs)),
          "1.5e-4 x (1 + cos(pi t / 30)), rel tol 1e-12")
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
        idx = [i for i, r in enumerate(c["rows"]) if r.get("epoch") == e and r.get("event") in ("train", "val", "epoch_end")]
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
          bool(c["rows"]) and all(isinstance(r.get("wall_clock"), float) and r["wall_clock"] > 1.7e9 for r in c["rows"]),
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
        check("d1_result_line", Q.result_line(c["summary"]) == "RESULT: QAT COMPLETE (E5, seed 42, clip 0.1, 15/15 epochs)",
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
                      for r in ends if r["epoch"] >= first_bad)
              and all(r.get("state_finite") is True for r in ends if r["epoch"] < first_bad),
              f"epoch_end state_finite false from epoch {first_bad}")
        check(f"d1_schedule_and_batches_continue_step{k}",
              len(steps) == 30 and len(clean_train) == 30
              and [r["lr"] for r in steps] == [r["lr"] for r in clean_train]
              and [r["batch_sha256"] for r in steps] == [r["batch_sha256"] for r in clean_train],
              "same 30 lr values and batch fingerprints as the clean run")
        if c["summary"]:
            check(f"d1_result_line_nonfinite_step{k}",
                  Q.result_line(c["summary"]).startswith(f"RESULT: QAT COMPLETE, STATE NON-FINITE since step {k} ("),
                  Q.result_line(c["summary"]))
        else:
            check(f"d1_result_line_nonfinite_step{k}", False, c["error"] or "")
        if k <= 24:
            errs = rows_of(c, "step_error")
            check("d1_step_errors_recorded_in_nonfinite_state",
                  bool(errs) and all(r["step"] > k and r.get("state_nonfinite") is True for r in errs),
                  f"{len(errs)} step_error rows (a raising forward once the state is non-finite)")
        shutil.rmtree(c["out"], ignore_errors=True)


# ---------------------------------------------------------------- launch gates, one refusal at a time
def code_of(fn, *a, **kw) -> str | None:
    try:
        fn(*a, **kw)
    except Q.QATRefused as e:
        return e.code
    return None


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
        ("refuses_clip_absent", ("E5", 42, None), dict(u4_pilot=True, clip_selection=None, clip_selection_sha256=None),
         "grad_clip_norm_invalid"),
        ("refuses_clip_not_candidate", ("E5", 42, 2.0), dict(u4_pilot=True, clip_selection=None,
                                                              clip_selection_sha256=None), "grad_clip_norm_not_candidate"),
        ("refuses_u4_pilot_outside_e5_s42", ("E5", 43, 1.0), dict(u4_pilot=True, clip_selection=None,
                                                                   clip_selection_sha256=None), "u4_pilot_not_e5_s42"),
        ("refuses_u4_pilot_with_selection", ("E5", 42, 1.0), dict(u4_pilot=True, clip_selection=str(sel),
                                                                   clip_selection_sha256=ssha),
         "u4_pilot_with_clip_selection"),
        ("refuses_e5_s42_without_u4_pilot", ("E5", 42, 1.0), dict(u4_pilot=False, clip_selection=str(sel),
                                                                   clip_selection_sha256=ssha), "u4_pilot_required"),
        ("refuses_missing_clip_selection", ("E5", 44, 5.0), dict(u4_pilot=False, clip_selection=None,
                                                                  clip_selection_sha256=None), "clip_selection_required"),
        ("refuses_clip_selection_sha_format", ("E5", 44, 5.0), dict(u4_pilot=False, clip_selection=str(sel),
                                                                     clip_selection_sha256="abc"),
         "clip_selection_sha256_format"),
        ("refuses_clip_selection_file_missing", ("E5", 44, 5.0), dict(u4_pilot=False, clip_selection=str(TMP / "no.json"),
                                                                       clip_selection_sha256=ssha), "clip_selection_missing"),
        ("refuses_clip_selection_sha_mismatch", ("E5", 44, 5.0), dict(u4_pilot=False, clip_selection=str(sel),
                                                                       clip_selection_sha256="f" * 64),
         "clip_selection_sha256_mismatch"),
        ("refuses_clip_selection_format", ("E5", 44, 5.0), dict(u4_pilot=False, clip_selection=str(bad),
                                                                 clip_selection_sha256=sha_file(bad)),
         "clip_selection_format"),
        ("refuses_clip_not_the_winner", ("E5", 44, 1.0), dict(u4_pilot=False, clip_selection=str(sel),
                                                               clip_selection_sha256=ssha), "clip_selection_winner_mismatch"),
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
        ("refuses_lambda_selection_missing", (None, lsha, "lambda_selection/1", "lambda", repo), "lambda_selection_missing"),
        ("refuses_alpha_selection_missing", (None, asha, "alpha_selection/1", "alpha", repo), "alpha_selection_missing"),
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
        ("refuses_selection_changed_since_head", ("sel/changed.json", sha_file(repo / "sel" / "changed.json"),
                                                  "lambda_selection/1", "lambda", repo), "selection_changed_since_head"),
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
        ("gate_refuses_out_dir_test_path", [a if a != out_ok else str(TMP / "latest_out") for a in base], {}, "out_dir"),
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
          gate(without(e6_base, "--alpha-selection", "--alpha-selection-sha256"), e6, patches=allp) == "alpha_selection_missing")
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
    # ---- main(): a refused launch prints one RESULT line and writes nothing
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = Q.main(base, "e5")
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    check("main_refusal_exit_2_one_result_line", rc == 2 and len(lines) == 1 and "REFUSED [" in lines[0]
          and not Path(out_ok).exists(), lines[0] if lines else buf.getvalue()[-200:])


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
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
