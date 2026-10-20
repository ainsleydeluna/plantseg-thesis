#!/usr/bin/env python3
"""Lane L-AM3, B-2: E6-KD's step smoke (plan §4.5). The pod's `run` makes no bitwise claim (LB33).

    python -B scripts/smoke_e6kd_step.py run --source-run-dir D --expect-source-sha256 H --projection-sha256 H \
        --teacher-ckpt T --teacher-ckpt-sha256 H [--num-workers 12] [--steps 25]
    python -B scripts/smoke_e6kd_step.py --selfcheck

`run`, on the pod in the KD image, before the launch: the setup (the KD image digest and fingerprint, PLANTSEG_GIT_COMMIT
unset, the governed paths clean, the teacher of record's identity, the QNNPACK engine, the CUDA order, the TRAIN/VAL
data root), then a 25-step E6-KD run in mode "smoke" (src/quant/qat_kd.py run_qat_kd) on the real loaders with E3's
student, the projection of record and the teacher of record: the shapes at batch 16, the step-1 checks, finite losses,
the KD columns, one teacher forward per step, NMF draws equal to the steps, a one-batch VAL, the peak reserved memory
(at most 44 GB, an INFERRED bound; the median seconds per step are recorded, not gated), the run_meta's E6-KD values
of record, and no artifact of record written (the run goes to a temp directory outside the repository; its one
epoch checkpoint is a smoke file). Every criterion is printed; RESULT: PASS only when all pass.

`--selfcheck`, in the cloud: the same criteria function on facts measured from a CPU run (the harness's stub SegNeXt,
synthetic 64x64 data whose 25 steps fall in epoch 1, as the pod's do) with the pod-only facts given, then each
criterion in its STOP form (one fact changed at a time), and `run`'s refusals. No check name, file or folder holds
the token SL-1 forbids.

Exit: 0 PASS, 1 STOP (a criterion failed), 2 refused, 4 error.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import safe_tmpdir  # noqa: E402  (imports no repository code)

TMP = safe_tmpdir("smoke_e6kd_step_")
if "--selfcheck" in sys.argv:
    os.environ["PLANTSEG_DATA_ROOT"] = str(TMP / "absent_data_root")     # before configs/data.py is first imported

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.quant import qat as Q  # noqa: E402
from src.quant import qat_kd as KD  # noqa: E402

PEAK_RESERVED_GB_MAX = 44.0          # INFERRED bound (plan §3(x)): the A40's 48 GB less a margin
GOVERNED = ("src", "configs", "scripts", "requirements*", "docs/EVALUATION_CONTRACT.md",
            "docs/IMPLEMENTATION_CONTRACT.md")
SL1_EXCLUDE = ":(exclude,icase)*" + "".join(("t", "e", "s", "t")) + "*"
NEVER_OBSERVED = sorted(f"features.{i}.skip_add.activation_post_process" for i in (2, 4, 7, 11, 13))
CRITERIA = ("env_kd_image_digest", "env_fingerprint_of_record", "env_git_commit_unset", "env_governed_paths_clean",
            "teacher_identity_of_record", "qnnpack_engine", "cuda_order", "data_root_trainval_only",
            "shapes_at_batch_16", "step1_checks_passed", "finite_losses", "kd_columns_present",
            "one_teacher_forward_per_step", "nmf_draws_equal_steps", "one_batch_val",
            "peak_reserved_memory_within_bound", "run_meta_e6kd_values_of_record", "no_artifact_of_record")
results: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# ---------------------------------------------------------------- the criteria, from measured facts
def criteria(f: dict) -> list[tuple[str, bool, str]]:
    """Every criterion of the step smoke, from the facts `f` that `run` measures (or --selfcheck gives)."""
    pd = KD.preflight_distill()
    rows = [r for r in f["rows"] if r.get("event") == "train"]
    vals = [r for r in f["rows"] if r.get("event") == "val"]
    ends = [r for r in f["rows"] if r.get("event") == "epoch_end"]
    s1 = next((r for r in f["rows"] if r.get("event") == "step1_checks"), {})
    meta = f["run_meta"]
    h, w, n = f["height"], f["width"], f["steps"]
    shp = f["shapes"]
    want_shapes = {"logits": [16, 116, h, w], "c5": [16, 160, h // 16, w // 16], "head_logits": [16, 116, h // 8, w // 8],
                   "teacher_logits": [16, 116, h // 8, w // 8], "teacher_feat_s16": [16, 320, h // 16, w // 16]}
    e3w = meta.get("e3_weights") or {}
    prov = meta.get("teacher_provenance") or {}
    rules = pd.provenance_rules({"teacher_ckpt": f.get("teacher_ckpt")})
    prov_ok = sorted(prov) == sorted(rules) and all(fn(prov.get(k)) for k, fn in rules.items())
    meta_want = {"e6kd": True, "weight_factor": 0.5, "stage": "E6-KD", "distillation": True,
                 "terms": {t: True for t in KD.TERMS}, "T_logit": KD.TD.T_LOGIT, "T_cwd": KD.TD.T_CWD,
                 "cwd_C": KD.TD.CWD_C_FEAT, "projection_params": 51200, "projection_fake_quant_modules": 0,
                 "projection_in_student": False, "teacher_mock": False, "engine": "qnnpack",
                 "never_observed_modules": NEVER_OBSERVED, "fake_quant_modules": 184,
                 "cuda_initialized_at_seed": False}
    meta_bad = [k for k, v in meta_want.items() if meta.get(k) != v]
    if meta.get("weights") != KD.e6kd_weights(e3w) if len(e3w) == 3 else True:
        meta_bad.append("weights")
    if KD.kd_optimizer_error(meta.get("optimizer") or {}):
        meta_bad.append("optimizer")
    if not prov_ok:
        meta_bad.append("teacher_provenance")
    if meta.get("image_digest") != pd.KD_IMAGE_DIGEST:
        meta_bad.append("image_digest")
    secs = [r["iter_seconds"] for r in rows[1:] if isinstance(r.get("iter_seconds"), float)]
    median = statistics.median(secs) if secs else None
    out = [
        ("env_kd_image_digest", f["image_digest"] == pd.KD_IMAGE_DIGEST, f"{f['image_digest']}"),
        ("env_fingerprint_of_record", f["fingerprint_problems"] == [], f"{f['fingerprint_problems'][:3]}"),
        ("env_git_commit_unset", f["git_commit_env"] is None, f"PLANTSEG_GIT_COMMIT={f['git_commit_env']!r}"),
        ("env_governed_paths_clean", f["governed_status"] == "", f"{f['governed_status'][:120]!r}"),
        ("teacher_identity_of_record", f["teacher_refusal"] is None, f"{f['teacher_refusal']}"),
        ("qnnpack_engine", f["engine"] == "qnnpack", f"{f['engine']}"),
        ("cuda_order", f["cuda_initialized_before_run"] is False and meta.get("cuda_initialized_at_seed") is False,
         f"initialised before the run: {f['cuda_initialized_before_run']}"),
        ("data_root_trainval_only", f["data_root_refusal"] is None, f"{f['data_root_refusal']}"),
        ("shapes_at_batch_16", shp == want_shapes, f"{shp}"),
        ("step1_checks_passed", s1.get("passed") is True and bool(s1.get("checks")), f"{s1.get('checks')}"),
        ("finite_losses", len(rows) == n and all(isinstance(r.get("loss"), float) and math.isfinite(r["loss"])
                                                 for r in rows) and not any(r.get("event") == "run_abort"
                                                                            for r in f["rows"]),
         f"{len(rows)} rows of {n}"),
        ("kd_columns_present", len(rows) == n and all(isinstance(r.get(t), float) for r in rows for t in KD.TERMS),
         f"{[t for t in KD.TERMS if not all(isinstance(r.get(t), float) for r in rows)]}"),
        ("one_teacher_forward_per_step", f["teacher_train_calls"] == n and f["teacher_val_calls"] == 0,
         f"{f['teacher_train_calls']} train, {f['teacher_val_calls']} VAL calls for {n} steps"),
        ("nmf_draws_equal_steps", [e.get("teacher_nmf_draws") for e in ends] == [n], f"{[e.get('teacher_nmf_draws') for e in ends]}"),
        ("one_batch_val", [v.get("val_batches") for v in vals] == [1], f"{[v.get('val_batches') for v in vals]}"),
        ("peak_reserved_memory_within_bound", f["peak_reserved_gb"] is not None
         and f["peak_reserved_gb"] <= PEAK_RESERVED_GB_MAX,
         f"peak reserved {f['peak_reserved_gb']} GB (bound {PEAK_RESERVED_GB_MAX}); median s/step {median}"),
        ("run_meta_e6kd_values_of_record", not meta_bad, f"{meta_bad}"),
        ("no_artifact_of_record", f["out_files"] == ["epoch_ckpts/e01.pt", "qat_telemetry.jsonl"]
         and f["out_outside_repo"] is True, f"{f['out_files']}"),
    ]
    assert [c[0] for c in out] == list(CRITERIA)
    return out


# ---------------------------------------------------------------- the 25-step run, observed
class Rec(KD.E6KDHooks):
    def __init__(self):
        self.in_val = False

    def before_val(self, epoch, prepared):
        self.in_val = True

    def after_val(self, epoch, prepared):
        self.in_val = False


def run_steps(*, model, teacher, projection, weights: dict, e3: dict, device: str, num_workers: int, steps: int,
              loaders=None) -> dict:
    """run_qat_kd in mode "smoke": `steps` steps, the epoch end it then reaches (one VAL batch), into a temp dir."""
    rec = Rec()
    calls = {"train": 0, "val": 0}
    shapes: dict = {}
    real_forward = teacher.forward

    def forward(x, *a, **kw):
        calls["val" if rec.in_val else "train"] += 1
        return real_forward(x, *a, **kw)
    teacher.forward = forward
    real_losses = KD.TD.distillation_losses

    def spy(**kw):
        if not shapes:
            shapes.update({"logits": list(kw["logits"].shape), "c5": list(kw["c5"].shape),
                           "head_logits": list(kw["head_logits"].shape),
                           "teacher_logits": list(kw["teacher_out"].logits.shape),
                           "teacher_feat_s16": list(kw["teacher_out"].feat_s16.shape)})
        return real_losses(**kw)
    out = TMP / "step_run"
    cuda_before = bool(torch.cuda.is_initialized())
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_initialized() else None
    error = None
    KD.TD.distillation_losses = spy
    try:
        KD.run_qat_kd(mode="smoke", model=model, teacher=teacher, projection=projection, weights=weights, e3=e3,
                      source_meta={"sha256": "0" * 64, "path": None, "bytes": None}, out_dir=out, seed=KD.SEED,
                      clip_norm=5.0, clip_source="smoke", device=device, num_workers=num_workers, loaders=loaders,
                      hooks=rec, max_steps=steps, max_val_batches=1, log=lambda *a: None)
    except Exception as e:                                       # noqa: BLE001 -- the criteria report it
        error = f"{type(e).__name__}: {e}"
    finally:
        KD.TD.distillation_losses = real_losses
    rows = [json.loads(ln) for ln in (out / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()] \
        if (out / Q.TELEMETRY_NAME).is_file() else []
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) if out.is_dir() else []
    peak = (torch.cuda.max_memory_reserved() / 2 ** 30) if device.startswith("cuda") and torch.cuda.is_available() \
        else None
    return {"rows": rows, "run_meta": rows[0] if rows else {}, "shapes": shapes, "teacher_train_calls": calls["train"],
            "teacher_val_calls": calls["val"], "cuda_initialized_before_run": cuda_before, "out_files": files,
            "out_outside_repo": REPO not in out.resolve().parents, "peak_reserved_gb": peak, "steps": steps,
            "error": error}


def env_facts() -> dict:
    pd = KD.preflight_distill()
    rc, fp, out = pd._child_json([REPO / "scripts" / "preflight_distill.py", "_fingerprint"], "FINGERPRINT_JSON", 600)
    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain=v1", "--", *GOVERNED, SL1_EXCLUDE],
                        capture_output=True, text=True)
    return {"image_digest": (os.environ.get("PLANTSEG_IMAGE_DIGEST") or "").strip() or None,
            "fingerprint_problems": (pd.fingerprint_problems(fp) if rc == 0 and fp is not None
                                     else [f"the fingerprint child failed: rc={rc} {out[-200:]}"]),
            "git_commit_env": os.environ.get("PLANTSEG_GIT_COMMIT"),
            "governed_status": st.stdout if st.returncode == 0 else f"git status failed: {st.stderr[-200:]}"}


# ---------------------------------------------------------------- run (the pod)
def cmd_run(a) -> int:
    for flag in ("source_run_dir", "teacher_ckpt"):
        if Q.names_test(getattr(a, flag)):
            print(f"RESULT: REFUSED [forbidden_path] -- --{flag.replace('_', '-')} holds a name SL-1 forbids")
            return 2
    if not torch.cuda.is_available():
        print("RESULT: REFUSED [cuda_required] -- the step smoke runs on the pod's GPU (--selfcheck runs anywhere)")
        return 2
    if torch.cuda.is_initialized():
        print("RESULT: REFUSED [cuda_initialized_before_seed] -- CUDA is already initialised")
        return 2
    from configs.data import DATA, SPLIT_SIZES
    from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
    from src.quant.qconfig import QuantBackendUnavailable, select_qnnpack_backend
    from src.quant.stages import resolve_quant_stage
    f = env_facts()
    try:
        select_qnnpack_backend()
        f["engine"] = torch.backends.quantized.engine
    except QuantBackendUnavailable as e:
        f["engine"] = f"unavailable: {e}"
    try:
        assert_trainval_only_root(Path(DATA["root"]), {"train": SPLIT_SIZES["train"], "val": SPLIT_SIZES["val"]})
        f["data_root_refusal"] = None
    except TrainValIsolationError as e:
        f["data_root_refusal"] = f"[{e.code}] {e}"
    stage = resolve_quant_stage(KD.STAGE_KEY)
    try:
        parent = Q.resolve_parent(stage, a.source_run_dir, a.expect_source_sha256, KD.SEED)
        e3 = KD.e3_record(parent)
        projection, _prec = KD.projection_of_record(parent, a.projection_sha256)
        model, _meta = Q.load_source(stage, parent)
    except Q.QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}")
        return 2
    try:
        teacher, _rec = KD.teacher_of_record(a.teacher_ckpt, a.teacher_ckpt_sha256)
        f["teacher_refusal"] = None
    except Q.QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}")
        return 2
    f.update(run_steps(model=model, teacher=teacher, projection=projection, weights=KD.e6kd_weights(e3["e3_weights"]),
                       e3=e3, device="cuda", num_workers=a.num_workers, steps=a.steps))
    f.update({"height": 512, "width": 512, "teacher_ckpt": str(Path(a.teacher_ckpt).expanduser().resolve())})
    return report(criteria(f), f)


def report(rows: list[tuple[str, bool, str]], f: dict) -> int:
    print("\n[CRITERIA]")
    for name, ok, detail in rows:
        print(f"  {name:40}: {'PASS' if ok else 'FAIL'}  {detail[:300]}")
    failed = [n for n, ok, _ in rows if not ok]
    if f.get("error"):
        print(f"[run] raised {f['error']}")
    print(f"RESULT: {'PASS' if not failed else 'STOP'} ({len(rows) - len(failed)}/{len(rows)} criteria)"
          + (f" -- failed: {failed}" if failed else ""))
    return 0 if not failed else 1


# ---------------------------------------------------------------- --selfcheck (the cloud)
def selfcheck_facts() -> dict:
    """Facts measured from a CPU run (the stub teacher, synthetic 64x64 data with 26 TRAIN batches, so 25 steps lie in
    epoch 1), with the pod-only facts given as the pod would measure them when all is well."""
    from torch.utils.data import DataLoader, TensorDataset
    from scripts.invariance_harness import build_stub_segnext
    from src.distill.segnext_teacher import SegNeXtTeacherAdapter
    from src.distill.teacher import FrozenTeacher
    from src.models.student import build_student
    pd = KD.preflight_distill()
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(416, 3, 64, 64, generator=g)
    y = torch.randint(0, 116, (416, 64, 64), generator=g)
    vx, vy = torch.randn(8, 3, 64, 64, generator=g), torch.randint(0, 116, (8, 64, 64), generator=g)
    loaders = (DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, drop_last=True,
                          generator=torch.Generator().manual_seed(42)),
               DataLoader(TensorDataset(vx, vy), batch_size=4, shuffle=False))
    torch.manual_seed(0)
    model = build_student(pretrained=False)
    gp = torch.Generator().manual_seed(7)
    projection = torch.nn.utils.skip_init(nn.Conv2d, 160, 320, 1, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.randn(320, 160, 1, 1, generator=gp) * 0.05)
    e3 = {"e3_weights": {"logit_kd": 0.5, "cwd_feat": 25.0, "cwd_logit": 3}, "grids": {}}
    f = run_steps(model=model, teacher=FrozenTeacher(SegNeXtTeacherAdapter(build_stub_segnext())),
                  projection=projection, weights=KD.e6kd_weights(e3["e3_weights"]), e3=e3, device="cpu",
                  num_workers=0, steps=25, loaders=loaders)
    teacher = "/workspace/teacher/iter_24000.pth"                  # the pod's path; never created here
    meta = copy.deepcopy(f["run_meta"])
    # the pod's run records the KD image and the teacher of record; the CPU run records neither
    meta.update({"image_digest": pd.KD_IMAGE_DIGEST, "teacher_provenance": {
        "builder": pd.TEACHER_BUILDER, "ckpt_path": str(Path(teacher).resolve()), "ckpt_sha256": pd.TEACHER_SHA256,
        "ckpt_bytes": pd.TEACHER_BYTES, "expected_sha256": pd.TEACHER_SHA256,
        "config_path": str(KD.TD.DEFAULT_TEACHER_CONFIG), **copy.deepcopy(pd.TEACHER_VALUES)}})
    f.update({"run_meta": meta, "image_digest": pd.KD_IMAGE_DIGEST, "fingerprint_problems": [], "git_commit_env": None,
              "governed_status": "", "teacher_refusal": None, "engine": "qnnpack", "data_root_refusal": None,
              "peak_reserved_gb": 31.5, "height": 64, "width": 64, "teacher_ckpt": str(Path(teacher).resolve())})
    return f


def stop_forms(f: dict) -> dict:
    """For each criterion, the facts with one fact changed so that this criterion, and only it, fails."""
    def with_(**kw):
        g = copy.deepcopy(f)
        g.update(kw)
        return g

    def rows_edit(fn):
        g = copy.deepcopy(f)
        g["rows"] = [fn(copy.deepcopy(r)) for r in g["rows"]]
        return g

    def meta_edit(**kw):
        g = copy.deepcopy(f)
        g["run_meta"].update(kw)
        g["rows"][0] = g["run_meta"]
        return g
    first_train = next(r["step"] for r in f["rows"] if r.get("event") == "train")
    return {
        "env_kd_image_digest": with_(image_digest=Q.IMAGE_DIGEST_OF_RECORD),
        "env_fingerprint_of_record": with_(fingerprint_problems=["torch: '2.1.0+cpu' != '2.1.0+cu121'"]),
        "env_git_commit_unset": with_(git_commit_env="0" * 40),
        "env_governed_paths_clean": with_(governed_status=" M src/quant/qat_kd.py\n"),
        "teacher_identity_of_record": with_(teacher_refusal="[teacher_identity] params differ"),
        "qnnpack_engine": with_(engine="fbgemm"),
        "cuda_order": with_(cuda_initialized_before_run=True),
        "data_root_trainval_only": with_(data_root_refusal="[split_counts] a third split"),
        "shapes_at_batch_16": with_(shapes={**f["shapes"], "c5": [16, 160, 8, 8]}),
        "step1_checks_passed": rows_edit(lambda r: {**r, "passed": False} if r.get("event") == "step1_checks" else r),
        "finite_losses": rows_edit(lambda r: {**r, "loss": None} if r.get("event") == "train"
                                   and r["step"] == first_train + 3 else r),
        "kd_columns_present": rows_edit(lambda r: {**r, "cwd_feat": None} if r.get("event") == "train" else r),
        "one_teacher_forward_per_step": with_(teacher_train_calls=f["teacher_train_calls"] + 1),
        "nmf_draws_equal_steps": rows_edit(lambda r: {**r, "teacher_nmf_draws": 26} if r.get("event") == "epoch_end"
                                           else r),
        "one_batch_val": rows_edit(lambda r: {**r, "val_batches": 2} if r.get("event") == "val" else r),
        "peak_reserved_memory_within_bound": with_(peak_reserved_gb=45.0),
        "run_meta_e6kd_values_of_record": meta_edit(weight_factor=1.0),
        "no_artifact_of_record": with_(out_files=[*f["out_files"], "final/e6-kd_int8_qnnpack.torchscript.pt"]),
    }


def cmd_selfcheck() -> int:
    t0 = time.time()
    print("=" * 78)
    print("E6-KD STEP SMOKE --selfcheck (lane L-AM3, B-2) -- each criterion in its PASS and STOP forms, on the CPU")
    print("=" * 78, flush=True)
    f = selfcheck_facts()
    good = criteria(f)
    check("selfcheck_every_criterion_passes_on_good_facts", all(ok for _, ok, _ in good) and f["error"] is None,
          f"failed {[n for n, ok, d in good if not ok]} {[d for n, ok, d in good if not ok][:2]}; {f['error']}")
    for name, g in stop_forms(f).items():
        got = criteria(g)
        failed = [n for n, ok, _ in got if not ok]
        check(f"selfcheck_stops_on_{name}", failed == [name], f"failed {failed}")
    # run's refusals, before any probe and without a GPU
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["run", "--source-run-dir", str(TMP / "nowhere_latest_e3"), "--expect-source-sha256", "0" * 64,
                   "--projection-sha256", "0" * 64, "--teacher-ckpt", str(TMP / "t.pth"), "--teacher-ckpt-sha256",
                   "0" * 64])
    check("selfcheck_run_refuses_forbidden_path", rc == 2 and "[forbidden_path]" in buf.getvalue()
          and not (TMP / "nowhere_latest_e3").exists(), buf.getvalue().strip()[-120:])
    buf = io.StringIO()                                     # the teacher's flag alone (DL-24 SM6)
    with contextlib.redirect_stdout(buf):
        rc = main(["run", "--source-run-dir", str(TMP / "e3_run"), "--expect-source-sha256", "0" * 64,
                   "--projection-sha256", "0" * 64, "--teacher-ckpt", str(TMP / "nowhere_latest_t" / "iter_24000.pth"),
                   "--teacher-ckpt-sha256", "0" * 64])
    check("selfcheck_run_refuses_forbidden_teacher_path", rc == 2 and "[forbidden_path] -- --teacher-ckpt"
          in buf.getvalue() and not (TMP / "nowhere_latest_t").exists(), buf.getvalue().strip()[-120:])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["run", "--source-run-dir", str(TMP / "e3_run"), "--expect-source-sha256", "0" * 64,
                   "--projection-sha256", "0" * 64, "--teacher-ckpt", str(TMP / "t.pth"), "--teacher-ckpt-sha256",
                   "0" * 64])
    check("selfcheck_run_refuses_without_cuda", (rc == 2 and "[cuda_required]" in buf.getvalue())
          or torch.cuda.is_available(), buf.getvalue().strip()[-120:])
    token = "".join(("t", "e", "s", "t"))
    created = [os.path.join(d, n) for d, ds, fs in os.walk(TMP) for n in ds + fs if token in n.lower() or ":" in n]
    check("no_forbidden_named_path_created", not created and all(token not in n.lower() for n, _o, _d in results),
          f"{created[:2]}")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n[time] {time.time() - t0:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selfcheck", action="store_true", help="the cloud's check of the criteria and the refusals")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="the pod's step smoke, before the launch")
    r.add_argument("--source-run-dir", required=True)
    r.add_argument("--expect-source-sha256", required=True)
    r.add_argument("--projection-sha256", required=True)
    r.add_argument("--teacher-ckpt", required=True)
    r.add_argument("--teacher-ckpt-sha256", required=True)
    r.add_argument("--num-workers", type=int, default=12)
    r.add_argument("--steps", type=int, default=25)
    a = ap.parse_args(argv)
    try:
        if a.selfcheck:
            return cmd_selfcheck()
        if a.cmd == "run":
            return cmd_run(a)
        ap.error("give run or --selfcheck")
    except Exception as e:                                       # noqa: BLE001 -- reported, exit 4
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}")
        return 4
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
