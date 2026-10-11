#!/usr/bin/env python3
"""Smoke for the KD arm launch path (lane L-TEACHER-ARMS, commit C2; ARMS plan section 4.2 with the Phase B-2 GO's
eleven cases and four mutations): the arm registry (src/training/kd_arms.py), the trainer's arm seam
(src/training/train_distill.py --arm), scripts/preflight_arms.py (entry, records, the gate's stages, check-run-meta,
not-launched) and scripts/run_ptq_arm.py.

CPU only, torch only, GPUs hidden, stub teachers. Synthetic inputs only: scratch git repositories built with the
selection smoke's AM-19 scenarios (scripts/smoke_select_sweeps.py, through scripts/smoke_preflight_distill.py's gate
helpers; both imported as libraries), a synthetic registry folder where a check needs an arm that is not registered,
stubbed child processes (the teacher child, src.quant.ptq.run_ptq), and two-step dry runs of the trainer on
scripts/invariance_harness.py's synthetic 16 + 4 TRAIN/VAL set with its stub teacher. No dataset, checkpoint, GPU,
network or teacher file is used, and nothing is written in the repository. Every folder this smoke makes lies under
its work dir (kept for inspection); no name it makes contains "test", and the "test" refusals are shown on path strings
that are never created, refused before any filesystem call.

Checks (61):
  registry (3)         cm_registry_loads_r1_r2, cm_registry_schema, cm_registry_tolerates_extra_entries
  trainer (9)          cm_trainer_arm_requires_e3, cm_trainer_arm_seed_42_only (main(), a real run),
                       cm_trainer_arm_horizon_80k, cm_run_meta_arm_fields, cm_run_meta_target_construction_k1_v1,
                       cm_default_run_meta_has_no_arm_key, cm_payload_arm_keys_only_in_arm_runs,
                       cm_r1_targets_none_loss_bitwise_e3, dt_two_dry_runs_identical
  entry (9)            cm_entry_arm_line_bytes, cm_arm_line_parsed_by_launchlog, cm_arm_line_not_alpha_candidate,
                       cm_launch_note_cut_place, cm_entry_parses_log_with_launchlog, cm_clock_offset_recorded,
                       dt_entry_twice_identical_bytes, rf_test_path_entry, rf_arm_repeat_after_divergence
  gate (17)            cm_gate_stage_list, cm_gate_teacher_identity_arm_values, cm_gate_go_rule_go_required,
                       cm_gate_approval_row_required, cm_gate_arms_pin_rule, cm_gate_lambda_alpha_inherited,
                       cm_gate_launch_block_has_arm, cm_alpha_cut_record_accepted, cm_gate_record_teacher_arm,
                       rf_record_teacher_arm_without_approval, rf_record_teacher_arm_with_arm_teacher,
                       rf_test_path_gate, rf_existing_output_gate_record, rf_missing_input_teacher_record,
                       rf_wrong_checkpoint_pin, rf_missing_go_rule_record, rf_go_rule_no_go
  arithmetic (3)       cm_arith_cut, cm_arith_no_measurement_not_cut, cm_arith_flip_reference_max
  run_meta, lines (3)  cm_check_run_meta_arm_rules, dt_check_run_meta_twice_identical,
                       cm_launched_and_not_launched_lines
  E7-X (4)             cm_e7_arm_wrapper_requires_arm_run_meta, cm_e7_arm_record_binds, rf_test_path_e7,
                       rf_existing_output_e7
  mutations (13)       each a monkeypatched variant; the smoke asserts that the named check, which first passes
                       again unmutated (the control, C5), FAILS under it:
                       mut_parent_true (cm_run_meta_arm_fields), mut_target_construction_missing
                       (cm_run_meta_target_construction_k1_v1), mut_arm_key_in_default_run_meta
                       (cm_default_run_meta_has_no_arm_key), mut_stage_e3_in_arm_line
                       (cm_arm_line_not_alpha_candidate), mut_gate_skips_go_rule (cm_gate_go_rule_go_required),
                       mut_gate_accepts_record_teacher (cm_gate_teacher_identity_arm_values),
                       mut_approval_marker_ignored (cm_gate_approval_row_required), mut_pin_rule_off
                       (cm_gate_arms_pin_rule), mut_median_replaced_by_mean (cm_arith_cut),
                       mut_divergence_repeat_allowed (rf_arm_repeat_after_divergence), mut_min_in_place_of_max
                       (cm_arith_flip_reference_max), mut_alpha_record_rejected (cm_alpha_cut_record_accepted),
                       mut_record_teacher_arm_accepts_arm_teacher (rf_record_teacher_arm_with_arm_teacher)
No check asserts the registry's size: every check passes with an extra registry file present (ruling R10). The
plan's three W gate checks are the record-teacher checks here, on a synthetic record-teacher entry (e3-rt) in a
scratch registry folder.

Usage:  python -B scripts/smoke_arms_kd.py [--work-dir DIR] [--threads N]
  --work-dir  a new folder outside the repository (default: <tempdir>/arms_kd_smoke_<pid>_<n>)
The smoke re-invokes itself with `--internal arm_worker ...` for the arm's dry runs; that is not part of its CLI.
Exit codes: 0 PASS; 1 a check failed; 2 refused, nothing written (a path containing "test", an existing --work-dir, a
--work-dir inside the repository).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only smoke: hide GPUs from every process it starts

import argparse  # noqa: E402
import contextlib  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shlex  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STEPS, VAL_INTERVAL = 2, 2
WORKER_TIMEOUT = 3600
X_NAME = "iter_28000.pth"                        # the synthetic arm teacher's selected checkpoint
X_BYTES = b"synthetic arm teacher checkpoint (smoke_arms_kd; never loaded)\n" * 64
RT_ENTRY = {"format": "kd_arm/1", "name": "e3-rt", "run_stage": "E3-RT", "e7_stage": "E7-RT", "stage": "e3",
            "seed": 42, "horizon": 80000, "teacher": "record", "arm_teacher": None, "go_rule": False, "K": 1,
            "views": 1, "targets": None, "cut_place": 2}          # a synthetic record-teacher arm (ruling R10)

results: list[tuple[str, bool, str]] = []
CASES: dict = {}
DIRS: dict = {}


def check(name: str, ok, detail="") -> None:
    """One verdict; its detail is kept for the summary file either way and printed on a FAIL."""
    results.append((name, bool(ok), str(detail)[:900]))


def case(name: str):
    """A named check whose verdict a mutation re-evaluates: fn() -> (ok, detail)."""
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def run_case(name: str) -> tuple:
    try:
        return CASES[name]()
    except (Exception, SystemExit) as e:  # noqa: BLE001 - a crash (or a scratch refusal) is a failed check
        import traceback
        return False, f"raised {type(e).__name__}: {e} | {traceback.format_exc()[-600:]}"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def patched(obj, name: str, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


# ------------------------------------------------------------------------------------- internal children
def arm_worker(argv) -> int:
    """invariance_harness's worker, unchanged, with run() given the registered arm (smoke_arms_invariance's method)."""
    ap = argparse.ArgumentParser(prog="smoke_arms_kd.py --internal arm_worker")
    ap.add_argument("--arm", required=True)
    a, rest = ap.parse_known_args(argv)
    if any("test" in str(v).lower() for v in rest):
        print("REFUSED: an internal path containing 'test'", file=sys.stderr)
        return 2
    import scripts.invariance_harness as harness
    install = harness._Recorder.install

    def install_with_arm(rec) -> None:
        install(rec)
        from src.training import kd_arms
        spec = kd_arms.load_arm(a.arm)
        run = rec.td.run

        @functools.wraps(run)
        def run_with_arm(*args, **kwargs):
            return run(*args, arm=spec, **kwargs)
        rec.td.run = run_with_arm
    harness._Recorder.install = install_with_arm
    return harness.worker(rest)


INTERNAL = {"arm_worker": arm_worker}


# ------------------------------------------------------------------------------------------- the libraries
def libs():
    import scripts.preflight_arms as pa
    import scripts.preflight_distill as pd
    import scripts.run_ptq_arm as rpa
    import scripts.smoke_preflight_distill as spd
    import scripts.smoke_select_sweeps as sss
    import src.training.sweep_select as ss
    import src.training.train_distill as td
    from src.training import kd_arms
    return SimpleNamespace(pa=pa, pd=pd, rpa=rpa, spd=spd, sss=sss, ss=ss, td=td, kd_arms=kd_arms)


L = None          # set in main()


def scratch(name: str) -> Path:
    return L.sss.scratch_dir(f"akd_{name}")


def spec_of(name: str, arm_dir=None):
    return L.kd_arms.load_arms(arm_dir)[name]


def rt_dir() -> Path:
    """A scratch registry folder holding only the synthetic record-teacher entry e3-rt (ruling R10)."""
    if "rt" not in DIRS:
        d = scratch("rt_registry")
        (d / "e3_rt.json").write_text(json.dumps(RT_ENTRY, indent=2) + "\n", encoding="utf-8")
        DIRS["rt"] = d
    return DIRS["rt"]


# ---------------------------------------------------------------------------------- A: the registry
@case("cm_registry_loads_r1_r2")
def c_registry_loads():
    arms = L.kd_arms.load_arms()
    want = {"e3-r1": ("E3-R1", "E7-R1", "arm", "R1", True, 4), "e3-r2": ("E3-R2", "E7-R2", "arm", "R2", True, 1)}
    got = {n: (s.run_stage, s.e7_stage, s.teacher, s.arm_teacher, s.go_rule, s.cut_place) for n, s in arms.items()
           if n in want}
    common = all(arms[n].stage == "e3" and arms[n].seed == 42 and arms[n].horizon == 80000 and arms[n].K == 1
                 and arms[n].views == 1 and arms[n].targets is None
                 and arms[n].target_construction == {"K": 1, "views": 1} for n in want if n in arms)
    return got == want and common, f"{got} common={common}"


@case("cm_registry_schema")
def c_registry_schema():
    base = json.loads((REPO / "configs" / "kd_arms" / "e3_r1.json").read_text(encoding="utf-8"))
    bad = {"k2": {"K": 2}, "views2_no_targets": {"views": 2}, "record_with_arm_teacher": {"teacher": "record"},
           "go_rule_false_for_arm": {"go_rule": False}, "seed43": {"seed": 43}, "horizon160k": {"horizon": 160000},
           "stage_e2": {"stage": "e2"}, "cut_place3": {"cut_place": 3}, "wrong_suffix": {"arm_teacher": "R2"},
           "extra_key": {"note": "x"}}
    refused = {}
    for label, over in bad.items():
        d = scratch(f"schema_{label}")
        (d / "e3_r1.json").write_text(json.dumps({**base, **over}), encoding="utf-8")
        try:
            L.kd_arms.load_arms(d)
            refused[label] = False
        except L.kd_arms.ArmSpecError:
            refused[label] = True
    d = scratch("schema_stem")
    (d / "e3_rx.json").write_text(json.dumps(base), encoding="utf-8")
    try:
        L.kd_arms.load_arms(d)
        refused["name_not_stem"] = False
    except L.kd_arms.ArmSpecError:
        refused["name_not_stem"] = True
    good = scratch("schema_good")
    (good / "e3_r1.json").write_text(json.dumps(base), encoding="utf-8")
    ok_good = list(L.kd_arms.load_arms(good)) == ["e3-r1"]
    return all(refused.values()) and ok_good, f"refused={refused} good={ok_good}"


@case("cm_registry_tolerates_extra_entries")
def c_registry_extra():
    """The real registry plus one more entry (the synthetic record-teacher arm, unless an entry at its cut place is
    registered already: then the real registry is the case): R1 and R2 load unchanged and entry prints the same line."""
    real = sorted((REPO / "configs" / "kd_arms").glob("*.json"))
    d = scratch("extra_registry")
    for p in real:
        (d / p.name).write_bytes(p.read_bytes())
    places = {s.cut_place for s in L.kd_arms.load_arms().values()}
    added = None
    if RT_ENTRY["cut_place"] not in places:
        (d / "e3_rt.json").write_text(json.dumps(RT_ENTRY, indent=2) + "\n", encoding="utf-8")
        added = "e3-rt"
    arms_real, arms_extra = L.kd_arms.load_arms(), L.kd_arms.load_arms(d)
    same = all(arms_extra[n].sha256 == arms_real[n].sha256 and arms_extra[n].run_stage == arms_real[n].run_stage
               for n in ("e3-r1", "e3-r2"))
    more = len(arms_extra) == len(arms_real) + (1 if added else 0) and len(arms_extra) >= 3
    sc = entry_scenario("extra_registry")
    out_real = entry(sc, "e3-r1")
    with patched(L.pa, "ARM_DIR", d):
        out_extra = entry(sc, "e3-r1")
    return (same and more and out_real[0] == 0 and out_real == out_extra,
            f"added={added} same={same} sizes {len(arms_real)}/{len(arms_extra)} entry {out_real[0]}/{out_extra[0]} "
            f"equal={out_real == out_extra}")


# ---------------------------------------------------------------------------------- B: the trainer
class _Stop(Exception):
    pass


def trainer_row(stage_key: str, arm=None):
    """The REAL train_distill.run() in dry mode up to its training loop (smoke_preflight_distill's method), with the
    arm when given: its one run_meta row, or the reason it has none."""
    td, spd = L.td, L.spd
    from src.distill import FrozenTeacher, MockTeacher
    ck = scratch("meta") / f"{stage_key}_s42_meta"
    saved = (td.build_student, td.build_dataloader, td.cycle, td._git_provenance, td.set_seed)

    def stop(*_a, **_k):
        raise _Stop()
    td.build_student, td.build_dataloader = (lambda *a, **k: spd._Tiny()), (lambda *a, **k: spd._Loader())
    td.cycle, td._git_provenance, td.set_seed = stop, (lambda: ("0" * 40, "smoke_stub")), (lambda s: None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            td.run(stage=td.resolve_stage(stage_key), mode="dry", device="cpu", pretrained=False,
                   teacher=FrozenTeacher(MockTeacher(116)), lambda_logit=1.0, batch_size=2, max_iters=2, val_interval=2,
                   max_val_batches=1, num_workers=0, ckpt_dir_arg=str(ck), grad_clip_norm=None, log_every=1, seed=42,
                   **({} if arm is None else {"arm": arm}))
        return "run() did not reach its training loop"
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001 - the reason is the check's detail
        return f"{type(e).__name__}: {e}"
    finally:
        td.build_student, td.build_dataloader, td.cycle, td._git_provenance, td.set_seed = saved
    lines = (ck / f"{stage_key}_run_meta.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]) if len(lines) == 1 else f"{len(lines)} run_meta lines"


def td_main(argv) -> tuple:
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        try:
            rc = L.td.main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, err.getvalue()


@case("cm_trainer_arm_requires_e3")
def c_trainer_e3():
    rc, err = td_main(["--stage", "e2", "--dry-run", "--arm", "e3-r1", "--lambda-logit", "1"])
    rc_a, err_a = td_main(["--stage", "a", "--dry-run", "--arm", "e3-r1", "--alpha", "50"])
    return rc == 2 and "[arm_stage]" in err and rc_a == 2 and "[arm_stage]" in err_a, f"{rc} {err[-200:]} {rc_a}"


@case("cm_trainer_arm_seed_42_only")
def c_trainer_seed():
    """[arm_seed] through main() in a real run (C1 open item 4): refused before M11, CUDA or any data."""
    rc, err = td_main(["--stage", "e3", "--real-run", "--confirm-real-run", "--arm", "e3-r1", "--seed", "43",
                       "--lambda-logit", "1", "--alpha", "50"])
    spec = spec_of("e3-r1")
    dry_ok = spec.run_error(stage_key="e3", seed=43, horizon=80000, mode="dry") is None
    return rc == 2 and "[arm_seed]" in err and "M11" not in err and dry_ok, f"{rc} {err[-240:]} dry={dry_ok}"


@case("cm_trainer_arm_horizon_80k")
def c_trainer_horizon():
    rc, err = td_main(["--stage", "e3", "--dry-run", "--arm", "e3-r1", "--iterations", "160000"])
    return rc == 2 and "[arm_iterations]" in err, f"{rc} {err[-200:]}"


@case("cm_run_meta_arm_fields")
def c_run_meta_arm():
    row = trainer_row("e3", spec_of("e3-r1"))
    ok = isinstance(row, dict) and row.get("arm") == "E3-R1" and row.get("descriptive") is True \
        and row.get("parent_of_e4_e7") is False
    return ok, row if not isinstance(row, dict) else {k: row.get(k) for k in ("arm", "descriptive", "parent_of_e4_e7")}


@case("cm_run_meta_target_construction_k1_v1")
def c_run_meta_tc():
    row = trainer_row("e3", spec_of("e3-r1"))
    ok = isinstance(row, dict) and list(row)[-1] == "target_construction" \
        and row["target_construction"] == {"K": 1, "views": 1} \
        and list(row)[:-1] == L.pd.run_meta_keys("e3")
    return ok, row if not isinstance(row, dict) else (list(row)[-3:], row.get("target_construction"))


@case("cm_default_run_meta_has_no_arm_key")
def c_run_meta_default():
    row = trainer_row("e3")
    ok = isinstance(row, dict) and list(row) == L.pd.run_meta_keys("e3") and row["arm"] is None \
        and row["descriptive"] is False and row["parent_of_e4_e7"] is True and "target_construction" not in row
    return ok, row if not isinstance(row, dict) else {k: row.get(k, "<absent>") for k in
                                                       ("arm", "descriptive", "parent_of_e4_e7", "target_construction")}


WORKERS: dict = {}


def run_workers(threads: int) -> None:
    """The default E3 run and two E3-R1 runs (2 steps, validation at step 2) on the synthetic set, stub teacher."""
    from scripts.invariance_harness import make_synthetic_dataset, run_worker
    w = DIRS["work"] / "workers"
    w.mkdir()
    data = w / "data"
    make_synthetic_dataset(data)
    try:
        WORKERS["e3"] = run_worker(code_root=REPO, data_root=data, out_dir=w / "w_e3", stage="e3", steps=STEPS,
                                   val_interval=VAL_INTERVAL, threads=threads, timeout=WORKER_TIMEOUT)
    except subprocess.TimeoutExpired:
        WORKERS["e3"] = {"returncode": None, "summary": None, "log": "timed out"}
    for label in ("arm1", "arm2"):
        out = w / f"w_{label}"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": str(threads), "PLANTSEG_DATA_ROOT": str(data)})
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--internal", "arm_worker", "--arm", "e3-r1",
               "--code-root", str(REPO), "--data-root", str(data), "--out", str(out), "--stage", "e3", "--steps",
               str(STEPS), "--val-interval", str(VAL_INTERVAL), "--threads", str(threads)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT, cwd=str(w), env=env)
            rc, text = proc.returncode, proc.stdout + "\n--- stderr ---\n" + proc.stderr
        except subprocess.TimeoutExpired:
            rc, text = None, "timed out"
        (w / f"w_{label}.worker.log").write_text(text, encoding="utf-8")
        sp = out / "run_summary.json"
        WORKERS[label] = {"returncode": rc, "summary": json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None,
                          "log": str(w / f"w_{label}.worker.log"), "out": out}
    WORKERS["e3"]["out"] = w / "w_e3"


def ran(r: dict) -> bool:
    sm = (r or {}).get("summary") or {}
    return (r or {}).get("returncode") == 0 and str(sm.get("trainer_result", "")).startswith("RESULT: PASS") \
        and sm.get("n_step_records") == STEPS


@case("cm_payload_arm_keys_only_in_arm_runs")
def c_payload():
    import torch
    e3, arm = WORKERS.get("e3"), WORKERS.get("arm1")
    if not (ran(e3) and ran(arm)):
        return False, f"runs: e3 {ran(e3)} arm {ran(arm)} logs {e3 and e3.get('log')} {arm and arm.get('log')}"
    pe, pa_ = e3["summary"]["checkpoints"], arm["summary"]["checkpoints"]
    keys_ok = bool(pe) and len(pe) == len(pa_) and all(
        "arm" not in c1["keys"] and "target_construction" not in c1["keys"]
        and set(c2["keys"]) == set(c1["keys"]) | {"arm", "target_construction"} for c1, c2 in zip(pe, pa_))
    values = []
    for p in sorted((arm["out"]).glob("ckpt_e3*/*_student_best_iter*.pt")):
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        values.append((ck.get("arm"), ck.get("target_construction")))
    vals_ok = bool(values) and all(v == ("E3-R1", {"K": 1, "views": 1}) for v in values)
    return keys_ok and vals_ok, f"keys_ok={keys_ok} values={values}"


@case("cm_r1_targets_none_loss_bitwise_e3")
def c_bitwise():
    from scripts.invariance_harness import compare_step_dirs
    e3, arm = WORKERS.get("e3"), WORKERS.get("arm1")
    if not (ran(e3) and ran(arm)):
        return False, "a run did not finish"
    cmp = compare_step_dirs(e3["out"], arm["out"])
    return cmp["identical"] and cmp["n_old"] == STEPS, f"{cmp['n_old']}/{cmp['n_new']} {cmp['mismatches'][:1]}"


@case("dt_two_dry_runs_identical")
def c_dt_runs():
    from scripts.invariance_harness import compare_step_dirs
    a, b = WORKERS.get("arm1"), WORKERS.get("arm2")
    if not (ran(a) and ran(b)):
        return False, "a run did not finish"
    cmp = compare_step_dirs(a["out"], b["out"])
    return cmp["identical"] and cmp["n_old"] == STEPS, f"{cmp['n_old']}/{cmp['n_new']} {cmp['mismatches'][:1]}"


# ------------------------------------------------------------------------- scenarios (scratch repositories)
def dl_rows(pin: str, *, approval=True, approval_status="RECORDED", approval_marker=True, arms_pin=True, token=True,
            pk=None, go_row=None) -> str:
    rows = []
    if approval:
        text = "AM-18 written approval" if approval_marker else "an acknowledgement of AM-18"
        rows.append(f"| DL-101 | {approval_status} (synthetic) | {text}, as reported (synthetic) | - | yes |")
    if arms_pin:
        rows.append(f"| DL-106 | RECORDED (synthetic) | Arms pin: P_K = {pk or pin}" + ("; AM-20 item 3: open" if token
                                                                                       else "") + " (synthetic) | - | no |")
    if go_row is not None:
        rows.append(f"| DL-110 | RECORDED (synthetic) | {go_row} | - | no |")
    return "".join(r + "\n" for r in rows)


def teacher_x() -> Path:
    if "x" not in DIRS:
        d = scratch("teacher")
        (d / X_NAME).write_bytes(X_BYTES)
        DIRS["x"] = d / X_NAME
    return DIRS["x"]


def arm_record_files(spec, *, status="complete", verdict="go", tr_arm=None, go_arm=None, select_sha=None,
                     select_name=X_NAME, go_ck=None) -> tuple:
    """(files, teacher record sha256, go-rule sha256) of the arm teacher's records at reports/arms."""
    pa = L.pa
    tr = {"format": pa.TEACHER_RECORD_FORMAT, "arm": tr_arm or spec.arm_teacher, "status": status,
          "selection": {"checkpoint": {"name": select_name, "sha256": select_sha or sha(X_BYTES), "bytes": len(X_BYTES)}}}
    tr_b = (json.dumps(tr, indent=2, sort_keys=True) + "\n").encode("utf-8")
    go = {"format": pa.GO_RULE_FORMAT, "arm": go_arm or spec.arm_teacher, "verdict": verdict,
          "checkpoint": {"sha256": go_ck or tr["selection"]["checkpoint"]["sha256"]},
          "teacher_record": {"sha256": sha(tr_b)}}
    go_b = (json.dumps(go, indent=2, sort_keys=True) + "\n").encode("utf-8")
    files = {pa.TEACHER_RECORD_REL.format(arm=spec.arm_teacher): tr_b, pa.GO_RULE_REL.format(arm=spec.arm_teacher): go_b}
    return files, sha(tr_b), sha(go_b)


class ArmCase:
    """One gate scenario: an alpha sweep of the selection smoke (alpha 50 launched; the lambda selection and the
    band; the alpha selection, or the alpha-cut decision record), then the arm's records and rows and its own launch
    line, committed as the records commit H; the gate's KD stages run in-process on a context like the gate's."""

    def __init__(self, label: str, spec, *, alpha_cut: bool = False):
        sss, ss, spd = L.sss, L.ss, L.spd
        self.spec = spec
        if alpha_cut:
            self.sw = spd.alpha_gate_sweep(label)
            rc, log, self.alpha_record_sha = self.sw.record(sss.NOW_A)
            assert rc == 0, sss.result_line(log)
        else:
            self.sw = spd.alpha_gate_sweep(label, files={**sss.alpha_files(),
                                                         ss.ALPHA_SELECTION_REL: spd.alpha_selection(50.0)})
        self.pin = self.sw.pin
        self.files: dict = {}
        self.dl_extra = ""

    def line(self, *, lam=1.0, alpha=50.0, attempt=1, report=None, code_pin=None, spec=None) -> dict:
        spec = spec or self.spec
        rid = L.pa.run_id_of(spec, lam, alpha, attempt)
        return L.pa.arm_launch_line(L.ss, spec, lam=lam, alpha=alpha, attempt=attempt, schedule=self.sw.schedule,
                                    code_pin=code_pin or self.pin, ckpt_dir=str(DIRS["ck"] / rid), am8a_report=report)

    def commit_h(self, extra_events=()) -> str:
        ss = L.ss
        dl = (self.sw.repo / ss.DECISION_LOG_REL).read_text(encoding="utf-8") + self.dl_extra
        self.sw.events.extend(extra_events)
        return self.sw.commit("the arm's records and launch line", {**self.files, ss.DECISION_LOG_REL: dl})

    def ctx(self, line: dict, *, now=None, teacher_ckpt=None, approval_dl="DL-101", arms_pin_dl="DL-106",
            go_rule_dl="DL-110", report=None, previous_runs=(), child=None, logged=True) -> dict:
        sss, pa = L.sss, L.pa
        h = self.commit_h([dict(line)] if logged else [])
        sss.git(self.sw.repo, "checkout", "-q", "--detach", self.pin)
        folder = scratch("records")
        with L.spd.git_root(self.sw.repo):
            rc, out = L.spd.quiet(pa.cmd_records, SimpleNamespace(records_commit=h, out=str(folder)))
        assert rc == 0, out
        spec = self.spec
        return {"arm": spec.name, "profile": pa.PROFILE, "stage": spec.stage, "seed": spec.seed,
                "ckpt_dir": line["ckpt_dir"], "evidence": str(DIRS["ev"]), "expect_head": self.pin, "records_commit": h,
                "records": str(folder), "teacher_ckpt": str(teacher_ckpt or teacher_x()), "clock_offset": 0.0,
                "record": None, "rehearsal": False, "lambda_logit": line["lambda_logit"], "alpha": line["alpha_cwd"],
                "attempt": line["attempt"], "am8a_report": report, "previous_runs": [str(p) for p in previous_runs],
                "decision_records": [], "approval_dl": approval_dl, "arms_pin_dl": arms_pin_dl,
                "go_rule_dl": go_rule_dl if spec.go_rule else None, "reference_telemetry": None, "reference_smokes": [],
                "git_root": self.sw.repo, "now": sss.NOW_A if now is None else now, "warnings": [],
                "child_json": child or child_stub(), "child_run": None, "root": DIRS["data"], "spec": spec,
                "ident": {"sweep": None, "value": None, "horizon": spec.horizon}, "run_id": line["run_id"]}


def standard_case(label: str, spec=None, *, alpha_cut=False, rows=None, records=None, **ctx_kw) -> dict:
    """An arm scenario with every record in place unless `rows` / `records` say otherwise; returns its context."""
    spec = spec or spec_of("e3-r1")
    c = ArmCase(label, spec, alpha_cut=alpha_cut)
    go_row = None
    if spec.teacher == "arm":
        files, tr_sha, go_sha = arm_record_files(spec, **(records or {}))
        c.files.update(files)
        go_row = f"{spec.arm_teacher} go rule: go; arm_go_rule sha256 {go_sha}; teacher record sha256 {tr_sha} (synthetic)"
    rkw = dict(rows or {})
    if "go_row" not in rkw:
        rkw["go_row"] = go_row
    c.dl_extra = dl_rows(c.pin, **rkw)
    return c.ctx(c.line(), **ctx_kw)


KD = ("records", "selection_inputs", "launch_order")


def run_stages(ctx: dict, names=KD) -> str:
    """The arm gate's stages on `ctx`, in order: PASS, or '<stage>:<code>' of the first that does not pass."""
    fns = dict(L.pa.STAGES)
    for name in names:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                status, _detail, _rec = fns[name](ctx)
        except L.pd.GateError as e:
            return f"{name}:{e.code}"
        except Exception as e:  # noqa: BLE001 - reported as such
            return f"{name}:crash:{type(e).__name__}: {e}"
        if status != "PASS":
            return f"{name}:{status}"
    return "PASS"


def child_record(path, sha256, nbytes) -> dict:
    """The teacher child's record for a teacher file that loads with DL-88's values and its own checkpoint fields."""
    return {"ok": True, "sha256": sha256, "bytes": nbytes, "params": L.pd.TEACHER_PARAMS, "trainable": 0,
            "training_modules": [], "state_entries": L.pd.TEACHER_STATE_ENTRIES,
            "provenance": {"builder": L.pd.TEACHER_BUILDER, "ckpt_path": str(path), "ckpt_sha256": sha256,
                           "ckpt_bytes": nbytes, "expected_sha256": sha256,
                           "config_path": str(REPO / "configs" / "teacher" / "segnext_mscan_b.py"),
                           **L.pd.TEACHER_VALUES}}


def child_stub(records=None):
    """ctx['child_json'] for the teacher child: the record of the file named, as the child would load it."""
    def stub(argv, tag, timeout, env=None):
        t = str(argv[argv.index("--teacher-ckpt") + 1])
        if records and t in records:
            return 0, records[t], ""
        p = Path(t)
        if p.is_file():
            data = p.read_bytes()
            return 0, child_record(p, sha(data), len(data)), ""
        return 0, child_record(p, L.pd.TEACHER_SHA256, L.pd.TEACHER_BYTES), ""
    return stub


# ---------------------------------------------------------------------------------------- C: entry
def entry_scenario(label: str, *, token=True, pk=None, extra_lines=(), files=None) -> SimpleNamespace:
    """A repository at the tip (no checkout of the pin): the alpha sweep, the approval and arms-pin rows, and
    `extra_lines` appended to the launch log, all committed at HEAD."""
    c = ArmCase(f"entry_{label}", spec_of("e3-r1"))
    c.dl_extra = dl_rows(c.pin, token=token, pk=pk)
    c.files.update(files or {})
    c.commit_h(list(extra_lines))
    return SimpleNamespace(case=c, repo=c.sw.repo, pin=c.pin)


def entry(sc, arm: str, *, lam=1.0, alpha=50.0, ckpt_dir=None, code_pin=None, offset=0.0, report=None,
          previous_runs=(), attempt=1) -> tuple:
    """pa.cmd_entry in-process against the scenario's repository: (rc or refusal code, stdout)."""
    spec = spec_of(arm, L.pa.ARM_DIR)
    rid = L.pa.run_id_of(spec, lam, alpha, attempt)
    args = SimpleNamespace(arm=arm, lambda_logit=lam, alpha=alpha, ckpt_dir=ckpt_dir or f"/workspace/kd_runs/{rid}",
                           code_pin=code_pin or sc.pin, arms_pin_dl="DL-106", clock_offset_seconds=offset,
                           am8a_report=report, previous_run=[str(p) for p in previous_runs] or None)
    with L.spd.git_root(sc.repo):
        return L.spd.quiet(L.pa.cmd_entry, args)


def printed_line(out: str) -> str | None:
    head = "LAUNCH LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it before the gate):\n"
    return out.split(head)[1].splitlines()[0] if head in out else None


@case("cm_entry_arm_line_bytes")
def c_entry_bytes():
    sc = entry_scenario("bytes")
    rc, out = entry(sc, "e3-r1")
    golden = ('{"format": "kd_launch_log/1", "event": "launch", "run_id": "e3-r1_s42_lambda1_alpha50_80k_a1", "stage": '
              '"E3-R1", "seed": 42, "horizon": 80000, "sweep": null, "value": null, "lambda_logit": 1.0, "alpha_cwd": '
              f'50.0, "attempt": 1, "schedule": "T", "code_pin": "{sc.pin}", "ckpt_dir": '
              '"/workspace/kd_runs/e3-r1_s42_lambda1_alpha50_80k_a1", "am8a_report": null}')
    return rc == 0 and printed_line(out) == golden, f"rc={rc} {printed_line(out)!r}"


@case("cm_arm_line_parsed_by_launchlog")
def c_line_parsed():
    ss = L.ss
    sc = entry_scenario("parsed")
    rc, out = entry(sc, "e3-r1")
    line = printed_line(out)
    data = (sc.repo / ss.LAUNCH_LOG_REL).read_bytes() + (line or "").encode("utf-8") + b"\n"
    try:
        log = ss.LaunchLog(data)
        rid = "e3-r1_s42_lambda1_alpha50_80k_a1"
        e3 = [r for r in log.order if log.launch[r]["stage"] == "E3"]
        ok = rc == 0 and rid in log.launch and log.launch[rid]["attempt"] == 1 and bool(e3) \
            and L.pd.group_of(log.launch[rid]) != L.pd.group_of(log.launch[e3[0]])
        return ok, f"rc={rc} attempt={log.launch.get(rid, {}).get('attempt')} e3 lines {e3}"
    except ss.SelectionRefused as e:
        return False, f"{e.code}: {e}"


@case("cm_arm_line_not_alpha_candidate")
def c_not_candidate():
    """An E3 seed-42 alpha-50 line and the arm's line with the same lambda and alpha: the alpha sweep's attempts hold
    the E3 line only (sweep_select.sweep_attempts skips a line whose stage is not E3 and whose sweep is null)."""
    ss, pa = L.ss, L.pa
    spec = spec_of("e3-r1")
    e3 = L.sss.launch_line("e3_s42_alpha50_a1", "E3", 50.0, 1.0, 50.0, 1, "c" * 40)
    arm = pa.arm_launch_line(ss, spec, lam=1.0, alpha=50.0, attempt=1, schedule="T", code_pin="c" * 40,
                             ckpt_dir="/workspace/kd_runs/" + pa.run_id_of(spec, 1.0, 50.0, 1), am8a_report=None)
    try:
        log = ss.LaunchLog(L.sss.launch_log_text([e3, arm]))
        atts = ss.sweep_attempts(log, L.sss.ALPHA, "alpha_cwd", {"in_force": "T"})
        ids = sorted(a.run_id for v in atts for a in atts[v])
        return ids == ["e3_s42_alpha50_a1"] and arm["sweep"] is None and arm["stage"] == "E3-R1", ids
    except ss.SelectionRefused as e:
        return False, f"{e.code}: {e}"


@case("cm_launch_note_cut_place")
def c_cut_place():
    sc = entry_scenario("cut_place")
    rc1, out1 = entry(sc, "e3-r1")
    rc2, out2 = entry(sc, "e3-r2")
    return (rc1 == 0 and rc2 == 0 and "cut-order place 4 (AM-18 item 7(b))" in out1
            and "cut-order place 1 (AM-18 item 7(b))" in out2 and "reverse of the cut order" in out1,
            f"{rc1} {rc2} {out1[-300:]}")


@case("cm_entry_parses_log_with_launchlog")
def c_entry_parses():
    """entry prints a line only after the would-be log parses (audit F39): a line built wrong (here its attempt)
    is refused as launch_log_format before anything is printed; the printed line extends the committed log."""
    pa, ss = L.pa, L.ss
    sc = entry_scenario("parses")
    rc, out = entry(sc, "e3-r1")
    data = (sc.repo / ss.LAUNCH_LOG_REL).read_bytes() + (printed_line(out) or "").encode("utf-8") + b"\n"
    try:
        ss.LaunchLog(data)
        extends = True
    except ss.SelectionRefused:
        extends = False
    orig = pa.arm_launch_line

    def wrong_attempt(ss_, spec, **kw):
        line = orig(ss_, spec, **kw)
        line["attempt"] += 1
        return line
    with patched(pa, "arm_launch_line", wrong_attempt):
        rc_bad, out_bad = entry(sc, "e3-r1")
    return (rc == 0 and extends and rc_bad == "launch_log_format" and printed_line(out_bad) is None,
            f"rc={rc} extends={extends} bad={rc_bad}")


@case("cm_clock_offset_recorded")
def c_clock():
    sc = entry_scenario("clock")
    rc, out = entry(sc, "e3-r1", offset=12.5)
    rc_far, _ = entry(sc, "e3-r1", offset=61.0)
    rc_nan, _ = entry(sc, "e3-r1", offset=float("nan"))
    ctx = standard_case("clock_gate")
    ctx["clock_offset"] = -7.25
    rec = L.pa.gate_record(ctx, [], "NO-GO")
    return (rc == 0 and "clock offset +12.5 s" in out and rc_far == "clock_offset" and rc_nan == "clock_offset"
            and rec.get("clock_offset") == -7.25 and rec.get("format") == L.pa.GATE_RECORD_FORMAT,
            f"{rc} {rc_far} {rc_nan} {rec.get('clock_offset')}")


@case("dt_entry_twice_identical_bytes")
def c_dt_entry():
    sc = entry_scenario("twice")
    a, b = entry(sc, "e3-r1"), entry(sc, "e3-r1")
    return a[0] == 0 and a == b, f"{a[0]} {b[0]} equal={a == b}"


@case("rf_test_path_entry")
def c_rf_entry():
    bad = DIRS["work"] / "plantseg_test_root" / "e3-r1_s42_lambda1_alpha50_80k_a1"
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(err):
        rc = L.pa.main(["entry", "--arm", "e3-r1", "--lambda-logit", "1", "--alpha", "50", "--ckpt-dir", str(bad),
                        "--code-pin", "c" * 40, "--arms-pin-dl", "DL-106", "--clock-offset-seconds", "0"])
    return (rc == 2 and "RESULT: REFUSED (test_path)" in out.getvalue() and not bad.parent.exists(),
            f"rc={rc} {out.getvalue()[-200:]}")


def repeat_case(label: str, *, attempt1: str, report: bool = True, previous_run: bool = True) -> tuple:
    """The arm's attempt 2 after attempt 1 ended as `attempt1`: "diverged" (launched; a valid AM-7a divergence
    record), "stopped" (launched; a pod stop with no trainer record) or "unresolved" (a launch line only); with or
    without its AM-8a report and --previous-run: (entry's result, the gate's launch_order)."""
    sss, ss, pa = L.sss, L.ss, L.pa
    spec = spec_of("e3-r1")
    diverged = attempt1 == "diverged"
    c = ArmCase(f"div_{label}", spec)
    files, tr_sha, go_sha = arm_record_files(spec)
    c.files.update(files)
    c.dl_extra = dl_rows(c.pin, go_row=f"R1 go rule: go; {go_sha}; {tr_sha} (synthetic)")
    rid1 = pa.run_id_of(spec, 1.0, 50.0, 1)
    runs = scratch(f"div_runs_{label}")
    d = sss.make_run(runs, "E3", 50.0, 0.4, name=rid1, finished=False, run_end=False, with_best=False, with_ckpt=False,
                     abort={"it": 5000, "rule": "AM-7(a)"} if diverged else None, nonfinite=diverged,
                     launch=sss.BASE_A + 7200.0)
    events = [c.line()]
    if attempt1 != "unresolved":
        events.append({"event": "launched", "run_id": rid1, "run_meta_sha256": ss.sha256_file(d / "e3_run_meta.jsonl"),
                       "launch_time_utc": sss.BASE_A + 7200.0, "git_head": c.pin, "records_commit": "e" * 40,
                       "decision_date": None, "launch_order_position": None})
    if attempt1 == "stopped":
        events.append({"event": "stopped", "run_id": rid1,
                       "report": c.sw.report(f"reports/stops/{rid1}.md", f"{rid1}: the pod was lost (synthetic)\n")})
    rep_rel = rep = None
    if report:
        rep_rel = f"reports/am8a/{pa.run_id_of(spec, 1.0, 50.0, 2)}.md"
        rep = c.sw.report(rep_rel, "AM-8a report before attempt 2 (synthetic)\n")
    prev = [d] if previous_run else []
    # entry, at the tip (attempt 1's lines and the report committed)
    c.commit_h(events)
    c.dl_extra = ""
    sc = SimpleNamespace(repo=c.sw.repo, pin=c.pin)
    got_entry = entry(sc, "e3-r1", attempt=2, report=rep_rel, previous_runs=prev)[0]
    # the gate, for attempt 2's committed line
    line2 = c.line(attempt=2, report=rep)
    ctx = c.ctx(line2, report=rep_rel, previous_runs=prev)
    got_gate = run_stages(ctx)
    return got_entry, got_gate


def divergence_case(label: str, *, diverged: bool) -> tuple:
    """Attempt 1 launched and ended in a valid AM-7a divergence record (or in a pod stop with no trainer record);
    attempt 2 with its AM-8a report and --previous-run."""
    return repeat_case(label, attempt1="diverged" if diverged else "stopped")


def dry_run_stub(seen: list, *, arm_line: bool = True):
    """ctx['child_run'] for pa's kd_dry_run: records the argv and prints what the trainer's dry run prints."""
    def run(argv, timeout):
        argv = [str(a) for a in argv]
        seen.append(argv)
        out = [f"[ckpt] dir={argv[argv.index('--ckpt-dir') + 1]} (verified OUTSIDE repo)"]
        if arm_line:
            out.append("[arm] E3-R1 (dry run stub)")
        return SimpleNamespace(returncode=0, stdout="\n".join(out + ["RESULT: PASS"]) + "\n", stderr="")
    return run


@case("rf_arm_repeat_after_divergence")
def c_rf_divergence():
    e_div, g_div = divergence_case("diverged", diverged=True)
    e_stop, g_stop = divergence_case("stopped", diverged=False)
    ok = (e_div == "arm_rerun_after_divergence" and g_div == "launch_order:arm_rerun_after_divergence"
          and e_stop == 0 and g_stop == "PASS")
    return ok, f"diverged: entry {e_div} gate {g_div}; stopped: entry {e_stop} gate {g_stop}"


# ------------------------------------------------------------------------------------------ D: the gate
@case("cm_gate_stage_list")
def c_stage_list():
    pa, pd, E1G = L.pa, L.pd, L.pd.E1G
    names = [n for n, _ in pa.STAGES]
    want = [*pd.STAGE_NAMES_ORDER[:-1], "arm_cut_arithmetic", "repo_unchanged"]
    kd, e1, ours = dict(pd.STAGES), dict(E1G.STAGES), dict(pa.STAGES)
    reused_e1 = ("data_isolation", "repo_state", "class_weights", "smoke_loss", "cuda", "imagenet_backbone",
                 "smoke_loader_seed", "smoke_dataloader", "seed_sequence_R5", "repo_unchanged")
    reused_pd = ("selection_inputs", "kd_image")
    arm_own = ("arguments", "module_provenance", "records", "launch_order", "teacher_identity", "kd_dry_run",
               "arm_cut_arithmetic")
    ok = (names == want and len(names) == 19 and all(ours[n] is e1[n] for n in reused_e1)
          and all(ours[n] is kd[n] for n in reused_pd) and all(ours[n] is not kd.get(n) for n in arm_own))
    # pa's own launch_order keeps pd's repeat rules (plan §2 entry 17), on both sides (review KDG-2) ...
    rep = {"no_report": repeat_case("noreport", attempt1="stopped", report=False),
           "unresolved": repeat_case("unresolved", attempt1="unresolved"),
           "no_previous_run": repeat_case("noprev", attempt1="diverged", previous_run=False)}
    want_rep = {"no_report": ("repeat_report_missing", "launch_order:repeat_report_missing"),
                "unresolved": ("previous_attempt_unresolved", "launch_order:previous_attempt_unresolved"),
                "no_previous_run": ("previous_run_missing", "launch_order:previous_run_missing")}
    # ... and its own kd_dry_run runs pd's dry run as the arm, with the arm teacher (plan §2 entry 4)
    ctx = standard_case("dry_run")
    pre = run_stages(ctx, ("records",))
    seen: list = []
    ctx["child_run"] = dry_run_stub(seen)
    dr_ok = run_stages(ctx, ("kd_dry_run",))
    argv = seen[-1] if seen else []
    ctx["child_run"] = dry_run_stub([], arm_line=False)
    dr_no_arm = run_stages(ctx, ("kd_dry_run",))
    dry = (pre == "PASS" and dr_ok == "PASS" and "--arm" in argv and argv[argv.index("--arm") + 1] == "e3-r1"
           and argv[argv.index("--teacher-ckpt-sha256") + 1] == sha(X_BYTES)
           and dr_no_arm == "kd_dry_run:kd_dry_run_failed")
    ok = ok and rep == want_rep and dry
    return ok, f"{names}; repeats {rep}; dry run {dr_ok}/{dr_no_arm} argv {argv[-6:]}"


@case("cm_gate_teacher_identity_arm_values")
def c_identity():
    pa, pd = L.pa, L.pd
    spec = spec_of("e3-r1")
    x = teacher_x()
    exp = {"name": X_NAME, "sha256": sha(X_BYTES), "bytes": len(X_BYTES)}
    good = pa.teacher_identity_problems(child_record(x, exp["sha256"], exp["bytes"]), str(x), spec, exp)
    rec_t = Path("/workspace/teacher") / pd.TEACHER_FILE
    record = pa.teacher_identity_problems(child_record(rec_t, pd.TEACHER_SHA256, pd.TEACHER_BYTES), str(rec_t), spec,
                                          {"name": pd.TEACHER_FILE, "sha256": pd.TEACHER_SHA256, "bytes": pd.TEACHER_BYTES})
    wrong = pa.teacher_identity_problems(child_record(x, "0" * 64, exp["bytes"]), str(x), spec, exp)
    got_record_selected = run_stages(standard_case("identity_record_selected", records={
        "select_sha": pd.TEACHER_SHA256, "select_name": X_NAME}), ("records",))
    ctx = standard_case("identity_stage")
    stage_ok = run_stages(ctx, ("records", "teacher_identity"))
    ctx_r = standard_case("identity_stage_record")
    ctx_r["child_json"] = child_stub({str(x): child_record(x, pd.TEACHER_SHA256, pd.TEACHER_BYTES)})
    stage_bad = run_stages(ctx_r, ("records", "teacher_identity"))
    ok = (good == [] and any("teacher of record" in p for p in record) and bool(wrong)
          and got_record_selected == "records:teacher_is_record" and stage_ok == "PASS"
          and stage_bad == "teacher_identity:teacher_identity_mismatch")
    return ok, f"good={good} record={record[:2]} wrong={wrong[:1]} selected={got_record_selected} {stage_ok} {stage_bad}"


@case("cm_gate_go_rule_go_required")
def c_go_rule():
    got_go = run_stages(standard_case("go_go"))
    got_no = run_stages(standard_case("go_no_go", records={"verdict": "no_go"}))
    got_ne = run_stages(standard_case("go_not_evaluable", records={"verdict": "not_evaluable"}))
    # the go-rule record is bound to X and to the teacher record (plan §3(v), (iv) check 4; review KDG-3)
    got_ck = run_stages(standard_case("go_other_ckpt", records={"go_ck": "0" * 64}))
    got_arm = run_stages(standard_case("go_other_arm", records={"go_arm": "R2"}))
    ok = (got_go == "PASS" and got_no == "records:go_rule_not_go" and got_ne == "records:go_rule_not_go"
          and got_ck == "records:go_rule_record_invalid" and got_arm == "records:go_rule_record_invalid")
    return ok, f"go={got_go} no_go={got_no} not_evaluable={got_ne} other_ckpt={got_ck} other_arm={got_arm}"


@case("cm_gate_approval_row_required")
def c_approval():
    got = {"present": run_stages(standard_case("approval_present")),
           "absent": run_stages(standard_case("approval_absent", rows={"approval": False})),
           "decided": run_stages(standard_case("approval_decided", rows={"approval_status": "DECIDED"})),
           "no_marker": run_stages(standard_case("approval_no_marker", rows={"approval_marker": False}))}
    ok = got["present"] == "PASS" and all(got[k] == "records:approval_missing" for k in ("absent", "decided", "no_marker"))
    return ok, got


@case("cm_gate_arms_pin_rule")
def c_arms_pin():
    got = {"pk_is_pin": run_stages(standard_case("pin_ok")),
           "pk_other": run_stages(standard_case("pin_other", rows={"pk": "f" * 40})),
           "no_token": run_stages(standard_case("pin_no_token", rows={"token": False}))}
    spec1, spec2 = spec_of("e3-r1"), spec_of("e3-r2")
    c = ArmCase("pin_other_arm_line", spec1)
    files, tr_sha, go_sha = arm_record_files(spec1)
    c.files.update(files)
    c.dl_extra = dl_rows(c.pin, go_row=f"R1 go rule: go; {go_sha}; {tr_sha} (synthetic)")
    other = c.line(spec=spec2, code_pin="f" * 40)
    c.sw.events.append(other)
    other_nl = {"event": "not_launched", "run_id": other["run_id"], "reason": "synthetic",
                "evidence": c.sw.report(f"reports/not_launched/{other['run_id']}.txt", "synthetic\n")}
    c.sw.events.append(other_nl)
    got["other_arm_line_other_pin"] = run_stages(c.ctx(c.line()))
    sc = entry_scenario("pin_entry")
    got["entry_code_pin_other"] = entry(sc, "e3-r1", code_pin=L.sss.template()[1])[0]
    ok = (got["pk_is_pin"] == "PASS" and got["pk_other"] == "records:arms_pin_mismatch"
          and got["no_token"] == "records:approval_recheck_missing"
          and got["other_arm_line_other_pin"] == "launch_order:code_pin_mismatch"
          and got["entry_code_pin_other"] == "code_pin_mismatch")
    return ok, got


@case("cm_gate_lambda_alpha_inherited")
def c_inherited():
    ok_ctx = standard_case("inherit_ok")
    got_ok = run_stages(ok_ctx)
    sel = ok_ctx.get("selection_files") or {}
    c = ArmCase("inherit_lambda", spec_of("e3-r1"))
    files, tr_sha, go_sha = arm_record_files(c.spec)
    c.files.update(files)
    c.dl_extra = dl_rows(c.pin, go_row=f"R1 go rule: go; {go_sha}; {tr_sha} (synthetic)")
    got_lam = run_stages(c.ctx(c.line(lam=0.5)))
    c = ArmCase("inherit_alpha", spec_of("e3-r1"))
    c.files.update(files)
    c.dl_extra = dl_rows(c.pin, go_row=f"R1 go rule: go; {go_sha}; {tr_sha} (synthetic)")
    got_alpha = run_stages(c.ctx(c.line(alpha=25.0)))
    ss = L.ss
    ok = (got_ok == "PASS" and (sel.get("lambda_logit") or {}).get("rel") == ss.LAMBDA_SELECTION_REL
          and (sel.get("alpha_cwd") or {}).get("rel") == ss.ALPHA_SELECTION_REL
          and got_lam == "selection_inputs:lambda_not_selected" and got_alpha == "selection_inputs:alpha_not_selected")
    return ok, f"ok={got_ok} sel={sel} lam={got_lam} alpha={got_alpha}"


def trainer_argv(block: str) -> list:
    line = next(ln for ln in block.splitlines() if " nohup " in ln)
    toks = shlex.split(line)
    return toks[toks.index("src/training/train_distill.py") + 1:toks.index(">")]


@case("cm_gate_launch_block_has_arm")
def c_block():
    pa, pd = L.pa, L.pd
    ctx = standard_case("block")
    got = run_stages(ctx)
    if got != "PASS":
        return False, got
    a = trainer_argv(pa.build_launch_block(ctx))
    pairs = {"--stage": "e3", "--seed": "42", "--arm": "e3-r1", "--lambda-logit": "1", "--alpha": "50",
             "--teacher-ckpt": str(teacher_x()), "--teacher-ckpt-sha256": sha(X_BYTES), "--num-workers": "12",
             "--log-every": "50", "--records-commit": ctx["records_commit"], "--ckpt-dir": ctx["ckpt_dir"]}
    ok = (all(k in a and a[a.index(k) + 1] == v for k, v in pairs.items())
          and "--real-run" in a and "--confirm-real-run" in a and not any(t in a for t in pd.NEVER_PASSED)
          and "--teacher-config" not in a and a[a.index("--lambda-selection") + 1].endswith(L.ss.LAMBDA_SELECTION_REL)
          and a[a.index("--alpha-selection") + 1].endswith(L.ss.ALPHA_SELECTION_REL))
    return ok, a


@case("cm_alpha_cut_record_accepted")
def c_alpha_cut():
    """Ruling R8 (audit F25): with the alpha sweep cut, the committed alpha decision record stands in for
    alpha_selection.json (pd's PL-31 path), and the launch block hands the trainer its records copy."""
    pa, ss = L.pa, L.ss
    ctx = standard_case("alpha_cut", alpha_cut=True)
    got = run_stages(ctx)
    sel = (ctx.get("selection_files") or {}).get("alpha_cwd") or {}
    block_ok = False
    if got == "PASS":
        a = trainer_argv(pa.build_launch_block(ctx))
        block_ok = a[a.index("--alpha-selection") + 1].endswith(ss.DECISION_RECORD_REL["alpha_cwd"])
    ok = got == "PASS" and sel.get("rel") == ss.DECISION_RECORD_REL["alpha_cwd"] and block_ok
    return ok, f"{got} {sel} block={block_ok}"


@case("cm_gate_record_teacher_arm")
def c_record_teacher():
    """A record-teacher arm (teacher "record"; ruling R10): the teacher of record's identity (DL-88 values), no go-rule
    row, the approval and arms-pin rows still required; the launch block names the arm and the teacher of record."""
    pa, pd = L.pa, L.pd
    with patched(pa, "ARM_DIR", rt_dir()):
        spec = spec_of("e3-rt", rt_dir())
        rec_t = DIRS["work"] / "record_teacher" / pd.TEACHER_FILE           # never created
        ctx = standard_case("record_teacher", spec, teacher_ckpt=rec_t, go_rule_dl=None)
        got = run_stages(ctx, ("records", "selection_inputs", "launch_order", "teacher_identity"))
        a = trainer_argv(pa.build_launch_block(ctx)) if got == "PASS" else []
    ok = (got == "PASS" and ctx.get("teacher_expect") == {"name": pd.TEACHER_FILE, "sha256": pd.TEACHER_SHA256,
                                                           "bytes": pd.TEACHER_BYTES}
          and ctx.get("go_rule") is None and ctx.get("teacher_record") is None
          and a[a.index("--arm") + 1] == "e3-rt" and a[a.index("--teacher-ckpt-sha256") + 1] == pd.TEACHER_SHA256)
    return ok, f"{got} {ctx.get('teacher_expect')} {a[:12]}"


@case("rf_record_teacher_arm_without_approval")
def c_rf_rt_approval():
    with patched(L.pa, "ARM_DIR", rt_dir()):
        spec = spec_of("e3-rt", rt_dir())
        rec_t = DIRS["work"] / "record_teacher" / L.pd.TEACHER_FILE
        got = run_stages(standard_case("rt_no_approval", spec, rows={"approval": False}, teacher_ckpt=rec_t,
                                       go_rule_dl=None))
    return got == "records:approval_missing", got


def arguments(ctx: dict) -> str:
    """The arm gate's arguments stage on `ctx` (E1's part reads PLANTSEG_DATA_ROOT and PYTHONPATH, set by main())."""
    return run_stages(ctx, ("arguments",))


@case("rf_record_teacher_arm_with_arm_teacher")
def c_rf_rt_arm_teacher():
    """A record-teacher arm given an arm teacher's file is refused at the arguments (its name) and by the identity
    check (its sha256): AM-18 item 3(b)'s W distils from the teacher of record."""
    pa = L.pa
    with patched(pa, "ARM_DIR", rt_dir()):
        spec = spec_of("e3-rt", rt_dir())
        ctx = standard_case("rt_arm_teacher", spec, teacher_ckpt=teacher_x(), go_rule_dl=None)
        got_args = arguments(dict(ctx, record=None))
        exp = pa.teacher_expectations(spec, None)
        probs = pa.teacher_identity_problems(child_record(teacher_x(), sha(X_BYTES), len(X_BYTES)), str(teacher_x()),
                                             spec, exp)
        ctx2 = standard_case("rt_go_rule_given", spec, teacher_ckpt=DIRS["work"] / "rt" / L.pd.TEACHER_FILE,
                             go_rule_dl=None)
        got_go = arguments(dict(ctx2, go_rule_dl="DL-110"))
    ok = (got_args == "arguments:teacher_ckpt_path" and any("distils from the teacher of record" in p for p in probs)
          and got_go == "arguments:go_rule_not_applicable")
    return ok, f"args={got_args} identity={probs[:2]} go={got_go}"


@case("rf_test_path_gate")
def c_rf_gate():
    bad = DIRS["work"] / "plantseg_test_root" / "evidence"
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        rc = L.pa.main(["gate", "--arm", "e3-r1", "--lambda-logit", "1", "--alpha", "50", "--ckpt-dir",
                        str(DIRS["ck"] / "e3-r1_s42_lambda1_alpha50_80k_a1"), "--evidence", str(bad), "--expect-head",
                        "c" * 40, "--records-commit", "d" * 40, "--records", str(DIRS["ev"]), "--teacher-ckpt",
                        str(teacher_x()), "--clock-offset-seconds", "0", "--approval-dl", "DL-101", "--arms-pin-dl",
                        "DL-106", "--go-rule-dl", "DL-110", "--rehearsal"])
    return rc == 2 and "RESULT: REFUSED (test_path)" in out.getvalue() and not bad.parent.exists(), \
        f"rc={rc} {out.getvalue()[-200:]}"


@case("rf_existing_output_gate_record")
def c_rf_record():
    ctx = standard_case("existing_record")
    rec = scratch("gate_record") / "gate_record.json"
    rec.write_text("{}\n", encoding="utf-8")
    got = arguments(dict(ctx, record=str(rec)))
    return got == "arguments:record_exists", got


@case("rf_missing_input_teacher_record")
def c_rf_teacher_record():
    spec = spec_of("e3-r1")
    c = ArmCase("no_teacher_record", spec)
    files, tr_sha, go_sha = arm_record_files(spec)
    files.pop(L.pa.TEACHER_RECORD_REL.format(arm="R1"))
    c.files.update(files)
    c.dl_extra = dl_rows(c.pin, go_row=f"R1 go rule: go; {go_sha}; {tr_sha} (synthetic)")
    got = run_stages(c.ctx(c.line()))
    # a teacher record that is not R1's complete record (plan §3(iv) check 1; review KDG-3)
    bad = {k: run_stages(standard_case(f"tr_{k}", records=v)) for k, v in
           (("diverged", {"status": "diverged"}), ("incomplete", {"status": "incomplete"}),
            ("other_arm", {"tr_arm": "R2"}))}
    ok = got == "records:teacher_record_missing" and set(bad.values()) == {"records:teacher_record_invalid"}
    return ok, f"missing={got} {bad}"


@case("rf_wrong_checkpoint_pin")
def c_rf_ckpt():
    other = scratch("other_teacher") / X_NAME
    other.write_bytes(X_BYTES + b"one more line\n")
    got_bytes = run_stages(standard_case("ckpt_other_bytes", teacher_ckpt=other))
    named = scratch("other_name") / "iter_32000.pth"
    named.write_bytes(X_BYTES)
    got_name = run_stages(standard_case("ckpt_other_name", teacher_ckpt=named))
    ok = got_bytes == "records:teacher_ckpt_mismatch" and got_name == "records:teacher_ckpt_mismatch"
    return ok, f"bytes={got_bytes} name={got_name}"


@case("rf_missing_go_rule_record")
def c_rf_go_missing():
    spec = spec_of("e3-r1")
    c = ArmCase("no_go_file", spec)
    files, tr_sha, go_sha = arm_record_files(spec)
    files.pop(L.pa.GO_RULE_REL.format(arm="R1"))
    c.files.update(files)
    c.dl_extra = dl_rows(c.pin, go_row=f"R1 go rule: go; {go_sha}; {tr_sha} (synthetic)")
    got_file = run_stages(c.ctx(c.line()))
    got_row = run_stages(standard_case("no_go_row", rows={"go_row": None}))
    got_shas = run_stages(standard_case("go_row_without_shas", rows={"go_row": "R1 go rule: go (synthetic)"}))
    ok = all(g == "records:go_rule_record_missing" for g in (got_file, got_row, got_shas))
    return ok, f"file={got_file} row={got_row} shas={got_shas}"


@case("rf_go_rule_no_go")
def c_rf_no_go():
    """The gate loop on a no-go record: NO-GO (exit 1) at the records stage, no launch block, the record written."""
    pa = L.pa
    ctx = standard_case("no_go_loop", records={"verdict": "no_go"})
    rec = scratch("no_go_record") / "gate_record.json"
    args = SimpleNamespace(rehearsal=False, record=str(rec), arm="e3-r1", lambda_logit=1.0, alpha=50.0, attempt=1,
                           records_commit=ctx["records_commit"], evidence=ctx["evidence"])
    stages = [("arguments", lambda c: ("PASS", "E1's part and the arm's, not re-run here", {}))] + \
        [s for s in pa.STAGES if s[0] in ("records", "selection_inputs", "launch_order")]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = pa.run_gate(args, stages=stages, ctx=ctx)
    written = json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else {}
    ok = (rc == 1 and "first failing stage: records [go_rule_not_go]" in out.getvalue()
          and "----- launch block -----" not in out.getvalue() and written.get("verdict") == "NO-GO"
          and written.get("format") == pa.GATE_RECORD_FORMAT)
    return ok, f"rc={rc} {out.getvalue()[-240:]} {written.get('verdict')}"


# ------------------------------------------------------------------------------------ E: the arithmetic
def telemetry_file(name: str, values, *, finished=True) -> Path:
    d = scratch(f"tel_{name}")
    rows = [{"event": "train", "iter": i + 1, "loss": 1.0, "wall_clock": 1000.0 + i, "iter_seconds": v}
            for i, v in enumerate(values)]
    if finished:
        rows.append({"event": "run_end", "iter": len(values), "checks_passed": True, "wall_clock_end": 9999.0})
    p = d / "e3_telemetry.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def smoke_json(name: str, fmt: str, key: str, median: float) -> Path:
    p = scratch(f"ref_{name}") / f"{name}.json"
    p.write_text(json.dumps({"format": fmt, "throughput": {key: {"iter_seconds": {"median": median, "n": 20}}}}),
                 encoding="utf-8")
    return p


def arith(spec, *, telemetry=None, smokes=(), left=100000.0) -> tuple:
    """arm_cut_arithmetic on a minimal context: (verdict, arithmetic record), with `left` seconds to the end of
    Schedule T's freeze date."""
    pa, ss = L.pa, L.ss
    now = float(ss.cutoff_instant(pa.FREEZE_DATES["T"])) - left
    ctx = {"spec": spec, "schedule": {"in_force": "T"}, "now": now, "reference_telemetry": telemetry,
           "reference_smokes": list(smokes)}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pa.stage_arm_cut_arithmetic(ctx)
        return "PASS", ctx.get("arithmetic")
    except L.pd.GateError as e:
        return e.code, ctx.get("arithmetic")


@case("cm_arith_cut")
def c_arith_cut():
    """80,000 iterations x the median s/iter against the time left: a telemetry whose median fits (1.0 s/iter:
    80,000 s of 100,000) while its mean would not (9.9 s/iter), and a kd_smoke whose median does not fit."""
    spec = spec_of("e3-r1")
    tel = telemetry_file("median_fits", [1.0] * 9 + [50.0] * 2)
    got_fit, ar_fit = arith(spec, telemetry=tel)
    got_cut, ar_cut = arith(spec, smokes=[smoke_json("e3_slow", "kd_smoke/1", "e3", 1.5)])
    unfinished = telemetry_file("unfinished", [1.0] * 4, finished=False)
    got_unf, _ = arith(spec, telemetry=unfinished)
    ok = (got_fit == "PASS" and ar_fit["applied_s_per_iter"] == 1.0 and ar_fit["references"][0]["rows"] == 11
          and got_cut == "arm_cut_by_arithmetic" and ar_cut["seconds_needed"] == 120000.0 and ar_cut["cut"] is True
          and got_unf == "reference_unusable")
    return ok, f"fit={got_fit} {ar_fit and ar_fit.get('applied_s_per_iter')} cut={got_cut} unfinished={got_unf}"


@case("cm_arith_no_measurement_not_cut")
def c_arith_none():
    got, ar = arith(spec_of("e3-r1"), left=1.0)
    return (got == "PASS" and ar["measured"] is False and ar["cut"] is False
            and ar["text"].startswith("no measurement: not cut")), f"{got} {ar and ar.get('text')}"


@case("cm_arith_flip_reference_max")
def c_arith_max():
    """Ruling R7 / audit F40 on a record-teacher arm with its own pod smoke: E3's figure (1.0, fits) and the arm's
    (1.5, does not fit) are both read with one reader; the larger applies, so the arm is cut. The arm's pod-smoke
    format is refused for another arm."""
    with patched(L.pa, "ARM_DIR", rt_dir()):
        spec = spec_of("e3-rt", rt_dir())
        e3 = smoke_json("e3_fast", "kd_smoke/1", "e3", 1.0)
        own = smoke_json("rt_pod", L.pa.pod_smoke_format(spec), "e3-rt", 1.5)
        got, ar = arith(spec, smokes=[e3, own])
    got_other, _ = arith(spec_of("e3-r1"), smokes=[own])
    ok = (got == "arm_cut_by_arithmetic" and ar["applied_s_per_iter"] == 1.5
          and sorted(r["key"] for r in ar["references"]) == ["e3", "e3-rt"] and got_other == "reference_format")
    return ok, f"{got} {ar and ar.get('applied_s_per_iter')} other={got_other}"


# ------------------------------------------------------------------------------ F: check-run-meta, lines
def arm_gate_record(spec, *, teacher=None) -> dict:
    """A GO record of the arm gate (kd_arm_gate_record/1) as check-run-meta reads it."""
    pa = L.pa
    rid = pa.run_id_of(spec, 1.0, 50.0, 1)
    folder = "/workspace/kd_records/abababababab"
    sel = {"lambda_logit": {"rel": L.ss.LAMBDA_SELECTION_REL, "path": f"{folder}/{L.ss.LAMBDA_SELECTION_REL}",
                            "sha256": "1" * 64},
           "alpha_cwd": {"rel": L.ss.ALPHA_SELECTION_REL, "path": f"{folder}/{L.ss.ALPHA_SELECTION_REL}",
                         "sha256": "2" * 64}}
    t = teacher or {"name": X_NAME, "sha256": sha(X_BYTES), "bytes": len(X_BYTES)}
    return {"format": pa.GATE_RECORD_FORMAT, "verdict": "GO", "profile": pa.PROFILE, "stage": spec.stage,
            "seed": spec.seed, "run_id": rid, "ckpt_dir": str(DIRS["work"] / "crm_runs" / rid), "pin": "c" * 40,
            "records_commit": "ab" * 20, "lambda_logit": 1.0, "alpha": 50.0, "selection_files": sel,
            "teacher_ckpt": f"/workspace/arm_teachers/{t['name']}", "decision_date": None,
            "launch_order_position": None, "arm": pa.arm_fields(spec), "teacher": t}


def good_arm_row(spec, exp: dict) -> dict:
    """A run_meta row that meets every arm rule for `exp`: pd's good E3 row with the arm's fields and teacher."""
    pd = L.pd
    row = L.spd.good_row("e3", exp)
    t = exp["teacher"]
    row["teacher_provenance"] = {**row["teacher_provenance"], "ckpt_path": exp["teacher_ckpt"],
                                 "ckpt_sha256": t["sha256"], "expected_sha256": t["sha256"], "ckpt_bytes": t["bytes"]}
    row.update(arm=spec.run_stage, descriptive=True, parent_of_e4_e7=False)
    row["target_construction"] = spec.target_construction
    assert list(row)[:-1] == pd.run_meta_keys("e3")
    return row


def crm(spec, rec: dict, row: dict) -> tuple:
    """pa.check_run_meta on a gate record and one run_meta row: (rc or refusal code, stdout)."""
    gr = scratch("crm") / "gate_record.json"
    gr.write_text(json.dumps(rec), encoding="utf-8")
    d = Path(rec["ckpt_dir"])
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{spec.stage}_run_meta.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return L.spd.quiet(L.pa.check_run_meta, SimpleNamespace(allow_smoke=False, gate_record=str(gr), ckpt_dir=str(d)))


@case("cm_check_run_meta_arm_rules")
def c_crm():
    pa, td = L.pa, L.td
    spec = spec_of("e3-r1")
    rec = arm_gate_record(spec)
    gr = scratch("crm_exp") / "gate_record.json"
    gr.write_text(json.dumps(rec), encoding="utf-8")
    _, exp = pa.run_meta_expectations(SimpleNamespace(allow_smoke=False, gate_record=str(gr)))
    good = good_arm_row(spec, exp)
    p_good, w_good = pa.arm_meta_problems(good, spec, exp, td)
    bad = {"arm": {**good, "arm": "E3"}, "descriptive": {**good, "descriptive": False},
           "parent_of_e4_e7": {**good, "parent_of_e4_e7": True},
           "target_construction": {**good, "target_construction": {"K": 1, "views": 2}},
           "teacher_provenance": {**good, "teacher_provenance": {**good["teacher_provenance"],
                                                                 "ckpt_sha256": L.pd.TEACHER_SHA256}}}
    missed = []
    for k, row in bad.items():
        p, _ = pa.arm_meta_problems(row, spec, exp, td)
        if not any(x.startswith(f"{k} (") for x in p):
            missed.append(k)
    no_tc = {k: v for k, v in good.items() if k != "target_construction"}
    p_no_tc, _ = pa.arm_meta_problems(no_tc, spec, exp, td)
    moved = {"target_construction": good["target_construction"], **no_tc}
    p_moved, _ = pa.arm_meta_problems(moved, spec, exp, td)
    rc_pass, out_pass = crm(spec, rec, good)
    rc_fail, out_fail = crm(spec, rec, {**good, "parent_of_e4_e7": True})
    ok = (p_good == [] and w_good == [] and not missed and any(x.startswith("keys:") for x in p_no_tc)
          and any("last key" in x for x in p_moved) and rc_pass == 0 and "RESULT: PASS" in out_pass
          and rc_fail == 1 and "MISMATCH: parent_of_e4_e7" in out_fail)
    return ok, f"good={p_good} missed={missed} no_tc={p_no_tc[:1]} moved={p_moved[:1]} {rc_pass} {rc_fail}"


@case("dt_check_run_meta_twice_identical")
def c_dt_crm():
    spec = spec_of("e3-r1")
    rec = arm_gate_record(spec)
    gr = scratch("crm_twice") / "gate_record.json"
    gr.write_text(json.dumps(rec), encoding="utf-8")
    _, exp = L.pa.run_meta_expectations(SimpleNamespace(allow_smoke=False, gate_record=str(gr)))
    row = good_arm_row(spec, exp)
    a, b = crm(spec, rec, row), crm(spec, rec, row)
    return a[0] == 0 and a == b, f"{a[0]} {b[0]} equal={a == b}"


@case("cm_launched_and_not_launched_lines")
def c_lines():
    """check-run-meta's launched line and not-launched's line each extend a log that holds the arm's launch line
    (sweep_select.LaunchLog parses both); not-launched refuses a run that wrote a real run_meta row. The case also
    covers the stopped line (audit F34): stopped's line extends a log with a launched arm run, and a run whose
    telemetry holds the trainer's end record has none."""
    pa, ss = L.pa, L.ss
    spec = spec_of("e3-r1")
    rec = arm_gate_record(spec)
    gr = scratch("crm_lines") / "gate_record.json"
    gr.write_text(json.dumps(rec), encoding="utf-8")
    _, exp = pa.run_meta_expectations(SimpleNamespace(allow_smoke=False, gate_record=str(gr)))
    row = good_arm_row(spec, exp)
    rc, out = crm(spec, rec, row)
    head = "LAUNCHED LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it):\n"
    launched = out.split(head)[1].splitlines()[0] if head in out else ""
    own = pa.arm_launch_line(ss, spec, lam=1.0, alpha=50.0, attempt=1, schedule="T", code_pin="c" * 40,
                             ckpt_dir=rec["ckpt_dir"], am8a_report=None)
    try:
        log = ss.LaunchLog(L.pd.line_bytes(own) + b"\n" + launched.encode("utf-8") + b"\n")
        launched_ok = own["run_id"] in log.launched
    except ss.SelectionRefused as e:
        launched_ok = f"{e.code}: {e}"
    # not-launched, at the tip of a repository that holds the arm's launch line and the evidence
    sc = entry_scenario("not_launched")
    line = sc.case.line()
    ev_rel = f"reports/not_launched/{line['run_id']}.txt"
    sc.case.sw.events.append(line)
    sc.case.dl_extra = ""
    sc.case.commit_h()
    sc.case.sw.commit("the evidence", {ev_rel: "the gate refused after the line was pushed (synthetic)\n"})
    run_dir = scratch("nl_run") / line["run_id"]
    run_dir.mkdir()
    nl_args = SimpleNamespace(arm="e3-r1", run_id=line["run_id"], reason="the gate cut it (synthetic)", evidence=ev_rel,
                              run_dir=str(run_dir))
    with L.spd.git_root(sc.repo):
        rc_nl, out_nl = L.spd.quiet(pa.cmd_not_launched, nl_args)
    nl_head = "NOT_LAUNCHED LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it):\n"
    nl = out_nl.split(nl_head)[1].splitlines()[0] if nl_head in out_nl else ""
    try:
        log2 = ss.LaunchLog((sc.repo / ss.LAUNCH_LOG_REL).read_bytes() + nl.encode("utf-8") + b"\n")
        nl_ok = line["run_id"] in log2.not_launched
    except ss.SelectionRefused as e:
        nl_ok = f"{e.code}: {e}"
    (run_dir / "e3_run_meta.jsonl").write_text(json.dumps({"event": "run_meta", "mode": "real"}) + "\n", encoding="utf-8")
    with L.spd.git_root(sc.repo):
        rc_started, _ = L.spd.quiet(pa.cmd_not_launched, nl_args)
    # the stopped line (audit F34): a launched arm run that ended without a trainer record, with its stop report
    sc2 = entry_scenario("stopped")
    line2 = sc2.case.line()
    rid2 = line2["run_id"]
    sc2.case.sw.events.extend([line2, {"event": "launched", "run_id": rid2, "run_meta_sha256": "0" * 64,
                                       "launch_time_utc": L.sss.BASE_A, "git_head": sc2.pin, "records_commit": "e" * 40,
                                       "decision_date": None, "launch_order_position": None}])
    sc2.case.dl_extra = ""
    sc2.case.commit_h()
    rep_rel = f"reports/stops/{rid2}.md"
    sc2.case.sw.commit("the stop report", {rep_rel: "the pod was lost (synthetic)\n"})
    run2 = scratch("st_run") / rid2
    run2.mkdir()
    (run2 / "e3_telemetry.jsonl").write_text(json.dumps({"event": "train", "iter": 1, "wall_clock": 1.0}) + "\n",
                                             encoding="utf-8")
    st_args = SimpleNamespace(arm="e3-r1", run_id=rid2, report=rep_rel, run_dir=str(run2))
    with L.spd.git_root(sc2.repo):
        rc_st, out_st = L.spd.quiet(pa.cmd_stopped, st_args)
    st_head = "STOPPED LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it with its report):\n"
    st = out_st.split(st_head)[1].splitlines()[0] if st_head in out_st else ""
    try:
        log3 = ss.LaunchLog((sc2.repo / ss.LAUNCH_LOG_REL).read_bytes() + st.encode("utf-8") + b"\n")
        st_ok = rid2 in log3.stopped
    except ss.SelectionRefused as e:
        st_ok = f"{e.code}: {e}"
    with (run2 / "e3_telemetry.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "run_end", "iter": 1, "wall_clock_end": 2.0}) + "\n")
    with L.spd.git_root(sc2.repo):
        rc_recorded, _ = L.spd.quiet(pa.cmd_stopped, st_args)
    ok = (rc == 0 and launched_ok is True and rc_nl == 0 and nl_ok is True and rc_started == "not_launched_but_started"
          and rc_st == 0 and st_ok is True and rc_recorded == "stopped_but_recorded")
    return ok, (f"launched={launched_ok} not_launched={rc_nl} {nl_ok} started={rc_started} stopped={rc_st} {st_ok} "
                f"recorded={rc_recorded}")


# ------------------------------------------------------------------------------------------ G: E7-X
def e7_fixture(label: str, *, spec=None, row_over=None, rows=1, payload_over=None, best_name=None) -> tuple:
    """A synthetic E3-X run directory (run_meta, best.json, the checkpoint): (run dir, checkpoint, sha256)."""
    import torch
    spec = spec or spec_of("e3-r1")
    rid = L.pa.run_id_of(spec, 1.0, 50.0, 1)
    d = scratch(f"e7_{label}") / rid
    d.mkdir()
    row = {"event": "run_meta", "stage": "E3", "mode": "real", "seed": 42, "max_iters": 80000, "poly_horizon": 80000,
           "git_head": "c" * 40, "records_commit": "d" * 40, "arm": spec.run_stage, "descriptive": True,
           "parent_of_e4_e7": False, "target_construction": spec.target_construction}
    row.update(row_over or {})
    (d / "e3_run_meta.jsonl").write_text("".join(json.dumps(row) + "\n" for _ in range(rows)), encoding="utf-8")
    ck = d / "e3_student_best_iter76000.pt"
    torch.save({"stage": "E3", "arm": spec.run_stage, "target_construction": spec.target_construction,
                "model_state_dict": {}, **(payload_over or {})}, ck)
    (d / "best.json").write_text(json.dumps({"best_ckpt": f"/workspace/kd_runs/{rid}/{best_name or ck.name}",
                                             "best_val_miou_all_class": 0.4}), encoding="utf-8")
    return d, ck, sha(ck.read_bytes())


def ptq_stub(**kw) -> dict:
    """src.quant.ptq.run_ptq's contract on disk and in its return value, without a model (the wrapper's checks)."""
    out = Path(kw["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    names = {"torchscript": "e7_int8_qnnpack.torchscript.pt", "state_dict": "e7_int8_student.pt",
             "x86_copy": "e7_int8_x86_latency.torchscript.pt", "run_meta": "e7_ptq_run_meta.json"}
    shas = {}
    for k, n in names.items():
        data = f"synthetic {k} of {kw['source_ckpt']}\n".encode("utf-8")
        (out / n).write_bytes(data)
        shas[k] = sha(data)
    return {"stage": "E7", "converted_artifact": names["torchscript"], "converted_artifact_sha256": shas["torchscript"],
            "state_dict_artifact": {"path": names["state_dict"], "sha256": shas["state_dict"]},
            "x86_latency_copy": {"path": names["x86_copy"], "sha256": shas["x86_copy"]}, "ptq_run_role": "am10_of_record",
            "calibration": {"list_path": str(kw["calibration_list"]), "list_file_sha256": "1" * 64,
                            "checksum_sha256": kw["expect_list_checksum"], "seed": 42, "role": "am10_list_of_record",
                            "list_artifact_status": "registered"}}


def e7_main(run_dir, ck, ck_sha, *, out=None, arm="e3-r1", list_ok=True, cal=None) -> tuple:
    """run_ptq_arm.main in-process with run_ptq stubbed: (rc, stdout, out dir)."""
    rpa = L.rpa
    out = out or (scratch("e7out") / "e7-r1_s42_a1")
    argv = ["--arm", arm, "--real-run", "--confirm-real-run", "--e3-run-dir", str(run_dir), "--source-ckpt", str(ck),
            "--expect-source-sha256", ck_sha, "--calibration-list", str(cal or (DIRS["work"] / "calibration.json")),
            "--expect-list-checksum", "a" * 64, "--data-root", str(DIRS["data"]), "--out-dir", str(out)]
    with patched(rpa, "ptq_run", ptq_stub), \
            patched(rpa, "list_of_record_problem", (lambda *a: None) if list_ok else rpa.list_of_record_problem):
        with contextlib.redirect_stdout(io.StringIO()) as buf, contextlib.redirect_stderr(io.StringIO()):
            rc = rpa.main(argv)
    return rc, buf.getvalue(), Path(out)


def refusal(out: str) -> str:
    line = next((ln for ln in out.splitlines() if ln.startswith("RESULT: REFUSED [")), "")
    return line[len("RESULT: REFUSED ["):line.find("]")] if line else out[-120:]


@case("cm_e7_arm_wrapper_requires_arm_run_meta")
def c_e7_requires():
    """E7-X takes only the arm's E3 run: one run_meta row with the arm's fields, best.json naming the source in the
    run directory, the arm's payload, and the AM-10 list of record (the real list check on a missing list)."""
    d, ck, s = e7_fixture("good")
    got = {"good": e7_main(d, ck, s)[0]}
    for label, kw in (("record_e3_row", {"row_over": {"arm": None, "descriptive": False, "parent_of_e4_e7": True}}),
                      ("other_arm_row", {"row_over": {"arm": "E3-R2"}}), ("two_rows", {"rows": 2}),
                      ("parent_true", {"row_over": {"parent_of_e4_e7": True}}),
                      ("tc_views2", {"row_over": {"target_construction": {"K": 1, "views": 2}}}),
                      ("best_names_other", {"best_name": "e3_student_best_iter4000.pt"}),
                      ("payload_not_arm", {"payload_over": {"arm": None}})):
        dd, cc, ss_ = e7_fixture(label, **kw)
        got[label] = refusal(e7_main(dd, cc, ss_)[1])
    got["wrong_sha"] = refusal(e7_main(d, ck, "0" * 64, out=scratch("e7out_sha") / "e7-r1_s42_a1")[1])
    got["list_not_of_record"] = refusal(e7_main(d, ck, s, out=scratch("e7out_list") / "e7-r1_s42_a1", list_ok=False)[1])
    want = {"good": 0, "record_e3_row": "arm_run_meta_mismatch", "other_arm_row": "arm_run_meta_mismatch",
            "two_rows": "arm_run_meta_mismatch", "parent_true": "arm_run_meta_mismatch",
            "tc_views2": "arm_run_meta_mismatch", "best_names_other": "source_not_best",
            "payload_not_arm": "source_not_arm", "wrong_sha": "source_sha256_mismatch",
            "list_not_of_record": "calibration_not_of_record"}
    return got == want, {k: v for k, v in got.items() if want.get(k) != v} or got


@case("cm_e7_arm_record_binds")
def c_e7_binds():
    d, ck, s = e7_fixture("binds")
    rc, out, od = e7_main(d, ck, s)
    rec_p = od / L.rpa.RECORD_FILE
    if rc != 0 or not rec_p.is_file():
        return False, f"rc={rc} {out[-300:]}"
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    arts = {k: sha((od / v["path"]).read_bytes()) == v["sha256"] for k, v in rec["artifacts"].items()}
    ok = (rec["format"] == "e7_arm_record/1" and rec["arm"] == "e3-r1" and rec["e7_stage"] == "E7-R1"
          and rec["e3_run"]["run_id"] == d.name and rec["e3_run"]["run_meta_sha256"] == sha((d / "e3_run_meta.jsonl").read_bytes())
          and rec["e3_run"]["best_json_sha256"] == sha((d / "best.json").read_bytes())
          and rec["source"]["sha256"] == s and rec["source"]["payload_arm"] == "E3-R1"
          and rec["calibration"]["of_record"] is True and rec["calibration"]["checksum_sha256"] == "a" * 64
          and set(arts) == {"torchscript", "state_dict", "x86_copy", "run_meta"} and all(arts.values())
          and rec["registry"]["sha256"] == spec_of("e3-r1").sha256)
    return ok, {k: rec.get(k) for k in ("arm", "e3_run", "source")} | {"arts": arts}


@case("rf_test_path_e7")
def c_rf_e7():
    d, ck, s = e7_fixture("sl1_guard")
    bad = DIRS["work"] / "plantseg_test_root" / "e7-r1_s42_a1"
    rc, out, _ = e7_main(d, ck, s, out=bad)
    return rc == 2 and "[test_path]" in out and not bad.parent.exists(), f"rc={rc} {out[-200:]}"


@case("rf_existing_output_e7")
def c_rf_e7_out():
    d, ck, s = e7_fixture("existing")
    out = scratch("e7out_existing") / "e7-r1_s42_a1"
    out.mkdir()
    (out / L.rpa.RECORD_FILE).write_text("{}\n", encoding="utf-8")
    rc1, o1, _ = e7_main(d, ck, s, out=out)
    out2 = scratch("e7out_nonempty") / "e7-r1_s42_a1"
    out2.mkdir()
    (out2 / "something.txt").write_text("x\n", encoding="utf-8")
    rc2, o2, _ = e7_main(d, ck, s, out=out2)
    rc3, o3, _ = e7_main(d, ck, s, out=scratch("e7out_name") / "e7-r2_s42_a1")
    ok = rc1 == rc2 == rc3 == 2 and refusal(o1) == refusal(o2) == "output_exists" and refusal(o3) == "out_dir_name"
    return ok, f"{rc1} {refusal(o1)} {rc2} {refusal(o2)} {rc3} {refusal(o3)}"


# ------------------------------------------------------------------------------------------ H: mutations
def launch_fields_mutant(**over):
    """td.launch_fields with an arm run's (or every run's) fields changed: a mutated trainer seam."""
    orig = L.td.launch_fields

    def mutated(stage, **kw):
        fields = orig(stage, **kw)
        for k, v in over.items():
            if k == "drop":
                fields.pop(v, None)
            elif kw.get("arm") is not None or k == "arm_default":
                fields[{"arm_default": "arm"}.get(k, k)] = v
        return fields
    return mutated


def is_record_never(_sha) -> bool:
    return False


MUTATIONS = [
    ("mut_parent_true", "cm_run_meta_arm_fields", lambda: patched(L.td, "launch_fields",
                                                                  launch_fields_mutant(parent_of_e4_e7=True))),
    ("mut_target_construction_missing", "cm_run_meta_target_construction_k1_v1",
     lambda: patched(L.td, "launch_fields", launch_fields_mutant(drop="target_construction"))),
    ("mut_arm_key_in_default_run_meta", "cm_default_run_meta_has_no_arm_key",
     lambda: patched(L.td, "launch_fields", launch_fields_mutant(arm_default="E3"))),
    ("mut_stage_e3_in_arm_line", "cm_arm_line_not_alpha_candidate", lambda: patched(
        L.pa, "arm_launch_line", (lambda orig: lambda ss_, spec, **kw: {**orig(ss_, spec, **kw), "stage": "E3"})(
            L.pa.arm_launch_line))),
    ("mut_gate_skips_go_rule", "cm_gate_go_rule_go_required", lambda: patched(
        L.pa, "go_rule_check", lambda ss_, src, dl, spec, dl_id, tr: {"dl": dl_id, "path": None, "sha256": None,
                                                                      "verdict": "go",
                                                                      "checkpoint_sha256": tr["checkpoint"]["sha256"]})),
    ("mut_gate_accepts_record_teacher", "cm_gate_teacher_identity_arm_values",
     lambda: patched(L.pa, "is_record_teacher", is_record_never)),
    ("mut_approval_marker_ignored", "cm_gate_approval_row_required", lambda: patched(L.pa, "APPROVAL_MARKER", "")),
    ("mut_pin_rule_off", "cm_gate_arms_pin_rule", lambda: _multi(patched(L.pa, "pin_problem", lambda *a: None),
                                                                 patched(L.pa, "arm_pins_problem", lambda *a: None))),
    ("mut_median_replaced_by_mean", "cm_arith_cut", lambda: patched(L.pa, "MEDIAN", statistics.mean)),
    ("mut_divergence_repeat_allowed", "rf_arm_repeat_after_divergence",
     lambda: patched(L.pa, "is_divergence", lambda run: False)),
    ("mut_min_in_place_of_max", "cm_arith_flip_reference_max", lambda: patched(L.pa, "APPLIED", min)),
    ("mut_alpha_record_rejected", "cm_alpha_cut_record_accepted", lambda: patched(
        L.pd, "alpha_cut_record_ok", _raise_alpha_record)),
    ("mut_record_teacher_arm_accepts_arm_teacher", "rf_record_teacher_arm_with_arm_teacher",
     lambda: patched(L.pa, "record_teacher_rule", lambda *a, **k: None)),
]


def _raise_alpha_record(*_a, **_k):
    raise L.pd.GateError("alpha_cut_record_invalid", "mutation: the alpha decision record is rejected")


@contextlib.contextmanager
def _multi(*cms):
    with contextlib.ExitStack() as stack:
        for cm in cms:
            stack.enter_context(cm)
        yield


# ------------------------------------------------------------------------------------------------ main
ORDER = ("cm_registry_loads_r1_r2", "cm_registry_schema", "cm_registry_tolerates_extra_entries",
         "cm_trainer_arm_requires_e3", "cm_trainer_arm_seed_42_only", "cm_trainer_arm_horizon_80k",
         "cm_run_meta_arm_fields", "cm_run_meta_target_construction_k1_v1", "cm_default_run_meta_has_no_arm_key",
         "cm_payload_arm_keys_only_in_arm_runs", "cm_r1_targets_none_loss_bitwise_e3", "dt_two_dry_runs_identical",
         "cm_entry_arm_line_bytes", "cm_arm_line_parsed_by_launchlog", "cm_arm_line_not_alpha_candidate",
         "cm_launch_note_cut_place", "cm_entry_parses_log_with_launchlog", "cm_clock_offset_recorded",
         "dt_entry_twice_identical_bytes", "rf_test_path_entry", "rf_arm_repeat_after_divergence",
         "cm_gate_stage_list", "cm_gate_teacher_identity_arm_values", "cm_gate_go_rule_go_required",
         "cm_gate_approval_row_required", "cm_gate_arms_pin_rule", "cm_gate_lambda_alpha_inherited",
         "cm_gate_launch_block_has_arm", "cm_alpha_cut_record_accepted", "cm_gate_record_teacher_arm",
         "rf_record_teacher_arm_without_approval", "rf_record_teacher_arm_with_arm_teacher", "rf_test_path_gate",
         "rf_existing_output_gate_record", "rf_missing_input_teacher_record", "rf_wrong_checkpoint_pin",
         "rf_missing_go_rule_record", "rf_go_rule_no_go", "cm_arith_cut", "cm_arith_no_measurement_not_cut",
         "cm_arith_flip_reference_max", "cm_check_run_meta_arm_rules", "dt_check_run_meta_twice_identical",
         "cm_launched_and_not_launched_lines", "cm_e7_arm_wrapper_requires_arm_run_meta", "cm_e7_arm_record_binds",
         "rf_test_path_e7", "rf_existing_output_e7")


def refused(text: str) -> int:
    print(f"REFUSED: {text}")
    print("RESULT: REFUSED -- nothing ran, nothing written")
    return 2


def main(argv=None) -> int:
    global L
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--internal"] and len(argv) > 1 and argv[1] in INTERNAL:
        return INTERNAL[argv[1]](argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
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
            n = 0
            while True:
                n += 1
                work = Path(tempfile.gettempdir()) / f"arms_kd_smoke_{os.getpid()}_{n}"
                refuse_test_path(work, "the default work dir (set TMPDIR elsewhere)")
                work = require_outside_repo(work, "the default work dir")
                if not work.exists():
                    break
    except Refused as e:
        return refused(str(e))
    work.mkdir(parents=True)
    (work / "scratch").mkdir()
    tempfile.tempdir = str(work / "scratch")          # every scratch folder of this process lies under the work dir
    L = libs()
    DIRS.update(work=work, ck=work / "ckpt", ev=work / "evidence", data=work / "data_root")
    for k in ("ck", "ev", "data"):
        DIRS[k].mkdir()
    os.environ["PLANTSEG_DATA_ROOT"] = str(DIRS["data"])     # E1's arguments stage reads it (absolute, outside both)
    t0 = time.time()
    print("=" * 78)
    print("KD ARMS SMOKE (L-TEACHER-ARMS C2) -- CPU, synthetic scratch repositories, stub teachers and children")
    print(f"work dir: {work}")
    print("=" * 78, flush=True)
    try:
        run_workers(a.threads)
    except Exception as e:  # noqa: BLE001 - the worker checks fail with this detail
        WORKERS["error"] = f"{type(e).__name__}: {e}"
    for name in ORDER:
        ok, detail = run_case(name)
        check(name, ok, detail)
    base = {n: ok for n, ok, _ in results}
    for mname, killed_by, make in MUTATIONS:
        if not base.get(killed_by):
            check(mname, False, f"{killed_by} does not pass unmutated, so the mutation proves nothing")
            continue
        ok_c, detail_c = run_case(killed_by)          # the control: it passes again without the mutation
        if not ok_c:
            check(mname, False, f"{killed_by} fails when re-run unmutated, so the mutation proves nothing: {detail_c}")
            continue
        with make():
            ok, detail = run_case(killed_by)
        check(mname, not ok, f"{killed_by} under the mutation: {'PASS (not killed)' if ok else 'FAIL (killed)'}: "
                             f"{detail}")
    seconds = round(time.time() - t0, 1)
    (work / "arms_kd_summary.json").write_text(json.dumps({
        "seconds": seconds, "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results],
        "workers": {k: {"returncode": v.get("returncode"), "log": v.get("log")} if isinstance(v, dict) else v
                    for k, v in WORKERS.items()}}, indent=1, default=str), encoding="utf-8")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}  {detail[:500] if not ok else ''}".rstrip())
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nsummary: {work / 'arms_kd_summary.json'} ({seconds} s)")
    print(f"RESULT: {'PASS' if results and passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if results and passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
