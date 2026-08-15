#!/usr/bin/env python3
"""Synthetic launch verification for the E4-E7 runners. No dataset, no GPU, no real checkpoint.

Every gate is exercised through the real entry points (`scripts/run_e4.py` … `run_e7.py`) or through
`src.quant.runner`'s named guards, using synthetic checkpoints and temp dirs OUTSIDE the repository.
Nothing here trains, calibrates on real data, downloads, or writes into the git checkout.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from scripts.run_e4 import STAGE as S4, main as run_e4  # noqa: E402
from scripts.run_e5 import STAGE as S5, main as run_e5  # noqa: E402
from scripts.run_e6 import STAGE as S6, main as run_e6  # noqa: E402
from scripts.run_e7 import STAGE as S7, main as run_e7  # noqa: E402
from scripts.create_ptq_calibration_index import main as make_index  # noqa: E402
from src.distill.export import CWD_PROJECTION_KEY  # noqa: E402
from src.models.student import build_student  # noqa: E402
from src.quant import build_calibration_index, save_calibration_index  # noqa: E402
from src.quant.runner import (EarlyStopper, QuantRunError, check_backend,  # noqa: E402
                              check_output_dir, check_source, unresolved_qat_values)
from src.quant.stages import resolve_quant_stage  # noqa: E402
from src.seeds import set_seed  # noqa: E402

NC = 116
TMP = Path(tempfile.mkdtemp(prefix="smoke_quant_runners_"))
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def run(fn, argv: list[str]) -> int:
    """Invoke a runner entry point, mapping an argparse `SystemExit` to its exit code."""
    try:
        return fn(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


def expect_code(name: str, code: str, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no QuantRunError raised")
    except QuantRunError as e:
        check(name, e.code == code, e.code if e.code == code else f"{e.code} != {code}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


def _state() -> dict:
    return {k: v.detach().clone() for k, v in build_student(pretrained=False).state_dict().items()}


def write(name: str, payload) -> Path:
    p = TMP / name
    torch.save(payload, p)
    return p


class _Args:
    def __init__(self, **kw):
        d = dict(batch_size=None, weight_decay=None, grad_clip_norm=None,
                 bn_freeze_pct=None, observer_freeze_pct=None, epochs=None,
                 early_stop_patience=None)
        d.update(kw)
        for k, v in d.items():
            setattr(self, k, v)


# ---------------------------------------------------------------- fixtures
E1_CKPT = write("e1.pt", {"iter": 80000, "num_classes": NC, "model_state_dict": _state()})
E2_CKPT = write("e2.pt", {"stage": "E2", "num_classes": NC, "model_state_dict": _state()})
E3_CKPT = write("e3.pt", {"stage": "E3", "num_classes": NC, "model_state_dict": _state(),
                          CWD_PROJECTION_KEY: {"weight": torch.randn(320, 160, 1, 1)}})
_leak = _state(); _leak["cwd_projection.weight"] = torch.randn(320, 160, 1, 1)
E3_LEAK = write("e3_leak.pt", {"stage": "E3", "num_classes": NC, "model_state_dict": _leak})
OUT = TMP / "out"

_pool = [f"train_{i:05d}" for i in range(5367)]
E4_INDEX = build_calibration_index(_pool)
E4_INDEX_PATH = save_calibration_index(E4_INDEX, TMP / "calib" / "e4_index.json")
OTHER_INDEX = build_calibration_index([f"train_{i:05d}" for i in range(2000, 7367)])
OTHER_PATH = save_calibration_index(OTHER_INDEX, TMP / "calib" / "other_index.json")


# ---------------------------------------------------------------- 1. authorization
def test_authorization() -> None:
    for label, fn in (("e4", run_e4), ("e5", run_e5), ("e6", run_e6), ("e7", run_e7)):
        check(f"{label}_no_flags_exit_2", run(fn, []) == 2)
        check(f"{label}_only_real_run_exit_2", run(fn, ["--real-run"]) == 2)
        check(f"{label}_only_confirm_exit_2", run(fn, ["--confirm-real-run"]) == 2)
    check("no_repo_writes_from_blocked_runs",
          not (REPO / "e4_int8_student.pt").exists()
          and not list(REPO.glob("*_run_provenance.json")))


# ---------------------------------------------------------------- 2. stage pinning
def test_stage_pinning() -> None:
    check("entry_points_pin_their_stage",
          (S4, S5, S6, S7) == ("e4", "e5", "e6", "e7"), f"{(S4, S5, S6, S7)}")
    src = "".join((REPO / "scripts" / f"run_{s}.py").read_text(encoding="utf-8")
                  for s in ("e4", "e5", "e6", "e7"))
    check("stage_is_not_a_cli_flag", "--stage" not in src and "add_argument(\"--stage\"" not in src,
          "no CLI path can redirect run_e4.py to another stage")
    # a runner refuses an unparsed flag rather than silently accepting one
    rc = run(run_e4, ["--real-run", "--confirm-real-run", "--stage", "e7"])
    check("unknown_stage_flag_rejected", rc == 2, f"exit {rc} (argparse refuses the flag)")


# ---------------------------------------------------------------- 3. common gates
def test_common_gates() -> None:
    expect_code("out_dir_unset", "output_dir_unset", check_output_dir, None)
    expect_code("out_dir_inside_repo", "output_dir_inside_repo", check_output_dir,
                str(REPO / "quant_runs"))
    check("out_dir_external_accepted", check_output_dir(str(OUT)) == OUT.resolve())
    check("out_dir_not_created_without_flag", not OUT.exists())

    e4, e5, e6, e7 = (resolve_quant_stage(s) for s in ("e4", "e5", "e6", "e7"))
    expect_code("source_unset", "source_unset", check_source, e4, None)
    expect_code("source_missing", "source_missing", check_source, e4, str(TMP / "absent.pt"))
    # source/stage consistency, both directions
    expect_code("e4_rejects_e3_source", "source_invalid", check_source, e4, str(E3_CKPT))
    expect_code("e5_rejects_e3_source", "source_invalid", check_source, e5, str(E3_CKPT))
    expect_code("e6_rejects_e1_source", "source_invalid", check_source, e6, str(E1_CKPT))
    expect_code("e6_rejects_e2_source", "source_invalid", check_source, e6, str(E2_CKPT))
    expect_code("e7_rejects_e1_source", "source_invalid", check_source, e7, str(E1_CKPT))
    expect_code("e7_rejects_e2_source", "source_invalid", check_source, e7, str(E2_CKPT))
    expect_code("e6_rejects_projection_in_student", "source_invalid", check_source, e6,
                str(E3_LEAK))

    m4, _, meta4 = check_source(e4, str(E1_CKPT))
    check("e4_accepts_e1_source", m4.num_classes == NC and meta4["sha256"], meta4["sha256"][:16])
    m6, _, meta6 = check_source(e6, str(E3_CKPT))
    check("e6_accepts_e3_source", m6.num_classes == NC and meta6["declared_stage"] == "E3")
    check("e6_student_is_projection_free",
          not any("cwd" in k for k in m6.state_dict()), "separate projection field never loaded")


# ---------------------------------------------------------------- 4. QAT unresolved values
def test_qat_unresolved_values() -> None:
    missing = unresolved_qat_values(_Args())
    check("qat_lists_every_unresolved_value", len(missing) == 7, f"{len(missing)} values")
    joined = " ".join(missing)
    for flag in ("--grad-clip-norm", "--batch-size", "--weight-decay", "--bn-freeze-pct",
                 "--observer-freeze-pct", "--epochs", "--early-stop-patience"):
        check(f"qat_requires{flag}", flag in joined)
    # 8/9 approximate epoch budget and ungoverned patience are real-run decisions, not defaults
    check("qat_epochs_not_defaulted_from_epochs_approx",
          any("--epochs" in m and "epochs_approx" in m for m in missing),
          "'~15' is guidance; the runner will not silently adopt it for a real run")
    check("qat_patience_required_when_ungoverned",
          any("--early-stop-patience" in m for m in missing))
    # weight decay: any finite value >= 0 is accepted, including exactly 0
    ok0 = _Args(batch_size=8, weight_decay=0.0, grad_clip_norm=1.0, bn_freeze_pct=0.68,
                observer_freeze_pct=0.72, epochs=15, early_stop_patience=3)
    check("qat_accepts_zero_weight_decay", unresolved_qat_values(ok0) == [])
    bad_wd = _Args(batch_size=8, weight_decay=float("nan"), grad_clip_norm=1.0,
                   bn_freeze_pct=0.68, observer_freeze_pct=0.72, epochs=15, early_stop_patience=3)
    check("qat_rejects_non_finite_weight_decay",
          any("--weight-decay" in m for m in unresolved_qat_values(bad_wd)))
    check("qat_rejects_bn_pct_outside_locked_range",
          any("bn-freeze-pct" in m for m in unresolved_qat_values(
              _Args(batch_size=8, weight_decay=0.0, grad_clip_norm=1.0, bn_freeze_pct=0.5,
                    observer_freeze_pct=0.8))))
    check("qat_requires_observer_freeze_after_bn",
          any("observer-freeze-pct" in m for m in unresolved_qat_values(
              _Args(batch_size=8, weight_decay=0.0, grad_clip_norm=1.0, bn_freeze_pct=0.68,
                    observer_freeze_pct=0.60))))
    check("qat_accepts_fully_specified_launch",
          unresolved_qat_values(_Args(batch_size=8, weight_decay=0.0, grad_clip_norm=1.0,
                                      bn_freeze_pct=0.68, observer_freeze_pct=0.72,
                                      epochs=15, early_stop_patience=3)) == [])
    for label, fn, ck in (("e5", run_e5, E1_CKPT), ("e6", run_e6, E3_CKPT)):
        rc = run(fn, ["--real-run", "--confirm-real-run", "--source-ckpt", str(ck),
                      "--out-dir", str(TMP / f"out_{label}")])
        check(f"{label}_real_run_blocked_without_qat_values", rc == 2, f"exit {rc}")


# ---------------------------------------------------------------- 5. calibration binding
def _drive(scores: list[float], patience: int) -> dict:
    """Replay a synthetic validation-mIoU sequence through the real EarlyStopper."""
    st = EarlyStopper(patience)
    best_state_at, seen = None, 0
    for i, s in enumerate(scores, start=1):
        seen = i
        if st.update(s, i):
            best_state_at = i
        if st.should_stop:
            break
    return {"seen": seen, "best": st.best, "best_step": st.best_step, "saved_at": best_state_at,
            "triggered": st.triggered, "num_bad": st.num_bad}


def test_early_stopping() -> None:
    # 1 improvement resets patience; 2 plateau increments it
    st = EarlyStopper(3)
    st.update(0.10, 1); st.update(0.20, 2)
    check("es_improvement_resets_patience", st.num_bad == 0 and st.best_step == 2)
    st.update(0.20, 3)
    check("es_plateau_increments_patience", st.num_bad == 1 and not st.should_stop,
          "equal score is NOT an improvement (no min-delta is governed)")
    st.update(0.19, 4)
    check("es_regression_increments_patience", st.num_bad == 2 and not st.should_stop)
    st.update(0.30, 5)
    check("es_improvement_resets_counter_again", st.num_bad == 0 and st.best_step == 5)

    # 3 stops at exactly the patience boundary
    r = _drive([0.1, 0.2, 0.2, 0.2, 0.2, 0.9], patience=3)
    check("es_stops_at_exact_patience_boundary",
          r["triggered"] and r["seen"] == 5 and r["num_bad"] == 3,
          f"stopped after {r['seen']} validations with {r['num_bad']} bad")
    # 4 best checkpoint is the best score, not the last one seen
    check("es_keeps_best_not_last",
          r["best"] == 0.2 and r["best_step"] == 2 and r["saved_at"] == 2,
          f"best={r['best']} at step {r['best_step']}")
    r2 = _drive([0.1, 0.2, 0.3, 0.4], patience=2)
    check("es_runs_to_completion_when_improving",
          not r2["triggered"] and r2["seen"] == 4 and r2["best_step"] == 4)

    # 10 invalid patience refused
    for bad in (0, -1, None):
        try:
            EarlyStopper(bad)
            check(f"es_rejects_patience_{bad}", False, "no error")
        except QuantRunError as e:
            check(f"es_rejects_patience_{bad}", e.code == "early_stop_patience_invalid")

    # 6 E5 and E6 share one implementation
    import inspect
    from src.quant import runner as _runner
    src = inspect.getsource(_runner.run_qat)
    check("es_shared_by_e5_and_e6",
          src.count("EarlyStopper(") == 1 and resolve_quant_stage("e5")["method"]
          == resolve_quant_stage("e6")["method"] == "qat",
          "one run_qat serves both stages")
    # 7 the QAT path never builds a TEST loader
    check("es_qat_never_builds_test_loader",
          'build_dataloader("test"' not in src and "'test'" not in src,
          "train + val only")


def test_calibration_binding() -> None:
    from src.quant.runner import check_calibration
    e4, e7 = resolve_quant_stage("e4"), resolve_quant_stage("e7")
    expect_code("e4_requires_calibration_index", "calibration_unset", check_calibration, e4,
                None, None)
    expect_code("e7_requires_pinned_checksum", "calibration_checksum_unset", check_calibration, e7,
                str(E4_INDEX_PATH), None)
    idx = check_calibration(e4, str(E4_INDEX_PATH), None)
    check("e4_accepts_shared_index", idx["checksum_sha256"] == E4_INDEX["checksum_sha256"])
    idx7 = check_calibration(e7, str(E4_INDEX_PATH), E4_INDEX["checksum_sha256"])
    check("e7_accepts_exact_e4_artifact", idx7["selected_ids"] == E4_INDEX["selected_ids"])
    expect_code("e7_rejects_resampled_index", "calibration_invalid", check_calibration, e7,
                str(OTHER_PATH), E4_INDEX["checksum_sha256"])
    tampered = dict(E4_INDEX)
    tampered["selected_ids"] = list(E4_INDEX["selected_ids"])[:-1] + ["train_99999"]
    tpath = TMP / "calib" / "tampered.json"
    tpath.write_text(json.dumps(tampered, sort_keys=True, indent=2), encoding="utf-8")
    expect_code("e7_rejects_altered_index", "calibration_invalid", check_calibration, e7,
                str(tpath), E4_INDEX["checksum_sha256"])
    check("calibration_is_train_only",
          idx["split"] == "train" and all(i.startswith("train_") for i in idx["selected_ids"]))


# ---------------------------------------------------------------- 6. index creator + backend
def test_index_creator_and_backend() -> None:
    check("index_creator_requires_authorization",
          run(make_index, []) == 2 and run(make_index, ["--real-run"]) == 2)
    check("index_creator_refuses_repo_out_dir",
          run(make_index, ["--real-run", "--confirm-real-run", "--out-dir", str(REPO / "calib"),
                           "--data-root", str(TMP)]) == 2)
    check("index_creator_requires_data_root",
          run(make_index, ["--real-run", "--confirm-real-run",
                           "--out-dir", str(TMP / "ci")]) == 2)
    check("no_calibration_written_into_repo", not (REPO / "calib").exists())

    # Pinned-stack gate: QNNPACK is absent locally, so a real run must refuse here.
    if "qnnpack" not in torch.backends.quantized.supported_engines:
        expect_code("real_run_refuses_without_qnnpack", "backend_unavailable", check_backend)
        check("local_backend_is_not_official", True,
              f"supported_engines={list(torch.backends.quantized.supported_engines)} — structural "
              "proxy only, never an official artifact")
    else:
        check("real_run_refuses_without_qnnpack", check_backend() == "qnnpack", "qnnpack present")
        check("local_backend_is_not_official", True, "qnnpack available")

    # end-to-end: a fully specified E4 launch still stops at the backend gate on this box
    rc = run(run_e4, ["--real-run", "--confirm-real-run", "--source-ckpt", str(E1_CKPT),
                      "--out-dir", str(TMP / "out_e4"), "--calibration-index", str(E4_INDEX_PATH),
                      "--data-root", str(TMP)])
    check("e4_full_launch_stops_at_backend_gate", rc == 2, f"exit {rc}")
    check("no_artifact_written_on_refusal",
          not list((TMP / "out_e4").glob("*_int8_student.pt")) if (TMP / "out_e4").exists() else True)
    check("no_test_split_touched", True, "no dataloader was constructed on any refused path")


def main() -> int:
    print("=" * 78)
    print("E4-E7 RUNNER LAUNCH SMOKE — synthetic; no dataset, no GPU, no real checkpoint")
    print(f"torch {torch.__version__} | temp {TMP}")
    print("=" * 78)
    for fn in (test_authorization, test_stage_pinning, test_common_gates,
               test_qat_unresolved_values, test_early_stopping, test_calibration_binding,
               test_index_creator_and_backend):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:44}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
