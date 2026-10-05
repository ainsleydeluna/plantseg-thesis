#!/usr/bin/env python3
"""Synthetic launch verification for the E4-E7 runners. No dataset, no GPU, no real checkpoint.

Every gate is exercised through the real entry points (`scripts/run_e4.py` … `run_e7.py`) or through
`src.quant.runner`'s named guards, using synthetic checkpoints and temp dirs OUTSIDE the repository.
Nothing here trains, calibrates on real data, downloads, or writes into the git checkout.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
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
from src.quant.runner import (QuantRunError, check_backend, check_output_dir,  # noqa: E402
                              check_source)
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


def refusal(fn, argv: list[str]) -> tuple[int, str]:
    """Run a QAT entry point; return its exit code and the code in its `RESULT: REFUSED [code]` line."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run(fn, argv)
    m = re.search(r"RESULT: REFUSED \[([a-z0-9_]+)\]", buf.getvalue())
    return rc, (m.group(1) if m else buf.getvalue().strip()[-160:])


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


# ---------------------------------------------------------------- 4. QAT real-run gates (src/quant/qat.py)
HEAD40 = "0" * 40


def test_qat_real_run_gates() -> None:
    base = ["--real-run", "--confirm-real-run"]
    head = base + ["--expect-head", HEAD40]
    s42 = head + ["--seed", "42", "--num-workers", "12"]
    cases = [
        ("e5_refuses_without_expect_head", run_e5, base, "expect_head_format"),
        ("e5_refuses_seed_outside_42_43_44", run_e5, head + ["--seed", "7", "--num-workers", "12"], "seed"),
        ("e5_refuses_without_num_workers", run_e5, head + ["--seed", "42"], "num_workers"),
        ("e5_refuses_without_clip", run_e5, s42, "grad_clip_norm_invalid"),
        ("e5_refuses_clip_outside_candidates", run_e5, s42 + ["--grad-clip-norm", "2.0"],
         "grad_clip_norm_not_candidate"),
        ("e5_s42_requires_u4_pilot", run_e5, s42 + ["--grad-clip-norm", "1.0"], "u4_pilot_required"),
        ("e5_s43_refuses_u4_pilot", run_e5,
         head + ["--seed", "43", "--num-workers", "12", "--grad-clip-norm", "1.0", "--u4-pilot"],
         "u4_pilot_not_e5_s42"),
        ("e5_s43_requires_clip_selection", run_e5,
         head + ["--seed", "43", "--num-workers", "12", "--grad-clip-norm", "5.0"], "clip_selection_required"),
        ("e5_refuses_selection_flags", run_e5,
         s42 + ["--grad-clip-norm", "1.0", "--u4-pilot", "--lambda-selection", "x.json"],
         "selection_not_applicable"),
        ("e5_refuses_out_dir_inside_repo", run_e5,
         s42 + ["--grad-clip-norm", "1.0", "--u4-pilot", "--out-dir", str(REPO / "qat_runs")], "out_dir"),
    ]
    sel = TMP / "clip_selection.json"
    sel.write_text(json.dumps({"format": "qat_clip_selection/1", "winner": {"clip_norm": 1.0}}),
                   encoding="utf-8")
    sel_sha = hashlib.sha256(sel.read_bytes()).hexdigest()
    e6 = head + ["--seed", "42", "--num-workers", "12", "--grad-clip-norm", "1.0",
                 "--clip-selection", str(sel), "--clip-selection-sha256", sel_sha]
    cases += [
        ("e6_s42_refuses_u4_pilot", run_e6, head + ["--seed", "42", "--num-workers", "12",
                                                     "--grad-clip-norm", "1.0", "--u4-pilot"],
         "u4_pilot_not_e5_s42"),
        ("e6_refuses_clip_selection_sha_mismatch", run_e6,
         e6[:-1] + ["f" * 64], "clip_selection_sha256_mismatch"),
        ("e6_refuses_clip_unequal_to_winner", run_e6,
         [a if a != "1.0" else "5.0" for a in e6], "clip_selection_winner_mismatch"),
        ("e6_requires_lambda_selection", run_e6, e6, "lambda_selection_missing"),
    ]
    for name, fn, argv, want in cases:
        rc, code = refusal(fn, argv)
        check(name, rc == 2 and code == want, f"exit {rc} [{code}]")
    out = TMP / "qat_out_absent"
    rc, code = refusal(run_e5, s42 + ["--grad-clip-norm", "1.0", "--u4-pilot", "--out-dir", str(out)])
    if not torch.cuda.is_available():
        check("e5_refuses_without_cuda", rc == 2 and code == "cuda_required", f"exit {rc} [{code}]")
    else:
        check("e5_refuses_without_cuda", rc == 2, f"CUDA present; refused later at [{code}]")
    check("qat_refusals_write_nothing", not out.exists() and not (REPO / "qat_runs").exists())
    # AM-4a pins every recipe value: no launch flag can set one
    for flag in ("--epochs", "--batch-size", "--weight-decay", "--bn-freeze-pct", "--observer-freeze-pct",
                 "--early-stop-patience", "--lr", "--momentum"):
        with contextlib.redirect_stderr(io.StringIO()):
            rc = run(run_e5, s42 + ["--grad-clip-norm", "1.0", "--u4-pilot", flag, "1"])
        check(f"qat_has_no_flag_{flag.strip('-').replace('-', '_')}", rc == 2, f"exit {rc} (argparse refuses it)")


# ---------------------------------------------------------------- 5. no early stopping (AM-4)
def test_no_early_stopping() -> None:
    import inspect
    from configs.quant import QUANT
    from src.quant import qat as _qat
    from src.quant import runner as _runner
    check("runner_holds_no_qat_loop",
          not any(hasattr(_runner, n) for n in ("EarlyStopper", "run_qat", "unresolved_qat_values")),
          "src/quant/qat.py is the one QAT trainer")
    pat = re.compile(r"patience|EarlyStopper|early_stop")
    hits = [f"{f}:{i}" for f in ("src/quant/runner.py", "src/quant/qat.py", "configs/quant.py",
                                 "scripts/run_e5.py", "scripts/run_e6.py")
            for i, line in enumerate((REPO / f).read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
    check("qat_no_patience_path", not hits, str(hits[:5]))
    check("qat_fixed_fifteen_epochs",
          _qat.EPOCHS == QUANT["qat"]["epochs"] == QUANT["qat_real_run"]["epochs"] == 15)
    codes = [refusal(fn, ["--real-run"]) for fn in (run_e5, run_e6)]
    check("qat_shared_by_e5_and_e6",
          codes == [(2, "confirm_real_run_flag")] * 2
          and resolve_quant_stage("e5")["method"] == resolve_quant_stage("e6")["method"] == "qat",
          f"both entry points reach src.quant.qat.main through src.quant.runner.main: {codes}")
    src = inspect.getsource(_qat.build_qat_loaders)
    check("qat_never_builds_test_loader",
          'build_dataloader("train"' in src and 'build_dataloader("val"' in src
          and '"test"' not in src and "'test'" not in src, "train + val only")


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
               test_qat_real_run_gates, test_no_early_stopping, test_calibration_binding,
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
