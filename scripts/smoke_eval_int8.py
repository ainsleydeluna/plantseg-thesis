#!/usr/bin/env python3
"""Synthetic verification of the INT8 (E4-E7) evaluation path. No PlantSeg data, no GPU.

Artifacts are produced locally in the EXACT schema `src/quant/runner.py` writes, in a temp dir
outside the repository. Every failure case is proven to raise before the dataset adapter is built,
using the evaluator's own injectable `Counters`. The PlantSeg TEST split is never listed or read.

LOCAL BACKEND: this build ships onednn only, so the OFFICIAL path (`require_qnnpack=True`) refuses
here — that refusal is itself asserted. The structural load/forward round-trip runs on the onednn
proxy and is NON-OFFICIAL evidence; the production gate is never weakened.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from scripts.evaluate_model import (INT8_PRECISIONS, SUPPORTED_PRECISIONS, CliError,  # noqa: E402
                                    Counters, build_parser, run, validate_cli_args)
from src.eval.model_loading import (CheckpointError, Int8ArtifactInfo, load_int8_student,  # noqa: E402
                                    select_int8_backend)
from src.eval.stage_artifacts import StageArtifactError  # noqa: E402
from src.models.student import build_student  # noqa: E402
from src.quant.prepare import convert_model, prepare_ptq, prepare_qat_model  # noqa: E402
from src.seeds import set_seed  # noqa: E402

NC = 116
TMP = Path(tempfile.mkdtemp(prefix="smoke_eval_int8_"))
QNNPACK = "qnnpack" in torch.backends.quantized.supported_engines
PROXY = "onednn" if "onednn" in torch.backends.quantized.supported_engines else None
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect(name: str, exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no exception raised")
    except exc as e:
        check(name, True, type(e).__name__)
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


# ---------------------------------------------------------------- fixtures
def make_int8_artifact(stage: str, method: str, name: str) -> Path:
    """Produce the EXACT runner schema: {stage, quantization, num_classes, model:<state_dict>}."""
    set_seed(42)
    if PROXY and not QNNPACK:
        torch.backends.quantized.engine = PROXY
    student = build_student(pretrained=False)
    prepared = (prepare_ptq(student, select_backend=False) if method == "ptq"
                else prepare_qat_model(student, select_backend=False))
    if method == "qat":
        prepared.train()
        with torch.no_grad():
            prepared(torch.randn(1, 3, 64, 64))     # size the per-channel fake-quant qparams
    converted = convert_model(prepared)
    p = TMP / name
    torch.save({"stage": stage, "quantization": method, "num_classes": NC,
                "model": converted.state_dict()}, p)
    return p


def make_prov(name: str, stage: str, source: str, method: str, art: Path, **over) -> Path:
    from src.eval.model_loading import sha256_file
    payload = {"stage": stage, "source_stage": source, "quantization_method": method,
               "random_init": False, "num_classes": NC, "backend": "qnnpack",
               "converted_artifact": str(art), "converted_artifact_sha256": sha256_file(art)}
    if stage in ("E6", "E7"):
        payload["cwd_projection_loaded"] = False
    payload.update(over)
    p = TMP / name
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def args_for(**kw):
    base = dict(stage="E4", precision="int8_ptq", split="val", out_dir=str(TMP / "out"),
                artifact_status="smoke", device="cpu", batch_size=2)
    base.update(kw)
    argv = []
    for k, v in base.items():
        if v is True:
            argv.append(f"--{k.replace('_', '-')}")
        elif v not in (None, False):
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------- 1. CLI contract
def test_cli() -> None:
    check("int8_precisions_supported",
          set(INT8_PRECISIONS) <= set(SUPPORTED_PRECISIONS)
          and "fp32" in SUPPORTED_PRECISIONS, str(SUPPORTED_PRECISIONS))
    # argparse `choices` already blocks an unknown precision at parse time; bypass it to prove
    # validate_cli_args carries its own guard rather than relying on the parser alone.
    bad_prec = args_for(stage="E1", precision="fp32", random_init=True)
    bad_prec.precision = "int4"
    expect("unsupported_precision_rejected", CliError, validate_cli_args, bad_prec)
    check("parser_also_restricts_precision",
          "int4" not in str(build_parser().parse_known_args(["--out-dir", "x"])[0].precision))
    expect("int8_without_provenance_rejected", CliError, validate_cli_args,
           args_for(stage="E4", precision="int8_ptq"))
    expect("int8_with_checkpoint_and_provenance_rejected", CliError, validate_cli_args,
           args_for(precision="int8_ptq", provenance="p.json", checkpoint="c.pt"))
    expect("int8_with_random_init_rejected", CliError, validate_cli_args,
           args_for(precision="int8_ptq", provenance="p.json", random_init=True))
    expect("int8_on_cuda_rejected", CliError, validate_cli_args,
           args_for(precision="int8_ptq", provenance="p.json", device="cuda"))
    expect("provenance_on_fp32_rejected", CliError, validate_cli_args,
           args_for(stage="E1", precision="fp32", provenance="p.json", checkpoint="c.pt"))
    validate_cli_args(args_for(precision="int8_ptq", provenance="p.json"))
    check("int8_with_provenance_accepted", True)

    # FP32 behaviour unchanged
    validate_cli_args(args_for(stage="E1", precision="fp32", random_init=True))
    check("fp32_random_init_smoke_unchanged", True)
    expect("fp32_needs_one_source", CliError, validate_cli_args,
           args_for(stage="E1", precision="fp32"))
    expect("fp32_both_sources_rejected", CliError, validate_cli_args,
           args_for(stage="E1", precision="fp32", random_init=True, checkpoint="c.pt"))

    # test-split guards untouched
    expect("test_requires_confirmation", CliError, validate_cli_args,
           args_for(stage="E4", precision="int8_ptq", provenance="p.json", split="test",
                    artifact_status="official"))
    expect("official_test_forbids_max_samples", CliError, validate_cli_args,
           args_for(stage="E4", precision="int8_ptq", provenance="p.json", split="test",
                    artifact_status="official", confirm_test_split=True, max_samples=10))
    expect("test_forbids_random_init", CliError, validate_cli_args,
           args_for(stage="E1", precision="fp32", split="test", artifact_status="official",
                    confirm_test_split=True, random_init=True))
    expect("test_refuses_smoke_status", CliError, validate_cli_args,
           args_for(stage="E4", precision="int8_ptq", provenance="p.json", split="test",
                    artifact_status="smoke", confirm_test_split=True))


# ---------------------------------------------------------------- 2. serialization round-trip
def test_serialization() -> None:
    art = make_int8_artifact("E4", "ptq", "e4.pt")
    obj = torch.load(art, map_location="cpu", weights_only=False)
    check("artifact_uses_runner_schema",
          sorted(obj) == ["model", "num_classes", "quantization", "stage"]
          and isinstance(obj["model"], dict), str(sorted(obj)))

    from src.eval.stage_artifacts import validate_int8_artifact
    resolved = validate_int8_artifact("E4", make_prov("e4.json", "E4", "E1", "ptq", art))
    model, info = load_int8_student(resolved, require_qnnpack=False)
    check("int8_roundtrip_loads", isinstance(info, Int8ArtifactInfo) and info.stage == "E4",
          f"{info.method}/{info.backend}")
    with torch.no_grad():
        y = model(torch.randn(1, 3, 64, 64))
    check("int8_structural_forward_116ch", tuple(y.shape) == (1, NC, 64, 64), str(tuple(y.shape)))

    # a QAT artifact round-trips through the QAT skeleton
    art5 = make_int8_artifact("E5", "qat", "e5.pt")
    r5 = validate_int8_artifact("E5", make_prov("e5.json", "E5", "E1", "qat", art5))
    m5, i5 = load_int8_student(r5, require_qnnpack=False)
    check("qat_artifact_roundtrips", i5.method == "qat" and i5.stage == "E5")

    # alternate representations are refused
    alt = TMP / "alt.pt"
    torch.save(torch.nn.Linear(2, 2), alt)
    r_alt = dict(r5); r_alt["artifact_path"] = alt
    expect("serialized_module_rejected", CheckpointError, load_int8_student, r_alt,
           require_qnnpack=False)
    bare = TMP / "bare.pt"
    torch.save({"model": obj["model"]}, bare)
    r_bare = dict(r5); r_bare["artifact_path"] = bare
    expect("missing_schema_keys_rejected", CheckpointError, load_int8_student, r_bare,
           require_qnnpack=False)


# ---------------------------------------------------------------- 3. backend / device gate
def test_backend_gate() -> None:
    if QNNPACK:
        check("official_backend_selected", select_int8_backend() == "qnnpack")
        check("pinned_qnnpack_forward_available", True, "qnnpack present locally")
    else:
        expect("official_int8_refuses_without_qnnpack", CheckpointError, select_int8_backend)
        check("proxy_is_non_official", PROXY is not None and PROXY != "qnnpack",
              f"structural proxy={PROXY}; PINNED-QNNPACK FORWARD — DEFERRED TO OFFICIAL STACK")
        art = TMP / "e4.pt"
        from src.eval.stage_artifacts import validate_int8_artifact
        r = validate_int8_artifact("E4", TMP / "e4.json")
        expect("loader_refuses_official_backend_locally", CheckpointError, load_int8_student, r,
               require_qnnpack=True)


# ---------------------------------------------------------------- 4. ordering: fail before dataset
def test_fails_before_dataset() -> None:
    art = TMP / "e6.pt"
    make_int8_artifact("E6", "qat", "e6.pt")
    prov_ok = make_prov("e6.json", "E6", "E3", "qat", art)

    def run_counted(a):
        c = Counters(dataset=[], model=[])
        try:
            run(a, counters=c)
        finally:
            globals()["_LAST"] = c
        return c

    # tampered artifact -> hash mismatch, before any dataset work
    torch.save({"stage": "E6", "quantization": "qat", "num_classes": NC, "model": {"x": torch.ones(1)}},
               art)
    expect("tampered_artifact_rejected", StageArtifactError, run_counted,
           args_for(stage="E6", precision="int8_qat", provenance=str(prov_ok)))
    check("no_dataset_built_on_tampered_artifact", globals()["_LAST"].dataset == [],
          "dataset adapter never constructed")

    # wrong source stage
    art2 = make_int8_artifact("E6", "qat", "e6b.pt")
    bad_src = make_prov("e6_badsrc.json", "E6", "E1", "qat", art2)
    expect("wrong_source_rejected_before_dataset", StageArtifactError, run_counted,
           args_for(stage="E6", precision="int8_qat", provenance=str(bad_src)))
    check("no_dataset_built_on_wrong_source", globals()["_LAST"].dataset == [])

    # stage mismatch between CLI and provenance
    good6 = make_prov("e6c.json", "E6", "E3", "qat", art2)
    expect("stage_mismatch_rejected_before_dataset", StageArtifactError, run_counted,
           args_for(stage="E7", precision="int8_ptq", provenance=str(good6)))
    check("no_dataset_built_on_stage_mismatch", globals()["_LAST"].dataset == [])

    # missing cwd_projection_loaded flag on an E6 artifact
    leak = make_prov("e6_leak.json", "E6", "E3", "qat", art2, cwd_projection_loaded=True)
    expect("projection_flag_rejected_before_dataset", StageArtifactError, run_counted,
           args_for(stage="E6", precision="int8_qat", provenance=str(leak)))
    check("no_dataset_built_on_projection_flag", globals()["_LAST"].dataset == [])

    # backend gate fires before dataset construction too (local box has no qnnpack)
    if not QNNPACK:
        expect("backend_gate_before_dataset", CheckpointError, run_counted,
               args_for(stage="E6", precision="int8_qat", provenance=str(good6)))
        check("no_dataset_built_on_backend_refusal", globals()["_LAST"].dataset == [])


def main() -> int:
    print("=" * 78)
    print("INT8 EVALUATION PATH SMOKE — synthetic artifacts; no PlantSeg data, no GPU")
    print(f"torch {torch.__version__} | engines={list(torch.backends.quantized.supported_engines)}")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_cli, test_serialization, test_backend_gate, test_fails_before_dataset):
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
