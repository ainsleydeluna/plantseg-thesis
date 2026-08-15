#!/usr/bin/env python3
"""Synthetic verification of the E1-E7 stage/artifact evaluation bridge. No dataset, no GPU.

Every fixture is a synthetic checkpoint or provenance JSON in a temp dir outside the repository.
The PlantSeg TEST split is never read: only row COUNTS are reasoned about, never rows.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.eval.model_loading import sha256_file  # noqa: E402
from src.eval.stage_artifacts import (OFFICIAL_ROWS, STAGE_ARTIFACTS, StageArtifactError,  # noqa: E402
                                      governed_paths_error, official_launch_error,
                                      resolve_evaluation_source, resolve_stage_artifact,
                                      validate_fp32_artifact, validate_int8_artifact)
from src.models.student import build_student  # noqa: E402

NC = 116
TMP = Path(tempfile.mkdtemp(prefix="smoke_eval_stage_"))
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect_code(name: str, code: str, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no StageArtifactError raised")
    except StageArtifactError as e:
        check(name, e.code == code, e.code if e.code == code else f"{e.code} != {code}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


def _state() -> dict:
    return {k: v.detach().clone() for k, v in build_student(pretrained=False).state_dict().items()}


def fp32_ckpt(name: str, stage: str | None, **extra) -> Path:
    p = TMP / name
    payload = {"model_state_dict": _state(), "num_classes": NC, **extra}
    if stage:
        payload["stage"] = stage
    torch.save(payload, p)
    return p


def int8_artifact(name: str) -> Path:
    p = TMP / name
    torch.save({"quantization": "int8", "num_classes": NC,
                "model": {"features.0.0._packed_params": torch.zeros(1)}}, p)
    return p


def prov(name: str, stage: str, source: str, method: str, art: Path, **over) -> Path:
    payload = {"stage": stage, "source_stage": source, "quantization_method": method,
               "random_init": False, "num_classes": NC, "backend": "qnnpack",
               "converted_artifact": str(art), "converted_artifact_sha256": sha256_file(art)}
    if stage in ("E6", "E7"):
        payload["cwd_projection_loaded"] = False
    payload.update(over)
    p = TMP / name
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 1. stage contract
def test_stage_contract() -> None:
    check("all_seven_stages_known", sorted(STAGE_ARTIFACTS) == ["E1", "E2", "E3", "E4", "E5",
                                                                "E6", "E7"])
    for s, kind, prec in (("E1", "fp32_checkpoint", "fp32"), ("E2", "fp32_checkpoint", "fp32"),
                          ("E3", "fp32_checkpoint", "fp32"), ("E4", "int8_artifact", "int8_ptq"),
                          ("E5", "int8_artifact", "int8_qat"), ("E6", "int8_artifact", "int8_qat"),
                          ("E7", "int8_artifact", "int8_ptq")):
        spec = resolve_stage_artifact(s)
        check(f"{s}_resolves_correctly", spec["kind"] == kind and spec["precision"] == prec,
              f"{kind}/{prec} from {spec['source_stage']}")
    check("e4_e5_derive_from_e1",
          resolve_stage_artifact("E4")["source_stage"] == "E1"
          and resolve_stage_artifact("E5")["source_stage"] == "E1")
    check("e6_e7_derive_from_e3",
          resolve_stage_artifact("E6")["source_stage"] == "E3"
          and resolve_stage_artifact("E7")["source_stage"] == "E3")
    expect_code("teacher_not_invented", "unknown_stage", resolve_stage_artifact, "teacher")
    expect_code("unknown_stage_rejected", "unknown_stage", resolve_stage_artifact, "E9")


# ---------------------------------------------------------------- 2. artifact-kind separation
def test_kind_separation() -> None:
    e1 = fp32_ckpt("e1.pt", None)
    art = int8_artifact("e4_int8.pt")
    pv = prov("e4_prov.json", "E4", "E1", "ptq", art)

    check("e1_fp32_accepted", validate_fp32_artifact("E1", e1)["spec"]["stage"] == "E1")
    check("e4_int8_accepted", validate_int8_artifact("E4", pv)["artifact_path"] == art)

    # FP32 stage cannot consume an INT8 artifact, and vice versa
    expect_code("fp32_stage_rejects_int8_artifact", "checkpoint_is_quantized",
                validate_fp32_artifact, "E1",
                _save_quantized_ckpt(TMP / "quantized_as_fp32.pt"))
    expect_code("int8_stage_rejects_raw_checkpoint", "provenance_required",
                resolve_evaluation_source, "E4", checkpoint=str(e1))
    expect_code("fp32_stage_requires_checkpoint", "checkpoint_required",
                resolve_evaluation_source, "E1")
    expect_code("int8_validator_refuses_fp32_stage", "stage_is_not_int8",
                validate_int8_artifact, "E1", pv)
    expect_code("fp32_validator_refuses_int8_stage", "stage_is_not_fp32",
                validate_fp32_artifact, "E4", e1)


def _save_quantized_ckpt(p: Path) -> Path:
    st = _state()
    st["features.0.0._packed_params"] = torch.zeros(1)
    torch.save({"model_state_dict": st, "num_classes": NC}, p)
    return p


# ---------------------------------------------------------------- 3. provenance validation
def test_provenance_validation() -> None:
    art = int8_artifact("e6_int8.pt")
    good = prov("e6_prov.json", "E6", "E3", "qat", art)
    check("e6_valid_provenance_accepted",
          validate_int8_artifact("E6", good)["provenance"]["cwd_projection_loaded"] is False)

    expect_code("wrong_stage_metadata_rejected", "stage_mismatch",
                validate_int8_artifact, "E7", good)
    expect_code("e6_wrong_source_rejected", "source_stage_mismatch", validate_int8_artifact, "E6",
                prov("e6_badsrc.json", "E6", "E1", "qat", art))
    expect_code("e4_wrong_source_rejected", "source_stage_mismatch", validate_int8_artifact, "E4",
                prov("e4_badsrc.json", "E4", "E3", "ptq", art))
    expect_code("method_mismatch_rejected", "method_mismatch", validate_int8_artifact, "E6",
                prov("e6_ptq.json", "E6", "E3", "ptq", art))
    expect_code("random_init_artifact_rejected", "random_init_artifact", validate_int8_artifact,
                "E6", prov("e6_rand.json", "E6", "E3", "qat", art, random_init=True))
    expect_code("num_classes_mismatch_rejected", "num_classes_mismatch", validate_int8_artifact,
                "E6", prov("e6_nc.json", "E6", "E3", "qat", art, num_classes=150))
    expect_code("non_qnnpack_backend_rejected", "backend_not_official", validate_int8_artifact,
                "E6", prov("e6_onednn.json", "E6", "E3", "qat", art, backend="onednn"))
    expect_code("missing_projection_flag_rejected", "projection_flag_missing",
                validate_int8_artifact, "E7",
                prov("e7_noflag.json", "E7", "E3", "ptq", art, cwd_projection_loaded=True))
    expect_code("missing_provenance_rejected", "provenance_missing", validate_int8_artifact, "E4",
                TMP / "absent.json")

    # hash mismatch: mutate the artifact after the provenance was written
    art2 = int8_artifact("e5_int8.pt")
    pv2 = prov("e5_prov.json", "E5", "E1", "qat", art2)
    torch.save({"quantization": "int8", "num_classes": NC, "model": {"tampered": torch.ones(1)}},
               art2)
    expect_code("artifact_hash_mismatch_rejected", "artifact_hash_mismatch",
                validate_int8_artifact, "E5", pv2)


# ---------------------------------------------------------------- 4. FP32 stage metadata
def test_fp32_metadata() -> None:
    expect_code("e2_ckpt_rejected_for_e3", "stage_mismatch", validate_fp32_artifact, "E3",
                fp32_ckpt("e2.pt", "E2"))
    check("e3_ckpt_accepted_for_e3",
          validate_fp32_artifact("E3", fp32_ckpt("e3.pt", "E3"))["declared_stage"] == "E3")
    # E3 with the training-only projection leaked into the student state is refused
    leaked = TMP / "e3_leak.pt"
    st = _state(); st["cwd_projection.weight"] = torch.randn(320, 160, 1, 1)
    torch.save({"stage": "E3", "num_classes": NC, "model_state_dict": st}, leaked)
    try:
        validate_fp32_artifact("E3", leaked)
        check("e3_projection_contamination_rejected", False, "no error")
    except Exception as e:  # noqa: BLE001  (CWDProjectionLeak from the export contract)
        check("e3_projection_contamination_rejected", "CWDProjectionLeak" in type(e).__name__,
              type(e).__name__)
    expect_code("missing_checkpoint_rejected", "checkpoint_missing", validate_fp32_artifact, "E1",
                TMP / "nope.pt")


# ---------------------------------------------------------------- 5. official preconditions
def test_official_launch_preconditions() -> None:
    """PRE-RUN facts only. `actual_rows` must not appear here — it does not exist yet."""
    import inspect

    sig = inspect.signature(official_launch_error)
    check("launch_gate_takes_no_actual_rows", "actual_rows" not in sig.parameters,
          f"params={list(sig.parameters)}")
    src = inspect.getsource(official_launch_error)
    check("launch_gate_does_not_invent_actual_rows", "actual_rows" not in src.split('"""')[-1],
          "no post-inference row rule in the pre-run gate")

    ok = official_launch_error(split="test", random_init=False, stage="E4",
                               expected_manifest_rows=OFFICIAL_ROWS)
    check("official_launch_accepts_full_test_request", ok is None, f"manifest={OFFICIAL_ROWS}")
    check("official_rejects_random_init",
          "random_init" in (official_launch_error(split="test", random_init=True, stage="E1",
                                                  expected_manifest_rows=OFFICIAL_ROWS) or ""))
    check("official_rejects_non_test_split",
          "split='test'" in (official_launch_error(split="val", random_init=False, stage="E1",
                                                   expected_manifest_rows=846) or ""))
    check("official_rejects_sample_cap",
          "sample cap" in (official_launch_error(split="test", random_init=False, stage="E1",
                                                 expected_manifest_rows=OFFICIAL_ROWS,
                                                 max_samples=100) or ""))
    check("official_rejects_short_expected_manifest",
          "1561" in (official_launch_error(split="test", random_init=False, stage="E1",
                                           expected_manifest_rows=1560) or ""))

    # The POST-RUN rule stays with the frozen artifact layer, and still refuses a mismatch there.
    from src.eval import artifacts as _artifacts
    finalize_src = inspect.getsource(_artifacts)
    check("actual_rows_owned_by_artifact_layer",
          'd["actual_rows"] != d["expected_rows"]' in finalize_src
          and '"actual_rows": len(result.rows)' in finalize_src,
          "artifacts.py computes actual_rows from the result and enforces equality")
    bridge_src = (REPO / "src/eval/stage_artifacts.py").read_text(encoding="utf-8")
    # Prose may NAME the rule it delegates; what must not exist is an enforcement of it here.
    enforcing = [ln.strip() for ln in bridge_src.splitlines()
                 if "actual_rows" in ln
                 and ln.lstrip().startswith(("if ", "elif ", "assert ", "raise ", "return "))]
    check("bridge_never_enforces_actual_rows", enforcing == [],
          "the bridge holds no second copy of the post-run row rule"
          + (f" — found {enforcing}" if enforcing else ""))
    # model/provenance validation must be constructible-free: it never reaches a dataset adapter
    check("bridge_never_constructs_dataset",
          all(tok not in bridge_src for tok in ("PlantSegEvalDataset", "build_eval_loader",
                                                "DataLoader")),
          "artifact validation cannot touch the dataset, so it always fails first")
    # governed-path gate is delegated, and the protected PDF must not trip it
    err = governed_paths_error()
    check("governed_gate_delegated_and_pdf_allowlisted",
          err is None or "docs/reference/reference.pdf" not in err,
          err or "governed set clean")


def main() -> int:
    print("=" * 78)
    print("E1-E7 EVALUATION BRIDGE SMOKE — synthetic artifacts only; TEST split never read")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_stage_contract, test_kind_separation, test_provenance_validation,
               test_fp32_metadata, test_official_launch_preconditions):
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
