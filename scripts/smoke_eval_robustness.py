#!/usr/bin/env python3
"""Synthetic verification of the corruption-robustness execution surface. No PlantSeg, no GPU.

Synthetic uint8 images/masks and temp fixtures only. No statistics test is called: inference stays
in the statistics stage.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.evaluate_corruptions import CorruptionCliError, build_parser, validate_args  # noqa: E402
from src.eval.robustness import (INFERENTIAL_ROBUSTNESS_PAIR, RobustnessError,  # noqa: E402
                                 apply_corruption, corruption_type_miou, inferential_pair_error,
                                 is_primary_severity, miou_c, rcd, real_corruption_transform,
                                 registered_corruptions, rpd, validate_condition)
from src.eval.stage_artifacts import OFFICIAL_ROWS  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="smoke_eval_robust_"))
results: list[tuple[str, bool, str]] = []
FROZEN = ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog")


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect(name: str, exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no exception raised")
    except exc as e:
        code = getattr(e, "code", type(e).__name__)
        check(name, True, str(code))
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


def deterministic_transform(seed: int = 7):
    """A stand-in corruption: deterministic, uint8-in/uint8-out. NOT a registered definition."""
    def _t(img: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(seed)
        noise = rng.randint(0, 16, size=img.shape, dtype=np.uint8)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return _t


# ---------------------------------------------------------------- 1. vocabulary / severity
def test_vocabulary() -> None:
    check("frozen_vocabulary_exact", registered_corruptions() == FROZEN, str(registered_corruptions()))
    for c in FROZEN:
        validate_condition(c, 1, official=True)
    check("all_registered_ids_accepted", True)
    expect("alias_rejected", RobustnessError, validate_condition, "brightness_variation", 1,
           official=True)
    expect("hyphen_alias_rejected", RobustnessError, validate_condition, "motion-blur", 1,
           official=True)
    expect("unknown_corruption_rejected", RobustnessError, validate_condition, "snow", 1,
           official=True)

    for s in (1, 2, 3):
        c = validate_condition("fog", s, official=True)
        check(f"severity_{s}_inferential", c["role"] == "inferential" and is_primary_severity(s))
    c4 = validate_condition("fog", 4, official=True)
    check("severity_4_descriptive_only",
          c4["role"] == "descriptive_only" and not is_primary_severity(4))
    expect("severity_5_rejected", RobustnessError, validate_condition, "fog", 5, official=True)
    expect("severity_0_rejected", RobustnessError, validate_condition, "fog", 0, official=True)
    check("condition_schema_shape",
          set(validate_condition("fog", 2, official=True)) >=
          {"type", "name", "severity"} and c4["type"] == "corruption")


# ---------------------------------------------------------------- 2. ordering contract
def test_ordering() -> None:
    img = (np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    mask = np.full((8, 8), 3, dtype=np.uint8)
    mask[0, 0] = 255
    mask_before = mask.copy()

    out, m = apply_corruption(img, deterministic_transform(), mask)
    check("corruption_returns_uint8", out.dtype == np.uint8 and out.shape == img.shape)
    check("mask_bytes_unchanged", np.array_equal(m, mask_before) and np.array_equal(mask, mask_before),
          "ground truth is never edited")
    check("ignore_255_preserved", m[0, 0] == 255)

    out2, _ = apply_corruption(img, deterministic_transform(), mask)
    check("corruption_deterministic_under_fixed_seed", np.array_equal(out, out2))

    # normalized float input is refused -- corruption must precede normalization
    normalized = ((img.astype(np.float32) / 255.0) - 0.485) / 0.229
    expect("normalized_input_refused", RobustnessError, apply_corruption, normalized,
           deterministic_transform())
    expect("non_rgb_refused", RobustnessError, apply_corruption,
           np.zeros((8, 8), dtype=np.uint8), deterministic_transform())
    expect("bad_transform_output_refused", RobustnessError, apply_corruption, img,
           lambda a: a.astype(np.float32))

    # the production transform is not vendored and says so
    expect("real_transform_not_vendored", RobustnessError, real_corruption_transform, "fog", 2)


# ---------------------------------------------------------------- 3. aggregation math
def test_aggregation() -> None:
    # one type: mean over severities 1-3 only; severity 4 must not leak in
    per_sev = {1: 0.30, 2: 0.20, 3: 0.10, 4: 0.00}
    check("type_miou_averages_sev_1_3", abs(corruption_type_miou(per_sev) - 0.20) < 1e-12,
          f"{corruption_type_miou(per_sev):.4f} (0.0 at severity 4 excluded)")
    expect("type_miou_needs_primary_severities", RobustnessError, corruption_type_miou, {4: 0.5})

    # mIoU-C: equal weight across the five types
    per_corruption = {c: {1: v, 2: v, 3: v} for c, v in zip(FROZEN, (0.10, 0.20, 0.30, 0.40, 0.50))}
    check("miou_c_equal_weights_types", abs(miou_c(per_corruption) - 0.30) < 1e-12,
          f"{miou_c(per_corruption):.4f}")
    # a lopsided severity spread inside one type still contributes exactly one type-weight
    skewed = dict(per_corruption)
    skewed["fog"] = {1: 0.90, 2: 0.30, 3: 0.30, 4: 0.99}
    check("miou_c_severity4_never_leaks", abs(miou_c(skewed) - 0.30) < 1e-12,
          f"{miou_c(skewed):.4f} — 0.99 at severity 4 ignored")
    expect("miou_c_requires_all_types", RobustnessError, miou_c,
           {c: {1: 0.1, 2: 0.1, 3: 0.1} for c in FROZEN[:4]})
    expect("miou_c_rejects_unknown_type", RobustnessError, miou_c,
           {**per_corruption, "snow": {1: 0.1, 2: 0.1, 3: 0.1}})

    # RPD
    check("rpd_formula", abs(rpd(0.50, 0.40) - 20.0) < 1e-12, f"{rpd(0.50, 0.40):.4f}")
    check("rpd_zero_when_unchanged", abs(rpd(0.5, 0.5)) < 1e-12)
    expect("rpd_rejects_nonpositive_clean", RobustnessError, rpd, 0.0, 0.1)

    # rCD: E1 is the reference and scores exactly 1; teacher is excluded
    check("rcd_e1_reference_is_one", abs(rcd("E1", 0.25, 0.25) - 1.0) < 1e-12)
    check("rcd_relative_to_e1", abs(rcd("E6", 0.20, 0.25) - 0.8) < 1e-12)
    expect("rcd_excludes_teacher", RobustnessError, rcd, "TEACHER", 0.2, 0.25)
    expect("rcd_rejects_bad_reference", RobustnessError, rcd, "E6", 0.2, 0.0)


# ---------------------------------------------------------------- 4. stage matrix / inference
def test_stage_matrix() -> None:
    check("inferential_pair_is_e1_e6", INFERENTIAL_ROBUSTNESS_PAIR == ("E1", "E6"))
    check("e1_e6_pair_allowed", inferential_pair_error("E1", "E6") is None
          and inferential_pair_error("E6", "E1") is None, "order-insensitive")
    for a, b in (("E1", "E3"), ("E4", "E7"), ("TEACHER", "E6"), ("E2", "E6")):
        check(f"pair_{a}_{b}_descriptive_only", inferential_pair_error(a, b) is not None)
    src = (REPO / "src/eval/robustness.py").read_text(encoding="utf-8")
    check("no_statistics_invoked_here",
          all(t not in src.lower() for t in ("wilcoxon", "ttest", "holm", "bootstrap")),
          "inferential tests stay in the statistics stage")


# ---------------------------------------------------------------- 5. runner gates
def test_runner_gates() -> None:
    def parse(**kw):
        base = dict(stage="E1", corruption="fog", severity=2, out_dir=str(TMP / "out"),
                    checkpoint=str(TMP / "e1.pt"))
        base.update(kw)
        argv = []
        for k, v in base.items():
            if v is True:
                argv.append(f"--{k.replace('_', '-')}")
            elif v not in (None, False):
                argv += [f"--{k.replace('_', '-')}", str(v)]
        return build_parser().parse_args(argv)

    expect("runner_rejects_unknown_corruption", RobustnessError, validate_args,
           parse(corruption="snow"))
    expect("runner_rejects_severity_5", RobustnessError, validate_args, parse(severity=5))
    expect("runner_test_requires_confirmation", CorruptionCliError, validate_args,
           parse(split="test", artifact_status="official"))
    expect("runner_test_refuses_smoke_status", CorruptionCliError, validate_args,
           parse(split="test", artifact_status="smoke", confirm_test_split=True))
    expect("runner_official_forbids_max_samples", CorruptionCliError, validate_args,
           parse(split="test", artifact_status="official", confirm_test_split=True,
                 max_samples=10))
    cond = validate_args(parse(split="test", artifact_status="official",
                               confirm_test_split=True))
    check("runner_accepts_official_full_test", cond["name"] == "fog" and cond["severity"] == 2)
    check("official_row_authority_is_1561", OFFICIAL_ROWS == 1561, str(OFFICIAL_ROWS))

    # every stage (teacher + E1-E7) is routable through the SAME resolver
    from src.eval.stage_artifacts import resolve_stage_artifact
    kinds = {s: resolve_stage_artifact(s)["kind"] for s in
             ("teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7")}
    check("all_stages_share_one_resolver",
          kinds["teacher"] == "teacher_checkpoint" and kinds["E1"] == "fp32_checkpoint"
          and kinds["E7"] == "int8_artifact", str(len(kinds)) + " stages")

    # a blocked artifact fails before any dataset work: the runner validates the source first
    runner_src = (REPO / "scripts/evaluate_corruptions.py").read_text(encoding="utf-8")
    body = runner_src[runner_src.index("def main("):]      # CALL sites, not the import block
    check("runner_validates_source_before_dataset",
          body.index("resolve_evaluation_source") < body.index("real_corruption_transform"),
          "model source proven before the corruption/dataset stage")
    check("runner_builds_no_dataset_yet",
          "PlantSegEvalDataset" not in runner_src and "build_eval_loader" not in runner_src)


def main() -> int:
    print("=" * 78)
    print("ROBUSTNESS EXECUTION SMOKE — synthetic uint8 fixtures; no PlantSeg, no GPU, no stats")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_vocabulary, test_ordering, test_aggregation, test_stage_matrix,
               test_runner_gates):
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
