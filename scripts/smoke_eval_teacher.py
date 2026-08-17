#!/usr/bin/env python3
"""Synthetic verification of the teacher clean-evaluation path. No mmseg, no GPU, no dataset.

The teacher is a DESCRIPTIVE REFERENCE: this proves it can be resolved, validated and evaluated as
an ordinary FP32 model, while never being turned into an inferential comparator. All fixtures are
synthetic checkpoints in a temp dir outside the repository; PlantSeg is never touched.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from scripts.evaluate_model import CliError, Counters, build_parser, run, validate_cli_args  # noqa: E402
from src.eval.model_loading import CheckpointError, TeacherEvalModel, load_teacher_model  # noqa: E402
from src.eval.stage_artifacts import (StageArtifactError, expected_model_role,  # noqa: E402
                                      is_descriptive_only, resolve_evaluation_source,
                                      resolve_stage_artifact, validate_fp32_artifact,
                                      validate_teacher_artifact)
from src.models.student import build_student  # noqa: E402

NC = 116
TMP = Path(tempfile.mkdtemp(prefix="smoke_eval_teacher_"))
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


# ---------------------------------------------------------------- stub teacher
class StubMSCAN(nn.Module):
    def __init__(self, dims=(64, 128, 320, 512), strides=(4, 8, 16, 32)):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(3, d, 1, bias=False) for d in dims])
        self.strides = strides

    def forward(self, x):
        return tuple(p(torch.nn.functional.adaptive_avg_pool2d(
            x, (max(x.shape[-2] // s, 1), max(x.shape[-1] // s, 1))))
            for p, s in zip(self.projs, self.strides))


class StubHead(nn.Module):
    def __init__(self, num_classes=NC):
        super().__init__()
        self.num_classes = num_classes
        self.squeeze = nn.Conv2d(128, 512, 1, bias=False)
        self.conv_seg = nn.Conv2d(512, num_classes, 1)

    def forward(self, feats):
        return self.conv_seg(self.squeeze(feats[1]))       # native OS8 logits


class StubSegNeXt(nn.Module):
    def __init__(self, num_classes=NC):
        super().__init__()
        self.backbone = StubMSCAN()
        self.decode_head = StubHead(num_classes)

    def extract_feat(self, x):
        return self.backbone(x)


def stub_builder(_path):
    from src.distill.segnext_teacher import SegNeXtTeacherAdapter
    return SegNeXtTeacherAdapter(StubSegNeXt())


def teacher_ckpt(name: str = "teacher.pth") -> Path:
    p = TMP / name
    torch.save({"meta": {"mmseg_version": "1.2.2"},
                "state_dict": {"backbone.projs.0.weight": torch.randn(4, 3, 1, 1),
                               "decode_head.conv_seg.weight": torch.randn(NC, 8, 1, 1),
                               "decode_head.conv_seg.bias": torch.randn(NC)}}, p)
    return p


def student_ckpt(name: str = "e1.pt") -> Path:
    p = TMP / name
    torch.save({"num_classes": NC,
                "model_state_dict": {k: v.detach().clone()
                                     for k, v in build_student(pretrained=False).state_dict().items()}}, p)
    return p


def args_for(**kw):
    base = dict(stage="teacher", model_role="teacher", precision="fp32", split="val",
                out_dir=str(TMP / "out"), artifact_status="smoke", device="cpu", batch_size=2)
    base.update(kw)
    argv = []
    for k, v in base.items():
        if v is True:
            argv.append(f"--{k.replace('_', '-')}")
        elif v not in (None, False):
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------- 1. resolution
def test_resolution() -> None:
    spec = resolve_stage_artifact("teacher")
    check("teacher_is_its_own_artifact_kind",
          spec["kind"] == "teacher_checkpoint" and spec["precision"] == "fp32", str(spec["kind"]))
    check("teacher_role_is_teacher", expected_model_role("teacher") == "teacher"
          and expected_model_role("E1") == "student")
    check("teacher_marked_descriptive_only",
          is_descriptive_only("teacher") and not is_descriptive_only("E6"))

    tk, sk = teacher_ckpt(), student_ckpt()
    check("teacher_checkpoint_accepted",
          validate_teacher_artifact(tk)["descriptive_only"] is True)
    expect("student_checkpoint_cannot_satisfy_teacher", StageArtifactError,
           validate_teacher_artifact, sk)
    expect("teacher_checkpoint_cannot_satisfy_e1", StageArtifactError,
           validate_fp32_artifact, "E1", tk)
    # an INT8 artifact is not a teacher
    q = TMP / "int8.pt"
    torch.save({"stage": "E4", "quantization": "ptq", "num_classes": NC,
                "model": {"features.0.0._packed_params": torch.zeros(1)}}, q)
    expect("int8_artifact_cannot_satisfy_teacher", StageArtifactError,
           validate_teacher_artifact, q)
    check("resolve_routes_teacher",
          resolve_evaluation_source("teacher", checkpoint=tk)["spec"]["kind"]
          == "teacher_checkpoint")


# ---------------------------------------------------------------- 2. provenance
def test_provenance() -> None:
    tk = teacher_ckpt("t2.pth")
    expect("missing_teacher_checkpoint_refused", StageArtifactError,
           validate_teacher_artifact, TMP / "absent.pth")
    bad = TMP / "malformed.pth"
    bad.write_bytes(b"\x00not-a-torch-file")
    expect("malformed_teacher_checkpoint_refused", StageArtifactError,
           validate_teacher_artifact, bad)
    unrelated = TMP / "unrelated.pth"
    torch.save({"state_dict": {"foo.weight": torch.randn(2, 2)}}, unrelated)
    expect("unrelated_checkpoint_refused", StageArtifactError,
           validate_teacher_artifact, unrelated)
    good = validate_teacher_artifact(tk)
    expect("tampered_hash_refused", StageArtifactError, validate_teacher_artifact, tk,
           expected_sha256="0" * 64)
    check("teacher_hash_recorded", len(good["checkpoint_sha256"]) == 64,
          good["checkpoint_sha256"][:16] + "…")


# ---------------------------------------------------------------- 3. loading / wrapper
def test_loading() -> None:
    check("evaluator_import_pulls_no_mmseg",
          "mmseg" not in sys.modules and "mmcv" not in sys.modules)

    resolved = validate_teacher_artifact(teacher_ckpt("t3.pth"))
    model, _ = load_teacher_model(resolved, builder=stub_builder)
    check("teacher_wrapper_type", isinstance(model, TeacherEvalModel))
    check("teacher_is_eval_mode", not model.training)
    check("teacher_params_frozen", all(not p.requires_grad for p in model.parameters()))

    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    check("wrapper_returns_bchw_116", tuple(y.shape) == (2, NC, 64, 64), str(tuple(y.shape)))
    check("wrapper_returns_tensor_not_teacheroutput", torch.is_tensor(y), type(y).__name__)
    check("feat_s16_not_exposed", not hasattr(y, "feat_s16"),
          "only segmentation logits reach the metric code")
    check("teacher_output_detached", not y.requires_grad)

    # a 150-class (stock ADE20K-shaped) head is refused against the 116-class contract
    def bad_builder(_p):
        from src.distill.segnext_teacher import SegNeXtTeacherAdapter
        return SegNeXtTeacherAdapter(StubSegNeXt(num_classes=150))   # expects 116, gets 150
    from src.distill.segnext_teacher import TeacherArchitectureMismatch
    expect("wrong_class_space_refused", TeacherArchitectureMismatch, load_teacher_model,
           resolved, builder=bad_builder)

    # and the evaluation wrapper independently refuses wrong-channel logits at forward
    class _WrongChannels(nn.Module):
        def __call__(self, x, **kw):
            from src.distill.teacher import TeacherOutput
            return TeacherOutput(logits=torch.randn(x.shape[0], 42, *x.shape[-2:]), feat_s16=None)
        trainable_parameters = staticmethod(lambda: [])
    expect("wrapper_refuses_wrong_logit_channels", CheckpointError,
           TeacherEvalModel(_WrongChannels()), torch.randn(1, 3, 32, 32))

    # the real builder path fails loudly without the teacher stack
    from src.distill.teacher import TeacherStackMissing
    expect("missing_teacher_stack_fails_loudly", TeacherStackMissing, load_teacher_model, resolved)
    # SUPERSEDED: this used to assert mmseg was STILL absent from sys.modules after deliberately
    # invoking the teacher loader. That only held on a machine without the teacher stack; inside the
    # official image mmseg is installed and the loader legitimately imports it, so the old assertion
    # failed on a correct environment.
    #
    # The invariant that actually matters is that ORDINARY STUDENT/EVAL imports never pull the MMSeg
    # stack. A fresh subprocess proves that regardless of whether mmseg is installed, which the
    # in-process check above cannot do once a teacher call has run.
    import subprocess

    probe = ("import sys; import src.models.student, src.eval.model_loading, src.eval.evaluate; "
             "print(int(any(m in sys.modules for m in ('mmseg', 'mmcv', 'mmengine'))))")
    proc = subprocess.run([sys.executable, "-B", "-c", probe], cwd=str(REPO),
                          capture_output=True, text=True)
    check("student_paths_never_import_mmseg",
          proc.returncode == 0 and proc.stdout.strip() == "0",
          "fresh interpreter: student/eval imports pull no mmseg/mmcv/mmengine"
          if proc.returncode == 0 else proc.stderr.strip()[:120])


# ---------------------------------------------------------------- 4. CLI + safety
def test_cli_and_safety() -> None:
    tk = teacher_ckpt("t4.pth")
    validate_cli_args(args_for(checkpoint=str(tk)))
    check("teacher_cli_accepted", True)
    expect("teacher_random_init_refused", CliError, validate_cli_args,
           args_for(random_init=True))
    expect("teacher_without_checkpoint_refused", CliError, validate_cli_args, args_for())
    expect("teacher_int8_precision_refused", CliError, validate_cli_args,
           args_for(precision="int8_ptq", checkpoint=str(tk)))
    expect("teacher_stage_role_mismatch_refused", CliError, validate_cli_args,
           args_for(stage="teacher", model_role="student", checkpoint=str(tk)))
    expect("student_stage_teacher_role_refused", CliError, validate_cli_args,
           args_for(stage="E1", model_role="teacher", checkpoint=str(tk)))
    expect("teacher_test_requires_confirmation", CliError, validate_cli_args,
           args_for(split="test", artifact_status="official", checkpoint=str(tk)))
    expect("teacher_official_test_forbids_cap", CliError, validate_cli_args,
           args_for(split="test", artifact_status="official", confirm_test_split=True,
                    max_samples=10, checkpoint=str(tk)))
    # official status is NOT forbidden for the teacher -- nothing in the contract prohibits it
    validate_cli_args(args_for(split="test", artifact_status="official",
                               confirm_test_split=True, checkpoint=str(tk)))
    check("teacher_official_status_permitted", True,
          "descriptive ROLE is a statistics concern, not an artifact-status ban")

    from src.eval.stage_artifacts import OFFICIAL_ROWS
    check("official_test_count_is_1561", OFFICIAL_ROWS == 1561, str(OFFICIAL_ROWS))

    # ordering: an invalid teacher checkpoint fails before the dataset adapter is built
    ctr = Counters(dataset=[], model=[])
    expect("invalid_teacher_fails_before_dataset", StageArtifactError, run,
           args_for(checkpoint=str(student_ckpt("e1b.pt"))), counters=ctr)
    check("no_dataset_built_for_invalid_teacher", ctr.dataset == [])

    # descriptive-only role is carried, and no inferential machinery is touched here
    src = (REPO / "src/eval/stage_artifacts.py").read_text(encoding="utf-8")
    check("no_inferential_machinery_in_bridge",
          all(t not in src for t in ("wilcoxon", "holm", "paired", "ttest", "rcd")),
          "teacher integration creates no significance test")


def main() -> int:
    print("=" * 78)
    print("TEACHER CLEAN-EVALUATION SMOKE — synthetic; no mmseg, no GPU, no PlantSeg")
    print(f"temp: {TMP}")
    print("=" * 78)
    for fn in (test_resolution, test_provenance, test_loading, test_cli_and_safety):
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
