#!/usr/bin/env python3
"""Synthetic verification of the SegNeXt-B / MSCAN-B teacher adapter. No mmseg, no GPU, no dataset.

The real teacher checkpoint does not exist yet and MMSegmentation is intentionally absent from the
E1/E2/E3 student stack, so every check here runs on stub modules and synthetic checkpoints written
to a temp dir OUTSIDE the repo. What is proven: the adapter's framework-independent behaviour
(checkpoint parsing, Stage-3 resolution by semantics, class-space validation, provenance, frozen
teacher, E3 integration) and that the mmseg-dependent path fails loudly rather than silently.

Kept separate from scripts/smoke_distill.py so teacher-specific checks do not bloat that suite.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.distill import (STAGE3_CHANNELS, FrozenTeacher, SegNeXtTeacherAdapter,  # noqa: E402
                         TeacherArchitectureMismatch, TeacherCheckpointInvalid,
                         TeacherCheckpointMissing, TeacherStackMissing, build_cwd_projection,
                         build_segnext_teacher, load_frozen_teacher, load_teacher_state_dict,
                         segnext_builder, select_stride16_feature)
from src.models.student import build_student  # noqa: E402
from src.seeds import set_seed  # noqa: E402
from src.training.train_distill import distillation_losses, resolve_stage  # noqa: E402

NC = 116
results: list[tuple[str, bool, str]] = []
TMP = Path(tempfile.mkdtemp(prefix="smoke_teacher_"))


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect_raises(name: str, exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no exception raised")
    except exc as e:
        check(name, True, f"{type(e).__name__}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


# ---------------------------------------------------------------- stubs
class StubMSCAN(nn.Module):
    """Stand-in for MSCAN-B: emits four stage features at strides 4/8/16/32."""

    def __init__(self, dims=(64, 128, 320, 512), strides=(4, 8, 16, 32)):
        super().__init__()
        self.dims, self.strides = dims, strides
        self.projs = nn.ModuleList([nn.Conv2d(3, d, 1, bias=False) for d in dims])

    def forward(self, x):
        out = []
        for proj, s in zip(self.projs, self.strides):
            h, w = max(x.shape[-2] // s, 1), max(x.shape[-1] // s, 1)
            out.append(proj(torch.nn.functional.adaptive_avg_pool2d(x, (h, w))))
        return tuple(out)


class StubLightHamHead(nn.Module):
    """Stand-in for LightHamHead: logits at stride 8 from the stride-8 stage feature."""

    def __init__(self, in_ch=128, channels=512, num_classes=NC):
        super().__init__()
        self.num_classes = num_classes
        self.squeeze = nn.Conv2d(in_ch, channels, 1, bias=False)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats):
        return self.conv_seg(self.squeeze(feats[1]))


class StubSegNeXt(nn.Module):
    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone or StubMSCAN()
        self.decode_head = head or StubLightHamHead()

    def extract_feat(self, x):
        return self.backbone(x)


def stub_factory(_config_path, _ckpt_path):
    return StubSegNeXt()


class StubIsolatedNMF(nn.Module):
    """Stand-in for IsolatedNMF2D: one torch.rand basis draw per forward, routed through a stream."""
    SUPPORTS_NMF_STREAM = True

    def __init__(self):
        super().__init__()
        self.nmf_stream = None

    def forward(self, x):
        draw = lambda: torch.rand(x.shape[0], x.shape[1], 1, 1)  # noqa: E731 — the upstream-style draw
        bases = draw() if self.nmf_stream is None else self.nmf_stream.draw(draw)
        return x * bases


class NMF2D(nn.Module):
    """Named like the stock mmseg NMF2D: its draw cannot be isolated, so the adapter must refuse it."""


class StubHamHead(StubLightHamHead):
    def __init__(self, nmf: nn.Module | None = None, **kw):
        super().__init__(**kw)
        self.ham = nmf if nmf is not None else StubIsolatedNMF()

    def forward(self, feats):
        return self.conv_seg(self.ham(self.squeeze(feats[1])))


def write_ckpt(name: str, payload) -> Path:
    p = TMP / name
    torch.save(payload, p)
    return p


def valid_state() -> dict:
    return {"backbone.projs.0.weight": torch.randn(4, 3, 1, 1),
            "decode_head.conv_seg.weight": torch.randn(NC, 8, 1, 1),
            "decode_head.conv_seg.bias": torch.randn(NC)}


# ---------------------------------------------------------------- 1. import isolation
def test_import_isolation() -> None:
    check("imports_without_mmseg_or_mmcv",
          "mmseg" not in sys.modules and "mmcv" not in sys.modules,
          "src.distill imported clean on the student stack")
    import src.distill.segnext_teacher as st
    check("adapter_module_importable", hasattr(st, "build_segnext_teacher"))
    check("mmseg_path_is_lazy",
          "mmseg" not in sys.modules and "mmcv" not in sys.modules,
          "importing the adapter still pulls no teacher stack")
    # the real mmseg builder must fail loudly when the stack is absent
    ck = write_ckpt("stack.pth", valid_state())
    expect_raises("mmseg_builder_raises_TeacherStackMissing", TeacherStackMissing,
                  build_segnext_teacher, ck, config_path="whatever.py")


# ---------------------------------------------------------------- 2. checkpoint handling
def test_checkpoint_formats() -> None:
    raw = write_ckpt("raw.pth", valid_state())
    wrapped = write_ckpt("wrapped.pth", {"state_dict": valid_state()})
    meta = write_ckpt("meta.pth", {"meta": {"mmseg_version": "1.2.2", "seed": 42},
                                   "state_dict": valid_state(), "optimizer": {"lr": 6e-5}})
    prefixed = write_ckpt("prefixed.pth",
                          {"state_dict": {f"module.{k}": v for k, v in valid_state().items()}})
    for label, p in (("raw_state_dict", raw), ("state_dict_wrapper", wrapped),
                     ("metadata_wrapper", meta)):
        state = load_teacher_state_dict(p)
        check(f"accepts_{label}", len(state) == 3 and all(torch.is_tensor(v) for v in state.values()),
              f"{len(state)} tensors")
    stripped = load_teacher_state_dict(prefixed)
    check("strips_module_prefix", all(not k.startswith("module.") for k in stripped)
          and "backbone.projs.0.weight" in stripped)

    expect_raises("rejects_missing_file", TeacherCheckpointInvalid,
                  load_teacher_state_dict, TMP / "nope.pth")
    expect_raises("rejects_non_dict_payload", TeacherCheckpointInvalid,
                  load_teacher_state_dict, write_ckpt("list.pth", [1, 2, 3]))
    expect_raises("rejects_empty_state_dict", TeacherCheckpointInvalid,
                  load_teacher_state_dict, write_ckpt("empty.pth", {"state_dict": {}}))
    expect_raises("rejects_tensorless_state_dict", TeacherCheckpointInvalid,
                  load_teacher_state_dict, write_ckpt("notensor.pth", {"a": 1, "b": "x"}))
    expect_raises("rejects_unrelated_checkpoint", TeacherCheckpointInvalid,
                  load_teacher_state_dict,
                  write_ckpt("unrelated.pth", {"features.0.weight": torch.randn(2, 2)}))
    expect_raises("rejects_corrupt_file", TeacherCheckpointInvalid,
                  load_teacher_state_dict, _write_garbage("corrupt.pth"))


def _write_garbage(name: str) -> Path:
    p = TMP / name
    p.write_bytes(b"\x00\x01not-a-torch-file")
    return p


# ---------------------------------------------------------------- 3. Stage-3 resolution
def test_stage3_resolution() -> None:
    set_seed(42)
    x = torch.randn(1, 3, 64, 64)
    feats = StubMSCAN()(x)
    shapes = [tuple(f.shape[1:]) for f in feats]
    check("stub_backbone_matches_mscan_b", [f.shape[1] for f in feats] == [64, 128, 320, 512],
          str(shapes))
    f16 = select_stride16_feature(feats, (64, 64))
    check("selects_320ch_stride16_feature",
          f16.shape[1] == STAGE3_CHANNELS and tuple(f16.shape[-2:]) == (4, 4),
          f"{tuple(f16.shape)}")

    # wrong channel count -> no match
    expect_raises("rejects_wrong_feature_channels", TeacherArchitectureMismatch,
                  select_stride16_feature, StubMSCAN(dims=(64, 128, 256, 512))(x), (64, 64))
    # target stage absent entirely
    expect_raises("rejects_missing_stride16_stage", TeacherArchitectureMismatch,
                  select_stride16_feature, StubMSCAN(dims=(64, 128), strides=(4, 8))(x), (64, 64))
    # ambiguous: two candidates both 320ch @ stride 16
    expect_raises("rejects_ambiguous_stage_match", TeacherArchitectureMismatch,
                  select_stride16_feature,
                  StubMSCAN(dims=(320, 320, 320, 512), strides=(16, 16, 16, 32))(x), (64, 64))
    expect_raises("rejects_non_sequence_features", TeacherArchitectureMismatch,
                  select_stride16_feature, torch.randn(1, 320, 4, 4), (64, 64))


# ---------------------------------------------------------------- 4. adapter behaviour
def test_adapter() -> None:
    set_seed(42)
    adapter = SegNeXtTeacherAdapter(StubSegNeXt())
    x = torch.randn(1, 3, 64, 64)
    out = adapter(x)
    check("adapter_feat_s16_is_320ch_os16",
          tuple(out["feat_s16"].shape) == (1, 320, 4, 4), f"{tuple(out['feat_s16'].shape)}")
    check("adapter_logits_are_116ch",
          tuple(out["logits"].shape) == (1, NC, 8, 8), f"{tuple(out['logits'].shape)}")

    # class-space validation at construction (stock ADE20K head has 150)
    expect_raises("rejects_wrong_num_classes", TeacherArchitectureMismatch,
                  SegNeXtTeacherAdapter, StubSegNeXt(head=StubLightHamHead(num_classes=150)))
    expect_raises("rejects_model_without_decode_head", TeacherArchitectureMismatch,
                  SegNeXtTeacherAdapter, nn.Sequential())

    # no forward hooks are registered anywhere: the adapter consumes the backbone's own outputs
    hooked = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
                 for m in adapter.modules())
    check("adapter_registers_no_hooks", hooked == 0,
          "Stage-3 comes from the backbone's returned tuple, so there is nothing to clean up")


# ---------------------------------------------------------------- 5. end-to-end via load_frozen_teacher
def test_frozen_and_provenance() -> None:
    set_seed(42)
    ck = write_ckpt("teacher_ok.pth", {"meta": {"mmseg_version": "1.2.2"},
                                       "state_dict": valid_state()})
    teacher = load_frozen_teacher(str(ck), builder=segnext_builder(model_factory=stub_factory))
    check("load_frozen_teacher_builds_adapter",
          isinstance(teacher, FrozenTeacher) and isinstance(teacher.teacher, SegNeXtTeacherAdapter))

    prov = teacher.provenance.as_dict()
    expected_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    check("provenance_records_path", prov["ckpt_path"] == str(ck.resolve()))
    check("provenance_records_sha256", prov["ckpt_sha256"] == expected_sha, expected_sha[:16] + "…")
    check("provenance_records_size", prov["ckpt_bytes"] == ck.stat().st_size,
          f"{prov['ckpt_bytes']} B")
    check("provenance_records_builder", prov["builder"] == "segnext_mscan_b_builder",
          prov["builder"])

    x = torch.randn(1, 3, 64, 64)
    out = teacher(x, feat_size=(4, 4))
    check("frozen_teacher_outputs_detached",
          not out.logits.requires_grad and not out.feat_s16.requires_grad)
    check("frozen_teacher_eval_mode", not teacher.teacher.training)
    teacher.train(True)
    check("frozen_teacher_cannot_be_untrained", not teacher.teacher.training)
    check("frozen_teacher_no_trainable_params", teacher.trainable_parameters() == [])

    expect_raises("missing_checkpoint_fails_loud", TeacherCheckpointMissing,
                  load_frozen_teacher, str(TMP / "absent.pth"),
                  builder=segnext_builder(model_factory=stub_factory))
    expect_raises("no_checkpoint_path_fails_loud", TeacherCheckpointMissing,
                  load_frozen_teacher, None, builder=segnext_builder(model_factory=stub_factory))


# ---------------------------------------------------------------- 6. E3 integration
def test_e3_integration() -> None:
    set_seed(42)
    ck = write_ckpt("teacher_e3.pth", {"state_dict": valid_state()})
    teacher = load_frozen_teacher(str(ck), builder=segnext_builder(model_factory=stub_factory))
    student = build_student(pretrained=False)
    proj = build_cwd_projection()
    img = torch.randn(1, 3, 64, 64)

    from src.distill import StudentTaps
    with StudentTaps(student) as taps:
        logits = student(img)
        c5, head_logits = taps.require()
    t_out = teacher(img, feat_size=c5.shape[-2:])

    total, parts = distillation_losses(stage=resolve_stage("e3"), logits=logits,
                                       head_logits=head_logits, c5=c5, mask=torch.randint(0, NC, (1, 64, 64)),
                                       teacher_out=t_out, projection=proj, lambda_logit=1.0, ramp=1.0)
    check("e3_consumes_adapter_output",
          set(parts) == {"logit_kd", "cwd_feat", "cwd_logit"} and bool(torch.isfinite(total)),
          " ".join(f"{k}={v:.4f}" for k, v in parts.items()))

    total.backward()
    t_params = list(teacher.teacher.parameters())
    check("no_teacher_gradient_after_backward", all(p.grad is None for p in t_params),
          f"{len(t_params)} teacher tensors")
    check("student_and_projection_got_gradients",
          proj.weight.grad is not None
          and any(p.grad is not None for p in student.parameters()))

    opt = torch.optim.SGD(list(student.parameters()) + list(proj.parameters()), lr=1e-2)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    check("teacher_not_in_optimizer", not (opt_ids & teacher.parameter_ids()),
          f"|optimizer|={len(opt_ids)} |teacher|={len(teacher.parameter_ids())}")


def test_m4_nmf_stream() -> None:
    """M4 (B61 §4) adapter semantics on stubs: stream-only draw, caller RNG untouched, fail closed."""
    from src.distill import MockTeacher
    expect_raises("m4_stock_nmf2d_refused", TeacherArchitectureMismatch,
                  SegNeXtTeacherAdapter, StubSegNeXt(head=StubHamHead(nmf=NMF2D())))
    adapter = SegNeXtTeacherAdapter(StubSegNeXt(head=StubHamHead()))
    teacher = FrozenTeacher(adapter)
    x = torch.randn(2, 3, 64, 64)
    expect_raises("m4_forward_without_stream_refused", RuntimeError, teacher, x)

    # No torch.manual_seed / CUDA seeding anywhere in the stream path.
    seeding = {"torch.manual_seed": torch.manual_seed, "torch.cuda.manual_seed_all": torch.cuda.manual_seed_all,
               "torch.cuda.manual_seed": torch.cuda.manual_seed}

    def forbidden(*_a, **_k):
        raise AssertionError("a global/CUDA seeding call was made")
    torch.manual_seed(123)
    caller = torch.get_rng_state()
    torch.manual_seed, torch.cuda.manual_seed_all, torch.cuda.manual_seed = forbidden, forbidden, forbidden
    try:
        info = teacher.begin_nmf_stream("M4-KD", 42)
        o1, o2 = teacher(x).logits, teacher(x).logits
        seeded_ok = True
    except AssertionError:
        seeded_ok = False
    finally:
        torch.manual_seed = seeding["torch.manual_seed"]
        torch.cuda.manual_seed_all = seeding["torch.cuda.manual_seed_all"]
        torch.cuda.manual_seed = seeding["torch.cuda.manual_seed"]
    check("m4_no_manual_seed_or_cuda_seeding", seeded_ok)
    check("m4kd_begin_describes_stream", info["policy"] == "M4-KD" and info["seed"] == 42, str(info))
    check("m4kd_caller_cpu_rng_untouched", torch.equal(caller, torch.get_rng_state()))
    check("m4kd_stream_advances", teacher.nmf_stream_state()["draws"] == 2 and not torch.equal(o1, o2),
          "fresh bases on successive calls; never reset per batch")
    teacher.begin_nmf_stream("M4-KD", 42)
    check("m4kd_replay_from_42_reproduces", torch.equal(teacher(x).logits, o1) and torch.equal(teacher(x).logits, o2))
    from src.distill import NMFStream
    fresh = NMFStream(42, "M4-V")
    via_stream = fresh.draw(lambda: torch.rand(2, 512, 1, 1))
    via_generator = torch.rand(2, 512, 1, 1, generator=torch.Generator().manual_seed(42))
    check("m4_stream_equals_seed_42_generator", torch.equal(via_stream, via_generator),
          "the private stream is the seed-42 CPU generator sequence")
    ended = adapter.end_nmf_stream()
    check("m4v_end_detaches", ended is not None and adapter.nmf_stream is None
          and adapter.model.decode_head.ham.nmf_stream is None)
    check("m4_no_nmf_model_has_nothing_to_isolate",
          SegNeXtTeacherAdapter(StubSegNeXt()).begin_nmf_stream("M4-V") is None)
    check("m4_mock_teacher_returns_none", FrozenTeacher(MockTeacher(NC)).begin_nmf_stream("M4-KD") is None)


def main() -> int:
    print("=" * 78)
    print("SEGNEXT-B / MSCAN-B TEACHER ADAPTER SMOKE — stubs only; no mmseg, no GPU, no dataset")
    print(f"torch {torch.__version__} | temp dir {TMP}")
    print("=" * 78)
    for fn in (test_import_isolation, test_checkpoint_formats, test_stage3_resolution,
               test_adapter, test_frozen_and_provenance, test_e3_integration, test_m4_nmf_stream):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:38}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
