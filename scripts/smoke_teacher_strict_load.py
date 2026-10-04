#!/usr/bin/env python3
"""Strict teacher load, checksum and provenance (lane L-CKPT-GUARD: R6, DL-50). CPU; no dataset, no GPU.

Every load of the teacher goes through src/distill/teacher.load_frozen_teacher. With `expected_sha256`
the file is hashed and compared before any torch.load or builder call; the default builder builds the
segmentor WITHOUT weights, takes its architecture_signature, and loads the checkpoint only if its state
matches model.state_dict() exactly under the repository's own comparison (five classes: missing,
unexpected, shape, dtype, non-tensor; plus keys the `module.` strip collapses).

Sections:
  S  stub shape (always): a small MSCAN-shaped stub with BatchNorm buffers, through
     build_segnext_teacher with an injected factory. Exact load (0 in every class, the signature
     recorded); one renamed, missing or extra key; one shape; one float64 tensor; one int64 tensor in a
     float32 slot; a state without num_batches_tracked (torch's own strict=True accepts it: the control);
     a non-tensor value; a key the `module.` strip duplicates; a `module.`-prefixed state equal to the
     unprefixed load; a raw state-dict file without `meta`.
  C  checksum (always): a wrong value is refused (TeacherChecksumMismatch with expected and actual)
     before any torch.load or builder call; "", 63 characters and uppercase are refused
     (TeacherChecksumFormatError); a matching value is the recorded ckpt_sha256 even with
     record_sha256=False.
  F  full shape (needs the MMSeg stack; without it every F/P case is a named SKIP, never a PASS): the
     thesis config built with init_model(config, None), its seeded default-init weights
     (torch.manual_seed(2026)) saved to a temp file. Exact load
     (854 entries, 0 in every class); the twelve provenance keys and their values (config_sha256 of the
     file, ham_kwargs from the resolved config as a plain dict, architecture_signature of a fresh build,
     teacher_components_sha256 and reused_module_hashes from the imported teacher_components,
     model_cfg_sha256 recomputed); the old path (mmengine's non-strict init_model(config, ckpt)) and the
     new path give bitwise-equal state; the signature is equal across two builds and changes at 150
     classes; renamed, missing, extra, conv_seg weight shape (1), a 150-class head (2), float64, int64 in
     float32, no num_batches_tracked (47 missing), non-tensor, a `module.` prefix (loads, equal), no
     `meta` (loads); a wrong value at full shape never calls torch.load.
  P  provenance in a KD run (needs the stack): the invariance harness worker with --teacher real on the
     full-shape fixture: run_meta.teacher_provenance holds the twelve keys equal to the loader's, the KD
     checkpoint payload holds the same dict, and the checkpoint loads through
     src/eval/model_loading.load_student_checkpoint (weights_only=True). A dry run with the explicit
     MockTeacher still records teacher_provenance null and teacher_mock true (always runs).

--of-record PATH --sha256 HEX (deferred check 1, local, teacher image): load the checkpoint of record
strictly and print the five counts and the duplicate count, the entry and dtype census, the
parameter count, the checkpoint's top-level keys, the twelve provenance values and the
old-path/new-path equality. A non-zero count (the duplicate count included), an inequality, a
frozen value (config_sha256, teacher_components_sha256, reused_module_hashes) other than the
runbook freeze, or ham_kwargs other than patch 22's row 5g is a FAIL; a refused load raises
(TeacherStateDictMismatch names every count) and exits 1. An architecture_signature or
model_cfg_sha256 other than the sandbox value is reported, not failed (the image's value becomes
the value of record).

Writes only under tempfile.mkdtemp(); never in the repository.
Run:  python -B scripts/smoke_teacher_strict_load.py [--of-record PATH --sha256 HEX]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import src.distill.segnext_teacher as st  # noqa: E402
from src.distill.segnext_teacher import (TeacherStateDictMismatch, architecture_signature,  # noqa: E402
                                         build_segnext_teacher, load_teacher_state_dict,
                                         segnext_builder)
from src.distill.teacher import (TeacherChecksumFormatError, TeacherChecksumMismatch,  # noqa: E402
                                 TeacherProvenance, load_frozen_teacher)

NC = 116
CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
PROV_KEYS = ["builder", "ckpt_path", "ckpt_sha256", "ckpt_bytes", "expected_sha256", "config_path",
             "config_sha256", "ham_kwargs", "architecture_signature", "teacher_components_sha256",
             "reused_module_hashes", "model_cfg_sha256"]
# Deferred check 1 (patch 22): the runbook freeze (frozen: a difference is a STOP) and the sandbox values
# (a difference is reported; the image's value becomes the value of record).
FREEZE = {
    "config_sha256": "1b94aa32a7d22be647d8e60a42daa1dd8dd756158b586feec6465385e299ae4a",
    "teacher_components_sha256": "1b46680c6d03982c84387cc3c3ea78eb14de7ec643ac3b55b1937ef7cbf63025",
    "reused_module_hashes": {
        "teacher_components": "1b46680c6d03982c84387cc3c3ea78eb14de7ec643ac3b55b1937ef7cbf63025",
        "src/data/transforms.py": "df920496cf8710442ae6311b7538c8d3f5d67152a06d0ec1f1c05afc3307193b",
        "configs/augment.py": "8c450f30aedb20b213916d90c054c155906bf4ae49302b5c8d0aedfaf0b1dded",
        "src/eval/metrics.py": "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9",
        "src/distill/nmf_stream.py": "053e53cc441d58a2477affbd281331d7321dd2023e9f09c9f75010b9b36a64d4",
        "src/data/isolation.py": "74a063313cc04c1e327dce22b0faa4ae3a9334a01e6cf7c58eb1bee13c2e5014"},
}
# patch 22: ham_kwargs is as in row 5g (the NMF settings of the resolved config); a difference is a FAIL
HAM_KWARGS = {"MD_S": 1, "MD_R": 16, "train_steps": 6, "eval_steps": 7, "inv_t": 100, "rand_init": True}
SANDBOX = {"architecture_signature": "aa3a73b9a3a34c60c05f0b89dc9ddd08b6280779f5a946992c5614f716429e76",
           "model_cfg_sha256": "d0bfa1215ef575e9f7e16354c8b06a684acf47c4e136af30cfc5c1daa716507b"}
FULL_ENTRIES, FULL_FLOAT32, FULL_INT64, FULL_PARAMS = 854, 807, 47, 27_618_868
TMP = Path(tempfile.mkdtemp(prefix="smoke_teacher_strict_load_"))
results: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def skip(name: str, reason: str) -> None:
    SKIPPED.append((name, reason))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mmseg_stack_importable() -> bool:
    try:
        from mmseg.apis import init_model  # noqa: F401
    except Exception:  # noqa: BLE001 — the builder treats any import failure as a missing stack
        return False
    return True


def refusal(fn, *a, **k):
    """The TeacherStateDictMismatch `fn` raises (or the outcome, as text, when it raises another
    or none)."""
    try:
        fn(*a, **k)
    except TeacherStateDictMismatch as e:
        return e
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"[:200]
    return "loaded"


def counts_are(got, **want) -> bool:
    if not isinstance(got, TeacherStateDictMismatch):
        return False
    full = {c: 0 for c in (*TeacherStateDictMismatch.CLASSES, "duplicate")}
    full.update(want)
    return got.counts == full


# ------------------------------------------------------------------------------- stub (section S)
class StubMSCAN(nn.Module):
    """Four stage features at strides 4/8/16/32, with BatchNorm (so num_batches_tracked buffers
    exist)."""

    def __init__(self, dims=(64, 128, 320, 512), strides=(4, 8, 16, 32)):
        super().__init__()
        self.strides = strides
        self.projs = nn.ModuleList([nn.Conv2d(3, d, 1, bias=False) for d in dims])
        self.norms = nn.ModuleList([nn.BatchNorm2d(d) for d in dims])

    def forward(self, x):
        out = []
        for proj, norm, s in zip(self.projs, self.norms, self.strides):
            h, w = max(x.shape[-2] // s, 1), max(x.shape[-1] // s, 1)
            out.append(norm(proj(nn.functional.adaptive_avg_pool2d(x, (h, w)))))
        return tuple(out)


class StubHead(nn.Module):
    def __init__(self, num_classes=NC):
        super().__init__()
        self.num_classes = num_classes
        self.squeeze = nn.Conv2d(128, 32, 1, bias=False)
        self.conv_seg = nn.Conv2d(32, num_classes, 1)

    def forward(self, feats):
        return self.conv_seg(self.squeeze(feats[1]))


class StubSegNeXt(nn.Module):
    def __init__(self, num_classes=NC):
        super().__init__()
        self.backbone = StubMSCAN()
        self.decode_head = StubHead(num_classes)

    def extract_feat(self, x):
        return self.backbone(x)


def stub_factory(_config_path, _ckpt_path):
    return StubSegNeXt()


def stub_state(seed: int = 7) -> dict:
    torch.manual_seed(seed)
    return {k: (torch.randn_like(v) if v.is_floating_point() else v.clone())
            for k, v in StubSegNeXt().state_dict().items()}


def save(name: str, state, *, meta: bool = True) -> Path:
    p = TMP / name
    torch.save({"meta": {"fixture": name}, "state_dict": state} if meta else state, p)
    return p


def build(path: Path, factory=stub_factory, config_path=None):
    return build_segnext_teacher(path, config_path=config_path, model_factory=factory)


def test_stub() -> None:
    state = stub_state()
    good = save("stub_ok.pth", state)
    adapter = build(good)
    rec = adapter.build_record
    loaded = adapter.model.state_dict()
    check("S_exact_state_loads_with_zero_in_every_class",
          rec["strict_load"] == {"missing": 0, "unexpected": 0, "shape": 0, "dtype": 0, "non_tensor": 0,
                                 "duplicate": 0, "entries": len(state)}
          and all(torch.equal(loaded[k], v) for k, v in state.items()), str(rec["strict_load"]))
    check("S_signature_recorded_equals_a_fresh_build",
          rec["architecture_signature"] == architecture_signature(StubSegNeXt()))
    first = sorted(state)[0]
    renamed = dict(state)
    renamed[first + "_renamed"] = renamed.pop(first)
    got = refusal(build, save("stub_renamed.pth", renamed))
    check("S_renamed_key_is_one_missing_one_unexpected",
          counts_are(got, missing=1, unexpected=1) and got.missing == [first]
          and got.unexpected == [first + "_renamed"], str(got)[:200])
    missing = {k: v for k, v in state.items() if k != "decode_head.conv_seg.bias"}
    got = refusal(build, save("stub_missing.pth", missing))
    check("S_missing_key_refused", counts_are(got, missing=1)
          and got.missing == ["decode_head.conv_seg.bias"], str(got)[:200])
    got = refusal(build, save("stub_extra.pth", {**state, "decode_head.extra.weight": torch.zeros(1)}))
    check("S_extra_key_refused", counts_are(got, unexpected=1)
          and got.unexpected == ["decode_head.extra.weight"], str(got)[:200])
    got = refusal(build, save("stub_shape.pth", {**state, "decode_head.conv_seg.weight":
                                                 torch.randn(NC, 16, 1, 1)}))
    check("S_conv_seg_shape_refused", counts_are(got, shape=1)
          and got.shape == ["decode_head.conv_seg.weight"], str(got)[:200])
    got = refusal(build, save("stub_f64.pth", {**state, "decode_head.conv_seg.bias":
                                               state["decode_head.conv_seg.bias"].double()}))
    check("S_float64_tensor_refused", counts_are(got, dtype=1)
          and got.dtype == ["decode_head.conv_seg.bias"], str(got)[:200])
    got = refusal(build, save("stub_i64.pth", {**state, "decode_head.conv_seg.bias":
                                               torch.zeros(NC, dtype=torch.int64)}))
    check("S_int64_tensor_in_float32_slot_refused", counts_are(got, dtype=1), str(got)[:200])
    nbt = sorted(k for k in state if k.endswith("num_batches_tracked"))
    no_nbt = {k: v for k, v in state.items() if k not in nbt}
    got = refusal(build, save("stub_no_nbt.pth", no_nbt))
    check("S_state_without_num_batches_tracked_refused", counts_are(got, missing=len(nbt))
          and got.missing == nbt and len(nbt) == 4, str(got)[:200])
    control = StubSegNeXt()
    try:
        control.load_state_dict(no_nbt, strict=True)
        torch_strict = "accepted"
    except RuntimeError as e:
        torch_strict = f"refused: {e}"[:120]
    check("S_control_torch_strict_alone_accepts_that_state", torch_strict == "accepted", torch_strict)
    got = refusal(build, save("stub_nontensor.pth", {**state, "decode_head.conv_seg.bias": [0.0] * NC}))
    check("S_non_tensor_value_refused", counts_are(got, non_tensor=1)
          and got.non_tensor == ["decode_head.conv_seg.bias"], str(got)[:200])
    dup = {**state, "module.decode_head.conv_seg.bias": state["decode_head.conv_seg.bias"].clone()}
    got = refusal(load_teacher_state_dict, save("stub_dup.pth", dup))
    check("S_duplicate_key_after_module_strip_refused", counts_are(got, duplicate=1)
          and got.duplicate == ["decode_head.conv_seg.bias"], str(got)[:200])
    prefixed = build(save("stub_prefixed.pth", {f"module.{k}": v for k, v in state.items()}))
    p_state = prefixed.model.state_dict()
    check("S_module_prefixed_state_loads_equal_to_unprefixed",
          all(torch.equal(p_state[k], v) for k, v in loaded.items()))
    raw = build(save("stub_raw.pth", state, meta=False))
    check("S_raw_state_file_without_meta_loads",
          all(torch.equal(raw.model.state_dict()[k], v) for k, v in loaded.items()))


# ---------------------------------------------------------------------------- checksum (section C)
class Calls:
    def __init__(self):
        self.torch_load = self.builder = 0


def counting(calls: Calls):
    real_load, real_builder = torch.load, st.segnext_builder

    def load(*a, **k):
        calls.torch_load += 1
        return real_load(*a, **k)

    def builder(*a, **k):
        calls.builder += 1
        return real_builder(*a, **k)
    return (real_load, real_builder), (load, builder)


def wrong_sha256_outcome(path: Path, config_path=None):
    """load_frozen_teacher with a wrong value through the DEFAULT builder, counting torch.load and
    builder calls."""
    calls = Calls()
    real, fake = counting(calls)
    torch.load, st.segnext_builder = fake
    try:
        load_frozen_teacher(str(path), config_path=config_path, expected_sha256="0" * 64)
        outcome = "loaded"
    except TeacherChecksumMismatch as e:
        outcome = e
    except Exception as e:  # noqa: BLE001
        outcome = f"{type(e).__name__}: {e}"[:160]
    finally:
        torch.load, st.segnext_builder = real
    return outcome, calls


def test_checksum() -> None:
    good = save("stub_sha.pth", stub_state())
    digest = sha256(good)
    outcome, calls = wrong_sha256_outcome(good, config_path=str(CONFIG))
    check("C_wrong_sha256_refused_before_any_torch_load_or_builder",
          isinstance(outcome, TeacherChecksumMismatch) and outcome.expected == "0" * 64
          and outcome.actual == digest and calls.torch_load == 0 and calls.builder == 0,
          f"{outcome} load={calls.torch_load} builder={calls.builder}")
    for label, value in (("empty", ""), ("63_chars", digest[:63]), ("uppercase", digest.upper())):
        try:
            load_frozen_teacher(str(good), builder=segnext_builder(model_factory=stub_factory),
                                expected_sha256=value)
            check(f"C_sha256_{label}_refused", False, "loaded")
        except TeacherChecksumFormatError as e:
            check(f"C_sha256_{label}_refused", e.code == "teacher_ckpt_sha256_format", str(e)[:100])
    teacher = load_frozen_teacher(str(good), builder=segnext_builder(model_factory=stub_factory),
                                  expected_sha256=digest, record_sha256=False)
    prov = teacher.provenance.as_dict()
    check("C_matching_sha256_is_the_recorded_hash_whatever_record_sha256",
          prov["ckpt_sha256"] == prov["expected_sha256"] == digest, str(prov)[:160])
    check("C_provenance_twelve_keys_in_order", list(prov) == PROV_KEYS
          and [f for f in TeacherProvenance.__dataclass_fields__] == PROV_KEYS, str(list(prov)))


# -------------------------------------------------------------------------- full shape (section F)
def build_thesis(cfg_options=None):
    from mmseg.apis import init_model
    return init_model(str(CONFIG), None, device="cpu", cfg_options=cfg_options)


def model_cfg_sha256_of(model) -> str:
    blob = json.dumps(model.cfg.model.to_dict(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def full_fixture():
    """The thesis teacher's own random initialisation, seeded: valid BatchNorm statistics (a N(0, 1)
    draw would make running_var negative and the teacher's outputs non-finite)."""
    torch.manual_seed(2026)
    state = {k: v.detach().clone() for k, v in build_thesis().state_dict().items()}
    return state, save("full_ok.pth", state)


def full_refusal(name: str, state) -> object:
    path = save(name, state)
    try:
        return refusal(load_frozen_teacher, str(path), config_path=str(CONFIG),
                       expected_sha256=sha256(path))
    finally:
        path.unlink()


def old_path_state(path: Path) -> dict:
    from mmseg.apis import init_model
    return init_model(str(CONFIG), str(path), device="cpu").state_dict()


# the F section's checks, each a named SKIP when the MMSeg stack is absent
F_CHECKS = ("F_exact_load_854_entries_zero_in_every_class",
            "F_provenance_twelve_keys_with_the_thesis_values",
            "F_ham_kwargs_is_the_resolved_config_as_a_plain_dict",
            "F_provenance_values_are_plain_types_weights_only_loadable",
            "F_old_path_and_new_path_give_bitwise_equal_state", "F_signature_equal_across_two_builds",
            "F_signature_changes_at_150_classes", "F_renamed_key_is_one_missing_one_unexpected",
            "F_missing_key_refused", "F_extra_key_refused", "F_conv_seg_weight_shape_refused",
            "F_150_class_head_refused", "F_float64_tensor_refused",
            "F_int64_tensor_in_float32_slot_refused",
            "F_state_without_num_batches_tracked_is_47_missing", "F_non_tensor_value_refused",
            "F_module_prefixed_loads_equal", "F_raw_without_meta_loads_equal",
            "F_wrong_sha256_never_calls_torch_load")


def test_full_shape(state: dict, good: Path) -> object:
    digest = sha256(good)
    teacher = load_frozen_teacher(str(good), config_path=str(CONFIG), expected_sha256=digest)
    rec = teacher.teacher.build_record
    model = teacher.teacher.model
    sd = model.state_dict()
    dtypes = [str(v.dtype) for v in sd.values()]
    check("F_exact_load_854_entries_zero_in_every_class",
          rec["strict_load"] == {"missing": 0, "unexpected": 0, "shape": 0, "dtype": 0, "non_tensor": 0,
                                 "duplicate": 0, "entries": FULL_ENTRIES}
          and dtypes.count("torch.float32") == FULL_FLOAT32 and dtypes.count("torch.int64") == FULL_INT64
          and sum(p.numel() for p in model.parameters()) == FULL_PARAMS, str(rec["strict_load"]))
    prov = teacher.provenance.as_dict()
    import src.training.teacher_components as tc
    fresh = build_thesis()
    check("F_provenance_twelve_keys_with_the_thesis_values",
          list(prov) == PROV_KEYS and prov["builder"] == "segnext_mscan_b_builder"
          and prov["ckpt_sha256"] == prov["expected_sha256"] == digest
          and prov["config_path"] == str(CONFIG.resolve()) and prov["config_sha256"] == sha256(CONFIG)
          and prov["architecture_signature"] == architecture_signature(fresh)
          and prov["teacher_components_sha256"] == tc.COMPONENTS_PROVENANCE["sha256"]
          and prov["reused_module_hashes"] == tc.reused_module_hashes()
          and prov["model_cfg_sha256"] == model_cfg_sha256_of(fresh), json.dumps(prov)[:300])
    ham = prov["ham_kwargs"]
    check("F_ham_kwargs_is_the_resolved_config_as_a_plain_dict",
          type(ham) is dict and ham == fresh.cfg.model.decode_head.ham_kwargs.to_dict()
          and json.loads(json.dumps(ham)) == ham, str(ham))
    buf = io.BytesIO()
    torch.save({"teacher_provenance": prov}, buf)
    buf.seek(0)
    try:
        back = torch.load(buf, weights_only=True)["teacher_provenance"]
        plain = back == prov
    except Exception as e:  # noqa: BLE001
        plain = f"{type(e).__name__}: {e}"[:120]
    check("F_provenance_values_are_plain_types_weights_only_loadable", plain is True, str(plain))
    old = old_path_state(good)
    check("F_old_path_and_new_path_give_bitwise_equal_state",
          sorted(old) == sorted(sd) and all(old[k].dtype == sd[k].dtype and torch.equal(old[k], sd[k])
                                            for k in sd), f"{len(old)} vs {len(sd)} tensors")
    check("F_signature_equal_across_two_builds",
          architecture_signature(build_thesis()) == architecture_signature(fresh))
    check("F_signature_changes_at_150_classes",
          architecture_signature(build_thesis({"model.decode_head.num_classes": 150}))
          != architecture_signature(fresh))
    first = "backbone.block1.0.attn.proj_1.weight" if "backbone.block1.0.attn.proj_1.weight" in state \
        else sorted(state)[0]
    renamed = dict(state)
    renamed[first + "_renamed"] = renamed.pop(first)
    got = full_refusal("full_renamed.pth", renamed)
    check("F_renamed_key_is_one_missing_one_unexpected", counts_are(got, missing=1, unexpected=1)
          and got.missing == [first], str(got)[:200])
    got = full_refusal("full_missing.pth", {k: v for k, v in state.items() if k != first})
    check("F_missing_key_refused", counts_are(got, missing=1), str(got)[:200])
    got = full_refusal("full_extra.pth", {**state, "decode_head.extra.weight": torch.zeros(1)})
    check("F_extra_key_refused", counts_are(got, unexpected=1), str(got)[:200])
    w = state["decode_head.conv_seg.weight"]
    got = full_refusal("full_conv_seg_shape.pth", {**state, "decode_head.conv_seg.weight":
                                                   torch.randn(w.shape[0], w.shape[1] // 2, 1, 1)})
    check("F_conv_seg_weight_shape_refused", counts_are(got, shape=1)
          and got.shape == ["decode_head.conv_seg.weight"], str(got)[:200])
    got = full_refusal("full_150.pth", {**state, "decode_head.conv_seg.weight":
                                        torch.randn(150, *w.shape[1:]),
                                        "decode_head.conv_seg.bias": torch.randn(150)})
    check("F_150_class_head_refused", counts_are(got, shape=2), str(got)[:200])
    got = full_refusal("full_f64.pth", {**state, first: state[first].double()})
    check("F_float64_tensor_refused", counts_are(got, dtype=1), str(got)[:200])
    got = full_refusal("full_i64.pth", {**state, first: torch.zeros(state[first].shape,
                                                                    dtype=torch.int64)})
    check("F_int64_tensor_in_float32_slot_refused", counts_are(got, dtype=1), str(got)[:200])
    got = full_refusal("full_no_nbt.pth", {k: v for k, v in state.items()
                                           if not k.endswith("num_batches_tracked")})
    check("F_state_without_num_batches_tracked_is_47_missing", counts_are(got, missing=FULL_INT64),
          str(got)[:200])
    got = full_refusal("full_nontensor.pth", {**state, first: "not a tensor"})
    check("F_non_tensor_value_refused", counts_are(got, non_tensor=1), str(got)[:200])
    for label, payload, meta in (("module_prefixed", {f"module.{k}": v for k, v in state.items()}, True),
                                 ("raw_without_meta", state, False)):
        path = save(f"full_{label}.pth", payload, meta=meta)
        try:
            other = load_frozen_teacher(str(path), config_path=str(CONFIG), expected_sha256=sha256(path))
            o_sd = other.teacher.model.state_dict()
            ok = all(torch.equal(o_sd[k], v) for k, v in sd.items())
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [{label}] {type(e).__name__}: {e}"[:200])
        finally:
            path.unlink()
        check(f"F_{label}_loads_equal", ok)
    outcome, calls = wrong_sha256_outcome(good, config_path=str(CONFIG))
    check("F_wrong_sha256_never_calls_torch_load",
          isinstance(outcome, TeacherChecksumMismatch) and calls.torch_load == 0 and calls.builder == 0,
          f"{outcome} load={calls.torch_load} builder={calls.builder}")
    return prov


# --------------------------------------------------------------------- KD-run provenance (section P)
def test_kd_run(good: Path, prov: dict | None) -> None:
    from scripts.invariance_harness import make_synthetic_dataset, run_worker
    from src.eval.model_loading import load_student_checkpoint
    work = TMP / "kd"
    work.mkdir()
    data = work / "data"
    make_synthetic_dataset(data)
    if prov is None:
        skip("P_kd_run_records_the_twelve_keys_in_run_meta_and_payload", "no MMSeg stack")
        skip("P_kd_checkpoint_loads_through_load_student_checkpoint", "no MMSeg stack")
    else:
        r = run_worker(code_root=REPO, data_root=data, out_dir=work / "real_e3", stage="e3", steps=2,
                       val_interval=2, teacher="real", teacher_ckpt=str(good.resolve()),
                       teacher_config=str(CONFIG.resolve()), teacher_ckpt_sha256=sha256(good))
        sm = r["summary"] or {}
        rows = sm.get("run_meta") or [{}]
        meta_prov = rows[0].get("teacher_provenance") or {}
        cks = sorted((work / "real_e3").rglob("e3_student_best_iter*.pt"))
        payload = torch.load(str(cks[-1]), map_location="cpu", weights_only=False) if cks else {}
        check("P_kd_run_records_the_twelve_keys_in_run_meta_and_payload",
              r["returncode"] == 0 and list(meta_prov) == PROV_KEYS and meta_prov == prov
              and payload.get("teacher_provenance") == meta_prov,
              f"rc={r['returncode']} {json.dumps(meta_prov)[:200]} log={r['log']}")
        try:
            load_student_checkpoint(cks[-1])
            loaded = True
        except Exception as e:  # noqa: BLE001
            loaded = f"{type(e).__name__}: {e}"[:160]
        check("P_kd_checkpoint_loads_through_load_student_checkpoint", loaded is True, str(loaded))
    # the explicit MockTeacher of a dry run without --teacher-ckpt: no provenance, teacher_mock true
    ckpt_dir = work / "mock_e2"
    code = (f"import sys; sys.path.insert(0, {str(REPO)!r})\n"
            "import configs.data as cdata\n"
            f"cdata.SPLIT_SIZES.update({{'train': 16, 'val': 4}})\n"
            "import src.training.train_distill as td\n"
            f"raise SystemExit(td.main(['--stage', 'e2', '--dry-run', '--max-iters', '2', "
            f"'--ckpt-dir', {str(ckpt_dir)!r}]))\n")
    env = {**__import__("os").environ, "PLANTSEG_DATA_ROOT": str(data), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env,
                          cwd=str(work))
    meta_p = ckpt_dir / "e2_run_meta.jsonl"
    rows = [json.loads(ln) for ln in meta_p.read_text().splitlines()] if meta_p.is_file() else [{}]
    check("P_dry_mock_teacher_records_no_provenance_and_teacher_mock",
          proc.returncode == 0 and rows[0].get("teacher_provenance", "absent") is None
          and rows[0].get("teacher_mock") is True, f"rc={proc.returncode} {proc.stderr.strip()[-200:]}")


# ------------------------------------------------------------------------ deferred check 1 (--of-record)
def of_record(path: Path, expected: str) -> int:
    from mmseg.apis import init_model
    teacher = load_frozen_teacher(str(path), config_path=str(CONFIG), expected_sha256=expected)
    rec, model = teacher.teacher.build_record, teacher.teacher.model
    sd = model.state_dict()
    dtypes = [str(v.dtype) for v in sd.values()]
    prov = teacher.provenance.as_dict()
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    old = init_model(str(CONFIG), str(path), device="cpu").state_dict()
    equal = sorted(old) == sorted(sd) and all(old[k].dtype == sd[k].dtype and torch.equal(old[k], sd[k])
                                              for k in sd)
    classes = (*TeacherStateDictMismatch.CLASSES, "duplicate")
    print("strict load counts:", {c: rec["strict_load"][c] for c in classes})
    print(f"entries: {rec['strict_load']['entries']} of {FULL_ENTRIES}; float32 "
          f"{dtypes.count('torch.float32')}, int64 {dtypes.count('torch.int64')}; parameters "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print("checkpoint top-level keys:", sorted(payload) if isinstance(payload, dict) else type(payload))
    for k in PROV_KEYS:
        print(f"  {k}: {json.dumps(prov[k])}")
    print(f"old path vs new path: {len(sd)} tensors, bitwise equal: {equal}")
    fails = [] if all(rec["strict_load"][c] == 0 for c in classes) else ["counts"]
    if not (rec["strict_load"]["entries"] == FULL_ENTRIES
            and dtypes.count("torch.float32") == FULL_FLOAT32
            and dtypes.count("torch.int64") == FULL_INT64
            and sum(p.numel() for p in model.parameters()) == FULL_PARAMS):
        fails.append("census")
    if not equal:
        fails.append("old_new_inequality")
    fails += [f"frozen:{k}" for k, v in FREEZE.items() if prov[k] != v]
    if prov["ham_kwargs"] != HAM_KWARGS:
        fails.append("ham_kwargs")
    reported = [f"{k} {prov[k]} (sandbox {v}; the image's value becomes the value of record)"
                for k, v in SANDBOX.items() if prov[k] != v]
    for line in reported:
        print("REPORT:", line)
    print(f"RESULT: {'FAIL (STOP) ' + str(fails) if fails else 'PASS'} (of record: {path.name}, "
          f"sha256 {expected})")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--of-record", default=None)
    ap.add_argument("--sha256", default=None)
    a = ap.parse_args(argv)
    if a.of_record:
        if not a.sha256:
            ap.error("--of-record needs --sha256")
        return of_record(Path(a.of_record), a.sha256)
    print("=" * 78)
    print("TEACHER STRICT LOAD SMOKE (R6, DL-50) — CPU, synthetic; no dataset, no GPU")
    print(f"torch {torch.__version__} | temp {TMP}")
    print("=" * 78)
    test_stub()
    test_checksum()
    prov = None
    if mmseg_stack_importable():
        state, good = full_fixture()
        prov = test_full_shape(state, good)
    else:
        good = TMP / "absent.pth"
        for name in F_CHECKS:
            skip(name, "no MMSeg stack: runs in the teacher image")
    test_kd_run(good, prov)
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:62}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    for name, reason in SKIPPED:
        print(f"  {name:62}: SKIP  {reason}")
    passed = sum(1 for _, ok, _ in results if ok)
    skipped = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)}{skipped})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
