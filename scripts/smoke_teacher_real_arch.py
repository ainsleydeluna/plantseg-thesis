#!/usr/bin/env python3
"""Smoke: the thesis teacher architecture, randomly initialised, through the DIAG seam (P35; lane
L-TEACHER-DIAG). No real weights: the thesis teacher config is built by MMSegmentation with random
weights, its state_dict is saved as a synthetic checkpoint in a temporary folder, and that checkpoint is
loaded through the real-mode load lines of src/eval/teacher_diag.load_teacher (load_frozen_teacher(ckpt,
config_path=cfg) and load_teacher_model(resolved, config_path=cfg), no builder). On one synthetic
512x512 image:

  - the decode head is IsolatedNMFLightHamHead with exactly one isolated NMF module holding the begun
    stream (the adapter's live object, C1);
  - K = 1: the split forward (backbone once, head once) equals the adapter's forward bit for bit;
  - eight head calls on one backbone output leave the feature hash and the caller's CPU RNG unchanged,
    advance the M4-KD stream by exactly 8, give finite [1, 116, 64, 64] logits that differ across draws;
  - evaluator form: stream 42's resized map equals TeacherEvalModel's output bit for bit;
  - P8 on the real path: both forms pass the real-mode after-load and record checks, and the record holds
    K-part's 12 fields in order, none None, "", {} or [], expected_sha256 the verified sha.

Needs the MMSeg stack (mmengine 0.10.7, mmcv 2.1.0, mmseg 1.2.2); without it the smoke FAILS, never
passes silently. Part of local command 0 (P36).

    python -B scripts/smoke_teacher_real_arch.py
"""
from __future__ import annotations

import shutil
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def cases(tmp: Path) -> None:
    import torch.nn.functional as F

    from src.distill.nmf_stream import NMFStream, isolated_nmf_modules
    from src.eval import teacher_diag as td
    try:
        from mmseg.apis import init_model
    except Exception as e:  # noqa: BLE001
        check("the MMSeg stack imports (mmseg.apis.init_model)", False, f"{type(e).__name__}: {e}")
        return
    cfg = td.REPO / td.TEACHER_CONFIG_REL
    t0 = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261003)
        model = init_model(str(cfg), None, device="cpu")
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ckpt = tmp / "random_thesis_teacher.pth"
    torch.save({"meta": {"synthetic": "random init, not a teacher of record"}, "state_dict": state}, ckpt)
    del model
    inputs = td.TeacherInputs(ckpt=ckpt, sha256=td.file_sha256(ckpt), ckpt_bytes=ckpt.stat().st_size, config=cfg,
                              config_sha256=td.file_sha256(cfg), config_blob=td.RECORD_CONFIG_BLOB, role="record",
                              arm_id=None, arm_dl_id=None)
    kd = td.load_teacher("kd", inputs)                           # the real load line: no builder
    head = type(kd.segmentor.decode_head).__name__
    mods = isolated_nmf_modules(kd.segmentor)
    n_params = sum(p.numel() for p in kd.frozen.parameters())
    check("P35 the thesis config builds IsolatedNMFLightHamHead with one isolated NMF module holding the begun "
          "M4-KD stream (the adapter's object)",
          head == "IsolatedNMFLightHamHead" and len(mods) == 1 and mods[0].nmf_stream is kd.stream
          and kd.stream is kd.adapter.nmf_stream and kd.stream.policy == "M4-KD" and kd.segmentor.training is False,
          f"{head}, {len(mods)} module(s), {n_params} parameters, built in {time.monotonic() - t0:.1f} s")
    check("P35 the loaded state equals the saved random state (loaded_state_sha256)",
          td.loaded_state_sha256(kd.segmentor) == td.loaded_state_sha256(types.SimpleNamespace(state_dict=lambda: state)))

    g = torch.Generator().manual_seed(7)
    x = torch.rand(1, 3, 512, 512, generator=g)
    split = td.SplitTeacher(kd)
    t1 = time.monotonic()
    feats = split.features(x)
    h = td.feature_sha256(feats)
    z1 = split.head(feats, kd.stream, feat_hash=h, expect_shape=(1, 116, 64, 64))
    kd2 = td.load_teacher("kd", inputs)
    z_full = kd2.frozen(x).logits
    check("P35 K = 1: the split forward equals the adapter's forward bit for bit; the streams end in one state",
          torch.equal(z1, z_full) and kd.stream.state_sha256() == kd2.stream.state_sha256(),
          f"max |diff| {float((z1 - z_full).abs().max()):.3e}")
    rng0 = td.rng_state_sha256()
    zs = [split.head(feats, kd.stream, feat_hash=h, expect_shape=(1, 116, 64, 64)) for _ in range(7)]
    check("P35 eight head calls: draws == 8, feature hash and caller RNG unchanged, finite 64x64 logits",
          kd.stream.draws == 8 and td.feature_sha256(feats) == h and td.rng_state_sha256() == rng0
          and all(bool(torch.isfinite(z).all()) for z in [z1, *zs]) and tuple(zs[-1].shape) == (1, 116, 64, 64),
          f"backbone + 8 heads in {time.monotonic() - t1:.1f} s")
    check("P35 different draws give different logits (the NMF basis draw is live)",
          not torch.equal(z1, zs[0]))

    ev = td.load_teacher("evaluator", inputs)                    # the evaluator's load line
    ev2 = td.load_teacher("evaluator", inputs)
    s_split = td.SplitTeacher(ev)
    f2 = s_split.features(x)
    z42 = s_split.head(f2, ev.stream, feat_hash=td.feature_sha256(f2))
    z42 = F.interpolate(z42, size=(512, 512), mode="bilinear", align_corners=False)
    with torch.no_grad():
        z_eval = ev2.eval_model(x)
    other = NMFStream(43, "M4-V")
    s_split.attach(other)
    z43 = s_split.head(f2, other, feat_hash=td.feature_sha256(f2))
    s_split.attach(ev.stream)
    check("P35 evaluator form: stream 42's resized map equals TeacherEvalModel's output bit for bit; stream 43 "
          "differs", ev.stream.policy == "M4-V" and torch.equal(z42, z_eval)
          and not torch.equal(F.interpolate(z43, size=(512, 512), mode="bilinear", align_corners=False), z42))

    recs = {mode: td.teacher_record(x, inputs, td.after_load_checks(x, inputs, stub=False), stub=False)["provenance"]
            for mode, x in (("kd", kd), ("evaluator", ev))}
    empty = {m: [n for n, v in r.items() if td.provenance_value_empty(v)] for m, r in recs.items()}
    check("P8 real path: both forms pass the real-mode checks and record the 12 K-part fields in order, none None, "
          "\"\", {} or [], with expected_sha256 == the verified sha (SCOPE a, b)",
          all(list(r) == list(td.TEACHER_PROVENANCE_FIELDS) and r["expected_sha256"] == inputs.sha256
              for r in recs.values()) and not any(empty.values()), str(empty))


def main() -> int:
    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    tmp = safe_tmpdir("diag_arch_")
    try:
        cases(tmp)
    except Exception as e:  # noqa: BLE001 -- a crash is a failed case, never a pass
        import traceback
        check("cases raised", False, f"{type(e).__name__}: {e} {traceback.format_exc()[-500:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    good = sum(ok for _, ok, _ in RESULTS)
    print(f"RESULT: {'PASS' if good == len(RESULTS) and RESULTS else 'FAIL'} ({good}/{len(RESULTS)})")
    return 0 if good == len(RESULTS) and RESULTS else 1


if __name__ == "__main__":
    sys.exit(main())
