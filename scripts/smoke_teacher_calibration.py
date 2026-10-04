#!/usr/bin/env python3
"""Smoke: pixel-level calibration (src/eval/calibration.py) and the D2 VAL pass (lane L-TEACHER-DIAG).

Acceptance (d) of the lane prompt as amended by P16, P17 and P32(d): zero ECE for a perfectly calibrated
case; a hand-computed two-bin case; bin edges at exactly m/15 (m = 1..14), 1.0 in bin 15, a bin outside
1..15 stops; pixels labelled 255 are ignored; the disease subset; the float32-logit vector of P32(d)
(0.2799999955678827, a float32 softmax would give 0.47999998927116394); the hook scores the tensor the
core argmaxes (sum correct == sum TP), keeps no reference to it and refuses an in-place change.

The D2 VAL pass (scripts/teacher_d2_calibration.py) on the stub teacher and a synthetic 5,367/846 root:
its evaluator artifact equals scripts/evaluate_model.py run()'s under EVALUATION_CONTRACT 10(d); the
gate runs before the write (P18: "not reproduced", no artifact, exit 1); control and arm (P3, P4);
--val-reference read summary-first, its manifest identity and verification; refusals before any load;
real mode (the CLI) refused unless pinned to HEAD (P26); the RNG watch of P9. The record role's
reproduction gate (src/eval/teacher_diag.reproduction_gate, shared by D2 and D1's VAL part): in real mode
the produced value itself must be within 1e-5 of R3 = 0.38576993346214294 (AM-PC-1 finding 27).
Synthetic inputs only; no PlantSeg data, no checkpoint of record, no GPU.

    python -B scripts/smoke_teacher_calibration.py
"""
from __future__ import annotations

import gc
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.eval.calibration import (ECE_EDGES, CalibratingBatchForward, CalibrationAccumulator,  # noqa: E402
                                  CalibrationStop, bin_index, ece_from_bins)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def acc_for(logits_rows, targets, num_classes):
    """One image of len(rows) pixels laid out as [1, C, 1, n]."""
    z = torch.tensor(np.asarray(logits_rows, dtype=np.float64).T[None, :, None, :])
    t = torch.tensor(np.asarray(targets, dtype=np.int64)[None, None, :])
    acc = CalibrationAccumulator(num_classes=num_classes, disease_classes=range(1, num_classes))
    acc.update(z, t)
    return acc.result()


A9, A3 = math.log(27.0), math.log(9.0 / 7.0)          # e^a / (e^a + 3) = 0.9 and 0.3 with four classes


def case_strict_099() -> bool:
    """The share > 0.99 is strict: one valid pixel of max-probability p*; with the threshold patched to p* it is
    not counted, with the next float below p* it is."""
    import src.eval.calibration as cal
    p_star = acc_for([[6.0, 0.0, 0.0]], [0], 3)["mean_max_prob_t1"]      # one pixel: the mean is its value
    old = cal.HIGH_CONFIDENCE
    try:
        cal.HIGH_CONFIDENCE = p_star
        at = acc_for([[6.0, 0.0, 0.0]], [0], 3)["n_max_prob_gt_0_99_t1"]
        cal.HIGH_CONFIDENCE = float(np.nextafter(p_star, 0.0))
        below = acc_for([[6.0, 0.0, 0.0]], [0], 3)["n_max_prob_gt_0_99_t1"]
    finally:
        cal.HIGH_CONFIDENCE = old
    return at == 0 and below == 1


def case_disease_by_ground_truth() -> bool:
    """The disease subset is chosen by the target, not the prediction: a disease pixel predicted as background
    (conf 0.9, wrong) counts, a background pixel predicted as a disease (conf 0.3) does not; with a disease
    pixel predicted right (conf 0.3): ECE_disease = 0.5*|0 - 0.9| + 0.5*|1 - 0.3| = 0.8 over n_disease = 2.
    (By prediction it would be 0.2.)"""
    r = acc_for([[A9, 0, 0, 0], [0, 0, A3, 0], [0, 0, A3, 0]], [2, 0, 2], 4)
    return r["n_disease"] == 2 and abs(r["ece_disease"] - 0.8) <= 1e-12


def unit_cases() -> None:
    # d1: perfectly calibrated -- logits (0, 0): confidence exactly 0.5 at class 0, half the pixels right
    r = acc_for([[0.0, 0.0]] * 4, [0, 0, 1, 1], 2)
    check("d1 perfectly calibrated set: ECE == 0 exactly", r["ece"] == 0.0, f"ece={r['ece']!r}")
    check("d1 ece_from_bins on one bin with C/n == S/n gives 0",
          ece_from_bins([0] * 8 + [4] + [0] * 7, [0] * 8 + [2] + [0] * 7, [0.0] * 8 + [2.0] + [0.0] * 7) == 0.0)

    # d2: two bins by hand -- C = 4; conf 0.9 (bin 14) one right one wrong; conf 0.3 (bin 5) both right
    a9, a3 = math.log(27.0), math.log(9.0 / 7.0)          # e^a / (e^a + 3) = 0.9 and 0.3
    r = acc_for([[a9, 0, 0, 0], [a9, 0, 0, 0], [a3, 0, 0, 0], [a3, 0, 0, 0]], [0, 1, 0, 0], 4)
    want = 0.5 * abs(0.5 - 0.9) + 0.5 * abs(1.0 - 0.3)
    check("d2 two-bin hand case: 0.5*|0.5-0.9| + 0.5*|1-0.3| = 0.55",
          abs(r["ece"] - want) <= 1e-12 and [b["n"] for b in r["bins"]][4] == 2 and r["bins"][13]["n"] == 2,
          f"ece={r['ece']!r}")

    # d3: edges
    ok = all(int(bin_index(np.array([m / 15.0]))[0]) == m for m in range(1, 15))
    check("d3 conf == m/15 lands in bin m (m = 1..14)", ok)
    ok = all(int(bin_index(np.array([np.nextafter(m / 15.0, 2.0)]))[0]) == m + 1 for m in range(1, 15))
    check("d3 the next float above m/15 lands in bin m+1 (m = 1..14)", ok)
    check("d3 conf == 1.0 lands in bin 15", int(bin_index(np.array([1.0]))[0]) == 15)
    check("d3 edges are np.arange(16)/15.0", np.array_equal(ECE_EDGES, np.arange(16) / 15.0))
    check("d3 conf 0.0 (bin 0) stops", raises(lambda: bin_index(np.array([0.0])), CalibrationStop))
    check("d3 conf above 1.0 (bin 16) stops", raises(lambda: bin_index(np.array([np.nextafter(1.0, 2.0)])),
                                                      CalibrationStop))
    check("d3 a NaN confidence stops", raises(lambda: bin_index(np.array([np.nan])), CalibrationStop))

    # d4: 255 ignored
    rows = [[a9, 0, 0, 0], [a3, 0, 1.0, 0], [0.2, 0.1, 0, 0]]
    base = acc_for(rows, [0, 2, 1], 4)
    more = acc_for(rows + [[50.0, -3, 2, 0], [0, 0, 0, 9.0]], [0, 2, 1, 255, 255], 4)
    same = {k: v for k, v in base.items() if k != "images"} == {k: v for k, v in more.items() if k != "images"}
    check("d4 pixels labelled 255 change nothing", same and more["n"] == 3)

    # d6: disease subset -- background pixels leave the disease ECE unchanged
    r = acc_for([[a9, 0, 0, 0], [a9, 0, 0, 0], [0, 0, a3, 0], [0, 0, a3, 0]], [0, 0, 2, 1], 4)
    want_d = 1.0 * abs(0.5 - 0.3)
    check("d6 disease-pixel ECE over target in 1..C-1 only",
          r["n_disease"] == 2 and abs(r["ece_disease"] - want_d) <= 1e-12, f"ece_disease={r['ece_disease']!r}")

    # d7: P32(d) float32 vector
    z32 = torch.tensor([[1.3862942457199097, 0.0], [1.3862942457199097, 0.0],
                        [1.1526795625686646, 0.0], [1.1526795625686646, 0.0]], dtype=torch.float32)
    acc = CalibrationAccumulator(num_classes=2, disease_classes=range(1, 2))
    acc.update(z32.T[None, :, None, :].contiguous(), torch.tensor([[[0, 0, 1, 1]]]))
    r = acc.result()
    check("d7 float32 logits: ECE == 0.2799999955678827 (float64 softmax)", r["ece"] == 0.2799999955678827,
          f"ece={r['ece']!r}")
    pred = z32.argmax(1).numpy()
    c32 = torch.softmax(z32, 1).double().numpy()[np.arange(4), pred]
    b32 = bin_index(c32)
    n_m, c_m, s_m = [0] * 16, [0] * 16, [0.0] * 16
    for i in range(4):
        n_m[b32[i]] += 1
        c_m[b32[i]] += int(pred[i] == [0, 0, 1, 1][i])
        s_m[b32[i]] += c32[i]
    check("d7 a float32 softmax would give 0.47999998927116394 (the vector separates them)",
          ece_from_bins(n_m, c_m, s_m) == 0.47999998927116394, f"{ece_from_bins(n_m, c_m, s_m)!r}")

    # means vs an independent numpy reference
    rng = np.random.default_rng(5)
    z = rng.normal(0, 3, size=(1, 6, 5, 7)).astype(np.float32)
    t = rng.integers(0, 6, size=(1, 5, 7))
    t[0, 0, :3] = 255
    acc = CalibrationAccumulator(num_classes=6, disease_classes=range(1, 6))
    acc.update(torch.tensor(z), torch.tensor(t))
    r = acc.result()
    zz = z[0].reshape(6, -1).astype(np.float64)[:, t.reshape(-1) != 255]
    ref = {}
    for tag, T in (("t1", 1.0), ("t4", 4.0)):
        p = np.exp(zz / T - zz.max(0) / T)
        p = p / p.sum(0)
        ref[f"mean_max_prob_{tag}"] = p.max(0).mean()
        ref[f"mean_entropy_{tag}"] = (-(p * np.log(p)).sum(0)).mean()
    ok = all(abs(r[k] - v) <= 1e-12 for k, v in ref.items())
    check("means: max-probability and entropy at T=1 and T=4 match numpy to 1e-12", ok,
          str({k: (r[k], v) for k, v in ref.items()}))
    pmax = np.exp(zz - zz.max(0)) / np.exp(zz - zz.max(0)).sum(0)
    check("share max-prob > 0.99 at T=1 matches numpy", r["n_max_prob_gt_0_99_t1"] == int((pmax.max(0) > 0.99).sum()))
    check("P16 the share > 0.99 is strict: a pixel exactly at the threshold is not counted, one just above it is",
          case_strict_099())
    check("d6 the disease subset follows the ground truth, not the prediction (ECE_disease 0.8, n 2)",
          case_disease_by_ground_truth())


def hook_cases() -> None:
    from src.eval.evaluate import Condition, EvalBatch, ManifestEntry, evaluate_model

    rng = np.random.default_rng(11)
    n_img, C = 5, 116
    ids = [f"syn_{i:03d}" for i in range(n_img)]
    targets = torch.tensor(rng.integers(0, C, size=(n_img, 6, 8)))
    targets[0, 0, :4] = 255
    logits_all = torch.tensor(rng.normal(0, 2, size=(n_img, C, 6, 8)).astype(np.float32))

    class Fwd:
        def __call__(self, model, images):
            return images.clone()                        # the "model" output is a fresh tensor

    acc = CalibrationAccumulator()
    hook = CalibratingBatchForward(Fwd(), acc)
    batches = [EvalBatch(images=logits_all[i:i + 1], targets=targets[i:i + 1], image_ids=[ids[i]],
                         clean_image_ids=[ids[i]], manifest_indices=[i]) for i in range(n_img)]
    manifest = [ManifestEntry(i, ids[i], ids[i]) for i in range(n_img)]
    res = evaluate_model(None, batches, expected_manifest=manifest, condition=Condition(), num_classes=C,
                         background_index=0, ignore_index=255, batch_forward=hook)
    r = acc.result()
    check("d5 sum correct == sum TP of the core (same tensor, same argmax)",
          r["correct"] == int(res.dataset_tp.sum()), f"{r['correct']} vs {int(res.dataset_tp.sum())}")
    check("d5 n == sum GT of the core (valid pixels)", r["n"] == int(res.dataset_gt.sum()))
    gc.collect()
    check("d5 the hook keeps no tensor: its weakref is dead after the core's del",
          hook.last_logits_ref is not None and hook.last_logits_ref() is None)
    check("d5 one hook call per batch", hook.calls == n_img)

    class Mutating(CalibrationAccumulator):
        def update(self, logits, target):
            logits.add_(0.0)                             # an in-place op bumps _version
            super().update(logits, target)

    bad = CalibratingBatchForward(Fwd(), Mutating())
    check("d5 an in-place change of the core's tensor stops",
          raises(lambda: bad(None, batches[0]), CalibrationStop))

    class Warny(CalibrationAccumulator):
        def update(self, logits, target):
            import warnings
            warnings.warn("synthetic accumulation warning")
            super().update(logits, target)

    import warnings
    w_hook = CalibratingBatchForward(Fwd(), Warny())
    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        w_hook(None, batches[0])
    check("P17 accumulation warnings are kept apart from the evaluator's capture",
          not outer and len(w_hook.accumulation_warnings) == 1, f"outer={len(outer)}")


# ---------------------------------------------------------------------------------------------------
# the D2 VAL pass (scripts/teacher_d2_calibration.py) on the stub teacher and a synthetic data root
# ---------------------------------------------------------------------------------------------------
def _reference(fx, td, tmp, ckpt, sha, n=3):
    """The R3-equivalent reference: scripts/evaluate_model.py run() on the same stub teacher, with the sha256
    write_stub_ckpt returned (the evaluator's teacher stage requires it, R6)."""
    from scripts import evaluate_model as em
    from src.distill.segnext_teacher import segnext_builder
    out = tmp / "ref_artifact"
    args = em.build_parser().parse_args([
        "--stage", "teacher", "--model-role", "teacher", "--split", "val", "--checkpoint", str(ckpt),
        "--teacher-ckpt-sha256", sha,
        "--teacher-config", str(td.REPO / td.TEACHER_CONFIG_REL), "--artifact-status", "smoke",
        "--max-samples", str(n), "--batch-size", "1", "--out-dir", str(out), "--run-id", "ref_run"])
    return em.run(args, teacher_builder=segnext_builder(model_factory=fx.stub_factory))


def _rewrite_manifest(d: Path) -> None:
    import hashlib
    lines = [f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n"
             for n in ("per_image.jsonl", "sufficient_stats.npz", "summary.json")]
    (d / "MANIFEST.sha256").write_text("".join(lines), encoding="utf-8", newline="\n")


PINNED = {"RECORD_SHA256": "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e",
          "R3_VAL_MIOU": 0.38576993346214294, "R3_TOLERANCE": 1e-5,
          "RECORD_CONFIG_BLOB": "3c6a7b28ebad5fce6187c307f87fd7fc91f81933"}


def case_pinned_constants() -> bool:
    """The real-mode pins equal the GO's literals (imported from PAIRING and FROZEN, never retyped)."""
    from src.eval import teacher_diag as td
    return all(getattr(td, k) == v for k, v in PINNED.items())


# ---------------------------------------------------------------------------------------------------
# the record role's reproduction gate (src/eval/teacher_diag.reproduction_gate; SCOPE d, AM-PC-1 finding 27)
# ---------------------------------------------------------------------------------------------------
LITERAL_R3 = 0.38576993346214294
GATE_KEYS = ["rule", "reference", "reference_source", "tolerance", "value", "delta", "passed"]


def case_gate_real_reference(td) -> bool:
    """Real mode: the reference is R3_VAL_MIOU itself, whatever the --val-reference artifact holds. A value equal
    to R3 passes against an artifact 2e-5 away; a value equal to that artifact is refused. Stub mode compares
    with the artifact. The block's keys are the outputs' gate fields, in order."""
    off = LITERAL_R3 + 2e-5
    a = td.reproduction_gate(LITERAL_R3, off, stub=False, rule="r")
    b = td.reproduction_gate(off, off, stub=False, rule="r")
    c = td.reproduction_gate(off, off, stub=True, rule="r")
    return (list(a) == GATE_KEYS and a["reference"] == LITERAL_R3 and a["reference_source"] == "R3" and a["passed"]
            and not b["passed"] and c["reference"] == off and c["reference_source"] == "--val-reference (stub mode)"
            and c["passed"])


def case_gate_constants(td) -> bool:
    """R3_VAL_MIOU is the literal 0.38576993346214294 and PAIRING's value; the tolerance is 1e-5 (PAIRING's)."""
    from scripts.gap_bootstrap_val import PAIRING
    return (td.R3_VAL_MIOU == LITERAL_R3 == PAIRING["teacher"]["reference"]
            and td.R3_TOLERANCE == 1e-5 == PAIRING["teacher"]["tolerance"]
            and td.reproduction_gate(LITERAL_R3, None, stub=False, rule="r")["reference"] == LITERAL_R3
            and td.reproduction_gate(LITERAL_R3, None, stub=False, rule="r")["tolerance"] == 1e-5)


def case_gate_tolerance(td) -> bool:
    """Real mode: a produced value 9e-6 from R3 passes and one 1.1e-5 away is refused, on both sides."""
    near = [td.reproduction_gate(LITERAL_R3 + s * 9e-6, None, stub=False, rule="r")["passed"] for s in (1, -1)]
    far = [td.reproduction_gate(LITERAL_R3 + s * 1.1e-5, None, stub=False, rule="r")["passed"] for s in (1, -1)]
    return near == [True, True] and far == [False, False]


def case_gate_none_nan(td) -> bool:
    """None and NaN are refused, in both modes."""
    return not any(td.reproduction_gate(v, LITERAL_R3, stub=s, rule="r")["passed"]
                   for v in (None, float("nan")) for s in (False, True))


def gate_cases() -> None:
    from src.eval import teacher_diag as td
    check("d the real branch's reference is R3 itself, not the --val-reference artifact: R3 passes against an "
          "artifact 2e-5 away, the artifact's own value is refused; stub mode uses the artifact; the gate fields",
          case_gate_real_reference(td))
    check("d R3_VAL_MIOU is the literal 0.38576993346214294 and PAIRING's value; the tolerance is 1e-5",
          case_gate_constants(td))
    check("d real mode: 9e-6 from R3 passes and 1.1e-5 is refused, on both sides", case_gate_tolerance(td))
    check("d None and NaN are refused", case_gate_none_nan(td))


def reference_variant(ref: Path, dst: Path, **edits) -> Path:
    """A copy of the stub --val-reference with summary.json edited (run.* or dataset_level.all_class_miou) and
    its manifest re-hashed, so only the edited field can be refused."""
    import json
    import shutil

    from scripts import teacher_diag_fixtures as fx
    shutil.copytree(ref, dst)
    sj = json.loads((dst / "summary.json").read_text())
    for k, v in edits.items():
        if k == "all_class_miou":
            sj["dataset_level"]["all_class_miou"] = v
        else:
            sj["run"][k] = v
    (dst / "summary.json").write_text(json.dumps(sj, indent=2) + "\n")
    fx.rehash_artifact(dst)
    return dst


def val_reference_real_outcomes(ref: Path, tmp: Path) -> dict:
    """check_val_reference's real branch on edited copies: True where it refuses."""
    from src.eval import teacher_diag as td
    record = dict(checkpoint_sha256=td.RECORD_SHA256, artifact_status="provisional", all_class_miou=td.R3_VAL_MIOU)
    variants = {"of record": record, "another checkpoint": dict(record, checkpoint_sha256="f" * 64),
                "a smoke status": dict(record, artifact_status="smoke"),
                "an mIoU 2e-5 from R3": dict(record, all_class_miou=td.R3_VAL_MIOU + 2e-5),
                "stage E1": dict(record, stage="E1")}
    out = {}
    for i, (name, edits) in enumerate(variants.items()):
        d = reference_variant(ref, tmp / f"ref_real_{i}", **edits)
        out[name] = raises(lambda d=d: td.check_val_reference(d, stub=False, expected_rows=3), td.Refused)
    return out


def d2_cases() -> None:
    import contextlib
    import io
    import json
    import shutil

    from scripts import teacher_d2_calibration as d2
    from scripts import teacher_diag_fixtures as fx
    from scripts.compare_eval_artifacts import identity_problems
    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    from src.eval import teacher_diag as td

    tmp = safe_tmpdir("diag_d2_")
    try:
        root, _ = fx.make_data_root("diag_d2_data_")
        fx.set_data_root(root)
        ckpt, sha = fx.write_stub_ckpt(tmp / "stub_teacher.pth")
        ref = _reference(fx, td, tmp, ckpt, sha)
        cfg = str(td.REPO / td.TEACHER_CONFIG_REL)
        utc = "2026-10-02T00:00:00Z"

        def call(*, factory=fx.stub_factory, **kw):
            base = dict(teacher_ckpt=str(ckpt), teacher_ckpt_sha256=sha, teacher_config=cfg, teacher_role="record",
                        purpose="item1", val_reference=str(ref), max_samples=3, generated_utc=utc)
            base.update(kw)
            args = d2.build_parser().parse_args(fx.argv(**base))
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = d2.run(args) if factory is None else d2.run(args, model_factory=factory)
            return code, err.getvalue()

        # item 1(b)
        out1, art1 = tmp / "out1", tmp / "art1"
        code, err = call(out_dir=out1, artifact_dir=art1)
        files = fx.output_files(out1)
        doc = json.loads(files[0].read_text()) if files else {}
        check("D2 item1: exit 0, one teacher_d2_<UTC>.json, status written, artifact_status smoke",
              code == 0 and [p.name for p in files] == ["teacher_d2_20261002T000000Z.json"]
              and doc.get("status") == "written" and doc.get("artifact_status") == "smoke", err)
        art = Path(doc.get("artifact", {}).get("dir", tmp / "missing"))
        problems, _, _ = identity_problems(ref, art) if art.is_dir() else (["no artifact"], {}, {})
        check("D2 its evaluator artifact equals scripts/evaluate_model.py run()'s under EVALUATION_CONTRACT 10(d)",
              problems == [], str(problems))
        eq = doc.get("artifact", {}).get("against_val_reference", {})
        check("D2 per_image_equal and npz_equal against --val-reference are recorded (True here)",
              eq.get("per_image_equal") is True and eq.get("npz_equal") is True)
        c, cal = doc.get("checks", {}), doc.get("calibration", {})
        check("D2 sum correct == sum TP and n == sum GT of the core; one hook call per image",
              c.get("sum_correct_equals_sum_tp") is True and cal.get("correct") == c.get("sum_tp")
              and cal.get("n") == c.get("sum_gt") and c.get("hook_calls") == c.get("forward_batches") == 3)
        s = c.get("nmf_stream", {})
        check("D2 M4-V stream seeded 42: one draw per image, the adapter's object, caller RNG unchanged",
              s.get("begin", {}).get("policy") == "M4-V" and s.get("begin", {}).get("seed") == 42
              and s.get("draws") == 3 and c.get("rng_state_unchanged") is True)
        g = doc.get("gates", {})
        check("D2 record gate against the reference mIoU passed before the write (delta 0.0)",
              isinstance(g, dict) and g.get("passed") is True and g.get("delta") == 0.0)
        t = doc.get("teacher", {})
        check("D2 teacher block: verified sha256, loaded_state_sha256, provenance, evaluator load",
              t.get("checkpoint", {}).get("sha256_verified") == sha and len(t.get("loaded_state_sha256", "")) == 64
              and t.get("provenance", {}).get("ckpt_sha256") == sha and t.get("load_mode") == "evaluator")
        check("D2 the evaluator artifact's four file hashes are recorded",
              sorted(doc.get("artifact", {}).get("files_sha256", {})) ==
              ["MANIFEST.sha256", "per_image.jsonl", "sufficient_stats.npz", "summary.json"])

        # the gate runs before the write (P18)
        import src.eval.artifacts as ea
        writes = []
        orig_write = ea.write_artifact
        ea.write_artifact = lambda *a, **k: writes.append(1) or orig_write(*a, **k)
        try:
            with fx.patched(td, R3_TOLERANCE=-1.0):
                code, err = call(out_dir=tmp / "out_gate", artifact_dir=tmp / "art_gate")
        finally:
            ea.write_artifact = orig_write
        gdoc = json.loads(fx.output_files(tmp / "out_gate")[0].read_text()) if fx.output_files(tmp / "out_gate") else {}
        check("P18 a failed gate: exit 1, 'not reproduced' with value and delta, no artifact, write_artifact never "
              "called", code == 1 and gdoc.get("status") == "not reproduced" and gdoc.get("artifact") is None
              and gdoc.get("gates", {}).get("delta") is not None and not writes
              and not (tmp / "art_gate").exists(), err)

        # control (P4)
        code, err = call(out_dir=tmp / "out_ctl", artifact_dir=tmp / "art_ctl", purpose="control",
                         of_record_output=str(files[0]))
        ctl_files = fx.output_files(tmp / "out_ctl")
        cdoc = json.loads(ctl_files[0].read_text()) if ctl_files else {}
        check("P4 control: teacher_d2_control_<UTC>.json, reproduction_check, ECE_B beside the of-record ECE",
              code == 0 and [p.name for p in ctl_files] == ["teacher_d2_control_20261002T000000Z.json"]
              and cdoc.get("reproduction_check") is True
              and cdoc.get("control", {}).get("ece_b") == cdoc.get("control", {}).get("ece_of_record") is not None
              and cdoc.get("inputs", {}).get("of_record_output", {}).get("sha256") == td.file_sha256(files[0]), err)
        check("P4 control without --of-record-output, item1 with one, and an of-record file that is not item1 "
              "are refused (exit 2)",
              call(out_dir=tmp / "o_c1", artifact_dir=tmp / "a_c1", purpose="control")[0] == 2
              and call(out_dir=tmp / "o_c2", artifact_dir=tmp / "a_c2", of_record_output=str(files[0]))[0] == 2
              and call(out_dir=tmp / "o_c3", artifact_dir=tmp / "a_c3", purpose="control",
                       of_record_output=str(ctl_files[0]))[0] == 2)

        # arm (P3)
        cfg_copy = Path(shutil.copy(cfg, tmp / "arm_cfg.py"))
        arm = dict(teacher_role="arm", purpose=None, arm_id="R2", arm_dl_id="DL-70", teacher_config=str(cfg_copy),
                   teacher_config_sha256=td.file_sha256(cfg_copy))
        code, err = call(out_dir=tmp / "out_arm", artifact_dir=tmp / "art_arm", **arm)
        arm_files = fx.output_files(tmp / "out_arm")
        adoc = json.loads(arm_files[0].read_text()) if arm_files else {}
        check("P3 arm: teacher_d2_arm-R2_<UTC>.json, gates nothing, artifact hashes recorded (R2)",
              code == 0 and [p.name for p in arm_files] == ["teacher_d2_arm-R2_20261002T000000Z.json"]
              and adoc.get("gates") == "nothing" and adoc.get("teacher", {}).get("arm_id") == "R2"
              and len(adoc.get("artifact", {}).get("files_sha256", {})) == 4, err)
        check("P3 arm with --purpose is refused",
              call(out_dir=tmp / "o_a1", artifact_dir=tmp / "a_a1", **dict(arm, purpose="item1"))[0] == 2)

        # the real-mode pins and --val-reference's real branch
        check("P3/P18 the real-mode pins equal the GO's literals (record sha256, R3, its tolerance, config blob)",
              case_pinned_constants())
        out = val_reference_real_outcomes(ref, tmp)
        check("P18 real --val-reference: the record checkpoint, a real status and an mIoU within 1e-5 of R3 pass; "
              "another checkpoint, a smoke status, an mIoU 2e-5 away and stage E1 are refused",
              out == {"of record": False, "another checkpoint": True, "a smoke status": True,
                      "an mIoU 2e-5 from R3": True, "stage E1": True}, str(out))

        # a control re-score of another teacher
        other = json.loads(files[0].read_text())
        other["teacher"]["loaded_state_sha256"] = "0" * 64
        other_p = tmp / "item1_other_teacher.json"
        other_p.write_text(json.dumps(other))
        code, err = call(out_dir=tmp / "o_c4", artifact_dir=tmp / "a_c4", purpose="control", of_record_output=str(other_p))
        check("P4 a control re-score whose teacher differs from the of-record output's is refused, nothing written",
              code == 2 and "teacher differs" in err and not fx.output_files(tmp / "o_c4"), err)

        # refusals before any load
        bad_split = tmp / "ref_split"
        shutil.copytree(ref, bad_split)
        sj = json.loads((bad_split / "summary.json").read_text())
        sj["dataset"]["split"] = "t" + "est"
        (bad_split / "summary.json").write_text(json.dumps(sj))
        verifies = []
        orig_verify = ea.verify_artifact
        ea.verify_artifact = lambda *a, **k: verifies.append(1) or orig_verify(*a, **k)
        try:
            with fx.count_loads() as calls:
                code, _ = call(out_dir=tmp / "o_r1", artifact_dir=tmp / "a_r1", val_reference=str(bad_split))
        finally:
            ea.verify_artifact = orig_verify
        check("P18 a --val-reference whose summary names another split is refused from summary.json alone "
              "(no verify_artifact, no load)", code == 2 and not verifies and sum(calls.values()) == 0)
        bad_man = tmp / "ref_manifest"
        shutil.copytree(ref, bad_man)
        sj = json.loads((bad_man / "summary.json").read_text())
        sj["dataset"]["split_manifest_sha256"] = "0" * 64
        (bad_man / "summary.json").write_text(json.dumps(sj, indent=2) + "\n")
        _rewrite_manifest(bad_man)
        with fx.count_loads() as calls:
            code, err = call(out_dir=tmp / "o_r2", artifact_dir=tmp / "a_r2", val_reference=str(bad_man))
        check("D0-3 the pass's VAL manifest must equal --val-reference's (manifest identity), before any load",
              code == 2 and "manifest" in err and sum(calls.values()) == 0, err)
        tampered = tmp / "ref_tampered"
        shutil.copytree(ref, tampered)
        (tampered / "per_image.jsonl").write_text((tampered / "per_image.jsonl").read_text() + "\n")
        check("P18 a --val-reference failing verify_artifact is refused (exit 2)",
              call(out_dir=tmp / "o_r3", artifact_dir=tmp / "a_r3", val_reference=str(tampered))[0] == 2)
        check("P18 a --val-reference with another row count is refused",
              call(out_dir=tmp / "o_r4", artifact_dir=tmp / "a_r4", max_samples=2)[0] == 2)
        with fx.count_loads() as calls:
            r_repo = call(out_dir=td.REPO / "d2_out_in_repo", artifact_dir=tmp / "a_r5")[0]
            r_same = call(out_dir=out1, artifact_dir=tmp / "a_r6")[0]
            r_sha = call(out_dir=tmp / "o_r7", artifact_dir=tmp / "a_r7", teacher_ckpt_sha256=sha.upper())[0]
        check("P27/P5 an in-repo --out-dir, an existing output name and an uppercase sha256 are refused, load "
              "counter 0", (r_repo, r_same, r_sha) == (2, 2, 2) and sum(calls.values()) == 0
              and not (td.REPO / "d2_out_in_repo").exists())
        commit = dict(script_commit="c" * 40, script_commit_dl_id="DL-61", generated_utc=None, max_samples=None)
        with fx.count_loads() as calls:
            r_real = call(factory=None, out_dir=tmp / "o_r8", artifact_dir=tmp / "a_r8", **commit)[0]
            r_c2 = call(factory=None, out_dir=tmp / "o_r9", artifact_dir=tmp / "a_r9",
                        **dict(commit, generated_utc=utc))[0]
            r_ms = call(factory=None, out_dir=tmp / "o_r10", artifact_dir=tmp / "a_r10", **dict(commit, max_samples=3))[0]
        check("P2/P8/C2 real mode (no stub): refused before any load -- the provenance count, --generated-utc, "
              "--max-samples", (r_real, r_c2, r_ms) == (2, 2, 2) and sum(calls.values()) == 0)
        quiet = td.RngWatchForward(lambda m, x: x)
        drawing = td.RngWatchForward(lambda m, x: x + torch.rand(1))
        for w in (quiet, quiet, drawing, drawing):
            w(None, torch.zeros(1))
        check("P9 the RNG watch passes a forward that leaves the CPU RNG alone and stops one that draws from it",
              quiet.unchanged() and not raises(quiet.require_unchanged, Exception)
              and raises(drawing.require_unchanged, td.Stop))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            m_usage = d2.main([])
            m_real = d2.main(fx.argv(out_dir=tmp / "o_m", artifact_dir=tmp / "a_m", teacher_ckpt=str(ckpt),
                                     teacher_ckpt_sha256=sha, teacher_config=cfg, teacher_role="record",
                                     purpose="item1", val_reference=str(ref), script_commit="c" * 40,
                                     script_commit_dl_id="DL-61"))
        check("P29 the CLI: a usage error exits 2; the CLI is real mode and refuses a --script-commit that is not HEAD "
              "(exit 2)",
              m_usage == 2 and m_real == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        root_base = Path(os.environ.get("PLANTSEG_DATA_ROOT", "")).parent
        if root_base.name.startswith("diag_d2_data_"):
            shutil.rmtree(root_base, ignore_errors=True)


def main() -> int:
    for fn in (unit_cases, hook_cases, gate_cases, d2_cases):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- a crash is a failed case, never a pass
            check(f"{fn.__name__} raised", False, f"{type(e).__name__}: {e}")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    good = sum(ok for _, ok, _ in RESULTS)
    print(f"RESULT: {'PASS' if good == len(RESULTS) else 'FAIL'} ({good}/{len(RESULTS)})")
    return 0 if good == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
