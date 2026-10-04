#!/usr/bin/env python3
"""D1: NMF-draw sensitivity of the distillation targets (AM-18 item 1(a); lane L-TEACHER-DIAG).

--part crops (record or arm role)                         teacher_d1_<UTC>.json, teacher_d1_arm-<id>_<UTC>.json
  Sample: random.Random(1801).sample(sorted TRAIN ids, 256), the AM-10 procedure
  (src/quant/calibration.build_calibration_index); the TRAIN split list's sha256 must equal the strata
  file's. Crop i (sample order) is one pass of the KD trainer's train_preprocess with
  np.random.RandomState(1801 + i), then finalize (the trainer's ImageNet normalisation). The teacher is
  loaded in the KD trainer's form (load_frozen_teacher; M4-KD stream seeded 42 begun once); per crop the
  backbone runs once and the decode head K = 8 times, each head call advancing the one stream by one
  draw (2,048 draws for 256 crops). Domain: the 64x64 logit grid restricted to the trainer's all-valid
  min-pooled mask (losses.downsample_validity). Statistics: src/eval/nmf_sensitivity.py (F decided in
  integers, F_lesion, F_halves, KL_logit, KL_cwd, d2_crops, the image-level bootstrap). The record role
  writes the branch booleans; the arm role gates nothing and writes N, D and F only (P3).
  --d2-val-output (P11): the same teacher's D2 JSON (record: the item 1(b) output; arm: that arm's), whose
  checkpoint sha256, config sha256, loaded_state_sha256 and provenance must equal this load's.
  D == 0 writes status "F not produced" and exits 1 (P12).

--part val (record role only)                            teacher_d1val_<UTC>.json
  VAL in the R3 manifest order, batch 1, canvas protocol, CPU. The evaluator's own load
  (load_teacher_model) begins the M4-V stream seeded 42; streams 43..49 are NMFStream(s, "M4-V"). Per
  image: the backbone once, then for each stream: attach, check identity, the head, draws == image
  index + 1, the bilinear resize to the canvas (FrozenTeacher's call), its confusion matrix, its ECE and
  its float64 softmax added to the mean-probability sum. The evaluator core scores stream 42's map (its
  confusion must equal stream 42's own). The forward is a real ModelDeviceForward handed to
  check_post_eval; no evaluator artifact is written (P10). Stream 42's all-class mIoU must reproduce R3
  within 1e-5, else status "not reproduced", the statistics are withheld and the exit code is 1.

Both parts take --correction-state {available,declined,no_approval} and --correction-dl-id DL-<n>
(P28; required in a real run, checked before any file is read). Exit codes: 0 written, 1 STOP, 2 refusal
or usage, 4 unexpected exception. Real mode (the CLI) refuses every run until K-part (P8); the stub
teacher is reachable only from scripts/smoke_teacher_d1.py through run(args, model_factory=...).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval import teacher_diag as td  # noqa: E402

SCRIPT = "scripts/teacher_d1_nmf_sensitivity.py"
D2_SCRIPT = "scripts/teacher_d2_calibration.py"
SCHEMA_CROPS = "plantseg-teacher-d1/1.0.0"
SCHEMA_VAL = "plantseg-teacher-d1val/1.0.0"
PARTS = ("crops", "val")
SAMPLE_COUNT, SAMPLE_SEED, CROP_SEED_BASE = 256, 1801, 1801
VAL_SEEDS = tuple(range(42, 50))
NUM_CLASSES, IGNORE_INDEX = 116, 255
GRID = (64, 64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="D1: NMF-draw sensitivity of the teacher (AM-18 item 1(a)).")
    p.add_argument("--part", required=True, help="crops or val")
    td.add_common_flags(p)
    td.add_teacher_flags(p)
    td.add_correction_flags(p)
    p.add_argument("--strata", help="crops: the AM-17 item 8 strata file (reports/strata/train_strata_v1.json)")
    p.add_argument("--d2-val-output", help="crops: the same teacher's teacher_d2 JSON (P11)")
    p.add_argument("--val-reference", help="val: the teacher-stage canvas VAL artifact (R3-equivalent)")
    p.add_argument("--n-crops", type=int, default=None, help="stub mode only: the first n of the 256 crops")
    p.add_argument("--max-samples", type=int, default=None, help="stub mode only: evenly spaced VAL subset")
    return p


def check_d1_flags(args, *, real: bool) -> None:
    if args.part not in PARTS:
        raise td.Refused(f"--part must be one of {PARTS}")
    if real or args.correction_state is not None or args.correction_dl_id is not None:
        td.check_correction_flags(args)
    if args.part == "crops":
        if not args.strata or not args.d2_val_output:
            raise td.Refused("--part crops requires --strata and --d2-val-output")
        if args.val_reference is not None or args.max_samples is not None:
            raise td.Refused("--val-reference and --max-samples belong to --part val")
        if args.n_crops is not None and (real or not 1 <= args.n_crops <= SAMPLE_COUNT):
            raise td.Refused(f"--n-crops is a stub-mode flag in 1..{SAMPLE_COUNT}")
    else:
        if not args.val_reference:
            raise td.Refused("--part val requires --val-reference")
        if args.strata is not None or args.d2_val_output is not None or args.n_crops is not None:
            raise td.Refused("--strata, --d2-val-output and --n-crops belong to --part crops")
        if args.max_samples is not None and (real or not 1 <= args.max_samples <= td.VAL_ROWS):
            raise td.Refused(f"--max-samples is a stub-mode flag in 1..{td.VAL_ROWS}")


def output_kind(args) -> str:
    if args.part == "val":
        return "teacher_d1val"
    return f"teacher_d1_arm-{args.arm_id}" if args.teacher_role == "arm" else "teacher_d1"


# --------------------------------------------------------------------------------------------------
# the crops part
# --------------------------------------------------------------------------------------------------
def sample_ids(stems, count: int = SAMPLE_COUNT, seed: int = SAMPLE_SEED) -> list[str]:
    """random.Random(seed).sample(sorted(ids), count): the AM-10 procedure, checked against its literal."""
    from src.quant.calibration import build_calibration_index
    picked = build_calibration_index(list(stems), count=count, seed=seed)["selected_ids"]
    if picked != random.Random(seed).sample(sorted(stems), count):
        raise td.Stop("build_calibration_index disagrees with random.Random(seed).sample(sorted(ids), count)")
    return picked


def crop(pair, i: int):
    """Crop i: the KD trainer's TRAIN branch with RandomState(1801 + i), then finalize."""
    import numpy as np
    from PIL import Image

    from configs.augment import AUGMENT
    from src.data.transforms import finalize, train_preprocess
    img_path, mask_path = pair
    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        img_np, mask_np = train_preprocess(im, mk, np.random.RandomState(CROP_SEED_BASE + i), AUGMENT)
    return finalize(img_np, mask_np)


def valid_cells(mask):
    """The trainer's all-valid min-pooled mask on the 64x64 logit grid (losses.downsample_validity)."""
    from src.training.losses import downsample_validity
    return downsample_validity(mask[None], GRID, ignore_index=IGNORE_INDEX)[0].numpy()


def _run_crops(args, *, model_factory, stub, start, t0, code, kind, repeat, out_path) -> int:
    import numpy as np
    import torch

    from src.eval import nmf_sensitivity as ns

    role = args.teacher_role
    d2doc, d2sha = td.read_diag_output(args.d2_val_output, script=D2_SCRIPT, stub=stub, what="--d2-val-output")
    d2t = d2doc.get("teacher") or {}
    if role == "record" and (d2doc.get("purpose") != "item1" or d2t.get("role") != "record"):
        raise td.Refused("--d2-val-output is not the record role's item 1(b) output")
    if role == "arm" and (d2t.get("role") != "arm" or d2t.get("arm_id") != args.arm_id):
        raise td.Refused(f"--d2-val-output is not arm {args.arm_id}'s D2 output")
    inputs = td.verify_teacher_inputs(args, stub=stub)

    from configs.data import DATA
    td.refuse_test_path(DATA["root"], "PLANTSEG_DATA_ROOT")
    m11 = td.check_m11(DATA["root"])
    from scripts.build_train_strata import train_split_list
    stems = [s for s, _ in train_split_list(DATA["root"])]
    strata = td.check_strata(args.strata, stems, stub=stub)
    sample_all = sample_ids(stems)
    sample = sample_all[:args.n_crops] if args.n_crops is not None else sample_all

    from src.data.dataset import PlantSegDataset
    pairs = PlantSegDataset("train").pairs
    if [p[0].stem for p in pairs] != stems:
        raise td.Stop("PlantSegDataset('train') pairs do not follow the TRAIN split list")
    index = {s: k for k, s in enumerate(stems)}

    loaded = td.load_teacher("kd", inputs, model_factory=model_factory)
    checks = td.after_load_checks(loaded, inputs, stub=stub)
    teacher = td.teacher_record(loaded, inputs, checks, stub=stub)
    diffs = td.same_teacher(teacher, d2t)
    if diffs:
        raise td.Refused(f"--d2-val-output names another teacher (P11): {diffs}")

    split, stream = td.SplitTeacher(loaded), loaded.stream
    crops, per_crop, rng_hashes = [], [], []
    for i, sid in enumerate(sample):
        image, mask = crop(pairs[index[sid]], i)
        valid = valid_cells(mask)
        rng_hashes.append(td.rng_state_sha256())
        feats = split.features(image[None])
        h = td.feature_sha256(feats)
        zs = [split.head(feats, stream, feat_hash=h, expect_shape=(1, NUM_CLASSES) + GRID)
              for _ in range(ns.K_DRAWS)]
        rng_hashes.append(td.rng_state_sha256())
        cs = ns.crop_statistics(torch.cat(zs).numpy(), valid)
        crops.append(cs)
        per_crop.append({"i": i, "id": sid, "seed": CROP_SEED_BASE + i, "n_valid": cs.n_valid, "N": cs.N, "D": cs.D,
                         "N_L": cs.N_L, "D_L": cs.D_L, "n_lesion": cs.n_lesion, "H": cs.H,
                         "kl_logit_sum": cs.kl_logit_sum, "kl_cwd_sum": cs.kl_cwd_sum, "n_cwd": cs.n_cwd,
                         "feature_sha256": h})
        if i + 1 == 8 and len(sample) > 8:
            per = (time.monotonic() - t0) / 8
            print(f"[d1 crops] 8 crops in {time.monotonic() - t0:.1f} s; about {per * (len(sample) - 8) / 60:.1f} "
                  "min to go", file=sys.stderr)

    n = len(sample)
    if stream.draws != ns.K_DRAWS * n or loaded.adapter.nmf_stream is not stream:
        raise td.Stop(f"M4-KD stream: {stream.draws} draws, expected {ns.K_DRAWS * n}")
    if split.backbone_calls != n or split.head_calls != ns.K_DRAWS * n:
        raise td.Stop(f"{split.backbone_calls} backbone and {split.head_calls} head calls for {n} crops")
    if len(set(rng_hashes)) != 1:
        raise td.Stop("the caller's CPU RNG state moved between the first and the last forward")

    gated = role == "record"
    stats = ns.summarize(crops, gated=gated)
    stats["F"]["ci95"] = ns.bootstrap_f([c.N for c in crops], [c.D for c in crops])
    stats["F"]["ci95"]["note"] = "image-level percentile bootstrap; only the point estimate selects the branch"
    produced = stats["F"]["status"] != "F not produced"
    doc = td.base_document(SCRIPT, args, stub=stub, start_utc=start, code=code, extra={
        "schema": SCHEMA_CROPS, "part": "crops", "role": role, "arm_id": args.arm_id, "purpose": None,
        "repeat": repeat,
        "gates": "the F branch (integers): K = 8 averaged targets or not" if gated else "nothing",
        "correction": {"state": args.correction_state, "dl_id": args.correction_dl_id},
        "teacher": teacher,
        "inputs": {"strata": strata, "d2_val_output": {"path": str(args.d2_val_output), "sha256": d2sha},
                   "data_root": str(DATA["root"]), "m11": m11},
        "sample": {"rule": "random.Random(1801).sample(sorted TRAIN ids, 256) (build_calibration_index)",
                   "count": n, "count_of_record": SAMPLE_COUNT, "ids": sample,
                   "ids_sha256": hashlib.sha256("".join(f"{s}\n" for s in sample).encode("utf-8")).hexdigest(),
                   "split_list_sha256": strata["split_list_sha256"],
                   "crop_rule": "train_preprocess(RandomState(1801 + i), AUGMENT) then finalize; batch 1"},
        "nmf": {"policy": "M4-KD", "begin": loaded.stream_description, "end": stream.describe(),
                "draws": stream.draws, "expected_draws": ns.K_DRAWS * n, "k": ns.K_DRAWS},
        "checks": {"backbone_calls": split.backbone_calls, "head_calls": split.head_calls,
                   "feature_hash_unchanged_across_head_calls": True, "rng_state_unchanged": len(set(rng_hashes)) == 1,
                   "rng_state_sha256": rng_hashes[0] if rng_hashes else None,
                   "logit_grid": list(GRID), "domain": "losses.downsample_validity all-valid min-pool"},
        "per_crop": per_crop,
        **stats,
        "status": "written" if produced else "F not produced",
    })
    td.finish_output(out_path, doc, start, t0)
    return td.EXIT_OK if produced else td.EXIT_STOP


# --------------------------------------------------------------------------------------------------
# the VAL part
# --------------------------------------------------------------------------------------------------
class MeanProbability:
    """The mean-probability prediction: float64 softmax (T = 1) of each draw, summed; argmax (first max).
    Probabilities are averaged, never logits (dividing by 8 cannot change the argmax)."""

    def __init__(self):
        self.total = None
        self.count = 0

    def add(self, logits) -> None:
        import torch
        p = torch.softmax(logits.to(torch.float64), dim=1)
        if self.total is None:
            self.total = p
        else:
            self.total.add_(p)
        self.count += 1

    def prediction(self):
        return self.total.argmax(1)


class EightStreamForward:
    """`evaluate_model(batch_forward=...)` of the D1 VAL part (P10): one backbone pass per image, the head
    once on each of eight streams, per-stream confusion and ECE, the mean-probability sum; it returns
    stream 42's resized map for the core to score. `fwd` is the ModelDeviceForward (or a wrapper of it)."""

    def __init__(self, split, streams, fwd):
        import torch

        from src.eval.calibration import CalibrationAccumulator
        self.split, self.streams, self.fwd = split, list(streams), fwd
        self.cms = [torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long) for _ in self.streams]
        self.cm_mean = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        self.accs = [CalibrationAccumulator() for _ in self.streams]
        self.images = 0
        self._targets = None

    def __call__(self, model, batch):
        if batch.images.shape[0] != 1:
            raise td.Stop("the D1 VAL part runs at batch size 1")
        self._targets = batch.targets.cpu()
        try:
            out = self.fwd(self._forward_all, batch.images)
        finally:
            self._targets = None
        self.images += 1
        return out

    def _forward_all(self, x):
        import torch.nn.functional as F

        from src.eval.metrics import confusion_matrix
        t = self._targets
        feats = self.split.features(x)
        h = td.feature_sha256(feats)
        mean, out42 = MeanProbability(), None
        for k, s in enumerate(self.streams):
            self.split.attach(s)
            z = self.split.head(feats, s, feat_hash=h, expect_shape=(1, NUM_CLASSES) + GRID)
            if s.draws != self.images + 1:
                raise td.Stop(f"stream {s.seed}: {s.draws} draws at image {self.images}, expected {self.images + 1}")
            if tuple(z.shape[-2:]) != tuple(x.shape[-2:]):          # FrozenTeacher's resize, same call
                z = F.interpolate(z, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)
            z = z.detach()
            self.cms[k] += confusion_matrix(z.argmax(1)[0].cpu(), t[0], NUM_CLASSES, IGNORE_INDEX)
            self.accs[k].update(z, t)
            mean.add(z)
            if k == 0:
                out42 = z
        self.split.attach(self.streams[0])                          # stream 42 back on the head
        self.cm_mean += confusion_matrix(mean.prediction()[0].cpu(), t[0], NUM_CLASSES, IGNORE_INDEX)
        return out42


def spread(values) -> dict | None:
    vals = [None if v is None else float(v) for v in values]
    if any(v is None for v in vals) or len(vals) < 2:
        return None
    mean = math.fsum(vals) / len(vals)
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return {"n": len(vals), "mean": mean, "sd": sd, "sd_ddof": 1, "min": min(vals), "max": max(vals),
            "range": max(vals) - min(vals)}


def _counts(cm):
    """(TP, GT, PRED) per class from a [GT, pred] confusion matrix, as the core's int64 totals."""
    import numpy as np
    return (cm.diag().numpy().astype(np.int64), cm.sum(1).numpy().astype(np.int64),
            cm.sum(0).numpy().astype(np.int64))


def _run_val(args, *, model_factory, stub, start, t0, code, kind, repeat, out_path) -> int:
    import numpy as np

    from src.distill.nmf_stream import NMFStream
    from src.eval.adapters import deterministic_subset
    source_indices = (deterministic_subset(td.VAL_ROWS, args.max_samples) if args.max_samples is not None
                      else list(range(td.VAL_ROWS)))
    ref = td.check_val_reference(args.val_reference, stub=stub, expected_rows=len(source_indices))
    inputs = td.verify_teacher_inputs(args, stub=stub)

    from configs.data import DATA
    td.refuse_test_path(DATA["root"], "PLANTSEG_DATA_ROOT")
    m11 = td.check_m11(DATA["root"])

    from src.eval import Condition, evaluate_model
    from src.eval.adapters import build_eval_loader, build_expected_manifest_for, build_protocol_adapter
    from src.eval.artifacts import hash_split_manifest
    from src.eval.eval_runtime import (EVAL_NUM_WORKERS, ModelDeviceForward, check_post_eval,
                                       evaluate_capturing_warnings, resolve_model_device)
    from src.eval.metrics import miou_from_confusion
    from src.eval.protocols import CANVAS, PROTOCOL_IDS
    expected_manifest = build_expected_manifest_for("val", source_indices)
    manifest_sha = hash_split_manifest(expected_manifest)
    if manifest_sha != ref["split_manifest_sha256"]:
        raise td.Refused("this pass's VAL manifest hash differs from --val-reference's")
    adapter = build_protocol_adapter(CANVAS, "val", source_indices)

    loaded = td.load_teacher("evaluator", inputs, model_factory=model_factory)
    checks = td.after_load_checks(loaded, inputs, stub=stub)
    teacher = td.teacher_record(loaded, inputs, checks, stub=stub)
    streams = [loaded.stream] + [NMFStream(s, "M4-V") for s in VAL_SEEDS[1:]]
    if [s.seed for s in streams] != list(VAL_SEEDS) or any(s.policy != "M4-V" for s in streams):
        raise td.Stop("the eight VAL streams must be M4-V seeded 42..49, stream 42 the evaluator's own")

    model_device = resolve_model_device("cpu", cpu_only=True)
    fwd = ModelDeviceForward(model_device)
    watch = td.RngWatchForward(fwd, total=len(source_indices), label="d1 val")
    split = td.SplitTeacher(loaded)
    eight = EightStreamForward(split, streams, watch)
    loader = build_eval_loader(adapter, 1, num_workers=EVAL_NUM_WORKERS)
    result, warn_summary = evaluate_capturing_warnings(
        evaluate_model, loaded.eval_model, loader, expected_manifest=expected_manifest,
        condition=Condition("clean", None, None), num_classes=NUM_CLASSES, background_index=0,
        ignore_index=IGNORE_INDEX, batch_forward=eight)
    check_post_eval(model=loaded.eval_model, model_device=model_device, fwd=fwd, policy_applied=False)

    n = len(source_indices)
    from src.distill.nmf_stream import isolated_nmf_modules
    mods = isolated_nmf_modules(loaded.segmentor)
    if any(s.draws != n for s in streams):
        raise td.Stop(f"stream draws {[s.draws for s in streams]}, expected {n} each")
    if len(mods) != 1 or mods[0].nmf_stream is not loaded.stream or loaded.adapter.nmf_stream is not loaded.stream:
        raise td.Stop("stream 42 is not re-attached to the head at the end of the pass")
    rng_state = watch.require_unchanged()
    tp, gt, pr = _counts(eight.cms[0])
    core_equal = (np.array_equal(tp, result.dataset_tp) and np.array_equal(gt, result.dataset_gt)
                  and np.array_equal(pr, result.dataset_pred))
    if not core_equal:
        raise td.Stop("stream 42's confusion differs from the evaluator core's")

    def finite(x):
        return None if x is None or math.isnan(x) else float(x)

    miou42 = result.dataset_level["all_class_miou"]
    reference = ref["all_class_miou"] if stub else td.R3_VAL_MIOU
    delta = None if miou42 is None else abs(miou42 - reference)
    passed = delta is not None and delta <= td.R3_TOLERANCE
    gate = {"rule": "stream 42's |all_class_miou - reference| <= tolerance", "reference": reference,
            "reference_source": "--val-reference (stub mode)" if stub else "R3", "tolerance": td.R3_TOLERANCE,
            "value": miou42, "delta": delta, "passed": passed}
    doc = td.base_document(SCRIPT, args, stub=stub, start_utc=start, code=code, extra={
        "schema": SCHEMA_VAL, "part": "val", "role": "record", "arm_id": None, "purpose": None, "repeat": repeat,
        "gates": "nothing (descriptive); stream 42 must reproduce R3 for the statistics to be written",
        "reproduction": gate,
        "correction": {"state": args.correction_state, "dl_id": args.correction_dl_id},
        "teacher": teacher,
        "inputs": {"val_reference": ref, "data_root": str(DATA["root"]), "m11": m11},
        "pass": {"split": "val", "rows": n, "protocol": PROTOCOL_IDS[CANVAS], "device": "cpu", "batch_size": 1,
                 "split_manifest_sha256": manifest_sha, "eval_warnings": warn_summary["eval_warnings"],
                 "evaluator_artifact": None},
        "checks": {"core_confusion_equals_stream_42": core_equal, "rng_state_unchanged": watch.unchanged(),
                   "rng_state_sha256": rng_state,
                   "backbone_calls": split.backbone_calls, "head_calls": split.head_calls,
                   "stream_42_reattached": mods[0].nmf_stream is loaded.stream, "input_devices": sorted(fwd.input_devices)},
        "nmf": {"policy": "M4-V", "seeds": list(VAL_SEEDS), "stream_42_begin": loaded.stream_description,
                "draws": [s.draws for s in streams], "end_state_sha256": [s.state_sha256() for s in streams]},
    })
    if not passed:
        doc["status"] = "not reproduced"
        td.finish_output(out_path, doc, start, t0)
        print(f"STOP: not reproduced: stream 42 all_class_miou {miou42!r}, delta {delta!r}", file=sys.stderr)
        return td.EXIT_STOP

    per_stream = []
    for s, cm, acc in zip(streams, eight.cms, eight.accs):
        cal = acc.result()
        per_stream.append({"seed": s.seed, "all_class_miou": finite(miou_from_confusion(cm)), "ece": cal["ece"],
                           "ece_disease": cal["ece_disease"], "n": cal["n"], "correct": cal["correct"],
                           "draws": s.draws, "end_state_sha256": s.state_sha256()})
    doc["per_stream"] = per_stream
    doc["spread"] = {key: spread([r[key] for r in per_stream]) for key in ("all_class_miou", "ece", "ece_disease")}
    doc["mean_probability"] = {"all_class_miou": finite(miou_from_confusion(eight.cm_mean)),
                               "definition": "argmax of the mean over the eight draws of softmax(z / 1) on the "
                                             "canvas (float64); probabilities averaged, not logits"}
    doc["status"] = "written"
    td.finish_output(out_path, doc, start, t0)
    return td.EXIT_OK


# --------------------------------------------------------------------------------------------------
def run(args, *, model_factory=None) -> int:
    return td.run_with_exit_codes(_run, args, model_factory=model_factory)


def _run(args, *, model_factory=None) -> int:
    stub = model_factory is not None
    real = not stub
    start, t0 = td.utc_now(), time.monotonic()
    td.check_common_flags(args, real=real)
    check_d1_flags(args, real=real)
    td.check_teacher_flags(args, roles=td.ROLES if args.part == "crops" else ("record",))
    td.require_provenance_field_count(real)
    out_dir = td.require_outside_repo(args.out_dir, "--out-dir")
    code = td.require_commit_binding(args.script_commit, args.script_commit_dl_id) if real else td.code_provenance()
    kind = output_kind(args)
    repeat = td.require_single_output(out_dir, kind, args)
    out_path = out_dir / f"{kind}_{td.output_stamp(args, start)}.json"
    if out_path.exists():
        raise td.Refused(f"{out_path.name} already exists")
    body = _run_crops if args.part == "crops" else _run_val
    return body(args, model_factory=model_factory, stub=stub, start=start, t0=t0, code=code, kind=kind,
                repeat=repeat, out_path=out_path)


def main(argv=None) -> int:
    return td.cli_main(build_parser(), run, argv)


if __name__ == "__main__":
    sys.exit(main())
