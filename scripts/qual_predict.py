#!/usr/bin/env python3
"""QUAL-1, step 2 of 3 (AM-27 items 1, 2, 4 and 5): one model of record on S, and the check that ties each
picture to the record.

One process per model, as on TEST day. The model file is hashed against reports/derived/tday_manifest.json
before anything parses it, then validated and loaded by the evaluator's own functions, in the evaluator's own
order (scripts/evaluate_model.py run(); scripts/evaluate_corruptions.py run()). Inference is the evaluator's
own pass: build_protocol_adapter / PlantSegCorruptedEvalDataset, build_eval_loader (batch 1, 0 workers),
ModelDeviceForward on the CPU, evaluate_capturing_warnings(evaluate_model, ...), check_post_eval, with T = 1
thread. A forward hook copies `logits.argmax(1)` (the evaluator's own prediction, evaluate.py:342) and a loader
tap copies each batch; nothing the evaluator computes is changed.

The teacher (M4-V): the record scored the whole split in one pass with one NMF draw per row from a private
stream seeded 42, so row m was predicted after m draws. Only the images of S are read here; before each of
them the stream is advanced by the draws of the rows not read, each a discarded call of the module's own
_build_bases with the arguments of the real call (QUAL-1 Q1). Draws are counted and held to the row's index.

The record of each condition is loaded and checked before its pass (ruling QUAL-AUD-1 F-02, F-03): its
identity, and on TEST its execution context (T = 1, batch 1, model device cpu, the image this run runs in), else
STOP record_context before anything of it is predicted. The check (AM-27 item 4; tile_verdict): for every image and
condition, the evaluator's per-image disease-only mIoU of this prediction against the official value: within 1e-6
when the record's device class (summary run.eval_runtime.model_device) is this run's, within 0.5 pp otherwise; on
the same device class the per-image confusion counts must also equal the record's NPZ, class by class (QUAL-A-1
Q6). The counts are computed once more from the saved arrays and held to the pass's (N-14). A failed tile ends
the step with STOP and the render step refuses to draw anything.

Writes into <run>: arrays/pred/<model>/<condition>/<position>.png (class-index PNG), arrays/input/<condition>/
<position>.png (the 512x512 canvas, proven to be the bytes the model saw), arrays/mask/<position>.png,
arrays/original/ (position 2: the image and its mask as read, held to the canvas through core_preprocess: N-09),
values/<model>.json (the per-image values of record that a figure may print), checks/<model>.json and
records/predict_<model>.json (per condition and position, without a name: the sha256 of the image file, of the mask
file and of the cached copy read: F-06). The console shows counts, hashes and PASS or STOP lines only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.qual_select import (BATCH, CELLS, CLEAN, CONDITIONS, DEVICE, MODELS, N_SPLIT,  # noqa: E402
                                 P50_POSITION, RECORD_FORMAT, THREADS, QualStop, add_cache_args, cache_args_given,
                                 canonical_bytes, check_record_identity, counts_map, expected_commit, expected_hashes,
                                 k_of, load_test_official, load_val_official, open_cell,
                                 open_test_inventory, outputs_absent, read_s, record_context_problems, record_label,
                                 repo_state, run_dir, run_step, same_device_class, say, sha256_file, version_guard,
                                 write_new)

TEACHER_CFG_REL = "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
TEACHER_CFG_SHA256 = "1b94aa32a7d22be647d8e60a42daa1dd8dd756158b586feec6465385e299ae4a"    # DL-88
SAME_DEVICE_TOL = 1e-6       # AM-27 item 4
OTHER_DEVICE_TOL = 0.005     # 0.5 pp, in the record's unit (mIoU in [0, 1])
M4V = ("M4-V", 42)


# ------------------------------------------------------------------------------------------ the model file
def hash_first(path, expected: str, what: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise QualStop("model_file_missing", f"{what}: the file is not there")
    got = sha256_file(p)
    if got != expected:
        raise QualStop("model_sha256", f"{what}: sha256 {got[:12]}... is not the manifest's {expected[:12]}...; "
                                       "it was not parsed")
    return p


def load_model(model: str, args, expected: dict):
    """The evaluator's validation and load, in its order (evaluate_model.py:449-594). Returns (model, sha256 of
    the loaded file, quant engine or None, inputs record)."""
    from src.eval.model_loading import sha256_file as ev_sha256
    inputs = {}
    if model == "E6":
        from src.eval.model_loading import load_int8_student, select_int8_backend
        from src.eval.stage_artifacts import int8_official_error, resolve_evaluation_source
        prov = hash_first(args.provenance, expected["provenance"], "E6 provenance")
        resolved = resolve_evaluation_source("E6", provenance=str(prov))
        err = int8_official_error(resolved, repo_root=REPO)
        if err:
            raise QualStop("int8_official_gate", "the official INT8 gate refused the artifact")
        engine = select_int8_backend(require_qnnpack=True)
        if resolved["artifact_sha256"] != expected["converted"]:
            raise QualStop("model_sha256", "the converted artifact is not the manifest's; it was not loaded")
        if ev_sha256(Path(resolved["artifact_path"])) != expected["converted"]:
            raise QualStop("model_changed", "the converted artifact changed before the load")
        net = load_int8_student(resolved, require_qnnpack=True)[0]
        if sha256_file(prov) != expected["provenance"]:
            raise QualStop("model_changed", "the provenance changed during validation")
        inputs["provenance"] = {"path": str(prov), "sha256": expected["provenance"]}
        inputs["converted"] = {"path": str(resolved["artifact_path"]), "sha256": expected["converted"]}
        return net, expected["converted"], engine, inputs
    ckpt = hash_first(args.checkpoint, expected["checkpoint"], f"{model} checkpoint")
    inputs["checkpoint"] = {"path": str(ckpt), "sha256": expected["checkpoint"]}
    if model == "teacher":
        from src.eval.model_loading import load_teacher_model
        from src.eval.stage_artifacts import validate_teacher_artifact
        cfg = REPO / TEACHER_CFG_REL
        if sha256_file(cfg) != TEACHER_CFG_SHA256:
            raise QualStop("teacher_config", "the teacher config is not the DL-88 file")
        inputs["teacher_config"] = {"path": TEACHER_CFG_REL, "sha256": TEACHER_CFG_SHA256}
        resolved = validate_teacher_artifact(str(ckpt), expected_sha256=expected["checkpoint"])
        if sha256_file(ckpt) != expected["checkpoint"]:
            raise QualStop("model_changed", "the teacher checkpoint changed during validation")
        net = load_teacher_model(resolved, builder=None, config_path=str(cfg),
                                 expected_sha256=expected["checkpoint"])[0]
        return net, expected["checkpoint"], None, inputs
    from src.eval.model_loading import load_student_checkpoint
    from src.eval.stage_artifacts import validate_fp32_artifact
    validate_fp32_artifact(MODELS[model]["stage"], str(ckpt))
    net, info = load_student_checkpoint(str(ckpt), map_location=DEVICE)
    if info.sha256 != expected["checkpoint"]:
        raise QualStop("model_changed", f"{model}: the checkpoint changed between the hash and the load")
    return net, expected["checkpoint"], None, inputs


# ------------------------------------------------------------------------------------- the teacher stream
class StreamSkip:
    """M4-V on a subset: before the forward of row m the private stream must stand at m draws. The module's
    own _build_bases is wrapped once; at the next real call it first makes the pending number of discarded
    calls with that call's own arguments (each one stream draw), then the real one."""

    def __init__(self, net):
        from src.distill.nmf_stream import isolated_nmf_modules
        mods = isolated_nmf_modules(net)
        if len(mods) != 1:
            raise QualStop("nmf_modules", f"the teacher holds {len(mods)} isolated NMF modules, not 1")
        self.module = mods[0]
        self.stream = self.module.nmf_stream
        if self.stream is None or (self.stream.policy, self.stream.seed) != M4V or self.stream.draws != 0:
            raise QualStop("nmf_stream", "the teacher's stream is not a fresh M4-V stream seeded 42")
        self.pending = 0
        self.skipped = 0
        original = self.module._build_bases

        def build_bases(B, S, D, R, device=None):
            if B != 1:
                raise QualStop("nmf_batch", "the stream is advanced only at batch size 1")
            for _ in range(self.pending):
                original(B, S, D, R, device=device)          # one discarded draw each
                self.skipped += 1
            self.pending = 0
            return original(B, S, D, R, device=device)

        self.module._build_bases = build_bases

    def before(self, manifest_index: int) -> None:
        gap = int(manifest_index) - self.stream.draws
        if gap < 0:
            raise QualStop("nmf_order", "the subset is not in ascending manifest order")
        self.pending = gap

    def after(self, manifest_index: int) -> None:
        if self.stream.draws != int(manifest_index) + 1 or self.pending != 0:
            raise QualStop("nmf_draws", "the stream does not stand at the row's draw count after its forward")


# --------------------------------------------------------------------------------------------- one pass
def predict(net, condition: str, split: str, indices: list, *, cell, cache_manifest, stream: StreamSkip | None):
    """The evaluator's pass over `indices` (ascending manifest order). Returns (result, taps)."""
    import torch
    from scripts.evaluate_corruptions import check_cache_sources
    from src.eval import Condition, evaluate_model
    from src.eval.adapters import build_eval_loader, build_expected_manifest_for, build_protocol_adapter
    from src.eval.corrupted import PlantSegCorruptedEvalDataset
    from src.eval.eval_runtime import (EVAL_NUM_WORKERS, ModelDeviceForward, check_post_eval,
                                       evaluate_capturing_warnings, resolve_model_device)
    indices = sorted(int(i) for i in indices)
    expected_manifest = build_expected_manifest_for(split, indices)
    if condition == CLEAN:
        dataset = build_protocol_adapter("canvas", split, indices)
        cond = Condition("clean")
    else:
        check_cache_sources(cache_manifest, expected_manifest, full=False)     # evaluate_corruptions.py:676
        dataset = PlantSegCorruptedEvalDataset(split, indices, cell)
        cond = Condition("corruption", cell.corruption, int(cell.severity))
    model_device = resolve_model_device(DEVICE, cpu_only=True)
    fwd = ModelDeviceForward(model_device)
    taps = []

    def tapped(loader):
        for batch in loader:
            if len(batch.manifest_indices) != 1:
                raise QualStop("batch_size", "the pass is made at batch size 1")
            taps.append({"manifest_index": int(batch.manifest_indices[0]), "image_id": batch.image_ids[0],
                         "image": batch.images[0].clone(), "target": batch.targets[0].clone()})
            yield batch

    def forward(model, images):
        m = taps[-1]["manifest_index"]
        if stream is not None:
            stream.before(m)
        logits = fwd(model, images)
        if stream is not None:
            stream.after(m)
        taps[-1]["pred"] = logits.argmax(1).cpu()[0].clone()       # evaluate.py:342's own argmax
        return logits

    loader = build_eval_loader(dataset, BATCH, num_workers=EVAL_NUM_WORKERS)
    result, _warn = evaluate_capturing_warnings(
        evaluate_model, net, tapped(loader), expected_manifest=expected_manifest, condition=cond,
        num_classes=116, background_index=0, ignore_index=255, forward=forward)
    check_post_eval(model=net, model_device=model_device, fwd=fwd, policy_applied=False)
    if [t["manifest_index"] for t in taps] != indices or len(result.rows) != len(indices):
        raise QualStop("pass_rows", "the pass did not score exactly the requested rows in order")
    if condition != CLEAN and not dataset.items_verified == dataset.source_digests_verified == len(indices):
        raise QualStop("cache_items", "the cached items verified do not equal the rows")
    if any("pred" not in t or t["pred"].dtype != torch.int64 for t in taps):
        raise QualStop("pred_missing", "a forward left no prediction")
    return result, taps


def result_counts(result, image_id: str) -> dict:
    """This pass's confusion counts of one image as {class: (tp, gt, pred)} (QUAL-A-1 Q6)."""
    pos = list(result.manifest_ids).index(image_id)
    sel = result.sparse_image_index == pos
    return counts_map(result.sparse_class_id[sel], result.sparse_tp[sel], result.sparse_gt[sel],
                      result.sparse_pred[sel], "this pass")


def counts_of_arrays(pred, mask, *, num_classes: int = 116, ignore: int = 255) -> dict:
    """QUAL-AUD-1 N-14: one image's confusion counts {class: (tp, gt, pred)} from a saved prediction and its saved
    mask, the ignore pixels (255) left out; every class with a nonzero count, as the evaluator's sparse rows."""
    import numpy as np
    valid = np.asarray(mask) != ignore
    g = np.asarray(mask)[valid].astype(np.int64)
    p = np.asarray(pred)[valid].astype(np.int64)
    if g.size and (g.min() < 0 or p.min() < 0 or g.max() >= num_classes or p.max() >= num_classes):
        raise QualStop("class_range", "a saved prediction or mask holds a class outside 0..115 (and 255)")
    gt = np.bincount(g, minlength=num_classes)
    pr = np.bincount(p, minlength=num_classes)
    tp = np.bincount(g[g == p], minlength=num_classes)
    nz = np.nonzero((tp != 0) | (gt != 0) | (pr != 0))[0]
    return {int(c): (int(tp[c]), int(gt[c]), int(pr[c])) for c in nz}


def tolerance(same_device: bool) -> float:
    """AM-27 item 4: 1e-6 on the record's device class, 0.5 pp otherwise."""
    return SAME_DEVICE_TOL if same_device else OTHER_DEVICE_TOL


def tile_verdict(recomputed, official, *, counts_equal: bool, same_device: bool) -> tuple:
    """AM-27 item 4 and QUAL-A-1 Q6 in one place (QUAL-AUD-1 T7 (1)): on the record's device class the gap must be
    within 1e-6 and the confusion counts equal class by class; on another device class within 0.5 pp. Both values
    must be finite floats. Returns (passed, gap or None, tolerance)."""
    import math
    tol = tolerance(same_device)
    ok = all(isinstance(v, float) and math.isfinite(v) for v in (recomputed, official))
    gap = abs(recomputed - official) if ok else None
    return bool(ok and gap <= tol and (counts_equal or not same_device)), gap, tol


# ------------------------------------------------------------------------------------- canvases, written
def png_bytes(arr) -> bytes:
    import io
    import numpy as np
    from PIL import Image
    a = np.ascontiguousarray(arr)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    data = buf.getvalue()
    with Image.open(io.BytesIO(data)) as im:                       # round trip, bit for bit
        if not np.array_equal(np.asarray(im), a):
            raise QualStop("png_roundtrip", "a PNG does not read back as its array")
    return data


def put_array(run: Path, rel: str, arr, outputs: dict) -> None:
    """Write an array once; a later step that computes the same array must find the same pixels."""
    import io
    import numpy as np
    from PIL import Image
    p = run / rel
    if p.exists():
        with Image.open(io.BytesIO(p.read_bytes())) as im:
            if not np.array_equal(np.asarray(im), np.ascontiguousarray(arr)):
                raise QualStop("shared_array_differs", f"{rel} differs from the array another step wrote")
        outputs[rel] = sha256_file(p)
        return
    outputs[rel] = write_new(p, png_bytes(arr))


_PAIRS = {}


def pairs_of(split: str):
    """The evaluator's own (image, mask) pairs of a split (src.data.dataset.PlantSegDataset), listed once."""
    if split not in _PAIRS:
        from src.data.dataset import PlantSegDataset
        base = PlantSegDataset(split)
        if len(base) != N_SPLIT[split]:
            raise QualStop("split_size", f"the {split.upper()} listing does not hold {N_SPLIT[split]} pairs")
        _PAIRS[split] = base.pairs
    return _PAIRS[split]


def display_canvases(split: str, tap: dict, cell) -> tuple:
    """The uint8 canvas and the mask behind one tapped sample, made by the evaluator's own functions and held
    bit for bit to the tensors the model saw (finalize(canvas) == the batch); and, for the manifest (QUAL-AUD-1
    F-06, AM-27 item 6), the whole-file sha256 of the image and of the mask, and under a corruption the cache
    entry's cached_file_sha256 (the file load_cached verified on its read). Returns (canvas, mask, hashes)."""
    import torch
    from PIL import Image
    from src.corruption_cache import cache_index
    from src.data.transforms import core_preprocess, finalize
    from src.eval.corrupted import corrupted_canvas
    img_path, mask_path = pairs_of(split)[tap["manifest_index"]]
    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        clean, mcanvas = core_preprocess(im, mk)
    canvas = clean if cell is None else corrupted_canvas(tap["image_id"], tap["manifest_index"], clean, cell)
    image, target = finalize(canvas, mcanvas)
    if not (torch.equal(image, tap["image"]) and torch.equal(target, tap["target"])):
        raise QualStop("canvas_differs", "a display canvas is not the input the model saw")
    hashes = {"image_file_sha256": sha256_file(img_path), "mask_file_sha256": sha256_file(mask_path)}
    if cell is not None:
        entry = cache_index(cell.manifest).entries.get((tap["image_id"], cell.corruption, int(cell.severity)))
        if entry is None:
            raise QualStop("cache_entry", "the cache holds no entry for an image of S")
        hashes["cached_file_sha256"] = entry["cached_file_sha256"]
    return canvas, mcanvas, hashes


def original_pair(split: str, manifest_index: int) -> tuple:
    """Figure 4.2's original image and mask as the evaluator reads them (EXIF on the image only, RGB; the mask's
    raw class indices), before any resize."""
    import numpy as np
    from PIL import Image, ImageOps
    img_path, mask_path = pairs_of(split)[manifest_index]
    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(mk)
    if mask.ndim != 2 or mask.max(initial=0) > 255:
        raise QualStop("original_mask", "the original mask is not a single-channel index image")
    return rgb, mask.astype(np.uint8)


def original_matches(rgb, mask, canvas, mcanvas) -> bool:
    """QUAL-AUD-1 N-09: Figure 4.2's "Image" and "Mask" panels are the pair the canvas was made from:
    core_preprocess of the original pair, as read, gives the saved canvas and mask bit for bit."""
    import numpy as np
    from PIL import Image
    from src.data.transforms import core_preprocess
    c2, m2 = core_preprocess(Image.fromarray(rgb), Image.fromarray(mask))
    return bool(np.array_equal(c2, canvas) and np.array_equal(m2, mcanvas))


# ---------------------------------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True, choices=tuple(MODELS))
    ap.add_argument("--split", required=True, choices=("test", "val"))
    ap.add_argument("--select-sha256", required=True, help="the sha256 of <run>/select/S.json that select printed")
    ap.add_argument("--checkpoint", help="teacher, E1, E3: the checkpoint of record")
    ap.add_argument("--provenance", help="E6: the converted artifact's run_meta (e15_run_meta.json)")
    ap.add_argument("--conditions", default=None,
                    help="comma list; default: every condition the figures need for this model")
    add_cache_args(ap)
    ap.add_argument("--export", help="TEST: reports/derived/tday_inventory.json")
    ap.add_argument("--export-sha256")
    ap.add_argument("--session-root", help="TEST: the session checkout holding reports/tday")
    ap.add_argument("--val-record", action="append", default=[],
                    help="VAL: CONDITION=DIR, a VAL per-image artifact of this model (Phase A only)")
    ap.add_argument("--val-indices", default=None,
                    help="VAL only: comma list of manifest indices for a stream check instead of S's positions")
    ap.add_argument("--tag", default=None, help="VAL stream check: the folder name under <run>/streamcheck")
    ap.add_argument("--expect-commit", help="the commit the run is made from (TEST: required, the audited commit)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_step(f"predict_{args.model}", args, _main)


def step_outputs(args) -> list:
    """The files and folders this step writes that no other step writes (QUAL-AUD-1 N-03)."""
    m = args.model
    if args.val_indices is not None:
        sub = f"streamcheck/{args.tag}"
        return [f"{sub}/checks/{m}.json", f"{sub}/records/predict_{m}.json"]
    return [f"values/{m}.json", f"checks/{m}.json", f"records/predict_{m}.json", f"arrays/pred/{m}"]


def _main(args) -> int:
    import platform
    import re
    import numpy as np
    import torch
    from scripts.evaluate_model import CliError, apply_threads
    run = run_dir(args.run)
    if args.tag is not None and (not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", args.tag) or args.tag in (".", "..")
                                 or "test" in args.tag.casefold()):
        raise QualStop("cli", "--tag is a plain folder name")
    if args.val_indices is not None and not args.tag:
        raise QualStop("streamcheck", "a VAL stream check needs --tag")
    outputs_absent(run, step_outputs(args))
    guard = version_guard()
    repo = repo_state(expected_commit(args))
    model = args.model
    try:
        threads = apply_threads(argparse.Namespace(threads=THREADS))
    except CliError as e:
        raise QualStop("threads", "torch.set_num_threads(1) did not hold", private=str(e)) from None
    if threads != THREADS or torch.get_num_threads() != THREADS:
        raise QualStop("threads", "torch threads are not 1")
    say("PASS", f"version guard and threads: CPython {guard['python']}, T {threads}, device {DEVICE}; checkout "
                f"{repo['commit'][:12]}... (governed paths clean: {repo['governed_paths_clean']})")
    streamcheck = args.val_indices is not None
    if args.split == "test":
        if args.conditions or streamcheck or args.val_record:
            raise QualStop("cli", "on TEST the conditions are the figures' own and the records come from the binding")
        conditions = CONDITIONS[model]
    else:
        conditions = tuple(args.conditions.split(",")) if args.conditions else CONDITIONS[model]
        allowed = (CLEAN, *CELLS) if streamcheck else CONDITIONS[model]
        if not conditions or not set(conditions) <= set(allowed):
            raise QualStop("conditions", f"{model}: conditions {list(conditions)} are not among {list(allowed)}")
    if model == "teacher" and len(conditions) != 1:
        raise QualStop("nmf_stream", "one teacher load serves one pass (M4-V): one condition per process")
    if any(c != CLEAN for c in conditions) and not cache_args_given(args):
        raise QualStop("cli", "a corrupted condition needs the cache and its three pins")
    s_doc, s_sha = read_s(run, args.select_sha256)
    if s_doc["split"] != args.split:
        raise QualStop("s_split", "S.json is of another split")
    by_pos = {p["position"]: p for p in s_doc["picks"]}

    expected = expected_hashes(model)
    net, model_sha, engine, inputs = load_model(model, args, expected)
    say("PASS", f"model {MODELS[model]['slot']}: sha256 {model_sha[:12]}... is the manifest's; loaded by the "
                f"evaluator's functions" + (f" (engine {engine})" if engine else ""))

    inv = None
    if args.split == "test":
        if not (args.export and args.export_sha256 and args.session_root):
            raise QualStop("cli", "TEST needs --export, --export-sha256 and --session-root")
        inv = open_test_inventory(args.export, args.export_sha256)
    val_records = dict(x.split("=", 1) for x in args.val_record)

    stream = StreamSkip(net) if model == "teacher" else None
    record = {"format": RECORD_FORMAT, "step": "predict", "model": model, "slot": MODELS[model]["slot"],
              "split": args.split, "guard": guard, "repo": repo, "select_sha256": s_sha, "streamcheck": args.tag,
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "pillow": __import__("PIL").__version__,
                           "threads": torch.get_num_threads(), "quantized_engine": engine},
              "inputs": dict(inputs), "outputs": {}, "conditions": {}}
    checks, values, outputs = [], {}, {}
    any_fail = False
    for condition in conditions:
        label = record_label(model, condition)
        if streamcheck:
            positions = None
            indices = sorted(int(i) for i in args.val_indices.split(","))
        else:
            positions = list(range(1, k_of(condition) + 1))
            indices = [by_pos[p]["manifest_index"] for p in positions]

        # ---- the record of this model and condition, before its pass (QUAL-AUD-1 F-02, F-03) ----
        if args.split == "test":
            off = load_test_official(inv, model, condition, args.session_root)
        elif condition in val_records:
            off = load_val_official(val_records[condition], f"VAL {label}")
        else:
            off = None
        cond_rec = {"rows": len(indices), "record": None}
        same, tol, binding = False, None, False
        if off is not None:
            check_record_identity(off, split=args.split, condition=condition, checkpoint_sha256=model_sha)
            bad = record_context_problems(off.summary, guard)
            if bad and args.split == "test":
                raise QualStop("record_context", f"{label}: the record's {bad} are not T {THREADS}, batch {BATCH}, "
                                                 f"model device {DEVICE} and this run's image; nothing of it was "
                                                 "predicted")
            same = same_device_class(off.summary)
            tol = tolerance(same)
            # TEST: every tile binds (AM-27 item 4), in the record's own context (F-03). VAL (Phase A): a record made
            # with other threads, another image or batch size is shown for information and does not bind the exit.
            binding = args.split == "test" or not bad
            er = off.summary["run"]["eval_runtime"]
            cond_rec["record"] = {"source": off.source, "files_sha256": off.files_sha256, "device": off.device,
                                  "model_device": off.model_device, "threads": er.get("torch_num_threads"),
                                  "batch_size": er.get("batch_size"), "image_digest": er.get("image_digest"),
                                  "repo_commit": off.summary["run"].get("repo_commit"), "context_problems": bad,
                                  "same_device_class": same, "tolerance": tol, "binding": binding}

        cell = cache_manifest = None
        if condition != CLEAN:
            cell, cache_manifest = open_cell(args, args.split, condition)
            record["inputs"]["cache"] = {"canonical_sha256": args.cache_canonical_sha256,
                                         "manifest_file_sha256": args.cache_manifest_file_sha256,
                                         "verification_record_sha256": args.verification_record_sha256}
        result, taps = predict(net, condition, args.split, indices, cell=cell, cache_manifest=cache_manifest,
                               stream=stream)
        if stream is not None:
            st = net.teacher.nmf_stream_state()
            if (st["policy"], st["seed"]) != M4V or st["draws"] != max(indices) + 1:
                raise QualStop("nmf_draws", "the teacher's stream does not end at the last row's draw count")
            record["nmf"] = {"policy": st["policy"], "seed": st["seed"], "draws": st["draws"],
                             "skipped": stream.skipped, "state_sha256": st["state_sha256"]}

        # ---- the check of AM-27 item 4 ----
        rows = {r.manifest_index: r for r in result.rows}
        if off is not None:
            for t in taps:
                m = t["manifest_index"]
                orow, mine = off.row(m), rows[m]
                if orow.get("image_id") != t["image_id"] or mine.image_id != t["image_id"]:
                    raise QualStop("record_row", "a record row is not the image predicted at its index")
                a, b = mine.disease_only_miou, orow.get("disease_only_miou")
                counts_equal = result_counts(result, t["image_id"]) == off.counts(t["image_id"])
                passed, gap, _ = tile_verdict(a, b, counts_equal=counts_equal, same_device=same)
                checks.append({"condition": condition, "manifest_index": m, "image_id": t["image_id"],
                               "recomputed": a, "official": b, "gap": gap, "tolerance": tol,
                               "same_device_class": same, "binding": binding, "pass": passed,
                               "counts_equal": counts_equal, "am5_excluded": orow.get("am5_excluded")})
                any_fail |= binding and not passed
        # ---- the arrays (not for a stream check) ----
        if positions is not None:
            pos_of = {by_pos[p]["manifest_index"]: p for p in positions}
            by_position = {}
            for t in taps:
                p = pos_of[t["manifest_index"]]
                pred = t["pred"].numpy()
                if pred.shape != (512, 512) or pred.min() < 0 or pred.max() > 115:
                    raise QualStop("pred_range", "a prediction is not a 512x512 map of classes 0..115")
                pred8 = pred.astype(np.uint8)
                put_array(run, f"arrays/pred/{model}/{condition}/{p}.png", pred8, outputs)
                canvas, mcanvas, hashes = display_canvases(args.split, t, cell)
                mask8 = mcanvas.astype(np.uint8)
                put_array(run, f"arrays/input/{condition}/{p}.png", canvas, outputs)
                put_array(run, f"arrays/mask/{p}.png", mask8, outputs)
                if counts_of_arrays(pred8, mask8) != result_counts(result, t["image_id"]):    # N-14
                    raise QualStop("saved_counts", "the saved prediction and mask do not give the pass's confusion "
                                                   "counts")
                by_position[str(p)] = hashes                                                  # F-06
                if condition == CLEAN and p == P50_POSITION:
                    rgb, mask = original_pair(args.split, t["manifest_index"])
                    if not original_matches(rgb, mask, canvas, mcanvas):                      # N-09
                        raise QualStop("original_differs", "core_preprocess of Figure 4.2's original pair is not "
                                                           "the saved canvas and mask")
                    put_array(run, f"arrays/original/{p}_image.png", rgb, outputs)
                    put_array(run, f"arrays/original/{p}_mask.png", mask, outputs)
                if off is not None:
                    values.setdefault(condition, {})[str(p)] = off.row(t["manifest_index"]).get("disease_only_miou")
            cond_rec["inputs_by_position"] = {k: by_position[k] for k in sorted(by_position, key=int)}
            for c in checks:
                if c["condition"] == condition and c["manifest_index"] in pos_of:
                    c["position"] = pos_of[c["manifest_index"]]
        record["conditions"][condition] = cond_rec
        n_c = [c for c in checks if c["condition"] == condition]
        if off is None:
            say("NOTE", f"{label}: no {args.split.upper()} record of this model; "
                        f"{len(indices)} predictions made, nothing compared")
        else:
            n_pass = sum(c["pass"] for c in n_c)
            largest = max((c["gap"] for c in n_c if c["gap"] is not None), default=None)
            n_eq = sum(c["counts_equal"] for c in n_c)
            r = cond_rec["record"]
            kind = ("PASS" if n_pass == len(n_c) else "STOP") if r["binding"] else "INFO"
            say(kind, f"check {label}: {n_pass} of {len(n_c)} tiles within {r['tolerance']:g} of the record "
                      f"(device {r['model_device']}, T {r['threads']}, batch {r['batch_size']}, image "
                      f"{str(r['image_digest'])[7:19]}){'' if r['binding'] else ' [other context: information only]'}; "
                      f"largest gap {largest!r}; confusion counts equal {n_eq} of {len(n_c)}")

    # ---- the step's files ----
    sub = f"streamcheck/{args.tag}/" if streamcheck else ""
    if not streamcheck:
        record["outputs"]["values"] = write_new(run / sub / "values" / f"{model}.json", canonical_bytes(values))
    record["outputs"]["checks"] = write_new(run / sub / "checks" / f"{model}.json", canonical_bytes(checks))
    record["outputs"]["arrays"] = outputs
    largest = max((c["gap"] for c in checks if c["gap"] is not None), default=None)
    record["summary"] = {"tiles_checked": len(checks), "tiles_passed": sum(c["pass"] for c in checks),
                         "largest_gap": largest, "counts_equal": sum(c["counts_equal"] for c in checks),
                         "all_pass": not any_fail}
    rec_sha = write_new(run / sub / "records" / f"predict_{model}.json", canonical_bytes(record))
    if any_fail:
        say("STOP", f"check_failed: {model}: a tile is outside its tolerance; nothing may be rendered "
                    f"(record sha256 {rec_sha})")
        return 1
    say("RESULT:", f"QUAL PREDICT {model} OK: {len(checks)} tiles checked, largest gap {largest!r}; "
                   f"{len(outputs)} arrays; record sha256 {rec_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
