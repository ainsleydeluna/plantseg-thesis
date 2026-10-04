#!/usr/bin/env python3
"""Teacher TRAIN scores: the R3 evaluation procedure on all 5,367 TRAIN images (AM-18 item 1(b), the D3
input; lane L-TEACHER-DIAG).

A diagnostic entry point beside scripts/evaluate_model.py, which is not changed: its --split gate (val,
test), the artifact writer's SPLITS and PlantSegDataset("train")'s augmentation are all left as they are.
Everything else of R3 is reused: the manifest is build_expected_manifest_for("train", range(5367)), the
evaluator's name-sorted listing (order_equals_sorted_stems is recorded, P19); each TRAIN pair goes
through the exact VAL branch (core_preprocess, finalize) in src/eval/teacher_diag.train_canvas_dataset;
the same collate and batch-size-1 loader, the evaluator-form teacher load (M4-V stream seeded 42, one
pass in name order), ModelDeviceForward on the CPU, evaluate_capturing_warnings(evaluate_model, ...,
forward=...), check_post_eval and the runtime record. New is only the writer: the core's result in the
evaluator's own layout (summary.json validated by artifacts.validate_summary, per_image.jsonl,
sufficient_stats.npz, MANIFEST.sha256) under schema_version "plantseg-teacher-train-diagnostic/1.0.0"
with split "train", so every strict VAL reader refuses it.

Outputs: the TRAIN artifact under --artifact-dir/<run id> and teacher_train_scores_<UTC>.json (the
all-class union-present mIoU, the per-class table for D3, the artifact's four file hashes). Record role
only (P3). A data root or TRAIN folder whose path contains "test" is refused by its string before any
filesystem call (P21). Exit codes: 0 written, 1 STOP, 2 refusal or usage, 4 unexpected exception. Real
mode (the CLI) refuses every run until K-part (P8); the stub teacher is reachable only from
scripts/smoke_score_teacher_train.py through run(args, model_factory=...).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval import teacher_diag as td  # noqa: E402

SCRIPT = "scripts/score_teacher_train.py"
SCHEMA = "plantseg-teacher-train-scores/1.0.0"
TRAIN_ARTIFACT_SCHEMA = "plantseg-teacher-train-diagnostic/1.0.0"
KIND = "teacher_train_scores"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Teacher TRAIN scores under the R3 evaluation procedure (D3 input).")
    td.add_common_flags(p)
    td.add_teacher_flags(p)
    p.add_argument("--strata", required=True, help="the AM-17 item 8 strata file (its TRAIN split list)")
    p.add_argument("--artifact-dir", required=True, help="parent of the TRAIN artifact; outside the repository")
    p.add_argument("--max-samples", type=int, default=None, help="stub mode only: evenly spaced TRAIN subset")
    return p


def guard_train_paths(root) -> None:
    """P21: the data root and the two TRAIN folders, each by its string first, then resolved."""
    td.refuse_test_path(root, "PLANTSEG_DATA_ROOT")
    for rel in ("images/train", "annotations/train"):
        td.refuse_test_path(Path(root) / rel, "TRAIN folder")


def train_provenance(req):
    """validate_artifact_request's provenance block, without its VAL/TEST split gate (which stays)."""
    from src.eval import artifacts as A
    if req.out_dir.exists():
        raise td.Refused(f"refusing to overwrite an existing artifact directory: {req.out_dir}")
    raw = A.git_porcelain_bytes(req.repo_root)
    paths = A.parse_porcelain_paths(raw)
    violations = tuple(sorted(p for p in paths if any(p.startswith(g) for g in A.GOVERNED_PREFIXES)))
    metric_sha, metric_path = A.hash_metric_impl()
    payload = A.build_config_payload(req)
    return A.Provenance(
        repo_commit=A.git_commit(req.repo_root), governed_paths_clean=not violations,
        governed_violations=violations, dirty_allowlisted=tuple(sorted(p for p in paths if p not in violations)),
        worktree_state_sha256=hashlib.sha256(raw).hexdigest(), metric_impl_sha256=metric_sha,
        metric_impl_path=metric_path, config_sha256=hashlib.sha256(A.canonical_json_bytes(payload)).hexdigest(),
        split_manifest_sha256=A.hash_split_manifest(req.expected_manifest),
        class_map_sha256=A.hash_class_map(req.class_map), config_payload=payload)


def write_train_artifact(result, req, prov, *, eval_runtime: dict, diagnostic: dict) -> Path:
    """The evaluator's layout (artifacts.write_artifact) under the TRAIN diagnostic schema; all or nothing."""
    import numpy as np

    from src.eval import artifacts as A
    if req.out_dir.exists():
        raise td.Refused(f"refusing to overwrite {req.out_dir}")
    A.validate_eval_runtime(eval_runtime)
    ts = datetime.now(timezone.utc).strftime(td.UTC_FORMAT)
    summary = A.build_summary(result, req, prov, ts, eval_runtime=eval_runtime)
    A.validate_summary(summary, req, result)           # the evaluator's content rules, before the rename
    summary["schema_version"] = TRAIN_ARTIFACT_SCHEMA
    summary["diagnostic"] = diagnostic
    tmp = req.out_dir.parent / f".{req.out_dir.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        (tmp / "summary.json").write_bytes(td.json_bytes(summary))
        (tmp / "per_image.jsonl").write_text(
            "".join(json.dumps(r.as_dict(), ensure_ascii=False, allow_nan=False) + "\n" for r in result.rows),
            encoding="utf-8", newline="\n")
        np.savez_compressed(
            tmp / "sufficient_stats.npz",
            image_index=result.sparse_image_index, class_id=result.sparse_class_id,
            tp=result.sparse_tp, gt=result.sparse_gt, pred=result.sparse_pred,
            dataset_tp=result.dataset_tp, dataset_gt=result.dataset_gt, dataset_pred=result.dataset_pred,
            manifest_ids=np.array(result.manifest_ids, dtype=np.str_))
        (tmp / A.MANIFEST_NAME).write_text(
            "".join(f"{td.file_sha256(tmp / n)}  {n}\n" for n in A.ARTIFACT_FILES), encoding="utf-8", newline="\n")
        tmp.rename(req.out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    A.verify_artifact(req.out_dir)
    return req.out_dir


def run(args, *, model_factory=None) -> int:
    return td.run_with_exit_codes(_run, args, model_factory=model_factory)


def _run(args, *, model_factory=None) -> int:
    stub = model_factory is not None
    real = not stub
    start, t0 = td.utc_now(), time.monotonic()
    td.check_common_flags(args, real=real)
    td.check_teacher_flags(args, roles=("record",))
    if args.max_samples is not None and (real or not 1 <= args.max_samples <= td.TRAIN_ROWS):
        raise td.Refused(f"--max-samples is a stub-mode flag in 1..{td.TRAIN_ROWS}")
    td.require_provenance_field_count(real)
    out_dir = td.require_outside_repo(args.out_dir, "--out-dir")
    art_root = td.require_outside_repo(args.artifact_dir, "--artifact-dir")
    code = td.require_commit_binding(args.script_commit, args.script_commit_dl_id) if real else td.code_provenance()
    repeat = td.require_single_output(out_dir, KIND, args)
    stamp = td.output_stamp(args, start)
    out_path = out_dir / f"{KIND}_{stamp}.json"
    if out_path.exists():
        raise td.Refused(f"{out_path.name} already exists")

    from configs.data import DATA
    root = DATA["root"]
    guard_train_paths(root)
    inputs = td.verify_teacher_inputs(args, stub=stub)
    m11 = td.check_m11(root)
    from scripts.build_train_strata import train_split_list
    stems = [s for s, _ in train_split_list(root)]
    strata = td.check_strata(args.strata, stems, stub=stub)

    from src.eval import Condition, DatasetMeta, RunMeta, evaluate_model
    from src.eval.adapters import (DATASET_DOI, DATASET_NAME, build_eval_loader, build_expected_manifest_for,
                                   deterministic_subset, list_split_stems, load_class_map)
    from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME, prepare_artifact_request
    from src.eval.eval_runtime import (EVAL_NUM_WORKERS, ModelDeviceForward, build_eval_runtime_record,
                                       check_post_eval, evaluate_capturing_warnings, resolve_model_device)
    from src.eval.protocols import CANVAS, PROTOCOL_IDS
    listing = list_split_stems("train")
    if listing != stems:
        raise td.Stop("the evaluator's TRAIN listing is not the strata TRAIN split list")
    order_sorted = listing == sorted(listing)          # P19, Q10: recorded, never gated
    source_indices = (deterministic_subset(td.TRAIN_ROWS, args.max_samples) if args.max_samples is not None
                      else list(range(td.TRAIN_ROWS)))
    expected_manifest = build_expected_manifest_for("train", source_indices)
    run_id = f"{KIND}_{stamp}"
    class_map = load_class_map()
    request = prepare_artifact_request(
        out_dir=art_root / run_id, artifact_status="smoke" if stub else "provisional", run_id=run_id,
        run=RunMeta(stage="teacher", model_role="teacher", precision="fp32", quant_backend=None,
                    checkpoint_path=str(inputs.ckpt), checkpoint_sha256=inputs.sha256, random_init=False,
                    device="cpu"),
        dataset=DatasetMeta(name=DATASET_NAME, doi=DATASET_DOI, split="train",
                            condition=Condition("clean", None, None), preprocess_protocol=PROTOCOL_IDS[CANVAS],
                            expected_rows=len(source_indices)),
        expected_manifest=expected_manifest, class_map=list(class_map.entries), repo_root=REPO)
    provenance = train_provenance(request)
    dataset = td.train_canvas_dataset(source_indices)

    loaded = td.load_teacher("evaluator", inputs, model_factory=model_factory)
    checks = td.after_load_checks(loaded, inputs, stub=stub)
    teacher = td.teacher_record(loaded, inputs, checks, stub=stub)
    model_device = resolve_model_device("cpu", cpu_only=True)
    fwd = ModelDeviceForward(model_device)
    watch = td.RngWatchForward(fwd, total=len(source_indices), label="train")
    loader = build_eval_loader(dataset, 1, num_workers=EVAL_NUM_WORKERS)
    result, warn_summary = evaluate_capturing_warnings(
        evaluate_model, loaded.eval_model, loader, expected_manifest=expected_manifest,
        condition=request.dataset.condition, num_classes=116, background_index=0, ignore_index=255,
        forward=watch)
    check_post_eval(model=loaded.eval_model, model_device=model_device, fwd=fwd, policy_applied=False)
    runtime = build_eval_runtime_record(
        model=loaded.eval_model, model_device=model_device, fwd=fwd, batch_size=1,
        forward_batches=result.forward_batches, policy_applied=False, policy_info=None, ckpt_info=None,
        warn_summary=warn_summary)
    n = len(source_indices)
    rng_state = watch.require_unchanged()
    if loaded.adapter.nmf_stream is not loaded.stream or loaded.stream.draws != n:
        raise td.Stop(f"M4-V stream: {loaded.stream.draws} draws for {n} images, or the stream object changed")
    if result.forward_batches != n or len(result.rows) != n:
        raise td.Stop(f"{result.forward_batches} forward batches and {len(result.rows)} rows for {n} images")

    diagnostic = {"lane": td.LANE, "script": SCRIPT, "split": "train", "order_equals_sorted_stems": order_sorted,
                  "manifest_rule": "build_expected_manifest_for('train', range(5367)): the evaluator's "
                                   "name-sorted listing", "preprocess": "core_preprocess then finalize (the VAL branch)",
                  "strict_val_readers": "refuse this artifact (schema_version and split)"}
    art_dir = write_train_artifact(result, request, provenance, eval_runtime=runtime, diagnostic=diagnostic)
    doc = td.base_document(SCRIPT, args, stub=stub, start_utc=start, code=code, extra={
        "schema": SCHEMA, "role": "record", "arm_id": None, "purpose": None, "repeat": repeat, "gates": "nothing (descriptive; D3 input)",
        "teacher": teacher,
        "inputs": {"strata": strata, "data_root": str(root), "m11": m11},
        "pass": {"split": "train", "rows": n, "protocol": PROTOCOL_IDS[CANVAS], "device": "cpu", "batch_size": 1,
                 "split_manifest_sha256": provenance.split_manifest_sha256, "order_equals_sorted_stems": order_sorted,
                 "eval_warnings": runtime["eval_warnings"]},
        "checks": {"rng_state_unchanged": watch.unchanged(), "rng_state_sha256": rng_state,
                   "nmf_stream": {"begin": loaded.stream_description, "end": loaded.stream.describe(),
                                  "draws": loaded.stream.draws, "expected_draws": n}},
        "dataset_level": dict(result.dataset_level),
        "per_class": dict(result.per_class),
        "artifact": {"dir": str(art_dir), "run_id": run_id, "schema_version": TRAIN_ARTIFACT_SCHEMA,
                     "files_sha256": {name: td.file_sha256(art_dir / name)
                                      for name in sorted([*ARTIFACT_FILES, MANIFEST_NAME])}},
        "status": "written",
    })
    td.finish_output(out_path, doc, start, t0)
    return td.EXIT_OK


def main(argv=None) -> int:
    return td.cli_main(build_parser(), run, argv)


if __name__ == "__main__":
    sys.exit(main())
