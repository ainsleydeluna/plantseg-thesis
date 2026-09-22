#!/usr/bin/env python3
"""R3 — teacher readiness floor on VAL (M5-R3 under M4-V; B60 §2.4, B61 §6, EVALUATION_CONTRACT §7.2-§7.3).

After the teacher fine-tune, the M12-selected checkpoint is re-evaluated ONCE with the repository's
thesis evaluator (`scripts/evaluate_model.py`, core_preprocess canvas, union-present dataset-level
mIoU), under M4-V: batch size 1, the frozen VAL manifest order, the teacher's private NMF stream
seeded 42 at the start of the pass.

    PASS  iff  teacher VAL all-class mIoU  >  0.36314016580581665  (E1 run of record; no margin)

What R3 is NOT: the same VAL partition took part in checkpoint selection, so R3 is an operational
competence floor — not an independent estimate, not an inferential comparison, not a thesis result,
and never TEST. Its evaluator artifact is therefore written with artifact_status = "provisional".

On FAIL: STOP and escalate as a methodology question. No automatic retraining, no hyperparameter
search, no band relaxation, no TEST inspection. The exit code is 3.

The selected checkpoint's SHA-256 must equal the one in the run's `teacher_selection.json`
(status "final"); anything else is refused before the evaluator is touched.

M4-V makes an image's NMF basis depend on its POSITION in the pass, so R3 must score the same ordered
VAL manifest that checkpoint selection scored. ONE identity is used throughout: the evaluator's own
`src.eval.artifacts.hash_split_manifest` over the ordered (manifest_index, image_id, clean_image_id)
rows. The selection hook writes it as `val_manifest_sha256`; the evaluator writes the same function
of the same rows as `dataset.split_manifest_sha256`. So:
  * BEFORE evaluation, the ordered VAL manifest the evaluator is about to process (resolved with the
    evaluator's own functions) must hash to the selection's `val_manifest_sha256`
    ([val_manifest_missing] / [val_manifest_mismatch] otherwise; nothing is evaluated);
  * AFTER evaluation, the artifact's `dataset.split_manifest_sha256` must be present and equal to it
    ([artifact_manifest_missing] / [artifact_manifest_mismatch] otherwise).

    python scripts/teacher_readiness_r3.py --checkpoint <work_dir>/iter_<N>.pth \
        --teacher-config configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py \
        --selection <work_dir>/teacher_selection.json --out-dir <outside repo>/r3_artifact \
        --readiness-json <outside repo>/teacher_r3_readiness.json --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

E1_VAL_ALL_CLASS_MIOU = 0.36314016580581665      # E1 seed-42 run of record (B52), the R3 comparator
ARTIFACT_STATUS = "provisional"                  # same-VAL readiness record, never an official artifact
EXIT_PASS, EXIT_REFUSED, EXIT_FAIL = 0, 2, 3


class R3Refused(RuntimeError):
    """R3 could not be run as specified (identity, selection, manifest or path problem)."""

    def __init__(self, message: str, code: str = "r3_refused"):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_selection(checkpoint: Path, selection_path: Path) -> dict:
    """The checkpoint must be the M12-selected one: final selection record, identical SHA-256."""
    if not checkpoint.is_file():
        raise R3Refused(f"checkpoint not found: {checkpoint}")
    if not selection_path.is_file():
        raise R3Refused(f"selection record not found: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "final":
        raise R3Refused(f"selection record status is {selection.get('status')!r}, not 'final'")
    sha = _sha256(checkpoint)
    if sha != selection.get("checkpoint_sha256"):
        raise R3Refused(f"checkpoint sha256 {sha} is not the M12-selected checkpoint "
                        f"{selection.get('checkpoint_sha256')} (iteration {selection.get('selected_iteration')})")
    return {**selection, "verified_checkpoint_sha256": sha}


def evaluator_val_manifest() -> list:
    """The ordered VAL manifest `scripts/evaluate_model.py` evaluates for R3, resolved with the evaluator's
    own functions from a filename listing (no image opened): R3 passes no --max-samples, so the
    evaluator takes every VAL row, `source_indices = range(EXPECTED_SPLIT_ROWS["val"])`, name-sorted."""
    from scripts.evaluate_model import EXPECTED_SPLIT_ROWS
    from src.eval.adapters import build_expected_manifest_for
    return build_expected_manifest_for("val", list(range(EXPECTED_SPLIT_ROWS["val"])))


def verify_val_manifest(selection: dict, rows) -> str:
    """M4-V identity gate, BEFORE any evaluation: `rows` (the ordered manifest about to be evaluated) must
    hash, under the one canonical hasher, to the selection's `val_manifest_sha256`. Order counts."""
    from src.eval.artifacts import hash_split_manifest
    want = selection.get("val_manifest_sha256")
    if not isinstance(want, str) or not want:
        raise R3Refused("teacher_selection.json carries no val_manifest_sha256, so the VAL manifest and "
                        "order scored during checkpoint selection cannot be proven", "val_manifest_missing")
    got = hash_split_manifest(rows)
    if got != want:
        raise R3Refused(f"the ordered VAL manifest to be evaluated ({len(rows)} rows, {got}) is not the one "
                        f"scored during checkpoint selection ({want}); under M4-V a different order or "
                        "membership changes every image's NMF draw", "val_manifest_mismatch")
    return got


def decide(all_class_miou: float) -> dict:
    """Strictly greater than the E1 comparator, with no margin."""
    passed = all_class_miou > E1_VAL_ALL_CLASS_MIOU
    return {"readiness_verdict": "PASS" if passed else "FAIL",
            "comparator_e1_val_all_class_miou": E1_VAL_ALL_CLASS_MIOU,
            "teacher_val_all_class_miou": all_class_miou,
            "gap_teacher_minus_e1": all_class_miou - E1_VAL_ALL_CLASS_MIOU,
            "rule": "teacher VAL all-class mIoU > comparator; no margin"}


def evaluator_args(args) -> argparse.Namespace:
    from scripts.evaluate_model import build_parser
    argv = ["--stage", "teacher", "--model-role", "teacher", "--split", "val",
            "--checkpoint", str(args.checkpoint), "--teacher-config", str(args.teacher_config),
            "--batch-size", "1", "--artifact-status", ARTIFACT_STATUS,
            "--out-dir", str(args.out_dir), "--device", args.device]
    if args.run_id:
        argv += ["--run-id", args.run_id]
    return build_parser().parse_args(argv)


def run_r3(args, *, evaluate=None, teacher_builder=None, resolve_manifest=None) -> tuple[int, dict]:
    selection = verify_selection(Path(args.checkpoint), Path(args.selection))
    rows = (resolve_manifest or evaluator_val_manifest)()
    manifest_sha = verify_val_manifest(selection, rows)          # refused here -> nothing is evaluated
    if evaluate is None:
        from scripts.evaluate_model import run as evaluate
    ev_args = evaluator_args(args)
    artifact_dir = Path(evaluate(ev_args, teacher_builder=teacher_builder))
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    level = summary["dataset_level"]
    if summary["dataset"]["split"] != "val":
        raise R3Refused(f"R3 is VAL-only; the artifact split is {summary['dataset']['split']!r}")
    run_meta = summary["run"]
    if run_meta["artifact_status"] != ARTIFACT_STATUS or run_meta["model_role"] != "teacher":
        raise R3Refused(f"unexpected evaluator artifact: status {run_meta['artifact_status']!r}, "
                        f"role {run_meta['model_role']!r}")
    if run_meta["checkpoint_sha256"] != selection["verified_checkpoint_sha256"]:
        raise R3Refused("the evaluator scored a different checkpoint than the one verified")
    # The frozen evaluator schema writes the split manifest under "dataset" (src/eval/artifacts.py).
    artifact_manifest = summary["dataset"].get("split_manifest_sha256")
    if not isinstance(artifact_manifest, str) or not artifact_manifest:
        raise R3Refused("the evaluator artifact carries no dataset.split_manifest_sha256",
                        "artifact_manifest_missing")
    if artifact_manifest != manifest_sha:
        raise R3Refused(f"the evaluator artifact scored manifest {artifact_manifest}, not the selection's "
                        f"{manifest_sha}", "artifact_manifest_mismatch")
    decision = decide(float(level["all_class_miou"]))
    record = {
        "role": "same_val_operational_floor",
        "not": ["independent estimate", "inferential comparison", "thesis result", "TEST evaluation"],
        **decision,
        "teacher_val_disease_only_miou": level.get("disease_only_miou"),
        "artifact_dir": str(artifact_dir),
        "artifact_status": run_meta["artifact_status"],
        "split_manifest_sha256": artifact_manifest,
        "val_manifest": {"selection_val_manifest_sha256": selection["val_manifest_sha256"],
                         "pre_evaluation_sha256": manifest_sha,
                         "artifact_split_manifest_sha256": artifact_manifest, "rows": len(rows),
                         "identity": "src.eval.artifacts.hash_split_manifest over the ordered "
                                     "(manifest_index, image_id, clean_image_id) rows"},
        "repo_commit": run_meta.get("repo_commit"),
        "governed_paths_clean": run_meta.get("governed_paths_clean"),
        "m4v": {"policy": "M4-V", "seed": 42, "batch_size": 1,
                "order": "frozen VAL manifest (name-sorted), one complete pass"},
        "selection": {k: selection.get(k) for k in ("selected_iteration", "selected_mIoU_full",
                                                    "checkpoint_sha256", "verified_checkpoint_sha256",
                                                    "val_manifest_sha256")},
        "on_fail": "STOP and escalate; no retraining, no search, no band relaxation, no TEST",
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = Path(args.readiness_json)
    if out.exists():
        raise R3Refused(f"refusing to overwrite an existing readiness record: {out}")
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return (EXIT_PASS if decision["readiness_verdict"] == "PASS" else EXIT_FAIL), record


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R3 teacher readiness floor on VAL (M4-V, batch 1).")
    p.add_argument("--checkpoint", required=True, help="the M12-selected teacher checkpoint")
    p.add_argument("--teacher-config", required=True, help="the thesis teacher config")
    p.add_argument("--selection", required=True, help="teacher_selection.json (status final)")
    p.add_argument("--out-dir", required=True, help="evaluator artifact directory (must not exist)")
    p.add_argument("--readiness-json", required=True, help="where to write teacher_r3_readiness.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-id", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        code, record = run_r3(args)
    except R3Refused as e:
        print(f"R3 REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(record, indent=2, sort_keys=True))
    if code == EXIT_FAIL:
        print("\nR3 FAIL — STOP. Escalate as a methodology question. No retraining, no search, no "
              "band relaxation, no TEST.", file=sys.stderr)
    else:
        print("\nR3 PASS — same-VAL operational floor cleared (not an independent estimate).")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
