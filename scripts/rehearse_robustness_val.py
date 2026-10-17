#!/usr/bin/env python3
"""The VAL rehearsal statistics of the robustness contrast (L-CORR B-4; plan sections 4 and 6(xii);
R-G5).

  python -B scripts/rehearse_robustness_val.py --artifacts-root <dir> --e1 <file_id>
      --stand-in <file_id> --out <new dir outside the repo>

Inputs: the clean VAL artifact and the 15 inferential cells of E1 at seed 42 (`--e1`) and of the
stand-in for E6 (`--stand-in`: E1 at seed 44, lane 8's mapping), named as src/eval/corrupted.py
names them (`val_clean_<file_id>`, `val_corr_<file_id>_<cell id>`). Each is read by
src.stats.val_artifacts.load_val_artifact under Policy.REHEARSAL -- VAL only, 846 rows, a real-run
artifact (provisional or official; never smoke, never random-init), canvas only -- and must be an E1
student fp32 artifact (require_role) holding its directory's condition, one checkpoint per model.

The stand-in's runs are relabelled to stage E6 as src.stats.rehearsal.relabel does (the condition
kept), and the frozen functions run unmodified: assemble_miou_c(..., policy=REHEARSAL, stage=...,
clean=...) for each model, align_miou_c, and run_comparison("robustness_e1_e6", ...,
baseline_stage="E1", candidate_stage="E6"). Nothing in src/stats is edited.

R-G5, checked and recorded: for each model, n = 846 - K_val, the mIoU-C image set is the clean
artifact's included set, and every per-image mIoU-C equals the flat mean of its 15 cells' per-image
disease-only values within 1e-12.

Outputs: rehearsal_robustness.json (the stand-in mapping, each input's file hashes, K_val and n, the
R-G5 result, the serialised ComparisonResult, the code provenance) and SHA256SUMS, written
atomically to a new directory outside the repository. The outputs are nonofficial: a rehearsal of
the pipeline on VAL, never evidence on the robustness contrast.

Refusals, by code, before anything is written: `stand_in_same_as_e1` (the same file_id, or one
checkpoint for both); `rehearsal_input_refused` (an input the reader refuses, a role or condition
that is not the cell's, more than one checkpoint within a model, or the frozen modules' analysis
errors); the out-dir and writer codes of src/eval/robustness_report (write_dir_atomic).

Exit codes: 0 written and R-G5 holds; 1 written and R-G5 fails; 2 refused; 4 an unexpected error
(nothing written).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import src.eval.robustness_report as RR  # noqa: E402
from src import corruption_roles  # noqa: E402
from src.eval.corrupted import clean_dir_name, corrupted_dir_name  # noqa: E402

SCHEMA = "plantseg-robustness-rehearsal/1.0.0"
OUT_NAME, SUMS_NAME = "rehearsal_robustness.json", "SHA256SUMS"
TOLERANCE = 1e-12
CELLS = corruption_roles.INFERENTIAL_CELL_IDS
EXIT_DONE, EXIT_RG5_FAILED, EXIT_REFUSED, EXIT_UNEXPECTED = 0, 1, 2, 4
CODE_FILES = ("scripts/rehearse_robustness_val.py", "src/eval/corrupted.py",
              "src/eval/robustness_report.py", "src/stats/robustness.py", "src/stats/rehearsal.py",
              "src/stats/val_artifacts.py", "src/stats/ingest.py", "src/stats/align.py",
              "src/stats/artifact.py", "src/stats/tests.py", "src/stats/corruption_protocol.py",
              "configs/corruption_protocol.json")


class RehearsalRefusal(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _refuse(code: str, message: str):
    raise RehearsalRefusal(code, message)


def utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def check_pair(e1: str, stand_in: str) -> None:
    if e1 == stand_in:
        _refuse("stand_in_same_as_e1", f"--e1 and --stand-in are both {e1!r}; the rehearsal "
                                       "pairs two models")


def load_model(root, file_id: str) -> dict:
    """The clean artifact and the 15 inferential cells of one model, each checked."""
    from src.stats.val_artifacts import ValArtifactError, load_val_artifact, require_role
    arts = {}
    for label in ("clean", *CELLS):
        name = (clean_dir_name("val", file_id) if label == "clean"
                else corrupted_dir_name("val", file_id, label))
        try:
            art = load_val_artifact(Path(root) / name, label=name)
            require_role(art, stage="E1", model_role="student", precision="fp32", label=name)
        except ValArtifactError as e:
            _refuse("rehearsal_input_refused", str(e))
        check_condition(name, label, art)
        arts[label] = art
    check_one_checkpoint(file_id, arts)
    return arts


def check_condition(name: str, label: str, art) -> None:
    cond = art.summary["dataset"]["condition"]
    if cond != RR.expected_condition(label):
        _refuse("rehearsal_input_refused", f"{name} holds condition {cond}")


def check_one_checkpoint(file_id: str, arts: dict) -> None:
    shas = {a.summary["run"]["checkpoint_sha256"] for a in arts.values()}
    if len(shas) != 1:
        _refuse("rehearsal_input_refused", f"{file_id}: its artifacts name {len(shas)} checkpoints")


def analysis_errors() -> tuple:
    """The frozen modules' named analysis errors (src/stats/rehearsal.ANALYSIS_ERRORS): refusals."""
    from src.stats.rehearsal import ANALYSIS_ERRORS
    return ANALYSIS_ERRORS


def as_stand_in(run):
    """lane 8's mapping: the stand-in's identity relabelled to E6, its condition kept."""
    from src.stats.rehearsal import relabel
    i = run.identity
    return relabel(run, stage="E6", condition=(i.condition_type, i.condition_name,
                                               i.condition_severity))


def flat_means(cells: dict) -> dict:
    """Per clean image id (AM-5 rows already dropped by REHEARSAL), the flat mean of the per-image
    disease-only values of the 15 cells."""
    per: dict = {}
    for cid in CELLS:
        for rec in cells[cid].records:
            per.setdefault(rec.clean_image_id, []).append(float(rec.disease_only_miou))
    return {k: float(np.mean(np.array(v, dtype=np.float64))) for k, v in per.items()
            if len(v) == len(CELLS)}


def rg5(mc, cells: dict, clean) -> dict:
    from src.stats.ingest import EXPECTED_ROWS_VAL
    flat = flat_means(cells)
    k_val = int(clean.n_excluded_am5)
    ids = list(mc.clean_image_ids)
    diffs = [abs(float(v) - flat[cid]) for cid, v in zip(ids, mc.values) if cid in flat]
    max_diff = max(diffs) if diffs else None
    included = sorted(r.clean_image_id for r in clean.records)
    holds = (len(ids) == EXPECTED_ROWS_VAL - k_val == len(included) and ids == included
             and set(flat) == set(ids) and max_diff is not None and max_diff <= TOLERANCE)
    return {"k_val": k_val, "n": len(ids), "n_expected": EXPECTED_ROWS_VAL - k_val,
            "max_abs_difference": max_diff, "tolerance": TOLERANCE, "holds": bool(holds)}


def rehearse(root, e1_id: str, stand_in_id: str) -> dict:
    from src.stats.artifact import serialize_a3a
    from src.stats.corruption_protocol import official_corruption_grid
    from src.stats.ingest import Policy
    from src.stats.robustness import align_miou_c, assemble_miou_c
    from src.stats.tests import run_comparison
    check_pair(e1_id, stand_in_id)
    e1, si = load_model(root, e1_id), load_model(root, stand_in_id)
    if e1["clean"].summary["run"]["checkpoint_sha256"] == si["clean"].summary["run"]["checkpoint_sha256"]:
        _refuse("stand_in_same_as_e1", f"{e1_id} and {stand_in_id} name one checkpoint")
    grid = official_corruption_grid()
    policy = Policy.REHEARSAL
    try:
        e1_cells = {c: e1[c].run for c in CELLS}
        e6_cells = {c: as_stand_in(si[c].run) for c in CELLS}
        e6_clean = as_stand_in(si["clean"].run)
        mc_e1 = assemble_miou_c([e1_cells[c] for c in CELLS], grid, policy=policy, stage="E1",
                                clean=e1["clean"].run)
        mc_e6 = assemble_miou_c([e6_cells[c] for c in CELLS], grid, policy=policy, stage="E6",
                                clean=e6_clean)
        pv = align_miou_c(mc_e1, mc_e6, policy=policy)
        result = run_comparison("robustness_e1_e6", pv, baseline_stage="E1", candidate_stage="E6")
        comparison = serialize_a3a(result)
    except analysis_errors() as e:
        _refuse("rehearsal_input_refused", f"the frozen path refused the inputs: "
                                           f"{type(e).__name__}: {e}")
    checks = {"E1": rg5(mc_e1, e1_cells, e1["clean"].run),
              "E6": rg5(mc_e6, e6_cells, e6_clean)}
    from src.stats.val_artifacts import code_provenance
    return {
        "schema": SCHEMA, "status": "nonofficial", "mode": "rehearsal", "split": "val",
        "banner": ("NONOFFICIAL - VAL rehearsal of the robustness pipeline (stand-in mapping); "
                   "not evidence on the robustness contrast"),
        "e1": {"file_id": e1_id, "stage": "E1", "seed_role": "E1 at seed 42"},
        "stand_in": {"E6": {"file_id": stand_in_id, "scored_as": "E1", "relabelled_to": "E6",
                            "rule": "src.stats.rehearsal.relabel: stage only, condition kept",
                            "mapping": "lane 8: E1 at seed 44 stands in for E6"}},
        "inputs": {f"{tag}:{label}": {"dir": art.path.name, "run_id": art.run_id,
                                      "artifact_status": art.summary["run"]["artifact_status"],
                                      "checkpoint_sha256": art.summary["run"]["checkpoint_sha256"],
                                      "file_sha256": dict(art.file_sha256s)}
                   for tag, arts in (("E1", e1), ("E6", si)) for label, art in arts.items()},
        "r_g5": {**checks, "holds": all(c["holds"] for c in checks.values())},
        "miou_c_per_image_mean": {"E1": float(np.mean(mc_e1.values)),
                                  "E6": float(np.mean(mc_e6.values))},
        "comparison": comparison,
        "provenance": {"code": code_provenance(CODE_FILES), "generated_utc": utc_iso()},
    }


def write_outputs(out_dir, doc: dict) -> Path:
    """The JSON and its SHA256SUMS through B-4's atomic writer (re-read before the rename; a
    failure leaves nothing behind)."""
    payload = (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
               + "\n").encode("utf-8")
    return RR.write_dir_atomic(out_dir, {OUT_NAME: payload})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--artifacts-root", required=True)
    p.add_argument("--e1", required=True, help="file_id of E1 at seed 42")
    p.add_argument("--stand-in", required=True, help="file_id of the E6 stand-in (E1 at seed 44)")
    p.add_argument("--out", required=True, help="a new folder outside the repository")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_pair(args.e1, args.stand_in)
        RR.check_out_dir(args.out)
        doc = rehearse(args.artifacts_root, args.e1, args.stand_in)
        out = write_outputs(args.out, doc)
    except (RehearsalRefusal, RR.ReportRefusal) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:                                   # noqa: BLE001 -- reported, exit 4
        print(f"unexpected error, nothing written: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    print(f"rehearsal outputs: {out}")
    for stage in ("E1", "E6"):
        c = doc["r_g5"][stage]
        print(f"R-G5 {stage}: n {c['n']} (846 - K_val {c['k_val']}); max |nested - flat| "
              f"{c['max_abs_difference']!r}; holds {c['holds']}")
    print(doc["banner"])
    return EXIT_DONE if doc["r_g5"]["holds"] else EXIT_RG5_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
