#!/usr/bin/env python3
"""Score an AM-29 artifact (lane L-AM29, P4): dataset-level values only, through the evaluator's own code.

    python -B scripts/am29_score.py --split val|test [--confirm-test-split --threads T --out-json PATH] \\
        --precision int8|mixed [--protocol canvas|upstream] \\
        (--am29-eval-dir E [--arm A] | --l-am24-blockb-dir B --run-id R | --provenance P --stage S
         | --qat-eval-dir E --qat-selection-sha256 H)
        int8:  --expect-artifact-sha256 H            (the converted artifact's sha256, from AM-29's manifest)
        mixed: --islands G1|G2|G3|G4 --run-dir R --expect-checkpoint-sha256 C   (the selected epoch's eNN.pt)
        [--max-samples N (VAL only)] [--out-json PATH] [--host-label TEXT] [--allow-smoke-inputs]

--am29-eval-dir scores an AM-29 arm's finalized block B (scripts/am29_blockb.py); --l-am24-blockb-dir the block B of
AM-29's runs on lane/l-am24 (the run IDs of L_AM24_RUNS); --provenance one converted epoch (block B's VAL scoring,
VAL only, int8 only). int8 keeps the evaluator's INT8 provenance checks (src/eval/stage_artifacts.py
validate_int8_artifact, then src/eval/model_loading.py load_int8_student); mixed converts the same weights with a P2
group's FP32 islands and is labelled "mixed precision", never "int8". The TEST split needs --confirm-test-split,
--threads and --out-json and takes no sample cap. Nothing per image is printed or written: the output is the
dataset-level block.

--protocol upstream scores under the upstream PlantSeg protocol (AM-23 item 5); canvas is the default, and block B's.
--qat-eval-dir, mixed only, is E6 seed 42's block B of record (AM-29 item 6's look of E6 seed 42): its qat_selection.json
is hashed before it is parsed and must equal --qat-selection-sha256 (of record ce435dea..., DL-131), its
qat_finalize.json must be of record (b2c7705c..., DL-173), and epoch_ckpts/e15.pt is bound through the telemetry by
--expect-checkpoint-sha256 (of record 74fd438a..., AM-29 item 2(a)); --allow-smoke-inputs lifts these at VAL only.
At --split test the governed paths must be clean (src/eval/stage_artifacts.py governed_paths_error; recorded) and a
mixed-precision group must be eligible (AM-29 item 2(d)); a score whose thread count is not --threads is refused.

Exit codes: 0 scored, 2 refused, 4 error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_REFUSED, EXIT_ERROR = 0, 2, 4
# E6 seed 42's block B of record, the one --qat-eval-dir takes (AUDIT-LAM29-1 F-03): qat_selection.json (DL-131,
# DL-173), qat_finalize.json (DL-173) and the selected checkpoint epoch_ckpts/e15.pt (AM-29 item 2(a))
E6_S42_SELECTION_SHA256 = "ce435dea034073a45ae7353846f2c0a84c277e6126a838eb4d36d18de19358eb"
E6_S42_FINALIZE_SHA256 = "b2c7705c32b800f6fbbefff40d201cf828d2736b053236297e550189a7830c03"
E6_S42_CHECKPOINT_SHA256 = "74fd438a9e953ce84117a61454b879940b0a6a8f357e7705c22a86ab0ed847a5"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True)
    p.add_argument("--confirm-test-split", action="store_true")
    p.add_argument("--precision", required=True, choices=["int8", "mixed"])
    p.add_argument("--protocol", default="canvas", choices=["canvas", "upstream"])
    p.add_argument("--am29-eval-dir", default=None)
    p.add_argument("--arm", default=None)
    p.add_argument("--l-am24-blockb-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--provenance", default=None)
    p.add_argument("--stage", default=None)
    p.add_argument("--qat-eval-dir", default=None)
    p.add_argument("--qat-selection-sha256", default=None)
    p.add_argument("--expect-artifact-sha256", default=None)
    p.add_argument("--islands", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--expect-checkpoint-sha256", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--host-label", default=None)
    p.add_argument("--allow-smoke-inputs", action="store_true")
    return p


def cli_error(a) -> tuple[str, str] | None:
    """Every rule knowable from the arguments, before anything is read (the TEST rules first)."""
    split = str(a.split)
    if "test" in split.lower():
        if split != "test":
            return "split", f"--split {split!r}: val or test"
        if not a.confirm_test_split:
            return "test_unconfirmed", "--split test requires --confirm-test-split (the TEST look is Ice's kit's)"
        if a.max_samples is not None:
            return "test_capped", "--split test takes no --max-samples: the full TEST split or nothing"
        if a.threads is None:
            return "test_threads", "--split test requires --threads (the execution context of record)"
        if a.provenance is not None:
            return "test_provenance", "--provenance is block B's VAL scoring of one epoch; it never reads TEST"
        if a.out_json is None:                               # AUDIT-LAM29-1 F-08 (a)
            return "test_out_json", "--split test requires --out-json (the look's record, written once)"
    elif split != "val":
        return "split", f"--split {split!r}: val or test"
    sources = [k for k in ("am29_eval_dir", "l_am24_blockb_dir", "qat_eval_dir", "provenance")
               if getattr(a, k) is not None]
    if len(sources) != 1:
        return "source", "exactly one of --am29-eval-dir, --l-am24-blockb-dir, --qat-eval-dir, --provenance"
    if a.l_am24_blockb_dir is not None and a.run_id is None:
        return "source", "--l-am24-blockb-dir requires --run-id"
    if a.provenance is not None and (a.stage is None or a.precision != "int8"):
        return "source", "--provenance takes --stage and scores int8 only"
    if a.provenance is not None and a.protocol != "canvas":  # AUDIT-LAM29-1 F-01
        return "protocol", "--provenance is block B's VAL scoring, which scores canvas only"
    if (a.qat_eval_dir is None) != (a.qat_selection_sha256 is None):          # AUDIT-LAM29-1 F-03
        return "source", "--qat-eval-dir and --qat-selection-sha256 are given together"
    if a.qat_eval_dir is not None and a.precision != "mixed":
        return "qat_eval_dir_precision", ("--qat-eval-dir scores E6 seed 42's selected weights under a P2 group: "
                                          "--precision mixed only")
    if a.precision == "int8":
        if a.expect_artifact_sha256 is None:
            return "expect_artifact_sha256", "--precision int8 requires --expect-artifact-sha256 (AM-29's manifest)"
        if any(v is not None for v in (a.islands, a.expect_checkpoint_sha256)):
            return "precision", "--islands and --expect-checkpoint-sha256 belong to --precision mixed"
    else:
        if a.islands is None or a.expect_checkpoint_sha256 is None or a.run_dir is None:
            return "precision", "--precision mixed requires --islands, --run-dir and --expect-checkpoint-sha256"
        if a.expect_artifact_sha256 is not None:
            return "precision", "--expect-artifact-sha256 names the int8 artifact; a mixed score names the checkpoint"
    if a.qat_eval_dir is not None and values_of_record(a):  # AUDIT-LAM29-1 F-03: the values of record, by argument
        if a.qat_selection_sha256 != E6_S42_SELECTION_SHA256:
            return "qat_selection_not_of_record", (f"--qat-selection-sha256 {a.qat_selection_sha256}: E6 seed 42's "
                                                   f"qat_selection.json of record is {E6_S42_SELECTION_SHA256} (DL-131)")
        if a.expect_checkpoint_sha256 != E6_S42_CHECKPOINT_SHA256:
            return "checkpoint_not_of_record", (f"--expect-checkpoint-sha256 {a.expect_checkpoint_sha256}: E6 seed 42's "
                                                f"epoch_ckpts/e15.pt of record is {E6_S42_CHECKPOINT_SHA256} "
                                                "(AM-29 item 2(a))")
    if a.max_samples is not None and a.max_samples < 1:
        return "max_samples", "--max-samples must be >= 1"
    if a.threads is not None and a.threads < 1:
        return "threads", "--threads must be >= 1"
    return None


def values_of_record(a) -> bool:
    """--qat-eval-dir's values of record bind at --split test always, and at --split val unless --allow-smoke-inputs."""
    return a.split == "test" or not a.allow_smoke_inputs


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    err = cli_error(a)
    if err:
        print(f"RESULT: REFUSED [{err[0]}] -- {err[1]}. Nothing was read.")
        return EXIT_REFUSED
    try:
        import torch
        if a.threads is not None:
            torch.set_num_threads(int(a.threads))
            if torch.get_num_threads() != int(a.threads):
                raise RuntimeError(f"torch.set_num_threads({a.threads}) did not take")
        from src.eval import am29_score as SC
        from src.quant import am29_registry as R
        R.register()                                          # AM-29's stages, in this process only
        governed = None
        if a.split == "test":
            # AUDIT-LAM29-1 F-08 (c), its first option: the evaluator's governed-path gate (EVALUATION_CONTRACT 7.1),
            # before anything is read, and recorded. Not called at VAL: it runs git status over the whole repository.
            from src.eval.stage_artifacts import governed_paths_error
            governed = governed_paths_error(REPO)
            if governed is not None:
                SC._refuse("governed_paths_dirty", f"--split test: {governed}")
        SC.refuse_test_paths({"--am29-eval-dir": a.am29_eval_dir, "--l-am24-blockb-dir": a.l_am24_blockb_dir,
                              "--provenance": a.provenance, "--run-dir": a.run_dir, "--out-json": a.out_json,
                              "--qat-eval-dir": a.qat_eval_dir})
        if a.out_json is not None:
            out_p = Path(a.out_json).resolve()
            if out_p.exists():
                SC._refuse("out_exists", f"{a.out_json} exists; a score record is written once")
            if out_p == REPO or REPO in out_p.parents:
                SC._refuse("out_in_repo", f"--out-json {out_p} is inside the repository; records live outside it")
        if a.provenance is not None:
            bound = SC.int8_provenance(a.stage, a.provenance, expect_artifact_sha256=a.expect_artifact_sha256,
                                       allowed_stages=R.AM29_STAGES)
            bound.update({"source": "provenance", "stage": a.stage, "am29_label": R.STAGE_ARMS.get(a.stage)})
        elif a.precision == "mixed":
            bound = None
        elif a.am29_eval_dir is not None:
            bound = SC.bind_am29(a.am29_eval_dir, expect_artifact_sha256=a.expect_artifact_sha256, arm=a.arm)
        else:
            bound = SC.bind_l_am24(a.l_am24_blockb_dir, run_id=a.run_id,
                                   expect_artifact_sha256=a.expect_artifact_sha256)
        if a.precision == "int8":
            model = SC.load_int8_model(bound)
            label, variant = SC.LABEL_INT8, None
            source = {"stage": bound["stage"], "am29_label": bound.get("am29_label"),
                      "artifact": str(bound["artifact_path"]), "artifact_sha256": bound["artifact_sha256"],
                      "provenance_sha256": bound["provenance_sha256"], "identity": bound["identity"],
                      "selection_sha256": bound.get("selection_sha256"), "finalize_sha256": bound.get("finalize_sha256")}
        else:
            sel_bound = _selection_only(a, SC)
            st = SC.selected_state(sel_bound, run_dir=a.run_dir, expect_checkpoint_sha256=a.expect_checkpoint_sha256)
            model, variant = SC.mixed_model(st["state"], a.islands)
            if a.split == "test" and variant.get("eligible") is not True:          # AUDIT-LAM29-1 F-08 (d)
                SC._refuse("group_not_eligible", f"{a.islands} keeps INT8 parameter share "
                                                 f"{(variant.get('shares') or {}).get('int8_param_share')!r}: a variant "
                                                 "with less than 90% of its parameters in INT8 is not eligible (AM-29 "
                                                 "item 2(d)) and is not scored on TEST")
            label = SC.LABEL_MIXED
            source = {"stage": sel_bound["stage"], "am29_label": sel_bound["am29_label"], "checkpoint": st["checkpoint"],
                      "checkpoint_sha256": st["checkpoint_sha256"], "epoch": st["epoch"], "run_id": st["run_id"],
                      "selection_sha256": sel_bound["selection_sha256"], "islands": a.islands}
            if a.qat_eval_dir is not None:                    # AUDIT-LAM29-1 F-03
                source["finalize_sha256"] = sel_bound["finalize_sha256"]
        t0 = time.time()
        score = SC.score_model(model, split=a.split, max_samples=a.max_samples, protocol=a.protocol)
        if a.threads is not None and score["torch_num_threads"] != int(a.threads):   # AUDIT-LAM29-1 F-08 (b)
            SC._refuse("threads_mismatch", f"the score ran on {score['torch_num_threads']} threads, not --threads "
                                           f"{a.threads}; nothing is printed or written")
        from src.quant.am29 import code_identity
        from src.quant.qat import cpu_model
        ident = code_identity()
        doc = {"format": SC.FORMAT, "label": label, "precision": a.precision, "source_kind": bound["source"]
               if a.precision == "int8" else ("am29" if a.am29_eval_dir else "qat" if a.qat_eval_dir else "l-am24"),
               "source": source,
               "variant": variant, "score": score, "seconds": time.time() - t0, "git_head": ident["git_head"],
               "code_clean_at_head": ident["code_clean_at_head"], "host_label": a.host_label, "cpu_model": cpu_model(),
               "smoke_inputs": bool(a.allow_smoke_inputs), "dataset_level_only": True}
        if a.split == "test":                                 # AUDIT-LAM29-1 F-08 (c): None, the gate passed
            doc["governed_paths_error"] = governed
        for line in SC.dataset_level_lines(label, source.get("am29_label") or source["stage"], score):
            print(line)
        print("AM29_SCORE_JSON " + json.dumps({"label": label, "name": source.get("am29_label"), "stage": source["stage"],
                                               "split": score["split"], "protocol": score["protocol"],
                                               "rows": score["actual_rows"],
                                               "dataset_level": score["dataset_level"]}, sort_keys=True))
        if a.out_json is not None:
            from src.quant.ptq import write_exclusive
            write_exclusive({Path(a.out_json): (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
                             .encode("utf-8")})
        print(f"RESULT: SCORED ({label}, {source.get('am29_label') or source['stage']}, {score['split']}, "
              f"all-class mIoU {score['dataset_level'].get('all_class_miou')!r})")
        return EXIT_OK
    except Exception as e:                                   # noqa: BLE001 -- every refusal and error is named
        code = getattr(e, "code", None)
        if code is not None:
            print(f"RESULT: REFUSED [{code}] -- {e}")
            return EXIT_REFUSED
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}")
        return EXIT_ERROR


def _selection_only(a, SC) -> dict:
    """The selection a mixed score binds the checkpoint through (no int8 artifact is loaded)."""
    from src.quant import am29_registry as R
    if a.qat_eval_dir is not None:
        return _e6_s42_selection(a, SC)
    if a.am29_eval_dir is not None:
        E = Path(a.am29_eval_dir)
        sel = SC.read_json(E / SC.AM29_SELECTION, "am29_selection_unreadable")
        fin = SC.read_json(E / SC.AM29_FINALIZE, "am29_finalize_unreadable")
        if sel.get("format") != SC.AM29_SELECTION_FORMAT or fin.get("status") != "finalized" \
                or (fin.get("am29_selection") or {}).get("sha256") != SC.sha256_file(E / SC.AM29_SELECTION):
            SC._refuse("am29_selection_format", f"{E} holds no finalized AM-29 selection")
        if sel.get("stage") not in R.AM29_STAGES or (a.arm is not None and R.ARM_STAGES.get(a.arm) != sel.get("stage")):
            SC._refuse("artifact_unknown", f"the selection's stage {sel.get('stage')!r} is not the AM-29 arm asked")
        return {"selection": sel, "stage": sel["stage"], "am29_label": R.STAGE_ARMS[sel["stage"]],
                "selection_sha256": SC.sha256_file(E / SC.AM29_SELECTION)}
    B = Path(a.l_am24_blockb_dir)
    if a.run_id not in SC.L_AM24_RUNS:
        SC._refuse("artifact_unknown", f"--run-id {a.run_id!r} is not one of {sorted(SC.L_AM24_RUNS)}")
    sel_p = B / "eval" / "qat_selection.json"
    sel = SC.read_json(sel_p, "blockb_selection_unreadable")
    xb = SC.read_json(B / "x_blockb.json", "blockb_unreadable")
    fin = SC.read_json(B / "eval" / "qat_finalize.json", "blockb_finalize_unreadable")
    if sel.get("format") != SC.QAT_SELECTION_FORMAT or sel.get("run_id") != a.run_id or xb.get("selection") != sel:
        SC._refuse("blockb_selection", f"{sel_p} is not run {a.run_id}'s selection of this block B")
    if fin.get("status") != "finalized" or fin.get("qat_selection") != {"path": "qat_selection.json",
                                                                        "sha256": SC.sha256_file(sel_p)}:
        SC._refuse("blockb_finalize", f"{B / 'eval' / 'qat_finalize.json'} is not the finalize record of {sel_p}")
    if sel.get("stage") not in SC.L_AM24_STAGES:
        SC._refuse("artifact_unknown", f"{sel_p} names stage {sel.get('stage')!r}, not one of {list(SC.L_AM24_STAGES)}")
    return {"selection": sel, "stage": sel.get("stage"), "am29_label": SC.L_AM24_RUNS[a.run_id],
            "selection_sha256": SC.sha256_file(sel_p)}


def _read_hashed(path: Path, code: str, SC) -> tuple[bytes, str]:
    """The file's bytes, read once, and their sha256 (the bytes parsed later are these, never a second read)."""
    try:
        raw = path.read_bytes()
    except OSError as e:
        SC._refuse(code, f"{path}: {e}")
    return raw, hashlib.sha256(raw).hexdigest()


def _parse_hashed(raw: bytes, path: Path, code: str, SC) -> dict:
    """Bytes already hashed, parsed as strict JSON (no NaN or Infinity), an object."""
    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=SC._no_constant)
    except ValueError as e:
        SC._refuse(code, f"{path}: {e}")
    if not isinstance(doc, dict):
        SC._refuse(code, f"{path} is not a JSON object")
    return doc


def _e6_s42_selection(a, SC) -> dict:
    """AUDIT-LAM29-1 F-03: E6 seed 42's block B of record, the pinned scripts/qat_epoch_eval.py's eval dir (DL-131).
    qat_selection.json is hashed before it is parsed and must equal --qat-selection-sha256; qat_finalize.json must be of
    record (DL-173) and name that selection; the selection is stage E6, seed 42. selected_state then binds the winner's
    checkpoint through the run's telemetry and --expect-checkpoint-sha256 (cli_error holds the values of record)."""
    E = Path(a.qat_eval_dir)
    sel_p, fin_p = E / "qat_selection.json", E / "qat_finalize.json"
    sel_raw, sel_sha = _read_hashed(sel_p, "qat_selection_unreadable", SC)
    if sel_sha != a.qat_selection_sha256:
        SC._refuse("qat_selection_sha256_mismatch", f"{sel_p} has sha256 {sel_sha}; --qat-selection-sha256 is "
                                                    f"{a.qat_selection_sha256} (checked before the file is parsed)")
    sel = _parse_hashed(sel_raw, sel_p, "qat_selection_unreadable", SC)
    fin_raw, fin_sha = _read_hashed(fin_p, "qat_finalize_unreadable", SC)
    if values_of_record(a) and fin_sha != E6_S42_FINALIZE_SHA256:
        SC._refuse("qat_finalize_not_of_record", f"{fin_p} has sha256 {fin_sha}; E6 seed 42's qat_finalize.json of "
                                                 f"record is {E6_S42_FINALIZE_SHA256} (DL-173)")
    fin = _parse_hashed(fin_raw, fin_p, "qat_finalize_unreadable", SC)
    if sel.get("format") != SC.QAT_SELECTION_FORMAT or fin.get("format") != SC.QAT_FINALIZE_FORMAT \
            or fin.get("status") != "finalized" \
            or fin.get("qat_selection") != {"path": "qat_selection.json", "sha256": sel_sha}:
        SC._refuse("qat_selection_format", f"{E} holds no {SC.QAT_SELECTION_FORMAT} selection finalized by its "
                                           f"{SC.QAT_FINALIZE_FORMAT} record")
    if sel.get("stage") != "E6" or sel.get("seed") != 42 or fin.get("stage") != "E6" or fin.get("seed") != 42:
        SC._refuse("qat_stage", f"{sel_p} is stage {sel.get('stage')!r}, seed {sel.get('seed')!r}; --qat-eval-dir takes "
                                "E6 seed 42 only")
    return {"selection": sel, "stage": "E6", "am29_label": "E6-s42", "selection_sha256": sel_sha,
            "finalize_sha256": fin_sha}


if __name__ == "__main__":
    raise SystemExit(main())
