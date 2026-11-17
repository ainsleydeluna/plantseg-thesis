"""AM-29's scoring path (lane L-AM29, P2-P4): the evaluator's own loading, preprocessing and metric code, unchanged.

scripts/evaluate_model.py cannot take an AM-29 stage: its --stage choices and the artifact schema's stages are
src/eval/artifacts.py STAGES, a file outside the stage-map exception of hard limit 3. So this module calls, unchanged,
the evaluator's pieces that evaluate_model.run() calls between its source validation and its write_artifact:

  source       src.eval.stage_artifacts.validate_int8_artifact (stage, source stage, method, random_init, classes,
               backend, format, the recorded sha256 against the file) -- AM-29's stages registered in this process
               only (src/quant/am29_registry.py); then src.eval.model_loading.load_int8_student, whose TorchScript load
               checks the embedded plantseg_int8.json identity (schema, stage, quantization, classes, engine, role)
  data         src.eval.adapters: load_class_map, deterministic_subset, build_expected_manifest_for,
               build_protocol_adapter (canvas: core_preprocess/1.0.0), build_eval_loader (batch 1, num_workers 0)
  metrics      src.eval.evaluate.evaluate_model through src.eval.eval_runtime.evaluate_capturing_warnings, then
               check_post_eval and build_eval_runtime_record

and returns DATASET-LEVEL values only (result.dataset_level: all-class mIoU, its eligible count, macro Dice, mAcc,
disease-only mIoU, aAcc), never a per-image row. Three sources:

  am29       an AM-29 arm's finalized block B (scripts/am29_blockb.py): am29_finalize.json -> am29_selection.json -> the
             selected epoch's converted all-INT8 TorchScript and its provenance (scored "int8")
  l-am24     lane/l-am24's block B (scripts/explore_qat_blockb.py) of a run AM-29 scores there (L_AM24_RUNS, by run
             ID), finalized through the pinned scripts/qat_epoch_eval.py: x_blockb.json ->
             eval/qat_selection.json and eval/qat_finalize.json -> the artifact of record (scored "int8")
  provenance one converted epoch's eNN_run_meta.json (block B's VAL scoring of every epoch; VAL only)

and, for am29 and l-am24, the same weights (the selected epoch's eNN.pt, bound through the selection) converted with an
adopted P2 group (src/quant/am29_islands.py), scored "mixed precision", never "int8". Every artifact is bound by the
sha256 AM-29's manifest records (--expect-artifact-sha256, --expect-checkpoint-sha256). An unknown, mixed-precision or
hash-mismatched artifact is refused as int8.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FORMAT = "am29_score/1"
LABEL_INT8 = "int8"
LABEL_MIXED = "mixed precision"
INT8_META = "plantseg_int8.json"            # = src.eval.model_loading.INT8_TORCHSCRIPT_META
MIXED_META = "x_mixed_precision.json"       # lane/l-am24's mixed-precision identity (explore_qat_blockb.py:144)
# the runs on lane/l-am24 that AM-29 scores: run ID -> AM-29's label (the run ID is the trainer's run directory, the
# run_id of every record and of the artifact's identity)
L_AM24_RUNS = {
    "x-n2-s42": "N2",
    "x-n2r2-s42": "E6-R2-M",
    "x-d-n3-s42": "D",
}
# the pinned QAT stages whose recipe is E6's (source E2 or E3) that such a run may record; any other is unknown here
L_AM24_STAGES = ("E6", "E6-E2", "E6-R1", "E6-R2")
L_AM24_BLOCKB_FORMAT = "x_qat_blockb/1"
QAT_SELECTION_FORMAT = "qat_selection/1"
QAT_FINALIZE_FORMAT = "qat_finalize/1"
AM29_SELECTION = "am29_selection.json"
AM29_FINALIZE = "am29_finalize.json"
AM29_SELECTION_FORMAT = "am29_selection/1"
AM29_FINALIZE_FORMAT = "am29_finalize/1"


class ScoreRefused(RuntimeError):
    """A named refusal before any dataset or model exists (exit 2)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _refuse(code: str, message: str):
    raise ScoreRefused(code, message)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def names_test(value) -> bool:
    """SL-1: a path holding "test", as given or resolved, is refused before it is opened."""
    p = Path(str(value))
    return "test" in str(p).lower() or "test" in str(p.resolve()).lower()


def refuse_test_paths(paths: dict) -> None:
    for what, value in paths.items():
        if value is not None and names_test(value):
            _refuse("forbidden_path", f"{what} {value!r} names 'test' in its given or resolved form")


def _no_constant(c):
    raise ValueError(f"non-strict JSON constant {c}")


def read_json(path: Path, code: str) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_no_constant)
    except (OSError, ValueError) as e:
        _refuse(code, f"{path}: {e}")
    if not isinstance(doc, dict):
        _refuse(code, f"{path} is not a JSON object")
    return doc


def torchscript_extra_names(path) -> list[str]:
    """The extra files inside a TorchScript zip, read with zipfile (nothing is loaded)."""
    with zipfile.ZipFile(path) as z:
        return sorted(n.split("/extra/", 1)[1] for n in z.namelist() if "/extra/" in n)


# ------------------------------------------------------------------ the INT8 source: the evaluator's checks, then ours
def int8_provenance(stage: str, provenance, *, expect_artifact_sha256: str, allowed_stages) -> dict:
    """An all-INT8 converted artifact through its provenance, with the evaluator's INT8 checks intact.

    validate_int8_artifact (unchanged), then: the stage is one this route allows (else unknown); the format is
    torchscript; cwd_projection_loaded is false; the artifact's sha256 is the manifest's; the conversion's own checks
    (per-channel INT8 weights, no undeclared float module, the float boundary, parity) all passed; the zip carries
    plantseg_int8.json and no mixed-precision identity, and that identity names the provenance's stage and run."""
    from src.eval import stage_artifacts as SA
    from src.quant.qat_artifacts import read_ts_identity
    if stage not in allowed_stages:
        _refuse("artifact_unknown", f"stage {stage!r} is not one this route scores as int8 ({list(allowed_stages)})")
    if not isinstance(expect_artifact_sha256, str) or len(expect_artifact_sha256) != 64:
        _refuse("expect_artifact_sha256", "--expect-artifact-sha256 (AM-29's manifest value) is required: 64 hex")
    try:
        resolved = SA.validate_int8_artifact(stage, provenance)
    except SA.StageArtifactError as e:
        _refuse(f"int8_{e.code}", str(e))
    prov = resolved["provenance"]
    if resolved["artifact_format"] != "torchscript":
        _refuse("int8_format", f"{provenance} declares artifact_format {resolved['artifact_format']!r}; the QAT "
                               "artifact of record is the QNNPACK TorchScript")
    if prov.get("cwd_projection_loaded") is not False:
        _refuse("int8_projection_flag", f"{provenance} records cwd_projection_loaded {prov.get('cwd_projection_loaded')!r}")
    if resolved["artifact_sha256"] != expect_artifact_sha256:
        _refuse("int8_hash_mismatch", f"{resolved['artifact_path']} has sha256 {resolved['artifact_sha256']}; AM-29's "
                                      f"manifest records {expect_artifact_sha256}")
    checks = (prov.get("verification") or {}).get("checks") or {}
    if not checks or not all(v is True for v in checks.values()):
        _refuse("int8_not_verified", f"{provenance}: the conversion's checks are {checks!r}; an all-INT8 artifact "
                                     "passed every one")
    if prov.get("precision_label") is not None or prov.get("mixed_precision"):
        _refuse("int8_mixed_precision", f"{provenance} labels its artifact {prov.get('precision_label')!r}")
    extras = torchscript_extra_names(resolved["artifact_path"])
    if MIXED_META in extras or INT8_META not in extras:
        _refuse("int8_mixed_precision", f"{resolved['artifact_path']} carries {extras}: a mixed-precision artifact "
                                        f"(or one without {INT8_META}) is never scored as int8")
    ident = read_ts_identity(resolved["artifact_path"])
    differ = [k for k in ("stage", "run_id", "epoch", "qat_checkpoint_sha256") if ident.get(k) != prov.get(k)]
    if prov.get("identity") != ident:
        differ.append("identity")
    if ident.get("stage") != stage or ident.get("quantization") != "qat":
        differ.append("stage/quantization")
    if differ:
        _refuse("int8_identity", f"{resolved['artifact_path']}'s identity {ident} disagrees with its provenance: {differ}")
    return {**resolved, "identity": ident, "provenance_sha256": sha256_file(provenance),
            "label": LABEL_INT8}


def load_int8_model(resolved: dict):
    """src.eval.model_loading.load_int8_student, unchanged (the QNNPACK engine, the TorchScript identity check)."""
    from src.eval.model_loading import load_int8_student
    return load_int8_student({k: resolved[k] for k in ("spec", "provenance", "artifact_path", "artifact_sha256",
                                                       "artifact_format")}, require_qnnpack=True)[0]


# ------------------------------------------------------------------ the two finalized sources
def bind_am29(eval_dir, *, expect_artifact_sha256: str, arm: str | None = None) -> dict:
    """An AM-29 arm's finalized block B: am29_finalize.json names the artifact of record and am29_selection.json, whose
    winner names the same files; the artifact's sha256 is the manifest's; its provenance passes int8_provenance."""
    from src.quant import am29_registry as R
    R.register()
    E = Path(eval_dir)
    refuse_test_paths({"--am29-eval-dir": E})
    fin = read_json(E / AM29_FINALIZE, "am29_finalize_unreadable")
    sel_p = E / AM29_SELECTION
    sel = read_json(sel_p, "am29_selection_unreadable")
    if fin.get("format") != AM29_FINALIZE_FORMAT or fin.get("status") != "finalized":
        _refuse("am29_finalize_format", f"{E / AM29_FINALIZE} is not a finalized {AM29_FINALIZE_FORMAT} record")
    if sel.get("format") != AM29_SELECTION_FORMAT:
        _refuse("am29_selection_format", f"{sel_p} is not an {AM29_SELECTION_FORMAT} record")
    if (fin.get("am29_selection") or {}).get("sha256") != sha256_file(sel_p):
        _refuse("am29_finalize_stale", f"{E / AM29_FINALIZE} names another am29_selection.json")
    stage = sel.get("stage")
    if stage not in R.AM29_STAGES:
        _refuse("artifact_unknown", f"{sel_p} names stage {stage!r}, not an AM-29 stage")
    if arm is not None and R.ARM_STAGES.get(arm) != stage:
        _refuse("arm_mismatch", f"--arm {arm} is stage {R.ARM_STAGES.get(arm)!r}; the selection is {stage!r}")
    win = sel.get("winner") or {}
    aor = (fin.get("artifact_of_record") or {})
    if aor.get("torchscript") != win.get("artifact_of_record") or aor.get("provenance") != win.get("provenance") \
            or aor.get("checkpoint") != {"path": win.get("checkpoint"), "sha256": win.get("checkpoint_sha256")}:
        _refuse("am29_finalize_disagrees", f"{E / AM29_FINALIZE}'s artifact of record is not the selection's winner")
    ts = E / str((win.get("artifact_of_record") or {}).get("path"))
    prov = E / str((win.get("provenance") or {}).get("path"))
    if sha256_file(prov) != (win.get("provenance") or {}).get("sha256"):
        _refuse("am29_provenance_changed", f"{prov} differs from the selection's sha256")
    if not ts.is_file() or sha256_file(ts) != (win.get("artifact_of_record") or {}).get("sha256"):
        _refuse("int8_hash_mismatch", f"{ts} is missing or differs from the selection's sha256")
    resolved = int8_provenance(stage, prov, expect_artifact_sha256=expect_artifact_sha256, allowed_stages=R.AM29_STAGES)
    if resolved["identity"].get("epoch") != win.get("epoch") or resolved["identity"].get("run_id") != sel.get("run_id"):
        _refuse("int8_identity", "the artifact's identity is not the selection's winner epoch and run")
    return {**resolved, "source": "am29", "stage": stage, "arm": R.STAGE_ARMS[stage], "am29_label": R.STAGE_ARMS[stage],
            "selection": sel, "selection_sha256": sha256_file(sel_p), "finalize_sha256": sha256_file(E / AM29_FINALIZE),
            "eval_dir": str(E.resolve())}


def bind_l_am24(blockb_dir, *, run_id: str, expect_artifact_sha256: str) -> dict:
    """lane/l-am24's block B of N2 or E6-R2-M: <out-dir>/x_blockb.json (x_qat_blockb/1, exploratory, every step exit 0,
    the telemetry unchanged) whose selection is <out-dir>/eval/qat_selection.json, finalized by the pinned
    qat_epoch_eval.py (eval/qat_finalize.json); the artifact of record named by both, of the run ID of record."""
    B = Path(blockb_dir)
    refuse_test_paths({"--l-am24-blockb-dir": B})
    if run_id not in L_AM24_RUNS:
        _refuse("artifact_unknown", f"--run-id {run_id!r} is not one of AM-29's runs on lane/l-am24 {sorted(L_AM24_RUNS)}")
    xb = read_json(B / "x_blockb.json", "blockb_unreadable")
    E = B / "eval"
    sel_p, fin_p = E / "qat_selection.json", E / "qat_finalize.json"
    sel, fin = read_json(sel_p, "blockb_selection_unreadable"), read_json(fin_p, "blockb_finalize_unreadable")
    steps = xb.get("steps") or []
    bad = [k for k, ok in (("format", xb.get("format") == L_AM24_BLOCKB_FORMAT), ("exploratory", xb.get("exploratory") is True),
                           ("split", xb.get("split") == "val"), ("telemetry_unchanged", xb.get("telemetry_unchanged") is True),
                           ("steps", [s.get("step") for s in steps] == ["convert", "score", "select", "finalize"]
                            and all(s.get("exit") == 0 for s in steps)),
                           ("selection", xb.get("selection") == sel)) if not ok]
    if bad:
        _refuse("blockb_record", f"{B / 'x_blockb.json'} is not a completed registered-path block B of this selection: {bad}")
    if sel.get("format") != QAT_SELECTION_FORMAT or sel.get("run_id") != run_id \
            or sel.get("telemetry_sha256") != xb.get("telemetry_sha256"):
        _refuse("blockb_selection", f"{sel_p} is not run {run_id}'s {QAT_SELECTION_FORMAT} selection of this block B")
    win = sel.get("winner") or {}
    if fin.get("format") != QAT_FINALIZE_FORMAT or fin.get("status") != "finalized" \
            or fin.get("qat_selection") != {"path": "qat_selection.json", "sha256": sha256_file(sel_p)} \
            or (fin.get("artifact_of_record") or {}).get("torchscript") != win.get("artifact_of_record") \
            or fin.get("run_id") != run_id:
        _refuse("blockb_finalize", f"{fin_p} is not the finalize record of {sel_p}")
    prov = E / str((win.get("provenance") or {}).get("path"))
    if not prov.is_file() or sha256_file(prov) != (win.get("provenance") or {}).get("sha256"):
        _refuse("blockb_provenance_changed", f"{prov} is missing or differs from the selection's sha256")
    pdoc = read_json(prov, "blockb_provenance_unreadable")
    stage = pdoc.get("stage")
    resolved = int8_provenance(stage, prov, expect_artifact_sha256=expect_artifact_sha256, allowed_stages=L_AM24_STAGES)
    if resolved["artifact_sha256"] != (win.get("artifact_of_record") or {}).get("sha256"):
        _refuse("int8_hash_mismatch", "the artifact is not the selection's artifact of record")
    if resolved["identity"].get("run_id") != run_id or resolved["identity"].get("epoch") != win.get("epoch"):
        _refuse("int8_identity", f"the artifact's identity names run {resolved['identity'].get('run_id')!r} epoch "
                                 f"{resolved['identity'].get('epoch')!r}; the selection {run_id!r} {win.get('epoch')!r}")
    return {**resolved, "source": "l-am24", "stage": stage, "am29_label": L_AM24_RUNS[run_id], "run_id": run_id,
            "selection": sel, "selection_sha256": sha256_file(sel_p), "finalize_sha256": sha256_file(fin_p),
            "blockb_sha256": sha256_file(B / "x_blockb.json"), "item": xb.get("item"), "eval_dir": str(E.resolve())}


# ------------------------------------------------------------------ mixed precision: the same weights, an adopted group
def selected_state(bound: dict, *, run_dir, expect_checkpoint_sha256: str) -> dict:
    """The selected epoch's pre-convert QAT state, bound through the selection: the run's telemetry is the selection's,
    the epoch's eNN.pt has the sha256 its epoch_end row, the selection's winner and AM-29's manifest record, it loads
    weights_only as a qat-train-state sidecar of the run's identity, and its state is finite."""
    from src.quant import qat_artifacts as A
    from src.quant.x86_latency import load_qat_sidecar
    sel = bound["selection"]
    win = sel.get("winner") or {}
    refuse_test_paths({"--run-dir": run_dir})
    rec = A.read_run_record(run_dir)
    A.require_complete(rec)
    if rec["telemetry_sha256"] != sel.get("telemetry_sha256") or rec["run_id"] != sel.get("run_id"):
        _refuse("mixed_run_mismatch", f"{run_dir} is run {rec['run_id']!r} with telemetry {rec['telemetry_sha256']}; "
                                      f"the selection names {sel.get('run_id')!r}, {sel.get('telemetry_sha256')}")
    ck, ck_sha = A.verified_checkpoint(rec, win.get("epoch"))
    if not (ck_sha == win.get("checkpoint_sha256") == expect_checkpoint_sha256):
        _refuse("mixed_checkpoint_mismatch", f"{ck} has sha256 {ck_sha}; the selection records "
                                             f"{win.get('checkpoint_sha256')}, AM-29's manifest {expect_checkpoint_sha256}")
    payload = load_qat_sidecar(ck)
    meta = rec["run_meta"]
    ident = {"epoch": win.get("epoch"), "run_id": rec["run_id"], "seed": meta.get("seed"), "stage": meta.get("stage"),
             "source_checkpoint_sha256": meta.get("source_checkpoint_sha256")}
    wrong = {k: payload.get(k) for k, v in ident.items() if payload.get(k) != v}
    if wrong or payload.get("state_finite") is not True:
        _refuse("mixed_checkpoint_identity", f"{ck} records {wrong} / state_finite {payload.get('state_finite')!r}")
    return {"state": payload["model_state_dict"], "checkpoint": str(ck), "checkpoint_sha256": ck_sha,
            "epoch": win.get("epoch"), "run_id": rec["run_id"]}


def mixed_model(state: dict, group: str) -> tuple[object, dict]:
    """The same weights converted with FP32 islands of `group` (src/quant/am29_islands.py), with its record."""
    from src.quant import am29_islands as I
    if group not in I.GROUPS or group == "G0":
        _refuse("mixed_group", f"a mixed-precision score takes a P2 group G1-G4, not {group!r} (G0 is the int8 path)")
    res = I.convert_qat_state(state, groups=(group,))
    model, rep = res["variants"][group]
    rec = I.variant_record(group, model, rep, g0=res["g0"], prepared_qat=res["prepared"])
    if not rec["checks_ok"]:
        _refuse("mixed_checks", f"the {group} conversion failed its checks: "
                                f"{[k for k, v in rec['checks'].items() if not v.get('ok')]}")
    return model, {k: rec[k] for k in ("group", "text", "members", "label", "shares", "eligible", "checks_ok")}


# ------------------------------------------------------------------ the scoring
def score_model(model, *, split: str, max_samples: int | None = None, batch_size: int = 1,
                protocol: str = "canvas") -> dict:
    """Score an already-built model on `split` with the evaluator's own data and metric code (the steps of
    scripts/evaluate_model.py run() between its validation and write_artifact). Dataset-level values only."""
    import torch

    from scripts.evaluate_model import EXPECTED_SPLIT_ROWS
    from src.eval import Condition, evaluate_model
    from src.eval.adapters import (build_eval_loader, build_expected_manifest_for, build_protocol_adapter,
                                   deterministic_subset, load_class_map, protocol_forward_kwargs)
    from src.eval.eval_runtime import (EVAL_NUM_WORKERS, ModelDeviceForward, build_eval_runtime_record,
                                       check_post_eval, evaluate_capturing_warnings, resolve_model_device)
    from src.eval.protocols import PROTOCOL_IDS
    if split not in EXPECTED_SPLIT_ROWS:
        _refuse("split", f"split {split!r} is not one of {sorted(EXPECTED_SPLIT_ROWS)}")
    class_map = load_class_map()
    n_total = EXPECTED_SPLIT_ROWS[split]
    source_indices = (deterministic_subset(n_total, max_samples) if max_samples is not None else list(range(n_total)))
    expected_manifest = build_expected_manifest_for(split, source_indices)
    adapter = build_protocol_adapter(protocol, split, source_indices)
    model_device = resolve_model_device("cpu", cpu_only=True)
    fwd = ModelDeviceForward(model_device)
    loader = build_eval_loader(adapter, batch_size, num_workers=EVAL_NUM_WORKERS)
    cond = Condition("clean", None, None)
    result, warn_summary = evaluate_capturing_warnings(
        evaluate_model, model, loader, expected_manifest=expected_manifest, condition=cond, num_classes=116,
        background_index=0, ignore_index=255, **protocol_forward_kwargs(protocol, fwd))
    check_post_eval(model=model, model_device=model_device, fwd=fwd, policy_applied=False)
    runtime = build_eval_runtime_record(model=model, model_device=model_device, fwd=fwd, batch_size=batch_size,
                                        forward_batches=result.forward_batches, policy_applied=False,
                                        policy_info=None, ckpt_info=None, warn_summary=warn_summary)
    ids = [m.image_id for m in expected_manifest]
    return {"split": split, "protocol": PROTOCOL_IDS[protocol], "expected_rows": len(expected_manifest),
            "actual_rows": len(result.rows), "max_samples": max_samples, "batch_size": batch_size,
            "forward_batches": result.forward_batches, "num_workers": EVAL_NUM_WORKERS,
            "gt_support_sum": int(result.dataset_gt.sum()), "class_map_sha256": class_map.semantic_sha256,
            "manifest_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            "dataset_level": dict(result.dataset_level), "torch_num_threads": int(torch.get_num_threads()),
            "torch": torch.__version__, "eval_runtime": runtime}


def dataset_level_lines(label: str, name: str, score: dict) -> list[str]:
    lv = score["dataset_level"]
    keys = ("all_class_miou", "all_class_miou_n_eligible", "disease_only_miou", "disease_only_miou_n_eligible",
            "all_class_macro_dice", "all_class_macc", "aacc_diagnostic")
    return [f"[am29-score] {name} ({label}) {score['split']} rows {score['actual_rows']}/{score['expected_rows']} "
            f"protocol {score['protocol']} threads {score['torch_num_threads']}"] + \
           [f"  {k} = {lv.get(k)!r}" for k in keys]


__all__ = ["AM29_FINALIZE", "AM29_FINALIZE_FORMAT", "AM29_SELECTION", "AM29_SELECTION_FORMAT", "FORMAT", "LABEL_INT8",
           "LABEL_MIXED", "L_AM24_RUNS", "L_AM24_STAGES", "MIXED_META", "ScoreRefused", "bind_am29", "bind_l_am24",
           "dataset_level_lines", "int8_provenance", "load_int8_model", "mixed_model", "names_test", "read_json",
           "refuse_test_paths", "score_model", "selected_state", "sha256_file", "torchscript_extra_names"]
