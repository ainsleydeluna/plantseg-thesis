"""Bridge between produced E1-E7 and A/F/G artifacts and the existing evaluator. Resolver + gates only.

This module does NOT re-implement evaluation, metrics, or the artifact contract. It answers three
questions the evaluator currently cannot, and delegates everything else:

  1. which artifact KIND does a stage produce (FP32 checkpoint vs converted INT8 artifact)?
  2. is a given artifact genuinely that stage's output (provenance, source stage, hash, backend)?
  3. may this run be LAUNCHED as `artifact_status="official"` — judged only on facts that exist
     before a model or dataset is constructed?

Question 3 is strictly PRE-RUN. Everything post-inference — `actual_rows == expected_rows`,
manifest/ID/order integrity, the final artifact schema — stays owned by `src/eval/artifacts.py`,
which computes `actual_rows` from the real result and refuses to finalise a mismatch. This module
deliberately holds no second copy of that rule. Governed-path cleanliness is likewise delegated to
the existing implementation, so no global repository-cleanliness requirement is introduced and the
protected reference PDF remains allowlisted.

For E4-E7 the input is the run-provenance JSON written by `src/quant/runner.py`; for E1-E3 and the
exploratory arms A, F, G it is the FP32 checkpoint read through the existing
`src.eval.model_loading` contract. For precision qat_fakequant_off (lane L-FQOFF) it is a selected QAT
model's pre-convert state, `<run>/epoch_ckpts/eNN.pt`, bound through its selection of record
(validate_qat_fakequant_off_source).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

NUM_CLASSES = 116
OFFICIAL_SPLIT = "test"
OFFICIAL_ROWS = 1561
OFFICIAL_BACKEND = "qnnpack"
# = src.eval.model_loading.INT8_ARTIFACT_FORMATS (restated so this module stays import-light);
# a provenance without the key is the src/quant/runner.py state_dict.
INT8_ARTIFACT_FORMATS = ("state_dict", "torchscript")

# Stage -> artifact contract (IMPLEMENTATION_CONTRACT B4 stage table).
STAGE_ARTIFACTS: dict[str, dict] = {
    "E1": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
    "E2": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
    "E3": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
    # The exploratory arms (AM-17 item 7; AM-17b items 1(a), 1(b); L-CKPT-GUARD): FP32 students that
    # train_distill writes like E2/E3. Descriptive only (DESCRIPTIVE_ONLY_STAGES).
    "A": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
    "F": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
    "G": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
    "E4": {"kind": "int8_artifact", "precision": "int8_ptq", "source_stage": "E1", "method": "ptq"},
    "E5": {"kind": "int8_artifact", "precision": "int8_qat", "source_stage": "E1", "method": "qat"},
    "E6": {"kind": "int8_artifact", "precision": "int8_qat", "source_stage": "E3", "method": "qat"},
    "E7": {"kind": "int8_artifact", "precision": "int8_ptq", "source_stage": "E3", "method": "ptq"},
    # AM-23 items 1-4: descriptive INT8 stages, never comparators (DESCRIPTIVE_ONLY_STAGES). E6-E2 and E7-E2 derive
    # from E2 seed 42, E6-R1 and E6-R2 from the arm runs E3-R1 and E3-R2 (stage "E3" in their run_meta).
    "E6-E2": {"kind": "int8_artifact", "precision": "int8_qat", "source_stage": "E2", "method": "qat"},
    "E7-E2": {"kind": "int8_artifact", "precision": "int8_ptq", "source_stage": "E2", "method": "ptq"},
    "E6-R1": {"kind": "int8_artifact", "precision": "int8_qat", "source_stage": "E3", "method": "qat"},
    "E6-R2": {"kind": "int8_artifact", "precision": "int8_qat", "source_stage": "E3", "method": "qat"},
    # AM-3 (lane L-AM3, B-2): E6-KD, E6's QAT with the teacher of record and E3's terms at 0.5x; descriptive, never
    # a comparator. Its conversion records method "qat", as E6's does (src/quant/qat_artifacts.py).
    "E6-KD": {"kind": "int8_artifact", "precision": "int8_qat", "source_stage": "E3", "method": "qat"},
    # The frozen schema (EVALUATION_CONTRACT 5.x) already admits stage/model_role "teacher" at fp32.
    # The teacher is a DESCRIPTIVE REFERENCE: its clean artifact may legitimately be `official`
    # (nothing in the committed contract forbids that), but it never becomes an inferential
    # comparator — that restriction lives in the statistics stage, not in artifact status.
    "TEACHER": {"kind": "teacher_checkpoint", "precision": "fp32", "source_stage": None,
                "method": None},
}
PROJECTION_FREE_STAGES = ("E6", "E7", "E6-E2", "E7-E2", "E6-R1", "E6-R2", "E6-KD")
# FP32 checkpoints train_distill writes, which record their stage (DL-52): refused when they record none.
# E1 (train_e1) records none and is not listed.
STAGE_DECLARING_FP32_STAGES = ("E2", "E3", "A", "F", "G")
# FP32 stages whose student state must hold no training-only CWD projection key.
PROJECTION_FREE_FP32_STAGES = ("E3", "A", "F", "G")
TEACHER_STAGE = "TEACHER"
STUDENT_ROLE, TEACHER_ROLE = "student", "teacher"
# Never a comparator in an inferential test. Nothing in the evaluator enforces this; the statistics
# driver must refuse these stages as comparators.
DESCRIPTIVE_ONLY_STAGES = ("TEACHER", "A", "F", "G", "E6-E2", "E7-E2", "E6-R1", "E6-R2", "E6-KD")


class StageArtifactError(RuntimeError):
    """A stage/artifact mismatch or a failed official precondition. `code` keeps gates testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str):
    raise StageArtifactError(code, message)


def resolve_stage_artifact(stage: str) -> dict:
    """Return the artifact contract for a stage: E1-E7, the arms A, F and G, or the teacher.

    Unknown stages fail loudly; nothing here invents a contract for a stage the table does not hold.
    """
    key = str(stage).upper()
    if key not in STAGE_ARTIFACTS:
        _fail("unknown_stage",
              f"no artifact contract for stage {stage!r}; known stages: {sorted(STAGE_ARTIFACTS)}")
    return {"stage": key, **STAGE_ARTIFACTS[key]}


# ------------------------------------------------------------------ INT8 (E4-E7)
def validate_int8_artifact(stage: str, provenance_path: str | Path) -> dict:
    """Validate an E4-E7 converted INT8 artifact through its runner provenance.

    Checks stage identity, source stage, random_init, class count, the artifact format, artifact
    existence, the recorded SHA-256 against the file on disk, the official backend, and — for E6/E7 —
    that the training-only CWD projection was never loaded. A relative `converted_artifact` (written
    by src/quant/ptq.py, so a run directory can move between a container and its host) is resolved
    against the provenance file's directory; the recorded SHA-256 stays the trust anchor either way.
    """
    from src.eval.model_loading import sha256_file

    spec = resolve_stage_artifact(stage)
    if spec["kind"] != "int8_artifact":
        _fail("stage_is_not_int8",
              f"{spec['stage']} produces a {spec['kind']}; INT8 provenance validation does not apply")

    p = Path(provenance_path)
    if not p.is_file():
        _fail("provenance_missing", f"run provenance not found: {p}")
    try:
        prov = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        _fail("provenance_unreadable", f"could not parse run provenance {p}: {e}")

    if str(prov.get("stage", "")).upper() != spec["stage"]:
        _fail("stage_mismatch",
              f"provenance is stage {prov.get('stage')!r}, requested {spec['stage']}")
    if str(prov.get("source_stage", "")).upper() != spec["source_stage"]:
        _fail("source_stage_mismatch",
              f"{spec['stage']} must derive from {spec['source_stage']}, provenance says "
              f"{prov.get('source_stage')!r}")
    if prov.get("quantization_method") != spec["method"]:
        _fail("method_mismatch",
              f"{spec['stage']} is {spec['method'].upper()}, provenance says "
              f"{prov.get('quantization_method')!r}")
    if prov.get("random_init") is not False:
        _fail("random_init_artifact",
              f"{spec['stage']} artifact records random_init={prov.get('random_init')!r}; a "
              "quantized artifact always derives from a trained source")
    if int(prov.get("num_classes", -1)) != NUM_CLASSES:
        _fail("num_classes_mismatch",
              f"artifact declares num_classes={prov.get('num_classes')!r}, expected {NUM_CLASSES}")
    if prov.get("backend") != OFFICIAL_BACKEND:
        _fail("backend_not_official",
              f"artifact was produced on backend {prov.get('backend')!r}; the official INT8 "
              f"artifact requires {OFFICIAL_BACKEND!r}")
    if spec["stage"] in PROJECTION_FREE_STAGES and prov.get("cwd_projection_loaded") is not False:
        _fail("projection_flag_missing",
              f"{spec['stage']} provenance must record cwd_projection_loaded=false, got "
              f"{prov.get('cwd_projection_loaded')!r}")

    fmt = prov.get("artifact_format", "state_dict")
    if fmt not in INT8_ARTIFACT_FORMATS:
        _fail("artifact_format_unknown",
              f"provenance declares artifact_format={fmt!r}; known formats: "
              f"{list(INT8_ARTIFACT_FORMATS)}")

    art = prov.get("converted_artifact")
    if not art:
        _fail("artifact_path_missing", "provenance records no converted_artifact")
    art_path = Path(art)
    if not art_path.is_absolute():
        art_path = p.resolve().parent / art_path
    if not art_path.is_file():
        _fail("artifact_file_missing", f"converted artifact not found: {art_path}")
    recorded = prov.get("converted_artifact_sha256")
    if not recorded:
        _fail("artifact_hash_missing", "provenance records no converted_artifact_sha256")
    actual = sha256_file(art_path)
    if actual != recorded:
        _fail("artifact_hash_mismatch",
              f"converted artifact hash {actual[:16]}… does not match the recorded "
              f"{str(recorded)[:16]}… — the artifact changed after the run")
    return {"spec": spec, "provenance": prov, "artifact_path": art_path,
            "artifact_sha256": actual, "artifact_format": fmt}


def int8_official_calibration_error(resolved: dict, repo_root: str | Path | None = None) -> str | None:
    """PRE-RUN: may this INT8 artifact be scored as the OFFICIAL E4/E7? None when it may.

    A PTQ artifact written by src/quant/ptq.py (artifact_format "torchscript") records its calibration
    list. Only the registered AM-10 list of record, as committed at
    configs/calibration/ptq_calibration_seed42.json, yields an official E4/E7 score; runs on the AM-16
    item 5 sensitivity lists or on synthetic lists stay descriptive. QAT artifacts and the
    src/quant/runner.py state_dict artifacts (whose index validator accepts seed 42 only) are unaffected.
    """
    prov = resolved["provenance"]
    if resolved["spec"]["method"] != "ptq" or prov.get("artifact_format", "state_dict") != "torchscript":
        return None
    from src.quant.ptq import list_of_record_error
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    err = list_of_record_error(prov.get("calibration") or {}, root)
    return None if err is None else f"refusing an official {resolved['spec']['stage']} score: {err}"


def int8_official_error(resolved: dict, repo_root: str | Path | None = None) -> str | None:
    """PRE-RUN: may this INT8 artifact be scored as the OFFICIAL E4-E7? None when it may.

    The one function an official INT8 path calls (LQP, ruling R2): int8_official_calibration_error's
    refusal first (the committed AM-10 list of record), then the PTQ record gate's (ruling Q1-R2,
    L-CKPT-GUARD Q-part). QAT artifacts (E5, E6) pass the record gate.
    """
    return int8_official_calibration_error(resolved, repo_root) or _ptq_record_error(resolved)


def _ptq_record_error(resolved: dict) -> str | None:
    """The record gate: an official E4, E7 or E7-E2 is scripts/run_ptq.py's TorchScript artifact whose directory
    holds its <stage>_ptq_record.json (src/quant/ptq_record.py, imported here only) of the stage's role of record
    (role_of_record: of_record; quant_descriptive for E7-E2, AM-23 item 2), written by the run that wrote this
    run provenance. legacy_ptq_artifact: a src/quant/runner.py state_dict PTQ artifact; ptq_record_missing: no
    record (an E7-X or E7-flip arm directory holds e7_arm_record.json and none); ptq_record_invalid;
    ptq_record_role: another role (calibration_sensitivity, d3_replicate, or quant_descriptive for E4/E7);
    ptq_record_mismatch: the stage, the run provenance, the TorchScript artifact, the source checkpoint or the
    list checksum differ.
    """
    spec, prov = resolved["spec"], resolved["provenance"]
    if spec["method"] != "ptq":
        return None
    stage = spec["stage"]
    lead = f"refusing an official {stage} score: "
    if prov.get("artifact_format", "state_dict") != "torchscript":
        return (lead + "[legacy_ptq_artifact] a src/quant/runner.py state_dict PTQ artifact makes no "
                f"{stage} of record (ruling Q1-R2); the PTQ of record is scripts/run_ptq.py's TorchScript artifact")
    from src.quant.ptq import artifact_names
    from src.quant.ptq_record import RECORD_NAME, read_record, role_of_record
    run_dir = Path(resolved["artifact_path"]).parent
    path = run_dir / RECORD_NAME.format(key=stage.lower())
    if not path.is_file():
        return (lead + f"[ptq_record_missing] {path} is absent: an official {stage} is made by scripts/run_ptq.py's "
                "bound path, which writes its PTQ record last")
    try:
        record, _ = read_record(path)
        run_meta = json.loads((run_dir / artifact_names(stage.lower())["run_meta"]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return lead + f"[ptq_record_invalid] {path}: {e}"
    if record["role"] != role_of_record(stage):
        return (lead + f"[ptq_record_role] {path} has role {record['role']!r}; only a PTQ record of role "
                f"{role_of_record(stage)!r} is official for {stage} (calibration-sensitivity runs and d3 "
                "replicates are descriptive)")
    differ = [k for k, ok in (
        ("stage", record["stage"] == stage), ("run provenance", run_meta == prov),
        ("torchscript", record["artifacts"]["torchscript"]["sha256"] == resolved["artifact_sha256"]),
        ("source checkpoint", record["parent"].get("checkpoint_sha256") == prov.get("source_checkpoint_sha256")),
        ("list checksum", record["calibration"].get("checksum_sha256")
         == (prov.get("calibration") or {}).get("checksum_sha256"))) if not ok]
    if differ:
        return lead + f"[ptq_record_mismatch] {path} is not this run's record: {differ} differ"
    return None


# ------------------------------------------------------------------ FP32 (E1-E3, A, F, G)
def validate_fp32_artifact(stage: str, checkpoint_path: str | Path) -> dict:
    """Validate an E1, E2, E3, A, F or G FP32 checkpoint. Uses the existing checkpoint contract.

    DL-52: a checkpoint of a stage in STAGE_DECLARING_FP32_STAGES must record its stage
    (`stage_undeclared`) and record the requested one (`stage_mismatch`); an E1 checkpoint records
    none, so it is refused under those stages. Then the projection-free check for
    PROJECTION_FREE_FP32_STAGES (`CWDProjectionLeak`)."""
    import torch

    from src.eval.model_loading import sha256_file

    spec = resolve_stage_artifact(stage)
    if spec["kind"] != "fp32_checkpoint":
        _fail("stage_is_not_fp32",
              f"{spec['stage']} produces a {spec['kind']}; pass its run provenance instead")
    p = Path(checkpoint_path)
    if not p.is_file():
        _fail("checkpoint_missing", f"{spec['stage']} checkpoint not found: {p}")
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        _fail("checkpoint_not_fp32_student",
              f"{p} is not a student training checkpoint (no model_state_dict)")
    state = ckpt["model_state_dict"]
    if any("_packed_params" in k or "activation_post_process" in k for k in state):
        _fail("checkpoint_is_quantized",
              f"{p} carries quantization state; an FP32 stage cannot consume an INT8 artifact")
    declared = ckpt.get("stage")
    if declared is None and spec["stage"] in STAGE_DECLARING_FP32_STAGES:
        _fail("stage_undeclared",
              f"{p} records no stage; a {spec['stage']} checkpoint written by train_distill records "
              "its stage (an E1 checkpoint records none and is not a "
              f"{spec['stage']} checkpoint)")
    if declared is not None and str(declared).upper() != spec["stage"]:
        _fail("stage_mismatch",
              f"checkpoint is stage {declared!r}, requested {spec['stage']}")
    if spec["stage"] in PROJECTION_FREE_FP32_STAGES:
        from src.distill.export import assert_clean_student_state
        assert_clean_student_state(state)          # projection must be absent from the student
    if ckpt.get("num_classes") is not None and int(ckpt["num_classes"]) != NUM_CLASSES:
        _fail("num_classes_mismatch",
              f"checkpoint declares num_classes={ckpt['num_classes']}, expected {NUM_CLASSES}")
    return {"spec": spec, "checkpoint_path": p, "checkpoint_sha256": sha256_file(p),
            "declared_stage": declared}


# ------------------------------------------------------------------ fake-quant-off (AM-4; lane L-FQOFF, FQR2)
# Restated so this module stays import-light: scripts/qat_epoch_eval.py's SELECTION_RECORD, FINALIZE_RECORD and
# FINALIZE_FORMAT. The src/quant modules load inside validate_qat_fakequant_off_source only.
QAT_SELECTION_NAME = "qat_selection.json"
QAT_FINALIZE_NAME = "qat_finalize.json"
QAT_FINALIZE_FORMAT = "qat_finalize/1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _no_json_constant(c):
    raise ValueError(f"non-strict JSON constant {c}")


def _is_torchscript_archive(path: Path) -> bool:
    """A TorchScript archive (torch.jit.save) holds constants.pkl and torch.save's archives never do. Read with zipfile:
    nothing is deserialised."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return any(n == "constants.pkl" or n.endswith("/constants.pkl") for n in z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def validate_qat_fakequant_off_source(stage: str, checkpoint: str | Path | None, *, run_dir: str | Path | None,
                                      selection: str | Path | None, selection_sha256: str | None,
                                      telemetry_sha256: str | None, allow_smoke_inputs: bool = False) -> dict:
    """Bind precision qat_fakequant_off's source (AM-4's fake-quant-off score; lane L-FQOFF, ruling FQR2): a selected
    QAT model's pre-convert state, `<run>/epoch_ckpts/eNN.pt`. PRE-RUN: no model and no dataset exist here.

    The binding is the selection of record, `<eval>/qat_selection.json` (src/quant/qat_select.py), its sha256 pinned
    to its decision-log row. A `qat_finalize.json` beside it is checked whenever it exists and is never required
    (E6-KD has none). In FQR12's order, each refusal a StageArtifactError with its fqoff code:
      6  fqoff_forbidden_path: a path this function would open names "test", in its given or resolved form (Q2's
         names_test); nothing has been opened yet
      7  fqoff_selection_sha256_mismatch: the selection is absent, or its bytes differ from the pin
      8  fqoff_selection_not_of_record: its name; its format; its rules_sha256 against the committed rules file of its
         stage (rules_path_for_stage); smoke_inputs, outside smoke status; its winner's fields; and, once the run is
         read, its run_id
      9  fqoff_stage_mismatch: the selection's stage, then the run's run_meta stage, then the payload's
      10 fqoff_run_record: read_run_record refuses (Q2's code kept in the message)
      11 fqoff_run_incomplete: require_complete refuses (AM-19)
      12 fqoff_run_not_real: the run's mode is not "real", outside smoke status
      13 fqoff_telemetry_sha256_mismatch: the telemetry against the pin (the pod's copy-back value) and the selection
      14 fqoff_finalize_disagrees: a finalize record beside the selection is unreadable, not qat_finalize/1 finalized,
         or differs from the pin, the winner's checkpoint {path, sha256}, the run_id, the telemetry or the stage
      15 fqoff_not_the_winner: winner.checkpoint is not epoch_ckpts/e<epoch:02d>.pt, or `checkpoint` is not that file of
         the run (nor the file its epoch_end row names)
      16 fqoff_checkpoint_missing
      17 fqoff_checkpoint_sha256_mismatch: the file against its epoch_end row and the selection's winner (with 15 and
         16, verified_checkpoint's checks)
      18 fqoff_not_qat_train_state: a TorchScript archive (a converted artifact), a payload load_qat_sidecar refuses
         (weights_only; its kind, state and role flags) or any unreadable file, or an artifact_role other than
         preconvert_qat_state
      19 fqoff_checkpoint_identity: the payload's epoch, run_id, seed or source checkpoint is not the run's
         (convert_epoch's identity)
      20 fqoff_state_not_finite: the stored state_finite is not True (AM-21 item 1(c))
    It reads the run's telemetry, the selection, the committed rules file of the stage, the finalize record when
    present and the winner's eNN.pt; no other epoch's file. `allow_smoke_inputs` is for artifact_status smoke only: it
    accepts a smoke-mode run and a selection made from smoke inputs.

    Returns the carrier {spec, checkpoint_path, checkpoint_sha256, epoch, run_id, selection_sha256, telemetry_sha256,
    finalize_sha256 (None when absent)}, which src.eval.model_loading.load_qat_fakequant_off_student takes.
    """
    from src.quant import qat as Q
    from src.quant import qat_artifacts as A
    from src.quant import qat_select as S
    from src.quant.x86_latency import QAT_SIDECAR_ROLE, load_qat_sidecar

    spec = resolve_stage_artifact(stage)
    want = spec["stage"]
    sel_p = Path(selection) if selection is not None else None
    # 6. no path this function opens names "test", before any file is opened
    paths = {"--checkpoint": checkpoint, "--qat-run-dir": run_dir, "--qat-selection": selection,
             "the run's telemetry": None if run_dir is None else Path(run_dir) / Q.TELEMETRY_NAME,
             "the finalize record": None if sel_p is None else sel_p.parent / QAT_FINALIZE_NAME}
    for what, value in paths.items():
        if value is not None and Q.names_test(value):
            _fail("fqoff_forbidden_path", f"{what} {value} names 'test' in its given or resolved form; nothing was "
                                          "opened")
    # 7. the selection's bytes against the pin
    raw = sel_p.read_bytes() if sel_p is not None and sel_p.is_file() else None
    sel_sha = Q.sha256_bytes(raw) if raw is not None else None
    if sel_sha is None or sel_sha != selection_sha256:
        got = "is absent" if sel_sha is None else f"has sha256 {sel_sha}"
        _fail("fqoff_selection_sha256_mismatch", f"the selection {sel_p} {got}; its pin is {selection_sha256!r}")
    # 8. a selection of record: its name, format, rules, smoke inputs and winner
    if sel_p.name != QAT_SELECTION_NAME:
        _fail("fqoff_selection_not_of_record", f"{sel_p} is not named {QAT_SELECTION_NAME}: only "
                                               "scripts/select_qat_epoch.py's record binds")
    try:
        sel = json.loads(raw.decode("utf-8"), parse_constant=_no_json_constant)
    except ValueError:
        sel = None
    if not isinstance(sel, dict) or sel.get("format") != S.EPOCH_SELECTION_FORMAT:
        _fail("fqoff_selection_not_of_record", f"{sel_p} is not a {S.EPOCH_SELECTION_FORMAT} record")
    rules_p = S.rules_path_for_stage(sel.get("stage"))
    rules_sha = Q.sha256_file(rules_p)
    if sel.get("rules_sha256") != rules_sha:
        _fail("fqoff_selection_not_of_record", f"{sel_p} records rules_sha256 {sel.get('rules_sha256')!r}; the "
                                               f"committed {rules_p.name} of its stage {sel.get('stage')!r} has "
                                               f"{rules_sha}")
    if sel.get("smoke_inputs") is not False and not allow_smoke_inputs:
        _fail("fqoff_selection_not_of_record", f"{sel_p} records smoke_inputs {sel.get('smoke_inputs')!r}; only a "
                                               "smoke artifact binds a selection made from smoke inputs")
    win = sel.get("winner")
    epoch = win.get("epoch") if isinstance(win, dict) else None
    if not (type(epoch) is int and 1 <= epoch <= Q.EPOCHS and isinstance(win.get("checkpoint"), str)
            and isinstance(win.get("checkpoint_sha256"), str) and _HEX64.fullmatch(win["checkpoint_sha256"])):
        _fail("fqoff_selection_not_of_record", f"{sel_p}'s winner is malformed: {str(win)[:200]}")
    # 9. its stage is the requested one
    if sel.get("stage") != want:
        _fail("fqoff_stage_mismatch", f"the selection is of stage {sel.get('stage')!r}; {want} was requested")
    # 10-12. the run's record (AM-19): read strictly, complete, a real run, of the stage
    try:
        if run_dir is None:
            raise Q.QATRefused("run_dir_missing", "no run directory was given")
        rec = A.read_run_record(run_dir)
    except (Q.QATRefused, OSError, ValueError) as e:
        _fail("fqoff_run_record", f"the run's record is refused: [{getattr(e, 'code', type(e).__name__)}] {e}")
    try:
        A.require_complete(rec)
    except A.QATIncomplete as e:
        _fail("fqoff_run_incomplete", f"[{e.code}] {e}: an incomplete run has no model of record (AM-19)")
    meta = rec["run_meta"]
    if meta.get("mode") != "real" and not allow_smoke_inputs:
        _fail("fqoff_run_not_real", f"the run's mode is {meta.get('mode')!r}; only a real run is scored")
    if meta.get("stage") != want:
        _fail("fqoff_stage_mismatch", f"the run is of stage {meta.get('stage')!r}; {want} was requested")
    # 13. the telemetry against the pin and the selection; then the selection's run
    tel_sha = rec["telemetry_sha256"]
    if tel_sha != telemetry_sha256 or tel_sha != sel.get("telemetry_sha256"):
        _fail("fqoff_telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {tel_sha}; the pin is "
                                                 f"{telemetry_sha256!r}, the selection records "
                                                 f"{sel.get('telemetry_sha256')!r}")
    if sel.get("run_id") != rec["run_id"]:
        _fail("fqoff_selection_not_of_record", f"the selection names run {sel.get('run_id')!r}; the run is "
                                               f"{rec['run_id']!r}")
    # 14. the finalize record beside the selection: checked when present, never required (E6-KD has none)
    fin_p = sel_p.parent / QAT_FINALIZE_NAME
    fin_sha = None
    if fin_p.exists() or fin_p.is_symlink():
        fin_raw = fin_p.read_bytes() if fin_p.is_file() else None
        try:
            fin = None if fin_raw is None else json.loads(fin_raw.decode("utf-8"), parse_constant=_no_json_constant)
        except ValueError:
            fin = None
        aor = fin.get("artifact_of_record") if isinstance(fin, dict) else None
        disagree = ["unreadable"] if not isinstance(fin, dict) else [k for k, ok in (
            ("format", fin.get("format") == QAT_FINALIZE_FORMAT),
            ("status", fin.get("status") == "finalized"),
            ("qat_selection", fin.get("qat_selection") == {"path": QAT_SELECTION_NAME, "sha256": sel_sha}),
            ("artifact_of_record.checkpoint", isinstance(aor, dict) and aor.get("checkpoint")
             == {"path": win["checkpoint"], "sha256": win["checkpoint_sha256"]}),
            ("run_id", fin.get("run_id") == rec["run_id"]),
            ("telemetry_sha256", fin.get("telemetry_sha256") == tel_sha),
            ("stage", fin.get("stage") == want)) if not ok]
        if disagree:
            _fail("fqoff_finalize_disagrees", f"{fin_p} disagrees with the selection {sel_sha}: {disagree}")
        fin_sha = Q.sha256_bytes(fin_raw)
    # 15-17. the winner's file: its epoch_end row's name, the file, its sha256 against the row and the selection
    want_ck = f"{Q.EPOCH_DIR}/e{epoch:02d}.pt"
    if win["checkpoint"] != want_ck:
        _fail("fqoff_not_the_winner", f"the selection's winner.checkpoint is {win['checkpoint']!r}; epoch "
                                      f"{epoch:02d}'s file is {want_ck!r}")
    end = rec["ends"][epoch]
    ck = Path(checkpoint) if checkpoint is not None else None
    winner_file = (Path(rec["run_dir"]) / want_ck).resolve()
    if ck is None or end.get("checkpoint") != want_ck or ck.resolve() != winner_file:
        _fail("fqoff_not_the_winner", f"--checkpoint {checkpoint} is not the winner's {want_ck} of {rec['run_dir']} "
                                      f"(its epoch_end row names {end.get('checkpoint')!r})")
    if not ck.is_file():
        _fail("fqoff_checkpoint_missing", f"{ck} is missing")
    ck_sha = Q.sha256_file(ck)
    if ck_sha != end.get("checkpoint_sha256") or ck_sha != win["checkpoint_sha256"]:
        _fail("fqoff_checkpoint_sha256_mismatch", f"{ck} has sha256 {ck_sha}; its epoch_end row records "
                                                  f"{end.get('checkpoint_sha256')}, the selection's winner "
                                                  f"{win['checkpoint_sha256']}")
    # 18. a pre-convert QAT state in the x86 sidecar schema, never a converted TorchScript artifact
    if _is_torchscript_archive(ck):
        _fail("fqoff_not_qat_train_state", f"{ck} is a TorchScript archive (a converted artifact), not the epoch's "
                                           "pre-convert QAT state")
    try:
        payload = load_qat_sidecar(ck)
    except Exception as e:                                   # noqa: BLE001 -- any unreadable payload is refused
        _fail("fqoff_not_qat_train_state", f"{ck} is not a qat-train-state payload: "
                                           f"[{getattr(e, 'code', type(e).__name__)}] {str(e)[:300]}")
    if payload.get("artifact_role") != QAT_SIDECAR_ROLE:
        _fail("fqoff_not_qat_train_state", f"{ck} has artifact_role {payload.get('artifact_role')!r}, not "
                                           f"{QAT_SIDECAR_ROLE!r}")
    if payload.get("stage") != want:
        _fail("fqoff_stage_mismatch", f"{ck} records stage {payload.get('stage')!r}; {want} was requested")
    # 19. its identity against the run's (convert_epoch's check)
    ident = {"epoch": epoch, "run_id": rec["run_id"], "seed": meta.get("seed"),
             "source_checkpoint_sha256": meta.get("source_checkpoint_sha256")}
    wrong = {k: payload.get(k) for k, v in ident.items() if v is None or payload.get(k) != v}
    if wrong:
        _fail("fqoff_checkpoint_identity", f"{ck} records {wrong}; the run {ident}")
    # 20. the stored state flag: a finite state only (AM-21 item 1(c))
    if payload.get("state_finite") is not True:
        _fail("fqoff_state_not_finite", f"{ck} stores state_finite {payload.get('state_finite')!r}; only a finite "
                                        "state is scored")
    return {"spec": spec, "checkpoint_path": ck, "checkpoint_sha256": ck_sha, "epoch": epoch, "run_id": rec["run_id"],
            "selection_sha256": sel_sha, "telemetry_sha256": tel_sha, "finalize_sha256": fin_sha}


def expected_model_role(stage: str) -> str:
    """`teacher` for the teacher stage, `student` for E1-E7 and the arms A, F, G."""
    return TEACHER_ROLE if resolve_stage_artifact(stage)["stage"] == TEACHER_STAGE else STUDENT_ROLE


def is_descriptive_only(stage: str) -> bool:
    """True when the stage may never act as an inferential comparator (the teacher, A, F, G)."""
    return resolve_stage_artifact(stage)["stage"] in DESCRIPTIVE_ONLY_STAGES


def validate_teacher_artifact(checkpoint_path: str | Path, *,
                              expected_sha256: str | None = None) -> dict:
    """Validate a fine-tuned SegNeXt-B / MSCAN-B teacher checkpoint for clean evaluation.

    Delegates the structural parse to `src.distill.segnext_teacher.load_teacher_state_dict` — the
    existing teacher-checkpoint authority — rather than adding a second incompatible parser. That
    loader already refuses non-dict payloads, empty/tensor-free states, and anything lacking both
    `backbone.*` and `decode_head.*` keys, so an unrelated or ADE20K-only-shaped file cannot be
    silently substituted.

    R6: when `expected_sha256` is not None it must be 64 lowercase hex characters
    (`teacher_ckpt_sha256_format`, "" included), and the file is hashed once and compared BEFORE the
    parse (`teacher_hash_mismatch`); that hash is the recorded `checkpoint_sha256`.
    """
    from src.distill.segnext_teacher import TeacherCheckpointInvalid, load_teacher_state_dict
    from src.distill.teacher import sha256_format_error
    from src.eval.model_loading import sha256_file

    spec = resolve_stage_artifact(TEACHER_STAGE)
    if expected_sha256 is not None:
        error = sha256_format_error(expected_sha256)
        if error is not None:
            _fail("teacher_ckpt_sha256_format", f"expected teacher sha256 {error}")
    p = Path(checkpoint_path)
    if not p.is_file():
        _fail("teacher_checkpoint_missing", f"teacher checkpoint not found: {p}")
    digest = sha256_file(p)
    if expected_sha256 is not None and digest != expected_sha256:
        _fail("teacher_hash_mismatch",
              f"teacher checkpoint hash {digest} does not match the expected {expected_sha256}; the "
              "file was not parsed")
    try:
        state = load_teacher_state_dict(p)
    except TeacherCheckpointInvalid as e:
        _fail("teacher_checkpoint_invalid", f"teacher checkpoint rejected: {e}")

    # A student or quantized artifact must never pass as the teacher.
    if any(k.startswith(("features.", "head.")) for k in state):
        _fail("teacher_is_student_checkpoint",
              "this is a PlantSegStudent checkpoint (features./head. keys), not a SegNeXt teacher")
    if any("_packed_params" in k or "activation_post_process" in k for k in state):
        _fail("teacher_is_quantized_artifact",
              "this is a quantized artifact; the teacher is evaluated in FP32")
    return {"spec": spec, "checkpoint_path": p, "checkpoint_sha256": digest,
            "state_keys": len(state), "descriptive_only": True}


def resolve_evaluation_source(stage: str, *, checkpoint: str | Path | None = None,
                              provenance: str | Path | None = None,
                              expected_sha256: str | None = None,
                              qat_run_dir: str | Path | None = None, qat_selection: str | Path | None = None,
                              qat_selection_sha256: str | None = None, qat_telemetry_sha256: str | None = None,
                              allow_smoke_inputs: bool = False) -> dict:
    """Single entry point: route a stage to the right validator and reject the wrong artifact kind.

    Any qat_* argument routes to validate_qat_fakequant_off_source (precision qat_fakequant_off, lane L-FQOFF): the
    selected epoch's pre-convert QAT state at `checkpoint`, bound through its selection; `allow_smoke_inputs` applies to
    that route only. A call without them routes as before.
    """
    if any(v is not None for v in (qat_run_dir, qat_selection, qat_selection_sha256, qat_telemetry_sha256)):
        return validate_qat_fakequant_off_source(stage, checkpoint, run_dir=qat_run_dir, selection=qat_selection,
                                                 selection_sha256=qat_selection_sha256,
                                                 telemetry_sha256=qat_telemetry_sha256,
                                                 allow_smoke_inputs=allow_smoke_inputs)
    spec = resolve_stage_artifact(stage)
    if spec["kind"] == "teacher_checkpoint":
        if checkpoint is None:
            _fail("checkpoint_required", "the teacher stage requires its fine-tuned checkpoint")
        return validate_teacher_artifact(checkpoint, expected_sha256=expected_sha256)
    if spec["kind"] == "int8_artifact":
        if provenance is None:
            _fail("provenance_required",
                  f"{spec['stage']} is an INT8 stage: pass its run-provenance JSON, not a raw "
                  "checkpoint")
        return validate_int8_artifact(stage, provenance)
    if checkpoint is None:
        _fail("checkpoint_required", f"{spec['stage']} is an FP32 stage: pass its checkpoint")
    return validate_fp32_artifact(stage, checkpoint)


# ------------------------------------------------------------------ official preconditions
def official_launch_error(*, split: str, random_init: bool, stage: str,
                          expected_manifest_rows: int, max_samples: int | None = None) -> str | None:
    """PRE-RUN only: the official-launch facts that genuinely exist before inference.

    PHASE SEPARATION, deliberate. This function may assert only what is true before a model or
    dataset exists: the request is not random-init, it targets the test split, it is uncapped, and
    the CANONICAL EXPECTED manifest holds 1561 entries. It must NOT assert `actual_rows`, because no
    row has been produced yet.

    The post-inference rules — `actual_rows == expected_rows`, manifest/ID/order integrity — are
    already owned end-to-end by `src/eval/artifacts.py` (`actual_rows` is computed there from
    `len(result.rows)` and enforced against `expected_rows` before an artifact is finalised). They
    are therefore NOT duplicated here; a second authority for the same rule would be a liability.
    """
    resolve_stage_artifact(stage)
    if random_init:
        return ("artifact_status='official' forbids random_init=True (contract 5.3: random_init "
                "requires artifact_status='smoke')")
    if split != OFFICIAL_SPLIT:
        return f"artifact_status='official' requires split='{OFFICIAL_SPLIT}', got {split!r}"
    if max_samples is not None:
        return ("artifact_status='official' forbids a sample cap; the full official test split is "
                f"required (--max-samples={max_samples})")
    if expected_manifest_rows != OFFICIAL_ROWS:
        return (f"the canonical expected test manifest must hold {OFFICIAL_ROWS} entries, got "
                f"{expected_manifest_rows}")
    return None


def governed_paths_error(repo: Path | None = None) -> str | None:
    """Delegate the section 7.1 governed-path gate to the frozen implementation."""
    from src.eval.artifacts import GOVERNED_PREFIXES, git_porcelain_bytes, parse_porcelain_paths

    root = Path(repo) if repo else Path(__file__).resolve().parents[2]
    paths = parse_porcelain_paths(git_porcelain_bytes(root))
    violations = sorted(p for p in paths if p.startswith(tuple(GOVERNED_PREFIXES)))
    if violations:
        return ("refusing 'official': governed paths are dirty or untracked -> "
                f"{violations}. Non-governed paths (including the protected reference PDF) are "
                "allowlisted and do not block.")
    return None
