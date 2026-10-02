"""Bridge between produced E1-E7 artifacts and the existing evaluator. Resolver + gates only.

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

For E4-E7 the input is the run-provenance JSON written by `src/quant/runner.py`; for E1-E3 it is the
FP32 checkpoint read through the existing `src.eval.model_loading` contract.
"""

from __future__ import annotations

import json
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
    # The frozen schema (EVALUATION_CONTRACT 5.x) already admits stage/model_role "teacher" at fp32.
    # The teacher is a DESCRIPTIVE REFERENCE: its clean artifact may legitimately be `official`
    # (nothing in the committed contract forbids that), but it never becomes an inferential
    # comparator — that restriction lives in the statistics stage, not in artifact status.
    "TEACHER": {"kind": "teacher_checkpoint", "precision": "fp32", "source_stage": None,
                "method": None},
}
PROJECTION_FREE_STAGES = ("E6", "E7")
# FP32 checkpoints train_distill writes, which record their stage (DL-52): refused when they record none.
# E1 (train_e1) records none and is not listed.
STAGE_DECLARING_FP32_STAGES = ("E2", "E3", "A", "F", "G")
# FP32 stages whose student state must hold no training-only CWD projection key.
PROJECTION_FREE_FP32_STAGES = ("E3", "A", "F", "G")
TEACHER_STAGE = "TEACHER"
STUDENT_ROLE, TEACHER_ROLE = "student", "teacher"
# Never a comparator in an inferential test. Nothing in the evaluator enforces this; the statistics
# driver must refuse these stages as comparators.
DESCRIPTIVE_ONLY_STAGES = ("TEACHER", "A", "F", "G")


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
                              expected_sha256: str | None = None) -> dict:
    """Single entry point: route a stage to the right validator and reject the wrong artifact kind."""
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
