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

# Stage -> artifact contract (IMPLEMENTATION_CONTRACT B4 stage table).
STAGE_ARTIFACTS: dict[str, dict] = {
    "E1": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
    "E2": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
    "E3": {"kind": "fp32_checkpoint", "precision": "fp32", "source_stage": None, "method": None},
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
TEACHER_STAGE = "TEACHER"
STUDENT_ROLE, TEACHER_ROLE = "student", "teacher"
DESCRIPTIVE_ONLY_STAGES = ("TEACHER",)


class StageArtifactError(RuntimeError):
    """A stage/artifact mismatch or a failed official precondition. `code` keeps gates testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str):
    raise StageArtifactError(code, message)


def resolve_stage_artifact(stage: str) -> dict:
    """Return the artifact contract for a stage. Teacher is deliberately absent.

    The existing evaluator rejects `model_role='teacher'`; inventing teacher-artifact behaviour here
    would misrepresent runtime support, so unknown stages fail loudly instead.
    """
    key = str(stage).upper()
    if key not in STAGE_ARTIFACTS:
        _fail("unknown_stage",
              f"no artifact contract for stage {stage!r}; known stages: {sorted(STAGE_ARTIFACTS)} "
              "(teacher evaluation is not implemented by the evaluator and is not invented here)")
    return {"stage": key, **STAGE_ARTIFACTS[key]}


# ------------------------------------------------------------------ INT8 (E4-E7)
def validate_int8_artifact(stage: str, provenance_path: str | Path) -> dict:
    """Validate an E4-E7 converted INT8 artifact through its runner provenance.

    Checks stage identity, source stage, random_init, class count, artifact existence, the recorded
    SHA-256 against the file on disk, the official backend, and — for E6/E7 — that the training-only
    CWD projection was never loaded.
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

    art = prov.get("converted_artifact")
    if not art:
        _fail("artifact_path_missing", "provenance records no converted_artifact")
    art_path = Path(art)
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
            "artifact_sha256": actual}


# ------------------------------------------------------------------ FP32 (E1-E3)
def validate_fp32_artifact(stage: str, checkpoint_path: str | Path) -> dict:
    """Validate an E1/E2/E3 FP32 checkpoint. Uses the existing checkpoint contract, not a new one."""
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
    if declared is not None and str(declared).upper() != spec["stage"]:
        _fail("stage_mismatch",
              f"checkpoint is stage {declared!r}, requested {spec['stage']}")
    if spec["stage"] == "E3":
        from src.distill.export import assert_clean_student_state
        assert_clean_student_state(state)          # projection must be absent from the student
    if ckpt.get("num_classes") is not None and int(ckpt["num_classes"]) != NUM_CLASSES:
        _fail("num_classes_mismatch",
              f"checkpoint declares num_classes={ckpt['num_classes']}, expected {NUM_CLASSES}")
    return {"spec": spec, "checkpoint_path": p, "checkpoint_sha256": sha256_file(p),
            "declared_stage": declared}


def expected_model_role(stage: str) -> str:
    """`teacher` for the teacher stage, `student` for E1-E7."""
    return TEACHER_ROLE if resolve_stage_artifact(stage)["stage"] == TEACHER_STAGE else STUDENT_ROLE


def is_descriptive_only(stage: str) -> bool:
    """True when the stage may never act as an inferential comparator (teacher)."""
    return resolve_stage_artifact(stage)["stage"] in DESCRIPTIVE_ONLY_STAGES


def validate_teacher_artifact(checkpoint_path: str | Path, *,
                              expected_sha256: str | None = None) -> dict:
    """Validate a fine-tuned SegNeXt-B / MSCAN-B teacher checkpoint for clean evaluation.

    Delegates the structural parse to `src.distill.segnext_teacher.load_teacher_state_dict` — the
    existing teacher-checkpoint authority — rather than adding a second incompatible parser. That
    loader already refuses non-dict payloads, empty/tensor-free states, and anything lacking both
    `backbone.*` and `decode_head.*` keys, so an unrelated or ADE20K-only-shaped file cannot be
    silently substituted.
    """
    from src.distill.segnext_teacher import TeacherCheckpointInvalid, load_teacher_state_dict
    from src.eval.model_loading import sha256_file

    spec = resolve_stage_artifact(TEACHER_STAGE)
    p = Path(checkpoint_path)
    if not p.is_file():
        _fail("teacher_checkpoint_missing", f"teacher checkpoint not found: {p}")
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

    digest = sha256_file(p)
    if expected_sha256 and digest != expected_sha256:
        _fail("teacher_hash_mismatch",
              f"teacher checkpoint hash {digest[:16]}… does not match the expected "
              f"{str(expected_sha256)[:16]}…")
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
