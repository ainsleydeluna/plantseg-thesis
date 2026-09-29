"""FP32 student construction and strict repository-checkpoint loading (A2b).

Separated from `adapters.py` so dataset identity and model/checkpoint concerns stay independent:
a future teacher / PTQ / QAT adapter replaces this module without touching the dataset layer, and
vice versa.

Only the ONE documented repository checkpoint schema is accepted -- the nested dict written by
`src/training/train_e1.py::save_checkpoint`:

    {"iter", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
     "scheduler", "best_val_miou_all_class", "num_classes"}

Bare state dicts and any other undocumented shape are rejected loudly rather than silently
supported.

Import-time behaviour is side-effect free: no model is constructed and no file is read on import.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..models.student import build_student

FROZEN_NUM_CLASSES = 116

# `torch.load` weights_only policy (torch 2.9.1 reports a default of None, i.e. version-dependent
# resolution). We therefore pass it EXPLICITLY rather than inheriting a shifting library default.
#
# weights_only=True is both the safest option and fully compatible with the repository schema:
# every value in the nested checkpoint is a plain int/float/str, a dict/list thereof, or a
# torch.Tensor -- all of which the weights_only unpickler allows. No custom classes, no
# numpy scalars, no lambdas are stored. If a future checkpoint needs a richer type, that is a
# deliberate schema change and must be handled explicitly, not by relaxing this flag.
TORCH_LOAD_WEIGHTS_ONLY = True

REQUIRED_KEYS = ("model_state_dict", "num_classes")


class CheckpointError(RuntimeError):
    """Any rejected checkpoint. Always fatal -- never degrade to a partial load."""


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata the artifact request needs. `path` is caller-supplied; normalise before recording."""
    path: str
    sha256: str
    num_classes: int
    iteration: int | None
    best_val_miou_all_class: float | None


def sha256_file(path: Path) -> str:
    """SHA-256 over the exact raw checkpoint bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_fp32_student(num_classes: int = FROZEN_NUM_CLASSES, device: str = "cpu"):
    """FP32 student via the repository factory. No download, no architecture duplication.

    `pretrained=False` is forced: ImageNet init belongs to training, not evaluation, and would
    otherwise reach for the torch-hub cache or the network.
    """
    if num_classes != FROZEN_NUM_CLASSES:
        raise CheckpointError(
            f"frozen contract requires num_classes={FROZEN_NUM_CLASSES}, got {num_classes}")
    model = build_student(num_classes=num_classes, pretrained=False)
    if getattr(model, "used_pretrained", False):
        raise CheckpointError("build_student unexpectedly loaded pretrained weights")
    return model.to(torch.device(device)).eval()


# ---------------------------------------------------------------------------------------------
# INT8 (E4-E7) -- two artifact formats, named by the run provenance's `artifact_format`
#
# "state_dict" (the key is absent from src/quant/runner.py provenance): the EXACT runner schema
#
#     {"stage": str, "quantization": "ptq"|"qat", "num_classes": 116, "model": <converted
#      state_dict>}
#
# That is a QUANTIZED STATE DICT, not a serialized module, so evaluation rebuilds the converted
# skeleton with the committed quantization helpers and loads the weights strictly into it. Nothing is
# recalibrated and no QAT step is run.
#
# "torchscript" (src/quant/ptq.py, lane 8 / L-AM10): the QNNPACK TorchScript module of record, traced
# from the converted model, carrying its identity as the extra file `plantseg_int8.json`. Its converted
# state_dict is written alongside it in the runner schema above, and src/quant/ptq.py proves the two
# give bitwise-equal outputs before the run provenance exists.
# ---------------------------------------------------------------------------------------------
INT8_REQUIRED_KEYS = ("stage", "quantization", "num_classes", "model")
INT8_METHODS = ("ptq", "qat")
OFFICIAL_QUANT_BACKEND = "qnnpack"
INT8_FORMAT_STATE_DICT = "state_dict"
INT8_FORMAT_TORCHSCRIPT = "torchscript"
INT8_ARTIFACT_FORMATS = (INT8_FORMAT_STATE_DICT, INT8_FORMAT_TORCHSCRIPT)
INT8_TORCHSCRIPT_META = "plantseg_int8.json"            # the extra file inside the TorchScript zip
INT8_TORCHSCRIPT_SCHEMA = "plantseg-int8-torchscript/1.0.0"
INT8_TORCHSCRIPT_META_KEYS = ("schema", "stage", "quantization", "num_classes", "engine",
                              "artifact_role")
ACCURACY_ARTIFACT_ROLE = "accuracy"                     # = src.quant.x86_latency.ACCURACY_ARTIFACT_ROLE


def int8_artifact_format(provenance: dict) -> str:
    """The artifact format a run provenance declares; the runner's provenance (no key) is a state_dict."""
    fmt = provenance.get("artifact_format", INT8_FORMAT_STATE_DICT)
    if fmt not in INT8_ARTIFACT_FORMATS:
        raise CheckpointError(
            f"unknown INT8 artifact_format {fmt!r}; known formats: {list(INT8_ARTIFACT_FORMATS)}")
    return fmt


@dataclass(frozen=True)
class Int8ArtifactInfo:
    """Metadata for a loaded converted INT8 student."""
    path: str
    sha256: str
    stage: str
    method: str
    num_classes: int
    backend: str


def select_int8_backend(*, require_qnnpack: bool = True) -> str:
    """Select the quantized engine for INT8 evaluation.

    The official E4-E7 accuracy artifact is the QNNPACK one, so `require_qnnpack=True` (the only
    value the evaluator uses) fails loudly when QNNPACK is absent rather than silently evaluating on
    a different backend. `require_qnnpack=False` exists solely for NON-OFFICIAL structural proxy
    tests and never reaches the evaluator's official path.
    """
    supported = list(torch.backends.quantized.supported_engines)
    if OFFICIAL_QUANT_BACKEND in supported:
        torch.backends.quantized.engine = OFFICIAL_QUANT_BACKEND
    elif require_qnnpack:
        raise CheckpointError(
            f"INT8 evaluation requires the {OFFICIAL_QUANT_BACKEND!r} backend, which this build "
            f"does not provide (supported_engines={supported}). Evaluating the official INT8 "
            "artifact on another backend would change the artifact and is refused.")
    if torch.backends.quantized.engine != OFFICIAL_QUANT_BACKEND and require_qnnpack:
        raise CheckpointError("could not select the qnnpack quantized engine")
    return torch.backends.quantized.engine


def rebuild_int8_from_state_dict(path, *, stage: str, method: str):
    """Rebuild a converted INT8 student from a runner-schema state_dict file. Returns it in eval mode.

    The caller has already selected the quantized engine and established trust in the file (the
    evaluator re-verifies its SHA-256 against the run provenance; src/quant/ptq.py re-reads the file
    it has just written and hashed).
    """
    from ..quant.prepare import convert_model, prepare_ptq, prepare_qat_model

    if method not in INT8_METHODS:
        raise CheckpointError(f"unsupported quantization method {method!r}")
    # weights_only=False is required here: a converted state_dict carries quantized packed-param
    # objects that the weights_only unpickler rejects. The file is trusted because the bridge just
    # re-verified its SHA-256 against the value recorded when the runner produced it.
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise CheckpointError(f"INT8 artifact must be a dict, got {type(obj).__name__}")
    missing = [k for k in INT8_REQUIRED_KEYS if k not in obj]
    if missing:
        raise CheckpointError(
            f"INT8 artifact is missing required key(s) {missing}; present keys: {sorted(obj)}. "
            "The evaluator accepts only the schema written by src/quant/runner.py.")
    if obj["quantization"] != method:
        raise CheckpointError(
            f"artifact records quantization={obj['quantization']!r} but {stage} is {method}")
    if str(obj["stage"]).upper() != stage:
        raise CheckpointError(
            f"artifact records stage={obj['stage']!r}, requested {stage}")
    if int(obj["num_classes"]) != FROZEN_NUM_CLASSES:
        raise CheckpointError(
            f"artifact num_classes={obj['num_classes']!r} != required {FROZEN_NUM_CLASSES}")
    if not isinstance(obj["model"], dict) or not obj["model"]:
        raise CheckpointError("INT8 artifact 'model' must be a non-empty converted state_dict")

    # Rebuild the converted skeleton the SAME way the runner produced it, without calibration or
    # training, then load the quantized weights strictly.
    skeleton = build_student(num_classes=FROZEN_NUM_CLASSES, pretrained=False)
    prepared = (prepare_ptq(skeleton, select_backend=False) if method == "ptq"
                else prepare_qat_model(skeleton, select_backend=False))
    converted = convert_model(prepared)
    try:
        converted.load_state_dict(obj["model"], strict=True)
    except Exception as e:                                    # noqa: BLE001 -- surfaced verbatim
        raise CheckpointError(
            f"strict INT8 state_dict load failed: {type(e).__name__}: {str(e)[:300]}") from e

    if getattr(converted, "num_classes", FROZEN_NUM_CLASSES) != FROZEN_NUM_CLASSES:
        raise CheckpointError(
            f"converted model declares {converted.num_classes} classes, expected "
            f"{FROZEN_NUM_CLASSES}")
    if any("cwd" in k.lower() for k in obj["model"]):
        raise CheckpointError(
            "converted INT8 artifact contains CWD projection tensors; the training-only projection "
            "must be absent from every evaluated model")
    return converted.eval()


def load_int8_torchscript(path, *, stage: str, method: str, engine: str = OFFICIAL_QUANT_BACKEND,
                          artifact_role: str = ACCURACY_ARTIFACT_ROLE):
    """Load an INT8 TorchScript module written by src/quant/ptq.py and verify its embedded identity.

    `engine` must already be the active quantized engine: TorchScript re-packs quantized weights for
    the active engine while it loads, so loading under another engine would build a different
    runtime model. The embedded `plantseg_int8.json` must name exactly this stage, method, class
    count, engine and artifact role. Returns `(module in eval mode, embedded metadata)`.
    """
    if method not in INT8_METHODS:
        raise CheckpointError(f"unsupported quantization method {method!r}")
    active = torch.backends.quantized.engine
    if active != engine:
        raise CheckpointError(
            f"the quantized engine is {active!r}; select {engine!r} before loading an INT8 "
            "TorchScript artifact (its weights are re-packed for the active engine at load time)")
    extra = {INT8_TORCHSCRIPT_META: ""}
    try:
        module = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
    except Exception as e:                                    # noqa: BLE001 -- surfaced verbatim
        raise CheckpointError(
            f"INT8 TorchScript load failed for {path}: {type(e).__name__}: {str(e)[:300]}") from e
    raw = extra[INT8_TORCHSCRIPT_META]
    if not raw:
        raise CheckpointError(
            f"{path} carries no {INT8_TORCHSCRIPT_META}; only TorchScript artifacts written by "
            "src/quant/ptq.py are accepted")
    try:
        meta = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except ValueError as e:
        raise CheckpointError(f"{path}: {INT8_TORCHSCRIPT_META} is not JSON ({e})") from e
    missing = [k for k in INT8_TORCHSCRIPT_META_KEYS if k not in meta]
    if missing:
        raise CheckpointError(f"{path}: {INT8_TORCHSCRIPT_META} lacks {missing}")
    expected = {"schema": INT8_TORCHSCRIPT_SCHEMA, "stage": stage, "quantization": method,
                "num_classes": FROZEN_NUM_CLASSES, "engine": engine,
                "artifact_role": artifact_role}
    wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        raise CheckpointError(
            f"{path}: embedded identity {wrong} does not match the expected "
            f"{ {k: expected[k] for k in wrong} }")
    return module.eval(), meta


def load_int8_student(resolved: dict, *, require_qnnpack: bool = True,
                      map_location: str = "cpu"):
    """Load a converted E4-E7 INT8 student from an ALREADY-VALIDATED resolved artifact.

    `resolved` must come from `src.eval.stage_artifacts.validate_int8_artifact`, which has already
    proven stage/source/method/class-count/backend provenance AND re-verified the artifact's
    SHA-256 against the file on disk — that hash check is the trust anchor for reading this file.
    The provenance's `artifact_format` selects the reader: the runner's state_dict (rebuilt into the
    converted skeleton) or the QNNPACK TorchScript module of src/quant/ptq.py. This function never
    chooses a stage of its own, never downloads, never calibrates and never trains. Returns
    `(model, Int8ArtifactInfo)` with the model in eval mode on CPU.
    """
    spec, prov = resolved["spec"], resolved["provenance"]
    path = Path(resolved["artifact_path"])
    method = spec["method"]
    if method not in INT8_METHODS:
        raise CheckpointError(f"unsupported quantization method {method!r}")
    if str(map_location) != "cpu":
        raise CheckpointError(
            f"converted eager INT8 models are CPU-only; got map_location={map_location!r}")
    fmt = int8_artifact_format(prov)

    backend = select_int8_backend(require_qnnpack=require_qnnpack)

    if fmt == INT8_FORMAT_TORCHSCRIPT:
        model, _meta = load_int8_torchscript(path, stage=spec["stage"], method=method,
                                             engine=OFFICIAL_QUANT_BACKEND)
    else:
        model = rebuild_int8_from_state_dict(path, stage=spec["stage"], method=method)

    info = Int8ArtifactInfo(path=str(path), sha256=resolved["artifact_sha256"],
                            stage=spec["stage"], method=method,
                            num_classes=FROZEN_NUM_CLASSES,
                            backend=str(prov.get("backend", backend)))
    return model, info


# ---------------------------------------------------------------------------------------------
# Teacher (descriptive reference) -- evaluation-side adapter
# ---------------------------------------------------------------------------------------------
class TeacherEvalModel(torch.nn.Module):
    """Expose a `FrozenTeacher` to the evaluator as a plain segmentation-logits callable.

    The distillation-side `FrozenTeacher` returns a `TeacherOutput(logits, feat_s16)` because KD and
    CWD need both. The evaluator needs neither that container nor the stride-16 feature, so this
    thin EVALUATION-SIDE wrapper takes only `.logits` and never surfaces `feat_s16`. Neither
    `FrozenTeacher` nor `SegNeXtTeacherAdapter` was modified for evaluation convenience.

    Teacher logits are produced at the head's native resolution; `FrozenTeacher` resamples them
    bilinearly onto the requested `logits_size`, which is the ordinary segmentation-logit
    upsampling used for MMSeg inference. No CWD/feature-alignment logic is involved.
    """

    def __init__(self, frozen_teacher, num_classes: int = FROZEN_NUM_CLASSES):
        super().__init__()
        self.teacher = frozen_teacher
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.teacher(x, logits_size=tuple(x.shape[-2:]))   # feat_size omitted on purpose
        logits = out.logits
        if logits.dim() != 4 or logits.shape[1] != self.num_classes:
            raise CheckpointError(
                f"teacher produced logits with shape {tuple(logits.shape)}; expected "
                f"[B, {self.num_classes}, H, W]")
        if logits.shape[-2:] != x.shape[-2:]:
            raise CheckpointError(
                f"teacher logits {tuple(logits.shape[-2:])} do not match the evaluation canvas "
                f"{tuple(x.shape[-2:])}")
        return logits


def load_teacher_model(resolved: dict, *, builder=None, config_path: str | None = None,
                       num_classes: int = FROZEN_NUM_CLASSES):
    """Build the frozen teacher from an ALREADY-VALIDATED resolved teacher artifact.

    `resolved` comes from `src.eval.stage_artifacts.validate_teacher_artifact`. `builder` is the
    injectable teacher factory used by synthetic tests; when omitted the real MMSeg path is used,
    which REQUIRES `config_path` (the thesis teacher config) and raises `TeacherStackMissing` loudly if
    the teacher environment is absent. Nothing here downloads or substitutes a model, and a random
    teacher is impossible: a validated checkpoint is required.

    M4-V (B61 §4): one load serves ONE complete evaluation pass. The teacher's private NMF stream is
    seeded 42 here, before the first forward; the caller runs the whole split once, batch size 1, in
    the frozen manifest order. Only the NMF basis draw consumes the stream; the caller's CPU RNG and
    every CUDA generator are untouched. The stream description is exposed as `model.nmf_policy`.
    """
    from ..distill.nmf_stream import M4_NMF_SEED
    from ..distill.teacher import load_frozen_teacher

    if builder is None and not config_path:
        raise CheckpointError("the real teacher evaluation requires the thesis teacher config "
                              "(--teacher-config): its IsolatedNMFLightHamHead implements M4-V")
    frozen = load_frozen_teacher(str(resolved["checkpoint_path"]), builder=builder,
                                 config_path=config_path)
    if frozen.trainable_parameters():
        raise CheckpointError("teacher exposes trainable parameters; it must be frozen")
    nmf = frozen.begin_nmf_stream("M4-V", M4_NMF_SEED)
    if builder is None and nmf is None:
        raise CheckpointError("M4-V: the real teacher exposes no isolated NMF stream")
    model = TeacherEvalModel(frozen, num_classes=num_classes).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.nmf_policy = nmf
    return model, resolved


def _looks_like_bare_state_dict(obj) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    return all(isinstance(k, str) for k in obj) and \
        all(isinstance(v, torch.Tensor) for v in obj.values())


def load_student_checkpoint(path, *, map_location: str = "cpu",
                            num_classes: int = FROZEN_NUM_CLASSES):
    """Strictly load the repository's nested FP32-student checkpoint.

    Returns `(model, CheckpointInfo)`. Raises CheckpointError on any deviation.
    """
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"checkpoint file not found: {p}")
    digest = sha256_file(p)

    obj = torch.load(p, map_location=map_location, weights_only=TORCH_LOAD_WEIGHTS_ONLY)

    if _looks_like_bare_state_dict(obj):
        raise CheckpointError(
            "this is a BARE state_dict (all values are tensors). The repository schema is a "
            f"nested dict containing {list(REQUIRED_KEYS)}; bare state dicts are rejected so an "
            "unverifiable class count can never be assumed.")
    if not isinstance(obj, dict):
        raise CheckpointError(
            f"checkpoint must be a nested dict, got {type(obj).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise CheckpointError(
            f"checkpoint is missing required key(s) {missing}; present keys: {sorted(obj)}")

    ck_classes = obj["num_classes"]
    if not isinstance(ck_classes, int) or ck_classes != FROZEN_NUM_CLASSES:
        raise CheckpointError(
            f"checkpoint num_classes={ck_classes!r} != required {FROZEN_NUM_CLASSES}")
    if num_classes != FROZEN_NUM_CLASSES:
        raise CheckpointError(f"requested num_classes={num_classes} != {FROZEN_NUM_CLASSES}")

    model = build_fp32_student(num_classes=FROZEN_NUM_CLASSES, device=map_location)
    try:
        model.load_state_dict(obj["model_state_dict"], strict=True)
    except Exception as e:                                    # noqa: BLE001 -- surfaced verbatim
        raise CheckpointError(
            f"strict state_dict load failed: {type(e).__name__}: {str(e)[:300]}") from e

    info = CheckpointInfo(
        path=str(p),
        sha256=digest,
        num_classes=FROZEN_NUM_CLASSES,
        iteration=obj.get("iter") if isinstance(obj.get("iter"), int) else None,
        best_val_miou_all_class=(float(obj["best_val_miou_all_class"])
                                 if isinstance(obj.get("best_val_miou_all_class"), (int, float))
                                 else None),
    )
    return model.eval(), info
