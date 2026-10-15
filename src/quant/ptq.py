"""Lane 8 (L-AM10): the static INT8 PTQ of record for E4 (from E1) and E7 (from E3).

    FP32 source checkpoint + a calibration list (configs/calibration/*.json)
      -> fuse -> the registered PTQ qconfig (src/quant/qconfig.py: histogram UINT8 activations,
         per-channel symmetric INT8 weights) -> observers
      -> calibrate on the list's 128 TRAIN images in list order, canvas preprocessing (the clean-test
         core_preprocess), no augmentation, ONE image per mini-batch (AM-10)
      -> convert under QNNPACK
      -> the artifact of record: the QNNPACK TorchScript module, traced at 1x3x512x512 and carrying its
         identity as the extra file `plantseg_int8.json`; the converted state_dict alongside it in the
         src/quant/runner.py schema that src/eval/model_loading.py rebuilds
      -> the x86 latency copy (IMPLEMENTATION_CONTRACT: "separate fbgemm/x86 INT8 copy
         (reduce_range=True)"): the same images in the same order through the x86 PTQ qconfig,
         converted under the x86 engine by src/quant/x86_latency.py

Before a run provenance exists, every artifact is re-read from disk and checked (lane 8 (f) STOPs):
  * weight scheme: every quantized conv weight, in the eager model and inside each TorchScript file, is
    per-channel symmetric INT8 on axis 0 with zero points 0. A per-tensor weight is a STOP;
  * census: at run time every executed module takes and returns quantized tensors except the declared
    input Quantize and output DeQuantize, and the TorchScript graph holds exactly one
    quantize_per_tensor, fed by the input, and one dequantize, feeding only the final float upsample.
    An undeclared float region is a STOP;
  * parity: the TorchScript artifact and the state_dict rebuild, both re-read from disk, give
    bitwise-equal outputs on the 128 calibration images and fixed synthetic inputs; the TorchScript
    artifact equals the in-memory model that was calibrated on the synthetic inputs; the x86
    TorchScript copy equals its in-memory model on the synthetic inputs and the first 16 calibration
    images. A mismatch is a STOP.

Nothing here reads VAL or TEST: calibration reads images/train and annotations/train only, and any data
path naming TEST or VAL is refused before an image is opened.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.quant import QUANT                                         # noqa: E402
from src.eval.model_loading import (ACCURACY_ARTIFACT_ROLE, INT8_TORCHSCRIPT_META,  # noqa: E402
                                    INT8_TORCHSCRIPT_SCHEMA, load_int8_torchscript,
                                    rebuild_int8_from_state_dict)
from .calibration import (CALIBRATION_COUNT, CALIBRATION_SCHEMA, CALIBRATION_SPLIT,  # noqa: E402
                          CalibrationIndexError, verify_calibration_index)
from .prepare import CALIBRATION_BATCH_SIZE, calibrate, convert_model, prepare_ptq  # noqa: E402
from .qconfig import (QUANT_BACKEND, QuantBackendUnavailable, X86_BACKENDS,  # noqa: E402
                      describe_qconfig, ptq_qconfig, select_qnnpack_backend, select_x86_backend,
                      x86_ptq_qconfig)
from .x86_latency import X86_LATENCY_ARTIFACT_ROLE, build_x86_latency_copy  # noqa: E402

PTQ = QUANT["ptq"]
NUM_CLASSES = 116
if int(PTQ["calibration_batch_size"]) != CALIBRATION_BATCH_SIZE:
    raise RuntimeError(f"configs/quant.py calibration_batch_size={PTQ['calibration_batch_size']!r} "
                       f"disagrees with src/quant/prepare.py ({CALIBRATION_BATCH_SIZE}); AM-10 fixes 1")

# ------------------------------------------------------------------ calibration lists (AM-10, AM-16)
SEED_OF_RECORD = int(PTQ["calibration_seed"])                                  # 42, AM-10
SENSITIVITY_SEEDS = tuple(int(s) for s in PTQ["calibration_sensitivity_seeds"])  # 43-45, AM-16 item 5
LIST_SEEDS = (SEED_OF_RECORD,) + SENSITIVITY_SEEDS
ROLE_OF_RECORD = "am10_list_of_record"
ROLE_SENSITIVITY = "am16_calibration_sensitivity"
LIST_SCHEMA = "plantseg-ptq-calibration-list/1.0.0"
LIST_FILENAME = "ptq_calibration_seed{seed}.json"      # under configs/calibration/ once committed
LIST_STATUS_REGISTERED = "registered"                 # built from the 5,367-image TRAIN split list
LIST_SHARED_BY = ["E4", "E7"]
LIST_PROCEDURE = ("random.Random(seed).sample(sorted(train_ids), 128) "
                  "(src/quant/calibration.py build_calibration_index)")
CHECKSUM_DEFINITION = ('sha256 of the selected ids in draw order joined by "\\n" (no trailing newline), '
                       "UTF-8 (src/quant/calibration.py _checksum)")
# Identical to scripts/build_train_strata.py (L-AM17-STRATA), so a list and the strata file that
# record the same split_list_sha256 were drawn from the same TRAIN listing.
SPLIT_LIST_DEFINITION = ("sha256 of the name-sorted TRAIN image stems (images/train, .jpg/.jpeg), one "
                         "per line, UTF-8, LF, trailing LF; each stem's mask is "
                         "annotations/train/<stem>.png")
CALIBRATION_ORDER = "selected_ids order (the draw order), never re-sorted"
PREPROCESSING = ("core_preprocess/1.0.0 canvas (EXIF transpose, long side -> 512, pad to 512x512, "
                 "ImageNet normalisation), the clean-test preprocessing; no augmentation")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

# ------------------------------------------------------------------ artifacts
RUN_META_SCHEMA = "plantseg-ptq-run-meta/1.0.0"
LANE = "L-AM10 (docs/lane_specs/part2.md lane 8)"
EXAMPLE_SHAPE = (1, 3, 512, 512)                      # the canvas of record; the trace example
# TorchScript registers traced classes per process and renames a class traced a second time
# (`___torch_mangle_N`), and those names are serialized. A PTQ run therefore needs a process in which
# the student was never traced, so that two runs write identical bytes (d3). (Loading a TorchScript file
# registers nothing here and renames nothing.) The files' debug records also keep the absolute source
# paths and the Python call stack of the trace, so identical bytes also need the same invocation:
# scripts/run_ptq.py from the same checkout path and interpreter.
STUDENT_TS_CLASS = "__torch__.src.models.student.PlantSegStudent"
PTQ_RUN_ROLES = {ROLE_OF_RECORD: "am10_of_record", ROLE_SENSITIVITY: "am16_calibration_sensitivity"}
PARITY_SEED = SEED_OF_RECORD
# Synthetic parity inputs besides the calibration images: canvas-size single images, a batch of two
# and one non-square input, so a trace that specialised the batch or the spatial size cannot pass.
PARITY_SYNTHETIC_SHAPES = ((1, 3, 512, 512), (1, 3, 512, 512), (2, 3, 512, 512), (1, 3, 384, 512))
# The x86 latency copy's own serialization parity uses the synthetic inputs and the first calibration
# images; the artifact of record's parity uses all 128.
X86_PARITY_CALIBRATION_IMAGES = 16

# ------------------------------------------------------------------ census
DECLARED_INPUT = "quant"                              # QuantStub -> Quantize (float in, quint8 out)
DECLARED_OUTPUT = "dequant"                           # DeQuantStub -> DeQuantize (quint8 in, float out)
SIGMOID_PATH = "head.context.2"                       # LR-ASPP context: pool -> 1x1 conv -> Sigmoid
HEAD_INTERP_PROBE = "head.high_logits"                # its input is the head's OS8 interpolate output
FINAL_UPSAMPLE = "F.interpolate after the output DeQuantize (float by design, src/models/student.py:148-149)"
REQUIRED_MODULE_TYPES = {
    "quantized Conv2d": "torch.ao.nn.quantized.modules.conv.Conv2d",
    "quantized ConvReLU2d": "torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d",
    "quantized Hardswish": "torch.ao.nn.quantized.modules.activation.Hardswish",
    "QFunctional": "torch.ao.nn.quantized.modules.functional_modules.QFunctional",
}
# Float module TYPES that eager conversion keeps but that run on quantized tensors (their inputs and
# outputs are checked at run time, never trusted by type).
RUNTIME_QUANTIZED_TYPES = {
    "Sigmoid": "torch.nn.modules.activation.Sigmoid",
    "Hardsigmoid": "torch.nn.modules.activation.Hardsigmoid",
    "adaptive avg pool": "torch.nn.modules.pooling.AdaptiveAvgPool2d",
}
QUANTIZE_TYPE = "torch.ao.nn.quantized.modules.Quantize"
DEQUANTIZE_TYPE = "torch.ao.nn.quantized.modules.DeQuantize"
REQUIRED_GRAPH_OPS = ("quantized::conv2d", "quantized::conv2d_relu", "quantized::add", "quantized::mul",
                      "quantized::hardswish", "aten::sigmoid", "aten::hardsigmoid",
                      "aten::adaptive_avg_pool2d")
FLOAT_ONLY_GRAPH_OPS = ("aten::_convolution", "aten::conv2d", "aten::convolution", "aten::linear",
                        "aten::batch_norm", "aten::layer_norm", "aten::matmul")
UPSAMPLE_OP = "aten::upsample_bilinear2d"
EXPECTED_UPSAMPLES = 2                                # head OS8 (quantized) + final (float)

CODE_PATHS = ("src/quant/ptq.py", "src/quant/prepare.py", "src/quant/qconfig.py",
              "src/quant/x86_latency.py", "src/quant/runner.py", "src/quant/calibration.py",
              "src/quant/checkpoint.py", "src/models/student.py", "src/data/transforms.py",
              "src/eval/model_loading.py", "configs/quant.py", "configs/model.py", "configs/data.py",
              "scripts/run_ptq.py")


class PTQRefused(RuntimeError):
    """Refused or unreadable input. Raised before anything is written (exit 2)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class PTQStop(RuntimeError):
    """A lane 8 STOP: a per-tensor weight, an undeclared float region or a parity mismatch (exit 1)."""

    def __init__(self, code: str, message: str, report: dict | None = None):
        super().__init__(message)
        self.code = code
        self.report = report or {}


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ================================================================== split-path refusal (TRAIN only)
_VAL_TOKENS = ("val", "valid", "validation")


class SplitPathRefused(PTQRefused):
    """A data path naming the VAL or TEST split."""

    def __init__(self, message: str):
        super().__init__("eval_split_path", message)


def _path_forms(path) -> tuple[str, ...]:
    return (str(path), os.path.realpath(path))


def refuse_test_path(path, what: str) -> None:
    """Refuse a path containing "test" (case-insensitive), as given or once symlinks are resolved.

    The same rule as scripts/build_train_strata.py applies to every TRAIN path it lists. Opens nothing.
    """
    for form in _path_forms(path):
        if "test" in form.lower():
            raise SplitPathRefused(f"refusing {what} {form!r}: it contains 'test'; calibration reads the "
                                   "TRAIN split only and TEST is never listed or opened")


def refuse_val_dir(path, what: str) -> None:
    """Refuse a directory whose path, as given or resolved, has a component naming VAL.

    A component (or one of its tokens delimited by non-alphanumerics, e.g. `annotation_val`) equal to
    val, valid or validation. Applied to directories, never to an image stem. Opens nothing.
    """
    for form in _path_forms(path):
        for comp in re.split(r"[\\/]+", form.lower()):
            if comp in _VAL_TOKENS or any(t in _VAL_TOKENS for t in re.split(r"[^a-z0-9]+", comp)):
                raise SplitPathRefused(f"refusing {what} {form!r}: the component {comp!r} names the VAL "
                                       "split; calibration reads the TRAIN split only")


def refuse_eval_split_dir(path, what: str) -> None:
    """A directory on the calibration path: neither TEST (any "test") nor VAL (a val component)."""
    refuse_test_path(path, what)
    refuse_val_dir(path, what)


def refuse_eval_split_file(path, what: str) -> None:
    """A TRAIN file: the TEST rule on the whole path, the VAL rule on its (resolved) directories."""
    refuse_test_path(path, what)
    refuse_val_dir(Path(path).parent, f"{what} directory")
    refuse_val_dir(Path(os.path.realpath(path)).parent, f"{what} directory (resolved)")


# ================================================================== calibration lists
def list_role(seed: int) -> str:
    if seed == SEED_OF_RECORD:
        return ROLE_OF_RECORD
    if seed in SENSITIVITY_SEEDS:
        return ROLE_SENSITIVITY
    raise CalibrationIndexError(f"seed {seed!r} is not a registered calibration seed {list(LIST_SEEDS)} "
                                "(AM-10: 42; AM-16 item 5: 43, 44, 45)")


def ids_checksum(ids: Sequence[str]) -> str:
    """= src/quant/calibration.py _checksum (the index checksum run_e4/run_e7 pin)."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


LIST_REQUIRED_KEYS = ("schema", "list_schema", "role", "split", "seed", "count", "shared_by",
                      "candidate_pool_size", "selected_ids", "checksum_sha256", "split_list_sha256",
                      "procedure")


def verify_calibration_list(doc: dict) -> dict:
    """Validate a calibration list written by scripts/build_calibration_lists.py. Raises on any fault.

    The seed-42 list must also pass the committed E4/E7 index validator (`verify_calibration_index`),
    so the legacy run_e4/run_e7 path accepts exactly the same list of record.
    """
    if not isinstance(doc, dict):
        raise CalibrationIndexError(f"a calibration list is a JSON object, got {type(doc).__name__}")
    missing = [k for k in LIST_REQUIRED_KEYS if k not in doc]
    if missing:
        raise CalibrationIndexError(f"calibration list lacks {missing}")
    if doc["schema"] != CALIBRATION_SCHEMA or doc["list_schema"] != LIST_SCHEMA:
        raise CalibrationIndexError(f"unknown calibration list schema {doc['schema']!r} / "
                                    f"{doc['list_schema']!r}")
    if doc["split"] != CALIBRATION_SPLIT:
        raise CalibrationIndexError(f"calibration list split is {doc['split']!r}; only "
                                    f"{CALIBRATION_SPLIT!r} is ever used (VAL and TEST never calibrate)")
    seed = doc["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CalibrationIndexError(f"calibration seed must be an integer, got {seed!r}")
    if doc["role"] != list_role(seed):
        raise CalibrationIndexError(f"seed {seed} carries role {doc['role']!r}, expected {list_role(seed)!r}")
    ids = doc["selected_ids"]
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise CalibrationIndexError("selected_ids must be a list of non-empty strings")
    if any(("/" in i) or ("\\" in i) for i in ids):
        raise CalibrationIndexError("a calibration id is an image stem, never a path")
    if len(ids) != CALIBRATION_COUNT or doc["count"] != CALIBRATION_COUNT:
        raise CalibrationIndexError(f"calibration list holds {len(ids)} ids (count={doc['count']}); "
                                    f"AM-10 fixes {CALIBRATION_COUNT}")
    if len(set(ids)) != len(ids):
        raise CalibrationIndexError("calibration list contains duplicate ids")
    if ids_checksum(ids) != doc["checksum_sha256"]:
        raise CalibrationIndexError("checksum_sha256 does not match selected_ids")
    if list(doc["shared_by"]) != LIST_SHARED_BY:
        raise CalibrationIndexError(f"shared_by is {doc['shared_by']!r}, expected {LIST_SHARED_BY}")
    if not isinstance(doc["candidate_pool_size"], int) or doc["candidate_pool_size"] < CALIBRATION_COUNT:
        raise CalibrationIndexError(f"candidate_pool_size {doc['candidate_pool_size']!r} is invalid")
    if not isinstance(doc["split_list_sha256"], str) or not _HEX64.match(doc["split_list_sha256"]):
        raise CalibrationIndexError("split_list_sha256 must be a sha256 hex digest")
    if seed == SEED_OF_RECORD:
        verify_calibration_index(doc)                 # the committed E4/E7 validator, unchanged
    return doc


def _strict_json(raw: bytes, origin: str):
    def _no_constants(c):
        raise CalibrationIndexError(f"{origin}: non-finite JSON constant {c!r}")
    return json.loads(raw.decode("utf-8"), parse_constant=_no_constants)


def load_calibration_list(path, *, expected_checksum: str | None = None) -> tuple[dict, str]:
    """Load and validate a calibration list. Returns (document, file sha256).

    `expected_checksum` pins `checksum_sha256` (the id checksum a decision-log row records); a
    different list is refused even when it is valid on its own.
    """
    p = Path(path)
    if not p.is_file():
        raise PTQRefused("calibration_list_missing", f"calibration list not found: {p}")
    refuse_test_path(p, "calibration list path")
    raw = p.read_bytes()
    try:
        doc = verify_calibration_list(_strict_json(raw, str(p)))
    except (CalibrationIndexError, ValueError) as e:
        raise PTQRefused("calibration_list_invalid", f"{p}: {e}") from e
    if expected_checksum is not None and doc["checksum_sha256"] != expected_checksum:
        raise PTQRefused("calibration_list_checksum_mismatch",
                         f"{p} has checksum_sha256 {doc['checksum_sha256']}, expected {expected_checksum}")
    return doc, _sha256_bytes(raw)


# ================================================================== calibration data (TRAIN only)
def calibration_dataset(data_root, ids: Sequence[str]):
    """The list's TRAIN images and masks, in list order, through the runner's calibration dataset.

    `src.quant.runner.CalibrationDataset` reads images/train and annotations/train with the clean-test
    core preprocessing and no augmentation. Every directory and file on the path is checked against
    the VAL/TEST rules before the dataset object exists, and again for every resolved pair.
    """
    from .runner import CalibrationDataset, QuantRunError

    root = Path(data_root)
    refuse_eval_split_dir(root, "data root")
    for d in (root / "images" / "train", root / "annotations" / "train"):
        refuse_eval_split_dir(d, "TRAIN directory")
        if not d.is_dir():
            raise PTQRefused("train_dir_missing", f"TRAIN directory not found: {d}")
    try:
        ds = CalibrationDataset(root, list(ids))
    except QuantRunError as e:
        raise PTQRefused(e.code, str(e)) from e
    for img, mask in ds.pairs:
        refuse_eval_split_file(img, "calibration image")
        refuse_eval_split_file(mask, "calibration mask")
    order = [img.stem for img, _ in ds.pairs]
    if order != list(ids):
        raise PTQRefused("calibration_order", "the calibration dataset does not follow the list order")
    return ds


def iter_calibration(ds) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """(image [1,3,512,512], mask [1,512,512]) in list order: one image per mini-batch (AM-10)."""
    loader = torch.utils.data.DataLoader(ds, batch_size=CALIBRATION_BATCH_SIZE, shuffle=False,
                                         num_workers=0, drop_last=False)
    for img, mask in loader:
        yield img, mask


def iter_calibration_images(ds) -> Iterator[torch.Tensor]:
    for img, _ in iter_calibration(ds):
        yield img


# ================================================================== quantization
class _Counted:
    """Count what an iterable hands out, so the batches actually seen are recorded, not assumed."""

    def __init__(self, it: Iterable):
        self._it, self.n = iter(it), 0

    def __iter__(self):
        for x in self._it:
            self.n += 1
            yield x


def _require_engine(allowed: Sequence[str], where: str) -> str:
    engine = torch.backends.quantized.engine
    if engine not in allowed:
        raise PTQRefused("wrong_engine", f"quantized engine is {engine!r} at {where}; expected {list(allowed)}")
    return engine


def quantize_qnnpack(model: nn.Module, images: Iterable[torch.Tensor]) -> tuple[nn.Module, int]:
    """Fuse -> registered PTQ qconfig -> calibrate (batch 1) -> convert, all under QNNPACK."""
    select_qnnpack_backend()
    prepared = prepare_ptq(model, select_backend=False)
    n = calibrate(prepared, images)
    _require_engine((QUANT_BACKEND,), "QNNPACK convert")
    return convert_model(prepared).eval(), n


def quantize_x86_copy(stage: str, model: nn.Module,
                      images: Iterable[torch.Tensor]) -> tuple[nn.Module, str, int]:
    """The x86 latency copy from the same images in the same order (reduce_range=True)."""
    counted = _Counted(images)
    copy_ = build_x86_latency_copy(stage, model, calibration_batches=counted)   # selects the x86 engine
    engine = _require_engine(X86_BACKENDS, "x86 convert")
    return copy_.eval(), engine, counted.n


def qconfig_summary(qconfig) -> dict:
    d = describe_qconfig(qconfig)
    flat = {k: {kk: str(vv) for kk, vv in v.items()} for k, v in d.items()}
    return {"activation": flat["activation"], "weight": flat["weight"],
            "fingerprint": _sha256_bytes(json.dumps(flat, sort_keys=True).encode("utf-8"))[:16]}


# ================================================================== serialization (in memory)
def torchscript_bytes(converted: nn.Module, meta: dict) -> tuple[bytes, list[str]]:
    """Trace the converted model at the canvas of record and serialize it with its identity embedded.

    Serialization happens in memory so every artifact exists as bytes, hashed, before any file is
    created. The trace records the ops, not the example's values; the TracerWarnings it raises are
    returned verbatim for the run provenance (parity, not the warnings, decides correctness).
    """
    example = torch.zeros(EXAMPLE_SHAPE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traced = torch.jit.trace(converted, example)
    notes = sorted({f"{Path(w.filename).name}:{w.lineno}: {w.category.__name__}: "
                    f"{str(w.message).splitlines()[0][:160]}" for w in caught})
    buf = io.BytesIO()
    torch.jit.save(traced, buf, _extra_files={
        INT8_TORCHSCRIPT_META: json.dumps(meta, sort_keys=True, separators=(",", ":"))})
    return buf.getvalue(), notes


def state_dict_bytes(converted: nn.Module, stage_name: str, method: str = "ptq") -> bytes:
    """The converted state_dict in the exact src/quant/runner.py schema."""
    buf = io.BytesIO()
    torch.save({"stage": stage_name, "quantization": method, "num_classes": NUM_CLASSES,
                "model": converted.state_dict()}, buf)
    return buf.getvalue()


def write_exclusive(files: dict) -> list[Path]:
    """Create every file exclusively (never replacing one); a failure removes what was created."""
    existing = [str(p) for p in files if Path(p).exists()]
    if existing:
        raise PTQRefused("output_exists", f"refusing to overwrite {existing}")
    created: list[Path] = []
    try:
        for path, data in files.items():
            fh = open(path, "xb")
            created.append(Path(path))
            with fh:
                fh.write(data)
    except BaseException:
        for p in created:
            p.unlink(missing_ok=True)
        raise
    return created


# ================================================================== checks: weight scheme
def _weight_row(name: str, w: torch.Tensor) -> dict:
    qs = w.qscheme()
    row = {"module": name, "qscheme": str(qs), "dtype": str(w.dtype), "out_channels": int(w.shape[0])}
    per_channel = qs in (torch.per_channel_symmetric, torch.per_channel_affine)
    if per_channel:
        zps = w.q_per_channel_zero_points()
        row.update(axis=int(w.q_per_channel_axis()), n_scales=int(w.q_per_channel_scales().numel()),
                   zero_points_all_zero=bool((zps == 0).all()))
    row["per_channel_int8"] = bool(per_channel and w.dtype == torch.qint8 and row.get("axis") == 0
                                   and row.get("zero_points_all_zero") is True
                                   and row.get("n_scales") == int(w.shape[0]))
    return row


def weight_scheme_report(module: nn.Module) -> dict:
    """Every quantized conv weight of an eager converted model or of a loaded TorchScript module.

    Eager: `m.weight()` of every quantized conv, plus any FP32 nn.Conv2d that survived conversion.
    TorchScript: each `_packed_params` object is unpacked through `quantized::conv2d_unpack`, so the
    check reads the weights inside the artifact file itself.
    """
    from torch.ao.nn.quantized.modules.conv import _ConvNd as QuantizedConvNd

    rows, float_convs, unreadable = [], [], []
    scripted = isinstance(module, torch.jit.ScriptModule)
    for name, m in module.named_modules():
        if scripted:
            if not m._c.hasattr("_packed_params"):
                continue
            try:
                w, _bias = torch.ops.quantized.conv2d_unpack(m._c.getattr("_packed_params"))
            except Exception as e:                    # noqa: BLE001 -- recorded; a STOP below
                unreadable.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
                continue
            rows.append(_weight_row(name, w))
        elif isinstance(m, nn.Conv2d):
            float_convs.append(name)
        elif isinstance(m, QuantizedConvNd):
            rows.append(_weight_row(name, m.weight()))
    violations = [r["module"] for r in rows if not r["per_channel_int8"]]
    return {"source": "torchscript" if scripted else "eager", "quantized_convs": len(rows),
            "per_channel_int8": sum(r["per_channel_int8"] for r in rows),
            "violations": violations, "fp32_convs": float_convs, "unreadable_packed_params": unreadable,
            "ok": bool(rows) and not violations and not float_convs and not unreadable,
            "qschemes": dict(Counter(r["qscheme"] for r in rows))}


# ================================================================== checks: census
def _qname(m) -> str:
    return f"{type(m).__module__}.{type(m).__name__}"


def _tensor_info(t) -> dict:
    if not torch.is_tensor(t):
        return {"type": type(t).__name__}
    info = {"is_quantized": bool(t.is_quantized), "dtype": str(t.dtype), "shape": list(t.shape)}
    if t.is_quantized and t.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        info.update(scale=float(t.q_scale()), zero_point=int(t.q_zero_point()))
    return info


@torch.no_grad()
def runtime_census(converted: nn.Module, x: torch.Tensor) -> dict:
    """Module census of an EAGER converted model plus run-time evidence of what ran quantized.

    Forward hooks on every leaf module record whether each call took and returned quantized tensors
    (QFunctional add/mul show through their `activation_post_process` leaf). A module that is never
    called (for example `skip_add` of a block without a residual) is listed as unused. The only float
    edges allowed are the declared input Quantize (float in, quint8 out) and output DeQuantize (quint8
    in, float out); the final upsample after DeQuantize is float by design.
    """
    types = Counter(_qname(m) for _, m in converted.named_modules())
    leaves = {n: m for n, m in converted.named_modules() if not list(m.children())}
    calls: dict[str, list[tuple[bool, bool]]] = {}
    sigmoid_io: list[dict] = []
    interp_in: list[dict] = []
    hooks = []
    for name, m in leaves.items():
        def hook(mod, inp, out, _name=name):
            i = inp[0] if isinstance(inp, tuple) and inp else inp
            calls.setdefault(_name, []).append(
                (bool(getattr(i, "is_quantized", False)), bool(getattr(out, "is_quantized", False))))
            if _name == SIGMOID_PATH:
                sigmoid_io.append({"input": _tensor_info(i), "output": _tensor_info(out)})
        hooks.append(m.register_forward_hook(hook))
    modules = dict(converted.named_modules())
    if HEAD_INTERP_PROBE in modules:
        hooks.append(modules[HEAD_INTERP_PROBE].register_forward_pre_hook(
            lambda mod, inp: interp_in.append(_tensor_info(inp[0]))))
    try:
        converted.eval()
        y = converted(x)
    finally:
        for h in hooks:
            h.remove()

    float_modules, boundary = [], {}
    for name, seen in calls.items():
        qtype = _qname(leaves[name])
        if name == DECLARED_INPUT:
            boundary[name] = {"type": qtype, "calls": seen,
                              "ok": qtype == QUANTIZE_TYPE and all(s == (False, True) for s in seen)}
        elif name == DECLARED_OUTPUT:
            boundary[name] = {"type": qtype, "calls": seen,
                              "ok": qtype == DEQUANTIZE_TYPE and all(s == (True, False) for s in seen)}
        elif any(s != (True, True) for s in seen):
            float_modules.append({"module": name, "type": qtype, "calls": seen})
    for declared in (DECLARED_INPUT, DECLARED_OUTPUT):
        boundary.setdefault(declared, {"type": None, "calls": [], "ok": False})
    unused = sorted(n for n in leaves if n not in calls)

    def _ran_quantized(qtype: str) -> tuple[int, bool]:
        names = [n for n, m in leaves.items() if _qname(m) == qtype and n in calls]
        return len(names), bool(names) and all(all(s == (True, True) for s in calls[n]) for n in names)

    required = {label: types.get(t, 0) > 0 for label, t in REQUIRED_MODULE_TYPES.items()}
    for label, t in RUNTIME_QUANTIZED_TYPES.items():
        n, ok = _ran_quantized(t)
        required[f"{label} ran quantized ({n} executed)"] = ok
    interp_q = bool(interp_in) and bool(interp_in[0].get("is_quantized"))
    required["head OS8 interpolate ran quantized"] = interp_q
    sig = sigmoid_io[0] if sigmoid_io else None
    sigmoid_quantized = bool(sig and sig["input"].get("is_quantized") and sig["output"].get("is_quantized"))
    counts_ok = types.get(QUANTIZE_TYPE, 0) == 1 and types.get(DEQUANTIZE_TYPE, 0) == 1
    ok = (not float_modules and all(b["ok"] for b in boundary.values()) and all(required.values())
          and sigmoid_quantized and counts_ok and torch.is_tensor(y) and not y.is_quantized)
    return {
        "module_census": dict(sorted(types.items())),
        "leaf_modules": len(leaves), "executed_leaf_modules": len(calls),
        "unused_leaf_modules": unused,
        "float_modules": float_modules,
        "declared_float_boundary": boundary,
        "quantize_modules": types.get(QUANTIZE_TYPE, 0), "dequantize_modules": types.get(DEQUANTIZE_TYPE, 0),
        "required": required,
        "sigmoid": {"module": SIGMOID_PATH, "quantized": sigmoid_quantized, **(sig or {}),
                    "u6_values": ({"scale": sig["output"].get("scale"),
                                   "zero_point": sig["output"].get("zero_point")}
                                  if sigmoid_quantized else None)},
        "head_interpolate": {"probe": f"forward pre-hook on {HEAD_INTERP_PROBE}", "quantized": interp_q,
                             "input": interp_in[0] if interp_in else None},
        "final_upsample": FINAL_UPSAMPLE,
        "output": _tensor_info(y),
        "ok": ok,
    }


def _walk(nodes):
    for n in nodes:
        yield n
        for b in n.blocks():
            yield from _walk(b.nodes())


def graph_census(ts: torch.jit.ScriptModule) -> dict:
    """Op census of a TorchScript INT8 artifact's inlined graph, with the float-boundary data flow.

    Exactly one aten::quantize_per_tensor, fed by the graph's tensor input, and exactly one
    aten::dequantize, whose output feeds only the final upsample that the graph returns. With one
    quantize and one dequantize there is no dequantize -> float op -> quantize detour: every other op
    in between runs on quantized tensors (a float kernel given a quantized tensor raises, it never falls
    back silently).
    """
    g = ts.inlined_graph
    nodes = list(_walk(g.nodes()))
    kinds = Counter(n.kind() for n in nodes)
    inputs, outputs = list(g.inputs()), list(g.outputs())
    q = [n for n in nodes if n.kind() == "aten::quantize_per_tensor"]
    dq = [n for n in nodes if n.kind() == "aten::dequantize"]
    x_in = inputs[1] if len(inputs) > 1 else None
    input_ok = bool(len(q) == 1 and x_in is not None and q[0].inputsAt(0).unique() == x_in.unique())
    output_ok = False
    if len(dq) == 1 and len(outputs) == 1:
        users = [u.user for u in dq[0].output().uses()]
        output_ok = (len(users) == 1 and users[0].kind() == UPSAMPLE_OP
                     and users[0].output().unique() == outputs[0].unique())
    float_ops = {k: kinds[k] for k in FLOAT_ONLY_GRAPH_OPS if kinds.get(k)}
    required = {op: kinds.get(op, 0) > 0 for op in REQUIRED_GRAPH_OPS}
    upsamples = kinds.get(UPSAMPLE_OP, 0)
    ok = (input_ok and output_ok and not float_ops and all(required.values())
          and upsamples == EXPECTED_UPSAMPLES)
    return {"op_census": {k: v for k, v in sorted(kinds.items()) if not k.startswith("prim::")},
            "quantize_per_tensor": len(q), "dequantize": len(dq),
            "input_quantize_fed_by_graph_input": input_ok,
            "dequantize_feeds_only_final_upsample_output": output_ok,
            "float_only_ops": float_ops, "required_ops": required, "upsample_bilinear2d": upsamples,
            "ok": ok}


# ================================================================== checks: parity
def synthetic_parity_inputs() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(PARITY_SEED)
    return [torch.randn(*s, generator=g) for s in PARITY_SYNTHETIC_SHAPES]


@torch.no_grad()
def output_parity(models: dict, inputs: Iterable[torch.Tensor],
                  pairs: Sequence[tuple[str, str]]) -> dict:
    """Bitwise output comparison of named models over the same inputs (torch.equal, exact)."""
    stats = {f"{a} == {b}": {"compared": 0, "equal": 0, "first_mismatch": None} for a, b in pairs}
    n = 0
    for i, x in enumerate(inputs):
        n += 1
        outs = {name: m(x) for name, m in models.items()}
        for a, b in pairs:
            ya, yb = outs[a], outs[b]
            s = stats[f"{a} == {b}"]
            s["compared"] += 1
            same = ya.shape == yb.shape and ya.dtype == yb.dtype and torch.equal(ya, yb)
            if same:
                s["equal"] += 1
            elif s["first_mismatch"] is None:
                diff = (float((ya.float() - yb.float()).abs().max()) if ya.shape == yb.shape
                        else None)
                s["first_mismatch"] = {"input": i, "shape_a": list(ya.shape), "shape_b": list(yb.shape),
                                       "max_abs_diff": diff}
    all_equal = n > 0 and all(s["equal"] == s["compared"] == n for s in stats.values())
    return {"inputs": n, "pairs": stats, "all_equal": all_equal}


@torch.no_grad()
def pixel_agreement(fp32: nn.Module, int8: nn.Module,
                    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]], ignore_index: int = 255) -> dict:
    """Descriptive argmax agreement between an FP32 model and its INT8 artifact (lane 8 (c))."""
    agree_valid = n_valid = agree_all = n_all = images = 0
    for img, mask in pairs:
        a, b = fp32(img).argmax(1), int8(img).argmax(1)
        same = a == b
        valid = mask != ignore_index
        agree_valid += int((same & valid).sum())
        n_valid += int(valid.sum())
        agree_all += int(same.sum())
        n_all += int(same.numel())
        images += 1
    return {"images": images, "valid_pixels": n_valid,
            "agreement_valid_pixels": (agree_valid / n_valid) if n_valid else None,
            "canvas_pixels": n_all, "agreement_canvas_pixels": (agree_all / n_all) if n_all else None,
            "ignore_index": ignore_index, "descriptive": True}


# ================================================================== x86 latency copy (descriptive)
def load_x86_latency_torchscript(path, *, stage: str):
    """Load the x86 latency copy under its own engine. Never an accuracy artifact."""
    engine = select_x86_backend()
    return load_int8_torchscript(path, stage=stage, method="ptq", engine=engine,
                                 artifact_role=X86_LATENCY_ARTIFACT_ROLE)


# ================================================================== provenance
def _git(*args) -> str | None:
    """git output from the repository root, or None. GIT_OPTIONAL_LOCKS=0: never writes the index."""
    try:
        p = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True, timeout=60,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def code_provenance(paths: Sequence[str] = CODE_PATHS) -> dict:
    """HEAD, whether these files are clean at it (explicit-path status, never repo-wide), their sha256.

    Inside the pinned image without a .git directory the commit and cleanliness are recorded as null.
    """
    head = (_git("rev-parse", "HEAD") or "").strip()
    commit = head if _HEX40.match(head) else None
    status = _git("status", "--porcelain=v1", "--", *paths) if commit else None
    return {"commit": commit, "clean_at_commit": None if status is None else status == "",
            "files_sha256": {p: sha256_file(REPO / p) for p in paths if (REPO / p).is_file()}}


def environment() -> dict:
    import numpy
    import PIL
    import torchvision
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "supported_engines": list(torch.backends.quantized.supported_engines),
            "image_digest": (os.environ.get("PLANTSEG_IMAGE_DIGEST", "").strip() or None)}


# ================================================================== the run
def artifact_names(stage_key: str) -> dict:
    return {"torchscript": f"{stage_key}_int8_qnnpack.torchscript.pt",
            "state_dict": f"{stage_key}_int8_student.pt",
            "x86_copy": f"{stage_key}_int8_x86_latency.torchscript.pt",
            "run_meta": f"{stage_key}_ptq_run_meta.json",
            "stop": f"{stage_key}_ptq_STOP.json"}


def _json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def run_ptq(*, stage: str, source_ckpt, expect_source_sha256: str, calibration_list,
            expect_list_checksum: str, data_root, out_dir, threads: int | None = None,
            log=print) -> dict:
    """E4/E7 PTQ of record: artifacts + run provenance, or PTQRefused / PTQStop. See the module doc."""
    from src.seeds import set_seed
    from .runner import QuantRunError, build_run_provenance, check_output_dir, check_source
    from .stages import resolve_quant_stage

    # ---------------- cheap gates: nothing is loaded or written before these pass
    try:
        st = resolve_quant_stage(stage)
    except ValueError as e:
        raise PTQRefused("stage_unknown", str(e)) from e
    if st["method"] != "ptq":
        raise PTQRefused("stage_not_ptq", f"{st['name']} is a {st['method'].upper()} stage; PTQ is E4/E7")
    if student_traced_in_process():
        raise PTQRefused("fresh_process_required",
                         "the student was already traced in this process, so this run's TorchScript "
                         "files would carry renamed classes and not reproduce byte for byte (d3); run "
                         "scripts/run_ptq.py once per fresh process")
    for name, value in (("--expect-source-sha256", expect_source_sha256),
                        ("--expect-list-checksum", expect_list_checksum)):
        if not isinstance(value, str) or not _HEX64.match(value):
            raise PTQRefused("expected_hash_invalid", f"{name} must be a sha256 hex digest, got {value!r}")
    try:
        out = check_output_dir(str(out_dir) if out_dir is not None else None, create=False)
    except QuantRunError as e:
        raise PTQRefused(e.code, str(e)) from e
    names = artifact_names(st["key"])
    present = [n for n in names.values() if (out / n).exists()]
    if present:
        raise PTQRefused("output_exists", f"{out} already holds {present}; a PTQ run never overwrites")
    if data_root is None:
        raise PTQRefused("data_root_unset", "--data-root is required (the TRAIN images of the list)")
    refuse_eval_split_dir(data_root, "data root")
    cal, cal_file_sha = load_calibration_list(calibration_list, expected_checksum=expect_list_checksum)
    src_path = Path(source_ckpt) if source_ckpt else None
    if src_path is None or not src_path.is_file():
        raise PTQRefused("source_missing", f"source checkpoint not found: {source_ckpt}")
    src_sha = sha256_file(src_path)
    if src_sha != expect_source_sha256:
        raise PTQRefused("source_sha256_mismatch",
                         f"{src_path} has sha256 {src_sha}, expected {expect_source_sha256}")
    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        raise PTQRefused("backend_unavailable", str(e)) from e
    if threads is not None:
        if threads < 1:
            raise PTQRefused("threads_invalid", f"--threads must be >= 1, got {threads}")
        torch.set_num_threads(threads)
    set_seed(SEED_OF_RECORD)                          # B6 protocol; PTQ itself draws no random number

    # ---------------- source and calibration data
    try:
        model, _ckpt, source_meta = check_source(st, str(src_path))
    except QuantRunError as e:
        raise PTQRefused(e.code, str(e)) from e
    if source_meta["sha256"] != src_sha:
        raise PTQRefused("source_changed", "the source checkpoint changed while it was being loaded")
    model.eval()
    ds = calibration_dataset(data_root, cal["selected_ids"])
    log(f"[ptq] {st['name']} from {st['source_stage']} {src_sha[:12]}; list seed {cal['seed']} "
        f"({cal['role']}) {cal['checksum_sha256'][:12]}; {len(ds)} TRAIN images, batch "
        f"{CALIBRATION_BATCH_SIZE}")

    # ---------------- quantize: QNNPACK artifact of record, then the x86 latency copy
    counted_q = _Counted(iter_calibration_images(ds))
    q_model, n_q = quantize_qnnpack(model, counted_q)
    log(f"[ptq] QNNPACK calibrated on {n_q} batches and converted")
    x_model, x_engine, n_x = quantize_x86_copy(st["key"], model, iter_calibration_images(ds))
    log(f"[ptq] x86 latency copy calibrated on {n_x} batches and converted under {x_engine!r}")
    if not (n_q == n_x == len(ds) == CALIBRATION_COUNT):
        raise PTQRefused("calibration_count", f"calibration saw {n_q}/{n_x} batches, expected "
                                              f"{CALIBRATION_COUNT}")

    # ---------------- serialize in memory, then create every file exclusively
    ptq_q, x86_q = qconfig_summary(ptq_qconfig()), qconfig_summary(x86_ptq_qconfig())
    identity = {"stage": st["name"], "quantization": "ptq", "num_classes": NUM_CLASSES,
                "source_stage": st["source_stage"], "source_checkpoint_sha256": src_sha,
                "calibration_checksum_sha256": cal["checksum_sha256"], "calibration_seed": cal["seed"],
                "calibration_role": cal["role"], "calibration_batch_size": CALIBRATION_BATCH_SIZE,
                "traced_example_shape": list(EXAMPLE_SHAPE),
                "input": "float32 [N, 3, H, W]; the 512x512 canvas of record",
                "output": f"float32 [N, {NUM_CLASSES}, H, W] logits", "writer": "src/quant/ptq.py",
                "schema": INT8_TORCHSCRIPT_SCHEMA}
    ts_meta = {**identity, "engine": QUANT_BACKEND, "artifact_role": ACCURACY_ARTIFACT_ROLE,
               "qconfig_fingerprint": ptq_q["fingerprint"], "state_dict_companion": names["state_dict"]}
    x86_meta = {**identity, "engine": x_engine, "artifact_role": X86_LATENCY_ARTIFACT_ROLE,
                "qconfig_fingerprint": x86_q["fingerprint"], "usable_for_accuracy": False,
                "reduce_range": True}
    select_qnnpack_backend()
    ts_b, ts_notes = torchscript_bytes(q_model, ts_meta)
    sd_b = state_dict_bytes(q_model, st["name"])
    select_x86_backend()
    x86_b, x86_notes = torchscript_bytes(x_model, x86_meta)
    select_qnnpack_backend()
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / v for k, v in names.items()}
    write_exclusive({paths["torchscript"]: ts_b, paths["state_dict"]: sd_b, paths["x86_copy"]: x86_b})
    shas = {k: _sha256_bytes(b) for k, b in (("torchscript", ts_b), ("state_dict", sd_b),
                                             ("x86_copy", x86_b))}
    for k in shas:
        if sha256_file(paths[k]) != shas[k]:
            raise PTQStop("artifact_write_mismatch", f"{paths[k]} on disk differs from what was serialized")
    log(f"[ptq] wrote {names['torchscript']} {shas['torchscript'][:12]}, {names['state_dict']} "
        f"{shas['state_dict'][:12]}, {names['x86_copy']} {shas['x86_copy'][:12]}")

    # ---------------- verification from disk: x86 copy (its engine), then the artifact of record
    x86_ts, _ = load_x86_latency_torchscript(paths["x86_copy"], stage=st["name"])
    x86_weights = weight_scheme_report(x86_ts)
    x86_graph = graph_census(x86_ts)
    x86_parity = output_parity({"x86_eager": x_model, "x86_torchscript": x86_ts},
                               _chain(synthetic_parity_inputs(),
                                      _take(iter_calibration_images(ds), X86_PARITY_CALIBRATION_IMAGES)),
                               [("x86_eager", "x86_torchscript")])
    select_qnnpack_backend()
    ts_model, _ = load_int8_torchscript(paths["torchscript"], stage=st["name"], method="ptq",
                                        engine=QUANT_BACKEND)
    sd_model = rebuild_int8_from_state_dict(paths["state_dict"], stage=st["name"], method="ptq")
    # The gate of record: the TorchScript artifact against its state_dict rebuild, both re-read from
    # disk, on the fixed synthetic inputs and all 128 calibration images; plus the TorchScript
    # artifact against the in-memory model that was calibrated, on the synthetic inputs.
    parity = {
        "torchscript_vs_state_dict": output_parity(
            {"torchscript": ts_model, "state_dict": sd_model},
            _chain(synthetic_parity_inputs(), iter_calibration_images(ds)),
            [("torchscript", "state_dict")]),
        "eager_vs_torchscript": output_parity(
            {"eager": q_model, "torchscript": ts_model}, synthetic_parity_inputs(),
            [("eager", "torchscript")]),
    }
    parity["all_equal"] = all(v["all_equal"] for v in parity.values())
    weights = {"eager": weight_scheme_report(q_model), "torchscript": weight_scheme_report(ts_model),
               "x86_torchscript": x86_weights}
    census = runtime_census(q_model, torch.zeros(EXAMPLE_SHAPE))
    graph = graph_census(ts_model)
    _require_engine((QUANT_BACKEND,), "verification")

    checks = {
        "weights_per_channel_int8": all(w["ok"] for w in weights.values()),
        "no_undeclared_float_module": census["ok"],
        "torchscript_float_boundary": graph["ok"],
        "x86_copy_float_boundary": x86_graph["ok"],
        "parity_torchscript_state_dict_eager": parity["all_equal"],
        "parity_x86_copy": x86_parity["all_equal"],
    }
    verification = {"checks": checks, "weights": weights, "census": census, "graph": graph,
                    "x86_graph": x86_graph, "parity": parity, "x86_parity": x86_parity,
                    "trace_warnings": {"torchscript": ts_notes, "x86_copy": x86_notes}}
    if not all(checks.values()):
        failed = [k for k, ok in checks.items() if not ok]
        per_tensor = any(w["violations"] for w in weights.values())
        float_region = (any(w["fp32_convs"] for w in weights.values())
                        or not (checks["no_undeclared_float_module"] and checks["torchscript_float_boundary"]
                                and checks["x86_copy_float_boundary"]))
        code = ("per_tensor_weight" if per_tensor
                else "undeclared_float_fallback" if float_region
                else "weights_unreadable" if not checks["weights_per_channel_int8"]
                else "parity_mismatch")
        stop = {"schema": RUN_META_SCHEMA, "status": "STOP", "code": code, "failed_checks": failed,
                "stage": st["name"], "artifacts": {k: {"path": names[k], "sha256": shas[k]} for k in shas},
                "verification": verification, "created_utc": _utc_now()}
        write_exclusive({paths["stop"]: _json_bytes(stop)})
        raise PTQStop(code, f"{st['name']} PTQ STOP ({code}): {failed}; no run provenance was written; "
                            f"see {paths['stop']}", stop)

    # ---------------- the run provenance (the evaluator's --provenance)
    prov = build_run_provenance(
        stage=st, source_meta=source_meta, method="ptq",
        qconfig_summary={"activation": ptq_q["activation"], "weight": ptq_q["weight"],
                         "fingerprint": ptq_q["fingerprint"]},
        calibration=None, training=None, converted_path=paths["torchscript"], coverage=None,
        forward={"ok": True, "output_shape": census["output"]["shape"], "dtype": census["output"]["dtype"],
                 "error": None},
        backend=QUANT_BACKEND)
    prov.update({
        "schema": RUN_META_SCHEMA,
        "lane": LANE,
        # am10_of_record only for the registered AM-10 list; an official E4/E7 score also requires that
        # list to be the committed one (src/eval/stage_artifacts.py int8_official_calibration_error)
        "ptq_run_role": (PTQ_RUN_ROLES[cal["role"]] if cal.get("artifact_status") == LIST_STATUS_REGISTERED
                         else "smoke"),
        "artifact_format": "torchscript",
        # relative to this file's directory (src/eval/stage_artifacts.py resolves it), so a run
        # directory moves between the pinned image and its host unchanged; the sha256 anchors it
        "converted_artifact": names["torchscript"],
        "converted_artifact_sha256": shas["torchscript"],
        "converted_artifact_bytes": len(ts_b),
        "converted_artifact_role": ACCURACY_ARTIFACT_ROLE,
        "state_dict_artifact": {"path": names["state_dict"], "sha256": shas["state_dict"],
                                "bytes": len(sd_b), "schema": "src/quant/runner.py converted state_dict",
                                "role": "companion of the TorchScript artifact of record"},
        "x86_latency_copy": {"path": names["x86_copy"], "sha256": shas["x86_copy"], "bytes": len(x86_b),
                             "engine": x_engine, "artifact_role": X86_LATENCY_ARTIFACT_ROLE,
                             "reduce_range": True, "qconfig": x86_q, "batches_seen": n_x,
                             "usable_for_accuracy": False, "usable_for_robustness": False,
                             "usable_for_size_reporting": False, "usable_for_latency": True},
        "engine": QUANT_BACKEND,
        "observers": {"activation": ptq_q["activation"]["observer_class"],
                      "weight": ptq_q["weight"]["observer_class"]},
        "calibration": {"list_path": str(calibration_list), "list_file_sha256": cal_file_sha,
                        "checksum_sha256": cal["checksum_sha256"], "checksum_definition": CHECKSUM_DEFINITION,
                        "seed": cal["seed"], "role": cal["role"], "split": cal["split"],
                        "count": cal["count"], "n_calib": n_q, "batch_size": CALIBRATION_BATCH_SIZE,
                        "batches_seen": n_q, "order": CALIBRATION_ORDER,
                        "split_list_sha256": cal["split_list_sha256"],
                        "list_artifact_status": cal.get("artifact_status"),
                        "preprocessing": PREPROCESSING, "augmentation": False,
                        "data_root": str(data_root), "val_or_test_read": False},
        "artifact_bytes_note": ("the TorchScript files' debug records keep absolute source paths and the "
                                "Python call stack of the trace, so their sha256 reproduces for the same "
                                "invocation (scripts/run_ptq.py, same checkout path and interpreter, one "
                                "run per process); state_dict_artifact.sha256 depends on none of these, "
                                "and the parity check ties the two files"),
        "verification": verification,
        "source_expected_sha256_matched": True,
        "code": code_provenance(),
        "environment": environment(),
    })
    run_meta_b = _json_bytes(prov)
    write_exclusive({paths["run_meta"]: run_meta_b})
    log(f"[ptq] all checks passed; run provenance {paths['run_meta']}")
    return prov


def _chain(*its):
    for it in its:
        yield from it


def _take(it, n: int):
    for i, x in enumerate(it):
        if i >= n:
            return
        yield x


# ================================================================== official E4/E7: the list of record
def list_of_record_error(calibration: dict, repo_root) -> str | None:
    """None iff a PTQ run's calibration record is the AM-10 list of record as committed in the repository.

    An official E4/E7 score needs the registered seed-42 list, and the one committed at
    configs/calibration/ptq_calibration_seed42.json (the file a decision-log row records): E4 and E7
    are then provably calibrated on one list. A sensitivity list (AM-16 item 5) or a synthetic list
    is refused; its runs stay descriptive (non-official) evaluations.
    """
    if calibration.get("role") != ROLE_OF_RECORD or calibration.get("seed") != SEED_OF_RECORD:
        return (f"calibrated on seed {calibration.get('seed')!r} ({calibration.get('role')!r}); an official "
                f"E4/E7 score needs the AM-10 list of record (seed {SEED_OF_RECORD})")
    if calibration.get("list_artifact_status") != LIST_STATUS_REGISTERED:
        return (f"calibrated on a list with artifact_status {calibration.get('list_artifact_status')!r}; an "
                f"official score needs the registered list (the 5,367-image TRAIN split)")
    committed = Path(repo_root) / PTQ["calibration_list_dir"] / LIST_FILENAME.format(seed=SEED_OF_RECORD)
    if not committed.is_file():
        return f"the list of record is not committed ({committed.as_posix()} is absent)"
    try:
        doc = verify_calibration_list(_strict_json(committed.read_bytes(), str(committed)))
    except (CalibrationIndexError, ValueError) as e:
        return f"the committed list of record {committed.as_posix()} does not validate: {e}"
    if doc.get("artifact_status") != LIST_STATUS_REGISTERED:
        return f"the committed list {committed.as_posix()} is not a registered build"
    if doc["seed"] != SEED_OF_RECORD or doc["role"] != ROLE_OF_RECORD:
        return f"the committed list {committed.as_posix()} is seed {doc['seed']} ({doc['role']}), not the list of record"
    if doc["checksum_sha256"] != calibration.get("checksum_sha256"):
        return (f"calibrated on list {str(calibration.get('checksum_sha256'))[:16]}..., but the committed list of "
                f"record is {doc['checksum_sha256'][:16]}...")
    return None


# ================================================================== d3: two runs compared
def compare_runs(meta_a, meta_b) -> dict:
    """d3: two PTQ runs from the same checkpoint and list must be bitwise identical.

    Compares the three artifact sha256s the two provenances record, re-hashes every file on disk, and
    checks that the runs share source, list, qconfig, engine and library versions.
    """
    docs, rows = [], []
    if Path(meta_a).resolve().parent == Path(meta_b).resolve().parent:
        raise PTQRefused("compare_same_run", f"{meta_a} and {meta_b} are the same run directory; d3 compares "
                                             "two separate runs")
    for m in (meta_a, meta_b):
        p = Path(m)
        docs.append((p, json.loads(p.read_text(encoding="utf-8"))))
    (pa, a), (pb, b) = docs

    def _artifacts(p, d):
        return {"torchscript": (p.parent / d["converted_artifact"], d["converted_artifact_sha256"]),
                "state_dict": (p.parent / d["state_dict_artifact"]["path"], d["state_dict_artifact"]["sha256"]),
                "x86_copy": (p.parent / d["x86_latency_copy"]["path"], d["x86_latency_copy"]["sha256"])}

    arts_a, arts_b = _artifacts(pa, a), _artifacts(pb, b)
    for k in arts_a:
        (fa, sa), (fb, sb) = arts_a[k], arts_b[k]
        on_disk = (fa.is_file() and fb.is_file() and sha256_file(fa) == sa and sha256_file(fb) == sb)
        rows.append({"artifact": k, "sha256_a": sa, "sha256_b": sb, "identical": sa == sb,
                     "files_match_records": on_disk})
    same = {key: a.get(key) == b.get(key) for key in ("stage", "source_checkpoint_sha256", "engine")}
    same["calibration"] = all(a["calibration"][k] == b["calibration"][k]
                              for k in ("checksum_sha256", "seed", "role", "batch_size", "n_calib"))
    same["qconfig"] = a["qconfig"]["fingerprint"] == b["qconfig"]["fingerprint"]
    same["versions"] = all(a["environment"][k] == b["environment"][k]
                           for k in ("torch", "torchvision", "numpy", "pillow"))
    identical = all(r["identical"] and r["files_match_records"] for r in rows) and all(same.values())
    return {"artifacts": rows, "same_inputs": same, "identical": identical,
            "threads": [a["environment"]["torch_num_threads"], b["environment"]["torch_num_threads"]]}


def student_traced_in_process() -> bool:
    """True once the student was traced in this process. TorchScript then renames its classes, so a
    second trace would not reproduce the first one's bytes (d3); QAT converts one epoch per process."""
    return torch.jit._state._python_cu.get_class(STUDENT_TS_CLASS) is not None


__all__ = ["CALIBRATION_BATCH_SIZE", "CHECKSUM_DEFINITION", "LIST_FILENAME", "LIST_SCHEMA", "LIST_SEEDS",
           "LIST_SHARED_BY", "LIST_PROCEDURE", "LIST_STATUS_REGISTERED", "PTQRefused", "PTQStop",
           "PTQ_RUN_ROLES", "ROLE_OF_RECORD", "ROLE_SENSITIVITY", "SEED_OF_RECORD", "SENSITIVITY_SEEDS",
           "SPLIT_LIST_DEFINITION", "STUDENT_TS_CLASS", "SplitPathRefused",
           "artifact_names", "calibration_dataset", "code_provenance", "compare_runs", "graph_census",
           "ids_checksum", "iter_calibration", "iter_calibration_images", "list_of_record_error", "list_role",
           "load_calibration_list", "load_x86_latency_torchscript", "output_parity", "pixel_agreement",
           "quantize_qnnpack", "quantize_x86_copy", "refuse_eval_split_dir", "refuse_eval_split_file",
           "refuse_test_path", "refuse_val_dir", "run_ptq", "runtime_census", "state_dict_bytes",
           "student_traced_in_process",
           "synthetic_parity_inputs", "torchscript_bytes", "verify_calibration_list",
           "weight_scheme_report", "write_exclusive"]
