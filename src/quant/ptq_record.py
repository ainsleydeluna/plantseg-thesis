"""LQP (L-CKPT-GUARD Q-part, L-AM16-CALIB): the E4/E7 PTQ of record, bound to its parent, and its record.

scripts/run_ptq.py takes this path when its calibration list is a registered build: artifact_status
"registered", which scripts/build_calibration_lists.py writes only for lists drawn from the 5,367-image TRAIN
split list (its :65, :118-120, :137 and :157). It imports this module inside that path only, so a run on a
synthetic list never loads it. The PTQ itself stays src.quant.ptq.run_ptq (L-AM10).

    python -B scripts/run_ptq.py --real-run --confirm-real-run --stage E4 --seed S \\
        --source-run-dir RUN_DIR --expect-source-sha256 SHA256 \\
        --calibration-list configs/calibration/ptq_calibration_seed42.json --expect-list-checksum SHA256 \\
        --expect-head COMMIT --data-root ROOT --out-dir /out/e4_sS_cal42_a1 [--threads N] [--replicate-of RECORD]
    E7 adds  --lambda-selection reports/derived/lambda_selection.json --lambda-selection-sha256 SHA256
             --alpha-selection reports/derived/alpha_selection.json --alpha-selection-sha256 SHA256

Roles: of_record (the seed-42 list, AM-10); calibration_sensitivity (a seed-43/44/45 list at --seed 42, AM-16
item 5); d3_replicate (--replicate-of: a second run of a record's inputs, L-AM10's d3). Only of_record is ever
official (src/eval/stage_artifacts.py int8_official_error); the other two are descriptive.

The gates, in order; each refuses with exit 2 before anything is written, and no image, mask or checkpoint is
opened before G8:
  G2  the list is configs/calibration/ptq_calibration_seed<S>.json, tracked and byte-equal at HEAD
      (calibration_list_not_committed)
  G3  seed_invalid (42, 43, 44); sensitivity_seed (a sensitivity list at a seed other than 42);
      source_run_dir_required; expect_head_format; selection_not_applicable (E4)
  G4  out_dir_relative; output_dir_inside_repo; out_dir_name (<stage>_s<seed>_cal<list seed>_a<n>, with _d3
      exactly when --replicate-of is given); out_dir_not_empty
  G5  one identity reader, ptq.code_provenance(RECORD_CODE_PATHS): code_identity_unavailable,
      expect_head_mismatch, code_not_clean; then image_digest_not_of_record (PLANTSEG_IMAGE_DIGEST)
  G6  data_root_has_test (src.data.isolation.assert_no_test_surfaces: existence only, nothing listed)
  G7  E7: the committed λ and α selections (qat.read_tracked_selection)
  G8  the parent run directory and its best.json checkpoint (qat.resolve_parent)
  G9  E4 <- E1: parent_horizon, parent_payload
  G10 E7 <- E3: qat.e6_parent_binding; parent_not_of_record; parent_teacher_not_of_record;
      parent_checkpoint_not_selected (seed 42); parent_launch_log; parent_payload
  G11 --replicate-of: replicate_mismatch
A reused Q2 refusal keeps its code; its message is led by the stage and the parent rule (ruling R9). Then the
core runs, and <stage>_ptq_record.json (format ptq_record/1) is created exclusively, last, after the run
provenance: STOP identity_changed when the code, the source or the list the run recorded differ from the gates'
(exit 1), STOP record_write_failed when the record cannot be created (exit 1). Without its record a run
directory is not of record. Nothing here reads VAL or TEST.

For the TEST-day manifest lane: read_record, manifest_entry and planned_entry (pure; no TEST path).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
# The committed records the binding reads (the calibration lists, the λ and α selections, the KD launch log) are
# read at HEAD of the repository at RECORDS_ROOT; None is this checkout. The smoke repoints it in-process (as
# sweep_select.GIT_ROOT); no CLI flag redirects it.
RECORDS_ROOT: Path | None = None

RECORD_FORMAT = "ptq_record/1"
RECORD_NAME = "{key}_ptq_record.json"
ROLE_OF_RECORD, ROLE_SENSITIVITY, ROLE_REPLICATE = "of_record", "calibration_sensitivity", "d3_replicate"
ROLES = (ROLE_OF_RECORD, ROLE_SENSITIVITY, ROLE_REPLICATE)
REPLICABLE_ROLES = (ROLE_OF_RECORD, ROLE_SENSITIVITY)
SEEDS = (42, 43, 44)                                # AM-1; AM-17 item 9: E4 and E7 at every seed
SCRIPT_REL = "scripts/run_ptq.py"
MODULE_REL = "src/quant/ptq_record.py"
# The code the bound path runs beyond src/quant/ptq.py's CODE_PATHS: its sha256 goes into the record, and its
# cleanliness at HEAD is a gate (G5). configs/e1_student.py holds the 80,000-iteration horizon the parent rules read.
RECORD_EXTRA_CODE_PATHS = (MODULE_REL, "src/quant/stages.py", "src/quant/qat.py", "src/training/sweep_select.py",
                           "src/data/isolation.py", "scripts/preflight_distill.py",
                           "scripts/preflight_e1_trainval.py", "configs/e1_student.py")
TEACHER_ARCH = "segnext_mscan-b"                     # part3 lane 3's teacher_arch, read as teacher_provenance
ISOLATION_METHOD = ("src.data.isolation.assert_no_test_surfaces: TEST-surface absence; TRAIN mounted for PTQ")
AUTHORITY = ("AM-10 (the list of record, shared by E4 and E7); AM-16 item 5 (calibration sensitivity, seeds 43-45, "
             "descriptive); AM-19 items 4(a)-(c) (parents of record; derived artifacts); ruling Q1-R2 "
             "(L-CKPT-GUARD Q-part); lane L-AM16-CALIB")
RECORD_KEYS = ("format", "stage", "seed", "role", "descriptive", "replicate_of", "authority", "parent",
               "calibration", "ptq_run", "artifacts", "qconfig_fingerprint", "observers", "engine", "environment",
               "isolation", "writer", "generated_utc")
ARTIFACT_KEYS = ("torchscript", "state_dict", "x86_copy")
ENVIRONMENT_KEYS = ("image_digest", "torch", "torchvision", "numpy", "pillow", "torch_num_threads")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class RecordInvalid(ValueError):
    """A file that is not an intact ptq_record/1 (read_record)."""


def _refuse(code: str, message: str):
    from src.quant.ptq import PTQRefused
    raise PTQRefused(code, message)


def _stop(code: str, message: str):
    from src.quant.ptq import PTQStop
    raise PTQStop(code, message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def records_root() -> Path:
    return Path(REPO if RECORDS_ROOT is None else RECORDS_ROOT)


def record_code_paths() -> tuple:
    from src.quant.ptq import CODE_PATHS
    return tuple(CODE_PATHS) + RECORD_EXTRA_CODE_PATHS


def horizon_of_record() -> int:
    """80,000: IMPLEMENTATION_CONTRACT (b) and AM-16 item 3; train_e1.py's default horizon (its :336)."""
    from configs.e1_student import E1_STUDENT
    return int(E1_STUDENT["iterations"])


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# ------------------------------------------------------------------ G2: the committed list
def committed_list(path, doc: dict, root: Path) -> str:
    """The repository path of the list run_ptq reads (the path as given). Refused unless it is
    configs/calibration/ptq_calibration_seed<its seed>.json under `root`, tracked and byte-equal at HEAD."""
    from configs.quant import QUANT
    from src.quant.ptq import LIST_FILENAME
    from src.training import sweep_select as ss
    src = ss.HeadSource(root)
    want = f"{QUANT['ptq']['calibration_list_dir']}/{LIST_FILENAME.format(seed=doc['seed'])}"
    rel, problem = src.rel(Path(path).resolve()), None
    if rel != want:
        problem = f"{path} is {rel or 'outside the repository'}, not {want} in {src.root}"
    else:
        try:
            src.read(rel, missing="missing", uncommitted="uncommitted")
        except ss.SelectionRefused as e:
            problem = f"[{e.code}] {e}"
    if problem is not None:
        _refuse("calibration_list_not_committed", f"a registered list runs only as committed: {problem}")
    return rel


# ------------------------------------------------------------------ G4: the out-dir
def check_out_dir(value: str, key: str, seed: int, list_seed: int, replicate: bool) -> Path:
    from .runner import QuantRunError, check_output_dir
    if not Path(value).is_absolute():
        _refuse("out_dir_relative", f"--out-dir must be absolute: {value!r}")
    try:
        out = check_output_dir(value, create=False)
    except QuantRunError as e:
        raise _as_refusal(e.code, str(e)) from e
    m = re.fullmatch(rf"{key}_s{seed}_cal{list_seed}_a([1-9][0-9]*)(_d3)?", out.name)
    if m is None or (m.group(2) is not None) != replicate:
        _refuse("out_dir_name", f"--out-dir {out.name!r}: this run's directory is named "
                                f"{key}_s{seed}_cal{list_seed}_a<n>" + ("_d3 (a d3 replicate)" if replicate else
                                                                         " (_d3 only with --replicate-of)"))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        _refuse("out_dir_not_empty", f"--out-dir {out} exists and is not an empty directory; a PTQ run starts "
                                     "in a fresh directory")
    return out


def _as_refusal(code: str, message: str):
    from src.quant.ptq import PTQRefused
    return PTQRefused(code, message)


# ------------------------------------------------------------------ G5: identity
def identity(expect_head: str) -> dict:
    """ptq.code_provenance over RECORD_CODE_PATHS: the one identity reader (ruling R7)."""
    from src.quant import ptq as _ptq
    from src.quant.qat import IMAGE_DIGEST_OF_RECORD
    ident = _ptq.code_provenance(record_code_paths())
    if ident["commit"] is None:
        _refuse("code_identity_unavailable", "git reports no HEAD commit for this checkout, so the record could not "
                                             "name the code it ran (a mounted clone owned by another user needs "
                                             "safe.directory)")
    if ident["commit"] != expect_head:
        _refuse("expect_head_mismatch", f"the checkout's HEAD is {ident['commit']}, --expect-head {expect_head}")
    if ident["clean_at_commit"] is not True:
        _refuse("code_not_clean", "the PTQ code paths (RECORD_CODE_PATHS) are not clean at HEAD")
    digest = os.environ.get("PLANTSEG_IMAGE_DIGEST", "").strip()
    if digest != IMAGE_DIGEST_OF_RECORD:
        _refuse("image_digest_not_of_record", f"PLANTSEG_IMAGE_DIGEST is {digest or 'unset'}; the PTQ of record "
                                              f"runs in the student image {IMAGE_DIGEST_OF_RECORD} (DL-21)")
    return ident


# ------------------------------------------------------------------ G6: the data root
def isolation(data_root) -> dict:
    from src.data.isolation import TrainValIsolationError, assert_no_test_surfaces
    try:
        result = assert_no_test_surfaces(Path(data_root))
    except TrainValIsolationError as e:
        raise _as_refusal("data_root_has_test", f"[{e.code}] {e}") from e
    return {"data_root": str(data_root), "method": ISOLATION_METHOD, "result": result}


# ------------------------------------------------------------------ G8-G10: the parent rules
def _q2(prefix: str, fn, *a):
    """A Q2 function; its refusal keeps its code, its message led by the stage and the rule (ruling R9)."""
    from src.quant.qat import QATRefused
    try:
        return fn(*a)
    except QATRefused as e:
        raise _as_refusal(e.code, f"{prefix}{e}") from e


def _run_meta(parent: dict) -> dict:
    from src.quant.qat import _run_meta_rows
    return _run_meta_rows(Path(parent["run_dir"]) / parent["run_meta_file"])[0]


def _payload(parent: dict):
    import torch
    data = Path(parent["checkpoint_path"]).read_bytes()
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def _parent_fields(parent: dict, meta: dict) -> dict:
    return {"run_id": parent["run_id"], "run_dir": parent["run_dir"],
            "best_json_sha256": parent["best_json_sha256"],
            "best_val_miou_all_class": parent["best_val_miou_all_class"], "checkpoint": parent["checkpoint"],
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "checkpoint_bytes": Path(parent["checkpoint_path"]).stat().st_size,
            "run_meta": {"file": parent["run_meta_file"], "sha256": parent["run_meta_sha256"]},
            "telemetry": {"file": parent["telemetry"], "sha256": parent["telemetry_sha256"]},
            "mode": parent["mode"], "seed": parent["seed"], "max_iters": meta.get("max_iters"),
            "poly_horizon": meta.get("poly_horizon")}


def bind_e1(ctx: dict) -> dict:
    """E4 <- E1 (AM-19 item 4(a), (b)): the E1 run of record is the 80,000-iteration run, its best.json
    checkpoint the one it names. An absent or null poly_horizon is train_e1's default 80,000 (ruling R3)."""
    parent, prefix, h = ctx["parent"], ctx["prefix"], horizon_of_record()
    meta = _run_meta(parent)
    mi, ph = meta.get("max_iters"), meta.get("poly_horizon")
    if not (_is_int(mi) and mi == h) or not (ph is None or (_is_int(ph) and ph == h)):
        _refuse("parent_horizon", f"{prefix}the parent ran max_iters {mi!r}, poly_horizon {ph!r}; the E1 run of "
                                  f"record runs {h} iterations on the {h} horizon")
    payload = _payload(parent)
    bad = (["not a checkpoint dict"] if not isinstance(payload, dict) else
           ([f"stage {payload.get('stage')!r}"] if payload.get("stage") not in (None, "E1") else [])
           + ([f"best_val_miou_all_class {payload.get('best_val_miou_all_class')!r} != best.json's "
               f"{parent['best_val_miou_all_class']!r}"]
              if payload.get("best_val_miou_all_class") != parent["best_val_miou_all_class"] else []))
    if bad:
        _refuse("parent_payload", f"{prefix}the checkpoint best.json names is not that E1 run's: {bad}")
    return {"rule": ctx["rule"]["id"], "stage": "E1", **_parent_fields(parent, meta)}


def bind_e3(ctx: dict) -> dict:
    """E7 <- E3 (AM-19 item 4(a), (b); AM-18 item 3(d)): the E3 run the λ and α selections name, on the
    80,000-iteration horizon, eligible as a parent of E4-E7, distilled from the teacher of record, launched as
    the committed KD launch log records, its checkpoint the one best.json names (and, at seed 42, the one the α
    selection chose)."""
    import scripts.preflight_distill as pd                      # the teacher of record (DL-88; ruling R17)
    from src.quant import qat
    parent, prefix, args, h = ctx["parent"], ctx["prefix"], ctx["args"], horizon_of_record()
    lam, alpha = ctx["selections"]
    _q2(prefix, qat.e6_parent_binding, parent, args.seed, lam, alpha)
    meta = _run_meta(parent)
    not_of_record = [k for k, ok in (("parent_of_e4_e7", meta.get("parent_of_e4_e7") is True),
                                     ("arm", meta.get("arm") is None),
                                     ("descriptive", meta.get("descriptive") is False),
                                     ("max_iters", _is_int(meta.get("max_iters")) and meta.get("max_iters") == h),
                                     ("poly_horizon", _is_int(meta.get("poly_horizon"))
                                      and meta.get("poly_horizon") == h),
                                     ("target_construction", "target_construction" not in meta)) if not ok]
    if not_of_record:
        _refuse("parent_not_of_record", f"{prefix}the parent's run_meta is not an E3 run of record: "
                                        f"{ {k: meta.get(k) for k in not_of_record} }")
    tp = meta.get("teacher_provenance")
    if (meta.get("teacher_mock") is not False or not isinstance(tp, dict) or tp.get("builder") != pd.TEACHER_BUILDER
            or tp.get("ckpt_sha256") != pd.TEACHER_SHA256 or tp.get("expected_sha256") != pd.TEACHER_SHA256):
        _refuse("parent_teacher_not_of_record", f"{prefix}the parent was not distilled from the teacher of record "
                                                f"{pd.TEACHER_FILE} ({pd.TEACHER_SHA256[:12]}..., "
                                                f"{pd.TEACHER_BUILDER}): teacher_mock "
                                                f"{meta.get('teacher_mock')!r}, teacher_provenance "
                                                f"{ {k: (tp or {}).get(k) for k in ('builder', 'ckpt_sha256')} }")
    if args.seed == 42 and alpha["winner"].get("ckpt_sha256") != parent["checkpoint_sha256"]:
        _refuse("parent_checkpoint_not_selected", f"{prefix}the α selection chose checkpoint "
                                                  f"{alpha['winner'].get('ckpt_sha256')}, best.json names "
                                                  f"{parent['checkpoint_sha256']}")
    launch = launch_log_binding(parent, meta, ctx["root"], prefix)
    payload = _payload(parent)
    bad = (["not a checkpoint dict"] if not isinstance(payload, dict) else
           ([f"stage {payload.get('stage')!r}"] if payload.get("stage") != "E3" else [])
           + [f"{k} key" for k in ("arm", "target_construction") if k in payload]
           + (["teacher_provenance differs from run_meta's"] if payload.get("teacher_provenance") != tp else [])
           + ([f"best_val_miou_all_class {payload.get('best_val_miou_all_class')!r} != best.json's "
               f"{parent['best_val_miou_all_class']!r}"]
              if payload.get("best_val_miou_all_class") != parent["best_val_miou_all_class"] else []))
    if bad:
        _refuse("parent_payload", f"{prefix}the checkpoint best.json names is not that E3 run's: {bad}")
    rows, _torn = qat.read_jsonl(Path(parent["run_dir"]) / parent["telemetry"])
    end = [r for r in rows if r.get("event") == "run_end"][-1]
    return {"rule": ctx["rule"]["id"], "stage": "E3", **_parent_fields(parent, meta),
            "lambda_logit": meta.get("lambda_logit"), "alpha_cwd": meta.get("alpha_cwd"),
            "parent_of_e4_e7": True, "teacher_arch": TEACHER_ARCH, "teacher_ckpt_sha256": tp["ckpt_sha256"],
            "run_end": {k: end.get(k) for k in ("iter", "checks_passed", "wall_clock_end", "best_ckpt")},
            "selections": {"lambda": lam, "alpha": alpha}, "launch_log": launch}


def launch_log_binding(parent: dict, meta: dict, root: Path, prefix: str) -> dict:
    """reports/kd_launch_log.jsonl at HEAD: the parent's launched line, no stopped line, and the fields
    sweep_select.check_run_against_log compares (sweep_select.py:1800-1818; ruling R8). That function takes a
    sweep Attempt, which no seed-43/44 line makes (sweep None, value None), so its fields are mirrored here."""
    from src.training import sweep_select as ss
    try:
        log = ss.load_launch_log(ss.HeadSource(root))
    except ss.SelectionRefused as e:
        raise _as_refusal("parent_launch_log", f"{prefix}[{e.code}] {e}") from e
    rid = parent["run_id"]
    line, ld = log.launch.get(rid), log.launched.get(rid)
    if line is None or ld is None:
        _refuse("parent_launch_log", f"{prefix}{ss.LAUNCH_LOG_REL} at HEAD has no launched line for {rid}")
    if rid in log.stopped:
        _refuse("parent_launch_log", f"{prefix}{ss.LAUNCH_LOG_REL} holds a stopped line for {rid}")
    if ld["run_meta_sha256"] != parent["run_meta_sha256"]:
        _refuse("parent_launch_log", f"{prefix}the launched line's run_meta_sha256 {ld['run_meta_sha256']} != the "
                                     f"directory's {parent['run_meta_file']} {parent['run_meta_sha256']}")
    fields = [("git_head", meta.get("git_head"), ld["git_head"]), ("code_pin", ld["git_head"], line["code_pin"]),
              ("records_commit", meta.get("records_commit"), ld["records_commit"]),
              ("stage", meta.get("stage"), line["stage"]), ("seed", meta.get("seed"), line["seed"]),
              ("horizon", meta.get("poly_horizon"), line["horizon"])]
    if line["sweep"] is not None:                                   # the swept key (None ⇔ value None, :910-912)
        fields.append((line["sweep"], meta.get(line["sweep"]), line["value"]))
    fields += [("lambda_logit", meta.get("lambda_logit"), line["lambda_logit"]),
               ("alpha_cwd", meta.get("alpha_cwd"), line["alpha_cwd"]),
               ("launch_time_utc", meta.get("wall_clock"), ld["launch_time_utc"])]
    bad = [(f, got, want) for f, got, want in fields if not ss._same(got, want)]
    if bad:
        _refuse("parent_launch_log", f"{prefix}run_meta disagrees with its launch-log lines: {bad} (run_meta, log)")
    return {"path": ss.LAUNCH_LOG_REL, "sha256": log.sha256, "launch": line, "launched": ld}


PARENT_RULES = {
    "E4": {"id": "e1_run_of_record/1", "source_stage": "E1", "selections": False, "bind": bind_e1},
    "E7": {"id": "e3_run_of_record/1", "source_stage": "E3", "selections": True, "bind": bind_e3},
}


# ------------------------------------------------------------------ G11: the d3 replicate
def replicate_binding(path, stage: str, seed: int, doc: dict, parent: dict, out: Path) -> dict:
    try:
        rec, sha = read_record(path)
    except (OSError, ValueError) as e:
        raise _as_refusal("replicate_mismatch", f"--replicate-of {path}: {e}") from e
    bad = [k for k, ok in (("role", rec["role"] in REPLICABLE_ROLES), ("stage", rec["stage"] == stage),
                           ("seed", rec["seed"] == seed),
                           ("list", rec["calibration"].get("checksum_sha256") == doc["checksum_sha256"]),
                           ("parent", rec["parent"].get("checkpoint_sha256") == parent["checkpoint_sha256"]),
                           ("directory", f"{Path(path).resolve().parent.name}_d3" == out.name)) if not ok]
    if bad:
        _refuse("replicate_mismatch", f"--replicate-of {path} is not this run's original (it differs in {bad}); "
                                      "a d3 replicate repeats an of-record or sensitivity record's stage, seed, "
                                      "list and parent into <its directory>_d3")
    return {"path": str(Path(path).resolve()), "sha256": sha}


# ------------------------------------------------------------------ the bound run
def bound_gates(args, doc: dict, list_sha256: str) -> dict:
    """G2-G11, in order. Nothing is written; no image, mask or checkpoint is opened before G8."""
    from src.quant.ptq import SEED_OF_RECORD
    from src.quant.qat import ALPHA_SELECTION_FORMAT, LAMBDA_SELECTION_FORMAT, read_tracked_selection
    from src.quant.stages import resolve_quant_stage
    st = resolve_quant_stage(args.stage)
    rule, root = PARENT_RULES[st["name"]], records_root()
    prefix = f"{st['name']} parent rule ({rule['id']}): "
    list_rel = committed_list(args.calibration_list, doc, root)                                       # G2
    if args.seed not in SEEDS:                                                                         # G3
        _refuse("seed_invalid", f"--seed {args.seed!r} is not one of {SEEDS}")
    if doc["seed"] != SEED_OF_RECORD and args.seed != SEED_OF_RECORD:
        _refuse("sensitivity_seed", f"a calibration-sensitivity list (seed {doc['seed']}) calibrates E4 and E7 at "
                                    f"seed {SEED_OF_RECORD} only (AM-16 item 5), not seed {args.seed}")
    if not args.source_run_dir:
        _refuse("source_run_dir_required", "a registered list binds the run to its parent: pass --source-run-dir "
                                           "(the run directory whose best.json names the checkpoint), not "
                                           "--source-ckpt")
    if not args.expect_head or not _HEX40.match(args.expect_head):
        _refuse("expect_head_format", "--expect-head must be the 40-hex commit of the checkout")
    selection_flags = (args.lambda_selection, args.lambda_selection_sha256, args.alpha_selection,
                       args.alpha_selection_sha256)
    if not rule["selections"] and any(v is not None for v in selection_flags):
        _refuse("selection_not_applicable", f"{st['name']} takes no λ or α selection file")
    out = check_out_dir(args.out_dir, st["key"], args.seed, doc["seed"], args.replicate_of is not None)  # G4
    ident = identity(args.expect_head)                                                                 # G5
    iso = isolation(args.data_root)                                                                    # G6
    selections = None
    if rule["selections"]:                                                                             # G7
        selections = (_q2(prefix, read_tracked_selection, args.lambda_selection, args.lambda_selection_sha256,
                          LAMBDA_SELECTION_FORMAT, "lambda", root),
                      _q2(prefix, read_tracked_selection, args.alpha_selection, args.alpha_selection_sha256,
                          ALPHA_SELECTION_FORMAT, "alpha", root))
    from src.quant.qat import resolve_parent
    parent = _q2(prefix, resolve_parent, st, args.source_run_dir, args.expect_source_sha256, args.seed)  # G8
    ctx = {"args": args, "rule": rule, "parent": parent, "prefix": prefix, "selections": selections, "root": root}
    bound_parent = rule["bind"](ctx)                                                                   # G9/G10
    replicate = None
    if args.replicate_of is not None:                                                                  # G11
        replicate = replicate_binding(args.replicate_of, st["name"], args.seed, doc, parent, out)
    role = (ROLE_REPLICATE if replicate is not None
            else ROLE_OF_RECORD if doc["seed"] == SEED_OF_RECORD else ROLE_SENSITIVITY)
    return {"stage": st, "seed": args.seed, "rule": rule, "list_rel": list_rel, "doc": doc,
            "list_sha256": list_sha256, "out": out, "identity": ident, "isolation": iso, "parent": parent,
            "bound_parent": bound_parent, "replicate": replicate, "role": role}


def run_bound(args, doc: dict, list_sha256: str, *, log=print) -> tuple[dict, dict, Path, str]:
    """The gates, src.quant.ptq.run_ptq on the parent's checkpoint, then the record. Returns (run provenance,
    record, record path, record sha256); raises PTQRefused (exit 2) or PTQStop (exit 1)."""
    from src.quant import ptq as _ptq
    g = bound_gates(args, doc, list_sha256)
    prov = _ptq.run_ptq(stage=g["stage"]["name"], source_ckpt=g["parent"]["checkpoint_path"],
                        expect_source_sha256=args.expect_source_sha256, calibration_list=args.calibration_list,
                        expect_list_checksum=args.expect_list_checksum, data_root=args.data_root,
                        out_dir=str(g["out"]), threads=args.threads, log=log)
    record = build_record(g, prov)
    path, sha = write_record(g["out"], g["stage"]["key"], record)
    return prov, record, path, sha


def build_record(g: dict, prov: dict) -> dict:
    """ptq_record/1. STOP identity_changed when the run provenance's code, source or list differ from what the
    gates bound, or the code paths changed during the run."""
    from src.quant import ptq as _ptq
    ident, names = g["identity"], _ptq.artifact_names(g["stage"]["key"])
    after = _ptq.code_provenance(record_code_paths())
    code, cal = prov.get("code") or {}, prov.get("calibration") or {}
    changed = [k for k, ok in (
        ("code commit", code.get("commit") == ident["commit"] and code.get("clean_at_commit") is True),
        ("code files", code.get("files_sha256") == {p: s for p, s in ident["files_sha256"].items()
                                                    if p in _ptq.CODE_PATHS}),
        ("code paths after the run", after == ident),
        ("source checkpoint", prov.get("source_checkpoint_sha256") == g["parent"]["checkpoint_sha256"]),
        ("calibration list", cal.get("list_file_sha256") == g["list_sha256"]
         and cal.get("checksum_sha256") == g["doc"]["checksum_sha256"])) if not ok]
    if changed:
        _stop("identity_changed", f"the run provenance does not match what the gates bound ({changed}); no "
                                  f"record was written, so {g['out']} is not of record")
    out = g["out"]
    arts = {"torchscript": (prov["converted_artifact"], prov["converted_artifact_sha256"]),
            "state_dict": (prov["state_dict_artifact"]["path"], prov["state_dict_artifact"]["sha256"]),
            "x86_copy": (prov["x86_latency_copy"]["path"], prov["x86_latency_copy"]["sha256"])}
    role = g["role"]
    return {
        "format": RECORD_FORMAT, "stage": g["stage"]["name"], "seed": g["seed"], "role": role,
        "descriptive": role != ROLE_OF_RECORD,
        "replicate_of": g["replicate"], "authority": AUTHORITY, "parent": g["bound_parent"],
        "calibration": {"list_path": g["list_rel"], "list_file_sha256": g["list_sha256"],
                        "checksum_sha256": g["doc"]["checksum_sha256"], "seed": g["doc"]["seed"],
                        "role": g["doc"]["role"], "list_artifact_status": g["doc"].get("artifact_status")},
        "ptq_run": {"run_meta": names["run_meta"], "sha256": _ptq.sha256_file(out / names["run_meta"]),
                    "ptq_run_role": prov.get("ptq_run_role"), "schema": prov.get("schema")},
        "artifacts": {k: {"path": p, "sha256": s, "bytes": (out / p).stat().st_size} for k, (p, s) in arts.items()},
        "qconfig_fingerprint": (prov.get("qconfig") or {}).get("fingerprint"), "observers": prov.get("observers"),
        "engine": prov.get("engine"),
        "environment": {k: (prov.get("environment") or {}).get(k) for k in ENVIRONMENT_KEYS},
        "isolation": g["isolation"],
        "writer": {"script": {"path": SCRIPT_REL, "sha256": ident["files_sha256"].get(SCRIPT_REL)},
                   "module": {"path": MODULE_REL, "sha256": ident["files_sha256"].get(MODULE_REL)},
                   "commit": ident["commit"], "clean_at_commit": ident["clean_at_commit"],
                   "code_sha256": ident["files_sha256"]},
        "generated_utc": _ptq._utc_now(),
    }


def write_record(out: Path, key: str, record: dict) -> tuple[Path, str]:
    """Create <stage>_ptq_record.json exclusively, last (STOP record_write_failed)."""
    from src.quant.ptq import _json_bytes
    path = out / RECORD_NAME.format(key=key)
    data = _json_bytes(record)
    try:
        with open(path, "xb") as fh:
            fh.write(data)
    except OSError as e:
        _stop("record_write_failed", f"{path} could not be created ({type(e).__name__}: {e}); the run directory "
                                     f"{out} is not of record")
    return path, _sha256_bytes(data)


# ------------------------------------------------------------------ readers (the official gate, the manifest lane)
def _strict_json(data: bytes):
    def _no_constant(c):
        raise RecordInvalid(f"non-finite JSON constant {c!r}")
    return json.loads(data.decode("utf-8"), parse_constant=_no_constant)


def read_record(path, expect_sha256: str | None = None) -> tuple[dict, str]:
    """(record, sha256) of an intact ptq_record/1: the format, its keys and role, and every file it lists
    (the three artifacts and the run provenance, beside it) re-hashed. Raises RecordInvalid or OSError."""
    p = Path(path)
    data = p.read_bytes()
    sha = _sha256_bytes(data)
    if expect_sha256 is not None and sha != expect_sha256:
        raise RecordInvalid(f"{p} has sha256 {sha}, expected {expect_sha256}")
    try:
        rec = _strict_json(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RecordInvalid(f"{p} is not JSON ({e})") from e
    if not isinstance(rec, dict) or rec.get("format") != RECORD_FORMAT or sorted(rec) != sorted(RECORD_KEYS):
        raise RecordInvalid(f"{p} is not a {RECORD_FORMAT} record")
    if rec["role"] not in ROLES or rec["descriptive"] is not (rec["role"] != ROLE_OF_RECORD):
        raise RecordInvalid(f"{p}: role {rec['role']!r}, descriptive {rec['descriptive']!r}")
    try:
        files = {k: (str(rec["artifacts"][k]["path"]), rec["artifacts"][k]["sha256"]) for k in ARTIFACT_KEYS}
        files["run_meta"] = (str(rec["ptq_run"]["run_meta"]), rec["ptq_run"]["sha256"])
        for k in ("parent", "calibration", "writer", "environment"):
            if not isinstance(rec[k], dict):
                raise TypeError(f"{k} is not an object")
    except (KeyError, TypeError) as e:
        raise RecordInvalid(f"{p}: malformed {RECORD_FORMAT} ({type(e).__name__}: {e})") from e
    for k, (name, want) in files.items():
        f = p.parent / PurePosixPath(name).name
        if not f.is_file() or _sha256_bytes(f.read_bytes()) != want:
            raise RecordInvalid(f"{p}: its {k} {name} is absent or differs from its recorded sha256")
    return rec, sha


def manifest_entry(record: dict) -> dict:
    """AM-19 item 4(c): a derived artifact as the TEST-day manifest lists it, by its inputs and its outputs."""
    return {"status": "made", "stage": record["stage"], "seed": record["seed"], "role": record["role"],
            "parent_checkpoint_sha256": record["parent"]["checkpoint_sha256"],
            "list_file_sha256": record["calibration"]["list_file_sha256"],
            "list_checksum_sha256": record["calibration"]["checksum_sha256"],
            "ptq_commit": record["writer"]["commit"], "ptq_script": record["writer"]["script"]["path"],
            "image_digest": record["environment"]["image_digest"],
            "qconfig_fingerprint": record["qconfig_fingerprint"],
            "artifacts": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in record["artifacts"].items()}}


def planned_entry(stage: str, seed: int, parent_checkpoint_sha256: str, list_file_sha256: str,
                  list_checksum_sha256: str, ptq_commit: str, image_digest: str) -> dict:
    """An E4/E7 not yet made at the freeze, listed by the same inputs; its qconfig fingerprint is the registered
    PTQ qconfig's (src/quant/ptq.py qconfig_summary)."""
    from src.quant.ptq import ptq_qconfig, qconfig_summary
    return {"status": "planned", "stage": str(stage).upper(), "seed": seed, "role": ROLE_OF_RECORD,
            "parent_checkpoint_sha256": parent_checkpoint_sha256, "list_file_sha256": list_file_sha256,
            "list_checksum_sha256": list_checksum_sha256, "ptq_commit": ptq_commit, "ptq_script": SCRIPT_REL,
            "image_digest": image_digest, "qconfig_fingerprint": qconfig_summary(ptq_qconfig())["fingerprint"],
            "artifacts": None}


__all__ = ["AUTHORITY", "ISOLATION_METHOD", "PARENT_RULES", "RECORDS_ROOT", "RECORD_EXTRA_CODE_PATHS",
           "RECORD_FORMAT", "RECORD_NAME", "ROLES", "ROLE_OF_RECORD", "ROLE_REPLICATE", "ROLE_SENSITIVITY",
           "RecordInvalid", "SEEDS", "bound_gates", "build_record", "manifest_entry", "planned_entry",
           "read_record", "record_code_paths", "run_bound", "write_record"]
