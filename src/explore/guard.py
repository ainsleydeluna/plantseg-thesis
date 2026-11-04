"""E0: the guard every exploratory entry point runs first (lane L-AM24; audit EXPLORE-AUDIT-1 §5.2 clauses 2, 4, 7).

`guarded_start` runs these steps in order and writes nothing until every one has passed:

  1. the item ID is one of ITEMS ([item_unknown]);
  2. the split: "test" in any case is refused first ([split_test]); then any split outside the tool's allowed set,
     at most TRAIN and VAL, is refused ([split_not_trainval]). Scoring tools allow VAL only;
  3. the run ID starts with "x-" ([run_id_prefix]), holds only letters, digits, ".", "_" and "-"
     ([run_id_format]) and no "test" ([run_id_test]);
  4. every input path the entry point names holds no "test" (SL-1), as given and as resolved, checked before any
     other filesystem call on it ([input_path_test]);
  5. the output directory is the item directory <root>/<run ID>: outside the repository ([out_dir_in_repo]); its
     last component is the run ID ([out_dir_not_run_id]); no other component names an official run or record
     folder (OFFICIAL_DIR_RE, x- folders excepted) and no ancestor holds an official run's records
     (OFFICIAL_RECORD_NAMES, existence checks only) ([out_dir_official]); it is not inside an input's directory
     ([out_dir_inside_input]); it is absent or an empty directory ([out_dir_not_fresh]);
  6. the data root the pinned loaders read (configs/data.py DATA["root"], i.e. PLANTSEG_DATA_ROOT) passes
     src.data.isolation.assert_trainval_only_root with the TRAIN/VAL counts of record before any data is read
     ([data_root_not_trainval_only]). A records-only tool (N4) reads no dataset; it skips this step and records why;
  7. x_run_meta.json is written exclusively into the item directory: exploratory true, amendment AM-24, split val,
     artifact_status exploratory, the item, the run ID, the full recipe, the claim line of clause 7, the git commit,
     the sha256 of every pinned module the item imports (src/quant/ptq.py code_provenance, as the QAT runs of
     record record their code_files_sha256), for the KD items the comparison with the KD pin's
     reused_module_hashes (scripts/preflight_distill.py TEACHER_VALUES), and the sha256 of every lane file.

The pinned trainers create their own run directory inside the item directory (named after the run ID, or for a
feature-map CWD run after the run ID and its alpha token), because each refuses a non-empty directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LANE = "L-AM24"
AMENDMENT = "AM-24"
RULING = "ORCH-5 EXPLORE-2 (audit EXPLORE-AUDIT-1); addendum L-AM24-ADD-1 (MVA-AUDIT-1)"
RUN_PREFIX = "x-"
ARTIFACT_STATUS = "exploratory"
EVAL_SPLIT = "val"
TRAINVAL = ("train", "val")
VAL_ONLY = ("val",)
RUN_META_NAME = "x_run_meta.json"
RUN_META_FORMAT = "x_run_meta/1"
EXPECTED_COUNTS = {"train": 5367, "val": 846}          # configs/data.py SPLIT_SIZES (TRAIN, VAL)
VAL_IMAGES = 846

# The items of AM-24 this lane builds (audit §2; addendum L-AM24-ADD-1). "A1pp" is A1''.
ITEMS = {
    "N4": "free decompositions from existing records (CPU; no dataset, no model)",
    "B1": "D-1 conversion-gap split: the ten sites in turn, all ten, TF32 off, reconciled (CPU or GPU, VAL)",
    "C1": "E6-E2 at seeds 43 and 44 (E6-E2's recipe of record, QAT seed = source seed)",
    "N1": "E6-E2-KD: E6-E2 with logit KD in the QAT loss (lambda 0.5, T 4; no CWD term)",
    "N2": "E6-E2-M: E6-E2 with the ten sites fake-quantized on their input's grid and TF32 off",
    "N5": "E6-E2 with an FP32 LR-ASPP logit head: a MIXED-PRECISION artifact, never called INT8",
    "D": "E6-E2's recipe of record on an exploratory FP32 winner (trigger: best VAL >= 0.3875)",
    "N3": "E2-R2: stage e2 (logit KD, lambda 0.5, T 4) from arm teacher R2 at its M12 iteration 24000",
    "A1pp": "A1'': E3's three terms with alpha = lambda * mean(L_logitKD) / mean(L_feat) from N4",
    "A2": "DIST (beta = gamma = 1, tau = 1, weight 1) in place of lambda * KL only, on E2's grid and mask",
    "KL3": "KL-3 parity: x86 fbgemm latency copies against their QNNPACK artifacts on VAL (DL-181)",
}
ITEM_DISPLAY = {"A1pp": "A1''"}

CLAIM_LINE = "exploratory; VAL (846 images); seed {seeds}; not pre-registered; not evaluated on TEST"

RUN_ID_RE = re.compile(r"x-[A-Za-z0-9][A-Za-z0-9._-]{0,150}")
# A folder name of an official run or record (the records' run IDs and folders): a stage token (e1..e7, with an
# optional -suffix such as e6-e2 or e3-r2) followed by "_", "-" or the end; teacher-*, kd_*, qat_*, ptq_*, valfq_*,
# keep_*, records_*, ev_* and tday_*. x- folders are never matched (they are checked first).
OFFICIAL_DIR_RE = re.compile(r"^(?:e[1-7](?:-[a-z0-9]+)?(?:[_.-]|$)|(?:teacher|kd|qat|ptq|valfq|keep|records|ev|tday)"
                             r"[_-])", re.IGNORECASE)
# Files and folders that mark a directory as an official run's or record's (existence checks only).
OFFICIAL_RECORD_NAMES = ("best.json", "qat_telemetry.jsonl", "e1_telemetry.jsonl", "e2_run_meta.jsonl",
                         "e3_run_meta.jsonl", "a_run_meta.jsonl", "f_run_meta.jsonl", "g_run_meta.jsonl",
                         "e2_telemetry.jsonl", "e3_telemetry.jsonl", "qat_selection.json", "qat_finalize.json",
                         "qat_epoch_eval.json", "qat_convert.json", "epoch_ckpts", "MANIFEST.sha256",
                         "tday_manifest.json", "projection.pt")

EXIT_OK, EXIT_STOP, EXIT_REFUSED, EXIT_ABORTED, EXIT_ERROR = 0, 1, 2, 3, 4


class GuardError(RuntimeError):
    """A refusal of the exploratory guard. `code` names the step, so each guard is testable alone."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def names_test(value) -> bool:
    """SL-1: a string holding "test" in any case."""
    return "test" in str(value).lower()


# ------------------------------------------------------------------ steps 1-4
def check_item(item: str) -> str:
    if item not in ITEMS:
        raise GuardError("item_unknown", f"item {item!r} is not one of {sorted(ITEMS)}")
    return item


def refuse_split(split, *, allowed: Iterable[str] = TRAINVAL) -> str:
    """Step 2: TEST first, by name in any case, then anything outside `allowed` (a subset of TRAIN/VAL)."""
    allowed = tuple(allowed)
    if not set(allowed) <= set(TRAINVAL) or not allowed:
        raise GuardError("split_allowed_set", f"allowed splits {allowed} must be a non-empty subset of {TRAINVAL}")
    if split is None or names_test(split):
        raise GuardError("split_test", f"--split {split!r}: TEST is never read by exploratory code (AM-24 clause 2)")
    if split not in allowed:
        raise GuardError("split_not_trainval", f"--split {split!r} is not one of {allowed}")
    return split


def check_run_id(run_id) -> str:
    if not isinstance(run_id, str) or not run_id.startswith(RUN_PREFIX):
        raise GuardError("run_id_prefix", f"run ID {run_id!r} does not start with {RUN_PREFIX!r} (AM-24 clause 7)")
    if names_test(run_id):
        raise GuardError("run_id_test", f"run ID {run_id!r} holds 'test' (SL-1)")
    if not RUN_ID_RE.fullmatch(run_id):
        raise GuardError("run_id_format", f"run ID {run_id!r}: x- then letters, digits, '.', '_' or '-' only")
    return run_id


def check_input_paths(paths: Mapping[str, object]) -> dict:
    """Step 4, SL-1: each named input path (None skipped) holds no "test", as given and as resolved. Returns the
    resolved paths."""
    out = {}
    for flag, value in paths.items():
        if value is None:
            continue
        if names_test(value):
            raise GuardError("input_path_test", f"{flag} {value!r} holds 'test' (SL-1); refused before any read")
        resolved = os.path.realpath(os.path.expanduser(str(value)))
        if names_test(resolved):
            raise GuardError("input_path_test", f"{flag} {value!r} resolves to {resolved!r}, which holds 'test'")
        out[flag] = resolved
    return out


# ------------------------------------------------------------------ step 5
def _official_components(parts: list[str]) -> list[str]:
    return [p for p in parts if p and not p.startswith(RUN_PREFIX) and OFFICIAL_DIR_RE.match(p)]


def out_dir_error(out_dir, run_id: str, *, inputs: Iterable[str] = ()) -> tuple[str, str] | None:
    """Step 5 as (code, text), or None when the item directory may be written."""
    if out_dir is None or str(out_dir) == "":
        return "out_dir_required", "an explicit --out-dir is required"
    if names_test(out_dir):
        return "out_dir_test", f"--out-dir {out_dir!r} holds 'test' (SL-1)"
    try:
        p = Path(os.path.expanduser(str(out_dir))).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as e:
        return "out_dir_unresolvable", f"--out-dir {out_dir!r} cannot be resolved ({type(e).__name__}: {e})"
    repo = REPO.resolve()
    if p == repo or repo in p.parents:
        return "out_dir_in_repo", f"--out-dir {p} is inside the repository {repo}: run outputs never live there"
    if p.name != run_id:
        return "out_dir_not_run_id", f"--out-dir {p}: its last component must be the run ID {run_id!r}"
    bad = _official_components(list(p.parts[:-1]))
    if bad:
        return "out_dir_official", f"--out-dir {p} lies under official run or record folder(s) {bad}"
    for anc in p.parents:
        if anc.name.startswith(RUN_PREFIX):
            continue
        marks = [n for n in OFFICIAL_RECORD_NAMES if os.path.lexists(anc / n)]
        if marks:
            return "out_dir_official", f"--out-dir {p}: its ancestor {anc} holds official run records {marks}"
    for inp in inputs:
        if inp is None:
            continue
        q = Path(inp).resolve(strict=False)
        base = q if q.is_dir() else q.parent
        if p == base or base in p.parents:
            return "out_dir_inside_input", f"--out-dir {p} is inside the input location {base}: inputs are read-only"
    if p.exists() and (not p.is_dir() or any(p.iterdir())):
        return "out_dir_not_fresh", f"--out-dir {p} exists and is not an empty directory"
    return None


def check_out_dir(out_dir, run_id: str, *, inputs: Iterable[str] = ()) -> Path:
    err = out_dir_error(out_dir, run_id, inputs=inputs)
    if err is not None:
        raise GuardError(*err)
    return Path(os.path.expanduser(str(out_dir))).resolve(strict=False)


# ------------------------------------------------------------------ step 6
def data_root() -> str:
    """The root the pinned loaders read: configs/data.py DATA["root"] (PLANTSEG_DATA_ROOT when it is set)."""
    from configs.data import DATA
    return str(DATA["root"])


def check_data_root(*, records_only: bool = False) -> dict:
    """Step 6: src.data.isolation.assert_trainval_only_root on the loaders' root, before any data is read."""
    if records_only:
        return {"checked": False, "reason": "records-only tool: no dataset is read, so no data root is opened",
                "data_root": None}
    root = data_root()
    if names_test(root):
        raise GuardError("data_root_test_path", f"the data root {root!r} holds 'test' (SL-1)")
    from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
    try:
        iso = assert_trainval_only_root(Path(root), dict(EXPECTED_COUNTS))
    except TrainValIsolationError as e:
        raise GuardError("data_root_not_trainval_only", f"[{e.code}] {e}") from e
    return {"checked": True, "data_root": root, "env_PLANTSEG_DATA_ROOT": os.environ.get("PLANTSEG_DATA_ROOT"),
            **iso}


# ------------------------------------------------------------------ step 7
def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def lane_files() -> list[str]:
    """The lane's own files (src/explore and scripts/*explore*), repository-relative, sorted."""
    files = sorted(p for p in (REPO / "src" / "explore").glob("*.py"))
    files += sorted(p for p in (REPO / "scripts").glob("explore_*.py"))
    files += sorted(p for p in (REPO / "scripts").glob("smoke_explore_*.py"))
    return [p.relative_to(REPO).as_posix() for p in files if not names_test(p.name)]


def code_record(pinned_paths: Iterable[str]) -> dict:
    """The commit, the pinned modules' sha256 (code_provenance, the QAT runs' own record) and the lane's files."""
    from src.quant.ptq import code_provenance
    pinned = [p for p in dict.fromkeys(pinned_paths) if not names_test(p)]
    prov = code_provenance(pinned)
    lane = lane_files()
    lane_prov = code_provenance(lane)
    return {"git_head": prov["commit"], "pinned_clean_at_head": prov["clean_at_commit"],
            "pinned_files_sha256": prov["files_sha256"],
            "pinned_missing": [p for p in pinned if p not in prov["files_sha256"]],
            "lane_clean_at_head": lane_prov["clean_at_commit"], "lane_files_sha256": lane_prov["files_sha256"]}


def kd_pin_reuse_check() -> dict:
    """The KD items: each module of the KD pin's reused_module_hashes (scripts/preflight_distill.py TEACHER_VALUES),
    hashed now and compared; a module that differs is listed, never hidden (audit §6.1)."""
    from src.quant.qat_kd import preflight_distill
    want = dict(preflight_distill().TEACHER_VALUES["reused_module_hashes"])
    rel = {"teacher_components": "src/training/teacher_components.py"}
    rows = {}
    for key, sha in want.items():
        p = REPO / rel.get(key, key)
        got = sha256_file(p) if p.is_file() else None
        rows[key] = {"path": rel.get(key, key), "pin": sha, "now": got, "match": got == sha}
    return {"source": "scripts/preflight_distill.py TEACHER_VALUES['reused_module_hashes'] (the KD pin 8487551)",
            "modules": rows, "all_match": all(r["match"] for r in rows.values()),
            "differ": sorted(k for k, r in rows.items() if not r["match"])}


def host_record() -> dict:
    import torch
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
            "torch_num_threads": int(torch.get_num_threads()),
            "quantized_engines": list(torch.backends.quantized.supported_engines),
            "image_digest": os.environ.get("PLANTSEG_IMAGE_DIGEST") or None}


def json_bytes(doc) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_exclusive(path: Path, data: bytes) -> Path:
    """Create `path`, refusing an existing file (an exploratory record is never overwritten)."""
    with open(path, "xb") as fh:
        fh.write(data)
    return path


def build_run_meta(*, item: str, run_id: str, seeds, recipe: dict, isolation: dict, code: dict,
                   split: str = EVAL_SPLIT, precision_label: str | None = None, extra: dict | None = None) -> dict:
    seeds_txt = "/".join(str(s) for s in seeds) if isinstance(seeds, (list, tuple)) else str(seeds)
    doc = {"format": RUN_META_FORMAT, "event": "x_run_meta", "exploratory": True, "amendment": AMENDMENT,
           "lane": LANE, "ruling": RULING, "artifact_status": ARTIFACT_STATUS, "split": split,
           "item": item, "item_display": ITEM_DISPLAY.get(item, item), "item_name": ITEMS[item],
           "run_id": run_id, "seeds": list(seeds) if isinstance(seeds, (list, tuple)) else seeds,
           "claim_line": CLAIM_LINE.format(seeds=seeds_txt), "recipe": recipe, "isolation": isolation,
           "registered": False, "holm_family": False, "test_scored": False, "parent_of_record": False,
           "precision_label": precision_label, "host": host_record(), **code, "wall_clock": time.time()}
    if extra:
        clash = sorted(set(extra) & set(doc))
        if clash:
            raise GuardError("run_meta_key_clash", f"extra run_meta keys {clash} would overwrite the guard's")
        doc.update(extra)
    return doc


def write_run_meta(out_dir: Path, doc: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_exclusive(out_dir / RUN_META_NAME, json_bytes(doc))


def guarded_start(*, item: str, run_id: str, out_dir, split: str, allowed_splits: Iterable[str] = TRAINVAL,
                  input_paths: Mapping[str, object] | None = None, records_only: bool = False,
                  recipe: dict, seeds, pinned_paths: Iterable[str] = (), kd_reuse_check: bool = False,
                  precision_label: str | None = None, extra: dict | None = None,
                  write: bool = True) -> dict:
    """Steps 1-7. Returns {"out_dir", "run_meta", "run_meta_path", "inputs", "isolation"}; raises GuardError."""
    check_item(item)
    refuse_split(split, allowed=allowed_splits)
    check_run_id(run_id)
    inputs = check_input_paths(dict(input_paths or {}))
    out = check_out_dir(out_dir, run_id, inputs=inputs.values())
    isolation = check_data_root(records_only=records_only)
    code = code_record(pinned_paths)
    if kd_reuse_check:
        code["kd_pin_reused_modules"] = kd_pin_reuse_check()
    doc = build_run_meta(item=item, run_id=run_id, seeds=seeds, recipe=recipe, isolation=isolation, code=code,
                         split=EVAL_SPLIT, precision_label=precision_label,
                         extra={"input_paths": inputs, **(extra or {})})
    path = write_run_meta(out, doc) if write else None
    return {"out_dir": out, "run_meta": doc, "run_meta_path": path, "inputs": inputs, "isolation": isolation}


def result_line(tag: str, ok: bool, detail: str = "") -> str:
    return f"RESULT: {tag} {'OK' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}"


__all__ = ["AMENDMENT", "ARTIFACT_STATUS", "CLAIM_LINE", "EVAL_SPLIT", "EXIT_ABORTED", "EXIT_ERROR", "EXIT_OK",
           "EXIT_REFUSED", "EXIT_STOP", "GuardError", "ITEMS", "LANE", "OFFICIAL_DIR_RE", "OFFICIAL_RECORD_NAMES",
           "RUN_META_NAME", "RUN_PREFIX", "TRAINVAL", "VAL_IMAGES", "VAL_ONLY", "build_run_meta", "check_data_root",
           "check_input_paths", "check_item", "check_out_dir", "check_run_id", "code_record", "data_root",
           "guarded_start", "json_bytes", "kd_pin_reuse_check", "lane_files", "names_test", "out_dir_error",
           "refuse_split", "sha256_file", "write_exclusive", "write_run_meta"]
