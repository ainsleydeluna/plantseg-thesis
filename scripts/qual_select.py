#!/usr/bin/env python3
"""QUAL-1, step 1 of 3 (AM-27; DL-182): the qualitative image set S, the preflight P1, and what the qual steps share.

DL-182's rule, as AM-27 item 1 registers it, read from one official artifact and the split manifest:
  (1) D = the AM-5-eligible images (the ground truth holds a disease pixel), in manifest order; eligibility is
      read from the official per-image rows (`am5_excluded`), and a row that does not carry it stops the step.
  (2) Three percentile picks over D by E6 seed 42's clean per-image disease-only mIoU: the 10th, 50th and 90th
      percentiles, nearest rank (rank ceil(p * |D| / 100)), ties to the lower manifest index (TIE_READING).
  (3) Six random picks: random.Random(42).sample(sorted(D's manifest indices), len(D)) under CPython 3.11.16 in
      the KD image of record; its first six entries that are not percentile picks. The interpreter and the image
      digest are asserted before the draw (version_guard).
  (4) S = the three percentile picks in percentile order, then the random picks in shuffle order.
On TEST the artifact is located and verified through the committed binding (src.stats.heldout_inputs:
open_inventory + load_bound, which hash the artifact's four files against the binding record); the manifest
order is the evaluator's own split listing (src.eval.adapters.build_expected_manifest_for), whose
split_manifest_sha256 must equal the artifact's. Right after the E6 record, the other twelve records of AM-27 item
1(a) are loaded and checked the same way (ruling QUAL-AUD-1 F-02); select/record.json keeps all thirteen
files_sha256. On VAL (the Phase A layout check, not a selection) S is the first nine eligible images in VAL manifest
order, read from a VAL per-image artifact given by path.

The preflight P1 (ruling QUAL-AUD-1 A): one function, run alone (--preflight-only, the kit's block P1) and as the
first thing of this step, before any per-image file is opened. It opens no image, mask, cached copy or per-image
file. It reads the export and binding through TMAN's reader, hashes the four files of each record against the
binding, reads the thirteen summary.json files (identity, execution context, one repo_commit and its evaluator
trees), opens the three cells' cache manifest and verification record (open_cell) and lists the split's images and
masks (names held in memory only). It prints counts, hashes and PASS or REFUSED only; a refusal exits 3 and is not
a stopped run (QUAL-AUD-1 A.5). On VAL it runs on the VAL records given (--preflight-record), where the execution
context and the commits are information only.

Output: <run>/select/S.json (the positions, manifest indices and image ids of S: data of the run, never
printed) and <run>/select/record.json (inputs and their sha256). The console shows counts, hashes and
PASS or STOP lines only: no image id, no file name of an image, no per-image value.

  python -B scripts/qual_select.py --run DIR --split test --export reports/derived/tday_inventory.json \
      --export-sha256 HEX --session-root DIR --expect-commit SHA CACHE [--preflight-only]
  python -B scripts/qual_select.py --run DIR --split val --val-artifact DIR \
      --preflight-record MODEL:CONDITION=DIR ... CACHE [--preflight-only]
  CACHE = --corruption-cache DIR --cache-canonical-sha256 HEX --cache-manifest-file-sha256 HEX \
      --verification-record FILE --verification-record-sha256 HEX
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ------------------------------------------------------------------------------------------------ shared
SELECT_FORMAT = "qual1-select/1"
RECORD_FORMAT = "qual1-record/1"
KD_DIGEST = "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b"   # DL-182 item 3
PYTHON = "3.11.16"                                                                       # DL-182 item 3
SEED = 42
PERCENTILES = (10, 50, 90)
N_RANDOM = 6
K_ALL = 9                     # AM-27 item 3: Figures 4.3 and 4.4 show k = 9
K_CELLS = 3                   # AM-27 item 3: Figures 4.5 and 4.6 show k = 3
P50_POSITION = 2              # Figure 4.2: the 50th-percentile pick, the second of S
N_SPLIT = {"val": 846, "test": 1561}
# The execution context of record (tday_manifest execution_context; every TEST summary.json, MEASURED by ORCH-7 on
# 13 Nov 2026): T = 1, batch 1, the CPU; and the image of record (version_guard). Ruling QUAL-AUD-1 F-03.
THREADS = 1
BATCH = 1
DEVICE = "cpu"
TDAY_MANIFEST_REL = "reports/derived/tday_manifest.json"
TDAY_MANIFEST_SHA256 = "3ca39b40dfba45a7e486cf40098994582b8d2a22e11c0768a345e8cf5676059d"   # SUMMARY.txt
# Ruling QUAL-A-1 Q2 (ORCH-7, 13 Nov 2026): the reading is "rank": D ordered by (value, manifest index), the picks
# the elements at the nearest ranks (three distinct images always). "lowest_index_at_value" (the percentile value at
# that rank, then the lowest manifest index holding it; it can pick one image twice) is computed beside it and only
# recorded, as one word (tie_readings_word); it selects nothing.
TIE_READINGS = ("rank", "lowest_index_at_value")
TIE_READING = "rank"
# AM-27 item 2: the four models of record at seed 42, by slot in reports/derived/tday_manifest.json; `file_id` is
# the TEST layout's folder name (reports/tday/<file_id>/<precision>/<scope>[/<cell>]); `stage` is LREP's stage.
MODELS = {
    "teacher": {"slot": "teacher:s42:40k", "stage": "teacher", "precision": "fp32", "header": "Teacher"},
    "E1": {"slot": "E1:s42:80k", "stage": "E1", "precision": "fp32", "header": "E1"},
    "E3": {"slot": "E3:s42:lambda0.5:alpha25:80k", "stage": "E3", "precision": "fp32", "header": "E3"},
    "E6": {"slot": "E6:s42:sel", "stage": "E6", "precision": "int8_qat", "header": "E6"},
}
MODEL_ORDER = ("teacher", "E1", "E3", "E6")
CLEAN = "clean"
CELLS = ("motion_blur_s2", "gaussian_noise_s2", "fog_s2")   # AM-27 item 1(a), Chapter III's order
CONDITIONS = {"teacher": (CLEAN,), "E1": (CLEAN, *CELLS), "E3": (CLEAN, *CELLS), "E6": (CLEAN, *CELLS)}
# AM-27 item 1(a): the thirteen official records, as (model, condition), in this order everywhere
RECORD_KEYS = tuple((m, c) for m in MODEL_ORDER for c in CONDITIONS[m])
_HEX40 = re.compile(r"[0-9a-f]{40}")


def k_of(condition: str) -> int:
    """How many images of S a condition is predicted on: all nine clean, the first three under a corruption."""
    return K_ALL if condition == CLEAN else K_CELLS


def record_label(model: str, condition: str) -> str:
    return f"{model} {condition}"


class QualStop(RuntimeError):
    """A named stop: the kit prints `STOP <code>: <detail>` and exits 1. `detail` is written in this kit and names
    no image id, image file name or per-image value; `private` (a library's own message, which may) is kept in the
    run's log on TEST and printed on VAL (run_step)."""
    kind = "STOP"
    exit_code = 1

    def __init__(self, code: str, detail: str, private: str | None = None):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.private = private


class PreflightRefusal(QualStop):
    """A refusal of the preflight P1 (ruling QUAL-AUD-1 A.5): no image, mask, cached copy or per-image file was
    opened, so it is not a stopped run. Printed `REFUSED <code>: <detail>`; exit 3."""
    kind = "REFUSED"
    exit_code = 3


def say(kind: str, text: str) -> None:
    print(f"{kind} {text}", flush=True)


def _private(step: str, args, text: str) -> str:
    """VAL: to stderr. TEST: appended to <run>/logs/<step>.txt, a file for Ice, never for an AI session."""
    if getattr(args, "split", None) == "val":
        print(text, file=sys.stderr, flush=True)
        return "printed on stderr"
    try:
        run = no_test(Path(args.run))
        if not run.is_dir():
            return "not kept (no run folder)"
        p = run / "logs" / f"{step}.txt"
        p.parent.mkdir(exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")
        return f"kept in <run>/logs/{p.name}; do not paste it to an AI session"
    except Exception:                                        # noqa: BLE001 -- the console line is printed anyway
        return "not kept"


def run_step(step: str, args, body) -> int:
    """Every qual step's exit. A named stop prints its kind, code and detail; any other error prints its type only,
    and its message and traceback (which may name an image) go to _private."""
    import traceback
    try:
        return body(args)
    except QualStop as e:
        say(e.kind, f"{e.code}: {e.detail}")
        if e.private:
            _private(step, args, f"{e.kind} {e.code}: {e.private}")
        return e.exit_code
    except Exception as e:                                   # noqa: BLE001 -- every other error is a STOP too
        where = _private(step, args, traceback.format_exc())
        say("STOP", f"unexpected_error: {type(e).__name__} (its message: {where})")
        return 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def strict_json(data: bytes | str, what: str):
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        return json.loads(text, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as e:
        raise QualStop("json_invalid", f"{what}: {type(e).__name__}") from None


def canonical_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def no_test(path) -> Path:
    """Every path a qual step names or creates (AM-27; hard limits): none holds "test", case-folded."""
    p = Path(path)
    if "test" in str(p).casefold():
        raise QualStop("test_path", "a qual path names 'test'; nothing was read or written")
    return p


def write_new(path, data: bytes) -> str:
    """Write a file that must not exist yet (no overwrite, ever); return its sha256."""
    p = no_test(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "xb") as f:
            f.write(data)
    except FileExistsError:
        raise QualStop("output_exists", f"{p.name} exists already in {p.parent.name}; nothing is overwritten") from None
    return sha256_bytes(data)


def run_dir(path) -> Path:
    """The run folder: it exists (the kit made it) and lies outside the repository (QUAL-AUD-1 N-04)."""
    p = no_test(path)
    if not p.is_dir():
        raise QualStop("run_missing", "the run folder does not exist (the kit creates it)")
    if p.resolve().is_relative_to(REPO.resolve()):
        raise QualStop("run_in_repo", "the run folder lies inside the repository; nothing was read or written")
    return p


def outputs_absent(run: Path, rels) -> None:
    """QUAL-AUD-1 N-03: before any read, a step whose outputs exist already in the run folder stops."""
    there = [r for r in rels if (run / r).exists()]
    if there:
        raise QualStop("output_exists", f"{len(there)} of this step's outputs exist already in the run folder "
                                        f"({there[0]}); nothing was read")


def repo_state(expect_commit: str | None) -> dict:
    """The checkout a step runs from, read with the evaluator's own helpers (src/eval/artifacts.py): HEAD and the
    governed paths' state (contract section 7.1). With `expect_commit` (TEST: the audited commit, AM-27 item 5) HEAD
    must be that commit and the governed paths clean."""
    from src.eval.artifacts import GOVERNED_PREFIXES, git_commit, git_porcelain_bytes, parse_porcelain_paths
    try:
        commit = git_commit(REPO)
        paths = parse_porcelain_paths(git_porcelain_bytes(REPO))
    except Exception as e:                                   # noqa: BLE001 -- a named stop
        raise QualStop("repo_state", "git could not read the checkout", private=str(e)) from None
    violations = sorted(p for p in paths if any(p.startswith(g) for g in GOVERNED_PREFIXES))
    if expect_commit is not None and (commit != expect_commit or violations):
        raise QualStop("repo_state", f"the checkout is {commit[:12]}... with {len(violations)} governed paths dirty "
                                     f"or untracked; the run is made only from {expect_commit[:12]}... clean")
    return {"commit": commit, "governed_paths_clean": not violations, "governed_violations": violations}


def version_guard(*, implementation=None, version=None, digest=None) -> dict:
    """DL-182 item 3: CPython 3.11.16 in the KD image of record, asserted before any random draw. The image is
    named by PLANTSEG_IMAGE_DIGEST, which the kit reads from the image it runs (docker image inspect; QUAL-AUD-1
    N-11), never typed by hand."""
    impl = platform.python_implementation() if implementation is None else implementation
    ver = platform.python_version() if version is None else version
    dig = os.environ.get("PLANTSEG_IMAGE_DIGEST") if digest is None else digest
    bad = []
    if impl != "CPython":
        bad.append(f"implementation {impl}")
    if ver != PYTHON:
        bad.append(f"python {ver}")
    if dig != KD_DIGEST:
        bad.append(f"image {dig!r}")
    if bad:
        raise QualStop("version_guard", f"DL-182 item 3 draws under CPython {PYTHON} in {KD_DIGEST[:19]}...; "
                                        f"here: {', '.join(bad)}; nothing was drawn")
    return {"implementation": impl, "python": ver, "image_digest": dig}


# ------------------------------------------------------------------------------------- the models of record
def tday_manifest(path=None) -> dict:
    """reports/derived/tday_manifest.json, held to its sha256 before it is parsed: {slot: entry}."""
    p = Path(path) if path is not None else REPO / TDAY_MANIFEST_REL
    data = p.read_bytes()
    if sha256_bytes(data) != TDAY_MANIFEST_SHA256:
        raise QualStop("tday_manifest_changed", "reports/derived/tday_manifest.json is not the file of record")
    return {e["slot"]: e for e in json.loads(data)["body"]["entries"]}


def expected_hashes(model: str, manifest_path=None) -> dict:
    """AM-27 item 2: the sha256 each model file must have, from the TEST-day manifest (itself held to its sha256).
    E6 has two files (QUAL-AUD-1 N-12): its provenance and its converted artifact."""
    e = tday_manifest(manifest_path).get(MODELS[model]["slot"])
    if e is None:
        raise QualStop("slot_missing", f"{MODELS[model]['slot']} is not in the TEST-day manifest")
    role = "teacher_reference" if model == "teacher" else "family"
    if role not in (e.get("role") or []):
        raise QualStop("slot_role", f"{MODELS[model]['slot']} is not of role {role}")
    if model == "E6":
        return {"converted": e["hashed"]["artifact_sha256"], "provenance": e["hashed"]["provenance"]["sha256"]}
    return {"checkpoint": e["checkpoint"]["sha256"]}


def record_model_sha256(model: str, manifest_path=None) -> str:
    """The sha256 a record of this model gives as run.checkpoint_sha256: the checkpoint, or E6's converted
    artifact."""
    h = expected_hashes(model, manifest_path)
    return h["converted"] if model == "E6" else h["checkpoint"]


# ------------------------------------------------------------------------------------------- official rows
class Official:
    """One official per-image artifact, verified: its summary, its rows by manifest index, its NPZ arrays and
    the sha256 of its files. Nothing here is printed."""

    def __init__(self, *, label, directory, summary, rows, npz, files_sha256, source):
        self.label, self.directory, self.summary = label, directory, summary
        self.rows, self.npz, self.files_sha256, self.source = rows, npz, files_sha256, source

    @property
    def device(self) -> str:
        """The device the record's command line asked for (summary run.env.device), kept for the record."""
        return self.summary["run"]["env"]["device"]

    @property
    def model_device(self):
        """The device the record's model ran on (summary run.eval_runtime.model_device): the device class."""
        return ((self.summary.get("run") or {}).get("eval_runtime") or {}).get("model_device")

    def row(self, manifest_index: int) -> dict:
        r = self.rows.get(int(manifest_index))
        if r is None:
            raise QualStop("record_row_missing", f"{self.label}: the record has no row for a manifest index of S")
        return r

    def counts(self, image_id: str) -> dict:
        """The record's confusion counts of one image (src/eval/artifacts.py :615-619) as {class: (tp, gt, pred)}."""
        ids = [str(x) for x in self.npz["manifest_ids"]]
        pos = ids.index(image_id)
        sel = self.npz["image_index"] == pos
        return counts_map(self.npz["class_id"][sel], self.npz["tp"][sel], self.npz["gt"][sel], self.npz["pred"][sel],
                          self.label)


def counts_map(class_id, tp, gt, pred, what: str) -> dict:
    """QUAL-A-1 Q6: one image's confusion counts as a mapping {class: (tp, gt, pred)}, so that two are compared
    class by class and never as ordered lists; a class listed twice for one image is refused."""
    out = {}
    for c, t, g, p in zip(class_id, tp, gt, pred):
        if int(c) in out:
            raise QualStop("counts_invalid", f"{what}: a class is listed twice for one image")
        out[int(c)] = (int(t), int(g), int(p))
    return out


def same_device_class(summary: dict) -> bool:
    """AM-27 item 4's device class (QUAL-AUD-1 F-03): the record's model ran on this run's device, as the evaluator
    recorded it (run.eval_runtime.model_device, str(torch.device): "cpu"), not as its command line spelled it."""
    return ((summary.get("run") or {}).get("eval_runtime") or {}).get("model_device") == DEVICE


def record_context_problems(summary: dict, guard: dict) -> list:
    """QUAL-AUD-1 F-03 (AM-27 item 5): the record was made with T = 1, batch 1, on the CPU, in the image this run
    runs in. Returns the conditions that fail (an empty list when all four hold)."""
    er = (summary.get("run") or {}).get("eval_runtime") or {}
    bad = []
    if not (type(er.get("torch_num_threads")) is int and er["torch_num_threads"] == THREADS):
        bad.append("torch_num_threads")
    if not (type(er.get("batch_size")) is int and er["batch_size"] == BATCH):
        bad.append("batch_size")
    if er.get("image_digest") != guard.get("image_digest") or not guard.get("image_digest"):
        bad.append("image_digest")
    if er.get("model_device") != DEVICE:
        bad.append("model_device")
    return bad


def _parse_rows(data: bytes, label: str) -> dict:
    rows = {}
    for n, line in enumerate(data.decode("utf-8").splitlines()):
        r = strict_json(line, f"{label} per_image line {n + 1}")
        m = r.get("manifest_index")
        if not isinstance(m, int) or isinstance(m, bool) or m in rows:
            raise QualStop("record_rows_invalid", f"{label}: a row without a unique integer manifest_index")
        rows[m] = r
    return rows


def _read_payload(directory: Path, files_sha256: dict, label: str):
    """per_image.jsonl and sufficient_stats.npz read once each, their bytes held to the verified sha256."""
    import numpy as np
    out = {}
    for name in ("per_image.jsonl", "sufficient_stats.npz", "summary.json"):
        data = (directory / name).read_bytes()
        if sha256_bytes(data) != files_sha256[name]:
            raise QualStop("record_changed", f"{label}: {name} differs from its verified sha256")
        out[name] = data
    with np.load(io.BytesIO(out["sufficient_stats.npz"]), allow_pickle=False) as z:
        npz = {k: z[k] for k in ("image_index", "class_id", "tp", "gt", "pred", "manifest_ids")}
    return strict_json(out["summary.json"], f"{label} summary"), _parse_rows(out["per_image.jsonl"], label), npz


def open_test_inventory(export, export_sha256):
    """The committed bound export through LREP's reader (TMAN's read_inventory under Policy.OFFICIAL): the binding
    is found beside it and verified against the export's binding.sha256."""
    from src.stats.heldout_inputs import LrepRefusal, open_inventory
    from src.stats.ingest import Policy
    try:
        return open_inventory(no_test(export), policy=Policy.OFFICIAL, expect_sha256=export_sha256)
    except LrepRefusal as e:
        raise QualStop("export_refused", str(e.code), private=str(e)) from None


def binding_key(model: str, condition: str) -> tuple:
    """The binding's key of a record: (slot, precision, scope, cell)."""
    spec = MODELS[model]
    cell = None if condition == CLEAN else condition
    return spec["slot"], spec["precision"], CLEAN if cell is None else "corrupted", cell


def load_test_official(inv, model: str, condition: str, session_root) -> Official:
    """One TEST artifact of record through the binding (load_bound: the four files against the record, the run
    manifest, the canvas protocol, the stage, model role and precision, the checkpoint against the model row)."""
    from src.stats.heldout_inputs import LrepRefusal, load_bound
    from src.stats.ingest import Policy
    spec = MODELS[model]
    slot, precision, scope, cell = binding_key(model, condition)
    label = record_label(model, condition)
    try:
        ba = load_bound(inv, slot=slot, stage=spec["stage"], precision=precision, scope=scope, cell=cell,
                        input_root=no_test(session_root), policy=Policy.OFFICIAL)
    except LrepRefusal as e:
        raise QualStop("record_refused", f"{label}: {e.code}", private=str(e)) from None
    files = dict(ba.record["files_sha256"])
    summary, rows, npz = _read_payload(ba.loaded.directory, files, label)
    return Official(label=label, directory=ba.loaded.directory, summary=summary, rows=rows, npz=npz,
                    files_sha256=files, source={"binding_key": [slot, precision, ba.record["scope"], ba.record["cell"]],
                                                "repo_relative_path": ba.record["repo_relative_path"]})


def load_val_official(directory, label: str) -> Official:
    """A VAL artifact by path (Phase A only): the evaluator's verify_artifact, then its files hashed once."""
    from src.eval.artifacts import verify_artifact
    from src.stats.driver import input_file_sha256
    d = no_test(directory)
    try:
        verify_artifact(d)
    except Exception as e:                                   # noqa: BLE001 -- a named stop
        raise QualStop("record_unverified", f"{label}: {type(e).__name__}", private=str(e)) from None
    files = input_file_sha256(d)
    summary, rows, npz = _read_payload(d, files, label)
    if summary["dataset"]["split"] != "val":
        raise QualStop("record_split", f"{label}: the VAL record is not a VAL artifact")
    return Official(label=label, directory=d, summary=summary, rows=rows, npz=npz, files_sha256=files,
                    source={"path": str(d)})


def identity_problems(summary: dict, *, split: str, condition: str, checkpoint_sha256: str | None) -> list:
    """The summary's identity fields against the figure's (QUAL-AUD-1 F-02, the part a summary carries): split,
    canvas protocol, condition, row count and, when given, the model's sha256 as the evaluator recorded it."""
    from src.eval.protocols import CANVAS_PROTOCOL_ID
    ds, run = summary.get("dataset") or {}, summary.get("run") or {}
    want = {"type": "clean", "name": None, "severity": None} if condition == CLEAN else {
        "type": "corruption", "name": condition.rpartition("_s")[0], "severity": int(condition.rpartition("_s")[2])}
    problems = []
    if ds.get("split") != split:
        problems.append("split")
    if ds.get("preprocess_protocol") != CANVAS_PROTOCOL_ID:
        problems.append("protocol")
    if ds.get("condition") != want:
        problems.append("condition")
    if ds.get("actual_rows") != N_SPLIT[split]:
        problems.append("rows")
    if checkpoint_sha256 is not None and run.get("checkpoint_sha256") != checkpoint_sha256:
        problems.append("checkpoint_sha256")
    return problems


def check_record_identity(off: Official, *, split: str, condition: str, checkpoint_sha256: str | None) -> None:
    """The record is the one the figure names: split, canvas protocol, condition, all rows, and (when given) the
    model's sha256 as the evaluator recorded it."""
    problems = identity_problems(off.summary, split=split, condition=condition, checkpoint_sha256=checkpoint_sha256)
    if len(off.rows) != N_SPLIT[split] and "rows" not in problems:
        problems.append("rows")
    if problems:
        raise QualStop("record_identity", f"{off.label}: the record differs in {problems}")


def manifest_order(split: str, off: Official) -> list:
    """AM-27 item 1(b): the split's manifest (the evaluator's filename listing; no image opened), held to the
    record's split_manifest_sha256, and the record's rows held to it id by id. Returns the ids by index."""
    from src.eval.adapters import build_expected_manifest_for
    from src.eval.artifacts import hash_split_manifest
    entries = build_expected_manifest_for(split, range(N_SPLIT[split]))
    if hash_split_manifest(entries) != off.summary["dataset"]["split_manifest_sha256"]:
        raise QualStop("manifest_mismatch", "the split listing's sha256 is not the record's split_manifest_sha256")
    ids = [e.image_id for e in sorted(entries, key=lambda e: e.manifest_index)]
    for m, image_id in enumerate(ids):
        r = off.row(m)
        if r.get("image_id") != image_id or r.get("clean_image_id") != image_id:
            raise QualStop("manifest_mismatch", "a record row's id differs from the split manifest at its index")
    return ids


# ----------------------------------------------------------------------------------------------- the cache
def cache_args_given(args) -> bool:
    return bool(getattr(args, "corruption_cache", None) and getattr(args, "cache_canonical_sha256", None)
                and getattr(args, "cache_manifest_file_sha256", None) and getattr(args, "verification_record", None)
                and getattr(args, "verification_record_sha256", None))


def open_cell(args, split: str, condition: str):
    """One cell of the corruption cache through the evaluator's own gates (scripts/evaluate_corruptions.py): the
    manifest (status official on TEST, provisional on VAL) against the canonical sha256, the manifest file and the
    verification record against their pins. No cached copy is read here."""
    from scripts.evaluate_corruptions import (cached_image_loader, check_verification_record,
                                              resolve_corruption_cache)
    from src.corruption_cache import MANIFEST_NAME
    from src.eval.corrupted import CacheCell
    name, _, sev = condition.rpartition("_s")
    status = "official" if split == "test" else "provisional"
    root = no_test(args.corruption_cache)
    manifest = resolve_corruption_cache(str(root), name, int(sev), split=split, official=status == "official",
                                        artifact_status=status, expected_sha256=args.cache_canonical_sha256)
    if sha256_file(Path(root) / MANIFEST_NAME) != args.cache_manifest_file_sha256:
        raise QualStop("cache_manifest_file", "the cache's manifest file is not the pinned one")
    rec_sha = check_verification_record(str(root), manifest, str(no_test(args.verification_record)),
                                        artifact_status=status)
    if rec_sha != args.verification_record_sha256:
        raise QualStop("cache_record", "the cache's verification record is not the pinned one")
    cell = CacheCell(Path(root).resolve(), manifest, name, int(sev),
                     load=cached_image_loader(str(root), manifest, name, int(sev)))
    cell.entry()
    return cell, manifest


# ------------------------------------------------------------------------------------------ the preflight
def evaluator_trees(commit) -> dict:
    """QUAL-AUD-1 N-13: where the records' repo_commit is in this clone, its src/ and configs/ trees against HEAD's
    (git objects only: a blob-less clone holds every tree). Otherwise {"compared": False}."""
    import subprocess
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1")

    def rev(spec):
        p = subprocess.run(["git", "rev-parse", "--verify", "--quiet", spec], cwd=str(REPO), capture_output=True,
                           text=True, env=env)
        return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None

    if not (isinstance(commit, str) and _HEX40.fullmatch(commit)) or rev(f"{commit}^{{commit}}") is None:
        return {"commit": commit, "compared": False}
    out = {"commit": commit, "compared": True}
    for path in ("src", "configs"):
        a, b = rev(f"{commit}:{path}"), rev(f"HEAD:{path}")
        out[path] = {"record_tree": a, "head_tree": b, "equal": a is not None and a == b}
    return out


def _preflight_sources(args) -> tuple:
    """TEST: the thirteen records through the binding (TMAN's reader). VAL: the records given by path."""
    if args.split == "test":
        from src.stats.heldout_inputs import LrepRefusal, binding_record, safe_relative_path
        if not (args.export and args.export_sha256 and args.session_root):
            raise QualStop("cli", "TEST needs --export, --export-sha256 and --session-root")
        inv = open_test_inventory(args.export, args.export_sha256)
        sources = {}
        for model, condition in RECORD_KEYS:
            label = record_label(model, condition)
            try:
                rec = binding_record(inv, *binding_key(model, condition))
                rel = safe_relative_path(rec["repo_relative_path"], label)
            except LrepRefusal as e:
                raise QualStop("binding_record", f"{label}: {e.code}", private=str(e)) from None
            sources[(model, condition)] = (no_test(Path(args.session_root) / rel), rec)
        return inv, sources
    sources = {}
    for spec in getattr(args, "preflight_record", None) or []:
        key, sep, d = spec.partition("=")
        model, _, condition = key.partition(":")
        if not sep or (model, condition) not in RECORD_KEYS or (model, condition) in sources:
            raise QualStop("cli", "--preflight-record takes MODEL:CONDITION=DIR, one per record of AM-27 item 1(a)")
        sources[(model, condition)] = (no_test(d), None)
    if not sources:
        raise QualStop("cli", "VAL: the preflight needs at least one --preflight-record")
    return None, {k: sources[k] for k in RECORD_KEYS if k in sources}


def split_listing(split: str) -> tuple:
    """QUAL-AUD-1 F-04: the split's image listing (the evaluator's manifest entries, build_expected_manifest_for),
    its image and mask pairs (src.data.dataset.PlantSegDataset: a mask for every image) and the mask folder's count
    of PNG files. Names and stat only, held in memory: never printed, logged or written."""
    from configs.data import DATA
    from src.data.dataset import PlantSegDataset
    from src.eval.adapters import build_expected_manifest_for
    try:
        entries = build_expected_manifest_for(split, range(N_SPLIT[split]))
    except Exception as e:                                   # noqa: BLE001 -- its message may name a file
        raise QualStop("listing_images", f"the {split.upper()} image listing failed: {type(e).__name__}",
                       private=str(e)) from None
    try:
        pair_stems = [img.stem for img, _ in PlantSegDataset(split).pairs]
    except Exception as e:                                   # noqa: BLE001 -- its message may name a file
        raise QualStop("listing_masks", f"the {split.upper()} image and mask listing failed: {type(e).__name__}",
                       private=str(e)) from None
    mask_dir = Path(DATA["root"]) / "annotations" / split
    n_masks = sum(1 for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    return entries, pair_stems, n_masks


def _preflight(args, guard: dict, repo: dict) -> dict:
    from src.eval.artifacts import hash_split_manifest
    from src.stats.driver import INPUT_FILES, input_file_sha256
    from scripts.evaluate_corruptions import check_cache_sources
    split, test = args.split, args.split == "test"
    n = N_SPLIT[split]
    out = {"split": split, "records": {}}
    inv, sources = _preflight_sources(args)
    if test:
        say("PASS", f"P1 export and binding read through TMAN's reader (export sha256 {args.export_sha256[:12]}...); "
                    f"{len(sources)} binding records of AM-27 item 1(a)")
    # ---- the records: four files hashed, summary.json read (no per-image file is parsed) ----
    model_sha = {m: record_model_sha256(m) for m in MODEL_ORDER}
    commits, splits, context_bad, n_files = {}, {}, {}, 0
    for (model, condition), (d, rec) in sources.items():
        label = record_label(model, condition)
        try:
            names = sorted(os.listdir(d))
        except OSError as e:
            raise QualStop("record_folder", f"{label}: its folder cannot be listed", private=str(e)) from None
        if names != sorted(INPUT_FILES):
            raise QualStop("record_folder", f"{label}: its folder holds {len(names)} entries, not the four files")
        try:
            files = input_file_sha256(d)
        except Exception as e:                               # noqa: BLE001 -- a named refusal
            raise QualStop("record_files", f"{label}: {type(e).__name__}", private=str(e)) from None
        if rec is not None and files != dict(rec["files_sha256"]):
            raise QualStop("record_files", f"{label}: its four files are not the binding record's")
        n_files += len(files)
        data = (d / "summary.json").read_bytes()
        if sha256_bytes(data) != files["summary.json"]:
            raise QualStop("record_files", f"{label}: summary.json changed while read")
        summary = strict_json(data, f"{label} summary")
        problems = identity_problems(summary, split=split, condition=condition, checkpoint_sha256=model_sha[model])
        ds, run = summary.get("dataset") or {}, summary.get("run") or {}
        if rec is not None:
            problems += [k for k, v in (("split_manifest_sha256", ds.get("split_manifest_sha256")),
                                        ("repo_commit", run.get("repo_commit")),
                                        ("checkpoint_sha256", run.get("checkpoint_sha256"))) if v != rec.get(k)]
        if problems:
            raise QualStop("record_identity", f"{label}: the summary differs in {sorted(set(problems))}")
        bad = record_context_problems(summary, guard)
        if bad:
            context_bad[label] = bad
            if test:
                raise QualStop("record_context", f"{label}: {bad} do not hold (T {THREADS}, batch {BATCH}, the image "
                                                 f"{guard['image_digest'][7:19]}, model device {DEVICE})")
        commits[label] = run.get("repo_commit")
        splits[label] = ds.get("split_manifest_sha256") if rec is None else rec.get("split_manifest_sha256")
        out["records"][label] = {"files_sha256": files, "repo_commit": run.get("repo_commit"),
                                 "context": "held" if not bad else bad}
    say("PASS", f"P1 records: {len(sources)} folders, {n_files} files hashed"
                + (" and equal to the binding" if test else "") + "; every summary's identity (split, canvas "
                f"protocol, condition, {n} rows, the model's sha256 of record) holds")
    held = len(sources) - len(context_bad)
    say("PASS" if not context_bad else "INFO",
        f"P1 context: {held} of {len(sources)} records made with T {THREADS}, batch {BATCH}, model device {DEVICE}, "
        f"image {guard['image_digest'][7:19]}" + ("" if not context_bad else
                                                  f"; the other {len(context_bad)}: information only (VAL)"))
    # ---- one repo_commit, and its evaluator trees (N-13) ----
    distinct = sorted({c for c in commits.values() if isinstance(c, str)})
    if test and (len(distinct) != 1 or None in commits.values()):
        raise QualStop("repo_commit", f"the {len(sources)} records name {len(distinct)} repo_commits, not one")
    trees = {c: evaluator_trees(c) for c in distinct}
    for c, t in trees.items():
        if not t["compared"]:
            say("PASS" if test else "INFO", f"P1 repo_commit {c[:12]}...: not in this clone; trees not compared")
            continue
        eq = t["src"]["equal"] and t["configs"]["equal"]
        if test and not eq:
            raise QualStop("evaluator_tree", f"repo_commit {c[:12]}...: its src/ or configs/ tree is not HEAD's")
        say("PASS" if eq else "INFO", f"P1 repo_commit {c[:12]}... ({sum(v == c for v in commits.values())} records): "
                                      f"src/ {'=' if t['src']['equal'] else '!='} HEAD, configs/ "
                                      f"{'=' if t['configs']['equal'] else '!='} HEAD")
    out["repo_commits"] = distinct
    out["evaluator_trees"] = trees
    # ---- the listings (F-04): names in memory only, never printed, logged or written ----
    entries, pair_stems, n_masks = split_listing(split)
    lsha = hash_split_manifest(entries)
    want = sorted(set(splits.values()))
    if want != [lsha]:
        raise QualStop("manifest_mismatch", f"the split listing's sha256 {lsha[:12]}... is not the records' "
                                            f"({len(want)} distinct values)")
    stems = [e.image_id for e in sorted(entries, key=lambda e: e.manifest_index)]
    if len(stems) != n or pair_stems != stems:
        raise QualStop("listing_masks", f"the image and mask pairs ({len(pair_stems)}) are not the split listing")
    if n_masks != n:
        raise QualStop("listing_masks", f"the {split.upper()} mask folder holds {n_masks} masks, not {n}")
    say("PASS", f"P1 listings: {len(stems)} images and {n_masks} masks, one mask per image (names in memory only); "
                f"split manifest sha256 {lsha[:12]}... = the {'binding' if test else 'records'}' "
                f"({len(splits)} records)")
    out["listing"] = {"images": len(stems), "masks": n_masks, "split_manifest_sha256": lsha}
    # ---- the cache (F-01): the three cells' manifest and verification record; no cached copy ----
    if not cache_args_given(args):
        raise QualStop("cli", "the preflight needs the cache and its three pins")
    for cell in CELLS:
        try:
            _, manifest = open_cell(args, split, cell)
            check_cache_sources(manifest, entries, full=True)
        except QualStop:
            raise
        except Exception as e:                               # noqa: BLE001 -- the evaluator's own refusal code
            code = getattr(e, "code", None)
            raise QualStop("cache_refused", f"{cell}: {code if isinstance(code, str) else type(e).__name__}",
                           private=str(e)) from None
    out["cache"] = {"status": "official" if test else "provisional", "cells": list(CELLS),
                    "canonical_sha256": args.cache_canonical_sha256,
                    "manifest_file_sha256": args.cache_manifest_file_sha256,
                    "verification_record_sha256": args.verification_record_sha256}
    say("PASS", f"P1 cache: {len(CELLS)} cells opened ({out['cache']['status']}): canonical "
                f"{args.cache_canonical_sha256[:12]}..., manifest file {args.cache_manifest_file_sha256[:12]}..., "
                f"verification record {args.verification_record_sha256[:12]}...; its sources are the split listing "
                f"({n}); no cached copy read")
    out["guard"], out["repo"] = guard, repo
    return out


def preflight(args, guard: dict, repo: dict) -> dict:
    """The preflight P1 (ruling QUAL-AUD-1 A): every stop inside it is a refusal (exit 3), never a stopped run. On
    TEST a refusal keeps no library message and no traceback, not even in the run's log, for such a text may name an
    image or a mask (A.2: names are held in memory only, never printed, logged or written): its code and its
    kit-written detail are all that is printed. On VAL the message goes to stderr, as elsewhere."""
    import traceback
    keep = args.split != "test"
    try:
        return _preflight(args, guard, repo)
    except PreflightRefusal as e:
        raise PreflightRefusal(e.code, e.detail, e.private if keep else None) from None
    except QualStop as e:
        raise PreflightRefusal(e.code, e.detail, e.private if keep else None) from None
    except Exception as e:                                   # noqa: BLE001 -- its type alone is printed
        raise PreflightRefusal("unexpected_error", type(e).__name__,
                               traceback.format_exc() if keep else None) from None


# --------------------------------------------------------------------------------------------- the rule
def eligible(off: Official, n: int) -> list:
    """DL-182 (1): D in manifest order, eligibility read from the record's rows; a row without it stops."""
    d = []
    for m in range(n):
        r = off.row(m)
        ex = r.get("am5_excluded")
        if not isinstance(ex, bool):
            raise QualStop("eligibility_absent", "a record row does not carry am5_excluded; no mask is read")
        if ex is not (r.get("n_eligible_disease_only") == 0):
            raise QualStop("eligibility_invalid", "a record row's am5_excluded disagrees with its eligible count")
        if not ex:
            d.append(m)
    if not d:
        raise QualStop("eligibility_empty", "no eligible image")
    return d


def scored(off: Official, d: list) -> list:
    """(manifest index, disease-only mIoU of record) for every image of D; every value finite."""
    out = []
    for m in d:
        r = off.row(m)
        v = r.get("disease_only_miou")
        if r.get("disease_only_miou_status") != "ok" or not isinstance(v, float) or not math.isfinite(v):
            raise QualStop("value_undefined", "an eligible row has no finite disease-only mIoU of record")
        out.append((m, v))
    return out


def nearest_ranks(n: int) -> dict:
    """Nearest rank of each percentile over n values, 1-based: ceil(p * n / 100), in integers."""
    return {p: -(-p * n // 100) for p in PERCENTILES}


def percentile_picks(pairs: list, reading: str = TIE_READING) -> tuple:
    """DL-182 (2) over [(manifest index, value)] in manifest order; returns (picks, ranks)."""
    if reading not in TIE_READINGS:
        raise QualStop("tie_reading", f"unknown tie reading {reading!r}")
    ranks = nearest_ranks(len(pairs))
    order = sorted(pairs, key=lambda t: (t[1], t[0]))          # value, then the lower manifest index
    picks = []
    for p in PERCENTILES:
        m, v = order[ranks[p] - 1]
        if reading == "lowest_index_at_value":
            m = min(i for i, val in pairs if val == v)
        picks.append(m)
    if len(set(picks)) != len(picks):
        raise QualStop("percentile_duplicate", "two percentile picks are the same image; DL-182 does not settle "
                                               "it (QUAL-1 Q2): STOP for ORCH")
    return picks, ranks


def random_picks(d: list, exclude, *, guard) -> tuple:
    """DL-182 (3): random.Random(42).sample(sorted(D), len(D)); its first six entries outside the percentile
    picks. `guard` (version_guard's result) must exist: the draw never happens without it."""
    if not guard:
        raise QualStop("version_guard", "the random picks are drawn only after the version guard")
    shuffled = random.Random(SEED).sample(sorted(d), len(d))
    out = [m for m in shuffled if m not in set(exclude)][:N_RANDOM]
    if len(out) != N_RANDOM:
        raise QualStop("random_short", "fewer than six random picks")
    return out, sha256_bytes(json.dumps(shuffled).encode("ascii"))


def tie_readings_word(pairs: list) -> str:
    """QUAL-A-1 Q2: the picks under the literal reading "lowest_index_at_value" beside those of the reading of record
    ("rank"): "same", "different", or "literal_undefined" when the literal reading picks one image twice. It selects
    nothing; select/record.json keeps the word and the console prints the word only."""
    rank, _ = percentile_picks(pairs, "rank")
    try:
        literal, _ = percentile_picks(pairs, "lowest_index_at_value")
    except QualStop as e:
        if e.code == "percentile_duplicate":
            return "literal_undefined"
        raise
    return "same" if literal == rank else "different"


def select_dl182(off: Official, n: int, *, guard, reading: str = TIE_READING) -> dict:
    d = eligible(off, n)
    pairs = scored(off, d)
    pct, ranks = percentile_picks(pairs, reading)
    rnd, shuffle_sha = random_picks(d, pct, guard=guard)
    roles = [f"p{p}" for p in PERCENTILES] + [f"r{i + 1}" for i in range(N_RANDOM)]
    return {"indices": pct + rnd, "roles": roles, "n_d": len(d), "ranks": ranks, "shuffle_sha256": shuffle_sha,
            "tie_readings": tie_readings_word(pairs)}


def select_first_eligible(off: Official, n: int) -> dict:
    """Phase A's VAL layout set: the first nine eligible images in VAL manifest order (a layout check, not a
    selection)."""
    d = eligible(off, n)
    if len(d) < K_ALL:
        raise QualStop("eligibility_short", "fewer than nine eligible VAL images")
    return {"indices": d[:K_ALL], "roles": [f"v{i + 1}" for i in range(K_ALL)], "n_d": len(d), "ranks": None,
            "shuffle_sha256": None}


# ----------------------------------------------------------------------------------------------- S file
def s_document(split: str, sel: dict, ids: list, *, rule: str, reading: str | None, nonce: str | None = None) -> dict:
    """S.json. `nonce` (QUAL-AUD-1 N-01: os.urandom(16).hex(), read by nothing) keeps the printed sha256 of S.json
    from naming S by enumeration of the few candidate sets."""
    picks = [{"position": i + 1, "role": role, "manifest_index": m, "image_id": ids[m]}
             for i, (role, m) in enumerate(zip(sel["roles"], sel["indices"]))]
    if len({p["manifest_index"] for p in picks}) != len(picks):
        raise QualStop("s_duplicate", "S names one image twice")
    return {"format": SELECT_FORMAT, "split": split, "rule": rule, "tie_reading": reading, "n_rows": len(ids),
            "n_eligible": sel["n_d"], "ranks": sel["ranks"], "shuffle_sha256": sel["shuffle_sha256"], "picks": picks,
            "nonce": os.urandom(16).hex() if nonce is None else nonce}


def read_s(run: Path, expected_sha256: str | None) -> tuple:
    """<run>/select/S.json, held to the sha256 the select step printed (when given). Returns (doc, sha256)."""
    p = run / "select" / "S.json"
    data = p.read_bytes()
    got = sha256_bytes(data)
    if expected_sha256 is not None and got != expected_sha256:
        raise QualStop("s_changed", "select/S.json is not the file the select step wrote")
    doc = strict_json(data, "S.json")
    if doc.get("format") != SELECT_FORMAT or [p["position"] for p in doc["picks"]] != list(range(1, K_ALL + 1)):
        raise QualStop("s_invalid", "select/S.json is not a qual1-select/1 file of nine positions")
    return doc, got


# ---------------------------------------------------------------------------------------------------- CLI
def add_cache_args(ap) -> None:
    ap.add_argument("--corruption-cache", help="the corruption cache's root (TEST: the session's cache)")
    ap.add_argument("--cache-canonical-sha256")
    ap.add_argument("--cache-manifest-file-sha256")
    ap.add_argument("--verification-record")
    ap.add_argument("--verification-record-sha256")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the run folder (the kit made it)")
    ap.add_argument("--split", required=True, choices=("test", "val"))
    ap.add_argument("--export", help="TEST: the committed bound export (reports/derived/tday_inventory.json)")
    ap.add_argument("--export-sha256", help="TEST: its sha256 (SUMMARY_E.txt's bound export)")
    ap.add_argument("--session-root", help="TEST: the session checkout that holds reports/tday (the binding's root)")
    ap.add_argument("--val-artifact", help="VAL: a clean VAL per-image artifact, for eligibility (Phase A only)")
    ap.add_argument("--preflight-record", action="append", default=[],
                    help="VAL: MODEL:CONDITION=DIR, a VAL record for the preflight (repeat; Phase A only)")
    ap.add_argument("--preflight-only", action="store_true", help="run the preflight P1 alone (the kit's block P1)")
    ap.add_argument("--expect-commit", help="the commit the run is made from (TEST: required, the audited commit)")
    add_cache_args(ap)
    return ap


def expected_commit(args):
    """TEST runs only from the audited commit, named on the command line (AM-27 item 5)."""
    c = getattr(args, "expect_commit", None)
    if args.split == "test" and not (isinstance(c, str) and len(c) == 40 and all(ch in "0123456789abcdef" for ch in c)):
        raise QualStop("cli", "TEST needs --expect-commit, the audited commit's 40-character sha")
    return c


SELECT_OUTPUTS = ("select/S.json", "select/record.json")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_step("preflight" if args.preflight_only else "select", args, _main)


def _main(args) -> int:
    # Up to the end of the preflight nothing per-image is opened: every stop there is a refusal (exit 3), not a
    # stopped run (QUAL-AUD-1 A.5). The run starts when the E6 record's per-image file is opened below.
    try:
        run = run_dir(args.run)
        if not args.preflight_only:
            outputs_absent(run, SELECT_OUTPUTS)
        guard = version_guard()
        repo = repo_state(expected_commit(args))
    except QualStop as e:
        raise PreflightRefusal(e.code, e.detail, e.private) from None
    say("PASS", f"version guard: {guard['implementation']} {guard['python']} in {guard['image_digest'][:19]}...; "
                f"checkout {repo['commit'][:12]}... (governed paths clean: {repo['governed_paths_clean']})")
    pre = preflight(args, guard, repo)                       # before any per-image file (QUAL-AUD-1 A.1)
    if args.preflight_only:
        say("RESULT:", f"QUAL PREFLIGHT P1 PASS: {len(pre['records'])} records, {pre['listing']['images']} images, "
                       f"{len(CELLS)} cells; nothing per-image opened")
        return 0
    record = {"format": RECORD_FORMAT, "step": "select", "split": args.split, "guard": guard, "repo": repo,
              "inputs": {}, "preflight": pre}
    if args.split == "test":
        inv = open_test_inventory(args.export, args.export_sha256)
        say("PASS", f"export and binding read through TMAN's reader (export sha256 {args.export_sha256[:12]}...)")
        model_sha = {m: record_model_sha256(m) for m in MODEL_ORDER}
        off = load_test_official(inv, "E6", CLEAN, args.session_root)
        check_record_identity(off, split="test", condition=CLEAN, checkpoint_sha256=model_sha["E6"])
        records = {record_label("E6", CLEAN): off.files_sha256}
        for model, condition in RECORD_KEYS:                 # QUAL-AUD-1 F-02: the other twelve, right after E6's
            if (model, condition) == ("E6", CLEAN):
                continue
            other = load_test_official(inv, model, condition, args.session_root)
            check_record_identity(other, split="test", condition=condition, checkpoint_sha256=model_sha[model])
            if other.files_sha256 != pre["records"][record_label(model, condition)]["files_sha256"]:
                raise QualStop("record_changed", f"{other.label}: its files changed since the preflight")
            records[other.label] = other.files_sha256
            del other
        say("PASS", f"records: {len(records)} loaded through the binding and checked (identity, rows, the model's "
                    f"sha256 of record); their files_sha256 kept in select/record.json")
        ids = manifest_order("test", off)
        sel = select_dl182(off, N_SPLIT["test"], guard=guard)
        doc = s_document("test", sel, ids, rule="DL-182 (AM-27 item 1)", reading=TIE_READING)
        record["inputs"]["export_sha256"] = args.export_sha256
        record["records"] = {lab: records[lab] for lab in (record_label(m, c) for m, c in RECORD_KEYS)}
        record["tie_readings"] = sel["tie_readings"]                 # QUAL-A-1 Q2: one word, selects nothing
    else:
        if not args.val_artifact:
            raise QualStop("cli", "VAL needs --val-artifact")
        off = load_val_official(args.val_artifact, "VAL eligibility record")
        check_record_identity(off, split="val", condition=CLEAN, checkpoint_sha256=None)
        ids = manifest_order("val", off)
        sel = select_first_eligible(off, N_SPLIT["val"])
        doc = s_document("val", sel, ids, rule="first nine eligible VAL images (Phase A layout check)",
                         reading=None)
    record["inputs"]["record"] = {"source": off.source, "files_sha256": off.files_sha256,
                                  "split_manifest_sha256": off.summary["dataset"]["split_manifest_sha256"]}
    s_sha = write_new(run / "select" / "S.json", canonical_bytes(doc))
    record["outputs"] = {"select/S.json": s_sha}
    rec_sha = write_new(run / "select" / "record.json", canonical_bytes(record))
    say("PASS", f"manifest order: {len(ids)} rows; the record's split_manifest_sha256 "
                f"{off.summary['dataset']['split_manifest_sha256'][:12]}... and every row id match")
    say("PASS", f"eligible {sel['n_d']} of {len(ids)}" + (
        f"; nearest ranks {sel['ranks'][10]}, {sel['ranks'][50]}, {sel['ranks'][90]} (tie reading "
        f"{TIE_READING}); shuffle sha256 {sel['shuffle_sha256'][:12]}..." if sel["ranks"] else ""))
    if "tie_readings" in sel:
        say("PASS", f"tie readings recorded: {sel['tie_readings']}")
    say("RESULT:", f"QUAL SELECT OK: {len(doc['picks'])} images; S.json sha256 {s_sha}; record sha256 {rec_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
