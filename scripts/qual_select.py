#!/usr/bin/env python3
"""QUAL-1, step 1 of 3 (AM-27; DL-182): the qualitative image set S, and what the three qual steps share.

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
split_manifest_sha256 must equal the artifact's. On VAL (the Phase A layout check, not a selection) S is the
first nine eligible images in VAL manifest order, read from a VAL per-image artifact given by path.

Output: <run>/select/S.json (the positions, manifest indices and image ids of S: data of the run, never
printed) and <run>/select/record.json (inputs and their sha256). The console shows counts, hashes and
PASS or STOP lines only: no image id, no file name of an image, no per-image value.

  python -B scripts/qual_select.py --run DIR --split test --export reports/derived/tday_inventory.json \
      --export-sha256 HEX --session-root DIR
  python -B scripts/qual_select.py --run DIR --split val --val-artifact DIR
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


def k_of(condition: str) -> int:
    """How many images of S a condition is predicted on: all nine clean, the first three under a corruption."""
    return K_ALL if condition == CLEAN else K_CELLS


class QualStop(RuntimeError):
    """A named stop: the kit prints `STOP <code>: <detail>` and exits 1. `detail` is written in this kit and names
    no image id, image file name or per-image value; `private` (a library's own message, which may) is kept in the
    run's log on TEST and printed on VAL (run_step)."""

    def __init__(self, code: str, detail: str, private: str | None = None):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.private = private


def say(kind: str, text: str) -> None:
    print(f"{kind} {text}", flush=True)


def _private(step: str, args, text: str) -> str:
    """VAL: to stderr. TEST: appended to <run>/logs/<step>.txt, a file for Ice, never for an AI session."""
    if getattr(args, "split", None) == "val":
        print(text, file=sys.stderr, flush=True)
        return "printed above"
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
    """Every qual step's exit. A named stop prints its code and detail; any other error prints its type only, and
    its message and traceback (which may name an image) go to _private."""
    import traceback
    try:
        return body(args)
    except QualStop as e:
        say("STOP", f"{e.code}: {e.detail}")
        if e.private:
            _private(step, args, f"STOP {e.code}: {e.private}")
        return 1
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
    p = no_test(path)
    if not p.is_dir():
        raise QualStop("run_missing", "the run folder does not exist (the kit creates it)")
    return p


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
    named by PLANTSEG_IMAGE_DIGEST, which the kit sets from the digest it runs (the repository's convention)."""
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


# ------------------------------------------------------------------------------------------- official rows
class Official:
    """One official per-image artifact, verified: its summary, its rows by manifest index, its NPZ arrays and
    the sha256 of its files. Nothing here is printed."""

    def __init__(self, *, label, directory, summary, rows, npz, files_sha256, source):
        self.label, self.directory, self.summary = label, directory, summary
        self.rows, self.npz, self.files_sha256, self.source = rows, npz, files_sha256, source

    @property
    def device(self) -> str:
        return self.summary["run"]["env"]["device"]

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


def load_test_official(inv, model: str, condition: str, session_root) -> Official:
    """One TEST artifact of record through the binding (load_bound: the four files against the record, the run
    manifest, the canvas protocol, the stage, model role and precision, the checkpoint against the model row)."""
    from src.stats.heldout_inputs import LrepRefusal, load_bound
    from src.stats.ingest import Policy
    spec = MODELS[model]
    cell = None if condition == CLEAN else condition
    label = f"{model} {condition}"
    try:
        ba = load_bound(inv, slot=spec["slot"], stage=spec["stage"], precision=spec["precision"],
                        scope=CLEAN if cell is None else "corrupted", cell=cell,
                        input_root=no_test(session_root), policy=Policy.OFFICIAL)
    except LrepRefusal as e:
        raise QualStop("record_refused", f"{label}: {e.code}", private=str(e)) from None
    files = dict(ba.record["files_sha256"])
    summary, rows, npz = _read_payload(ba.loaded.directory, files, label)
    return Official(label=label, directory=ba.loaded.directory, summary=summary, rows=rows, npz=npz,
                    files_sha256=files, source={"binding_key": [spec["slot"], spec["precision"],
                                                                ba.record["scope"], ba.record["cell"]],
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


def check_record_identity(off: Official, *, split: str, condition: str, checkpoint_sha256: str | None) -> None:
    """The record is the one the figure names: split, canvas protocol, condition, all rows, and (when given) the
    model's sha256 as the evaluator recorded it."""
    from src.eval.protocols import CANVAS_PROTOCOL_ID
    ds, run = off.summary["dataset"], off.summary["run"]
    want = {"type": "clean", "name": None, "severity": None} if condition == CLEAN else {
        "type": "corruption", "name": condition.rpartition("_s")[0], "severity": int(condition.rpartition("_s")[2])}
    problems = []
    if ds["split"] != split:
        problems.append("split")
    if ds["preprocess_protocol"] != CANVAS_PROTOCOL_ID:
        problems.append("protocol")
    if ds["condition"] != want:
        problems.append("condition")
    if ds["actual_rows"] != N_SPLIT[split] or len(off.rows) != N_SPLIT[split]:
        problems.append("rows")
    if checkpoint_sha256 is not None and run["checkpoint_sha256"] != checkpoint_sha256:
        problems.append("checkpoint_sha256")
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
def s_document(split: str, sel: dict, ids: list, *, rule: str, reading: str | None) -> dict:
    picks = [{"position": i + 1, "role": role, "manifest_index": m, "image_id": ids[m]}
             for i, (role, m) in enumerate(zip(sel["roles"], sel["indices"]))]
    if len({p["manifest_index"] for p in picks}) != len(picks):
        raise QualStop("s_duplicate", "S names one image twice")
    return {"format": SELECT_FORMAT, "split": split, "rule": rule, "tie_reading": reading, "n_rows": len(ids),
            "n_eligible": sel["n_d"], "ranks": sel["ranks"], "shuffle_sha256": sel["shuffle_sha256"], "picks": picks}


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
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the run folder (the kit made it)")
    ap.add_argument("--split", required=True, choices=("test", "val"))
    ap.add_argument("--export", help="TEST: the committed bound export (reports/derived/tday_inventory.json)")
    ap.add_argument("--export-sha256", help="TEST: its sha256 (SUMMARY_E.txt's bound export)")
    ap.add_argument("--session-root", help="TEST: the session checkout that holds reports/tday (the binding's root)")
    ap.add_argument("--val-artifact", help="VAL: a clean VAL per-image artifact, for eligibility (Phase A only)")
    ap.add_argument("--expect-commit", help="the commit the run is made from (TEST: required, the audited commit)")
    return ap


def expected_commit(args):
    """TEST runs only from the audited commit, named on the command line (AM-27 item 5)."""
    c = getattr(args, "expect_commit", None)
    if args.split == "test" and not (isinstance(c, str) and len(c) == 40 and all(ch in "0123456789abcdef" for ch in c)):
        raise QualStop("cli", "TEST needs --expect-commit, the audited commit's 40-character sha")
    return c


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_step("select", args, _main)


def _main(args) -> int:
    run = run_dir(args.run)
    guard = version_guard()
    repo = repo_state(expected_commit(args))
    say("PASS", f"version guard: {guard['implementation']} {guard['python']} in {guard['image_digest'][:19]}...; "
                f"checkout {repo['commit'][:12]}... (governed paths clean: {repo['governed_paths_clean']})")
    record = {"format": RECORD_FORMAT, "step": "select", "split": args.split, "guard": guard, "repo": repo,
              "inputs": {}}
    if args.split == "test":
        if not (args.export and args.export_sha256 and args.session_root):
            raise QualStop("cli", "TEST needs --export, --export-sha256 and --session-root")
        inv = open_test_inventory(args.export, args.export_sha256)
        say("PASS", f"export and binding read through TMAN's reader (export sha256 {args.export_sha256[:12]}...)")
        off = load_test_official(inv, "E6", CLEAN, args.session_root)
        check_record_identity(off, split="test", condition=CLEAN, checkpoint_sha256=None)
        ids = manifest_order("test", off)
        sel = select_dl182(off, N_SPLIT["test"], guard=guard)
        doc = s_document("test", sel, ids, rule="DL-182 (AM-27 item 1)", reading=TIE_READING)
        record["inputs"]["export_sha256"] = args.export_sha256
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
