#!/usr/bin/env python3
"""The efficiency table of record (LEFF B-2: the LEFF Phase A plan §2 decision (v); LEFF-AUDIT-RULE-1 A2).

    python -B scripts/efficiency_table.py build --session DIR [--session DIR2] --pins FILE --out-dir DIR

A sitting is one session directory of the operator's block: <session>/<stage>/<group>/ holds the one record that
`scripts/profile_efficiency.py --of-record` wrote for that stage and metric group (static, latency, memory). Up to two
sittings of one host type make one table (A2; LAT-HOST-1). The pins file (efficiency_pins/1) names, per stage of the
efficiency analysis, the sha256 of the file each run is handed and, for an INT8 stage, of its converted artifact and
of its x86 latency copy. Every record is hashed when it is listed and again after all are read. Refused:
  [table_too_many_sessions]      a third --session
  [path_forbidden_name]          a path naming "test", as given or resolved, before it is opened (a session's
                                 records included)
  [out_dir_inside_repository]    an --out-dir inside the repository; [out_dir_not_empty] an existing, non-empty one
  [table_pins_invalid]           a pins file that is not efficiency_pins/1 with a sha256 for every stage it needs
  [table_record_not_of_record]   a record that is not plantseg-efficiency/2.0.0 of record (or not of the analysis,
                                 or with a deferred metric, or another role than its stage and group take)
  [table_sha256_mismatch]        a record changed after it was listed
  [table_threads_mixed] [table_interop_mixed] [table_host_mixed] [table_image_mixed] [table_commit_mixed]
                                 the records differ in the thread count, the inter-op count, the host type
                                 (host_type_fingerprint), the image (digest and PL-26 fingerprint) or the commit
  [table_stage_missing] [table_stage_duplicate]   a stage and group with no record, or with two
  [table_artifact_not_pinned]    a record whose handed file, converted artifact or x86 copy differs from the pins
It writes efficiency_table.json (efficiency_table/1), created exclusively outside the repository: one row per stage
of EFFICIENCY_ANALYSIS_STAGES, every quantity with its record's path and sha256, and the sittings with every
record's host label; then a RESULT line. Exit 0 written, 2 refused. Descriptive: no test, no threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.profile_efficiency import METRIC_GROUPS, forbidden_path_error  # noqa: E402
from src.eval.efficiency import (ARTIFACT_ROLE_ACCURACY, ARTIFACT_ROLE_X86_LATENCY,  # noqa: E402
                                 EFFICIENCY_ANALYSIS_STAGES, EFFICIENCY_SCHEMA, INT8_STAGES)

TABLE_FORMAT = "efficiency_table/1"
PINS_FORMAT = "efficiency_pins/1"
TABLE_NAME = "efficiency_table.json"
MAX_SESSIONS = 2
EXIT_OK, EXIT_REFUSED = 0, 2


class TableRefused(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def refuse(code: str, message: str):
    raise TableRefused(code, message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def role_of(stage: str, group: str) -> str:
    """The role a stage and group take of record (the which-artifact rule): INT8 latency and memory on the x86 copy."""
    return ARTIFACT_ROLE_X86_LATENCY if stage in INT8_STAGES and group != "static" else ARTIFACT_ROLE_ACCURACY


def check_paths(sessions, pins, out_dir) -> Path:
    for label, value in [("--session", s) for s in sessions] + [("--pins", pins), ("--out-dir", out_dir)]:
        error = forbidden_path_error(label, value)
        if error:
            refuse("path_forbidden_name", error)
    out = Path(out_dir).resolve()
    if REPO == out or REPO in out.parents:
        refuse("out_dir_inside_repository", "the table is written outside the repository")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        refuse("out_dir_not_empty", f"--out-dir {out} exists and is not an empty directory")
    return out


def read_pins(path) -> tuple[dict, str]:
    try:
        raw = Path(path).read_bytes()
        pins = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        refuse("table_pins_invalid", f"{path} cannot be read as JSON: {e}")
    stages = pins.get("stages") if isinstance(pins, dict) else None
    if not isinstance(pins, dict) or pins.get("format") != PINS_FORMAT or not isinstance(stages, dict):
        refuse("table_pins_invalid", f"{path} is not an {PINS_FORMAT} file")
    for stage in EFFICIENCY_ANALYSIS_STAGES:
        need = ("checkpoint_sha256", "converted_sha256", "x86_sha256") if stage in INT8_STAGES else ("checkpoint_sha256",)
        bad = [k for k in need if not _hex64((stages.get(stage) or {}).get(k))]
        if bad:
            refuse("table_pins_invalid", f"{path} pins no 64-hex {bad} for {stage}")
    return stages, sha256_bytes(raw)


def list_records(sessions) -> list:
    """Every record of every sitting: (session, stage, group directory, path, sha256 at listing, parsed record)."""
    found = []
    for session in sessions:
        root = Path(session)
        for stage in EFFICIENCY_ANALYSIS_STAGES:
            for group in METRIC_GROUPS:
                d = root / stage / group
                if not d.is_dir():
                    continue
                for name in sorted(os.listdir(d)):
                    path = d / name
                    error = forbidden_path_error("a session record", path)     # SL-1: before it is opened
                    if error:
                        refuse("path_forbidden_name", error)
                    if not (name.startswith("efficiency_") and name.endswith(".json") and path.is_file()):
                        refuse("table_record_not_of_record", f"{path} is not an efficiency record")
                    raw = path.read_bytes()
                    try:
                        rec = json.loads(raw.decode("utf-8"))
                    except ValueError as e:
                        refuse("table_record_not_of_record", f"{path} is not JSON: {e}")
                    found.append({"session": str(root), "dir": (stage, group), "path": path,
                                  "sha256": sha256_bytes(raw), "record": rec})
    return found


def record_problem(entry: dict) -> str | None:
    rec = entry["record"]
    if not isinstance(rec, dict) or rec.get("schema") != EFFICIENCY_SCHEMA or rec.get("of_record") is not True:
        return f"schema {rec.get('schema') if isinstance(rec, dict) else None!r}, of_record " \
               f"{rec.get('of_record') if isinstance(rec, dict) else None!r}"
    stage, group = (rec.get("provenance") or {}).get("stage"), rec.get("metric_group")
    if not rec.get("efficiency_analysis_member") or stage not in EFFICIENCY_ANALYSIS_STAGES:
        return f"stage {stage!r} is not of the efficiency analysis"
    if group not in METRIC_GROUPS:
        return f"metric group {group!r}"
    if rec.get("deferred"):
        return f"deferred {sorted(rec['deferred'])}"
    if (rec.get("provenance") or {}).get("artifact_role") != role_of(stage, group):
        return f"role {(rec.get('provenance') or {}).get('artifact_role')!r}; {stage} {group} takes {role_of(stage, group)}"
    return None


def _one(values, code: str, what: str):
    distinct = sorted({json.dumps(v, sort_keys=True) for v in values})
    if len(distinct) != 1:
        refuse(code, f"the records differ in {what}: {distinct[:4]}")
    return json.loads(distinct[0])


def build(sessions, pins_path, out_dir) -> tuple[Path, dict]:
    if len(sessions) > MAX_SESSIONS:
        refuse("table_too_many_sessions",
               f"{len(sessions)} sittings: a table takes at most {MAX_SESSIONS} (LEFF-AUDIT-RULE-1 A2)")
    out = check_paths(sessions, pins_path, out_dir)
    pins, pins_sha = read_pins(pins_path)
    entries = list_records(sessions)
    for e in entries:
        problem = record_problem(e)
        if problem:
            refuse("table_record_not_of_record", f"{e['path']}: {problem}")
    recs = [e["record"] for e in entries]
    tp = [r.get("thread_policy") or {} for r in recs]
    threads = _one([t.get("intra_op_requested") for t in tp] + [t.get("intra_op_observed") for t in tp],
                   "table_threads_mixed", "the intra-op thread count (requested and observed)")
    interop = _one([(t.get("inter_op") or {}).get("observed") for t in tp], "table_interop_mixed",
                   "the inter-op thread count")
    host = _one([(r.get("host") or {}).get("fingerprint") for r in recs], "table_host_mixed",
                "the host type (host_type_fingerprint)")
    image = _one([r.get("image") for r in recs], "table_image_mixed", "the image (digest and PL-26 fingerprint)")
    commit = _one([(r.get("code") or {}).get("repo_commit") for r in recs], "table_commit_mixed", "the commit")
    slots = {}
    for e in entries:
        key = ((e["record"]["provenance"]).get("stage"), e["record"]["metric_group"])
        slots.setdefault(key, []).append(e)
    for stage in EFFICIENCY_ANALYSIS_STAGES:
        for group in METRIC_GROUPS:
            got = slots.get((stage, group), [])
            if not got:
                refuse("table_stage_missing", f"no record of {stage} {group} in {list(map(str, sessions))}")
            if len(got) > 1:
                refuse("table_stage_duplicate", f"{stage} {group} has {len(got)} records: {[str(e['path']) for e in got]}")
    for e in entries:
        rec, stage = e["record"], e["record"]["provenance"]["stage"]
        prov, pin = rec["provenance"], pins[stage]
        bad = []
        if (prov.get("handed") or {}).get("sha256") != pin["checkpoint_sha256"]:
            bad.append("handed file")
        if stage in INT8_STAGES and prov.get("source_artifact_sha256") != pin["converted_sha256"]:
            bad.append("converted artifact")
        if prov.get("artifact_role") == ARTIFACT_ROLE_X86_LATENCY and \
                ((prov.get("binding") or {}).get("x86_copy") or {}).get("sha256") != pin["x86_sha256"]:
            bad.append("x86 latency copy")
        if bad:
            refuse("table_artifact_not_pinned", f"{e['path']}: the {bad} differ from the pins")
    for e in entries:
        if sha256_bytes(Path(e["path"]).read_bytes()) != e["sha256"]:
            refuse("table_sha256_mismatch", f"{e['path']} changed after it was listed")

    def cite(e, value):
        return {"value": value, "record": str(e["path"]), "record_sha256": e["sha256"]}

    rows = []
    for stage in EFFICIENCY_ANALYSIS_STAGES:
        st, la, me = (slots[(stage, g)][0] for g in ("static", "latency", "memory"))
        srec, lrec, mrec = st["record"], la["record"], me["record"]
        lat = lrec["latency"] or {}
        rows.append({
            "stage": stage,
            "parameters": cite(st, srec["parameters"]),
            "footprint": cite(st, srec["footprint"]),
            "footprint_source": cite(st, srec["footprint_source"]),
            "serialized_size": cite(st, srec["artifact"]),
            "flops": cite(st, srec["compute"]),
            "latency": cite(la, {k: lat.get(k) for k in ("latency_median_ms", "latency_iqr_ms", "latency_p95_ms",
                                                         "latency_q1_ms", "latency_q3_ms", "latency_samples",
                                                         "intra_op_observed", "protocol")}),
            "peak_memory": cite(me, {k: (mrec["memory"] or {}).get(k) for k in (
                "peak_rss_delta_bytes", "negative_delta", "slack_bytes", "forwards", "ru_maxrss_consistent")}),
            "artifact_role_latency_memory": role_of(stage, "latency"),
        })
    table = {
        "format": TABLE_FORMAT,
        "measurement_role": "descriptive", "cpu_proxy": True, "on_device": False,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sittings": [{"session": str(s), "records": sum(1 for e in entries if e["session"] == str(Path(s))),
                      "host_labels": sorted({(e["record"].get("host") or {}).get("label") or "" for e in entries
                                             if e["session"] == str(Path(s))})} for s in sessions],
        "host_type_fingerprint": host, "threads": threads, "inter_op": interop, "image": image,
        "repo_commit": commit, "pins": {"path": str(pins_path), "sha256": pins_sha},
        "records": [{"path": str(e["path"]), "sha256": e["sha256"], "host_label": (e["record"].get("host") or {})
                     .get("label"), "host_note": (e["record"].get("host") or {}).get("note")} for e in entries],
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / TABLE_NAME
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(table, indent=2, sort_keys=True))
    return path, table


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The efficiency table of record (LEFF B-2).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build efficiency_table.json from up to two sittings")
    b.add_argument("--session", action="append", required=True, help="a sitting's session directory (repeatable)")
    b.add_argument("--pins", required=True, help="the efficiency_pins/1 file")
    b.add_argument("--out-dir", required=True, help="a fresh directory outside the repository")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path, table = build(args.session, args.pins, args.out_dir)
    except TableRefused as e:
        print(f"RESULT: REFUSED [{e.code}] {e}")
        return EXIT_REFUSED
    digest = sha256_bytes(path.read_bytes())
    print(f"RESULT: TABLE WRITTEN ({len(table['rows'])} stages, {len(table['records'])} records, "
          f"{len(table['sittings'])} sitting(s)) {path} sha256 {digest}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
