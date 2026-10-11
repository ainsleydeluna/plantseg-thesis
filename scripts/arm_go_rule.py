#!/usr/bin/env python3
"""AM-18 item 5's go rule for an arm teacher (lane L-TEACHER-ARMS, commit C4; ARMS plan §1.1 and §2 entries 12 and
18; Phase B-2 rulings R10, R11).

    python -B scripts/arm_go_rule.py --arm R1|R2 --teacher-record PATH --teacher-record-sha256 HEX
        --arm-output teacher_d2_arm-<X>_<UTC>.json --control-output teacher_d2_control_<UTC>.json
        --item1b-output teacher_d2_<UTC>.json --out PATH [--host-fault-dl DL-n] [--void-session-dl DL-n]
        [--generated-utc UTC]

E3-X and E7-X run if either (literal constants, exact float comparisons):
    VAL_X > 0.38576993346214294; or
    VAL_X >= 0.37576993346214294 and ECE_X < ECE_B.
Otherwise X is a teacher-only result (no_go). VAL_X and ECE_X come from DIAG's D2 arm output for X (its
all_class_miou and ECE), ECE_B from the control output of the same session (the teacher of record re-scored
first, at the same commit), and item 1(b)'s ECE is reported beside it. Before any decision:
  * the teacher record is arm_teacher_record/1 of this arm, complete, with the given sha256, and the arm output
    scored exactly its M12-selected checkpoint (by sha256; cm_go_rule_binds_m12_checkpoint);
  * the outputs are DIAG's D2 outputs (lane, script, roles: arm X; record control; record item1); the control
    was checked against this item 1(b) output; control and arm ran at the same code commit, the control first;
  * all three ran on one host, the item 1(b) host (DIAG's environment cpu_model; the hostnames, which name a
    container inside Docker, are recorded only), unless --host-fault-dl names a RECORDED row (at HEAD of this
    clone) that documents that host's fault and names the item 1(b) output's sha256;
  * the control reproduced 0.38576993346214294 within 1e-5. If not, the session is void: a STOP, nothing written
    (exit 1); when --void-session-dl names the RECORDED row of an earlier void session of this X, the recurrence
    makes X not evaluable: the record is written with verdict not_evaluable (exit 1).
Exit: 0 written (go or no_go); 1 STOP (void session) or not_evaluable (written); 2 refused, nothing written.
The record, arm_go_rule/1, is created exclusively at --out (outside the repository; the orchestrator commits a
copy at reports/arms/arm_go_rule_<arm>.json and enters the outcome with the values and the artifacts' sha256 in
the decision log before E3-X launches). --generated-utc is for synthetic runs: it accepts DIAG smoke outputs only
and a real run refuses them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2
GIT_ROOT = REPO                         # the clone whose HEAD holds the decision log (a smoke points it elsewhere)
RECORD_FORMAT = "arm_go_rule/1"
TEACHER_RECORD_FORMAT = "arm_teacher_record/1"
SCRIPT_REL = "scripts/arm_go_rule.py"
D2_SCRIPT, D2_LANE = "scripts/teacher_d2_calibration.py", "L-TEACHER-DIAG"
VAL_GATE = 0.38576993346214294          # AM-18 item 5: the teacher of record's VAL all-class mIoU (R3)
VAL_FLOOR = 0.37576993346214294         # AM-18 item 5: 1 pp below it
REPRO_TOLERANCE = 1e-5                  # AM-18 item 5: the control reproduces VAL_GATE within 1e-5
ARMS = ("R1", "R2")


class Refused(RuntimeError):
    pass


def decide(val_x: float, ece_x: float, ece_b: float) -> tuple[str, dict]:
    """The two conditions, as written: strict > for the first, >= and strict < for the second."""
    c1 = val_x > VAL_GATE
    c2 = val_x >= VAL_FLOOR and ece_x < ece_b
    return ("go" if c1 or c2 else "no_go"), {"val_x_gt_val_gate": c1, "val_x_ge_val_floor": val_x >= VAL_FLOOR,
                                             "ece_x_lt_ece_b": ece_x < ece_b}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_d2(path, *, synthetic: bool, what: str, statuses=("written",)) -> tuple[dict, str]:
    """A DIAG D2 output: this lane's script, the run mode the session requires, one of `statuses`."""
    p = Path(path)
    if not p.is_file():
        raise Refused(f"{what} {p} is not a file")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Refused(f"{what}: not JSON ({e})") from e
    if not isinstance(doc, dict) or doc.get("lane") != D2_LANE or doc.get("script") != D2_SCRIPT:
        raise Refused(f"{what} is not a {D2_SCRIPT} output")
    smoke = doc.get("artifact_status") == "smoke"
    if smoke != synthetic:
        raise Refused(f"{what}: artifact_status {doc.get('artifact_status')!r}; "
                      + ("a synthetic run takes DIAG smoke outputs only" if synthetic else "a real run refuses a smoke "
                                                                                          "output"))
    if doc.get("status") not in statuses:
        raise Refused(f"{what} has status {doc.get('status')!r}")
    return doc, _sha(p)


def _num(doc: dict, *keys):
    v = doc
    for k in keys:
        v = v.get(k) if isinstance(v, dict) else None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _host(doc: dict):
    """The machine a DIAG output ran on: its CPU model (DIAG's environment block). The hostname is recorded for
    information only: inside a container it names the container, which changes from run to run."""
    return (doc.get("environment") or {}).get("cpu_model")


def dl_row(dl_id: str, need: tuple, what: str) -> dict:
    """A RECORDED decision-log row at HEAD of this clone carrying every string in `need`."""
    import scripts.preflight_arms as pa
    from src.training.sweep_select import DECISION_LOG_REL, HeadSource
    src = HeadSource(GIT_ROOT)
    text = src.show("HEAD", DECISION_LOG_REL)
    if text is None:
        raise Refused(f"{DECISION_LOG_REL} is not at HEAD of this clone")
    try:
        row = pa.recorded_row(text.decode("utf-8", "replace"), dl_id, what=what, code="dl_row", need=need)
    except pa.GateError as e:
        raise Refused(str(e)) from e
    return {"dl": dl_id, "head": src.head(), "row_sha256": hashlib.sha256(row.encode("utf-8")).hexdigest()}


def evaluate(args) -> tuple[int, dict | None]:
    from src.eval.teacher_diag import utc_now
    synthetic = args.generated_utc is not None
    rec_p = Path(args.teacher_record)
    if not rec_p.is_file() or _sha(rec_p) != args.teacher_record_sha256:
        raise Refused(f"--teacher-record {rec_p}: missing, or its sha256 is not --teacher-record-sha256")
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    if not isinstance(rec, dict):
        raise Refused("--teacher-record is not a JSON object")
    sel = rec.get("selection")
    ck = sel.get("checkpoint") if isinstance(sel, dict) else None
    ck = ck if isinstance(ck, dict) else {}
    if rec.get("format") != TEACHER_RECORD_FORMAT or rec.get("arm") != args.arm or rec.get("status") != "complete" \
            or not isinstance(ck.get("sha256"), str):
        raise Refused(f"--teacher-record: format {rec.get('format')!r}, arm {rec.get('arm')!r}, status "
                      f"{rec.get('status')!r}: a complete record of {args.arm} with its selected checkpoint is scored")
    arm, arm_sha = read_d2(args.arm_output, synthetic=synthetic, what="--arm-output")
    item1b, item1b_sha = read_d2(args.item1b_output, synthetic=synthetic, what="--item1b-output")
    control, control_sha = read_d2(args.control_output, synthetic=synthetic, what="--control-output",
                                   statuses=("written", "not reproduced"))
    if arm.get("role") != "arm" or arm.get("arm_id") != args.arm:
        raise Refused(f"--arm-output is role {arm.get('role')!r} arm {arm.get('arm_id')!r}, not arm {args.arm}")
    if item1b.get("role") != "record" or item1b.get("purpose") != "item1":
        raise Refused("--item1b-output is not the record role's item 1(b) output")
    if control.get("role") != "record" or control.get("purpose") != "control":
        raise Refused("--control-output is not the record role's control re-score")
    x_sha = ((arm.get("teacher") or {}).get("checkpoint") or {}).get("sha256_verified")
    if x_sha != ck["sha256"]:
        raise Refused(f"the arm output scored checkpoint {x_sha}; {args.arm}'s M12 selection is {ck.get('name')} "
                      f"{ck['sha256']} (AM-18 item 2(c): X at its M12-selected checkpoint)")
    if ((control.get("inputs") or {}).get("of_record_output") or {}).get("sha256") != item1b_sha:
        raise Refused("the control was not checked against this --item1b-output")
    head_c, head_a = (control.get("code") or {}).get("head"), (arm.get("code") or {}).get("head")
    if not head_c or head_c != head_a:
        raise Refused(f"the control ran at {head_c} and the arm at {head_a}: the same commit is required")
    end_c, start_a = (control.get("environment") or {}).get("end_utc"), arm.get("start_utc")
    if not (isinstance(end_c, str) and isinstance(start_a, str) and end_c <= start_a):
        raise Refused(f"the control ended {end_c} and the arm started {start_a}: the control is scored first")
    hosts = {"item1b": _host(item1b), "control": _host(control), "arm": _host(arm)}
    hostnames = {k: (d.get("environment") or {}).get("hostname") for k, d in (("item1b", item1b),
                                                                              ("control", control), ("arm", arm))}
    fault = None
    if not hosts["item1b"] or len(set(hosts.values())) != 1:
        if args.host_fault_dl is None:
            raise Refused(f"hosts {hosts}: the session runs on the item 1(b) host, another only after a documented "
                          "fault of that host (--host-fault-dl)")
        fault = dl_row(args.host_fault_dl, (item1b_sha,), "--host-fault-dl (a documented fault of the host that "
                                                          "produced the item 1(b) output, naming its sha256)")
    val_x, ece_x = _num(arm, "pass", "all_class_miou"), _num(arm, "calibration", "ece")
    ece_b, miou_b = _num(control, "calibration", "ece"), _num(control, "pass", "all_class_miou")
    ece_1b = _num(item1b, "calibration", "ece")
    if val_x is None or ece_x is None:
        raise Refused("the arm output has no finite all_class_miou and ECE")
    gate = control.get("gates") if isinstance(control.get("gates"), dict) else {}
    reproduced = (control.get("status") == "written" and gate.get("passed") is True and miou_b is not None
                  and abs(miou_b - VAL_GATE) <= REPRO_TOLERANCE)
    void = None if args.void_session_dl is None else dl_row(
        args.void_session_dl, (args.teacher_record_sha256,), "--void-session-dl (an earlier void session of this X)")
    if not reproduced and void is None:
        print(f"STOP: the control did not reproduce {VAL_GATE!r} within {REPRO_TOLERANCE} (all_class_miou "
              f"{miou_b!r}, status {control.get('status')!r}): the session is void; nothing is written (AM-18 item 5)")
        return EXIT_STOP, None
    if reproduced and ece_b is None:
        raise Refused("the control output has no finite ECE")
    verdict, conditions = (decide(val_x, ece_x, ece_b) if reproduced else ("not_evaluable", None))
    doc = {"format": RECORD_FORMAT, "arm": args.arm, "verdict": verdict,
           "rule": {"val_gate": VAL_GATE, "val_floor": VAL_FLOOR, "reproduction_tolerance": REPRO_TOLERANCE,
                    "text": "VAL_X > val_gate, or VAL_X >= val_floor and ECE_X < ECE_B (AM-18 item 5)"},
           "values": {"val_x": val_x, "ece_x": ece_x, "ece_b": ece_b, "ece_item1b": ece_1b,
                      "control_all_class_miou": miou_b, "control_reproduced": reproduced,
                      "reprs": {k: repr(v) for k, v in (("val_x", val_x), ("ece_x", ece_x), ("ece_b", ece_b),
                                                         ("control_all_class_miou", miou_b))}},
           "conditions": conditions,
           "checkpoint": {"name": ck.get("name"), "sha256": ck["sha256"], "bytes": ck.get("bytes")},
           "teacher_record": {"path": str(rec_p), "sha256": args.teacher_record_sha256},
           "inputs": {"arm_output": {"path": str(args.arm_output), "sha256": arm_sha},
                      "control_output": {"path": str(args.control_output), "sha256": control_sha},
                      "item1b_output": {"path": str(args.item1b_output), "sha256": item1b_sha}},
           "session": {"hosts": hosts, "hostnames": hostnames, "host_fault": fault, "commit": head_a,
                       "control_end_utc": end_c,
                       "arm_start_utc": start_a, "void_session": void},
           "writer": {"script_rel": SCRIPT_REL, "script_sha256": _sha(Path(__file__).resolve())},
           "generated_utc": args.generated_utc or utc_now()}
    return (EXIT_OK if verdict in ("go", "no_go") else EXIT_STOP), doc


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AM-18 item 5 go rule (L-TEACHER-ARMS C4).", allow_abbrev=False)
    p.add_argument("--arm", choices=ARMS, required=True)
    for name in ("--teacher-record", "--teacher-record-sha256", "--arm-output", "--control-output",
                 "--item1b-output", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--host-fault-dl", default=None)
    p.add_argument("--void-session-dl", default=None)
    p.add_argument("--generated-utc", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.eval.teacher_diag import Refused as DiagRefused
    from src.eval.teacher_diag import refuse_test_path, require_outside_repo, write_json_exclusive
    try:
        for value, flag in ((args.teacher_record, "--teacher-record"), (args.arm_output, "--arm-output"),
                            (args.control_output, "--control-output"), (args.item1b_output, "--item1b-output"),
                            (args.out, "--out")):
            refuse_test_path(value, flag)
        out = require_outside_repo(args.out, "--out")
        if out.exists():
            raise Refused(f"--out {out} exists: the record is written once")
        rc, doc = evaluate(args)
    except (Refused, DiagRefused, OSError, ValueError) as e:
        print(f"RESULT: REFUSED -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    if doc is None:
        print("RESULT: STOP (void session)")
        return rc
    path = write_json_exclusive(out, doc)
    print(f"  VAL_X {doc['values']['reprs']['val_x']}  ECE_X {doc['values']['reprs']['ece_x']}  ECE_B "
          f"{doc['values']['reprs']['ece_b']}  (item 1(b) ECE {doc['values']['ece_item1b']!r})")
    print(f"RESULT: {doc['verdict'].upper()} -- {path} sha256 {_sha(path)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
