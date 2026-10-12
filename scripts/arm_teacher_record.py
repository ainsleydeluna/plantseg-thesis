#!/usr/bin/env python3
"""The arm teacher's provenance record and its divergence watch (lane L-TEACHER-ARMS, commit C4; AM-18 item 2;
ARMS plan decisions (iii) and (iv); DL-24 audit F20, F41; Phase B-2 rulings R4(b), R10, R11).

    python -B scripts/arm_teacher_record.py progress --work-dir DIR
    python -B scripts/arm_teacher_record.py record --arm R1|R2 --work-dir DIR --config PATH --config-sha256 HEX
        --eval-config PATH --out PATH [--generated-utc UTC]

The run's own logs decide (no hook is added; plan decision (iii)): mmengine's <work>/<timestamp>/vis_data/
scalars.json (a training row every 50 iterations with "iter" and the losses; a VAL row per validation with the
scores) and the M12 hook's teacher_selection_records.jsonl. A logged training loss (every key with "loss" in its
name) or VAL score (mIoU_full, mIoU_disease_full, mIoU) that is not finite is a divergence (AM-18 item 2: recorded,
reported and not rerun; the launcher refuses a repeat of a diverged arm, ruling R4(b)).

progress   read-only, during the run: the last logged iteration, the VAL rows, and the first non-finite value with
           its iteration. Exit 0 none, 1 one found, 2 refused.
record     writes arm_teacher_record/1 exclusively at --out (outside the repository; the orchestrator commits a
           copy at reports/arms/arm_teacher_record_<arm>.json). status:
             complete    40,000 iterations logged, ten VAL rows at 4,000..40,000, a final M12 selection whose
                         checkpoint is the interval checkpoint it names (sha256 and size), all ten interval
                         checkpoints present, no non-finite value, and the selected checkpoint's EMA structure
                         (state_dict, ema_state_dict with steps = its iteration; AM-18 item 2(a));
             diverged    the first non-finite value is recorded (first_nonfinite);
             incomplete  anything else: a stop by pod or host, repeated under AM-8a in a new _a<n> work dir.
           Exit 0 complete, 1 diverged or incomplete (the record is still written), 2 refused (nothing written).
The record names the work dir and attempt, both configurations (the arm's training config and the record
config it is evaluated with; plan decision (iv)), the registered difference (REG[arm]'s canonical JSON and its
sha256, which must equal the launch provenance's), the launch provenance files' sha256 and their rows, the logs'
sha256 (R2's launcher binds its reference log to R1's), the selection, every interval checkpoint's sha256, the
EMA checks and the writer. The launch provenance is bound to --work-dir by the work dir's name and attempt (the
laptop records a copy of the pod's folder, ruling R11). --generated-utc is for synthetic runs: a work dir whose
provenance records a launch refuses it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2
RECORD_FORMAT = "arm_teacher_record/1"
SCRIPT_REL = "scripts/arm_teacher_record.py"
ARM_CONFIGS = {"R1": "configs/teacher/arms/r1_bema.py", "R2": "configs/teacher/arms/r2_bema_rare.py"}
RECORD_CONFIG_REL = "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
WORK_DIR_RE = re.compile(r"teacher-(r1|r2)_s42_40k_a([1-9][0-9]*)")
MAX_ITERS, VAL_INTERVAL = 40000, 4000
VAL_ITERS = list(range(VAL_INTERVAL, MAX_ITERS + 1, VAL_INTERVAL))
SCORE_KEYS = ("mIoU_full", "mIoU_disease_full", "mIoU")
ARM_PROVENANCE, LAUNCH_PROVENANCE = "teacher_arm_launch_provenance.json", "teacher_launch_provenance.json"
RECORDS_FILE, SELECTION_FILE = "teacher_selection_records.jsonl", "teacher_selection.json"


class Refused(RuntimeError):
    pass


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def scalars_path(work: Path) -> Path | None:
    """The run's one <timestamp>/vis_data/scalars.json (a fresh work dir holds one run); None before the first log
    row (a stop before iteration 50)."""
    if not work.is_dir():
        raise Refused(f"--work-dir {work} is not a directory")
    found = sorted(work.glob("*/vis_data/scalars.json"))
    if len(found) > 1:
        raise Refused(f"{work}: {len(found)} <timestamp>/vis_data/scalars.json files; a launched arm run has one")
    return found[0] if found else None


def read_logs(work: Path) -> dict:
    """The training rows, VAL rows and selection records, and the first non-finite logged value (by iteration)."""
    sp = scalars_path(work)
    train, val, bad = [], [], []
    for n, ln in enumerate(sp.read_text(encoding="utf-8").splitlines() if sp else [], 1):
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError as e:
            raise Refused(f"{sp} line {n} is not JSON ({e})") from e
        if not isinstance(row, dict):
            raise Refused(f"{sp} line {n} is not an object")
        if isinstance(row.get("iter"), int) and not isinstance(row.get("iter"), bool):
            train.append(row)
            it = row["iter"]
            bad += [(it, "train", k, row[k]) for k in sorted(row) if "loss" in k and not _finite(row[k])]
        elif any(k in row for k in SCORE_KEYS):
            val.append(row)
            it = row.get("step")
            bad += [(it, "val", k, row[k]) for k in SCORE_KEYS if k in row and not _finite(row[k])]
    recs = []
    rp = work / RECORDS_FILE
    if rp.is_file():
        for n, ln in enumerate(rp.read_text(encoding="utf-8").splitlines(), 1):
            if ln.strip():
                try:
                    recs.append(json.loads(ln))
                except json.JSONDecodeError as e:
                    raise Refused(f"{rp} line {n} is not JSON ({e})") from e
        bad += [(r.get("iteration"), "selection_record", k, r.get(k)) for r in recs if isinstance(r, dict)
                for k in ("mIoU_full", "mIoU_disease_full") if k in r and not _finite(r[k])]
    first = min(bad, key=lambda b: (b[0] if isinstance(b[0], int) else float("inf"))) if bad else None
    return {"scalars": sp, "train": train, "val": val, "records": recs,
            "last_iter": max((r["iter"] for r in train), default=0),
            "first_nonfinite": None if first is None else {"iteration": first[0], "source": first[1], "key": first[2],
                                                           "value": repr(first[3])}}


def cmd_progress(args) -> int:
    from src.eval.teacher_diag import refuse_test_path
    refuse_test_path(args.work_dir, "--work-dir")
    logs = read_logs(Path(args.work_dir))
    print(f"last logged iteration {logs['last_iter']} of {MAX_ITERS}; VAL rows {len(logs['val'])}; selection records "
          f"{len(logs['records'])}")
    if logs["first_nonfinite"] is not None:
        f = logs["first_nonfinite"]
        print(f"NON-FINITE: {f['source']} {f['key']} = {f['value']} at iteration {f['iteration']}: the fine-tune is "
              "recorded as diverged (AM-18 item 2); the operator may stop it (plan decision (iii))")
        return EXIT_STOP
    print("RESULT: no non-finite logged value")
    return EXIT_OK


def _file(p: Path) -> dict:
    from src.eval.teacher_diag import file_sha256
    return {"name": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size}


def _git_blob(rel: str) -> str | None:
    from src.training.sweep_select import HeadSource
    try:
        return HeadSource(REPO).blob_at("HEAD", rel)
    except OSError:
        return None


def ema_checks(ckpt: Path, iteration: int) -> dict:
    """AM-18 item 2(a), checked on the selected checkpoint: state_dict holds the EMA weights, ema_state_dict the
    raw weights under module.* plus steps (= the iterations averaged), and the two differ."""
    import torch
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = payload.get("state_dict") if isinstance(payload, dict) else None
    ema = payload.get("ema_state_dict") if isinstance(payload, dict) else None
    out = {"state_dict_entries": len(sd) if isinstance(sd, dict) else None,
           "ema_state_dict_present": isinstance(ema, dict), "steps": None, "steps_equal_iteration": False,
           "state_dict_differs_from_raw": None}
    if isinstance(sd, dict) and isinstance(ema, dict):
        steps = ema.get("steps")
        out["steps"] = int(steps) if torch.is_tensor(steps) and steps.numel() == 1 else None
        out["steps_equal_iteration"] = out["steps"] == iteration
        raw = {k[len("module."):]: v for k, v in ema.items() if k.startswith("module.")}
        out["raw_entries"] = len(raw)
        out["state_dict_differs_from_raw"] = sorted(raw) == sorted(sd) and any(
            torch.is_tensor(v) and v.is_floating_point() and not torch.equal(v, raw[k]) for k, v in sd.items())
    return out


def cmd_record(args) -> int:
    from src.eval.teacher_diag import (file_sha256, refuse_test_path, require_outside_repo, utc_now,
                                       write_json_exclusive)
    for value, flag in ((args.work_dir, "--work-dir"), (args.config, "--config"), (args.eval_config, "--eval-config"),
                        (args.out, "--out")):
        refuse_test_path(value, flag)
    work = require_outside_repo(args.work_dir, "--work-dir")
    out = require_outside_repo(args.out, "--out")
    if out.exists():
        raise Refused(f"--out {out} exists: a record is written once")
    if _inside(out, work):
        raise Refused("--out lies inside the work dir; the record is written beside it")
    m = WORK_DIR_RE.fullmatch(work.name)
    if not work.is_dir() or m is None or m.group(1) != args.arm.lower():
        raise Refused(f"--work-dir {work.name!r} is not an existing teacher-{args.arm.lower()}_s42_40k_a<n> directory")
    cfg = Path(args.config).resolve()
    if cfg != (REPO / ARM_CONFIGS[args.arm]).resolve() or file_sha256(cfg) != args.config_sha256:
        raise Refused(f"--config must be {ARM_CONFIGS[args.arm]} with sha256 --config-sha256")
    ev = Path(args.eval_config).resolve()
    if ev != (REPO / RECORD_CONFIG_REL).resolve():
        raise Refused(f"--eval-config must be the record config {RECORD_CONFIG_REL} (plan decision (iv))")
    prov_p, launch_p = work / ARM_PROVENANCE, work / LAUNCH_PROVENANCE
    if not prov_p.is_file():
        raise Refused(f"{prov_p.name} is missing: {work.name} is not a launched arm run")
    prov = json.loads(prov_p.read_text(encoding="utf-8"))
    # The work dir is bound by its name and attempt, not its absolute path: R11 records a copy of the pod's
    # /workspace/arm_runs/<name> on the laptop (review VREC-2).
    if prov.get("arm") != args.arm or Path(str(prov.get("work_dir"))).name != work.name \
            or prov.get("attempt") != int(m.group(2)) or (prov.get("config") or {}).get("sha256") != args.config_sha256:
        raise Refused(f"{prov_p.name} names arm {prov.get('arm')!r}, work dir {prov.get('work_dir')!r} (attempt "
                      f"{prov.get('attempt')!r}) and config {(prov.get('config') or {}).get('sha256')!r}")
    if args.generated_utc is not None and prov.get("launch") is True:
        raise Refused("--generated-utc is for synthetic runs; this work dir's provenance records a launch")
    rd = prov.get("registered_difference") if isinstance(prov.get("registered_difference"), dict) else {}
    diff_b = _reg_bytes(args.arm)
    diff_sha = hashlib.sha256(diff_b).hexdigest()
    if rd.get("sha256") != diff_sha or rd.get("reg_sha256") != diff_sha:
        raise Refused(f"{prov_p.name}: registered difference sha256 {rd.get('sha256')!r} (REG {rd.get('reg_sha256')!r})"
                      f" is not REG[{args.arm}]'s {diff_sha} at this checkout")
    logs = read_logs(work)
    sel_p = work / SELECTION_FILE
    if logs["records"] and not sel_p.is_file():
        raise Refused(f"{SELECTION_FILE} is missing although {RECORDS_FILE} holds {len(logs['records'])} validation(s)")
    sel = json.loads(sel_p.read_text(encoding="utf-8")) if sel_p.is_file() else None
    intervals = [{"iteration": it, **_file(work / f"iter_{it}.pth")} for it in VAL_ITERS
                 if (work / f"iter_{it}.pth").is_file()]
    selection = ema = None
    if sel is not None:
        name = Path(str(sel.get("checkpoint_path"))).name
        p = work / name
        selection = {"teacher_selection_json_sha256": file_sha256(sel_p), "status": sel.get("status"),
                     "selected_iteration": sel.get("selected_iteration"),
                     "selected_mIoU_full": repr(sel.get("selected_mIoU_full")),
                     "checkpoint": _file(p) if p.is_file() else {"name": name, "sha256": None, "bytes": None},
                     "recorded_checkpoint_sha256": sel.get("checkpoint_sha256")}
    complete = (logs["first_nonfinite"] is None and logs["last_iter"] == MAX_ITERS
                and [r.get("iteration") for r in logs["records"]] == VAL_ITERS and len(logs["val"]) == len(VAL_ITERS)
                and selection is not None and selection["status"] == "final"
                and selection["checkpoint"]["name"] == f"iter_{selection['selected_iteration']}.pth"
                and selection["checkpoint"]["sha256"] == selection["recorded_checkpoint_sha256"]
                and [c["iteration"] for c in intervals] == VAL_ITERS)
    if complete:
        ema = ema_checks(work / selection["checkpoint"]["name"], selection["selected_iteration"])
        complete = ema["ema_state_dict_present"] and ema["steps_equal_iteration"] and ema["state_dict_differs_from_raw"]
    status = "diverged" if logs["first_nonfinite"] is not None else ("complete" if complete else "incomplete")
    launch = json.loads(launch_p.read_text(encoding="utf-8")) if launch_p.is_file() else {}
    rows = prov.get("rows") or {}
    doc = {"format": RECORD_FORMAT, "arm": args.arm, "status": status, "first_nonfinite": logs["first_nonfinite"],
           "work_dir": {"path": str(work), "name": work.name, "attempt": int(m.group(2)),
                        "launched_path": str(prov.get("work_dir"))},
           "dl": {"config": (rows.get("config") or {}).get("dl"),
                  "table": (rows.get("r2_table") or {}).get("dl") if args.arm == "R2" else None},
           "train_config": {"path_rel": ARM_CONFIGS[args.arm], "sha256": args.config_sha256,
                            "git_blob": _git_blob(ARM_CONFIGS[args.arm])},
           "eval_config": {"path_rel": RECORD_CONFIG_REL, "sha256": file_sha256(ev), "git_blob": _git_blob(RECORD_CONFIG_REL)},
           "registered_difference": dict(rd, json=json.loads(diff_b.decode("utf-8"))),
           "launch": {"teacher_arm_launch_provenance_sha256": file_sha256(prov_p),
                      "teacher_launch_provenance_sha256": file_sha256(launch_p) if launch_p.is_file() else None,
                      "launcher_sha256s": {k: (v or {}).get("sha256") for k, v in (prov.get("modules") or {}).items()},
                      "checkout_git_head": prov.get("checkout_git_head"), "records_commit": prov.get("records_commit"),
                      "image_digest_declared": launch.get("image_digest_declared"),
                      "r2_table_sha256": (prov.get("r2_table") or {}).get("sha256") if args.arm == "R2" else None},
           "progress": {"last_logged_iteration": logs["last_iter"], "val_rows": len(logs["val"]),
                        "selection_records": len(logs["records"])},
           "logs": {"scalars_rel": logs["scalars"].relative_to(work).as_posix() if logs["scalars"] else None,
                    "scalars_sha256": file_sha256(logs["scalars"]) if logs["scalars"] else None},
           "selection": selection, "interval_checkpoints": intervals, "ema_checks": ema,
           "writer": {"script_rel": SCRIPT_REL, "script_sha256": file_sha256(Path(__file__).resolve()),
                      "commit": _head()},
           "generated_utc": args.generated_utc or utc_now()}
    path = write_json_exclusive(out, doc)
    print(f"  {args.arm}: status {status}; last logged iteration {logs['last_iter']}; VAL rows {len(logs['val'])}"
          + (f"; selected {selection['checkpoint']['name']} {selection['checkpoint']['sha256']}" if selection else ""))
    if logs["first_nonfinite"] is not None:
        print(f"  first non-finite: {logs['first_nonfinite']}")
    print(f"RESULT: RECORD WRITTEN ({status}) -- {path} sha256 {file_sha256(path)}")
    return EXIT_OK if status == "complete" else EXIT_STOP


def _inside(p: Path, d: Path) -> bool:
    return p == d or d in p.parents


def _reg_bytes(arm: str) -> bytes:
    """REG[arm]'s canonical JSON, as the launcher's configuration gate and the configuration row hold it (plan
    decision (iv): registered_difference {json, sha256}; review REC-7)."""
    import scripts.launch_teacher_arm as lta
    return lta.canonical_bytes(lta.reg(arm))


def _head() -> str | None:
    from src.training.sweep_select import HeadSource
    try:
        return HeadSource(REPO).head() or None
    except OSError:
        return None


def build_parser():
    p = argparse.ArgumentParser(description="Arm teacher record (L-TEACHER-ARMS C4).", allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("progress", allow_abbrev=False)
    g.add_argument("--work-dir", required=True)
    r = sub.add_parser("record", allow_abbrev=False)
    r.add_argument("--arm", choices=sorted(ARM_CONFIGS), required=True)
    for name in ("--work-dir", "--config", "--config-sha256", "--eval-config", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--generated-utc", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from src.eval.teacher_diag import Refused as DiagRefused
    try:
        return cmd_progress(args) if args.cmd == "progress" else cmd_record(args)
    except (Refused, DiagRefused, OSError, ValueError) as e:
        print(f"RESULT: REFUSED -- {type(e).__name__}: {e}")
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
