#!/usr/bin/env python3
"""E7-X: the static INT8 PTQ of an AM-18 item 3 arm's E3 run (lane L-TEACHER-ARMS, commit C2; ARMS plan decision
(v); AM-18 item 3(a): "E7-X = PTQ of E3-X with the AM-10 calibration list of record").

    python -B scripts/run_ptq_arm.py --arm NAME --real-run --confirm-real-run --e3-run-dir DIR \\
        --source-ckpt CKPT --expect-source-sha256 SHA256 \\
        --calibration-list configs/calibration/ptq_calibration_seed42.json --expect-list-checksum SHA256 \\
        --data-root ROOT --out-dir OUT [--threads N]

Before src.quant.ptq.run_ptq(stage="E7", ...) runs, unchanged, every one of these holds (else exit 2, nothing
written):
  * --real-run and --confirm-real-run; every path argument passes the SL-1 guard (as given, then resolved);
  * NAME is a registered arm (configs/kd_arms/; any registry entry, ruling R10);
  * --out-dir is absolute, outside the repository, named <e7 stage>_s<seed>_a<n> in lower case (e7-r1_s42_a1),
    absent or empty, and holds no e7_arm_record.json;
  * --e3-run-dir is the arm's E3 run: absolute, outside the repository, named <arm>_s<seed>_..., and its
    <stage>_run_meta.jsonl holds exactly one row: event run_meta, mode real, stage E3, the arm's seed, max_iters
    and poly_horizon = the arm's horizon, arm = the arm's run stage, descriptive true, parent_of_e4_e7 false and
    target_construction = the arm's {K, views} (the trainer's arm fields);
  * that run's best.json names --source-ckpt, which is a file in the run directory; its sha256 is
    --expect-source-sha256, and its payload carries stage E3, arm = the arm's run stage and the arm's
    target_construction (the trainer's arm payload keys);
  * the calibration list is the AM-10 list of record as committed (src.quant.ptq.load_calibration_list with the
    expected checksum, then list_of_record_error).
After run_ptq writes its artifacts, e7_arm_record.json (format e7_arm_record/1) is created exclusively in
--out-dir. It binds the arm, the E3-X run (run_id, run_meta and best.json sha256), the source checkpoint, the
calibration list and the sha256 of every PTQ artifact and of the PTQ run provenance.
Exit: 0 written; 1 STOP (run_ptq's STOP: no run provenance, nothing evaluable); 2 refused, nothing written.
Nothing here trains, downloads, uses a GPU or reads VAL or TEST.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2
ARM_DIR = None                     # the registry folder (None: configs/kd_arms/); a smoke points it at a scratch copy
RECORD_FILE = "e7_arm_record.json"
RECORD_FORMAT = "e7_arm_record/1"
PTQ_STAGE = "E7"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ArmPTQRefused(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E7-X: PTQ of an AM-18 item 3 arm's E3 run (L-TEACHER-ARMS C2).")
    p.add_argument("--arm", default=None)
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--e3-run-dir", default=None)
    p.add_argument("--source-ckpt", default=None)
    p.add_argument("--expect-source-sha256", default=None)
    p.add_argument("--calibration-list", default=None)
    p.add_argument("--expect-list-checksum", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--threads", type=int, default=None)
    return p


def _sha256(path) -> str:
    from src.eval.teacher_diag import file_sha256
    return file_sha256(path)


def _head() -> str | None:
    """HEAD of this checkout, read-only (sweep_select.HeadSource: safe.directory, no optional locks); None off git."""
    from src.training.sweep_select import HeadSource
    try:
        out = HeadSource(REPO).head()
    except (OSError, subprocess.SubprocessError):
        return None
    return out if re.fullmatch(r"[0-9a-f]{40}", out or "") else None


def arm_spec(name: str):
    from src.training import kd_arms
    try:
        arms = kd_arms.load_arms(ARM_DIR)
    except kd_arms.ArmSpecError as e:
        raise ArmPTQRefused("arm_spec", str(e)) from e
    if name not in arms:
        raise ArmPTQRefused("arm_spec", f"--arm {name!r} is not a registered arm; registered: {sorted(arms) or 'none'}")
    return arms[name]


def check_out_dir(spec, value: str) -> Path:
    from src.eval.teacher_diag import Refused, require_outside_repo
    p = Path(value)
    if not p.is_absolute():
        raise ArmPTQRefused("out_dir_relative", f"--out-dir must be absolute: {value!r}")
    try:
        out = require_outside_repo(p, "--out-dir")
    except Refused as e:
        raise ArmPTQRefused("out_dir_inside_repo", str(e)) from e
    stem = f"{spec.e7_stage.lower()}_s{spec.seed}_a"
    if not (out.name.startswith(stem) and out.name[len(stem):].isdigit() and int(out.name[len(stem):]) >= 1):
        raise ArmPTQRefused("out_dir_name", f"--out-dir {out.name!r}: E7-X's run directory is named {stem}<n>")
    if (out / RECORD_FILE).exists():
        raise ArmPTQRefused("output_exists", f"{out / RECORD_FILE} already exists; an E7-X run is written once")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ArmPTQRefused("output_exists", f"--out-dir {out} exists and is not an empty directory")
    return out


def read_arm_run(spec, run_dir: str) -> dict:
    """The E3-X run directory: its one run_meta row with the arm's fields, and its best.json."""
    from src.eval.teacher_diag import Refused, require_outside_repo
    d = Path(run_dir)
    if not d.is_absolute():
        raise ArmPTQRefused("e3_run_dir_relative", f"--e3-run-dir must be absolute: {run_dir!r}")
    try:
        d = require_outside_repo(d, "--e3-run-dir")
    except Refused as e:
        raise ArmPTQRefused("e3_run_dir_inside_repo", str(e)) from e
    if not d.name.startswith(f"{spec.name}_s{spec.seed}_"):
        raise ArmPTQRefused("arm_run_meta_mismatch", f"--e3-run-dir {d.name!r} is not a run of {spec.run_stage} "
                                                     f"({spec.name}_s{spec.seed}_...)")
    meta_p = d / f"{spec.stage}_run_meta.jsonl"
    rows = []
    try:
        for ln in meta_p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    except (OSError, ValueError) as e:
        raise ArmPTQRefused("arm_run_meta_mismatch", f"{meta_p}: unreadable ({type(e).__name__}: {e})") from e
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ArmPTQRefused("arm_run_meta_mismatch", f"{meta_p}: {len(rows)} rows; E3-X's run_meta has exactly one")
    row = rows[0]
    want = {"event": "run_meta", "mode": "real", "stage": "E3", "seed": spec.seed, "max_iters": spec.horizon,
            "poly_horizon": spec.horizon, "arm": spec.run_stage, "descriptive": True, "parent_of_e4_e7": False,
            "target_construction": spec.target_construction}
    wrong = {k: (row.get(k, "<absent>"), v) for k, v in want.items() if row.get(k, "<absent>") != v
             or type(row.get(k)) is not type(v)}
    if wrong:
        raise ArmPTQRefused("arm_run_meta_mismatch", f"{meta_p}: not an {spec.run_stage} run of record: {wrong} "
                                                     "(got, want)")
    best_p = d / "best.json"
    try:
        best = json.loads(best_p.read_text(encoding="utf-8"))
        best_name = Path(str(best["best_ckpt"])).name
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ArmPTQRefused("best_json_invalid", f"{best_p}: unreadable ({type(e).__name__}: {e})") from e
    return {"dir": d, "run_meta": meta_p, "row": row, "best_json": best_p, "best_ckpt": best_name}


def check_source(spec, run: dict, source: str, expect_sha256: str) -> dict:
    """best.json names the source, a file in the run directory, with the expected sha256 and the arm's payload."""
    if not (isinstance(expect_sha256, str) and SHA256_RE.fullmatch(expect_sha256)):
        raise ArmPTQRefused("expected_hash_invalid", f"--expect-source-sha256 {expect_sha256!r}")
    p = Path(source)
    if not p.is_absolute() or not p.is_file():
        raise ArmPTQRefused("source_missing", f"--source-ckpt {source!r} is not an existing absolute file")
    if p.name != run["best_ckpt"] or p.resolve().parent != run["dir"]:
        raise ArmPTQRefused("source_not_best", f"--source-ckpt {p} is not the checkpoint {run['best_ckpt']} that "
                                               f"{run['best_json']} names in {run['dir']}")
    sha = _sha256(p)
    if sha != expect_sha256:
        raise ArmPTQRefused("source_sha256_mismatch", f"{p}: sha256 {sha}, expected {expect_sha256}")
    import torch
    try:
        payload = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        raise ArmPTQRefused("source_invalid", f"{p}: unreadable ({type(e).__name__}: {e})") from e
    if not isinstance(payload, dict) or str(payload.get("stage")).upper() != "E3" \
            or payload.get("arm") != spec.run_stage or payload.get("target_construction") != spec.target_construction:
        got = {k: payload.get(k) for k in ("stage", "arm", "target_construction")} if isinstance(payload, dict) else None
        raise ArmPTQRefused("source_not_arm", f"{p}: payload {got}; E7-X takes the checkpoint of {spec.run_stage} "
                                              f"(stage E3, arm {spec.run_stage}, {spec.target_construction})")
    return {"path": str(p), "sha256": sha, "bytes": p.stat().st_size, "payload_arm": payload["arm"],
            "target_construction": payload["target_construction"]}


def list_of_record_problem(calibration_list: str, checksum: str) -> str | None:
    """None iff the list is the AM-10 list of record as committed (AM-18 item 3(a))."""
    from src.quant.ptq import PTQRefused, list_of_record_error, load_calibration_list
    try:
        doc, _ = load_calibration_list(calibration_list, expected_checksum=checksum)
    except PTQRefused as e:
        return f"[{e.code}] {e}"
    return list_of_record_error({"role": doc["role"], "seed": doc["seed"],
                                 "list_artifact_status": doc.get("artifact_status"),
                                 "checksum_sha256": doc["checksum_sha256"]}, REPO)


def ptq_run(**kw) -> dict:
    """src.quant.ptq.run_ptq, unchanged (stage E7)."""
    from src.quant.ptq import run_ptq
    return run_ptq(**kw)


def e7_record(spec, run: dict, source: dict, prov: dict, out: Path) -> dict:
    from src.eval.teacher_diag import utc_now
    from src.quant.ptq import artifact_names
    names = artifact_names(PTQ_STAGE.lower())
    cal = prov.get("calibration") or {}
    arts = {"torchscript": {"path": prov["converted_artifact"], "sha256": prov["converted_artifact_sha256"]},
            "state_dict": {k: prov["state_dict_artifact"][k] for k in ("path", "sha256")},
            "x86_copy": {k: prov["x86_latency_copy"][k] for k in ("path", "sha256")},
            "run_meta": {"path": names["run_meta"], "sha256": _sha256(out / names["run_meta"])}}
    return {"format": RECORD_FORMAT, "arm": spec.name, "run_stage": spec.run_stage, "e7_stage": spec.e7_stage,
            "registry": {"file": Path(spec.path).name, "sha256": spec.sha256},
            "e3_run": {"run_id": run["dir"].name, "dir": str(run["dir"]), "run_meta_sha256": _sha256(run["run_meta"]),
                       "best_json_sha256": _sha256(run["best_json"]), "git_head": run["row"].get("git_head"),
                       "records_commit": run["row"].get("records_commit")},
            "source": source,
            "calibration": {k: cal.get(k) for k in ("list_path", "list_file_sha256", "checksum_sha256", "seed", "role",
                                                    "list_artifact_status")} | {"of_record": True},
            "ptq_run_role": prov.get("ptq_run_role"), "artifacts": arts,
            "writer": {"script_rel": "scripts/run_ptq_arm.py", "script_sha256": _sha256(Path(__file__).resolve()),
                       "commit": _head()},
            "generated_utc": utc_now()}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.real_run or not args.confirm_real_run:
        print("REFUSING: E7-X requires --real-run --confirm-real-run. Nothing was read or written.", file=sys.stderr)
        return EXIT_REFUSED
    missing = [f for f, v in (("--arm", args.arm), ("--e3-run-dir", args.e3_run_dir), ("--source-ckpt", args.source_ckpt),
                              ("--expect-source-sha256", args.expect_source_sha256),
                              ("--calibration-list", args.calibration_list),
                              ("--expect-list-checksum", args.expect_list_checksum), ("--data-root", args.data_root),
                              ("--out-dir", args.out_dir)) if not v]
    if missing:
        print(f"REFUSING: missing {missing}. Nothing was read or written.", file=sys.stderr)
        return EXIT_REFUSED
    from src.eval.teacher_diag import Refused, refuse_test_path, write_json_exclusive
    try:
        for value, what in ((args.e3_run_dir, "--e3-run-dir"), (args.source_ckpt, "--source-ckpt"),
                            (args.calibration_list, "--calibration-list"), (args.data_root, "--data-root"),
                            (args.out_dir, "--out-dir")):
            refuse_test_path(value, what)
    except Refused as e:
        print(f"RESULT: REFUSED [test_path] -- {e}")
        return EXIT_REFUSED
    from src.quant.ptq import PTQRefused, PTQStop
    try:
        spec = arm_spec(args.arm)
        out = check_out_dir(spec, args.out_dir)
        run = read_arm_run(spec, args.e3_run_dir)
        source = check_source(spec, run, args.source_ckpt, args.expect_source_sha256)
        problem = list_of_record_problem(args.calibration_list, args.expect_list_checksum)
        if problem is not None:
            raise ArmPTQRefused("calibration_not_of_record", f"{args.calibration_list}: {problem} (AM-18 item 3(a): "
                                                             "E7-X uses the AM-10 calibration list of record)")
    except ArmPTQRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}")
        return EXIT_REFUSED
    print(f"[e7-arm] {spec.e7_stage}: PTQ of {run['dir'].name} ({spec.run_stage}) {source['sha256'][:12]} -> {out}")
    try:
        prov = ptq_run(stage=PTQ_STAGE, source_ckpt=args.source_ckpt, expect_source_sha256=args.expect_source_sha256,
                       calibration_list=args.calibration_list, expect_list_checksum=args.expect_list_checksum,
                       data_root=args.data_root, out_dir=str(out), threads=args.threads)
    except PTQRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}")
        return EXIT_REFUSED
    except PTQStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return EXIT_STOP
    rec = e7_record(spec, run, source, prov, out)
    path = write_json_exclusive(out / RECORD_FILE, rec)
    print(f"  arm record          {path}")
    print(f"  artifact of record  {rec['artifacts']['torchscript']['path']}  sha256 "
          f"{rec['artifacts']['torchscript']['sha256']}")
    print(f"RESULT: E7 ARM WRITTEN ({spec.e7_stage} from {run['dir'].name})")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
