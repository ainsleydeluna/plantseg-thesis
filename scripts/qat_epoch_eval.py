#!/usr/bin/env python3
"""Lane 6 (L-AM4 + L-AM1q): convert and score every epoch of a QAT run (AM-4a item 2).

    convert        --run-dir R --eval-dir E --purpose record|timing --host-label TEXT
                   [--expect-telemetry-sha256 H]   (required for record)   [--epochs N ...]   (timing only)
    convert-epoch  the fresh-process worker `convert` spawns, one epoch per process
    score          --run-dir R --eval-dir E --purpose record|timing --host-label TEXT

`convert` checks the run is complete (AM-19) and its telemetry hash, proves the freezes took (BN
statistics of e11-e15 equal e10's, observer buffers of e13-e15 equal e12's), then converts each epoch in
a fresh process (src/quant/qat_artifacts.py). Conversion is data-free. `score` runs the literal evaluator
command once per converted epoch, on the VAL split at batch size 1 on the CPU, into scores/eNN, and writes
qat_epoch_eval.json. An eval directory is used once: a STOP ends it and a re-run starts a new one.

Every subcommand ends with one RESULT line. Exit codes: 0 done, 1 STOP, 2 refused, 3 incomplete or
aborted, 4 error.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant.qat import EXIT_ABORTED, EXIT_ERROR, EXIT_OK, EXIT_REFUSED, EXIT_STOP, QATRefused, QATStop  # noqa: E402

SCRIPT = Path(__file__).resolve()
EVALUATOR_SCRIPT = "scripts/evaluate_model.py"          # run from the repository root, verbatim
EVAL_FORMAT = "qat_epoch_eval/1"
CONVERT_FORMAT = "qat_convert/1"
CONVERT_RECORD = "qat_convert.json"
CONVERT_STOP = "convert_STOP.json"
EVAL_RECORD = "qat_epoch_eval.json"
TIMING_MAX_SAMPLES = 64
THREADS_NOTE = ("no --threads and no thread environment variable: torch_num_threads is the evaluator's own "
                "record (summary eval_runtime); QNNPACK's thread pool keeps its default")


def _base_parser(sub, name: str, help_: str):
    p = sub.add_parser(name, help=help_)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--purpose", required=True, choices=list(A.PURPOSES))
    p.add_argument("--host-label", required=True)
    p.add_argument("--allow-smoke-inputs", action="store_true",
                   help="smokes only: accept a run in mode 'smoke'; recorded in every output")
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = _base_parser(sub, "convert", "convert every epoch, one fresh process each")
    c.add_argument("--expect-telemetry-sha256", default=None)
    c.add_argument("--epochs", type=int, nargs="+", default=None)
    w = _base_parser(sub, "convert-epoch", "convert one epoch (the worker)")
    w.add_argument("--epoch", type=int, required=True)
    w.add_argument("--expect-telemetry-sha256", default=None)
    s = _base_parser(sub, "score", "score every converted epoch with scripts/evaluate_model.py")
    s.add_argument("--epochs", type=int, nargs="+", default=None)
    return ap


def _fresh_dir_error(d: Path) -> str | None:
    if d.exists() and (not d.is_dir() or any(d.iterdir())):
        return f"{d} exists and is not empty; every conversion starts in a new eval directory"
    return None


def _epochs(args) -> list[int]:
    if args.purpose == "record":
        if args.epochs:
            raise QATRefused("epochs_with_record", "--epochs is for --purpose timing; a record conversion "
                                                   "covers all 15 epochs")
        return list(range(1, Q.EPOCHS + 1))
    eps = sorted(set(args.epochs or range(1, Q.EPOCHS + 1)))
    if any(e < 1 or e > Q.EPOCHS for e in eps):
        raise QATRefused("epochs_range", f"--epochs must lie in 1..{Q.EPOCHS}, got {eps}")
    return eps


def _host(args) -> dict:
    ident = Q.code_identity()
    return {"purpose": args.purpose, "host_label": args.host_label, "cpu_model": Q.cpu_model(),
            "git_head": ident["git_head"], "smoke_inputs": bool(args.allow_smoke_inputs)}


# ------------------------------------------------------------------ convert
def cmd_convert(args) -> int:
    eps = _epochs(args)
    if args.purpose == "record" and not args.expect_telemetry_sha256:
        raise QATRefused("expect_telemetry_sha256_required", "--purpose record requires --expect-telemetry-sha256 "
                                                             "(the run's qat_telemetry.jsonl sha256 recorded after it)")
    A.refuse_test_path(args.eval_dir, "eval_dir")
    E = Path(args.eval_dir)
    err = _fresh_dir_error(E)
    if err:
        raise QATRefused("eval_dir_not_fresh", err)
    rec = A.read_run_record(args.run_dir)
    A.require_complete(rec)
    if args.expect_telemetry_sha256 and rec["telemetry_sha256"] != args.expect_telemetry_sha256:
        raise QATRefused("telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {rec['telemetry_sha256']}, "
                                                      f"expected {args.expect_telemetry_sha256}")
    if rec["run_meta"].get("mode") != "real" and not args.allow_smoke_inputs:
        raise QATRefused("run_not_real", f"the run's mode is {rec['run_meta'].get('mode')!r}; only a real run "
                                         "converts")
    cross = A.freeze_cross_check(rec)
    E.mkdir(parents=True, exist_ok=True)
    head = {"format": CONVERT_FORMAT, "run_dir": rec["run_dir"], "run_id": rec["run_id"],
            "stage": rec["run_meta"]["stage"], "telemetry_sha256": rec["telemetry_sha256"], **_host(args),
            "freeze_cross_check": cross, "epochs_requested": eps}
    if not cross["ok"]:
        from src.quant.ptq import write_exclusive
        write_exclusive({E / CONVERT_STOP: A.json_bytes({**head, "status": "STOP", "code": "freeze_cross_check"})})
        raise QATStop("freeze_cross_check", f"the freezes did not take: {cross['first_differences']}; see "
                                            f"{E / CONVERT_STOP}")
    outcomes = []
    for e in eps:
        argv = [sys.executable, "-B", str(SCRIPT), "convert-epoch", "--run-dir", str(args.run_dir), "--eval-dir",
                str(args.eval_dir), "--epoch", str(e), "--purpose", args.purpose, "--host-label", args.host_label]
        if args.expect_telemetry_sha256:
            argv += ["--expect-telemetry-sha256", args.expect_telemetry_sha256]
        if args.allow_smoke_inputs:
            argv.append("--allow-smoke-inputs")
        t0 = time.time()
        p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
        tail = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT:")]
        names = A.epoch_names(e)
        status = ("converted" if (E / A.CONVERTED_DIR / names["run_meta"]).is_file()
                  else "not convertible" if (E / A.CONVERTED_DIR / names["not_convertible"]).is_file()
                  else "STOP" if (E / A.CONVERTED_DIR / names["stop"]).is_file() else "failed")
        outcomes.append({"epoch": e, "status": status, "exit_code": p.returncode, "seconds": time.time() - t0,
                         "result": tail[-1] if tail else None})
        print(f"[convert] e{e:02d}: {status} (exit {p.returncode}, {time.time() - t0:.1f}s)")
        if p.returncode != EXIT_OK:
            sys.stderr.write(p.stderr[-2000:])
            print(f"RESULT: {'STOP' if p.returncode == EXIT_STOP else 'INCOMPLETE'} -- e{e:02d} worker exited "
                  f"{p.returncode}: {tail[-1] if tail else p.stderr.strip()[-300:]}. The eval directory {E} is "
                  "spent; a re-run uses a new one.")
            return p.returncode if p.returncode in (EXIT_STOP, EXIT_REFUSED, EXIT_ABORTED) else EXIT_ERROR
    from src.quant.ptq import write_exclusive
    write_exclusive({E / CONVERT_RECORD: A.json_bytes({**head, "status": "converted", "outcomes": outcomes,
                                                       "created_wall_clock": time.time()})})
    n_conv = sum(o["status"] == "converted" for o in outcomes)
    excluded = [o["epoch"] for o in outcomes if o["status"] == "not convertible"]
    print(f"RESULT: QAT CONVERT DONE ({rec['run_meta']['stage']}, {rec['run_id']}: {n_conv} converted, "
          f"{len(excluded)} not convertible {excluded}, purpose {args.purpose})")
    return EXIT_OK


def cmd_convert_epoch(args) -> int:
    if args.expect_telemetry_sha256:
        rec = A.read_run_record(args.run_dir)
        if rec["telemetry_sha256"] != args.expect_telemetry_sha256:
            raise QATRefused("telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {rec['telemetry_sha256']}")
    out = A.convert_epoch(args.run_dir, args.eval_dir, args.epoch, purpose=args.purpose,
                          host_label=args.host_label, allow_smoke_inputs=args.allow_smoke_inputs)
    print(f"RESULT: QAT EPOCH {args.epoch:02d} {out['status'].upper()}")
    return EXIT_OK


# ------------------------------------------------------------------ score
def evaluator_command(stage_name: str, provenance: Path, out_dir: Path, run_id: str, purpose: str) -> list[str]:
    status = "provisional" if purpose == "record" else "smoke"
    argv = [sys.executable, "-B", EVALUATOR_SCRIPT, "--stage", stage_name, "--model-role", "student",
            "--precision", "int8_qat", "--split", "val", "--protocol", "canvas", "--provenance", str(provenance),
            "--artifact-status", status, "--device", "cpu", "--batch-size", "1", "--out-dir", str(out_dir),
            "--run-id", run_id]
    if purpose == "timing":
        argv += ["--max-samples", str(TIMING_MAX_SAMPLES)]
    return argv


def cmd_score(args) -> int:
    A.refuse_test_path(args.eval_dir, "eval_dir")
    E = Path(args.eval_dir)
    rec = A.read_run_record(args.run_dir)
    A.require_complete(rec)
    conv_rec = E / CONVERT_RECORD
    if (E / CONVERT_STOP).exists() or list((E / A.CONVERTED_DIR).glob("e*_STOP.json")):
        raise QATRefused("stop_present", f"{E} holds a STOP file; it is spent and is never scored")
    if not conv_rec.is_file():
        raise A.QATIncomplete("convert_record_missing", f"{conv_rec} is missing: run `convert` first")
    conv = json.loads(conv_rec.read_text(encoding="utf-8"))
    if conv.get("telemetry_sha256") != rec["telemetry_sha256"] or conv.get("run_id") != rec["run_id"]:
        raise QATRefused("convert_record_mismatch", f"{conv_rec} names telemetry {conv.get('telemetry_sha256')} of "
                                                    f"run {conv.get('run_id')!r}; the run now has {rec['telemetry_sha256']}")
    if conv.get("purpose") != args.purpose or bool(conv.get("smoke_inputs")) != bool(args.allow_smoke_inputs):
        raise QATRefused("purpose_mismatch", f"{conv_rec} was written for purpose {conv.get('purpose')!r} "
                                             f"(smoke inputs {conv.get('smoke_inputs')})")
    scores = E / A.SCORES_DIR
    if scores.exists() and any(scores.iterdir()):
        raise QATRefused("scores_not_fresh", f"{scores} is not empty; an eval directory is scored once")
    if (E / EVAL_RECORD).exists():
        raise QATRefused("output_exists", f"{E / EVAL_RECORD} exists")
    stage = rec["run_meta"]["stage"]
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    rows = []
    for o in conv["outcomes"]:
        e = o["epoch"]
        names = A.epoch_names(e)
        ck_sha = rec["ends"][e]["checkpoint_sha256"]
        row = {"epoch": e, "checkpoint_sha256": ck_sha}
        if o["status"] == "not convertible":
            rows.append({**row, "status": "excluded", "rule": A.NOT_CONVERTIBLE_RULE,
                         "not_convertible": f"{A.CONVERTED_DIR}/{names['not_convertible']}"})
            print(f"[score] e{e:02d}: {A.NOT_CONVERTIBLE_RULE}")
            continue
        prov = E / A.CONVERTED_DIR / names["run_meta"]
        out = scores / f"e{e:02d}"
        run_id = f"{rec['run_id']}_e{e:02d}_{ck_sha[:12]}"
        argv = evaluator_command(stage, prov, out, run_id, args.purpose)
        scores.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
        seconds = time.time() - t0
        summ = out / "summary.json"
        if p.returncode != 0 or not summ.is_file():
            sys.stderr.write(p.stdout[-1500:] + p.stderr[-2500:])
            print(f"RESULT: INCOMPLETE -- the evaluator exited {p.returncode} on e{e:02d}; {E} is spent")
            return EXIT_ABORTED
        s = json.loads(summ.read_text(encoding="utf-8"))
        rt = (s.get("run") or {}).get("eval_runtime") or {}
        rows.append({**row, "status": "scored", "run_id": run_id,
                     "provenance": {"path": f"{A.CONVERTED_DIR}/{names['run_meta']}", "sha256": Q.sha256_file(prov)},
                     "converted_artifact_sha256": json.loads(prov.read_text(encoding="utf-8"))["converted_artifact_sha256"],
                     "score_dir": f"{A.SCORES_DIR}/e{e:02d}",
                     "summary": {"path": f"{A.SCORES_DIR}/e{e:02d}/summary.json", "sha256": Q.sha256_file(summ)},
                     "all_class_miou": (s.get("dataset_level") or {}).get("all_class_miou"),
                     "evaluator_command": argv[1:], "evaluator_python": argv[0], "seconds": seconds,
                     "torch_num_threads": rt.get("torch_num_threads"), "affinity_cpus": affinity})
        print(f"[score] e{e:02d}: all-class mIoU {rows[-1]['all_class_miou']} ({seconds:.1f}s)")
    doc = {"format": EVAL_FORMAT, "run_dir": rec["run_dir"], "run_id": rec["run_id"], "stage": stage,
           "telemetry_sha256": rec["telemetry_sha256"], **_host(args), "threads_note": THREADS_NOTE,
           "affinity_cpus": affinity, "epochs": rows, "created_wall_clock": time.time()}
    from src.quant.ptq import write_exclusive
    write_exclusive({E / EVAL_RECORD: A.json_bytes(doc)})
    n = sum(r["status"] == "scored" for r in rows)
    print(f"RESULT: QAT SCORE DONE ({stage}, {rec['run_id']}: {n} scored, "
          f"{sum(r['status'] == 'excluded' for r in rows)} excluded, purpose {args.purpose})")
    return EXIT_OK


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"convert": cmd_convert, "convert-epoch": cmd_convert_epoch, "score": cmd_score}
    try:
        return handlers[args.cmd](args)
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}")
        return EXIT_REFUSED
    except A.QATIncomplete as e:
        print(f"RESULT: INCOMPLETE [{e.code}] -- {e}")
        return EXIT_ABORTED
    except QATStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return EXIT_STOP
    except Exception as e:                                   # noqa: BLE001 -- reported, exit 4
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
