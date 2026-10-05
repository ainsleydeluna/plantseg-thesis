#!/usr/bin/env python3
"""Lane 6 (L-AM4 + L-AM1q): convert, score and finalize the epochs of a QAT run (AM-4a items 2-3).

    convert         --run-dir R --eval-dir E --purpose record|timing --host-label TEXT
                    [--expect-telemetry-sha256 H]   (required for record)   [--epochs N ...]   (timing only)
    convert-epoch   the fresh-process worker `convert` spawns, one epoch per process
    score           --run-dir R --eval-dir E --purpose record|timing --host-label TEXT   [--epochs N ...]   (timing)
    finalize        --run-dir R --eval-dir E   [--clip-selection F [--clip-selection-sha256 H]]   (E5 seed 42)
    check-run-meta  --run-dir R --stage E5|E6 --seed S --grad-clip-norm C --expect-head P --expect-source-sha256 H
                    (--u4-pilot | --clip-selection F --clip-selection-sha256 H)   [E6: the λ and α selections]

`convert` checks the run is complete (AM-19) and its telemetry hash, proves the freezes took (BN
statistics of e11-e15 equal e10's, observer buffers of e13-e15 equal e12's), then converts each epoch in
a fresh process (src/quant/qat_artifacts.py). Conversion is data-free. `score` runs the literal evaluator
command once per converted epoch, on the VAL split at batch size 1 on the CPU, into scores/eNN, and writes
qat_epoch_eval.json. An eval directory is used once: a STOP ends it and a re-run starts a new one.
`finalize` runs after scripts/select_qat_epoch.py (and, for the E5 seed-42 pilot, scripts/select_clip.py):
it names the selected epoch's scored artifact as the artifact of record, never tracing it again, and
translates the sha-checked eNN.pt into the x86 latency copy with zero optimizer steps (P28).
`check-run-meta` compares a launched run's run_meta row with the launch of record (P14, P10, O2).

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
FINALIZE_FORMAT = "qat_finalize/1"
CONVERT_RECORD = "qat_convert.json"
CONVERT_STOP = "convert_STOP.json"
EVAL_RECORD = "qat_epoch_eval.json"
SELECTION_RECORD = "qat_selection.json"
FINALIZE_RECORD = "qat_finalize.json"
FINALIZE_STOP = "finalize_STOP.json"
FINAL_DIR = "final"
TIMING_MAX_SAMPLES = 64
THREADS_NOTE = ("no --threads and no thread environment variable: torch_num_threads is the evaluator's own "
                "record (summary eval_runtime); QNNPACK's thread pool keeps its default")
# the QAT launch of record (P14): the observer census of the prepared student and the loader workers
EXPECTED_CENSUS = {"FixedQParamsObserver": 9, "MovingAverageMinMaxObserver": 110,
                   "MovingAveragePerChannelMinMaxObserver": 65}
NUM_WORKERS_OF_RECORD = 12


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
    f = sub.add_parser("finalize", help="the artifact of record and its x86 latency copy (after the selections)")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--eval-dir", required=True)
    f.add_argument("--clip-selection", default=None)
    f.add_argument("--clip-selection-sha256", default=None)
    f.add_argument("--allow-smoke-inputs", action="store_true",
                   help="smokes only: accept a run in mode 'smoke'; recorded in the output")
    m = sub.add_parser("check-run-meta", help="compare a launched run's run_meta row with the launch of record")
    m.add_argument("--run-dir", required=True)
    m.add_argument("--stage", required=True, choices=["E5", "E6", "e5", "e6"])
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--grad-clip-norm", type=float, required=True)
    m.add_argument("--expect-head", required=True)
    m.add_argument("--expect-source-sha256", required=True)
    m.add_argument("--u4-pilot", action="store_true")
    m.add_argument("--clip-selection", default=None)
    m.add_argument("--clip-selection-sha256", default=None)
    m.add_argument("--lambda-selection", default=None)
    m.add_argument("--lambda-selection-sha256", default=None)
    m.add_argument("--alpha-selection", default=None)
    m.add_argument("--alpha-selection-sha256", default=None)
    return ap


def _fresh_dir_error(d: Path) -> str | None:
    if d.exists() and (not d.is_dir() or any(d.iterdir())):
        return f"{d} exists and is not empty; every conversion starts in a new eval directory"
    return None


def _epochs(args) -> list[int] | None:
    """Record: None (every epoch). Timing: the --epochs given, or None."""
    if args.purpose == "record":
        if args.epochs:
            raise QATRefused("epochs_with_record", "--epochs is for --purpose timing; a record pass covers all "
                                                   f"{Q.EPOCHS} epochs")
        return None
    if not args.epochs:
        return None
    eps = sorted(set(args.epochs))
    if any(e < 1 or e > Q.EPOCHS for e in eps):
        raise QATRefused("epochs_range", f"--epochs must lie in 1..{Q.EPOCHS}, got {eps}")
    return eps


def _host(args) -> dict:
    ident = Q.code_identity()
    return {"purpose": args.purpose, "host_label": args.host_label, "cpu_model": Q.cpu_model(),
            "git_head": ident["git_head"], "smoke_inputs": bool(args.allow_smoke_inputs)}


def _epoch_list(eps) -> str:
    return " ".join(f"e{e:02d}" for e in eps)


# ------------------------------------------------------------------ convert
def cmd_convert(args) -> int:
    eps = _epochs(args) or list(range(1, Q.EPOCHS + 1))
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
    tail = f"; not convertible: {_epoch_list(excluded)}" if excluded else ""
    print(f"RESULT: CONVERTED {n_conv}/{len(outcomes)} ({args.purpose}{tail})")
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
    asked = _epochs(args)
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
                                                    f"run {conv.get('run_id')!r}; the run now has "
                                                    f"{rec['telemetry_sha256']}")
    if conv.get("purpose") != args.purpose or bool(conv.get("smoke_inputs")) != bool(args.allow_smoke_inputs):
        raise QATRefused("purpose_mismatch", f"{conv_rec} was written for purpose {conv.get('purpose')!r} "
                                             f"(smoke inputs {conv.get('smoke_inputs')})")
    outcomes = {o["epoch"]: o for o in conv["outcomes"]}
    if args.purpose == "record" and sorted(outcomes) != list(range(1, Q.EPOCHS + 1)):
        raise QATRefused("convert_record_epochs", f"{conv_rec} covers epochs {sorted(outcomes)}; a record pass "
                                                  f"scores all {Q.EPOCHS}")
    if asked is not None:
        missing = [e for e in asked if (outcomes.get(e) or {}).get("status") != "converted"]
        if missing:
            raise QATRefused("epoch_not_converted", f"--epochs {asked}: {_epoch_list(missing)} not converted in {E}")
    todo = asked if asked is not None else sorted(outcomes)
    scores = E / A.SCORES_DIR
    if scores.exists() and any(scores.iterdir()):
        raise QATRefused("scores_not_fresh", f"{scores} is not empty; an eval directory is scored once")
    if (E / EVAL_RECORD).exists():
        raise QATRefused("output_exists", f"{E / EVAL_RECORD} exists")
    stage = rec["run_meta"]["stage"]
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    rows = []
    for e in todo:
        o = outcomes[e]
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
        prov_doc = json.loads(prov.read_text(encoding="utf-8"))
        rows.append({**row, "status": "scored", "run_id": run_id,
                     "provenance": {"path": f"{A.CONVERTED_DIR}/{names['run_meta']}", "sha256": Q.sha256_file(prov)},
                     "converted_artifact_sha256": prov_doc["converted_artifact_sha256"],
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
    excluded = [r["epoch"] for r in rows if r["status"] == "excluded"]
    what = f"{args.purpose}, {TIMING_MAX_SAMPLES} samples" if args.purpose == "timing" else args.purpose
    tail = f"; excluded: {_epoch_list(excluded)}" if excluded else ""
    print(f"RESULT: SCORED {n}/{len(rows)} ({what}{tail})")
    return EXIT_OK


# ------------------------------------------------------------------ finalize (P28)
def _read_json(path: Path, code: str) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QATRefused(code, f"{path}: {e}") from e
    if not isinstance(doc, dict):
        raise QATRefused(code, f"{path} is not a JSON object")
    return doc


def _pilot_clip_winner(args, rec: dict, sel_sha: str) -> dict:
    """E5 seed 42: clip_selection.json must name this run, and this qat_selection.json, as the U4 winner."""
    A.refuse_test_path(args.clip_selection, "clip_selection")
    p = Path(args.clip_selection)
    if not p.is_file():
        raise A.QATIncomplete("clip_selection_missing", f"{p} is missing: run scripts/select_clip.py first")
    sha = Q.sha256_file(p)
    if args.clip_selection_sha256 is not None and sha != args.clip_selection_sha256:
        raise QATRefused("clip_selection_sha256_mismatch", f"{p} has sha256 {sha}, expected "
                                                           f"{args.clip_selection_sha256}")
    doc = _read_json(p, "clip_selection_format")
    w = doc.get("winner") or {}
    if doc.get("format") != Q.CLIP_SELECTION_FORMAT or w.get("clip_norm") not in Q.CLIP_CANDIDATES:
        raise QATRefused("clip_selection_format", f"{p} is not a {Q.CLIP_SELECTION_FORMAT} file with a winner "
                                                  f"among {Q.CLIP_CANDIDATES}")
    if w.get("run_id") != rec["run_id"] or w.get("telemetry_sha256") != rec["telemetry_sha256"]:
        raise QATRefused("not_the_clip_winner", f"the U4 winner is run {w.get('run_id')!r} (clip "
                                                f"{w.get('clip_norm')}); {rec['run_id']!r} is retained and reported, "
                                                "never served")
    if (w.get("qat_selection") or {}).get("sha256") != sel_sha:
        raise QATRefused("clip_selection_stale", f"{p} names qat_selection.json sha256 "
                                                 f"{(w.get('qat_selection') or {}).get('sha256')}; the eval directory "
                                                 f"holds {sel_sha}")
    return {"path": str(p.resolve()), "sha256": sha, "winner_clip_norm": w["clip_norm"],
            "winner_run_id": w["run_id"], "tie": doc.get("tie"),
            "sha256_pinned": args.clip_selection_sha256 is not None}


def cmd_finalize(args) -> int:
    from src.eval.model_loading import load_int8_torchscript
    from src.models.student import build_student
    from src.quant import qat_select as S
    from src.quant.prepare import convert_model
    from src.quant.ptq import (graph_census, output_parity, qconfig_summary, student_traced_in_process,
                               synthetic_parity_inputs, torchscript_bytes, weight_scheme_report, write_exclusive)
    from src.quant.qconfig import X86_BACKENDS, select_x86_backend, x86_qat_qconfig
    from src.quant.x86_latency import (X86_LATENCY_ARTIFACT_ROLE, load_qat_sidecar, translate_qat_state_to_x86,
                                       x86_artifact_provenance)
    if student_traced_in_process():
        raise QATRefused("fresh_process_required", "finalize traces the x86 copy; run it in a fresh process")
    A.refuse_test_path(args.eval_dir, "eval_dir")
    E = Path(args.eval_dir)
    rec = A.read_run_record(args.run_dir)
    A.require_complete(rec)
    meta = rec["run_meta"]
    if meta.get("mode") != "real" and not args.allow_smoke_inputs:
        raise QATRefused("run_not_real", f"the run's mode is {meta.get('mode')!r}")
    if (E / FINALIZE_RECORD).exists() or (E / FINAL_DIR).exists() or (E / FINALIZE_STOP).exists():
        raise QATRefused("output_exists", f"{E} is already finalized (or its finalize stopped)")
    sel_p = E / SELECTION_RECORD
    if not sel_p.is_file():
        raise A.QATIncomplete("epoch_selection_missing", f"{sel_p} is missing: run scripts/select_qat_epoch.py first")
    sel = _read_json(sel_p, "epoch_selection_format")
    _rules, rules_sha = S.load_rules()
    if sel.get("format") != S.EPOCH_SELECTION_FORMAT or sel.get("rules_sha256") != rules_sha:
        raise QATRefused("epoch_selection_format", f"{sel_p} is not a {S.EPOCH_SELECTION_FORMAT} file of the "
                                                   "committed rules")
    if sel.get("telemetry_sha256") != rec["telemetry_sha256"] or sel.get("run_id") != rec["run_id"]:
        raise QATRefused("epoch_selection_mismatch", f"{sel_p} names run {sel.get('run_id')!r}, telemetry "
                                                     f"{sel.get('telemetry_sha256')}")
    if bool(sel.get("smoke_inputs")) and not args.allow_smoke_inputs:
        raise QATRefused("smoke_inputs", f"{sel_p} was selected from smoke inputs")
    sel_sha = Q.sha256_file(sel_p)
    # the pilot serves the U4 winner only (AM-4a item 3)
    is_pilot = meta.get("stage") == Q.U4_STAGE and meta.get("seed") == Q.U4_SEED
    clip_doc = None
    if is_pilot:
        if not args.clip_selection:
            raise QATRefused("clip_selection_required", "an E5 seed-42 run is a U4 pilot run: finalize takes "
                                                        "--clip-selection and serves the winner only")
        clip_doc = _pilot_clip_winner(args, rec, sel_sha)
    elif args.clip_selection or args.clip_selection_sha256:
        raise QATRefused("clip_selection_not_applicable", "only the E5 seed-42 pilot runs take --clip-selection here; "
                                                          "every other run was bound to it at launch (P10)")
    # the artifact of record: re-hashed, never traced again
    win = sel.get("winner") or {}
    ck, ck_sha = A.verified_checkpoint(rec, win.get("epoch"))
    if ck_sha != win.get("checkpoint_sha256"):
        raise QATRefused("winner_checkpoint_changed", f"{ck} has sha256 {ck_sha}; the selection names "
                                                      f"{win.get('checkpoint_sha256')}")
    for part in ("artifact_of_record", "provenance", "summary"):
        p = E / (win.get(part) or {}).get("path", "")
        if not p.is_file() or Q.sha256_file(p) != (win.get(part) or {}).get("sha256"):
            raise QATRefused("winner_files_changed", f"{p} is missing or differs from the selection's sha256")
    epoch = win["epoch"]
    names = A.epoch_names(epoch)
    prov = json.loads((E / win["provenance"]["path"]).read_text(encoding="utf-8"))
    sd = E / A.CONVERTED_DIR / names["state_dict"]
    sd_sha = Q.sha256_file(sd) if sd.is_file() else None
    if sd_sha is None or sd_sha != (prov.get("state_dict_artifact") or {}).get("sha256"):
        raise QATRefused("winner_files_changed", f"{sd} is missing or differs from its provenance's sha256")
    record = {"epoch": epoch, "value": win["value"], "torchscript": win["artifact_of_record"],
              "provenance": win["provenance"], "summary": win["summary"],
              "state_dict_companion": {"path": f"{A.CONVERTED_DIR}/{names['state_dict']}", "sha256": sd_sha},
              "checkpoint": {"path": win["checkpoint"], "sha256": ck_sha}}
    # the x86 latency copy, translated from the sha-checked epoch checkpoint (zero optimizer steps)
    payload = load_qat_sidecar(ck)
    student = build_student(num_classes=Q.NUM_CLASSES, pretrained=False)
    prepared, report = translate_qat_state_to_x86(meta["stage"], student, payload)
    prepared.eval()
    engine = select_x86_backend()
    if engine not in X86_BACKENDS:
        raise QATRefused("x86_backend_unavailable", f"engine {engine!r}")
    x86_model = convert_model(prepared)
    xq = qconfig_summary(x86_qat_qconfig())
    x86_name = f"e{epoch:02d}_int8_x86_latency.torchscript.pt"
    x86_meta = {"schema": "plantseg-int8-torchscript/1.0.0", "stage": meta["stage"], "quantization": "qat",
                "num_classes": Q.NUM_CLASSES, "engine": engine, "artifact_role": X86_LATENCY_ARTIFACT_ROLE,
                "source_checkpoint_sha256": meta["source_checkpoint_sha256"], "qat_checkpoint_sha256": ck_sha,
                "epoch": epoch, "run_id": rec["run_id"], "qconfig_fingerprint": xq["fingerprint"],
                "usable_for_accuracy": False, "reduce_range": True}
    x86_b, notes = torchscript_bytes(x86_model, x86_meta)
    x86_sha = Q.sha256_bytes(x86_b)
    out_dir = E / FINAL_DIR
    out_dir.mkdir(parents=True)
    write_exclusive({out_dir / x86_name: x86_b})
    ts, _ = load_int8_torchscript(out_dir / x86_name, stage=meta["stage"], method="qat", engine=engine,
                                  artifact_role=X86_LATENCY_ARTIFACT_ROLE)
    parity = output_parity({"x86_eager": x86_model, "x86_torchscript": ts}, synthetic_parity_inputs(),
                           [("x86_eager", "x86_torchscript")])
    graph, weights = graph_census(ts), weight_scheme_report(ts)
    checks = {"zero_optimizer_steps": report.get("optimizer_steps_for_translation") == 0
              and report.get("gradient_updates") == 0 and report.get("retrained") is False,
              "parity_x86_copy": parity["all_equal"], "x86_copy_float_boundary": graph["ok"],
              "x86_weights_per_channel_int8": weights["ok"],
              "file_matches_serialization": Q.sha256_file(out_dir / x86_name) == x86_sha}
    # describe_qconfig holds torch dtypes and qschemes: the record carries the JSON summary of the same qconfig
    x86_prov = {**x86_artifact_provenance(stage=meta["stage"], engine=engine, source_sha256=ck_sha,
                                          artifact_path=f"{FINAL_DIR}/{x86_name}", translation=report),
                "qconfig": xq}
    ident = Q.code_identity()
    doc = {"format": FINALIZE_FORMAT, "stage": meta["stage"], "run_id": rec["run_id"], "seed": meta["seed"],
           "clip_norm": meta["clip_norm"], "telemetry_sha256": rec["telemetry_sha256"],
           "qat_selection": {"path": SELECTION_RECORD, "sha256": sel_sha}, "clip_selection": clip_doc,
           "artifact_of_record": record, "artifact_of_record_note": ("the scored TorchScript file itself, re-hashed; "
                                                                     "never traced again (P28)"),
           "x86_latency_copy": {"path": f"{FINAL_DIR}/{x86_name}", "sha256": x86_sha, "engine": engine,
                                "provenance": x86_prov, "checks": checks, "parity": parity, "graph": graph,
                                "weights": weights, "trace_warnings": notes, "usable_for_accuracy": False},
           "cpu_model": Q.cpu_model(), "git_head": ident["git_head"], "smoke_inputs": bool(args.allow_smoke_inputs),
           "created_wall_clock": time.time()}
    if not all(checks.values()):
        write_exclusive({E / FINALIZE_STOP: A.json_bytes({**doc, "status": "STOP"})})
        raise QATStop("x86_copy_checks", f"the x86 latency copy failed {[k for k, v in checks.items() if not v]}; "
                                         f"see {E / FINALIZE_STOP}")
    write_exclusive({E / FINALIZE_RECORD: A.json_bytes({**doc, "status": "finalized"})})
    aor = win["artifact_of_record"]
    print(f"[finalize] artifact of record {aor['path']} {aor['sha256'][:12]}; x86 latency copy "
          f"{FINAL_DIR}/{x86_name} {x86_sha[:12]} under {engine!r}")
    print(f"RESULT: FINALIZED ({meta['stage']}, seed {meta['seed']}, epoch {epoch:02d})")
    return EXIT_OK


# ------------------------------------------------------------------ check-run-meta (P14, P10, O2)
def _dotted(d, k):
    for part in k.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def profile_mismatches(meta: dict, args, *, repo_root: Path = REPO) -> dict:
    """Every constant of a real QAT launch's run_meta row, against the launch flags. {key: (got, want)}."""
    stage = args.stage.upper()
    want = {"event": "run_meta", "mode": "real", "stage": stage, "seed": args.seed, "clip_norm": args.grad_clip_norm,
            "git_head": args.expect_head, "code_clean_at_head": True,
            "source_checkpoint_sha256": args.expect_source_sha256, "image_digest": Q.IMAGE_DIGEST_OF_RECORD,
            "num_workers": NUM_WORKERS_OF_RECORD, "persistent_workers": True, "epochs": Q.EPOCHS,
            "steps_per_epoch": Q.REAL_STEPS_PER_EPOCH, "total_steps": Q.REAL_T_MAX,
            "scheduler.class": "CosineAnnealingLR", "scheduler.T_max": Q.REAL_T_MAX, "scheduler.eta_min": 0.0,
            "bn_freeze_after_step": Q.BN_FREEZE_EPOCH * Q.REAL_STEPS_PER_EPOCH,
            "obs_freeze_after_step": Q.OBS_FREEZE_EPOCH * Q.REAL_STEPS_PER_EPOCH,
            "bn_frozen_from_epoch": Q.BN_FREEZE_EPOCH + 1, "obs_frozen_from_epoch": Q.OBS_FREEZE_EPOCH + 1,
            "lr": Q.LEARNING_RATE, "momentum": Q.MOMENTUM, "weight_decay": Q.WEIGHT_DECAY, "nesterov": False,
            "dampening": 0.0, "optimizer.class": "SGD", "optimizer.param_groups": 1,
            "optimizer.group_holds_every_model_param": True, "batch_size": Q.BATCH_SIZE, "drop_last": True,
            "engine": "qnnpack", "observer_census": EXPECTED_CENSUS, "fake_quant_modules": 184,
            "unfused_batchnorm": 0, "tf32": Q.TF32_DEFAULTS, "cuda_initialized_at_seed": False,
            "parent.checkpoint_sha256": args.expect_source_sha256, "parent.seed": args.seed,
            "parent.mode": "real", "val_batches_cap": None}
    bad = {}
    for k, v in want.items():
        got = _dotted(meta, k)
        if got != v or type(got) is not type(v):
            bad[k] = (got, v)
    tv = str(meta.get("torch", ""))
    if tv.split("+")[0] != Q.TORCH_OF_RECORD:
        bad["torch"] = (tv, Q.TORCH_OF_RECORD)
    # the clip and its source (P10)
    if stage == Q.U4_STAGE and args.seed == Q.U4_SEED:
        if not (args.u4_pilot and meta.get("u4_pilot") is True and meta.get("clip_source") == "u4_pilot"
                and meta.get("clip_selection") is None and args.clip_selection is None):
            bad["clip_source"] = ((meta.get("clip_source"), meta.get("u4_pilot")),
                                  "u4_pilot: E5 seed 42 is a U4 pilot run, launched and checked with --u4-pilot")
    else:
        cs = meta.get("clip_selection") or {}
        if args.u4_pilot or meta.get("u4_pilot") is not False or meta.get("clip_source") != "clip_selection":
            bad["clip_source"] = ((meta.get("clip_source"), meta.get("u4_pilot")),
                                  f"clip_selection: {stage} seed {args.seed} takes the U4 winner")
        elif cs.get("sha256") != args.clip_selection_sha256 or cs.get("winner") != args.grad_clip_norm:
            bad["clip_selection"] = ((cs.get("sha256"), cs.get("winner")),
                                     (args.clip_selection_sha256, args.grad_clip_norm))
        else:
            try:
                doc = Q.read_clip_selection(args.clip_selection, args.clip_selection_sha256)
                if doc["winner"] != args.grad_clip_norm:
                    bad["clip_selection_winner"] = (doc["winner"], args.grad_clip_norm)
            except QATRefused as e:
                bad["clip_selection_file"] = (e.code, str(e))
    # E6: the λ and α selections and the parent they name (O2)
    if stage == "E6":
        try:
            lam = Q.read_tracked_selection(args.lambda_selection, args.lambda_selection_sha256,
                                           Q.LAMBDA_SELECTION_FORMAT, "lambda", repo_root)
            alpha = Q.read_tracked_selection(args.alpha_selection, args.alpha_selection_sha256,
                                             Q.ALPHA_SELECTION_FORMAT, "alpha", repo_root)
            Q.e6_parent_binding(meta.get("parent") or {}, args.seed, lam, alpha)
            sels = meta.get("e6_selections") or {}
            for what, doc in (("lambda_selection", lam), ("alpha_selection", alpha)):
                if (sels.get(what) or {}).get("sha256") != doc["sha256"]:
                    bad[f"e6_selections.{what}"] = ((sels.get(what) or {}).get("sha256"), doc["sha256"])
        except QATRefused as e:
            bad["e6_binding"] = (e.code, str(e))
        if meta.get("cwd_projection_loaded") is not False:
            bad["cwd_projection_loaded"] = (meta.get("cwd_projection_loaded"), False)
    elif any(v is not None for v in (args.lambda_selection, args.lambda_selection_sha256, args.alpha_selection,
                                     args.alpha_selection_sha256)) or meta.get("e6_selections") is not None:
        bad["e6_selections"] = (meta.get("e6_selections"), f"none: {stage} takes no λ or α selection")
    return bad


def cmd_check_run_meta(args) -> int:
    A.refuse_test_path(args.run_dir, "run_dir")
    tel = Path(args.run_dir) / Q.TELEMETRY_NAME
    if not tel.is_file():
        raise A.QATIncomplete("telemetry_missing", f"{tel} is missing (the run has not written its run_meta row)")
    with open(tel, encoding="utf-8") as fh:
        first = fh.readline()
    def non_strict(c):
        raise ValueError(f"non-strict JSON constant {c}")
    try:
        meta = json.loads(first, parse_constant=non_strict)
    except ValueError as e:
        raise QATRefused("run_meta_rows", f"{tel}'s first line is not a strict JSON row ({e})") from e
    if not isinstance(meta, dict) or meta.get("event") != "run_meta":
        raise QATRefused("run_meta_rows", f"{tel}'s first row is not the run_meta row")
    bad = profile_mismatches(meta, args)
    if bad:
        for k, (g, w) in sorted(bad.items()):
            print(f"  {k}: run_meta {g!r}, expected {w!r}")
        raise QATStop("run_meta_profile", f"{len(bad)} run_meta value(s) differ from the QAT launch of record: "
                                          f"{sorted(bad)}. Stop the run")
    print(f"[check-run-meta] {meta['stage']} seed {meta['seed']} clip {meta['clip_norm']} ({meta['clip_source']}), "
          f"HEAD {meta['git_head']}, every profile value as recorded")
    print("RESULT: CHECK-RUN-META PASS")
    return EXIT_OK


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"convert": cmd_convert, "convert-epoch": cmd_convert_epoch, "score": cmd_score,
                "finalize": cmd_finalize, "check-run-meta": cmd_check_run_meta}
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
