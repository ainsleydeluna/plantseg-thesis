#!/usr/bin/env python3
"""Block B of an AM-29 arm (lane L-AM29, P3): convert, score all-INT8 on VAL, select under AM-29's rules, finalize.

    convert         --run-dir R --eval-dir E --host-label TEXT --expect-telemetry-sha256 H
    convert-epoch   the fresh-process worker `convert` spawns, one epoch per process
    score           --run-dir R --eval-dir E --host-label TEXT [--threads T]
    select          --run-dir R --eval-dir E --expect-telemetry-sha256 H
    finalize        --run-dir R --eval-dir E
    (smokes only: --allow-smoke-inputs on each, and --max-samples N on score)

The pinned scripts/qat_epoch_eval.py and scripts/select_qat_epoch.py refuse AM-29's stages (resolve_quant_stage, and
scripts/evaluate_model.py's --stage choices), so these are new entry points over the pinned functions:
  convert   the run's record (AM-19) and its telemetry sha256, the freeze cross-check (qat_artifacts.freeze_cross_check),
            then src/quant/qat_artifacts.py convert_epoch, unchanged, for each epoch in a fresh process, with the AM-29
            stage registered in that process only (src/quant/am29_registry.py): the QNNPACK TorchScript artifact, its
            state_dict companion and eNN_run_meta.json, every check of record; am29_convert.json
  score     every converted epoch scored all-INT8 on VAL, batch 1, CPU, canvas, by scripts/am29_score.py --provenance in a
            fresh process (the evaluator's own loading, preprocessing and metric code, its INT8 provenance checks);
            scores/eNN.json per epoch and am29_epoch_eval.json
  select    configs/qat_selection_rules_am29.json, read through qat_select.load_rules: the highest all-INT8 converted
            VAL all-class mIoU of the 15 epochs, strict >, a tie keeps the earlier epoch; an epoch whose checkpoint
            fails AM-21 item 1(a) is excluded (qat_select.checkpoint_states decides); the fake-quant VAL never selects;
            every input re-verified (checkpoint -> provenance -> TorchScript identity -> score); am29_selection.json
  finalize  am29_finalize.json names the selected epoch's scored TorchScript as the artifact of record (re-hashed, never
            traced again) with its provenance, state_dict companion and checkpoint. There is no x86 latency copy.

Every subcommand ends with one RESULT line. Exit codes: 0 done, 1 STOP, 2 refused, 3 incomplete, 4 error.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.quant import am29_registry as R  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant.qat import EXIT_ABORTED, EXIT_ERROR, EXIT_OK, EXIT_REFUSED, EXIT_STOP, QATRefused, QATStop  # noqa: E402

SCRIPT = Path(__file__).resolve()
SCORER = "scripts/am29_score.py"
CONVERT_RECORD, CONVERT_STOP = "am29_convert.json", "am29_convert_STOP.json"
EVAL_RECORD, SCORE_STOP = "am29_epoch_eval.json", "am29_score_STOP.json"
SELECTION_RECORD, FINALIZE_RECORD = "am29_selection.json", "am29_finalize.json"
CONVERT_FORMAT, EVAL_FORMAT = "am29_convert/1", "am29_epoch_eval/1"
SELECTION_FORMAT, FINALIZE_FORMAT = "am29_selection/1", "am29_finalize/1"
RULES_PATH = REPO / R.RULES_REL
EVAL_IDENTITY_KEYS = ("git_head", "host_label", "cpu_model", "purpose")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("convert", "convert-epoch", "score", "select", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--eval-dir", required=True)
        p.add_argument("--allow-smoke-inputs", action="store_true")
        if name in ("convert", "convert-epoch", "score"):
            p.add_argument("--host-label", required=True)
        if name in ("convert", "convert-epoch", "select"):
            p.add_argument("--expect-telemetry-sha256", required=name != "convert-epoch", default=None)
        if name == "convert-epoch":
            p.add_argument("--epoch", type=int, required=True)
        if name == "score":
            p.add_argument("--threads", type=int, default=None)
            p.add_argument("--max-samples", type=int, default=None)
    return ap


def _json_bytes(doc: dict) -> bytes:
    return A.json_bytes(doc)


def _write(path: Path, doc: dict) -> None:
    from src.quant.ptq import write_exclusive
    write_exclusive({path: _json_bytes(doc)})


def _read(path: Path, code: str) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QATRefused(code, f"{path}: {e}") from e
    if not isinstance(doc, dict):
        raise QATRefused(code, f"{path} is not a JSON object")
    return doc


def _identity(args, purpose: str = "record") -> dict:
    from src.quant.am29 import code_identity
    return {"git_head": code_identity()["git_head"], "host_label": getattr(args, "host_label", None),
            "cpu_model": Q.cpu_model(), "purpose": purpose, "smoke_inputs": bool(args.allow_smoke_inputs)}


def _am29_run(args) -> dict:
    """The run's record, complete, real (or smoke with --allow-smoke-inputs), of an AM-29 stage and arm."""
    A.refuse_test_path(args.eval_dir, "eval_dir")
    rec = A.read_run_record(args.run_dir)
    A.require_complete(rec)
    meta = rec["run_meta"]
    if meta.get("mode") != "real" and not args.allow_smoke_inputs:
        raise QATRefused("run_not_real", f"the run's mode is {meta.get('mode')!r}; only a real run is converted")
    stage = meta.get("stage")
    if stage not in R.AM29_STAGES or (meta.get("am29") or {}).get("arm") != R.STAGE_ARMS.get(stage):
        raise QATRefused("stage_not_am29", f"the run records stage {stage!r}, arm {(meta.get('am29') or {}).get('arm')!r}; "
                                           f"this block B takes AM-29's stages {list(R.AM29_STAGES)}")
    return rec


# ------------------------------------------------------------------ convert
def cmd_convert(args) -> int:
    rec = _am29_run(args)
    if rec["telemetry_sha256"] != args.expect_telemetry_sha256:
        raise QATRefused("telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {rec['telemetry_sha256']}, "
                                                      f"expected {args.expect_telemetry_sha256}")
    E = Path(args.eval_dir).resolve()
    if E.exists() and (not E.is_dir() or any(E.iterdir())):
        raise QATRefused("eval_dir_not_fresh", f"{E} exists and is not empty; every conversion starts in a new eval dir")
    cross = A.freeze_cross_check(rec)
    E.mkdir(parents=True)
    head = {"format": CONVERT_FORMAT, "run_dir": rec["run_dir"], "run_id": rec["run_id"],
            "stage": rec["run_meta"]["stage"], "arm": rec["run_meta"]["am29"]["arm"],
            "telemetry_sha256": rec["telemetry_sha256"], **_identity(args), "freeze_cross_check": cross}
    if not cross["ok"]:
        _write(E / CONVERT_STOP, {**head, "status": "STOP", "code": "freeze_cross_check"})
        raise QATStop("freeze_cross_check", f"the freezes did not take: {cross['first_differences']}")
    outcomes = []
    for e in range(1, Q.EPOCHS + 1):
        argv = [sys.executable, "-B", str(SCRIPT), "convert-epoch", "--run-dir", rec["run_dir"], "--eval-dir", str(E),
                "--epoch", str(e), "--host-label", args.host_label, "--expect-telemetry-sha256", rec["telemetry_sha256"]]
        if args.allow_smoke_inputs:
            argv.append("--allow-smoke-inputs")
        t0 = time.time()
        p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
        names = A.epoch_names(e)
        status = ("converted" if (E / A.CONVERTED_DIR / names["run_meta"]).is_file()
                  else "not convertible" if (E / A.CONVERTED_DIR / names["not_convertible"]).is_file()
                  else "STOP" if (E / A.CONVERTED_DIR / names["stop"]).is_file() else "failed")
        tail = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT:")]
        outcomes.append({"epoch": e, "status": status, "exit_code": p.returncode, "seconds": time.time() - t0,
                         "result": tail[-1] if tail else None})
        print(f"[am29-convert] e{e:02d}: {status} (exit {p.returncode}, {time.time() - t0:.1f}s)")
        if p.returncode != EXIT_OK or status == "failed":
            sys.stderr.write(p.stdout[-1500:] + p.stderr[-1500:])
            _write(E / CONVERT_STOP, {**head, "status": "STOP", "code": "worker_failed", "outcomes": outcomes})
            print(f"RESULT: STOP -- e{e:02d} worker exited {p.returncode} ({status}); {E} is spent")
            return EXIT_STOP
    _write(E / CONVERT_RECORD, {**head, "status": "converted", "outcomes": outcomes, "created_wall_clock": time.time()})
    n = sum(o["status"] == "converted" for o in outcomes)
    print(f"RESULT: CONVERTED {n}/{len(outcomes)} (AM-29 {head['arm']})")
    return EXIT_OK


def cmd_convert_epoch(args) -> int:
    R.register()                                          # AM-29's stage resolves in this worker only
    if args.expect_telemetry_sha256:
        rec = A.read_run_record(args.run_dir)
        if rec["telemetry_sha256"] != args.expect_telemetry_sha256:
            raise QATRefused("telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {rec['telemetry_sha256']}")
    out = A.convert_epoch(args.run_dir, args.eval_dir, args.epoch, purpose="record", host_label=args.host_label,
                          allow_smoke_inputs=args.allow_smoke_inputs)
    print(f"RESULT: AM-29 EPOCH {args.epoch:02d} {out['status'].upper()}")
    return EXIT_OK


# ------------------------------------------------------------------ score
def scorer_command(stage: str, prov: Path, artifact_sha: str, out: Path, args) -> list[str]:
    argv = [sys.executable, "-B", SCORER, "--split", "val", "--precision", "int8", "--provenance", str(prov),
            "--stage", stage, "--expect-artifact-sha256", artifact_sha, "--out-json", str(out),
            "--host-label", args.host_label]
    if args.threads is not None:
        argv += ["--threads", str(args.threads)]
    if args.max_samples is not None:
        argv += ["--max-samples", str(args.max_samples), "--allow-smoke-inputs"]
    return argv


def cmd_score(args) -> int:
    if args.max_samples is not None and not args.allow_smoke_inputs:
        raise QATRefused("max_samples", "--max-samples is for smokes (with --allow-smoke-inputs); a record pass scores "
                                        "the whole VAL split")
    rec = _am29_run(args)
    E = Path(args.eval_dir).resolve()
    if (E / CONVERT_STOP).exists() or (E / SCORE_STOP).exists() or list((E / A.CONVERTED_DIR).glob("e*_STOP.json")):
        raise QATRefused("stop_present", f"{E} holds a STOP file; it is spent and is never scored")
    conv = _read(E / CONVERT_RECORD, "convert_record_missing")
    if conv.get("telemetry_sha256") != rec["telemetry_sha256"] or conv.get("run_id") != rec["run_id"]:
        raise QATRefused("convert_record_mismatch", f"{E / CONVERT_RECORD} is not this run's")
    if bool(conv.get("smoke_inputs")) != bool(args.allow_smoke_inputs):
        raise QATRefused("smoke_inputs_mismatch", f"{E / CONVERT_RECORD} smoke inputs {conv.get('smoke_inputs')}")
    scores = E / A.SCORES_DIR
    if scores.exists() or (E / EVAL_RECORD).exists():
        raise QATRefused("scores_not_fresh", f"{scores} or {E / EVAL_RECORD} exists: an eval directory is scored once")
    scores.mkdir()
    stage = rec["run_meta"]["stage"]
    rows = []
    for o in conv["outcomes"]:
        e = o["epoch"]
        names = A.epoch_names(e)
        ck_sha = rec["ends"][e]["checkpoint_sha256"]
        row = {"epoch": e, "checkpoint_sha256": ck_sha}
        if o["status"] == "not convertible":
            rows.append({**row, "status": "excluded", "rule": A.NOT_CONVERTIBLE_RULE,
                         "not_convertible": f"{A.CONVERTED_DIR}/{names['not_convertible']}"})
            print(f"[am29-score] e{e:02d}: {A.NOT_CONVERTIBLE_RULE}")
            continue
        prov = E / A.CONVERTED_DIR / names["run_meta"]
        pdoc = _read(prov, "provenance_unreadable")
        out = scores / f"e{e:02d}.json"
        argv = scorer_command(stage, prov, pdoc["converted_artifact_sha256"], out, args)
        t0 = time.time()
        p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
        if p.returncode != 0 or not out.is_file():
            sys.stderr.write(p.stdout[-1500:] + p.stderr[-1500:])
            _write(E / SCORE_STOP, {"format": EVAL_FORMAT, "status": "STOP", "run_id": rec["run_id"], "epoch": e,
                                    "scorer_exit_code": p.returncode, "scored": rows, **_identity(args)})
            print(f"RESULT: INCOMPLETE -- the scorer exited {p.returncode} on e{e:02d}; {E} is spent ({SCORE_STOP})")
            return EXIT_ABORTED
        sdoc = _read(out, "score_unreadable")
        rows.append({**row, "status": "scored", "score": {"path": f"{A.SCORES_DIR}/{out.name}", "sha256": Q.sha256_file(out)},
                     "provenance": {"path": f"{A.CONVERTED_DIR}/{names['run_meta']}", "sha256": Q.sha256_file(prov)},
                     "converted_artifact_sha256": pdoc["converted_artifact_sha256"],
                     "all_class_miou": sdoc["score"]["dataset_level"].get("all_class_miou"),
                     "scorer_command": argv[1:], "seconds": time.time() - t0})
        print(f"[am29-score] e{e:02d}: all-INT8 VAL all-class mIoU {rows[-1]['all_class_miou']!r} "
              f"({time.time() - t0:.1f}s)")
    _write(E / EVAL_RECORD, {"format": EVAL_FORMAT, "run_dir": rec["run_dir"], "run_id": rec["run_id"], "stage": stage,
                             "telemetry_sha256": rec["telemetry_sha256"], **_identity(args),
                             "max_samples": args.max_samples, "epochs": rows, "created_wall_clock": time.time()})
    n = sum(r["status"] == "scored" for r in rows)
    print(f"RESULT: SCORED {n}/{len(rows)} (all-INT8, VAL)")
    return EXIT_OK


# ------------------------------------------------------------------ select
def load_rules() -> tuple[dict, str]:
    """configs/qat_selection_rules_am29.json through the pinned load_rules, the file registered for AM-29's stages."""
    from src.quant import qat_select as S
    R.register()
    rules, sha = S.load_rules(RULES_PATH)
    ev = rules["qat_epoch"]["evaluator"]
    want = {"split": "val", "precision": "int8", "quant_backend": "qnnpack", "device": "cpu",
            "preprocess_protocol": "core_preprocess/1.0.0", "expected_rows": 846}
    bad = {k: ev.get(k) for k, v in want.items() if ev.get(k) != v}
    if bad or rules["qat_epoch"].get("stages") != list(R.AM29_STAGES):
        raise QATRefused("rules_mismatch", f"{RULES_PATH} is not AM-29's rules: {bad}")
    return rules, sha


def _score_errors(s: dict, *, rules: dict, smoke: bool) -> list[str]:
    ev = rules["qat_epoch"]["evaluator"]
    sc = s.get("score") or {}
    bad = []
    want = {"label": "int8", "precision": "int8", "source_kind": "provenance", "dataset_level_only": True}
    bad += [f"{k} = {s.get(k)!r}" for k, v in want.items() if s.get(k) != v]
    want_sc = {"split": ev["split"], "protocol": ev["preprocess_protocol"], "batch_size": ev["eval_runtime"]["batch_size"],
               "num_workers": ev["eval_runtime"]["num_workers"]}
    bad += [f"score.{k} = {sc.get(k)!r}" for k, v in want_sc.items() if sc.get(k) != v]
    rows = (sc.get("expected_rows"), sc.get("actual_rows"), sc.get("forward_batches"))
    if not smoke:
        if rows != (ev["expected_rows"], ev["expected_rows"], ev["eval_runtime"]["forward_batches"]) \
                or sc.get("max_samples") is not None:
            bad.append(f"rows {rows} (cap {sc.get('max_samples')!r}); the rules require {ev['expected_rows']}")
        if sc.get("gt_support_sum") != ev["gt_support_sum"]:
            bad.append(f"gt_support_sum {sc.get('gt_support_sum')!r}, the rules' {ev['gt_support_sum']}")
    elif not (rows[0] == rows[1] == rows[2]):
        bad.append(f"rows {rows} differ")
    return bad


def am29_epoch_selection(args, *, rules: dict, rules_sha256: str) -> dict:
    from src.quant import qat_select as S
    smoke = bool(args.allow_smoke_inputs)
    rec = _am29_run(args)
    if rec["telemetry_sha256"] != args.expect_telemetry_sha256:
        raise QATRefused("telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {rec['telemetry_sha256']}")
    meta = rec["run_meta"]
    stage = meta["stage"]
    if stage not in rules["qat_epoch"]["stages"]:
        raise QATRefused("stage", f"stage {stage!r} is not a stage of the AM-29 rules")
    E = Path(args.eval_dir)
    stops = [n for n in (CONVERT_STOP, SCORE_STOP) if (E / n).exists()]
    stops += sorted(p.name for p in (E / A.CONVERTED_DIR).glob("e*_STOP.json")) if (E / A.CONVERTED_DIR).is_dir() else []
    if stops:
        raise QATRefused("stop_present", f"{E} holds STOP files {stops}; it is never selected")
    conv, evd = _read(E / CONVERT_RECORD, "eval_record_missing"), _read(E / EVAL_RECORD, "eval_record_missing")
    for name, doc in ((CONVERT_RECORD, conv), (EVAL_RECORD, evd)):
        if doc.get("telemetry_sha256") != rec["telemetry_sha256"] or doc.get("run_id") != rec["run_id"]:
            raise QATRefused("eval_record_mismatch", f"{name} is not this run's")
        if doc.get("purpose") != "record" or bool(doc.get("smoke_inputs")) != smoke:
            raise QATRefused("eval_record_purpose", f"{name}: purpose {doc.get('purpose')!r}, smoke {doc.get('smoke_inputs')}")
    split = {k: sorted(map(str, {conv.get(k), evd.get(k)})) for k in EVAL_IDENTITY_KEYS if conv.get(k) != evd.get(k)}
    if split:
        raise QATRefused("eval_dir_mixed", f"one selection uses one commit, one host and one eval dir: {split}")
    eval_identity = {k: evd.get(k) for k in EVAL_IDENTITY_KEYS}
    rows = {r["epoch"]: r for r in evd.get("epochs", [])}
    if sorted(rows) != list(range(1, Q.EPOCHS + 1)):
        raise A.QATIncomplete("eval_incomplete", f"{EVAL_RECORD} covers epochs {sorted(rows)}")
    checks = S.checkpoint_states(rec)                       # AM-21 item 1(a) on every checkpoint: it decides
    scored, excluded, trace, docs = {}, [], [], {}
    for e in range(1, Q.EPOCHS + 1):
        row = rows[e]
        finite, _fails, _ck, ck_sha = checks[e]
        if row.get("checkpoint_sha256") != ck_sha:
            raise QATRefused("chain_checkpoint", f"e{e:02d}: {EVAL_RECORD} names {row.get('checkpoint_sha256')}, "
                                                 f"the file has {ck_sha}")
        if not finite:
            if row.get("status") != "excluded":
                trace.append(f"e{e:02d} {A.NOT_CONVERTIBLE_RULE}: its checkpoint fails item 1(a) though its record says "
                             f"{row.get('status')!r}; the checkpoint decides")
            else:
                trace.append(f"e{e:02d} {A.NOT_CONVERTIBLE_RULE}; not scanned")
            excluded.append(e)
            continue
        if row.get("status") != "scored":
            raise QATRefused("exclusion_inconsistent", f"e{e:02d} is {row.get('status')!r} but its state is finite")
        prov_p, sc_p = E / row["provenance"]["path"], E / row["score"]["path"]
        if Q.sha256_file(prov_p) != row["provenance"]["sha256"] or Q.sha256_file(sc_p) != row["score"]["sha256"]:
            raise QATRefused("chain_files", f"e{e:02d}: the provenance or the score record changed")
        prov, sdoc = _read(prov_p, "provenance_unreadable"), _read(sc_p, "score_unreadable")
        ts_p = prov_p.parent / str(prov.get("converted_artifact"))
        ts_sha = Q.sha256_file(ts_p)
        ident = A.read_ts_identity(ts_p)
        src = sdoc.get("source") or {}
        chain = {"provenance checkpoint": prov.get("qat_checkpoint_sha256") == ck_sha,
                 "identity checkpoint": ident.get("qat_checkpoint_sha256") == ck_sha,
                 "artifact sha256": ts_sha == prov.get("converted_artifact_sha256") == row.get("converted_artifact_sha256")
                 == src.get("artifact_sha256"),
                 "score provenance": src.get("provenance_sha256") == row["provenance"]["sha256"],
                 "epoch": prov.get("epoch") == ident.get("epoch") == (src.get("identity") or {}).get("epoch") == e,
                 "run_id": prov.get("run_id") == ident.get("run_id") == rec["run_id"],
                 "stage": prov.get("stage") == ident.get("stage") == src.get("stage") == stage,
                 "verification": bool((prov.get("verification") or {}).get("checks"))
                 and all((prov.get("verification") or {}).get("checks", {}).values()),
                 "score git_head": sdoc.get("git_head") == eval_identity["git_head"],
                 "no STOP file": not (E / A.CONVERTED_DIR / A.epoch_names(e)["stop"]).exists()}
        if not all(chain.values()):
            raise QATRefused("chain_broken", f"e{e:02d}: {[k for k, v in chain.items() if not v]}")
        bad = _score_errors(sdoc, rules=rules, smoke=smoke)
        if bad:
            raise QATRefused("score_values", f"e{e:02d}: {bad}")
        v = sdoc["score"]["dataset_level"].get("all_class_miou")
        if not isinstance(v, float) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise QATRefused("score_invalid", f"e{e:02d}: all_class_miou {v!r} is not a float in [0, 1]")
        docs[e] = sdoc
        scored[e] = {"value": v, "checkpoint_sha256": ck_sha, "provenance": row["provenance"], "score": row["score"],
                     "artifact_of_record": {"path": f"{A.CONVERTED_DIR}/{prov['converted_artifact']}", "sha256": ts_sha}}
    if not scored:
        raise QATRefused("no_convertible_epoch", f"{S.NO_MODEL_ENTRY}: every epoch of {rec['run_id']} is excluded "
                                                 "(non-finite state, AM-21 item 2(a)); no epoch is selected")
    eq = {}
    for k in rules["qat_epoch"]["equal_across_scores"]:
        vals = set()
        for d in docs.values():
            sc = d["score"]
            got = {"git_head": d.get("git_head"), "torch": sc.get("torch"),
                   "torch_num_threads": sc.get("torch_num_threads"), "class_map_sha256": sc.get("class_map_sha256"),
                   "manifest_ids_sha256": sc.get("manifest_ids_sha256"),
                   "image_digest": (sc.get("eval_runtime") or {}).get("image_digest")}[k]
            vals.add(json.dumps(got, sort_keys=True))
        eq[k] = len(vals) == 1
    if not all(eq.values()):
        raise QATRefused("scores_differ", f"fields not equal across the scored epochs: {[k for k, v in eq.items() if not v]}")
    trace.insert(0, f"rule {R.RULES_REL} qat_epoch (sha256 {rules_sha256}): highest all-INT8 converted VAL all-class "
                    "mIoU, strict >, tie -> the earlier epoch; fake-quant VAL never selects; an epoch whose checkpoint "
                    "fails item 1(a) is excluded (AM-21 item 2(a))")
    best, ties = None, []
    for e in sorted(scored):
        v = scored[e]["value"]
        if best is None:
            best, ties = e, [e]
            trace.append(f"e{e:02d} all-INT8 {v!r} -> best e{e:02d}")
        elif v > scored[best]["value"]:
            trace.append(f"e{e:02d} all-INT8 {v!r} > best {scored[best]['value']!r} (e{best:02d}) -> best e{e:02d}")
            best, ties = e, [e]
        elif v == scored[best]["value"]:
            ties.append(e)
            trace.append(f"e{e:02d} all-INT8 {v!r} == best (e{best:02d}) -> tie; the earlier epoch e{best:02d} is kept")
        else:
            trace.append(f"e{e:02d} all-INT8 {v!r} < best {scored[best]['value']!r} (e{best:02d})")
    w = scored[best]
    if Q.sha256_file(E / w["artifact_of_record"]["path"]) != w["artifact_of_record"]["sha256"]:
        raise QATRefused("winner_changed", f"e{best:02d}'s artifact changed during selection")
    fq = {e: r.get("fake_quant_val_all_class_miou") for e, r in rec["ends"].items()
          if isinstance(r.get("fake_quant_val_all_class_miou"), float)}
    fq_argmax = max(sorted(fq), key=lambda e: fq[e]) if fq else None
    trace.append(f"winner e{best:02d} (checkpoint sha256 {w['checkpoint_sha256']}); fake-quant VAL argmax "
                 f"{('e%02d' % fq_argmax) if fq_argmax else None} recorded, not used")
    return {"format": SELECTION_FORMAT, "rules": {"path": R.RULES_REL, "sha256": rules_sha256}, "rules_sha256": rules_sha256,
            "stage": stage, "arm": meta["am29"]["arm"], "run_dir": rec["run_dir"], "run_id": rec["run_id"],
            "seed": meta["seed"], "clip_norm": meta["clip_norm"], "lr": meta.get("lr"),
            "telemetry_sha256": rec["telemetry_sha256"], "eval_dir": str(E.resolve()), "eval_identity": eval_identity,
            "winner": {"epoch": best, "value": w["value"], "checkpoint": f"{Q.EPOCH_DIR}/e{best:02d}.pt",
                       "checkpoint_sha256": w["checkpoint_sha256"], "artifact_of_record": w["artifact_of_record"],
                       "provenance": w["provenance"], "score": w["score"]},
            "tied_epochs": ties, "excluded_epochs": excluded,
            "values": {f"e{e:02d}": scored[e]["value"] for e in sorted(scored)},
            "fake_quant_val_argmax_epoch": fq_argmax, "fake_quant_val_used": False, "label": "int8",
            "smoke_inputs": smoke, "rule_trace": trace}


def cmd_select(args) -> int:
    E = Path(args.eval_dir)
    out = E / SELECTION_RECORD
    if out.exists():
        raise QATRefused("output_exists", f"{out} exists; a selection is written once")
    rules, sha = load_rules()
    sel = am29_epoch_selection(args, rules=rules, rules_sha256=sha)
    _write(out, sel)
    for line in sel["rule_trace"]:
        print(f"  {line}")
    w = sel["winner"]
    print(f"RESULT: SELECTED epoch {w['epoch']:02d} ({sel['stage']}, seed {sel['seed']}, all-INT8 VAL {w['value']!r})")
    return EXIT_OK


# ------------------------------------------------------------------ finalize
def cmd_finalize(args) -> int:
    rec = _am29_run(args)
    E = Path(args.eval_dir)
    if (E / FINALIZE_RECORD).exists():
        raise QATRefused("output_exists", f"{E / FINALIZE_RECORD} exists")
    sel_p = E / SELECTION_RECORD
    sel = _read(sel_p, "selection_missing")
    rules, rules_sha = load_rules()
    if sel.get("format") != SELECTION_FORMAT or sel.get("rules_sha256") != rules_sha:
        raise QATRefused("selection_format", f"{sel_p} is not an {SELECTION_FORMAT} record of the committed AM-29 rules")
    if sel.get("telemetry_sha256") != rec["telemetry_sha256"] or sel.get("run_id") != rec["run_id"]:
        raise QATRefused("selection_mismatch", f"{sel_p} is not this run's")
    if bool(sel.get("smoke_inputs")) != bool(args.allow_smoke_inputs):
        raise QATRefused("smoke_inputs", f"{sel_p} smoke inputs {sel.get('smoke_inputs')}")
    w = sel["winner"]
    ck, ck_sha = A.verified_checkpoint(rec, w["epoch"])
    if ck_sha != w["checkpoint_sha256"]:
        raise QATRefused("winner_checkpoint_changed", f"{ck} has sha256 {ck_sha}")
    for part in ("artifact_of_record", "provenance", "score"):
        p = E / w[part]["path"]
        if not p.is_file() or Q.sha256_file(p) != w[part]["sha256"]:
            raise QATRefused("winner_files_changed", f"{p} is missing or differs from the selection's sha256")
    names = A.epoch_names(w["epoch"])
    prov = _read(E / w["provenance"]["path"], "provenance_unreadable")
    sd = E / A.CONVERTED_DIR / names["state_dict"]
    if not sd.is_file() or Q.sha256_file(sd) != (prov.get("state_dict_artifact") or {}).get("sha256"):
        raise QATRefused("winner_files_changed", f"{sd} differs from its provenance's sha256")
    doc = {"format": FINALIZE_FORMAT, "status": "finalized", "stage": sel["stage"], "arm": sel["arm"],
           "run_id": rec["run_id"], "seed": sel["seed"], "telemetry_sha256": rec["telemetry_sha256"],
           "am29_selection": {"path": SELECTION_RECORD, "sha256": Q.sha256_file(sel_p)},
           "artifact_of_record": {"torchscript": w["artifact_of_record"], "provenance": w["provenance"],
                                  "checkpoint": {"path": w["checkpoint"], "sha256": ck_sha}, "score": w["score"],
                                  "state_dict_companion": {"path": f"{A.CONVERTED_DIR}/{names['state_dict']}",
                                                           "sha256": Q.sha256_file(sd)}},
           "label": "int8", "x86_latency_copy": None,
           "note": "the scored TorchScript itself, re-hashed, never traced again; no x86 latency copy (not in AM-29's P3)",
           **_identity(args), "created_wall_clock": time.time()}
    _write(E / FINALIZE_RECORD, doc)
    print(f"RESULT: FINALIZED ({sel['stage']}, seed {sel['seed']}, epoch {w['epoch']:02d}, artifact "
          f"{w['artifact_of_record']['sha256'][:12]})")
    return EXIT_OK


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"convert": cmd_convert, "convert-epoch": cmd_convert_epoch, "score": cmd_score, "select": cmd_select,
                "finalize": cmd_finalize}
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
