#!/usr/bin/env python3
"""Block B of an exploratory QAT run (lane L-AM24; C1, N1, N2, N2R2, N5, D): 15 converted epochs scored on the CPU,
AM-23's selection rules, finalize.

    python -B -I scripts/explore_qat_blockb.py --run-dir <item dir>/<run-id> --expect-telemetry-sha256 <sha>
        --host-label TEXT --run-id x-...-bb --out-dir <root>/x-...-bb [--threads 8]

The guard runs first (VAL, the data root's TRAIN/VAL-only check, x- run ID, the output outside the repository,
every official folder and the run directory). The run is identified from its x_run_meta.json (the item folder that
holds the trainer's run directory): it must be an exploratory C1, N1, N2, N2R2, N5 or D run.

  C1, N1, N2, N2R2, D: the existing scripts, unchanged, as subprocesses from the checkout, into <out-dir>/eval:
                 qat_epoch_eval.py convert --purpose record, score (the evaluator at --artifact-status provisional:
                 the evaluator has no exploratory status), select_qat_epoch.py (the rules file of the run's stage:
                 AM-23's for E6-E2 and E6-R2), qat_epoch_eval.py finalize. They accept the run because its telemetry
                 is the pinned trainer's own (stage E6-E2, E6-R2 or E6, mode real); the x- run ID travels in every
                 artifact name.
  N2, N2R2 also: the run's stage is its item's (N2 E6-E2, N2R2 E6-R2), the selection's rules are AM-23's file, and,
                 after finalize, card fix 1 of the lane's audit (DL-251; AM-24 item 10(b)): the selected epoch's
                 fake-quant VAL (its epoch_end row) and fake-quant-off VAL, the latter by the pinned
                 scripts/evaluate_model.py at precision qat_fakequant_off (lane L-FQOFF), unchanged, as a subprocess
                 into <out-dir>/fqoff/eNN, each value recorded with its file and sha256 (DL-176's form). The
                 evaluator's own stage list (src/eval/artifacts.py FQOFF_STAGES) decides: it holds E6-E2, not E6-R2,
                 so for N2R2 the evaluator refuses and block B ends INCOMPLETE with that refusal recorded.
  N5:            its own steps (the registered converter refuses the FP32 head's key set and its float fallback):
                 each epoch's state, sha-checked against its epoch_end row, converted through the pinned copy and
                 convert onto the FP32-head skeleton (src/explore/fp32_head.py), the islands report, VAL scored on the
                 CPU through the evaluator library (src/explore/valscore.py); selection: the highest VAL, strict >,
                 a tie keeps the earlier epoch, a non-finite-state epoch excluded (AM-23's rule as written); finalize:
                 the selected converted model traced to TorchScript with an x_mixed_precision.json identity (never
                 plantseg_int8.json), MIXED PRECISION in every record.

x_blockb.json records every step, the winner and artifact_status exploratory. Exit codes: 0 done, 2 refused, 3 a
step failed or an input changed, 4 error.
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

from src.explore import guard as G                                   # noqa: E402

REGISTERED_ITEMS = ("C1", "N1", "N2", "N2R2", "D")
FORMAT = "x_qat_blockb/1"
# Card fix 1 (DL-251; AM-24 item 10(b)), for N2 and N2R2: the selected epoch's fake-quant and fake-quant-off VAL.
FQOFF_ITEMS = ("N2", "N2R2")
ITEM_STAGE = {"N2": "E6-E2", "N2R2": "E6-R2"}
AM23_RULES_REL = "configs/qat_selection_rules_am23.json"
EVALUATOR_SCRIPT = "scripts/evaluate_model.py"          # run from the checkout, unchanged (as qat_epoch_eval.py does)
FQOFF_PINNED = ("src/eval/artifacts.py", "src/eval/stage_artifacts.py", "src/eval/model_loading.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--expect-telemetry-sha256", required=True)
    p.add_argument("--host-label", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--threads", type=int, default=None)
    return p


def run_item(run_dir: Path) -> dict:
    meta = run_dir.parent / G.RUN_META_NAME
    if not run_dir.parent.name.startswith(G.RUN_PREFIX) or not meta.is_file():
        raise G.GuardError("run_not_exploratory", f"{run_dir} is not the trainer folder of an x- item folder")
    doc = json.loads(meta.read_text(encoding="utf-8"))
    if doc.get("exploratory") is not True or doc.get("item") not in (*REGISTERED_ITEMS, "N5"):
        raise G.GuardError("run_item", f"{meta} records item {doc.get('item')!r}; block B takes C1, N1, N2, N2R2, N5, "
                                       "D")
    return doc


def run_stage(run_dir: Path) -> str | None:
    """The trainer's stage, from its telemetry's first row (the run_meta row)."""
    with open(run_dir / "qat_telemetry.jsonl", encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    return first.get("stage") if isinstance(first, dict) and first.get("event") == "run_meta" else None


def selection_rules_record(eval_dir: Path, stage: str) -> dict:
    """The rules file select_qat_epoch.py read for this stage (src/quant/qat_select.py rules_path_for_stage), its
    sha256 against the selection's rules_sha256: AM-23's file for E6-E2 and E6-R2."""
    from src.quant import qat_select as S
    path = Path(S.rules_path_for_stage(stage))
    rel = path.resolve().relative_to(REPO.resolve()).as_posix()
    sha = G.sha256_file(path)
    sel = json.loads((eval_dir / "qat_selection.json").read_text(encoding="utf-8"))
    return {"rules_file": rel, "rules_sha256": sha, "selection_rules_sha256": sel.get("rules_sha256"),
            "ok": rel == AM23_RULES_REL and sel.get("rules_sha256") == sha}


def fake_quant_off_argv(*, stage: str, run_dir: Path, winner: dict, selection: Path, selection_sha256: str,
                        telemetry_sha256: str, out_dir: Path, run_id: str) -> list[str]:
    """The pinned evaluator's command for the selected epoch's fake-quant-off VAL (precision qat_fakequant_off, lane
    L-FQOFF), as qat_epoch_eval.py's score step runs the evaluator: from the checkout, on the CPU, canvas protocol,
    batch size 1, artifact status provisional (the evaluator has no exploratory status) and no thread flag (the thread
    count is the environment's, which the summary records). The selection, its sha256 and the telemetry's bind the
    selected epoch's eNN.pt before any dataset or model exists; --expected-checkpoint-sha256 is the winner's."""
    return [sys.executable, "-B", EVALUATOR_SCRIPT, "--stage", stage, "--model-role", "student",
            "--precision", "qat_fakequant_off", "--split", "val", "--protocol", "canvas",
            "--checkpoint", str(Path(run_dir) / str(winner.get("checkpoint"))), "--qat-run-dir", str(run_dir),
            "--qat-selection", str(selection), "--qat-selection-sha256", selection_sha256,
            "--qat-telemetry-sha256", telemetry_sha256,
            "--expected-checkpoint-sha256", str(winner.get("checkpoint_sha256")),
            "--artifact-status", "provisional", "--device", "cpu", "--batch-size", "1",
            "--out-dir", str(out_dir), "--run-id", run_id]


def selected_epoch_vals(run_dir: Path, eval_dir: Path, out: Path, args, *, stage: str, runner=subprocess.run) -> dict:
    """Card fix 1 (DL-251; AM-24 item 10(b)): the selected epoch's fake-quant VAL, the value its epoch_end row
    records, and its fake-quant-off VAL, scored by the pinned scripts/evaluate_model.py, as DL-176 records them: each
    value with its file and sha256. Returns {"fake_quant", "fake_quant_off", "ok"}."""
    from src.quant import qat_artifacts as A
    sel_p = eval_dir / "qat_selection.json"
    sel = json.loads(sel_p.read_text(encoding="utf-8"))
    sel_sha = G.sha256_file(sel_p)
    win = sel.get("winner") or {}
    epoch = win.get("epoch")
    rec = A.read_run_record(run_dir)
    end = rec["ends"].get(epoch) or {}
    fq_v = end.get("fake_quant_val_all_class_miou")
    fq = {"epoch": epoch, "value": fq_v, "source": "the selected epoch's epoch_end row, fake_quant_val_all_class_miou",
          "telemetry": {"path": str(run_dir / "qat_telemetry.jsonl"), "sha256": rec["telemetry_sha256"]},
          "checkpoint_sha256": end.get("checkpoint_sha256")}
    fq_ok = isinstance(fq_v, float) and math.isfinite(fq_v)
    if type(epoch) is not int or end.get("checkpoint_sha256") != win.get("checkpoint_sha256"):
        return {"fake_quant": fq, "ok": False,
                "fake_quant_off": {"status": "failed", "message": f"the selection's winner (epoch {epoch!r}) is not an "
                                                                  "epoch_end row of the run"}}
    fo_dir = out / "fqoff" / f"e{epoch:02d}"
    run_id = f"{rec['run_id']}_fqoff_e{epoch:02d}_{str(win.get('checkpoint_sha256'))[:12]}"
    argv = fake_quant_off_argv(stage=stage, run_dir=run_dir, winner=win, selection=sel_p, selection_sha256=sel_sha,
                               telemetry_sha256=args.expect_telemetry_sha256, out_dir=fo_dir, run_id=run_id)
    fo_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = runner(argv, cwd=str(REPO), capture_output=True, text=True)
    row = {"step": "fake_quant_off", "argv": argv[1:], "python": argv[0], "exit": p.returncode,
           "seconds": time.time() - t0, "out_dir": str(fo_dir), "selection": {"path": str(sel_p), "sha256": sel_sha},
           "evaluator_status_note": "artifact_status provisional: the evaluator has no exploratory status"}
    summ = fo_dir / "summary.json"
    if p.returncode != 0 or not summ.is_file():
        lines = [ln for ln in (p.stderr or "").splitlines() + (p.stdout or "").splitlines() if ln.strip()]
        row.update({"status": "failed", "message": lines[-1][:500] if lines else None})
        return {"fake_quant": fq, "fake_quant_off": row, "ok": False}
    s = json.loads(summ.read_text(encoding="utf-8"))
    run, ds = s.get("run") or {}, s.get("dataset") or {}
    v = (s.get("dataset_level") or {}).get("all_class_miou")
    checks = {"split_val": ds.get("split") == "val",
              "precision_qat_fakequant_off": run.get("precision") == "qat_fakequant_off",
              "stage": run.get("stage") == stage,
              "checkpoint_is_the_winner": run.get("checkpoint_sha256") == win.get("checkpoint_sha256"),
              "rows_846": ds.get("expected_rows") == ds.get("actual_rows") == G.VAL_IMAGES,
              "value_in_0_1": isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= 1.0}
    row.update({"status": "scored" if all(checks.values()) else "refused", "value": v, "checks": checks,
                "summary": {"path": str(summ), "sha256": G.sha256_file(summ)},
                "torch_num_threads": (run.get("eval_runtime") or {}).get("torch_num_threads")})
    return {"fake_quant": fq, "fake_quant_off": row, "ok": all(checks.values()) and fq_ok}


def registered_block_b(run_dir: Path, eval_dir: Path, args) -> list[dict]:
    py = [sys.executable, "-B"]
    steps = [
        ("convert", py + ["scripts/qat_epoch_eval.py", "convert", "--run-dir", str(run_dir), "--eval-dir", str(eval_dir),
                          "--purpose", "record", "--host-label", args.host_label, "--expect-telemetry-sha256",
                          args.expect_telemetry_sha256]),
        ("score", py + ["scripts/qat_epoch_eval.py", "score", "--run-dir", str(run_dir), "--eval-dir", str(eval_dir),
                        "--purpose", "record", "--host-label", args.host_label]),
        ("select", py + ["scripts/select_qat_epoch.py", "--run-dir", str(run_dir), "--eval-dir", str(eval_dir),
                         "--expect-telemetry-sha256", args.expect_telemetry_sha256]),
        ("finalize", py + ["scripts/qat_epoch_eval.py", "finalize", "--run-dir", str(run_dir), "--eval-dir",
                           str(eval_dir)]),
    ]
    rows = []
    for name, argv in steps:
        t0 = time.time()
        p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
        result = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
        rows.append({"step": name, "argv": argv[1:], "exit": p.returncode, "result": result[-1] if result else None,
                     "seconds": time.time() - t0})
        print(f"[blockb] {name}: exit {p.returncode} {rows[-1]['result']}")
        if p.returncode != 0:
            sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
            break
    return rows


def n5_block_b(run_dir: Path, eval_dir: Path, args) -> dict:
    import torch
    from src.explore import fp32_head as H
    from src.explore.valscore import score_val
    from src.quant import qat_artifacts as A
    from src.quant.x86_latency import load_qat_sidecar
    rec = A.read_run_record(run_dir)
    A.require_complete(rec)
    if rec["telemetry_sha256"] != args.expect_telemetry_sha256:
        raise G.GuardError("telemetry_sha256_mismatch", f"the telemetry is {rec['telemetry_sha256']}")
    eval_dir.mkdir(parents=True)
    rows = {}
    for e in range(1, 16):
        ck, ck_sha = A.verified_checkpoint(rec, e)
        payload = load_qat_sidecar(ck)
        if payload.get("state_finite") is not True or rec["ends"][e].get("state_finite") is not True:
            rows[e] = {"epoch": e, "status": "excluded", "rule": "non-finite state (AM-21 item 2(a))"}
            continue
        conv, copy_rep = H.convert_state(payload["model_state_dict"])
        isl = H.converted_islands_report(conv, torch.zeros(1, 3, 512, 512))
        if not isl["ok"]:
            raise G.GuardError("islands_not_float", f"e{e:02d}: the converted FP32 islands report fails: {isl}")
        summ, _ = score_val(conv, device="cpu")
        sd = eval_dir / f"e{e:02d}_mixed_fp32head_state_dict.pt"
        torch.save({"model": conv.state_dict(), "precision_label": H.PRECISION_LABEL, "epoch": e,
                    "qat_checkpoint_sha256": ck_sha, "run_id": rec["run_id"]}, sd)
        rows[e] = {"epoch": e, "status": "scored", "all_class_miou": summ["all_class_miou"], "checkpoint_sha256": ck_sha,
                   "state_dict": sd.name, "state_dict_sha256": G.sha256_file(sd), "islands": isl,
                   "seconds": summ["seconds"]}
        print(f"[n5] e{e:02d}: VAL all-class mIoU {summ['all_class_miou']!r} (MIXED PRECISION)")
    scored = {e: r for e, r in rows.items() if r["status"] == "scored" and isinstance(r["all_class_miou"], float)}
    if not scored:
        raise G.GuardError("no_convertible_epoch", "no epoch was scored: nothing is selected")
    best = None
    for e in sorted(scored):                                  # strict >: a tie keeps the earlier epoch
        if best is None or scored[e]["all_class_miou"] > scored[best]["all_class_miou"]:
            best = e
    ck, ck_sha = A.verified_checkpoint(rec, best)
    conv, _ = H.convert_state(load_qat_sidecar(ck)["model_state_dict"])
    final = eval_dir / "final"
    final.mkdir()
    ident = {"format": "x_mixed_precision/1", "precision_label": H.PRECISION_LABEL, "never_int8": True,
             "stage": rec["run_meta"].get("stage"), "run_id": rec["run_id"], "epoch": best,
             "qat_checkpoint_sha256": ck_sha, "value": scored[best]["all_class_miou"], "amendment": G.AMENDMENT,
             "exploratory": True, "fp32_islands": list(H.FP32_ISLANDS), "dequant_points": list(H.DEQUANT_POINTS)}
    traced = torch.jit.trace(conv, torch.zeros(1, 3, 512, 512), check_trace=False)
    ts = final / f"e{best:02d}_mixed_fp32head_qnnpack.torchscript.pt"
    torch.jit.save(traced, str(ts), _extra_files={"x_mixed_precision.json": json.dumps(ident, sort_keys=True)})
    return {"epochs": [rows[e] for e in sorted(rows)], "winner": {**ident, "artifact": f"final/{ts.name}",
                                                                   "artifact_sha256": G.sha256_file(ts)},
            "rule": "highest converted VAL all-class mIoU, strict >, a tie keeps the earlier epoch; non-finite excluded"}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        G.refuse_split(args.split, allowed=G.VAL_ONLY)          # TEST first, by name, before anything is read
        import torch
        if args.threads is not None:
            torch.set_num_threads(int(args.threads))
        G.check_input_paths({"--run-dir": args.run_dir})
        run_dir = Path(args.run_dir).resolve()
        item_doc = run_item(run_dir)
        fqoff_item = item_doc["item"] in FQOFF_ITEMS
        st = G.guarded_start(item=item_doc["item"], run_id=args.run_id, out_dir=args.out_dir, split=args.split,
                             allowed_splits=G.VAL_ONLY, input_paths={"--run-dir": str(run_dir.parent)},
                             seeds=item_doc.get("seeds"), pinned_paths=("scripts/qat_epoch_eval.py",
                                                                        "scripts/select_qat_epoch.py",
                                                                        "src/quant/qat_artifacts.py",
                                                                        "src/quant/qat_select.py",
                                                                        "scripts/evaluate_model.py")
                             + (FQOFF_PINNED if fqoff_item else ()),
                             precision_label=item_doc.get("precision_label"),
                             recipe={"what": "block B of an exploratory QAT run", "run_dir": str(run_dir),
                                     "run_item": item_doc["item"], "run_x_run_meta": item_doc.get("run_id"),
                                     "expect_telemetry_sha256": args.expect_telemetry_sha256})
        out = st["out_dir"]
        eval_dir = out / "eval"
        tel_before = G.sha256_file(run_dir / "qat_telemetry.jsonl")
        if tel_before != args.expect_telemetry_sha256:
            raise G.GuardError("telemetry_sha256_mismatch", f"the run's telemetry is {tel_before}")
        stage = None
        if fqoff_item:                                    # N2 is E6-E2's run, N2R2 E6-R2's
            stage = run_stage(run_dir)
            if stage != ITEM_STAGE[item_doc["item"]]:
                raise G.GuardError("run_stage", f"{item_doc['item']}'s run is stage {ITEM_STAGE[item_doc['item']]}; "
                                                f"this run's run_meta records {stage!r}")
        if item_doc["item"] == "N5":
            res = n5_block_b(run_dir, eval_dir, args)
            ok = True
        else:
            steps = registered_block_b(run_dir, eval_dir, args)
            ok = all(s["exit"] == 0 for s in steps) and len(steps) == 4
            sel = eval_dir / "qat_selection.json"
            res = {"steps": steps, "selection": (json.loads(sel.read_text(encoding="utf-8")) if sel.is_file() else None),
                   "evaluator_status_note": ("the evaluator wrote artifact_status 'provisional' (src/eval/artifacts.py "
                                             "ARTIFACT_STATUSES has no exploratory value); this record is exploratory")}
            if fqoff_item and ok:                         # card fix 1 (DL-251; AM-24 item 10(b))
                res["selection_rules"] = selection_rules_record(eval_dir, stage)
                vals = selected_epoch_vals(run_dir, eval_dir, out, args, stage=stage)
                res["selected_epoch_fake_quant_val"] = vals["fake_quant"]
                res["selected_epoch_fake_quant_off_val"] = vals["fake_quant_off"]
                ok = ok and vals["ok"] and res["selection_rules"]["ok"]
                fo = vals["fake_quant_off"]
                what = repr(fo.get("value")) if fo.get("status") == "scored" else fo.get("message")
                print(f"[blockb] fake_quant_off: exit {fo.get('exit')} {fo.get('status')} {what}")
            elif fqoff_item:
                res["selected_epoch_fake_quant_off_val"] = {"status": "not run",
                                                            "reason": "a registered step did not finish"}
        tel_after = G.sha256_file(run_dir / "qat_telemetry.jsonl")
        doc = {"format": FORMAT, "exploratory": True, "amendment": G.AMENDMENT, "artifact_status": G.ARTIFACT_STATUS,
               "split": "val", "item": item_doc["item"], "run_dir": str(run_dir), "telemetry_sha256": tel_before,
               "telemetry_unchanged": tel_after == tel_before, "precision_label": item_doc.get("precision_label"),
               "claim_line": item_doc.get("claim_line"), **res}
        G.write_exclusive(out / "x_blockb.json", G.json_bytes(doc))
        if not ok or tel_after != tel_before:
            fo_txt = ""
            if fqoff_item:
                fo = res.get("selected_epoch_fake_quant_off_val") or {}
                why = fo.get("message") or fo.get("reason")
                fo_txt = f" (fake-quant-off VAL: {fo.get('status')}{', ' + why if why else ''})"
                rules = res.get("selection_rules")
                if rules is not None and not rules["ok"]:
                    fo_txt += f" (selection rules {rules['rules_file']}, not {AM23_RULES_REL} as selected)"
            print("RESULT: X BLOCK B INCOMPLETE -- see x_blockb.json" + fo_txt)
            return G.EXIT_ABORTED
        w = res.get("winner") or ((res.get("selection") or {}).get("winner") or {})
        vals_txt = ""
        if fqoff_item:
            vals_txt = (f"; selected epoch fake-quant VAL {res['selected_epoch_fake_quant_val']['value']!r}, "
                        f"fake-quant-off VAL {res['selected_epoch_fake_quant_off_val']['value']!r}")
        print(f"RESULT: X BLOCK B DONE ({item_doc['item']}): epoch {w.get('epoch')}, VAL {w.get('value')!r}" + vals_txt
              + (" -- MIXED PRECISION, not INT8" if item_doc["item"] == "N5" else ""))
        return G.EXIT_OK
    except G.GuardError as e:
        print(f"RESULT: REFUSED {e}")
        return G.EXIT_REFUSED
    except Exception as e:                                   # noqa: BLE001
        print(f"RESULT: ERROR [{type(e).__name__}] {e}")
        return G.EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
