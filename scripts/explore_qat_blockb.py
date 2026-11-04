#!/usr/bin/env python3
"""Block B of an exploratory QAT run (lane L-AM24; C1, N1, N2, N5, D): 15 converted epochs scored on the CPU, AM-23's selection
rules, finalize.

    python -B -I scripts/explore_qat_blockb.py --run-dir <item dir>/<run-id> --expect-telemetry-sha256 <sha>
        --host-label TEXT --run-id x-...-bb --out-dir <root>/x-...-bb [--threads 8]

The guard runs first (VAL, the data root's TRAIN/VAL-only check, x- run ID, the output outside the repository,
every official folder and the run directory). The run is identified from its x_run_meta.json (the item folder that
holds the trainer's run directory): it must be an exploratory C1, N1, N2, N5 or D run.

  C1, N1, N2, D: the existing scripts, unchanged, as subprocesses from the checkout, into <out-dir>/eval:
                 qat_epoch_eval.py convert --purpose record, score (the evaluator at --artifact-status provisional:
                 the evaluator has no exploratory status), select_qat_epoch.py (the rules file of the run's stage:
                 AM-23's for E6-E2), qat_epoch_eval.py finalize. They accept the run because its telemetry is the
                 pinned trainer's own (stage E6-E2 or E6, mode real); the x- run ID travels in every artifact name.
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
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.explore import guard as G                                   # noqa: E402

REGISTERED_ITEMS = ("C1", "N1", "N2", "D")
FORMAT = "x_qat_blockb/1"


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
        raise G.GuardError("run_item", f"{meta} records item {doc.get('item')!r}; block B takes C1, N1, N2, N5, D")
    return doc


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
        st = G.guarded_start(item=item_doc["item"], run_id=args.run_id, out_dir=args.out_dir, split=args.split,
                             allowed_splits=G.VAL_ONLY, input_paths={"--run-dir": str(run_dir.parent)},
                             seeds=item_doc.get("seeds"), pinned_paths=("scripts/qat_epoch_eval.py",
                                                                        "scripts/select_qat_epoch.py",
                                                                        "src/quant/qat_artifacts.py",
                                                                        "src/quant/qat_select.py",
                                                                        "scripts/evaluate_model.py"),
                             precision_label=item_doc.get("precision_label"),
                             recipe={"what": "block B of an exploratory QAT run", "run_dir": str(run_dir),
                                     "run_item": item_doc["item"], "run_x_run_meta": item_doc.get("run_id"),
                                     "expect_telemetry_sha256": args.expect_telemetry_sha256})
        out = st["out_dir"]
        eval_dir = out / "eval"
        tel_before = G.sha256_file(run_dir / "qat_telemetry.jsonl")
        if tel_before != args.expect_telemetry_sha256:
            raise G.GuardError("telemetry_sha256_mismatch", f"the run's telemetry is {tel_before}")
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
        tel_after = G.sha256_file(run_dir / "qat_telemetry.jsonl")
        doc = {"format": FORMAT, "exploratory": True, "amendment": G.AMENDMENT, "artifact_status": G.ARTIFACT_STATUS,
               "split": "val", "item": item_doc["item"], "run_dir": str(run_dir), "telemetry_sha256": tel_before,
               "telemetry_unchanged": tel_after == tel_before, "precision_label": item_doc.get("precision_label"),
               "claim_line": item_doc.get("claim_line"), **res}
        G.write_exclusive(out / "x_blockb.json", G.json_bytes(doc))
        if not ok or tel_after != tel_before:
            print("RESULT: X BLOCK B INCOMPLETE -- see x_blockb.json")
            return G.EXIT_ABORTED
        w = res.get("winner") or ((res.get("selection") or {}).get("winner") or {})
        print(f"RESULT: X BLOCK B DONE ({item_doc['item']}): epoch {w.get('epoch')}, VAL {w.get('value')!r}"
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
