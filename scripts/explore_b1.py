#!/usr/bin/env python3
"""B1 (lane L-AM24, build item E2): the D-1 conversion-gap split of one QAT model's selected epoch, on VAL.

    python -I scripts/explore_b1.py --qat-run-dir <QAT run dir> --selection <qat_selection.json>
        --selection-sha256 <sha> --artifact <converted/eNN_int8_qnnpack.torchscript.pt> --stage E6-E2
        --run-id x-b1-<model> --out-dir <root>/x-b1-<model> [--device cpu|cuda] [--threads 8] [--passes ...]

The E0 guard runs first (VAL only; the data root's TRAIN/VAL-only check; x- run ID; the output outside the
repository, every official folder and every input). The QAT run, the selection and the artifact are read only: the
artifact's and the selected checkpoint's sha256 are checked before they are loaded and again after the last pass.
The artifact must be the selection's winner (its artifact_of_record sha256). Each pass's summary is appended to
b1_passes.jsonl as it ends; b1_split.json and b1_split.txt follow the last pass. TF32 passes run only with
--device cuda. Exit codes: 0 done, 2 refused, 3 an input changed during the run, 4 error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.explore import b1_split as B                                # noqa: E402
from src.explore import guard as G                                   # noqa: E402

PINNED = ("src/quant/qat.py", "src/quant/qat_artifacts.py", "src/quant/prepare.py", "src/quant/qconfig.py",
          "src/quant/x86_latency.py", "src/models/student.py", "src/eval/evaluate.py", "src/eval/adapters.py",
          "src/eval/eval_runtime.py", "src/eval/model_loading.py", "src/eval/metrics.py", "src/data/transforms.py",
          "src/data/dataset.py", "src/data/isolation.py", "configs/data.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--qat-run-dir", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--selection-sha256", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--stage", required=True, help="the artifact's stage, e.g. E6-E2 (its embedded identity)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--passes", nargs="+", default=None, help=f"a subset of {list(B.PASS_ORDER)}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    try:
        if args.threads is not None:
            torch.set_num_threads(int(args.threads))
        plan = B.pass_plan(args.device, args.passes)
        st = G.guarded_start(
            item="B1", run_id=args.run_id, out_dir=args.out_dir, split=args.split, allowed_splits=G.VAL_ONLY,
            input_paths={"--qat-run-dir": args.qat_run_dir, "--selection": args.selection,
                         "--artifact": args.artifact},
            seeds="the QAT run's", pinned_paths=PINNED,
            recipe={"what": "D-1 conversion-gap split (audit §2 B1 items (i)-(v))", "passes": plan,
                    "device": args.device, "threads": torch.get_num_threads(), "stage": args.stage,
                    "selection_sha256": args.selection_sha256, "scorer": "src/explore/valscore.py (canvas, VAL 846)",
                    "site_definition": "src/explore/sites.py (derived from the code; D-1 names none in the repo)"})
        out, inp = st["out_dir"], st["inputs"]
        sel = B.read_selection(inp["--selection"], args.selection_sha256)
        win = sel["doc"]["winner"]
        art_sha = ((win.get("artifact_of_record") or {}).get("sha256"))
        fq, fq_meta = B.load_selected_fake_quant(inp["--qat-run-dir"], win["epoch"])
        if fq_meta["checkpoint_sha256"] != win.get("checkpoint_sha256"):
            raise B.B1Error("selection_checkpoint_mismatch", f"the run's e{win['epoch']:02d}.pt is "
                                                            f"{fq_meta['checkpoint_sha256']}, the selection names "
                                                            f"{win.get('checkpoint_sha256')}")
        conv, conv_meta = B.load_converted(inp["--artifact"], stage=args.stage, expect_sha256=art_sha)
        fq = fq.to(args.device)
        from src.explore.valscore import score_val
        log = out / "b1_passes.jsonl"

        def on_pass(name, summ):
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(summ, sort_keys=True) + "\n")

        scores = B.run_passes(fq, conv, lambda m, d: score_val(m, device=d)[0], device=args.device, passes=plan,
                              on_pass=on_pass)
        after = {"checkpoint": G.sha256_file(fq_meta["checkpoint"]), "artifact": G.sha256_file(inp["--artifact"]),
                 "selection": G.sha256_file(inp["--selection"])}
        changed = [k for k, v in (("checkpoint", fq_meta["checkpoint_sha256"]), ("artifact", conv_meta["sha256"]),
                                  ("selection", sel["sha256"])) if after[k] != v]
        rec = B.reconcile(scores, converted_of_record=win.get("value"), device=args.device)
        doc = {"format": B.FORMAT, "run_id": args.run_id, "exploratory": True, "amendment": G.AMENDMENT,
               "split": "val", "item": "B1", "selection": {k: sel[k] for k in ("path", "sha256")},
               "winner": win, "fake_quant": fq_meta, "converted": conv_meta, "sites": B.describe_sites(fq.cpu()),
               "scores": scores, "reconciliation": rec, "inputs_unchanged": not changed, "changed": changed,
               "item_iv_hardware_match": (str(args.device).startswith("cuda")
                                          and str(fq_meta.get("recorded_device") or "").startswith("cuda")),
               "claim_line": st["run_meta"]["claim_line"]}
        G.write_exclusive(out / "b1_split.json", G.json_bytes(doc))
        text = B.table_text(scores, rec)
        G.write_exclusive(out / "b1_split.txt", (text + "\n").encode("utf-8"))
        print(text)
        if changed:
            print(f"RESULT: B1 INPUT CHANGED {changed}: the record is evidence only")
            return G.EXIT_ABORTED
        print(f"RESULT: B1 DONE ({len(scores)} passes) -> {out / 'b1_split.json'}")
        return G.EXIT_OK
    except (G.GuardError, B.B1Error) as e:
        print(f"RESULT: REFUSED {e}")
        return G.EXIT_REFUSED
    except Exception as e:                                   # noqa: BLE001
        print(f"RESULT: ERROR [{type(e).__name__}] {e}")
        return G.EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
