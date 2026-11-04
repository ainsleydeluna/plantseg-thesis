#!/usr/bin/env python3
"""N4 (lane L-AM24, build item E1): free decompositions from existing records. CPU; reads records only.

    terms      --run-dir <KD run folder> --run-id x-... --out-dir <root>/<run-id> [--stage-key e3] [--window 10000]
               per-term loss magnitudes over the last 10,000 iterations of the run's telemetry and the A1'' alpha
               (alpha = lambda * mean(L_logitKD) / mean(L_feat)), printed with its inputs
    qat-table  --spec <models.json> --run-id x-... --out-dir <root>/<run-id>
               parent, fake-quant (selected epoch), converted and fake-quant-off VAL per QAT model, one table;
               models.json is a list of {"label", "parent_best_json", "qat_run_dir", "selection", "fqoff_summary"}

Both run the E0 guard first (records-only: no data root is opened) and write x_run_meta.json, then their outputs,
into the item directory. Exit codes: 0 done, 2 refused, 4 error.
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

from src.explore import decompose as D                               # noqa: E402
from src.explore import guard as G                                   # noqa: E402

PINNED = ("src/training/train_distill.py", "src/quant/qat.py", "src/quant/qat_select.py",
          "src/quant/qat_artifacts.py", "scripts/evaluate_model.py", "src/eval/evaluate.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("terms", "qat-table"):
        s = sub.add_parser(name)
        s.add_argument("--run-id", required=True)
        s.add_argument("--out-dir", required=True)
        s.add_argument("--split", default="val")
        if name == "terms":
            s.add_argument("--run-dir", required=True)
            s.add_argument("--stage-key", default="e3")
            s.add_argument("--window", type=int, default=D.WINDOW)
        else:
            s.add_argument("--spec", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "terms":
            st = G.guarded_start(item="N4", run_id=args.run_id, out_dir=args.out_dir, split=args.split,
                                 allowed_splits=G.VAL_ONLY, input_paths={"--run-dir": args.run_dir},
                                 records_only=True, seeds="as the run's", pinned_paths=PINNED,
                                 recipe={"part": "a", "what": "per-term loss magnitudes and the A1'' alpha",
                                         "run_dir": args.run_dir, "stage_key": args.stage_key,
                                         "window": args.window, "formula": "alpha = lambda * mean(L_logitKD) / "
                                                                           "mean(L_feat)"})
            res = D.term_magnitudes(st["inputs"]["--run-dir"], stage_key=args.stage_key, last=args.window)
            out = st["out_dir"]
            G.write_exclusive(out / "n4_terms.json", G.json_bytes(res))
            text = D.terms_text(res)
            G.write_exclusive(out / "n4_terms.txt", (text + "\n").encode("utf-8"))
            print(text)
            ok = res["alpha_a1pp"]["status"] == "ok"
            print(f"RESULT: N4 TERMS {'DONE' if ok else 'DONE, ALPHA NOT FORMED'} -> {out / 'n4_terms.json'}")
            return G.EXIT_OK
        spec_path = args.spec
        st0 = G.check_input_paths({"--spec": spec_path})
        specs = json.loads(Path(st0["--spec"]).read_text(encoding="utf-8"))
        paths = {f"{i}.{k}": v for i, s in enumerate(specs) for k, v in s.items() if k != "label"}
        st = G.guarded_start(item="N4", run_id=args.run_id, out_dir=args.out_dir, split=args.split,
                             allowed_splits=G.VAL_ONLY, input_paths={"--spec": spec_path, **paths},
                             records_only=True, seeds="as each model's", pinned_paths=PINNED,
                             recipe={"part": "b", "what": "parent / fake-quant / converted / fake-quant-off VAL",
                                     "spec": specs, "spec_sha256": G.sha256_file(st0["--spec"])})
        tab = D.qat_table(specs)
        out = st["out_dir"]
        G.write_exclusive(out / "n4_qat_table.json", G.json_bytes(tab))
        text = D.table_text(tab)
        G.write_exclusive(out / "n4_qat_table.txt", (text + "\n").encode("utf-8"))
        print(text)
        print(f"RESULT: N4 QAT TABLE DONE ({len(tab['rows'])} models) -> {out / 'n4_qat_table.json'}")
        return G.EXIT_OK
    except (G.GuardError, D.DecomposeError) as e:
        print(f"RESULT: REFUSED {e}. Nothing was written after the refusal.")
        return G.EXIT_REFUSED
    except Exception as e:                                   # noqa: BLE001 -- reported, not swallowed
        print(f"RESULT: ERROR [{type(e).__name__}] {e}")
        return G.EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
