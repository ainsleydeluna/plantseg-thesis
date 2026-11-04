#!/usr/bin/env python3
"""E5 (lane L-AM24, addendum L-AM24-ADD-1): KL-3 parity of one x86 latency copy against its QNNPACK artifact, on VAL.

    python -B -I scripts/explore_kl3_parity.py --slot E6-E2:s42:sel --stage E6-E2
        --qnnpack-artifact <eval dir>/converted/eNN_int8_qnnpack.torchscript.pt --qnnpack-sha256 <sha>
        --x86-copy <eval dir>/final/eNN_int8_x86_latency.torchscript.pt --x86-sha256 <sha>
        --test-results-dl DL-<n> --test-results-commit <hex> --run-id x-kl3-<model> --out-dir <root>/<run-id>
        [--threads 8]

Order: the split (TEST refused first, by name); AM-24 clause 3 (the decision-log row of the committed TEST results,
RECORDED at HEAD, naming TEST and the commit, the commit in HEAD's history); the E0 guard (VAL, the data root's
TRAIN/VAL-only check, x- run ID, the output outside the repository, every official folder and both artifacts'
folders); with --slot, both sha256 must be the TEST-day manifest's for that slot. Both artifacts are read only:
hashed before the load and after the last pass. Writes kl3_parity.json and kl3_parity.txt (exploratory) into the
item directory. Exit codes: 0 done, 2 refused, 3 an artifact changed during the run, 4 error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.explore import guard as G                                   # noqa: E402
from src.explore import kl3_parity as P                              # noqa: E402

PINNED = ("src/eval/model_loading.py", "src/eval/evaluate.py", "src/eval/adapters.py", "src/eval/eval_runtime.py",
          "src/eval/metrics.py", "src/quant/qat_artifacts.py", "src/quant/x86_latency.py", "src/quant/qconfig.py",
          "src/data/transforms.py", "src/data/dataset.py", "src/data/isolation.py", "configs/data.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--slot", default=None)
    p.add_argument("--stage", required=True)
    p.add_argument("--qnnpack-artifact", required=True)
    p.add_argument("--qnnpack-sha256", required=True)
    p.add_argument("--x86-copy", required=True)
    p.add_argument("--x86-sha256", required=True)
    p.add_argument("--test-results-dl", required=True)
    p.add_argument("--test-results-commit", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--threads", type=int, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        G.refuse_split(args.split, allowed=G.VAL_ONLY)
        clause3 = P.clause3_gate(args.test_results_dl, args.test_results_commit)
        import torch
        if args.threads is not None:
            torch.set_num_threads(int(args.threads))
        st = G.guarded_start(item="KL3", run_id=args.run_id, out_dir=args.out_dir, split=args.split,
                             allowed_splits=G.VAL_ONLY,
                             input_paths={"--qnnpack-artifact": args.qnnpack_artifact, "--x86-copy": args.x86_copy},
                             seeds="the model's", pinned_paths=PINNED,
                             recipe={"what": "KL-3 parity (DL-181): the x86 latency copy against its QNNPACK artifact",
                                     "slot": args.slot, "stage": args.stage, "scorer": "src/explore/valscore.py",
                                     "threads": torch.get_num_threads(), "clause3": clause3})
        if args.slot is not None:
            m = P.manifest_pair(args.slot)
            if (m["artifact_sha256"], m["x86_copy_sha256"]) != (args.qnnpack_sha256, args.x86_sha256):
                raise P.KL3Error("manifest_pair_mismatch", f"slot {args.slot}'s pair is {m['artifact_sha256']} / "
                                                           f"{m['x86_copy_sha256']}")
        inp = st["inputs"]
        pair = P.load_pair(inp["--qnnpack-artifact"], inp["--x86-copy"], stage=args.stage,
                           expect_q_sha256=args.qnnpack_sha256, expect_x_sha256=args.x86_sha256)
        from src.explore.valscore import score_val
        res = P.parity(pair, score_q=lambda mdl: score_val(mdl, device="cpu")[0],
                       score_x_with_forward=lambda mdl, fwd: score_val(mdl, device="cpu", forward=fwd)[0])
        after = {"qnnpack": G.sha256_file(inp["--qnnpack-artifact"]), "x86": G.sha256_file(inp["--x86-copy"])}
        unchanged = after == pair["sha256_before"]
        out = st["out_dir"]
        doc = {"format": P.FORMAT, "exploratory": True, "amendment": G.AMENDMENT, "item": "KL3", "split": "val",
               "artifact_status": G.ARTIFACT_STATUS, "slot": args.slot, "stage": args.stage,
               "identities": pair["identities"], "sha256_before": pair["sha256_before"], "sha256_after": after,
               "read_only_unchanged": unchanged, "clause3": clause3, **res,
               "claim_line": st["run_meta"]["claim_line"],
               "note": "descriptive parity of a latency proxy; the QNNPACK artifact stays the accuracy artifact"}
        G.write_exclusive(out / "kl3_parity.json", G.json_bytes(doc))
        text = (f"{args.slot or args.stage}: QNNPACK {res['qnnpack'].get('all_class_miou')!r}  x86 "
                f"{res['x86'].get('all_class_miou')!r}  x86 - QNNPACK {res['x86_minus_qnnpack_pp']!r} pp  argmax "
                f"agreement {res['agreement']['argmax_agreement']!r}  max |logit diff| "
                f"{res['agreement']['max_abs_logit_diff']!r}")
        G.write_exclusive(out / "kl3_parity.txt", (text + "\n").encode("utf-8"))
        print(text)
        if not unchanged:
            print("RESULT: KL3 ARTIFACT CHANGED during the run: the record is evidence only")
            return G.EXIT_ABORTED
        print(f"RESULT: KL3 PARITY DONE -> {out / 'kl3_parity.json'}")
        return G.EXIT_OK
    except (G.GuardError, P.KL3Error) as e:
        print(f"RESULT: REFUSED {e}. Nothing was written.")
        return G.EXIT_REFUSED
    except Exception as e:                                   # noqa: BLE001
        print(f"RESULT: ERROR [{type(e).__name__}] {e}")
        return G.EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
