#!/usr/bin/env python3
"""The descriptive robustness outputs: B-4's CLI (L-CORR; plan section 4; src/eval/robustness_report.py).

  python -B scripts/report_robustness.py --inventory <export.json> --artifacts-root <dir>
      --policy {official,rehearsal,smoke} --split {test,val} --out <new dir outside the repo>
      [--expected-cache-sha256 <64 hex>]

The models come only from the inventory export (`tday-inventory/1`, read by
scripts/run_corruption_grid.load_inventory); an official read also checks the export's TL section 9
scopes (run_corruption_grid.check_scopes). The policy fixes the split (official: test; rehearsal and
smoke: val), and the request and the out-dir are checked before any input is read. The artifacts are
read where they are; nothing is copied into the repository (LE14).

Exit codes: 0 written; 2 refused, by code or by a frozen module's analysis error (nothing written).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.run_corruption_grid as RG  # noqa: E402
import src.eval.robustness_report as RR  # noqa: E402
from src.eval.robustness import RobustnessError  # noqa: E402

EXIT_DONE, EXIT_REFUSED = 0, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--inventory", required=True, help="the tday-inventory/1 export")
    p.add_argument("--artifacts-root", required=True,
                   help="the folder holding <prefix>_clean_<file_id> and "
                        "<prefix>_corr_<file_id>_<cell id> with their cell records")
    p.add_argument("--policy", required=True, choices=RR.POLICIES)
    p.add_argument("--split", required=True, choices=("val", "test"))
    p.add_argument("--out", required=True, help="a new folder outside the repository")
    p.add_argument("--expected-cache-sha256", default=None,
                   help="the recorded canonical cache checksum (required by the official policy)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from src.stats.bootstrap import BootstrapError
    try:
        RR.check_request(args.policy, args.split)
        RR.check_out_dir(args.out)
        export, inventory_sha = RG.load_inventory(args.inventory)
        if args.policy == "official":
            RG.check_scopes(export)
        doc = RR.build_report(export, args.artifacts_root, split=args.split, policy=args.policy,
                              expected_cache=args.expected_cache_sha256,
                              inventory_sha256=inventory_sha)
        out = RR.write_outputs(args.out, doc)
    except (RR.ReportRefusal, RG.GridRefusal) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except (BootstrapError, RobustnessError) as e:
        print(f"refused by an analysis error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"robustness outputs: {out}")
    print(f"status: {doc['status']}; models: {len(doc['models'])}; cache "
          f"{doc['cache']['canonical_manifest_sha256']}")
    if doc["banner"]:
        print(doc["banner"])
    return EXIT_DONE


if __name__ == "__main__":
    raise SystemExit(main())
