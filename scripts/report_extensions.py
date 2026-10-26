#!/usr/bin/env python3
"""The LREP extension report (lane LREP, B-3; B-5): seed stability and the per-seed tables, the
rare-class strata TEST tables, the default candidates, GT-present for every model of record and every
other clean canvas record, the fq_off table, the MDE entry, and B-5's descriptive parts (AM-18's arm
tables, the duplicate-free TEST, AM-23's rows), as a separate document beside report.py's report (K1;
src/stats/report_extensions.py). B-5's intervals use bootstrap.PRODUCTION_B replicates.

    python -B scripts/report_extensions.py --stats-artifact DIR --inventory FILE --out-dir DIR
        [--inventory-sha256 HEX] [--strata FILE] [--mde-entry FILE] [--parts seed,strata,...]
        [--synthetic-inputs [--repo-root DIR]]

  --stats-artifact  a statistics artifact (family.json, bootstrap.npz, MANIFEST.sha256), verified with
                    its 37 inputs and the contract (all thirteen checks)
  --inventory       TMAN's inventory export tday-inventory/1 (IF§8, IF§11 v2); the path arrives here,
                    LREP never builds it (IF§11.6). Until TMAN merges it is read by LREP's stub, which
                    is refused under Policy.OFFICIAL
  --repo-root       resolves the inputs' repository-relative paths; accepted only with
                    --synthetic-inputs (P16). Otherwise this code's repository
  --strata          the strata file (default: the committed reports/strata/train_strata_v1.json)
  --mde-entry       the MDE entry (default: the one committed reports/derived/mde_entry_*.json, if
                    exactly one is tracked)
  --parts           a comma-separated subset of seed,strata,defaults,am6,fqoff,mde,arms,dedup,quant
                    (default all)
  --out-dir         a new directory, whose parent exists, outside the repository and every input
                    directory

Exit codes: 0 written (and --help, which writes nothing); 2 refused by name, or an argparse usage error
without a RESULT line (nothing written); 4 an unexpected failure (nothing written).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_REFUSED, EXIT_ERROR = 0, 2, 4


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LREP extension report (lane LREP, B-3; B-5).")
    p.add_argument("--stats-artifact", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--inventory-sha256", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--strata", type=Path, default=REPO / "reports" / "strata" / "train_strata_v1.json")
    p.add_argument("--mde-entry", type=Path, default=None)
    p.add_argument("--parts", default=None)
    p.add_argument("--synthetic-inputs", action="store_true")
    p.add_argument("--repo-root", type=Path, default=None)
    return p


def default_mde_entry():
    from src.stats.report import tracked_mde_entries
    tracked = tracked_mde_entries()
    return REPO / tracked[0] if tracked is not None and len(tracked) == 1 else None


def main(argv=None) -> int:
    try:
        args = parser().parse_args(argv)
    except SystemExit as e:                                   # argparse: usage errors exit 2
        return EXIT_OK if e.code == 0 else EXIT_REFUSED
    try:
        from src.stats import heldout_inputs as H
        from src.stats import report_extensions as RX
    except Exception as e:                                    # noqa: BLE001 -- exit 4, named
        print(f"ERROR: import failed: {type(e).__name__}: {e}", file=sys.stderr)
        print("RESULT: ERROR (import)")
        return EXIT_ERROR
    try:
        mde = args.mde_entry if args.mde_entry is not None else default_mde_entry()
        path, doc = RX.run(stats_artifact=args.stats_artifact, inventory=args.inventory,
                           out_dir=args.out_dir, repo_root=args.repo_root,
                           synthetic=args.synthetic_inputs, strata=args.strata, mde_entry=mde,
                           parts=args.parts, expect_inventory_sha256=args.inventory_sha256)
    except H.LrepRefusal as e:
        print(f"REFUSED {e.code}: {e.detail}", file=sys.stderr)
        print(f"RESULT: REFUSED ({e.code}); nothing written")
        return EXIT_REFUSED
    except Exception as e:                                    # noqa: BLE001 -- exit 4, named
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print("RESULT: ERROR; nothing written")
        return EXIT_ERROR
    print(f"profile: {doc['profile']}; parts: {[p['part'] for p in doc['parts']]}")
    for name, detail in doc["unmet_official_conditions"]:
        print(f"  unmet: {name}: {detail}")
    print(f"RESULT: REPORT EXTENSIONS WRITTEN ({doc['profile']}) {path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
