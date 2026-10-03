#!/usr/bin/env python3
"""Statistics driver command line (lane L-STATS-OFFICIAL).

  --mode official  --inputs LIST.json --out-dir DIR --run-id ID --confirm-official-test-analysis
                   --synthetic-inputs [--repo-root DIR] [--B 10000] [--jobs N]
      The 37 inputs of STATISTICAL_ANALYSIS_CONTRACT section 12.4.6, named in an input-list file
      (plantseg-stats-inputs/1.0.0), under the OFFICIAL policy and the official namespace, written as
      a section 12 artifact at DIR/ID. Door order (P13): the confirmation flag; the binding rule (while
      the TEST-manifest binding is unbound, only inputs declared synthetic, so no official artifact
      can be written); the out-dir pre-check; the pinned stack; then reads. --repo-root (accepted only
      with --synthetic-inputs) resolves input paths only; the contract, the corruption protocol and
      the code provenance always come from this repository (P16).
  --mode smoke     --inputs LIST.json --out-dir DIR --run-id ID [--repo-root DIR] [--B N] [--jobs N]
      Smoke-status inputs only, never a PlantSeg TEST input (P15); NONOFFICIAL_SMOKE; nonofficial.

Every mode refuses a non-canvas input by name (EVALUATION_CONTRACT section 11(a)).

Exit codes (P18, A3): 0 OK; 1 a failed rehearsal structural criterion (rehearsal mode only); 2 a
refusal, with nothing written; 4 an unexpected error -- an import failure, a worker failure, or an
artifact that was written but fails its re-verification (named). No test hook is exposed (P11).

Run:  python -B scripts/run_stats.py --mode smoke --inputs inputs.json --out-dir OUT --run-id smk-0001
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXIT_OK, EXIT_CRITERIA, EXIT_REFUSED, EXIT_ERROR = 0, 1, 2, 4


class Refused(Exception):
    """A command line the driver refuses before doing anything."""


def parse(argv):
    ap = argparse.ArgumentParser(description="PlantSeg statistics driver (section 12 artifact)")
    ap.add_argument("--mode", required=True, choices=("official", "smoke"))
    ap.add_argument("--inputs", type=Path, help="input-list file (plantseg-stats-inputs/1.0.0)")
    ap.add_argument("--out-dir", type=Path, help="parent directory of the artifact directory")
    ap.add_argument("--run-id", help="the artifact directory name, ^[a-z0-9][a-z0-9._-]{0,63}$")
    ap.add_argument("--confirm-official-test-analysis", action="store_true")
    ap.add_argument("--synthetic-inputs", action="store_true",
                    help="declare the inputs synthetic (official mode, while the binding is unbound)")
    ap.add_argument("--repo-root", type=Path, help="resolve input paths here (synthetic or smoke)")
    ap.add_argument("--B", type=int, default=10_000, help="bootstrap replicates (official: 10000)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes, whole tasks each")
    return ap.parse_args(argv)


def require(args, *names) -> None:
    missing = [n for n in names if getattr(args, n) is None]
    if missing:
        raise Refused(f"--mode {args.mode} requires " + ", ".join(
            "--" + n.replace("_", "-") for n in missing))


def validate_counts(args) -> None:
    for name in ("B", "jobs"):
        v = getattr(args, name)
        if v < 1:
            raise Refused(f"--{name} must be a positive integer, got {v}")


def report(res) -> None:
    fam = res.family
    tasks = fam["bootstrap_tasks"]
    per_image = sorted({t["jackknife_count"] for t in tasks
                        if not t["task_id"].endswith("__dataset_miou_delta")})
    dataset = sorted({t["jackknife_count"] for t in tasks
                      if t["task_id"].endswith("__dataset_miou_delta")})
    print(f"mode: {res.mode}")
    print(f"inputs resolved under: {res.input_root}")
    print(f"AM-5 k (zero-disease images excluded from per-image tasks): {res.am5_excluded_count}")
    print(f"jackknife counts: per-image {per_image}, dataset-level {dataset}; bootstrap replicates "
          f"{sorted({t['bootstrap_replicates'] for t in tasks})}")
    print(f"artifact_status: {fam['artifact_status']}; warnings: {fam['warnings']}")
    if res.mode == "official":
        print("unmet official conditions:")
        for name, detail in res.unmet_official:
            print(f"  - {name}: {detail}")
    print(f"integrity checks re-established after the rename: {len(res.established)}/13")
    print(f"ARTIFACT: {res.artifact}")
    print(f"UNMET_OFFICIAL: {json.dumps([n for n, _ in res.unmet_official])}")
    print(f"WALL_SECONDS: {res.seconds:.1f}")


def main(argv=None) -> int:
    try:
        args = parse(argv)
    except SystemExit as e:                                   # argparse: usage error or --help
        return EXIT_REFUSED if e.code else EXIT_OK
    try:                                                      # P18: imports inside the wrapper
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from src.stats import artifact as A
        from src.stats import driver as D
    except Exception:                                         # noqa: BLE001
        traceback.print_exc()
        print("ERROR: the statistics modules could not be imported", file=sys.stderr)
        return EXIT_ERROR
    refusals = (Refused, D.DriverRefusal, A.ArtifactError)
    try:
        if args.mode == "official":
            D.official_doors(confirm=args.confirm_official_test_analysis,      # doors 1 and 2
                             synthetic=args.synthetic_inputs)
            require(args, "inputs", "out_dir", "run_id")
            validate_counts(args)
            res = D.run_official(inputs_list=args.inputs, out_dir=args.out_dir,
                                 run_id=args.run_id, confirm=args.confirm_official_test_analysis,
                                 synthetic=args.synthetic_inputs, repo_root=args.repo_root,
                                 B=args.B, jobs=args.jobs)
        else:
            if args.confirm_official_test_analysis or args.synthetic_inputs:
                raise Refused("--confirm-official-test-analysis and --synthetic-inputs belong to "
                              "official mode")
            require(args, "inputs", "out_dir", "run_id")
            validate_counts(args)
            res = D.run_smoke(inputs_list=args.inputs, out_dir=args.out_dir, run_id=args.run_id,
                              repo_root=args.repo_root, B=args.B, jobs=args.jobs)
    except refusals as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except (D.PostWriteVerifyFailed, D.WorkerFailure) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:                                         # noqa: BLE001
        traceback.print_exc()
        print("ERROR: unexpected failure", file=sys.stderr)
        return EXIT_ERROR
    report(res)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
