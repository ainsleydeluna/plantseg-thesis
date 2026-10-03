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
  --mode rehearsal --s42 DIR --s43 DIR --s44 DIR --out-root DIR [--B 10000] [--expect-k-val K]
                   [--mde-entry FILE] [--jobs N]
      The AM-17 item 9 dress rehearsal on the E1 seed 42/43/44 VAL artifacts (src/stats/rehearsal.py):
      nonofficial outputs under OUT-ROOT/<UTC>/, the out-root outside the repository; no section 12
      artifact. Exit 1 when a structural criterion fails (the outputs are kept).
  --mode report    --artifact DIR --report-out DIR [--mde-entry FILE] [--repo-root DIR
                   --synthetic-inputs]
      Renders a section 12 artifact (src/stats/report.py) after the full verifier with the 37 inputs
      and the contract; the profile is the verified status, never a flag. --repo-root (with
      --synthetic-inputs only) resolves the inputs the artifact's records name.

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
    ap.add_argument("--mode", required=True, choices=("official", "smoke", "rehearsal", "report"))
    ap.add_argument("--inputs", type=Path, help="input-list file (plantseg-stats-inputs/1.0.0)")
    ap.add_argument("--out-dir", type=Path, help="parent directory of the artifact directory")
    ap.add_argument("--run-id", help="the artifact directory name, ^[a-z0-9][a-z0-9._-]{0,63}$")
    ap.add_argument("--confirm-official-test-analysis", action="store_true")
    ap.add_argument("--synthetic-inputs", action="store_true",
                    help="declare the inputs synthetic (official mode, while the binding is unbound)")
    ap.add_argument("--repo-root", type=Path, help="resolve input paths here (synthetic or smoke)")
    ap.add_argument("--B", type=int, default=10_000, help="bootstrap replicates (official: 10000)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes, whole tasks each")
    for seed in ("s42", "s43", "s44"):
        ap.add_argument(f"--{seed}", type=Path, help=f"rehearsal: the E1 {seed} VAL artifact")
    ap.add_argument("--out-root", type=Path, help="rehearsal: outputs under OUT-ROOT/<UTC>/")
    ap.add_argument("--expect-k-val", type=int, help="rehearsal: refuse unless K_val equals this")
    ap.add_argument("--mde-entry", type=Path, help="the committed MDE entry (reported, see Q9)")
    ap.add_argument("--artifact", type=Path, help="report: the statistics artifact directory")
    ap.add_argument("--report-out", type=Path, help="report: a new directory for the report")
    return ap.parse_args(argv)


MODE_ARGS = {"official": ("inputs", "out_dir", "run_id", "repo_root", "B", "jobs",
                          "confirm_official_test_analysis", "synthetic_inputs"),
             "smoke": ("inputs", "out_dir", "run_id", "repo_root", "B", "jobs"),
             "rehearsal": ("s42", "s43", "s44", "out_root", "B", "jobs", "expect_k_val",
                           "mde_entry"),
             "report": ("artifact", "report_out", "repo_root", "synthetic_inputs", "mde_entry")}
DEFAULTS = {"B": 10_000, "jobs": 1, "confirm_official_test_analysis": False,
            "synthetic_inputs": False}


def refuse_foreign(args) -> None:
    """An argument that belongs to another mode is refused by name, never ignored."""
    allowed = set(MODE_ARGS[args.mode])
    foreign = [n for n in vars(args) if n != "mode" and n not in allowed
               and getattr(args, n) not in (None, DEFAULTS.get(n))]
    if foreign:
        raise Refused(f"--mode {args.mode} does not accept " + ", ".join(
            "--" + n.replace("_", "-") for n in sorted(foreign)))


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
        from src.stats import rehearsal as RH
        from src.stats import report as RP
    except Exception:                                         # noqa: BLE001
        traceback.print_exc()
        print("ERROR: the statistics modules could not be imported", file=sys.stderr)
        return EXIT_ERROR
    refusals = (Refused, D.DriverRefusal, A.ArtifactError, RH.RehearsalRefusal, RP.ReportError)
    try:
        if args.mode == "official":
            D.official_doors(confirm=args.confirm_official_test_analysis,      # doors 1 and 2
                             synthetic=args.synthetic_inputs)
            refuse_foreign(args)
            require(args, "inputs", "out_dir", "run_id")
            validate_counts(args)
            res = D.run_official(inputs_list=args.inputs, out_dir=args.out_dir,
                                 run_id=args.run_id, confirm=args.confirm_official_test_analysis,
                                 synthetic=args.synthetic_inputs, repo_root=args.repo_root,
                                 B=args.B, jobs=args.jobs)
        elif args.mode == "smoke":
            if args.confirm_official_test_analysis or args.synthetic_inputs:
                raise Refused("--confirm-official-test-analysis and --synthetic-inputs belong to "
                              "official mode")
            refuse_foreign(args)
            require(args, "inputs", "out_dir", "run_id")
            validate_counts(args)
            res = D.run_smoke(inputs_list=args.inputs, out_dir=args.out_dir, run_id=args.run_id,
                              repo_root=args.repo_root, B=args.B, jobs=args.jobs)
        elif args.mode == "report":
            if args.confirm_official_test_analysis:
                raise Refused("--confirm-official-test-analysis belongs to official mode")
            if args.repo_root is not None and not args.synthetic_inputs:
                raise Refused("--repo-root is accepted only with --synthetic-inputs (P16)")
            refuse_foreign(args)
            require(args, "artifact", "report_out")
            out, doc = RP.run_report(artifact=args.artifact, report_out=args.report_out,
                                     repo_root=args.repo_root, synthetic=args.synthetic_inputs,
                                     mde_entry=args.mde_entry)
            print(f"report: {out}")
            print(f"profile (the verified status): {doc['profile']}; warnings: {doc['warnings']}")
            print(f"k (AM-5): {doc['am5_excluded_count_k']}")
            for r in doc["family"]:
                print(f"  {r['comparison_id']}: Holm reject {r['holm_reject']}, dataset-level "
                      f"difference {r['dataset_level_difference']!r}: {r['label']}")
            print(f"REPORT_DIR: {out}")
            return EXIT_OK
        else:
            if args.repo_root is not None:
                raise Refused("--repo-root is not accepted in rehearsal mode (P16: synthetic "
                              "official mode and smoke mode only)")
            if args.confirm_official_test_analysis or args.synthetic_inputs:
                raise Refused("--confirm-official-test-analysis and --synthetic-inputs belong to "
                              "official mode")
            refuse_foreign(args)
            validate_counts(args)
            reh = RH.run_rehearsal(s42=args.s42, s43=args.s43, s44=args.s44,
                                   out_root=args.out_root, B=args.B, jobs=args.jobs,
                                   expect_k=args.expect_k_val, mde_entry=args.mde_entry)
            print(f"rehearsal outputs: {reh.out_dir}")
            print(f"K_val (AM-5 zero-disease VAL images): {reh.k_val}")
            print("INVESTIGATE (AM-17 item 9): " + (
                "FIRED -- " + "; ".join(reh.investigate) if reh.investigate else "not fired"))
            print(f"criteria: {'all passed' if reh.criteria_passed else 'FAILED (exit 1)'}")
            print(f"REHEARSAL_DIR: {reh.out_dir}")
            print(f"WALL_SECONDS: {reh.seconds:.1f}")
            return EXIT_OK if reh.criteria_passed else EXIT_CRITERIA
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
