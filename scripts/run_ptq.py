#!/usr/bin/env python3
"""E4/E7 static INT8 PTQ of record (lane 8, L-AM10; LQP): parent + calibration list -> artifacts + record.

    python -B scripts/run_ptq.py --real-run --confirm-real-run --stage E4 --seed S \\
        --source-run-dir RUN_DIR --expect-source-sha256 SHA256 \\
        --calibration-list configs/calibration/ptq_calibration_seed42.json --expect-list-checksum SHA256 \\
        --expect-head COMMIT --data-root ROOT --out-dir /out/e4_sS_cal42_a1 [--threads N] [--replicate-of RECORD]
    (E7 adds --lambda-selection, --lambda-selection-sha256, --alpha-selection, --alpha-selection-sha256)
    python -B scripts/run_ptq.py --compare OUT_A/e4_ptq_run_meta.json OUT_B/e4_ptq_run_meta.json

BOUND PATH (LQP: L-CKPT-GUARD Q-part, L-AM16-CALIB). A run whose calibration list is a registered build
(artifact_status "registered": the four lists committed in configs/calibration/) is bound to its parent by
src/quant/ptq_record.py, imported here for that path only: the E1 run of record (E4) or the E3 run of record
the λ and α selections and the KD launch log name (E7), read from its run directory (--source-run-dir; its
best.json names the checkpoint), at a clean commit (--expect-head) in the student image, on a data root
without TEST surfaces. After the run provenance it writes <stage>_ptq_record.json (ptq_record/1; role
of_record, calibration_sensitivity or d3_replicate), without which a run directory is not of record. The
legacy runners scripts/run_e4.py and run_e7.py refuse every run (ruling Q1-R2).

UNBOUND PATH. A run on a list that is not a registered build (a synthetic list: artifact_status "smoke") takes
--source-ckpt and refuses every binding flag (binding_needs_registered_list); it is never of record.

Before either path, exactly one of --source-ckpt and --source-run-dir is given (source_exactly_one), and no
path argument contains "test", as given or resolved (test_path), checked by name before it is opened.

RUN. E4 takes the E1 FP32 checkpoint and E7 the projection-free E3 student (the strict loaders of
src/quant/checkpoint.py). The list's 128 TRAIN images are read from ROOT/images/train and
ROOT/annotations/train in list order, one image per mini-batch (AM-10), with the clean-test canvas
preprocessing and no augmentation; any data path naming VAL or TEST is refused before an image is
opened. OUT, outside the repository and absent or empty, receives:

    <stage>_int8_qnnpack.torchscript.pt   the QNNPACK TorchScript artifact of record
    <stage>_int8_student.pt               its converted state_dict (the src/quant/runner.py schema)
    <stage>_int8_x86_latency.torchscript.pt   the x86/fbgemm latency copy (reduce_range=True)
    <stage>_ptq_run_meta.json             the run provenance: evaluate_model.py --provenance
    <stage>_ptq_record.json               the bound path's record (ptq_record/1), written last

The run provenance is written only after every artifact has been re-read from disk and checked: all
conv weights per-channel INT8, no undeclared float module or graph region, and bitwise-equal outputs
from the TorchScript artifact and its state_dict rebuild on the 128 calibration images and fixed
synthetic inputs (plus the TorchScript artifact against the in-memory model, and the x86 copy against
its in-memory model). Any failure is a STOP: <stage>_ptq_STOP.json is written instead and nothing is
evaluable. --expect-source-sha256 and
--expect-list-checksum pin the checkpoint and the list (checksum_sha256, the value its decision-log row
records); the E1 seed-42 checkpoint of record is cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03.

COMPARE (d3). Two runs from the same checkpoint and list: PASS iff the three artifact sha256s are equal,
every file re-hashes to its record, and the source, list, qconfig, engine and library versions agree.
Run both in the same environment and checkout path, each as its own invocation: the TorchScript files'
debug records keep absolute source paths and the call stack of the trace, and TorchScript renames a
class traced twice in one process, so a run is refused in a process that has already traced the
student. The state_dict's sha256 depends on none of this.

OFFICIAL E4/E7. A run on a sensitivity list (seeds 43-45, AM-16 item 5) or a synthetic list is recorded
as such (ptq_run_role) and can only be scored descriptively: scripts/evaluate_model.py refuses an
official score unless the run used the registered seed-42 list committed at
configs/calibration/ptq_calibration_seed42.json and its directory holds its PTQ record of role of_record.

Exit codes: 0 written / PASS; 1 STOP or compare FAIL (inspect before anything else runs); 2 refused or
unreadable input (nothing written). Nothing here trains, downloads, uses a GPU or reads VAL or TEST.
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

EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2
# The bound path's flags: refused on a list that is not a registered build (binding_needs_registered_list)
BINDING_FLAGS = ("seed", "source_run_dir", "lambda_selection", "lambda_selection_sha256", "alpha_selection",
                 "alpha_selection_sha256", "expect_head", "replicate_of")
PATH_FLAGS = ("source_ckpt", "source_run_dir", "calibration_list", "data_root", "out_dir", "lambda_selection",
              "alpha_selection", "replicate_of")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E4/E7 PTQ of record (lane 8, L-AM10; LQP).")
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--stage", choices=["E4", "E7", "e4", "e7"], default=None)
    p.add_argument("--source-ckpt", default=None, help="unbound path: E1 checkpoint (E4) or E3 checkpoint (E7)")
    p.add_argument("--source-run-dir", default=None,
                   help="bound path: the E1 (E4) or E3 (E7) run directory of record; its best.json names the "
                        "checkpoint")
    p.add_argument("--expect-source-sha256", default=None)
    p.add_argument("--calibration-list", default=None,
                   help="configs/calibration/ptq_calibration_seed<S>.json (scripts/build_calibration_lists.py)")
    p.add_argument("--expect-list-checksum", default=None,
                   help="the list's checksum_sha256 (sha256 of its ids joined by newlines)")
    p.add_argument("--data-root", default=None, help="PlantSeg root holding images/train, annotations/train")
    p.add_argument("--out-dir", default=None,
                   help="directory OUTSIDE the repository, absent or empty; an artifact is never overwritten")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (recorded either way)")
    p.add_argument("--seed", type=int, default=None, help="bound path: the parent's seed (42, 43 or 44)")
    p.add_argument("--lambda-selection", default=None,
                   help="E7 bound path: reports/derived/lambda_selection.json (repo-relative, committed)")
    p.add_argument("--lambda-selection-sha256", default=None)
    p.add_argument("--alpha-selection", default=None,
                   help="E7 bound path: reports/derived/alpha_selection.json (repo-relative, committed)")
    p.add_argument("--alpha-selection-sha256", default=None)
    p.add_argument("--expect-head", default=None, help="bound path: the 40-hex commit of the checkout")
    p.add_argument("--replicate-of", default=None,
                   help="bound path: the original's <stage>_ptq_record.json; this run is its d3 replicate")
    p.add_argument("--compare", nargs=2, metavar="RUN_META", default=None,
                   help="d3: compare two run provenances instead of running")
    return p


def print_run(prov: dict) -> None:
    v = prov["verification"]
    print(f"  artifact of record  {prov['converted_artifact']}  sha256 {prov['converted_artifact_sha256']}")
    print(f"  state_dict          {prov['state_dict_artifact']['path']}  "
          f"sha256 {prov['state_dict_artifact']['sha256']}")
    print(f"  x86 latency copy    {prov['x86_latency_copy']['path']}  sha256 {prov['x86_latency_copy']['sha256']} "
          f"({prov['x86_latency_copy']['engine']})")
    print(f"  calibration         seed {prov['calibration']['seed']} ({prov['calibration']['role']}), "
          f"{prov['calibration']['batches_seen']} batches of {prov['calibration']['batch_size']}; "
          f"run role {prov['ptq_run_role']}")
    print(f"  checks              {sum(v['checks'].values())}/{len(v['checks'])} "
          f"{', '.join(k for k, ok in v['checks'].items() if ok)}")
    print(f"  sigmoid quantized   {v['census']['sigmoid']['quantized']} u6={v['census']['sigmoid']['u6_values']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        from src.quant.ptq import PTQRefused, compare_runs
        try:
            result = compare_runs(*args.compare)
        except PTQRefused as e:
            print(f"RESULT: REFUSED [{e.code}] -- {e}")
            return EXIT_REFUSED
        except Exception as e:                        # noqa: BLE001 -- unreadable, never a verdict
            print(f"RESULT: ERROR -- cannot compare: {type(e).__name__}: {e}")
            return EXIT_REFUSED
        print(json.dumps(result, indent=2))
        verdict = "PTQ d3 PASS (bitwise identical)" if result["identical"] else "PTQ d3 FAIL -- STOP"
        print(f"RESULT: {verdict}")
        return EXIT_OK if result["identical"] else EXIT_STOP

    if not args.real_run:
        print("REFUSING: a PTQ run requires --real-run --confirm-real-run. Nothing was read or written.",
              file=sys.stderr)
        return EXIT_REFUSED
    if not args.confirm_real_run:
        print("REFUSING: --real-run requires --confirm-real-run. Nothing was read or written.",
              file=sys.stderr)
        return EXIT_REFUSED
    missing = [f for f, v in (("--stage", args.stage), ("--expect-source-sha256", args.expect_source_sha256),
                              ("--calibration-list", args.calibration_list),
                              ("--expect-list-checksum", args.expect_list_checksum),
                              ("--data-root", args.data_root), ("--out-dir", args.out_dir)) if not v]
    if missing:
        print(f"REFUSING: missing {missing}. Nothing was read or written.", file=sys.stderr)
        return EXIT_REFUSED
    if bool(args.source_ckpt) == bool(args.source_run_dir):
        print("RESULT: REFUSED [source_exactly_one] -- pass exactly one of --source-run-dir (a registered list: "
              "the parent's run directory) and --source-ckpt (a synthetic list). Nothing was read or written.")
        return EXIT_REFUSED
    from src.quant.qat import names_test
    for flag in PATH_FLAGS:
        value = getattr(args, flag)
        if value and names_test(value):
            print(f"RESULT: REFUSED [test_path] -- --{flag.replace('_', '-')} {value!r} contains 'test' (as given "
                  "or resolved); nothing under TEST is ever named, opened or listed. Nothing was read or written.")
            return EXIT_REFUSED

    from src.quant.ptq import LIST_STATUS_REGISTERED, PTQRefused, PTQStop, load_calibration_list, run_ptq
    record = None
    try:
        doc, list_sha256 = load_calibration_list(args.calibration_list, expected_checksum=args.expect_list_checksum)
        if doc.get("artifact_status") == LIST_STATUS_REGISTERED:
            from src.quant.ptq_record import run_bound
            prov, record, record_path, record_sha256 = run_bound(args, doc, list_sha256)
        else:
            given = [f"--{f.replace('_', '-')}" for f in BINDING_FLAGS if getattr(args, f) is not None]
            if given:
                raise PTQRefused("binding_needs_registered_list",
                                 f"{given} bind a run of record, which only a registered list (artifact_status "
                                 f"{LIST_STATUS_REGISTERED!r}) makes; this list is {doc.get('artifact_status')!r}")
            prov = run_ptq(stage=args.stage.upper(), source_ckpt=args.source_ckpt,
                           expect_source_sha256=args.expect_source_sha256,
                           calibration_list=args.calibration_list,
                           expect_list_checksum=args.expect_list_checksum, data_root=args.data_root,
                           out_dir=args.out_dir, threads=args.threads)
    except PTQRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}")
        return EXIT_REFUSED
    except PTQStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return EXIT_STOP
    print_run(prov)
    if record is None:
        n = len(prov["verification"]["checks"])
        print(f"RESULT: PTQ WRITTEN ({prov['stage']}, {n}/{n} checks)")
        return EXIT_OK
    tag = (f"{record['stage']} s{record['seed']}, role {record['role']}, parent {record['parent']['run_id']}, "
           f"record {record_sha256[:12]}")
    print(f"  record              {record_path}  sha256 {record_sha256}")
    print(f"RESULT: PTQ OF RECORD WRITTEN ({tag})" if record["role"] == "of_record"
          else f"RESULT: PTQ RECORD WRITTEN ({tag})")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
