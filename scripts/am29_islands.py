#!/usr/bin/env python3
"""AM-29 Step 1's FP32 islands (lane L-AM29, P2): convert with G0-G4 kept in FP32, report the INT8 shares, score VAL.

    python -B scripts/am29_islands.py --split val --model E6-s42-e15 --qat-checkpoint <run>/epoch_ckpts/e15.pt \\
        --expect-checkpoint-sha256 <64 hex> [--run-dir <run>] --out <report.json outside the repo> [--groups G0 G1 ...]
    python -B scripts/am29_islands.py --split val --model E7|E7-R2 --fp32-checkpoint <best ckpt> \\
        --expect-source-sha256 <64 hex> --calibration-list configs/calibration/ptq_calibration_seed42.json \\
        --expect-list-checksum <64 hex> --out <report.json> [--groups ...] [--threads T]

(a) --model E6-s42-e15: E6 seed 42's selected epoch-15 checkpoint (a QAT state; DL-131, DL-176). Its sha256 of record
is 74fd438a9e953ce84117a61454b879940b0a6a8f357e7705c22a86ab0ed847a5 (AM-29 item 2(a)); --expect-checkpoint-sha256 is
required and, outside smoke inputs, must equal it.
(b) --model E7 (E3 seed 42, 4baa5bad..., DL-121) or E7-R2 (E3-R2 seed 42, da70d82f...): the FP32 checkpoint through
the PTQ of record (prepare_ptq, calibrate on the AM-10 list of record's 128 TRAIN images in list order, one per batch,
convert under QNNPACK). VAL only: --split test is refused by name before anything is read. Each variant's INT8 share of
parameters and of MACs is reported; a variant is eligible only if at least 90% of its parameters are INT8; G0 is "int8"
and every other variant "mixed precision". Exit codes: 0 done, 1 a check failed, 2 refused, 4 error.
--threads is read back after it is set, and a run without --allow-smoke-inputs requires code_clean_at_head true
(AUDIT-LAM29-1 F-14); E7-R2's G0 state_dict is labelled under the stage of record, E7, as E7-R2 is made.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FORMAT = "am29_islands/1"
EXIT_OK, EXIT_STOP, EXIT_REFUSED, EXIT_ERROR = 0, 1, 2, 4
MODELS = {
    "E6-s42-e15": {"source": "qat", "stage": "E6", "seed": 42, "epoch": 15,
                   "sha256": "74fd438a9e953ce84117a61454b879940b0a6a8f357e7705c22a86ab0ed847a5",
                   "record": "DL-131 (E6 seed 42 selected epoch 15); AM-29 item 2(a) (epoch_ckpts/e15.pt, its sha256)"},
    "E7": {"source": "ptq", "parent": "E3 seed 42", "loader_stage": "e7",
           "sha256": "4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d", "record": "DL-121"},
    "E7-R2": {"source": "ptq", "parent": "E3-R2 seed 42", "loader_stage": "e7", "stage": "E7",
              "sha256": "da70d82fcba6543488d6748691fbe9f56a4bf8c29b1c9d29d92f0d5287291ef4",
              "record": "the L-AM29 prompt (DL-136 holds the value)"},
}


class Refused(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="val")
    p.add_argument("--model", required=True)
    p.add_argument("--groups", nargs="+", default=["G0", "G1", "G2", "G3", "G4"])
    p.add_argument("--qat-checkpoint", default=None)
    p.add_argument("--expect-checkpoint-sha256", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--fp32-checkpoint", default=None)
    p.add_argument("--expect-source-sha256", default=None)
    p.add_argument("--calibration-list", default=None)
    p.add_argument("--expect-list-checksum", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-score", action="store_true", help="conversion and checks only (no VAL pass)")
    p.add_argument("--allow-smoke-inputs", action="store_true",
                   help="smokes only: a checkpoint or list not of record, a sample cap; recorded in the output")
    return p


def refuse_split(split) -> None:
    """TEST first, by name in any case, before anything is read; then anything but val."""
    if split is None or "test" in str(split).lower():
        raise Refused("split_test", f"--split {split!r}: AM-29 Step 1 scores VAL only; TEST is never read here")
    if split != "val":
        raise Refused("split_not_val", f"--split {split!r}: val only")


def _sha_ok(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def qat_source(a, spec: dict) -> tuple[dict, dict]:
    """(a) the QAT state: hashed before it is parsed, loaded weights_only as a qat-train-state sidecar, its identity
    E6 seed 42 epoch 15, its state flag finite; with --run-dir, also the file its epoch_end row names."""
    import io

    import torch

    from src.quant import qat as Q
    from src.quant import qat_artifacts as A
    from src.quant.x86_latency import validate_qat_sidecar
    if not a.qat_checkpoint or not _sha_ok(a.expect_checkpoint_sha256):
        raise Refused("qat_source_flags", "--qat-checkpoint and --expect-checkpoint-sha256 (64 hex) are required")
    if not a.allow_smoke_inputs and a.expect_checkpoint_sha256 != spec["sha256"]:   # AUDIT-LAM29-1 F-14
        raise Refused("checkpoint_not_of_record", f"{a.model}'s checkpoint sha256 is {spec['sha256']} "
                                                  f"({spec['record']}); got {a.expect_checkpoint_sha256}")
    raw = Path(a.qat_checkpoint).read_bytes()
    sha = Q.sha256_bytes(raw)
    if sha != a.expect_checkpoint_sha256:
        raise Refused("checkpoint_sha256_mismatch", f"{a.qat_checkpoint} has sha256 {sha}, expected "
                                                    f"{a.expect_checkpoint_sha256}")
    payload = validate_qat_sidecar(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True),
                                   origin=str(a.qat_checkpoint))
    want = {"stage": spec["stage"], "seed": spec["seed"], "epoch": spec["epoch"]}
    wrong = {k: payload.get(k) for k, v in want.items() if payload.get(k) != v}
    if (wrong and not a.allow_smoke_inputs) or payload.get("state_finite") is not True:
        raise Refused("checkpoint_identity", f"{a.qat_checkpoint} records {wrong or payload.get('state_finite')!r}; "
                                             f"{a.model} is {want} with a finite state")
    run = None
    if a.run_dir is not None:
        rec = A.read_run_record(a.run_dir)
        A.require_complete(rec)
        ck, ck_sha = A.verified_checkpoint(rec, payload.get("epoch"))
        if ck_sha != sha or Path(ck).resolve() != Path(a.qat_checkpoint).resolve():
            raise Refused("checkpoint_not_the_runs", f"{a.qat_checkpoint} is not the run's epoch checkpoint {ck}")
        run = {"run_id": rec["run_id"], "telemetry_sha256": rec["telemetry_sha256"]}
    return payload["model_state_dict"], {"kind": "qat", "checkpoint": str(a.qat_checkpoint), "sha256": sha,
                                         "stage": payload.get("stage"), "seed": payload.get("seed"),
                                         "epoch": payload.get("epoch"), "run": run, "record": spec["record"]}


def ptq_source(a, spec: dict):
    """(b) the FP32 checkpoint through the PTQ of record: the strict E3 loader, the AM-10 list of record's TRAIN
    images in list order, one per batch."""
    from configs.data import DATA
    from src.quant import ptq as P
    from src.quant.runner import QuantRunError, check_source
    from src.quant.stages import resolve_quant_stage
    if not a.fp32_checkpoint or not _sha_ok(a.expect_source_sha256):
        raise Refused("ptq_source_flags", "--fp32-checkpoint and --expect-source-sha256 (64 hex) are required")
    if not a.allow_smoke_inputs and a.expect_source_sha256 != spec["sha256"]:
        raise Refused("source_not_of_record", f"{a.model}'s parent is {spec['parent']} ({spec['sha256']}, "
                                              f"{spec['record']}); got {a.expect_source_sha256}")
    if not a.calibration_list or not _sha_ok(a.expect_list_checksum):
        raise Refused("calibration_flags", "--calibration-list and --expect-list-checksum (64 hex) are required")
    cal, cal_sha = P.load_calibration_list(a.calibration_list, expected_checksum=a.expect_list_checksum)
    rec = {"role": cal["role"], "seed": cal["seed"], "list_artifact_status": cal.get("artifact_status"),
           "checksum_sha256": cal["checksum_sha256"]}
    err = P.list_of_record_error(rec, REPO)
    if err and not a.allow_smoke_inputs:
        raise Refused("calibration_not_of_record", err)
    sha = P.sha256_file(a.fp32_checkpoint)
    if sha != a.expect_source_sha256:
        raise Refused("source_sha256_mismatch", f"{a.fp32_checkpoint} has sha256 {sha}, expected {a.expect_source_sha256}")
    try:
        model, _ck, meta = check_source(resolve_quant_stage(spec["loader_stage"]), a.fp32_checkpoint)
    except QuantRunError as e:
        raise Refused(e.code, str(e)) from e
    model.eval()
    ds = P.calibration_dataset(DATA["root"], cal["selected_ids"])
    return model, ds, {"kind": "ptq", "checkpoint": str(a.fp32_checkpoint), "sha256": sha, "parent": spec["parent"],
                       "calibration": {"list": str(a.calibration_list), "list_file_sha256": cal_sha, **rec,
                                       "list_of_record_error": err}}


def clean_identity(a) -> dict:
    """AUDIT-LAM29-1 F-14: the code identity this run records (src/quant/am29.py code_identity), taken once its source is
    bound and before any conversion; a run without --allow-smoke-inputs requires code_clean_at_head true."""
    from src.quant.am29 import code_identity
    ident = code_identity()
    if not a.allow_smoke_inputs and ident.get("code_clean_at_head") is not True:
        raise Refused("code_not_clean", f"code_clean_at_head is {ident.get('code_clean_at_head')!r} at HEAD "
                                        f"{ident.get('git_head')}; a Step 1 record runs on code clean at HEAD")
    return ident


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        refuse_split(a.split)                                 # TEST first, by name, before anything is read
        if a.model not in MODELS:
            raise Refused("model_unknown", f"--model {a.model!r} is not one of {sorted(MODELS)}")
        spec = MODELS[a.model]
        from src.eval import am29_score as SC
        from src.quant import am29_islands as I
        bad = [g for g in a.groups if g not in I.GROUPS]
        if bad or len(set(a.groups)) != len(a.groups):
            raise Refused("groups", f"--groups {a.groups}: each of {sorted(I.GROUPS)} at most once")
        if (a.max_samples is not None) and not a.allow_smoke_inputs:
            raise Refused("max_samples", "--max-samples is for smokes (with --allow-smoke-inputs); Step 1 scores VAL whole")
        SC.refuse_test_paths({"--qat-checkpoint": a.qat_checkpoint, "--run-dir": a.run_dir, "--out": a.out,
                              "--fp32-checkpoint": a.fp32_checkpoint, "--calibration-list": a.calibration_list})
        out = Path(a.out).resolve()
        if out == REPO or REPO in out.parents:
            raise Refused("out_in_repo", f"--out {out} is inside the repository")
        if out.exists():
            raise Refused("out_exists", f"{out} exists; a report is written once")
        import torch
        if a.threads is not None:
            torch.set_num_threads(int(a.threads))
            if torch.get_num_threads() != int(a.threads):          # AUDIT-LAM29-1 F-14: read back, before anything
                raise Refused("threads_not_set", f"torch.set_num_threads({a.threads}) left {torch.get_num_threads()} "
                                                 "threads; nothing was read")
        from src.quant import ptq as P
        t0 = time.time()
        if spec["source"] == "qat":
            state, source = qat_source(a, spec)
            ident = clean_identity(a)                              # AUDIT-LAM29-1 F-14
            res = I.convert_qat_state(state, groups=tuple(a.groups))
            prepared_qat = res["prepared"]
        else:
            model, ds, source = ptq_source(a, spec)
            ident = clean_identity(a)                              # AUDIT-LAM29-1 F-14
            prepared, n = I.calibrate_fp32(model, P.iter_calibration_images(ds))
            source["calibration"]["batches_seen"] = n
            res = I.convert_ptq_prepared(prepared, groups=tuple(a.groups))
            prepared_qat = None
        # the G0 state_dict in the runner schema, under the stage of record, so that its sha256 compares with the
        # companion the conversion of record wrote (E6 seed 42's e15_int8_student.pt; E7's e7_int8_student.pt)
        g0_bytes = P.state_dict_bytes(res["g0"], spec.get("stage") or a.model, "qat" if spec["source"] == "qat" else "ptq")
        rows = []
        for g in a.groups:
            model_g, rep = res["variants"][g]
            rec = I.variant_record(g, model_g, rep, g0=res["g0"], prepared_qat=prepared_qat)
            if not a.no_score:
                rec["val"] = SC.score_model(model_g, split="val", max_samples=a.max_samples)
                for line in SC.dataset_level_lines(rec["label"], f"{a.model} {g}", rec["val"]):
                    print(line)
            sh = rec["shares"]
            print(f"[islands] {a.model} {g} ({rec['label']}): INT8 params {sh['int8_param_share']:.4f}, INT8 MACs "
                  f"{sh['int8_mac_share']:.4f} ({sh['int8_mac_share_with_boundary']:.4f} with the final float "
                  f"upsample), eligible {rec['eligible']}, checks {'OK' if rec['checks_ok'] else 'FAIL'}")
            rows.append(rec)
        doc = {"format": FORMAT, "split": "val", "model": a.model, "spec": spec, "source": source,
               "groups": a.groups, "g0_state_dict_sha256": P._sha256_bytes(g0_bytes), "variants": rows,
               "eligibility_rule": f"int8_param_share >= {I.ELIGIBLE_PARAM_SHARE}", "seconds": time.time() - t0,
               "git_head": ident["git_head"], "code_clean_at_head": ident["code_clean_at_head"],
               "smoke_inputs": bool(a.allow_smoke_inputs), "max_samples": a.max_samples}
        P.write_exclusive({out: (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
                           .encode("utf-8")})
        ok = all(r["checks_ok"] for r in rows)
        print(f"RESULT: ISLANDS {'DONE' if ok else 'CHECK FAILED'} ({a.model}: "
              + ", ".join(f"{r['group']} {'eligible' if r['eligible'] else 'not eligible'}" for r in rows) + f") -> {out}")
        return EXIT_OK if ok else EXIT_STOP
    except Exception as e:                                   # noqa: BLE001 -- named refusals and errors
        code = getattr(e, "code", None)
        if code is not None:
            print(f"RESULT: REFUSED [{code}] -- {e}")
            return EXIT_REFUSED
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
