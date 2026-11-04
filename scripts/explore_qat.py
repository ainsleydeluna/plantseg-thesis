#!/usr/bin/env python3
"""E3 (lane L-AM24): the exploratory QAT entry for C1, N1, N2, N5 and D (E6-E2's recipe of record, one named change).

    python -B -I scripts/explore_qat.py --real-run --confirm-real-run --item C1|N1|N2|N5|D --seed S
        --source-run-dir <E2 run dir | D: the exploratory winner's trainer dir>
        (--expect-source-sha256 <E2 of record's checkpoint sha256> | D: --parent-sha256 <its checkpoint sha256>)
        --clip-selection <clip_selection.json of record> --clip-selection-sha256 c433145a...
        --num-workers 12 --device cuda --expect-head <40-hex HEAD> --run-id x-... --out-dir <root>/<run-id>
        [N1: --kd-logit --teacher-ckpt <.../iter_24000.pth> --teacher-ckpt-sha256 8c0e649a...]
        [N2: --sites-on-input-grid --tf32-off]   [N5: --fp32-head]

Order: the item's flags and seed; the E0 guard (split VAL, the data root's TRAIN/VAL-only check, x- run ID, output
outside the repository and official folders); the real-run gates of src/quant/qat.py real_run_gates that apply (the
config pins, HEAD = --expect-head, the pinned QAT paths and the lane's files clean at HEAD, workers, CUDA present and
uninitialised, TF32 at torch defaults before launch, QNNPACK, the clip selection of record with clip 5.0); the
exploratory parent check; the strict source load; for N1 the teacher of record through src/quant/qat_kd.py
teacher_of_record. Nothing is written before all pass. Then <out-dir>/x_run_meta.json, and the pinned trainer's run
directory <out-dir>/<run-id>. Block B: scripts/explore_qat_blockb.py. Exit codes as src/quant/qat.py main: 0 done,
1 STOP, 2 refused, 3 aborted or incomplete, 4 error.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.explore import guard as G                                   # noqa: E402
from src.explore import qat_explore as X                             # noqa: E402

FLAG_NAMES = ("kd_logit", "sites_on_input_grid", "tf32_off", "fp32_head")
_HEX40 = re.compile(r"[0-9a-f]{40}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--item", required=True, choices=list(X.ITEMS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--source-run-dir", required=True)
    p.add_argument("--expect-source-sha256", default=None)
    p.add_argument("--parent-sha256", default=None)
    p.add_argument("--clip-selection", required=True)
    p.add_argument("--clip-selection-sha256", required=True)
    p.add_argument("--teacher-ckpt", default=None)
    p.add_argument("--teacher-ckpt-sha256", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expect-head", default=None)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="val")
    for f in FLAG_NAMES:
        p.add_argument("--" + f.replace("_", "-"), dest=f, action="store_true")
    return p


def pinned_paths(item: str) -> tuple[str, ...]:
    from src.quant import qat as Q
    paths = list(Q.QAT_CODE_PATHS)
    if item == "N1":
        from src.quant import qat_kd as K
        paths += list(K.E6KD_CODE_PATHS)
    return tuple(dict.fromkeys(paths))


def gates(args, flags: set) -> dict:
    """Everything before the first write. Returns what the run needs; raises GuardError, ExploreQATError, QATRefused."""
    import torch
    from src.quant import qat as Q
    from src.quant.qconfig import QuantBackendUnavailable, select_qnnpack_backend
    err = X.item_flag_error(args.item, flags, args.seed if args.item != "D" else None)
    if err:
        raise X.ExploreQATError(*err)
    if not (args.real_run and args.confirm_real_run):
        raise X.ExploreQATError("real_run_flags", "an exploratory QAT launch needs --real-run --confirm-real-run")
    if args.item == "N1" and not (args.teacher_ckpt and args.teacher_ckpt_sha256):
        raise X.ExploreQATError("teacher_required", "N1 needs --teacher-ckpt and --teacher-ckpt-sha256 (of record)")
    if args.item != "N1" and (args.teacher_ckpt or args.teacher_ckpt_sha256):
        raise X.ExploreQATError("teacher_not_applicable", f"{args.item} runs no teacher")
    from src.explore.fp32_head import PRECISION_LABEL
    st = G.guarded_start(
        item=args.item, run_id=args.run_id, out_dir=args.out_dir, split=args.split, allowed_splits=G.VAL_ONLY,
        input_paths={"--source-run-dir": args.source_run_dir, "--clip-selection": args.clip_selection,
                     "--teacher-ckpt": args.teacher_ckpt},
        seeds=[args.seed], pinned_paths=pinned_paths(args.item),
        precision_label=PRECISION_LABEL if args.item == "N5" else "int8 QNNPACK (as E6-E2)",
        recipe={"recipe_of_record": X.RECIPE_OF_RECORD, "change": X.CHANGES[args.item], "flags": sorted(flags),
                "seed": args.seed, "clip": X.CLIP_OF_RECORD, "comparators": X.COMPARATORS,
                "lambda_logit": X.N1_LAMBDA if args.item == "N1" else None,
                "d_trigger": str(X.D_TRIGGER) if args.item == "D" else None},
        write=False)
    pins = Q.config_pins_error()
    if pins:
        raise X.ExploreQATError("config_pins", pins)
    if not args.expect_head or not _HEX40.fullmatch(args.expect_head):
        raise X.ExploreQATError("expect_head_format", "--expect-head must be the 40-hex commit of the checkout")
    code = st["run_meta"]
    if code["git_head"] != args.expect_head:
        raise X.ExploreQATError("expect_head_mismatch", f"HEAD is {code['git_head']!r}, --expect-head {args.expect_head}")
    if code["pinned_clean_at_head"] is not True or code["lane_clean_at_head"] is not True:
        raise X.ExploreQATError("code_not_clean", "the pinned QAT paths or the lane's files are not clean at HEAD")
    if args.num_workers is None or args.num_workers < 1:
        raise X.ExploreQATError("num_workers", "--num-workers must be given and at least 1")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise X.ExploreQATError("cuda_required", f"--device {args.device!r}: a QAT run needs CUDA")
    if torch.cuda.is_initialized():
        raise X.ExploreQATError("cuda_initialized_before_seed", "CUDA is already initialised")
    tf = Q.tf32_error()
    if tf:
        raise X.ExploreQATError("tf32_not_default", f"{tf} (N2 switches TF32 off itself, after this check)")
    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        raise X.ExploreQATError("backend_unavailable", str(e)) from e
    if args.clip_selection_sha256 != X.CLIP_SELECTION_SHA256:
        raise X.ExploreQATError("clip_selection_not_of_record", f"--clip-selection-sha256 is not DL-139's "
                                                                f"{X.CLIP_SELECTION_SHA256}")
    clip = Q.clip_binding("E6-E2", args.seed, X.CLIP_OF_RECORD, u4_pilot=False, clip_selection=args.clip_selection,
                          clip_selection_sha256=args.clip_selection_sha256)
    stage, parent = X.exploratory_parent(args.item, st["inputs"]["--source-run-dir"], seed=args.seed,
                                         expect_source_sha256=args.expect_source_sha256,
                                         parent_sha256=args.parent_sha256)
    model, source_meta = Q.load_source(stage, parent)
    teacher = teacher_rec = None
    if args.item == "N1":
        from src.quant import qat_kd as K
        teacher, teacher_rec = K.teacher_of_record(args.teacher_ckpt, args.teacher_ckpt_sha256)
    return {"guard": st, "clip": clip, "stage": stage, "parent": parent, "model": model, "source_meta": source_meta,
            "teacher": teacher, "teacher_record": teacher_rec}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    flags = {f for f in FLAG_NAMES if getattr(args, f)}
    from src.quant import qat as Q
    try:
        g = gates(args, flags)
    except (G.GuardError, X.ExploreQATError, Q.QATRefused) as e:
        print(f"RESULT: REFUSED {e}. Nothing was written.")
        return Q.EXIT_REFUSED
    except Exception as e:                                   # noqa: BLE001
        print(f"RESULT: ERROR [{type(e).__name__}] {e}. Nothing was written.")
        return Q.EXIT_ERROR
    st = g["guard"]
    out = st["out_dir"]
    model = g["model"]
    substitution = None
    extra_x = {}
    if args.item == "N5":
        from src.explore.fp32_head import PRECISION_LABEL, apply_fp32_head, mac_share
        import copy
        extra_x["x_mac_share"] = mac_share(copy.deepcopy(model))
        apply_fp32_head(model)
        extra_x["x_precision_label"] = PRECISION_LABEL
    doc = dict(st["run_meta"])
    doc.update({"parent_check": g["parent"], "clip_binding": g["clip"], "quant_stage": g["stage"]["name"],
                "teacher_record": g["teacher_record"], **extra_x})
    path = G.write_run_meta(out, doc)
    inner = out / args.run_id
    extra = {"clip_selection": g["clip"]["clip_selection"], "u4_pilot": False, "e6_selections": None,
             "data_root": st["isolation"].get("data_root"), "isolation_counts": st["isolation"].get("counts"),
             "expect_head": args.expect_head, "expect_source_sha256": args.expect_source_sha256 or args.parent_sha256,
             "cwd_projection_loaded": False, "x_exploratory": True, "x_amendment": G.AMENDMENT, "x_item": args.item,
             "x_flags": sorted(flags), "x_change": X.CHANGES[args.item], "x_run_meta_sha256": G.sha256_file(path),
             "x_artifact_status": G.ARTIFACT_STATUS, **extra_x}
    try:
        if args.item == "N1":
            summary = X.run_qat_logit_kd(mode="real", model=model, teacher=g["teacher"], source_meta=g["source_meta"],
                                         out_dir=inner, seed=args.seed, clip_norm=X.CLIP_OF_RECORD,
                                         clip_source=g["clip"]["clip_source"], device=args.device,
                                         num_workers=args.num_workers, parent=g["parent"], extra_meta=extra)
        elif args.item == "N2":
            from src.explore.patching import substituted
            from src.explore.sites import SITE_NAMES, SiteMatcher, set_tf32
            holder = {}
            pinned_prepare = Q.prepare_qat_model

            def prepare_with_sites(m, *, select_backend=True):
                prepared = pinned_prepare(m, select_backend=select_backend)
                holder["matcher"] = SiteMatcher(prepared, SITE_NAMES)
                return prepared

            set_tf32(False)
            with substituted(Q, "prepare_qat_model", prepare_with_sites, item="N2",
                             why="install the ten input-grid site fake-quants on the prepared student") as rec:
                substitution = rec
                extra["x_substitution"] = rec
                summary = Q.run_qat(stage=g["stage"]["key"], mode="real", model=model, source_meta=g["source_meta"],
                                    out_dir=inner, seed=args.seed, clip_norm=X.CLIP_OF_RECORD,
                                    clip_source=g["clip"]["clip_source"], device=args.device,
                                    num_workers=args.num_workers, parent=g["parent"], extra_meta=extra)
            print(f"[n2] site hook calls: {holder['matcher'].calls}")
        else:
            summary = Q.run_qat(stage=g["stage"]["key"], mode="real", model=model, source_meta=g["source_meta"],
                                out_dir=inner, seed=args.seed, clip_norm=X.CLIP_OF_RECORD,
                                clip_source=g["clip"]["clip_source"], device=args.device,
                                num_workers=args.num_workers, parent=g["parent"], extra_meta=extra)
    except (X.ExploreQATError, Q.QATRefused) as e:
        print(f"RESULT: REFUSED {e}")
        return Q.EXIT_REFUSED
    except Q.QATStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return Q.EXIT_STOP
    except Exception as e:                                   # noqa: BLE001 -- after the run_meta row: run_abort
        print(f"RESULT: ABORTED [exception] -- {type(e).__name__}: {e}")
        return Q.EXIT_ABORTED
    print(f"[x] {args.item}: {G.CLAIM_LINE.format(seeds=args.seed)}"
          + (f"; substitution {substitution['original']}" if substitution else ""))
    print(Q.result_line(summary).replace("RESULT: QAT", f"RESULT: X-{args.item} QAT"))
    return Q.EXIT_OK if summary["complete"] else Q.EXIT_ABORTED


if __name__ == "__main__":
    raise SystemExit(main())
