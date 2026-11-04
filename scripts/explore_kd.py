#!/usr/bin/env python3
"""N3, A1'' and A2 (lane L-AM24): the exploratory KD entry (E2 / E3 seed 42's recipe, one named change).

    python -B -I scripts/explore_kd.py --real-run --confirm-real-run --item N3|A1pp|A2
        --teacher-ckpt <absolute .../iter_24000.pth> --teacher-ckpt-sha256 <N3: R2's d5085961... | the record's>
        --num-workers 12 --expect-head <40-hex HEAD> --run-id x-... --out-dir <root>/<run-id>
        [A1pp: --alpha-from <N4 item dir>/n4_terms.json --alpha-from-sha256 <its sha256>]

AM-24 v2.4: every launch has the futility flag OFF and no item stops early on a VAL value. The flag stays in the code
(audit §6.3's optional rule: --futility-stop --futility-comparator-telemetry <E2 s42's or E3 s42's telemetry>
--futility-comparator-sha256 <the TEST-day manifest's completion record>) and is not used by any launch.

Order: the item's rules (seed 42; the item's teacher by sha256; A1'''s alpha from N4); the guard (VAL, the data
root's TRAIN/VAL-only check, x- run ID, output outside the repository and official folders); HEAD = --expect-head
with the pinned KD modules and the lane's files clean; CUDA present and uninitialised; TF32 at torch defaults;
the teacher path absolute, outside the repository and named iter_24000.pth; the record's teacher config by sha256.
Nothing is written before these pass. Then <out-dir>/x_run_meta.json, and the pinned trainer's main() with the
trainer directory <out-dir>/<run-id> (A1'': <run-id>_alpha<value>), which applies its own real-run gates unchanged.
Exit codes: train_distill's (0 PASS, 1 FAIL, 2 refused, 3 aborted), 5 stopped by the futility rule.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.explore import guard as G                                   # noqa: E402
from src.explore import kd_explore as K                              # noqa: E402

_HEX40 = re.compile(r"[0-9a-f]{40}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--item", required=True, choices=list(K.ITEMS))
    p.add_argument("--seed", type=int, default=K.SEED)
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--teacher-ckpt-sha256", required=True)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--expect-head", default=None)
    p.add_argument("--alpha-from", default=None)
    p.add_argument("--alpha-from-sha256", default=None)
    p.add_argument("--futility-stop", action="store_true",
                   help="audit §6.3's optional 40k rule; OFF in every AM-24 v2.4 launch (no item stops early)")
    p.add_argument("--futility-comparator-telemetry", default=None)
    p.add_argument("--futility-comparator-sha256", default=None)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="val")
    return p


def gates(args) -> dict:
    import torch
    from src.training import train_distill as TD
    if not (args.real_run and args.confirm_real_run):
        raise K.ExploreKDError("real_run_flags", "an exploratory KD launch needs --real-run --confirm-real-run")
    if args.seed != K.SEED:
        raise K.ExploreKDError("seed", f"{args.item} runs seed {K.SEED} only (audit §6.1), not {args.seed}")
    want_sha, who = K.teacher_expectation(args.item)
    if args.teacher_ckpt_sha256 != want_sha:
        raise K.ExploreKDError("teacher_not_item", f"{args.item}'s teacher is {who}, sha256 {want_sha}; got "
                                                   f"{args.teacher_ckpt_sha256}")
    if (args.item == "A1pp") != bool(args.alpha_from and args.alpha_from_sha256):
        raise K.ExploreKDError("alpha_source_flags", "--alpha-from and --alpha-from-sha256 go with A1'' and only A1''")
    fut = (args.futility_comparator_telemetry, args.futility_comparator_sha256)
    if args.futility_stop != all(fut) or (not args.futility_stop and any(fut)):
        raise K.ExploreKDError("futility_flags", "--futility-stop takes --futility-comparator-telemetry and "
                                                 "--futility-comparator-sha256, and they need it")
    st = G.guarded_start(
        item=args.item, run_id=args.run_id, out_dir=args.out_dir, split=args.split, allowed_splits=G.VAL_ONLY,
        input_paths={"--teacher-ckpt": args.teacher_ckpt, "--alpha-from": args.alpha_from,
                     "--futility-comparator-telemetry": args.futility_comparator_telemetry},
        seeds=[K.SEED], pinned_paths=K.KD_CODE_PATHS, kd_reuse_check=True,
        recipe={"recipe_of_record": "E3 seed 42's (A1'')" if args.item == "A1pp" else "E2 seed 42's",
                "audit": "§6.1: everything in the recipe except the one named change", "change": K.CHANGES[args.item],
                "stage": K.ITEM_STAGE[args.item], "lambda_logit": K.ITEM_LAMBDA[args.item], "seed": K.SEED,
                "teacher": who, "teacher_sha256": want_sha, "comparators": K.COMPARATORS,
                "futility": ({"iter": K.FUTILITY_ITER, "margin": str(K.FUTILITY_MARGIN),
                              "comparator_slot": K.COMPARATOR_SLOT[args.item]} if args.futility_stop else None)},
        write=False)
    code = st["run_meta"]
    if not args.expect_head or not _HEX40.fullmatch(args.expect_head) or code["git_head"] != args.expect_head:
        raise K.ExploreKDError("expect_head", f"--expect-head {args.expect_head!r} is not HEAD {code['git_head']!r}")
    if code["pinned_clean_at_head"] is not True or code["lane_clean_at_head"] is not True:
        raise K.ExploreKDError("code_not_clean", "the pinned KD modules or the lane's files are not clean at HEAD")
    if args.num_workers is None or args.num_workers < 1:
        raise K.ExploreKDError("num_workers", "--num-workers must be given and at least 1 (the record's 12)")
    if not torch.cuda.is_available():
        raise K.ExploreKDError("cuda_required", "a real KD run needs CUDA")
    if torch.cuda.is_initialized():
        raise K.ExploreKDError("cuda_initialized", "CUDA is already initialised")
    tf = TD.tf32_gate_error(TD.tf32_state())
    if tf:
        raise K.ExploreKDError("tf32_not_default", tf)
    tp = Path(args.teacher_ckpt)
    if not os.path.isabs(args.teacher_ckpt) or tp.name != K.TEACHER_FILE or tp.resolve().name != K.TEACHER_FILE \
            or G.REPO.resolve() in tp.resolve().parents:
        raise K.ExploreKDError("teacher_ckpt_path", f"--teacher-ckpt {args.teacher_ckpt!r}: absolute, outside the "
                                                    f"repository, named {K.TEACHER_FILE}")
    from src.quant.qat_kd import preflight_distill
    cfg_sha = G.sha256_file(TD.DEFAULT_TEACHER_CONFIG)
    if cfg_sha != preflight_distill().TEACHER_VALUES["config_sha256"]:
        raise K.ExploreKDError("teacher_config", f"the default teacher config's sha256 {cfg_sha} is not the record's")
    alpha = K.a1pp_alpha(args.alpha_from, args.alpha_from_sha256) if args.item == "A1pp" else None
    comp = (K.comparator_at(args.futility_comparator_telemetry, args.futility_comparator_sha256, args.item)
            if args.futility_stop else None)
    return {"guard": st, "alpha": alpha, "comparator": comp, "teacher_label": who, "teacher_config_sha256": cfg_sha}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        g = gates(args)
    except (G.GuardError, K.ExploreKDError) as e:
        print(f"RESULT: REFUSED {e}. Nothing was written.")
        return G.EXIT_REFUSED
    except Exception as e:                                   # noqa: BLE001
        print(f"RESULT: ERROR [{type(e).__name__}] {e}. Nothing was written.")
        return G.EXIT_ERROR
    from src.explore.patching import substituted
    from src.training import train_distill as TD
    st = g["guard"]
    out = st["out_dir"]
    alpha = None if g["alpha"] is None else g["alpha"]["value"]
    inner = out / K.trainer_dir_name(args.item, args.run_id, alpha)
    doc = dict(st["run_meta"])
    doc.update({"alpha_a1pp": g["alpha"], "futility_comparator": g["comparator"], "trainer_dir": str(inner),
                "teacher_config_sha256": g["teacher_config_sha256"]})
    subs = []
    with contextlib.ExitStack() as stack:
        if args.item == "A1pp":
            subs.append(stack.enter_context(substituted(
                TD, "alpha_gate_error", K.alpha_gate_admitting(TD.alpha_gate_error, alpha), item="A1pp",
                why=f"admit exactly the N4 alpha {g['alpha']['repr']} (off the AM-16 grid)")))
        if args.item == "A2":
            from src.explore.dist_loss import dist_distillation_losses
            subs.append(stack.enter_context(substituted(
                TD, "distillation_losses", dist_distillation_losses, item="A2",
                why="DIST (beta = gamma = 1, tau = 1, weight 1) in place of lambda * KL")))
        if args.futility_stop:
            subs.append(stack.enter_context(substituted(
                TD, "_jsonl", K.futility_jsonl(TD._jsonl, comparator=g["comparator"], item=args.item), item=args.item,
                why="the optional 40k futility stop (audit §6.3): record and stop, nothing else")))
        doc["substitutions"] = subs
        G.write_run_meta(out, doc)
        argv_t = K.trainer_argv(args.item, teacher_ckpt=args.teacher_ckpt, teacher_sha256=args.teacher_ckpt_sha256,
                                num_workers=args.num_workers, ckpt_dir=inner, alpha=alpha)
        print(f"[x] {args.item}: train_distill main {' '.join(argv_t)}")
        try:
            rc = TD.main(argv_t)
        except K.FutilityStop as e:
            G.write_exclusive(out / "x_futility.json", G.json_bytes({**e.record, "result": str(e),
                                                                     "status": "stopped: futility rule"}))
            print(f"RESULT: X-{args.item} {e}; its rows are evidence, not a result")
            return K.EXIT_FUTILITY
    G.write_exclusive(out / "x_result.json", G.json_bytes({"item": args.item, "trainer_exit": rc,
                                                           "trainer_dir": str(inner)}))
    print(f"RESULT: X-{args.item} trainer exit {rc} ({G.CLAIM_LINE.format(seeds=K.SEED)})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
