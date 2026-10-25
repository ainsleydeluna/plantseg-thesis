#!/usr/bin/env python3
"""Lane L-AM3, B-2: E6-KD's check-run-meta (plan §1.4 as LB18 and LB23 amend it; E6-KD's other block-B steps are
Q2's: scripts/qat_epoch_eval.py convert and score, scripts/select_qat_epoch.py under the E6-KD rules file).

    check-run-meta  --run-dir R --seed 42 --grad-clip-norm C --expect-head P --expect-source-sha256 H
                    --clip-selection F --clip-selection-sha256 H
                    --lambda-selection REL --lambda-selection-sha256 H --alpha-selection REL --alpha-selection-sha256 H
                    --trigger REL --trigger-sha256 H --projection-sha256 H --teacher-ckpt T --teacher-ckpt-sha256 H

The launched run's run_meta row (the first row of its qat_telemetry.jsonl) against the E6-KD launch of record, one
rule per key: Q2's QAT profile (scripts/qat_epoch_eval.py profile_mismatches: the recipe, the census, DL-85's
never-observed modules, 12 workers, the clip binding), except where E6-KD differs by design, and E6-KD's own keys:
the KD image (DL-87), two optimizer groups, E6's λ and α binding, the weights and T (AM-3), the ramp (DL-39), the
projection of record (DL-51), the teacher of record and its M4-KD stream (DL-88), and the trigger binding. Run on
the pod within 5 minutes of the launch. The stage is E6-KD; there is no --stage flag (LB23).

Ends with one RESULT line: CHECK-RUN-META PASS (exit 0), STOP (exit 1), REFUSED (exit 2), INCOMPLETE (exit 3) or
ERROR (exit 4).
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

import scripts.qat_epoch_eval as QEE  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant import qat_kd as KD  # noqa: E402
from src.quant.e6kd_trigger import read_trigger_record, trigger_binding  # noqa: E402
from src.quant.qat import EXIT_ABORTED, EXIT_ERROR, EXIT_OK, EXIT_REFUSED, EXIT_STOP, QATRefused, QATStop  # noqa: E402

# Q2's profile keys E6-KD replaces with its own rule (they differ by design: the KD image, two optimizer groups, and
# the λ/α selections Q2 checks only for E6)
REPLACED_Q2_KEYS = ("image_digest", "optimizer.param_groups", "e6_selections")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("check-run-meta", help="compare a launched E6-KD run's run_meta row with the launch of record")
    m.add_argument("--run-dir", required=True)
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--grad-clip-norm", type=float, required=True)
    m.add_argument("--expect-head", required=True)
    m.add_argument("--expect-source-sha256", required=True)
    m.add_argument("--clip-selection", required=True)
    m.add_argument("--clip-selection-sha256", required=True)
    m.add_argument("--lambda-selection", required=True)
    m.add_argument("--lambda-selection-sha256", required=True)
    m.add_argument("--alpha-selection", required=True)
    m.add_argument("--alpha-selection-sha256", required=True)
    m.add_argument("--trigger", required=True)
    m.add_argument("--trigger-sha256", required=True)
    m.add_argument("--projection-sha256", required=True)
    m.add_argument("--teacher-ckpt", required=True)
    m.add_argument("--teacher-ckpt-sha256", required=True)
    return ap


def q2_args(args) -> argparse.Namespace:
    """The flags Q2's profile reads, for stage E6-KD (a non-pilot run bound to clip_selection.json)."""
    return argparse.Namespace(stage=KD.STAGE_NAME, seed=args.seed, grad_clip_norm=args.grad_clip_norm,
                              expect_head=args.expect_head, expect_source_sha256=args.expect_source_sha256,
                              u4_pilot=False, clip_selection=args.clip_selection,
                              clip_selection_sha256=args.clip_selection_sha256,
                              lambda_selection=args.lambda_selection,
                              lambda_selection_sha256=args.lambda_selection_sha256,
                              alpha_selection=args.alpha_selection, alpha_selection_sha256=args.alpha_selection_sha256,
                              condition_record=None, condition_sha256=None, condition_dl=None)


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def e6kd_profile_mismatches(meta: dict, args, *, repo_root: Path = REPO) -> dict:
    """Every constant of E6-KD's launch of record against its run_meta row. {key: (got, want)}."""
    bad = {k: v for k, v in QEE.profile_mismatches(meta, q2_args(args), repo_root=repo_root).items()
           if k not in REPLACED_Q2_KEYS}
    pd = KD.preflight_distill()
    if meta.get("image_digest") != pd.KD_IMAGE_DIGEST:
        bad["image_digest"] = (meta.get("image_digest"), pd.KD_IMAGE_DIGEST)
    opt = meta.get("optimizer") or {}
    if KD.kd_optimizer_error(opt):
        bad["optimizer"] = (opt, KD.kd_optimizer_error(opt))
    # E6's λ and α binding (O2), the trigger binding and the weights read from the selections
    try:
        lam = Q.read_tracked_selection(args.lambda_selection, args.lambda_selection_sha256, Q.LAMBDA_SELECTION_FORMAT,
                                       "lambda", repo_root)
        alpha = Q.read_tracked_selection(args.alpha_selection, args.alpha_selection_sha256, Q.ALPHA_SELECTION_FORMAT,
                                         "alpha", repo_root)
        parent = meta.get("parent") or {}
        Q.e6_parent_binding(parent, args.seed, lam, alpha)
        sels = meta.get("e6_selections") or {}
        for what, doc in (("lambda_selection", lam), ("alpha_selection", alpha)):
            if (sels.get(what) or {}).get("sha256") != doc["sha256"]:
                bad[f"e6_selections.{what}"] = ((sels.get(what) or {}).get("sha256"), doc["sha256"])
        trig = read_trigger_record(args.trigger, args.trigger_sha256, repo_root=repo_root)
        want_bound = trigger_binding(trig, parent=parent, clip_norm=args.grad_clip_norm,
                                     clip_selection_sha256=args.clip_selection_sha256, lam=lam, alpha=alpha)
        if not _same(meta.get("trigger_binding"), want_bound):
            bad["trigger_binding"] = (meta.get("trigger_binding"), want_bound)
        if meta.get("trigger_sha256") != args.trigger_sha256 or not _same(meta.get("trigger"), KD.trigger_meta(trig)):
            bad["trigger"] = ((meta.get("trigger_sha256"), meta.get("trigger")), args.trigger_sha256)
        e3w = {"logit_kd": lam["winner"].get("lambda"), "cwd_feat": alpha["winner"].get("alpha"),
               "cwd_logit": KD.TD.BETA_CWD_LOGIT}
        got_e3w = meta.get("e3_weights")
        # by value: the selections and E3's run_meta may record an integral weight as 25 or 25.0
        if not (isinstance(got_e3w, dict) and sorted(got_e3w) == sorted(e3w)
                and all(_number(got_e3w[t]) and _number(e3w[t]) and float(got_e3w[t]) == float(e3w[t]) for t in e3w)):
            bad["e3_weights"] = (got_e3w, e3w)
        want_w = KD.e6kd_weights(e3w)
        if not _same(meta.get("weights"), want_w):
            bad["weights"] = (meta.get("weights"), want_w)
    except QATRefused as e:
        bad["e6kd_binding"] = (e.code, str(e))
    proj = meta.get("projection_source") or {}
    parent = meta.get("parent") or {}
    want_proj = {"sha256": args.projection_sha256, "format": "cwd_projection/1", "stage": "E3",
                 "checkpoint": parent.get("checkpoint"), "paired": True, "optimizer_state_loaded": False,
                 "in_channels": 160, "out_channels": 320, "params": 51200}
    wrong_proj = {k: proj.get(k) for k, v in want_proj.items() if not _same(proj.get(k), v)}
    if wrong_proj or type(proj.get("iter")) is not int:
        bad["projection_source"] = ({**wrong_proj, "iter": proj.get("iter")}, want_proj)
    # the teacher of record (DL-88) and its stream
    prov = meta.get("teacher_provenance") or {}
    rules = pd.provenance_rules({"teacher_ckpt": str(Path(args.teacher_ckpt).expanduser().resolve())})
    wrong_t = sorted(k for k, rule in rules.items() if not rule(prov.get(k)))
    if wrong_t or sorted(prov) != sorted(rules) or args.teacher_ckpt_sha256 != pd.TEACHER_SHA256:
        bad["teacher_provenance"] = ({k: prov.get(k) for k in wrong_t}, f"DL-88's values ({pd.TEACHER_SHA256})")
    nmf = meta.get("teacher_nmf") or {}
    if not (nmf.get("policy") == KD.NMF_POLICY and nmf.get("seed") == 42 and nmf.get("draws") == 0):
        bad["teacher_nmf"] = (nmf, {"policy": KD.NMF_POLICY, "seed": 42, "draws": 0})
    want = {"schema": KD.TELEMETRY_SCHEMA, "lane": KD.LANE, "stage": KD.STAGE_NAME, "source_stage": "E3",
            "distillation": True, "objective": KD.OBJECTIVE, "e6kd": True, "weight_factor": KD.WEIGHT_FACTOR,
            "descriptive": True, "holm_family": False, "terms": {t: True for t in KD.TERMS},
            "T_logit": KD.TD.T_LOGIT, "T_cwd": KD.TD.T_CWD, "cwd_C": KD.TD.CWD_C_FEAT,
            "logit_kd_semantics": KD.LOGIT_KD_SEMANTICS, "ramp_iters": Q.REAL_STEPS_PER_EPOCH,
            "ramp_rule": KD.RAMP_RULE, "projection_params": 51200, "projection_fake_quant_modules": 0,
            "projection_in_student": False, "teacher_mock": False, "abort_rule": KD.ABORT_RULE_TEXT,
            "u4_pilot": False, "clip_source": "clip_selection"}
    for k, v in want.items():
        if not _same(meta.get(k), v) or type(meta.get(k)) is not type(v):
            bad[k] = (meta.get(k), v)
    if "cwd_projection_loaded" in meta:                       # E6's flag: this run trains its projection
        bad["cwd_projection_loaded"] = (meta.get("cwd_projection_loaded"), "absent")
    return bad


def cmd_check_run_meta(args) -> int:
    # SL-1: each path's given and resolved forms, before the path is read (the profile resolves the teacher's)
    if Q.names_test(args.run_dir):
        raise QATRefused("run_dir_forbidden_path", f"--run-dir {args.run_dir!r} holds a name SL-1 forbids")
    if Q.names_test(args.teacher_ckpt) or Q.names_test(Path(args.teacher_ckpt).expanduser()):
        raise QATRefused("teacher_forbidden_path", f"--teacher-ckpt {args.teacher_ckpt!r} holds a name SL-1 forbids")
    tel = Path(args.run_dir) / Q.TELEMETRY_NAME
    if not tel.is_file():
        raise A.QATIncomplete("telemetry_missing", f"{tel} is missing (the run has not written its run_meta row)")
    with open(tel, encoding="utf-8") as fh:
        first = fh.readline()

    def non_strict(c):
        raise ValueError(f"non-strict JSON constant {c}")
    try:
        meta = json.loads(first, parse_constant=non_strict)
    except ValueError as e:
        raise QATRefused("run_meta_rows", f"{tel}'s first line is not a strict JSON row ({e})") from e
    if not isinstance(meta, dict) or meta.get("event") != "run_meta":
        raise QATRefused("run_meta_rows", f"{tel}'s first row is not the run_meta row")
    bad = e6kd_profile_mismatches(meta, args)
    if bad:
        for k, (g, w) in sorted(bad.items()):
            print(f"  {k}: run_meta {g!r}, expected {w!r}")
        raise QATStop("run_meta_profile", f"{len(bad)} run_meta value(s) differ from the E6-KD launch of record: "
                                          f"{sorted(bad)}. Stop the run")
    print(f"[check-run-meta] {meta['stage']} seed {meta['seed']} clip {meta['clip_norm']} ({meta['clip_source']}), "
          f"weights {meta['weights']}, HEAD {meta['git_head']}, every profile value as recorded")
    print("RESULT: CHECK-RUN-META PASS")
    return EXIT_OK


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_check_run_meta(args)
    except QATRefused as e:
        print(f"RESULT: REFUSED [{e.code}] -- {e}")
        return EXIT_REFUSED
    except A.QATIncomplete as e:
        print(f"RESULT: INCOMPLETE [{e.code}] -- {e}")
        return EXIT_ABORTED
    except QATStop as e:
        print(f"RESULT: STOP [{e.code}] -- {e}")
        return EXIT_STOP
    except Exception as e:                                   # noqa: BLE001 -- reported, exit 4
        print(f"RESULT: ERROR [{type(e).__name__}] -- {e}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
