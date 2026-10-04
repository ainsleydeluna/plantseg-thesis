#!/usr/bin/env python3
"""Invariance smoke for the per-term distillation switches (L-AM17B-FG) and the alpha grid
(L-AM16-ALPHA). CPU, synthetic TRAIN/VAL data, stub teacher by default; no dataset, checkpoint or GPU.

Lane checks (docs/lane_specs/part2.md lanes 1-2; smoke convention per errata E-7):

default mode, on this checkout (or --code-root DIR):
  H0   repeat determinism: E3 run twice, every step JSON byte-identical (the wall-clock fields are
       dropped by the harness before writing and comparing; L-KD-HARDEN item 4e).
  d2   optimizer parameters = the student only (E2, G) or the student + 51,200 (E3, A, F).
  d6   exact telemetry key sets per stage (E2, E3, A, F, G): train rows with ce, dice, the grad-norm
       split and E1's wall-clock fields; val rows in E1's schema; run_end with the wall-clock and
       GPU-hour fields (L-KD-HARDEN item 4).
  d7   one teacher forward per step for every stage; the same `_build_bases` count, NMF stream
       position and teacher output hashes at every step for every stage.
  d8   no projection keys in any student state_dict and no projection in the checkpoint payload;
       projection.pt exists iff cwd_feat and pairs with the latest checkpoint; the E6/E7 loader
       takes E3's checkpoint as it is (nothing to strip) and refuses the other stages.
  R    projection.pt restores the projection and its optimizer group bit for bit.
  run_meta: terms, projection_params, and lambda/alpha/beta/T only for instantiated terms; E1's
       provenance and recipe keys plus ramp_iters, class_weights_sha256, tf32, teacher_provenance and
       teacher_mock, with their dry-harness values (L-KD-HARDEN item 5); best.json in E1's schema.

--cross-commit OLD NEW  (or --old-root DIR --new-root DIR for trees exported beforehand):
  d4   E2 and E3 step JSONs of NEW byte-identical to OLD: telemetry (losses, ramp, lr, grad norm),
       gradient and weight hashes, teacher output hashes, NMF position, RNG state. Fields the lane
       adds are named with --declared-new (removed from NEW only; an OLD field always has to match).
       E3 is also compared with the dry global-norm clipping path on (max_norm 1.0).
  L2d3 NEW E3 with an explicit alpha = 50 byte-identical to OLD E3.
  Guards: OLD and NEW must be different code (train_distill.py bytes differ), and a negative
       control (NEW E3 at alpha 25) must differ from OLD E3 at every step after the first.
  OLD and NEW are exported with `git archive <commit> -- <explicit paths>`; NEW may be WORKTREE
  (this checkout, for development).

N = 8 steps with the stub teacher (one full epoch of the 16-image set at batch 2, so the whole
first-epoch ramp is exercised); the acceptance run with --teacher real uses --steps 4. Outputs go to
--work-dir (a new temp dir by default), never into the repository.

--teacher real (L-CKPT-GUARD) needs --teacher-ckpt and --teacher-config as absolute paths and
--teacher-ckpt-sha256: the driver hashes the checkpoint before any worker starts, each worker hands
the value to load_frozen_teacher when the code under test takes it, and every worker's run_meta must
record that hash as teacher_provenance.ckpt_sha256.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs.e1_student import E1_STUDENT  # noqa: E402
from scripts.invariance_harness import (BATCH, N_TRAIN, compare_step_dirs, export_commit,  # noqa: E402
                                        make_synthetic_dataset, run_worker, sha256_file)

TERMS = ("logit_kd", "cwd_feat", "cwd_logit")
STAGE_TERMS = {"e2": (True, False, False), "e3": (True, True, True), "a": (False, True, True),
               "f": (False, True, False), "g": (False, False, True)}
PROJECTION_NUMEL = 160 * 320                     # bias-free 1x1 conv, student C5 -> teacher Stage 3
STUDENT_PARAMS = 2_933_688                       # AM-17 item 10(b)
BEST_JSON_KEYS = ["best_ckpt", "best_val_miou_all_class"]      # train_e1.write_best_pointer
DATA_LAYOUT = ["annotations/train", "annotations/val", "images/train", "images/val"]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def outside_repo(p: Path) -> Path:
    p = p.resolve()
    if p == REPO or REPO in p.parents:
        raise SystemExit(f"--work-dir must be outside the repository: {p}")
    return p


def expected_keys(flags) -> list[str]:
    """Train-row keys in order (L-KD-HARDEN item 4): the 44c05dc keys with ce/dice after sup, the
    grad-norm split after grad_norm, and E1's wall-clock fields last."""
    cf = flags[1]
    return (["event", "iter", "loss", "sup", "ce", "dice"] + [t for t, on in zip(TERMS, flags) if on]
            + ["ramp", "lr", "grad_norm", "grad_norm_student"] + (["grad_norm_projection"] if cf else [])
            + ["wall_clock", "iter_seconds", "samples_per_sec"])


VAL_KEYS = ["event", "iter", "all_class_miou", "disease_only_miou_PROVISIONAL", "per_class_iou",
            "per_class_eligible", "n_eligible_classes", "val_batches", "val_total_px", "val_seconds",
            "wall_clock"]                                    # train_e1's val row (train_e1.py:492-500)
RUN_END_KEYS = ["event", "iter", "best_val_miou_all_class", "best_ckpt", "checks_passed",
                "wall_clock_start", "wall_clock_end", "wall_seconds", "train_seconds", "gpu_hours"]
# E1's run_meta keys, with persistent_workers (E1's TRAIN-loader argument) after num_workers (item 5)
E1_META_KEYS = ["wall_clock", "git_head", "git_head_source", "image_digest", "torch", "numpy", "device",
                "cuda_available", "gpu_name", "num_workers", "persistent_workers", "val_interval",
                "max_val_batches", "learning_rate", "momentum", "weight_decay", "lr_power", "poly_horizon",
                "grad_clip_norm", "used_pretrained", "params", "ignore_index"]
KD_META_KEYS = ["ramp_iters", "class_weights_sha256", "tf32", "teacher_provenance", "teacher_mock"]


def host_facts() -> dict:
    """What run_meta records about the HOST rather than the run's device: the workers train on the CPU,
    but cuda_available, gpu_name and the TF32 state describe the machine and its environment, which the
    workers inherit. A worker that set a TF32 flag would differ from this fresh-process reading."""
    import os

    import torch
    cuda = torch.cuda.is_available()
    return {"cuda_available": cuda, "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "tf32": {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                     "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                     "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")}}


def check_recorded_teacher_sha256(runs: dict, a) -> None:
    """--teacher real (L-CKPT-GUARD): every worker's run_meta records the checkpoint the driver hashed."""
    if a.teacher != "real":
        return
    for k, r in runs.items():
        rows = (r["summary"] or {}).get("run_meta") or [{}]
        prov = rows[0].get("teacher_provenance") or {}
        check(f"teacher_sha256_recorded_{k}", prov.get("ckpt_sha256") == a.teacher_ckpt_sha256,
              f"{prov.get('ckpt_sha256')} vs {a.teacher_ckpt_sha256}")


def load_steps(d: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("step_*.json"))]


# ------------------------------------------------------------------------------ single commit
def single(work: Path, data_root: Path, code_root: Path, a) -> dict:
    runs = {}
    for s in a.stages:
        runs[s] = run_worker(code_root=code_root, data_root=data_root, out_dir=work / f"run_{s}",
                             stage=s, steps=a.steps, val_interval=a.val_interval, teacher=a.teacher,
                             teacher_ckpt=a.teacher_ckpt,
                             teacher_ckpt_sha256=a.teacher_ckpt_sha256,
                             teacher_config=a.teacher_config, threads=a.threads)
    if not a.no_repeat and "e3" in a.stages:
        runs["e3_repeat"] = run_worker(code_root=code_root, data_root=data_root,
                                       out_dir=work / "run_e3_repeat", stage="e3", steps=a.steps,
                                       val_interval=a.val_interval, teacher=a.teacher,
                                       teacher_ckpt=a.teacher_ckpt,
                                       teacher_ckpt_sha256=a.teacher_ckpt_sha256,
                                       teacher_config=a.teacher_config, threads=a.threads)
        rep = compare_step_dirs(work / "run_e3", work / "run_e3_repeat")
        check("H0_repeat_e3_step_jsons_byte_identical", rep["identical"],
              f"{rep['n_old']} steps; mismatches={rep['mismatches'][:1]}")

    ref = load_steps(work / "run_e3") if "e3" in a.stages else None
    for s in a.stages:
        flags = STAGE_TERMS[s]
        lk, cf, cl = flags
        r, sm = runs[s], runs[s]["summary"] or {}
        steps = load_steps(work / f"run_{s}")
        ok_run = (r["returncode"] == 0 and str(sm.get("trainer_result", "")).startswith("RESULT: PASS")
                  and sm.get("n_step_records") == a.steps and len(steps) == a.steps)
        check(f"run_{s}_completes", ok_run, f"rc={r['returncode']} {sm.get('trainer_result')} "
                                            f"steps={len(steps)} log={r['log']}")
        if not ok_run:
            continue
        # d6 — telemetry keys, in order
        want = expected_keys(flags)
        check(f"d6_{s}_telemetry_keys", sm["telemetry_keys"] == [want],
              f"{sm['telemetry_keys']} vs {want}")
        check(f"d6_{s}_telemetry_finite", sm["telemetry_finite"])
        # d7 — one full teacher forward per step; identical teacher and NMF stream across stages
        check(f"d7_{s}_one_teacher_forward_per_step", all(x["teacher_forwards"] == 1 for x in steps),
              str([x["teacher_forwards"] for x in steps]))
        check(f"d7_{s}_teacher_features_and_logits_every_step",
              all(x["teacher_shapes"] == [[2, 116, 64, 64], [2, 320, 32, 32]] for x in steps),
              str(steps[0]["teacher_shapes"]))
        bb = [x["build_bases_calls"] for x in steps]
        check(f"d7_{s}_build_bases_per_step", bb == [1] * a.steps
              and sm.get("stub_build_bases_calls") in (None, a.steps), f"{bb} stub_total="
                                                                      f"{sm.get('stub_build_bases_calls')}")
        if ref is not None and s != "e3":
            same = all((x["nmf"], x["teacher_logits_sha256"], x["teacher_stage3_sha256"],
                        x["build_bases_calls"]) == (y["nmf"], y["teacher_logits_sha256"],
                                                    y["teacher_stage3_sha256"], y["build_bases_calls"])
                       for x, y in zip(steps, ref))
            check(f"d7_{s}_teacher_stream_equals_e3", same,
                  "NMF position, teacher logits/Stage-3 hashes and _build_bases counts per step")
        # d2 — optimizer parameters
        groups = sm.get("optimizer_groups") or []
        numel = {g["name"]: g["numel"] for g in groups}
        want_groups = ["student"] + (["cwd_projection"] if cf else [])
        check(f"d2_{s}_optimizer_groups", [g["name"] for g in groups] == want_groups
              and numel.get("student") == sm["student_params"] == STUDENT_PARAMS
              and numel.get("cwd_projection", 0) == (PROJECTION_NUMEL if cf else 0),
              f"{groups}")
        # d8 — checkpoint layout and projection.pt
        cks = sm["checkpoints"]
        check(f"d8_{s}_no_projection_keys_in_student", bool(cks)
              and all(c["projection_keys_in_student"] == [] for c in cks))
        check(f"d8_{s}_no_projection_in_payload",
              all("cwd_projection_state_dict" not in c["keys"] for c in cks)
              and all(c["optimizer_groups"] == ["student"] for c in cks),
              f"{[(c['file'], c['optimizer_groups']) for c in cks]}")
        pj = sm["projection_pt"]
        check(f"d8_{s}_projection_pt_iff_cwd_feat", (pj is not None) == cf,
              f"projection.pt present={pj is not None} cwd_feat={cf}")
        if cf and pj is not None:
            check(f"d8_{s}_projection_pt_pairs_latest_checkpoint",
                  pj["checkpoint"] == cks[-1]["file"] and pj["state_keys"] == ["weight"]
                  and pj["numel"] == PROJECTION_NUMEL and pj["optimizer_groups"] == ["cwd_projection"],
                  json.dumps(pj))
            rs = sm["restore"]
            check(f"R_{s}_projection_restore_bitwise", rs.get("ran") and rs.get(
                "projection_weights_bitwise") and rs.get("optimizer_state_bitwise"), json.dumps(rs))
        e6 = sm["e6e7_loader"]
        check(f"d8_{s}_e6e7_loader", e6.get("ran") and (e6.get("ok") is (s == "e3")),
              json.dumps(e6))
        # run_meta and best.json
        rows = sm["run_meta"]
        rm = rows[0] if len(rows) == 1 else {}
        present = set(rm)
        ok_meta = (len(rows) == 1 and rm.get("terms") == dict(zip(TERMS, flags))
                   and rm.get("projection_params") == (PROJECTION_NUMEL if cf else 0)
                   and (("lambda_logit" in present) == lk) and (("T_logit" in present) == lk)
                   and (("logit_kd_semantics" in present) == lk)
                   and (("alpha_cwd" in present) == cf) and (("cwd_C" in present) == cf)
                   and (("beta_cwd" in present) == cl) and (("T_cwd" in present) == (cf or cl))
                   and (rm.get("alpha_cwd") == 50.0 if cf else True))
        check(f"run_meta_{s}_terms_and_instantiated_fields", ok_meta, json.dumps(rm)[:400])
        # L-KD-HARDEN item 5: E1's provenance/recipe keys and the KD carriers, with dry-harness values
        want_meta = {"mode": "dry", "seed": 42, "batch_size": BATCH, "max_iters": a.steps,
                     "num_workers": 0, "persistent_workers": False, "val_interval": a.val_interval,
                     "max_val_batches": None, "poly_horizon": E1_STUDENT["iterations"],
                     "grad_clip_norm": None, "used_pretrained": False, "params": STUDENT_PARAMS,
                     "ignore_index": 255, **{k: E1_STUDENT[k] for k in ("learning_rate", "momentum",
                                                                        "weight_decay", "lr_power")},
                     "ramp_iters": N_TRAIN // BATCH, "device": "cpu", **host_facts(), "teacher_mock": False,
                     "torch": sm.get("torch"),
                     "class_weights_sha256": sha256_file(code_root / "reports" / "e1_class_weights.json")}
        missing = [k for k in E1_META_KEYS + KD_META_KEYS if k not in rm]
        wrong = {k: (rm.get(k), v) for k, v in want_meta.items()
                 if k not in rm or rm[k] != v or type(rm[k]) is not type(v)}
        prov = rm.get("teacher_provenance", "absent")
        prov_ok = prov is None if a.teacher == "stub" else (isinstance(prov, dict)
                                                            and bool(prov.get("ckpt_sha256")))
        check(f"run_meta_{s}_e1_keys_and_kd_carriers", len(rows) == 1 and not missing and not wrong
              and prov_ok, f"missing={missing} wrong={wrong} teacher_provenance={prov!r}")
        # L-KD-HARDEN item 4: E1-schema val rows at every validation, run_end with the GPU-hour carrier
        want_val = sorted({i for i in range(1, a.steps + 1) if i % a.val_interval == 0} | {a.steps})
        check(f"val_rows_{s}_e1_schema", sm.get("val_keys") == [VAL_KEYS]
              and sm.get("val_iters") == want_val, f"{sm.get('val_keys')} at {sm.get('val_iters')} "
                                                   f"(want {want_val})")
        end = sm.get("run_end") or {}
        check(f"run_end_{s}_wall_clock_and_gpu_hours", list(end) == RUN_END_KEYS
              and end["wall_clock_end"] >= end["wall_clock_start"]
              and end["wall_seconds"] == end["wall_clock_end"] - end["wall_clock_start"]
              and end["gpu_hours"] == end["wall_seconds"] / 3600.0
              and 0.0 <= end["train_seconds"] <= end["wall_seconds"], json.dumps(end)[:300])
        check(f"best_json_{s}_e1_schema", sorted(sm["best_json"] or {}) == BEST_JSON_KEYS,
              json.dumps(sm["best_json"]))
        end, best = sm.get("run_end") or {}, sm["best_json"] or {}
        check(f"run_end_{s}_last_row_matches_best_json",
              end.get("iter") == a.steps and end.get("checks_passed") is True
              and end.get("best_val_miou_all_class") == best.get("best_val_miou_all_class")
              and end.get("best_ckpt") == Path(str(best.get("best_ckpt", ""))).name, json.dumps(end))
        check(f"no_test_surface_{s}", sm["data_layout"] == DATA_LAYOUT, str(sm["data_layout"]))
    check_recorded_teacher_sha256(runs, a)
    return runs


# ------------------------------------------------------------------------------- cross commit
def cross(work: Path, data_root: Path, old_root: Path, new_root: Path, a) -> dict:
    runs = {}
    for label, root in (("old", old_root), ("new", new_root)):
        for s in ("e2", "e3"):
            runs[f"{label}_{s}"] = run_worker(code_root=root, data_root=data_root,
                                              out_dir=work / f"{label}_{s}", stage=s, steps=a.steps,
                                              val_interval=a.val_interval, teacher=a.teacher,
                                              teacher_ckpt=a.teacher_ckpt,
                                              teacher_ckpt_sha256=a.teacher_ckpt_sha256,
                                              teacher_config=a.teacher_config, threads=a.threads)
        # the dry global-norm clipping path (a real run refuses --grad-clip-norm under AM-7)
        runs[f"{label}_e3_clip"] = run_worker(code_root=root, data_root=data_root,
                                              out_dir=work / f"{label}_e3_clip", stage="e3",
                                              steps=a.steps, val_interval=a.val_interval,
                                              teacher=a.teacher, teacher_ckpt=a.teacher_ckpt,
                                              teacher_ckpt_sha256=a.teacher_ckpt_sha256,
                                              teacher_config=a.teacher_config, grad_clip_norm=1.0,
                                              threads=a.threads)
    for tag, alpha in (("alpha50", 50.0), ("alpha25", 25.0)):
        runs[f"new_e3_{tag}"] = run_worker(code_root=new_root, data_root=data_root,
                                           out_dir=work / f"new_e3_{tag}", stage="e3", steps=a.steps,
                                           val_interval=a.val_interval, teacher=a.teacher,
                                           teacher_ckpt=a.teacher_ckpt,
                                           teacher_ckpt_sha256=a.teacher_ckpt_sha256,
                                           teacher_config=a.teacher_config, alpha=alpha,
                                           threads=a.threads)
    check_recorded_teacher_sha256(runs, a)
    for k, r in runs.items():
        sm = r["summary"] or {}
        check(f"cross_{k}_completes", r["returncode"] == 0
              and str(sm.get("trainer_result", "")).startswith("RESULT: PASS")
              and sm.get("n_step_records") == a.steps, f"rc={r['returncode']} "
                                                       f"{sm.get('trainer_result')} log={r['log']}")
    # The gate is vacuous if OLD and NEW are the same code: require the trainer bytes to differ.
    old_sha = (runs["old_e3"]["summary"] or {}).get("train_distill_sha256")
    new_sha = (runs["new_e3"]["summary"] or {}).get("train_distill_sha256")
    check("cross_old_and_new_code_differ", bool(old_sha) and bool(new_sha) and old_sha != new_sha,
          f"train_distill.py sha256 old={old_sha} new={new_sha}")
    new = tuple(a.declared_new)
    comps = {"d4_e2": compare_step_dirs(work / "old_e2", work / "new_e2", new),
             "d4_e3": compare_step_dirs(work / "old_e3", work / "new_e3", new),
             "d4_e3_clipped": compare_step_dirs(work / "old_e3_clip", work / "new_e3_clip", new),
             "lane2_d3_e3_alpha50": compare_step_dirs(work / "old_e3", work / "new_e3_alpha50", new)}
    for name, c in comps.items():
        check(f"{name}_step_jsons_byte_identical", c["identical"],
              f"{c['n_old']}/{c['n_new']} steps; first mismatch: {c['mismatches'][:1]}")
    # Negative control: the comparison must be able to fail. alpha 25 changes the feature-term weight
    # from step 2 on (the ramp is exactly 0 at step 1), so those step JSONs must differ from OLD E3.
    ctrl = compare_step_dirs(work / "old_e3", work / "new_e3_alpha25", new)
    comps["control_e3_alpha25"] = ctrl
    diff_files = [m["file"] for m in ctrl["mismatches"]]
    check("control_e3_alpha25_detected", ctrl["n_old"] == ctrl["n_new"] == a.steps
          and "step_0001.json" not in diff_files and len(diff_files) == a.steps - 1,
          f"differing steps: {diff_files}")
    return {"runs": runs, "comparisons": comps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--val-interval", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--teacher", choices=("stub", "real"), default="stub")
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--teacher-ckpt-sha256", default=None,
                    help="--teacher real: the checkpoint's sha256, checked by the driver before any "
                         "worker and against every worker's recorded teacher_provenance.ckpt_sha256")
    ap.add_argument("--teacher-config", default=None)
    ap.add_argument("--code-root", default=None, help="single-commit mode: the code root (default: "
                                                      "this checkout)")
    ap.add_argument("--cross-commit", nargs=2, metavar=("OLD", "NEW"), default=None)
    ap.add_argument("--old-root", default=None)
    ap.add_argument("--new-root", default=None)
    ap.add_argument("--stages", nargs="+", default=list(STAGE_TERMS), choices=list(STAGE_TERMS))
    ap.add_argument("--no-repeat", action="store_true")
    ap.add_argument("--declared-new", nargs="*", default=[], metavar="FIELD",
                    help="cross-commit only: dotted step-JSON fields the lane under test adds (e.g. "
                         "telemetry.ce); removed from NEW before the byte comparison, never from OLD")
    a = ap.parse_args(argv)
    if a.teacher == "real" and not (a.teacher_ckpt and a.teacher_config and a.teacher_ckpt_sha256):
        ap.error("--teacher real needs --teacher-ckpt, --teacher-config and --teacher-ckpt-sha256")
    if a.teacher == "real":
        if not (Path(a.teacher_ckpt).is_absolute() and Path(a.teacher_config).is_absolute()):
            ap.error("--teacher-ckpt and --teacher-config must be absolute: each worker runs with its "
                     "working directory at the output folder's parent")
        if sha256_file(Path(a.teacher_ckpt)) != a.teacher_ckpt_sha256:
            ap.error(f"--teacher-ckpt {a.teacher_ckpt} does not hash to --teacher-ckpt-sha256 "
                     f"{a.teacher_ckpt_sha256}; no worker was started")
    if bool(a.old_root) != bool(a.new_root) or (a.old_root and a.cross_commit):
        ap.error("use either --cross-commit OLD NEW or both --old-root and --new-root")

    work = outside_repo(Path(a.work_dir) if a.work_dir else
                        Path(tempfile.mkdtemp(prefix="k1_invariance_")))
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = make_synthetic_dataset(work / "data")
    out = {"mode": None, "steps": a.steps, "teacher": a.teacher, "threads": a.threads,
           "declared_new": list(a.declared_new),
           "data_manifest_sha256": data["manifest_sha256"], "work_dir": str(work)}
    print("=" * 78)
    print(f"DISTILL INVARIANCE SMOKE — CPU, synthetic 16+4 set, batch 2, seed 42, "
          f"teacher={a.teacher}, N={a.steps}")
    print(f"work dir: {work}")
    print("=" * 78)

    if a.cross_commit or a.old_root:
        out["mode"] = "cross-commit"
        if a.cross_commit:
            old, new = a.cross_commit
            old_root = work / "export_old"
            out["old"] = export_commit(REPO, old, old_root)
            if new.upper() == "WORKTREE":
                new_root, out["new"] = REPO, {"commit": "WORKTREE", "root": str(REPO)}
            else:
                new_root = work / "export_new"
                out["new"] = export_commit(REPO, new, new_root)
        else:
            old_root, new_root = Path(a.old_root).resolve(), Path(a.new_root).resolve()
            out["old"], out["new"] = {"root": str(old_root)}, {"root": str(new_root)}
        res = cross(work, work / "data", old_root, new_root, a)
        out["comparisons"] = res["comparisons"]
    else:
        out["mode"] = "single"
        code_root = Path(a.code_root).resolve() if a.code_root else REPO
        out["code_root"] = str(code_root)
        runs = single(work, work / "data", code_root, a)
        out["summaries"] = {k: r["summary"] for k, r in runs.items()}

    out["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in results]
    out["seconds"] = round(time.time() - t0, 1)
    (work / "invariance_summary.json").write_text(json.dumps(out, indent=1, default=str),
                                                  encoding="utf-8")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail[:160]) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\nsummary: {work / 'invariance_summary.json'} ({out['seconds']} s)")
    print(f"RESULT: {'PASS' if passed == total and total else 'FAIL'} ({passed}/{total})")
    return 0 if passed == total and total else 1


if __name__ == "__main__":
    raise SystemExit(main())
