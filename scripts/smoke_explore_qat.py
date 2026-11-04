#!/usr/bin/env python3
"""Smoke for C1, N1, N2, N5 and D, the exploratory QAT entry (src/explore/qat_explore.py, src/explore/fp32_head.py,
scripts/explore_qat.py, scripts/explore_qat_blockb.py). Synthetic students, loaders and fixture records; no dataset.

  * the parent constants against the records (DL-146 / DL-109, DL-218 and the TEST-day manifest, DL-139);
  * each item's flags and seeds (one named change); D's trigger as a number (0.3875, Decimal of the repr);
  * the exploratory parent check on fixture run folders: it accepts a listed hash and refuses others, a wrong
    identity, a projection.pt; D accepts an x- N3/A1''/A2 winner above the trigger and refuses one below it, a
    non-exploratory folder, another item and an E2 of record;
  * N1: run_qat_logit_kd on a random student with MockTeacher: the loss of step 2 equals CE + Dice + r * 0.5 *
    logit_kd_kl (recomputed independently), zero distillation at step 1, no CWD term anywhere (parts, telemetry,
    run_meta), no projection in the optimizer; a real run refuses injected loaders;
  * N2: run_qat (pinned) with prepare_qat_model substituted: the ten sites installed and firing, TF32 off recorded
    in run_meta, the substitution recorded, the pinned name restored afterwards;
  * N5: the FP32 islands carry no fake-quant in training; in the converted model the classifiers and the add run in
    float on float inputs, a DeQuantize precedes them and every other conv stays quantized (the registered head
    fails the same check); the FP32 MAC share; an epoch checkpoint of a run_qat smoke converts the same way;
  * the CLI refuses a wrong item seed, a non-x- run ID, TEST and a data root that fails the TRAIN/VAL-only check,
    writing nothing.
Mutations: a permissive flag check, an always-met trigger, a no-op FP32 head and a KD term dropped each fail a case.

    python -I scripts/smoke_explore_qat.py        (exit 0 PASS, 1 FAIL)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402
from torch.utils.data import DataLoader, TensorDataset              # noqa: E402

from src.explore import fp32_head as H                               # noqa: E402
from src.explore import guard as G                                   # noqa: E402
from src.explore import qat_explore as X                             # noqa: E402
from src.explore.smokekit import Cases, mutation_killed, patched, scratch_dir  # noqa: E402
from src.explore.synthetic import calibrated_prepared, converted_of, random_student  # noqa: E402


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fixture_parent(base: Path, *, stage="E2", seed=43, run_id="e2_s43_lambda0.5_80k_a1",
                   ckpt="e2_student_best_iter72000.pt", value=0.3716566562652588, projection=False,
                   inner=None) -> tuple[Path, str]:
    d = (base / run_id) if inner is None else inner
    d.mkdir(parents=True)
    key = stage.lower()
    _jsonl(d / f"{key}_run_meta.jsonl", [{"event": "run_meta", "stage": stage, "mode": "real", "seed": seed,
                                          "lambda_logit": 0.5}])
    _jsonl(d / f"{key}_telemetry.jsonl", [{"event": "train", "iter": 1}, {"event": "run_end", "iter": 80000,
                                                                           "checks_passed": True}])
    (d / ckpt).write_bytes(os.urandom(64))
    (d / "best.json").write_text(json.dumps({"best_ckpt": ckpt, "best_val_miou_all_class": value}))
    if projection:
        (d / "projection.pt").write_bytes(b"x")
    return d, G.sha256_file(d / ckpt)


def loaders(n=8, size=64, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randn(n, 3, size, size, generator=g)
    masks = torch.randint(0, 116, (n, size, size), generator=g)
    masks[:, :4, :] = 255
    ds = TensorDataset(imgs, masks)
    return (DataLoader(ds, batch_size=batch, shuffle=False, drop_last=True),
            DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False))


def run_cases(verbose: bool = False) -> Cases:
    c = Cases("SMOKE_EXPLORE_QAT", verbose=verbose)
    base = scratch_dir("x_qat_")
    # ---- constants against the records
    dl = (REPO / "docs" / "DECISION_LOG.md").read_text(encoding="utf-8")
    row146 = next(ln for ln in dl.splitlines() if ln.startswith("| DL-146 |"))
    row139 = next(ln for ln in dl.splitlines() if ln.startswith("| DL-139 |"))
    man = json.loads((REPO / "reports" / "derived" / "tday_manifest.json").read_text(encoding="utf-8"))
    ents = {e.get("run_id"): e for e in man["body"]["entries"] if isinstance(e, dict)}
    c.check("E2 s42's parent hash is DL-146's", X.E2_PARENTS_OF_RECORD[42]["sha256"] in row146)
    c.check("E2 s43/s44's parent hashes are the TEST-day manifest's",
            all(ents[X.E2_PARENTS_OF_RECORD[s]["run_id"]]["checkpoint"]["sha256"] == X.E2_PARENTS_OF_RECORD[s]["sha256"]
                and ents[X.E2_PARENTS_OF_RECORD[s]["run_id"]]["checkpoint"]["name"] == X.E2_PARENTS_OF_RECORD[s]["checkpoint"]
                for s in (42, 43, 44)))
    c.check("the clip selection's hash is DL-139's", X.CLIP_SELECTION_SHA256 in row139)
    c.check("the E6-E2 comparator is DL-146's value", repr(X.COMPARATORS["E6-E2 s42"]) in row146)
    # ---- flags, seeds, trigger
    c.check("C1 with --kd-logit refused", X.item_flag_error("C1", {"kd_logit"}, 43)[0] == "item_flags")
    c.check("N2 with --tf32-off alone refused", X.item_flag_error("N2", {"tf32_off"}, 42)[0] == "item_flags")
    c.check("N2 with both flags accepted", X.item_flag_error("N2", {"tf32_off", "sites_on_input_grid"}, 42) is None)
    c.check("C1 at seed 42 refused", X.item_flag_error("C1", set(), 42)[0] == "item_seed")
    c.check("C1 at seed 43 and 44 accepted", X.item_flag_error("C1", set(), 43) is None
            and X.item_flag_error("C1", set(), 44) is None)
    c.check("N1 at seed 43 refused", X.item_flag_error("N1", {"kd_logit"}, 43)[0] == "item_seed")
    c.check("N5 with --fp32-head accepted", X.item_flag_error("N5", {"fp32_head"}, 42) is None)
    c.check("D trigger: E2 s42's 0.38749080896377563 is below 0.3875", not X.d_trigger_met(0.38749080896377563))
    c.check("D trigger: 0.3875 and 0.39 meet it; NaN does not", X.d_trigger_met(0.3875) and X.d_trigger_met(0.39)
            and not X.d_trigger_met(float("nan")))
    # ---- the exploratory parent check
    d43, sha43 = fixture_parent(base / "p1")
    table = {**X.E2_PARENTS_OF_RECORD, 43: {**X.E2_PARENTS_OF_RECORD[43], "sha256": sha43}}
    with patched(X, "E2_PARENTS_OF_RECORD", table):
        st, par = c.passes("C1 parent: a listed hash is accepted", X.exploratory_parent, "C1", d43, seed=43,
                           expect_source_sha256=sha43) or (None, None)
        c.check("C1 parent: stage E6-E2 from E2", st is not None and st["name"] == "E6-E2" and par["seed"] == 43)
        c.refuses("C1 parent: an unlisted hash refused", "parent_not_of_record", X.exploratory_parent, "C1", d43,
                  seed=43, expect_source_sha256="0" * 64)
        dw, shaw = fixture_parent(base / "p2", run_id="e2_s43_other_run")
        table2 = {**table, 43: {**table[43], "sha256": shaw}}
        with patched(X, "E2_PARENTS_OF_RECORD", table2):
            c.refuses("C1 parent: a listed hash under another run's name refused", "parent_identity",
                      X.exploratory_parent, "C1", dw, seed=43, expect_source_sha256=shaw)
        dp, shap = fixture_parent(base / "p3", projection=True)
        with patched(X, "E2_PARENTS_OF_RECORD", {**table, 43: {**table[43], "sha256": shap}}):
            c.refuses("C1 parent: a projection.pt in the E2 folder refused", "parent_projection_present",
                      X.exploratory_parent, "C1", dp, seed=43, expect_source_sha256=shap)
    c.refuses("C1 parent with the real table: a fixture hash refused", "parent_not_of_record", X.exploratory_parent,
              "C1", d43, seed=43, expect_source_sha256=sha43)
    # D
    item = base / "kd" / "x-n3-s42"
    item.mkdir(parents=True)
    (item / G.RUN_META_NAME).write_text(json.dumps({"item": "N3", "amendment": "AM-24", "exploratory": True,
                                                    "run_id": "x-n3-s42"}))
    dd, shad = fixture_parent(base, stage="E2", seed=42, run_id="x-n3-s42-train", ckpt="e2_student_best_iter76000.pt",
                              value=0.391, inner=item / "x-n3-s42")
    st, par = c.passes("D parent: an x- N3 winner at 0.391 accepted", X.exploratory_parent, "D", dd, seed=42,
                       parent_sha256=shad) or (None, None)
    c.check("D parent: stage E6-E2 for an E2-stage winner", st is not None and st["name"] == "E6-E2"
            and par["x_item"] == "N3")
    low = base / "kd2" / "x-a2-s42"
    low.mkdir(parents=True)
    (low / G.RUN_META_NAME).write_text(json.dumps({"item": "A2", "amendment": "AM-24", "exploratory": True}))
    dl2, shal = fixture_parent(base, stage="E2", seed=42, run_id="r", ckpt="e2_student_best_iter4000.pt",
                               value=0.3874, inner=low / "x-a2-s42")
    c.refuses("D parent below the trigger refused", "d_trigger_not_met", X.exploratory_parent, "D", dl2, seed=42,
              parent_sha256=shal)
    other = base / "kd3" / "x-c1-s43"
    other.mkdir(parents=True)
    (other / G.RUN_META_NAME).write_text(json.dumps({"item": "C1", "amendment": "AM-24", "exploratory": True}))
    do, shao = fixture_parent(base, stage="E2", seed=42, run_id="r2", ckpt="e2_student_best_iter8000.pt",
                              value=0.40, inner=other / "x-c1-s43")
    c.refuses("D parent of another item refused", "d_parent_item", X.exploratory_parent, "D", do, seed=42,
              parent_sha256=shao)
    c.refuses("D parent outside an x- item folder refused", "d_parent_not_exploratory", X.exploratory_parent, "D",
              d43, seed=43, parent_sha256=sha43)
    c.refuses("D with an E2 of record's hash refused", "d_parent_of_record", X.exploratory_parent, "D", dd, seed=42,
              parent_sha256=X.E2_PARENTS_OF_RECORD[42]["sha256"])
    # ---- N1
    from src.distill.teacher import FrozenTeacher, MockTeacher
    from src.training import train_distill as TD
    from src.training.losses import downsample_validity, logit_kd_kl
    cap = {}

    def on_step(step, d):
        cap[step] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in d.items() if k != "teacher_out"}
        cap[step]["t_logits"] = d["teacher_out"].logits.detach().clone()

    model = random_student(seed=3)
    teacher = FrozenTeacher(MockTeacher(116))
    out1 = base / "n1" / "x-n1-smoke"
    res = c.passes("N1 loop runs 3 smoke steps", X.run_qat_logit_kd, mode="smoke", model=model, teacher=teacher,
                   source_meta={"sha256": "0" * 64}, out_dir=out1, seed=42, clip_norm=5.0, clip_source="smoke",
                   device="cpu", num_workers=0, loaders=loaders(), max_steps=3, max_val_batches=1, on_step=on_step,
                   log=lambda *a: None)
    if res:
        s1, s2 = cap[1], cap[2]
        c.check("N1 step 1: zero distillation (ramp 0)", float(s1["distill"]) == 0.0 and s1["ramp"] == 0.0)
        valid = downsample_validity(s2["mask"], s2["head_logits"].shape[-2:])
        kl = logit_kd_kl(s2["head_logits"], s2["t_logits"], valid, T=TD.T_LOGIT)
        want = s2["ce"] + s2["dice"] + s2["ramp"] * 0.5 * kl
        c.check("N1 step 2: loss = CE + Dice + r * 0.5 * KL(T 4), recomputed", s2["ramp"] == 1.0 / 3.0
                and torch.allclose(s2["loss"], want, rtol=1e-6, atol=1e-7),
                f"{float(s2['loss'])!r} vs {float(want)!r}")
        c.check("N1: the only term is logit_kd", set(s2["parts"]) == {"logit_kd"})
        rows = [json.loads(ln) for ln in (out1 / "qat_telemetry.jsonl").read_text().splitlines()]
        meta = rows[0]
        c.check("N1 run_meta: terms logit_kd only, lambda 0.5, T 4, no projection",
                meta["terms"] == {"logit_kd": True, "cwd_feat": False, "cwd_logit": False}
                and meta["lambda_logit"] == 0.5 and meta["T_logit"] == 4 and meta["cwd_projection_loaded"] is False)
        trains = [r for r in rows if r.get("event") == "train"]
        c.check("N1 telemetry: no CWD column", trains and all("cwd_feat" not in r and "cwd_logit" not in r
                                                              for r in trains) and all("logit_kd" in r for r in trains))
        c.check("N1: the optimizer holds the student only (one group of record)",
                meta["optimizer"]["param_groups"] == 1 and meta["optimizer"]["group_holds_every_model_param"] is True)
        c.check("N1: an epoch checkpoint and a fake-quant VAL row were written",
                (out1 / "epoch_ckpts" / "e01.pt").is_file() and any(r.get("event") == "val" for r in rows))
    c.refuses("N1 real mode refuses injected loaders", "real_run_overrides", X.run_qat_logit_kd, mode="real",
              model=random_student(seed=3), teacher=teacher, source_meta={"sha256": "0" * 64},
              out_dir=base / "n1r" / "x-n1-r", seed=42, clip_norm=5.0, clip_source="clip_selection", device="cpu",
              num_workers=1, loaders=loaders())
    # ---- N2: run_qat with the substituted prepare and TF32 off
    from src.explore.patching import substituted
    from src.explore.sites import SITE_NAMES, SiteMatcher, set_tf32, tf32_flags
    from src.quant import qat as Q
    holder = {}
    pinned_prepare = Q.prepare_qat_model

    def prepare_with_sites(m, *, select_backend=True):
        p = pinned_prepare(m, select_backend=select_backend)
        holder["m"] = SiteMatcher(p, SITE_NAMES)
        return p

    before = tf32_flags()
    set_tf32(False)
    out2 = base / "n2" / "x-n2-smoke"
    try:
        with substituted(Q, "prepare_qat_model", prepare_with_sites, item="N2", why="smoke") as rec:
            Q.run_qat(stage="e6-e2", mode="smoke", model=random_student(seed=4), source_meta={"sha256": "0" * 64},
                      out_dir=out2, seed=42, clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0,
                      loaders=loaders(), extra_meta={"x_substitution": rec}, max_steps=2, max_val_batches=1,
                      log=lambda *a: None)
    finally:
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
    meta2 = json.loads((out2 / "qat_telemetry.jsonl").read_text().splitlines()[0])
    c.check("N2: the ten sites were installed and fired", "m" in holder and all(v > 0 for v in holder["m"].calls.values()),
            str(holder.get("m").calls if "m" in holder else None))
    c.check("N2: run_meta records TF32 off (cuDNN and matmul)", meta2["tf32"].get("cudnn_allow_tf32") is False
            and meta2["tf32"].get("cuda_matmul_allow_tf32") is False, str(meta2["tf32"]))
    c.check("N2: run_meta records the substitution and the pinned file's sha256",
            meta2.get("x_substitution", {}).get("pinned_file") == "src/quant/qat.py"
            and len(meta2["x_substitution"].get("pinned_file_sha256", "")) == 64)
    c.check("N2: the pinned prepare_qat_model is restored afterwards", Q.prepare_qat_model is pinned_prepare)
    from src.explore.sites import calibrated_copy, require_site_set
    c.passes("N2 launch gate (AM-24 v2.4): the parent's calibrated copy gives the ten sites", require_site_set,
             calibrated_copy(random_student(seed=4)))
    import torch.nn as nn
    bad = random_student(seed=4)
    ctx = bad.head.context
    bad.head.context = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.AdaptiveAvgPool2d(1), ctx[1], ctx[2])
    c.refuses("N2 launch gate STOPs when the code gives a different site set", "site_set_mismatch", require_site_set,
              calibrated_copy(bad))
    # ---- N5
    pf = calibrated_prepared(random_student(seed=5, fp32_head=True), size=96)
    tr = H.training_islands_report(pf)
    c.check("N5 training: the islands carry no fake-quant and qconfig None", tr["ok"]
            and all(v["qconfig_none"] for v in tr["islands"].values()), str(tr["islands"]))
    cf = converted_of(pf)
    isl = H.converted_islands_report(cf, torch.randn(1, 3, 96, 96))
    c.check("N5 converted: classifiers and add in float on float inputs, DeQuantize before them",
            isl["islands_float_in_float_out"] and isl["islands_are_float_modules"] and isl["dequant_points_converted"],
            str({k: isl[k] for k in ("island_types", "dequant_types")}))
    c.check("N5 converted: every other conv quantized, fed quantized tensors", isl["rest_quantized_in_and_out"]
            and isl["quantized_convs"] > 50 and not isl["rest_not_quantized"], str(isl["quantized_convs"]))
    c.check("N5 converted: the output is float", isl["output_float"] and isl["ok"])
    preg = calibrated_prepared(random_student(seed=5), size=96)
    try:
        isl_reg = H.converted_islands_report(converted_of(preg), torch.randn(1, 3, 96, 96))
        discriminates = not isl_reg["ok"]
    except Exception:                                        # noqa: BLE001 -- the registered head has no dequant points
        discriminates = True
    c.check("N5 check discriminates: the registered head fails it", discriminates)
    ms = H.mac_share(random_student(seed=5))
    c.check("N5 FP32 MAC share in (0, 0.2)", 0.0 < ms["fp32_mac_share"] < 0.2, f"{ms['fp32_mac_share']:.4f}")
    c.check("N5 precision label says mixed and not INT8", "MIXED PRECISION (not INT8)" in H.PRECISION_LABEL)
    out5 = base / "n5" / "x-n5-smoke"
    Q.run_qat(stage="e6-e2", mode="smoke", model=random_student(seed=6, fp32_head=True),
              source_meta={"sha256": "0" * 64}, out_dir=out5, seed=42, clip_norm=5.0, clip_source="smoke",
              device="cpu", num_workers=0, loaders=loaders(), max_steps=2, max_val_batches=1, log=lambda *a: None)
    from src.quant.x86_latency import load_qat_sidecar
    st5 = load_qat_sidecar(out5 / "epoch_ckpts" / "e01.pt")["model_state_dict"]
    conv5, _ = H.convert_state(st5)
    isl5 = H.converted_islands_report(conv5, torch.randn(1, 3, 64, 64))
    c.check("N5: an epoch checkpoint of a run_qat smoke converts with the islands in float", isl5["ok"])
    c.check("N5: the registered converter would refuse the key set (fake-quant entries absent)",
            any(k.startswith("head.high_logits.weight_fake_quant") for k in preg.state_dict())
            and not any(k.startswith("head.high_logits.weight_fake_quant") for k in st5))
    # ---- CLI refusals (no CUDA, no data root in this session)
    py, cli = sys.executable, str(REPO / "scripts" / "explore_qat.py")
    (base / "clipdir").mkdir()
    common = ["--real-run", "--confirm-real-run", "--source-run-dir", str(d43), "--expect-source-sha256", "0" * 64,
              "--clip-selection", str(base / "clipdir" / "clip.json"), "--clip-selection-sha256",
              X.CLIP_SELECTION_SHA256,
              "--num-workers", "1", "--expect-head", "0" * 40]
    env = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}

    def cli_run(*extra):
        return subprocess.run([py, "-I", cli, *common, *extra], capture_output=True, text=True, env=env)
    r = cli_run("--item", "C1", "--seed", "42", "--run-id", "x-c1-s42", "--out-dir", str(base / "o" / "x-c1-s42"))
    c.check("CLI: C1 at seed 42 refused (exit 2)", r.returncode == 2 and "item_seed" in r.stdout)
    r = cli_run("--item", "C1", "--seed", "43", "--run-id", "c1-s43", "--out-dir", str(base / "o" / "c1-s43"))
    c.check("CLI: a run ID without x- refused (exit 2)", r.returncode == 2 and "run_id_prefix" in r.stdout)
    r = cli_run("--item", "C1", "--seed", "43", "--split", "te" + "st", "--run-id", "x-c1-s43",
                "--out-dir", str(base / "o" / "x-c1-s43"))
    c.check("CLI: --split TEST refused (exit 2)", r.returncode == 2 and "split_test" in r.stdout)
    r = cli_run("--item", "C1", "--seed", "43", "--run-id", "x-c1-s43", "--out-dir", str(base / "o" / "x-c1-s43"))
    c.check("CLI: a data root failing the TRAIN/VAL-only check refused, nothing written",
            r.returncode == 2 and "data_root_not_trainval_only" in r.stdout and not (base / "o" / "x-c1-s43").exists(),
            r.stdout.strip()[-160:])
    bb = str(REPO / "scripts" / "explore_qat_blockb.py")
    r = subprocess.run([py, "-I", bb, "--run-dir", str(d43), "--expect-telemetry-sha256", "0" * 64, "--host-label",
                        "smoke", "--run-id", "x-bb", "--out-dir", str(base / "bb" / "x-bb")], capture_output=True,
                       text=True, env=env)
    c.check("block B: a run outside an x- item folder refused (exit 2)", r.returncode == 2
            and "run_not_exploratory" in r.stdout)
    r = subprocess.run([py, "-I", bb, "--run-dir", str(out5), "--expect-telemetry-sha256", "0" * 64, "--host-label",
                        "smoke", "--split", "TE" + "ST", "--run-id", "x-bb2", "--out-dir", str(base / "bb" / "x-bb2")],
                       capture_output=True, text=True, env=env)
    c.check("block B: --split TEST refused first, by name (exit 2)", r.returncode == 2 and "split_test" in r.stdout)
    return c


def main() -> int:
    print("[smoke_explore_qat] cases")
    c = run_cases(verbose=True)
    print("[smoke_explore_qat] mutations")
    from src.training import train_distill as TD

    def no_kd(**kw):
        tot, parts = TD_ORIG(**kw)
        return tot * 0.0, parts
    TD_ORIG = TD.distillation_losses
    for name, mut in {
        "item flags unchecked": lambda: patched(X, "item_flag_error", lambda item, flags, seed: None),
        "D trigger always met": lambda: patched(X, "d_trigger_met", lambda v: True),
        "FP32 head a no-op": lambda: patched(H, "apply_fp32_head", lambda s: s),
        "N1's KD term dropped": lambda: patched(TD, "distillation_losses", no_kd),
    }.items():
        killed, failed = mutation_killed(lambda: run_cases(False), mut)
        c.check(f"mutation KILLED: {name}", killed, f"e.g. {failed[:2]}")
    return c.result()


if __name__ == "__main__":
    raise SystemExit(main())
