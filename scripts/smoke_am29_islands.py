#!/usr/bin/env python3
"""L-AM29 P2 smoke: src/quant/am29_islands.py and scripts/am29_islands.py on synthetic data, on the CPU.

  g0       G0 equals the existing conversion paths: (a) the state_dict companion scripts/qat_epoch_eval.py convert-epoch
           writes for a QAT epoch (byte for byte, by sha256) and its TorchScript's outputs; (b) src/quant/ptq.py
           quantize_qnnpack's converted state_dict on the same calibration images (byte for byte)
  islands  G1-G4 from the QAT state: no FakeQuantize (nor observer, nor BN) in the converted variant; BN folded from the
           frozen statistics (and a ConvBn folded by the same path); each island equals its fake-quant-off module within
           1e-5; every re-entry Quantize reuses the stored qparams; no add or mul mixes quantized and float tensors;
           the INT8 shares of parameters and MACs computed and asserted; the >= 90% parameter rule; the labels
  ptq      the same variants from the PTQ path (re-entry, mixing, shares)
  cli      --split test refused by name before anything is read; a QAT run and a PTQ run of the CLI scoring VAL
           through the evaluator's own code (synthetic VAL tree, 2 samples)

    python -B scripts/smoke_am29_islands.py [--sections g0,islands,ptq,cli]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import smoke_am29_fixtures as FX  # noqa: E402

TMP = FX.tmpdir("smoke_am29_islands_")
DATA = TMP / "plantseg_root"
os.environ["PLANTSEG_DATA_ROOT"] = str(DATA)                 # before configs.data is imported

import torch  # noqa: E402

from src.quant import am29_islands as I  # noqa: E402
from src.quant import ptq as P  # noqa: E402
from src.quant import qat as Q  # noqa: E402

SECTIONS = ("g0", "islands", "ptq", "cli")
check = FX.Checks("L-AM29 P2 (src/quant/am29_islands.py, scripts/am29_islands.py)")
SRC = {"sha256": "0" * 64, "path": None}
STATE: dict = {}


def e6_run() -> dict:
    """A complete smoke E6 run (run_qat, stage e6, 15 epochs x 2 steps) and its pinned conversion of epoch 15."""
    if STATE:
        return STATE
    run = TMP / "e6_s42_smoke"
    Q.run_qat(stage="e6", mode="smoke", model=FX.student(), source_meta=SRC, out_dir=run, seed=42, clip_norm=5.0,
              clip_source="clip_selection", device="cpu", num_workers=0, loaders=FX.loaders(), max_val_batches=1,
              log=FX.quiet)
    ev = TMP / "e6_eval"
    p = subprocess.run([sys.executable, "-B", "scripts/qat_epoch_eval.py", "convert-epoch", "--run-dir", str(run),
                        "--eval-dir", str(ev), "--epoch", "15", "--purpose", "timing", "--host-label", "smoke",
                        "--allow-smoke-inputs"], cwd=str(REPO), capture_output=True, text=True)
    ck = run / "epoch_ckpts" / "e15.pt"
    STATE.update({"run": run, "eval": ev, "rc": p.returncode, "out": p.stdout[-400:] + p.stderr[-400:], "ck": ck,
                  "state": torch.load(ck, map_location="cpu", weights_only=True)["model_state_dict"]})
    return STATE


# ------------------------------------------------------------------ g0
def sec_g0() -> None:
    s = e6_run()
    check("g0_pinned_convert_ran", s["rc"] == 0, s["out"][-200:])
    res = I.convert_qat_state(s["state"], groups=("G0",))
    mine = P.state_dict_bytes(res["g0"], "E6", "qat")
    pinned = s["eval"] / "converted" / "e15_int8_student.pt"
    check("g0_qat_equals_pinned_state_dict_bytes", P._sha256_bytes(mine) == FX.sha_file(pinned),
          f"G0 {P._sha256_bytes(mine)[:16]} == qat_epoch_eval convert-epoch's e15_int8_student.pt "
          f"{FX.sha_file(pinned)[:16]}")
    from src.eval.model_loading import load_int8_torchscript
    from src.quant.qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    ts, _ = load_int8_torchscript(s["eval"] / "converted" / "e15_int8_qnnpack.torchscript.pt", stage="E6", method="qat")
    par = P.output_parity({"g0": res["g0"], "torchscript": ts}, P.synthetic_parity_inputs(), [("g0", "torchscript")])
    check("g0_qat_outputs_equal_pinned_torchscript", par["all_equal"], f"{par['pairs']}")
    model, _ = res["variants"]["G0"]
    check("g0_variant_is_g0", P._sha256_bytes(P.state_dict_bytes(model, "E6", "qat")) == P._sha256_bytes(mine),
          "the G0 variant record holds the same converted model")
    # (b) the PTQ of record's conversion
    g = torch.Generator().manual_seed(4)
    imgs = [torch.randn(1, 3, 64, 64, generator=g) for _ in range(4)]
    fp32 = FX.student(21).eval()
    q, n = P.quantize_qnnpack(fp32, iter(imgs))
    prepared, n2 = I.calibrate_fp32(fp32, iter(imgs))
    r2 = I.convert_ptq_prepared(prepared, groups=("G0",))
    a, b = P.state_dict_bytes(q, "E7", "ptq"), P.state_dict_bytes(r2["g0"], "E7", "ptq")
    check("g0_ptq_equals_quantize_qnnpack_bytes", n == n2 == 4 and P._sha256_bytes(a) == P._sha256_bytes(b),
          f"{P._sha256_bytes(a)[:16]} == {P._sha256_bytes(b)[:16]} after {n} calibration batches")
    STATE["ptq_prepared"] = prepared


# ------------------------------------------------------------------ islands (QAT)
def sec_islands() -> None:
    s = e6_run()
    from src.quant import qat_artifacts as A
    rec = A.read_run_record(s["run"])
    cross = A.freeze_cross_check(rec)
    check("islands_bn_statistics_frozen", cross["ok"], "BN statistics of e11-e15 equal e10's (the freeze took)")
    res = I.convert_qat_state(s["state"])
    recs = {}
    for g, (model, rep) in res["variants"].items():
        recs[g] = I.variant_record(g, model, rep, g0=res["g0"], prepared_qat=res["prepared"])
    for g in ("G1", "G2", "G3", "G4"):
        r = recs[g]
        c = r["checks"]
        check(f"islands_{g}_no_fake_quant", c["no_fake_quant_observer_or_bn"]["ok"]
              and not any(isinstance(m, torch.ao.quantization.FakeQuantizeBase) for isl in
                          I._island_bodies(res["variants"][g][0]).values() for m in isl.modules()),
              f"islands {r['islands'][:3]}{'...' if len(r['islands']) > 3 else ''}")
        check(f"islands_{g}_bn_folded", c["bn_folded_from_frozen_statistics"]["ok"],
              f"{[x['kind'] for x in c['bn_folded_from_frozen_statistics']['convs']][:2]} "
              f"({len(c['bn_folded_from_frozen_statistics']['convs'])} convs)")
        eq = c["equals_fake_quant_off_module"]
        check(f"islands_{g}_equals_fake_quant_off", eq["ok"], f"max |diff| {eq['max_abs_diff']:.3g} <= {eq['tolerance']}")
        check(f"islands_{g}_reentry_stored_qparams", c["reentry_reuses_stored_qparams"]["ok"]
              and (g == "G1" or c["reentry_reuses_stored_qparams"]["rows"]),
              f"{[(x['site'], x['quantize']) for x in c['reentry_reuses_stored_qparams']['rows'][:2]]}")
        check(f"islands_{g}_no_mixed_add_mul", c["no_mixed_add_mul"]["ok"], f"{c['no_mixed_add_mul']['calls']} add/mul calls")
        check(f"islands_{g}_label", r["label"] == "mixed precision", r["label"])
    check("islands_G0_label_and_checks", recs["G0"]["label"] == "int8" and recs["G0"]["checks_ok"], recs["G0"]["label"])
    # a ConvBn through the same fold (features.0.0, ConvBn2d): folded from its running statistics, within 1e-5
    from torch.ao.quantization import disable_fake_quant, disable_observer
    from torch.nn.utils.fusion import fuse_conv_bn_weights
    qm = res["prepared"].get_submodule("features.0.0")
    fm = I.float_from_qat(qm)
    w, b = fuse_conv_bn_weights(qm.weight, qm.bias, qm.bn.running_mean, qm.bn.running_var, qm.bn.eps, qm.bn.weight,
                                qm.bn.bias)
    ref = __import__("copy").deepcopy(qm)
    ref.apply(disable_fake_quant)
    ref.apply(disable_observer)
    ref.eval()
    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        d = float((fm(x) - ref(x)).abs().max())
    check("islands_convbn_fold", type(qm).__name__ == "ConvBn2d" and torch.equal(fm.weight, w.detach())
          and torch.equal(fm.bias, b.detach()) and d <= I.EQUIVALENCE_TOL and not list(fm.children()),
          f"ConvBn2d -> Conv2d with BN folded from running stats; max |diff| vs fake-quant-off {d:.3g}")
    # the shares
    sh = {g: r["shares"] for g, r in recs.items()}
    totals = {g: v["params_total"] for g, v in sh.items()}
    check("shares_params_total_equal", len(set(totals.values())) == 1, f"{totals}")
    check("shares_g0_all_int8", sh["G0"]["int8_param_share"] == 1.0 and sh["G0"]["int8_mac_share"] == 1.0
          and 0 < sh["G0"]["int8_mac_share_with_boundary"] < 1.0,
          f"G0 params 1.0, network MACs 1.0, with the final float upsample {sh['G0']['int8_mac_share_with_boundary']:.4f}")
    table = {g: (round(v["int8_param_share"], 4), round(v["int8_mac_share"], 4)) for g, v in sh.items()}
    check("shares_g1_g4_below_g0", all(sh[g]["int8_mac_share"] < 1.0 for g in ("G1", "G2", "G3", "G4"))
          and all(sh[g]["int8_param_share"] < 1.0 for g in ("G1", "G2", "G3")) and sh["G4"]["int8_param_share"] == 1.0,
          f"(params, MACs) {table}")
    want_elig = {"G0": True, "G1": True, "G2": False, "G3": True, "G4": True}
    elig = {g: r["eligible"] for g, r in recs.items()}
    check("shares_eligibility_90pct", elig == want_elig and recs["G2"]["shares"]["int8_param_share"] < 0.9,
          f"eligible {elig} (G2 keeps the SE gates' parameters, over 10% of the model, in FP32)")
    fp32_g1 = sh["G1"]["params"]["fp32"]
    check("shares_g1_param_count", fp32_g1 == (256 * 116 + 116) + (40 * 116 + 116),
          f"G1 keeps {fp32_g1} parameters in FP32: high_logits 256x116+116, low_logits 40x116+116")
    STATE["recs"] = recs


# ------------------------------------------------------------------ ptq
def sec_ptq() -> None:
    prepared = STATE.get("ptq_prepared")
    if prepared is None:
        g = torch.Generator().manual_seed(4)
        prepared, _ = I.calibrate_fp32(FX.student(21).eval(), iter([torch.randn(1, 3, 64, 64, generator=g)
                                                                   for _ in range(4)]))
    res = I.convert_ptq_prepared(prepared)
    for g in ("G0", "G1", "G2", "G3", "G4"):
        model, rep = res["variants"][g]
        r = I.variant_record(g, model, rep, g0=res["g0"], prepared_qat=None)
        check(f"ptq_{g}_checks", r["checks_ok"], f"{ {k: v.get('ok') for k, v in r['checks'].items()} }; params "
                                               f"{r['shares']['int8_param_share']:.4f}, eligible {r['eligible']}")


# ------------------------------------------------------------------ cli
def cli(argv) -> tuple[int, str]:
    from scripts import am29_islands as C
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = C.main(argv)
    return rc, buf.getvalue()


def sec_cli() -> None:
    for split in ("test", "TEST", "Test", "val_test"):
        rc, out = cli(["--split", split, "--model", "E6-s42-e15", "--qat-checkpoint", str(TMP / "absent.pt"),
                       "--expect-checkpoint-sha256", "a" * 64, "--out", str(TMP / "never.json")])
        check(f"cli_split_{split}_refused", rc == 2 and "[split_test]" in out and not (TMP / "never.json").exists(),
              out.strip().splitlines()[-1][:120])
    rc, out = cli(["--split", "train", "--model", "E7", "--out", str(TMP / "never.json")])
    check("cli_split_train_refused", rc == 2 and "[split_not_val]" in out, out.strip()[-100:])
    s = e6_run()
    FX.make_val_tree(DATA, 846)
    from scripts.synthetic_ptq_fixtures import make_tree
    ids = make_tree(DATA, n_train=128, n_val=0, seed=5)["train"]
    sha = FX.sha_file(s["ck"])
    rc, out = cli(["--split", "val", "--model", "E6-s42-e15", "--qat-checkpoint", str(s["ck"]),
                   "--expect-checkpoint-sha256", "ab" * 32, "--out", str(TMP / "never.json")])
    check("cli_qat_prefix_refused", rc == 2 and "[checkpoint_not_of_record]" in out,
          "a sha256 not beginning 74fd438a9e95 (DL-176's artifact name) is refused outside smoke inputs")
    rep = TMP / "islands_qat.json"
    rc, out = cli(["--split", "val", "--model", "E6-s42-e15", "--qat-checkpoint", str(s["ck"]),
                   "--expect-checkpoint-sha256", sha, "--run-dir", str(s["run"]), "--out", str(rep),
                   "--max-samples", "2", "--allow-smoke-inputs"])
    doc = json.loads(rep.read_text()) if rep.is_file() else {}
    vs = {v["group"]: v for v in doc.get("variants", [])}
    check("cli_qat_ran", rc == 0 and len(vs) == 5, out.strip().splitlines()[-1][:200] if out.strip() else "")
    check("cli_qat_scored_val", all(v["val"]["split"] == "val" and v["val"]["actual_rows"] == 2
                                    and isinstance(v["val"]["dataset_level"]["all_class_miou"], float)
                                    for v in vs.values()) and doc.get("split") == "val",
          f"{ {g: round(v['val']['dataset_level']['all_class_miou'], 4) for g, v in vs.items()} }")
    pinned = s["eval"] / "converted" / "e15_int8_student.pt"
    check("cli_qat_g0_sha256_is_pinned", doc.get("g0_state_dict_sha256") == FX.sha_file(pinned),
          "the report's G0 state_dict sha256 is the pinned conversion's companion")
    check("cli_qat_labels_shares", all(v["label"] == ("int8" if g == "G0" else "mixed precision")
                                       and 0 < v["shares"]["int8_param_share"] <= 1 for g, v in vs.items())
          and [g for g, v in vs.items() if not v["eligible"]] == ["G2"], "G2 alone is not eligible")
    # the PTQ path through the CLI: a synthetic list in the list schema over 128 synthetic TRAIN images
    cal = TMP / "cal_seed42.json"
    from src.quant.calibration import CALIBRATION_SCHEMA
    doc_l = {"schema": CALIBRATION_SCHEMA, "list_schema": P.LIST_SCHEMA, "role": P.ROLE_OF_RECORD, "split": "train",
             "seed": 42, "count": 128, "shared_by": ["E4", "E7"], "candidate_pool_size": 128, "selected_ids": ids,
             "checksum_sha256": P.ids_checksum(ids), "split_list_sha256": "0" * 64, "procedure": P.LIST_PROCEDURE,
             "artifact_status": "smoke"}
    cal.write_text(json.dumps(doc_l), encoding="utf-8")
    e3 = TMP / "e3_fixture.pt"
    torch.save({"stage": "E3", "iter": 76000, "num_classes": FX.NC, "model_state_dict": FX.student(33).state_dict()}, e3)
    rep2 = TMP / "islands_ptq.json"
    rc, out = cli(["--split", "val", "--model", "E7", "--fp32-checkpoint", str(e3), "--expect-source-sha256",
                   FX.sha_file(e3), "--calibration-list", str(cal), "--expect-list-checksum", P.ids_checksum(ids),
                   "--out", str(rep2), "--max-samples", "2", "--allow-smoke-inputs", "--groups", "G0", "G1", "G3"])
    d2 = json.loads(rep2.read_text()) if rep2.is_file() else {}
    check("cli_ptq_ran", rc == 0 and [v["group"] for v in d2.get("variants", [])] == ["G0", "G1", "G3"]
          and d2.get("source", {}).get("calibration", {}).get("batches_seen") == 128,
          out.strip().splitlines()[-1][:200] if out.strip() else "")
    rc, out = cli(["--split", "val", "--model", "E7", "--fp32-checkpoint", str(e3), "--expect-source-sha256",
                   FX.sha_file(e3), "--calibration-list", str(cal), "--expect-list-checksum", P.ids_checksum(ids),
                   "--out", str(TMP / "never2.json")])
    check("cli_ptq_source_not_of_record_refused", rc == 2 and "[source_not_of_record]" in out,
          "outside smoke inputs E7's source must be 4baa5bad... (DL-121)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS))
    want = [s for s in ap.parse_args(argv).sections.split(",") if s]
    try:
        for s in SECTIONS:
            if s in want:
                globals()[f"sec_{s}"]()
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("no_unexpected_exception", False, f"{type(e).__name__}: {e}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
