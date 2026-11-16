#!/usr/bin/env python3
"""L-AM29 P3 + P4 smoke: scripts/am29_blockb.py and scripts/am29_score.py on synthetic data, on the CPU.

A complete smoke T1 run (run_am29_qat, 15 epochs x 2 steps) goes through block B (convert, score, select, finalize);
its selected all-INT8 artifact is scored (int8) and its selected weights under a P2 group (mixed precision). A
synthetic artifact in lane/l-am24's block B format (a smoke E6-E2 run named x-n2-s42, one epoch converted by the
pinned scripts/qat_epoch_eval.py, its qat_selection.json, qat_finalize.json and x_blockb.json) is scored through the
same path. Refusals: an unknown, a mixed-precision and a hash-mismatched artifact as int8; the TEST rules (refused
before anything is read; TEST is never scored here). VAL is a synthetic 846-image tree, scored on 2 samples.

    python -B scripts/smoke_am29_blockb.py [--sections blockb,rule,nonfinite,score,lam24,test]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import smoke_am29_fixtures as FX  # noqa: E402

TMP = FX.tmpdir("smoke_am29_blockb_")
DATA = TMP / "plantseg_root"
os.environ["PLANTSEG_DATA_ROOT"] = str(DATA)

import torch  # noqa: E402

from src.quant import am29 as M  # noqa: E402
from src.quant import am29_registry as R  # noqa: E402
from src.quant import qat as Q  # noqa: E402

SECTIONS = ("blockb", "rule", "nonfinite", "score", "lam24", "test")
check = FX.Checks("L-AM29 P3 + P4 (scripts/am29_blockb.py, scripts/am29_score.py)")
SRC = {"sha256": "0" * 64, "path": None}
ST: dict = {}


def run(script: str, *args) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-B", script, *map(str, args)], cwd=str(REPO), capture_output=True, text=True,
                       env={**os.environ, "PLANTSEG_DATA_ROOT": str(DATA)})
    return p.returncode, p.stdout + p.stderr


def last_result(out: str) -> str:
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT")]
    return lines[-1] if lines else out.strip()[-200:]


def t1_run(name: str = "am29-t1_s42_smoke", inject=()) -> Path:
    out = TMP / name
    m = FX.student()
    M.run_am29_qat(arm="T1", mode="smoke", model=m, source_meta=SRC, out_dir=out, seed=42, clip_norm=5.0,
                   clip_source="clip_selection", device="cpu", num_workers=0, lr=3e-4, factor=0.5, e3_lambda=0.5,
                   teacher=M.SelfTeacher(m, checkpoint_sha256="1" * 64), loaders=FX.loaders(), max_val_batches=1,
                   inject_nan_at_steps=inject, log=FX.quiet)
    return out


def blockb(run_dir: Path, ev: Path, *, max_samples: int = 2) -> dict:
    tel = FX.sha_file(run_dir / "qat_telemetry.jsonl")
    res = {}
    res["convert"] = run("scripts/am29_blockb.py", "convert", "--run-dir", run_dir, "--eval-dir", ev, "--host-label",
                         "smoke", "--expect-telemetry-sha256", tel, "--allow-smoke-inputs")
    res["score"] = run("scripts/am29_blockb.py", "score", "--run-dir", run_dir, "--eval-dir", ev, "--host-label",
                       "smoke", "--max-samples", max_samples, "--allow-smoke-inputs", "--threads", 2)
    res["select"] = run("scripts/am29_blockb.py", "select", "--run-dir", run_dir, "--eval-dir", ev,
                        "--expect-telemetry-sha256", tel, "--allow-smoke-inputs")
    res["finalize"] = run("scripts/am29_blockb.py", "finalize", "--run-dir", run_dir, "--eval-dir", ev,
                          "--allow-smoke-inputs")
    res["tel"] = tel
    return res


# ------------------------------------------------------------------ blockb
def sec_blockb() -> None:
    FX.make_val_tree(DATA, 846)
    rd = t1_run()
    ev = TMP / "eval_t1"
    res = blockb(rd, ev)
    ST.update({"run": rd, "eval": ev, "tel": res["tel"]})
    for step in ("convert", "score", "select", "finalize"):
        rc, out = res[step]
        check(f"blockb_{step}", rc == 0, last_result(out))
    conv = json.loads((ev / "am29_convert.json").read_text())
    check("blockb_converted_15", [o["status"] for o in conv["outcomes"]] == ["converted"] * 15
          and conv["stage"] == "AM29-T1" and conv["freeze_cross_check"]["ok"], f"{conv['stage']} {conv['arm']}")
    from src.quant.qat_artifacts import read_ts_identity
    ids = [read_ts_identity(ev / "converted" / f"e{e:02d}_int8_qnnpack.torchscript.pt") for e in range(1, 16)]
    provs = [json.loads((ev / "converted" / f"e{e:02d}_run_meta.json").read_text()) for e in range(1, 16)]
    check("blockb_identity_stage_am29", all(i["stage"] == "AM29-T1" and i["quantization"] == "qat" for i in ids)
          and all(p["stage"] == "AM29-T1" and p["quantization_method"] == "qat" and p["cwd_projection_loaded"] is False
                  and all(p["verification"]["checks"].values()) for p in provs),
          "every converted epoch: plantseg_int8.json stage AM29-T1, the conversion's checks of record all passed")
    evd = json.loads((ev / "am29_epoch_eval.json").read_text())
    scores = [json.loads((ev / r["score"]["path"]).read_text()) for r in evd["epochs"] if r["status"] == "scored"]
    check("blockb_every_epoch_scored_int8", len(scores) == 15 and all(s["label"] == "int8" and s["precision"] == "int8"
                                                                      and s["score"]["split"] == "val" for s in scores),
          "15/15 scored all-INT8 on VAL")
    flat = json.dumps(scores)
    check("blockb_scores_dataset_level_only", all(s["dataset_level_only"] is True for s in scores)
          and "per_image" not in flat and "image_id" not in flat,
          f"score record keys {sorted(scores[0]['score'])[:8]}...")
    sel = json.loads((ev / "am29_selection.json").read_text())
    rules_sha = FX.sha_file(REPO / R.RULES_REL)
    vals = sel["values"]
    best = max(vals.values())
    first = min(int(k[1:]) for k, v in vals.items() if v == best)
    check("blockb_selection_rule", sel["rules_sha256"] == rules_sha and sel["winner"]["epoch"] == first
          and sel["stage"] == "AM29-T1" and sel["fake_quant_val_used"] is False,
          f"winner e{sel['winner']['epoch']:02d} = earliest of the max {best!r} (ties {sel['tied_epochs']})")
    fin = json.loads((ev / "am29_finalize.json").read_text())
    check("blockb_finalize", fin["status"] == "finalized" and fin["am29_selection"]["sha256"] == FX.sha_file(
        ev / "am29_selection.json") and fin["artifact_of_record"]["torchscript"] == sel["winner"]["artifact_of_record"]
          and fin["x86_latency_copy"] is None, f"artifact {fin['artifact_of_record']['torchscript']['sha256'][:12]}")
    ST["sel"], ST["fin"] = sel, fin
    # the pinned entry points refuse the AM-29 stage (hence the wrappers)
    rc, out = run("scripts/select_qat_epoch.py", "--run-dir", rd, "--eval-dir", ev, "--expect-telemetry-sha256",
                  res["tel"], "--allow-smoke-inputs", "--out", TMP / "pinned_sel.json")
    rc2, out2 = run("scripts/qat_epoch_eval.py", "convert-epoch", "--run-dir", rd, "--eval-dir", TMP / "pinned_ev",
                    "--epoch", "1", "--purpose", "timing", "--host-label", "smoke", "--allow-smoke-inputs")
    check("blockb_pinned_tools_refuse_am29", rc != 0 and rc2 != 0, f"select_qat_epoch: {last_result(out)[:90]}; "
                                                                   f"qat_epoch_eval convert-epoch: {last_result(out2)[:90]}")


# ------------------------------------------------------------------ rule (mutated scores: tie and strict >)
def _mutated_select(values: dict, tag: str) -> dict:
    src = ST["eval"]
    ev = TMP / f"eval_rule_{tag}"
    shutil.copytree(src, ev)
    for f in ("am29_selection.json", "am29_finalize.json"):
        (ev / f).unlink()
    evd = json.loads((ev / "am29_epoch_eval.json").read_text())
    for r in evd["epochs"]:
        if r["status"] != "scored" or r["epoch"] not in values:
            continue
        p = ev / r["score"]["path"]
        s = json.loads(p.read_text())
        s["score"]["dataset_level"]["all_class_miou"] = values[r["epoch"]]
        r["score"]["sha256"] = FX.write_json(p, s)
        r["all_class_miou"] = values[r["epoch"]]
    FX.write_json(ev / "am29_epoch_eval.json", evd)
    rc, out = run("scripts/am29_blockb.py", "select", "--run-dir", ST["run"], "--eval-dir", ev,
                  "--expect-telemetry-sha256", ST["tel"], "--allow-smoke-inputs")
    return {"rc": rc, "out": out, "sel": json.loads((ev / "am29_selection.json").read_text()) if rc == 0 else None}


def sec_rule() -> None:
    base = {e: 0.1 for e in range(1, 16)}
    tie = _mutated_select({**base, 5: 0.5, 9: 0.5}, "tie")
    check("rule_tie_to_earlier", tie["rc"] == 0 and tie["sel"]["winner"]["epoch"] == 5 and tie["sel"]["tied_epochs"] == [5, 9],
          last_result(tie["out"]))
    strict = _mutated_select({**base, 5: 0.5, 9: 0.5000001}, "strict")
    check("rule_strict_greater", strict["rc"] == 0 and strict["sel"]["winner"]["epoch"] == 9, last_result(strict["out"]))
    late = _mutated_select({**base, 15: 0.9}, "late")
    check("rule_highest_of_15", late["rc"] == 0 and late["sel"]["winner"]["epoch"] == 15, last_result(late["out"]))
    from src.quant import qat_select as S
    unregistered = None
    if not R.registered():                               # this process has not registered AM-29's stages yet
        try:
            S.load_rules(REPO / R.RULES_REL)
        except Q.QATRefused as e:
            unregistered = e.code
    check("rule_pinned_loader_refuses_unregistered", unregistered == "rules_mismatch",
          f"without register(), the pinned load_rules checks the file as Q2's and refuses it: [{unregistered}]")
    R.register()
    rules, sha = S.load_rules(REPO / R.RULES_REL)
    check("rule_rules_file_through_pinned_loader", sha == FX.sha_file(REPO / R.RULES_REL)
          and rules["qat_epoch"]["compare"] == "strict_greater" and rules["qat_epoch"]["tie"] == "earliest_epoch"
          and rules["qat_epoch"]["stages"] == list(R.AM29_STAGES), f"rules sha256 {sha[:16]}")
    # a changed score record breaks the chain
    ev = TMP / "eval_rule_late"
    p = ev / "scores" / "e03.json"
    s = json.loads(p.read_text())
    s["score"]["dataset_level"]["all_class_miou"] = 0.99
    p.write_text(json.dumps(s))
    (ev / "am29_selection.json").unlink()
    rc, out = run("scripts/am29_blockb.py", "select", "--run-dir", ST["run"], "--eval-dir", ev,
                  "--expect-telemetry-sha256", ST["tel"], "--allow-smoke-inputs")
    check("rule_changed_score_refused", rc == 2 and "[chain_files]" in out, last_result(out))


# ------------------------------------------------------------------ nonfinite (AM-21 item 2(a))
def sec_nonfinite() -> None:
    rd = t1_run("am29-t1_s42_nan", inject=(27,))
    ends = FX.rows(rd, "epoch_end")
    bad = [r["epoch"] for r in ends if r["state_finite"] is False]
    ev = TMP / "eval_nan"
    res = blockb(rd, ev, max_samples=1)
    rcs = {k: res[k][0] for k in ("convert", "score", "select", "finalize")}
    sel = json.loads((ev / "am29_selection.json").read_text()) if (ev / "am29_selection.json").is_file() else {}
    check("nonfinite_excluded", bad == [14, 15] and rcs == {"convert": 0, "score": 0, "select": 0, "finalize": 0}
          and sel.get("excluded_epochs") == [14, 15] and sel["winner"]["epoch"] <= 13,
          f"non-finite states at epochs {bad}; exits {rcs}; excluded {sel.get('excluded_epochs')}")


# ------------------------------------------------------------------ score (P4, the AM-29 arm)
def sec_score() -> None:
    ev, sel = ST["eval"], ST["sel"]
    art = sel["winner"]["artifact_of_record"]["sha256"]
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--am29-eval-dir", ev, "--arm", "T1",
                  "--expect-artifact-sha256", art, "--max-samples", 2, "--threads", 2)
    js = [json.loads(ln.split(" ", 1)[1]) for ln in out.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    check("score_am29_int8", rc == 0 and js and js[0]["label"] == "int8" and js[0]["name"] == "T1"
          and js[0]["stage"] == "AM29-T1" and set(js[0]) == {"label", "name", "stage", "split", "rows", "dataset_level"},
          last_result(out))
    check("score_prints_dataset_level_only", js and "all_class_miou" in js[0]["dataset_level"]
          and not [ln for ln in out.splitlines() if "plant_vleaf" in ln], "no image identifier is printed")
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--am29-eval-dir", ev,
                  "--expect-artifact-sha256", "ab" * 32, "--max-samples", 2)
    check("score_am29_hash_mismatch_refused", rc == 2 and "int8_hash_mismatch" in out, last_result(out))
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--am29-eval-dir", ev, "--arm", "T2",
                  "--expect-artifact-sha256", art, "--max-samples", 2)
    check("score_am29_other_arm_refused", rc == 2 and "[arm_mismatch]" in out, last_result(out))
    tam = TMP / "eval_tampered"
    shutil.copytree(ev, tam)
    tsp = tam / sel["winner"]["artifact_of_record"]["path"]
    b = bytearray(tsp.read_bytes())
    b[-200] ^= 0x01
    tsp.write_bytes(bytes(b))
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--am29-eval-dir", tam,
                  "--expect-artifact-sha256", art, "--max-samples", 2)
    check("score_am29_tampered_refused", rc == 2 and "int8_hash_mismatch" in out, last_result(out))
    ck = sel["winner"]["checkpoint_sha256"]
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "mixed", "--am29-eval-dir", ev, "--arm", "T1",
                  "--islands", "G1", "--run-dir", ST["run"], "--expect-checkpoint-sha256", ck, "--max-samples", 2)
    js = [json.loads(ln.split(" ", 1)[1]) for ln in out.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    check("score_am29_mixed_precision", rc == 0 and js and js[0]["label"] == "mixed precision"
          and "(mixed precision, T1" in last_result(out) and "int8" not in last_result(out).split("(")[1].split(",")[0],
          last_result(out))
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "mixed", "--am29-eval-dir", ev,
                  "--islands", "G0", "--run-dir", ST["run"], "--expect-checkpoint-sha256", ck, "--max-samples", 2)
    check("score_mixed_g0_refused", rc == 2 and "[mixed_group]" in out, last_result(out))
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "mixed", "--am29-eval-dir", ev,
                  "--islands", "G3", "--run-dir", ST["run"], "--expect-checkpoint-sha256", "cd" * 32, "--max-samples", 2)
    check("score_mixed_checkpoint_mismatch_refused", rc == 2 and "[mixed_checkpoint_mismatch]" in out, last_result(out))
    out_json = TMP / "score_t1.json"
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--am29-eval-dir", ev,
                  "--expect-artifact-sha256", art, "--max-samples", 2, "--out-json", out_json)
    doc = json.loads(out_json.read_text()) if out_json.is_file() else {}
    check("score_record_binding", rc == 0 and doc.get("source", {}).get("artifact_sha256") == art
          and doc["source"]["selection_sha256"] == FX.sha_file(ev / "am29_selection.json")
          and doc["dataset_level_only"] is True, "the record names the manifest sha256, the selection and finalize")


# ------------------------------------------------------------------ lam24 (lane/l-am24's block B format)
def _write_lam24_records(bb: Path, run_dir: Path, epoch: int) -> dict:
    E = bb / "eval"
    tel = FX.sha_file(run_dir / "qat_telemetry.jsonl")
    ck_sha = FX.rows(run_dir, "epoch_end")[epoch - 1]["checkpoint_sha256"]
    ts = E / "converted" / f"e{epoch:02d}_int8_qnnpack.torchscript.pt"
    prov = E / "converted" / f"e{epoch:02d}_run_meta.json"
    sel = {"format": "qat_selection/1", "rules_sha256": "0" * 64, "stage": "E6-E2", "run_dir": str(run_dir),
           "run_id": run_dir.name, "seed": 42, "clip_norm": 5.0, "telemetry_sha256": tel, "eval_dir": str(E),
           "winner": {"epoch": epoch, "value": 0.31, "checkpoint": f"epoch_ckpts/e{epoch:02d}.pt",
                      "checkpoint_sha256": ck_sha,
                      "artifact_of_record": {"path": f"converted/{ts.name}", "sha256": FX.sha_file(ts)},
                      "provenance": {"path": f"converted/{prov.name}", "sha256": FX.sha_file(prov)},
                      "summary": {"path": f"scores/e{epoch:02d}/summary.json", "sha256": "0" * 64},
                      "run_id": f"{run_dir.name}_e{epoch:02d}_{ck_sha[:12]}"},
           "smoke_inputs": False}
    ssha = FX.write_json(E / "qat_selection.json", sel)
    FX.write_json(E / "qat_finalize.json", {"format": "qat_finalize/1", "status": "finalized", "stage": "E6-E2",
                                            "run_id": run_dir.name, "qat_selection": {"path": "qat_selection.json",
                                                                                     "sha256": ssha},
                                            "artifact_of_record": {"torchscript": sel["winner"]["artifact_of_record"]}})
    FX.write_json(bb / "x_blockb.json", {"format": "x_qat_blockb/1", "exploratory": True, "amendment": "AM-24",
                                         "artifact_status": "exploratory", "split": "val", "item": "N2",
                                         "run_dir": str(run_dir), "telemetry_sha256": tel, "telemetry_unchanged": True,
                                         "steps": [{"step": s, "exit": 0} for s in ("convert", "score", "select",
                                                                                     "finalize")],
                                         "selection": sel})
    return sel


def sec_lam24() -> None:
    item = TMP / "x-n2-s42-item"
    rd = item / "x-n2-s42"
    item.mkdir()
    Q.run_qat(stage="e6-e2", mode="smoke", model=FX.student(3), source_meta=SRC, out_dir=rd, seed=42, clip_norm=5.0,
              clip_source="clip_selection", device="cpu", num_workers=0, loaders=FX.loaders(), max_val_batches=1,
              log=FX.quiet)
    bb = TMP / "x-n2-bb"
    rc, out = run("scripts/qat_epoch_eval.py", "convert-epoch", "--run-dir", rd, "--eval-dir", bb / "eval", "--epoch", 3,
                  "--purpose", "record", "--host-label", "smoke", "--allow-smoke-inputs")
    check("lam24_pinned_convert", rc == 0, last_result(out))
    sel = _write_lam24_records(bb, rd, 3)
    art = sel["winner"]["artifact_of_record"]["sha256"]
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--l-am24-blockb-dir", bb,
                  "--run-id", "x-n2-s42", "--expect-artifact-sha256", art, "--max-samples", 2)
    js = [json.loads(ln.split(" ", 1)[1]) for ln in out.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    check("lam24_int8_scored", rc == 0 and js and js[0]["label"] == "int8" and js[0]["name"] == "N2"
          and js[0]["stage"] == "E6-E2", last_result(out))
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--l-am24-blockb-dir", bb,
                  "--run-id", "x-n2-s42", "--expect-artifact-sha256", "ef" * 32, "--max-samples", 2)
    check("lam24_hash_mismatch_refused", rc == 2 and "int8_hash_mismatch" in out, last_result(out))
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--l-am24-blockb-dir", bb,
                  "--run-id", "x-n5-s42", "--expect-artifact-sha256", art, "--max-samples", 2)
    check("lam24_unknown_run_refused", rc == 2 and "[artifact_unknown]" in out, last_result(out))
    # an unknown stage: the provenance says E5
    unk = TMP / "x-unknown-bb"
    shutil.copytree(bb, unk)
    pp = unk / "eval" / "converted" / "e03_run_meta.json"
    pd = json.loads(pp.read_text())
    pd["stage"] = "E5"
    FX.write_json(pp, pd)
    _write_lam24_records(unk, rd, 3)
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--l-am24-blockb-dir", unk,
                  "--run-id", "x-n2-s42", "--expect-artifact-sha256", art, "--max-samples", 2)
    check("lam24_unknown_stage_refused", rc == 2 and "[artifact_unknown]" in out, last_result(out))
    # a mixed-precision artifact: the TorchScript re-saved with x_mixed_precision.json, the records made consistent
    mix = TMP / "x-mixed-bb"
    shutil.copytree(bb, mix)
    tsp = mix / "eval" / "converted" / "e03_int8_qnnpack.torchscript.pt"
    from src.quant.qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    mod = torch.jit.load(str(tsp))
    tsp.unlink()
    torch.jit.save(mod, str(tsp), _extra_files={"x_mixed_precision.json": json.dumps({"precision_label": "mixed precision"})})
    pp = mix / "eval" / "converted" / "e03_run_meta.json"
    pd = json.loads(pp.read_text())
    pd["converted_artifact_sha256"] = FX.sha_file(tsp)
    FX.write_json(pp, pd)
    msel = _write_lam24_records(mix, rd, 3)
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--l-am24-blockb-dir", mix,
                  "--run-id", "x-n2-s42", "--expect-artifact-sha256", msel["winner"]["artifact_of_record"]["sha256"],
                  "--max-samples", 2)
    check("lam24_mixed_precision_refused_as_int8", rc == 2 and "[int8_mixed_precision]" in out, last_result(out))
    ck = sel["winner"]["checkpoint_sha256"]
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "mixed", "--l-am24-blockb-dir", bb,
                  "--run-id", "x-n2-s42", "--islands", "G4", "--run-dir", rd, "--expect-checkpoint-sha256", ck,
                  "--max-samples", 2)
    js = [json.loads(ln.split(" ", 1)[1]) for ln in out.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    check("lam24_mixed_scored_as_mixed", rc == 0 and js and js[0]["label"] == "mixed precision" and js[0]["name"] == "N2",
          last_result(out))


# ------------------------------------------------------------------ test (refused before anything is read)
def sec_test() -> None:
    absent = TMP / "absent_dir"
    cases = [(["--split", "test", "--precision", "int8", "--am29-eval-dir", absent, "--expect-artifact-sha256", "a" * 64],
              "test_unconfirmed"),
             (["--split", "test", "--confirm-test-split", "--precision", "int8", "--am29-eval-dir", absent,
               "--expect-artifact-sha256", "a" * 64], "test_threads"),
             (["--split", "test", "--confirm-test-split", "--threads", 1, "--max-samples", 4, "--precision", "int8",
               "--am29-eval-dir", absent, "--expect-artifact-sha256", "a" * 64], "test_capped"),
             (["--split", "test", "--confirm-test-split", "--threads", 1, "--precision", "int8", "--provenance", absent,
               "--stage", "AM29-T1", "--expect-artifact-sha256", "a" * 64], "test_provenance"),
             (["--split", "TEST", "--confirm-test-split", "--threads", 1, "--precision", "int8", "--am29-eval-dir", absent,
               "--expect-artifact-sha256", "a" * 64], "split"),
             (["--split", "val", "--precision", "int8", "--am29-eval-dir", absent], "expect_artifact_sha256"),
             (["--split", "val", "--precision", "mixed", "--am29-eval-dir", absent, "--islands", "G1"], "precision")]
    miss = []
    for argv, code in cases:
        rc, out = run("scripts/am29_score.py", *argv)
        if rc != 2 or f"[{code}]" not in out or "Nothing was read" not in out:
            miss.append((code, rc, last_result(out)[:100]))
    check("test_rules_refuse_before_reading", not miss and not absent.exists(),
          f"{len(cases)} argument sets refused before any path is opened; misses {miss}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS))
    want = [s for s in ap.parse_args(argv).sections.split(",") if s]
    try:
        for s in SECTIONS:
            if s in want or (s == "blockb" and set(want) & {"rule", "score"}):
                globals()[f"sec_{s}"]()
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("no_unexpected_exception", False, f"{type(e).__name__}: {e}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
