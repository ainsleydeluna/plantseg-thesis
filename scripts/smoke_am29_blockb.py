#!/usr/bin/env python3
"""L-AM29 P3 + P4 smoke: scripts/am29_blockb.py and scripts/am29_score.py on synthetic data, on the CPU.

A complete smoke T1 run (run_am29_qat, 15 epochs x 2 steps) goes through block B (convert, score, select, finalize);
its selected all-INT8 artifact is scored (int8) and its selected weights under a P2 group (mixed precision). A
synthetic artifact in lane/l-am24's block B format (a smoke E6-E2 run named x-n2-s42, one epoch converted by the
pinned scripts/qat_epoch_eval.py, its qat_selection.json, qat_finalize.json and x_blockb.json) is scored through the
same path. Refusals: an unknown, a mixed-precision and a hash-mismatched artifact as int8; the TEST rules (refused
before anything is read; TEST is never scored here). VAL is a synthetic 846-image tree, scored on 2 samples.
L-AM29-FIX (AUDIT-LAM29-1) adds: --protocol upstream (F-01); D's run ID through lane/l-am24's mixed and int8 routes
(F-02); E6 seed 42's block B of record as a mixed-precision source, on synthetic files (F-03); the TEST guards of F-08,
those that need a model run in this process with src/eval/stage_artifacts.py governed_paths_error stubbed (it runs git
status over the whole repository, which no smoke runs) and score_model replaced, so that no TEST file can be read;
block B's --threads, and outside smokes its selection's clean-code and image-digest rules (F-09).

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
    # L-AM29-FIX (AUDIT-LAM29-1 F-09): block B's score takes --threads (required); outside smokes the selection requires
    # every score record to hold code_clean_at_head true and an image digest (this run's e01 record, three ways)
    no_threads = TMP / "eval_no_threads"
    rc, out = run("scripts/am29_blockb.py", "score", "--run-dir", ST["run"], "--eval-dir", no_threads, "--host-label",
                  "smoke", "--max-samples", 2, "--allow-smoke-inputs")
    check("blockb_score_requires_threads", rc == 2 and "--threads" in out and not no_threads.exists(),
          out.strip().splitlines()[-1][:160] if out.strip() else "")
    from scripts import am29_blockb as BB
    bb_rules, _ = BB.load_rules()
    s01 = json.loads((ST["eval"] / "scores" / "e01.json").read_text())

    def score_errors(clean, digest, smoke=False) -> list:
        d = json.loads(json.dumps(s01))
        d["code_clean_at_head"] = clean
        d["score"]["eval_runtime"]["image_digest"] = digest
        return BB._score_errors(d, rules=bb_rules, smoke=smoke)
    digest = "sha256:" + "b8" * 32
    good, dirty, no_digest = score_errors(True, digest), score_errors(False, digest), score_errors(True, None)
    smoke_both = score_errors(False, None, smoke=True)

    def names(errs, key) -> list:
        return [e for e in errs if key in e]
    check("blockb_select_requires_clean_code", bool(names(dirty, "code_clean_at_head"))
          and not names(good, "code_clean_at_head") and not names(smoke_both, "code_clean_at_head"),
          f"outside smokes: {names(dirty, 'code_clean_at_head')}; a smoke selection: none")
    check("blockb_select_requires_image_digest", bool(names(no_digest, "image_digest"))
          and not names(good, "image_digest") and not names(smoke_both, "image_digest"),
          f"outside smokes: {names(no_digest, 'image_digest')}; a smoke selection: none")


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
    # L-AM29-FIX (AUDIT-LAM29-1 F-01): --protocol upstream through the same path; the line and the record carry it
    up_json = TMP / "score_t1_upstream.json"
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--am29-eval-dir", ev, "--arm", "T1",
                  "--expect-artifact-sha256", art, "--max-samples", 2, "--threads", 2, "--protocol", "upstream",
                  "--out-json", up_json)
    js = [json.loads(ln.split(" ", 1)[1]) for ln in out.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    up_doc = json.loads(up_json.read_text()) if up_json.is_file() else {}
    check("score_protocol_upstream", rc == 0 and bool(js) and js[0].get("protocol") == "upstream/1.0.0"
          and (up_doc.get("score") or {}).get("protocol") == "upstream/1.0.0" and "protocol upstream/1.0.0" in out,
          f"{last_result(out)}; AM29_SCORE_JSON protocol {js[0].get('protocol') if js else None!r}")
    # F-08 (b): a score that ran on another thread count than --threads is refused; nothing is printed or written
    from src.eval import am29_score as SC
    real_score = SC.score_model

    def one_more_thread(model, **kw):
        torch.set_num_threads(torch.get_num_threads() + 1)
        return real_score(model, **kw)
    tm_json = TMP / "score_threads_mismatch.json"
    r = score_in_process(["--split", "val", "--precision", "int8", "--am29-eval-dir", ev, "--arm", "T1",
                          "--expect-artifact-sha256", art, "--max-samples", 2, "--threads", 2, "--out-json", tm_json],
                         scorer=one_more_thread)
    check("score_threads_mismatch_refused", r["rc"] == 2 and "[threads_mismatch]" in r["out"]
          and "AM29_SCORE_JSON" not in r["out"] and not tm_json.exists() and r["scorer"] == ["val"],
          last_result(r["out"]))
    # F-08 (c): at --split test the governed-path gate is called and its value recorded (the gate stubbed, the scorer one
    # that reads nothing); at --split val it is not called; a dirty governed path refuses --split test before any read
    rec_t, rec_v = TMP / "score_record_stub.json", TMP / "score_record_stub_v.json"
    rt = score_in_process(["--split", "test", "--confirm-test-split", "--threads", 1, "--out-json", rec_t,
                           "--precision", "int8", "--am29-eval-dir", ev, "--arm", "T1", "--expect-artifact-sha256", art],
                          governed=None, scorer=stub_score)
    rv = score_in_process(["--split", "val", "--precision", "int8", "--am29-eval-dir", ev, "--arm", "T1",
                           "--expect-artifact-sha256", art, "--max-samples", 2, "--out-json", rec_v],
                          governed=None, scorer=stub_score)
    dt = json.loads(rec_t.read_text()) if rec_t.is_file() else {}
    dv = json.loads(rec_v.read_text()) if rec_v.is_file() else {}
    check("score_governed_paths_recorded", rt["rc"] == 0 and "governed_paths_error" in dt
          and dt["governed_paths_error"] is None and rt["governed"] == [str(REPO)] and rt["scorer"] == ["test"]
          and rv["rc"] == 0 and rv["governed"] == [] and "governed_paths_error" not in dv,
          f"--split test: called on {rt['governed']}, recorded {dt.get('governed_paths_error', 'absent')!r}; "
          f"--split val: called {len(rv['governed'])} times, key "
          f"{'present' if 'governed_paths_error' in dv else 'absent'}")
    absent = TMP / "absent_eval_dir"
    never = TMP / "never_governed.json"
    rg = score_in_process(["--split", "test", "--confirm-test-split", "--threads", 1, "--out-json", never,
                           "--precision", "int8", "--am29-eval-dir", absent, "--expect-artifact-sha256", art],
                          governed="refusing 'official': governed paths are dirty or untracked -> ['scripts/x.py']",
                          scorer=tripwire)
    check("score_governed_paths_dirty_refused", rg["rc"] == 2 and "[governed_paths_dirty]" in rg["out"]
          and rg["scorer"] == [] and not absent.exists() and not never.exists(), last_result(rg["out"]))
    # F-08 (d): at --split test a group whose variant record is not eligible (G2; AM-29 item 2(d)) is refused before the
    # scorer; an eligible group (G1) passes that check and meets the scorer, a tripwire here
    base_t = ["--split", "test", "--confirm-test-split", "--threads", 1, "--precision", "mixed", "--am29-eval-dir", ev,
              "--arm", "T1", "--run-dir", ST["run"], "--expect-checkpoint-sha256", ck]
    g2 = score_in_process(base_t + ["--islands", "G2", "--out-json", TMP / "never_g2.json"], governed=None,
                          scorer=tripwire)
    g1 = score_in_process(base_t + ["--islands", "G1", "--out-json", TMP / "never_g1.json"], governed=None,
                          scorer=tripwire)
    check("score_test_group_not_eligible_refused", g2["rc"] == 2 and "[group_not_eligible]" in g2["out"]
          and g2["scorer"] == [] and g1["rc"] == 4 and "tripwire" in g1["out"] and g1["scorer"] == ["test"]
          and not (TMP / "never_g2.json").exists() and not (TMP / "never_g1.json").exists(),
          f"G2: {last_result(g2['out'])[:150]}; G1: {last_result(g1['out'])[:90]}")
    qat_e6_s42_source()                                       # F-03


# ------------------------------------------------------------------ L-AM29-FIX: scripts/am29_score.py in this process
def score_in_process(argv, *, governed=None, scorer=None) -> dict:
    """scripts/am29_score.py main() in this process. src/eval/stage_artifacts.py governed_paths_error is replaced by a
    stub returning `governed` (the real one runs git status over the whole repository, which no smoke runs), and
    src/eval/am29_score.py score_model by `scorer` when one is given; both are restored after, and so is the thread
    count."""
    import contextlib
    import io
    from scripts import am29_score as S
    from src.eval import am29_score as SC
    from src.eval import stage_artifacts as SA
    calls = {"governed": [], "scorer": []}
    saved = (SA.governed_paths_error, SC.score_model, torch.get_num_threads())

    def stub_governed(repo=None):
        calls["governed"].append(str(repo))
        return governed

    def counted(model, **kw):
        calls["scorer"].append(kw.get("split"))
        return scorer(model, **kw)
    buf = io.StringIO()
    try:
        SA.governed_paths_error = stub_governed
        if scorer is not None:
            SC.score_model = counted
        with contextlib.redirect_stdout(buf):
            rc = S.main([str(x) for x in argv])
    finally:
        SA.governed_paths_error, SC.score_model = saved[0], saved[1]
        torch.set_num_threads(saved[2])
    return {"rc": rc, "out": buf.getvalue(), **calls}


def tripwire(model, **kw):
    """In place of score_model where the split is test: reaching it raises, so no TEST file is read."""
    raise AssertionError(f"tripwire: score_model reached with split {kw.get('split')!r}; nothing was scored")


def stub_score(model, *, split, max_samples=None, batch_size=1, protocol="canvas"):
    """In place of score_model for the record checks: a score of no image, so nothing is read."""
    from src.eval.protocols import PROTOCOL_IDS
    return {"split": split, "protocol": PROTOCOL_IDS[protocol], "expected_rows": 0, "actual_rows": 0,
            "max_samples": max_samples, "batch_size": batch_size, "forward_batches": 0, "num_workers": 0,
            "gt_support_sum": 0, "class_map_sha256": None, "manifest_ids_sha256": None, "dataset_level": {},
            "torch_num_threads": int(torch.get_num_threads()), "torch": torch.__version__,
            "eval_runtime": {"stub": "no image was read"}}


# ------------------------------------------------------------------ L-AM29-FIX F-03: E6 seed 42's block B, synthetic
def qat_e6_s42_source() -> None:
    """AUDIT-LAM29-1 F-03: --qat-eval-dir on synthetic files, a complete smoke E6 run at seed 42 and an eval dir in the
    pinned block B's format (qat_selection.json naming epoch 15, qat_finalize.json naming it by sha256). The values of
    record (DL-131, DL-173, AM-29 item 2(a)) bind outside smoke inputs, and at --split test always."""
    from scripts import am29_score as S
    run_dir = TMP / "e6_s42_smoke"
    Q.run_qat(stage="e6", mode="smoke", model=FX.student(5), source_meta=SRC, out_dir=run_dir, seed=42, clip_norm=5.0,
              clip_source="clip_selection", device="cpu", num_workers=0, loaders=FX.loaders(), max_val_batches=1,
              log=FX.quiet)
    ck = FX.rows(run_dir, "epoch_end")[-1]["checkpoint_sha256"]

    def eval_dir(name: str, stage: str = "E6") -> tuple[Path, str]:
        E = TMP / name
        E.mkdir()
        sel = {"format": "qat_selection/1", "rules_sha256": "0" * 64, "stage": stage, "run_dir": str(run_dir),
               "run_id": run_dir.name, "seed": 42, "clip_norm": 5.0,
               "telemetry_sha256": FX.sha_file(run_dir / "qat_telemetry.jsonl"), "eval_dir": str(E),
               "winner": {"epoch": 15, "value": 0.33, "checkpoint": "epoch_ckpts/e15.pt", "checkpoint_sha256": ck},
               "smoke_inputs": True}
        ssha = FX.write_json(E / "qat_selection.json", sel)
        FX.write_json(E / "qat_finalize.json", {"format": "qat_finalize/1", "status": "finalized", "stage": stage,
                                                "run_id": run_dir.name, "seed": 42,
                                                "qat_selection": {"path": "qat_selection.json", "sha256": ssha}})
        return E, ssha
    E, ssha = eval_dir("e6_s42_eval")
    mixed = ["--split", "val", "--precision", "mixed", "--islands", "G1", "--run-dir", run_dir,
             "--expect-checkpoint-sha256", ck, "--max-samples", 2, "--allow-smoke-inputs"]
    out_json = TMP / "score_e6_s42_g1.json"
    rc, out = run("scripts/am29_score.py", *mixed, "--qat-eval-dir", E, "--qat-selection-sha256", ssha,
                  "--out-json", out_json)
    js = [json.loads(ln.split(" ", 1)[1]) for ln in out.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    doc = json.loads(out_json.read_text()) if out_json.is_file() else {}
    src = doc.get("source") or {}
    check("qat_e6_s42_scored_mixed", rc == 0 and bool(js) and js[0]["label"] == "mixed precision"
          and js[0]["name"] == "E6-s42" and js[0]["stage"] == "E6" and doc.get("source_kind") == "qat"
          and src.get("selection_sha256") == ssha and src.get("finalize_sha256") == FX.sha_file(E / "qat_finalize.json")
          and src.get("checkpoint_sha256") == ck and src.get("epoch") == 15, last_result(out))
    rc, out = run("scripts/am29_score.py", *mixed, "--qat-eval-dir", E, "--qat-selection-sha256", "ab" * 32)
    check("qat_e6_s42_selection_sha256_refused", rc == 2 and "[qat_selection_sha256_mismatch]" in out, last_result(out))
    E2, ssha2 = eval_dir("e6e2_s42_eval", stage="E6-E2")
    rc, out = run("scripts/am29_score.py", *mixed, "--qat-eval-dir", E2, "--qat-selection-sha256", ssha2)
    check("qat_e6_s42_wrong_stage_refused", rc == 2 and "[qat_stage]" in out, last_result(out))
    rc, out = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--qat-eval-dir", E,
                  "--qat-selection-sha256", ssha, "--expect-artifact-sha256", "a" * 64, "--allow-smoke-inputs")
    check("qat_e6_s42_int8_refused", rc == 2 and "[qat_eval_dir_precision]" in out and "Nothing was read" in out,
          last_result(out))
    # the values of record, by argument and before anything is read: outside smoke inputs, and at --split test even with
    # them
    absent = TMP / "absent_qat_eval_dir"
    rec = ["--precision", "mixed", "--islands", "G1", "--run-dir", absent, "--qat-eval-dir", absent]
    cases = [(["--split", "val", *rec, "--qat-selection-sha256", ssha, "--expect-checkpoint-sha256",
               S.E6_S42_CHECKPOINT_SHA256], "qat_selection_not_of_record"),
             (["--split", "val", *rec, "--qat-selection-sha256", S.E6_S42_SELECTION_SHA256,
               "--expect-checkpoint-sha256", ck], "checkpoint_not_of_record"),
             (["--split", "test", "--confirm-test-split", "--threads", 1, "--out-json", TMP / "never_qat.json", *rec,
               "--qat-selection-sha256", ssha, "--expect-checkpoint-sha256", ck, "--allow-smoke-inputs"],
              "qat_selection_not_of_record")]
    miss = []
    for argv, code in cases:
        rc, out = run("scripts/am29_score.py", *argv)
        if rc != 2 or f"[{code}]" not in out or "Nothing was read" not in out:
            miss.append((code, rc, last_result(out)[:100]))
    check("qat_e6_s42_values_of_record", not miss and not absent.exists(),
          f"{len(cases)} argument sets refused before any path is opened; misses {miss}")
    # qat_finalize.json of record (DL-173): with this process's selection and checkpoint constants set to the synthetic
    # ones, outside smoke inputs the synthetic finalize record is refused, before the run is read
    saved = (S.E6_S42_SELECTION_SHA256, S.E6_S42_CHECKPOINT_SHA256)
    try:
        S.E6_S42_SELECTION_SHA256, S.E6_S42_CHECKPOINT_SHA256 = ssha, ck
        r = score_in_process(["--split", "val", "--precision", "mixed", "--islands", "G1", "--run-dir", run_dir,
                              "--expect-checkpoint-sha256", ck, "--qat-eval-dir", E, "--qat-selection-sha256", ssha,
                              "--max-samples", 2], scorer=tripwire)
    finally:
        S.E6_S42_SELECTION_SHA256, S.E6_S42_CHECKPOINT_SHA256 = saved
    dl173 = "b2c7705c32b800f6fbbefff40d201cf828d2736b053236297e550189a7830c03"
    check("qat_e6_s42_finalize_of_record", r["rc"] == 2 and "[qat_finalize_not_of_record]" in r["out"]
          and r["scorer"] == [] and S.E6_S42_FINALIZE_SHA256 == dl173, last_result(r["out"]))


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
    # L-AM29-FIX (AUDIT-LAM29-1 F-02): D (x-d-n3-s42; AM-29 item 3(d)) is one of L_AM24_RUNS. A smoke E6-E2 run named
    # x-d-n3-s42 in lane/l-am24's block B format: the mixed route refuses an unknown run ID and scores D's weights under
    # a P2 group; the int8 route scores its artifact
    d_item = TMP / "x-d-n3-s42-item"
    d_rd = d_item / "x-d-n3-s42"
    d_item.mkdir()
    Q.run_qat(stage="e6-e2", mode="smoke", model=FX.student(6), source_meta=SRC, out_dir=d_rd, seed=42, clip_norm=5.0,
              clip_source="clip_selection", device="cpu", num_workers=0, loaders=FX.loaders(), max_val_batches=1,
              log=FX.quiet)
    d_bb = TMP / "x-d-n3-bb"
    rc_c, out_c = run("scripts/qat_epoch_eval.py", "convert-epoch", "--run-dir", d_rd, "--eval-dir", d_bb / "eval",
                      "--epoch", 3, "--purpose", "record", "--host-label", "smoke", "--allow-smoke-inputs")
    d_sel = _write_lam24_records(d_bb, d_rd, 3)
    d_mixed = ["--split", "val", "--precision", "mixed", "--l-am24-blockb-dir", d_bb, "--islands", "G1",
               "--run-dir", d_rd, "--expect-checkpoint-sha256", d_sel["winner"]["checkpoint_sha256"], "--max-samples", 2]
    rc_u, out_u = run("scripts/am29_score.py", *d_mixed, "--run-id", "x-n5-s42")
    rc_m, out_m = run("scripts/am29_score.py", *d_mixed, "--run-id", "x-d-n3-s42")
    jm = [json.loads(ln.split(" ", 1)[1]) for ln in out_m.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    check("lam24_d_mixed_route", rc_c == 0 and rc_u == 2 and "[artifact_unknown]" in out_u and rc_m == 0 and bool(jm)
          and jm[0]["label"] == "mixed precision" and jm[0]["name"] == "D",
          f"x-n5-s42: {last_result(out_u)[:100]}; x-d-n3-s42: {last_result(out_m)}")
    rc_i, out_i = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--l-am24-blockb-dir", d_bb,
                      "--run-id", "x-d-n3-s42", "--expect-artifact-sha256",
                      d_sel["winner"]["artifact_of_record"]["sha256"], "--max-samples", 2)
    ji = [json.loads(ln.split(" ", 1)[1]) for ln in out_i.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    check("lam24_d_int8_route", rc_i == 0 and bool(ji) and ji[0]["label"] == "int8" and ji[0]["name"] == "D"
          and ji[0]["stage"] == "E6-E2", last_result(out_i))
    # AM-29a item 7(a): E6-R2-MT is bound by its run ID, x-n2r2t-s42. A smoke run so named (an E6-E2 run: the stage is
    # the fixture's) in lane/l-am24's block B format is scored as int8; an unknown run ID is still refused
    t_item = TMP / "x-n2r2t-s42-item"
    t_rd = t_item / "x-n2r2t-s42"
    t_item.mkdir()
    Q.run_qat(stage="e6-e2", mode="smoke", model=FX.student(7), source_meta=SRC, out_dir=t_rd, seed=42, clip_norm=5.0,
              clip_source="clip_selection", device="cpu", num_workers=0, loaders=FX.loaders(), max_val_batches=1,
              log=FX.quiet)
    t_bb = TMP / "x-n2r2t-bb"
    rc_tc, out_tc = run("scripts/qat_epoch_eval.py", "convert-epoch", "--run-dir", t_rd, "--eval-dir", t_bb / "eval",
                        "--epoch", 3, "--purpose", "record", "--host-label", "smoke", "--allow-smoke-inputs")
    t_sel = _write_lam24_records(t_bb, t_rd, 3)
    t_int8 = ["--split", "val", "--precision", "int8", "--l-am24-blockb-dir", t_bb, "--expect-artifact-sha256",
              t_sel["winner"]["artifact_of_record"]["sha256"], "--max-samples", 2]
    rc_t, out_t = run("scripts/am29_score.py", *t_int8, "--run-id", "x-n2r2t-s42")
    rc_x, out_x = run("scripts/am29_score.py", *t_int8, "--run-id", "x-n2r2t-s43")
    jt = [json.loads(ln.split(" ", 1)[1]) for ln in out_t.splitlines() if ln.startswith("AM29_SCORE_JSON ")]
    check("lam24_e6r2mt_bound_by_run_id", rc_tc == 0 and rc_t == 0 and bool(jt) and jt[0]["label"] == "int8"
          and jt[0]["name"] == "E6-R2-MT" and rc_x == 2 and "[artifact_unknown]" in out_x,
          f"x-n2r2t-s42: {last_result(out_t)}; x-n2r2t-s43: {last_result(out_x)[:110]}")


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
    # L-AM29-FIX: --split test without --out-json (AUDIT-LAM29-1 F-08 (a)); --protocol upstream with --provenance (F-01)
    rc_o, out_o = run("scripts/am29_score.py", "--split", "test", "--confirm-test-split", "--threads", 1, "--precision",
                      "int8", "--am29-eval-dir", absent, "--expect-artifact-sha256", "a" * 64)
    check("test_rules_out_json_required", rc_o == 2 and "[test_out_json]" in out_o and "Nothing was read" in out_o
          and not absent.exists(), last_result(out_o))
    rc_p, out_p = run("scripts/am29_score.py", "--split", "val", "--precision", "int8", "--provenance", absent, "--stage",
                      "AM29-T1", "--expect-artifact-sha256", "a" * 64, "--protocol", "upstream")
    check("test_rules_provenance_canvas_only", rc_p == 2 and "[protocol]" in out_p and "Nothing was read" in out_p
          and not absent.exists(), last_result(out_p))


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
