#!/usr/bin/env python3
"""d3 + d5 (lane 6, L-AM4 + L-AM1q): the QAT epoch selection and the U4 clip selection, on synthetic inputs.

Real QAT runs of the trainer in mode "smoke" (E5 seed 42 at clip 1.0 and 5.0, the same seed so the same
batches, plus runs made non-finite) give the run records and checkpoints. Each record is then re-labelled
mode "real" (a synthetic real-mode record; the checkpoints are untouched), so the strict selection path
runs without any smoke allowance. The eval directories are fabricated: per-epoch provenance, a TorchScript
zip carrying only its identity, and evaluator artifacts whose summaries hold chosen values, with every
hash consistent, so each rule can be driven exactly.

d3: the epoch rule (strict >, tie -> the earlier epoch, the fake-quant VAL argmax never selects),
excluded non-finite epochs, and every refusal of the chain. d5: the exact 0.1 pp clip band in both
orientations, the AM-21 item 3 rejection, and the clip refusals. No PlantSeg data, no checkpoint of
record; temp files outside the repository.

Ends with one RESULT line; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_qat_fixtures import confident_student, make_qat_run, safe_tmpdir  # noqa: E402

TMP = safe_tmpdir("smoke_qat_selection_")
os.environ["PLANTSEG_IMAGE_DIGEST"] = "sha256:" + "5" * 64       # the runs record a non-null digest

import numpy as np  # noqa: E402

import scripts.select_clip as SC  # noqa: E402
import scripts.select_qat_epoch as SE  # noqa: E402
from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant import qat_select as S  # noqa: E402

results: list[tuple[str, bool, str]] = []
SECTIONS = ("d3", "oq1", "refusals", "d5")
GT_SUM = 159_279_104


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def run_cli(main, argv) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    return rc, (lines[-1] if lines else buf.getvalue()[-300:])


# ---------------------------------------------------------------- synthetic real-mode run records
def relabel(run: Path, *, fq: dict | None = None, meta: dict | None = None, fingerprints=None,
            host: dict | None = None) -> str:
    """Re-label a smoke run's record mode "real" (+ optional edits). Returns the new telemetry sha256."""
    tel = run / Q.TELEMETRY_NAME
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    rows[0].update({"mode": "real", "u4_pilot": True, "clip_source": "u4_pilot", **(meta or {})})
    if host:
        rows[0]["host"] = {**rows[0]["host"], **host}
    for r in rows:
        if r.get("event") == "epoch_end" and fq:
            r["fake_quant_val_all_class_miou"] = fq[r["epoch"]]
        if fingerprints and r.get("event") in ("train", "step_error"):
            r["batch_sha256"] = fingerprints(r["step"], r["batch_sha256"])
    tel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Q.sha256_file(tel)


def clone_run(src: Path, dst: Path) -> Path:
    """A copy of a run: checkpoints hard-linked (never rewritten), the record copied."""
    (dst / Q.EPOCH_DIR).mkdir(parents=True)
    for p in (src / Q.EPOCH_DIR).iterdir():
        os.link(p, dst / Q.EPOCH_DIR / p.name)
    shutil.copy2(src / Q.TELEMETRY_NAME, dst / Q.TELEMETRY_NAME)
    return dst


# ---------------------------------------------------------------- fabricated eval directories
def summary(value: float, run_id: str, ts_sha: str, stage: str, *, rows: int = 846, split: str = "val",
            commit: str = "a" * 40, digest: str | None = "sha256:" + "6" * 64, protocol="core_preprocess/1.0.0",
            status="provisional", metric: str = "b" * 64) -> dict:
    gt = [1000] * 116
    gt[0] = (GT_SUM if rows == 846 else rows * 100_000) - 115_000
    return {"schema_version": "plantseg-eval/1.0.0",
            "run": {"run_id": run_id, "artifact_status": status, "stage": stage, "model_role": "student",
                    "precision": "int8_qat", "quant_backend": "qnnpack", "checkpoint_sha256": ts_sha,
                    "random_init": False, "repo_commit": commit, "governed_paths_clean": True,
                    "metric_impl_sha256": metric, "config_sha256": "c" * 64,
                    "env": {"torch": "2.1.0+cpu", "device": "cpu"},
                    "eval_runtime": {"batch_size": 1, "forward_batches": rows, "num_workers": 0,
                                     "torch_num_threads": 4, "image_digest": digest}},
            "dataset": {"split": split, "expected_rows": rows, "actual_rows": rows, "preprocess_protocol": protocol,
                        "split_manifest_sha256": "d" * 64, "class_map_sha256": "e" * 64},
            "dataset_level": {"all_class_miou": value}, "per_class": {"gt_support": gt}}


def write_artifact(d: Path, s: dict) -> None:
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(s, sort_keys=True), encoding="utf-8")
    (d / "per_image.jsonl").write_text("", encoding="utf-8")
    with open(d / "sufficient_stats.npz", "wb") as fh:
        np.savez(fh, tp=np.zeros(116, dtype=np.int64))
    lines = [f"{Q.sha256_file(d / n)}  {n}" for n in ARTIFACT_FILES]
    (d / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_summary(ev: Path, e: int, **edits) -> str:
    """Edit scores/eNN/summary.json (dotted keys) and re-write its MANIFEST; returns the new sha256."""
    d = ev / "scores" / f"e{e:02d}"
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    for k, v in edits.items():
        node = s
        parts = k.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = v
    (d / "summary.json").write_text(json.dumps(s, sort_keys=True), encoding="utf-8")
    lines = [f"{Q.sha256_file(d / n)}  {n}" for n in ARTIFACT_FILES]
    (d / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Q.sha256_file(d / "summary.json")


def edit_eval_record(ev: Path, fn) -> None:
    p = ev / "qat_epoch_eval.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    fn(doc)
    p.write_text(json.dumps(doc), encoding="utf-8")


def row_of(doc: dict, e: int) -> dict:
    return next(r for r in doc["epochs"] if r["epoch"] == e)


def fabricate(run: Path, ev: Path, values: dict, *, excluded=(), tamper: dict | None = None) -> None:
    """converted/, scores/, qat_convert.json and qat_epoch_eval.json for `run`, scored with `values`."""
    tamper = tamper or {}
    commit, host_label = tamper.get("commit_all", "a" * 40), tamper.get("host_label", "fab-host")
    rec = A.read_run_record(run)
    meta = rec["run_meta"]
    conv, scores = ev / A.CONVERTED_DIR, ev / A.SCORES_DIR
    conv.mkdir(parents=True)
    outcomes, rows = [], []
    for e in range(1, 16):
        names = A.epoch_names(e)
        ck_sha = rec["ends"][e]["checkpoint_sha256"]
        if e in excluded:
            (conv / names["not_convertible"]).write_text(json.dumps(
                {"status": "not convertible", "rule": A.NOT_CONVERTIBLE_RULE, "qat_checkpoint_sha256": ck_sha,
                 "failing": ["fabricated"]}), encoding="utf-8")
            outcomes.append({"epoch": e, "status": "not convertible"})
            rows.append({"epoch": e, "checkpoint_sha256": ck_sha, "status": "excluded", "rule": A.NOT_CONVERTIBLE_RULE,
                         "not_convertible": f"{A.CONVERTED_DIR}/{names['not_convertible']}"})
            continue
        ident = {"schema": "plantseg-int8-torchscript/1.0.0", "stage": meta["stage"], "quantization": "qat",
                 "num_classes": 116, "engine": "qnnpack", "artifact_role": "accuracy",
                 "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
                 "qat_checkpoint_sha256": tamper.get(("identity_ck", e), ck_sha), "epoch": e,
                 "run_id": rec["run_id"], "qconfig_fingerprint": "f" * 16, "state_dict_companion": names["state_dict"]}
        ts = conv / names["torchscript"]
        with zipfile.ZipFile(ts, "w") as z:
            z.writestr(f"e{e:02d}/extra/plantseg_int8.json", json.dumps(ident))
            z.writestr(f"e{e:02d}/data.pkl", f"fabricated {rec['run_id']} e{e:02d}")
        ts_sha = Q.sha256_file(ts)
        sd = conv / names["state_dict"]
        sd.write_bytes(f"fabricated state_dict {rec['run_id']} e{e:02d}".encode())
        prov = {"stage": meta["stage"], "epoch": e, "run_id": rec["run_id"],
                "state_dict_artifact": {"path": names["state_dict"], "sha256": Q.sha256_file(sd)},
                "git_head": tamper.get(("prov_head", e), commit), "host_label": host_label, "cpu_model": "fab-cpu",
                "purpose": tamper.get("purpose", "record"),
                "qat_checkpoint_sha256": tamper.get(("prov_ck", e), ck_sha),
                "converted_artifact": names["torchscript"], "converted_artifact_sha256": ts_sha,
                "verification": {"checks": {"weights_per_channel_int8": True, "no_undeclared_float_module": True,
                                            "torchscript_float_boundary": True,
                                            "parity_torchscript_state_dict_eager": True}}}
        pp = conv / names["run_meta"]
        pp.write_text(json.dumps(prov), encoding="utf-8")
        run_id = f"{rec['run_id']}_e{e:02d}_{ck_sha[:12]}"
        kw = {"commit": commit, **tamper.get(("summary", e), {})}
        write_artifact(scores / f"e{e:02d}", summary(values[e], run_id, ts_sha, meta["stage"], **kw))
        outcomes.append({"epoch": e, "status": "converted"})
        rows.append({"epoch": e, "checkpoint_sha256": ck_sha, "status": "scored", "run_id": run_id,
                     "provenance": {"path": f"{A.CONVERTED_DIR}/{names['run_meta']}", "sha256": Q.sha256_file(pp)},
                     "converted_artifact_sha256": ts_sha, "score_dir": f"{A.SCORES_DIR}/e{e:02d}",
                     "summary": {"path": f"{A.SCORES_DIR}/e{e:02d}/summary.json",
                                 "sha256": Q.sha256_file(scores / f"e{e:02d}" / "summary.json")},
                     "all_class_miou": values[e],
                     "evaluator_command": tamper.get("command", ["-B", "scripts/evaluate_model.py", "--stage", meta["stage"]]),
                     "torch_num_threads": 4, "affinity_cpus": 4})
    head = {"run_dir": rec["run_dir"], "run_id": rec["run_id"], "stage": meta["stage"],
            "telemetry_sha256": rec["telemetry_sha256"], "purpose": tamper.get("purpose", "record"),
            "host_label": host_label, "cpu_model": "fab-cpu", "git_head": commit,
            "smoke_inputs": tamper.get("smoke_inputs", False)}
    (ev / "qat_convert.json").write_text(json.dumps({"format": "qat_convert/1", **head, "outcomes": outcomes,
                                                     "git_head": tamper.get("convert_head", commit)}),
                                         encoding="utf-8")
    if tamper.get("drop_epoch"):
        rows = [r for r in rows if r["epoch"] != tamper["drop_epoch"]]
    (ev / "qat_epoch_eval.json").write_text(json.dumps({"format": "qat_epoch_eval/1", **head, "epochs": rows}),
                                            encoding="utf-8")


def select(run: Path, ev: Path, sha: str, out: Path | None = None) -> tuple[int, str, dict]:
    argv = ["--run-dir", str(run), "--eval-dir", str(ev), "--expect-telemetry-sha256", sha]
    if out:
        argv += ["--out", str(out)]
    rc, res = run_cli(SE.main, argv)
    p = out or ev / "qat_selection.json"
    return rc, res, (json.loads(p.read_text()) if rc == 0 and p.is_file() else {})


def flat(v: float) -> dict:
    return {e: v for e in range(1, 16)}


def prepare_runs(cache: Path | None) -> dict:
    """The four raw smoke-mode runs, trained once (or cloned from --cache-dir, where they are kept raw)."""
    names = ("run_clip1.0", "run_clip5.0", "run_nan10", "run_nan1")
    if cache is not None and (cache / "READY").is_file():
        return {n: clone_run(cache / n, TMP / n) for n in names}
    src = confident_student(seed=0, steps=30)            # any trained source: the scores are fabricated
    plan = {"run_clip1.0": dict(clip=1.0), "run_clip5.0": dict(clip=5.0),
            "run_nan10": dict(clip=1.0, nan_at=(10,)), "run_nan1": dict(clip=1.0, nan_at=(1,))}
    out = {}
    for n, kw in plan.items():
        make_qat_run(TMP / n, model=src, stage="e5", **kw)
        out[n] = TMP / n
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        for n in names:
            clone_run(TMP / n, cache / n)
        (cache / "READY").write_text("raw smoke runs\n", encoding="utf-8")
    return out


def vals(**kw) -> dict:
    v = {e: 0.30 for e in range(1, 16)}
    for k, x in kw.items():
        v[int(k[1:])] = x
    return v


# ---------------------------------------------------------------- the cases
CASES: list[tuple[str, str, object]] = []          # (name, section, fn); a case reports one or more checks
FX: dict = {}


def case(section: str):
    def deco(fn):
        CASES.append((fn.__name__, section, fn))
        return fn
    return deco


def register(section: str, name: str, fn) -> None:
    CASES.append((name, section, fn))


def clip_cli(pairs, out: Path) -> tuple[int, str]:
    argv = []
    for run, ev in pairs:
        argv += ["--candidate", str(run), str(ev)]
    return run_cli(SC.main, argv + ["--out", str(out)])


def selected_eval(run: Path, sha: str, name: str, values: dict, **kw) -> Path:
    ev = TMP / name
    fabricate(run, ev, values, **kw)
    select(run, ev, sha)
    return ev


@functools.cache
def ev_clip1() -> Path:
    return selected_eval(FX["base"][1.0], FX["sha"][1.0], "ev_clip1_selected", vals(e07=0.30))


@functools.cache
def ev_clip5() -> Path:
    return selected_eval(FX["base"][5.0], FX["sha"][5.0], "ev_clip5_selected", vals(e07=0.31))


@functools.cache
def nan5_run() -> Path:
    """Clip 5.0's run with its record non-finite from epoch 14 (AM-21 item 3 rejects it)."""
    run = clone_run(FX["base"][5.0], TMP / "run_clip5_nan")
    tel = run / Q.TELEMETRY_NAME
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r.get("event") == "epoch_end" and r["epoch"] >= 14:
            r["state_finite"] = False
            r["nonfinite_since_step"] = 27
    tel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return run


# ---- d3: the epoch rule
@case("d3")
def d3_rules_agree_with_config():
    rules = FX["rules"]
    check("d3_rules_agree_with_config", rules["qat_epoch"]["epochs"] == 15
          and rules["qat_clip"]["candidates"] == [1.0, 5.0], FX["rules_sha"][:12])


def _rule_case(name: str, values: dict, expect: int):
    def run():
        r1 = FX["base"][1.0]
        ev = TMP / f"ev_{name}"
        fabricate(r1, ev, values)
        rc, res, sel = select(r1, ev, FX["sha"][1.0])
        ok = rc == 0 and sel["winner"]["epoch"] == expect
        extra = ""
        if name == "d3_decoy_converted_wins":
            ok = ok and sel["fake_quant_val_argmax_epoch"] == 9 and sel["fake_quant_val_used"] is False
            extra = f"; fake-quant argmax e{sel.get('fake_quant_val_argmax_epoch')}"
        check(name, ok, f"{res}{extra}")
    return run


for _n, _v, _e in [("d3_tie_goes_to_earlier_epoch", vals(e05=0.41, e09=0.41), 5),
                   ("d3_higher_later_epoch_wins", vals(e05=0.41, e09=0.42), 9),
                   ("d3_decoy_converted_wins", vals(e04=0.45), 4),
                   ("d3_strictly_greater_by_one_ulp", vals(e03=0.4, e11=float(np.nextafter(0.4, 1.0))), 11)]:
    register("d3", _n, _rule_case(_n, _v, _e))


@case("d3")
def winner_is_existing_epoch():
    r1 = FX["base"][1.0]
    ev = TMP / "ev_ok"
    fabricate(r1, ev, vals(e07=0.5))
    rc, res, sel = select(r1, ev, FX["sha"][1.0])
    w = sel.get("winner", {})
    check("winner_is_existing_epoch",
          rc == 0 and w.get("epoch") == 7 and w.get("checkpoint_sha256") == Q.sha256_file(r1 / Q.EPOCH_DIR / "e07.pt")
          and w.get("artifact_of_record", {}).get("sha256") == Q.sha256_file(ev / w["artifact_of_record"]["path"])
          and sel.get("rule_trace", [""])[0].startswith("rule configs/qat_selection_rules.json qat_epoch")
          and res == "RESULT: SELECTED epoch 07 (E5, seed 42, clip 1.0)", res)


@case("d3")
def d3_refuses_existing_output():
    r1 = FX["base"][1.0]
    ev = TMP / "ev_twice"
    fabricate(r1, ev, vals(e07=0.5))
    select(r1, ev, FX["sha"][1.0])
    rc, res, _ = select(r1, ev, FX["sha"][1.0])
    check("d3_refuses_existing_output", rc == 2 and "[output_exists]" in res, res)


# ---- d3 / OQ1: non-finite epochs
@case("oq1")
def d3_nonfinite_epochs_excluded():
    ev = TMP / "ev_nan10"
    fabricate(FX["nan10"], ev, vals(e02=0.33), excluded=range(5, 16))
    rc, res, sel = select(FX["nan10"], ev, FX["sha_n10"])
    check("d3_nonfinite_epochs_excluded", rc == 0 and sel["winner"]["epoch"] == 2
          and sel["excluded_epochs"] == list(range(5, 16))
          and any("excluded: non-finite state (AM-19 item 3(a))" in t for t in sel["rule_trace"]), res)


@case("oq1")
def d3_scored_nonfinite_epoch_refused():
    ev = TMP / "ev_nan10_scored"
    fabricate(FX["nan10"], ev, vals(e02=0.33))
    rc, res, _ = select(FX["nan10"], ev, FX["sha_n10"])
    check("d3_scored_nonfinite_epoch_refused", rc == 2 and "[scored_nonfinite_state]" in res, res)


@case("oq1")
def d3_excluding_a_finite_epoch_refused():
    ev = TMP / "ev_nan10_finite_excluded"
    fabricate(FX["nan10"], ev, vals(), excluded=(2,) + tuple(range(5, 16)))
    rc, res, _ = select(FX["nan10"], ev, FX["sha_n10"])
    check("d3_excluding_a_finite_epoch_refused", rc == 2 and "[exclusion_inconsistent]" in res, res)


@case("oq1")
def d3_no_convertible_epoch_refused_exit_2():
    ev = TMP / "ev_nan1"
    fabricate(FX["nan1"], ev, vals(), excluded=range(1, 16))
    rc, res, _ = select(FX["nan1"], ev, FX["sha_n1"])
    check("d3_no_convertible_epoch_refused_exit_2", rc == 2 and "[no_convertible_epoch]" in res, res)


# ---- d3: refusals of the chain (P23-P26), one link at a time
def _tamper_case(name: str, tamper: dict, expect_rc: int, code: str):
    def run():
        r1 = FX["base"][1.0]
        ev = TMP / f"ev_{name}"
        fabricate(r1, ev, vals(e07=0.5), tamper=tamper)
        rc, res, _ = select(r1, ev, FX["sha"][1.0])
        check(name, rc == expect_rc and f"[{code}]" in res, res)
    return run


for _n, _t, _rc, _c in [
        ("d3_partial_eval_incomplete", {"drop_epoch": 9}, 3, "eval_incomplete"),
        ("d3_identity_sha_mismatch_refused", {("identity_ck", 6): "0" * 64}, 2, "chain_broken"),
        ("d3_provenance_checkpoint_mismatch_refused", {("prov_ck", 6): "0" * 64}, 2, "chain_broken"),
        ("d3_capped_summary_refused", {("summary", 4): {"rows": 64}}, 2, "summary_values"),
        ("d3_non_val_split_refused", {("summary", 4): {"split": "train"}}, 2, "summary_values"),
        ("d3_upstream_protocol_refused", {("summary", 4): {"protocol": "upstream/1.0.0"}}, 2, "summary_values"),
        ("d3_smoke_status_refused", {("summary", 4): {"status": "smoke"}}, 2, "summary_values"),
        ("d3_summary_commit_not_eval_commit_refused", {("summary", 8): {"commit": "9" * 40}}, 2, "eval_dir_mixed"),
        ("d3_metric_impl_differs_across_summaries", {("summary", 8): {"metric": "9" * 64}}, 2, "summaries_differ"),
        ("d3_null_image_digest_refused", {("summary", 8): {"digest": None}}, 2, "summaries_differ"),
        ("d3_provenance_other_commit_refused", {("prov_head", 6): "9" * 40}, 2, "eval_dir_mixed"),
        ("d3_evaluator_not_literal_refused", {"command": ["-B", "/tmp/driver.py"]}, 2, "evaluator_not_literal"),
        ("d3_timing_purpose_refused", {"purpose": "timing"}, 2, "purpose_not_record"),
        ("d3_two_commits_refused", {"convert_head": "9" * 40}, 2, "eval_dir_mixed"),
        ("d3_smoke_input_records_refused", {"smoke_inputs": True}, 2, "smoke_inputs")]:
    register("refusals", _n, _tamper_case(_n, _t, _rc, _c))


def _edit_case(name: str, edit, expect_rc: int, code: str, needle: str = ""):
    """Fabricate a clean eval directory, apply `edit(ev)`, then select."""
    def run():
        r1 = FX["base"][1.0]
        ev = TMP / f"ev_{name}"
        fabricate(r1, ev, vals(e07=0.5))
        edit(ev)
        rc, res, _ = select(r1, ev, FX["sha"][1.0])
        check(name, rc == expect_rc and f"[{code}]" in res and needle in res, res)
    return run


def _unlink(rel: str):
    return lambda ev: (ev / rel).unlink()


def _append_space(rel: str):
    return lambda ev: (ev / rel).write_text((ev / rel).read_text(encoding="utf-8") + " ", encoding="utf-8")


def _summary_identity(ev: Path) -> None:
    new_sha = rewrite_summary(ev, 6, run__run_id="another_run_e06_000000000000")
    edit_eval_record(ev, lambda d: row_of(d, 6)["summary"].update(sha256=new_sha))


_N5 = A.epoch_names(5)
for _n, _ed, _rc, _c, _needle in [
        ("d3_sha_mismatch_refused", lambda ev: (ev / "scores" / "e03" / "per_image.jsonl").write_text("{}\n"), 2,
         "score_artifact_invalid", ""),
        ("d3_stop_file_refused", lambda ev: (ev / "converted" / "e05_STOP.json").write_text("{}"), 2, "stop_present", ""),
        ("d3_extra_scores_entry_refused", lambda ev: (ev / "scores" / "tmp_e07").mkdir(), 2, "scores_dir_contents", ""),
        ("d3_unreadable_record_refused", lambda ev: (ev / "qat_epoch_eval.json").write_text("{"), 2,
         "record_unreadable", ""),
        ("d3_missing_eval_record_exit_3", _unlink("qat_epoch_eval.json"), 3, "eval_record_missing", ""),
        ("d3_row_checkpoint_mismatch_refused",
         lambda ev: edit_eval_record(ev, lambda d: row_of(d, 3).update(checkpoint_sha256="0" * 64)), 2,
         "chain_checkpoint", ""),
        ("d3_unscored_epoch_exit_3", lambda ev: edit_eval_record(ev, lambda d: row_of(d, 4).update(status="pending")),
         3, "score_missing", ""),
        ("d3_provenance_not_scored_one_refused", _append_space(f"{A.CONVERTED_DIR}/{_N5['run_meta']}"), 2,
         "chain_provenance", ""),
        ("d3_summary_not_scored_one_refused", lambda ev: rewrite_summary(ev, 6, dataset_level__all_class_miou=0.31), 2,
         "chain_summary", "not the scored"),
        ("d3_summary_identity_refused", _summary_identity, 2, "chain_summary", "expected"),
        ("d3_missing_provenance_exit_3", _unlink(f"{A.CONVERTED_DIR}/{_N5['run_meta']}"), 3, "conversion_missing", ""),
        ("d3_missing_torchscript_exit_3", _unlink(f"{A.CONVERTED_DIR}/{_N5['torchscript']}"), 3, "conversion_missing",
         ""),
        ("d3_missing_score_exit_3", _unlink(f"{A.SCORES_DIR}/e05/summary.json"), 3, "score_missing", "summary.json")]:
    register("refusals", _n, _edit_case(_n, _ed, _rc, _c, _needle))


@case("refusals")
def d3_score_outside_unit_interval_refused():
    _tamper_case_values("d3_score_outside_unit_interval_refused", vals(e08=1.5))


@case("refusals")
def d3_score_not_float_refused():
    _tamper_case_values("d3_score_not_float_refused", vals(e08=1))


def _tamper_case_values(name: str, values: dict) -> None:
    r1 = FX["base"][1.0]
    ev = TMP / f"ev_{name}"
    fabricate(r1, ev, values)
    rc, res, _ = select(r1, ev, FX["sha"][1.0])
    check(name, rc == 2 and "[score_invalid]" in res, res)


@case("refusals")
def d3_swapped_eval_dir_refused():
    ev5 = TMP / "ev_of_clip5"
    fabricate(FX["base"][5.0], ev5, vals(e07=0.5))
    rc, res, _ = select(FX["base"][1.0], ev5, FX["sha"][1.0])
    check("d3_swapped_eval_dir_refused", rc == 2 and "[eval_record_mismatch]" in res, res)


@case("refusals")
def d3_test_path_refused():
    rc, res, _ = select(FX["base"][1.0], TMP / "latest_eval", FX["sha"][1.0])
    check("d3_test_path_refused", rc == 2 and "[eval_dir_test_path]" in res and not (TMP / "latest_eval").exists(),
          res)


@case("refusals")
def d3_telemetry_sha_mismatch_refused():
    r1 = FX["base"][1.0]
    ev = TMP / "ev_wrong_sha"
    fabricate(r1, ev, vals(e07=0.5))
    rc, res, _ = select(r1, ev, "f" * 64)
    check("d3_telemetry_sha_mismatch_refused", rc == 2 and "[telemetry_sha256_mismatch]" in res, res)


@case("refusals")
def d3_incomplete_run_refused_exit_3():
    r1 = FX["base"][1.0]
    ev = TMP / "ev_for_incomplete"
    fabricate(r1, ev, vals(e07=0.5))
    inc = clone_run(r1, TMP / "run_incomplete")
    lines = (inc / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()[:-1]
    (inc / Q.TELEMETRY_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc, res, _ = select(inc, ev, Q.sha256_file(inc / Q.TELEMETRY_NAME), out=TMP / "sel_inc.json")
    check("d3_incomplete_run_refused_exit_3", rc == 3 and "[run_incomplete]" in res, res)


@case("refusals")
def d3_rules_disagreeing_with_config_refused():
    bad_rules = TMP / "rules_bad.json"
    doc = json.loads(S.RULES_PATH.read_text(encoding="utf-8"))
    doc["qat_epoch"]["epochs"] = 14
    bad_rules.write_text(json.dumps(doc), encoding="utf-8")
    try:
        S.load_rules(bad_rules)
        code = None
    except Q.QATRefused as e:
        code = e.code
    check("d3_rules_disagreeing_with_config_refused", code == "rules_mismatch", str(code))


@case("refusals")
def d3_smoke_run_refused():
    run = clone_run(FX["base"][1.0], TMP / "run_mode_smoke")
    sha = relabel(run, meta={"mode": "smoke"})
    ev = TMP / "ev_mode_smoke"
    fabricate(run, ev, vals(e07=0.5))
    rc, res, _ = select(run, ev, sha)
    check("d3_smoke_run_refused", rc == 2 and "[run_not_real]" in res, res)


@case("refusals")
def d3_non_qat_stage_refused():
    run = clone_run(FX["base"][1.0], TMP / "run_stage_e1")
    sha = relabel(run, meta={"stage": "E1"})
    ev = TMP / "ev_stage_e1"
    fabricate(run, ev, vals(e07=0.5))
    rc, res, _ = select(run, ev, sha)
    check("d3_non_qat_stage_refused", rc == 2 and "[stage]" in res, res)


@case("refusals")
def d3_winner_changed_during_selection_refused():
    r1 = FX["base"][1.0]
    ev = TMP / "ev_winner_changed"
    fabricate(r1, ev, vals(e07=0.5))
    ts7 = (ev / A.CONVERTED_DIR / A.epoch_names(7)["torchscript"]).resolve()
    real_sha, seen = Q.sha256_file, {"n": 0}

    def racing_sha(path, *a, **k):
        if Path(path).resolve() == ts7:
            seen["n"] += 1
            if seen["n"] > 1:
                return "0" * 64
        return real_sha(path, *a, **k)
    Q.sha256_file = racing_sha
    try:
        rc, res, _ = select(r1, ev, FX["sha"][1.0])
    finally:
        Q.sha256_file = real_sha
    check("d3_winner_changed_during_selection_refused", rc == 2 and "[winner_changed]" in res, res)


# ---- d5: the U4 clip
@case("d5")
def d5_candidates_are_1_and_5():
    rules = FX["rules"]
    check("d5_candidates_are_1_and_5", rules["qat_clip"]["candidates"] == [1.0, 5.0] and rules["qat_clip"]["tie"] == 5.0)


def _pair_case(name: str, v1: float, v5: float, expect: float, tie: bool):
    def run():
        base, sha = FX["base"], FX["sha"]
        evs = {}
        for clip, v in ((1.0, v1), (5.0, v5)):
            evs[clip] = TMP / f"ev_{name}_{clip}"
            fabricate(base[clip], evs[clip], {**flat(0.0), 7: v})
            rc0, res0, _ = select(base[clip], evs[clip], sha[clip])
            if rc0 != 0:
                print(f"[d5] {name} clip {clip}: epoch selection failed: {res0}")
        outs = []
        for order in ((1.0, 5.0), (5.0, 1.0)):
            out = TMP / f"clip_{name}_{order[0]}.json"
            rc, res = clip_cli([(base[c], evs[c]) for c in order], out)
            outs.append((rc, res, json.loads(out.read_text()) if rc == 0 else {}))
        want_line = f"RESULT: CLIP SELECTED {expect} (tie {str(tie).lower()})"
        ok = all(rc == 0 and d["winner"]["clip_norm"] == expect and d["tie"] is tie and res == want_line
                 for rc, res, d in outs)
        d = outs[0][2]
        check(name, ok, f"{outs[0][1]}; d = {d.get('difference', {}).get('fraction') if d else None}")
        if name == "d5_boundary_pair_ties_to_5":
            check("d5_loser_retained", d.get("loser", {}).get("clip_norm") == 1.0 and d["loser"].get("retained") is True
                  and d["loser"].get("value") == 0.40099999999999997)
            check("d5_clipped_steps_and_max_norm_recorded",
                  isinstance(d["winner"].get("clipped_steps"), int)
                  and isinstance(d["winner"].get("max_pre_clip_norm"), float))
    return run


for _n, _v1, _v5, _e, _tie in [("d5_boundary_pair_ties_to_5", 0.40099999999999997, 0.4, 5.0, True),
                               ("d5_just_above_not_tie", 0.401, 0.4, 1.0, False),
                               ("d5_exact_not_naive_float", 0.001, 0.0, 1.0, False),
                               ("d5_tie_to_5_either_order", 0.4, 0.40099999999999997, 5.0, True),
                               ("d5_higher_5_wins", 0.30, 0.35, 5.0, False)]:
    register("d5", _n, _pair_case(_n, _v1, _v5, _e, _tie))


def _rejected_case(tag: str, nan_first: bool):
    def run():
        live = (FX["base"][1.0], ev_clip1())
        dead = (nan5_run(), TMP / "ev_none")
        out = TMP / f"clip_o1_{tag}.json"
        rc, res = clip_cli([dead, live] if nan_first else [live, dead], out)
        d = json.loads(out.read_text()) if rc == 0 else {}
        check(f"d5_nonfinite_candidate_rejected_{tag}",
              rc == 0 and d["winner"]["clip_norm"] == 1.0 and d["loser"]["rejected"] is True
              and any("rejected: non-finite state (AM-21 item 3)" in t for t in d["rule_trace"])
              and "rejected: non-finite state (AM-21 item 3)" in res, res)
    return run


register("d5", "d5_nonfinite_candidate_rejected_a", _rejected_case("a", False))
register("d5", "d5_nonfinite_candidate_rejected_b", _rejected_case("b", True))


def _both_rejected_case(tag: str, nan5_first: bool):
    def run():
        nan1b = clone_run(FX["nan10"], TMP / f"run_clip1_nan_{tag}")
        pairs = [(nan1b, TMP / "ev_none1"), (nan5_run(), TMP / "ev_none")]
        out = TMP / f"clip_none_{tag}.json"
        rc, res = clip_cli(pairs[::-1] if nan5_first else pairs, out)
        check(f"d5_both_rejected_no_winner_exit_2_{tag}", rc == 2 and "[no_winner]" in res
              and "no winner (AM-21 item 3): a new amendment decides the clip value" in res and not out.exists(), res)
    return run


register("d5", "d5_both_rejected_no_winner_exit_2_a", _both_rejected_case("a", False))
register("d5", "d5_both_rejected_no_winner_exit_2_b", _both_rejected_case("b", True))


@case("d5")
def d5_missing_run_refused():
    rc, res = clip_cli([(FX["base"][1.0], ev_clip1()), (FX["base"][5.0], TMP / "ev_missing")], TMP / "clip_missing.json")
    check("d5_missing_run_refused", rc == 3 and "[epoch_selection_missing]" in res, res)


@case("d5")
def d5_incomplete_run_refused():
    inc5 = clone_run(FX["base"][5.0], TMP / "run_clip5_incomplete")
    lines = (inc5 / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()[:-1]
    (inc5 / Q.TELEMETRY_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc, res = clip_cli([(FX["base"][1.0], ev_clip1()), (inc5, ev_clip5())], TMP / "clip_inc.json")
    check("d5_incomplete_run_refused", rc == 3 and "[run_incomplete]" in res, res)


def _variant_case(name: str, code: str, needle: str = "", *, meta=None, host=None, fingerprints=None, tamper=None,
                  expect_rc: int = 2):
    """Clip 1.0's selected run against a clip-5.0 variant (its own record and eval directory)."""
    def run():
        rn = clone_run(FX["base"][5.0], TMP / f"run_{name}")
        sh = relabel(rn, meta=meta, host=host, fingerprints=fingerprints)
        evx = selected_eval(rn, sh, f"ev_{name}", vals(e07=0.31), tamper=tamper)
        out = TMP / f"clip_{name}.json"
        rc, res = clip_cli([(FX["base"][1.0], ev_clip1()), (rn, evx)], out)
        if expect_rc == 0:
            d = json.loads(out.read_text()) if rc == 0 else {}
            hd = d.get("host_differences", {})
            check(name, rc == 0 and hd.get("host.hostname", [None, None])[1] == "pod-b"
                  and hd.get("host.pod_id", [None, None])[1] == "b", res)
        else:
            check(name, rc == expect_rc and f"[{code}]" in res and needle in res, res)
    return run


for _n, _c, _needle, _kw in [
        ("d5_recipe_mismatch_refused", "recipe_mismatch", "num_workers", {"meta": {"num_workers": 1}}),
        ("d5_batch_order_mismatch_refused", "batch_order_differs", "",
         {"fingerprints": lambda step, h: ("0" * 64) if step == 13 else h}),
        ("d5_non_pilot_run_refused", "not_a_pilot_run", "", {"meta": {"u4_pilot": False, "clip_source": "clip_selection"}}),
        ("d5_pilot_summary_commits_differ_refused", "pilot_evals_differ", "", {"tamper": {"commit_all": "9" * 40}}),
        ("d5_pilot_host_labels_differ_refused", "pilot_evals_differ", "", {"tamper": {"host_label": "other-host"}}),
        ("d5_cpu_model_difference_refused", "recipe_mismatch", "host", {"host": {"cpu_model": "Other CPU"}}),
        ("d5_hostname_and_pod_id_differences_reported", "", "", {"host": {"hostname": "pod-b", "pod_id": "b"},
                                                                  "expect_rc": 0})]:
    register("d5", _n, _variant_case(_n, _c, _needle, **_kw))


@case("d5")
def d5_one_candidate_refused():
    rc, res = clip_cli([(FX["base"][1.0], ev_clip1())], TMP / "clip_one.json")
    check("d5_one_candidate_refused", rc == 2 and "[candidates]" in res and not (TMP / "clip_one.json").exists(), res)


@case("d5")
def d5_two_clip_1_runs_refused():
    rc, res = clip_cli([(FX["base"][1.0], ev_clip1()), (FX["base"][1.0], ev_clip1())], TMP / "clip_same.json")
    check("d5_two_clip_1_runs_refused", rc == 2 and "[clip_values]" in res, res)


def _edited_selection_case(name: str, edit, code: str):
    def run():
        evx = selected_eval(FX["base"][5.0], FX["sha"][5.0], f"ev_{name}", vals(e07=0.31))
        selp = evx / "qat_selection.json"
        doc = json.loads(selp.read_text())
        edit(doc)
        selp.write_text(json.dumps(doc), encoding="utf-8")
        rc, res = clip_cli([(FX["base"][1.0], ev_clip1()), (FX["base"][5.0], evx)], TMP / f"clip_{name}.json")
        check(name, rc == 2 and f"[{code}]" in res, res)
    return run


register("d5", "d5_edited_epoch_selection_refused",
         _edited_selection_case("d5_edited_epoch_selection_refused",
                                lambda d: d["winner"].update(value=0.99), "selection_differs"))
register("d5", "d5_selection_of_other_rules_refused",
         _edited_selection_case("d5_selection_of_other_rules_refused",
                                lambda d: d.update(rules_sha256="0" * 64), "epoch_selection_format"))


@case("d5")
def d5_null_git_head_refused():
    nulls = {}
    for c in (1.0, 5.0):
        rn = clone_run(FX["base"][c], TMP / f"run_clip{c}_nullhead")
        sh = relabel(rn, meta={"git_head": None})
        nulls[c] = (rn, selected_eval(rn, sh, f"ev_nullhead_{c}", vals(e07=0.31)))
    rc, res = clip_cli([nulls[1.0], nulls[5.0]], TMP / "clip_nullhead.json")
    check("d5_null_git_head_refused", rc == 2 and "[recipe_identity_null]" in res, res)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--only", default=None, help="comma-separated case names (scripts/smoke_qat_mutations.py)")
    ap.add_argument("--cache-dir", default=None, help="keep (or reuse) the trained raw runs here")
    args = ap.parse_args()
    want = set(args.sections.split(","))
    names = [n for n, _, _ in CASES]
    if want - set(SECTIONS) or len(set(names)) != len(names):
        print(f"RESULT: ERROR unknown sections {sorted(want - set(SECTIONS))} or duplicate case names")
        return 4
    if args.only:
        only = args.only.split(",")
        if set(only) - set(names):
            print(f"RESULT: ERROR unknown cases {sorted(set(only) - set(names))}")
            return 4
        todo = [c for c in CASES if c[0] in only]
    else:
        todo = [c for c in CASES if c[1] in want]
    t_all = time.time()
    print("=" * 78)
    print("QAT SELECTION SMOKE (d3 + d5) — synthetic real-mode records, fabricated eval directories")
    print(f"{len(todo)} cases | sections {sorted({c[1] for c in todo})} | temp {TMP}")
    print("=" * 78)
    runs = prepare_runs(Path(args.cache_dir) if args.cache_dir else None)
    base = {1.0: runs["run_clip1.0"], 5.0: runs["run_clip5.0"]}
    fq = {e: 0.1 + 0.01 * e for e in range(1, 16)}
    fq[9] = 0.9                                            # the fake-quant VAL argmax: e09
    FX.update(base=base, sha={c: relabel(base[c], fq=fq) for c in base}, nan10=runs["run_nan10"],
              nan1=runs["run_nan1"])
    FX.update(sha_n10=relabel(FX["nan10"], fq=fq), sha_n1=relabel(FX["nan1"], fq=fq))
    FX["rules"], FX["rules_sha"] = S.load_rules()
    for name, _section, fn in todo:
        before = len(results)
        try:
            fn()
        except Exception as e:                               # noqa: BLE001 -- the case fails, the smoke goes on
            check(name, False, f"the case raised {type(e).__name__}: {e}")
        if len(results) == before:
            check(name, False, "the case reported no check")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:46}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n[time] {time.time() - t_all:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
