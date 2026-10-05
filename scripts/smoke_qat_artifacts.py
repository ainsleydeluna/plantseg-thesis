#!/usr/bin/env python3
"""d4 (lane 6, L-AM4 + L-AM1q): QAT epochs converted to the INT8 artifacts of record, then scored.

A confidently trained synthetic source (scripts/synthetic_qat_fixtures.py) runs the QAT trainer for 15
epochs in mode "smoke", as E5 and as E6. scripts/qat_epoch_eval.py converts every epoch in fresh
processes; one epoch is converted twice more to show the bytes reproduce. The evaluator scores the
converted epochs through a driver that stubs only its git calls, with VAL rows capped. A run made
non-finite at step 10 shows the not-convertible path, and an injected parity failure the STOP path.

Checks the identity and provenance of record, bitwise TorchScript / state_dict / eager parity read from
disk, per-channel INT8 weights, no float region, the fresh-process guard, the evaluator accepting the
provenance, and the fake-quant vs converted agreement on a confident fixture (with two negative controls
that must disagree). No PlantSeg data, no checkpoint of record; temp files outside the repository.

Ends with one RESULT line; exit 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_qat_fixtures import (color_task, confident_student, evaluator_driver,  # noqa: E402
                                            make_qat_run, make_tree, safe_tmpdir)

TMP = safe_tmpdir("smoke_qat_artifacts_")
ROOT = TMP / "data"
make_tree(ROOT, n_train=16, n_val=846, seed=3)              # no TEST split
os.environ["PLANTSEG_DATA_ROOT"] = str(ROOT)                 # before configs/data.py is imported

import torch  # noqa: E402

import scripts.qat_epoch_eval as QEE  # noqa: E402
from src.eval.model_loading import load_int8_torchscript, rebuild_int8_from_state_dict  # noqa: E402
from src.eval.stage_artifacts import validate_int8_artifact  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant.prepare import convert_model  # noqa: E402
from src.quant.ptq import (graph_census, output_parity, runtime_census, synthetic_parity_inputs,  # noqa: E402
                           torchscript_bytes, weight_scheme_report)
from src.quant.qconfig import select_qnnpack_backend  # noqa: E402

results: list[tuple[str, bool, str]] = []
SECTIONS = ("units", "records", "e5", "e6", "nan", "stop", "guard")
DRIVER = evaluator_driver(TMP / "evaluator_driver.py")
os.environ["SMOKE_REPO"] = str(REPO)
HOST = "cloud-smoke"


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def qee(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = QEE.main(argv)
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT:")]
    print(out.rstrip())
    return rc, (lines[-1] if lines else "")


def convert(run: Path, ev: Path, purpose: str = "record", epochs=None) -> tuple[int, str]:
    argv = ["convert", "--run-dir", str(run), "--eval-dir", str(ev), "--purpose", purpose, "--host-label", HOST,
            "--allow-smoke-inputs"]
    if purpose == "record":
        argv += ["--expect-telemetry-sha256", Q.sha256_file(run / Q.TELEMETRY_NAME)]
    if epochs:
        argv += ["--epochs", *map(str, epochs)]
    return qee(argv)


def score(run: Path, ev: Path, purpose: str, cap: str | None) -> tuple[int, str]:
    old = QEE.EVALUATOR_SCRIPT
    QEE.EVALUATOR_SCRIPT = str(DRIVER)
    if cap:
        os.environ["SMOKE_MAX_SAMPLES"] = cap
    try:
        return qee(["score", "--run-dir", str(run), "--eval-dir", str(ev), "--purpose", purpose,
                    "--host-label", HOST, "--allow-smoke-inputs"])
    finally:
        QEE.EVALUATOR_SCRIPT = old
        os.environ.pop("SMOKE_MAX_SAMPLES", None)


def state_of(run: Path, epoch: int) -> dict:
    return torch.load(run / Q.EPOCH_DIR / f"e{epoch:02d}.pt", map_location="cpu", weights_only=True)["model_state_dict"]


def converted_without(state: dict, drop: set[str]):
    """A converted student built from the state with `drop` left at the fresh prepared values."""
    from src.quant.x86_latency import copy_qat_state_by_name
    m = A.prepared_skeleton()
    copy_qat_state_by_name(m, {k: v for k, v in state.items() if k not in drop})
    return convert_model(m)


def test_units() -> None:
    """The state predicate, the by-name copy and its key-set comparison, without training or converting."""
    prepared = A.prepared_skeleton().train()
    with torch.no_grad():
        prepared(torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(1)))   # observers populated
    never = Q.never_observed_modules(prepared)
    kinds = Q.state_kinds(prepared)
    state = Q.cpu_state(prepared)
    pred = lambda st: Q.state_predicate(st, kinds, never)[0]              # noqa: E731
    check("d4_predicate_accepts_an_observed_state", pred(state) and len(never) == 5, f"never observed: {never}")
    bn = next(k for k, v in kinds.items() if v == "bn_buffer" and k.endswith("running_var"))
    obs_max = next(k for k, v in kinds.items() if v == "observer_max")
    obs_min = obs_max[: -len("max_val")] + "min_val"

    def edited(key, fn):
        st = dict(state)
        st[key] = fn(st[key].clone())
        return st
    check("d4_predicate_flags_nonfinite_bn_with_finite_params",
          not pred(edited(bn, lambda t: t.index_fill_(0, torch.tensor([0]), float("inf")))), bn)
    check("d4_predicate_flags_nonfinite_observer",
          not pred(edited(obs_max, lambda t: t.fill_(float("nan")))), obs_max)
    check("d4_predicate_flags_inverted_observer_range",
          not pred({**state, obs_min: state[obs_max].clone() + 1.0}), obs_min)
    never_key = f"{never[0]}.activation_post_process.min_val"
    check("d4_predicate_allows_unobserved_only_where_never_observed",
          float(state[never_key]) == float("inf") and not pred({
              **state, "quant.activation_post_process.activation_post_process.min_val": torch.tensor(float("inf")),
              "quant.activation_post_process.activation_post_process.max_val": torch.tensor(float("-inf"))}),
          never_key)
    report = A.load_state_by_name(A.prepared_skeleton(), state)
    check("d4_copy_exact_with_alias_keys", report == {"entries": 1593, "copied": 1575, "alias_keys": 18,
                                                      "equal_entries": 1593}, str(report))
    for name, st in (("d4_key_set_missing_key_refused", {k: v for k, v in state.items() if k != bn}),
                     ("d4_key_set_extra_key_refused", {**state, "head.extra.weight": torch.zeros(1)})):
        try:
            A.load_state_by_name(A.prepared_skeleton(), st)
            got = None
        except Q.QATStop as e:
            got = e.code
        check(name, got == "state_key_mismatch", str(got))
    nan_t = torch.tensor([float("nan"), 1.0])
    check("d4_bytes_equal_is_bitwise", A.bytes_equal(nan_t, nan_t.clone()) and not torch.equal(nan_t, nan_t.clone())
          and not A.bytes_equal(nan_t, torch.tensor([float("nan"), 2.0])))


def clone_run(src: Path, dst: Path) -> Path:
    """A copy of a run (the checkpoints copied, so one may be rewritten)."""
    shutil.copytree(src, dst)
    return dst


def rewrite_checkpoint(run: Path, epoch: int, edit) -> None:
    """Rewrite eNN.pt with `edit(state)` applied and record its new sha256 in the run's epoch_end row."""
    p = run / Q.EPOCH_DIR / f"e{epoch:02d}.pt"
    payload = torch.load(p, map_location="cpu", weights_only=True)
    edit(payload["model_state_dict"])
    torch.save(payload, p)
    tel = run / Q.TELEMETRY_NAME
    rows = [json.loads(line) for line in tel.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r.get("event") == "epoch_end" and r["epoch"] == epoch:
            r["checkpoint_sha256"] = Q.sha256_file(p)
    tel.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fake_convert_record(run: Path, ev: Path, *, purpose: str = "record", epochs=range(1, 16), **edit) -> None:
    """A qat_convert.json as `convert` writes it (no conversion): score's own checks run on it."""
    rec = A.read_run_record(run)
    ev.mkdir(parents=True)
    doc = {"format": QEE.CONVERT_FORMAT, "run_dir": rec["run_dir"], "run_id": rec["run_id"],
           "stage": rec["run_meta"]["stage"], "telemetry_sha256": rec["telemetry_sha256"], "purpose": purpose,
           "host_label": HOST, "cpu_model": Q.cpu_model(), "git_head": None, "smoke_inputs": True,
           "outcomes": [{"epoch": e, "status": "converted"} for e in epochs], **edit}
    (ev / QEE.CONVERT_RECORD).write_text(json.dumps(doc), encoding="utf-8")


def test_records(cache: Path | None) -> None:
    """convert's and score's own refusals (P24, P26) and the freeze cross-check STOP (P16): nothing converts."""
    run = TMP / "run_records"
    if cache is not None and (cache / "READY").is_file():
        clone_run(cache / "run_records", run)
    else:
        torch.manual_seed(0)
        from src.models.student import build_student
        make_qat_run(run, model=build_student(pretrained=False).eval(), stage="e5", clip=1.0)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            clone_run(run, cache / "run_records")
            (cache / "READY").write_text("raw smoke run\n", encoding="utf-8")
    sha = Q.sha256_file(run / Q.TELEMETRY_NAME)
    base = ["--run-dir", str(run), "--purpose", "record", "--host-label", HOST, "--allow-smoke-inputs"]

    def refused(name: str, argv: list[str], code: str, rc_want: int = 2, absent: Path | None = None) -> None:
        rc, res = qee(argv)
        check(name, rc == rc_want and f"[{code}]" in res and (absent is None or not absent.exists()), res)

    ev = TMP / "ev_records"
    refused("d4_convert_record_requires_telemetry_sha", ["convert", "--eval-dir", str(ev), *base],
            "expect_telemetry_sha256_required", absent=ev)
    refused("d4_convert_record_takes_no_epochs",
            ["convert", "--eval-dir", str(ev), *base, "--expect-telemetry-sha256", sha, "--epochs", "3"],
            "epochs_with_record", absent=ev)
    refused("d4_convert_epochs_out_of_range",
            ["convert", "--eval-dir", str(ev), *[a if a != "record" else "timing" for a in base], "--epochs", "16"],
            "epochs_range", absent=ev)
    refused("d4_convert_telemetry_sha_mismatch",
            ["convert", "--eval-dir", str(ev), *base, "--expect-telemetry-sha256", "f" * 64],
            "telemetry_sha256_mismatch", absent=ev)
    refused("d4_convert_smoke_run_refused",
            ["convert", "--eval-dir", str(ev), *[a for a in base if a != "--allow-smoke-inputs"],
             "--expect-telemetry-sha256", sha], "run_not_real", absent=ev)
    refused("d4_convert_test_path_refused",
            ["convert", "--eval-dir", str(TMP / "latest_ev"), *base, "--expect-telemetry-sha256", sha],
            "eval_dir_test_path", absent=TMP / "latest_ev")
    used = TMP / "ev_used"
    used.mkdir()
    (used / "note.txt").write_text("used", encoding="utf-8")
    refused("d4_convert_eval_dir_not_fresh",
            ["convert", "--eval-dir", str(used), *base, "--expect-telemetry-sha256", sha], "eval_dir_not_fresh")
    inc = clone_run(run, TMP / "run_records_incomplete")
    lines = (inc / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()[:-1]
    (inc / Q.TELEMETRY_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    refused("d4_convert_incomplete_run_exit_3",
            ["convert", "--eval-dir", str(ev), *[a if a != str(run) else str(inc) for a in base],
             "--expect-telemetry-sha256", Q.sha256_file(inc / Q.TELEMETRY_NAME)], "run_incomplete", 3, absent=ev)
    # P16: a BN statistic that moved after the epoch-11 freeze stops the conversion before any epoch converts
    moved = clone_run(run, TMP / "run_records_bn_moved")
    bn_key = next(k for k in state_of(run, 12) if k.endswith("running_mean"))
    rewrite_checkpoint(moved, 12, lambda st: st[bn_key].add_(1.0))
    ev_m = TMP / "ev_bn_moved"
    rc, res = qee(["convert", "--eval-dir", str(ev_m), *[a if a != str(run) else str(moved) for a in base],
                   "--expect-telemetry-sha256", Q.sha256_file(moved / Q.TELEMETRY_NAME)])
    stop = ev_m / QEE.CONVERT_STOP
    check("d4_convert_stops_when_a_freeze_did_not_take",
          rc == 1 and "[freeze_cross_check]" in res and stop.is_file() and not (ev_m / A.CONVERTED_DIR).exists()
          and "bn e12" in json.dumps(json.loads(stop.read_text()).get("freeze_cross_check", {})), res)
    # score refuses on the records alone (no evaluator runs)
    sbase = ["score", *[a for a in base]]

    def score_refused(name: str, ev_dir: Path, code: str, rc_want: int = 2, extra=(), purpose="record") -> None:
        argv = [a if a != "record" else purpose for a in sbase] + ["--eval-dir", str(ev_dir), *extra]
        rc, res = qee(argv)
        check(name, rc == rc_want and f"[{code}]" in res and not (ev_dir / A.SCORES_DIR).exists(), res)

    empty = TMP / "ev_score_empty"
    empty.mkdir()
    score_refused("d4_score_without_convert_record_exit_3", empty, "convert_record_missing", 3)
    other = TMP / "ev_score_other_run"
    fake_convert_record(run, other, telemetry_sha256="0" * 64)
    score_refused("d4_score_convert_record_of_another_run", other, "convert_record_mismatch")
    timing = TMP / "ev_score_timing"
    fake_convert_record(run, timing, purpose="timing", epochs=[15])
    score_refused("d4_score_purpose_mismatch", timing, "purpose_mismatch")
    score_refused("d4_score_unconverted_epoch_refused", timing, "epoch_not_converted", extra=("--epochs", "14"),
                  purpose="timing")
    part = TMP / "ev_score_partial"
    fake_convert_record(run, part, epochs=range(1, 15))
    score_refused("d4_score_record_needs_all_15_epochs", part, "convert_record_epochs")
    score_refused("d4_score_record_takes_no_epochs", part, "epochs_with_record", extra=("--epochs", "3"))
    stopped = TMP / "ev_score_stopped"
    fake_convert_record(run, stopped)
    (stopped / A.CONVERTED_DIR).mkdir()
    (stopped / A.CONVERTED_DIR / A.epoch_names(5)["stop"]).write_text("{}", encoding="utf-8")
    score_refused("d4_score_refuses_a_stop_file", stopped, "stop_present")
    busy = TMP / "ev_score_busy"
    fake_convert_record(run, busy)
    (busy / A.SCORES_DIR / "e01").mkdir(parents=True)
    rc, res = qee([a for a in sbase] + ["--eval-dir", str(busy)])
    check("d4_score_scores_dir_not_fresh", rc == 2 and "[scores_not_fresh]" in res, res)
    done = TMP / "ev_score_done"
    fake_convert_record(run, done)
    (done / QEE.EVAL_RECORD).write_text("{}", encoding="utf-8")
    score_refused("d4_score_runs_once", done, "output_exists")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--cache-dir", default=None, help="keep (or reuse) the records section's trained run here")
    args = ap.parse_args()
    want = set(args.sections.split(","))
    ran: set[str] = set()
    if want - set(SECTIONS):
        print(f"RESULT: ERROR unknown sections {sorted(want - set(SECTIONS))}")
        return 4
    t_all = time.time()
    print("=" * 78)
    print("QAT ARTIFACTS SMOKE (d4) — confident synthetic source; convert in fresh processes; score capped")
    print(f"torch {torch.__version__} | sections {sorted(want)} | temp {TMP}")
    print("=" * 78)
    if "units" in want:
        ran.add("units")
        test_units()
    if "records" in want:
        ran.add("records")
        test_records(Path(args.cache_dir) if args.cache_dir else None)
    src = r5 = r6 = None
    if want & {"e5", "e6", "nan", "stop", "guard"}:
        t0 = time.time()
        src = confident_student(seed=0)
        print(f"[source] confident FP32 student in {time.time() - t0:.0f}s")
    for stage, need in (("e5", {"e5", "stop", "guard"}), ("e6", {"e6"})):
        if want & need:
            t0 = time.time()
            run = TMP / f"run_{stage}"
            make_qat_run(run, model=src, stage=stage)
            print(f"[train] {stage}: 15 epochs in {time.time() - t0:.0f}s")
            r5, r6 = (run, r6) if stage == "e5" else (r5, run)
    names = A.epoch_names(15)

    if "e5" in want:
        ran.add("e5")
        rec5 = A.read_run_record(r5)
        sha15 = rec5["ends"][15]["checkpoint_sha256"]
        # ---------------- convert E5 (record), then e15 twice more in fresh processes (bytes reproduce)
        ev5 = TMP / "eval_e5"
        t0 = time.time()
        rc, res = convert(r5, ev5)
        conv = json.loads((ev5 / QEE.CONVERT_RECORD).read_text()) if (ev5 / QEE.CONVERT_RECORD).is_file() else {}
        check("convert_e5_all_15_epochs", rc == 0 and res == "RESULT: CONVERTED 15/15 (record)"
              and [o["status"] for o in conv.get("outcomes", [])] == ["converted"] * 15
              and conv.get("freeze_cross_check", {}).get("ok") is True, f"{res} ({time.time() - t0:.0f}s)")
        names = A.epoch_names(15)
        cdir = ev5 / A.CONVERTED_DIR
        shas = {}
        for label in ("b", "c"):
            ev = TMP / f"eval_e5_e15_{label}"
            rc_t, res_t = convert(r5, ev, purpose="timing", epochs=[15])
            shas[label] = tuple(Q.sha256_file(ev / A.CONVERTED_DIR / names[k])
                                if (ev / A.CONVERTED_DIR / names[k]).is_file()
                                else None for k in ("torchscript", "state_dict"))
        rec_shas = tuple(Q.sha256_file(cdir / names[k]) if (cdir / names[k]).is_file() else None
                         for k in ("torchscript", "state_dict"))
        check("d4_reconversion_bytes_reproduce", None not in rec_shas and shas["b"] == shas["c"] == rec_shas,
              f"TorchScript {rec_shas[0] and rec_shas[0][:12]}, state_dict {rec_shas[1] and rec_shas[1][:12]}; "
              "three fresh processes")

        # ---------------- identity and provenance of record (e15)
        ident = A.read_ts_identity(cdir / names["torchscript"]) if (cdir / names["torchscript"]).is_file() else {}
        check("d4_identity_keys_exact",
              tuple(sorted(ident)) == tuple(sorted(A.IDENTITY_KEYS)) and ident.get("quantization") == "qat"
              and ident.get("engine") == "qnnpack" and ident.get("artifact_role") == "accuracy"
              and ident.get("schema") == "plantseg-int8-torchscript/1.0.0" and ident.get("num_classes") == 116
              and ident.get("stage") == "E5" and ident.get("qat_checkpoint_sha256") == sha15
              and ident.get("epoch") == 15
              and ident.get("run_id") == rec5["run_id"] and ident.get("state_dict_companion") == names["state_dict"],
              str(sorted(ident)))
        prov_p = cdir / names["run_meta"]
        prov = json.loads(prov_p.read_text()) if prov_p.is_file() else {}
        need = {"stage": "E5", "source_stage": "E1", "quantization_method": "qat", "random_init": False,
                "num_classes": 116, "backend": "qnnpack", "artifact_format": "torchscript", "epoch": 15,
                "run_id": rec5["run_id"], "qat_checkpoint_sha256": sha15, "purpose": "record", "host_label": HOST,
                "converted_artifact": names["torchscript"]}
        wrong = {k: prov.get(k) for k, v in need.items() if prov.get(k) != v}
        check("d4_provenance_fields", not wrong and prov.get("converted_artifact_sha256") == rec_shas[0]
              and (prov.get("state_dict_artifact") or {}).get("sha256") == rec_shas[1]
              and all(k in prov for k in ("verification", "code", "environment", "cpu_model", "git_head")),
              f"wrong {wrong}")
        try:
            resolved = validate_int8_artifact("E5", prov_p)
            ok = resolved["artifact_sha256"] == rec_shas[0]
        except Exception as e:                                   # noqa: BLE001
            ok, resolved = False, {"error": str(e)}
        check("evaluator_accepts_qat_provenance", ok, str(resolved.get("error", "")))

        # ---------------- round trip from disk, the eager model, weights and census (e15)
        select_qnnpack_backend()
        s15 = state_of(r5, 15)
        ts, _ = load_int8_torchscript(cdir / names["torchscript"], stage="E5", method="qat", engine="qnnpack")
        sd = rebuild_int8_from_state_dict(cdir / names["state_dict"], stage="E5", method="qat")
        eager = convert_model(A.fake_quant_model(s15))
        par_ts_sd = output_parity({"torchscript": ts, "state_dict": sd}, synthetic_parity_inputs(),
                                  [("torchscript", "state_dict")])
        par_eager = output_parity({"eager": eager, "torchscript": ts}, synthetic_parity_inputs(),
                                  [("eager", "torchscript")])
        check("d4_torchscript_equals_state_dict_bitwise", par_ts_sd["all_equal"], json.dumps(par_ts_sd["pairs"])[:160])
        check("d4_eager_equals_torchscript", par_eager["all_equal"], json.dumps(par_eager["pairs"])[:160])
        w_ts, w_e = weight_scheme_report(ts), weight_scheme_report(eager)
        check("d4_weights_per_channel_int8", w_ts["ok"] and w_e["ok"],
              f"{w_ts['per_channel_int8']}/{w_ts['quantized_convs']} in the TorchScript file")
        g, c = graph_census(ts), runtime_census(eager, torch.zeros(1, 3, 512, 512))
        check("d4_no_float_region", g["ok"] and c["ok"], f"graph ok {g['ok']}, runtime census ok {c['ok']}")

        # ---------------- fake-quant vs converted agreement on the confident fixture (e15)
        fq = A.fake_quant_model(s15)
        lsb = A.head_lsb(fq)
        x, y = color_task(8, 128, seed=99, band=8)               # valid: 8 px or more from a cell border
        inputs = [x[i:i + 1] for i in range(8)]
        valid = [y[i:i + 1] != 255 for i in range(8)]
        agr = A.argmax_agreement(fq, ts, inputs, lsb, valid_masks=valid)
        skel = A.prepared_skeleton()
        bn_keys, _obs_keys = A.freeze_key_sets(skel)
        fq_prefixes = tuple(f"{n}." for n, _ in Q.fake_quant_modules(skel))
        fq_keys = {k for k in s15 if k.startswith(fq_prefixes)}     # every fake-quant buffer, flags included
        ctl_obs = A.argmax_agreement(fq, converted_without(s15, fq_keys), inputs, lsb, valid_masks=valid)
        ctl_bn = A.argmax_agreement(fq, converted_without(s15, set(bn_keys)), inputs, lsb, valid_masks=valid)
        print(f"[agreement] LSB {lsb:.6g}: overall {agr['agreement']}, confident fraction {agr['confident_fraction']}, "
              f"confident agreement {agr['confident_agreement']}, max |diff| {agr['max_logit_diff_lsb']:.2f} LSB, "
              f"p99.9 {agr['p999_logit_diff_lsb']:.2f} LSB")
        print(f"[agreement] by margin band: {json.dumps(agr['by_margin_band'])}")
        print(f"[agreement] controls: observers not carried {ctl_obs['agreement']}, "
              f"BN not carried {ctl_bn['agreement']}")
        check("d4_confident_pixels_agree", agr["confident_pixels"] > 0 and agr["confident_agreement"] == 1.0,
              f"margin >= 8 LSB: {agr['confident_pixels']} pixels, agreement {agr['confident_agreement']}")
        check("d4_fixture_confident", (agr["confident_fraction"] or 0) >= 0.90,
              f"{agr['confident_fraction']} of valid pixels have margin >= 8 LSB"
              + ("" if (agr["confident_fraction"] or 0) >= 0.90 else " -- fixture not confident"))
        check("d4_argmax_agreement_ge_99pct", (agr["agreement"] or 0) >= 0.99, f"{agr['agreement']}")
        check("d4_control_observers_not_carried_below_50pct", (ctl_obs["agreement"] or 1) < 0.5,
              f"{ctl_obs['agreement']}")
        check("d4_control_bn_not_carried_below_50pct", (ctl_bn["agreement"] or 1) < 0.5, f"{ctl_bn['agreement']}")

        # ---------------- score E5 (record, every epoch; VAL rows capped by the driver)
        t0 = time.time()
        rc, res = score(r5, ev5, "record", cap="2")
        ev_doc = json.loads((ev5 / QEE.EVAL_RECORD).read_text()) if (ev5 / QEE.EVAL_RECORD).is_file() else {}
        rows = ev_doc.get("epochs", [])
        s_ok = len(rows) == 15 and all(r["status"] == "scored" for r in rows)
        summ = {}
        if s_ok:
            summ = json.loads((ev5 / rows[-1]["summary"]["path"]).read_text())
        from src.eval.artifacts import verify_artifact
        try:
            verified = all(verify_artifact(ev5 / r["score_dir"]) is not None for r in rows) if s_ok else False
        except Exception as e:                                   # noqa: BLE001
            verified = False
            res += f" verify_artifact: {e}"
        run_ids_ok = s_ok and all(r["run_id"] == f"{rec5['run_id']}_e{r['epoch']:02d}_{r['checkpoint_sha256'][:12]}"
                                  for r in rows)
        check("evaluate_model_scores_e5_val",
              rc == 0 and res == "RESULT: SCORED 15/15 (record)" and s_ok and verified and run_ids_ok
              and summ.get("dataset", {}).get("split") == "val"
              and summ["run"]["precision"] == "int8_qat" and summ["run"]["quant_backend"] == "qnnpack"
              and summ["run"]["checkpoint_sha256"] == rec_shas[0] and summ["run"]["run_id"] == rows[-1]["run_id"],
              f"{res} ({time.time() - t0:.0f}s)")
        cmd = rows[-1]["evaluator_command"] if s_ok else []
        of_record = ["--stage", "E5", "--model-role", "student", "--precision", "int8_qat", "--split", "val",
                     "--protocol", "canvas", "--artifact-status", "provisional", "--device", "cpu", "--batch-size", "1"]
        check("score_runs_the_evaluator_flags_of_record",
              all(cmd[cmd.index(a) + 1] == b for a, b in zip(of_record[0::2], of_record[1::2]) if a in cmd)
              and all(a in cmd for a in of_record[0::2]) and "--threads" not in cmd
              and all(isinstance(r.get("torch_num_threads"), int) and isinstance(r.get("affinity_cpus"), int)
                      for r in rows)
              and ev_doc.get("threads_note", "").endswith("QNNPACK's thread pool keeps its default"),
              " ".join(cmd[1:]) if cmd else "")
        ev_t = TMP / "eval_e5_e15_b"
        rc_t, res_t = score(r5, ev_t, "timing", cap=None)
        trows = json.loads((ev_t / QEE.EVAL_RECORD).read_text())["epochs"] if (ev_t / QEE.EVAL_RECORD).is_file() else []
        tsum = json.loads((ev_t / trows[0]["summary"]["path"]).read_text()) if trows else {}
        check("timing_purpose_is_smoke_status_64_rows",
              rc_t == 0 and res_t == "RESULT: SCORED 1/1 (timing, 64 samples)" and len(trows) == 1
              and tsum.get("dataset", {}).get("actual_rows") == 64
              and tsum.get("run", {}).get("artifact_status") == "smoke"
              and "--max-samples" in trows[0]["evaluator_command"], res_t)

    if "e6" in want:
        ran.add("e6")
        # ---------------- E6 end to end: provenance projection-free
        ev6 = TMP / "eval_e6"
        rc6, res6 = convert(r6, ev6)
        p6 = ev6 / A.CONVERTED_DIR / names["run_meta"]
        prov6 = json.loads(p6.read_text()) if p6.is_file() else {}
        try:
            ok6 = validate_int8_artifact("E6", p6)["provenance"]["cwd_projection_loaded"] is False
        except Exception:                                         # noqa: BLE001
            ok6 = False
        check("d4_e6_converts_projection_free", rc6 == 0 and prov6.get("source_stage") == "E3"
              and prov6.get("cwd_projection_loaded") is False and ok6, res6)
        rc6s, res6s = score(r6, ev6, "record", cap="1")
        check("d4_e6_scores", rc6s == 0, res6s)

    if "nan" in want:
        ran.add("nan")
        # ---------------- a non-finite run: epochs from the bad one on are recorded, never converted
        rn = TMP / "run_e5_nan"
        make_qat_run(rn, model=src, stage="e5", nan_at=(10,))
        evn = TMP / "eval_e5_nan"
        rcn, resn = convert(rn, evn)
        convn = json.loads((evn / QEE.CONVERT_RECORD).read_text()) if (evn / QEE.CONVERT_RECORD).is_file() else {}
        statuses = [o["status"] for o in convn.get("outcomes", [])]
        nc = evn / A.CONVERTED_DIR / A.epoch_names(5)["not_convertible"]
        ncd = json.loads(nc.read_text()) if nc.is_file() else {}
        check("d4_not_convertible_recorded",
              rcn == 0 and resn == ("RESULT: CONVERTED 4/15 (record; not convertible: "
                                    + " ".join(f"e{e:02d}" for e in range(5, 16)) + ")")
              and statuses == ["converted"] * 4 + ["not convertible"] * 11
              and ncd.get("status") == "not convertible" and ncd.get("rule") == A.NOT_CONVERTIBLE_RULE
              and ncd.get("qat_checkpoint_sha256") == A.read_run_record(rn)["ends"][5]["checkpoint_sha256"]
              and ncd.get("failing"), f"{resn}; {statuses.count('not convertible')} not convertible")
        rcns, resns = score(rn, evn, "record", cap="1")
        nrows = json.loads((evn / QEE.EVAL_RECORD).read_text())["epochs"] if (evn / QEE.EVAL_RECORD).is_file() else []
        check("d4_not_convertible_excluded_from_scoring",
              rcns == 0 and [r["status"] for r in nrows] == ["scored"] * 4 + ["excluded"] * 11
              and all(r.get("rule") == A.NOT_CONVERTIBLE_RULE for r in nrows[4:]), resns)

    if "stop" in want:
        ran.add("stop")
        # ---------------- an injected parity failure: a STOP file, no provenance, the eval dir spent
        stop_driver = TMP / "stop_driver.py"
        stop_driver.write_text(
            "import sys\nsys.dont_write_bytecode = True\nsys.path.insert(0, %r)\n"
            "import src.quant.ptq as ptq\nreal = ptq.output_parity\n"
            "def failing(models, inputs, pairs):\n    r = real(models, inputs, pairs)\n"
            "    r['all_equal'] = False\n    return r\n"
            "ptq.output_parity = failing\n"
            "from scripts.qat_epoch_eval import main\nraise SystemExit(main(sys.argv[1:]))\n" % str(REPO),
            encoding="utf-8")
        old = QEE.SCRIPT
        QEE.SCRIPT = stop_driver
        try:
            evs = TMP / "eval_e5_stop"
            rcs, ress = convert(r5, evs, purpose="timing", epochs=[15])
        finally:
            QEE.SCRIPT = old
        sname = A.epoch_names(15)
        stop = evs / A.CONVERTED_DIR / sname["stop"]
        check("d4_stop_writes_stop_file_and_no_provenance",
              rcs == 1 and stop.is_file() and not (evs / A.CONVERTED_DIR / sname["run_meta"]).exists()
              and json.loads(stop.read_text()).get("code") == "parity_mismatch", ress)
        rcs2, ress2 = score(r5, evs, "timing", cap=None)
        check("d4_score_refuses_spent_eval_dir", rcs2 == 2 and "[stop_present]" in ress2, ress2)
        used = TMP / "eval_used"
        used.mkdir()
        (used / "note.txt").write_text("used", encoding="utf-8")
        rc_again, res_again = convert(r5, used)
        check("d4_eval_dir_never_reused", rc_again == 2 and "[eval_dir_not_fresh]" in res_again, res_again)

    if "guard" in want:
        ran.add("guard")
        # ---------------- the fresh-process guard (this process traces now; nothing in-process converts after)
        select_qnnpack_backend()
        torchscript_bytes(convert_model(A.prepared_skeleton().eval()), {"schema": "smoke"})
        try:
            A.convert_epoch(r5, TMP / "eval_guard", 15, purpose="timing", host_label=HOST, allow_smoke_inputs=True)
            guard = "not refused"
        except Q.QATRefused as e:
            guard = e.code
        check("d4_fresh_process_guard", guard == "fresh_process_required" and not (TMP / "eval_guard").exists(), guard)

    check("all_requested_sections_ran", ran == want, f"ran {sorted(ran)} of {sorted(want)}")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n[time] {time.time() - t_all:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
