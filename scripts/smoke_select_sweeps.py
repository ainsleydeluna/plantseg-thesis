#!/usr/bin/env python3
"""Smoke for the sweep selection scripts (lane L-AM16-ALPHA d2; the lambda rule of AM-2 / DL-06).

Synthetic inputs only: rule units on candidate records, and end-to-end runs of scripts/select_alpha.py
and scripts/select_lambda.py on synthetic run directories (best.json, run_meta, telemetry and a tiny
checkpoint) under a temp dir. No dataset, no real checkpoint, no GPU; nothing is written in the repo.

Checks (docs/lane_specs/part2.md lane 2 (d) d2, and the cross-lane selection contract):
  alpha  clear winner; tie containing 50 -> 50; tie without 50 -> the smallest alpha; band from s
         above and below the 0.5 pp floor; the band edge is inclusive (decimal); boundary 25/100
         flagged; missing band -> refuse; two candidates -> refuse (lane 2 STOP, exit 3).
  lambda clear winner; tie -> the smallest lambda; boundary 0.25/4 flagged; fewer than five finished
         candidates -> refusal naming AM-17 item 9 (exit 3), never a selection.
  inputs a run with no run_end record (it died during its final validation) or a torn last
         telemetry line is a shortfall (exit 3); refused (exit 2, never a traceback): a torn middle
         telemetry line, a run_end that disagrees with best.json, a run whose own checks failed, a
         dry run, another seed, --allow-offgrid, a Logit-KD semantics override, a second run_meta
         row, a checkpoint whose best value disagrees with best.json, a missing or unreadable
         checkpoint, a path under a 'test' directory, a band file that is not an object, a missing
         lambda selection file, alpha runs whose lambda is not the lambda selection's winner, an
         existing selection file, a rule file that disagrees with configs/distill.py.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import scripts.select_alpha as sa  # noqa: E402
import scripts.select_lambda as sl  # noqa: E402
from configs.distill import LOGIT_KD_SEMANTICS  # noqa: E402
from src.training.sweep_select import (RULES_PATH, SelectionRefused, apply_rule,  # noqa: E402
                                       dl27_band, load_rules, sha256_file)

RULES = load_rules()
ALPHA, LAMBDA = RULES["alpha_cwd"], RULES["lambda_logit"]
E1_VALS = {"42": 0.36314016580581665, "43": 0.3598, "44": 0.3662}
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def cands(pairs) -> list[dict]:
    return [{"value": float(v), "run_id": f"run_{v:g}", "best_val": b, "ckpt_sha256": f"sha{v:g}"}
            for v, b in pairs]


def refused(fn, *args, **kwargs) -> str | None:
    try:
        fn(*args, **kwargs)
    except SelectionRefused as e:
        return e.code
    return None


# ------------------------------------------------------------------------------ rule units
def test_alpha_rule() -> None:
    r = apply_rule(cands([(25, 0.40), (50, 0.43), (100, 0.41)]), ALPHA, 0.005, "alpha")
    check("alpha_clear_winner", r["winner"]["value"] == 50 and not r["tie"] and not r["boundary"], str(r["tied"]))
    r = apply_rule(cands([(25, 0.430), (50, 0.428), (100, 0.431)]), ALPHA, 0.005, "alpha")
    check("alpha_tie_with_50_goes_to_50", r["winner"]["value"] == 50 and r["tie"]
          and r["tied"] == [25.0, 50.0, 100.0], str(r["tied"]))
    r = apply_rule(cands([(25, 0.430), (50, 0.420), (100, 0.432)]), ALPHA, 0.005, "alpha")
    check("alpha_tie_without_50_goes_to_smallest", r["winner"]["value"] == 25 and r["tie"]
          and r["tied"] == [25.0, 100.0] and r["boundary"], str(r["tied"]))
    r = apply_rule(cands([(25, 0.40), (50, 0.41), (100, 0.43)]), ALPHA, 0.005, "alpha")
    check("alpha_boundary_100_flagged", r["winner"]["value"] == 100 and r["boundary"] and not r["tie"])
    r = apply_rule(cands([(25, 0.44), (50, 0.43), (100, 0.41)]), ALPHA, 0.005, "alpha")
    check("alpha_boundary_25_flagged", r["winner"]["value"] == 25 and r["boundary"] and not r["tie"])
    r = apply_rule(cands([(25, 0.40), (50, 0.395), (100, 0.39)]), ALPHA, 0.005, "alpha")
    check("alpha_band_edge_inclusive_in_decimal", r["winner"]["value"] == 50 and r["tied"] == [25.0, 50.0]
          and (0.40 - 0.395) > 0.005, "0.40 - 0.395 is 0.0050000000000000044 in binary floats")
    # band from s below and above the 0.5 pp floor, and its effect on the same three values
    low = dict(E1_VALS, **{})
    s_low = __import__("statistics").stdev(low.values())
    s_doc = {"e1_best_val": low, "s": s_low}
    s, band, _ = dl27_band(s_doc, 0.005)
    check("band_floor_when_sqrt2_s_below", math.sqrt(2) * s < 0.005 and band == 0.005, f"s={s!r}")
    wide = {"42": 0.36314016580581665, "43": 0.3431, "44": 0.3802}
    s_w = __import__("statistics").stdev(wide.values())
    s2, band2, _ = dl27_band({"e1_best_val": wide, "s": s_w, "band": math.sqrt(2) * s_w}, 0.005)
    check("band_sqrt2_s_when_above", band2 == math.sqrt(2) * s2 and band2 > 0.005, f"band={band2!r}")
    vals = [(25, 0.430), (50, 0.421), (100, 0.431)]
    narrow = apply_rule(cands(vals), ALPHA, band, "alpha")["winner"]["value"]
    widened = apply_rule(cands(vals), ALPHA, band2, "alpha")["winner"]["value"]
    check("band_width_changes_the_tie", narrow == 25 and widened == 50, f"{narrow} vs {widened}")
    check("band_s_mismatch_refused", refused(dl27_band, {"e1_best_val": low, "s": s_low + 1e-6}, 0.005)
          == "band_s_mismatch")
    check("band_value_mismatch_refused", refused(dl27_band, {"e1_best_val": low, "s": s_low,
                                                             "band": 0.006}, 0.005) == "band_mismatch")
    check("band_missing_seed_refused", refused(dl27_band, {"e1_best_val": {"42": 0.36, "43": 0.36},
                                                           "s": 0.0}, 0.005) == "band_format")
    check("alpha_two_candidates_refused",
          refused(apply_rule, cands([(25, 0.4), (50, 0.4)]), ALPHA, 0.005, "alpha") == "partial_input")
    check("alpha_offgrid_candidate_refused",
          refused(apply_rule, cands([(25, 0.4), (30, 0.4), (100, 0.4)]), ALPHA, 0.005, "alpha")
          == "partial_input")


def test_lambda_rule() -> None:
    grid = [0.25, 0.5, 1, 2, 4]
    r = apply_rule(cands(zip(grid, [0.40, 0.41, 0.43, 0.42, 0.40])), LAMBDA, 0.005, "lambda")
    check("lambda_clear_winner", r["winner"]["value"] == 1 and not r["tie"] and not r["boundary"])
    r = apply_rule(cands(zip(grid, [0.40, 0.428, 0.43, 0.427, 0.41])), LAMBDA, 0.005, "lambda")
    check("lambda_tie_goes_to_smallest", r["winner"]["value"] == 0.5 and r["tied"] == [0.5, 1.0, 2.0])
    r = apply_rule(cands(zip(grid, [0.44, 0.41, 0.42, 0.40, 0.40])), LAMBDA, 0.005, "lambda")
    check("lambda_boundary_0.25_flagged", r["winner"]["value"] == 0.25 and r["boundary"])
    r = apply_rule(cands(zip(grid, [0.40, 0.41, 0.42, 0.40, 0.43])), LAMBDA, 0.005, "lambda")
    check("lambda_boundary_4_flagged", r["winner"]["value"] == 4 and r["boundary"])
    check("lambda_four_candidates_refused",
          refused(apply_rule, cands(zip(grid[:4], [0.4] * 4)), LAMBDA, 0.005, "lambda") == "partial_input")


# ------------------------------------------------------------------- synthetic run dirs
def make_run(root: Path, stage: str, value: float, best_val: float, *, name: str | None = None,
             finished=True, mode="real", seed=42, offgrid=False, rows=1, ckpt_best=None,
             with_ckpt=True, lam=1.0, override=False, run_end=True, checks_passed=True,
             end_best=None, torn="", corrupt_ckpt=False) -> Path:
    sk = stage.lower()
    d = root / (name or (f"{sk}_s42_alpha{value:g}" if stage == "E3" else f"{sk}_s42_lambda{value:g}"))
    d.mkdir(parents=True)
    terms = {"logit_kd": True, "cwd_feat": stage == "E3", "cwd_logit": stage == "E3"}
    meta = {"event": "run_meta", "stage": stage, "mode": mode, "seed": seed, "terms": terms,
            "projection_params": 51200 if stage == "E3" else 0, "lambda_logit": lam if stage == "E3" else value,
            "logit_kd_semantics": LOGIT_KD_SEMANTICS, "logit_kd_semantics_declared": None,
            "logit_kd_semantics_override_used": override, "max_iters": 80000}
    if stage == "E3":
        meta.update({"alpha_cwd": value, "alpha_offgrid": offgrid, "beta_cwd": 3})
    (d / f"{sk}_run_meta.jsonl").write_text("".join(json.dumps(meta) + "\n" for _ in range(rows)),
                                            encoding="utf-8")
    last = 80000 if finished else 79999
    ck_name = f"{sk}_student_best_iter76000.pt"
    lines = [json.dumps({"event": "train", "iter": last - 1, "loss": 1.0}),
             json.dumps({"event": "train", "iter": last, "loss": 1.0})]
    if torn == "middle":
        lines.insert(1, '{"event": "train", "iter": 7')
    if finished and run_end:
        lines.append(json.dumps({"event": "run_end", "iter": 80000,
                                 "best_val_miou_all_class": best_val if end_best is None else end_best,
                                 "best_ckpt": ck_name, "checks_passed": checks_passed}))
    text = "\n".join(lines) + "\n"
    if torn == "last":
        text += '{"event": "train", "it'
    (d / f"{sk}_telemetry.jsonl").write_text(text, encoding="utf-8")
    if with_ckpt and corrupt_ckpt:
        (d / ck_name).write_bytes(b"\x00not a checkpoint")
    elif with_ckpt:
        payload = {"stage": stage, "iter": 76000,
                   "best_val_miou_all_class": best_val if ckpt_best is None else ckpt_best,
                   ("alpha_cwd" if stage == "E3" else "lambda_logit"): value}
        torch.save(payload, d / ck_name)
    (d / "best.json").write_text(json.dumps({"best_ckpt": f"/workspace/{d.name}/{ck_name}",
                                             "best_val_miou_all_class": best_val}, indent=2),
                                 encoding="utf-8")
    return d


def band_file(root: Path) -> Path:
    s = __import__("statistics").stdev(E1_VALS.values())
    p = root / "dl27_band.json"
    p.write_text(json.dumps({"e1_best_val": E1_VALS, "s": s}), encoding="utf-8")
    return p


def lambda_selection_file(root: Path, lam: float = 1.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / "lambda_selection.json"
    p.write_text(json.dumps({"format": "lambda_selection/1",
                             "winner": {"lambda": lam, "run_id": "e2_s42_lambda1"}}), encoding="utf-8")
    return p


def run_cli(module, argv) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = module.main(argv)
    return rc, out.getvalue() + err.getvalue()


def test_alpha_cli(tmp: Path) -> None:
    root = tmp / "alpha_ok"
    runs = [make_run(root, "E3", a, b) for a, b in ((25, 0.430), (50, 0.428), (100, 0.431))]
    band, lam, out = band_file(root), lambda_selection_file(root), tmp / "alpha_selection.json"
    common = ["--band", str(band), "--lambda-selection", str(lam)]
    rc, log = run_cli(sa, ["--runs", *map(str, runs), *common, "--out", str(out)])
    doc = json.loads(out.read_text()) if out.exists() else {}
    want_keys = {"candidates", "s", "band", "rule_trace", "winner", "tie", "boundary"}
    check("alpha_cli_selects", rc == 0 and want_keys <= set(doc) and doc["winner"]["alpha"] == 50
          and doc["tie"] and doc["band"] == 0.005 and doc["lambda_logit"] == 1.0
          and doc.get("lambda_selection_file", {}).get("lambda") == 1.0, log[-200:])
    check("alpha_cli_candidates_record_existing_checkpoints",
          [c["alpha"] for c in doc.get("candidates", [])] == [25.0, 50.0, 100.0]
          and all(c["ckpt_sha256"] == sha256_file(r / "e3_student_best_iter76000.pt")
                  for c, r in zip(doc.get("candidates", []), runs))
          and doc.get("winner", {}).get("run_id") == "e3_s42_alpha50")
    rc, log = run_cli(sa, ["--runs", *map(str, runs), *common, "--out", str(out)])
    check("alpha_cli_never_overwrites", rc == 2 and "output_exists" in log)
    rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(root / "absent.json"),
                           "--lambda-selection", str(lam), "--out", str(tmp / "a_nob.json")])
    check("alpha_cli_missing_band_refused", rc == 2 and "band_missing" in log)
    listed = root / "band_list.json"
    listed.write_text("[0.005]", encoding="utf-8")
    rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(listed), "--lambda-selection",
                           str(lam), "--out", str(tmp / "a_list.json")])
    check("alpha_cli_band_not_an_object_refused", rc == 2 and "band_format" in log)
    rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(band), "--lambda-selection",
                           str(root / "absent_lambda.json"), "--out", str(tmp / "a_nol.json")])
    check("alpha_cli_missing_lambda_selection_refused", rc == 2 and "lambda_selection_missing" in log)
    rc, log = run_cli(sa, ["--runs", *map(str, runs), "--band", str(band), "--lambda-selection",
                           str(lambda_selection_file(tmp / "lam_half", 0.5)), "--out",
                           str(tmp / "a_lam.json")])
    check("alpha_cli_lambda_other_than_selection_refused", rc == 2 and "lambda_mismatch" in log)
    rc, log = run_cli(sa, ["--runs", *map(str, runs[:2]), *common, "--out", str(tmp / "a_two.json")])
    check("alpha_cli_two_runs_refused_lane2_stop", rc == 3 and "Lane 2 STOP" in log
          and not (tmp / "a_two.json").exists())
    cases = {"unfinished": (dict(finished=False), 3),
             "no_run_end_died_in_final_val": (dict(run_end=False), 3),
             "torn_last_telemetry_line": (dict(run_end=False, torn="last"), 3),
             "torn_middle_telemetry_line": (dict(torn="middle"), 2),
             "run_end_best_mismatch": (dict(end_best=0.2), 2),
             "run_checks_failed": (dict(checks_passed=False), 2),
             "dry_run": (dict(mode="dry"), 2), "seed_43": (dict(seed=43), 2),
             "offgrid": (dict(offgrid=True), 2), "two_run_meta_rows": (dict(rows=2), 2),
             "semantics_override": (dict(override=True), 2),
             "checkpoint_value_mismatch": (dict(ckpt_best=0.1), 2),
             "checkpoint_missing": (dict(with_ckpt=False), 2),
             "checkpoint_unreadable": (dict(corrupt_ckpt=True), 2),
             "mixed_lambda": (dict(lam=0.5), 2)}
    for label, (kw, code) in cases.items():
        r2 = tmp / f"alpha_{label}"
        bad = [make_run(r2, "E3", 25, 0.43), make_run(r2, "E3", 50, 0.42),
               make_run(r2, "E3", 100, 0.41, **kw)]
        o = tmp / f"a_{label}.json"
        rc, log = run_cli(sa, ["--runs", *map(str, bad), *common, "--out", str(o)])
        check(f"alpha_cli_refuses_{label}", rc == code and not o.exists() and "Traceback" not in log,
              f"rc={rc} {log.strip()[-160:]}")
    test_dir = tmp / "x" / "test"
    trap = [make_run(test_dir, "E3", a, 0.4) for a in (25, 50, 100)]
    rc, log = run_cli(sa, ["--runs", *map(str, trap), *common, "--out", str(tmp / "a_t.json")])
    check("alpha_cli_refuses_test_path", rc == 2 and "test_path" in log)


def test_lambda_cli(tmp: Path) -> None:
    root = tmp / "lambda_ok"
    grid = [0.25, 0.5, 1, 2, 4]
    runs = [make_run(root, "E2", v, b) for v, b in zip(grid, [0.40, 0.428, 0.43, 0.427, 0.41])]
    out = tmp / "lambda_selection.json"
    rc, log = run_cli(sl, ["--runs", *map(str, runs), "--out", str(out)])
    doc = json.loads(out.read_text()) if out.exists() else {}
    check("lambda_cli_selects", rc == 0 and doc.get("winner", {}).get("lambda") == 0.5
          and doc.get("tie") and doc.get("band") == 0.005 and "rule_trace" in doc, log[-200:])
    rc, log = run_cli(sl, ["--runs", *map(str, runs[:4]), "--out", str(tmp / "l_four.json")])
    check("lambda_cli_four_runs_refused_naming_am17_item9", rc == 3 and "AM-17 item 9" in log
          and not (tmp / "l_four.json").exists(), log.strip()[-160:])
    r2 = tmp / "lambda_unfinished"
    runs2 = [make_run(r2, "E2", v, 0.4, finished=(v != 4)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs2), "--out", str(tmp / "l_unf.json")])
    check("lambda_cli_unfinished_refused_naming_am17_item9", rc == 3 and "AM-17 item 9" in log
          and "run_unfinished" in log and not (tmp / "l_unf.json").exists())
    r3 = tmp / "lambda_override"
    runs3 = [make_run(r3, "E2", v, 0.4, override=(v == 1)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs3), "--out", str(tmp / "l_ovr.json")])
    check("lambda_cli_semantics_override_refused", rc == 2 and "semantics" in log)
    r4 = tmp / "lambda_died_in_final_val"
    runs4 = [make_run(r4, "E2", v, 0.4, run_end=(v != 2)) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs4), "--out", str(tmp / "l_end.json")])
    check("lambda_cli_no_run_end_refused_naming_am17_item9", rc == 3 and "AM-17 item 9" in log
          and not (tmp / "l_end.json").exists(), log.strip()[-160:])
    r5 = tmp / "lambda_torn_last"
    runs5 = [make_run(r5, "E2", v, 0.4, run_end=(v != 1), torn=("last" if v == 1 else "")) for v in grid]
    rc, log = run_cli(sl, ["--runs", *map(str, runs5), "--out", str(tmp / "l_torn.json")])
    check("lambda_cli_torn_last_line_is_shortfall_naming_am17_item9", rc == 3 and "AM-17 item 9" in log
          and "Traceback" not in log, log.strip()[-160:])


def test_rules_file(tmp: Path) -> None:
    check("rules_file_agrees_with_config", RULES["format"] == "sweep_rules/1"
          and ALPHA["grid"] == [25, 50, 100] and LAMBDA["grid"] == [0.25, 0.5, 1, 2, 4])
    doc = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    doc["alpha_cwd"]["grid"] = [25, 50, 75]
    bad = tmp / "rules_bad.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    check("rules_file_disagreeing_with_config_refused", refused(load_rules, bad) == "rules_mismatch")


def main() -> int:
    print("=" * 78)
    print("SWEEP SELECTION SMOKE — synthetic inputs only; nothing written in the repository")
    print("=" * 78)
    tmp = Path(tempfile.mkdtemp(prefix="k1_select_"))
    for fn in (test_alpha_rule, test_lambda_rule):
        fn()
    for fn in (test_alpha_cli, test_lambda_cli, test_rules_file):
        fn(tmp)
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:56}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
