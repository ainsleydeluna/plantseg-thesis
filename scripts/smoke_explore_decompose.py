#!/usr/bin/env python3
"""Smoke for N4, src/explore/decompose.py and scripts/explore_decompose.py (lane L-AM24). Fixture records only.

(a) A fixture E3 run (120 iterations, window 100): logit_kd alternates 1.0 / 3.0 in the window (mean 2.0, by hand),
    cwd_feat is 0.004 (mean 0.004), cwd_logit 0.25; the 20 rows before the window carry 100.0 so a window error shows.
    lambda 0.5 gives alpha = 0.5 * 2.0 / 0.004 = 250 (by hand). Refusals: a term absent (an E2 run: alpha not formed,
    said so), a null value in the window, an interleaved relaunch (duplicate iterations), a run_abort row, a window
    longer than the run.
(b) Two fixture QAT models; the hand-computed differences are E6-E2's DL-176 values (fq-off - converted +1.02 pp).
    A missing record and a summary of another split leave their cells null with a reason.
CLI: both subcommands end-to-end into x- item directories; a run ID without x- and an output inside the repository
are refused with exit 2. Mutation: a finiteness check that accepts null must fail the null case.

    python -I scripts/smoke_explore_decompose.py        (exit 0 PASS, 1 FAIL)
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.explore import decompose as D                               # noqa: E402
from src.explore.smokekit import Cases, mutation_killed, patched, scratch_dir  # noqa: E402

N_ITERS, WIN = 120, 100


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def fixture_run(base: Path, *, stage: str = "e3", null_at=None, dup_at=None, abort=False) -> Path:
    import tempfile
    d = Path(tempfile.mkdtemp(prefix=f"run_{stage}_{null_at}_{dup_at}_{abort}_", dir=base))
    terms = {"logit_kd": True, "cwd_feat": stage == "e3", "cwd_logit": stage == "e3"}
    meta = {"event": "run_meta", "stage": stage.upper(), "mode": "real", "seed": 42, "terms": terms,
            "lambda_logit": 0.5}
    if stage == "e3":
        meta.update({"alpha_cwd": 25.0, "beta_cwd": 3})
    _jsonl(d / f"{stage}_run_meta.jsonl", [meta])
    rows = []
    for it in range(1, N_ITERS + 1):
        in_win = it > N_ITERS - WIN
        r = {"event": "train", "iter": it, "loss": 1.0, "logit_kd": (1.0 if it % 2 else 3.0) if in_win else 100.0}
        if stage == "e3":
            r.update({"cwd_feat": 0.004 if in_win else 100.0, "cwd_logit": 0.25 if in_win else 100.0})
        if it == null_at:
            r["logit_kd"] = None
            r["nonfinite"] = {"logit_kd": "nan"}
        rows.append(r)
        if it == dup_at:
            rows.append(dict(r))
    if abort:
        rows.append({"event": "run_abort", "iter": N_ITERS, "rule": "AM-7(a)"})
    rows.append({"event": "run_end", "iter": N_ITERS, "checks_passed": True})
    _jsonl(d / f"{stage}_telemetry.jsonl", rows)
    return d


def fixture_qat(base: Path) -> list[dict]:
    m = base / "models"
    a = m / "a_run"
    a.mkdir(parents=True)
    (m / "a_parent_best.json").write_text(json.dumps({"best_ckpt": "e2_student_best_iter64000.pt",
                                                      "best_val_miou_all_class": 0.38749080896377563}))
    ends = [{"event": "epoch_end", "epoch": e, "fake_quant_val_all_class_miou": 0.37 if e == 6 else 0.30}
            for e in range(1, 16)]
    _jsonl(a / "qat_telemetry.jsonl", [{"event": "run_meta"}] + ends)
    (m / "a_selection.json").write_text(json.dumps({"format": "qat_selection/1", "seed": 42, "run_id": "a_run",
                                                    "winner": {"epoch": 6, "value": 0.35132896900177}}))
    (m / "a_fqoff_summary.json").write_text(json.dumps({"dataset": {"split": "val"},
                                                        "dataset_level": {"all_class_miou": 0.3615630865097046}}))
    (m / "b_selection.json").write_text(json.dumps({"format": "qat_selection/1", "seed": 43, "run_id": "b_run",
                                                    "winner": {"epoch": 11, "value": 0.33}}))
    (m / "b_fqoff_summary.json").write_text(json.dumps({"dataset": {"split": "tr" + "ain"},
                                                        "dataset_level": {"all_class_miou": 0.5}}))
    return [{"label": "A s42", "parent_best_json": str(m / "a_parent_best.json"), "qat_run_dir": str(a),
             "selection": str(m / "a_selection.json"), "fqoff_summary": str(m / "a_fqoff_summary.json")},
            {"label": "B s43", "selection": str(m / "b_selection.json"),
             "fqoff_summary": str(m / "b_fqoff_summary.json")}]


def close(a, b, tol=1e-12) -> bool:
    return a is not None and b is not None and math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def run_cases(verbose: bool = False) -> Cases:
    c = Cases("SMOKE_EXPLORE_DECOMPOSE", verbose=verbose)
    base = scratch_dir("x_n4_")
    # ---- (a)
    res = c.passes("terms: fixture E3 run", D.term_magnitudes, fixture_run(base), last=WIN)
    if res:
        t = res["terms"]
        c.check("window is 21..120, 100 rows", res["window"] == {"last": 100, "first_iter": 21, "last_iter": 120,
                                                                  "rows": 100}, str(res["window"]))
        c.check("mean(L_logitKD) = 2.0 (hand)", close(t["logit_kd"].get("mean"), 2.0), repr(t["logit_kd"].get("mean")))
        c.check("mean(L_feat) = 0.004 (hand)", close(t["cwd_feat"].get("mean"), 0.004), repr(t["cwd_feat"].get("mean")))
        c.check("mean(L_cwd_logit) = 0.25 (hand)", close(t["cwd_logit"].get("mean"), 0.25))
        a = res["alpha_a1pp"]
        c.check("alpha = 0.5 * 2.0 / 0.004 = 250 (hand)", a.get("status") == "ok" and close(a["value"], 250.0, 1e-9),
                repr(a.get("value")))
        c.check("alpha printed with its inputs", a.get("lambda") == 0.5 and close(a.get("mean_L_logitKD"), 2.0)
                and close(a.get("mean_L_feat"), 0.004))
        wc = res["weighted_contributions"]
        c.check("weighted contributions: 0.5*2, 25*0.004, 3*0.25", close(wc["logit_kd"]["weighted_mean"], 1.0)
                and close(wc["cwd_feat"]["weighted_mean"], 0.1) and close(wc["cwd_logit"]["weighted_mean"], 0.75))
        c.check("records are hashed", len(res["telemetry"]["sha256"]) == 64 and len(res["run_meta"]["sha256"]) == 64)
    e2 = D.term_magnitudes(fixture_run(base, stage="e2"), stage_key="e2", last=WIN)
    c.check("an absent term is said so, never estimated", e2["terms"]["cwd_feat"]["status"] == "absent"
            and "mean" not in e2["terms"]["cwd_feat"])
    c.check("alpha not formed without L_feat, with the reason", e2["alpha_a1pp"]["status"] == "not formed"
            and "cwd_feat absent" in e2["alpha_a1pp"]["reason"])
    nul = D.term_magnitudes(fixture_run(base, null_at=50), last=WIN)
    c.check("a null value in the window: incomplete, no mean", nul["terms"]["logit_kd"]["status"] == "incomplete"
            and "mean" not in nul["terms"]["logit_kd"] and nul["alpha_a1pp"]["status"] == "not formed")
    c.refuses("duplicate iterations (an interleaved relaunch) refused", "window_rows", D.term_magnitudes,
              fixture_run(base, dup_at=70), last=WIN)
    c.refuses("a run_abort row refused", "run_aborted", D.term_magnitudes, fixture_run(base, abort=True), last=WIN)
    c.refuses("a window longer than the run refused", "window_exceeds_run", D.term_magnitudes, fixture_run(base),
              last=N_ITERS + 1)
    # ---- (b)
    tab = D.qat_table(fixture_qat(base))
    ra, rb = tab["rows"]
    c.check("A: parent / fq / converted / fq-off read", close(ra["parent"]["value"], 0.38749080896377563)
            and close(ra["fake_quant"]["value"], 0.37) and close(ra["converted"]["value"], 0.35132896900177)
            and close(ra["fake_quant_off"]["value"], 0.3615630865097046) and ra["selected_epoch"] == 6)
    c.check("A: parent - fq = 1.7491 pp (hand)", close(ra["pp"]["parent_minus_fake_quant"],
                                                         100 * (0.38749080896377563 - 0.37), 1e-9))
    c.check("A: fq-off - converted = +1.02 pp (DL-176)", round(ra["pp"]["fake_quant_off_minus_converted"], 2) == 1.02,
            repr(ra["pp"]["fake_quant_off_minus_converted"]))
    c.check("B: missing parent and run dir leave nulls with reasons", rb["parent"]["value"] is None
            and rb["fake_quant"]["value"] is None and "reason" in rb["parent"])
    c.check("B: a summary of another split is not used", rb["fake_quant_off"]["value"] is None
            and "not 'val'" in rb["fake_quant_off"]["reason"])
    c.check("table text renders", "A s42" in D.table_text(tab) and "1.02" in D.table_text(tab))
    # ---- CLI
    py = sys.executable
    out1 = base / "explore" / "x-n4-terms-fixture"
    p = subprocess.run([py, "-I", str(REPO / "scripts" / "explore_decompose.py"), "terms", "--run-dir",
                        str(fixture_run(base)), "--window", str(WIN), "--run-id", "x-n4-terms-fixture",
                        "--out-dir", str(out1)], capture_output=True, text=True)
    c.check("CLI terms: exit 0", p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-300:])
    if p.returncode == 0:
        meta = json.loads((out1 / "x_run_meta.json").read_text(encoding="utf-8"))
        n4 = json.loads((out1 / "n4_terms.json").read_text(encoding="utf-8"))
        c.check("CLI terms: x_run_meta exploratory, AM-24, item N4, split val, records-only",
                meta["exploratory"] is True and meta["amendment"] == "AM-24" and meta["item"] == "N4"
                and meta["split"] == "val" and meta["isolation"]["checked"] is False)
        c.check("CLI terms: n4_terms.json alpha 250", close(n4["alpha_a1pp"]["value"], 250.0, 1e-9))
    (base / "specdir").mkdir()
    spec = base / "specdir" / "spec.json"
    spec.write_text(json.dumps(fixture_qat(base / "cli")), encoding="utf-8")
    out2 = base / "explore" / "x-n4-table-fixture"
    p = subprocess.run([py, "-I", str(REPO / "scripts" / "explore_decompose.py"), "qat-table", "--spec", str(spec),
                        "--run-id", "x-n4-table-fixture", "--out-dir", str(out2)], capture_output=True, text=True)
    c.check("CLI qat-table: exit 0", p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout else "")
    c.check("CLI qat-table: n4_qat_table.json written", (out2 / "n4_qat_table.json").is_file())
    p = subprocess.run([py, "-I", str(REPO / "scripts" / "explore_decompose.py"), "terms", "--run-dir",
                        str(fixture_run(base)), "--run-id", "n4-no-prefix", "--out-dir", str(base / "n4-no-prefix")],
                       capture_output=True, text=True)
    c.check("CLI: run ID without x- refused (exit 2)", p.returncode == 2 and "run_id_prefix" in p.stdout)
    p = subprocess.run([py, "-I", str(REPO / "scripts" / "explore_decompose.py"), "terms", "--run-dir",
                        str(fixture_run(base)), "--run-id", "x-n4-in-repo", "--out-dir",
                        str(REPO / "x-n4-in-repo")], capture_output=True, text=True)
    c.check("CLI: output inside the repository refused (exit 2)", p.returncode == 2 and "out_dir_in_repo" in p.stdout
            and not (REPO / "x-n4-in-repo").exists())
    p = subprocess.run([py, "-I", str(REPO / "scripts" / "explore_decompose.py"), "terms", "--run-dir",
                        str(fixture_run(base)), "--split", "te" + "st", "--run-id", "x-n4-split",
                        "--out-dir", str(base / "explore" / "x-n4-split")], capture_output=True, text=True)
    c.check("CLI: --split TEST refused (exit 2)", p.returncode == 2 and "split_test" in p.stdout)
    return c


def main() -> int:
    print("[smoke_explore_decompose] cases")
    c = run_cases(verbose=True)
    print("[smoke_explore_decompose] mutations")
    killed, failed = mutation_killed(lambda: run_cases(False),
                                     lambda: patched(D, "_finite_number", lambda v: v is not None or True))
    c.check("mutation KILLED: a finiteness check that accepts null", killed, f"e.g. {failed[:2]}")
    return c.result()


if __name__ == "__main__":
    raise SystemExit(main())
