#!/usr/bin/env python3
"""Dress-rehearsal mode on synthetic VAL inputs (lane L-STATS-OFFICIAL; acceptance (e); AM-17 item 9).

SYNTHETIC ONLY: three synthetic E1 VAL evaluation artifacts ("seeds" 42/43/44: 846 rows, k = 4 AM-5
images, status provisional, float32 per-image values, bare stems, the canvas protocol) are written
through the unmodified evaluator writer into a temporary clean fixture repository outside the
working tree. No PlantSeg data, checkpoint, model or GPU; the out-root is a temporary directory
outside the repository.

  R  `--mode rehearsal` through scripts/run_stats.py: exit 0, outputs only under <out-root>/<UTC>/,
     the five rehearsal files and nothing named like a section 12 artifact; the manifest's P25 fields;
     `not_rehearsed`, the placeholders at p = 1 and outside every criterion and label; the counts
     846 - k and 846; every criterion passed; SHA256SUMS; the three raw p-values and the 3-member Holm
  I  INVESTIGATE fires (printed, exit 0) when a rehearsed test rejects
  N  P21: the 15-cell invariant is exact, and a one-float32-ulp perturbation of one cell breaks it
  H  the hand-rolled strict Holm, including the equality boundary
  X  refusals (exit 2, nothing written): an out-root inside the repository, a value that is not
     float32-representable, a row with clean_image_id != image_id, a TEST artifact, the same seed
     twice, --expect-k-val mismatch, a non-canvas input; and exit 1 (outputs kept) on a failed
     criterion
  P  P22: the rehearsal module imports neither build_family nor the statistics writer

Run:  python -B scripts/smoke_stats_rehearsal.py
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                       # noqa: E402

import run_stats                                                         # noqa: E402
import stats_fixtures as F                                               # noqa: E402
from src.eval.evaluate import Condition                                  # noqa: E402
from src.stats import rehearsal as R                                     # noqa: E402
from src.stats.corruption_protocol import official_corruption_grid       # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
N, K, B = 846, 4, 200
PY = sys.executable


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def seeds(root: Path, *, k=K, quality=(0.70, 0.70, 0.70), split="val", tag="r") -> dict:
    """Three synthetic E1 VAL artifacts sharing one ground truth; equal mean quality by default, so
    the seed differences are noise."""
    repo = F.clean_git_fixture(root)
    truth = F.make_truth(N, k, F.int_seed(tag, "truth"))
    out = {}
    for label, q in zip(R.SEEDS, quality):
        scored = F.score(truth, F.predict(truth, quality=q, seed=F.int_seed(tag, label)),
                         Condition())
        out[label] = F.write_fixture(
            repo / f"runs/{split}/e1_{label}", scored, N, stage="E1", condition=Condition(),
            split=split, status="provisional", repo_root=repo, run_id=f"{tag}-e1-{label}",
            checkpoint_sha256=hashlib.sha256(f"{tag} {label}".encode()).hexdigest())
    return out


def cli(*args):
    p = subprocess.run([PY, "-B", str(REPO / "scripts" / "run_stats.py"), *map(str, args)],
                       cwd=str(REPO), capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return p.returncode, p.stdout, p.stderr


def rehearse(s, out_root, *extra):
    return cli("--mode", "rehearsal", "--s42", s["s42"], "--s43", s["s43"], "--s44", s["s44"],
               "--out-root", out_root, "--B", B, *extra)


def only_run_dir(out_root: Path):
    dirs = [d for d in out_root.iterdir()] if out_root.exists() else []
    return dirs[0] if len(dirs) == 1 and dirs[0].is_dir() else None


def refused(name, res, needle, out_root: Path):
    code, so, se = res
    nothing = not out_root.exists() or not any(out_root.iterdir())
    check(f"{name} -> exit 2 naming it; nothing written", code == 2 and needle in se and nothing,
          f"exit {code}: {se.strip()[-300:]}")


def edit_row(d: Path, index: int, fn) -> None:
    p = Path(d) / "per_image.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    fn(rows[index])
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                 encoding="utf-8", newline="\n")
    F.rehash(d)


# --------------------------------------------------------------------------------------------------
def part_r(work: Path) -> None:
    s = seeds(work / "fx")
    out_root = work / "reh_out"
    code, so, se = rehearse(s, out_root, "--jobs", 2)
    check("R1 rehearsal on three synthetic VAL seeds: exit 0", code == 0, se[-400:])
    run = only_run_dir(out_root)
    names = sorted(p.name for p in run.iterdir()) if run else []
    check("R2 outputs only under <out-root>/<UTC>/: the four files and SHA256SUMS, nothing named "
          "family.json, bootstrap.npz or MANIFEST.sha256",
          run is not None and names == sorted([*R.FILES, R.SUMS])
          and len(run.name) == 16 and run.name.endswith("Z"), names)
    if run is None:
        return
    sums_ok = all(hashlib.sha256((run / n).read_bytes()).hexdigest() == dgst for dgst, n in (
        ln.split("  ", 1) for ln in (run / R.SUMS).read_text(encoding="utf-8").splitlines()))
    check("R3 SHA256SUMS covers the four files and verifies", sums_ok)
    m = json.loads((run / "rehearsal_manifest.json").read_text(encoding="utf-8"))
    r = json.loads((run / "rehearsal_results.json").read_text(encoding="utf-8"))
    check("R4 manifest: mode rehearsal, artifact_status nonofficial, not registered as a result",
          m["mode"] == "rehearsal" and m["artifact_status"] == "nonofficial"
          and m["registered_as_result"] is False and r["artifact_status"] == "nonofficial")
    p25 = ("inputs", "code_provenance", "software_environment", "cpu_features", "B", "K_val",
           "mapping", "statistical_contract_sha256", "src_stats_path")
    check("R5 manifest carries P25: input identities and file hashes, code provenance, stack, CPU "
          "features, B, K_val, the mapping, the contract sha256, the resolved src.stats path",
          all(k in m for k in p25) and m["B"] == B and m["K_val"] == K
          and Path(m["src_stats_path"]) == (REPO / "src" / "stats").resolve()
          and all(len(m["inputs"][x]["artifact_sha256s"]) == 4 for x in R.SEEDS)
          and m["mapping"]["noninferiority_e3_e6"] == ["s44", "s42"], sorted(m))
    holm = {x["comparison_id"]: x for x in r["holm_8"]["members"]}
    roles = {c["comparison_id"]: c["role"] for c in r["comparisons"]}
    check("R6 not_rehearsed: the five ids; each a placeholder with p = 1 and no label",
          r["not_rehearsed"] == list(R.NOT_REHEARSED)
          and all(roles[c] == "placeholder" and holm[c]["p_raw"] == 1.0
                  and c not in r["labels"] for c in R.NOT_REHEARSED)
          and sorted(r["labels"]) == sorted(R.REHEARSED))
    crit_names = " ".join(c["name"] for c in r["criteria"])
    check("R7 placeholders appear in no criterion", not any(c in crit_names
                                                            for c in R.NOT_REHEARSED[:4]))
    check("R8 every criterion passed", r["criteria_passed"]
          and all(c["passed"] for c in r["criteria"]),
          [c for c in r["criteria"] if not c["passed"]][:2])
    pi = {t["jackknife_count"] for t in r["tasks"] if not t["task_id"].endswith("dataset_miou_delta")}
    ds = {t["jackknife_count"] for t in r["tasks"] if t["task_id"].endswith("dataset_miou_delta")}
    check(f"R9 19 tasks bootstrapped; per-image counts 846 - k = {N - K}, dataset-level 846; the "
          "16 placeholder tasks not run", len(r["tasks"]) == 19 and pi == {N - K} and ds == {N}
          and len(r["tasks_not_run"]) == 16, (pi, ds))
    report = (run / "report.md").read_text(encoding="utf-8")
    check("R10 the report prints the three raw p-values and their 3-member Holm (P24), and a "
          "banner on every table", "3-member Holm" in report
          and all(x["comparison_id"] in report for x in r["holm_3"]) and len(r["holm_3"]) == 3
          and report.count("REHEARSAL -- NONOFFICIAL") >= 6)
    acc, st = r["comparisons"][CANONICAL.index("accuracy_e1_e6")], r["structural_robustness"]
    check("R11 the structural robustness comparison equals accuracy_e1_e6 except comparison_id and "
          "metric", {k for k in acc if k not in ("role",) and acc[k] != st.get(k)}
          == {"comparison_id", "metric"})
    check("R12 investigate not fired on seed noise here (printed flag, never a failure)",
          r["investigate"]["fired"] is False and "not fired" in report)


def part_i(work: Path) -> None:
    s = seeds(work / "fx_eff", quality=(0.60, 0.80, 0.81), tag="eff")
    out_root = work / "reh_eff"
    code, so, se = rehearse(s, out_root)
    run = only_run_dir(out_root)
    r = json.loads((run / "rehearsal_results.json").read_text(encoding="utf-8")) if run else {}
    check("I1 a real effect: INVESTIGATE fires on a rejection, is printed, and the exit stays 0",
          code == 0 and r.get("investigate", {}).get("fired") is True
          and "INVESTIGATE" in so and "FIRED" in (run / "report.md").read_text(encoding="utf-8"),
          (code, se[-300:]))


def part_n(work: Path) -> None:
    s = seeds(work / "fx_n", tag="n")
    art = R.load_seeds(s)["s42"]
    grid = official_corruption_grid()
    mc = R.structural_miou_c(art.run, "E1", grid)
    check("N1 P21: mIoU-C of 15 identical cells equals the clean values exactly (==)",
          R.invariant_mismatches(mc, art.run) == [])
    cells = R.structural_cells(art.run, "E1", grid)
    recs = list(cells[7].records)
    v = recs[11].disease_only_miou
    toward = np.float32(0.0) if v > 0.5 else np.float32(1.0)
    recs[11] = dataclasses.replace(recs[11], disease_only_miou=float(
        np.nextafter(np.float32(v), toward)))
    cells[7] = dataclasses.replace(cells[7], records=tuple(recs))
    bad = R.invariant_mismatches(R.structural_miou_c(art.run, "E1", grid, cells=cells), art.run)
    check("N2 P21 negative: one cell moved by one float32 ulp breaks the invariant for exactly that "
          "image", bad == [recs[11].clean_image_id], bad)


def part_h() -> None:
    check("H1 hand-rolled strict Holm: rejects while p < alpha / (m - rank), stops at the first "
          "non-rejection", R.hand_holm([0.001, 0.02, 0.04]) == [True, True, True]
          and R.hand_holm([0.001, 0.02, 0.06]) == [True, True, False]
          and R.hand_holm([0.004, 0.2, 0.01]) == [True, False, True]
          and R.hand_holm([0.06, 0.001, 0.01]) == [False, True, True])
    eq = R.holm_members([("a", 0.025), ("b", 0.5)])
    check("H2 equality never rejects: p = alpha / 2 at rank 1 (built as a literal)",
          R.hand_holm([0.025, 0.5]) == [False, False] and [m["reject"] for m in eq] == [False, False])
    mm = R.holm_members([("x", 0.001), ("y", 0.02), ("z", 0.04)])
    check("H3 the 3-member family from the frozen holm_audit agrees with the hand-rolled one",
          [m["reject"] for m in mm] == R.hand_holm([0.001, 0.02, 0.04]))


def part_x(work: Path) -> None:
    s = seeds(work / "fx_x", tag="x")
    refused("X1 an out-root inside the repository (Q10)", rehearse(s, REPO / "reports" / "x_reh"),
            "inside the repository", REPO / "reports" / "x_reh")

    def variant(tag, label, fn):
        root = work / f"x_{tag}"
        shutil.copytree(s[label].parents[2], root)
        d = root / s[label].relative_to(s[label].parents[2])
        fn(d)
        return dict(s, **{label: d})
    v = variant("f32", "s43", lambda d: edit_row(d, 5, lambda o: o.__setitem__(
        "disease_only_miou", 0.1) if o["disease_only_miou"] is not None else o.__setitem__(
        "all_class_miou", 0.1)))
    refused("X2 a per-image value 0.1 that is not float32-representable (P20)",
            rehearse(v, work / "o_x2"), "float32-representable", work / "o_x2")
    v = variant("stem", "s44", lambda d: edit_row(d, 3, lambda o: o.__setitem__(
        "clean_image_id", o["image_id"] + "_c")))
    refused("X3 a row with clean_image_id != image_id (P20)", rehearse(v, work / "o_x3"),
            "bare stems", work / "o_x3")
    t = seeds(work / "fx_test", split="test", tag="t")
    refused("X4 TEST artifacts (REHEARSAL reads VAL only)", rehearse(t, work / "o_x4"),
            "VAL", work / "o_x4")
    refused("X5 the same seed twice", rehearse(dict(s, s43=s["s42"]), work / "o_x5"),
            "three runs", work / "o_x5")
    refused("X6 --expect-k-val 0 while K_val = 4", rehearse(s, work / "o_x6", "--expect-k-val", 0),
            "K_val is 4", work / "o_x6")
    v = variant("canvas", "s42", lambda d: F.edit_summary(d, lambda j: dict(
        j, protocol={"name": "upstream"})))
    refused("X7 a non-canvas input (EVALUATION_CONTRACT 11(a))", rehearse(v, work / "o_x7"),
            "canvas", work / "o_x7")
    refused("X8 --repo-root is not accepted in rehearsal mode",
            rehearse(s, work / "o_x8", "--repo-root", work), "--repo-root", work / "o_x8")
    real = R.criteria

    def failing(reh, runs):
        return real(reh, runs) + [("injected failing criterion (smoke)", False, "test")]
    R.criteria = failing
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_stats.main(["--mode", "rehearsal", "--s42", str(s["s42"]), "--s43",
                                   str(s["s43"]), "--s44", str(s["s44"]), "--out-root",
                                   str(work / "o_x9"), "--B", "50"])
    finally:
        R.criteria = real
    run = only_run_dir(work / "o_x9")
    check("X9 a failed structural criterion -> exit 1, outputs kept", code == 1 and run is not None
          and not json.loads((run / "rehearsal_results.json").read_text(encoding="utf-8"))[
              "criteria_passed"], (code, err.getvalue()[-300:]))


def part_p() -> None:
    tree = ast.parse((REPO / "src" / "stats" / "rehearsal.py").read_text(encoding="utf-8"))
    imported = {a.asname or a.name for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {"build_family", "write_statistics_artifact"}
    check("P1 P22: the rehearsal module imports neither build_family nor the statistics writer",
          not forbidden & (imported | attrs | set(vars(R))), sorted(forbidden & imported))
    check("P2 P22: its files are not named family.json, bootstrap.npz or MANIFEST.sha256",
          not {"family.json", "bootstrap.npz", "MANIFEST.sha256"} & set(R.FILES + (R.SUMS,)))
    check("P3 namespace plantseg_primary_analysis_v1 (Q4)",
          R.NAMESPACE == "plantseg_primary_analysis_v1")


CANONICAL = ("accuracy_e1_e2", "accuracy_e2_e3", "accuracy_e4_e5", "accuracy_e7_e6",
             "accuracy_e4_e7", "accuracy_e5_e6", "accuracy_e1_e6", "robustness_e1_e6")


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="smoke_stats_rehearsal_"))
    try:
        part_r(work)
        part_i(work)
        part_n(work)
        part_h()
        part_x(work)
        part_p()
    except Exception:                                                    # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=3))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check("Z1 temporary fixtures and outputs removed", not work.exists()
          and not (REPO / "reports" / "x_reh").exists())

    print()
    for name, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if det and not ok:
            print(f"         {det[:400]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'STATS REHEARSAL OK' if allok else 'STATS REHEARSAL FAILED'} "
          f"({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic VAL inputs; rehearsal outputs are nonofficial by construction.")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
