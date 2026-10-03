#!/usr/bin/env python3
"""Full-size synthetic official run and the driver's doors (lane L-STATS-OFFICIAL; acceptance (a), (d)).

SYNTHETIC ONLY: 37 synthetic evaluation artifacts (n = 1561, k = 7 AM-5 images, status official,
split test, a non-PlantSeg dataset name) are written through the unmodified evaluator writer into a
temporary clean fixture repository outside the working tree (scripts/stats_fixtures.py). No PlantSeg
data, checkpoint, model or GPU is used, and nothing is written into this repository.

  D  doors and refusals through scripts/run_stats.py (exit 2, nothing written): the confirmation flag,
     the binding rule and the out-dir pre-check fire before any read (a non-existent --inputs path),
     the pinned-stack door (Q6; also simulated in process on any stack), --repo-root without
     --synthetic-inputs, both halves of the canvas guard, the smoke-mode refusals (P15); exit 4 for
     an import failure, a worker failure and an artifact that fails its post-rename verification
     (named)
  A  acceptance (a): `--mode official --synthetic-inputs` at B = 10,000 and --jobs min(4, CPUs): exit
     0; all 13 checks re-established; every per-image task 1561 - k and every dataset-level task 1561;
     the only warning synthetic_input_data; each comparison's mean_delta, the descriptive delta and
     the NI observed_delta equal the fixture arrays exactly; the unmet official conditions are exactly
     the synthetic declaration and the binding (O3: this needs a committed clean tree); then
     `--mode report` on that artifact: k = 7 reported, the profile the verified status, and the P28
     dataset-level mIoU-C contrast equal to the fixture's nested mean exactly

On a running stack other than the section 10.1 pin, official mode is refused up front (Q6): the A part
is not run and the smoke checks that refusal instead.

Run:  python -B scripts/smoke_stats_official_fullsize.py        (about 20-30 minutes, 4 CPUs)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                       # noqa: E402

import run_stats                                                         # noqa: E402
import stats_fixtures as F                                               # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import bootstrap as BS                                    # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats.corruption_protocol import official_corruption_grid       # noqa: E402
from src.stats.noninferiority import pooled_miou_from_totals             # noqa: E402
from src.stats.robustness import INFERENTIAL_SEVERITIES                  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
N, K = 1561, 7
PINNED_STACK = A.observed_environment() == A.PINNED_ENVIRONMENT
JOBS = max(1, min(4, os.cpu_count() or 1))
PY = sys.executable


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def cli(*args, timeout=None):
    """scripts/run_stats.py in a fresh interpreter: (exit code, stdout, stderr)."""
    p = subprocess.run([PY, "-B", str(REPO / "scripts" / "run_stats.py"), *map(str, args)],
                       cwd=str(REPO), capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return p.returncode, p.stdout, p.stderr


def in_process(args):
    """run_stats.main in this process: (exit code, stdout + stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_stats.main([str(a) for a in args])
    return code, out.getvalue() + err.getvalue()


def refused(name, res, needle, out: Path | None = None):
    code, so, se = res
    ok = code == 2 and needle in se and (out is None or not out.exists())
    check(f"{name} -> exit 2 naming it; nothing written", ok, f"exit {code}: {se.strip()[-300:]}")


# --------------------------------------------------------------------------------------------------
def doors(work: Path, fs, lst: Path) -> None:
    out = work / "doors"
    missing = work / "no_such_inputs.json"
    base = ["--mode", "official", "--inputs", missing, "--out-dir", out, "--run-id", "door-0001"]
    refused("D1 no --confirm-official-test-analysis (non-existent --inputs: no read happened)",
            cli(*base), "--confirm-official-test-analysis", out / "door-0001")
    refused("D2 confirmed, not declared synthetic: the binding rule (still before any read)",
            cli(*base, "--confirm-official-test-analysis"), "binding is unbound", out / "door-0001")
    refused("D3 --repo-root without --synthetic-inputs (P16)",
            cli(*base, "--confirm-official-test-analysis", "--repo-root", fs.input_root),
            "binding is unbound", out / "door-0001")
    (out / "door-0001").mkdir(parents=True)
    refused("D4 an existing target: the out-dir pre-check fires before any read (P10, P13)",
            cli(*base, "--confirm-official-test-analysis", "--synthetic-inputs"), "overwrite")
    (out / ".door-0002.tmp-0123456789ab").mkdir()
    refused("D5 a stale temporary sibling: refused before any read (P10)",
            cli(*base[:-1], "door-0002", "--confirm-official-test-analysis", "--synthetic-inputs"),
            "stale", out / "door-0002")
    if PINNED_STACK:
        refused("D6 every door passed: the first read is the input list (the missing file)",
                cli(*base[:-1], "door-0003", "--confirm-official-test-analysis",
                    "--synthetic-inputs"), "does not exist", out / "door-0003")
    else:
        refused("D6 an unpinned running stack: official mode refused before any read (Q6)",
                cli(*base[:-1], "door-0003", "--confirm-official-test-analysis",
                    "--synthetic-inputs"), "pinned statistics stack", out / "door-0003")
    real_obs = A.observed_environment
    A.observed_environment = lambda: dict(real_obs(), numpy=real_obs()["numpy"] + "+drifted")
    try:
        code, text = in_process(base[:-1] + ["door-0005", "--confirm-official-test-analysis",
                                             "--synthetic-inputs"])
    finally:
        A.observed_environment = real_obs
    check("D6b a running stack other than the pin (simulated in process, any stack): official mode "
          "refused before any read (Q6; the missing --inputs is never reached)",
          code == 2 and "pinned statistics stack" in text and "does not exist" not in text
          and not (out / "door-0005").exists(), text[-300:])
    refused("D7 --jobs 0", cli("--mode", "official", "--confirm-official-test-analysis",
                               "--synthetic-inputs", "--inputs", lst, "--out-dir", out, "--run-id",
                               "door-0004", "--jobs", "0"), "--jobs", out / "door-0004")


def canvas_and_smoke_mode(work: Path) -> None:
    """Both halves of the canvas guard and the smoke-mode refusals, in smoke mode (any stack)."""
    sfs = F.build_inventory(work / "smk", n=30, k=0, split="val", status="smoke", run_prefix="smk")
    out = work / "smk_out"

    def variant(tag, rel_index, fn):
        root = work / f"smk_{tag}"
        shutil.copytree(sfs.input_root, root)
        rel = sfs.entries[rel_index][2]
        F.edit_summary(root / rel, fn)
        return root, rel

    lst = F.write_input_list(work / "smk_inputs.json", sfs)

    def smoke(root, run_id, *extra):
        return cli("--mode", "smoke", "--inputs", lst, "--out-dir", out, "--run-id", run_id,
                   "--repo-root", root, "--B", "20", *extra)
    root, rel = variant("pp", 2, lambda s: (s["dataset"].__setitem__(
        "preprocess_protocol", "letterbox_512/1.0.0"), s)[1])
    refused(f"D8 canvas guard, half 1: {rel} with preprocess_protocol letterbox_512/1.0.0",
            smoke(root, "cv-0001"), rel, out / "cv-0001")
    root, rel = variant("block", 9, lambda s: dict(s, protocol={"name": "upstream"}))
    refused(f"D9 canvas guard, half 2: {rel} carrying a summary.protocol block",
            smoke(root, "cv-0002"), rel, out / "cv-0002")
    root, rel = variant("status", 4, lambda s: (s["run"].__setitem__("artifact_status",
                                                                     "provisional"), s)[1])
    refused(f"D10 smoke mode: a provisional input ({rel}) (P15)", smoke(root, "sm-0001"),
            "smoke-status", out / "sm-0001")
    root, rel = variant("test", 6, lambda s: (s["dataset"].update(name="PlantSeg", split="test"),
                                              s)[1])
    refused(f"D11 smoke mode: an input named PlantSeg with split test ({rel}) (P15)",
            smoke(root, "sm-0002"), "never reads a PlantSeg TEST", out / "sm-0002")
    kfs = F.build_inventory(work / "smk_k", n=30, k=2, split="val", status="smoke",
                            run_prefix="smkk")
    klst = F.write_input_list(work / "smk_k.json", kfs)
    refused("D12 smoke mode: inputs with AM-5 exclusions (k = 2)",
            cli("--mode", "smoke", "--inputs", klst, "--out-dir", out, "--run-id", "sm-0003",
                "--repo-root", kfs.input_root, "--B", "20"), "AM-5", out / "sm-0003")
    refused("D13 smoke mode: an official-mode flag", smoke(sfs.input_root, "sm-0004",
                                                           "--synthetic-inputs"),
            "official mode", out / "sm-0004")
    code, so, se = smoke(sfs.input_root, "sm-0005")
    check("D14 smoke mode on smoke inputs: exit 0, a nonofficial artifact, 13/13 re-established",
          code == 0 and "artifact_status: nonofficial" in so and "13/13" in so, se[-300:])

    # exit 4 (P18)
    script = ("import sys; sys.modules['src.stats.driver'] = None; "
              f"sys.path.insert(0, {str(REPO / 'scripts')!r}); import run_stats; "
              "sys.exit(run_stats.main(['--mode', 'smoke']))")
    p = subprocess.run([PY, "-B", "-c", script], cwd=str(REPO), capture_output=True, text=True)
    check("D15 an import failure inside the wrapper -> exit 4", p.returncode == 4,
          p.stderr[-300:])
    args = ["--mode", "smoke", "--inputs", lst, "--out-dir", out, "--repo-root", sfs.input_root,
            "--B", "20"]

    class BrokenPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, *a, **k):
            raise RuntimeError("worker died (simulated)")
    real_pool = D.ProcessPoolExecutor
    D.ProcessPoolExecutor = BrokenPool
    try:
        code, text = in_process(args + ["--run-id", "ex-0001", "--jobs", "2"])
    finally:
        D.ProcessPoolExecutor = real_pool
    check("D16 a worker failure -> exit 4 with the named error (no traceback), nothing written",
          code == 4 and "ERROR: a bootstrap worker failed" in text and "Traceback" not in text
          and not (out / "ex-0001").exists(), text[-300:])
    real_verify = A.verify_statistics_artifact

    def failing_after_rename(path, **kw):
        if not Path(path).name.startswith("."):
            raise A.IntegrityError(A.C_MANIFEST, "simulated post-rename failure")
        return real_verify(path, **kw)
    A.verify_statistics_artifact = failing_after_rename
    try:
        code, text = in_process(args + ["--run-id", "ex-0002"])
    finally:
        A.verify_statistics_artifact = real_verify
    check("D17 an artifact that fails its post-rename verification -> exit 4 naming the directory",
          code == 4 and str(out / "ex-0002") in text, text[-300:])


def official_canvas(work: Path, fs, lst: Path) -> None:
    """Both halves of the canvas guard in official mode (pinned stack only: the stack door first)."""
    out = work / "off_cv"
    for tag, idx, fn, needle in (
            ("pp", 2, lambda s: (s["dataset"].__setitem__("preprocess_protocol",
                                                          "letterbox_512/1.0.0"), s)[1], "pp"),
            ("block", 9, lambda s: dict(s, protocol={"name": "upstream"}), "block")):
        root = work / f"off_{tag}"
        shutil.copytree(fs.input_root, root)
        rel = fs.entries[idx][2]
        F.edit_summary(root / rel, fn)
        refused(f"D18 official mode, canvas guard ({needle}): {rel}",
                cli("--mode", "official", "--confirm-official-test-analysis", "--synthetic-inputs",
                    "--inputs", lst, "--out-dir", out, "--run-id", f"cv-{tag}", "--repo-root", root,
                    "--B", "20"), rel, out / f"cv-{tag}")


def fixture_expectations(fs) -> dict:
    """Every value P30 names, computed from the fixture arrays with the frozen arithmetic."""
    keep = ~fs.truth.am5

    def dv(stage, cond=D.CLEAN):
        return fs.scored[(stage, cond)].disease_values[keep]

    def pooled(stage):
        return pooled_miou_from_totals(*fs.scored[(stage, D.CLEAN)].totals)
    grid = official_corruption_grid()

    def miou_c(stage):
        return np.stack([np.stack([dv(stage, ("corruption", n, s)) for s in INFERENTIAL_SEVERITIES]
                                  ).mean(axis=0) for n in grid.names]).mean(axis=0)
    out = {}
    for cid in D.CLEAN_COMPARISON_IDS:
        b, c, _ = D.COMPARISON_TABLE[cid]
        out[cid] = float(np.mean(dv(c) - dv(b)))
    out[D.ROBUSTNESS_ID] = float(np.mean(miou_c("E6") - miou_c("E1")))
    out[BS.DESCRIPTIVE_E1_E3] = float(np.mean(dv("E3") - dv("E1")))
    out[BS.NONINFERIORITY_E3_E6] = pooled("E6") - pooled("E3")
    return out


def fixture_miou_c(fs, stage: str) -> float:
    """P28 from the fixture arrays: the mean over the five corruptions of the mean over severities 1-3
    of each cell's pooled all-class mIoU (float64)."""
    grid = official_corruption_grid()
    per = [float(np.mean(np.array([pooled_miou_from_totals(*fs.scored[(stage, ("corruption", n, s))]
                                                           .totals)
                                   for s in INFERENTIAL_SEVERITIES], dtype=np.float64)))
           for n in grid.names]
    return float(np.mean(np.array(per, dtype=np.float64)))


def report_mode(work: Path, fs, path: Path) -> None:
    """`--mode report` on the (a) artifact: the full verifier with the 37 inputs and the contract."""
    rdir = work / "report"
    t0 = time.time()
    code, so, se = cli("--mode", "report", "--artifact", path, "--report-out", rdir, "--repo-root",
                       fs.input_root, "--synthetic-inputs")
    wall = time.time() - t0
    ok = code == 0 and (rdir / "report.json").is_file() and (rdir / "report.md").is_file()
    check(f"A8 report mode on that artifact: exit 0, report.json and report.md (wall {wall:.0f} s)",
          ok, se[-600:])
    if not ok:
        return
    doc = json.loads((rdir / "report.json").read_text(encoding="utf-8"))
    md = (rdir / "report.md").read_text(encoding="utf-8")
    check(f"A9 k = {K} reported by the report layer: stdout, report.json and report.md",
          f"k (AM-5): {K}" in so and doc["am5_excluded_count_k"] == K
          and f"excluded from per-image analyses) = {K}." in md, so[-300:])
    check("A10 the profile is the verified status (nonofficial, synthetic_input_data only) and all "
          "13 checks were re-established", doc["profile"] == "nonofficial"
          and doc["warnings"] == [A.W_SYNTHETIC]
          and doc["integrity_checks_established"] == list(A.INTEGRITY_CHECKS), doc["warnings"])
    e1, e6 = fixture_miou_c(fs, "E1"), fixture_miou_c(fs, "E6")
    dm = doc["accuracy_on_corrupted_images"]["dataset_level_miou_c"]
    check("A11 P28: each model's dataset-level mIoU-C and the contrast E6 - E1 equal the fixture's "
          "nested mean of 15 cells exactly", (dm["E1"], dm["E6"], dm["difference_e6_minus_e1"])
          == (e1, e6, e6 - e1), (dm, e1, e6))


def full_run(work: Path, fs, lst: Path) -> None:
    out = work / "official"
    t0 = time.time()
    code, so, se = cli("--mode", "official", "--confirm-official-test-analysis", "--synthetic-inputs",
                       "--inputs", lst, "--out-dir", out, "--run-id", "syn-official-0001",
                       "--repo-root", fs.input_root, "--B", BS.PRODUCTION_B, "--jobs", JOBS)
    wall = time.time() - t0
    print(so)
    check(f"A1 official mode, synthetic inputs, B = 10000, --jobs {JOBS}: exit 0 "
          f"(wall {wall:.0f} s)", code == 0, se[-600:])
    if code != 0:
        return
    lines = dict(ln.split(": ", 1) for ln in so.splitlines() if ln.startswith(("ARTIFACT: ",
                                                                               "UNMET_OFFICIAL: ")))
    path = Path(lines["ARTIFACT"])
    rep = A.verify_statistics_artifact(path, input_root=fs.input_root,
                                       contract_bytes=A.contract_sha256()[1])
    fam = rep.family
    check("A2 all 13 integrity checks re-established from disk with the inputs and the contract",
          rep.established == A.INTEGRITY_CHECKS and not rep.not_reestablished)
    tasks = fam["bootstrap_tasks"]
    pi = {t["jackknife_count"] for t in tasks if not t["task_id"].endswith("dataset_miou_delta")}
    ds = {t["jackknife_count"] for t in tasks if t["task_id"].endswith("dataset_miou_delta")}
    check(f"A3 counts: the 27 per-image tasks 1561 - k = {N - K}, the 8 dataset-level tasks 1561, "
          "every task B = 10000", pi == {N - K} and ds == {N}
          and {t["bootstrap_replicates"] for t in tasks} == {BS.PRODUCTION_B}
          and sum(1 for t in tasks if t["jackknife_count"] == N - K) == 27, (pi, ds))
    check("A4 the only warning is synthetic_input_data; status nonofficial",
          fam["warnings"] == [A.W_SYNTHETIC] and fam["artifact_status"] == "nonofficial",
          fam["warnings"])
    want = fixture_expectations(fs)
    comps = {c["comparison_id"]: c for c in fam["a3a_family"]["comparisons"]}
    got = {cid: comps[cid]["shifts"]["mean_delta"] for cid in comps}
    got[BS.DESCRIPTIVE_E1_E3] = fam["descriptive_e1_e3"]["mean_delta"]
    got[BS.NONINFERIORITY_E3_E6] = fam["noninferiority_e3_e6"]["observed_delta"]
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    check("A5 each comparison's mean_delta, the descriptive delta and the NI observed_delta equal "
          "the fixture arrays exactly (P30)", not bad, bad)
    unmet = set(json.loads(lines["UNMET_OFFICIAL"]))
    check("A6 the unmet official conditions are exactly the synthetic declaration and the binding "
          "(P30; needs a committed clean tree, O3)",
          unmet == {A.W_SYNTHETIC, A.BINDING_CONDITION}, sorted(unmet))
    check(f"A7 k = {K} reported by the driver (never a family.json field)",
          f"AM-5 k (zero-disease images excluded from per-image tasks): {K}" in so
          and "am5" not in json.dumps(list(fam)))
    print(f"FULL-SIZE WALL SECONDS: {wall:.1f}")
    report_mode(work, fs, path)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="smoke_stats_fullsize_"))
    try:
        fs = F.build_inventory(work / "fx", n=N, k=K, split="test", status="official",
                               run_prefix="full")
        lst = F.write_input_list(work / "inputs.json", fs)
        doors(work, fs, lst)
        canvas_and_smoke_mode(work)
        if PINNED_STACK:
            official_canvas(work, fs, lst)
            full_run(work, fs, lst)
        else:
            check("A0 (a) not run: official mode is refused on an unpinned running stack (Q6; D6)",
                  True)
    except Exception:                                                    # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=3))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check("Z1 temporary fixtures and artifacts removed", not work.exists())

    print()
    for name, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if det and not ok:
            print(f"         {det[:600]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'STATS OFFICIAL FULL-SIZE OK' if allok else 'STATS OFFICIAL FULL-SIZE FAILED'} "
          f"({good}/{len(CHECKS)}; "
          + ("pinned stack, (a) run" if PINNED_STACK else "unpinned stack, (a) not run") + ")")
    print("NONOFFICIAL: synthetic inputs; no statistics artifact with status official exists.")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
