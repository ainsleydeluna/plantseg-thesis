#!/usr/bin/env python3
"""Single pre-flight gate for the real E1 run (B31c-3).

ONE command, ONE verdict. Run this on the pod immediately before launching E1.

It ORCHESTRATES existing scripts and re-implements none of their checks, with ONE exception noted
at stage 1 below:

  1. CE class-weight artifact       — IN-PROCESS (B34b). The only stage that is not a delegation.
                                        No other script asserts anything about
                                        reports/e1_class_weights.json, which is exactly the gap the
                                        B34 audit found (its section A5): train_e1.py loads the file
                                        unconditionally and aborts without it, but nothing verified
                                        its CONTENT. Runs first because it is the cheapest check
                                        here (one JSON read) and guards an artifact E1 hard-depends
                                        on.
  2. scripts/verify_env.py            — platform, pinned versions, the B31-3 CUDA compute-capability
                                        gate (sm<=9.0) and the B31-4 dataset verification.
  3. scripts/smoke_dataloader.py      — one train + one val batch: shapes, dtypes, label range.
  4. scripts/smoke_aug_stochasticity.py — augmentation varies across epochs, val is bit-identical,
                                        seeds reproduce across processes. On the pod this doubles as
                                        the R5 PINNED-STACK re-measurement: the observed seed
                                        sequence is compared against the B30 §9 baseline.
  5. src/training/train_e1.py --dry-run — the end-to-end scaffold floor.

Stage 1 is a REPO-ARTIFACT check, not a dataset check: it proves the committed weight vector is the
one B18a computed. It does NOT prove the pod's dataset matches the histogram those weights came from
— identical split counts are compatible with a different pixel histogram. That binding is the job of
scripts/verify_class_weights_pod.py, which is ADVISORY and deliberately NOT a stage here.

Stops at the first hard failure. Exits non-zero on NO-GO.

Why the seed comparison is a hard gate: every seed-dependent artifact under reports/ was produced on
the development stack. If the pinned stack draws a different sequence, those artifacts describe an
augmentation schedule the pod will not run, and they would all need regenerating on the pod before
any of them could be cited.

Usage:
    python scripts/preflight_e1.py                  # real launch gate
    python scripts/preflight_e1.py --dev-rehearsal  # CPU dev box: exercise the gate, never a GO
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# B34b: the CE class-weight artifact the E1 loop loads unconditionally (train_e1.py:49,55,293).
CLASS_WEIGHTS_JSON = REPO / "reports" / "e1_class_weights.json"
EXPECTED_NUM_CLASSES = 116          # configs/data.py DATA["num_classes"]; background = index 0
BACKGROUND_INDEX = 0

# B30 §9 / B31-5: the first two per-sample augmentation seeds of pass 1 and pass 2 over the fixed
# probe subset, measured on the development stack. smoke_aug_stochasticity.py prints these.
B30_SEED_BASELINE = [[1608637542, 1273642419], [1935803228, 787846414]]

HARD_CHECKS_TOTAL = 6          # the train_e1 scaffold's hard-check count


def _run(cmd, label):
    print("\n" + "=" * 78)
    print(f"STAGE {label}")
    print("=" * 78)
    print("$ " + " ".join(str(c) for c in cmd) + "\n", flush=True)
    t0 = time.time()
    # encoding must be explicit: text=True otherwise decodes with the locale codec (cp1252 on
    # Windows), which mangles the non-ASCII characters the child scripts print.
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=3600)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stdout.write("\n--- stderr ---\n" + r.stderr)
    return r, time.time() - t0


def _banner(label):
    """Stage banner for an IN-PROCESS stage. Mirrors _run()'s header so the transcript reads the
    same whether a stage delegates to a subprocess or not."""
    print("\n" + "=" * 78)
    print(f"STAGE {label}")
    print("=" * 78)
    print(f"# in-process check of {CLASS_WEIGHTS_JSON.relative_to(REPO).as_posix()}\n", flush=True)
    return time.time()


def check_class_weights(path=CLASS_WEIGHTS_JSON):
    """B34b Tier 1. Assert the CE class-weight artifact is the one B18a computed.

    Returns (ok, detail). NEVER raises: a malformed artifact is a NO-GO, not a traceback.
    `path` is a parameter so the check can be exercised against a deliberately corrupted copy
    without touching the real artifact.
    """
    import torch                       # lazy: keeps the gate's own import cost near zero

    def fail(assertion, found):
        print(f"  [FAIL] {assertion}")
        print(f"         artifact contained: {found}")
        return False, f"{assertion} (artifact contained: {found})"

    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:                                            # noqa: BLE001
        return fail("artifact unreadable", f"{type(e).__name__}: {e}")

    for key in ("weights", "pixel_counts", "total_nonignore_pixels"):
        if key not in doc:
            return fail(f"artifact key {key!r} present", f"keys={sorted(doc)}")
    w, counts = doc["weights"], doc["pixel_counts"]

    # --- T1a: length ------------------------------------------------------------------------
    if len(w) != EXPECTED_NUM_CLASSES:
        return fail(f"T1a len(weights) == {EXPECTED_NUM_CLASSES}", f"len={len(w)}")
    if len(counts) != EXPECTED_NUM_CLASSES:
        return fail(f"T1a len(pixel_counts) == {EXPECTED_NUM_CLASSES}", f"len={len(counts)}")

    # --- T1b: finite + float32 after the conversion train_e1.load_ce_weights performs ---------
    t = torch.tensor(w, dtype=torch.float32)
    if not bool(torch.isfinite(t).all()):
        bad = [i for i, v in enumerate(w) if not math.isfinite(v)]
        return fail("T1b all weights finite", f"non-finite at indices {bad[:8]}")
    if t.dtype is not torch.float32:
        return fail("T1b dtype is float32", f"dtype={t.dtype}")

    # --- T1c: PROVENANCE fingerprint -- NOT a correctness check --------------------------------
    # This records WHICH generator produced the artifact; it cannot affect training. A uniform
    # scale on CE weights cancels in F.cross_entropy(reduction='mean'), which normalizes by the
    # sum of per-pixel weights (measured in B34 section 4: gradient-norm delta 9.77e-08, below
    # float32 eps). Kept because it pins provenance, not because the number matters numerically.
    #
    # DO NOT "SIMPLIFY" THIS TO np.median / torch.median. num_classes is 116, i.e. EVEN, and the
    # conventions disagree: torch.median returns the LOWER middle (0.996273398399 here) and would
    # NO-GO a perfectly correct artifact. The artifact was median-normalized with the
    # average-of-the-two-middles convention, so the two-element form below is the only correct one.
    srt = sorted(w)
    lo, hi = srt[EXPECTED_NUM_CLASSES // 2 - 1], srt[EXPECTED_NUM_CLASSES // 2]
    median_two_element = (lo + hi) / 2
    if not math.isclose(median_two_element, 1.0, abs_tol=1e-9):
        return fail("T1c median-normalization fingerprint == 1.0 (two-element form)",
                    f"(sorted[57]+sorted[58])/2 = {median_two_element!r}")

    # --- T1d: CORRECTNESS + dataset binding ----------------------------------------------------
    argmin, argmax = int(t.argmin()), int(t.argmax())
    if argmin != BACKGROUND_INDEX:
        return fail(f"T1d argmin == {BACKGROUND_INDEX} (background is the most frequent class)",
                    f"argmin={argmin}, w[{argmin}]={w[argmin]!r}")
    if argmax != 42:
        return fail("T1d argmax == 42", f"argmax={argmax}, w[{argmax}]={w[argmax]!r}")
    # Cross-check the recorded argmax against the histogram rather than trusting the constant:
    # w_c is proportional to 1/sqrt(N_c), so the largest weight must be the rarest class.
    rarest = min(range(EXPECTED_NUM_CLASSES), key=lambda c: counts[c])
    if rarest != argmax:
        return fail("T1d argmax(weights) == argmin(pixel_counts)",
                    f"argmax={argmax} but rarest class is {rarest} (N={counts[rarest]})")
    # The ratio is scale-invariant (it survives ANY renormalization) yet is a pure function of the
    # pixel histogram, so it breaks the moment the class distribution changes. Re-derived here from
    # the artifact's own pixel_counts -- deliberately NOT hardcoded.
    expected_ratio = math.sqrt(counts[BACKGROUND_INDEX] / counts[argmax])
    actual_ratio = w[argmax] / w[BACKGROUND_INDEX]
    if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-9):
        return fail("T1d w[42]/w[0] == sqrt(N_0/N_42)",
                    f"ratio={actual_ratio!r} vs sqrt({counts[BACKGROUND_INDEX]}/"
                    f"{counts[argmax]})={expected_ratio!r}")

    # --- T1e: index 0 is the small background weight, not a 1.0 placeholder --------------------
    # 1.0 is exactly what an absent/placeholder class receives (it equals the median), so a
    # background weight at or near 1.0 means the vector was NOT computed over the real train set.
    if not w[BACKGROUND_INDEX] < 0.5:
        return fail("T1e w[0] is the small background weight, not a 1.0 placeholder",
                    f"w[0]={w[BACKGROUND_INDEX]!r}")

    # --- index alignment: reproduce every weight from the artifact's own histogram --------------
    # Closes the remaining B34 A5 item. w_c = sqrt(total / (num_classes * N_c)), median-normalized.
    total = doc["total_nonignore_pixels"]
    if sum(counts) != total:
        return fail("sum(pixel_counts) == total_nonignore_pixels",
                    f"sum={sum(counts)} vs total={total}")
    raw = [math.sqrt(total / (EXPECTED_NUM_CLASSES * c)) if c else float("inf") for c in counts]
    rs = sorted(raw)
    rmed = (rs[EXPECTED_NUM_CLASSES // 2 - 1] + rs[EXPECTED_NUM_CLASSES // 2]) / 2
    worst = max(range(EXPECTED_NUM_CLASSES), key=lambda c: abs(raw[c] / rmed - w[c]))
    worst_err = abs(raw[worst] / rmed - w[worst])
    if worst_err > 1e-9:
        return fail("index alignment: weights reproduce from pixel_counts",
                    f"largest deviation {worst_err:.3e} at class {worst}")

    print(f"  [PASS] T1a  len(weights) == {EXPECTED_NUM_CLASSES}")
    print(f"  [PASS] T1b  all finite; dtype={t.dtype}")
    print(f"  [PASS] T1c  provenance fingerprint (sorted[57]+sorted[58])/2 == "
          f"{median_two_element:.12f}  [NOT a correctness check]")
    print(f"  [PASS] T1d  argmin={argmin} argmax={argmax}; "
          f"w[{argmax}]/w[{BACKGROUND_INDEX}] = {actual_ratio:.12f} == "
          f"sqrt({counts[BACKGROUND_INDEX]}/{counts[argmax]})")
    print(f"  [PASS] T1e  w[0] = {w[BACKGROUND_INDEX]:.12f} (background, {100.0 * counts[0] / total:.2f}% of pixels)")
    print(f"  [PASS] index alignment: all {EXPECTED_NUM_CLASSES} weights reproduce "
          f"(max deviation {worst_err:.3e})")
    return True, f"{EXPECTED_NUM_CLASSES} weights verified; ratio {actual_ratio:.6f}"


def stage_class_weights(_dev):
    t0 = _banner("1/5 — CE class-weight artifact (B34b Tier 1)")
    ok, detail = check_class_weights()
    return ok, detail, time.time() - t0


def stage_verify_env(dev_rehearsal):
    r, secs = _run([PY, str(REPO / "scripts" / "verify_env.py")], "2/5 — verify_env.py")
    # verify_env returns 0 regardless of verdict, so the RESULT line is the authority.
    line = next((x for x in reversed(r.stdout.splitlines()) if x.startswith("RESULT:")), "")
    verdict = line.split("RESULT:", 1)[1].split("|")[0].strip() if line else "UNKNOWN"
    if verdict == "PASS":
        return True, f"verdict={verdict}", secs
    if verdict == "PARTIAL" and dev_rehearsal:
        return True, f"verdict={verdict} (accepted under --dev-rehearsal ONLY)", secs
    return False, f"verdict={verdict} — needs PASS (CUDA GPU, sm<=9.0, dataset OK)", secs


def stage_dataloader(_dev):
    r, secs = _run([PY, str(REPO / "scripts" / "smoke_dataloader.py")],
                   "3/5 — smoke_dataloader.py")
    ok = r.returncode == 0 and "[PASS] dataloader smoke test" in r.stdout
    return ok, f"exit={r.returncode}", secs


def stage_stochasticity(_dev):
    r, secs = _run([PY, str(REPO / "scripts" / "smoke_aug_stochasticity.py")],
                   "4/5 — smoke_aug_stochasticity.py  (R5 pinned-stack seed re-measurement)")
    if r.returncode != 0 or "RESULT: PASS" not in r.stdout:
        return False, f"exit={r.returncode}, gate did not pass", secs

    m = re.search(r"process A seeds\s*:\s*(\[\[.*?\]\])", r.stdout)
    if not m:
        return False, "could not read the seed sequence from stdout", secs
    observed = ast.literal_eval(m.group(1))

    print("\n--- R5: pinned-stack seed comparison ---")
    print(f"  B30 §9 baseline : {B30_SEED_BASELINE}")
    print(f"  observed here   : {observed}")
    if observed == B30_SEED_BASELINE:
        print("  VERDICT         : MATCH — the pinned stack draws the development-stack sequence.")
        print("                    Seed-dependent artifacts under reports/ remain valid.")
        return True, "seed sequence MATCH", secs
    print("  VERDICT         : DEVIATION — the pinned stack draws a DIFFERENT sequence.")
    print("                    This is a NO-GO. Every seed-dependent artifact under reports/")
    print("                    describes a schedule this machine will not run and must be")
    print("                    regenerated here before it can be cited.")
    return False, "seed sequence DEVIATION vs B30 §9 baseline", secs


def stage_dryrun(_dev):
    r, secs = _run([PY, str(REPO / "src" / "training" / "train_e1.py"), "--dry-run"],
                   "5/5 — train_e1.py --dry-run")
    line = next((x for x in reversed(r.stdout.splitlines()) if x.startswith("RESULT:")), "")
    print(f"\n--- final line: {line!r} ---")
    if r.returncode != 0:
        return False, f"exit={r.returncode}", secs
    if line.startswith("RESULT: NOOP"):
        return False, "NOOP — zero checks exercised; not a launch gate", secs
    if not line.startswith("RESULT: PASS"):
        return False, f"non-PASS: {line}", secs
    # A3: assert the checks-exercised COUNT, not merely a zero exit. A reduced-coverage dry-run
    # (e.g. lr_non_increasing SKIPPED) still exits 0 and still says PASS.
    m = re.search(r"\((\d+)/(\d+) checks exercised, (\d+) skipped\)", line)
    if not m:
        return False, "RESULT line carries no checks-exercised count (stale train_e1.py?)", secs
    exercised, total, skipped = (int(x) for x in m.groups())
    if total != HARD_CHECKS_TOTAL or exercised != total or skipped != 0:
        return False, f"reduced coverage: {exercised}/{total} exercised, {skipped} skipped", secs
    return True, f"{exercised}/{total} checks exercised, {skipped} skipped", secs


STAGES = [
    ("class_weights", stage_class_weights),
    ("verify_env", stage_verify_env),
    ("smoke_dataloader", stage_dataloader),
    ("smoke_aug_stochasticity", stage_stochasticity),
    ("train_e1 --dry-run", stage_dryrun),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E1 pre-flight gate.")
    ap.add_argument("--dev-rehearsal", action="store_true",
                    help="accept verify_env PARTIAL (no CUDA) so the gate can be exercised on a CPU "
                         "development box. NEVER produces a GO.")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("E1 PRE-FLIGHT GATE")
    print(f"repo={REPO}")
    print(f"python={PY}")
    if args.dev_rehearsal:
        print("MODE: --dev-rehearsal — exercising the gate on a non-GPU box. NOT a launch "
              "authorization.")
    print("=" * 78)

    results, first_failure = [], None
    for name, fn in STAGES:
        ok, detail, secs = fn(args.dev_rehearsal)
        results.append((name, ok, detail, secs))
        if not ok:
            first_failure = name
            print(f"\n*** STOPPING: stage '{name}' failed ({detail}) ***")
            break

    print("\n" + "=" * 78)
    print("PRE-FLIGHT SUMMARY")
    print("=" * 78)
    print(f"  {'stage':<26}{'result':<10}{'secs':>7}  detail")
    for name, ok, detail, secs in results:
        print(f"  {name:<26}{'PASS' if ok else 'FAIL':<10}{secs:>7.1f}  {detail}")
    for name, _fn in STAGES[len(results):]:
        print(f"  {name:<26}{'SKIPPED':<10}{'-':>7}  not reached")

    all_ok = all(ok for _n, ok, _d, _s in results) and len(results) == len(STAGES)
    go = all_ok and not args.dev_rehearsal

    print()
    if go:
        print("VERDICT: GO — every gate passed. Proceed to the launch command in "
              "reports/e1_launch_runbook_v2.md.")
        print("NOTE: two residuals are NOT code-gated and remain the operator's responsibility —")
        print("      the SegNeXt teacher checkpoint is absent (blocks E2/E3, not E1), and the")
        print("      ImageNet backbone may not be cached (needed for --init imagenet).")
        return 0
    if all_ok and args.dev_rehearsal:
        print("VERDICT: NO-GO (--dev-rehearsal) — all stages passed, but this run was a rehearsal "
              "on a non-GPU box and is not a launch authorization.")
        print("         Re-run WITHOUT --dev-rehearsal on the pod to obtain a real GO.")
        return 1
    print(f"VERDICT: NO-GO — first failing stage: {first_failure}.")
    print("         Fix it and re-run. Do not launch E1.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
