#!/usr/bin/env python3
"""Statistics report layer (lane L-STATS-OFFICIAL; acceptance (g), and (d) for report mode).

SYNTHETIC ONLY: 37 synthetic evaluation artifacts in a temporary clean fixture repository outside
the working tree (scripts/stats_fixtures.py), a nonofficial statistics artifact written by
`run_stats.py --mode smoke`, and report mode on it. No PlantSeg data, checkpoint, model or GPU.

  G  the AM-17 item 2 / 3(f) labels on crafted values built as literals or atoms (never by
     subtracting decimals or from constant non-dyadic vectors): the 0.010 boundary, one ulp below it,
     a rejection with a negative difference, a case where the Holm decision differs from the
     library's, the inconclusive caveat; the MDE validation and the official MDE refusal on
     in-memory data (P35)
  R  report mode end to end: report.json and report.md; the code commit, contract sha256, MDE sha256
     and commit, every input's file hashes (P29); a banner on every table; the family table's W, z,
     r_rb, dz and BCa columns; the P28 dataset-level mIoU-C against the fixture totals; the robustness
     label from it; the item 3(f) caveat pending without a committed MDE entry; z0 and an explicit
     containment column per interval; the stored NI sensitivity flags; E6-KD launches nothing; k;
     AM-6; the three extension points
  K  k > 0: an OFFICIAL-policy synthetic artifact with k = 3 AM-5 images (cheap placeholder tasks):
     k reported, and the AM-6 table equal to rule_variants for all seven stages, both rules and both
     class spaces; on 30 images with k = 3 the GT-present columns differ from the union-present ones
  N  acceptance (g) on a crafted artifact: an NI lower bound of exactly -0.020 (a literal) fails, its
     four sensitivity decisions come from that one bound, and a stored flag that contradicts the
     bound is refused (observed_source_mapping_verified)
  D  refusals (exit 2): --repo-root without --synthetic-inputs, the canvas guard, an artifact the
     verifier reads as official (relabelled in memory; no smoke writes an official status, P2), a
     tampered artifact, an existing --report-out, a synthetic declaration the artifact does not
     carry, an official-mode flag

Run:  python -B scripts/smoke_stats_report.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                       # noqa: E402

import stats_fixtures as F                                               # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import bootstrap as BS                                    # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import report as RP                                       # noqa: E402
from src.stats.corruption_protocol import official_corruption_grid       # noqa: E402
from src.stats.eligibility import GT_PRESENT, UNION_PRESENT, rule_variants  # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402
from src.stats.noninferiority import pooled_miou_from_totals             # noqa: E402
from src.stats.robustness import INFERENTIAL_SEVERITIES                  # noqa: E402
from src.stats.tests import CANONICAL_COMPARISON_IDS, holm_family        # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
PY = sys.executable


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def raises(fn, exc=RP.ReportError, needle=""):
    try:
        fn()
    except exc as e:
        return needle in str(e)
    except Exception:                                                    # noqa: BLE001
        return False
    return False


def in_process(*args):
    """run_stats.main in this process: (exit code, stdout, stderr)."""
    import contextlib
    import io

    import run_stats
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_stats.main([str(a) for a in args])
    return code, out.getvalue(), err.getvalue()


class verifier_reads_official:
    """The verifier reads family.json as status official with no warnings: relabelled in memory right
    after the strict parse (check 10), so no smoke writes an official status to disk (P2)."""

    def __enter__(self):
        self.real = real = A._check_finite_values

        def relabel(ctx):
            real(ctx)
            ctx.fam = dict(ctx.fam, artifact_status="official", warnings=[])
        A._check_finite_values = relabel

    def __exit__(self, *exc):
        A._check_finite_values = self.real


def outcome_ok(fn) -> bool:
    try:
        fn()
        return True
    except Exception:                                                    # noqa: BLE001
        return False


def cli(*args):
    p = subprocess.run([PY, "-B", str(REPO / "scripts" / "run_stats.py"), *map(str, args)],
                       cwd=str(REPO), capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return p.returncode, p.stdout, p.stderr


# --------------------------------------------------------------------------------------------------
def part_g() -> None:
    L = RP.contrast_label
    below = float(np.nextafter(0.010, 0.0))
    check("G1 a rejection with a difference of exactly 0.010 (literal) is sizable",
          L(True, 0.010) == RP.LABEL_SIZABLE and RP.SESOI == 0.010)
    check("G2 one ulp below 0.010 (an atom: nextafter) is detected, below the smallest effect of "
          "interest", L(True, below) == RP.LABEL_BELOW_SESOI, repr(below))
    check("G3 a rejection with a negative difference is detected, below the smallest effect of "
          "interest", L(True, -0.0125) == RP.LABEL_BELOW_SESOI)
    check("G4 not rejected; with the MDE power caveat: inconclusive at the smallest effect of "
          "interest", L(False, 0.05) == RP.LABEL_NOT_REJECTED
          and L(False, 0.05, power_caveat=True) == RP.LABEL_INCONCLUSIVE
          and L(False, 0.05, power_caveat=False) == RP.LABEL_NOT_REJECTED)
    check("G5 labels refuse a non-bool decision and a non-finite or non-float difference",
          raises(lambda: L(1, 0.02)) and raises(lambda: L(True, float("nan")))
          and raises(lambda: L(True, 1)))
    # the Holm decision differs from the library's: p = alpha / 8 exactly (a literal)
    carriers = [types.SimpleNamespace(comparison_id=c, primary_p=p, alignment_status="ok")
                for c, p in zip(CANONICAL_COMPARISON_IDS, [0.00625] + [0.5] * 7)]
    holm = A.serialize_a3a(holm_family(carriers))
    fam = {"a3a_family": {"holm": holm, "comparisons": [
        {"comparison_id": c, "baseline_stage": "B", "candidate_stage": "C", "metric": "m",
         "n_paired": 10, "wilcoxon": {"statistic": 40.0, "zstatistic": 2.5, "p_value": p,
                                      "status": "ok"},
         "rank_biserial": {"value": 0.5}, "cohens_dz": {"value": 0.75},
         "shifts": {"mean_delta": 0.03, "median_delta": 0.02}, "hodges_lehmann_shift": 0.02,
         "t_test": {"statistic": 2.0, "p_value": 0.03}}
        for c, p in zip(CANONICAL_COMPARISON_IDS, [0.00625] + [0.5] * 7)]},
        "bootstrap_tasks": [{"task_id": f"{c}__dataset_miou_delta", "observed": 0.05}
                            for c in CANONICAL_COMPARISON_IDS[:7]]
        + [{"task_id": f"{c}__mean_delta", "observed": 0.03, "lower_bound": 0.0125,
            "upper_bound": 0.0625} for c in CANONICAL_COMPARISON_IDS]}
    rows = RP.family_rows(fam, 0.05)
    m0 = holm["members"][0]
    check("G6 reject != library_reject (p = 0.00625 = alpha / 8): the label follows the strict Holm "
          "decision", m0["reject"] is False and m0["library_reject"] is True
          and rows[0]["label"] == RP.LABEL_NOT_REJECTED, (m0["reject"], m0["library_reject"]))
    fam2 = copy.deepcopy(fam)
    fam2["a3a_family"]["holm"]["members"][7]["reject"] = True
    rows2 = RP.family_rows(fam2, float(np.nextafter(0.010, 0.0)))
    check("G7 the robustness label uses the P28 dataset-level mIoU-C difference passed in",
          rows2[7]["label"] == RP.LABEL_BELOW_SESOI and RP.family_rows(fam2, 0.010)[7]["label"]
          == RP.LABEL_SIZABLE)
    caveat_rows = RP.family_rows(fam, 0.05, power_caveat=True, mde_w=0.0125)
    check("G8 with the power caveat every non-rejection is inconclusive and states the MDE",
          all(r["label"] == RP.LABEL_INCONCLUSIVE and r["mde_w"] == 0.0125 for r in caveat_rows))
    pending = RP.family_rows(fam2, 0.05, power_caveat="pending")
    check("G12 without a committed MDE entry every non-rejection reads 'power caveat pending'; a "
          "rejection keeps its item 2 label (lane item 7)",
          all((RP.CAVEAT_PENDING in r["label"]) is (not r["holm_reject"]) for r in pending)
          and pending[7]["label"] == RP.LABEL_SIZABLE)
    check("G13 the family table carries W, z, r_rb, dz and the mean delta's BCa interval",
          rows[0]["wilcoxon_statistic"] == 40.0 and rows[0]["wilcoxon_z"] == 2.5
          and rows[0]["rank_biserial"] == 0.5 and rows[0]["cohens_dz"] == 0.75
          and rows[0]["mean_delta_bca"] == [0.0125, 0.0625])
    fam3 = copy.deepcopy(fam)
    fam3["a3a_family"]["holm"]["members"][1]["reject"] = True
    t1 = next(t for t in fam3["bootstrap_tasks"]
              if t["task_id"] == f"{CANONICAL_COMPARISON_IDS[1]}__dataset_miou_delta")
    t1["observed"] = float(np.nextafter(0.010, 0.0))
    fam3["a3a_family"]["comparisons"][1]["shifts"]["mean_delta"] = 0.03
    lab_below = RP.family_rows(fam3, 0.05)[1]["label"]
    t1["observed"] = 0.010
    fam3["a3a_family"]["comparisons"][1]["shifts"]["mean_delta"] = 0.005
    lab_sizable = RP.family_rows(fam3, 0.05)[1]["label"]
    check("G16 a rejected clean contrast is labelled from its __dataset_miou_delta value, not its "
          "per-image shifts (one ulp below 0.010 with mean delta 0.03; 0.010 with mean delta 0.005)",
          lab_below == RP.LABEL_BELOW_SESOI and lab_sizable == RP.LABEL_SIZABLE,
          (lab_below, lab_sizable))
    E = RP.estimate_inside
    check("G17 containment is a closed comparison, never assumed: one-sided lower == estimate is "
          "inside, lower > estimate is not; two-sided estimate == upper is inside, above or below is "
          "not", E(0.25, None, 0.25) is True and E(0.5, None, 0.25) is False
          and E(0.125, 0.25, 0.25) is True and E(0.125, 0.25, 0.5) is False
          and E(0.125, 0.25, 0.0625) is False)
    deep = Path(tempfile.mkdtemp(prefix="smoke_report_mde_")) / "mde_entry_deep.json"
    try:
        deep.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
        check("G19 an MDE entry nested beyond the parser's depth is refused by name (exit 2, never "
              "4)", raises(lambda: RP.load_mde_entry(deep), needle="RecursionError"))
    finally:
        shutil.rmtree(deep.parent, ignore_errors=True)
    C = RP.committed_caveat
    ok = {"mde_w": 0.0125, "tau_p": 0.010, "power_caveat": True, "problems": [],
          "inside_repository": True, "tracked": True, "clean_at_head": True}
    check("G14 the caveat is read from a committed, valid entry only; otherwise it is pending",
          C(None) == ("pending", None) and C(ok) == (True, 0.0125)
          and C(dict(ok, tracked=False)) == ("pending", None)
          and C(dict(ok, clean_at_head=False)) == ("pending", None)
          and C(dict(ok, inside_repository=False)) == ("pending", None)
          and C(dict(ok, problems=["x"])) == ("pending", None))
    check("G15 a malformed MDE entry is recorded JSON-safe, never fatal (NaN, a list, a non-object)",
          RP.mde_fields({"mde_w": float("nan"), "tau_p": [1], "power_caveat": True})
          == {"mde_w": "nan", "tau_p": "[1]", "power_caveat": True}
          and RP.mde_fields([0.1]) == {"mde_w": None, "tau_p": None, "power_caveat": None}
          and RP.validate_mde_entry([0.1]) != ())
    V = RP.validate_mde_entry
    edge = float(np.nextafter(0.010, 1.0))
    check("G9 MDE validation: mde_w, tau_p = 0.010 and power_caveat == (mde_w > tau_p), the "
          "boundary mde_w = 0.010 (literal) with no caveat and one ulp above with the caveat",
          V({"mde_w": 0.010, "tau_p": 0.010, "power_caveat": False}) == ()
          and V({"mde_w": edge, "tau_p": 0.010, "power_caveat": True}) == ()
          and V({"mde_w": 0.010, "tau_p": 0.010, "power_caveat": True}) != ()
          and V({"mde_w": 0.02, "tau_p": 0.02, "power_caveat": False}) != ()
          and V({"mde_w": 1, "tau_p": 0.010, "power_caveat": False}) != ()
          and V({"mde_w": 0.02, "tau_p": 0.010, "power_caveat": "yes"}) != ())
    ok = {"mde_w": 0.0125, "tau_p": 0.010, "power_caveat": True}
    st = {"path": "/r/reports/derived/mde_entry_20261010T000000Z.json", "tracked": True,
          "clean_at_head": True}
    tracked = ["reports/derived/mde_entry_20261010T000000Z.json"]

    def R(*a):
        return RP.require_official_mde(*a, repo=Path("/r"))
    cases = [("no entry given", (None, None, (), tracked), "requires --mde-entry"),
             ("none tracked", (ok, st, (), []), "exactly one"),
             ("two tracked", (ok, st, (), tracked + ["reports/derived/mde_entry_x.json"]),
              "exactly one"),
             ("not committed", (ok, dict(st, tracked=False), (), tracked), "not committed"),
             ("modified at HEAD", (ok, dict(st, clean_at_head=False), (), tracked),
              "not committed"),
             ("another file", (ok, dict(st, path="/r/elsewhere.json"), (), tracked),
              "not the committed entry"),
             ("a file whose path only ends with the committed one",
              (ok, dict(st, path="/r/xreports/derived/mde_entry_20261010T000000Z.json"), (),
               tracked), "not the committed entry"),
             ("the same name under another directory",
              (ok, dict(st, path="/r/scratch/reports/derived/mde_entry_20261010T000000Z.json"), (),
               tracked), "not the committed entry"),
             ("invalid fields", (ok, st, ("tau_p 0.02 is not 0.01",), tracked), "tau_p"),
             ("Git unavailable", (ok, st, (), None), "Git")]
    for name, args, needle in cases:
        check(f"G10 official MDE refusal on in-memory data: {name}",
              raises(lambda a=args: R(*a), needle=needle))
    try:
        R(ok, st, (), tracked)
        passed = True
    except RP.ReportError:
        passed = False
    check("G11 exactly one committed, valid MDE entry is accepted", passed)


def fixture_miou_c(fs, stage) -> float:
    grid = official_corruption_grid()
    per = []
    for n in grid.names:
        per.append(float(np.mean(np.array([pooled_miou_from_totals(*fs.scored[(
            stage, ("corruption", n, s))].totals) for s in INFERENTIAL_SEVERITIES]))))
    return float(np.mean(np.array(per)))


def part_r(work: Path) -> tuple:
    fs = F.build_inventory(work / "fx", n=30, k=0, split="val", status="smoke", run_prefix="rep")
    lst = F.write_input_list(work / "inputs.json", fs)
    code, so, se = cli("--mode", "smoke", "--inputs", lst, "--out-dir", work / "stats", "--run-id",
                       "rep-0001", "--repo-root", fs.input_root, "--B", 60)
    art = work / "stats" / "rep-0001"
    check("R0 a nonofficial statistics artifact written by smoke mode", code == 0, se[-300:])
    mde = work / "mde_entry_20261003T000000Z.json"
    mde.write_text(json.dumps({"mde_w": 0.0125, "tau_p": 0.010, "power_caveat": True}),
                   encoding="utf-8")
    out = work / "report_1"
    code, so, se = cli("--mode", "report", "--artifact", art, "--report-out", out, "--repo-root",
                       fs.input_root, "--synthetic-inputs", "--mde-entry", mde)
    check("R1 report mode: exit 0, report.json and report.md", code == 0
          and sorted(p.name for p in out.iterdir()) == sorted(RP.REPORT_FILES)
          if out.exists() else False, se[-300:])
    if code != 0:
        return fs, art
    doc = json.loads((out / "report.json").read_text(encoding="utf-8"))
    md = (out / "report.md").read_text(encoding="utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                          text=True).stdout.strip()
    fam = json.loads((art / "family.json").read_text(encoding="utf-8"))
    check("R2 report.json records the code commit, the contract sha256, the MDE sha256 and its Git "
          "state, and every input's four file hashes (P29)",
          doc["code_commit"] == head and doc["statistical_contract_sha256"]
          == fam["statistical_contract_sha256"] == A.contract_sha256()[0]
          and doc["mde_entry"]["sha256"] == hashlib.sha256(mde.read_bytes()).hexdigest()
          and doc["mde_entry"]["tracked"] is False and len(doc["inputs"]) == 37
          and all(len(v) == 4 for v in doc["inputs"].values()))
    lines = md.splitlines()
    seps = [i for i, ln in enumerate(lines) if ln.startswith("|---")]
    unbannered = []
    for i in seps:
        j = i - 2                                            # the line above the header row
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j < 0 or not lines[j].startswith("> **NONOFFICIAL**"):
            unbannered.append(lines[i - 1][:60])
    check("R3 the profile is the verified status (nonofficial) and the banner sits directly above "
          "every table", doc["profile"] == "nonofficial" and len(seps) >= 8 and not unbannered,
          unbannered)
    want = {s: fixture_miou_c(fs, s) for s in ("E1", "E6")}
    dm = doc["accuracy_on_corrupted_images"]["dataset_level_miou_c"]
    check("R4 P28: the dataset-level mIoU-C of each model is the nested mean of the 15 cells' "
          "pooled float64 all-class mIoU, and the contrast is E6 - E1 (exact)",
          dm["E1"] == want["E1"] and dm["E6"] == want["E6"]
          and dm["difference_e6_minus_e1"] == want["E6"] - want["E1"]
          and "union-present" in dm["class_space"], (dm, want))
    check("R5 the rejected alias appears nowhere in the report",
          "dataset_miou_c_delta" not in md and "dataset_miou_c_delta" not in json.dumps(doc))
    rob = [r for r in doc["family"] if r["comparison_id"] == "robustness_e1_e6"][0]
    others = [r for r in doc["family"] if r["comparison_id"] != "robustness_e1_e6"]
    obs = {t["task_id"]: t["observed"] for t in fam["bootstrap_tasks"]}
    check("R6 labels: the robustness contrast from the P28 difference, the others from their "
          "__dataset_miou_delta observed values; the given MDE entry is not committed, so the item "
          "3(f) caveat is pending on every non-rejection and the entry is recorded",
          rob["dataset_level_difference"] == dm["difference_e6_minus_e1"]
          and all(r["dataset_level_difference"] == obs[f"{r['comparison_id']}__dataset_miou_delta"]
                  for r in others)
          and doc["power_caveat"] == "pending"
          and all((RP.CAVEAT_PENDING in r["label"]) is (not r["holm_reject"])
                  for r in doc["family"])
          and doc["mde_entry"]["mde_w"] == 0.0125 and "MDE_W" not in md)
    check("R6b the family table renders W, z, r_rb, dz and the mean delta's BCa interval",
          "| W | z |" in md and "r_rb" in md and "| dz |" in md and "[BCa 95 %]" in md
          and all(r["mean_delta_bca"][0] is not None for r in doc["family"]))
    iv = doc["intervals"]
    check("R7 every interval reports z0 and an explicit containment flag (never assumed)",
          len(iv) == 35 and all("z0" in i and isinstance(i["estimate_inside_interval"], bool)
                                for i in iv) and "estimate inside" in md)
    real_inside = RP.estimate_inside
    RP.estimate_inside = lambda lo, up, est: False
    try:
        forced = RP.build_report(art, input_root=fs.input_root)
    finally:
        RP.estimate_inside = real_inside
    check("R7b build_report takes every containment flag from estimate_inside (never assumed)",
          all(i["estimate_inside_interval"] is False for i in forced["intervals"])
          and [i["estimate_inside_interval"] for i in iv]
          == [RP.estimate_inside(i["lower_bound"], i["upper_bound"], i["estimate"]) for i in iv])
    doc2 = copy.deepcopy(doc)
    rob2 = next(r for r in doc2["family"] if r["comparison_id"] == "robustness_e1_e6")
    rob2.update(label=RP.LABEL_INCONCLUSIVE, mde_w=0.0125)
    doc2["accuracy_on_corrupted_images"]["per_image_test"] = rob2
    corrupted = [ln for ln in RP.render_markdown(doc2).splitlines()
                 if ln.startswith("| per-image mIoU-C test")]
    check("R7c the 'accuracy on corrupted images' restatement states the MDE next to an inconclusive "
          "label", len(corrupted) == 1 and "(MDE_W = 0.0125)" in corrupted[0], corrupted)
    check("R8 non-inferiority shows the stored sensitivity flags",
          doc["noninferiority_e3_e6"]["sensitivity"] == fam["noninferiority_e3_e6"]["sensitivity"])
    check("R9 'accuracy on corrupted images', the descriptive E1 -> E3 block and the E6-KD "
          "observation that launches nothing are reported",
          "Accuracy on corrupted images" in md and "Descriptive E1 -> E3" in md
          and "launches nothing" in md and doc["e6_kd_observation"] == fam["e6_kd_trigger"])
    check("R10 k is reported by the report layer (never a family.json field)",
          doc["am5_excluded_count_k"] == 0 and "k (AM-5" in md
          and not any("am5" in k for k in fam))
    totals = {r["stage"]: r for r in doc["gt_present_am6"]}
    rv = rule_variants(np.load(fs.path("E3") / "sufficient_stats.npz", allow_pickle=False))
    check("R11 AM-6: the GT-present table for the seven clean stages equals rule_variants",
          len(totals) == 7 and totals["E3"]["gt_present_all_class"]
          == rv[GT_PRESENT]["all_class"].value_float64
          and totals["E3"]["union_present_all_class"]
          == rv[UNION_PRESENT]["all_class"].value_float64)
    check("R12 the three extension points are listed, not populated",
          [e["name"] for e in doc["extension_points"]] == [n for n, _ in RP.EXTENSION_POINTS]
          and len(RP.EXTENSION_POINTS) == 3
          and all(e["populated"] is False for e in doc["extension_points"]))
    return fs, art


def placeholder_tasks(obs, analysis_id, B):
    """Cheap tasks with real seeds and counts (constant replicates; the P3 fallback): the report reads
    counts and stored values, never replicate distributions."""
    out = []
    for spec in BS.FROZEN_TASK_MATRIX:
        v = D.observed_value(obs, spec)
        n = (obs.pooled[spec.comparison_id].n_images if spec.statistic_name == BS.DATASET_MIOU_DELTA
             else obs.paired[spec.comparison_id].n)
        reps, jack = np.full(B, v), np.full(n, v)
        out.append(BS.BootstrapTaskResult(
            spec, analysis_id, BS.derive_seed(analysis_id, spec.comparison_id, spec.statistic_name),
            v, reps, jack, BS.bca_interval(reps, jack, v, spec.interval_type)))
    return tuple(out)


def part_k(work: Path) -> None:
    """k > 0 needs the OFFICIAL row policy (smoke mode refuses AM-5 exclusions): a synthetic official
    inventory, n = 1561 and k = 3, written as a nonofficial artifact through the writer."""
    fs = F.build_inventory(work / "fxk", n=1561, k=3, split="test", status="official",
                           run_prefix="repk")
    entries = D.load_input_list(F.write_input_list(work / "repk.json", fs))
    inputs = D.load_inputs(entries, fs.input_root, Policy.OFFICIAL)
    obs = D.assemble_observed(inputs, Policy.OFFICIAL)
    inp = D.artifact_inputs(obs, inputs, placeholder_tasks(obs, BS.ANALYSIS_ID_OFFICIAL, 64),
                            analysis_id=BS.ANALYSIS_ID_OFFICIAL, run_id="repk-0001",
                            created_at_utc="2026-10-03T00:00:00Z",
                            contract_sha256=A.contract_sha256()[0], synthetic=True,
                            input_root=fs.input_root)
    art = A.write_statistics_artifact(work / "statsk" / "repk-0001", inp)
    out = work / "report_k"
    code, so, se = cli("--mode", "report", "--artifact", art, "--report-out", out, "--repo-root",
                       fs.input_root, "--synthetic-inputs")
    doc = json.loads((out / "report.json").read_text(encoding="utf-8")) if code == 0 else {}
    md = (out / "report.md").read_text(encoding="utf-8") if code == 0 else ""
    check("K1 k = 3 AM-5 images reported by the report layer (stdout, report.json, report.md)",
          code == 0 and doc["am5_excluded_count_k"] == 3 and "k (AM-5): 3" in so
          and "per-image analyses) = 3." in md, se[-300:])
    if code != 0:
        return
    rows = {g["stage"]: g for g in doc["gt_present_am6"]}
    rv = {st: rule_variants(np.load(fs.path(st) / "sufficient_stats.npz", allow_pickle=False))
          for st in D.STAGES}
    same = all(rows[st][f"{rule}_{space}"] == rv[st][rule][space].value_float64
               and rows[st][f"{rule}_{space}_n_classes"] == rv[st][rule][space].n_classes
               for st in D.STAGES for rule in (UNION_PRESENT, GT_PRESENT)
               for space in ("all_class", "disease_only"))
    check("K2 AM-6 in that report: all seven stages, both rules and both class spaces equal "
          "rule_variants of each stage's clean statistics", sorted(rows) == sorted(D.STAGES) and same)
    small = F.build_inventory(work / "fxk30", n=30, k=3, split="val", status="smoke",
                              run_prefix="repk30")
    stats = {st: np.load(small.path(st) / "sufficient_stats.npz", allow_pickle=False)
             for st in D.STAGES}
    table = {g["stage"]: g for g in RP.gt_present_table(stats, key="stage")}
    rv30 = {st: rule_variants(v) for st, v in stats.items()}
    equal = all(table[st][f"{rule}_{space}"] == rv30[st][rule][space].value_float64
                for st in D.STAGES for rule in (UNION_PRESENT, GT_PRESENT)
                for space in ("all_class", "disease_only"))
    check("K3 gt_present_table (the AM-6 table) on 30 images with k = 3: equal to rule_variants, and "
          "GT-present differs from union-present for some stage, so the columns cannot be swapped "
          "unnoticed", equal and any(table[st]["gt_present_all_class"]
                                     != table[st]["union_present_all_class"] for st in D.STAGES))


def part_n(work: Path, fs, art: Path) -> None:
    """Acceptance (g): an NI lower bound of exactly -0.020 (a literal) fails, the four sensitivity
    decisions come from that one stored bound, and a flag contradicting it is refused."""
    ni_task = "noninferiority_e3_e6__dataset_miou_delta"

    def crafted(tag, passed=None, flip=None):
        d = work / f"stats_ni_{tag}" / art.name
        shutil.copytree(art, d)
        f = json.loads((d / A.FAMILY_JSON).read_text(encoding="utf-8"))
        ni = f["noninferiority_e3_e6"]
        ni["lower_bound"] = -0.02
        ni["passed"] = -0.02 > -ni["primary_margin"] if passed is None else passed
        for i, sd in enumerate(ni["sensitivity"]):
            sd["passed"] = -0.02 > -sd["margin"]
            if flip == i:
                sd["passed"] = not sd["passed"]
        t = next(t for t in f["bootstrap_tasks"] if t["task_id"] == ni_task)
        t["lower_bound"] = -0.02
        (d / A.FAMILY_JSON).write_text(A.canonical_json(f), encoding="utf-8", newline="\n")
        with np.load(d / A.BOOTSTRAP_NPZ, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays[f"{ni_task}__bounds"] = np.array([-0.02], dtype=np.float64)
        np.savez_compressed(d / A.BOOTSTRAP_NPZ, **arrays)
        (d / A.MANIFEST_NAME).write_text(A._manifest_text(d), encoding="utf-8", newline="\n")
        return d

    def report(d, out):
        return cli("--mode", "report", "--artifact", d, "--report-out", out, "--repo-root",
                   fs.input_root, "--synthetic-inputs")
    out = work / "report_ni"
    code, so, se = report(crafted("ok"), out)
    doc = json.loads((out / "report.json").read_text(encoding="utf-8")) if code == 0 else {}
    ni = doc.get("noninferiority_e3_e6", {})
    check("N1 an NI lower bound of exactly -0.020 (literal) fails the primary margin; the four "
          "sensitivity decisions (1.0/1.5/2.0/2.5 pp) come from that one bound: F, F, F, T",
          code == 0 and ni.get("lower_bound") == -0.02 and ni.get("passed") is False
          and [x["passed"] for x in ni.get("sensitivity", [])] == [False, False, False, True]
          and [x["margin"] for x in ni.get("sensitivity", [])] == [0.01, 0.015, 0.02, 0.025],
          se[-300:])
    code, so, se = report(crafted("passed", passed=True), work / "report_ni_p")
    check("N2 the same bound stored with passed true -> refused (observed_source_mapping_verified)",
          code == 2 and "observed_source_mapping_verified" in se
          and not (work / "report_ni_p").exists(), se[-300:])
    code, so, se = report(crafted("flip", flip=2), work / "report_ni_f")
    check("N3 one sensitivity flag contradicting the one stored bound -> refused",
          code == 2 and "observed_source_mapping_verified" in se
          and not (work / "report_ni_f").exists(), se[-300:])


def part_d(work: Path, fs, art: Path) -> None:
    def report(out, *extra):
        return cli("--mode", "report", "--artifact", art, "--report-out", out, *extra)

    def refused(name, res, needle, out):
        code, so, se = res
        check(f"{name} -> exit 2 naming it; nothing written",
              code == 2 and needle in se and not Path(out).exists(),
              f"exit {code}: {se.strip()[-300:]}")
    refused("D1 --repo-root without --synthetic-inputs (P16)",
            report(work / "r_d1", "--repo-root", fs.input_root), "--synthetic-inputs",
            work / "r_d1")
    moved = work / "moved"
    shutil.copytree(fs.input_root, moved)
    rel = fs.rel("E5")
    F.edit_summary(moved / rel, lambda s: (s["dataset"].__setitem__(
        "preprocess_protocol", "letterbox_512/1.0.0"), s)[1])
    refused("D2 the canvas guard in report mode (an input relabelled non-canvas)",
            report(work / "r_d2", "--repo-root", moved, "--synthetic-inputs"), "canvas",
            work / "r_d2")
    with verifier_reads_official():
        res = in_process("--mode", "report", "--artifact", art, "--report-out", work / "r_d3",
                         "--repo-root", fs.input_root, "--synthetic-inputs")
    refused("D3 an artifact the verifier reads as official (relabelled in memory; nothing official "
            "on disk, P2): refused while the binding is unbound", res, "unbound", work / "r_d3")
    tam = work / "stats_tam" / "rep-0001"
    shutil.copytree(art, tam)
    with open(tam / A.FAMILY_JSON, "a", encoding="utf-8") as fh:
        fh.write(" ")
    refused("D4 a tampered artifact", cli("--mode", "report", "--artifact", tam, "--report-out",
                                          work / "r_d4", "--repo-root", fs.input_root,
                                          "--synthetic-inputs"), "manifest_verified",
            work / "r_d4")
    (work / "r_d5").mkdir()
    code, so, se = report(work / "r_d5", "--repo-root", fs.input_root, "--synthetic-inputs")
    check("D5 an existing --report-out -> exit 2", code == 2 and "overwrite" in se, se[-200:])
    pfs = F.build_inventory(work / "pfx", n=30, k=0, split="val", status="smoke",
                            run_prefix="pl", dataset_name="PlantSeg")
    plst = F.write_input_list(work / "pl.json", pfs)
    code, so, se = cli("--mode", "smoke", "--inputs", plst, "--out-dir", work / "pstats",
                       "--run-id", "pl-0001", "--repo-root", pfs.input_root, "--B", 30)
    refused("D6 --synthetic-inputs declared for an artifact without synthetic_input_data",
            cli("--mode", "report", "--artifact", work / "pstats" / "pl-0001", "--report-out",
                work / "r_d6", "--repo-root", pfs.input_root, "--synthetic-inputs"),
            "synthetic_input_data", work / "r_d6")
    refused("D7 an official-mode flag in report mode",
            report(work / "r_d7", "--confirm-official-test-analysis"), "official mode",
            work / "r_d7")
    refused("D10 --report-out inside the artifact directory (its exact file set would break)",
            report(art / "rendered", "--repo-root", fs.input_root, "--synthetic-inputs"),
            "is inside", art / "rendered")
    check("D10b ... the artifact still verifies after that refusal",
          outcome_ok(lambda: A.verify_statistics_artifact(art, input_root=fs.input_root)))
    dangling = work / "r_d11"
    dangling.symlink_to(work / "no_such_target")
    code, so, se = report(dangling)
    check("D11 a dangling symlink as --report-out -> exit 2 (never exit 4); nothing written",
          code == 2 and "overwrite" in se and dangling.is_symlink()
          and not (work / "no_such_target").exists(), se[-300:])
    refused("D8 --B 10000 (its official-mode default) in report mode is foreign, never ignored",
            report(work / "r_d8", "--B", "10000"), "does not accept --B", work / "r_d8")
    refused("D9 --jobs 1 in report mode is foreign, never ignored",
            report(work / "r_d9", "--jobs", "1"), "does not accept --jobs", work / "r_d9")
    block = work / "moved_block"
    shutil.copytree(fs.input_root, block)
    rel2 = fs.rel("E2")
    F.edit_summary(block / rel2, lambda s: dict(s, protocol={"name": "upstream"}))
    refused("D2b the canvas guard in report mode, half 2 (an input carrying a summary.protocol "
            "block)", report(work / "r_d2b", "--repo-root", block, "--synthetic-inputs"), rel2,
            work / "r_d2b")


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="smoke_stats_report_"))
    try:
        part_g()
        fs, art = part_r(work)
        part_k(work)
        part_n(work, fs, art)
        part_d(work, fs, art)
    except Exception:                                                    # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=3))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check("Z1 temporary fixtures and outputs removed", not work.exists())

    print()
    for name, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if det and not ok:
            print(f"         {det[:400]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'STATS REPORT OK' if allok else 'STATS REPORT FAILED'} ({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic inputs; every report rendered here is nonofficial.")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
