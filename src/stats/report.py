"""Statistics report layer (lane L-STATS-OFFICIAL): report mode, the AM-17 labels and the MDE entry.

Report mode renders a section 12 statistics artifact into report.md and report.json. It always runs
the full verifier with the 37 inputs and the contract bytes (all thirteen checks, the canvas guard of
EVALUATION_CONTRACT section 11(a) on every input); the profile comes from the verified status, never
from a flag, and while the TEST-manifest binding is unbound the verifier refuses every `official`
artifact, so every report is nonofficial and every table carries a status banner (P26, P29).

Sections (P26): the family table with the t-test column and each contrast's AM-17 item 2 label; the
bootstrap intervals with z0 and an explicit containment column (an interval is never assumed to
contain its estimate); non-inferiority with the stored sensitivity flags; "accuracy on corrupted
images"; descriptive E1 -> E3; the E6-KD observation, which launches nothing; k; the AM-6 GT-present
table; the three extension points (AM-8 seeds 43/44, the rare-class strata TEST tables, the AM-17
item 6 default candidates).

Labels (AM-17 item 2 and item 3(f); P27): a Holm superiority contrast is labelled from its finalized
Holm decision (`holm.members[].reject`, never the library's) and its dataset-level difference -- the
`<comparison>__dataset_miou_delta` observed value, or for the robustness contrast the dataset-level
mIoU-C difference of P28 -- compared as stored fractions, never in percentage points:

  rejected and difference >= 0.010   "sizable"
  rejected, smaller difference       "detected, below the smallest effect of interest"
  not rejected                       "not rejected", or, when the MDE entry's power_caveat is true
                                     (MDE_W > tau_P = 0.010), "inconclusive at the smallest effect of
                                     interest" with the MDE stated next to it

P28 (orchestrator Q7): the dataset-level mIoU-C of a model is the mean over the five corruptions of
the mean over severities 1-3 of each cell's pooled all-class union-present mIoU
(`pooled_miou_from_totals` on the cell's dataset totals, float64); the contrast is E6 - E1. It is a
descriptive point value of the report layer only: never in family.json, never bootstrapped (D22), and
never named after the rejected section 8.7.1 alias.

The MDE entry (docs/lane_specs/part1.md lane 6: reports/derived/mde_entry_<UTC>.json) is validated
only for mde_w, tau_p = 0.010 and power_caveat == (mde_w > tau_p) (orchestrator answer Q9). An
official report requires exactly one committed entry; a nonofficial report records its state.

Import-time behaviour is side-effect free; Git runs only inside the functions that name it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

#: AM-17 item 2: 1.0 pp, as the stored fraction it is compared with.
SESOI = 0.010
#: AM-17 item 3(f): the per-image planning threshold, as a stored fraction.
TAU_P = 0.010
LABEL_SIZABLE = "sizable"
LABEL_BELOW_SESOI = "detected, below the smallest effect of interest"
LABEL_NOT_REJECTED = "not rejected"
LABEL_INCONCLUSIVE = "inconclusive at the smallest effect of interest"
LABELS = (LABEL_SIZABLE, LABEL_BELOW_SESOI, LABEL_NOT_REJECTED, LABEL_INCONCLUSIVE)
CODE_REPO = Path(__file__).resolve().parents[2]
MDE_GLOB = "reports/derived/mde_entry_*.json"
REPORT_SCHEMA = "plantseg-stats-report/1.0.0"
REPORT_FILES = ("report.json", "report.md")
CLASS_SPACE = "all-class, union-present (116 classes: background 0 and diseases 1..115)"
EXTENSION_POINTS = (
    ("AM-8 seed stability", "seeds 43 and 44 reported per seed as descriptive tables of the eight "
                            "contrasts (AM-8; AM-17 item 4)"),
    ("Rare-class strata TEST tables", "per-class IoU changes by TRAIN pixel-share stratum "
                                      "(AM-17 item 8)"),
    ("Default candidates", "E2 with lambda = 1 and E3 with alpha = 50 scored descriptively at TEST "
                           "(AM-17 item 6)"),
)


class ReportError(RuntimeError):
    """A report request or input the report layer refuses, by name."""


# --------------------------------------------------------------------------------------------------
# 1. labels and the MDE entry
# --------------------------------------------------------------------------------------------------
def contrast_label(reject, difference, *, power_caveat=None) -> str:
    """AM-17 item 2 and item 3(f). `reject` is the finalized Holm decision; `difference` the stored
    dataset-level difference (candidate - baseline) as a fraction; `power_caveat` the MDE entry's
    flag (None or False: no caveat)."""
    if type(reject) is not bool:
        raise ReportError(f"reject must be a bool, got {reject!r}")
    if type(difference) is not float or not math.isfinite(difference):
        raise ReportError(f"the dataset-level difference must be a finite float, got {difference!r}")
    if reject:
        return LABEL_SIZABLE if difference >= SESOI else LABEL_BELOW_SESOI
    return LABEL_INCONCLUSIVE if power_caveat is True else LABEL_NOT_REJECTED


def validate_mde_entry(doc) -> tuple[str, ...]:
    """The problems of an MDE entry: mde_w a finite non-negative float, tau_p exactly 0.010 and
    power_caveat a bool equal to (mde_w > tau_p). Nothing else is validated (Q9)."""
    if not isinstance(doc, dict):
        return ("the MDE entry is not a JSON object",)
    problems = []
    mde_w, tau_p, caveat = doc.get("mde_w"), doc.get("tau_p"), doc.get("power_caveat")
    if type(mde_w) is not float or not math.isfinite(mde_w) or mde_w < 0.0:
        problems.append(f"mde_w {mde_w!r} is not a finite non-negative float")
    if type(tau_p) is not float or tau_p != TAU_P:
        problems.append(f"tau_p {tau_p!r} is not {TAU_P}")
    if type(caveat) is not bool:
        problems.append(f"power_caveat {caveat!r} is not a bool")
    elif not problems and caveat is not (mde_w > tau_p):
        problems.append(f"power_caveat {caveat} != (mde_w {mde_w} > tau_p {tau_p})")
    return tuple(problems)


def _git(repo: Path, *args):
    try:
        p = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True,
                           timeout=60, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return p


def mde_entry_status(path, repo: Path = CODE_REPO) -> dict:
    """The MDE file's sha256 and its Git state in the code repository: inside it, tracked, unmodified
    at HEAD, the commit that last changed it and HEAD (None where Git cannot tell)."""
    p = Path(path).resolve()
    info = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "inside_repository": p.is_relative_to(Path(repo).resolve()), "tracked": None,
            "clean_at_head": None, "last_commit": None, "head": None}
    if not info["inside_repository"]:
        info["tracked"] = info["clean_at_head"] = False
        return info
    rel = p.relative_to(Path(repo).resolve()).as_posix()
    head = _git(repo, "rev-parse", "HEAD")
    tracked = _git(repo, "ls-files", "--error-unmatch", "--", rel)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "--", rel)
    last = _git(repo, "log", "-1", "--format=%H", "--", rel)
    if None in (head, tracked, status, last) or head.returncode != 0 or status.returncode != 0:
        return info
    info.update(head=head.stdout.strip(), tracked=tracked.returncode == 0,
                clean_at_head=status.stdout.strip() == "",
                last_commit=(last.stdout.strip() or None) if last.returncode == 0 else None)
    return info


def load_mde_entry(path, repo: Path = CODE_REPO) -> tuple[dict, dict, tuple[str, ...]]:
    """(entry, git status, validation problems). Refuses only an unreadable file."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ReportError(f"MDE entry {path}: cannot be read as JSON ({e})") from e
    return doc, mde_entry_status(path, repo), validate_mde_entry(doc)


def tracked_mde_entries(repo: Path = CODE_REPO):
    """The MDE entry files tracked in the code repository (None when Git cannot tell)."""
    p = _git(repo, "ls-files", "--", MDE_GLOB)
    if p is None or p.returncode != 0:
        return None
    return sorted(x for x in p.stdout.splitlines() if x)


def require_official_mde(doc, status, problems, tracked) -> None:
    """P29: an official report needs exactly one committed MDE entry, valid on mde_w, tau_p and
    power_caveat. Pure: every input is passed in, so the refusal is testable on in-memory data."""
    if doc is None or status is None:
        raise ReportError("an official report requires --mde-entry (exactly one committed MDE entry)")
    if tracked is None:
        raise ReportError("the MDE entry cannot be shown to be committed: Git is unavailable")
    if len(tracked) != 1:
        raise ReportError(f"an official report requires exactly one committed MDE entry; "
                          f"{len(tracked)} are tracked: {tracked}")
    if not (status.get("tracked") and status.get("clean_at_head")):
        raise ReportError(f"MDE entry {status.get('path')} is not committed and unmodified at HEAD")
    if not str(status.get("path", "")).replace("\\", "/").endswith(tracked[0]):
        raise ReportError(f"MDE entry {status.get('path')} is not the committed entry {tracked[0]}")
    if problems:
        raise ReportError(f"MDE entry {status.get('path')}: {list(problems)}")


# --------------------------------------------------------------------------------------------------
# 2. descriptive values computed from the inputs (P28, AM-6)
# --------------------------------------------------------------------------------------------------
def dataset_miou_c(views, stage: str) -> float:
    """P28: mean over the five corruptions of the mean over severities 1-3 of each cell's pooled
    all-class union-present mIoU (float64), for one stage."""
    from .corruption_protocol import official_corruption_grid
    from .noninferiority import pooled_miou_from_totals
    from .robustness import INFERENTIAL_SEVERITIES
    grid = official_corruption_grid()
    per_corruption = []
    for name in grid.names:
        cells = []
        for sev in INFERENTIAL_SEVERITIES:
            st = views.loaded[(stage, ("corruption", name, sev))].run.stats
            cells.append(pooled_miou_from_totals(st.dataset_tp, st.dataset_gt, st.dataset_pred))
        per_corruption.append(float(np.mean(np.array(cells, dtype=np.float64))))
    return float(np.mean(np.array(per_corruption, dtype=np.float64)))


def dataset_miou_c_contrast(views) -> dict:
    e1, e6 = dataset_miou_c(views, "E1"), dataset_miou_c(views, "E6")
    return {"E1": e1, "E6": e6, "difference_e6_minus_e1": e6 - e1, "class_space": CLASS_SPACE,
            "cells_per_model": 15}


def gt_present_rows(views) -> list[dict]:
    """AM-6 (AM-17 item 1(b)), descriptive: each clean stage's dataset-level mIoU under the
    union-present headline rule and the GT-present rule (exact float64 means)."""
    from .driver import CLEAN, STAGES
    from .eligibility import GT_PRESENT, UNION_PRESENT, rule_variants
    rows = []
    for stage in STAGES:
        rv = rule_variants(views.loaded[(stage, CLEAN)].run.stats)
        row = {"stage": stage}
        for rule in (UNION_PRESENT, GT_PRESENT):
            for space in ("all_class", "disease_only"):
                d = rv[rule][space]
                row[f"{rule}_{space}"] = d.value_float64
                row[f"{rule}_{space}_n_classes"] = d.n_classes
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------
# 3. report mode
# --------------------------------------------------------------------------------------------------
def _fmt(v, digits=6) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def family_rows(fam: dict, robustness_difference: float, *, power_caveat=None, mde_w=None
                ) -> list[dict]:
    """The family table: one row per Holm contrast, labelled from the finalized Holm decision
    (`holm.members[].reject`, never `library_reject`) and the dataset-level difference: the
    `__dataset_miou_delta` observed value, or `robustness_difference` (P28) for the robustness
    contrast (P27). Pure: reads only `fam`."""
    from .bootstrap import DATASET_MIOU_DELTA
    from .tests import CANONICAL_COMPARISON_IDS
    holm = {m["comparison_id"]: m for m in fam["a3a_family"]["holm"]["members"]}
    obs = {t["task_id"]: t["observed"] for t in fam["bootstrap_tasks"]}
    rows = []
    for c in fam["a3a_family"]["comparisons"]:
        cid = c["comparison_id"]
        diff = (robustness_difference if cid == CANONICAL_COMPARISON_IDS[7]
                else obs[f"{cid}__{DATASET_MIOU_DELTA}"])
        label = contrast_label(holm[cid]["reject"], diff, power_caveat=power_caveat)
        rows.append({"comparison_id": cid, "baseline_stage": c["baseline_stage"],
                     "candidate_stage": c["candidate_stage"], "metric": c["metric"],
                     "n_paired": c["n_paired"], "wilcoxon_p": c["wilcoxon"]["p_value"],
                     "wilcoxon_status": c["wilcoxon"]["status"],
                     "holm_p_adjusted": holm[cid]["p_adjusted"], "holm_reject": holm[cid]["reject"],
                     "library_reject": holm[cid]["library_reject"],
                     "mean_delta": c["shifts"]["mean_delta"],
                     "median_delta": c["shifts"]["median_delta"],
                     "hodges_lehmann_shift": c["hodges_lehmann_shift"],
                     "t_statistic": c["t_test"]["statistic"], "t_p_value": c["t_test"]["p_value"],
                     "dataset_level_difference": diff, "label": label,
                     "mde_w": mde_w if label == LABEL_INCONCLUSIVE else None})
    return rows


def build_report(artifact_dir, *, input_root, mde_entry=None, synthetic_declared: bool = False
                 ) -> dict:
    """Verify the artifact fully, then assemble report.json's content. Refuses by name."""
    from . import artifact as A
    from .bootstrap import DESCRIPTIVE_E1_E3
    from .tests import CANONICAL_COMPARISON_IDS
    try:
        rep = A.verify_statistics_artifact(artifact_dir, input_root=input_root,
                                           contract_bytes=A.contract_sha256()[1])
    except A.ArtifactError as e:
        raise ReportError(f"the artifact does not verify: {e}") from e
    if rep.not_reestablished or rep.views is None:
        raise ReportError(f"checks {rep.not_reestablished} were not re-established")
    fam, views = rep.family, rep.views
    profile = fam["artifact_status"]
    if synthetic_declared and A.W_SYNTHETIC not in fam["warnings"]:
        raise ReportError("--synthetic-inputs was declared but the artifact does not carry "
                          "synthetic_input_data")
    mde = None
    doc = status = problems = None
    if mde_entry is not None:
        doc, status, problems = load_mde_entry(mde_entry)
        mde = dict(status, mde_w=doc.get("mde_w") if isinstance(doc, dict) else None,
                   tau_p=doc.get("tau_p") if isinstance(doc, dict) else None,
                   power_caveat=doc.get("power_caveat") if isinstance(doc, dict) else None,
                   problems=list(problems))
    if profile == A.STATUS_OFFICIAL:                         # unreachable while the binding is unbound
        require_official_mde(doc, status, problems, tracked_mde_entries())
    caveat = mde["power_caveat"] if mde and not mde["problems"] else None
    robust = dataset_miou_c_contrast(views)
    rows = family_rows(fam, robust["difference_e6_minus_e1"], power_caveat=caveat,
                       mde_w=mde["mde_w"] if mde else None)
    intervals = []
    for t in fam["bootstrap_tasks"]:
        lo, up, est = t["lower_bound"], t.get("upper_bound"), t["observed"]
        inside = (lo <= est) if up is None else (lo <= est <= up)
        intervals.append({"task_id": t["task_id"], "estimate": est, "lower_bound": lo,
                          "upper_bound": up, "z0": t["z0"], "interval_method": t["interval_method"],
                          "estimate_inside_interval": inside, "warnings": t["warnings"]})
    head = _git(CODE_REPO, "rev-parse", "HEAD")
    manifest_lines = (Path(artifact_dir) / A.MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    return {
        "schema": REPORT_SCHEMA, "profile": profile, "artifact_status": profile,
        "warnings": fam["warnings"], "run_id": fam["run_id"], "analysis_id": fam["analysis_id"],
        "artifact": {"path": str(Path(artifact_dir).resolve()), "manifest": manifest_lines},
        "code_commit": head.stdout.strip() if head is not None and head.returncode == 0 else None,
        "statistical_contract_sha256": fam["statistical_contract_sha256"],
        "integrity_checks_established": list(rep.established),
        "mde_entry": mde,
        "inputs": {li.rel_path: dict(li.file_sha256) for li in views.loaded.values()},
        "am5_excluded_count_k": views.am5_excluded_count,
        "family": rows, "intervals": intervals,
        "noninferiority_e3_e6": fam["noninferiority_e3_e6"],
        "accuracy_on_corrupted_images": {
            "per_image_test": next(r for r in rows if r["comparison_id"]
                                   == CANONICAL_COMPARISON_IDS[7]),
            "dataset_level_miou_c": robust},
        "descriptive_e1_e3": fam["descriptive_e1_e3"],
        "descriptive_e1_e3_intervals": [i for i in intervals
                                        if i["task_id"].startswith(DESCRIPTIVE_E1_E3)],
        "e6_kd_observation": fam["e6_kd_trigger"],
        "gt_present_am6": gt_present_rows(views),
        "extension_points": [{"name": n, "content": d, "populated": False}
                             for n, d in EXTENSION_POINTS],
    }


def banner(doc: dict) -> str:
    if doc["profile"] == "official":
        return ""
    return (f"> **NONOFFICIAL** -- statistics artifact `{doc['run_id']}` has status "
            f"`{doc['artifact_status']}` (warnings: {', '.join(doc['warnings'])}). Not a thesis "
            "result.\n")


def render_markdown(doc: dict) -> str:
    b = banner(doc)
    out = [f"# Statistics report -- {doc['run_id']}", "", b,
           f"Analysis `{doc['analysis_id']}`; contract sha256 `{doc['statistical_contract_sha256']}`;"
           f" code commit `{doc['code_commit']}`; all 13 integrity checks re-established. k (AM-5 "
           f"zero-disease images excluded from per-image analyses) = {doc['am5_excluded_count_k']}.",
           "", "## Holm family (eight superiority contrasts)", "", b,
           "| contrast | n | Wilcoxon p | Holm p | reject | mean delta | median delta | HL shift | "
           "t (p) | dataset-level delta | label |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["family"]:
        label = r["label"] + (f" (MDE_W = {r['mde_w']})" if r["mde_w"] is not None else "")
        out.append(f"| {r['comparison_id']} ({r['baseline_stage']} -> {r['candidate_stage']}) | "
                   f"{r['n_paired']} | {_fmt(r['wilcoxon_p'])} | {_fmt(r['holm_p_adjusted'])} | "
                   f"{r['holm_reject']} | {_fmt(r['mean_delta'])} | {_fmt(r['median_delta'])} | "
                   f"{_fmt(r['hodges_lehmann_shift'])} | {_fmt(r['t_statistic'])} "
                   f"({_fmt(r['t_p_value'])}) | {_fmt(r['dataset_level_difference'])} | {label} |")
    out += ["", "Labels: AM-17 item 2 (Holm decision and the dataset-level difference against 0.010, "
                "compared as stored fractions); item 3(f) when the MDE entry's power caveat holds.",
            "", "## Bootstrap intervals (95 % BCa or the percentile fallback; z0 per interval)", "",
            b, "| task | estimate | lower | upper | z0 | method | estimate inside |",
            "|---|---|---|---|---|---|---|"]
    for i in doc["intervals"]:
        out.append(f"| {i['task_id']} | {_fmt(i['estimate'])} | {_fmt(i['lower_bound'])} | "
                   f"{_fmt(i['upper_bound'])} | {_fmt(i['z0'], 4)} | {i['interval_method']} | "
                   f"{i['estimate_inside_interval']} |")
    ni = doc["noninferiority_e3_e6"]
    out += ["", "## Non-inferiority E3 -> E6 (stored decisions)", "", b,
            "| baseline mIoU | candidate mIoU | delta | drop | retention | lower bound | margin | "
            "passed |", "|---|---|---|---|---|---|---|---|",
            f"| {_fmt(ni['baseline_dataset_miou'])} | {_fmt(ni['candidate_dataset_miou'])} | "
            f"{_fmt(ni['observed_delta'])} | {_fmt(ni['observed_drop'])} | "
            f"{_fmt(ni['relative_retention'])} | {_fmt(ni['lower_bound'])} | "
            f"{_fmt(ni['primary_margin'])} | {ni['passed']} |", "", b,
            "| sensitivity margin | passed (stored) |", "|---|---|"]
    out += [f"| {_fmt(s['margin'])} | {s['passed']} |" for s in ni["sensitivity"]]
    ac = doc["accuracy_on_corrupted_images"]
    r, d = ac["per_image_test"], ac["dataset_level_miou_c"]
    out += ["", "## Accuracy on corrupted images (E1 -> E6, mIoU-C)", "", b,
            "| quantity | E1 | E6 | difference |", "|---|---|---|---|",
            f"| per-image mIoU-C test: mean delta (Holm p {_fmt(r['holm_p_adjusted'])}, label "
            f"{r['label']}) | | | {_fmt(r['mean_delta'])} |",
            f"| dataset-level mIoU-C ({d['class_space']}; 15 cells per model, nested mean) | "
            f"{_fmt(d['E1'])} | {_fmt(d['E6'])} | {_fmt(d['difference_e6_minus_e1'])} |"]
    de = doc["descriptive_e1_e3"]
    out += ["", "## Descriptive E1 -> E3 (outside the Holm family)", "", b,
            "| n | mean delta | mean delta (pp) | median delta | HL shift |", "|---|---|---|---|---|",
            f"| {de['n_paired']} | {_fmt(de['mean_delta'])} | {_fmt(de['mean_delta_pp'])} | "
            f"{_fmt(de['median_delta'])} | {_fmt(de['hodges_lehmann_shift'])} |"]
    e6 = doc["e6_kd_observation"]
    out += ["", "## E6-KD observation (descriptive; launches nothing)", "", b,
            "| observed drop E3 - E6 | threshold | rule | triggered (stored) |", "|---|---|---|---|",
            f"| {_fmt(e6['observed_drop'])} | {_fmt(e6['trigger_threshold'])} | "
            f"{e6['decision_rule']} | {e6['triggered']} |",
            "", "## AM-6: dataset-level mIoU under the GT-present rule (descriptive)", "", b,
            "| stage | union-present all-class | GT-present all-class | union-present disease-only "
            "| GT-present disease-only |", "|---|---|---|---|---|"]
    for g in doc["gt_present_am6"]:
        out.append(f"| {g['stage']} | {_fmt(g['union_present_all_class'])} | "
                   f"{_fmt(g['gt_present_all_class'])} | {_fmt(g['union_present_disease_only'])} | "
                   f"{_fmt(g['gt_present_disease_only'])} |")
    out += ["", "## Extension points (not populated by this report)", "", b]
    out += [f"- {e['name']}: {e['content']}" for e in doc["extension_points"]]
    m = doc["mde_entry"]
    out += ["", "## MDE entry", "",
            (f"`{m['path']}` sha256 `{m['sha256']}`, tracked {m['tracked']}, unmodified at HEAD "
             f"{m['clean_at_head']}, last commit `{m['last_commit']}`; mde_w {m['mde_w']}, tau_p "
             f"{m['tau_p']}, power_caveat {m['power_caveat']}; problems {m['problems']}"
             if m else "none given"), ""]
    return "\n".join(out)


def write_report(report_out, doc: dict) -> Path:
    """Atomic: a temporary sibling with report.json and report.md, renamed to --report-out."""
    from .artifact import canonical_json
    final = Path(report_out)
    if final.exists():
        raise ReportError(f"refusing to overwrite {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir()
    try:
        (tmp / "report.json").write_text(canonical_json(doc), encoding="utf-8", newline="\n")
        (tmp / "report.md").write_text(render_markdown(doc), encoding="utf-8", newline="\n")
        tmp.rename(final)
        return final
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def run_report(*, artifact, report_out, repo_root=None, synthetic: bool = False, mde_entry=None):
    """--mode report. --repo-root (accepted only with --synthetic-inputs; P16) resolves the inputs
    the artifact's records name; the contract comes from this code's repository."""
    if repo_root is not None and not synthetic:
        raise ReportError("--repo-root is accepted only with --synthetic-inputs (P16)")
    if artifact is None or report_out is None:
        raise ReportError("report mode requires --artifact and --report-out")
    if Path(report_out).exists():
        raise ReportError(f"refusing to overwrite {report_out}")
    from .artifact import CODE_REPO as ROOT
    doc = build_report(artifact, input_root=Path(repo_root) if repo_root is not None else ROOT,
                       mde_entry=mde_entry, synthetic_declared=synthetic)
    return write_report(report_out, doc), doc


__all__ = ["SESOI", "TAU_P", "LABEL_SIZABLE", "LABEL_BELOW_SESOI", "LABEL_NOT_REJECTED",
           "LABEL_INCONCLUSIVE", "LABELS", "MDE_GLOB", "REPORT_SCHEMA", "REPORT_FILES",
           "CLASS_SPACE", "EXTENSION_POINTS", "ReportError", "contrast_label",
           "validate_mde_entry", "mde_entry_status", "load_mde_entry", "tracked_mde_entries",
           "require_official_mde", "family_rows", "dataset_miou_c", "dataset_miou_c_contrast",
           "gt_present_rows", "build_report", "banner", "render_markdown", "write_report",
           "run_report"]
