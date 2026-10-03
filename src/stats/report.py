"""Statistics report layer (lane L-STATS-OFFICIAL): the AM-17 labels and the MDE entry.

PREREGISTRATION_AMENDMENTS AM-17 item 2 (smallest effect of interest) and item 3(f) (power caveat).
A Holm superiority contrast is labelled from its finalized Holm decision (`holm.members[].reject`) and
the observed dataset-level difference (`<comparison>__dataset_miou_delta`, or the dataset-level
mIoU-C difference for the robustness contrast), compared as stored fractions, never in percentage
points (P27):

  rejected and difference >= 0.010   "sizable"
  rejected, smaller difference       "detected, below the smallest effect of interest"
  not rejected                       "not rejected", or, when the MDE entry's power_caveat is true
                                     (MDE_W > tau_P = 0.010), "inconclusive at the smallest effect of
                                     interest" with the MDE stated next to it

The MDE entry (docs/lane_specs/part1.md lane 6: reports/derived/mde_entry_<UTC>.json) is validated
only for mde_w, tau_p = 0.010 and power_caveat == (mde_w > tau_p) (orchestrator answer Q9).

Import-time behaviour is side-effect free; Git runs only inside `mde_entry_status`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

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


class ReportError(RuntimeError):
    """A report input the report layer refuses, by name."""


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


__all__ = ["SESOI", "TAU_P", "LABEL_SIZABLE", "LABEL_BELOW_SESOI", "LABEL_NOT_REJECTED",
           "LABEL_INCONCLUSIVE", "LABELS", "ReportError", "contrast_label", "validate_mde_entry",
           "mde_entry_status", "load_mde_entry"]
