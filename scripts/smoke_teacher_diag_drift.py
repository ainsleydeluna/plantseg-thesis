#!/usr/bin/env python3
"""Smoke: drift pins of lane L-TEACHER-DIAG (P34, P1, P6, P7, R1).

  P34  sha256 of inspect.getsource(scripts.evaluate_model.run) equals the value fixed in this lane, so a
       later change to the function D2 mirrors (scripts/teacher_d2_calibration.py) fails here first. The
       value was measured at afd2d33 (b3313391...) and re-pinned by the P41 follow-up at 60c1417, after
       K-part's two run() hunks (expected_sha256 to validate_teacher_artifact and load_teacher_model).
  P6   src/eval/teacher_diag.py calls load_frozen_teacher and load_teacher_model exactly once each (the two
       load lines), and no DIAG script calls either.
  P7   frozen.provenance.as_dict() is called at exactly one site (teacher_record), and no DIAG script
       calls it.
  P1   the teacher-loading scripts (D1, D2, TRAIN) and their modules import nothing from src.stats, in a
       fresh interpreter.
  R1   every CODE_FILES path exists, so every output of one pin carries one code digest.

    python -B scripts/smoke_teacher_diag_drift.py
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS: list[tuple[str, bool, str]] = []
EVALUATE_MODEL_RUN_SHA256 = "d1c809d4ad588d6efffd7fd98a1240fa22ade18d729ec6f187963125de97de2a"   # at 60c1417
DIAG_SCRIPTS = ("scripts/teacher_d1_nmf_sensitivity.py", "scripts/teacher_d2_calibration.py",
                "scripts/score_teacher_train.py", "scripts/teacher_d3_perclass_strata.py",
                "scripts/hash_split_files.py", "scripts/dedup_val_scores.py")
TEACHER_LOADING = ("scripts.teacher_d1_nmf_sensitivity", "scripts.teacher_d2_calibration", "scripts.score_teacher_train",
                   "src.eval.teacher_diag", "src.eval.calibration", "src.eval.nmf_sensitivity")


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def calls(path: str) -> dict:
    """Counts of the load calls and of provenance.as_dict() in one file (AST, not text)."""
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    out = {"load_frozen_teacher": 0, "load_teacher_model": 0, "provenance.as_dict": 0}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        if name in ("load_frozen_teacher", "load_teacher_model"):
            out[name] += 1
        if (isinstance(f, ast.Attribute) and f.attr == "as_dict" and isinstance(f.value, ast.Attribute)
                and f.value.attr == "provenance"):
            out["provenance.as_dict"] += 1
    return out


def imported_repo_modules() -> list:
    """Repository .py files imported (directly, anywhere in the file) by the DIAG scripts and modules."""
    out = set()
    for path in (*DIAG_SCRIPTS, "src/eval/teacher_diag.py", "src/eval/calibration.py", "src/eval/nmf_sensitivity.py"):
        tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            for name in names:
                if name.split(".")[0] in ("src", "scripts", "configs"):
                    rel = name.replace(".", "/") + ".py"
                    if (REPO / rel).is_file():
                        out.add(rel)
    return sorted(out)


def main() -> int:
    import scripts.evaluate_model as em
    from src.eval import teacher_diag as td
    got = hashlib.sha256(inspect.getsource(em.run).encode("utf-8")).hexdigest()
    check("P34 sha256(inspect.getsource(scripts.evaluate_model.run)) equals the lane's pin", got == EVALUATE_MODEL_RUN_SHA256,
          f"got {got}")
    seam = calls("src/eval/teacher_diag.py")
    check("P6 the seam has exactly two load lines: load_frozen_teacher once, load_teacher_model once",
          seam["load_frozen_teacher"] == 1 and seam["load_teacher_model"] == 1, str(seam))
    check("P7 frozen.provenance.as_dict() is called at exactly one site", seam["provenance.as_dict"] == 1, str(seam))
    script_calls = {p: calls(p) for p in DIAG_SCRIPTS}
    check("P6/P7 no DIAG script loads the teacher or reads its provenance itself",
          all(sum(c.values()) == 0 for c in script_calls.values()), str({p: c for p, c in script_calls.items() if sum(c.values())}))
    probe = ("import sys\n" + "".join(f"import {m}\n" for m in TEACHER_LOADING)
             + "print(sorted(m for m in sys.modules if m == 'src.stats' or m.startswith('src.stats.')))")
    p = subprocess.run([sys.executable, "-B", "-c", probe], cwd=str(REPO), capture_output=True, text=True, timeout=300)
    check("P1 the teacher-loading scripts and modules import nothing from src.stats (fresh interpreter)",
          p.returncode == 0 and p.stdout.strip() == "[]", (p.stdout + p.stderr)[-300:])
    missing = [r for r in td.CODE_FILES if not (td.REPO / r).is_file()]
    check("R1 every CODE_FILES path exists (one code digest per pin)", not missing, str(missing))
    uncovered = sorted(set(imported_repo_modules()) - set(td.CODE_FILES))
    check("R1 every repository module a DIAG script or module imports (AST, direct imports) is in CODE_FILES",
          not uncovered, str(uncovered))
    check("R1 CODE_FILES covers every DIAG script and module",
          all(p in td.CODE_FILES for p in DIAG_SCRIPTS) and all(
              p in td.CODE_FILES for p in ("src/eval/teacher_diag.py", "src/eval/calibration.py",
                                           "src/eval/nmf_sensitivity.py")))
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    good = sum(ok for _, ok, _ in RESULTS)
    print(f"RESULT: {'PASS' if good == len(RESULTS) else 'FAIL'} ({good}/{len(RESULTS)})")
    return 0 if good == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
