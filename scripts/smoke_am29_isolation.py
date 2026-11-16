#!/usr/bin/env python3
"""L-AM29 isolation smoke: AM-28's code is untouched and AM-29's registration reaches its own processes only.

  code      every file of src/quant/qat.py QAT_CODE_PATHS: its sha256 in the checkout equals its sha256 at the base commit
            (git show <base>:<path>), both printed
  tree      in src/, scripts/ and configs/ (test-named paths excluded), the branch adds files and changes none:
            git diff --name-status <base>..HEAD plus the untracked files, every entry an addition
  registry  a fresh process holds the pinned tables as their files define them (no AM-29 key; the pinned tools refuse
            an AM-29 stage); a process that registers AM-29 holds every existing key unchanged, resolves every existing
            stage exactly as before, and adds only AM-29's keys

    python -B scripts/smoke_am29_isolation.py [--base aa8cac283c1d7bda7aceb0844b47f6c3cf1ab1b7] [--print-hashes]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import smoke_am29_fixtures as FX  # noqa: E402

BASE = "aa8cac283c1d7bda7aceb0844b47f6c3cf1ab1b7"
check = FX.Checks("L-AM29 isolation")
SCOPE = ("src", "scripts", "configs", ":(exclude,icase)*test*", ":(exclude)docs/reference")

DUMP = r"""
import json, sys
sys.path.insert(0, %r)
from src.quant import stages as ST, qat_select as S
from src.eval import stage_artifacts as SA, artifacts as AR
if %r:
    from src.quant import am29_registry as R
    R.register()
def resolve_all():
    out = {}
    for k in sorted(ST.QUANT_STAGES):
        out["quant:" + k] = ST.resolve_quant_stage(k)
    for k in sorted(SA.STAGE_ARTIFACTS):
        out["artifact:" + k] = SA.resolve_stage_artifact(k)
        out["rules:" + k] = str(S.rules_path_for_stage(k))
    for rel in sorted(S.RULES_FILES):
        out["spec:" + rel] = S.rules_spec(%r + "/" + rel)
    return out
doc = {"QUANT_STAGES": ST.QUANT_STAGES, "RULES_FILES": S.RULES_FILES, "STAGE_ARTIFACTS": SA.STAGE_ARTIFACTS,
       "PROJECTION_FREE_STAGES": list(SA.PROJECTION_FREE_STAGES),
       "DESCRIPTIVE_ONLY_STAGES": list(SA.DESCRIPTIVE_ONLY_STAGES), "ARTIFACT_SCHEMA_STAGES": list(AR.STAGES),
       "resolved": resolve_all()}
try:
    ST.resolve_quant_stage("AM29-T1"); doc["am29_resolves"] = True
except Exception as e:
    doc["am29_resolves"] = type(e).__name__
print("DUMP " + json.dumps(doc, sort_keys=True, default=str))
"""


def dump(register: bool) -> dict:
    p = subprocess.run([sys.executable, "-B", "-c", DUMP % (str(REPO), register, str(REPO))], cwd=str(REPO),
                       capture_output=True, text=True)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("DUMP ")]
    if p.returncode != 0 or not line:
        raise RuntimeError(p.stderr[-800:])
    return json.loads(line[-1][5:])


def git(*args) -> str:
    import os
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-400:])
    return p.stdout.decode("utf-8")


def git_bytes(spec: str) -> bytes:
    p = subprocess.run(["git", "-C", str(REPO), "show", spec], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-300:])
    return p.stdout


def qat_code_paths() -> tuple[str, ...]:
    src = (REPO / "src" / "quant" / "qat.py").read_text(encoding="utf-8")
    m = re.search(r"QAT_CODE_PATHS = \((.*?)\n\)", src, re.S)
    return ast.literal_eval("(" + m.group(1) + ")")


def sec_code(base: str, show: bool) -> None:
    paths = qat_code_paths()
    rows, differ = [], []
    for p in paths:
        if "test" in p.lower() or p.startswith("docs/reference"):
            differ.append(f"{p}: a forbidden path in QAT_CODE_PATHS")
            continue
        before = hashlib.sha256(git_bytes(f"{base}:{p}")).hexdigest()
        after = hashlib.sha256((REPO / p).read_bytes()).hexdigest()
        rows.append((p, before, after))
        if before != after:
            differ.append(p)
    if show:
        print(f"QAT_CODE_PATHS sha256 at {base[:7]} (before) and in the checkout (after):")
        for p, b, a in rows:
            print(f"  {b}  {a}  {'=' if a == b else 'DIFFERS'}  {p}")
    check("code_qat_code_paths_byte_identical", not differ and len(rows) == len(paths) == 44,
          f"{len(rows)} files, before == after for all; differ: {differ}")
    from src.quant import am29 as M
    from src.quant import qat_kd as KD
    pinned_kd = [p for p in KD.E6KD_CODE_PATHS if p not in paths]
    kd_differ = [p for p in pinned_kd if hashlib.sha256(git_bytes(f"{base}:{p}")).hexdigest()
                 != hashlib.sha256((REPO / p).read_bytes()).hexdigest()]
    check("code_e6kd_code_paths_byte_identical", not kd_differ, f"{len(pinned_kd)} further E6-KD files; differ {kd_differ}")
    check("code_am29_files_new", all(subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{base}:{p}"],
                                                    capture_output=True).returncode != 0 for p in M.AM29_FILES)
          and all("test" not in p.lower() for p in M.AM29_FILES), f"{list(M.AM29_FILES)} absent at the base")


def sec_tree(base: str) -> None:
    committed = [ln.split("\t") for ln in git("diff", "--name-status", f"{base}..HEAD", "--", *SCOPE).splitlines() if ln]
    status = [ln for ln in git("status", "--porcelain=v1", "--untracked-files=all", "--", *SCOPE).splitlines() if ln]
    bad_c = [r for r in committed if r[0] != "A"]
    bad_s = [ln for ln in status if not ln.startswith("?? ")]
    added = sorted({r[1] for r in committed} | {ln[3:] for ln in status})
    check("tree_only_added_files", not bad_c and not bad_s,
          f"committed {[r[1] for r in committed]}; untracked {[ln[3:] for ln in status]}; not additions "
          f"{bad_c + bad_s}")
    check("tree_no_test_named_file_added", all("test" not in a.lower() for a in added), f"{len(added)} files")


def sec_registry() -> None:
    a, b = dump(False), dump(True)
    from src.quant import am29_registry as R
    check("registry_fresh_process_pinned", not any(k.startswith("am29") for k in a["QUANT_STAGES"])
          and R.RULES_REL not in a["RULES_FILES"] and not any(k.startswith("AM29") for k in a["STAGE_ARTIFACTS"])
          and a["am29_resolves"] == "QuantStageError",
          "without registration: no AM-29 key, resolve_quant_stage('AM29-T1') raises QuantStageError")
    same = all(b[t][k] == v for t in ("QUANT_STAGES", "RULES_FILES", "STAGE_ARTIFACTS") for k, v in a[t].items())
    check("registry_existing_keys_unchanged", same, "every pre-existing entry of the three tables, value for value")
    added = {t: sorted(set(b[t]) - set(a[t])) for t in ("QUANT_STAGES", "RULES_FILES", "STAGE_ARTIFACTS")}
    check("registry_adds_only_am29", added == {"QUANT_STAGES": sorted(R.QUANT_ENTRIES),
                                               "RULES_FILES": [R.RULES_REL],
                                               "STAGE_ARTIFACTS": sorted(R.STAGE_ARTIFACT_ENTRIES)}, f"{added}")
    check("registry_existing_stages_resolve_identically", all(b["resolved"].get(k) == v for k, v in a["resolved"].items()),
          f"{len(a['resolved'])} resolutions (quant stage, artifact contract, rules path, rules spec) equal")
    check("registry_tuples_untouched", all(a[t] == b[t] for t in ("PROJECTION_FREE_STAGES", "DESCRIPTIVE_ONLY_STAGES",
                                                                   "ARTIFACT_SCHEMA_STAGES")),
          "PROJECTION_FREE_STAGES, DESCRIPTIVE_ONLY_STAGES and src/eval/artifacts.py STAGES are never extended")
    check("registry_am29_resolves_when_registered", b["am29_resolves"] is True, "resolve_quant_stage('AM29-T1') after "
                                                                                "register()")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--print-hashes", action="store_true")
    ap.add_argument("--sections", default="code,tree,registry")
    a = ap.parse_args(argv)
    try:
        want = a.sections.split(",")
        if "code" in want:
            sec_code(a.base, a.print_hashes)
        if "tree" in want:
            sec_tree(a.base)
        if "registry" in want:
            sec_registry()
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("no_unexpected_exception", False, f"{type(e).__name__}: {e}")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
