#!/usr/bin/env python3
"""Smoke for B-3b's mutation tests (lane TMAN, B-3b; B3R12): every refusal and rule of src/tman/binding.py and of
scripts/tday_manifest.py's bind has a mutation killed by a named case of smoke_tman_binding.

smoke_tman_b3_mutations.py's form: a mutation replaces exact texts (each must occur once) in a scratch copy of the
two files, under the system temp dir; a fresh interpreter imports the copy as src.tman.binding and
scripts.tday_manifest (the lane's other modules, and src.stats, stay the checkout's; src.stats.artifact's import of
BINDING_REL then reaches the copy) and runs one case of smoke_tman_binding up to its named check. The mutation is
killed when that named check fails; a crash, or another check failing, is no kill. Each killing case first passes
alone on the unmutated copy, its named checks among those it records. The repository's files are read, never
written; the scratch copies are removed.

Run with TMPDIR set to a new folder:
  TMPDIR=<new folder> python -B scripts/smoke_tman_b3b_mutations.py
  (a worker is this file with --worker <spec>; --only <id>,<id> runs those mutations and their cases alone, a
  development aid whose RESULT is not the record)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BIN, CLI = "src/tman/binding.py", "scripts/tday_manifest.py"
SMOKES = {"binding": "scripts.smoke_tman_binding"}
RESULT_TAG = "MUTATION-RESULT "
WORD = "te" + "st"                                   # project rule 4's word, never written whole here
INV = "binding_invalid"

# (id, file, [(old, new), ...], smoke, case, the check the case runs to, the refusal codes it guards)
M = [
    # -------------------------------------------------------------------------- binding.py: names and rules (B3R2-B3R5)
    ("binding_rel_other", BIN, [('BINDING_REL: Final[str] = "reports/derived/tday_binding.json"',
                                 'BINDING_REL: Final[str] = "reports/derived/tday_binding_v2.json"')],
     "binding", "case_one_definition", "binding_path_is_reports_derived_tday_binding_json", []),
    ("cell_rule_other", BIN, [('else f"{name}_s{severity}"', 'else f"{name}_{severity}"')],
     "binding", "case_stats_restated", "cell_rule_agrees_with_corruption_roles_on_the_24_cells_B3R5", [INV]),
    ("identities_e6_first", BIN, [("for stage in STATS_CELL_STAGES for cond in corr",
                                   "for stage in reversed(STATS_CELL_STAGES) for cond in corr")],
     "binding", "case_stats_restated", "stats_identities_are_official_inventory_s_37_in_order_B3R5", [INV]),
    ("identities_clean_last", BIN, [("    return tuple((stage, CLEAN) for stage in STATS_STAGES) + tuple(\n"
                                     "        (stage, cond) for stage in STATS_CELL_STAGES for cond in corr)",
                                     "    return tuple(\n        (stage, cond) for stage in STATS_CELL_STAGES for cond in "
                                     "corr) + tuple((stage, CLEAN) for stage in STATS_STAGES)")],
     "binding", "case_stats_restated", "stats_identities_are_official_inventory_s_37_in_order_B3R5", [INV]),
    ("precision_e4_other", BIN, [('"E4": "int8_ptq", "E5"', '"E4": "int8_qat", "E5"')],
     "binding", "case_stats_restated", "stats_precision_stages_and_input_list_schema_are_the_driver_s", [INV]),
    # (dropping sorted() alone is equivalent: MANIFEST_NAME sorts before the lower-case ARTIFACT_FILES, already sorted)
    ("file_names_reversed", BIN, [("    return tuple(sorted((MANIFEST_NAME, *ARTIFACT_FILES)))",
                                   "    return tuple(sorted((MANIFEST_NAME, *ARTIFACT_FILES), reverse=True))")],
     "binding", "case_stats_restated", "files_sha256_keys_are_the_driver_s_input_files_and_the_evaluator_s_B3R3",
     [INV]),
    ("artifact_rel_without_its_cell", BIN, [('+ ("" if cell is None else f"/{cell}")', '+ ""')],
     "binding", "case_stats_restated", "artifact_rel_layout_B3R9", []),
    ("artifact_rel_word_guard_dropped", BIN, [("    S.refuse_test_path(rel)\n", "")],
     "binding", "case_stats_restated", "artifact_rel_refuses_the_word_B3R9", [WORD + "_path"]),
    ("read_not_binding_invalid", BIN, [('code="binding_invalid")', 'code="manifest_unreadable")')],
     "binding", "case_file_check", "binding_duplicate_key_read_strictly_B3R3", [INV]),
    # -------------------------------------------------------------------------------- binding.py: the file check
    ("file_spec_unchecked", BIN, [('    errs = S.field_errors(doc, S.BINDING_FILE_SPEC, "binding")\n',
                                   "    errs = []\n")],
     "binding", "case_file_check", "format_other_B3R3", [INV]),
    ("record_twice_allowed", BIN, [("        if key_of(r) in recs:", "        if False:")],
     "binding", "case_file_check", "record_twice_B3R3", [INV]),
    ("files_keys_unchecked", BIN, [('        if sorted(r["files_sha256"]) != names:', "        if False:")],
     "binding", "case_file_check", "files_sha256_three_files_B3R3", [INV]),
    ("cell_rule_unchecked", BIN, [('        if (r["cell"] is not None) != (r["scope"] in CELL_SCOPES):',
                                   "        if False:")],
     "binding", "case_file_check", "corrupted_record_naming_no_cell", [INV]),
    ("protocol_rule_unchecked", BIN, [('        if (r["protocol"] == UPSTREAM) != (r["scope"] == UPSTREAM):',
                                       "        if False:")],
     "binding", "case_file_check", "clean_record_under_the_upstream_protocol", [INV]),
    ("record_path_form_unchecked", BIN, [('        err = S.relative_path_error(r["repo_relative_path"])\n',
                                          "        err = None\n")],
     "binding", "case_file_check", "record_path_with_a_dotdot_P2R5", [INV]),
    ("record_commit_unchecked", BIN, [('        if r["repo_commit"] != commit:', "        if False:")],
     "binding", "case_file_check", "record_repo_commit_not_the_evaluator_s_B3R5", [INV]),
    ("split_digests_unchecked", BIN, [("        if len(found) > 1:", "        if False:")],
     "binding", "case_file_check", "canvas_records_two_split_digests_DL74", [INV]),
    ("split_digest_groups_merged", BIN, [('digests[UPSTREAM if r["scope"] == UPSTREAM else "canvas"]',
                                          'digests["canvas"]')],
     "binding", "case_file_check", "fixture_binding_passes_the_file_check_B3R5", [INV]),
    ("upstream_digests_dropped", BIN, [('digests[UPSTREAM if r["scope"] == UPSTREAM else "canvas"].add(',
                                        '(set() if r["scope"] == UPSTREAM else digests["canvas"]).add(')],
     "binding", "case_file_check", "upstream_records_two_split_digests_B3R5", [INV]),
    ("identities_unchecked", BIN, [("    if got != list(identities):", "    if False:")],
     "binding", "case_file_check", "inputs_36_B3R5", [INV]),
    ("identities_order_free", BIN, [("    if got != list(identities):",
                                     "    if sorted(got, key=repr) != sorted(identities, key=repr):")],
     "binding", "case_file_check", "inputs_out_of_order_B3R5", [INV]),
    ("input_key_shape_unchecked", BIN, [(" and all(k is None or isinstance(k, str) for k in it[\"key\"])", "")],
     "binding", "case_file_check", "input_key_holding_a_list_B3R5", [INV]),
    ("input_key_record_unchecked", BIN, [("        if key not in recs:", "        if False:")],
     "binding", "case_file_check", "input_key_naming_no_record_B3R5", [INV]),
    ("input_key_rule_unchecked", BIN, [("        elif key[1:] != want:", "        elif False:")],
     "binding", "case_file_check", "input_key_upstream_for_clean_B3R5", [INV]),
    ("input_precision_unchecked", BIN, [("want = (precision.get(stage),", "want = (key[1] if key else None,")],
     "binding", "case_file_check", "input_key_precision_not_the_stage_s_B3R5", [INV]),
    ("input_scope_unchecked", BIN, [('"clean" if cond == CLEAN else "corrupted", cell_of(cond))',
                                    'key[2] if key else None, cell_of(cond))')],
     "binding", "case_file_check", "input_key_sev45_scope_for_a_cell_B3R5", [INV]),
    ("input_cell_unchecked", BIN, [('"clean" if cond == CLEAN else "corrupted", cell_of(cond))',
                                   '"clean" if cond == CLEAN else "corrupted", key[3] if key else None)')],
     "binding", "case_file_check", "input_key_cell_not_the_condition_s_B3R5", [INV]),
    ("input_named_twice_allowed", BIN, [("        elif key in named:", "        elif False:")],
     "binding", "case_file_check", "two_inputs_naming_one_record_B3R5", [INV]),
    # ------------------------------------------------------------------------------ binding.py: the export check
    ("export_without_the_file_check", BIN, [("    errs = file_errors(doc, identities, precision)\n    if errs:\n"
                                             "        return errs\n    rows =", "    errs = []\n    rows =")],
     "binding", "case_export_check", "export_check_carries_the_file_check_B3R5", [INV]),
    ("export_row_unchecked", BIN, [("        if row is None:\n            errs.append(",
                                    "        if False:\n            errs.append(")],
     "binding", "case_export_check", "record_of_no_export_row_B3R5", [INV]),
    ("export_cell_unchecked", BIN, [('        elif r["cell"] is not None and r["cell"] not in row["cells"]:',
                                     "        elif False:")],
     "binding", "case_export_check", "record_cell_not_its_row_s_B3R5", [INV]),
    ("export_commit_unchecked", BIN, [('        if r["repo_commit"] != export["evaluator_commit"]:',
                                       "        if False:")],
     "binding", "case_export_check", "record_repo_commit_not_the_export_s_B3R5", [INV]),
    ("export_body_unchecked", BIN, [('    if refs["manifest"]["body_sha256"] != export["manifest_sha256"]:',
                                     "    if False:")],
     "binding", "case_export_check", "references_of_another_manifest_body_B3R5", [INV]),
    ("export_context_unchecked", BIN, [('    if refs["execution_context"] != export["execution_context"]:',
                                        "    if False:")],
     "binding", "case_export_check", "references_of_another_execution_context_B3R5", [INV]),
]
# The refusal codes the binding's code raises (schema.TmanRefused, and strict_json's code=), and B-1's guard's: each
# has a killed mutation (a static case ties this list to the files)
CODES = (INV,)
GUARD_CODES = (WORD + "_path",)                     # raised by B-1's guard from artifact_rel


# ================================================================================================ the worker
def worker(spec: dict) -> dict:
    """Run in a fresh interpreter: import the scratch copy at spec["root"] as src.tman.binding and
    scripts.tday_manifest (the lane's other modules stay the checkout's), run spec["case"] of spec["smoke"] up to
    spec["check"] (None: the whole case) and return the checks it recorded and any exception."""
    import importlib
    import importlib.util
    import src.tman
    root = Path(spec["root"])

    def load(name: str, path: Path):
        sp = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(sp)
        sys.modules[name] = mod
        sp.loader.exec_module(mod)
        return mod
    src.tman.binding = load("src.tman.binding", root / BIN)
    load("scripts.tday_manifest", root / CLI).REPO = REPO
    import scripts.smoke_tman_schema as FX
    first, target = FX.check, spec["check"]

    class Stop(Exception):
        pass

    def check(name, ok, detail=""):
        first(name, ok, detail)
        if name == target:
            raise Stop
    FX.check = check
    mod = importlib.import_module(SMOKES[spec["smoke"]])
    for m in list(sys.modules.values()):
        if getattr(m, "check", None) is first:
            m.check = check
    raised = None
    try:
        getattr(mod, spec["case"])()
    except Stop:
        pass
    except Exception as e:  # noqa: BLE001 -- a case that raises fails by name
        raised = f"{type(e).__name__}: {e}"[:300]
    FX.remove_scratch()
    return {"results": [(n, ok, d[:200]) for n, ok, d in FX.RESULTS], "raised": raised}


# ================================================================================================ the parent
def scratch_root(edits_by_file: dict, FX) -> Path:
    """A scratch copy of the two files with the edits applied; each old text must occur exactly once (ValueError
    otherwise)."""
    root = FX.scratch_dir("tmanbbmut")
    for rel in (BIN, CLI):
        text = (REPO / rel).read_bytes().decode("utf-8")
        for old, new in edits_by_file.get(rel, ()):
            if text.count(old) != 1:
                raise ValueError(f"{rel}: {text.count(old)} occurrences of {old[:70]!r}")
            text = text.replace(old, new)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(text.encode("utf-8"))
    return root


def run_worker(spec: dict) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    try:
        r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
                           capture_output=True, text=True, env=env, cwd=str(REPO), timeout=900)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(RESULT_TAG)]
    if not lines:
        return {"error": (r.stderr or r.stdout)[-400:]}
    return json.loads(lines[-1][len(RESULT_TAG):])


def files_sha256() -> dict:
    return {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in (BIN, CLI)}


def raised_codes(path: Path, functions=None) -> set:
    """The literal codes a file refuses with -- schema.TmanRefused's first argument and strict_json's code= -- in the
    named `functions` (all of the file when None)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scopes = [tree] if functions is None else [n for n in ast.walk(tree)
                                               if isinstance(n, ast.FunctionDef) and n.name in functions]
    out = set()
    for scope in scopes:
        for n in ast.walk(scope):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr == "TmanRefused" and n.args and isinstance(n.args[0], ast.Constant):
                    out.add(n.args[0].value)
                if n.func.attr == "strict_json":
                    out |= {k.value.value for k in n.keywords if k.arg == "code" and isinstance(k.value, ast.Constant)}
    return out


ONLY: set = set()        # --only id,id: a development run of those mutations and their cases; never the record


def case_mutations():
    import scripts.smoke_tman_schema as FX
    check = FX.check
    check("codes_listed_are_the_files_codes", set(CODES) == raised_codes(REPO / BIN), str(sorted(raised_codes(
        REPO / BIN))))
    before = files_sha256()
    clean = scratch_root({}, FX)
    mutations = [m for m in M if not ONLY or m[0] in ONLY]
    cases = sorted({(m[3], m[4]) for m in mutations})
    jobs = [("baseline", (smoke, case), {"root": str(clean), "smoke": smoke, "case": case, "check": None})
            for smoke, case in cases]
    for mid, rel, edits, smoke, case, target, codes in mutations:
        try:
            root = scratch_root({rel: edits}, FX)
        except ValueError as e:
            check(f"mut_{mid}_text_occurs_once", False, str(e))
            continue
        jobs.append(("mutation", mid, {"root": str(root), "smoke": smoke, "case": case, "check": target}))
    with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1))) as pool:
        outs = list(pool.map(lambda job: run_worker(job[2]), jobs))
    targets = {}
    for m in mutations:
        targets.setdefault((m[3], m[4]), set()).add(m[5])
    killed_codes = set()
    for (kind, key, spec), out in zip(jobs, outs):
        if kind == "baseline":
            smoke, case = key
            names = {n for n, _, _ in out.get("results", [])}
            bad = [n for n, ok, _ in out.get("results", []) if not ok]
            check(f"baseline_{smoke}_{case}_passes_alone", "error" not in out and out.get("raised") is None
                  and not bad and targets[key] <= names,
                  str(out.get("error") or out.get("raised") or bad or sorted(targets[key] - names))[:300])
            continue
        m = next(x for x in M if x[0] == key)
        fails = [n for n, ok, _ in out.get("results", []) if not ok]
        killed = "error" not in out and m[5] in fails          # the named check itself fails
        how = out.get("error") or (f"raised {out['raised']}" if out.get("raised") else f"failed {fails[:3]}")
        print(f"MUTATION {key:34} {m[1]:28} {'KILLED' if killed else 'SURVIVED'} by {m[3]}:{m[4]} -- {how[:150]}")
        check(f"mut_{key}", killed, f"survived: {m[3]}:{m[4]}'s check {m[5]} did not fail ({how[:200]})"
              if "error" not in out else str(out.get("error"))[:300])
        if killed:
            killed_codes |= set(m[6])
    if not ONLY:
        need = set(CODES) | set(GUARD_CODES)
        check("every_refusal_code_has_a_killed_mutation", need <= killed_codes and len(need) == 2,
              str(sorted(need - killed_codes)))
        check("every_mutation_has_its_own_id", len({m[0] for m in M}) == len(M))
    check("the_repository_files_are_unchanged", files_sha256() == before)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(RESULT_TAG + json.dumps(worker(json.loads(sys.argv[2]))))
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        ONLY.update(sys.argv[2].split(","))
        unknown = ONLY - {m[0] for m in M}
        if unknown:
            raise SystemExit(f"--only names no mutation: {sorted(unknown)}")
    import scripts.smoke_tman_schema as FX
    return FX.run_cases("TMAN B-3b MUTATIONS SMOKE (lane TMAN, B-3b; B3R12)", (case_mutations,))


if __name__ == "__main__":
    raise SystemExit(main())
