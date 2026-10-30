#!/usr/bin/env python3
"""Smoke for B-3b's binding tday_binding/1 (lane TMAN, B-3b; src/tman/binding.py; B3R2-B3R5, B3R9; IF§11.4).

1. One definition (B3R2): the binding's path is a literal once among the named files -- manifest.LANE_CODE_FILES'
   Python modules, src/stats/artifact.py, driver.py, report.py and scripts/run_stats.py -- in binding.py, and
   validate.py imports it; no lane module imports src.stats; binding.py's module level loads the standard library
   only (a fresh `python -I` that imports it and lists the modules it loaded).
2. The STATS restatements (B3R3, B3R5): stats_identities, STATS_PRECISION, the stages, the input-list schema and the
   four file names equal src.stats.driver's (imported here as a program, as the stats smokes import it); the cell
   rule agrees with src.corruption_roles on the 24 cells; artifact_rel's layout (B3R9).
3. The file check and the export check (B3R5) on a fixture binding: each passes, and each rule is refused alone.

Synthetic inputs only, plain dicts or under the system temp dir (removed at the end); TEST is never read, and
nothing is written in the repository.

Run:  python -B scripts/smoke_tman_binding.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import copy  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_pins as PINS  # noqa: E402
import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)

check, expect_refused = FX.check, FX.expect_refused
WORD = "te" + "st"                                   # project rule 4's word, never written whole here
STATS_FILES = ("src/stats/artifact.py", "src/stats/driver.py", "src/stats/report.py", "scripts/run_stats.py")
INFERENTIAL = [c for c in PINS.IF6_CELLS if int(c.rpartition("_s")[2]) <= 3]       # the corrupted scope's 15
COMMIT = "e1" * 20
EC = {"model_device": "cpu", "input_devices": ["cpu"], "determinism_policy_applied": False, "batch_size": 1,
      "num_workers": 0, "torch_num_threads": 4, "image_digest": "sha256:" + "c" * 64, "evaluator_commit": COMMIT,
      "host": {"name": "tday-host", "cpu_model": "x86-64 fixture"}, "processes": 2}
FAMILY = {"E1": ("E1:s42:80k", "fp32"), "E2": ("E2:s42:lambda0.5:80k", "fp32"),
          "E3": ("E3:s42:lambda0.5:alpha25:80k", "fp32"), "E4": ("E4:s42:list42", "int8_ptq"),
          "E5": ("E5:s42:sel", "int8_qat"), "E6": ("E6:s42:sel", "int8_qat"), "E7": ("E7:s42:list42", "int8_ptq")}


def h(n: int) -> str:
    return f"{n:064x}"


# ============================================================================================== fixtures
def fixture_export() -> dict:
    """An export's binding-relevant part (IF§8): the seven family rows of seed 42 with the 24 cells, the two selected
    QAT models' qat_fakequant_off rows and a seed-43 row with no corrupted scope."""
    rows = []
    for n, (slot, prec) in enumerate(FAMILY.values()):
        int8 = prec in ("int8_ptq", "int8_qat")
        rows.append({"slot": slot, "precision": prec, "cells": list(PINS.IF6_CELLS),
                     "source": {"kind": "provenance" if int8 else "checkpoint", "path": f"r{n}/x", "sha256": h(100 + n)},
                     "converted_artifact_sha256": h(200 + n) if int8 else None})
        if prec == "int8_qat":
            rows.append({"slot": slot, "precision": "qat_fakequant_off", "cells": [],
                         "source": {"kind": "checkpoint", "path": f"r{n}/epoch_ckpts/e10.pt", "sha256": h(300 + n)},
                         "converted_artifact_sha256": None})
    rows.append({"slot": "E1:s43:80k", "precision": "fp32", "cells": [],
                 "source": {"kind": "checkpoint", "path": "r9/x", "sha256": h(109)}, "converted_artifact_sha256": None})
    return {"evaluator_commit": COMMIT, "manifest_sha256": h(1), "execution_context": copy.deepcopy(EC),
            "models": rows}


def identity(row: dict) -> str:
    """TB17's identity of a row (B1GO :196-197): the converted artifact for INT8, else source.sha256."""
    return row["converted_artifact_sha256"] if row["precision"] in ("int8_ptq", "int8_qat") else row["source"]["sha256"]


def fixture_binding(export: dict) -> dict:
    """A binding of every record the export's rows ask for (clean and upstream; each cell under its scope; fq_off
    for a qat_fakequant_off row), the 37 inputs naming the family rows' records."""
    from src.tman import binding as B
    from src.tman import schema as S
    names = B.file_names()
    arts = []
    for m in export["models"]:
        keys = [("fq_off", None)] if m["precision"] == "qat_fakequant_off" else (
            [("clean", None), ("upstream", None)]
            + [("corrupted" if int(c.rpartition("_s")[2]) <= 3 else "corrupted_sev45", c) for c in m["cells"]])
        for scope, cell in keys:
            arts.append({"slot": m["slot"], "precision": m["precision"], "scope": scope, "cell": cell,
                         "protocol": "upstream" if scope == "upstream" else "canvas",
                         "repo_relative_path": B.artifact_rel(S.file_id_of(m["slot"]), m["precision"], scope, cell),
                         "files_sha256": {n: h(400 + i) for i, n in enumerate(names)},
                         "checkpoint_sha256": identity(m), "repo_commit": export["evaluator_commit"],
                         "split_manifest_sha256": h(6) if scope == "upstream" else h(5)})
    inputs = []
    for stage, cond in B.stats_identities(INFERENTIAL):
        slot, prec = FAMILY[stage]
        inputs.append({"stage": stage, "condition": list(cond),
                       "key": [slot, prec, "clean" if cond == B.CLEAN else "corrupted", B.cell_of(cond)]})
    return {"format": S.BINDING_FORMAT,
            "references": {"manifest": {"path": S.MANIFEST_REL, "sha256": h(2), "body_sha256": export["manifest_sha256"],
                                        "freeze_dl": "DL-150"},
                           "evaluator": {"commit": export["evaluator_commit"]},
                           "execution_context": copy.deepcopy(export["execution_context"]),
                           "session": {"dedup": {"sha256": h(3)}},
                           "mde_entry": {"path": "reports/derived/mde_entry_20261009T034825Z.json", "sha256": h(4)},
                           "written_utc": "2026-11-05T06:00:00Z"},
            "artifacts": arts, "inputs": inputs}


def tman_data() -> tuple:
    """The 37 identities and stage precisions TMAN passes (its restatement, src.stats never imported)."""
    from src.tman import binding as B
    return B.stats_identities(INFERENTIAL), B.STATS_PRECISION


def record(doc: dict, slot: str, scope: str, cell=None, precision=None) -> dict:
    return next(r for r in doc["artifacts"] if r["slot"] == slot and r["scope"] == scope and r["cell"] == cell
                and (precision is None or r["precision"] == precision))


def errors_of(fn, *a) -> list:
    """`fn`'s error list; a crash where a refusal is due is its own error, so the check fails by name."""
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001 -- a crash fails its check
        return [f"raised {type(e).__name__}: {e}"]


def file_refused(name: str, edit, detail: str) -> None:
    """The fixture binding with `edit` applied: file_errors reports `detail` (and the base passes, case_file_check)."""
    from src.tman import binding as B
    doc = fixture_binding(fixture_export())
    edit(doc)
    errs = errors_of(B.file_errors, doc, *tman_data())
    check(name, any(detail in e for e in errs), str(errs[:3]))


def precision_input(doc: dict) -> None:
    """E5's clean input naming a clean record of its slot under another precision (one added): only the precision
    is not the stage's."""
    rec = dict(record(doc, "E5:s42:sel", "clean", precision="int8_qat"), precision="qat_fakequant_off")
    rec["repo_relative_path"] = rec["repo_relative_path"].replace("/int8_qat/", "/qat_fakequant_off/")
    doc["artifacts"].append(rec)
    doc["inputs"][4]["key"] = ["E5:s42:sel", "qat_fakequant_off", "clean", None]


def sev45_input(doc: dict) -> None:
    """E1's first corruption input naming a record of its cell under the corrupted_sev45 scope (one added)."""
    cell = doc["inputs"][7]["key"][3]
    rec = dict(record(doc, "E1:s42:80k", "corrupted", cell), scope="corrupted_sev45")
    rec["repo_relative_path"] = rec["repo_relative_path"].replace("/corrupted/", "/corrupted_sev45/")
    doc["artifacts"].append(rec)
    doc["inputs"][7]["key"] = ["E1:s42:80k", "fp32", "corrupted_sev45", cell]


def export_refused(name: str, edit, detail: str, edit_export=None) -> None:
    from src.tman import binding as B
    export = fixture_export()
    doc = fixture_binding(export)
    edit(doc)
    if edit_export is not None:
        edit_export(export)
    errs = errors_of(B.export_errors, doc, export, *tman_data())
    check(name, any(detail in e for e in errs), str(errs[:3]))


# ================================================================================================== cases
def lane_python() -> dict:
    from src.tman import manifest as M
    return {rel: (M.CODE_ROOT / rel).read_text(encoding="utf-8") for rel in M.LANE_CODE_FILES if rel.endswith(".py")}


def case_one_definition():
    """B3R2: binding.py defines the path once; validate.py imports it; no lane module imports src.stats."""
    from src.tman import binding as B
    from src.tman import validate as V
    texts = dict(lane_python(), **{rel: (REPO / rel).read_text(encoding="utf-8") for rel in STATS_FILES})
    hits = sorted((rel, t.count(B.BINDING_REL)) for rel, t in texts.items() if B.BINDING_REL in t)
    check("binding_path_literal_once_in_binding_py_B3R2", hits == [("src/tman/binding.py", 1)]
          and len(texts) == 13 + len(STATS_FILES), f"{hits}; {len(texts)} files")
    check("binding_path_is_reports_derived_tday_binding_json", B.BINDING_REL == "reports/derived/tday_binding.json")
    tree = ast.parse(texts["src/tman/validate.py"])
    imports = [(n.module, n.level, [a.name for a in n.names]) for n in tree.body if isinstance(n, ast.ImportFrom)]
    check("validate_imports_binding_rel_from_binding_B3R2", ("binding", 1, ["BINDING_REL"]) in imports
          and V.BINDING_REL is B.BINDING_REL, str(imports))
    found = []
    for rel, t in lane_python().items():
        for n in ast.walk(ast.parse(t)):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                    [n.module or ""] if isinstance(n, ast.ImportFrom) and n.level == 0 else [])
            found += [(rel, m) for m in mods if m == "src.stats" or m.startswith("src.stats.")]
    check("no_lane_module_imports_src_stats_B3R2", not found, str(found))


def case_import_closure():
    """B3R2: a fresh interpreter (-I) that puts the checkout first on sys.path and imports src.tman.binding loads the
    standard library and src, src.tman and src.tman.binding, and nothing else."""
    code = ("import sys\nbefore = set(sys.modules)\nsys.path.insert(0, %r)\nimport src.tman.binding\n"
            "new = sorted(set(sys.modules) - before)\n"
            "own = ('src', 'src.tman', 'src.tman.binding')\n"
            "print(repr((new, [m for m in new if m not in own and m.split('.')[0] not in sys.stdlib_module_names])))\n"
            ) % str(REPO)
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(REPO),
                       timeout=120)
    new, other = ast.literal_eval(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else ([], ["failed"])
    check("binding_module_level_loads_the_standard_library_only_B3R2", r.returncode == 0 and not other
          and {"src", "src.tman", "src.tman.binding"} <= set(new), (r.stderr or str(other))[-300:])
    tree = ast.parse((REPO / "src/tman/binding.py").read_text(encoding="utf-8"))
    top = []                                       # each module-level import's module ("." for a relative one)
    for n in tree.body:
        if isinstance(n, ast.Import):
            top += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            top.append(n.module if n.level == 0 and n.module else ".")
    check("binding_module_level_imports_name_the_standard_library_only_B3R2",
          bool(top) and all(m.split(".")[0] in sys.stdlib_module_names for m in top), str(top))


def case_stats_restated():
    """B3R3, B3R5: TMAN's restatements equal the statistics driver's (imported as a program); the cell rule agrees
    with src.corruption_roles; artifact_rel's layout (B3R9)."""
    import src.corruption_roles as CR
    from src.eval import artifacts as EA
    from src.stats import driver as D
    from src.tman import binding as B
    from src.tman import manifest as M
    from src.tman import schema as S
    want = D.official_inventory()
    check("stats_identities_are_official_inventory_s_37_in_order_B3R5",
          B.stats_identities(CR.INFERENTIAL_CELL_IDS) == want == B.stats_identities(INFERENTIAL) and len(want) == 37)
    check("stats_identities_from_the_registered_cells_B3R5",
          B.stats_identities(M.cells_for({"corrupted"})) == want, str(M.cells_for({"corrupted"}))[:200])
    check("stats_precision_stages_and_input_list_schema_are_the_driver_s",
          B.STATS_PRECISION == D.STAGE_PRECISION and B.STATS_STAGES == D.STAGES
          and B.STATS_CELL_STAGES == D.CORRUPTION_STAGES and B.STATS_INPUT_LIST == D.INPUT_LIST_SCHEMA)
    check("files_sha256_keys_are_the_driver_s_input_files_and_the_evaluator_s_B3R3",
          B.file_names() == tuple(sorted(D.INPUT_FILES)) == tuple(sorted((EA.MANIFEST_NAME, *EA.ARTIFACT_FILES)))
          and len(B.file_names()) == 4, str(B.file_names()))
    bad = [c for c in CR.CELL_IDS if B.cell_of(B.condition_of(c)) != c or B.condition_of(c)[1:] != CR.parse_cell_id(c)
           or B.cell_of(("corruption", *CR.parse_cell_id(c))) != CR.cell_id(*CR.parse_cell_id(c))]
    check("cell_rule_agrees_with_corruption_roles_on_the_24_cells_B3R5", not bad and len(CR.CELL_IDS) == 24, str(bad))
    check("clean_condition_names_no_cell", B.cell_of(B.CLEAN) is None)
    check("artifact_rel_layout_B3R9", B.artifact_rel("E1_s42_80k", "fp32", "clean", None)
          == "reports/tday/E1_s42_80k/fp32/clean" and B.artifact_rel("E6_s42_sel", "int8_qat", "corrupted", "fog_s1")
          == "reports/tday/E6_s42_sel/int8_qat/corrupted/fog_s1")
    expect_refused("artifact_rel_refuses_the_word_B3R9", "test_path", B.artifact_rel, f"E1_{WORD}", "fp32", "clean",
                   None)
    doc = fixture_binding(fixture_export())
    check("fixture_records_sit_at_artifact_rel", all(r["repo_relative_path"] == B.artifact_rel(
        S.file_id_of(r["slot"]), r["precision"], r["scope"], r["cell"]) for r in doc["artifacts"]))


def case_file_check():
    """B3R3, B3R5: the file check passes the fixture and reports each rule broken alone; the reader is strict."""
    from src.tman import binding as B
    from src.tman import schema as S
    export = fixture_export()
    doc = fixture_binding(export)
    errs = errors_of(B.file_errors, doc, *tman_data())
    check("fixture_binding_passes_the_file_check_B3R5", errs == [] and len(doc["inputs"]) == 37
          and len(doc["artifacts"]) == 7 * 26 + 2 + 2, f"{errs[:3]} {len(doc['artifacts'])}")
    check("binding_file_spec_holds_the_reserved_fields_TB11", set(S.BINDING_RESERVED["references"])
          <= set(S.BINDING_FILE_SPEC["references"][1]) and S.BINDING_FILE_SPEC["artifacts"][1][1]
          == S.BINDING_ARTIFACT_FIELDS and sorted(S.BINDING_FILE_SPEC) == ["artifacts", "format", "inputs",
                                                                           "references"])
    e1 = ("E1:s42:80k", "clean")
    for name, edit, detail in (
            ("format_other_B3R3", lambda d: d.update(format="tday_binding/2"), "binding.format"),
            ("top_level_extra_key_B3R3", lambda d: d.update(checkpoints={}), "unexpected ['checkpoints']"),
            ("top_level_without_inputs_B3R16", lambda d: d.pop("inputs"), "missing ['inputs']"),
            ("references_extra_key_B3R3", lambda d: d["references"].update(session_record={}), "unexpected"),
            ("references_without_mde_entry_TB11", lambda d: d["references"].pop("mde_entry"), "missing ['mde_entry']"),
            ("manifest_without_freeze_dl_B3R3", lambda d: d["references"]["manifest"].pop("freeze_dl"), "freeze_dl"),
            ("manifest_freeze_dl_not_a_dl_id_B3R3", lambda d: d["references"]["manifest"].update(freeze_dl="150"),
             "freeze_dl"),
            ("evaluator_commit_not_40_hex_B3R3", lambda d: d["references"]["evaluator"].update(commit="e1" * 19),
             "commit"),
            ("session_dedup_not_64_hex_B3R3", lambda d: d["references"]["session"]["dedup"].update(sha256="0" * 63),
             "sha256"),
            ("written_utc_not_utc_B3R3", lambda d: d["references"].update(written_utc="2026-11-05 06:00:00"),
             "written_utc"),
            ("record_extra_key_B3R3", lambda d: d["artifacts"][0].update(extra_field="x"), "unexpected"),
            ("record_without_split_digest_B3R3", lambda d: d["artifacts"][0].pop("split_manifest_sha256"), "missing"),
            ("record_scope_dedup_IF11_4", lambda d: d["artifacts"][0].update(scope="dedup"), "scope"),
            ("record_files_value_not_64_hex_B3R3",
             lambda d: d["artifacts"][0]["files_sha256"].update({"summary.json": "0" * 63}), "files_sha256"),
            ("record_checkpoint_null_B3R3", lambda d: d["artifacts"][0].update(checkpoint_sha256=None),
             "checkpoint_sha256"),
            ("record_twice_B3R3", lambda d: d["artifacts"].append(copy.deepcopy(d["artifacts"][0])), "a second record"),
            ("files_sha256_three_files_B3R3", lambda d: d["artifacts"][0]["files_sha256"].pop("MANIFEST.sha256"),
             "not keyed by"),
            ("clean_record_naming_a_cell", lambda d: record(d, *e1).update(cell="fog_s1"), "names a cell exactly"),
            ("corrupted_record_naming_no_cell", lambda d: record(d, "E1:s42:80k", "corrupted", "fog_s1").update(
                cell=None), "names a cell exactly"),
            ("clean_record_under_the_upstream_protocol", lambda d: record(d, *e1).update(protocol="upstream"),
             "upstream protocol"),
            ("upstream_record_under_the_canvas_protocol", lambda d: record(d, "E1:s42:80k", "upstream").update(
                protocol="canvas"), "upstream protocol"),
            ("record_path_with_a_dotdot_P2R5", lambda d: record(d, *e1).update(repo_relative_path="reports/../x"),
             "repo_relative_path"),
            ("record_path_absolute_P2R5", lambda d: record(d, *e1).update(repo_relative_path="/reports/x"),
             "repo_relative_path"),
            ("record_path_backslash_P2R5", lambda d: record(d, *e1).update(repo_relative_path="reports\\x"),
             "repo_relative_path"),
            ("record_path_holding_the_word_P2R5", lambda d: record(d, *e1).update(
                repo_relative_path=f"reports/{WORD}x"), "repo_relative_path"),
            ("record_repo_commit_not_the_evaluator_s_B3R5", lambda d: record(d, *e1).update(repo_commit="e2" * 20),
             "not the evaluator commit"),
            ("canvas_records_two_split_digests_DL74", lambda d: record(d, *e1).update(split_manifest_sha256=h(7)),
             "the canvas records score 2 split manifests"),
            ("upstream_records_two_split_digests_B3R5", lambda d: record(d, "E1:s42:80k", "upstream").update(
                split_manifest_sha256=h(7)), "the upstream records score 2 split manifests"),
            ("inputs_36_B3R5", lambda d: d["inputs"].pop(), "are not the 37 identities"),
            ("inputs_out_of_order_B3R5", lambda d: d["inputs"].insert(0, d["inputs"].pop(1)),
             "are not the 37 identities"),
            ("input_condition_other_B3R5", lambda d: d["inputs"][7].update(condition=["corruption", "fog", 4]),
             "are not the 37 identities"),
            ("input_key_naming_no_record_B3R5", lambda d: d["inputs"][0].update(key=["E1:s44:80k", "fp32", "clean",
                                                                                      None]), "names no record"),
            ("input_key_of_three_B3R5", lambda d: d["inputs"][0].update(key=["E1:s42:80k", "fp32", "clean"]),
             "names no record"),
            ("input_key_holding_a_list_B3R5", lambda d: d["inputs"][0]["key"].__setitem__(3, []), "names no record"),
            ("input_key_precision_not_the_stage_s_B3R5", precision_input, "is not of (precision, scope, cell)"),
            ("input_key_upstream_for_clean_B3R5", lambda d: d["inputs"][0].update(
                key=["E1:s42:80k", "fp32", "upstream", None]), "is not of (precision, scope, cell)"),
            ("input_key_cell_not_the_condition_s_B3R5", lambda d: d["inputs"][7].update(
                key=["E1:s42:80k", "fp32", "corrupted", "fog_s1"]), "is not of (precision, scope, cell)"),
            ("input_key_sev45_scope_for_a_cell_B3R5", sev45_input, "is not of (precision, scope, cell)"),
            ("two_inputs_naming_one_record_B3R5", lambda d: d["inputs"][2].update(key=list(d["inputs"][1]["key"])),
             "is another input's")):
        file_refused(name, edit, detail)
    for name, data in (("binding_duplicate_key_read_strictly_B3R3", b'{"format": 1, "format": 2}'),
                       ("binding_nan_read_strictly_B3R3", b'{"format": NaN}'),
                       ("binding_not_utf8_read_strictly_B3R3", b'{"format": "\xff"}')):
        expect_refused(name, "binding_invalid", B.read, data)
    check("binding_read_returns_the_document", B.read(S.json_file_bytes(doc)) == doc)


def case_export_check():
    """B3R5: the export check passes the fixture against its export, carries the file check, and refuses each record
    whose row, cell or commit is not the export's, and references of another manifest."""
    from src.tman import binding as B
    export = fixture_export()
    doc = fixture_binding(export)
    errs = errors_of(B.export_errors, doc, export, *tman_data())
    check("fixture_binding_passes_the_export_check_B3R5", errs == [], str(errs[:3]))
    export_refused("export_check_carries_the_file_check_B3R5", lambda d: d["inputs"].pop(), "are not the 37 identities")
    export_refused("record_of_no_export_row_B3R5", lambda d: d["artifacts"].append(dict(
        d["artifacts"][0], slot="E1:s44:80k", repo_relative_path="reports/tday/E1_s44_80k/fp32/clean")),
        "no row of the export")
    export_refused("record_cell_not_its_row_s_B3R5", lambda d: None, "is not one of its row's cells",
                   edit_export=lambda e: e["models"][0]["cells"].remove("fog_s1"))
    export_refused("record_repo_commit_not_the_export_s_B3R5", lambda d: None, "not the export's evaluator_commit",
                   edit_export=lambda e: e.update(evaluator_commit="e2" * 20))
    export_refused("references_of_another_manifest_body_B3R5", lambda d: None, "body_sha256 is not the export's",
                   edit_export=lambda e: e.update(manifest_sha256=h(9)))
    export_refused("references_of_another_execution_context_B3R5", lambda d: None,
                   "execution_context is not the export's",
                   edit_export=lambda e: e["execution_context"].update(torch_num_threads=8))


# ===================================================================================== the statistics cases
STATS: dict = {}                                     # the full-size fixture, built once per process
STATS_RUN = "tman-b3b-0001"


def stats_fixture() -> dict:
    """The stats smokes' fixture form (smoke_stats_integrity's Env, as its "off" inventory): the 37 inputs of section
    12.4.6 written through the unmodified evaluator writer (scripts/stats_fixtures.py) into a fresh clean scratch git
    repository -- full size (n 1561, k 3), status official, the evaluator's dataset name, the TEST split's name --
    at paths free of that name (runs/tday/...), loaded under the OFFICIAL policy, with placeholder tasks at the
    production B. The repository is also the code root the cases put a binding in (never the checkout's reports/)."""
    if STATS:
        return STATS
    import scripts.smoke_stats_integrity as SI
    from src.eval.adapters import DATASET_NAME
    from src.eval.evaluate import Condition
    from src.stats import artifact as A
    from src.stats import bootstrap as BS
    from src.stats import driver as D
    from src.stats.ingest import Policy
    F = SI.F
    root = FX.scratch_dir("tmanstats") / "fx"
    repo = F.clean_git_fixture(root)
    truth = F.make_truth(1561, 3, F.int_seed(20261030, "truth"))
    fs = F.FixtureSet(input_root=repo, truth=truth)
    for stage, cond in D.official_inventory():
        q = F.STAGE_QUALITY[stage] - (F.SEVERITY_DROP * cond[2] if cond[0] == "corruption" else 0.0)
        scored = F.score(truth, F.predict(truth, quality=q, seed=F.int_seed(20261030, stage, *cond)), Condition(*cond))
        tail = "clean" if cond[0] == "clean" else f"{cond[1]}_s{cond[2]}"
        rel = f"runs/tday/{stage.lower()}_{tail}"
        F.write_fixture(repo / rel, scored, 1561, stage=stage, condition=Condition(*cond), split=WORD,
                        status="official", repo_root=repo, run_id=f"tman-{stage.lower()}-{tail.replace('_', '-')}",
                        dataset_name=DATASET_NAME)
        fs.entries.append((stage, cond, rel))
    lst = F.write_input_list(root.parent / "inputs.json", fs)
    inputs = D.load_inputs(D.load_input_list(lst), repo, Policy.OFFICIAL)
    obs = D.assemble_observed(inputs, Policy.OFFICIAL)
    tasks = SI.placeholder_tasks(obs, BS.ANALYSIS_ID_OFFICIAL, BS.PRODUCTION_B)
    csha, craw = A.contract_sha256()
    inp = D.artifact_inputs(obs, inputs, tasks, analysis_id=BS.ANALYSIS_ID_OFFICIAL, run_id=STATS_RUN,
                            created_at_utc="2026-11-05T07:00:00Z", contract_sha256=csha, synthetic=False,
                            input_root=repo)
    STATS.update(SI=SI, A=A, D=D, F=F, fs=fs, repo=repo, list=lst, inp=inp, craw=craw, policy=Policy.OFFICIAL)
    return STATS


def stats_binding(fx: dict, edit=None) -> dict:
    """A binding of the fixture's 37 inputs (each record read from its artifact; `edit` changes the doc), written in
    canonical JSON at the fixture repository's BINDING_REL; returns the doc."""
    import json
    from src.eval.artifacts import canonical_json_bytes
    from src.tman import binding as B
    from src.tman import schema as S
    repo, D = fx["repo"], fx["D"]
    arts, inputs = [], []
    for stage, cond, rel in fx["fs"].entries:
        s = json.loads((repo / rel / "summary.json").read_text(encoding="utf-8"))
        slot, prec = FAMILY[stage]
        scope, cell = ("clean", None) if cond == B.CLEAN else ("corrupted", B.cell_of(cond))
        arts.append({"slot": slot, "precision": prec, "scope": scope, "cell": cell, "protocol": "canvas",
                     "repo_relative_path": rel, "files_sha256": D.input_file_sha256(repo / rel),
                     "checkpoint_sha256": s["run"]["checkpoint_sha256"], "repo_commit": s["run"]["repo_commit"],
                     "split_manifest_sha256": s["dataset"]["split_manifest_sha256"]})
        inputs.append({"stage": stage, "condition": list(cond), "key": [slot, prec, scope, cell]})
    doc = {"format": S.BINDING_FORMAT,
           "references": {"manifest": {"path": S.MANIFEST_REL, "sha256": h(2), "body_sha256": h(1),
                                       "freeze_dl": "DL-150"},
                          "evaluator": {"commit": arts[0]["repo_commit"]}, "execution_context": copy.deepcopy(EC),
                          "session": {"dedup": {"sha256": h(3)}},
                          "mde_entry": {"path": "reports/derived/mde_entry_20261009T034825Z.json", "sha256": h(4)},
                          "written_utc": "2026-11-05T06:00:00Z"},
           "artifacts": arts, "inputs": inputs}
    if edit is not None:
        edit(doc)
    f = repo / B.BINDING_REL
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(canonical_json_bytes(doc))
    return doc


class code_repo:
    """src.stats.artifact.CODE_REPO pointed at `root` for the block (the loader's default root, B3R6(a)); restored."""

    def __init__(self, A, root):
        self.A, self.root = A, root

    def __enter__(self):
        self.saved, self.A.CODE_REPO = self.A.CODE_REPO, self.root

    def __exit__(self, *exc):
        self.A.CODE_REPO = self.saved


def conditions(fx: dict, views, root) -> dict:
    """official_conditions over the fixture's family and `views`, the binding read under `root`: {name: detail}; a
    crash is its own entry, so a check fails by name."""
    A = fx["A"]
    try:
        return dict(A.official_conditions(A.build_family(fx["inp"]), views, fx["craw"], code_status_repo=root))
    except Exception as e:  # noqa: BLE001 -- a crash fails its check
        return {A.BINDING_CONDITION: f"raised {type(e).__name__}: {e}", "raised": ""}


def case_stats_loader():
    """B3R6(a), (d): the loader over a scratch repository holding a bound tree whose 37 inputs equal the binding
    (BINDING_CONDITION absent), and each way it is listed: unbound, one byte changed in one input's per_image.jsonl,
    a binding whose inputs name another record for one identity, a binding the file check refuses."""
    import shutil
    fx = stats_fixture()
    A, F, repo = fx["A"], fx["F"], fx["repo"]
    fam = A.build_family(fx["inp"])
    views = A.load_input_views(fam["input_artifacts"], repo, fx["policy"])
    stats_binding(fx)
    got = conditions(fx, views, repo)
    check("bound_tree_equal_to_the_binding_lists_no_binding_condition_B3R6d",
          A.binding_reason(views=views, repo_root=repo) is None and A.BINDING_CONDITION not in got
          and len(fam["input_artifacts"]) == 37, str(got.get(A.BINDING_CONDITION)))
    empty = FX.scratch_dir("tmanstatsnb")
    got = conditions(fx, views, empty)
    check("no_binding_file_lists_the_binding_unbound_B3R6a", "unbound" in got.get(A.BINDING_CONDITION, "")
          and list(got)[-1] == A.BINDING_CONDITION, str(got.get(A.BINDING_CONDITION)))
    copied = FX.scratch_dir("tmanstatscp") / "fx"
    shutil.copytree(repo, copied)
    d = copied / fx["fs"].entries[1][2]                                          # E2's clean input
    data = (d / "per_image.jsonl").read_bytes()
    (d / "per_image.jsonl").write_bytes(data.replace(b": ", b":\t", 1))         # one byte, the rows unchanged
    F.rehash(d)
    views2 = A.load_input_views(fam["input_artifacts"], copied, fx["policy"])
    got = conditions(fx, views2, repo)
    check("one_byte_changed_in_one_per_image_jsonl_listed_B3R6d", "files_sha256" in got.get(A.BINDING_CONDITION, "")
          and "e2_clean" in got.get(A.BINDING_CONDITION, ""), str(got.get(A.BINDING_CONDITION)))

    def another(doc):
        e3 = next(r for r in doc["artifacts"] if r["slot"] == FAMILY["E3"][0] and r["scope"] == "clean")
        doc["artifacts"].append(dict(e3, slot="E2:s43:lambda0.5:80k"))
        doc["inputs"][1]["key"] = ["E2:s43:lambda0.5:80k", "fp32", "clean", None]
    stats_binding(fx, another)
    got = conditions(fx, views, repo)
    check("binding_naming_another_record_for_one_identity_listed_B3R6d",
          "is not its binding record" in got.get(A.BINDING_CONDITION, "")
          and "repo_relative_path" in got.get(A.BINDING_CONDITION, ""), str(got.get(A.BINDING_CONDITION)))
    stats_binding(fx, lambda doc: doc.update(format="tday_binding/2"))
    got = conditions(fx, views, repo)
    check("binding_the_file_check_refuses_listed_with_its_reason_B3R6a",
          "binding_invalid" in got.get(A.BINDING_CONDITION, "") and "binding.format" in got.get(A.BINDING_CONDITION, ""),
          str(got.get(A.BINDING_CONDITION)))
    from src.tman import binding as B
    (repo / B.BINDING_REL).write_bytes(b'{"format": 1, "format": 2}')
    got = conditions(fx, views, repo)
    check("binding_not_strict_json_listed_with_its_reason_B3R3", "binding_invalid" in got.get(A.BINDING_CONDITION, "")
          and "duplicate key" in got.get(A.BINDING_CONDITION, ""), str(got.get(A.BINDING_CONDITION)))
    stats_binding(fx)


# B3R6(d)'s fields beside the four files' sha256, held here, not read from the module under test
RULED_FIELDS = ("repo_relative_path", "checkpoint_sha256", "repo_commit", "split_manifest_sha256")


def reason_of(fn, *a) -> str | None:
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001 -- a crash fails its check
        return f"raised {type(e).__name__}: {e}"


def case_inputs_reason():
    """B3R6(d), (e) on plain dicts: binding.inputs_reason holds the binding's inputs to the identities given, on
    INPUT_FIELDS, and on the four files' sha256 where they are given."""
    from src.tman import binding as B
    doc = fixture_binding(fixture_export())
    recs = {B.key_of(r): r for r in doc["artifacts"]}

    def given(files: bool) -> dict:
        out = {}
        for it in doc["inputs"]:
            rec = recs[tuple(it["key"])]
            out[(it["stage"], tuple(it["condition"]))] = dict({k: rec[k] for k in RULED_FIELDS}, **(
                {"files_sha256": dict(rec["files_sha256"])} if files else {}))
        return out
    check("input_fields_are_b3r6d_s_four_B3R6d", B.INPUT_FIELDS == RULED_FIELDS, str(B.INPUT_FIELDS))
    check("inputs_equal_on_the_fields_and_the_files_give_no_reason_B3R6d",
          reason_of(B.inputs_reason, doc, given(True)) is None and reason_of(B.inputs_reason, doc, given(False)) is None)
    got = given(True)
    got.pop(next(iter(got)))
    check("inputs_of_other_identities_reason_B3R6d",
          "not the binding's 37 identities" in str(reason_of(B.inputs_reason, doc, got)), str(reason_of(
              B.inputs_reason, doc, got)))
    for field in RULED_FIELDS:
        got = given(False)
        got[("E3", B.CLEAN)][field] = "x"
        check(f"input_{field}_differing_reason_B3R6d", f"'{field}'" in str(reason_of(B.inputs_reason, doc, got)),
              str(reason_of(B.inputs_reason, doc, got)))
    got = given(True)
    got[("E3", B.CLEAN)]["files_sha256"]["summary.json"] = h(9)
    check("input_files_differing_reason_where_given_B3R6e",
          "files_sha256" in str(reason_of(B.inputs_reason, doc, got)), str(reason_of(B.inputs_reason, doc, got)))


def case_stats_doors():
    """B3R6(b), (c): before any read, inputs not declared synthetic need a file at TEST_MANIFEST_BINDING (no open);
    declared synthetic, official mode runs bound or unbound; run_official passes the declaration through."""
    fx = stats_fixture()
    A, D = fx["A"], fx["D"]
    stats_binding(fx)
    empty = FX.scratch_dir("tmanstatsdr")
    outcome = {}
    for name, root, synthetic in (("unbound_real", empty, False), ("unbound_synthetic", empty, True),
                                  ("bound_real", fx["repo"], False), ("bound_synthetic", fx["repo"], True)):
        with code_repo(A, root):
            try:
                D.official_doors(confirm=True, synthetic=synthetic)
                outcome[name] = None
            except D.DriverRefusal as e:
                outcome[name] = str(e)
    check("bound_door_refuses_real_inputs_with_no_binding_file_B3R6b",
          "binding is unbound: no" in (outcome["unbound_real"] or ""), str(outcome))
    check("bound_door_passes_real_inputs_with_a_binding_file_B3R6b", outcome["bound_real"] is None, str(outcome))
    check("synthetic_inputs_pass_the_door_bound_or_unbound_B3R6b",
          outcome["unbound_synthetic"] is None and outcome["bound_synthetic"] is None, str(outcome))

    class Stop(Exception):
        pass
    seen, saved = [], (D.run_tasks, D.artifact_inputs)

    def capture(*a, **kw):
        seen.append(kw["synthetic"])
        raise Stop
    D.run_tasks = lambda obs, aid, B, jobs=1: fx["SI"].placeholder_tasks(obs, aid, B)
    D.artifact_inputs = capture
    try:
        out = FX.scratch_dir("tmanstatsro")
        for synthetic, repo_root in ((False, None), (True, fx["repo"])):
            with code_repo(A, fx["repo"]):
                try:
                    D.run_official(inputs_list=fx["list"], out_dir=out, run_id=f"tman-pass-{len(seen)}",
                                   confirm=True, synthetic=synthetic, repo_root=repo_root, B=24, log=lambda *a: None)
                except Stop:
                    pass
    finally:
        D.run_tasks, D.artifact_inputs = saved
    check("run_official_passes_the_synthetic_declaration_through_B3R6c", seen == [False, True], str(seen))


class reads_official:
    """The stats smokes' verifier_reads_official, its software block read as pinned too: family.json read as status
    official with no warnings, relabelled in memory right after the strict parse, so nothing official is ever on disk
    (P2). The artifact is written with a stack read as unpinned (its one warning)."""

    def __init__(self, A):
        self.A = A

    def __enter__(self):
        A = self.A
        self.real = real = A._check_finite_values

        def relabel(ctx):
            real(ctx)
            sw = dict(ctx.fam["software_environment"], observed=dict(A.PINNED_ENVIRONMENT), matches_pinned=True)
            ctx.fam = dict(ctx.fam, artifact_status="official", warnings=[], software_environment=sw)
        A._check_finite_values = relabel

    def __exit__(self, *exc):
        self.A._check_finite_values = self.real


def case_stats_verifier():
    """B3R6(e): an official artifact hand-built in the stats smokes' form, verified with and without its inputs:
    accepted when its 37 records equal the binding's (and, with the inputs, their four files' sha256 too); refused,
    by officiality_policy_verified, when the binding is absent or another record differs; a difference in the files
    alone refused with the inputs only."""
    fx = stats_fixture()
    A = fx["A"]
    art = FX.scratch_dir("tmanstatsva") / STATS_RUN
    saved = A.observed_environment
    A.observed_environment = lambda: dict(A.PINNED_ENVIRONMENT, numpy="0.0.0")
    try:
        A.write_statistics_artifact(art, fx["inp"])
    finally:
        A.observed_environment = saved
    import json
    on_disk = json.loads((art / A.FAMILY_JSON).read_text(encoding="utf-8"))
    check("the_artifact_on_disk_is_nonofficial_P2", on_disk["artifact_status"] == "nonofficial"
          and on_disk["warnings"] == [A.W_SOFTWARE], str(on_disk["warnings"]))

    def verdicts(root) -> list:
        out = []
        for kw in ({"input_root": fx["repo"], "contract_bytes": fx["craw"]}, {}):
            with code_repo(A, root), reads_official(A):
                try:
                    rep = A.verify_statistics_artifact(art, **kw)
                    out.append(("accepted", rep.family["artifact_status"]))
                except A.IntegrityError as e:
                    out.append((e.check, str(e)))
        return out
    stats_binding(fx)
    got = verdicts(fx["repo"])
    check("official_artifact_accepted_with_and_without_its_inputs_when_bound_equal_B3R6e",
          got == [("accepted", "official"), ("accepted", "official")], str(got)[:300])
    got = verdicts(FX.scratch_dir("tmanstatsvb"))
    check("official_artifact_refused_with_and_without_its_inputs_when_unbound_B3R6e",
          all(c == A.C_POLICY and "unbound" in d for c, d in got), str(got)[:300])
    stats_binding(fx, lambda doc: doc["artifacts"][0].update(checkpoint_sha256=h(9)))
    got = verdicts(fx["repo"])
    check("official_artifact_refused_when_a_record_differs_B3R6e",
          all(c == A.C_POLICY and "checkpoint_sha256" in d for c, d in got), str(got)[:300])
    stats_binding(fx, lambda doc: doc["artifacts"][0]["files_sha256"].update({"summary.json": h(9)}))
    got = verdicts(fx["repo"])
    check("files_differing_refused_with_the_inputs_only_B3R6e", got[0][0] == A.C_POLICY and "files_sha256" in got[0][1]
          and got[1] == ("accepted", "official"), str(got)[:300])
    stats_binding(fx)


def main() -> int:
    return FX.run_cases("TMAN BINDING SMOKE (lane TMAN, B-3b)",
                        (case_one_definition, case_import_closure, case_stats_restated, case_file_check,
                         case_export_check, case_inputs_reason, case_stats_loader, case_stats_doors,
                         case_stats_verifier))


if __name__ == "__main__":
    raise SystemExit(main())
