#!/usr/bin/env python3
"""Smoke for the generator pins (AM-19 Code: "the generator pins of item 1(c)"; AM-19 item 1(c); IF§3, IF§5, IF§6;
ruling TB6) (lane TMAN, B-1, C4).

The corrupted-TEST cache pin is L-CORR's JSON, embedded verbatim: the schema and each of AM-19 item 1(c)'s four
fields (script, commit, library versions, parameters) with IF§3's keys; pin_sha256 recomputed over
canonical_json_bytes (src/eval/artifacts.py:93) without pin_sha256; the cells against the registered ordered set,
which comes from src.corruption_roles (imported lazily) or, here, through a parameter holding IF§6's 24 cells
(TB6), with one static case on the module name and, since L-CORR merged, one live case (B-2, B2R10; audit M5:
the live set is IF§6's in L-CORR's corruption-major order); the cells each scope asks for (15 inferential, 9
descriptive); the image hasher's pin (B-3a's); and no value before the unlock record (M11; IF§5's four values and
the hasher's).

Run:  python -B scripts/smoke_tman_pins.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)

check, expect_refused = FX.check, FX.expect_refused
# IF§6's 24 cells in protocol order: the 15 inferential cells, gaussian_noise at 4, the four others at 4 and 5
CORRUPTIONS = ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog")
IF6_CELLS = ([f"{c}_s{s}" for c in CORRUPTIONS for s in (1, 2, 3)] + ["gaussian_noise_s4"]
             + [f"{c}_s{s}" for c in CORRUPTIONS if c != "gaussian_noise" for s in (4, 5)])


def pin(**kw) -> dict:
    """A pin as IF§3 states it, with its pin_sha256."""
    from src.tman import manifest as M
    from src.tman import schema as S
    p = {"schema": "plantseg-corruption-generator-pin/1.0.0",
         "script": {"path": "scripts/build_corruption_cache.py", "git_blob": "1" * 40},
         "commit": "2" * 40,
         "library_versions": {"python": "3.11", "numpy": "1.26.4", "pillow": "12.3.0", "scikit_image": "0.23.2",
                              "image_digest": S.STUDENT_IMAGE},
         "parameters": {"split": "te" + "st", "expected_images": 1561,
                        "protocol": {"path": "configs/corruption_protocol.json", "sha256": "3" * 64,
                                     "schema_version": "2.0.0"},
                        "cells": list(IF6_CELLS), "source_resize": "core_preprocess/1.0.0 image path",
                        "vendor_runtime_sha256": "caad6535" + "0" * 56, "seed_policy_id": "per_image_cell/1",
                        "master_seed": 42, "cache_schema": "2.0.0",
                        "storage": {"relative_path": "corr_cache_tday"}}}
    p.update(kw)
    p["pin_sha256"] = M.pin_sha256(p)
    return p


def case_pin_fields():
    from src.tman import manifest as M
    p = pin()
    check("pin_complete_IF3", not M.pin_incomplete_errors(p), str(M.pin_incomplete_errors(p)))
    check("if6_cells_24_in_protocol_order_R055", len(IF6_CELLS) == 24 and len(set(IF6_CELLS)) == 24
          and IF6_CELLS[:3] == ["motion_blur_s1", "motion_blur_s2", "motion_blur_s3"]
          and "gaussian_noise_s5" not in IF6_CELLS and IF6_CELLS[15] == "gaussian_noise_s4")
    for k in ("script", "commit", "library_versions", "parameters", "schema", "pin_sha256"):
        bad = {x: v for x, v in p.items() if x != k}
        check(f"pin_without_{k}_incomplete", bool(M.pin_incomplete_errors(bad)))
    check("pin_schema_other_refused", bool(M.pin_incomplete_errors(dict(p, schema="x/1"))))
    check("pin_script_without_git_blob", bool(M.pin_incomplete_errors(dict(p, script={"path": "s.py"}))))
    check("pin_commit_not_hex", bool(M.pin_incomplete_errors(dict(p, commit="HEAD"))))
    for block, key in (("library_versions", "image_digest"), ("library_versions", "scikit_image"),
                       ("parameters", "cells"), ("parameters", "master_seed"), ("parameters", "storage")):
        q = copy.deepcopy(p)
        del q[block][key]
        check(f"pin_{block}_without_{key}_incomplete", bool(M.pin_incomplete_errors(q)))
    check("pin_not_an_object", bool(M.pin_incomplete_errors(["x"])))
    q = copy.deepcopy(p)
    q["parameters"]["storage"]["relative_path"] = "elsewhere"
    check("pin_values_are_l_corr_s_A4", not M.pin_incomplete_errors(q))


def case_pin_sha256():
    from src.eval.artifacts import canonical_json_bytes
    from src.tman import manifest as M
    from src.tman import schema as S
    p = pin()
    check("pin_sha256_recomputed_over_canonical_json_TB6", M.pin_sha256_error(p) is None
          and p["pin_sha256"] == S.sha256_bytes(canonical_json_bytes({k: v for k, v in p.items()
                                                                      if k != "pin_sha256"})))
    q = copy.deepcopy(p)
    q["commit"] = "4" * 40
    check("pin_sha256_mismatch_after_an_edit", M.pin_sha256_error(q) is not None)
    reordered = dict(reversed(list(p.items())))
    check("pin_sha256_ignores_key_order", M.pin_sha256_error(reordered) is None)
    check("pin_sha256_of_a_verbatim_extra_key", M.pin_sha256_error(pin(note="verbatim")) is None)


def case_cells():
    from src.tman import manifest as M
    p = pin()
    check("cells_equal_the_registered_set", not M.pin_cells_errors(p, IF6_CELLS))
    check("cells_reordered_refused", bool(M.pin_cells_errors(pin(parameters=dict(
        pin()["parameters"], cells=list(reversed(IF6_CELLS)))), IF6_CELLS)))
    check("cells_23_refused_R055", bool(M.pin_cells_errors(pin(parameters=dict(
        pin()["parameters"], cells=[c for c in IF6_CELLS if c != "gaussian_noise_s4"])), IF6_CELLS)))
    check("cells_with_gaussian_noise_s5_refused", bool(M.pin_cells_errors(pin(parameters=dict(
        pin()["parameters"], cells=IF6_CELLS + ["gaussian_noise_s5"])), IF6_CELLS)))
    check("cells_missing_refused", bool(M.pin_cells_errors({"parameters": {}}, IF6_CELLS)))
    check("corruption_roles_module_name_static_TB6", M.CORRUPTION_ROLES_MODULE == "src.corruption_roles"
          and "importlib.import_module(CORRUPTION_ROLES_MODULE)" in (REPO / "src/tman/manifest.py")
          .read_text(encoding="utf-8"))
    saved = M.CORRUPTION_ROLES_MODULE
    try:
        M.CORRUPTION_ROLES_MODULE = "src.tman_absent_module"
        expect_refused("registered_set_unavailable_refused", "generator_cells_mismatch", M.pin_cells_errors, p)
    finally:
        M.CORRUPTION_ROLES_MODULE = saved
    check("cells_for_corrupted_15_inferential", M.cells_for({"corrupted"}, IF6_CELLS) == IF6_CELLS[:15])
    check("cells_for_sev45_9_descriptive", M.cells_for({"corrupted_sev45"}, IF6_CELLS) == IF6_CELLS[15:]
          and len(IF6_CELLS[15:]) == 9)
    check("cells_for_both_24", M.cells_for({"corrupted", "corrupted_sev45", "clean"}, IF6_CELLS) == IF6_CELLS)
    check("cells_for_no_corrupted_scope_empty", M.cells_for({"clean", "upstream", "dedup", "fq_off"}, IF6_CELLS) == [])
    check("cell_severity_parses", [M.cell_severity(c) for c in ("fog_s5", "gaussian_noise_s4")] == [5, 4])
    expect_refused("cell_id_malformed_refused", "generator_cells_mismatch", M.cell_severity, "fog-5")


def case_live_cells():
    """B2R10 (closing report §18 item 1; audit M5): with no parameter, registered_cells() is L-CORR's live ordered set,
    list(src.corruption_roles.CELL_IDS) -- 24 ids, the IF§6 set these cases inject, in L-CORR's protocol order
    (corruption-major), which differs from IF6_CELLS' severity-grouped order; so a pin carrying IF6_CELLS' order is
    refused against the live set. src/corruption_cache.py's build_pin holds the cells under parameters.cells, the key
    pin_cells_errors reads (read statically: of L-CORR's modules only src.corruption_roles is imported)."""
    import ast
    import importlib
    from src.tman import manifest as M
    live = M.registered_cells()
    roles = importlib.import_module(M.CORRUPTION_ROLES_MODULE)
    check("registered_cells_live_is_corruption_roles_cell_ids_B2R10_M5", live == list(roles.CELL_IDS)
          and len(live) == 24 and len(set(live)) == 24, str(live[:3]))
    check("live_set_is_if6_s_set_in_corruption_major_order_M5", set(live) == set(IF6_CELLS) and live != IF6_CELLS
          and live[15] == "brightness_s2" and IF6_CELLS[15] == "gaussian_noise_s4", f"{live[15]} {IF6_CELLS[15]}")
    check("pin_in_if6_order_refused_against_the_live_set_M5", bool(M.pin_cells_errors(pin()))
          and not M.pin_cells_errors(pin(parameters=dict(pin()["parameters"], cells=list(live)))))
    tree = ast.parse((REPO / "src/corruption_cache.py").read_text(encoding="utf-8"))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_pin"), None)
    keys = [x.value for node in (ast.walk(fn) if fn else ()) if isinstance(node, ast.Dict)
            for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant) and k.value == "parameters"
            and isinstance(v, ast.Dict) for x in v.keys if isinstance(x, ast.Constant)]
    check("build_pin_holds_the_cells_under_parameters_cells_B2R10", M.PIN_CELLS_KEY in keys, str(keys))


def case_values_and_hasher():
    from src.tman import manifest as M
    hasher = {"script": {"path": "scripts/tday_hash_images.py", "commit": "5" * 40, "sha256": "6" * 64},
              "python": "3.11", "parameters": {"folder": "images/<split>", "suffixes": [".jpg", ".jpeg"]}}
    pins = {"corrupted_cache": {"pin": pin(), "value": None}, "image_hasher": {"pin": hasher, "value": None}}
    check("no_value_before_the_unlock_M11", not M.test_value_errors(pins))
    for name in ("corrupted_cache", "image_hasher"):
        q = copy.deepcopy(pins)
        q[name]["value"] = {"x": "7" * 64}
        check(f"{name}_value_before_the_unlock_refused", bool(M.test_value_errors(q)))
    for key in M.PIN_VALUES:
        q = copy.deepcopy(pins)
        q["corrupted_cache"]["pin"]["parameters"][key] = "8" * 64
        check(f"pin_holding_{key}_refused_IF5", bool(M.test_value_errors(q)))
    q = copy.deepcopy(pins)
    q["image_hasher"]["pin"]["parameters"]["hash_list_sha256"] = "9" * 64
    check("hasher_pin_holding_its_value_refused", bool(M.test_value_errors(q)))
    check("hasher_pin_complete_B3a", not M.hasher_pin_errors(hasher))
    check("hasher_pin_without_script_sha256", bool(M.hasher_pin_errors(dict(hasher, script={
        "path": "scripts/tday_hash_images.py", "commit": "5" * 40}))))
    check("hasher_pin_without_python", bool(M.hasher_pin_errors({k: v for k, v in hasher.items() if k != "python"})))
    check("hasher_pin_without_parameters", bool(M.hasher_pin_errors(dict(hasher, parameters={}))))
    check("hasher_pin_not_an_object", bool(M.hasher_pin_errors(None)))


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN PINS SMOKE (lane TMAN, B-1, C4)",
                                  (case_pin_fields, case_pin_sha256, case_cells, case_live_cells,
                                   case_values_and_hasher)))
