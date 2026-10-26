#!/usr/bin/env python3
"""The mutation harness of lane LREP (B-1 to B-3; LRB8): every mutation names the case that kills it.

Each mutation rewrites one LREP function from its own source (an exact, unique text replacement,
compiled into the module's namespace and restored afterwards; scripts/smoke_teacher_diag_mutations.py's
harness) and runs its named case, which must pass unmutated and fail under the mutation (a failed
check or an unexpected error both count as killed). Three kinds:

  C  one per refusal code of src.stats.heldout_inputs.CODES: its raise site disabled
     (`raise refusal(` -> `if False: raise refusal(`), killed by the case that reaches that code with
     every earlier gate passing (the cases of the lane's smokes, or the in-process cases below)
  R  a further raise site of a code, added by the DL-24 review's fixes, disabled the same way and
     killed by the case that reaches that site
  S  semantic mutations of the formulas, states, labels and rules (the plan's §7.1 list as LRB2-LRB13
     amend it, and the DL-24 fixes), each killed by its named case

The cases run on one synthetic world (scripts/lrep_fixtures.py) under a new directory in TMPDIR (left
in place); no created path contains "test".

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_lrep_mutations.py
"""
from __future__ import annotations

import __future__ as _future
import contextlib
import copy
import inspect
import sys
import textwrap
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import lrep_fixtures as LF                                               # noqa: E402
import smoke_am6_tables as S_AM6                                         # noqa: E402
import smoke_default_candidates as S_DC                                  # noqa: E402
import smoke_report_extensions as S_RX                                   # noqa: E402
import smoke_seed_stability as S_SEED                                    # noqa: E402
import smoke_strata_tables as S_ST                                       # noqa: E402
from src.stats import am6_tables as AM6                                  # noqa: E402
from src.stats import default_candidates as DC                           # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import report_extensions as RX                            # noqa: E402
from src.stats import seed_stability as SS                               # noqa: E402
from src.stats import strata_tables as ST                                # noqa: E402

CTX = types.SimpleNamespace(world=None)


@contextlib.contextmanager
def mutate(owner, name: str, *pairs):
    """Replace `owner.name` (a module function) by its own source with each (old, new) replaced exactly
    once, compiled into its module's namespace; restore it afterwards (the L-TEACHER-DIAG harness)."""
    original = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    module = sys.modules[original.__module__]
    src = textwrap.dedent(inspect.getsource(original))
    for old, new in pairs:
        if src.count(old) != 1:
            raise AssertionError(f"mutation anchor occurs {src.count(old)} times in {name}: {old!r}")
        src = src.replace(old, new)
    ns = module.__dict__
    had, prev = name in ns, ns.get(name)
    exec(compile(src, f"<mutant {module.__name__}.{name}>", "exec", flags=_future.annotations.compiler_flag,
                 dont_inherit=True), ns)
    mutant = ns[name]
    if isinstance(owner, type):
        if had:
            ns[name] = prev
        else:
            del ns[name]
    setattr(owner, name, mutant)
    try:
        yield
    finally:
        setattr(owner, name, original)


def refused(code, fn):
    try:
        fn()
    except H.LrepRefusal as e:
        return e.code == code, f"{e.code}: {e.detail[:200]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


# --------------------------------------------------------------------------------------------------
# in-process cases for the report's codes and the seed-42 cross-check
# --------------------------------------------------------------------------------------------------
def m_out_dir():
    w = CTX.world
    return refused("out_dir_invalid", lambda: RX.check_out_dir(
        w.input_root / "runs" / "lrep_mutation_out", [w.stats_path, w.input_root, w.inventory_path.parent]))


def m_mde():
    d = CTX.world.fresh("mut_mde")
    return refused("mde_entry_invalid", lambda: RX.build_mde(d / "mde_entry_absent.json", official=False))


def m_usage():
    w = CTX.world
    out = CTX.world.fresh("mut_usage") / "rx"
    return refused("usage_invalid", lambda: RX.run(stats_artifact=w.stats_path, inventory=w.inventory_path,
                                                   out_dir=out, repo_root=w.input_root, synthetic=False,
                                                   strata=w.strata_path, parts="mde"))


def m_seed42_tamper():
    """A family whose stored accuracy_e1_e2 delta is one ulp off is refused by the seed-42 check."""
    import math
    st = S_SEED.stats_artifact()
    fam = copy.deepcopy(st.family)
    t = next(t for t in fam["bootstrap_tasks"] if t["task_id"] == "accuracy_e1_e2__dataset_miou_delta")
    t["observed"] = math.nextafter(t["observed"], 1.0)
    fake = types.SimpleNamespace(**{**st.__dict__, "family": fam})
    inv = H.open_inventory(CTX.world.inventory_path, policy=st.policy)
    inputs = SS.load_seed_inputs(st, inv, input_root=CTX.world.input_root)
    names = H.corruption_names()
    m = {key: SS.m_of(li.run.stats) for key, li in inputs.clean.items()}
    mc = {key: SS.miou_c({k: li.run.stats for k, li in cells.items()}, names)
          for key, cells in inputs.corrupted.items()}
    return refused("seed42_mismatch", lambda: SS.seed42_cross_checks(fake, m, mc))


LOCAL = {"mut:out_dir": m_out_dir, "mut:mde": m_mde, "mut:usage": m_usage,
         "mut:seed42_tamper": m_seed42_tamper}


def all_cases() -> dict:
    out = dict(LOCAL)
    for tag, mod in (("seed", S_SEED), ("strata", S_ST), ("defaults", S_DC), ("am6", S_AM6),
                     ("report", S_RX)):
        for cid, _, fn in mod.cases():
            out[f"{tag}:{cid}"] = fn
    return out


# --------------------------------------------------------------------------------------------------
# the mutations: (id, description, [(owner, function, [(old, new), ...]), ...], killing case)
# --------------------------------------------------------------------------------------------------
def off(code, tail=""):
    """Disable a raise site: `raise refusal("code",<tail>` -> `if False: raise refusal("code",<tail>`."""
    return (f'raise refusal("{code}",{tail}', f'if False: raise refusal("{code}",{tail}')


def off_h(code, tail=""):
    return (f'raise H.refusal("{code}",{tail}', f'if False: raise H.refusal("{code}",{tail}')


MUTATIONS = [
    # C: one per refusal code
    ("C01", "stats_artifact_unverified not raised",
     [(H, "load_stats_artifact", [off("stats_artifact_unverified")])], "seed:R1"),
    ("C02", "stub_reader_official not raised", [(H, "open_inventory", [off("stub_reader_official")])],
     "seed:R2a"),
    ("C03", "inventory_unreadable not raised (unreadable file)",
     [(H, "read_inventory", [off("inventory_unreadable", ' f"{p}: {type(e)')])], "seed:R2b"),
    ("C04", "inventory_sha256_mismatch not raised",
     [(H, "read_inventory", [off("inventory_sha256_mismatch")])], "seed:R2c"),
    ("C05", "inventory_invalid not raised (reader and open_inventory)",
     [(H, "read_inventory", [off("inventory_invalid")]),
      (H, "open_inventory", [off("inventory_invalid", ' f"{path}: {problem}')])], "seed:R2d"),
    ("C06", "binding_invalid not raised (binding file)", [(H, "read_inventory", [off("binding_invalid")])],
     "seed:R2e"),
    ("C07", "binding_unbound not raised", [(H, "open_inventory", [off("binding_unbound")])], "seed:R2f"),
    ("C08", "freeze_mismatch not raised", [(H, "open_inventory", [off("freeze_mismatch")])], "seed:R2g"),
    ("C09", "planned_slot_missing not raised", [(H, "record_slots", [off("planned_slot_missing")])],
     "seed:R3a"),
    ("C10", "planned_slot_duplicate not raised", [(H, "record_slots", [off("planned_slot_duplicate")])],
     "seed:R3b"),
    ("C11", "unplanned_slot not raised", [(H, "record_slots", [off("unplanned_slot")])], "seed:R3c"),
    ("C12", "status_invalid not raised (a missing record field)",
     [(H, "_checked_kind", [off("status_invalid", ' f"slot {slot!r} ({kind})')])], "seed:R4b"),
    ("C13", "superseded not raised", [(H, "_checked_kind", [off("superseded")])], "seed:R5"),
    ("C14", "parent_rule not raised (not run while the parent is complete)",
     [(H, "_not_run_status", [off("parent_rule", ' f"slot {slot!r} is not run')])], "seed:R12"),
    ("C15", "seed42_not_complete not raised", [(H, "check_seed42", [off("seed42_not_complete")])],
     "seed:R13"),
    ("C16", "seed_inconsistent not raised", [(H, "check_rows", [off("seed_inconsistent")])], "seed:R14"),
    ("C17", "artifact_missing not raised (no record)", [(H, "binding_record", [off("artifact_missing")])],
     "seed:R6a"),
    ("C18", "artifact_for_missing_run not raised",
     [(H, "check_missing_runs", [off("artifact_for_missing_run")])], "seed:R6b"),
    ("C19", "artifact_record_mismatch not raised (files)",
     [(H, "load_bound", [off("artifact_record_mismatch", ' f"{label}: files_sha256')])], "seed:R7a"),
    ("C20", "not_canvas not raised (record protocol)", [(H, "load_bound", [off("not_canvas")])], "seed:R8"),
    ("C21", "artifact_unreadable not raised (files)",
     [(H, "load_bound", [off("artifact_unreadable", ' f"{label}: {type(e)')])], "seed:R9"),
    ("C22", "precision_mismatch not raised", [(H, "load_bound", [off("precision_mismatch")])], "seed:R18"),
    ("C23", "path_unsafe not raised", [(H, "safe_relative_path", [off("path_unsafe")])], "seed:R19"),
    ("C24", "cross_artifact_mismatch not raised", [(H, "cross_check", [off("cross_artifact_mismatch")])],
     "seed:R10a"),
    ("C25", "e1_checkpoint_record not raised", [(H, "check_e1_record", [off("e1_checkpoint_record")])],
     "seed:R10b"),
    ("C26", "seed42_mismatch not raised (identity of the 37)",
     [(H, "check_seed42_identity", [off("seed42_mismatch")])], "seed:R11"),
    ("C27", "telemetry_sha256_mismatch not raised",
     [(H, "read_telemetry", [off("telemetry_sha256_mismatch")])], "seed:R16"),
    ("C28", "telemetry_invalid not raised (no attempt of the run of record)",
     [(SS, "_of_record_attempt", [off_h("telemetry_invalid")])], "seed:R17"),
    ("C29", "robustness_cells_missing not raised",
     [(SS, "load_seed_inputs", [off_h("robustness_cells_missing")])], "seed:R20"),
    ("C30", "strata_invalid not raised", [(ST, "load_strata_checked", [off_h("strata_invalid")])],
     "strata:R4"),
    ("C31", "strata_not_of_record not raised", [(ST, "load_strata_checked", [off_h("strata_not_of_record")])],
     "strata:R3"),
    ("C32", "strata_gt_mismatch not raised", [(ST, "model_checks", [off_h("strata_gt_mismatch")])],
     "strata:R1"),
    ("C33", "strata_integrity not raised", [(ST, "model_checks", [off_h("strata_integrity")])], "strata:R2"),
    ("C34", "default_candidate_missing not raised (no slot)",
     [(DC, "default_slots", [off_h("default_candidate_missing", ' f"no default_candidate slot')])],
     "defaults:R3"),
    ("C35", "default_candidate_invalid not raised (another candidate)",
     [(DC, "default_slots", [off_h("default_candidate_invalid", ' f"{stage} ({slot!r}): "')])],
     "defaults:R1"),
    ("C36", "selection_record_mismatch not raised",
     [(DC, "check_selection_record", [off_h("selection_record_mismatch")])], "defaults:R4"),
    ("C37", "gt_present_unfaithful not raised", [(AM6, "am6_rows", [off_h("gt_present_unfaithful")])],
     "am6:R1"),
    ("C38", "out_dir_invalid not raised (inside an input)",
     [(RX, "check_out_dir", [off_h("out_dir_invalid", ' f"--out-dir {out_dir} is inside')])], "mut:out_dir"),
    ("C39", "parts_invalid not raised", [(RX, "parse_parts", [off_h("parts_invalid")])], "report:R1"),
    ("C40", "mde_entry_invalid not raised (unreadable)",
     [(RX, "build_mde", [off_h("mde_entry_invalid", ' f"cannot be read')])], "mut:mde"),
    ("C41", "usage_invalid not raised (--repo-root without --synthetic-inputs)",
     [(RX, "run", [off_h("usage_invalid", ' "--repo-root is accepted')])], "mut:usage"),
    # R: further raise sites (DL-24)
    ("R01", "binding_invalid not raised (a bound export without binding.artifacts)",
     [(H, "open_inventory", [off("binding_invalid", ' f"{path}: a bound export')])], "seed:R2h"),
    ("R02", "status_invalid not raised (two entries for one planned slot)",
     [(H, "record_slots", [off("status_invalid")])], "seed:R4c"),
    ("R03", "status_invalid not raised (a divergence in the wrong unit)",
     [(H, "_check_unit", [off("status_invalid")])], "seed:R4d"),
    ("R04", "parent_rule not raised (a selection whose parent is not its QAT run)",
     [(H, "resolve_status", [off("parent_rule", ' f"selection {slot!r}: its parent')])], "seed:R12c"),
    ("R05", "parent_rule not raised (a PTQ model with kind null whose parent is another stage)",
     [(H, "resolve_status", [off("parent_rule", ' f"PTQ slot {slot!r}: its parent')])], "seed:R12e"),
    ("R06", "seed_inconsistent not raised (a PTQ model with kind null whose parent is at another seed)",
     [(H, "resolve_status", [off("seed_inconsistent")])], "seed:R14b"),
    ("R07", "parent_rule not raised (a descendant of a cut run that is not 'not run')",
     [(H, "check_parentage", [off("parent_rule", ' f"{stage} seed {seed} ({slot!r}) reads')])],
     "seed:R12d"),
    ("R08", "parent_rule not raised (a model that descends from the wrong run)",
     [(H, "check_parentage", [off("parent_rule", ' f"{stage} seed {seed} ({slot!r}) descends')])],
     "seed:R12b"),
    ("R09", "precision_mismatch not raised (a model row whose model_role is not student)",
     [(H, "check_rows", [off("precision_mismatch")])], "seed:R18b"),
    ("R10", "out_dir_invalid not raised by the out-dir check (a parent that does not exist)",
     [(RX, "check_out_dir", [off_h("out_dir_invalid", ' f"--out-dir {out_dir}: its parent')])], "report:P5"),
    ("R11", "seed_inconsistent not raised (an fq_off row that is not its int8_qat row's model)",
     [(RX, "fqoff_row", [off_h("seed_inconsistent")])], "report:Q6"),
    ("R12", "out_dir_invalid not raised by write_report (a parent that does not exist)",
     [(RX, "write_report", [off_h("out_dir_invalid", ' f"{final}: its parent')])], "report:P6"),
    # S: semantic mutations
    ("S01", "condition 1: g > 0 becomes g >= 0",
     [(SS, "condition_1", [("not g[k] > 0.0", "not g[k] >= 0.0")])], "seed:S2"),
    ("S02", "condition 2: mean > max SD becomes >=",
     [(SS, "condition_2", [("if mean > max(s1, s3)", "if mean >= max(s1, s3)")])], "seed:S7"),
    ("S03", "the SDs become population SDs (pstdev)",
     [(SS, "mean_and_sds", [("statistics.stdev(e1), statistics.stdev(e3)",
                             "statistics.pstdev(e1), statistics.pstdev(e3)")])], "seed:F9"),
    ("S04", "the mean gain becomes a naive sum", [(SS, "mean_and_sds", [("math.fsum(gains) / 3",
                                                                        "sum(gains) / 3")])], "seed:F7"),
    ("S05", "condition 3 compares with 2.0 (pp) instead of 0.020",
     [(SS, "condition_3", [("not d[k] < DROP_THRESHOLD", "not d[k] < 2.0")])], "seed:F11"),
    ("S06", "condition 3: d < 0.020 becomes d <= 0.020",
     [(SS, "condition_3", [("not d[k] < DROP_THRESHOLD", "not d[k] <= DROP_THRESHOLD")])], "seed:S9"),
    ("S07", "the drop's sign flipped",
     [(SS, "build_tables", [('d = {k: m[("E3", k)] - m[("E6", k)]', 'd = {k: m[("E6", k)] - m[("E3", k)]')])],
     "seed:F6"),
    ("S08", "a missing seed counted as passing (condition 1)",
     [(SS, "condition_1", [("elif all(k in g for k in SEEDS):", "elif True:")])], "seed:S5"),
    ("S09", "the E3-divergence override removed",
     [(SS, "verdict", [("and not e3_divergences)", ")")])], "seed:S14"),
    ("S10", "an E2 divergence changes the verdict",
     [(SS, "verdict", [("and not e3_divergences)", "and not e3_divergences and not e2_divergences)")])],
     "seed:S17"),
    ("S11", "the reason order changed (condition 1 first of all)",
     [(SS, "verdict", [('reasons += [f"condition 1 fails at seed {k}"',
                        'reasons[:0] = [f"condition 1 fails at seed {k}"')])], "seed:S16"),
    ("S12", "condition 2 computed over the seeds present (two), its guards removed",
     [(SS, "condition_2", [("if not all(k in m_e1 and k in m_e3 for k in SEEDS):", "if False:"),
                           ("mean_and_sds([m_e1[k] for k in SEEDS], [m_e3[k] for k in SEEDS])",
                            "mean_and_sds([m_e1[k] for k in SEEDS if k in m_e1 and k in m_e3], "
                            "[m_e3[k] for k in SEEDS if k in m_e1 and k in m_e3])"),
                           ("{k: m_e3[k] - m_e1[k] for k in SEEDS}",
                            "{k: m_e3[k] - m_e1[k] for k in SEEDS if k in m_e1 and k in m_e3}")]),
      (SS, "mean_and_sds", [("if len(e1) != 3 or len(e3) != 3:", "if False:"),
                            ("math.fsum(gains) / 3", "math.fsum(gains) / len(gains)")])],
     "seed:S8"),
    ("S13", "the missing-cell labels in the wrong order (candidate first)",
     [(SS, "_cell", [("for st in (b, c)", "for st in (c, b)")])], "seed:L7"),
    ("S14", "comparator values interpolated (a missing point filled)",
     [(SS, "_comparator", [('out["values"] = [{unit: u, "all_class_miou": have.get(u)}',
                            'out["values"] = [{unit: u, "all_class_miou": have.get(u, '
                            'next(iter(have.values()), None))}')])], "seed:W4"),
    ("S15", "the cut entry's comparator not flagged",
     [(SS, "cut_curves", [('hit[0]["cut_entry_comparator"] = True',
                           'hit[0]["cut_entry_comparator"] = False')])], "seed:W4"),
    ("S16", "a cut QAT run compared with its own seed, not seed 42",
     [(SS, "cut_curves", [("comps.append((stage, 42,", "comps.append((stage, k,")])], "seed:W5"),
    ("S17", "the robustness refusal fires whatever the pair's status (LRB11)",
     [(SS, "load_seed_inputs", [("if all(cells.has_model(st, seed) for st in D.CORRUPTION_STAGES):",
                                 "if True:")])], "seed:W5"),
    ("S18", "the seed-42 cross-check disabled",
     [(SS, "seed42_cross_checks", [("    if bad:", "    if False:")])], "mut:seed42_tamper"),
    ("S19", "a QAT selection reads its own kind, not its QAT run's (LRB5)",
     [(H, "resolve_status", [("re_, rk = _checked_kind(inv, run_slot)", "re_, rk = e, kind")])], "seed:L6"),
    ("S20", "a PTQ model with kind null ignores a diverged parent",
     [(H, "resolve_status", [('if pk in ("cut", "diverged"):', 'if pk in ("cut",):')])], "seed:L5"),
    ("S21", "slots without a family/per_seed role are no longer skipped (LRB4)",
     [(H, "record_slots", [("if not roles:", "if not roles and False:")])], "seed:W1"),
    ("S22", "fq_off rows checked against the stage precision (LRB7)",
     [(H, "load_bound", [("precision if precision == PRECISION_FQOFF else D.STAGE_PRECISION.get(",
                          "D.STAGE_PRECISION.get(")])], "report:Q3"),
    ("S23", "the non-finite label replaced (D-6)",
     [(H, "_label", [("return LABEL_NON_FINITE", "return LABEL_PARENT_CUT")])], "seed:L6"),
    ("S24", "the binding's sha256 comparison skipped",
     [(H, "read_inventory", [('if hashlib.sha256(braw).hexdigest() != b["sha256"]:', "if False:")])],
     "seed:R2e"),
    ("S25", "strata delta direction swapped",
     [(ST, "pair_table", [('"delta": (ic - ib) if evaluable else None',
                           '"delta": (ib - ic) if evaluable else None')])], "strata:H1"),
    ("S26", "the zero-IoU list includes classes already at 0",
     [(ST, "pair_table", [('r["iou_baseline"] > 0', 'r["iou_baseline"] >= 0')])], "strata:H17"),
    ("S27", "the zero-IoU list includes E1 -> E6 (D-8)",
     [(ST, "pair_table", [("if (b, c) in ZERO_IOU_PAIRS:", "if True:")])], "strata:H16"),
    ("S28", "not-evaluable classes summarised as 0",
     [(ST, "pair_table", [('_summary([r["delta"] for r in sel if r["status"] == OK],',
                           '_summary([(r["delta"] or 0.0) for r in sel],')])], "strata:H10"),
    ("S29", "GT-present float64 values computed with the union rule",
     [(AM6, "gt_present_row", [('_float64": rv[rule][space].value_float64',
                                '_float64": rv["union_present"][space].value_float64')])], "am6:F4"),
    ("S30", "a model row's entry_kind no longer checked against its slot's entry (IF§11.1)",
     [(H, "model_row", [('row.get("entry_kind") in SCORED_ENTRY_KINDS and row.get("entry_kind") == '
                         'e.get("kind")', "True")])], "seed:R2i"),
    ("S31", "a telemetry item's path no longer required before it is read (LRB2)",
     [(H, "read_telemetry", [(' or not _is_text(it.get("path"))', "")])], "seed:R17b"),
    ("S32", "the strict parse accepts a repeated JSON key",
     [(H, "strict_json_text", [("if len(set(keys)) != len(keys):", "if False:")])], "seed:R2j"),
    ("S33", "R-12 checked before R-13 (a seed-42 run of record without a model named parent_rule)",
     [(H, "planned_cells", [("check_seed42(status)\n    check_parentage(inv, slots, status)",
                             "check_parentage(inv, slots, status)\n    check_seed42(status)")])],
     "seed:R13b"),
    ("S34", "a comparator without a model never shows the VAL rows it wrote (AM-19 item 3(c))",
     [(SS, "_comparator", [("if slot is None or st.kind not in TELEMETRY_KINDS:", "if True:")])],
     "seed:W9"),
    ("S35", "the markdown drops the no-shared-point statement beside empty values (LRB6)",
     [(SS, "render_markdown", [('("; ".join(x for x in (vals, c["note"]) if x) or "—")',
                                '(vals or c["note"] or "—")')])], "seed:W10"),
    ("S36", "a missing run's reason drops its entry's decision-log rows (D-6)",
     [(SS.Missing, "text", [('{self.label}{rows}"', '{self.label}"')])], "seed:S21"),
    ("S37", "the markdown prints the slot key of each model (IF§2)",
     [(SS, "render_markdown", [("f\"{_fmt(r['miou_c'])} | {r['label'] or 'value'} |\")",
                                "f\"{_fmt(r['miou_c'])} | `{r['slot']}` | {r['label'] or 'value'} |\")")])],
     "seed:X6"),
    ("S38", "the JSON's limitation_divergences left empty (plan §5 a16)",
     [(SS, "build_tables", [('"limitation_divergences": [x.text() for x in divs if x.stage in ("E2", "E3")]',
                             '"limitation_divergences": []')])], "seed:W6"),
    ("S39", "a table cell's effect is baseline - candidate",
     [(SS, "_cell", [('{"value": values[(c, k)] - values[(b, k)]',
                      '{"value": values[(b, k)] - values[(c, k)]')])], "seed:F2"),
    ("S40", "mC pools each corruption's totals over its severities instead of averaging the cells",
     [(SS, "miou_c", [("cells = [m_of(cell_stats[(name, sev)]) for sev in INFERENTIAL_SEVERITIES]",
                       "cells = [pooled_miou_from_totals(*(sum(np.asarray(getattr(cell_stats[(name, s)], f), "
                       "np.int64) for s in INFERENTIAL_SEVERITIES) for f in ('dataset_tp', 'dataset_gt', "
                       "'dataset_pred')))]")])], "seed:F4"),
    ("S41", "condition 2 compares the mean gain with the smaller SD",
     [(SS, "condition_2", [("mean > max(s1, s3)", "mean > min(s1, s3)")])], "seed:F10"),
    ("S42", "strata: summary.per_class no longer checked against the npz (plan §2.1)",
     [(ST, "model_checks", [("if totals is not None:", "if False:")])], "strata:R6"),
    ("S43", "strata: the markdown appendix loses its per-class rows (plan §2.4)",
     [(ST, "render_markdown", [('for r in [p["background"], *p["rows"]]:', "for r in []:")])],
     "strata:E11"),
    ("S44", "strata: the pairs in reverse order",
     [(ST, "build_strata_tables", [("for b, c in PAIRS]", "for b, c in PAIRS[::-1]]")])], "strata:H20"),
    ("S45", "am6: the GT-present float32 values computed with the union rule",
     [(AM6, "gt_present_row", [('row = {f"{rule}_{space}": rv[rule][space].value',
                                'row = {f"{rule}_{space}": rv["union_present"][space].value')])], "am6:F4"),
    ("S46", "am6: the GT-present class counts taken from the union rule",
     [(AM6, "gt_present_row", [('_n_classes": rv[rule][space].n_classes',
                                '_n_classes": rv["union_present"][space].n_classes')])], "am6:F4"),
    ("S47", "am6: DL-99's E1 checkpoint record not checked",
     [(AM6, "build_am6_tables", [("H.check_e1_record({", "(lambda _r: None)({")])], "am6:R5"),
    ("S48", "defaults: the candidate's GT-present all-class value from the union rule",
     [(DC, "candidate_values", [('"candidate_gt_present_all_class": rv[GT_PRESENT]',
                                 '"candidate_gt_present_all_class": rv[UNION_PRESENT]')])], "defaults:V14"),
    ("S49", "defaults: the candidate's GT-present disease-only value from the union rule",
     [(DC, "candidate_values", [('"candidate_gt_present_disease_only": rv[GT_PRESENT]',
                                 '"candidate_gt_present_disease_only": rv[UNION_PRESENT]')])],
     "defaults:V14"),
    ("S50", "defaults: a coinciding row has 13 cells under the 12-column header",
     [(DC, "render_markdown", [('" |" * 9', '" |" * 10')])], "defaults:V11"),
    ("S51", "report: the --strata file's directory no longer guarded against the out-dir (D-14)",
     [(RX, "run", [("Path(strata).parent if strata is not None else None,", "None,")])], "report:P4"),
    ("S52", "report: the --mde-entry's directory no longer guarded against the out-dir (D-14)",
     [(RX, "run", [("Path(mde_entry).parent if mde_entry is not None else None]", "None]")])], "report:P4"),
    ("S53", "report: an fq_off row without its binding record reads 'not available' (LRB3)",
     [(RX, "build_fqoff", [("if not inv.has_row(slot, H.PRECISION_FQOFF):",
                            'if not (inv.has_row(slot, H.PRECISION_FQOFF) and '
                            'inv.has_record(slot, H.PRECISION_FQOFF, "fq_off", None)):')])], "report:Q5"),
    ("S54", "report: --parts fqoff no longer checks the 37's identities (R-11)",
     [(RX, "build_fqoff", [("H.check_seed42_identity(inv, cells, stats, stages=H.QAT_STAGES)", "None")])],
     "report:Q7"),
    ("S55", "report: the banner no longer names a split that is not TEST",
     [(RX, "banner", [('on = "" if split == "test" else f" Its inputs are the {split!r} split, not TEST."',
                       'on = ""')])], "report:F4"),
    ("S56", "report: the frozen code's provenance no longer gates the profile (O3)",
     [(RX, "official_unmet", [('unmet += [tuple(u) for u in code.get("report_code_unmet", [])]', "pass")])],
     "report:H8"),
    ("S57", "seed: the Conditions table loses its banner (D-13)",
     [(SS, "render_markdown", [('"### Conditions", "", banner,', '"### Conditions", "",')])], "report:F1"),
]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2)[-200:]}"


def run_mutated(targets, fn):
    with contextlib.ExitStack() as stack:
        for owner, name, pairs in targets:
            stack.enter_context(mutate(owner, name, *pairs))
        return run_case(fn)


def setup(root: Path) -> None:
    w = LF.build_world(root)
    CTX.world = w
    for mod in (S_SEED, S_ST, S_DC, S_AM6, S_RX):
        mod.setup(root, world=w)


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_mutations_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_lrep_mutations: temporary root {root} (left in place)", flush=True)
    setup(root)
    cases = all_cases()
    results = []
    covered = set()
    for mid, desc, targets, killer in MUTATIONS:
        for owner, name, pairs in targets:
            for old, _ in pairs:
                if old.startswith(("raise refusal(\"", "raise H.refusal(\"")):
                    covered.add(old.split('"')[1])
    results.append(("X1", "every refusal code has a mutation", set(H.CODES) == covered,
                    str(sorted(set(H.CODES) ^ covered))))
    killers = sorted({m[3] for m in MUTATIONS})
    base = {}
    for k in killers:
        ok, detail = run_case(cases[k])
        base[k] = ok
        results.append((f"K {k}", "the killing case passes unmutated", ok, detail))
    for mid, desc, targets, killer in MUTATIONS:
        try:
            ok, detail = run_mutated(targets, cases[killer])
        except AssertionError as e:                                    # an anchor that is not unique
            ok, detail = True, f"harness: {e}"
            results.append((mid, f"{desc} -- anchor", False, detail))
            continue
        killed = base[killer] and not ok
        results.append((mid, f"'{desc}' killed by {killer}", killed, "" if killed else detail))
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:400]}")
    good = sum(1 for r in results if r[2])
    ids = {m[0] for m in MUTATIONS}
    n_mut = sum(1 for r in results if r[0] in ids and r[2])
    print(f"\nRESULT: {'LREP MUTATIONS OK' if good == len(results) else 'LREP MUTATIONS FAILED'} "
          f"({good}/{len(results)}; mutations killed {n_mut}/{len(MUTATIONS)}; refusal codes "
          f"covered {len(covered & set(H.CODES))}/{len(H.CODES)}); {time.time() - t0:.0f} s")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
