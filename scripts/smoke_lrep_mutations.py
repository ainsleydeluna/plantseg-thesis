#!/usr/bin/env python3
"""The mutation harness of lane LREP (B-1 to B-3, B-5; LRB8): every mutation names the case that kills it.

Each mutation rewrites one LREP function from its own source (an exact, unique text replacement,
compiled into the module's namespace and restored afterwards; scripts/smoke_teacher_diag_mutations.py's
harness) and runs its named case, which must pass unmutated and fail under the mutation (a failed
check or an unexpected error both count as killed). Three kinds:

  C  one per refusal code of src.stats.heldout_inputs.CODES: its raise site disabled
     (`raise refusal(` -> `if False: raise refusal(`), killed by the case that reaches that code with
     every earlier gate passing (the cases of the lane's smokes, or the in-process cases below)
  R  a further raise site of a code (added by the DL-24 review's fixes, by B-5 or by B-5b), disabled the
     same way and killed by the case that reaches that site
  S  semantic mutations of the formulas, states, labels and rules (the plan's §7.1 list as LRB2-LRB13
     amend it, the DL-24 fixes, B-5's rulings: B5-D1 to B5-D4, B5-S1, B5-S2, B5-S4, the subset, (c), (d)
     and (e), and B-5b's ruling LREP-B5-RULE-4: R4-1's DT read, its pair, load_slot, the read column, the
     label and the guard; R4-2; R4-5's stages, roles, skips and diverged shape), each killed by its named
     case

The cases run on one synthetic world (scripts/lrep_fixtures.py) under a new directory in TMPDIR (left
in place), and scripts/smoke_lrep_official_profile.py's on its own official world in a sub-directory
(B-5b); no created path contains "test" or ":". A smoke that keeps a document between its cases (the
arms, dedup, quant and report smokes' the_doc, am6's cov_doc, the official smoke's inventory) has it
dropped before a mutated run of one of its cases and restored after, so that the mutant builds its own.

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
import smoke_arm_tables as S_ARM                                         # noqa: E402
import smoke_quant_rows as S_QR                                          # noqa: E402
import smoke_dedup_subset as S_DS                                        # noqa: E402
import smoke_default_candidates as S_DC                                  # noqa: E402
import smoke_lrep_descriptive as S_LD                                    # noqa: E402
import smoke_lrep_official_profile as S_OFF                              # noqa: E402
import smoke_report_extensions as S_RX                                   # noqa: E402
import smoke_seed_stability as S_SEED                                    # noqa: E402
import smoke_strata_tables as S_ST                                       # noqa: E402
from src.stats import am6_tables as AM6                                  # noqa: E402
from src.stats import arm_tables as AT                                   # noqa: E402
from src.stats import dedup_subset as DS                                 # noqa: E402
from src.stats import default_candidates as DC                           # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import lrep_descriptive as LD                             # noqa: E402
from src.stats import quant_rows as QR                                   # noqa: E402
from src.stats import report_extensions as RX                            # noqa: E402
from src.stats import seed_stability as SS                               # noqa: E402
from src.stats import strata_tables as ST                                # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402

CTX = types.SimpleNamespace(world=None)
#: The documents B-5's smokes keep between their cases, by case tag: dropped before a mutated run of one
#: of that smoke's cases (the mutant builds its own) and restored after it.
KEPT = {"arms": (S_ARM.CTX, "doc"), "dedup": (S_DS.CTX, "doc"), "quant": (S_QR.CTX, "doc"),
        "report": (S_RX.CTX, "doc"), "am6": (S_AM6.CTX, "cov"), "official": (S_OFF.CTX, "inv")}


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
# in-process cases for the report's codes, the seed-42 cross-check and B5-S1's map
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


def m_stage_unmapped():
    """B5-S1: a bound arm row read at a stage outside the map (E3-K8; K8 is out, DL-96) is refused once
    its row, record, path, protocol, files and manifest have passed."""
    w = CTX.world
    inv = H.open_inventory(w.inventory_path, policy=Policy.NONOFFICIAL_SMOKE)
    return refused("stage_unmapped", lambda: H.load_bound(
        inv, slot="E3-R1:s42:lambda0.5:alpha25:80k", stage="E3-K8", precision="fp32", scope="clean",
        input_root=w.input_root, policy=Policy.NONOFFICIAL_SMOKE))


def m_dedup_invalid():
    """B5-S2: an export whose dedup.sha256 is not its file's is refused before the file is parsed."""
    w = CTX.world
    export, binding = copy.deepcopy(w.export), copy.deepcopy(w.binding)
    export["dedup"]["sha256"] = LF.syn_sha("not the dedup file")
    path = LF.write_inventory(w.fresh("mut_dedup_sha"), export, binding, w.dedup)
    inv = H.open_inventory(path, policy=Policy.NONOFFICIAL_SMOKE)
    return refused("dedup_invalid", lambda: H.read_dedup(inv))


def m_stage_map():
    """B5-S1's static case (am6 M1) on the map as heldout_inputs._stage_map builds it now, not on the copy
    made at import: STAGE_MAP and STAGE_ORDER are set to it for the case and restored after."""
    fresh = H._stage_map()
    saved = H.STAGE_MAP, H.STAGE_ORDER
    H.STAGE_MAP, H.STAGE_ORDER = fresh, tuple(fresh)
    try:
        return S_AM6.m_stage_map()
    finally:
        H.STAGE_MAP, H.STAGE_ORDER = saved


LOCAL = {"mut:out_dir": m_out_dir, "mut:mde": m_mde, "mut:usage": m_usage,
         "mut:seed42_tamper": m_seed42_tamper, "mut:stage_unmapped": m_stage_unmapped,
         "mut:dedup_invalid": m_dedup_invalid, "mut:stage_map": m_stage_map}


def all_cases() -> dict:
    out = dict(LOCAL)
    for tag, mod in (("seed", S_SEED), ("strata", S_ST), ("defaults", S_DC), ("am6", S_AM6),
                     ("report", S_RX), ("arms", S_ARM), ("quant", S_QR), ("dedup", S_DS),
                     ("descriptive", S_LD), ("official", S_OFF)):
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


#: B-5b (R4-1): am6's coverage read by load_bound under the profile's policy instead of load_slot.
COVERAGE_BY_LOAD_BOUND = (
    "ba = H.load_slot(stats, inv, slot=slot, stage=stage, precision=precision, input_root=input_root)",
    'ba = H.load_bound(inv, slot=slot, stage=stage, precision=precision, scope="clean", '
    "input_root=input_root, policy=stats.policy)")


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
    ("C22", "precision_mismatch not raised",
     [(H, "load_bound", [off("precision_mismatch", ' f"{label}: (model_role, precision) is')])], "seed:R18"),
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
    ("C42", "stage_unmapped not raised (B5-S1)", [(H, "stage_spec", [off("stage_unmapped")])],
     "mut:stage_unmapped"),
    ("C43", "dedup_invalid not raised (the dedup file's sha256; B5-S2)",
     [(H, "read_dedup", [off("dedup_invalid", ' f"{where}: {f} has sha256')])], "mut:dedup_invalid"),
    ("C44", "arm_slot_invalid not raised (two slots for one (X, stage))",
     [(AT, "arm_slots", [off_h("arm_slot_invalid", ' f"two slots for one')])], "arms:P3"),
    ("C45", "arm_record_invalid not raised (a go-rule record not at its sha256)",
     [(AT, "read_record", [off_h("arm_record_invalid", ' f"{what} {rel}: sha256')])], "arms:T3"),
    ("C46", "quant_slot_invalid not raised (two slots for one stage)",
     [(QR, "quant_slots", [off_h("quant_slot_invalid", ' f"two quant_descriptive')])], "quant:R2"),
    # C: B-5b's codes (ruling LREP-B5-RULE-4 R4-1)
    ("C47", "descriptive_input_invalid not raised (the DT read's step (b): an official DT artifact)",
     [(H, "read_descriptive_test", [off("descriptive_input_invalid", ' f"{where}: (b)')])], "official:B1"),
    ("C48", "descriptive_input_in_official not raised (a DT run given to dedup's (ii)-(iii))",
     [(H, "require_official_inputs", [off("descriptive_input_in_official")])], "dedup:O1"),
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
     [(H, "check_rows", [off("precision_mismatch")])], "seed:R18c"),
    ("R10", "out_dir_invalid not raised by the out-dir check (a parent that does not exist)",
     [(RX, "check_out_dir", [off_h("out_dir_invalid", ' f"--out-dir {out_dir}: its parent')])], "report:P5"),
    ("R11", "seed_inconsistent not raised (an fq_off row that is not its int8_qat row's model)",
     [(RX, "fqoff_row", [off_h("seed_inconsistent")])], "report:Q6"),
    ("R12", "out_dir_invalid not raised by write_report (a parent that does not exist)",
     [(RX, "write_report", [off_h("out_dir_invalid", ' f"{final}: its parent')])], "report:P6"),
    # R: B-5's further raise sites (B5-S2's reader and invariants, (d), (a), (c))
    ("R13", "dedup_invalid not raised (an export dedup without its three fields)",
     [(H, "read_dedup", [off("dedup_invalid", ' f"{where} is not an object')])], "dedup:I3"),
    ("R14", "dedup_invalid not raised (a dedup file that cannot be read)",
     [(H, "read_dedup", [off("dedup_invalid", ' f"{where}: {f} cannot be read')])], "dedup:I5"),
    ("R15", "dedup_invalid not raised (a dedup file that is not UTF-8)",
     [(H, "read_dedup", [off("dedup_invalid", ' f"{where}: {f} is not UTF-8')])], "dedup:I7"),
    ("R16", "dedup_invalid not raised (kept_ids unsorted, repeated or empty)",
     [(H, "read_dedup", [off("dedup_invalid", ' f"{where}: kept_ids are not')])], "dedup:I9"),
    ("R17", "dedup_invalid not raised (the file's removed_by_reason is not the export's)",
     [(H, "read_dedup", [off("dedup_invalid", " f\"{where}: the file's removed_by_reason")])],
     "dedup:I10"),
    ("R18", "dedup_invalid not raised (a scored artifact's id set is not the 37's)",
     [(DS, "check_ids", [off_h("dedup_invalid", ' f"the scored artifacts do not share')])], "dedup:I12"),
    ("R19", "dedup_invalid not raised (a kept id outside the scored artifacts)",
     [(DS, "check_ids", [off_h("dedup_invalid", ' f"{len(outside)} kept id(s)')])], "dedup:I12"),
    ("R20", "dedup_invalid not raised (removed_by_reason counts fewer than removed)",
     [(DS, "check_ids", [off_h("dedup_invalid", ' f"{removed} images are not kept')])], "dedup:I12"),
    ("R21", "dedup_invalid not raised by am6's subset mode (a kept id outside the artifact)",
     [(AM6, "subset_values", [off_h("dedup_invalid", ' f"{len(missing)} kept id(s)')])], "am6:S3"),
    ("R22", "seed_inconsistent not raised (a coverage row's model row at another seed; R-14)",
     [(H, "check_row_at_entry", [off("seed_inconsistent")])], "am6:R8"),
    # R23 and R24 retired at B5b-1: R4-5(b) skips an arm slot at E3-K8 and an arm teacher outside
    # teacher-R1 and teacher-R2 (LRB4), so their raise sites are gone (arms:P2 and P4 now pin the skip)
    ("R25", "arm_slot_invalid not raised (a row the export has no slot for)",
     [(AT, "arm_slots", [off_h("arm_slot_invalid", ' f"the export has no slot')])], "arms:P5"),
    ("R26", "arm_slot_invalid not raised (an arm candidate at seed 43)",
     [(AT, "arm_slots", [off_h("arm_slot_invalid", ' f"slot {slot!r} at {stage} has seed')])], "arms:P7"),
    ("R27", "arm_record_invalid not raised (a go-rule record that is missing)",
     [(AT, "read_record", [off_h("arm_record_invalid", ' f"{what} {rel}: cannot be read')])], "arms:T5"),
    ("R28", "arm_record_invalid not raised (a band that is not √2·s; B5-D1)",
     [(AT, "seed_noise", [off_h("arm_record_invalid", ' f"√2·s =')])], "arms:T4"),
    ("R29", "arm_record_invalid not raised (a record at its sha256 whose fields are not R1's)",
     [(AT, "teacher_values", [off_h("arm_record_invalid", " f\"{x}'s go-rule record")])], "arms:T6"),
    ("R30", "artifact_for_missing_run not raised (a cut arm whose clean artifact is still bound; R-6)",
     [(AT, "arm_row", [off_h("artifact_for_missing_run")])], "arms:L5"),
    ("R31", "quant_slot_invalid not raised (a quant_descriptive slot outside the four stages)",
     [(QR, "quant_slots", [off_h("quant_slot_invalid", ' f"quant_descriptive slot {slot!r}')])],
     "quant:R1"),
    ("R32", "artifact_for_missing_run not raised (AM-23: not made, its clean artifact still bound)",
     [(QR, "build_quant_rows", [off_h("artifact_for_missing_run")])], "quant:L4"),
    # R: B-5's DL-24 fixes (B5-F1 to B5-F4)
    ("R33", "parent_rule not raised (a B-5 model whose parent run has none; R-12)",
     [(H, "_b5_with_model", [off("parent_rule")])], "arms:P8"),
    ("R34", "parent_rule not raised (AM-23's E2 parent that is not the family slot)",
     [(H, "_b5_parent", [off("parent_rule",
                             ' f"{slot!r} ({stage}): its parent {parent!r} is not the family')])],
     "quant:R5"),
    ("R35", "precision_mismatch not raised (a model row whose model_role is not its stage's; B5-S1)",
     [(H, "load_bound", [off("precision_mismatch", ' f"{label}: its model row')])], "am6:R10"),
    ("R36", "artifact_for_missing_run not raised by the fq_off table (an AM-23 slot without a model)",
     [(RX, "build_fqoff", [off_h("artifact_for_missing_run")])], "report:Q12"),
    # R: B-5b's further raise sites (ruling LREP-B5-RULE-4)
    ("R37", "quant_slot_invalid not raised (an AM-23 slot absent from the export; R4-2)",
     [(QR, "quant_slots", [off_h("quant_slot_invalid", ' f"the export has no quant_descriptive')])],
     "quant:R7"),
    ("R38", "descriptive_input_invalid not raised at (a) (an artifact the DT read's loader refuses)",
     [(H, "read_descriptive_test", [off("descriptive_input_invalid", ' f"{where}: (a)')])], "official:A1"),
    ("R39", "descriptive_input_invalid not raised at (f) (a record whose files differ; it falls to (a))",
     [(H, "read_descriptive_test", [off("descriptive_input_invalid", ' f"{where}: (f)')])], "official:F1"),
    ("R40", "descriptive_input_invalid not raised at (c) (an undefined value AM-5 does not exclude)",
     [(H, "read_descriptive_test", [off("descriptive_input_invalid", ' f"{where}: (c)')])], "official:C1"),
    ("R41", "descriptive_input_invalid not raised at (d) (another metric implementation)",
     [(H, "read_descriptive_test", [off("descriptive_input_invalid", ' f"{where}: (d)')])], "official:D2"),
    ("R42", "descriptive_input_invalid not raised at (e) (the reference's files changed)",
     [(H, "dt_pair", [off("descriptive_input_invalid", ' f"{where}: (e)')])], "official:E1"),
    # R: B-5b's DL-24 fixes (B5b-F1)
    ("R43", "descriptive_input_invalid not raised at (d) (a DT-classed slot with no family reference)",
     [(H, "load_slot", [off("descriptive_input_invalid", ' f"the DT read of')])], "official:D4"),
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
     [(H, "load_bound", [("if precision == PRECISION_FQOFF:", "if False:")])], "report:Q3"),
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
    # S: B-5 (B5-9)
    ("S58", "B5-D1: the reading's boundary exclusive (|Δ| < √2·s)",
     [(AT, "reading", [("abs(delta) <= scale", "abs(delta) < scale")])], "arms:S2"),
    ("S59", "B5-D1: the signed Δ read against √2·s, not |Δ|",
     [(AT, "reading", [("abs(delta) <= scale", "delta <= scale")])], "arms:S2"),
    ("S60", "B5-D1: the scale is s, not √2·s (its band check off)",
     [(AT, "seed_noise", [("scale = math.sqrt(2) * s", "scale = s"),
                          ('if not (scale == band == record["band"]):', "if False:")])], "arms:S1"),
    ("S61", "B5-D2: the per-image interval one-sided",
     [(LD, "mean_delta_task", [("BS.MEAN_DELTA, BS.TWO_SIDED,", "BS.MEAN_DELTA, BS.ONE_SIDED_LOWER,")])],
     "descriptive:M1"),
    ("S62", "B5-D2: the observed value is the median, not the A3a mean",
     [(LD, "mean_delta_task", [("float(engineering_shifts(d).mean_delta)", "float(np.median(d))")])],
     "descriptive:M1"),
    ("S63", "B5-D2: the default B typed, not bootstrap.PRODUCTION_B imported",
     [(LD, "mean_delta_task", [("B: int = BS.PRODUCTION_B)", "B: int = 10_000)")])], "descriptive:M2"),
    ("S64", "B5-D2: the arm rows' per-image vector all-class, not disease-only",
     [(AT, "arm_row", [("from .align import METRIC_DISEASE_ONLY,",
                        "from .align import METRIC_ALL_CLASS as METRIC_DISEASE_ONLY,")])], "arms:V5"),
    ("S65", "B5-D3: p <= 0.05 rejects (not strictly below)",
     [(DS, "unadjusted", [("bool(p < ALPHA)", "bool(p <= ALPHA)")])], "dedup:T1"),
    ("S66", "B5-D3: a None p-value reads as a False decision, not 'not evaluable'",
     [(DS, "unadjusted", [("return None if p is None", "return False if p is None")])], "dedup:T1"),
    ("S67", "B5-D3: a zero dataset-level difference counts as a sign disagreement",
     [(DS, "compare", [("== -1", "<= 0")])], "dedup:T3"),
    ("S68", "B5-D3: a different label no longer disagrees",
     [(DS, "compare", [('why.append("label")', "pass")])], "dedup:T3"),
    ("S69", "B5-D3: a subset that is not evaluable disagrees whatever the sign",
     [(DS, "compare", [("(True if opposite else None)", "True")])], "dedup:T3"),
    ("S70", "B5-D3: the pending caveat dropped from a not-rejected label",
     [(DS, "label_of", [('label = f"{label}; {RP.CAVEAT_PENDING}"', "pass")])], "dedup:T2"),
    ("S71", "B5-D3: report.contrast_label fed the difference's own type (np.float64)",
     [(DS, "label_of", [("float(difference)", "difference")])], "dedup:T2"),
    ("S72", "B5-D3: NI's disagreement inverted",
     [(DS, "build_dedup", [('"disagrees": ni.passed is not off_ni["passed"]',
                            '"disagrees": ni.passed is off_ni["passed"]')])], "dedup:N1"),
    ("S73", "B5-D3: the official decision read from Holm's adjusted p, not p_raw",
     [(DS, "build_dedup", [('holm[cid]["p_raw"]', 'holm[cid]["p_adjusted"]')])], "dedup:T4"),
    ("S74", "B5-D4: a trailing NUL after the statistic name",
     [(LD, "seed_bytes", [('+ statistic_name.encode("utf-8"))',
                           '+ statistic_name.encode("utf-8") + b"\\x00")')])], "descriptive:B1"),
    ("S75", "B5-D4: the seed bytes start with the official prefix",
     [(LD, "seed_bytes", [("return (SEED_PREFIX", 'return (b"plantseg-stats/1.0.0\\x00"')])],
     "descriptive:N4"),
    ("S76", "B5-D4: the entropy from the digest's last 16 bytes",
     [(LD, "seed_material", [("entropy = digest[:16]", "entropy = digest[16:]")])], "descriptive:B2"),
    ("S77", "B5-D4: the entropy read little-endian",
     [(LD, "seed_material", [('"big", signed=False', '"little", signed=False')])], "descriptive:B2"),
    ("S78", "B5-D4: the jackknife in reverse order (the loop is no longer run_bootstrap_task's)",
     [(LD, "run_task", [("jack[i] = jackknife_fn(i)", "jack[i] = jackknife_fn(n_images - 1 - i)")])],
     "descriptive:L1"),
    ("S79", "B5-D4: a contrast id without '__' accepted",
     [(LD, "check_contrast_id", [("not CONTRAST_ID_RE.fullmatch(contrast_id)",
                                  'not re.fullmatch(r"[a-z0-9_]+", contrast_id)')])], "descriptive:N6"),
    ("S80", "B5-S1: E6-KD read at evaluator stage E6 (B-4's retired assumption)",
     [(H, "stage_spec", [("STAGE_MAP.get(stage)", 'STAGE_MAP.get("E6" if stage == "E6-KD" else stage)')])],
     "am6:V4"),
    ("S81", "B5-S1: the teachers expected as students",
     [(H, "stage_spec", [("return spec", 'return (spec[0], "student", spec[2])')])], "am6:V4"),
    ("S82", "B5-S1: the map built with E6-KD at E6",
     [(H, "_stage_map", [("(*QUANT_STAGES, E6KD_STAGE, *EXPLORATORY_STAGES)",
                          "(*QUANT_STAGES, *EXPLORATORY_STAGES)"),
                         ("    return out",
                          '    out[E6KD_STAGE] = ("E6", STUDENT_ROLE, D.STAGE_PRECISION["E6"])\n'
                          "    return out")])],
     "mut:stage_map"),
    ("S83", "B5-S1: E7-X mapped at E3's precision, not E7's",
     [(H, "_stage_map", [("D.STAGE_PRECISION[parent])", 'D.STAGE_PRECISION["E3"])')])], "mut:stage_map"),
    ("S84", "B5-S2: unsorted kept ids accepted (T4: sorted and unique)",
     [(H, "read_dedup", [("ids == sorted(set(ids))", "len(ids) == len(set(ids))")])], "dedup:I9"),
    ("S85", "B5-S2: dedup.path resolved against the directory above the export's (H:10)",
     [(H, "read_dedup", [('f = inv.path.parent / d["path"]', 'f = inv.path.parent.parent / d["path"]')])],
     "dedup:I1"),
    ("S86", "(b): the subset's delta not restricted by id (its first rows)",
     [(DS, "restrict_pair", [("delta=pv.delta[mask]", "delta=pv.delta[:int(mask.sum())]")])], "dedup:C2"),
    ("S87", "(b): the full pair's Am5Pair kept on the subset",
     [(DS, "restrict_pair", [("am5 = pv.am5 if pv.am5 is None else am5_on_kept(*runs, kept)",
                              "am5 = pv.am5")])], "dedup:C6"),
    ("S88", "(b): the Am5Pair counts the removed images' exclusions",
     [(DS, "am5_on_kept", [("excluded = tuple(i for i in run.am5.excluded_ids if i in kept)",
                            "excluded = tuple(run.am5.excluded_ids)")])], "dedup:C6"),
    ("S89", "(b): the pooled rows not restricted (the full pooled delta)",
     [(DS, "restrict_pooled", [("rows = np.asarray(sorted(pos[i] for i in kept), dtype=np.int64)",
                                "rows = np.arange(len(manifest_ids), dtype=np.int64)")])], "dedup:C3"),
    ("S90", "(b): mC from every row's totals, not the kept rows'",
     [(DS, "miou_c_on_kept", [("totals[key] = types.SimpleNamespace(",
                               "totals[key] = st or types.SimpleNamespace(")])], "dedup:C4"),
    ("S91", "(b): robustness_e1_e6's vector not restricted to the kept ids",
     [(DS, "subset_contrast", [("pv = restrict_pair(align_miou_c(mc[b], mc[c], policy=policy), kept)",
                                "pv = align_miou_c(mc[b], mc[c], policy=policy)")])], "dedup:C4"),
    ("S92", "(b): a scalar interval's observed value is the mean Δ, not its statistic's",
     [(DS, "subset_interval", [("BS.comparison_observed(result, spec.statistic_name)",
                                "float(np.mean(delta))")])], "dedup:C5"),
    ("S93", "(b)(iii): the NI bound two-sided",
     [(DS, "subset_noninferiority", [("BS.ONE_SIDED_LOWER, ps.n_images", "BS.TWO_SIDED, ps.n_images")])],
     "dedup:N1"),
    ("S94", "(b)(i): the arms, teachers and AM-23 slots left out of the subset rows",
     [(DS, "model_rows", [("set(roles) & set(SUBSET_ROLES)", 'set(roles) & {"family", "per_seed"}')])],
     "dedup:V1"),
    ("S95", "(b)(i): E6-KD's row left out (its role is none of the six)",
     [(DS, "model_rows", [(' or r["stage"] == H.E6KD_STAGE', "")])], "dedup:V1"),
    ("S96", "(d): the planned slots' records read again as coverage rows",
     [(AM6, "coverage_items", [(' and r.get("slot") not in rec_slots', "")])], "am6:V1"),
    ("S97", "(d): a coverage row's model row checked at its stage only, not its seed (R-14)",
     [(H, "check_row_at_entry", [(' or ba.row["seed"] != e.get("seed")', "")])], "am6:R8"),
    ("S98", "(d): the seed-42 faithfulness check applied to the coverage rows at the family's stages",
     [(AM6, "build_am6_tables", [('if r.get("role") != "family":',
                                  'if r.get("role") == "per_seed" or r["stage"] not in ref:')])], "am6:V1"),
    ("S99", "(d): a null seed sorted first",
     [(AM6, "_sort_key", [('return (r["seed"] is None,', 'return (r["seed"] is not None,')])], "am6:V5"),
    ("S100", "(d): DL-99's E1 record checked over every E1 row (a 160k control's among them)",
     [(AM6, "build_am6_tables", [(' and ident.get("role") in PLANNED_ROLES', "")])], "am6:V1"),
    ("S101", "(d): the coverage rows lose their slots' roles",
     [(AM6, "coverage_items", [('"roles": list(e["role"])', '"roles": []')])], "am6:V2"),
    ("S102", "(d): the planned table prints the coverage rows too",
     [(AM6, "render_markdown", [("if not is_planned(r):", "if False:")])], "am6:V6"),
    ("S103", "(e) R1-12: a null freeze accepted under Policy.OFFICIAL",
     [(H, "open_inventory", [("(freeze is None and policy is Policy.OFFICIAL)", "False")])], "seed:R2k"),
    ("S104", "(e) R2-04: a cut run's attempt without VAL rows has no note",
     [(SS, "cut_curves", [('"note": None if a.points else WROTE_NO_VAL_ROWS}', '"note": None}')])],
     "seed:W11"),
    ("S105", "(e) R2-04: a comparator's attempt without VAL rows has no note",
     [(SS, "_comparator", [('"note": None if a.points else WROTE_NO_VAL_ROWS}', '"note": None}')])],
     "seed:W11"),
    ("S106", "(e) R2-09: a comparator's non-finite tag dropped",
     [(SS, "_comparator", [('v["nonfinite"] = tags.get(v[unit])', 'v["nonfinite"] = None')])], "seed:W12"),
    ("S107", "(e) R2-09: the markdown prints a non-finite row as the absent mark",
     [(SS, "_val", [('if v.get("nonfinite") else', "if False else")])], "seed:W12"),
    ("S108", "(e) R2-09: TelemetryAttempt.tags returns no tag",
     [(H.TelemetryAttempt, "tags", [("if t is not None}", "if False}")])], "seed:W12"),
    ("S109", "B5-S4: the am6 part built without (d)'s coverage",
     [(RX, "build_report", [("coverage=True)", "coverage=False)")])], "report:S3"),
    ("S110", "B5-S4: the quant part's section loses 'descriptive; outside protocol 1.0.0'",
     [(QR, "render_markdown", [("f\"> {doc['banner']}\", ", "")])], "report:S2"),
    ("S111", "B5-S4: the dedup part's caveat not read from the MDE part (B5-D3)",
     [(RX, "build_report", [('power_caveat=mde["power_caveat"],', 'power_caveat="pending",')])],
     "report:S5"),
    ("S112", "B5-D2: a B other than PRODUCTION_B no longer an unmet official condition",
     [(RX, "official_unmet", [('p["B"] != PRODUCTION_B:', "False:")])], "report:S7"),
    ("S113", "(c): the quant_descriptive QAT slots' fq_off rows left out",
     [(RX, "build_fqoff", [('"quant") for stage in H.QUANT_STAGES', '"quant") for stage in ()')])],
     "report:Q9"),
    ("S114", "(c): E2 − E3 oriented as the official delta (E3 − E2)",
     [(QR, "build_quant_rows", [('differences(sides["E2"], sides["E3"])',
                                 'differences(sides["E3"], sides["E2"])')])], "quant:Q2"),
    # S: B-5's DL-24 fixes (B5-F2 to B5-F4)
    ("S115", "(a): an arm candidate's model row not checked against its entry (R-14)",
     [(AT, "arm_row", [("H.check_row_at_entry(inv, ba)", "None")])], "arms:P9"),
    ("S116", "(c): an AM-23 candidate's model row not checked against its entry (R-14)",
     [(QR, "build_quant_rows", [("H.check_row_at_entry(inv, ba)", "None")])], "quant:R6"),
    ("S117", "(c): the fq_off table's E6-KD and AM-23 model rows not checked against their entries (R-14)",
     [(RX, "build_fqoff", [("H.check_row_at_entry(inv, ba)", "None")])], "report:Q11"),
    ("S118", "(d): a coverage row's model row not checked against its entry (R-14)",
     [(AM6, "coverage_items", [("H.check_row_at_entry(inv, ba)", "None")])], "am6:R8"),
    ("S119", "(a): an arm slot at E6-X placed as the E6-X candidate",
     [(AT, "arm_slots", [('for x, st in ROWS if st != "E6"}', "for x, st in ROWS}")])], "arms:P10"),
    ("S120", "(a): the E6-X candidates read without quant_slots' rule",
     [(AT, "arm_slots", [("quant = quant_slots(inv)",
                          'quant = {inv.entry(s)["stage"]: s '
                          'for s in H.role_slots(inv, "quant_descriptive")}')])], "arms:P11"),
    # S121 retired at B5b-3: R4-2 refuses an AM-23 slot absent from the export, so the fq_off table's
    # NO_SLOT row is gone (R37, S129 and S130 pin the refusal)
    ("S122", "B5-D3: a subset that is not evaluable skips the sign test",
     [(DS, "compare", [("(True if opposite else None)", "None")])], "dedup:T3"),
    ("S123", "(b)(i): the part reads every role's coverage rows (LRB4)",
     [(DS, "model_rows", [("roles=SUBSET_ROLES, stages=(H.E6KD_STAGE,)", "roles=None")])], "dedup:V5"),
    ("S124", "(b)(i): am6's coverage role filter ignored",
     [(AM6, "coverage_items", [("if roles is not None and not", "if False and not")])], "dedup:V5"),
    ("S125", "(e) R2-04: the comparator's attempt of record without VAL rows has no note",
     [(SS, "_comparator", [("if not a.points:", "if False:")])], "seed:W13"),
    ("S126", "(e) R2-09: a no-model comparator's attempt drops its rows' non-finite tags",
     [(SS, "_comparator", [('"nonfinite": tags.get(u)} for u in points]',
                            '"nonfinite": None} for u in points]')])], "seed:W14"),
    ("S127", "(e) R2-09: a cut run's own points drop their non-finite tags",
     [(SS, "cut_curves", [('"nonfinite": t}', '"nonfinite": None}')])], "seed:W14"),
    ("S128", "(d): an fq_off record filed under scope clean read as a coverage row",
     [(AM6, "coverage_items", [(' and r.get("precision") != H.PRECISION_FQOFF', "")])], "am6:V7"),
    # S: B-5b (ruling LREP-B5-RULE-4)
    ("S129", "R4-2: the quant part requires none of the four AM-23 slots",
     [(QR, "build_quant_rows", [("quant_slots(inv, required=H.QUANT_STAGES)", "quant_slots(inv)")])],
     "quant:R7"),
    ("S130", "R4-2: the fq_off table requires none of AM-23's QAT slots",
     [(RX, "build_fqoff", [("quant_slots(inv, required=FQOFF_QUANT_STAGES)", "quant_slots(inv)")])],
     "report:Q10"),
    # S: B-5b, R4-1 (B5b-2): the DT read, its pair, load_slot, the read column, the guard
    ("S131", "R4-1 (b): an official DT artifact accepted",
     [(H, "read_descriptive_test", [('(ident.artifact_status == "provisional",',
                                     '(ident.artifact_status in ("provisional", "official"),')])],
     "official:B1"),
    ("S132", "R4-1 (b): a DT artifact on another split accepted",
     [(H, "read_descriptive_test", [('(ident.split == "test",', "(True,")])], "official:B1"),
    ("S133", "R4-1 (b): a random_init DT artifact accepted",
     [(H, "read_descriptive_test", [("(ident.random_init is False,", "(True,")])], "official:B1"),
    ("S134", "R4-1 (b): a DT artifact of another row count accepted",
     [(H, "read_descriptive_test", [("(ident.expected_rows == ident.actual_rows == EXPECTED_ROWS_TEST,",
                                     "(ident.expected_rows == ident.actual_rows,")])], "official:B1"),
    ("S135", "R4-1 (c): the DT rows under NONOFFICIAL_SMOKE's rule (nothing dropped, nothing aborts)",
     [(H, "read_descriptive_test", [("apply_am5_policy(ba.run.records, Policy.OFFICIAL)",
                                     "apply_am5_policy(ba.run.records, Policy.NONOFFICIAL_SMOKE)")])],
     "official:C1"),
    ("S136", "R4-1 (d): a reference not read under Policy.OFFICIAL accepted",
     [(H, "read_descriptive_test", [("ref.policy is not Policy.OFFICIAL or ", "")])], "official:D1"),
    ("S137", "R4-1 (d): a reference of another evaluator stage accepted",
     [(H, "read_descriptive_test", [(" or ref.identity.stage != ev_stage", "")])], "official:D1"),
    ("S138", "R4-1 (d): a reference that is not the 37's driver.LoadedInput accepted",
     [(H, "read_descriptive_test", [("or not isinstance(reference, D.LoadedInput)", "or False")])],
     "official:D1"),
    ("S139", "R4-1 (d): assert_compatible under NONOFFICIAL_SMOKE (a metric difference only warns)",
     [(H, "read_descriptive_test", [("run.identity, Policy.OFFICIAL)",
                                     "run.identity, Policy.NONOFFICIAL_SMOKE)")])],
     "official:D2"),
    ("S140", "R4-1 (d): am5_pair not called",
     [(H, "read_descriptive_test", [("am5_pair(ref, run)", "None")])], "official:D3"),
    ("S141", "R4-1 (e): the reference's files not compared with its official read",
     [(H, "dt_pair", [("if dict(again.file_sha256) != dict(reference.file_sha256):", "if False:")])],
     "official:E1"),
    ("S142", "R4-1 (e): the reference re-loaded under Policy.OFFICIAL",
     [(H, "dt_pair", [("again = D.load_input(",
                       "again = (lambda *a: D.load_input(*a[:4], Policy.OFFICIAL))(")])],
     "official:H2"),
    ("S143", "R4-1 (e): the re-loaded reference keeps its AM-5 rows (not (c)'s)",
     [(H, "dt_pair", [("records=apply_am5_policy(again.run.records, Policy.OFFICIAL))",
                       "records=again.run.records)")])], "official:H2"),
    ("S144", "R4-1 (e): the DT pair aligned under Policy.OFFICIAL",
     [(H, "dt_pair", [("policy=Policy.NONOFFICIAL_SMOKE, metric=metric)",
                       "policy=Policy.OFFICIAL, metric=metric)")])],
     "official:H2"),
    ("S145", "R4-1 item 3: a DT slot under Policy.OFFICIAL read by load_bound, never the DT read",
     [(H, "load_slot", [("if dt_read(stats.policy, read):", "if False:")])], "official:H1"),
    ("S146", "R4-1 (d): every DT slot's reference the 37's E7 (the E4 lists' too)",
     [(H, "load_slot", [("reference=stats.clean(ev_stage))", 'reference=stats.clean("E7"))')])],
     "official:H1"),
    ("S147", "R4-1 item 2: load_slot drops the slot's read (a DT slot reads official in the smoke profile)",
     [(H, "load_slot", [("return dataclasses.replace(ba, read=read)", "return ba")])], "arms:G1"),
    ("S148", "R4-1 item 3: the DT read in the smoke profile too",
     [(H, "dt_read", [("return policy is Policy.OFFICIAL and read == READ_DESCRIPTIVE_TEST",
                       "return read == READ_DESCRIPTIVE_TEST")])], "arms:G1"),
    ("S149", "R4-1 item 2: role calibration_sensitivity no longer reads descriptive_test",
     [(H, "slot_read", [("if DT_ROLE in roles or (", "if (")])], "official:G1"),
    ("S150", "R4-1 item 2: every arm slot reads descriptive_test, whatever its stage's precision",
     [(H, "slot_read", [("spec[2] == DT_ARM_PRECISION", "True")])], "official:G1"),
    ("S151", "R4-1 item 6: a DT read prints no label",
     [(H, "read_label", [("return DT_LABEL if read == READ_DESCRIPTIVE_TEST else None", "return None")])],
     "arms:G1"),
    ("S152", "R4-1 item 5: the guard compares the 37's LoadedInputs only, not their runs",
     [(H, "require_official_inputs", [("for x in (li, li.run)", "for x in (li,)")])], "dedup:O1"),
    ("S153", "R4-1 item 5: dedup's (ii)-(iii) take their inputs unguarded",
     [(DS, "build_dedup", [("H.require_official_inputs(stats, {", "(lambda *a: None)(stats, {")])],
     "dedup:O1"),
    ("S154", "R4-1 (e): the arm rows pair a DT run with align_runs under the profile's policy",
     [(AT, "arm_row", [("if H.dt_read(stats.policy, ba.read):", "if False:")])], "official:H4"),
    ("S155", "R4-1 (e): an arm row records its pair's policy as official",
     [(AT, "arm_row", [('"pair_policy": pv.policy.value,', '"pair_policy": "official",')])], "arms:G1"),
    ("S156", "R4-1 item 6: an arm row from a DT run carries no label",
     [(AT, "arm_row", [("read_label=ba.read_label,", "read_label=None,")])], "arms:G1"),
    ("S157", "R4-1 item 6: the arm table's candidate cell prints no label",
     [(AT, "render_markdown", [(' if r.get("read_label") else ""', ' if False else ""')])], "arms:G1"),
    ("S158", "R4-1: am6's coverage read by load_bound (its rows lose their read)",
     [(AM6, "coverage_items", [COVERAGE_BY_LOAD_BOUND])], "am6:V8"),
    ("S159", "R4-1 item 6: am6's coverage rows from a DT run carry no label",
     [(AM6, "coverage_items", [('"read_label": ba.read_label,', '"read_label": None,')])], "am6:V8"),
    ("S160", "R4-1 item 6: am6's coverage table prints no label",
     [(AM6, "render_markdown", [('stage = r["stage"] if not r.get("read_label") else ',
                                 'stage = r["stage"] if True else ')])], "am6:V8"),
    ("S161", "R4-1 items 2 and 6: dedup's (i) rows lose their read and label",
     [(DS, "model_rows", [('"read": r.get("read", H.READ_OFFICIAL), "read_label": r.get("read_label"),',
                           '"read": H.READ_OFFICIAL, "read_label": None,')])], "dedup:O2"),
    ("S162", "R4-1 item 6: dedup's (i) table prints no label",
     [(DS, "render_markdown", [('stage = m["stage"] if not m.get("read_label") else ',
                                'stage = m["stage"] if True else ')])], "dedup:O2"),
    # S: B-5b, R4-5 (B5b-1): the catalogue's stages, roles and diverged shape
    ("S163", "R4-5(a): the arm teachers mapped at B5-S1's retired stages R1 and R2",
     [(H, "_stage_map", [("for st in TEACHER_STAGES:", 'for st in ("teacher", "R1", "R2"):')])],
     "mut:stage_map"),
    ("S164", "R4-5(c): role e6kd outside the model-row role check",
     [(H, "model_row", [('set(row["role"]) <= set(ROLES)', 'set(row["role"]) <= set(ROLES) - {"e6kd"}')])],
     "am6:V4"),
    ("S165", "R4-5(f): a divergence without both unit keys accepted",
     [(H, "_kind_fields_problem", [('if "iteration" not in e or "epoch" not in e:', "if False:")])],
     "arms:L8"),
    ("S166", "R4-5(f): a divergence with both units null or both non-null accepted",
     [(H, "_kind_fields_problem", [("if len(units) != 1 or ", "if ")])], "arms:L9"),
    ("S167", "R4-5(f): a divergence without its run_id or cause accepted",
     [(H, "_kind_fields_problem", [('("run_id", "rule", "cause")', '("rule",)')])], "arms:L9"),
    ("S168", "R4-5(f): a divergence without its telemetry_sha256 accepted",
     [(H, "_kind_fields_problem", [('if not (isinstance(e.get("telemetry_sha256"), str)',
                                    'if False and (isinstance(e.get("telemetry_sha256"), str)')])],
     "arms:L9"),
    ("S169", "R4-5(f): a divergence's label always by iteration",
     [(H, "_label", [('unit = "iteration" if e["iteration"] is not None else "epoch"',
                      'unit = "iteration"')])],
     "arms:L10"),
    ("S170", "R4-5(b): a skipped scored slot labelled",
     [(H, "entry_label", [("return None", 'return "scored"')])], "arms:P12"),
    ("S171", "R4-5(b): a skipped not_run slot's label loses its reason",
     [(H, "entry_label", [("return f\"not run: {e.get('reason')}\"", 'return "not run"')])], "arms:P12"),
    ("S172", "R4-5(b): a skipped cut slot labelled by its kind only",
     [(H, "entry_label", [(" and _kind_fields_problem(e) is None:", " and False:")])], "arms:P12"),
    ("S173", "R4-5(f): the seed part's divergences always by iteration",
     [(SS, "divergences", [('unit = "iteration" if st.record.get("iteration") is not None else "epoch"',
                            'unit = "iteration"')])], "seed:W15"),
    ("S174", "R4-5(a): the arm teachers looked for at the retired stages R1 and R2",
     [(AT, "arm_slots", [("if stage not in ARM_TEACHER_STAGES:", 'if stage not in ("R1", "R2"):')])],
     "arms:P13"),
    ("S175", "R4-5(a): X not taken from the arm teacher's stage name",
     [(AT, "teacher_x", [('return stage.split("-", 1)[1]', "return stage")])], "arms:P13"),
    ("S176", "R4-5(b): an arm slot outside E3-X and E7-X no longer skipped",
     [(AT, "arm_slots", [("if stage not in want:", "if False:")])], "arms:P12"),
    ("S177", "R4-5(b): an arm_teacher slot outside teacher-R1 and teacher-R2 no longer skipped",
     [(AT, "arm_slots", [("if stage not in ARM_TEACHER_STAGES:", "if False:")])], "arms:P12"),
    ("S178", "R4-5(b): the skipped slots listed without their labels",
     [(AT, "build_arm_tables", [('"label": H.entry_label(e)})', '"label": None})')])], "arms:P12"),
    ("S179", "R4-5(b): the markdown no longer names the skipped slots",
     [(AT, "render_markdown", [('if doc["skipped"]:', "if False:")])], "arms:P12"),
    # S: B-5b, R4-1 item 8 (B5b-4) and the cases of items 3(e), 4 and the official coverage
    ("S180", "R4-1 item 8: report_extensions.run drops its B (the parts at PRODUCTION_B)",
     [(RX, "run", [("expect_inventory_sha256=expect_inventory_sha256, B=B)",
                    "expect_inventory_sha256=expect_inventory_sha256)")])], "report:S11"),
    ("S181", "R4-1 item 4: every slot under Policy.OFFICIAL read through the DT read (a downgrade path)",
     [(H, "load_slot", [("if dt_read(stats.policy, read):", "if stats.policy is Policy.OFFICIAL:")])],
     "official:K1"),
    ("S182", "R4-1 (e): align_runs' refusal escapes the DT pair unnamed",
     [(H, "dt_pair", [("except (D.DriverRefusal, IngestError, AlignmentError) as e:",
                       "except (D.DriverRefusal, IngestError) as e:")])], "official:E2"),
    ("S183", "R4-1: am6's coverage read by load_bound under Policy.OFFICIAL (its DT slots refused)",
     [(AM6, "coverage_items", [COVERAGE_BY_LOAD_BOUND])], "official:H5"),
    # S: B-5b's DL-24 fixes (B5b-F1)
    ("S184", "R4-1 item 5: am6's seed-42 rows take their inputs unguarded (a DT run as E7's)",
     [(AM6, "build_am6_tables", [("H.require_official_inputs(stats, {", "(lambda *a: None)(stats, {")])],
     "am6:R11"),
]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2)[-200:]}"


def run_mutated(targets, fn, killer=""):
    """Run the killing case under the mutation, its smoke's kept document dropped for the run (KEPT)."""
    kept = KEPT.get(killer.split(":")[0])
    with contextlib.ExitStack() as stack:
        for owner, name, pairs in targets:
            stack.enter_context(mutate(owner, name, *pairs))
        saved = getattr(*kept) if kept else None
        if kept:
            setattr(*kept, None)
        try:
            return run_case(fn)
        finally:
            if kept:
                setattr(*kept, saved)


def setup(root: Path) -> None:
    w = LF.build_world(root)
    CTX.world = w
    for mod in (S_SEED, S_ST, S_DC, S_AM6, S_RX, S_ARM, S_QR, S_DS, S_LD):
        mod.setup(root, world=w)
    S_OFF.setup(LF.new_dir(root, "official"))          # its own official world (R4-1 item 8)


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
            ok, detail = run_mutated(targets, cases[killer], killer)
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
