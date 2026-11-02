#!/usr/bin/env python3
"""LREP's conformance with TMAN's committed formats (lane LREP, B-4: ruling LB4-13, with LB4-A1, LB4-3 and LB4-4).

SYNTHETIC ONLY (DL-41). TMAN's own fixture world and builders, by import, never hand-copied
(scripts/lrep_fixtures.tman_world and tman_exports: scripts/smoke_tman_validate.World, src.tman.manifest.build_inventory
with its freeze record, smoke_tman_validate.dedup_fixture and scripts/smoke_tman_binding.fixture_binding), under a new
directory in TMPDIR, left in place. TMAN's bound export, which src.tman.validate.read_inventory accepts, is read at
reader level: no report runs on artifacts. The catalogue is the checkout's, read through TMAN's own expression,
schema.load_catalogue of manifest.CODE_ROOT / schema.CATALOGUE_REL. No created path contains "test" or ":".

  I  the imports: heldout_inputs' and report_extensions' own imports load none of src.tman.validate, schema, manifest
     and dedup, and no src.training (B-4 imports TMAN's modules when called: LB4-1, LB4-5, LB4-7, LB4-8); each case
     also names whether torch and src.eval.artifacts are loaded, as they already were at MBL (B-4's S2)
  E  end to end at reader level on TMAN's accepted bound export: open_inventory with TMAN's reader, its default, under
     Policy.NONOFFICIAL_SMOKE and Policy.OFFICIAL; every model row through model_row and check_row_at_entry;
     read_dedup and check_ids against the file's own id set; E5:s42:sel's parent under LB4-8, with the 21 planned
     cells (R-3 to R-6, R-12 to R-14); the records indexed by (slot, precision, scope, cell); LB4-3's paths against the
     export's folder, on both sides; LB4-4's binding file; LREP's own checks after TMAN's reader returns (LB4-1)
  C  the checkout's catalogue: B5-S1's map against every scored slot's evaluator_stage, model_role and precision
     (49/49, none unmapped); R4-1's DT set is exactly Part 3 §5's nine
  D  drift guards against TMAN's definitions, imported: LREP's own copies asserted equal to them (ROLES, the
     fixtures' SLOT_KEY_RE with its flags, the format strings, the dedup fields, the keys model_row reads against
     ROW_SPEC, the fixtures' catalogue copy against the checkout's), since heldout_inputs' import loads no TMAN
     schema (LB4-A3); row_identity is TMAN's binding.identity itself, checked on every TMAN-built row
  A  LB4-A1: scripts/smoke_tman_parent.py at MBL refuses no arm or quant_descriptive descendant (B-4's S2), so these
     two go through TMAN's validator, its freeze gates on TMAN's world: an arm descendant (E7-R1) and a
     quant_descriptive descendant (E6-E2's selection), each entered derived under its cut parent, each
     parent_rule_violation with every earlier gate passing
  X  no created path contains 'test' or ':', under this smoke's root and TMAN's world's folders
LREP reads neither catalogue.never_cut nor catalogue.cut_place (LB4-13); nothing here tests them.

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_lrep_conformance.py
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import os
import shutil
import subprocess
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
from src.stats import am6_tables as AM6                                  # noqa: E402
from src.stats import dedup_subset as DS                                 # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402

CTX = types.SimpleNamespace(root=None, world=None, path=None, doc=None, inv=None, plain=None)
#: R4-1's DT set, typed from Part 3 §5's table (docs/lane_reports/lrep.md :2304-2305): the nine slots the evaluator
#: scores descriptively only (role arm at int8_ptq; role calibration_sensitivity).
DT_NINE = ("E7-R1:s42:list42", "E7-R2:s42:list42", "E7-flip:s42:list42", "E4:s42:list43", "E4:s42:list44",
           "E4:s42:list45", "E7:s42:list43", "E7:s42:list44", "E7:s42:list45")
#: Part 3 §5 (lrep.md :2296): the catalogue's scored slots, evaluator_stage not null.
SCORED_SLOTS = 49
#: What B-4's imports, made when called, keep out of heldout_inputs' and report_extensions' own imports.
NOT_AT_IMPORT = ("src.tman.validate", "src.tman.schema", "src.tman.manifest", "src.tman.dedup")
E5SEL = "E5:s42:sel"
#: LB4-A1's descendants: an arm's PTQ model under its cut arm run (from a derived PTQ entry of the world), and AM-23's
#: QAT selection under its cut QAT run (from the world's derived selection E6:s42:sel).
ARM_CHILD, ARM_PARENT, ARM_SOURCE = "E7-R1:s42:list42", "E3-R1:s42:lambda0.5:alpha25:80k", "E7:s42:list42"
QUANT_CHILD, QUANT_SOURCE = "E6-E2:s42:sel", "E6:s42:sel"
WANT_GATE = "parent_rule_violation"


def refused(code, fn, head="", contains=""):
    try:
        fn()
    except H.LrepRefusal as e:
        return (e.code == code and e.detail.startswith(head) and contains in e.detail), f"{e.code}: {e.detail[:240]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


def catalogue():
    """The checkout's catalogue through TMAN's own expression (heldout_inputs.catalogue_pilot_runs reads it so)."""
    from src.tman import manifest as M
    from src.tman import schema as S
    return S.load_catalogue((M.CODE_ROOT / S.CATALOGUE_REL).read_bytes(), S.CATALOGUE_REL)


def scored_slots() -> list:
    return [s for s in catalogue().slots.values() if s["evaluator_stage"] is not None]


# --------------------------------------------------------------------------------------------------
# I: the imports
# --------------------------------------------------------------------------------------------------
def imported(module: str) -> set:
    """The modules a fresh interpreter holds after importing `module` alone (this environment, -B)."""
    code = f"import json, sys; import {module}; print(json.dumps(sorted(sys.modules)))"
    p = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(REPO), capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"import {module}: exit {p.returncode}: {p.stderr[-400:]}")
    return set(json.loads(p.stdout.strip().splitlines()[-1]))


def import_case(module: str):
    def case():
        mods = imported(module)
        bad = sorted(m for m in mods if m in NOT_AT_IMPORT or m == "src.training" or m.startswith("src.training."))
        already = {m: m in mods for m in ("torch", "src.eval.artifacts", "src.tman", "src.tman.binding")}
        return module in mods and not bad, f"of B-4's set: {bad}; loaded as at MBL: {already}"
    return case


# --------------------------------------------------------------------------------------------------
# E: end to end at reader level, on TMAN's accepted bound export
# --------------------------------------------------------------------------------------------------
def e_open():
    """open_inventory with TMAN's reader, its default, reads TMAN's bound export under both policies: the reader's
    name, bound, its freeze recording its manifest_sha256, one entry per catalogue slot (IF§11.2)."""
    got = {}
    slots = len(catalogue().slots)
    for pol in (Policy.NONOFFICIAL_SMOKE, Policy.OFFICIAL):
        inv = H.open_inventory(CTX.path, policy=pol)
        got[pol.name] = (inv.reader, inv.bound, inv.doc["freeze"]["manifest_sha256"] == inv.doc["manifest_sha256"],
                         len(inv.doc["entries"]) == slots)
    counts = (len(CTX.inv.doc["models"]), len(CTX.inv.doc["entries"]), len(CTX.inv.doc["binding"]["artifacts"]))
    return all(v == (H.TMAN_READER, True, True, True) for v in got.values()), \
        f"{got}; rows, entries, records {counts}; catalogue slots {slots}"


def e_rows():
    """Every TMAN-built model row passes model_row, and check_row_at_entry at its slot's entry (LB4-6, LB4-7); the
    qat_fakequant_off rows carry fqoff_source, accepted and never read."""
    inv, bad = CTX.inv, []
    for r in inv.doc["models"]:
        try:
            got = H.model_row(inv, r["slot"], r["precision"])
            H.check_row_at_entry(inv, types.SimpleNamespace(slot=r["slot"], precision=r["precision"], row=got))
        except H.LrepRefusal as e:
            bad.append((r["slot"], r["precision"], e.code))
    fq = [r for r in inv.doc["models"] if r["precision"] == H.PRECISION_FQOFF]
    teacher = [r for r in inv.doc["models"] if r["model_role"] == "teacher"]
    return (not bad and len(inv.doc["models"]) >= 20 and len(fq) >= 2 and all("fqoff_source" in r for r in fq)
            and teacher and all(r["seed"] is None for r in teacher)), \
        f"{len(inv.doc['models'])} rows, {len(fq)} fq_off, {len(teacher)} teacher; refused {bad}"


def e_dedup():
    """read_dedup (TMAN's parser, LB4-5) and check_ids against the file's own id set: its kept ids and its removed ids
    by reason, which the reasons partition."""
    from src.tman import dedup as TD
    rec = H.read_dedup(CTX.inv)
    doc = TD.read_dedup(rec.file.read_bytes(), expect_sha256=rec.sha256)
    every = list(doc["kept_ids"]) + [i for r in sorted(doc["removed_ids_by_reason"])
                                     for i in doc["removed_ids_by_reason"][r]]
    got = DS.check_ids(rec, {"the file's own id set": AM6.ids_sha256(every)}, every)
    return (got["n_all"] == len(set(every)) == doc["n_images"] and got["n_removed"] == got["sum_removed_by_reason"] > 0
            and got["partition_stated"] is True and rec.reasons_partition is True), str(got)


def e_e5_parent():
    """LB4-8 on TMAN's export, read under Policy.OFFICIAL: E5:s42:sel's catalogue.parent is null, the checkout
    catalogue's pilot.runs are read, its derived entry's own parent is one of them and its inputs.run_slot, parent_slot
    gives it, and the 21 planned cells read (R-3 to R-6, R-12 to R-14: check_parentage among them)."""
    inv = H.open_inventory(CTX.path, policy=Policy.OFFICIAL)
    e = inv.entry(E5SEL)
    runs = catalogue().slots[E5SEL]["pilot"]["runs"]
    cells = H.planned_cells(inv)
    return (e["catalogue"]["parent"] is None and e["kind"] == "derived" and inv.pilot_runs.get(E5SEL) == tuple(runs)
            and e["parent"] in runs and e["parent"] == e["inputs"]["run_slot"]
            and H.parent_slot(inv, E5SEL) == e["parent"]
            and len(cells.slots) == 21 and cells.status[("E5", 42)].has_model), \
        f"own parent {e['parent']}; pilot.runs {runs}; {len(cells.slots)} planned cells"


def e_records():
    """The binding's records, read_inventory's doc["binding"]["artifacts"] (LB4-4), indexed by (slot, precision, scope,
    cell): each key once, and binding_record returns each record by its key."""
    inv = CTX.inv
    recs = inv.doc["binding"]["artifacts"]
    keys = [(r["slot"], r["precision"], r["scope"], r.get("cell")) for r in recs]
    differ = [k for k, r in zip(keys, recs) if H.binding_record(inv, *k) != r]
    return (len(set(keys)) == len(keys) == sum(len(v) for v in inv.records.values()) > 0 and not differ), \
        f"{len(keys)} records, {len(set(keys))} keys; differ {differ[:3]}"


def e_folder():
    """LB4-3: binding.path and dedup.path resolve against the export file's folder, on both sides. The export beside
    its dedup file only is binding_invalid, and beside its binding file only dedup_invalid (TMAN's reader, its code
    kept); LREP's read_dedup of the export moved away is dedup_invalid (its own), and in place reads the file beside
    it; LREP's stub, the synthetic path's binding.path site (H:463), reads the binding file beside the export in place
    (refusing only its own schema key, which TMAN's file names format: LB4-4) and finds none when it is moved away."""
    doc, path = CTX.doc, CTX.path
    out = []
    for name, keep, code in (("beside_dedup_only", "dedup", "binding_invalid"),
                             ("beside_binding_only", "binding", "dedup_invalid")):
        d = LF.new_dir(CTX.root, name)
        shutil.copy2(path, d / path.name)
        shutil.copy2(path.parent / doc[keep]["path"], d / doc[keep]["path"])
        out.append(refused(code, lambda d=d: H.open_inventory(d / path.name, policy=Policy.NONOFFICIAL_SMOKE),
                           head=f"tman:{code}: "))
    moved = dataclasses.replace(CTX.inv, path=CTX.root / "beside_binding_only" / path.name)
    out.append(refused("dedup_invalid", lambda: H.read_dedup(moved), contains="cannot be read"))
    out.append(refused("binding_invalid", lambda: H.read_inventory_stub(CTX.root / "beside_dedup_only" / path.name),
                       contains="FileNotFoundError"))
    out.append(refused("binding_invalid", lambda: H.read_inventory_stub(path), contains="not a tday_binding/1 object"))
    here = H.read_dedup(CTX.inv).file == path.parent / doc["dedup"]["path"]
    return all(ok for ok, _ in out) and here, str([d for _, d in out])


def e_binding_file():
    """LB4-4: the binding file is tday_binding/1 under the key `format`, with exactly TMAN's four top keys; LREP reads
    the records from read_inventory's doc["binding"]["artifacts"], and the file's `inputs` (the 37) are not returned:
    the statistics loader checks them (B3R5)."""
    from src.tman import schema as S
    raw = (CTX.path.parent / CTX.doc["binding"]["path"]).read_bytes()
    bfile = json.loads(raw)
    b = CTX.inv.doc["binding"]
    return (bfile["format"] == S.BINDING_FORMAT == H.BINDING_SCHEMA and sorted(bfile) == sorted(S.BINDING_FILE_SPEC)
            and b["artifacts"] == bfile["artifacts"] and "inputs" not in b and S.sha256_bytes(raw) == b["sha256"]), \
        f"file keys {sorted(bfile)}; returned {sorted(b)}"


def e_unbound_official():
    """LB4-1: LREP's own checks still run after TMAN's reader returns: TMAN's plain export (binding unbound), which
    its reader accepts, is binding_unbound under Policy.OFFICIAL and reads under the smoke policy."""
    ok, detail = refused("binding_unbound", lambda: H.open_inventory(CTX.plain, policy=Policy.OFFICIAL))
    inv = H.open_inventory(CTX.plain, policy=Policy.NONOFFICIAL_SMOKE)
    return ok and inv.reader == H.TMAN_READER and not inv.bound, detail


# --------------------------------------------------------------------------------------------------
# C: the checkout's catalogue
# --------------------------------------------------------------------------------------------------
def c_map():
    """B5-S1's map against every scored slot of the checkout's catalogue (evaluator_stage not null): (evaluator_stage,
    model_role, precision) equal, none unmapped."""
    scored = scored_slots()
    unmapped = sorted(s["slot"] for s in scored if s["stage"] not in H.STAGE_MAP)
    differ = sorted(s["slot"] for s in scored if s["stage"] in H.STAGE_MAP
                    and H.STAGE_MAP[s["stage"]] != (s["evaluator_stage"], s["model_role"], s["precision"]))
    return (len(scored) == SCORED_SLOTS and not unmapped and not differ), \
        f"{len(scored) - len(unmapped) - len(differ)}/{len(scored)} agree; unmapped {unmapped}; differ {differ}"


def c_dt_set():
    """R4-1's DT set on the checkout's catalogue (slot_read of each scored slot's stage and roles) is exactly Part 3
    §5's nine."""
    dt = sorted(s["slot"] for s in scored_slots()
                if H.slot_read({"stage": s["stage"], "role": s["roles"]}) == H.READ_DESCRIPTIVE_TEST)
    return dt == sorted(DT_NINE), str(dt)


# --------------------------------------------------------------------------------------------------
# D: drift guards: LREP's copies against TMAN's definitions, imported
# --------------------------------------------------------------------------------------------------
def d_roles():
    from src.tman import schema as S
    return H.ROLES == S.ROLES, f"{H.ROLES} vs {S.ROLES}"


def d_slot_key():
    """LREP's fixtures' SLOT_KEY_RE is TMAN's (schema.SLOT_KEY_RE: pattern and flags), and every key of the checkout's
    catalogue and of TMAN's export matches it."""
    from src.tman import schema as S
    keys = list(catalogue().slots) + [e["slot"] for e in CTX.doc["entries"]]
    bad = sorted({k for k in keys if not LF.SLOT_KEY_RE.fullmatch(k)})
    return (LF.SLOT_KEY_RE.pattern == S.SLOT_KEY_RE.pattern and LF.SLOT_KEY_RE.flags == S.SLOT_KEY_RE.flags
            and not bad), f"{len(set(keys))} keys; {bad[:3]}"


def d_catalogue_copy():
    """The fixtures' typed catalogue (lrep_fixtures.CATALOGUE, from TMAN's ee5593e: CATALOGUE_SOURCE) is the checkout's:
    the same sha256, and slot by slot the same stage, seed, class, roles, evaluator_stage, model_role, precision and
    parent."""
    cat = catalogue()
    got = {k: (s["stage"], s["seed"], s["class"], tuple(s["roles"]), s["evaluator_stage"], s["model_role"],
               s["precision"], s["parent"]) for k, s in cat.slots.items()}
    differ = sorted(k for k in set(got) | set(LF.CATALOGUE) if got.get(k) != LF.CATALOGUE.get(k))
    return cat.sha256 == LF.CATALOGUE_SOURCE["sha256"] and not differ, \
        f"sha256 {cat.sha256[:16]}; {len(got)} slots; differ {differ[:4]}"


def d_formats():
    """The format strings: tday-inventory/1 (TMAN's export's schema), tday_binding/1, and tday_dedup/1, which LREP reads
    only through TMAN's parser."""
    from src.tman import dedup as TD
    from src.tman import schema as S
    dfile = TD.read_dedup(H.read_dedup(CTX.inv).file.read_bytes())
    return (H.INVENTORY_SCHEMA == S.INVENTORY_FORMAT == CTX.doc["schema"] and H.BINDING_SCHEMA == S.BINDING_FORMAT
            and dfile["format"] == S.DEDUP_FORMAT), f"{CTX.doc['schema']}, {S.BINDING_FORMAT}, {dfile['format']}"


def d_dedup_fields():
    """DEDUP_EXPORT_FIELDS are validate.DEDUP_SPEC's keys; the file's fields LREP reads (DEDUP_FILE_FIELDS) are among
    dedup.DOC_KEYS."""
    from src.tman import dedup as TD
    from src.tman import validate as V
    return (sorted(H.DEDUP_EXPORT_FIELDS) == sorted(V.DEDUP_SPEC[1])
            and set(H.DEDUP_FILE_FIELDS) <= set(TD.DOC_KEYS)), \
        f"{sorted(H.DEDUP_EXPORT_FIELDS)} vs {sorted(V.DEDUP_SPEC[1])}; file fields {H.DEDUP_FILE_FIELDS}"


def keys_read(fn, var: str) -> set:
    """The constant keys `fn` reads from the name `var`: var["k"], var.get("k") and "k" in var (AST)."""
    out = set()
    for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == var
                and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)):
            out.add(n.slice.value)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
              and isinstance(n.func.value, ast.Name) and n.func.value.id == var and n.args
              and isinstance(n.args[0], ast.Constant)):
            out.add(n.args[0].value)
        elif (isinstance(n, ast.Compare) and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str)
              and len(n.ops) == 1 and isinstance(n.ops[0], ast.In) and isinstance(n.comparators[0], ast.Name)
              and n.comparators[0].id == var):
            out.add(n.left.value)
    return out


def d_row_keys():
    """The keys model_row reads of a row, and of its source, are ROW_SPEC's (src.tman.validate)."""
    from src.tman import validate as V
    row, src = keys_read(H.model_row, "row"), keys_read(H.model_row, "src")
    return (len(row) >= 9 and row <= set(V.ROW_SPEC) and bool(src) and src <= set(V.ROW_SPEC["source"][1])), \
        f"row {sorted(row)}; source {sorted(src)}"


def d_identity():
    """row_identity is TMAN's binding.identity (TB17) on every TMAN-built row."""
    from src.tman import binding as B
    rows = CTX.inv.doc["models"]
    bad = [(r["slot"], r["precision"]) for r in rows if H.row_identity(r) != B.identity(r)]
    return not bad and len(rows) >= 20, f"{len(rows)} rows; {bad[:3]}"


# --------------------------------------------------------------------------------------------------
# A: LB4-A1, through TMAN's validator
# --------------------------------------------------------------------------------------------------
def gate_refusal(mutate):
    """TMAN's validator in freeze mode on TMAN's world, `mutate` applied to a copy of its manifest: refused by
    parent_rule_violation with every earlier freeze gate passing (scripts/smoke_tman_validate.expect_gate's rule), the
    world itself passing before."""
    import scripts.smoke_tman_validate as TV
    w = CTX.world
    order = TV.gate_codes("freeze")
    doc = copy.deepcopy(w.doc)
    mutate(w, doc)
    before = w.run("freeze")[0]
    code, passed, e = w.run("freeze", doc)
    ok = before is None and code == WANT_GATE and passed == order[:order.index(WANT_GATE)]
    return ok, f"world {before}; {code} after {len(passed or [])} gates: {(e.detail if e else '')[:200]}"


def arm_derived(w, doc):
    """E7-R1, entered not run under its cut arm run E3-R1, entered derived instead, as TMAN's own
    derived_under_a_cut_parent case enters E7:s43 (from a derived PTQ entry of the world)."""
    e, src = w.entry(doc, ARM_CHILD), w.entry(doc, ARM_SOURCE)
    for k in ("reason", "parent_record", "entry"):
        e.pop(k, None)
    e.update(kind="derived", method="ptq", inputs=copy.deepcopy(src["inputs"]), hashed=copy.deepcopy(src["hashed"]),
             ptq_record=None, evaluator_stage=w.cat.slots[ARM_CHILD]["evaluator_stage"], parent=ARM_PARENT)


def quant_derived(w, doc):
    """E6-E2:s42:sel, which has no entry under its cut QAT run E6-E2:s42:15ep (IF§11.2), entered derived: the world's
    derived selection E6:s42:sel with this slot's stage, catalogue block, roles, parent, evaluator stage and
    run_slot."""
    cs = w.cat.slots[QUANT_CHILD]
    e = copy.deepcopy(w.entry(doc, QUANT_SOURCE))
    e.update(slot=QUANT_CHILD, stage=cs["stage"], seed=cs["seed"], catalogue=w.cat.entry_block(QUANT_CHILD),
             role=list(cs["roles"]), parent=cs["parent"], evaluator_stage=cs["evaluator_stage"])
    e["inputs"]["run_slot"] = cs["parent"]
    doc["body"]["entries"].append(e)


def a_arm():
    """LB4-A1: an arm descendant (E7-R1, role arm) whose entry contradicts its cut parent's: parent_rule_violation."""
    w = CTX.world
    pre = (w.cat.slots[ARM_CHILD]["parent"] == ARM_PARENT and "arm" in w.cat.slots[ARM_CHILD]["roles"]
           and w.entry(w.doc, ARM_PARENT)["kind"] == "cut" and w.entry(w.doc, ARM_CHILD)["kind"] == "not_run")
    ok, detail = gate_refusal(arm_derived)
    return pre and ok and ARM_CHILD in detail, detail


def a_quant():
    """LB4-A1: a quant_descriptive descendant (E6-E2's selection) whose entry contradicts its cut parent's:
    parent_rule_violation."""
    w = CTX.world
    cs = w.cat.slots[QUANT_CHILD]
    pre = ("quant_descriptive" in cs["roles"] and w.entry(w.doc, cs["parent"])["kind"] == "cut"
           and all(e["slot"] != QUANT_CHILD for e in w.doc["body"]["entries"]))
    ok, detail = gate_refusal(quant_derived)
    return pre and ok and QUANT_CHILD in detail, detail


# --------------------------------------------------------------------------------------------------
def x_no_test_path():
    """LRB20, with ':' (R4-6): no created path contains 'test' or ':', under this smoke's root and TMAN's world's
    folders (its scratch repository and runs root, which TMAN's fixture library makes in TMPDIR). Run last."""
    roots = [CTX.root, CTX.world.repo, CTX.world.runs]
    bad = [os.path.join(d, n) for r in roots for d, dirs, files in os.walk(r) for n in dirs + files
           if "test" in n.lower() or ":" in n]
    return not bad and not any("test" in str(r).lower() for r in roots), str(bad[:3])


def setup(root: Path, world) -> None:
    """TMAN's world, held open by the caller (its validator reads the patched teacher of record), and its exports
    under `root`: the bound one is read with TMAN's reader."""
    CTX.root, CTX.world = root, world
    ex = LF.tman_exports(world, LF.new_dir(root, "exports"))
    (path, doc), CTX.plain = ex["bound"], ex["plain"][0]
    CTX.path, CTX.doc = path, doc
    CTX.inv = H.open_inventory(path, policy=Policy.NONOFFICIAL_SMOKE)


def cases():
    return [("I1", "heldout_inputs' import loads none of TMAN's validate, schema, manifest, dedup or src.training",
             import_case("src.stats.heldout_inputs")),
            ("I2", "report_extensions' import likewise", import_case("src.stats.report_extensions")),
            ("E1", "open_inventory with TMAN's reader reads TMAN's bound export (both policies)", e_open),
            ("E2", "every TMAN-built row through model_row and check_row_at_entry (LB4-6, LB4-7)", e_rows),
            ("E3", "read_dedup and check_ids against the file's own id set (LB4-5)", e_dedup),
            ("E4", "E5:s42:sel's parent under LB4-8; the 21 planned cells under Policy.OFFICIAL", e_e5_parent),
            ("E5", "the records indexed by (slot, precision, scope, cell)", e_records),
            ("E6", "LB4-3: binding.path and dedup.path against the export's folder, on both sides", e_folder),
            ("E7", "LB4-4: the binding file's format and TMAN's four top keys; the records read from it",
             e_binding_file),
            ("E8", "LB4-1: LREP's own checks after TMAN's reader: an unbound export under Policy.OFFICIAL",
             e_unbound_official),
            ("C1", "B5-S1's map against every scored catalogue slot (49/49, none unmapped)", c_map),
            ("C2", "R4-1's DT set is exactly Part 3 §5's nine", c_dt_set),
            ("D1", "ROLES is schema.ROLES", d_roles),
            ("D2", "SLOT_KEY_RE is schema.SLOT_KEY_RE; every key matches", d_slot_key),
            ("D3", "the format strings", d_formats),
            ("D4", "DEDUP_EXPORT_FIELDS is DEDUP_SPEC's keys; DEDUP_FILE_FIELDS among DOC_KEYS", d_dedup_fields),
            ("D5", "the keys model_row reads are ROW_SPEC's", d_row_keys),
            ("D6", "row_identity is binding.identity on every TMAN-built row", d_identity),
            ("D7", "the fixtures' catalogue copy is the checkout's (sha256 and every slot)", d_catalogue_copy),
            ("A1", "LB4-A1: an arm descendant under its cut parent: parent_rule_violation (TMAN's validator)", a_arm),
            ("A2", "LB4-A1: a quant_descriptive descendant under its cut parent: parent_rule_violation", a_quant),
            ("X1", "no created path contains 'test' or ':' (LRB20; R4-6; run last)", x_no_test_path)]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_conformance_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_lrep_conformance: temporary root {root} (left in place)", flush=True)
    with LF.tman_world() as world:
        setup(root, world)
        results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'LREP CONFORMANCE OK' if good == len(results) else 'LREP CONFORMANCE FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; nothing here is a thesis result.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
