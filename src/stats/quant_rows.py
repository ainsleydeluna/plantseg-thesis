"""AM-23's descriptive rows on TEST (lane LREP, B-5 (c): AM-23 items 5-6, PA :2607-2615; LRB22).

AM-23 item 5 (PA :2607-2611): E6-E2, E7-E2 and E6-X, when they exist, are scored once at the single TEST
evaluation, clean, canvas protocol, descriptive; E6-X is reported with AM-18 item 3(d)'s quantities;
AM-18 item 8(c)'s duplicate-free subset reports them as it reports every arm; their converted VAL scores
are also reported. Item 6 (PA :2612-2615): Chapter 4 reports E6-E2 − E6 and E7-E2 − E7 seed 42 as
descriptive contrasts of the parent under one quantization recipe, beside E2 − E3, on VAL and on clean
TEST, with no test and no claim beyond them; E6-X − E6 seed 42 joins AM-18 item 3(d)'s contrasts, read
the same way.

  * Rows at seed 42: E6-E2 − E6 and E7-E2 − E7, with their dataset-level differences -- all-class from
    Part 1's float64 m, GT-present and disease-only from eligibility.rule_variants (float64) -- beside
    E2 − E3 from the 37 inputs. All three keep item 6's orientation, the E2 parent minus the E3 parent,
    so E2 − E3 = m(E2) − m(E3): the negative of the official accuracy_e2_e3 delta (E3 − E2;
    driver.py:55). No interval, no test, no claim beyond them (the quant__<stage> ids stay reserved in
    lrep_descriptive).
  * E6-X − E6 (E6-R1, E6-R2) is reported with AM-18 item 3(d)'s quantities in the arm tables
    (src.stats.arm_tables), whose rows this part names.
  * The quant_descriptive slots (IF§11.1) are E6-E2, E7-E2, E6-R1 and E6-R2 at seed 42; a slot at
    another stage or seed, or two slots for one stage, is quant_slot_invalid. A slot without a model
    carries its IF§11.2 label from TMAN's entries (cut, not made, ...), never a refusal, unless the
    binding lists its clean artifact anyway (artifact_for_missing_run, R-6); a stage the export has no
    slot for reads so ("when they exist", item 5). Their fq_off rows (E6-E2, E6-R1, E6-R2; IF§9) join
    the extension report's fq_off table (report_extensions.build_fqoff).
  * TEST side only: item 6's VAL side goes to the LRB10 block with AM-18 item 3(d)'s, and item 5's
    converted VAL scores read "not on record" (no committed record at M; DL-146 is a draft until the
    records catch-up lands).

Descriptive. Import-time behaviour is side-effect free.
"""
from __future__ import annotations

from pathlib import Path

from . import heldout_inputs as H
from . import lrep_descriptive as LD

#: (candidate stage, the E3 parent's quantization it sits beside) at seed 42 (AM-23 item 6).
ROWS = (("E6-E2", "E6"), ("E7-E2", "E7"))
REFERENCE = ("E2", "E3")
E6_X_STAGES = ("E6-R1", "E6-R2")
QUANT_SEED = 42
NO_SLOT = "no quant_descriptive slot in the export"
CONVERTED_VAL = ("not on record: no committed record of the converted VAL scores at M (DL-146 is a draft "
                 "until the records catch-up lands)")
TITLE = "AM-23's descriptive quantizations on TEST (items 5-6; descriptive)"
FOOTNOTE = ("Descriptive (AM-23 item 6): no interval, no test and no claim beyond the differences. "
            "Orientation: the E2 parent minus the E3 parent; E2 − E3 = m(E2) − m(E3), the negative of the "
            "official accuracy_e2_e3 delta (E3 − E2).")


def quant_slots(inv: H.Inventory) -> dict:
    """{stage: slot} of the quant_descriptive slots; a stage outside the four or a seed other than 42,
    or two slots for one stage, is quant_slot_invalid."""
    found: dict = {}
    for slot in H.role_slots(inv, "quant_descriptive"):
        e = inv.entry(slot)
        stage = e.get("stage")
        if stage not in H.QUANT_STAGES or e.get("seed") != QUANT_SEED:
            raise H.refusal("quant_slot_invalid", f"quant_descriptive slot {slot!r} at stage {stage!r}, "
                                                  f"seed {e.get('seed')!r}: the slots are "
                                                  f"{list(H.QUANT_STAGES)} at seed {QUANT_SEED} (AM-23 "
                                                  "items 1-3)")
        found.setdefault(stage, []).append(slot)
    two = {k: v for k, v in found.items() if len(v) > 1}
    if two:
        raise H.refusal("quant_slot_invalid", f"two quant_descriptive slots for one stage: {two}")
    return {k: v[0] for k, v in found.items()}


def differences(cand: dict, base: dict) -> dict:
    """candidate − baseline of two arm_tables.side_values: all-class from m, GT-present and disease-only
    from the float64 values (None where a side has no eligible class)."""
    def sub(a, b):
        return None if a is None or b is None else a - b
    return {"all_class": cand["m"] - base["m"],
            "gt_present": sub(cand["gt_present"]["value_float64"], base["gt_present"]["value_float64"]),
            "disease_only": sub(cand["disease_only"]["value_float64"], base["disease_only"]["value_float64"])}


def build_quant_rows(stats: H.StatsArtifact, inv: H.Inventory, cells: H.PlannedCells, *,
                     input_root: Path) -> dict:
    """The "AM-23 descriptive quantizations" part."""
    from .arm_tables import side_values
    from .lrep_descriptive import arm_contrast_id
    family = H.family_inputs(stats, inv, cells, stages=H.B5_FAMILY_STAGES)
    slots = quant_slots(inv)
    sides = {st: side_values(family[st].run.stats, family[st].summary) for st in H.B5_FAMILY_STAGES}
    reference = {"candidate_stage": REFERENCE[0], "baseline_stage": REFERENCE[1],
                 "candidate": sides["E2"], "baseline": sides["E3"],
                 "differences": differences(sides["E2"], sides["E3"]),
                 "note": "E2 − E3 = m(E2) − m(E3): the negative of the official accuracy_e2_e3 delta "
                         "(E3 − E2; driver.py:55)"}
    rows, loaded = [], {f"{st} seed 42": family[st] for st in H.B5_FAMILY_STAGES}
    for cand, base in ROWS:
        row = {"candidate_stage": cand, "baseline_stage": base, "slot": slots.get(cand), "kind": None,
               "label": None, "candidate": None, "baseline": sides[base], "differences": None}
        rows.append(row)
        if row["slot"] is None:
            row["label"] = NO_SLOT
            continue
        st = H.b5_status(inv, row["slot"])
        row.update(kind=st.kind, label=st.label)
        precision = H.stage_spec(cand)[2]
        if not st.has_model:
            if inv.has_record(row["slot"], precision, "clean", None):
                raise H.refusal("artifact_for_missing_run", f"{cand} ({row['slot']!r}) reads {st.label!r} "
                                                            "but the binding lists a clean artifact for it "
                                                            "(R-6)")
            continue
        ba = H.load_bound(inv, slot=row["slot"], stage=cand, precision=precision, scope="clean",
                          input_root=input_root, policy=stats.policy)
        loaded[f"{cand} {ba.slot}"] = ba
        row["candidate"] = side_values(ba.run.stats, ba.summary)
        row["differences"] = differences(row["candidate"], row["baseline"])
    e6x = []
    for stage in E6_X_STAGES:
        slot = slots.get(stage)
        st = H.b5_status(inv, slot) if slot is not None else None
        e6x.append({"stage": stage, "slot": slot, "label": st.label if st else NO_SLOT,
                    "arms_row": arm_contrast_id(stage.split("-", 1)[1], "E6")})
    H.cross_check(loaded, {}, evaluator_commit=inv.evaluator_commit, official=stats.profile == "official")
    return {"part": "quant", "title": TITLE, "banner": LD.BANNER, "rows": rows, "reference": reference,
            "e6_x": e6x, "converted_val_scores": CONVERTED_VAL, "footnote": FOOTNOTE}


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))


def render_markdown(doc: dict, banner: str = "") -> list[str]:
    out = [f"## {doc['title']}", "", banner, f"> {doc['banner']}", "",
           "| contrast | m candidate | m baseline | Δ all-class | Δ GT-present | Δ disease-only | note |",
           "|---|---|---|---|---|---|---|"]
    for r in [*doc["rows"], doc["reference"]]:
        dd, c = r["differences"] or {}, r["candidate"]
        note = r.get("label") or r.get("note") or ""
        out.append(f"| {r['candidate_stage']} − {r['baseline_stage']} | {_fmt(c['m'] if c else None)} | "
                   f"{_fmt(r['baseline']['m'])} | {_fmt(dd.get('all_class'))} | "
                   f"{_fmt(dd.get('gt_present'))} | {_fmt(dd.get('disease_only'))} | {note} |")
    e6x = "; ".join(f"{e['stage']} ({e['label'] or 'scored'}): the arms part, row `{e['arms_row']}`"
                    for e in doc["e6_x"])
    out += ["", f"E6-X − E6: {e6x}.", "",
            f"Converted VAL scores (AM-23 item 5): {doc['converted_val_scores']}.", "",
            f"_{doc['footnote']}_", ""]
    return out


__all__ = ["ROWS", "REFERENCE", "E6_X_STAGES", "QUANT_SEED", "NO_SLOT", "CONVERTED_VAL", "quant_slots",
           "differences", "build_quant_rows", "render_markdown"]
