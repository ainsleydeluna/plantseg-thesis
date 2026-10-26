"""AM-18's arm tables on TEST (lane LREP, B-5 (a): AM-18 items 3(d) and 11; TL-13; AM-23 item 6).

AM-18 item 3(d) (PA :1078-1086): for each arm X, "VAL and TEST clean dataset-level mIoU (all-class,
GT-present, disease-only); the dataset-level all-class differences E3-X − E3 and E7-X − E7, each read
next to the AM-16 item 2 seed-noise scale (√2·s for a two-run contrast); a difference inside that scale
is reported as not distinguishable from seed noise; per-image mean ΔmIoU with BCa 95% CIs for the same
contrasts; each arm teacher's VAL_X and ECE_X alongside", with no test and outside the Holm family
(:1085). Item 11 (:1283-1288): every arm with its teacher's VAL score and ECE, its go/no-go outcome and
its KD results next to the seed-noise scale. AM-23 item 6 (:2612-2615): E6-X − E6 at seed 42 joins those
contrasts, read the same way.

  * X ∈ {R1, R2, flip}: K8 is out (DL-96; PA :1116) and avg is out (DL-66; PA :2561). Rows at seed 42:
    E3-X − E3 and E7-X − E7 for every X, E6-X − E6 for R1 and R2 only; no E6-flip row is made. The
    candidates are the `arm` slots at E3-X and E7-X and AM-23's `quant_descriptive` slots at E6-R1 and
    E6-R2 (IF§11.1 :25); the comparators E3, E7 and E6 are the family slots, read from the 37 inputs of
    the verified statistics artifact with their identity checked (R-11). An arm or arm_teacher slot or a
    row the tables cannot place -- an arm at a stage other than E3-X or E7-X for the three X, a candidate
    at a seed other than 42, an arm teacher other than R1 and R2, two slots for one (X, stage), or a row
    the export has no slot for -- is arm_slot_invalid.
  * Each row: both sides' clean dataset-level mIoU, m (Part 1's float64 m, seed_stability.m_of) and
    eligibility.rule_variants' all-class, GT-present and disease-only values; the all-class difference
    Δ = m(candidate) - m(baseline) beside √2·s with B5-D1's reading; the per-image mean Δ on
    align.METRIC_DISEASE_ONLY, the official family's per-image vector of the seven clean contrasts
    (driver.py:54-60), with B5-D2's two-sided BCa 95% interval in lrep_descriptive's namespace (contrast
    id arm__<x>__<stage>); the arm teacher's VAL_X, ECE_X and go/no-go. A candidate without a model
    carries its IF§11.2 label and no value, never a refusal; one whose binding lists a clean artifact
    anyway is artifact_for_missing_run (R-6).
  * B5-D1: "not distinguishable from seed noise" iff |Δ| <= √2·s (inclusive), in float64. s is read from
    reports/derived/dl27_band.json by exact repo path, its sha256 DL-99's checked before it is parsed;
    √2·s = math.sqrt(2) * s must equal its "band" and 0.015621041685101825 bitwise.
  * Teacher values: R1 and R2 from reports/arms/arm_go_rule_R1.json and _R2.json (DL-124, DL-125; format
    arm_go_rule/1, scripts/arm_go_rule.py), each checked against its sha256 before a strict parse: VAL_X
    = values.val_x, ECE_X = values.ece_x, go/no-go = verdict. flip: "not on record" (DL-133 and DL-134
    name none; W keeps the teacher of record's weights, DL-88). Nothing is computed anew. A record that
    is missing, unreadable, not at its sha256 or malformed, or a band that is not √2·s, is
    arm_record_invalid.
  * TEST side only. AM-18 item 3(d)'s VAL side is a later local block (LRB10; LREP-B5-RULE-2 item 7),
    which runs side_values, reading and lrep_descriptive.mean_delta_task on the VAL artifacts.

Descriptive; outside protocol 1.0.0: no test, no p-value, not in the Holm family (PA :1085).
Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from . import bootstrap as BS
from . import heldout_inputs as H
from . import lrep_descriptive as LD

CODE_REPO = Path(__file__).resolve().parents[2]
ARM_XS = H.ARM_XS
ARM_TEACHERS = ("R1", "R2")
ARM_SEED = 42
#: The rows, in order: per X, E3-X, E7-X and (R1, R2) E6-X (AM-18 item 3(d); AM-23 item 6).
ROWS = tuple((x, st) for x in ARM_XS for st in ("E3", "E7", "E6") if not (st == "E6" and x == "flip"))
#: DL-99: the AM-16 item 2 seed-noise record and its band.
SEED_NOISE_RECORD = {"path": "reports/derived/dl27_band.json", "dl": "DL-99", "band": 0.015621041685101825,
                     "sha256": "3f33ab525037e9744b21b9abc19b60190f58f7832e3006783d759ec23ddf6ee4"}
#: DL-124 and DL-125: the arm teachers' go-rule records (AM-18 item 5).
TEACHER_RECORDS = {
    "R1": {"path": "reports/arms/arm_go_rule_R1.json", "dl": "DL-124",
           "sha256": "71d024d1ba17e68ebf679dcf178ddb84472e5147815850e8c4ab3748c0a35e9f"},
    "R2": {"path": "reports/arms/arm_go_rule_R2.json", "dl": "DL-125",
           "sha256": "0e3552c622b219c34c4ca9b289b2d960775a8f7922f731e4bc8fce684a6f3288"}}
RECORD_FORMAT = "arm_go_rule/1"                       # scripts/arm_go_rule.py
VERDICTS = ("go", "no_go", "not_evaluable")           # scripts/arm_go_rule.py:79, :244
NOT_ON_RECORD = ("not on record: DL-133 and DL-134 name no VAL_X or ECE_X; W keeps the teacher of record's "
                 "weights (DL-88)")
READING_INSIDE = "not distinguishable from seed noise"
READING_OUTSIDE = "larger than the seed-noise scale"
TITLE = "AM-18's arm tables on TEST (items 3(d) and 11; AM-23 item 6; descriptive; outside protocol 1.0.0)"
FOOTNOTE = ("Descriptive, outside protocol 1.0.0: no test, no p-value, not in the Holm family (PA :1085). "
            "Δ = m(candidate) - m(baseline), all-class, read beside √2·s (DL-99): inside it reads 'not "
            "distinguishable from seed noise' (|Δ| <= √2·s). Per-image mean Δ: disease-only, BCa 95%.")
HEADER = ("X", "stage", "candidate", "m baseline", "m candidate", "Δ all-class", "reading",
          "mean Δ per image [BCa 95%]", "GT-present b / c", "disease-only b / c", "VAL_X", "ECE_X", "go")


def candidate_stage(x: str, stage: str) -> str:
    """The TMAN stage of row (x, stage): E3-X, E7-X or E6-X (IF§9's names; B5-S1)."""
    return f"{stage}-{x}"


def read_record(rel: str, sha256: str, what: str, *, repo=CODE_REPO) -> dict:
    """A committed record by exact repo path, its sha256 checked before a strict parse."""
    p = Path(repo) / rel
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise H.refusal("arm_record_invalid", f"{what} {rel}: cannot be read ({type(e).__name__})") from e
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise H.refusal("arm_record_invalid", f"{what} {rel}: sha256 {digest}, its record's is {sha256}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise H.refusal("arm_record_invalid", f"{what} {rel}: not UTF-8") from e
    doc = H.strict_json_text(text, f"{what} {rel}", "arm_record_invalid")
    if not isinstance(doc, dict):
        raise H.refusal("arm_record_invalid", f"{what} {rel}: not a JSON object")
    return doc


def seed_noise(*, repo=CODE_REPO, record=SEED_NOISE_RECORD) -> dict:
    """B5-D1's scale: √2·s from DL-99's record, equal to its band and to the record's value bitwise."""
    doc = read_record(record["path"], record["sha256"], "the seed-noise record", repo=repo)
    s, band = doc.get("s"), doc.get("band")
    if type(s) is not float or type(band) is not float or not (math.isfinite(s) and s > 0.0):
        raise H.refusal("arm_record_invalid", f"the seed-noise record: s {s!r} and band {band!r} are not "
                                              "positive finite floats")
    scale = math.sqrt(2) * s
    if not (scale == band == record["band"]):
        raise H.refusal("arm_record_invalid", f"√2·s = {scale!r} differs from the band {band!r} or "
                                              f"{record['band']!r} (B5-D1; {record['dl']})")
    return {"path": record["path"], "sha256": record["sha256"], "dl": record["dl"], "s": s, "band": band,
            "scale": scale}


def reading(delta: float, scale: float) -> str:
    """B5-D1: inside the seed-noise scale iff |Δ| <= √2·s, inclusive, on the float64 values."""
    return READING_INSIDE if abs(delta) <= scale else READING_OUTSIDE


def teacher_values(x: str, *, repo=CODE_REPO, records=TEACHER_RECORDS) -> dict:
    """The arm teacher's VAL_X, ECE_X and go/no-go from its committed record; flip: not on record."""
    if x not in records:
        return {"x": x, "val_x": None, "ece_x": None, "go": None, "record": None, "note": NOT_ON_RECORD}
    rec = records[x]
    doc = read_record(rec["path"], rec["sha256"], f"{x}'s go-rule record", repo=repo)
    values = doc.get("values") if isinstance(doc.get("values"), dict) else {}
    val_x, ece_x, verdict = values.get("val_x"), values.get("ece_x"), doc.get("verdict")
    if (doc.get("format") != RECORD_FORMAT or doc.get("arm") != x or type(val_x) is not float
            or type(ece_x) is not float or verdict not in VERDICTS):
        raise H.refusal("arm_record_invalid", f"{x}'s go-rule record {rec['path']}: format "
                                              f"{doc.get('format')!r}, arm {doc.get('arm')!r}, val_x "
                                              f"{val_x!r}, ece_x {ece_x!r}, verdict {verdict!r}")
    return {"x": x, "val_x": val_x, "ece_x": ece_x, "go": verdict,
            "record": {"path": rec["path"], "sha256": rec["sha256"], "dl": rec["dl"]}, "note": None}


def arm_slots(inv: H.Inventory) -> tuple[dict, dict]:
    """({(x, stage): slot} of the rows' candidates, {x: slot} of the arm teachers R1 and R2): the `arm`
    slots at E3-X or E7-X and the `quant_descriptive` slots at E6-R1 and E6-R2 (IF§11.1 :25), at seed 42."""
    want = {candidate_stage(x, st): (x, st) for x, st in ROWS}
    found: dict = {}
    for slot in dict.fromkeys(H.role_slots(inv, "arm") + H.role_slots(inv, "quant_descriptive")):
        e = inv.entry(slot)
        stage = e.get("stage")
        if stage in want and e.get("seed") != ARM_SEED:
            raise H.refusal("arm_slot_invalid", f"slot {slot!r} at {stage} has seed {e.get('seed')!r}; the "
                                                f"arm tables place seed {ARM_SEED} only (AM-18 item 3(d))")
        if stage in want:
            found.setdefault(want[stage], []).append(slot)
        elif "arm" in e["role"]:
            raise H.refusal("arm_slot_invalid", f"arm slot {slot!r} at stage {stage!r}: the arm tables place "
                                                f"E3-X and E7-X for X in {list(ARM_XS)} only (K8 is out, "
                                                "DL-96; avg is out, DL-66)")
    teachers: dict = {}
    for slot in H.role_slots(inv, "arm_teacher"):
        stage = inv.entry(slot).get("stage")
        if stage not in ARM_TEACHERS:
            raise H.refusal("arm_slot_invalid", f"arm_teacher slot {slot!r} at stage {stage!r}: the arm "
                                                f"teachers are {list(ARM_TEACHERS)}")
        teachers.setdefault(stage, []).append(slot)
    two = {str(k): v for k, v in {**found, **teachers}.items() if len(v) > 1}
    if two:
        raise H.refusal("arm_slot_invalid", f"two slots for one (X, stage): {two}")
    missing = [candidate_stage(x, st) for x, st in ROWS if (x, st) not in found]
    missing += [x for x in ARM_TEACHERS if x not in teachers]
    if missing:
        raise H.refusal("arm_slot_invalid", f"the export has no slot for {missing}: a row the arm tables "
                                            "cannot place (IF§11.1: every catalogue slot is exported)")
    return {k: v[0] for k, v in found.items()}, {k: v[0] for k, v in teachers.items()}


def side_values(stats, summary: dict) -> dict:
    """One side's clean dataset-level mIoU from any artifact's statistics: m (Part 1's float64 m) and
    rule_variants' all-class, GT-present and disease-only values (float32 `value` and float64)."""
    from .eligibility import GT_PRESENT, UNION_PRESENT, rule_variants
    from .seed_stability import m_of
    rv = rule_variants(stats, background_index=int(summary["dataset"]["background_index"]))
    out = {"m": m_of(stats), "run_id": summary["run"]["run_id"],
           "checkpoint_sha256": summary["run"].get("checkpoint_sha256")}
    for name, (rule, space) in (("all_class", (UNION_PRESENT, "all_class")),
                                ("gt_present", (GT_PRESENT, "all_class")),
                                ("disease_only", (UNION_PRESENT, "disease_only"))):
        d = rv[rule][space]
        out[name] = {"value": d.value, "value_float64": d.value_float64, "n_classes": d.n_classes}
    return out


def arm_row(stats: H.StatsArtifact, inv: H.Inventory, x: str, stage: str, slot: str, base, *,
            input_root: Path, B: int, scale: float, teacher: dict) -> tuple[dict, object]:
    """One row (x, stage) against the 37's `base`: its candidate's status, values and per-image mean Δ,
    or its IF§11.2 label."""
    from .align import METRIC_DISEASE_ONLY, AlignmentError, align_runs
    cand = candidate_stage(x, stage)
    precision = H.stage_spec(cand)[2]
    st = H.b5_status(inv, slot)
    row = {"x": x, "stage": stage, "contrast_id": LD.arm_contrast_id(x, stage), "candidate_stage": cand,
           "candidate_slot": slot, "baseline_stage": stage,
           "baseline": side_values(base.run.stats, base.summary), "candidate": None, "kind": st.kind,
           "label": st.label, "delta_all_class": None, "seed_noise_scale": scale, "reading": None,
           "per_image": None, "teacher": teacher}
    if not st.has_model:
        if inv.has_record(slot, precision, "clean", None):
            raise H.refusal("artifact_for_missing_run", f"{cand} ({slot!r}) reads {st.label!r} but the "
                                                        "binding lists a clean artifact for it (R-6)")
        return row, None
    ba = H.load_bound(inv, slot=slot, stage=cand, precision=precision, scope="clean", input_root=input_root,
                      policy=stats.policy)
    row["candidate"] = side_values(ba.run.stats, ba.summary)
    delta = row["candidate"]["m"] - row["baseline"]["m"]
    row.update(delta_all_class=delta, reading=reading(delta, scale))
    try:
        pv = align_runs(base.run, ba.run, policy=stats.policy, metric=METRIC_DISEASE_ONLY)
    except AlignmentError as e:
        raise H.refusal("cross_artifact_mismatch", f"{cand} vs {stage}: the pair cannot be aligned: "
                                                   f"{e}") from e
    r = LD.mean_delta_task(row["contrast_id"], pv.delta, B=B)
    row["per_image"] = {"metric": METRIC_DISEASE_ONLY, "n_paired": pv.n,
                        "alignment_status": pv.alignment_status, **LD.task_json(r)}
    return row, ba


def build_arm_tables(stats: H.StatsArtifact, inv: H.Inventory, cells: H.PlannedCells, *, input_root: Path,
                     B: int = BS.PRODUCTION_B, repo=CODE_REPO) -> dict:
    """The "AM-18 arm tables" part."""
    noise = seed_noise(repo=repo)
    family = H.family_inputs(stats, inv, cells, stages=("E3", "E6", "E7"))
    candidates, teacher_slots = arm_slots(inv)
    teachers = {x: teacher_values(x, repo=repo) for x in ARM_XS}
    rows, loaded = [], {f"{st} seed 42": family[st] for st in ("E3", "E6", "E7")}
    for x, st in ROWS:
        row, ba = arm_row(stats, inv, x, st, candidates[(x, st)], family[st], input_root=input_root, B=B,
                          scale=noise["scale"], teacher=teachers[x])
        rows.append(row)
        if ba is not None:
            loaded[f"{row['candidate_stage']} {ba.slot}"] = ba
    teacher_rows = []
    for x in ARM_XS:
        slot = teacher_slots.get(x)
        st = H.b5_status(inv, slot) if slot is not None else None
        teacher_rows.append({**teachers[x], "slot": slot, "kind": st.kind if st else None,
                             "label": st.label if st else None})
    H.cross_check(loaded, {}, evaluator_commit=inv.evaluator_commit, official=stats.profile == "official")
    return {"part": "arms", "title": TITLE, "banner": LD.BANNER, "analysis_id": LD.ANALYSIS_ID, "B": B,
            "seed_noise": noise, "rows": rows, "teachers": teacher_rows, "footnote": FOOTNOTE}


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))


def _line(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown(doc: dict, banner: str = "") -> list[str]:
    n = doc["seed_noise"]
    out = [f"## {doc['title']}", "", banner, f"> {doc['banner']}", "",
           f"√2·s = {_fmt(n['scale'])} (s = {_fmt(n['s'])}; {n['dl']}, `{n['path']}`).", "",
           _line(HEADER), _line(["---"] * len(HEADER)).replace(" ", "")]
    for r in doc["rows"]:
        t, b, c = r["teacher"], r["baseline"], r["candidate"]
        if c is None:
            mid = [_fmt(b["m"]), r["label"], "", "", "", "", ""]
        else:
            p = r["per_image"]
            mid = [_fmt(b["m"]), _fmt(c["m"]), _fmt(r["delta_all_class"]), r["reading"],
                   f"{_fmt(p['observed'])} [{_fmt(p['lower_bound'])}, {_fmt(p['upper_bound'])}]",
                   f"{_fmt(b['gt_present']['value_float64'])} / {_fmt(c['gt_present']['value_float64'])}",
                   f"{_fmt(b['disease_only']['value_float64'])} / {_fmt(c['disease_only']['value_float64'])}"]
        out.append(_line([r["x"], r["stage"], r["candidate_stage"], *mid, _fmt(t["val_x"]), _fmt(t["ece_x"]),
                          _fmt(t["go"])]))
    out += ["", "| arm teacher | VAL_X | ECE_X | go/no-go | record | TEST slot |",
            "|---|---|---|---|---|---|"]
    for t in doc["teachers"]:
        rec = t["record"]["dl"] if t["record"] else t["note"]
        out.append(_line([t["x"], _fmt(t["val_x"]), _fmt(t["ece_x"]), _fmt(t["go"]), rec,
                          t["label"] or ("scored" if t["slot"] else "—")]))
    out += ["", f"_{doc['footnote']}_", ""]
    return out


__all__ = ["ARM_XS", "ARM_TEACHERS", "ARM_SEED", "ROWS", "SEED_NOISE_RECORD", "TEACHER_RECORDS",
           "RECORD_FORMAT", "VERDICTS", "NOT_ON_RECORD", "READING_INSIDE", "READING_OUTSIDE", "HEADER",
           "candidate_stage", "read_record", "seed_noise", "reading", "teacher_values", "arm_slots",
           "side_values", "arm_row", "build_arm_tables", "render_markdown"]
