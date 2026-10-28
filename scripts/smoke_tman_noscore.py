#!/usr/bin/env python3
"""Smoke for decision (vi), no check reads a score (plan §4 (vi); lane TMAN, B-1, C5).

1. The static scan: the lane's code (src/tman/*.py and scripts/tday_manifest.py) holds no score-key literal outside
   its docstrings; the one allowlist that keeps a value(s) key is records.DECISION_RECORD, whose value is a grid
   value; the lane references no function of src/training/sweep_select.py that reads a score.
2. The sentinel: the readers' JSON objects raise on a score key (scripts/smoke_tman_schema.SentinelDict) while the
   world is built and every mode runs -- draft, freeze, unlock and session -- with the builder, the inventory
   export, the cut-entry check and the arithmetic printer; a sweep decision record with scores beside its grid
   values is read too.
3. The mutation test: every score field of the fixtures -- the KD, E1 and QAT telemetry rows, the E1 and teacher
   logs' scores, the selection files, the ptq_record's parent score -- set to random finite values, to "nan" (as
   the trainers write a non-finite value) or deleted: every verdict (draft and freeze, and an unlock world's unlock
   and session, its records made over the changed files), the arithmetic, the manifest and inventory bytes, and the
   command line's printed freeze record and post-freeze append stay identical, up to the sha256 of the files the
   mutation changed, of the QAT selection records that cite a changed telemetry and the commit a printer names.
   Removing a record changes the verdict.

Synthetic inputs only, under the system temp dir (removed at the end); TEST is never read.

Run:  python -B scripts/smoke_tman_noscore.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_pins as PINS  # noqa: E402
import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)
import scripts.smoke_tman_validate as VW  # noqa: E402  (the fixture world)

check = FX.check
SCORE = FX.SCORE_KEY_RE
FREEZE_FIELD_SITES = ("entries.py", "am22.py")          # where kind freeze's field `conditions` is named (B2R5)
SCORE_TEXT_RE = re.compile(r"(?P<key>[A-Za-z_]*(?:miou|loss)[A-Za-z_]*)(?P<sep>=|: )"
                           r"(?P<num>-?[0-9.]+(?:[eE]-?\d+)?|nan)", re.I)
SELECTION_RELS = ("reports/derived/lambda_selection.json", "reports/derived/alpha_selection.json")
NOW = "2026-10-21T00:00:00Z"


# ================================================================================================ helpers
def mutate_obj(o, mode: str, rng: random.Random):
    """`o` (parsed JSON) with every score key's value set to a random finite number, to "nan", or deleted."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if isinstance(k, str) and SCORE.search(k):
                if mode != "deleted":
                    out[k] = rng.uniform(-5.0, 5.0) if mode == "finite" else "nan"
            else:
                out[k] = mutate_obj(v, mode, rng)
        return out
    if isinstance(o, list):
        return [mutate_obj(x, mode, rng) for x in o]
    return o


def mutate_text(text: str, mode: str, rng: random.Random) -> str:
    def sub(m):
        if mode == "deleted":
            return ""
        return m["key"] + m["sep"] + (f"{rng.uniform(0.0, 1.0):.6f}" if mode == "finite" else "nan")
    return SCORE_TEXT_RE.sub(sub, text)


def run_files(w) -> list[str]:
    """The world's run files that hold scores, by name (no directory is listed)."""
    from src.tman import completion as C
    out = []
    for slot, e in w.entries.items():
        rid = e.get("run_id")
        kind = w.cat.slots[slot]["record_kind"]
        if not rid:
            continue
        if kind == "kd_run_end":
            out.append(f"{rid}/{C.kd_telemetry_name(w.cat.slots[slot]['stage'])}")
        elif kind == "e1_val_result":
            out += [f"{rid}/{C.E1_TELEMETRY}", f"{rid}/{C.E1_LOG}"]
        elif kind == "qat_epoch15":
            out.append(f"{rid}/{C.QAT_TELEMETRY}")
        elif kind == "teacher_m12":
            out.append(f"{rid}/20261012_010203/20261012_010203.log")
    return out


def replace_all(o, mapping: dict):
    if isinstance(o, dict):
        return {k: replace_all(v, mapping) for k, v in o.items()}
    if isinstance(o, list):
        return [replace_all(v, mapping) for v in o]
    return mapping.get(o, o) if isinstance(o, str) else o


def mutate_world(w, mode: str, seed: int) -> dict:
    """Mutate every score field of the world's run files, selection files and ptq_records; the QAT selection records
    under --runs then cite the changed telemetry's sha256 (as a selection made on it would), and the sources cite
    every changed file's sha256. Returns {new sha256: old sha256} of every file changed."""
    S, rng, shas = w.S, random.Random(seed), {}
    for rel in run_files(w):
        p = w.runs / rel
        old = p.read_bytes()
        if rel.endswith(".jsonl"):
            new = b"".join(json.dumps(mutate_obj(json.loads(x), mode, rng)).encode() + b"\n"
                           for x in old.split(b"\n") if x.strip())
        else:
            new = mutate_text(old.decode("utf-8"), mode, rng).encode("utf-8")
        if new != old:
            p.write_bytes(new)
            shas[S.sha256_bytes(new)] = S.sha256_bytes(old)
    for rel in w.qat_records:                      # each before the records citing it (the clip selection last)
        p = w.runs / rel
        old = p.read_bytes()
        new = S.json_file_bytes(replace_all(json.loads(old), {o: n for n, o in shas.items()}))
        if new != old:
            p.write_bytes(new)
            shas[S.sha256_bytes(new)] = S.sha256_bytes(old)
    files = {}
    for rel in SELECTION_RELS:
        old = (w.repo / rel).read_bytes()
        files[rel] = S.json_file_bytes(mutate_obj(json.loads(old), mode, rng))
        shas[S.sha256_bytes(files[rel])] = S.sha256_bytes(old)
    w.commit(files, f"scores mutated ({mode})")
    for rel, doc in list(w.ptq_docs.items()):
        old = S.json_file_bytes(doc)
        w.ptq_docs[rel] = mutate_obj(doc, mode, rng)
        shas[S.sha256_bytes(S.json_file_bytes(w.ptq_docs[rel]))] = S.sha256_bytes(old)
    w.sources = replace_all(w.sources, {old: new for new, old in shas.items()})
    return shas


def normalised(data: bytes, shas: dict) -> bytes:
    text = data.decode("utf-8")
    for new, old in shas.items():
        text = text.replace(new, old)
    return text.encode("utf-8")


def outputs(w) -> dict:
    """What the lane prints and decides over the world: the freeze and draft verdicts, the manifest built, the
    command line's printed freeze record and post-freeze append (DL-24 #59), the inventory export and the arithmetic
    of a KD reference run."""
    from src.tman import entries as E
    from src.tman import manifest as M
    S = w.S
    doc = M.build_manifest(w.records, w.cat, w.sources, runs_root=w.runs)
    data = S.json_file_bytes(doc)
    out = {"manifest": data, "hashes": (S.sha256_bytes(data), S.canonical_sha256(doc["body"]))}
    for mode in ("freeze", "draft"):
        code, passed, e = w.run(mode, data=data)
        out[mode] = (code, passed)
    out["head"] = FX.git(w.repo, "rev-parse", "HEAD")          # the commit a printer names (printed_by)
    out["printed_freeze"], out["append"] = cli_outputs(w, doc, data)
    fr = w.freeze_record(doc["body"])                          # committed first: w.records then reads it
    out["freeze_dl"] = fr["dl"]
    out["inventory"] = S.json_file_bytes(M.build_inventory(doc, data, w.records, w.cat, runs_root=w.runs,
                                                           registered=PINS.IF6_CELLS, freeze_record=fr["path"]))
    ref = E.reference_telemetry(w.runs, "e2_s42_lambda0.5_80k_a1", w.cat.slots["E2:s42:lambda0.5:80k"])
    out["arithmetic"] = S.json_file_bytes(E.cut_arithmetic(80000, [ref], "T", FX.utc(NOW)))
    return out


def cli_outputs(w, doc: dict, data: bytes) -> tuple:
    """The files the command line writes over the manifest `data` (DL-24 #59): its freeze record printed by `entry
    freeze`, and the manifest `append` writes for E4 seed 43's PTQ model made after the freeze, with its ptq_record
    (smoke_tman_paths.post_freeze_ptq_append's recipe). None where the command refuses."""
    from src.tman import records as R
    S, out = w.S, FX.scratch_dir("tmannscli")
    (out / "m.json").write_bytes(data)
    (out / "fields.json").write_bytes(S.json_file_bytes({"body_sha256": S.canonical_sha256(doc["body"]),
                                                          "freeze_date": VW.FREEZE_DAY, **VW.FREEZE_1F}))
    (out / "entry").mkdir()
    rc, _ = VW._cli(["--root", str(w.repo), "entry", "freeze", "--date", VW.FREEZE_DAY, "--fields",
                     str(out / "fields.json"), "--amendment", "AM-19", "--summary", "printed by a smoke.", "--out",
                     str(out / "entry"), "--nn", "60", "--manifest", str(out / "m.json"), "--runs", str(w.runs),
                     "--now", VW.AFTER_1102])
    printed = b"".join(p.read_bytes() for p in sorted((out / "entry").iterdir())) if rc == 0 else None
    rel = "E4_s43_list42/E4_s43_list42_ptq_record.json"
    (w.runs / rel).write_bytes(S.json_file_bytes(w.ptq_docs[rel]))
    d = json.loads(data)
    e = next(x for x in d["body"]["entries"] if x["slot"] == "E4:s43:list42")
    prov = VW.runs_variant(w, e["hashed"]["provenance"], f"variant_made_{VW.stamp(VW.LATE)}.json",
                           {"created_utc": VW.LATE})
    (out / "hashed.json").write_bytes(S.json_file_bytes(dict(e["hashed"], appended="append", made_utc=VW.LATE,
                                                             provenance=prov)))
    e.update(hashed=None, ptq_record=None)
    (out / "m_post.json").write_bytes(S.json_file_bytes(d))
    (out / "append").mkdir()
    saved = R.ptq_record_reader
    try:
        R.ptq_record_reader = lambda: w.reader()
        rc, _ = VW._cli(["--root", str(w.repo), "append", "--manifest", str(out / "m_post.json"), "--slot",
                         "E4:s43:list42", "--hashed", str(out / "hashed.json"), "--runs", str(w.runs), "--out",
                         str(out / "append"), "--ptq-record", rel])
    finally:
        R.ptq_record_reader = saved
    return printed, (out / "append" / "tday_manifest.json").read_bytes() if rc == 0 else None


def unlock_verdicts(edit=None) -> tuple:
    """An unlock world's unlock and session verdicts (smoke_tman_unlock.Unlock; `edit` changes its world before its
    freeze record and unlock entry are made)."""
    import scripts.smoke_tman_unlock as UL
    u = UL.Unlock(edit_world=edit)
    return u.run("unlock")[:2], u.run("session", u.start_session())[:2]


def score_mutated(mode: str, seed: int):
    """An edit_world: the world's score fields mutated (mutate_world), its manifest built again over the changed
    files and committed."""
    def edit(w):
        mutate_world(w, mode, seed)
        w.doc = w.M.build_manifest(w.records, w.cat, w.sources, runs_root=w.runs)
        w.data = w.S.json_file_bytes(w.doc)
        w.commit({w.S.MANIFEST_REL: w.data}, f"the manifest over the files mutated ({mode})")
    return edit


# ======================================================================================================= cases
def lane_python() -> list[Path]:
    from src.tman import manifest as M
    return [M.CODE_ROOT / rel for rel in M.LANE_CODE_FILES if rel.endswith(".py")]       # the code as loaded


def code_literals(path: Path) -> list[str]:
    """The string literals of a module, docstrings excepted (f-strings' literal parts included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            docs.add(id(node.body[0].value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def case_static_scan():
    from src.tman import records as R
    files = lane_python()
    check("static_scan_reads_the_lane_s_ten_modules_B2R11", len(files) == 10 and all(f.is_file() for f in files)
          and "am22.py" in [f.name for f in files], str([f.name for f in files]))
    # B2R5's field of kind freeze, AM-22's five conditions with their records: the one literal of that name, at the
    # freeze kind's sites only (its fields, entries.py; its printer and reader, am22.py; DL-24 R3); the go-rule
    # record's score-derived block of the same name is kept by no allowlist (checked below)
    hits = {f.name: [s for s in code_literals(f) if SCORE.search(s)
                     and not (s == "conditions" and f.name in FREEZE_FIELD_SITES)] for f in files}
    check("no_score_key_literal_in_the_lane_s_code", not any(hits.values()), str({k: v for k, v in hits.items() if v}))
    from src.tman import entries as E
    keepers = [n for n in dir(R) if n.isupper() and isinstance(getattr(R, n), dict)
               and "conditions" in _allow_keys(getattr(R, n))]
    check("conditions_is_the_freeze_entry_s_field_and_no_reader_keeps_it_B2R5",
          "conditions" in E.ENTRY_FILE_FIELDS["freeze"] and not keepers, str(keepers))
    holders = []
    for name in dir(R):
        a = getattr(R, name)
        if name.isupper() and isinstance(a, dict) and _allow_keys(a) & {"value", "values"}:
            holders.append(name)
    check("only_the_decision_record_keeps_a_value_key", holders == ["DECISION_RECORD"], str(holders))
    sweep = ast.parse((REPO / "src/training/sweep_select.py").read_text(encoding="utf-8"))
    readers = sorted({fn.name for fn in ast.walk(sweep) if isinstance(fn, ast.FunctionDef)
                      and any(isinstance(n, ast.Constant) and isinstance(n.value, str) and SCORE.search(n.value)
                              for n in ast.walk(fn))})
    used = sorted({n.attr for f in files for n in ast.walk(ast.parse(f.read_text(encoding="utf-8")))
                   if isinstance(n, ast.Attribute)} & set(readers))
    check("no_sweep_select_score_reader_is_referenced", "_run_end_ending" in readers and not used,
          f"readers {len(readers)}; used {used}")


def _allow_keys(a) -> set:
    out = set()
    if isinstance(a, dict):
        for k, v in a.items():
            out.add(k)
            out |= _allow_keys(v)
    elif isinstance(a, list) and a:
        out |= _allow_keys(a[0])
    return out


def case_sentinel():
    """Every mode, the builder, the export and the printers under the sentinel."""
    from src.tman import records as R
    import scripts.smoke_tman_unlock as UL
    try:
        FX.with_sentinel(lambda: R.project(R.strict_json(b'{"best_val": 0.3}', "x", code="entry_field"),
                                           {"best_val": True}))
        fired = False
    except AssertionError:
        fired = True
    check("sentinel_fires_on_a_reader_of_best_val", fired)

    def run_all():
        w = VW.World()
        res = {"freeze": w.run("freeze")[:2], "draft": w.run("draft")[:2]}
        decision = w.S.json_file_bytes({"format": "sweep_decision_record/1", "sweep": "alpha_cwd",
                                        "decision_date": "2026-10-22", "values": [
                                            {"value": v, "status": "finished", "reason": None,
                                             "directories": [{"best_val": 0.3, "all_class_miou": 0.3}]}
                                            for v in (25.0, 50.0, 100.0)]})
        w.commit({"reports/derived/alpha_decision_record.json": decision}, "a decision record with scores")
        res["freeze_with_a_decision_record"] = w.run("freeze")[:2]
        res["decision_record"] = w.records.json("reports/derived/alpha_decision_record.json", R.DECISION_RECORD,
                                                code="decision_date_cut_mismatch")[0]
        res["outputs"] = outputs(w)
        u = UL.Unlock()
        res["unlock"] = u.run("unlock")[:2]
        res["session"] = u.run("session", u.start_session())[:2]
        return res
    try:
        res = FX.with_sentinel(run_all)
        err = ""
    except AssertionError as e:
        res, err = None, str(e)
    check("sentinel_quiet_through_every_mode_builder_export_and_printer", res is not None, err)
    if res is None:
        return
    for mode in ("freeze", "draft", "freeze_with_a_decision_record", "unlock", "session"):
        want = VW.gate_codes(mode.split("_")[0])
        check(f"sentinel_{mode}_passes", res[mode] == (None, want), str(res[mode][0]))
    check("decision_record_keeps_grid_values_and_statuses_only", res["decision_record"] == {
        "decision_date": "2026-10-22", "values": [{"value": v, "status": "finished"} for v in (25.0, 50.0, 100.0)]})
    check("sentinel_builder_export_and_arithmetic_complete", all(res["outputs"][k] for k in (
        "manifest", "inventory", "arithmetic")) and res["outputs"]["freeze"] == (None, VW.gate_codes("freeze")))


def case_mutation_invariance():
    base_unlock = unlock_verdicts()
    check("unlock_world_passes_unlock_and_session_B59", base_unlock == (
        (None, VW.gate_codes("unlock")), (None, VW.gate_codes("session"))), str([v[0] for v in base_unlock]))
    for n, mode in enumerate(("finite", "nan", "deleted")):
        w = VW.World()
        base = outputs(w)
        shas = mutate_world(w, mode, seed=1009 + n)
        after = outputs(w)
        check(f"mutation_{mode}_changes_the_score_files", len(shas) >= 10, str(len(shas)))
        check(f"mutation_{mode}_verdicts_identical", (after["freeze"], after["draft"]) == (base["freeze"],
                                                                                          base["draft"])
              and base["freeze"] == (None, VW.gate_codes("freeze")), f"{after['freeze'][0]}, {after['draft'][0]}")
        # the export cites the manifest itself: its file and body sha256 follow the files it cites, and its freeze
        # row is the decision-log row of that body's freeze record
        shas.update(zip(after["hashes"], base["hashes"]))
        shas[f'"dl_row": "{after["freeze_dl"]}"'] = f'"dl_row": "{base["freeze_dl"]}"'
        diffs = [k for k in ("manifest", "inventory", "arithmetic") if normalised(after[k], shas) != base[k]]
        check(f"mutation_{mode}_manifest_inventory_arithmetic_identical_but_hashes", not diffs, str(diffs))
        shas[after["head"]] = base["head"]
        cli = [k for k in ("printed_freeze", "append") if after[k] is None or normalised(after[k], shas) != base[k]]
        check(f"mutation_{mode}_printed_freeze_record_and_append_identical_but_hashes_B59", not cli, str(cli))
        got = unlock_verdicts(score_mutated(mode, 2009 + n))
        check(f"mutation_{mode}_unlock_and_session_verdicts_identical_B59", got == base_unlock,
              str([v[0] for v in got]))
        if n == 0:
            check("the_manifest_and_export_cite_the_changed_files", after["manifest"] != base["manifest"]
                  and after["inventory"] != base["inventory"])
            check("the_cli_printed_the_freeze_record_and_wrote_the_append_B59", all(
                base[k] is not None and base[k].startswith(b"{") for k in ("printed_freeze", "append")),
                  str([base[k] is None for k in ("printed_freeze", "append")]))


def case_removal_changes_the_verdict():
    w = VW.World()
    f = w.runs / "e1_s43_80k" / "e1_telemetry.jsonl"
    saved = f.read_bytes()
    try:
        f.unlink()
        code, _, _ = w.run("freeze")
    finally:
        f.write_bytes(saved)
    check("telemetry_removed_refused", code == "completion_record_invalid", str(code))
    t = w.repo / "reports/derived/am3_trigger.json"
    saved = t.read_bytes()
    try:
        t.unlink()
        code, _, _ = w.run("freeze")
    finally:
        t.write_bytes(saved)
    check("trigger_record_removed_refused", code == "record_uncommitted", str(code))
    check("world_passes_again_after_the_removals", w.run("freeze")[0] is None)


def main() -> int:
    try:
        return FX.run_cases("TMAN NOSCORE SMOKE (lane TMAN, B-1, C5)",
                            (case_static_scan, case_sentinel, case_mutation_invariance,
                             case_removal_changes_the_verdict))
    finally:
        from src.tman import manifest as M
        if VW.REAL_TEACHER is not None:
            M.TEACHER_OF_RECORD.clear()
            M.TEACHER_OF_RECORD.update(VW.REAL_TEACHER)


if __name__ == "__main__":
    raise SystemExit(main())
