"""The LREP extension report (lane LREP, B-3): the tables report.py's three extension points name,
written as a separate document (K1; LRB18), plus the fq_off table (LRB7; TL-12) and the MDE label.

report.py, rehearsal.py and their smokes are untouched: report.build_report keeps its three
`extension_points` unpopulated (scripts/smoke_stats_report.py:383-385), and this report names the
verified statistics artifact it extends.

Parts, in order (`--parts` selects a subset; plan §1.6, §2.4, §3.3):
  seed      Seed stability (AM-8; AM-19 item 6) and the per-seed tables -- src.stats.seed_stability
  strata    Rare-class strata on TEST (AM-17 item 8) -- src.stats.strata_tables
  defaults  Default candidates on TEST (AM-17 item 6) -- src.stats.default_candidates
  am6       GT-present on TEST for every model of record (AM-6; AM-17 item 1(b)) -- src.stats.am6_tables
  fqoff     AM-4's fake-quant-off TEST score of each selected QAT model of record beside its int8_qat
            value (TL-12; LRB7): the qat_fakequant_off rows of family/per_seed E5/E6 slots, and
            E6-KD's if it exists; "not available: no fq_off artifact" otherwise
  mde       the committed MDE entry's mde_w, tau_p, power_caveat, label and each pair's share_ties and
            n_zero (AM-17 item 3(c), (e), (f); DL-99; LRB18)

Every input is verified first: the statistics artifact with all thirteen checks (R-1), then the
export through the reader (R-2), then the 21 planned cells (R-3 to R-6, R-12 to R-14). The profile
follows the verified statistics artifact and is never set by a flag (D-13): "official" only if that
artifact is official, the binding is bound and read by TMAN's reader, every LREP input is official,
the strata file is the committed DL-63 file, the MDE entry is committed, and LREP's code is clean at
HEAD. While the TEST-manifest binding is unbound the statistics verifier refuses every official
artifact, so every extension report is nonofficial and every table carries a banner.

Output (D-14): a new directory outside the repository and outside every input directory, holding
exactly report_extensions.json, report_extensions.md and SHA256SUMS, written in a temporary sibling
and renamed; the temporary sibling is removed on any failure, as report.write_report does.
Import-time behaviour is side-effect free.
"""
from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

from . import heldout_inputs as H

SCHEMA = "plantseg-lrep-report-extensions/1.0.0"
PARTS = ("seed", "strata", "defaults", "am6", "fqoff", "mde")
OUT_FILES = ("SHA256SUMS", "report_extensions.json", "report_extensions.md")
LREP_CODE_PATHS = ("src/stats/heldout_inputs.py", "src/stats/seed_stability.py",
                   "src/stats/strata_tables.py", "src/stats/default_candidates.py",
                   "src/stats/am6_tables.py", "src/stats/report_extensions.py",
                   "scripts/report_extensions.py")
FQOFF_NONE = "not available: no fq_off artifact"
EXTENSION_POINTS = ("AM-8 seed stability", "Rare-class strata TEST tables", "Default candidates")


def parse_parts(value) -> tuple[str, ...]:
    """`--parts` as a comma-separated list: known names, each once; None or "all" gives every part."""
    if value is None or value == "all":
        return PARTS
    names = [p.strip() for p in str(value).split(",")]
    if not names or any(n not in PARTS for n in names) or len(set(names)) != len(names):
        raise H.refusal("parts_invalid", f"--parts {value!r}: each of {list(PARTS)} at most once")
    return tuple(p for p in PARTS if p in names)


# --------------------------------------------------------------------------------------------------
# the fq_off part (LRB7; TL-12; IF§11.1)
# --------------------------------------------------------------------------------------------------
def _score(li) -> dict:
    from .seed_stability import m_of
    run = li.summary["run"]
    return {"m": m_of(li.run.stats), "all_class_miou_summary": li.summary["dataset_level"]["all_class_miou"],
            "run_id": run["run_id"], "precision": run["precision"],
            "checkpoint_sha256": run.get("checkpoint_sha256")}


def build_fqoff(stats: H.StatsArtifact, inv: H.Inventory, cells: H.PlannedCells, *,
                input_root: Path) -> dict:
    """One row per family/per_seed E5/E6 run of record, and E6-KD's if the export lists it."""
    rows, clean, others = [], {}, {}
    targets = [(stage, seed, cells.slots[(stage, seed)], False) for stage in H.QAT_STAGES
               for seed in H.SEEDS]
    kd = sorted({r["slot"] for r in inv.doc["models"] if r.get("stage") == "E6-KD"})
    targets += [("E6-KD", None, slot, True) for slot in kd]
    for stage, seed, slot, is_kd in targets:
        out = {"stage": stage, "seed": seed, "slot": slot, "int8_qat": None, "fq_off": None,
               "difference_m": None, "note": None}
        rows.append(out)
        if not is_kd:
            st = cells.status[(stage, seed)]
            if not st.has_model:
                out["note"] = st.label
                continue
            int8 = (stats.clean(stage) if seed == 42 else
                    H.load_clean_of_record(inv, cells, stage, seed, input_root=input_root,
                                           policy=stats.policy).loaded)
        else:
            out["seed"] = H.model_row(inv, slot, "int8_qat")["seed"]
            int8 = H.load_bound(inv, slot=slot, stage=stage, precision="int8_qat", scope="clean",
                                input_root=input_root, policy=stats.policy).loaded
        label = f"{stage} {slot}"
        clean[label] = int8
        out["int8_qat"] = _score(int8)
        if not (inv.has_row(slot, H.PRECISION_FQOFF) and inv.has_record(slot, H.PRECISION_FQOFF,
                                                                         "fq_off", None)):
            out["note"] = FQOFF_NONE
            continue
        fq = H.load_bound(inv, slot=slot, stage=stage, precision=H.PRECISION_FQOFF, scope="fq_off",
                          input_root=input_root, policy=stats.policy)
        others[f"{label} fq_off"] = fq
        out["fq_off"] = _score(fq.loaded)
        out["difference_m"] = out["fq_off"]["m"] - out["int8_qat"]["m"]
    H.cross_check(clean, others, evaluator_commit=inv.evaluator_commit,
                  official=stats.profile == "official")
    return {"part": "fqoff", "title": "Fake quantization off on TEST (AM-4 supplementary; TL-12; "
                                      "descriptive)",
            "rows": rows, "footnote": "Descriptive, no test (TL-12). m is the pooled float64 "
                                      "all-class mIoU; difference = fq_off - int8_qat."}


def render_fqoff(doc: dict, banner: str = "") -> list[str]:
    out = [f"## {doc['title']}", "", banner, "| stage | seed | int8_qat m | fq_off m | fq_off - int8_qat "
           "| note |", "|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        a = r["int8_qat"]["m"] if r["int8_qat"] else None
        b = r["fq_off"]["m"] if r["fq_off"] else None
        out.append(f"| {r['stage']} | {r['seed']} | {_fmt(a)} | {_fmt(b)} | {_fmt(r['difference_m'])} | "
                   f"{r['note'] or ''} |")
    out += ["", f"_{doc['footnote']}_", ""]
    return out


# --------------------------------------------------------------------------------------------------
# the MDE part (LRB18)
# --------------------------------------------------------------------------------------------------
def build_mde(mde_entry, *, official: bool) -> dict:
    from .report import (ReportError, committed_caveat, load_mde_entry, mde_fields,
                         require_official_mde, tracked_mde_entries)
    if mde_entry is None:
        if official:
            raise H.refusal("mde_entry_invalid", "the official profile reads the committed MDE entry")
        return {"part": "mde", "title": "MDE entry (AM-17 item 3)", "entry": None,
                "power_caveat": "pending", "pairs": {}, "label": None}
    try:
        doc, status, problems = load_mde_entry(mde_entry)
    except ReportError as e:
        raise H.refusal("mde_entry_invalid", f"cannot be read: {e}") from e
    if official:
        try:
            require_official_mde(doc, status, problems, tracked_mde_entries())
        except ReportError as e:
            raise H.refusal("mde_entry_invalid", f"not the committed entry: {e}") from e
    entry = dict(status, **mde_fields(doc), problems=list(problems))
    caveat, mde_w = committed_caveat(entry)
    pairs = {}
    for name, p in sorted((doc.get("pairs") or {}).items()):
        if isinstance(p, dict):
            pairs[name] = {k: p.get(k) for k in ("n", "n_zero", "share_ties", "mde_w")}
    return {"part": "mde", "title": "MDE entry (AM-17 item 3)", "entry": entry, "power_caveat": caveat,
            "mde_w_committed": mde_w, "label": doc.get("label"), "ties_note": doc.get("ties_note"),
            "mde_w_pair": doc.get("mde_w_pair"), "pairs": pairs}


def render_mde(doc: dict, banner: str = "") -> list[str]:
    e = doc["entry"]
    out = [f"## {doc['title']}", "", banner]
    if e is None:
        return out + ["No MDE entry given: power caveat pending.", ""]
    out += [f"`{e['path']}` sha256 `{e['sha256']}`, tracked {e['tracked']}, unmodified at HEAD "
            f"{e['clean_at_head']}; mde_w {e['mde_w']} (pair {doc['mde_w_pair']}), tau_p {e['tau_p']}, "
            f"power_caveat {e['power_caveat']}; power caveat used: {doc['power_caveat']}.", "",
            f"Label: {doc['label']}", "", f"Ties: {doc['ties_note']}", "",
            "| pair | n | n_zero | share_ties | mde_w |", "|---|---|---|---|---|"]
    for name, p in doc["pairs"].items():
        out.append(f"| {name} | {p['n']} | {p['n_zero']} | {_fmt(p['share_ties'])} | {_fmt(p['mde_w'])} |")
    return out + [""]


# --------------------------------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------------------------------
def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))


def official_unmet(stats: H.StatsArtifact, inv: H.Inventory, parts_docs: dict, code: dict) -> list:
    """D-13: every condition an official extension report needs that this one does not meet."""
    unmet = []
    if stats.profile != "official":
        unmet.append(("statistics_artifact_nonofficial", f"status {stats.profile!r}, warnings "
                                                         f"{stats.family['warnings']}"))
    if not inv.bound or inv.reader == "lrep_stub":
        unmet.append(("binding_not_read_by_tman", f"binding {inv.doc['binding']['state']!r}, reader "
                                                  f"{inv.reader}"))
    strata = parts_docs.get("strata")
    if strata is not None and not strata["strata_file"]["of_record"]:
        unmet.append(("strata_not_of_record", strata["strata_file"]["path"]))
    mde = parts_docs.get("mde")
    if mde is not None and mde["power_caveat"] == "pending":
        unmet.append(("mde_entry_not_committed", str((mde["entry"] or {}).get("path"))))
    if code.get("code_clean_at_commit") is not True:
        unmet.append(("lrep_code_not_clean_at_head", str(code.get("code_clean_at_commit"))))
    return unmet


def build_report(*, stats_artifact, inventory, input_root, strata=None, mde_entry=None, parts=None,
                 reader=None, expect_inventory_sha256=None) -> dict:
    """Verify every input, build the selected parts and the profile. Refuses by code."""
    from . import seed_stability, strata_tables, default_candidates, am6_tables
    from .val_artifacts import code_provenance
    parts = parse_parts(parts)
    stats = H.load_stats_artifact(stats_artifact, input_root)
    inv = H.open_inventory(inventory, policy=stats.policy, reader=reader,
                           expect_sha256=expect_inventory_sha256)
    cells = H.planned_cells(inv)
    official = stats.profile == "official"
    docs = {}
    for part in parts:
        if part == "seed":
            docs[part] = seed_stability.build_seed_stability(stats, inv, input_root=Path(input_root))
        elif part == "strata":
            if strata is None:
                raise H.refusal("usage_invalid", "the strata part needs --strata")
            docs[part] = strata_tables.build_strata_tables(stats, inv, cells, strata_path=strata,
                                                           official=official)
        elif part == "defaults":
            docs[part] = default_candidates.build_default_candidates(stats, inv, cells,
                                                                     input_root=Path(input_root))
        elif part == "am6":
            docs[part] = am6_tables.build_am6_tables(stats, inv, cells, input_root=Path(input_root))
        elif part == "fqoff":
            docs[part] = build_fqoff(stats, inv, cells, input_root=Path(input_root))
        elif part == "mde":
            docs[part] = build_mde(mde_entry, official=official)
    code = code_provenance(LREP_CODE_PATHS)
    unmet = official_unmet(stats, inv, docs, code)
    profile = "official" if official and not unmet else "nonofficial"
    manifest = (Path(stats_artifact) / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    return {
        "schema": SCHEMA, "profile": profile, "unmet_official_conditions": [list(u) for u in unmet],
        "statistics_artifact": {"path": str(Path(stats_artifact).resolve()), "run_id": stats.family["run_id"],
                                "artifact_status": stats.profile, "warnings": stats.family["warnings"],
                                "analysis_id": stats.family["analysis_id"], "manifest": manifest,
                                "policy": stats.policy.value,
                                "am5_excluded_count_k": stats.views.am5_excluded_count},
        "inventory": {"path": str(Path(inventory).resolve()), "sha256": H.sha256_file(Path(inventory)),
                      "reader": inv.reader, "binding_state": inv.doc["binding"]["state"],
                      "binding_sha256": inv.doc["binding"].get("sha256"),
                      "manifest_sha256": inv.doc["manifest_sha256"], "freeze": inv.doc.get("freeze"),
                      "evaluator_commit": inv.evaluator_commit},
        "input_root": str(Path(input_root).resolve()), "code": code,
        "extends": {"report_extension_points": list(EXTENSION_POINTS)},
        "parts": [docs[p] for p in parts],
    }


def banner(doc: dict) -> str:
    if doc["profile"] == "official":
        return ""
    names = ", ".join(n for n, _ in doc["unmet_official_conditions"])
    return (f"> **NONOFFICIAL** -- extends statistics artifact `{doc['statistics_artifact']['run_id']}` "
            f"(status `{doc['statistics_artifact']['artifact_status']}`); unmet: {names}. Not a thesis "
            "result.\n")


def render_markdown(doc: dict) -> str:
    from . import seed_stability, strata_tables, default_candidates, am6_tables
    b = banner(doc)
    st = doc["statistics_artifact"]
    out = [f"# LREP extension report -- {st['run_id']}", "", b,
           f"Statistics artifact `{st['path']}` ({st['analysis_id']}, policy {st['policy']}, k = "
           f"{st['am5_excluded_count_k']}); export `{doc['inventory']['path']}` sha256 "
           f"`{doc['inventory']['sha256']}` (binding {doc['inventory']['binding_state']}, reader "
           f"{doc['inventory']['reader']}); code commit `{doc['code']['code_commit']}`, clean "
           f"{doc['code']['code_clean_at_commit']}.", ""]
    render = {"seed": seed_stability.render_markdown, "strata": strata_tables.render_markdown,
              "defaults": default_candidates.render_markdown, "am6": am6_tables.render_markdown,
              "fqoff": render_fqoff, "mde": render_mde}
    for part in doc["parts"]:
        out += render[part["part"]](part, b)
    return "\n".join(out) + "\n"


def check_out_dir(out_dir, inputs) -> Path:
    """D-14: a new directory, outside this repository and outside every input directory."""
    from .artifact import CODE_REPO
    out = Path(out_dir).resolve()
    if Path(out_dir).exists() or Path(out_dir).is_symlink():
        raise H.refusal("out_dir_invalid", f"refusing to overwrite {out_dir}")
    for d in [CODE_REPO.resolve(), *[Path(x).resolve() for x in inputs if x is not None]]:
        if out == d or out.is_relative_to(d):
            raise H.refusal("out_dir_invalid", f"--out-dir {out_dir} is inside {d}: the repository and "
                                               "the input directories keep their exact file sets")
    if H.path_problem(out.name) is not None:
        raise H.refusal("out_dir_invalid", f"--out-dir name {out.name!r}: {H.path_problem(out.name)}")
    stale = sorted(p for p in out.parent.glob(f".{out.name}.tmp-*")) if out.parent.is_dir() else []
    if stale:
        raise H.refusal("out_dir_invalid", f"stale temporary sibling {stale[0]}")
    return out


def write_report(out_dir, doc: dict) -> Path:
    """Atomic: a temporary sibling with the three files, renamed to `out_dir`."""
    from .artifact import canonical_json
    final = Path(out_dir)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir()
    try:
        (tmp / "report_extensions.json").write_text(canonical_json(doc), encoding="utf-8", newline="\n")
        (tmp / "report_extensions.md").write_text(render_markdown(doc), encoding="utf-8", newline="\n")
        sums = "".join(f"{hashlib.sha256((tmp / n).read_bytes()).hexdigest()}  {n}\n"
                       for n in OUT_FILES[1:])
        (tmp / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
        if final.exists() or final.is_symlink():
            raise H.refusal("out_dir_invalid", f"refusing to overwrite {final}")
        tmp.rename(final)
        return final
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def run(*, stats_artifact, inventory, out_dir, repo_root=None, synthetic: bool = False, strata=None,
        mde_entry=None, parts=None, reader=None, expect_inventory_sha256=None):
    """The CLI's body (scripts/report_extensions.py): --repo-root only with --synthetic-inputs (P16's
    rule, as report.run_report); the out-dir is checked before anything is read."""
    from .artifact import CODE_REPO, W_SYNTHETIC
    if repo_root is not None and not synthetic:
        raise H.refusal("usage_invalid", "--repo-root is accepted only with --synthetic-inputs (P16)")
    if stats_artifact is None or inventory is None or out_dir is None:
        raise H.refusal("usage_invalid", "--stats-artifact, --inventory and --out-dir are required")
    input_root = Path(repo_root) if repo_root is not None else CODE_REPO
    check_out_dir(out_dir, [stats_artifact, input_root, Path(inventory).parent])
    doc = build_report(stats_artifact=stats_artifact, inventory=inventory, input_root=input_root,
                       strata=strata, mde_entry=mde_entry, parts=parts, reader=reader,
                       expect_inventory_sha256=expect_inventory_sha256)
    if synthetic and W_SYNTHETIC not in doc["statistics_artifact"]["warnings"]:
        raise H.refusal("usage_invalid", "--synthetic-inputs was declared but the statistics artifact "
                                         "does not carry synthetic_input_data")
    return write_report(out_dir, doc), doc


__all__ = ["SCHEMA", "PARTS", "OUT_FILES", "LREP_CODE_PATHS", "FQOFF_NONE", "parse_parts", "build_fqoff",
           "render_fqoff", "build_mde", "render_mde", "official_unmet", "build_report", "banner",
           "render_markdown", "check_out_dir", "write_report", "run"]
