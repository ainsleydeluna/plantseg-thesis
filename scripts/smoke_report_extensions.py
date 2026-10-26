#!/usr/bin/env python3
"""The LREP extension report end to end (lane LREP, B-3): src/stats/report_extensions.py and
scripts/report_extensions.py against the plan of record's §1.6, §2.4, §3.3, D-13 and D-14, and
LRB7 (fq_off) and LRB18 (a separate document; the MDE label and zero-tie shares).

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's world; the committed MDE entry
(reports/derived/mde_entry_20261009T034825Z.json) is read as the MDE input. Created paths live under a
new directory in TMPDIR (left in place); none contains "test".

  W  writing through the CLI: exit 0, exactly three files, SHA256SUMS verifies, the markdown is the
     rendering of the JSON
  P  placement: refuse an existing out-dir, one inside the repository, one inside an input directory
     (the strata file's and the MDE entry's included), one whose parent does not exist
  F  the profile: nonofficial with a banner on every table and the unmet conditions named; an export
     read by LREP's stub keeps it nonofficial even beside an official statistics artifact (D-13; an
     unbound binding under Policy.OFFICIAL is seed R2f's); the inputs' split in the banner
  C  each of the six parts present and populated
  Q  the fq_off table (LRB7): its values, driver.load_input on a qat_fakequant_off artifact, a slot
     without one, E6-KD's row; a row without its record, a row of another seed, the 37's identities;
     AM-23's quant_descriptive QAT slots' rows and a cut one's label (B-5 (c))
  M  the MDE part from the committed entry (LRB18)
  N  no regression: report.build_report keeps its three extension points unpopulated, and no LREP
     file imports rehearsal.py or writes report.py's names (K1)
  H  code provenance, exit codes 0 / 2 / 4, --parts, no temporary directory left after a failure,
     determinism, no 'test' path, the frozen code's provenance in the profile (O3)
  R  refusals: parts_invalid, mde_entry_invalid, usage_invalid

Run:  TMPDIR=<a scratch directory without 'test'> python -B scripts/smoke_report_extensions.py
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import traceback
import types
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import lrep_fixtures as LF                                               # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats import heldout_inputs as H                                # noqa: E402
from src.stats import report as RP                                       # noqa: E402
from src.stats import report_extensions as RX                            # noqa: E402
from src.stats import seed_stability as SS                               # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402

CTX = types.SimpleNamespace(world=None, out=None, doc=None, paths={}, cli=None)
MDE = REPO / "reports" / "derived" / "mde_entry_20261009T034825Z.json"
CLI = REPO / "scripts" / "report_extensions.py"
#: Texts written out here rather than read from the module under test (R4-03): LRB7's note and the
#: seven LREP code paths.
FQOFF_NONE = "not available: no fq_off artifact"
LREP_FILES = ["src/stats/heldout_inputs.py", "src/stats/seed_stability.py", "src/stats/strata_tables.py",
              "src/stats/default_candidates.py", "src/stats/am6_tables.py", "src/stats/report_extensions.py",
              "scripts/report_extensions.py"]
BANNER = "**NONOFFICIAL**"


def refused(code, fn):
    try:
        fn()
    except H.LrepRefusal as e:
        return e.code == code, f"{e.code}: {e.detail[:300]}"
    except Exception as e:                                              # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, f"no refusal (expected {code})"


def cli(*args):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.run([sys.executable, "-B", str(CLI), *map(str, args)], capture_output=True, text=True,
                       env=env, cwd=str(REPO), timeout=1800)
    return p.returncode, p.stdout, p.stderr


def base_args(out, inventory=None, **extra):
    w = CTX.world
    args = ["--stats-artifact", w.stats_path, "--inventory", inventory or w.inventory_path,
            "--out-dir", out, "--strata", w.strata_path, "--mde-entry", MDE, "--synthetic-inputs",
            "--repo-root", w.input_root]
    for k, v in extra.items():
        args += [f"--{k.replace('_', '-')}", v]
    return args


def new_out(name):
    return CTX.world.fresh("outs") / name


def variant(name, edit):
    if name not in CTX.paths:
        export, binding = copy.deepcopy(CTX.world.export), copy.deepcopy(CTX.world.binding)
        edit(export, binding)
        CTX.paths[name] = LF.write_inventory(CTX.world.fresh(name), export, binding)
    return CTX.paths[name]


def part(doc, name):
    return next(p for p in doc["parts"] if p["part"] == name)


def build(**kw):
    w = CTX.world
    args = dict(stats_artifact=w.stats_path, inventory=w.inventory_path, input_root=w.input_root,
                strata=w.strata_path, mde_entry=MDE)
    args.update(kw)
    return RX.build_report(**args)


def the_doc():
    """The CLI's JSON (W1), or, when W1 has not run (the mutation harness), the same report built in
    process once and kept."""
    if CTX.doc is None:
        CTX.doc = build()
    return CTX.doc


# --------------------------------------------------------------------------------------------------
def w_written():
    out = new_out("rx_main")
    code, so, se = cli(*base_args(out))
    CTX.out = out
    if code == 0:
        CTX.doc = json.loads((out / "report_extensions.json").read_text(encoding="utf-8"))
    return code == 0 and "RESULT: REPORT EXTENSIONS WRITTEN (nonofficial)" in so, se[-400:]


def w_three_files():
    return sorted(p.name for p in CTX.out.iterdir()) == sorted(RX.OUT_FILES), ""


def w_sums():
    lines = (CTX.out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    ok = all(hashlib.sha256((CTX.out / n).read_bytes()).hexdigest() == d
             for d, n in (ln.split("  ", 1) for ln in lines))
    return ok and len(lines) == 2, str(lines)


def p_overwrite():
    code, so, _ = cli(*base_args(CTX.out))
    return code == 2 and "REFUSED (out_dir_invalid)" in so, so[-200:]


def p_in_repo():
    code, so, _ = cli(*base_args(REPO / "reports" / "lrep_never_written"))
    return code == 2 and "out_dir_invalid" in so and not (REPO / "reports" / "lrep_never_written").exists(), \
        so[-200:]


def p_in_input():
    out = CTX.world.input_root / "runs" / "lrep_never_written"
    code, so, _ = cli(*base_args(out))
    return code == 2 and "out_dir_invalid" in so and not out.exists(), so[-200:]


def p_in_strata_or_mde_dir():
    """An out-dir inside the directory of an explicit --strata file, or of an --mde-entry, outside the
    repository is refused, and nothing is written there (R3-08)."""
    w = CTX.world
    mde_copy = w.fresh("p_mde") / MDE.name
    mde_copy.write_bytes(MDE.read_bytes())
    res = []
    for inside in (w.strata_path.parent, mde_copy.parent):
        out = inside / f"rx_never_written_{uuid.uuid4().hex[:12]}"       # a mutant writes its report here
        ok, det = refused("out_dir_invalid", lambda: RX.run(
            stats_artifact=w.stats_path, inventory=w.inventory_path, out_dir=out, repo_root=w.input_root,
            synthetic=True, strata=w.strata_path, mde_entry=mde_copy, parts="mde"))
        res.append((ok and not out.exists(), det))
    return all(ok for ok, _ in res), str([d for _, d in res])


def p_missing_parent():
    """The out-dir check refuses an out-dir whose parent does not exist, and nothing is created
    (R3-09)."""
    base = CTX.world.fresh("p_parent")
    ok, det = refused("out_dir_invalid", lambda: RX.check_out_dir(base / "never_made" / "rx",
                                                                  [CTX.world.input_root]))
    return ok and not (base / "never_made").exists(), det


def p_write_parent():
    """write_report refuses an out-dir whose parent does not exist, and nothing is created (R3-09)."""
    base = CTX.world.fresh("p_write")
    ok, det = refused("out_dir_invalid", lambda: RX.write_report(base / "never_made" / "rx", the_doc()))
    return ok and not (base / "never_made").exists(), det


def w_rendering():
    """The markdown the CLI wrote is the rendering of the JSON it wrote."""
    md = (CTX.out / "report_extensions.md").read_text(encoding="utf-8")
    return md == RX.render_markdown(CTX.doc), "the markdown differs from the rendering of the JSON"


def f_banner():
    """Every table of the markdown (a block between headings with a '|---' line) carries the banner,
    and so does the preamble (D-13; R3-03, R3-04, R4-02). Rendered in process from the CLI's JSON."""
    doc = the_doc()
    md = RX.render_markdown(doc)
    blocks, cur = [], []
    for ln in md.splitlines():
        if ln.startswith("#") and cur:
            blocks.append(cur)
            cur = []
        cur.append(ln)
    blocks.append(cur)
    tables = [b for b in blocks if any(x.startswith("|---") for x in b)]
    bare = [b[0] for b in tables if not any(BANNER in x for x in b)]
    # seed 3, strata 1 and its 8 appendix pairs, defaults 1, am6 1, fqoff 1, mde 1
    return (doc["profile"] == "nonofficial" and any(BANNER in x for x in blocks[0])
            and len(tables) == 16 and not bare), f"{len(tables)} tables; without a banner: {bare}"


def f_split():
    """The inputs' split is recorded, and the banner names it when it is not TEST (R3-15). Rendered in
    process from the CLI's JSON."""
    doc = the_doc()
    md = RX.render_markdown(doc)
    return (doc["statistics_artifact"]["split"] == "val"
            and "Its inputs are the 'val' split, not TEST." in md), doc["statistics_artifact"]["split"]


def f_unmet():
    names = [n for n, _ in CTX.doc["unmet_official_conditions"]]
    return {"statistics_artifact_nonofficial", "binding_not_read_by_tman"} <= set(names), str(names)


def f_stub_read():
    """D-13: an official statistics artifact beside a stub-read export stays nonofficial."""
    st = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    fake = types.SimpleNamespace(**{**st.__dict__, "profile": "official"})
    inv = H.open_inventory(CTX.world.inventory_path, policy=st.policy)
    unmet = RX.official_unmet(fake, inv, {}, {"code_clean_at_commit": True})
    return [n for n, _ in unmet] == ["binding_not_read_by_tman"], str(unmet)


def c_part(name, check):
    def fn():
        p = part(CTX.doc, name)
        return check(p), str({k: (len(v) if isinstance(v, list) else v) for k, v in p.items()})[:300]
    return fn


def q_values():
    p = part(CTX.doc, "fqoff")
    r = next(r for r in p["rows"] if (r["stage"], r["seed"]) == ("E5", 43))
    w = CTX.world
    li_i = D.load_input("E5", D.CLEAN, w.art("E5", 43).rel, w.input_root, Policy.NONOFFICIAL_SMOKE)
    li_f = D.load_input("E5", D.CLEAN, w.art("E5", 43, H.PRECISION_FQOFF, "fq_off").rel, w.input_root,
                        Policy.NONOFFICIAL_SMOKE)
    mi, mf = SS.m_of(li_i.run.stats), SS.m_of(li_f.run.stats)
    return (r["int8_qat"]["m"] == mi and r["fq_off"]["m"] == mf and r["difference_m"] == mf - mi
            and r["fq_off"]["precision"] == H.PRECISION_FQOFF), str(r)[:300]


def q_load_input():
    """LRB7's confirmation: the frozen driver loads a qat_fakequant_off artifact unchanged."""
    w = CTX.world
    li = D.load_input("E6", D.CLEAN, w.art("E6", 42, H.PRECISION_FQOFF, "fq_off").rel, w.input_root,
                      Policy.NONOFFICIAL_SMOKE)
    return li.run.identity.precision == H.PRECISION_FQOFF and li.summary["run"]["model_role"] == "student", \
        li.run.identity.precision


def q_not_available():
    def edit(export, binding):
        export["models"] = [r for r in export["models"]
                            if (r["slot"], r["precision"]) != ("E6:s44:sel", H.PRECISION_FQOFF)]
        binding["artifacts"] = [r for r in binding["artifacts"]
                                if (r["slot"], r["precision"]) != ("E6:s44:sel", H.PRECISION_FQOFF)]
    doc = build(inventory=variant("no_fqoff", edit), parts="fqoff")
    r = next(r for r in part(doc, "fqoff")["rows"] if (r["stage"], r["seed"]) == ("E6", 44))
    return r["note"] == FQOFF_NONE and r["fq_off"] is None and r["int8_qat"] is not None, str(r)


def q_e6kd():
    r = [r for r in part(CTX.doc, "fqoff")["rows"] if r["stage"] == "E6-KD"]
    return (len(r) == 1 and r[0]["seed"] == 42 and r[0]["fq_off"] is not None
            and r[0]["int8_qat"]["precision"] == "int8_qat"), str(r)[:300]


def q_row_without_record():
    """A slot whose qat_fakequant_off row has no binding record is refused, never 'not available'
    (LRB3; R1r1-09)."""
    def edit(export, binding):
        binding["artifacts"] = [r for r in binding["artifacts"]
                                if (r["slot"], r["precision"]) != ("E6:s44:sel", H.PRECISION_FQOFF)]
    return refused("artifact_missing", lambda: build(inventory=variant("fqoff_no_record", edit),
                                                     parts="fqoff"))


def q_row_other_seed():
    """A slot's qat_fakequant_off row of another seed than its int8_qat row is refused (IF§11.1; LRB5;
    R1r1-01)."""
    def edit(export, binding):
        r = next(r for r in export["models"] if (r["slot"], r["precision"]) == ("E5:s43:sel",
                                                                              H.PRECISION_FQOFF))
        r["seed"] = 44
    return refused("seed_inconsistent", lambda: build(inventory=variant("fqoff_seed", edit), parts="fqoff"))


def q_quant():
    """B-5 (c): AM-23's quant_descriptive QAT slots (E6-E2, E6-R1, E6-R2) join the fq_off table at seed
    42, each fq_off m beside its int8_qat m, both loaded here at the evaluator's own stage."""
    w = CTX.world
    rows = {r["stage"]: r for r in part(CTX.doc, "fqoff")["rows"]
            if r["stage"] in ("E6-E2", "E6-R1", "E6-R2")}
    bad = []
    for stage, slot in (("E6-E2", "E6-E2:s42:sel"), ("E6-R1", "E6-R1:s42:sel"), ("E6-R2", "E6-R2:s42:sel")):
        mi = SS.m_of(D.load_input(stage, D.CLEAN, w.written[(slot, "int8_qat", "clean", None)].rel,
                                  w.input_root, Policy.NONOFFICIAL_SMOKE).run.stats)
        mf = SS.m_of(D.load_input(stage, D.CLEAN, w.written[(slot, H.PRECISION_FQOFF, "fq_off", None)].rel,
                                  w.input_root, Policy.NONOFFICIAL_SMOKE).run.stats)
        r = rows.get(stage)
        if not (r and r["seed"] == 42 and r["slot"] == slot and r["int8_qat"]["m"] == mi
                and r["fq_off"]["m"] == mf and r["difference_m"] == mf - mi and r["note"] is None):
            bad.append(stage)
    return not bad and list(rows) == ["E6-E2", "E6-R1", "E6-R2"], str(bad)


def q_quant_label():
    """A quant_descriptive QAT run that was cut (DL-137's shape for E6-R1) reads its IF§11.2 label from
    TMAN's entries in the fq_off table, with no value and no refusal."""
    def edit(export, binding):
        e = next(e for e in export["entries"] if e["slot"] == "E6-R1:s42")
        e.update(kind="cut", cut_date="2026-10-24", rule="AM-22 item 2(a): the launch cutoff",
                 comparator=None)
        next(e for e in export["entries"] if e["slot"] == "E6-R1:s42:sel")["kind"] = None
        export["models"] = [r for r in export["models"] if r["slot"] != "E6-R1:s42:sel"]
        binding["artifacts"] = [r for r in binding["artifacts"] if r["slot"] != "E6-R1:s42:sel"]
    doc = build(inventory=variant("e6r1_cut", edit), parts="fqoff")
    r = next(r for r in part(doc, "fqoff")["rows"] if r["stage"] == "E6-R1")
    return (r["note"] == "cut (2026-10-24, AM-22 item 2(a): the launch cutoff)" and r["int8_qat"] is None
            and r["fq_off"] is None and r["seed"] == 42), str(r)


def q_seed42_identity():
    """--parts fqoff alone checks the 37's E5 and E6 identities against the family slots (R-11;
    R1r1-01)."""
    def edit(export, binding):
        r = next(r for r in export["models"] if (r["slot"], r["precision"]) == ("E6:s42:sel", "int8_qat"))
        r["converted_artifact_sha256"] = LF.syn_sha("another converted artifact")
    return refused("seed42_mismatch", lambda: build(inventory=variant("fqoff_identity", edit),
                                                    parts="fqoff"))


def m_mde():
    """The MDE part and its markdown: the label line and one row per pair (LRB18)."""
    p = part(CTX.doc, "mde")
    src = json.loads(MDE.read_text(encoding="utf-8"))
    md = (CTX.out / "report_extensions.md").read_text(encoding="utf-8").splitlines()
    pairs_ok = all(p["pairs"][k][f] == src["pairs"][k][f] for k in src["pairs"]
                   for f in ("n", "n_zero", "share_ties"))
    rows = [f"| {k} | {v['n']} | {v['n_zero']} | {format(v['share_ties'], '.6g')} | "
            for k, v in sorted(src["pairs"].items())]
    rows_ok = all(sum(1 for ln in md if ln.startswith(r)) == 1 for r in rows)
    return (pairs_ok and rows_ok and f"Label: {src['label']}" in md and p["label"] == src["label"]
            and p["entry"]["mde_w"] == src["mde_w"] and p["power_caveat"] is False), str(rows)[:300]


def n_report_points():
    rep = RP.build_report(CTX.world.stats_path, input_root=CTX.world.input_root)
    pts = rep["extension_points"]
    return len(pts) == 3 and all(e["populated"] is False for e in pts), str(pts)


def n_no_rehearsal():
    """No LREP file imports rehearsal.py in any form (`import ...rehearsal`, `from ... import
    rehearsal`, `from ...rehearsal import ...`) or assigns a frozen module's attribute (K1)."""
    bad = []
    for name in LREP_FILES:
        tree = ast.parse((REPO / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and ((node.module or "").split(".")[-1] == "rehearsal"
                                                     or any(a.name == "rehearsal" for a in node.names)):
                bad.append((name, node.module, [a.name for a in node.names]))
            if isinstance(node, ast.Import) and any(a.name.split(".")[-1] == "rehearsal" for a in node.names):
                bad.append((name, [a.name for a in node.names]))
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("D", "H", "A", "RP", "report", "artifact", "driver")):
                bad.append((name, node.attr))
    return not bad, str(bad)


def h_code():
    code = CTX.doc["code"]
    return (sorted(code["code_files"]) == sorted(LREP_FILES) == sorted(RX.LREP_CODE_PATHS)
            and code["code_commit"]), str(code)[:200]


def h_report_code():
    """The frozen code that computes the numbers (artifact.OFFICIAL_CODE_PATHS) and the CLI are recorded,
    and a path of them not clean at HEAD keeps the report nonofficial (O3; R3-01)."""
    want = ["src/stats", "src/stats/driver.py", "scripts/run_stats.py", "configs/corruption_protocol.json",
            "docs/STATISTICAL_ANALYSIS_CONTRACT.md", "scripts/report_extensions.py"]
    code = the_doc()["code"]
    st = H.load_stats_artifact(CTX.world.stats_path, CTX.world.input_root)
    fake = types.SimpleNamespace(**{**st.__dict__, "profile": "official"})
    inv = types.SimpleNamespace(bound=True, reader="tman.read_inventory", doc={"binding": {"state": "bound"}})
    dirty = ["code_not_clean_at_head", "src/stats/driver.py:  M src/stats/driver.py"]
    unmet = RX.official_unmet(fake, inv, {}, {"code_clean_at_commit": True, "report_code_unmet": [dirty]})
    return (code["report_code_paths"] == want == [*A.OFFICIAL_CODE_PATHS, want[-1]]
            and isinstance(code["report_code_unmet"], list) and unmet == [tuple(dirty)]), str(unmet)


def h_exit_2():
    code, so, _ = cli(*base_args(new_out("rx_parts_bad"), parts="seed,seed"))
    return code == 2 and "REFUSED (parts_invalid)" in so, so[-200:]


def h_exit_4():
    spec = importlib.util.spec_from_file_location("lrep_cli", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    orig = RX.run
    RX.run = lambda **kw: (_ for _ in ()).throw(RuntimeError("injected"))
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            code = mod.main([str(a) for a in base_args(new_out("rx_exit4"))])
    finally:
        RX.run = orig
    return code == 4 and "RESULT: ERROR" in buf_out.getvalue(), str(code)


def h_parts():
    doc = build(parts="seed,mde")
    return [p["part"] for p in doc["parts"]] == ["seed", "mde"], ""


def h_no_tmp_left():
    out = new_out("rx_fail")
    orig = RX.render_markdown
    RX.render_markdown = lambda doc: (_ for _ in ()).throw(RuntimeError("injected"))
    try:
        try:
            RX.write_report(out, CTX.doc)
            raised = False
        except RuntimeError:
            raised = True
    finally:
        RX.render_markdown = orig
    left = [p.name for p in out.parent.iterdir() if p.name.startswith(f".{out.name}.tmp-")]
    return raised and not out.exists() and not left, str(left)


def h_deterministic():
    out = new_out("rx_again")
    code, _, se = cli(*base_args(out))
    if code != 0:
        return False, se[-300:]
    a = (CTX.out / "report_extensions.json").read_bytes()
    b = (out / "report_extensions.json").read_bytes()
    return a == b, "the JSON differs between two runs"


def h_no_test_path():
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(CTX.world.root) for n in dirs + files
           if "test" in n.lower()]
    return not bad, str(bad[:3])


def r_parts():
    try:
        RX.parse_parts("seed,bogus")
    except H.LrepRefusal as e:
        return e.code == "parts_invalid", str(e)
    return False, "accepted"


def r_mde():
    d = CTX.world.fresh("mde_missing")
    code, so, _ = cli(*base_args(new_out("rx_mde"), mde_entry=d / "mde_entry_absent.json"))
    return code == 2 and "REFUSED (mde_entry_invalid)" in so, so[-200:]


def r_usage():
    args = [a for a in base_args(new_out("rx_usage")) if a != "--synthetic-inputs"]
    code, so, _ = cli(*args)
    return code == 2 and "REFUSED (usage_invalid)" in so, so[-200:]


def setup(root: Path, world=None) -> None:
    CTX.world = world or LF.build_world(root)
    CTX.out, CTX.doc, CTX.paths = None, None, {}


def cases():
    return [("W1", "the CLI writes the report (exit 0, nonofficial)", w_written),
            ("W2", "exactly report_extensions.json, report_extensions.md and SHA256SUMS", w_three_files),
            ("W3", "SHA256SUMS verifies", w_sums),
            ("W4", "the markdown is the rendering of the JSON", w_rendering),
            ("P1", "an existing out-dir is refused (exit 2)", p_overwrite),
            ("P2", "an out-dir inside the repository is refused", p_in_repo),
            ("P3", "an out-dir inside an input directory is refused", p_in_input),
            ("P4", "an out-dir inside the --strata file's or the --mde-entry's directory is refused",
             p_in_strata_or_mde_dir),
            ("P5", "the out-dir check refuses a missing parent and creates nothing", p_missing_parent),
            ("P6", "write_report refuses a missing parent and creates nothing", p_write_parent),
            ("F1", "profile nonofficial and a banner on every table", f_banner),
            ("F2", "the unmet official conditions are named", f_unmet),
            ("F3", "an official statistics artifact beside a stub-read export stays nonofficial (D-13)",
             f_stub_read),
            ("F4", "the inputs' split is recorded and named in the banner", f_split),
            ("C1", "seed: ten rows, a verdict sentence", c_part("seed", lambda p: len(p["rows"]) == 10
                                                                  and p["verdict_sentence"])),
            ("C2", "strata: eight pairs", c_part("strata", lambda p: len(p["pairs"]) == 8)),
            ("C3", "defaults: D1 and D2", c_part("defaults", lambda p: [r["row"] for r in p["rows"]]
                                                 == ["D1", "D2"])),
            ("C4", "am6: 21 rows", c_part("am6", lambda p: len(p["rows"]) == 21)),
            ("C5", "fqoff: six E5/E6 rows, E6-KD and the three quant_descriptive QAT slots (B-5 (c))",
             c_part("fqoff", lambda p: len(p["rows"]) == 10)),
            ("C6", "mde: the three pairs", c_part("mde", lambda p: sorted(p["pairs"]) == ["43-42", "44-42",
                                                                                         "44-43"])),
            ("Q1", "fq_off values beside int8_qat (m and the difference)", q_values),
            ("Q2", "driver.load_input passes a qat_fakequant_off artifact (LRB7)", q_load_input),
            ("Q3", "no fq_off row: 'not available: no fq_off artifact'", q_not_available),
            ("Q4", "E6-KD's row, whatever its role", q_e6kd),
            ("Q5", "an fq_off row without its binding record: artifact_missing", q_row_without_record),
            ("Q6", "an fq_off row of another seed than its int8_qat row: seed_inconsistent",
             q_row_other_seed),
            ("Q7", "--parts fqoff: an E6 family identity that is not the 37's: seed42_mismatch",
             q_seed42_identity),
            ("Q8", "B-5 (c): the quant_descriptive QAT slots' fq_off rows beside their int8_qat", q_quant),
            ("Q9", "B-5 (c): a cut quant_descriptive QAT run reads its label in the fq_off table",
             q_quant_label),
            ("M1", "the MDE label line, and share_ties and n_zero per pair from the committed entry (LRB18)",
             m_mde),
            ("N1", "report.build_report keeps three unpopulated extension points (K1)", n_report_points),
            ("N2", "no LREP file imports rehearsal.py or assigns a module attribute", n_no_rehearsal),
            ("H1", "code provenance of the seven LREP paths is recorded", h_code),
            ("H2", "exit 2 on a refusal (--parts repeated)", h_exit_2),
            ("H3", "exit 4 on an unexpected failure", h_exit_4),
            ("H4", "--parts selects parts", h_parts),
            ("H5", "no temporary directory left after a failure", h_no_tmp_left),
            ("H6", "byte-identical JSON from two runs", h_deterministic),
            ("H7", "no created path contains 'test'", h_no_test_path),
            ("H8", "the frozen code and the CLI are recorded and gate the profile (O3)", h_report_code),
            ("R1", "parts_invalid", r_parts),
            ("R2", "mde_entry_invalid (an MDE path that does not exist)", r_mde),
            ("R3", "usage_invalid (--repo-root without --synthetic-inputs)", r_usage)]


def run_case(fn):
    try:
        ok, detail = fn()
        return bool(ok), str(detail)
    except Exception as e:                                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def main() -> int:
    t0 = time.time()
    root, why = LF.make_root("lrep_report_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    print(f"smoke_report_extensions: temporary root {root} (left in place)", flush=True)
    setup(root)
    results = [(cid, desc, *run_case(fn)) for cid, desc, fn in cases()]
    for cid, desc, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {desc}")
        if not ok:
            print(f"         {detail[:500]}")
    good = sum(1 for r in results if r[2])
    print(f"\nRESULT: {'REPORT EXTENSIONS OK' if good == len(results) else 'REPORT EXTENSIONS FAILED'} "
          f"({good}/{len(results)}); {time.time() - t0:.0f} s")
    print("NONOFFICIAL: synthetic inputs; no report here is a thesis result.")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
