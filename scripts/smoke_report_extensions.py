#!/usr/bin/env python3
"""The LREP extension report end to end (lane LREP, B-3): src/stats/report_extensions.py and
scripts/report_extensions.py against the plan of record's §1.6, §2.4, §3.3, D-13 and D-14, and
LRB7 (fq_off) and LRB18 (a separate document; the MDE label and zero-tie shares).

SYNTHETIC ONLY (DL-41): scripts/lrep_fixtures.py's world; the committed MDE entry
(reports/derived/mde_entry_20261009T034825Z.json) is read as the MDE input. Created paths live under a
new directory in TMPDIR (left in place); none contains "test".

  W  writing through the CLI: exit 0, exactly three files, SHA256SUMS verifies
  P  placement: refuse an existing out-dir, one inside the repository, one inside an input directory
  F  the profile: nonofficial with a banner on every table and the unmet conditions named; an
     unbound binding keeps it nonofficial even beside an official statistics artifact (D-13)
  C  each of the six parts present and populated
  Q  the fq_off table (LRB7): its values, driver.load_input on a qat_fakequant_off artifact, a slot
     without one, E6-KD's row
  M  the MDE part from the committed entry (LRB18)
  N  no regression: report.build_report keeps its three extension points unpopulated, and no LREP
     module imports rehearsal.py or writes report.py's names (K1)
  H  code provenance, exit codes 0 / 2 / 4, --parts, no temporary directory left after a failure,
     determinism, no 'test' path
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


def f_banner():
    md = (CTX.out / "report_extensions.md").read_text(encoding="utf-8")
    tables = md.count("\n## ")
    return CTX.doc["profile"] == "nonofficial" and md.count("**NONOFFICIAL**") >= tables + 1 >= 7, \
        f"{md.count('**NONOFFICIAL**')} banners, {tables} sections"


def f_unmet():
    names = [n for n, _ in CTX.doc["unmet_official_conditions"]]
    return {"statistics_artifact_nonofficial", "binding_not_read_by_tman"} <= set(names), str(names)


def f_unbound_official():
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
    return r["note"] == RX.FQOFF_NONE and r["fq_off"] is None and r["int8_qat"] is not None, str(r)


def q_e6kd():
    r = [r for r in part(CTX.doc, "fqoff")["rows"] if r["stage"] == "E6-KD"]
    return (len(r) == 1 and r[0]["seed"] == 42 and r[0]["fq_off"] is not None
            and r[0]["int8_qat"]["precision"] == "int8_qat"), str(r)[:300]


def m_mde():
    p = part(CTX.doc, "mde")
    src = json.loads(MDE.read_text(encoding="utf-8"))
    pairs_ok = all(p["pairs"][k][f] == src["pairs"][k][f] for k in src["pairs"]
                   for f in ("n", "n_zero", "share_ties"))
    return (pairs_ok and p["label"] == src["label"] and p["entry"]["mde_w"] == src["mde_w"]
            and p["power_caveat"] is False and "share_ties" in
            (CTX.out / "report_extensions.md").read_text(encoding="utf-8")), str(p)[:300]


def n_report_points():
    rep = RP.build_report(CTX.world.stats_path, input_root=CTX.world.input_root)
    pts = rep["extension_points"]
    return len(pts) == 3 and all(e["populated"] is False for e in pts), str(pts)


def n_no_rehearsal():
    bad = []
    for name in ("heldout_inputs", "seed_stability", "strata_tables", "default_candidates", "am6_tables",
                 "report_extensions"):
        tree = ast.parse((REPO / "src" / "stats" / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("rehearsal"):
                bad.append((name, node.module))
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("D", "H", "A", "RP", "report", "artifact", "driver")):
                bad.append((name, node.attr))
    return not bad, str(bad)


def h_code():
    code = CTX.doc["code"]
    return sorted(code["code_files"]) == sorted(RX.LREP_CODE_PATHS) and code["code_commit"], str(code)[:200]


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
            ("P1", "an existing out-dir is refused (exit 2)", p_overwrite),
            ("P2", "an out-dir inside the repository is refused", p_in_repo),
            ("P3", "an out-dir inside an input directory is refused", p_in_input),
            ("F1", "profile nonofficial and a banner on every table", f_banner),
            ("F2", "the unmet official conditions are named", f_unmet),
            ("F3", "an official statistics artifact beside a stub-read export stays nonofficial (D-13)",
             f_unbound_official),
            ("C1", "seed: ten rows, a verdict sentence", c_part("seed", lambda p: len(p["rows"]) == 10
                                                                  and p["verdict_sentence"])),
            ("C2", "strata: eight pairs", c_part("strata", lambda p: len(p["pairs"]) == 8)),
            ("C3", "defaults: D1 and D2", c_part("defaults", lambda p: [r["row"] for r in p["rows"]]
                                                 == ["D1", "D2"])),
            ("C4", "am6: 21 rows", c_part("am6", lambda p: len(p["rows"]) == 21)),
            ("C5", "fqoff: six E5/E6 rows and E6-KD", c_part("fqoff", lambda p: len(p["rows"]) == 7)),
            ("C6", "mde: the three pairs", c_part("mde", lambda p: sorted(p["pairs"]) == ["43-42", "44-42",
                                                                                         "44-43"])),
            ("Q1", "fq_off values beside int8_qat (m and the difference)", q_values),
            ("Q2", "driver.load_input passes a qat_fakequant_off artifact (LRB7)", q_load_input),
            ("Q3", "no fq_off row: 'not available: no fq_off artifact'", q_not_available),
            ("Q4", "E6-KD's row, whatever its role", q_e6kd),
            ("M1", "the MDE label, share_ties and n_zero from the committed entry (LRB18)", m_mde),
            ("N1", "report.build_report keeps three unpopulated extension points (K1)", n_report_points),
            ("N2", "no LREP module imports rehearsal.py or assigns a module attribute", n_no_rehearsal),
            ("H1", "code provenance of the seven LREP paths is recorded", h_code),
            ("H2", "exit 2 on a refusal (--parts repeated)", h_exit_2),
            ("H3", "exit 4 on an unexpected failure", h_exit_4),
            ("H4", "--parts selects parts", h_parts),
            ("H5", "no temporary directory left after a failure", h_no_tmp_left),
            ("H6", "byte-identical JSON from two runs", h_deterministic),
            ("H7", "no created path contains 'test'", h_no_test_path),
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
