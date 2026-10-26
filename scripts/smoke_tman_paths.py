#!/usr/bin/env python3
"""Smoke for the TMAN lane's path rules (lane TMAN, B-1; C1: names and "test" refusals; C5: the code's file access
and the command line's outputs).

- Every path the lane creates or names for creation is free of "test" in any case, of ":", and of <>"/\\|?* inside
  a component (TB7; IF§2): the catalogue's file_ids, every entry-file name, the record paths, the out-dir names of
  IF§2 and the lane's own files (a static case: nothing is created).
- A path containing "test" is refused as given before any filesystem call: a tracer counts os.stat, os.lstat,
  os.scandir, os.listdir, open and io.open on never-created strings and must record zero (LANGUAGE=C, as DL-90).
  The resolved form is shown with os.path.realpath patched for one call and restored in a finally block; no file
  or folder whose name contains "test" is created.
- C5, static: the lane's code walks no directory (one listing: a teacher work dir's log folders), searches neither
  the repository root nor docs/ (its docs/ literals are two record paths), and writes only through two exclusive
  opens ("xb"): no write path reaches docs/DECISION_LOG.md.
- C5, the command line: every command writes only new files in --out, a folder outside the repository at --root
  and this checkout; a second write is refused; the scratch repository is unchanged and its decision log is never
  written (the entry command prints the row); a "test" path given to a command or a reader is refused before any
  filesystem call.

Run:  python -B scripts/smoke_tman_paths.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import builtins  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["LANGUAGE"] = "C"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)

check, expect_refused = FX.check, FX.expect_refused

# The IF§6 cell set (24 cells; R-055), injected until master has src.corruption_roles (TB6)
IF6_CELLS = ([f"{c}_s{s}" for c in ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog")
              for s in (1, 2, 3)] + ["gaussian_noise_s4"]
             + [f"{c}_s{s}" for c in ("motion_blur", "jpeg_compression", "brightness", "fog") for s in (4, 5)])
LANE_FILES = ("src/tman/__init__.py", "src/tman/schema.py", "src/tman/records.py", "src/tman/completion.py",
              "src/tman/schedule.py", "src/tman/entries.py", "src/tman/manifest.py", "src/tman/validate.py",
              "configs/tday_catalogue.json", "scripts/tday_manifest.py", "docs/lane_reports/tman.md") + tuple(
    f"scripts/smoke_tman_{n}.py" for n in ("schema", "cuts", "parent", "derived", "complete", "pins", "validate",
                                           "unlock", "noscore", "determinism", "paths", "mutations"))
MARKER = "nonexistent_tman_fx"          # a top-level folder that is never created


class Tracer:
    """Counts filesystem calls whose path argument holds `marker`, then calls through."""

    NAMES = (("os", "stat"), ("os", "lstat"), ("os", "scandir"), ("os", "listdir"), ("builtins", "open"),
             ("io", "open"))

    def __init__(self, marker: str):
        self.marker, self.calls, self.saved = marker, [], {}

    def __enter__(self):
        mods = {"os": os, "builtins": builtins, "io": io}
        for mod, name in self.NAMES:
            orig = getattr(mods[mod], name)
            self.saved[(mod, name)] = orig

            def wrapper(*a, _orig=orig, _name=f"{mod}.{name}", **kw):
                if a and self.marker in str(a[0]):
                    self.calls.append((_name, str(a[0])))
                return _orig(*a, **kw)
            setattr(mods[mod], name, wrapper)
        return self

    def __exit__(self, *exc):
        mods = {"os": os, "builtins": builtins, "io": io}
        for (mod, name), orig in self.saved.items():
            setattr(mods[mod], name, orig)
        return False


def case_created_names():
    from src.tman import schema as S
    cat = S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())
    names = [S.MANIFEST_REL, S.LANES_REL, S.ENTRIES_DIR_REL, S.CATALOGUE_REL, *LANE_FILES]
    file_ids = list(cat.by_file_id) + [S.SESSION_FILE_ID]
    for fid in file_ids:
        for kind in S.ENTRY_FILE_KINDS:
            for nn in (1, 99):
                names.append(S.entry_file_rel(S.entry_file_name(kind, fid, "2026-11-03", nn)))
        names += [f"tday_clean_{fid}", f"tday_upstream_{fid}", f"tday_fqoff_{fid}"]
        names += [f"tday_corr_{fid}_{c}" for c in IF6_CELLS]
    bad = [n for n in names if S.created_name_error(n)]
    want = 4 + len(LANE_FILES) + len(file_ids) * (2 * len(S.ENTRY_FILE_KINDS) + 3 + len(IF6_CELLS))
    check("created_names_free_of_test_colon_and_reserved_TB7", not bad and len(file_ids) == 70
          and len(IF6_CELLS) == 24 and len(names) == want == 3317, f"{len(names)} names (want {want}); bad {bad[:5]}")
    check("lane_files_free_of_test_TB7", all(S.created_name_error(p) is None for p in LANE_FILES)
          and len(LANE_FILES) == 23)


def case_test_paths_refused_before_any_call():
    from src.tman import records as R
    from src.tman import schema as S
    probes = [f"/{MARKER}/images/test/a.jpg", f"/{MARKER}/TEST/run", f"/{MARKER}/annotation_test.json",
              f"/{MARKER}/x/Test_dir/e1_telemetry.jsonl"]
    for i, p in enumerate(probes):
        with Tracer(MARKER) as t:
            code = FX.refused_code(S.refuse_test_path, p)
        check(f"test_path_refused_as_given_{i}", code == "test_path" and not t.calls, f"{code} {t.calls}")
    for name, fn, a, kw in (("read_jsonl", R.read_jsonl, (Path(probes[3]), R.KD_ROWS),
                             {"code": "completion_record_invalid"}),
                            ("read_json_file", R.read_json_file, (Path(probes[2]), R.SELECTION),
                             {"code": "reference_mismatch"}),
                            ("read_ptq_record", R.read_ptq_record, (probes[1], "0" * 64), {"reader": print})):
        with Tracer(MARKER) as t:
            code = FX.refused_code(fn, *a, **kw)
        check(f"reader_{name}_refuses_test_path_before_any_call", code == "test_path" and not t.calls,
              f"{code} {t.calls}")
    saved = os.path.realpath
    try:
        os.path.realpath = lambda p, *a, **kw: f"/{MARKER}/resolved/te" + "st/target"
        code = FX.refused_code(S.refuse_test_path, f"/{MARKER}/plain/link")
    finally:
        os.path.realpath = saved
    check("test_path_refused_as_resolved", code == "test_path", str(code))
    check("plain_path_accepted", S.test_path_error(f"/{MARKER}/plain/run") is None)
    check("marker_never_created", not os.path.lexists(f"/{MARKER}"))


# ======================================================================================================== C5
LISTING_CALLS = ("walk", "glob", "rglob", "listdir", "scandir", "iterdir")
WRITE_CALLS = ("write_bytes", "write_text", "mkdir", "makedirs", "unlink", "rmdir", "rename", "remove", "rmtree",
               "touch", "symlink_to", "copyfile", "copytree", "move", "chmod")


def lane_python() -> dict:
    import ast
    from src.tman import manifest as M
    return {rel: ast.parse((M.CODE_ROOT / rel).read_text(encoding="utf-8")) for rel in M.LANE_CODE_FILES
            if rel.endswith(".py")}


def calls(tree):
    """(name, node) of each call; the ast module's own walk is not a file access."""
    import ast
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "ast":
                continue
            yield (f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None), n


def docstrings(tree) -> set:
    import ast
    return {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}


def listing_sites(mods: dict) -> list:
    """(file, enclosing function, call) of each directory listing in the lane's code."""
    import ast
    return sorted((rel, fn.name, name) for rel, t in mods.items() for fn in ast.walk(t)
                  if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                  for name, _ in calls(fn) if name in LISTING_CALLS)


def case_static_file_access():
    import ast
    from src.tman import schema as S
    mods = lane_python()
    listing = listing_sites(mods)
    check("listings_are_the_teacher_log_and_the_runs_folder_only", len(mods) == 9
          and listing == [("src/tman/completion.py", "__init__", "iterdir"),
                          ("src/tman/completion.py", "teacher_state", "iterdir")], str(listing))
    writes = sorted((rel, name) for rel, t in mods.items() for name, n in calls(t)
                    if name in WRITE_CALLS or (name == "replace" and isinstance(n.func, ast.Attribute)
                                               and isinstance(n.func.value, ast.Name) and n.func.value.id == "os"))
    opens = sorted((rel, n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else None)
                   for rel, t in mods.items() for name, n in calls(t) if name == "open")
    check("writes_only_through_two_exclusive_opens", not writes
          and opens == [("scripts/tday_manifest.py", "xb"), ("src/tman/entries.py", "xb")], f"{writes} {opens}")
    docs = sorted({n.value for t in mods.values() for n in ast.walk(t) if isinstance(n, ast.Constant)
                   and isinstance(n.value, str) and n.value.startswith("docs") and id(n) not in docstrings(t)})
    check("docs_literals_are_two_record_paths_no_search_root", docs == sorted([S.DECISION_LOG_REL,
                                                                              S.AMENDMENTS_REL]), str(docs))


class ClosureSource:
    """An in-memory stand-in for sweep_select's source at one commit: {path: bytes}; `show` of a path holding
    "test" raises (the closure must record it by blob id without reading it). Nothing touches the filesystem."""

    def __init__(self, files: dict):
        self.files, self.shown = files, []

    def blob_at(self, commit, rel):
        return f"{abs(hash(rel)) % 10 ** 12:040d}" if rel in self.files else None

    def show(self, commit, rel):
        if "test" in rel.lower():
            raise AssertionError(f"the closure read {rel}")
        self.shown.append(rel)
        return self.files[rel]


def case_evaluator_closure_never_reads_a_test_named_module():
    """src/stats/tests.py is in the real evaluator's closure (src/stats/__init__.py imports .tests): the closure
    records such a module by the blob id git's tree gives, never reads or parses it, and follows no import of
    it; the rest of the closure is unchanged."""
    from src.tman import manifest as M
    hidden = "src/stats/te" + "sts.py"
    src = ClosureSource({"scripts/evaluate_model.py": b"from src.stats import artifact\n",
                         "src/__init__.py": b"", "src/stats/__init__.py": b"from .te" + b"sts import holm\n",
                         "src/stats/artifact.py": b"import json\n", hidden: b"",
                         "src/stats/only_from_hidden.py": b""})
    try:
        files = M.evaluator_files(src, "0" * 40)
        err = ""
    except AssertionError as e:
        files, err = {}, str(e)
    check("closure_records_a_test_named_module_by_blob_id_unread", not err and hidden in files
          and files[hidden] == src.blob_at("0" * 40, hidden) and hidden not in src.shown
          and "src/stats/only_from_hidden.py" not in files
          and sorted(files) == sorted(["scripts/evaluate_model.py", "src/__init__.py", "src/stats/__init__.py",
                                       "src/stats/artifact.py", hidden]), err or str(sorted(files)))
    check("hash_only_is_a_name_test", M.hash_only(hidden) and M.hash_only("x/TEST_y.py")
          and not M.hash_only("src/stats/artifact.py"))


def post_freeze_ptq_append(w, out: Path, base: list) -> None:
    """The append command for a PTQ model of record made after the freeze (TB19 Q9; A2): with --ptq-record, its
    intact ptq_record stands in for the isolation record; without, refused. Lane LQP's reader is the world's
    stand-in for the calls (records.ptq_record_reader patched and restored)."""
    import json
    from src.tman import records as R
    S = w.S
    rel = "E4_s43_list42/E4_s43_list42_ptq_record.json"
    data = S.json_file_bytes(w.ptq_docs[rel])
    (w.runs / rel).write_bytes(data)
    d = json.loads(w.data)
    e = next(x for x in d["body"]["entries"] if x["slot"] == "E4:s43:list42")
    (out / "hashed_post.json").write_bytes(S.json_file_bytes(dict(e["hashed"], appended="append",
                                                                  made_utc="2026-11-04T10:00:00Z")))
    e.update(hashed=None, ptq_record=None)
    (out / "m_inputs_post.json").write_bytes(S.json_file_bytes(d))
    argv = base + ["append", "--manifest", str(out / "m_inputs_post.json"), "--slot", "E4:s43:list42", "--hashed",
                   str(out / "hashed_post.json"), "--runs", str(w.runs)]
    got = []
    saved = R.ptq_record_reader
    try:
        R.ptq_record_reader = lambda: w.reader()
        for n, extra in ((1, ["--ptq-record", rel]), (2, [])):
            folder = out / f"post{n}"
            folder.mkdir()
            got.append((_cli(argv + ["--out", str(folder)] + extra), folder))
    finally:
        R.ptq_record_reader = saved
    (rc, txt), folder = got[0]
    made = json.loads((folder / "tday_manifest.json").read_bytes())["appends"] if rc == 0 else []
    check("cli_post_freeze_ptq_append_with_its_record_TB19_Q9", rc == 0 and [a["ptq_record"] for a in made] == [
        {"path": rel, "sha256": S.sha256_bytes(data)}], txt.strip()[-200:])
    (rc, txt), folder = got[1]
    check("cli_post_freeze_ptq_append_without_its_record_refused_A2", rc == 2 and "ptq_record_invalid" in txt
          and not (folder / "tday_manifest.json").exists(), txt.strip()[-200:])


def _cli(argv) -> tuple[int, str]:
    import contextlib
    import scripts.tday_manifest as CLI
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = CLI.main(argv)
    return rc, buf.getvalue()


def case_cli_outputs():
    import json
    import scripts.smoke_tman_pins as PINS
    import scripts.smoke_tman_validate as VW
    from src.tman import validate as V
    w = VW.World()
    S = w.S
    try:
        status0 = FX.git(w.repo, "status", "--porcelain", "--untracked-files=all")
        dl0 = (w.repo / S.DECISION_LOG_REL).read_bytes()
        out = FX.scratch_dir("tmanpout")
        (out / "sources.json").write_bytes(S.json_file_bytes(w.sources))
        (out / "cells.json").write_text(json.dumps(PINS.IF6_CELLS), encoding="utf-8")
        (out / "m.json").write_bytes(w.data)
        (out / "fields.json").write_bytes(S.json_file_bytes({"reason": "code not merged", "lane": "L-AM23"}))
        d = json.loads(w.data)
        e = next(x for x in d["body"]["entries"] if x["slot"] == "E4:s42:list43")
        (out / "hashed.json").write_bytes(S.json_file_bytes(dict(e["hashed"], appended="append")))
        e["hashed"] = None
        (out / "m_inputs.json").write_bytes(S.json_file_bytes(d))
        root = FX.scratch_dir("tmanroot")
        for split in ("train", "val"):
            for sub, ext in (("images", ".jpg"), ("annotations", ".png")):
                (root / sub / split).mkdir(parents=True)
                (root / sub / split / f"a{ext}").write_bytes(b"x")
        base = ["--root", str(w.repo)]
        dirs = {k: out / k for k in ("build", "entry", "inventory", "append", "isolation", "arith")}
        for p in dirs.values():
            p.mkdir()
        entry = base + ["entry", "not_made", "--slot", "E7-E2:s42:list42", "--date", "2026-11-03", "--fields",
                        str(out / "fields.json"), "--amendment", "AM-23", "--summary", "not made: code not merged.",
                        "--out", str(dirs["entry"]), "--nn", "7"]
        runs = [("build", base + ["build", "--sources", str(out / "sources.json"), "--out", str(dirs["build"]),
                                  "--runs", str(w.runs), "--registered-cells", str(out / "cells.json")]),
                ("validate", base + ["validate", "--manifest", str(out / "m.json"), "--mode", "draft",
                                     "--registered-cells", str(out / "cells.json")]),
                ("entry", entry),
                ("inventory", base + ["inventory", "--manifest", str(out / "m.json"), "--out", str(dirs["inventory"]),
                                      "--runs", str(w.runs), "--registered-cells", str(out / "cells.json")]),
                ("append", base + ["append", "--manifest", str(out / "m_inputs.json"), "--slot", "E4:s42:list43",
                                   "--hashed", str(out / "hashed.json"), "--out", str(dirs["append"]), "--runs",
                                   str(w.runs)]),
                ("isolation-record", base + ["isolation-record", "--data-root", str(root), "--out",
                                             str(dirs["isolation"])]),
                ("arithmetic", base + ["arithmetic", "--slot", "G:s42:80k", "--iterations", "80000",
                                       "--reference-telemetry", "e2_s42_lambda0.5_80k_a1", "--reference-stage", "E2",
                                       "--runs", str(w.runs), "--now", "2026-10-21T00:00:00Z", "--out",
                                       str(dirs["arith"])])]
        import src.data.isolation as ISO
        saved = ISO.DEFAULT_EXPECTED_COUNTS
        try:
            ISO.DEFAULT_EXPECTED_COUNTS = {"train": 1, "val": 1}      # the fixture root's counts
            got = {name: _cli(argv) for name, argv in runs}
        finally:
            ISO.DEFAULT_EXPECTED_COUNTS = saved
        check("every_command_passes_once", all(rc == 0 for rc, _ in got.values()),
              str({k: (rc, txt.strip().splitlines()[-1:]) for k, (rc, txt) in got.items() if rc}))
        made = {k: sorted(p.name for p in v.iterdir()) for k, v in dirs.items()}
        check("outputs_are_new_files_in_out_only", made["build"] == ["tday_manifest.json"]
              and made["inventory"] == ["tday_inventory.json"] and made["append"] == ["tday_manifest.json"]
              and len(made["entry"]) == 1 and len(made["isolation"]) == 1 and made["arith"] == [], str(made))
        first = (dirs["entry"] / made["entry"][0]).read_bytes() if made["entry"] else b""
        rc, txt = _cli(entry)
        check("a_second_write_is_refused_and_the_first_kept", rc == 2 and "entry_file_rewritten" in txt
              and (dirs["entry"] / made["entry"][0]).read_bytes() == first if made["entry"] else False)
        check("the_entry_command_prints_its_row_and_never_writes_the_log", "DECISION-LOG ROW" in got["entry"][1]
              and (w.repo / S.DECISION_LOG_REL).read_bytes() == dl0)
        appended = json.loads((dirs["append"] / "tday_manifest.json").read_bytes())
        check("cli_append_writes_the_append_with_its_ptq_record_key", [a["slot"] for a in appended["appends"]]
              == ["E4:s42:list43"] and appended["appends"][0]["ptq_record"] is None)
        post_freeze_ptq_append(w, out, base)
        inside = [("out_in_the_repository_at_root", w.repo / "reports"),
                  ("out_in_this_checkout", REPO / "nonexistent_tman_out")]
        for name, p in inside:
            rc, txt = _cli(base + ["build", "--sources", str(out / "sources.json"), "--out", str(p)])
            check(f"{name}_refused", rc == 2 and "inside the repository" in txt, txt.strip()[-160:])
        check("nothing_created_in_the_checkout", not os.path.lexists(REPO / "nonexistent_tman_out"))
        bad = f"/{MARKER}/te" + "st_out"
        for name, argv in (("out", base + ["build", "--sources", str(out / "sources.json"), "--out", bad]),
                           ("manifest", base + ["validate", "--manifest", f"/{MARKER}/te" + "st_m.json", "--mode",
                                                "draft"]),
                           ("data_root", base + ["isolation-record", "--data-root", bad, "--out", str(out)])):
            with Tracer(MARKER) as t:
                rc, txt = _cli(argv)
            check(f"cli_{name}_test_path_refused_before_any_call", rc == 2 and "test_path" in txt and not t.calls,
                  f"{rc} {t.calls}")
        with Tracer(MARKER) as t:
            code = FX.refused_code(V.read_inventory, f"/{MARKER}/te" + "st_inventory.json")
        check("read_inventory_test_path_refused_before_any_call", code == "test_path" and not t.calls)
        with Tracer(MARKER) as t:
            code = FX.refused_code(V.validate, w.ctx("freeze", runs_root=f"/{MARKER}/te" + "st_runs"))
        check("validator_runs_root_test_path_refused_before_any_call", code == "test_path" and not t.calls)
        check("the_scratch_repository_is_unchanged", FX.git(w.repo, "status", "--porcelain",
                                                             "--untracked-files=all") == status0 == "")
    finally:
        if VW.REAL_TEACHER is not None:
            w.M.TEACHER_OF_RECORD.clear()
            w.M.TEACHER_OF_RECORD.update(VW.REAL_TEACHER)


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN PATHS SMOKE (lane TMAN, B-1) -- names, 'test' refusals, file access, outputs",
                                  (case_created_names, case_test_paths_refused_before_any_call,
                                   case_static_file_access, case_evaluator_closure_never_reads_a_test_named_module,
                                   case_cli_outputs)))
