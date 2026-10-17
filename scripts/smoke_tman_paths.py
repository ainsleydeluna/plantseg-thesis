#!/usr/bin/env python3
"""Smoke for the TMAN lane's path rules (lane TMAN, B-1; C1 part: names and "test" refusals).

- Every path the lane creates or names for creation is free of "test" in any case, of ":", and of <>"/\\|?* inside
  a component (TB7; IF§2): the catalogue's file_ids, every entry-file name, the record paths, the out-dir names of
  IF§2 and the lane's own files (a static case: nothing is created).
- A path containing "test" is refused as given before any filesystem call: a tracer counts os.stat, os.lstat,
  os.scandir, os.listdir, open and io.open on never-created strings and must record zero (LANGUAGE=C, as DL-90).
  The resolved form is shown with os.path.realpath patched for one call and restored in a finally block; no file
  or folder whose name contains "test" is created.

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


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN PATHS SMOKE (lane TMAN, B-1) -- names and 'test' refusals",
                                  (case_created_names, case_test_paths_refused_before_any_call)))
