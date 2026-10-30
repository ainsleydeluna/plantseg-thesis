#!/usr/bin/env python3
"""LEFF B-1's mutation runner (LF35): every mutation of B-1's set, written into a copy of its file in a shadow of the
repository outside it, must make its named check of scripts/smoke_efficiency_record.py FAIL.

The shadow holds SHADOW_PATHS only: every file under the repository that the smoke and its child processes open,
recorded once with an audit hook on `open`. They are copied by explicit path (nothing is listed), and none names
"test"; a file missing from the list fails the control run, never a mutation. The smoke runs in the shadow with its
`--only` filter, so its child processes import the mutated copy too, and its TMPDIR inside the runner's temporary
tree, so every root a shadow smoke creates is walked at the end. Per mutation:
  <id>_anchor   each FIND occurs exactly once in its file (the edit goes into the shadow's copy only)
  <id>_killed   each named check reports FAIL in the mutated shadow (a check that never reports is not killed)
and once: control_checks_pass (every named check passes in an unmutated shadow), shadow_roots_walked (the shadow
smokes' temporary roots lie inside the walked tree), repo_state_unchanged and no_forbidden_named_path_created over the
runner's temporary tree.

    python -B scripts/smoke_efficiency_record_mutations.py [--only M01,M02]

Prints `RESULT: PASS (n/n checks; k/k edits killed)`; exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True    # no __pycache__ in the repository, with or without -B

from scripts.synthetic_ptq_fixtures import repo_state, safe_tmpdir, state_verdict  # noqa: E402

SMOKE = "scripts/smoke_efficiency_record.py"
EFF = "src/eval/efficiency.py"
PE = "scripts/profile_efficiency.py"
FORBIDDEN = "test"
LINE = re.compile(r"^  (\S+)\s*: (PASS|FAIL)")

# Every file the smoke and its children open under the repository (an audit hook on `open`, all processes).
SHADOW_PATHS: tuple[str, ...] = (
    "configs/data.py",
    "configs/model.py",
    "configs/quant.py",
    "scripts/profile_efficiency.py",
    "scripts/smoke_efficiency_record.py",
    "scripts/synthetic_ptq_fixtures.py",
    "src/__init__.py",
    "src/distill/__init__.py",
    "src/distill/cwd_projection.py",
    "src/distill/export.py",
    "src/distill/features.py",
    "src/distill/nmf_stream.py",
    "src/distill/segnext_teacher.py",
    "src/distill/teacher.py",
    "src/eval/__init__.py",
    "src/eval/efficiency.py",
    "src/eval/evaluate.py",
    "src/eval/metrics.py",
    "src/eval/model_loading.py",
    "src/eval/stage_artifacts.py",
    "src/models/__init__.py",
    "src/models/student.py",
    "src/quant/__init__.py",
    "src/quant/calibration.py",
    "src/quant/checkpoint.py",
    "src/quant/prepare.py",
    "src/quant/ptq.py",
    "src/quant/qconfig.py",
    "src/quant/stages.py",
    "src/quant/x86_latency.py",
)

# (id, file, [(FIND, REPLACE), ...], named checks, the edit). Each FIND occurs exactly once in its file.
MUTATIONS = [
    ("M01", EFF, [('STAGES = ("teacher", "E1", "E2", "E3", "A", "F", "G", "E4", "E5", "E6", "E7")',
                   'STAGES = ("teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7")')],
     ("stages_include_arms",), "A, F and G dropped from STAGES"),
    ("M02", PE, [("        if teacher_sha is None:\n", "        if teacher_sha is None:\n            return\n        if False:\n")],
     ("teacher_sha_required",), "a missing teacher sha256 accepted (the run goes on to write its record)"),
    ("M03", PE, [("                                         expected_sha256=args.teacher_ckpt_sha256)",
                  "                                         expected_sha256=None)")],
     ("teacher_hash_before_parse",), "the resolver handed no expected sha256: the file parsed before any compare"),
    ("M04", PE, [("config_path=teacher_config,\n                                              "
                  "expected_sha256=teacher_sha256)",
                  "config_path=teacher_config,\n                                              expected_sha256=None)")],
     ("teacher_hash_rechecked_at_load",), "no expected sha256 at the teacher's load"),
    ("M05", PE, [('load_student_checkpoint(resolved["checkpoint_path"])', 'load_student_checkpoint(resolved["path"])')],
     ("fp32_dispatch_no_keyerror",), 'the loader reads resolved["path"] again (F1)'),
    ("M06", PE, [("        reported, source = from_artifact, FOOTPRINT_SOURCE_ARTIFACT",
                  "        reported, source = from_companion, FOOTPRINT_SOURCE_COMPANION")],
     ("int8_footprint_from_artifact_equals_companion",), "the footprint reported from the companion alone"),
    ("M07", EFF, [("    start = read_proc_status(status)\n    with torch.inference_mode():",
                   "    start = read_proc_status(status)\n    end = read_proc_status(status)\n"
                   "    with torch.inference_mode():"),
                  ("            model(inputs)\n    end = read_proc_status(status)\n", "            model(inputs)\n")],
     ("memory_sees_forward_allocation", "adjacent_samples_removed"), "the two samples adjacent, no forward between"),
    ("M08", EFF, [("    reset = reset_rss_high_water(slack_bytes=slack, clear_refs=clear_refs, status=status)",
                   '    reset = {"slack_bytes": slack}')],
     ("memory_ignores_load_spike",), "the high-water mark not reset after the load"),
    ("M09", EFF, [('EFFICIENCY_SCHEMA = "plantseg-efficiency/2.0.0"', 'EFFICIENCY_SCHEMA = "plantseg-efficiency/1.0.0"')],
     ("schema_2_0_0",), "the record's schema 1.0.0"),
    ("M10", PE, [("        if FORBIDDEN_NAME in form.lower():", "        if False:")],
     ("forbidden_path_name_refused",), 'the "test" guard dropped'),
    ("M11", PE, [('        if actual != resolved["artifact_sha256"]:', "        if False:")],
     ("int8_converted_changed_before_load",), "the converted artifact's re-hash before its load dropped (P3)"),
    ("M12", EFF, [('STAGES = ("teacher", "E1", "E2", "E3", "A", "F", "G", "E4", "E5", "E6", "E7")',
                   'STAGES = ("teacher", "E1", "E2", "E3", "A", "F", "G", "E4", "E5", "E6", "E7", "E6-KD")')],
     ("e6kd_not_profiled",), "E6-KD admitted to STAGES"),
    ("M13", EFF, [('    if after["vm_hwm_bytes"] > after["vm_rss_bytes"] + slack_bytes:',
                   '    if after["vm_hwm_bytes"] > after["vm_rss_bytes"] + os.sysconf("SC_PAGE_SIZE"):')],
     ("hwm_slack_recorded",), "a one-page tolerance in place of the recorded slack"),
    ("M14", PE, [("    elif not footprint_counts_equal(from_artifact, from_companion):", "    elif False:")],
     ("int8_footprint_counts_disagree_refused",), "the footprint's cross-check against the companion skipped"),
    ("M15", EFF, [('                unreadable.append({"module": name or "<root>", "type": _packed_params_type(packed),'
                   ' "error": str(e)})', "                pass")],
     ("int8_footprint_fallback_named",), "an unreadable packed-params object skipped silently"),
    ("M16", PE, [('    if artifact["sha256"] != source_sha256:', "    if False:")],
     ("size_read_rehash_refused",), "the size read's re-hash dropped"),
    ("M17", EFF, [("    if digest != expected_sha256:\n        _fail(\"checkpoint_changed_after_validation\",\n"
                   "              f\"checkpoint bytes changed between validation ({expected_sha256}) and the size "
                   "companion's read \"\n              f\"({digest}); the file was not parsed\")\n"
                   "    payload = torch.load(io.BytesIO(raw), map_location=\"cpu\", weights_only=False)\n",
                   "    payload = torch.load(io.BytesIO(raw), map_location=\"cpu\", weights_only=False)\n"
                   "    if digest != expected_sha256:\n        _fail(\"checkpoint_changed_after_validation\", "
                   "\"parsed first\")\n")],
     ("size_companion_hashed_before_parse",), "the companion parses the bytes before it compares their hash"),
    ("M18", PE, [("    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):", "    if False:")],
     ("out_dir_not_empty_refused",), "an existing, non-empty --out-dir accepted"),
    ("M19", PE, [('    if info.sha256 != resolved["checkpoint_sha256"]:', "    if False:")],
     ("checkpoint_changed_after_validation",), "the FP32 checkpoint's hash not compared again at the load"),
    ("M20", EFF, [('"contaminated": excess > PARENT_CONTAMINATION_BYTES}',
                   '"contaminated": excess >= PARENT_CONTAMINATION_BYTES}')],
     ("parent_contamination_detected",), "contaminated at exactly 64 MiB"),
    ("M21", PE, [("    if REPO == out_dir or REPO in out_dir.parents:", "    if False:")],
     ("out_dir_inside_repository_refused",), "an --out-dir inside the repository accepted"),
    ("M22", PE, [('        if not getattr(args, "teacher_config", None):', "        if False:")],
     ("teacher_config_required",), "the teacher run without --teacher-config accepted"),
    ("M23", PE, [("    elif teacher_sha is not None:", "    elif False:")],
     ("teacher_sha_not_teacher",), "a teacher sha256 accepted on another stage"),
    ("M24", PE, [('        if sha_error is not None:\n            raise EfficiencyCliError(f"--teacher-ckpt-sha256 {sha_error}"',
                  '        if False:\n            raise EfficiencyCliError(f"--teacher-ckpt-sha256 {sha_error}"')],
     ("teacher_sha_format_uppercase", "teacher_sha_format_empty"),
     "the profiler's own format check dropped (the resolver's message is not evaluate_model.py's)"),
    ("M25", EFF, [("        if isinstance(mod, FloatFunctional):\n            continue\n", "")],
     ("fp32_student_counted_as_fp32",), "FloatFunctional counted as quantized again (the FP32 student refused)"),
    ("M26", PE, [("    refuse_forbidden_converted(resolved)\n", "")],
     ("converted_artifact_forbidden_name_refused",), "the converted artifact's \"test\" guard dropped"),
    ("M27", PE, [('    error = forbidden_path_error("the state_dict companion", path)\n    if error:\n'
                  '        raise EfficiencyError("path_forbidden_name", error)\n', "")],
     ("companion_forbidden_name_refused",), "the state_dict companion's \"test\" guard dropped"),
    ("M28", PE, [("    torch.set_num_threads(args.threads)\n    got = torch.get_num_threads()\n", "    got = args.threads\n")],
     ("memory_at_requested_threads",), "--threads not applied before the memory phase"),
    ("M29", PE, [('    if artifact is not None and kind != "int8_artifact":\n        artifact["model_only_companion"] = '
                  "model_only_companion(resolved)     # parses the checkpoint: after memory\n", ""),
                 ('    artifact = measure_artifact_size(resolved) if "size" in metrics else None\n',
                  '    artifact = measure_artifact_size(resolved) if "size" in metrics else None\n'
                  '    if artifact is not None and kind != "int8_artifact":\n'
                  '        artifact["model_only_companion"] = model_only_companion(resolved)\n')],
     ("memory_precedes_companion",), "the model-only companion measured before the load and the memory phase"),
    ("M30", PE, [(',\n                                 code="metrics_unknown")', ")")],
     ("metric_refusals_coded",), "the unknown-metric refusal uncoded again"),
    ("M33", PE, [('    if not rec.get("path") or not rec.get("sha256"):\n        raise EfficiencyError("state_dict_companion_'
                  'mismatch",\n', '    if False:\n        raise EfficiencyError("state_dict_companion_mismatch",\n')],
     ("state_dict_companion_mismatch",), "a run provenance recording no companion not refused"),
    ("M31", EFF, [('PROC_CLEAR_REFS = "/proc/self/clear_refs"', 'PROC_CLEAR_REFS = "/proc/self/clear_ref"')],
     ("memory_sees_forward_allocation", "hwm_reset_effective"),
     "the code's clear_refs path wrong: refused 'is absent' where the smoke's probe writes clear_refs"),
    ("M32", PE, [("        reported, source = from_artifact, FOOTPRINT_SOURCE_ARTIFACT",
                  "        reported, source = from_companion, FOOTPRINT_SOURCE_ARTIFACT")],
     ("int8_footprint_from_artifact_equals_companion",), "the companion's count reported under the artifact's label"),
]

RESULTS: list[tuple[str, bool, str]] = []
TMP: Path = None


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {name:46}: {'PASS' if ok else 'FAIL'}  {detail}", flush=True)


def make_shadow(tag: str) -> Path:
    root = TMP / tag
    for rel in SHADOW_PATHS:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO / rel).read_bytes())
    return root


def run_smoke(root: Path, checks) -> tuple[dict, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(TMP / "smoke_tmp")     # the shadow smoke's safe_tmpdir roots, walked by the last check
    p = subprocess.run([sys.executable, "-B", str(root / SMOKE), "--only", ",".join(checks)], cwd=str(root),
                       env=env, capture_output=True, text=True, timeout=3600)
    verdicts = {}
    for line in p.stdout.splitlines():
        m = LINE.match(line)
        if m:
            verdicts[m.group(1)] = m.group(2)
    return verdicts, (p.stdout + p.stderr)[-300:]


def main(argv=None) -> int:
    global TMP
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated mutation ids")
    args = ap.parse_args(argv)
    chosen = [m for m in MUTATIONS if args.only is None or m[0] in args.only.split(",")]
    before = repo_state(REPO)
    TMP = safe_tmpdir("leff_b1_mut_")
    (TMP / "smoke_tmp").mkdir()
    print("=" * 78)
    print(f"LEFF B-1 mutation runner: {len(chosen)} mutations of {len(MUTATIONS)}; shadow {len(SHADOW_PATHS)} files")
    print(f"temp {TMP}")
    print("=" * 78)
    t0 = time.time()
    check("shadow_paths_name_no_forbidden_token",
          not any(FORBIDDEN in p.lower() or ":" in p for p in SHADOW_PATHS) and len(set(SHADOW_PATHS)) == len(SHADOW_PATHS),
          f"{len(SHADOW_PATHS)} explicit paths")
    named = sorted({c for m in chosen for c in m[3]})
    control, tail = run_smoke(make_shadow("control"), named)
    bad = [c for c in named if control.get(c) != "PASS"]
    check("control_checks_pass", not bad, f"{len(named)} named checks" + (f"; not PASS: {bad} -- {tail}" if bad else ""))
    killed = 0
    for mid, rel, edits, checks, what in chosen:
        root = make_shadow(mid.lower())
        target = root / rel
        text = target.read_text(encoding="utf-8")
        counts = [text.count(find) for find, _new in edits]
        check(f"{mid}_anchor", counts == [1] * len(edits), f"{rel}: {counts}")
        if counts != [1] * len(edits):
            check(f"{mid}_killed", False, "not applied")
            continue
        for find, new in edits:
            text = text.replace(find, new, 1)
        target.write_text(text, encoding="utf-8")
        verdicts, tail = run_smoke(root, checks)
        got = {c: verdicts.get(c, "absent") for c in checks}
        ok = all(v == "FAIL" for v in got.values())
        killed += ok
        check(f"{mid}_killed", ok, f"{what}: {got}" + ("" if ok else f" -- {tail}"))
    roots = [n for n in os.listdir(TMP / "smoke_tmp") if n.startswith("leff_b1_")]
    check("shadow_roots_walked", len(roots) == len(chosen) + 1,
          f"{len(roots)} shadow smoke roots under {TMP / 'smoke_tmp'} ({len(chosen) + 1} runs)")
    check("repo_state_unchanged", state_verdict(before, repo_state(REPO)) == "unchanged",
          state_verdict(before, repo_state(REPO)))
    bad_names = [os.path.join(r, n) for r, ds, fs in os.walk(TMP) for n in ds + fs
                 if FORBIDDEN in n.lower() or ":" in n]
    check("no_forbidden_named_path_created", not bad_names, f"{len(bad_names)} under {TMP}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n({time.time() - t0:.0f} s)")
    print(f"RESULT: {'PASS' if passed == len(RESULTS) else 'FAIL'} ({passed}/{len(RESULTS)} checks; "
          f"{killed}/{len(chosen)} edits killed)")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
