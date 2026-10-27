#!/usr/bin/env python3
"""LEFF B-2's mutation runner (LF35): every mutation of B-2's set, written into a copy of its file in a shadow of the
repository outside it, must make its named check of scripts/smoke_latency_threads.py FAIL.

The shadow holds SHADOW_PATHS only: every file under the repository that the smoke and its child processes open
(scripts/run_ptq.py's PTQ run included), recorded once with an audit hook on `open`. They are copied by explicit path
(nothing is listed), and none names "test"; a file missing from the list fails the control run, never a mutation. The
smoke runs in the shadow with its `--only` filter, so its child processes import the mutated copy too, and its TMPDIR
inside the runner's temporary tree, so every root a shadow smoke creates is walked at the end. Per mutation:
  <id>_anchor   each FIND occurs exactly once in its file (the edit goes into the shadow's copy only)
  <id>_killed   each named check reports FAIL in the mutated shadow (a check that never reports is not killed)
and once: control_checks_pass (every named check passes in an unmutated shadow), shadow_roots_walked (the shadow
smokes' temporary roots lie inside the walked tree), repo_state_unchanged and no_forbidden_named_path_created over the
runner's temporary tree.

    python -B scripts/smoke_latency_threads_mutations.py [--only M01,M02]

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

SMOKE = "scripts/smoke_latency_threads.py"
EFF = "src/eval/efficiency.py"
PE = "scripts/profile_efficiency.py"
EFT = "scripts/efficiency_table.py"
FORBIDDEN = "test"
LINE = re.compile(r"^  (\S+)\s*: (PASS|FAIL)")

# Every file the smoke and its children open under the repository (an audit hook on `open`, all processes), and
# scripts/efficiency_table.py, which the smoke imports at its top (the recording took it from a bytecode cache).
SHADOW_PATHS: tuple[str, ...] = (
    "configs/augment.py",
    "configs/data.py",
    "configs/e1_student.py",
    "configs/model.py",
    "configs/quant.py",
    "scripts/build_calibration_lists.py",
    "scripts/efficiency_table.py",
    "scripts/preflight_distill.py",
    "scripts/preflight_e1.py",
    "scripts/preflight_e1_trainval.py",
    "scripts/profile_efficiency.py",
    "scripts/qat_epoch_eval.py",
    "scripts/run_ptq.py",
    "scripts/smoke_latency_threads.py",
    "scripts/synthetic_ptq_fixtures.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/transforms.py",
    "src/distill/__init__.py",
    "src/distill/cwd_projection.py",
    "src/distill/export.py",
    "src/distill/features.py",
    "src/distill/nmf_stream.py",
    "src/distill/segnext_teacher.py",
    "src/distill/teacher.py",
    "src/eval/__init__.py",
    "src/eval/artifacts.py",
    "src/eval/efficiency.py",
    "src/eval/evaluate.py",
    "src/eval/metrics.py",
    "src/eval/model_loading.py",
    "src/eval/protocols.py",
    "src/eval/stage_artifacts.py",
    "src/models/__init__.py",
    "src/models/student.py",
    "src/quant/__init__.py",
    "src/quant/calibration.py",
    "src/quant/checkpoint.py",
    "src/quant/prepare.py",
    "src/quant/ptq.py",
    "src/quant/qat.py",
    "src/quant/qconfig.py",
    "src/quant/runner.py",
    "src/quant/stages.py",
    "src/quant/x86_latency.py",
    "src/seeds.py",
    "src/training/__init__.py",
    "src/training/losses.py",
    "src/training/train_e1.py",
)

# (id, file, [(FIND, REPLACE), ...], named checks, the edit): the plan's M1-M12, the GO's M13-M16 (LF35) and the lane's
# M17-M18. Each FIND occurs exactly once in its file.
MUTATIONS = [
    ("M01", EFF, [('"probe": probe},\n        num_threads=num_threads)', '"probe": probe})')],
     ("timer_runs_at_requested_threads",), "the Timer without num_threads (its default 1, F2)"),
    ("M02", EFF, [("\nTHREADS_OF_RECORD = 1\n", "\nTHREADS_OF_RECORD = 2\n")],
     ("threads_of_record_constants",), "THREADS_OF_RECORD = 2"),
    ("M03", PE, [('    if args.threads is None:\n        refuse("threads_required_of_record",',
                  '    if False:\n        refuse("threads_required_of_record",')],
     ("threads_required_of_record",), "--threads None accepted of record"),
    ("M04", EFF, [('        if value not in (None, "") and value.strip() != str(threads):', "        if False:")],
     ("thread_env_conflict",), "the thread-variable check skipped"),
    ("M05", EFF, [('        "intra_op_observed": (seen[0] if len(seen) == 1 else seen) if seen else None,',
                   '        "intra_op_observed": num_threads,')],
     ("observed_threads_recorded",), "the requested count recorded as the observed one"),
    ("M06", PE, [('    if actual != rec["sha256"]:\n        refuse("x86_copy_hash_mismatch",',
                  '    if False:\n        refuse("x86_copy_hash_mismatch",')],
     ("x86_copy_hash_mismatch",), "the x86 copy bound by its path only"),
    ("M07", PE, [("    if binding is not None:\n        model, backend = load_x86_copy(binding, resolved)",
                  "    if False:\n        model, backend = load_x86_copy(binding, resolved)")],
     ("e4_x86_copy_bound",), "the QNNPACK artifact timed for role x86"),
    ("M08", EFT, [('    threads = _one([t.get("intra_op_requested") for t in tp] + [t.get("intra_op_observed") for t in '
                   'tp],\n                   "table_threads_mixed", "the intra-op thread count (requested and '
                   'observed)")',
                   '    threads = (tp[0] if tp else {}).get("intra_op_requested")')],
     ("table_threads_mixed",), "the table builder ignores the thread count"),
    ("M09", EFT, [('    host = _one([(r.get("host") or {}).get("fingerprint") for r in recs], "table_host_mixed",\n'
                   '                "the host type (host_type_fingerprint)")',
                   '    host = (recs[0].get("host") or {}).get("fingerprint") if recs else None')],
     ("table_host_mixed",), "the table builder ignores the host fingerprint"),
    ("M10", PE, [('    if stage not in EFFICIENCY_ANALYSIS_STAGES:\n        refuse("stage_not_in_efficiency_analysis",',
                  '    if stage not in EFFICIENCY_ANALYSIS_STAGES + ("A",):\n'
                  '        refuse("stage_not_in_efficiency_analysis",')],
     ("stage_not_in_efficiency_analysis",), "arm A admitted of record"),
    ("M11", PE, [("    code = governed_gate(GOVERNED_ROOT)", "    code = None")],
     ("governed_paths_dirty",), "the governed-path gate skipped"),
    ("M12", PE, [('        if bound != (handed_sha, resolved["artifact_sha256"]):', "        if False:")],
     ("qat_finalize_mismatch",), "the finalize record's binding to the artifact of record skipped"),
    ("M13", PE, [("    require_unchanged(handed, handed_sha)\n", "")],
     ("source_changed_during_validation",), "the handed file not re-hashed after validation (A5)"),
    ("M14", PE, [("    except Exception as e:  # noqa: BLE001 - no usable worktree is a refusal (A7)\n",
                  "    except Exception as e:  # noqa: BLE001 - no usable worktree is a refusal (A7)\n        raise\n")],
     ("governed_paths_unavailable",), "a missing worktree raises instead of refusing (A7)"),
    ("M15", EFF, [('HOST_TYPE_KEYS = ("machine", "vendor", "model_name", "isa_flags", "threads_per_core", "usable_cpus")',
                   'HOST_TYPE_KEYS = ("machine", "vendor", "model_name", "isa_flags", "threads_per_core", "usable_cpus",'
                   ' "kernel")')],
     ("host_type_fingerprint_ignores_kernel_and_memtotal",), "the kernel fingerprinted"),
    ("M16", EFF, [('        stmt="with torch.inference_mode():\\n    model(x)",\n        setup="thread_probe(probe)",',
                   '        stmt="thread_probe(probe)\\nwith torch.inference_mode():\\n    model(x)",\n'
                   '        setup="pass",')],
     ("timer_probe_in_setup_untimed",), "the probe inside the timed statement (A15)"),
    ("M17", EFT, [('                    error = forbidden_path_error("a session record", path)     # SL-1: before it '
                   'is opened\n                    if error:\n', '                    error = None\n'
                   '                    if error:\n')],
     ("table_record_forbidden_name_refused",), "a session record's \"test\" guard dropped"),
    ("M18", EFF, [("    if _INTEROP_CALLS:\n", "    if False:\n")],
     ("interop_second_call_refused",), "torch called again on a second call in one process (torch 2.1.0 aborts)"),
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
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("PLANTSEG_")}
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
    TMP = safe_tmpdir("leff_b2_mut_")
    (TMP / "smoke_tmp").mkdir()
    print("=" * 78)
    print(f"LEFF B-2 mutation runner: {len(chosen)} mutations of {len(MUTATIONS)}; shadow {len(SHADOW_PATHS)} files")
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
    roots = [n for n in os.listdir(TMP / "smoke_tmp") if n.startswith("leff_b2_")]
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
