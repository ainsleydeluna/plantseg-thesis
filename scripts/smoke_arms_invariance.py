#!/usr/bin/env python3
"""Default-off invariance of the KD arm seam (lane L-TEACHER-ARMS, commit C1; ARMS plan decision (vii), section 4.3).

With --arm absent, the trainer (src/training/train_distill.py), `entry`, the gate and check-run-meta must
behave as at OLD (8487551 by default, the alpha sweep's pin; audit ruling A5). This smoke extends
scripts/smoke_invariance_distill.py, whose --cross-commit mode compares E2 and E3 step by step (d4, L2d3), with
run_meta, payload, best.json, blob, check-run-meta and launch-line comparisons and a negative control. CPU only,
GPUs hidden: the synthetic 16 + 4 TRAIN/VAL set and the stub teacher of scripts/invariance_harness.py (batch 2,
seed 42, 2 steps with a validation at step 2); no dataset, checkpoint, GPU or download. This smoke's own export
of OLD is `git archive <OLD> -- src configs scripts reports/e1_class_weights.json` with the two exclude
suffixes (errata E-5; SL-1), refused before anything is written if a member name still contains "test".
Check 1's child runs as plan (vii) item 1 names it, `--cross-commit <OLD> WORKTREE`, and makes its own export
of src and configs without the SL-1 exclude (smoke_invariance_distill.py, unchanged): OLD's test-named files
are extracted into <work>/child/export_old and never opened. With --new-root the child is given this smoke's
export instead (--old-root).

Checks (16):
  inv_cross_commit_child_pass           smoke_invariance_distill.py --cross-commit <OLD> WORKTREE (with
                                        --new-root: --old-root <this smoke's export> --new-root <dir>) exits 0,
                                        RESULT PASS: E2 and E3 step JSONs byte-identical to OLD's (losses,
                                        targets, gradients, weights, teacher outputs, NMF stream position, RNG
                                        states), E3 with an explicit alpha 50 too, and its alpha-25 negative
                                        control detected
  inv_run_meta_equal_<s>      x5        s = e2, e3, a, f, g: NEW's run_meta row is byte-identical to OLD's once
                                        the run-identity keys wall_clock, git_head and git_head_source are dropped
  inv_payload_keys_equal_<s>  x5        each checkpoint payload's key set, with its stage, terms, projection and
                                        optimizer-group facts, equals OLD's
  inv_best_json_equal                   each stage's best.json equals OLD's, key by key (best_ckpt compared by
                                        file name)
  inv_blob_ids_unchanged                every file OLD has under src/, scripts/ and configs/ (a name containing
                                        "test" is skipped as a string, never opened) is in NEW with OLD's git blob
                                        id, except src/training/train_distill.py (the seam); the five C1 paths
                                        exist; no file NEW adds there is a module that shadows one of OLD's. So
                                        `entry`, the gate, check-run-meta, the selection code, the losses and the
                                        frozen files are OLD's code on any later tree that only adds files
  inv_check_run_meta_default_pass       check-run-meta's rule table (preflight_distill.meta_problems, the
                                        --allow-smoke expectations) on NEW's default E3 row: the row holds exactly
                                        the E3 keys, every key it flags is one of HARNESS_FACTS, and its verdicts
                                        equal those on OLD's row (run identity aligned). A literal PASS needs the
                                        pod's dry row: CUDA, batch 16, 12 workers, the KD image, the teacher of
                                        record and 25 iterations of the 5,367-image TRAIN set
  inv_launch_line_bytes_equal           `entry`'s launch line (preflight_distill.launch_line) for the five lambda
                                        and the three alpha candidates has the same bytes from OLD's and NEW's code
  inv_negative_control_arm_run_differs  the comparisons tell an arm run apart: a NEW E3 run with arm e3-r1
                                        differs from OLD's E3 run_meta in exactly arm "E3-R1", descriptive true,
                                        parent_of_e4_e7 false and a last key target_construction {K 1, views 1};
                                        its payloads carry OLD's keys and facts plus arm "E3-R1" and that
                                        target_construction; its step JSONs equal NEW E3's (an arm without a
                                        target builder trains as E3); and main(), with run() recorded, hands run()
                                        the registered e3-r1 for `--arm e3-r1`, while without --arm it hands run()
                                        OLD main()'s keywords and never imports the registry
Real-run-only paths (the AM-7 abort record and main()'s real-run gates) are outside this smoke. C1 extends the
abort record's teacher_finite with `and (targets is None or bool(targets.finite))`, which is `and True`, the old
value, in every run without targets.

Usage:  python -B scripts/smoke_arms_invariance.py [--old SHA] [--new-root DIR] [--work-dir DIR] [--threads N]
                                                     [--section child|workers|static ...]
  --new-root  the tree under test (default: this checkout); a mutation run points it at a scratch copy
  --work-dir  a new folder outside the repository (default: <tempdir>/arms_smoke_<pid>_<n>)
  --section   only these sections (repeatable): child (check 1), workers (the worker-run checks), static
              (blob ids, launch lines); the count printed is the count of the checks run
The smoke re-invokes itself with `--internal NAME ...` in fresh processes (the arm worker, check-run-meta's
rule table, the launch lines, the payload fields and the main() probe); that is not part of its CLI.
Exit codes: 0 PASS; 1 a check failed (a run that raises or times out fails its section's checks); 2 refused,
nothing written (a path containing "test", an existing --work-dir, a --work-dir inside the repository, a
--new-root without the trainer, scripts/ or configs/, an OLD that does not resolve or does not export).
Outputs, kept for inspection, go to --work-dir, never into the repository.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.machinery
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import types
from pathlib import Path, PurePosixPath

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only smoke: hide GPUs from every process it starts

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Standard-library-only modules: nothing under src/ or configs/ is imported at module level, so the internal
# children import the trainer from the code root under test.
from scripts.invariance_harness import (EXPORT_EXCLUDE, compare_step_dirs, export_commit,  # noqa: E402
                                        make_synthetic_dataset, run_worker)
from scripts.smoke_frozen_blobs import git_blob_id  # noqa: E402

OLD_DEFAULT = "8487551ca67bd50ffbe911d1f3e93244841cbc9f"
C1_PATHS = ("configs/kd_arms/e3_r1.json", "configs/kd_arms/e3_r2.json", "scripts/smoke_arms_invariance.py",
            "src/training/kd_arms.py", "src/training/train_distill.py")
SEAM_PATHS = ("src/training/train_distill.py",)          # the one file of OLD that C1 modifies
TEST_EXCLUDE = ":(exclude,icase)*test*"
EXPORT_PATHS = ("src", "configs", "scripts", "reports/e1_class_weights.json")
BLOB_TOPS = ("src", "scripts", "configs")
STAGES = ("e2", "e3", "a", "f", "g")
STEPS, VAL_INTERVAL = 2, 2
WORKER_TIMEOUT, CHILD_TIMEOUT, SUB_TIMEOUT = 3600, 7200, 1800       # seconds; below STEP 5's 10,800
IDENTITY_KEYS = ("wall_clock", "git_head", "git_head_source")
# The run_meta keys whose value on this smoke's CPU row is a fact of the harness or of the pod, not of the
# code: batch 2, 2 steps, a validation every 2, no loader workers, CPU torch, no KD image, the stub teacher
# (no provenance) and the 16-image TRAIN set (ramp_iters is one epoch of it).
HARNESS_FACTS = ("batch_size", "max_iters", "val_interval", "num_workers", "persistent_workers", "device",
                 "cuda_available", "torch", "image_digest", "teacher_provenance", "ramp_iters")
ARM = "e3-r1"
ARM_FIELDS = {"arm": "E3-R1", "descriptive": True, "parent_of_e4_e7": False,
              "target_construction": {"K": 1, "views": 1}}
LAUNCH_CASES = (("e2", 1.0, None), ("e2", 0.5, None), ("e2", 2.0, None), ("e2", 0.25, None), ("e2", 4.0, None),
                ("e3", 1.0, 50.0), ("e3", 1.0, 25.0), ("e3", 1.0, 100.0))
PROBE_ARGV = ["--stage", "e3", "--dry-run", "--lambda-logit", "1"]     # main()'s dry E3 launch, run() recorded
SECTIONS = ("child", "workers", "static")
SECTION_CHECKS = {
    "child": ["inv_cross_commit_child_pass"],
    "workers": ([f"inv_{kind}_{s}" for s in STAGES for kind in ("run_meta_equal", "payload_keys_equal")]
                + ["inv_best_json_equal", "inv_check_run_meta_default_pass", "inv_negative_control_arm_run_differs"]),
    "static": ["inv_blob_ids_unchanged", "inv_launch_line_bytes_equal"],
}

results: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail="") -> None:
    results.append((name, bool(ok), str(detail)))


# ------------------------------------------------------------------------------- internal children
def _child_paths(*values) -> None:
    """An internal child takes only paths the smoke built from its guarded roots; each is refused as a string
    (SL-1) before any use."""
    for v in values:
        if v is not None and "test" in str(v).lower():
            print("REFUSED: an internal path containing 'test'", file=sys.stderr)
            raise SystemExit(2)


def _code_root_first(code_root: Path) -> None:
    """invariance_harness.worker's rule: the code root first, no other checkout of this repository after it."""
    sys.path[:] = [str(code_root)] + [p for p in sys.path if p and Path(p).resolve() != code_root
                                      and not (Path(p) / "src" / "training" / "train_distill.py").exists()]


def arm_worker(argv) -> int:
    """invariance_harness's worker, unchanged, with run() given the registered arm: the worker's own imports
    and calls run in its own order, and the arm is bound to run() when the worker installs its recorder,
    just before its run() call."""
    ap = argparse.ArgumentParser(prog="smoke_arms_invariance.py --internal arm_worker")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--code-root", required=True)
    a, rest = ap.parse_known_args(argv)
    _child_paths(a.code_root, *rest)
    code_root = Path(a.code_root).resolve()
    import scripts.invariance_harness as harness
    install = harness._Recorder.install

    def install_with_arm(rec) -> None:
        install(rec)
        from src.training import kd_arms
        if code_root not in Path(kd_arms.__file__).resolve().parents:
            raise SystemExit(f"kd_arms was imported from {kd_arms.__file__}, outside the code root {code_root}")
        spec = kd_arms.load_arm(a.arm)
        run = rec.td.run

        @functools.wraps(run)
        def run_with_arm(*args, **kwargs):
            return run(*args, arm=spec, **kwargs)
        rec.td.run = run_with_arm
    harness._Recorder.install = install_with_arm
    return harness.worker(["--code-root", str(code_root), *rest])


def _load_preflight(code_root: Path):
    """The code root's scripts/preflight_distill.py, loaded by file location (its own pattern for the E1 gate)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("preflight_distill_of_root",
                                                  str(code_root / "scripts" / "preflight_distill.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crm_child(argv) -> int:
    """check-run-meta's rule table from the code root, applied to the rows of a JSON file."""
    ap = argparse.ArgumentParser(prog="smoke_arms_invariance.py --internal crm")
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    a = ap.parse_args(argv)
    _child_paths(a.code_root, a.rows, a.ckpt_dir)
    code_root = Path(a.code_root).resolve()
    _code_root_first(code_root)
    import src.training.train_distill as td
    pd = _load_preflight(code_root)
    rows = json.loads(Path(a.rows).read_text(encoding="utf-8"))
    exp = pd.run_meta_expectations(argparse.Namespace(
        allow_smoke=True, profile="distill_80k", stage="e3", seed=42, expect_head="0" * 40, lambda_logit=1.0,
        alpha=50.0, ckpt_dir=a.ckpt_dir, gate_record=None))
    out = {"run_meta_keys_e3": pd.run_meta_keys("e3")}
    for label, row in rows.items():
        problems, warns = pd.meta_problems(row, "e3", exp, td)
        out[label] = {"problems": problems, "warnings": warns}
    print(json.dumps(out))
    return 0


def launch_lines_child(argv) -> int:
    """`entry`'s launch line bytes (preflight_distill.launch_line) for the eight sweep candidates."""
    ap = argparse.ArgumentParser(prog="smoke_arms_invariance.py --internal launch_lines")
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--code-pin", required=True)
    a = ap.parse_args(argv)
    _child_paths(a.code_root)
    code_root = Path(a.code_root).resolve()
    _code_root_first(code_root)
    import src.training.sweep_select as ss
    pd = _load_preflight(code_root)
    lines = []
    for stage, lam, alpha in LAUNCH_CASES:
        rid = pd.run_id_of(stage, 42, 80000, lam, alpha, 1)
        line = pd.launch_line(ss, stage=stage, seed=42, profile="distill_80k", lam=lam, alpha=alpha, attempt=1,
                              schedule="T", code_pin=a.code_pin, ckpt_dir=f"/workspace/kd_runs/{rid}",
                              am8a_report=None)
        lines.append(pd.line_bytes(line).decode("utf-8"))
    print(json.dumps(lines))
    return 0


def payload_child(argv) -> int:
    """The arm fields of each checkpoint in a run's checkpoint folder ("<absent>" when a payload has none)."""
    ap = argparse.ArgumentParser(prog="smoke_arms_invariance.py --internal payload")
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    a = ap.parse_args(argv)
    _child_paths(a.code_root, a.ckpt_dir)
    _code_root_first(Path(a.code_root).resolve())
    import torch
    out = []
    for p in sorted(Path(a.ckpt_dir).glob("*_student_best_iter*.pt")):
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        out.append({"file": p.name, **{k: ck.get(k, "<absent>") for k in ("arm", "target_construction")}})
    print(json.dumps(out))
    return 0


def main_probe(argv) -> int:
    """The code root's main() for a dry E3 launch, with run() replaced by a recorder: the keywords run() gets
    without --arm (and with --arm, when given) and whether the registry module was imported by then."""
    ap = argparse.ArgumentParser(prog="smoke_arms_invariance.py --internal main_probe")
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--arm", default=None)
    a = ap.parse_args(argv)
    _child_paths(a.code_root)
    _code_root_first(Path(a.code_root).resolve())
    import src.training.train_distill as td
    out = {}
    for label, extra in [("default", [])] + ([("arm", ["--arm", a.arm])] if a.arm else []):
        calls, err = [], io.StringIO()
        td.run = lambda **kw: calls.append(kw) or 0      # noqa: B023 (one launch at a time)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            try:
                rc = td.main([*PROBE_ARGV, *extra])
            except SystemExit as e:                     # argparse errors
                rc = e.code
        kw = calls[0] if len(calls) == 1 else {}
        spec = kw.get("arm")
        out[label] = {"rc": rc, "run_calls": len(calls), "registry_imported": "src.training.kd_arms" in sys.modules,
                      "keys": sorted(kw),
                      "values": json.loads(json.dumps({k: v for k, v in kw.items() if k not in ("teacher", "arm")},
                                                      default=repr)),
                      "arm": None if spec is None else {"run_stage": getattr(spec, "run_stage", None),
                                                        "target_construction": getattr(spec, "target_construction",
                                                                                       None)},
                      "stderr_tail": err.getvalue()[-300:]}
    print(json.dumps(out))
    return 0


INTERNAL = {"arm_worker": arm_worker, "crm": crm_child, "launch_lines": launch_lines_child,
            "payload": payload_child, "main_probe": main_probe}


# --------------------------------------------------------------------------------------- helpers
def _subcommand(name: str, args: list, *, threads: int, cwd: Path, data_root: Path, timeout: int = SUB_TIMEOUT):
    """An internal child in a fresh process, in invariance_harness.run_worker's environment; a timeout comes
    back as returncode None."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads), "PLANTSEG_DATA_ROOT": str(data_root)})
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--internal", name, *map(str, args)]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd), env=env)
    except subprocess.TimeoutExpired:
        return types.SimpleNamespace(returncode=None, stdout="", stderr=f"timed out after {timeout} s")


def _last_json(proc):
    """The JSON an internal child prints on its last line; None when it failed or printed none."""
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else None
    except (ValueError, IndexError):
        return None


def _log(path: Path, proc) -> None:
    path.write_text(f"rc={proc.returncode}\n{proc.stdout}\n--- stderr ---\n{proc.stderr}", encoding="utf-8")


def safe_worker(**kw) -> dict:
    """invariance_harness.run_worker; a timeout comes back as a failed run (returncode None)."""
    try:
        return run_worker(timeout=WORKER_TIMEOUT, **kw)
    except subprocess.TimeoutExpired:
        return {"returncode": None, "summary": None, "log": f"{kw['out_dir']}: timed out after {WORKER_TIMEOUT} s"}


def run_arm_worker(*, work: Path, data_root: Path, code_root: Path, threads: int) -> dict:
    """run_worker's contract (exit code, run summary, log) for the arm run of stage e3."""
    out_dir = work / "arm_e3"
    proc = _subcommand("arm_worker", ["--arm", ARM, "--code-root", code_root, "--data-root", data_root,
                                      "--out", out_dir, "--stage", "e3", "--steps", STEPS,
                                      "--val-interval", VAL_INTERVAL, "--threads", threads],
                       threads=threads, cwd=work, data_root=data_root, timeout=WORKER_TIMEOUT)
    log = work / f"{out_dir.name}.worker.log"
    _log(log, proc)
    summary_path = out_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {"returncode": proc.returncode, "summary": summary, "log": str(log)}


def ran(r: dict) -> bool:
    sm = r.get("summary") or {}
    return (r.get("returncode") == 0 and str(sm.get("trainer_result", "")).startswith("RESULT: PASS")
            and sm.get("n_step_records") == STEPS)


def meta_row(r: dict):
    rows = (r.get("summary") or {}).get("run_meta") or []
    return rows[0] if len(rows) == 1 and isinstance(rows[0], dict) else None


def without_identity(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in IDENTITY_KEYS}


def canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), allow_nan=True).encode("utf-8")


def meta_equal(a, b) -> bool:
    return a is not None and b is not None and canonical(without_identity(a)) == canonical(without_identity(b))


def differing_keys(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if canonical(a.get(k, "<absent>")) != canonical(b.get(k, "<absent>")))


def payloads(r: dict) -> list:
    return (r.get("summary") or {}).get("checkpoints") or []


def best_json(r: dict):
    """The run's best.json, with best_ckpt (an absolute path) reduced to its file name."""
    best = (r.get("summary") or {}).get("best_json")
    if not isinstance(best, dict):
        return None
    return {**best, "best_ckpt": Path(str(best.get("best_ckpt"))).name}


def _no_test(name: str) -> bool:
    return "test" not in name.lower()


def archive_old(old: str) -> bytes:
    """`git archive` of OLD with the SL-1 exclude; nothing is written. A failed archive, or a member name that
    still contains "test" (each name compared as a string), is a refusal."""
    proc = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", old, "--", *EXPORT_PATHS,
                           TEST_EXCLUDE, EXPORT_EXCLUDE], capture_output=True)
    if proc.returncode != 0:
        raise RefusedExport(f"git archive of {old} failed: {proc.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
        named = sum(1 for n in tf.getnames() if not _no_test(n))
    if named:
        raise RefusedExport(f"git archive of {old} kept {named} member(s) whose name contains 'test'")
    return proc.stdout


class RefusedExport(RuntimeError):
    """OLD cannot be exported under the SL-1 exclude (exit 2, nothing written)."""


def blob_map_dir(root: Path) -> dict:
    """{path: git blob id} for every file under root/{src,scripts,configs}. A name containing "test" is
    skipped as a string before any call on it; __pycache__ is skipped."""
    out = {}
    for top in BLOB_TOPS:
        stack = [root / top]
        while stack:
            d = stack.pop()
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
            for e in entries:
                if not _no_test(e.name) or e.name == "__pycache__":
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    p = Path(e.path)
                    out[p.relative_to(root).as_posix()] = git_blob_id(p.read_bytes())
    return out


def blob_map_checkout(root: Path) -> dict:
    """{path: git blob id} for the checkout's tracked and untracked (not ignored) files under src/, scripts/
    and configs/, from `git ls-files` with the SL-1 exclude: no path containing "test" is listed."""
    proc = subprocess.run(["git", "-C", str(root), "--no-optional-locks", "ls-files", "-z", "--cached", "--others",
                           "--exclude-standard", "--", *BLOB_TOPS, TEST_EXCLUDE], capture_output=True, check=True)
    out = {}
    for rel in sorted({p for p in proc.stdout.decode("utf-8").split("\0") if p}):
        if not _no_test(rel):
            continue
        p = root / rel
        if p.is_file():
            out[rel] = git_blob_id(p.read_bytes())
    return out


def module_name(rel: str):
    """The dotted module a file defines (a package's __init__.py, a source or an extension module), else None."""
    p = PurePosixPath(rel)
    if p.name == "__init__.py":
        return ".".join(p.parent.parts)
    for suffix in (".py", *importlib.machinery.EXTENSION_SUFFIXES):
        if p.name.endswith(suffix):
            return ".".join((*p.parent.parts, p.name[:-len(suffix)]))
    return None


# ----------------------------------------------------------------------------------------- sections
def section_child(work: Path, old: str, old_root: Path, new_root: Path, threads: int) -> None:
    cmd = [sys.executable, "-B", str(REPO / "scripts" / "smoke_invariance_distill.py")]
    if new_root == REPO:
        cmd += ["--cross-commit", old, "WORKTREE"]
    else:
        cmd += ["--old-root", str(old_root), "--new-root", str(new_root)]
    cmd += ["--work-dir", str(work / "child"), "--threads", str(threads)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    t0 = time.time()
    log = work / "child.log"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), env=env, timeout=CHILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        log.write_text(" ".join(cmd) + f"\ntimed out after {CHILD_TIMEOUT} s\n", encoding="utf-8")
        check("inv_cross_commit_child_pass", False, f"timed out after {CHILD_TIMEOUT} s; log {log}")
        return
    log.write_text(" ".join(cmd) + "\n" + proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    result = next((ln.strip() for ln in reversed(proc.stdout.splitlines()) if ln.startswith("RESULT")), None)
    check("inv_cross_commit_child_pass", proc.returncode == 0 and str(result).startswith("RESULT: PASS"),
          f"rc={proc.returncode} {result} ({time.time() - t0:.0f} s; log {log})")


def section_workers(work: Path, data_root: Path, old_root: Path, new_root: Path, threads: int) -> None:
    runs = {}
    for label, root in (("old", old_root), ("new", new_root)):
        for s in STAGES:
            runs[f"{label}_{s}"] = safe_worker(code_root=root, data_root=data_root, out_dir=work / f"{label}_{s}",
                                               stage=s, steps=STEPS, val_interval=VAL_INTERVAL, threads=threads)
    runs["arm_e3"] = run_arm_worker(work=work, data_root=data_root, code_root=new_root, threads=threads)
    (work / "workers_summary.json").write_text(json.dumps({k: {"returncode": r["returncode"], "log": r["log"],
                                                               "summary": r["summary"]} for k, r in runs.items()},
                                                          indent=1, default=str), encoding="utf-8")
    failed = {k: r["log"] for k, r in runs.items() if not ran(r)}
    for s in STAGES:
        o, n = runs[f"old_{s}"], runs[f"new_{s}"]
        ok_runs = ran(o) and ran(n)
        a, b = meta_row(o), meta_row(n)
        diff = None
        if a is not None and b is not None and not meta_equal(a, b):
            diff = differing_keys(without_identity(a), without_identity(b)) or "key order"
        check(f"inv_run_meta_equal_{s}", ok_runs and meta_equal(a, b),
              f"runs ok={ok_runs} rows={a is not None}/{b is not None} differing={diff} failed={failed}")
        pa, pb = payloads(o), payloads(n)
        check(f"inv_payload_keys_equal_{s}", ok_runs and bool(pa) and pa == pb,
              f"old={[c.get('keys') for c in pa]} new={[c.get('keys') for c in pb]}")
    best = {s: (best_json(runs[f"old_{s}"]), best_json(runs[f"new_{s}"])) for s in STAGES}
    best_bad = [s for s, (bo, bn) in best.items() if bo is None or bn is None or canonical(bo) != canonical(bn)]
    check("inv_best_json_equal", all(ran(r) for k, r in runs.items() if k != "arm_e3") and not best_bad,
          f"differing or missing: {best_bad}; " + "; ".join(f"{s}: old={best[s][0]} new={best[s][1]}"
                                                            for s in best_bad))

    # check-run-meta's rule table on the default E3 rows (the code root's own table)
    o_row, n_row = meta_row(runs["old_e3"]), meta_row(runs["new_e3"])
    crm_ok, crm_detail = False, "no rows"
    if o_row is not None and n_row is not None:
        aligned = {}
        for label, row in (("old", o_row), ("new", n_row)):
            r = dict(row)
            r.update(wall_clock=1.0, git_head="0" * 40, git_head_source="git_checkout")
            aligned[label] = r
        rows_file = work / "crm_rows.json"
        rows_file.write_text(json.dumps(aligned), encoding="utf-8")
        ck_dir = work / "crm_run" / "e3_s42_lambda1_alpha50_80k_a1"
        proc = _subcommand("crm", ["--code-root", new_root, "--rows", rows_file, "--ckpt-dir", ck_dir],
                           threads=threads, cwd=work, data_root=data_root)
        _log(work / "crm.log", proc)
        got = _last_json(proc)
        if got is None:
            crm_detail = f"rc={proc.returncode}: no verdicts; log {work / 'crm.log'}"
        else:
            want_keys = got["run_meta_keys_e3"]
            new_p = got["new"]["problems"]
            flagged = [p.split(" ")[0] for p in new_p]
            outside = [p for p in new_p if p.split(" ")[0] not in HARNESS_FACTS]
            crm_ok = sorted(n_row) == sorted(want_keys) and not outside and got["new"] == got["old"]
            crm_detail = (f"keys exact={sorted(n_row) == sorted(want_keys)} flagged outside HARNESS_FACTS={outside} "
                          f"same verdicts as OLD={got['new'] == got['old']}; flagged ({len(flagged)}): {flagged}")
    check("inv_check_run_meta_default_pass", crm_ok, crm_detail)

    # the negative control: the comparisons tell an arm run apart, by its arm fields only; main() hands run()
    # the arm only for --arm
    arm = runs["arm_e3"]
    a_row = meta_row(arm)
    parts = {"arm run": ran(arm)}
    details = [f"arm run ok={ran(arm)} log {arm.get('log')}"]
    if ran(arm) and a_row is not None and o_row is not None:
        o_cmp, a_cmp = without_identity(o_row), without_identity(a_row)
        differing = differing_keys(o_cmp, a_cmp)
        restored = dict(a_cmp)
        restored.pop("target_construction", None)
        restored.update({k: o_cmp[k] for k in ("arm", "descriptive", "parent_of_e4_e7") if k in o_cmp})
        parts["run_meta"] = (not meta_equal(o_row, a_row) and differing == sorted(ARM_FIELDS)
                             and all(k in a_row and canonical(a_row[k]) == canonical(v) for k, v in ARM_FIELDS.items())
                             and list(a_row)[-1] == "target_construction"
                             and canonical(restored) == canonical(o_cmp))
        po, pa_ = payloads(runs["old_e3"]), payloads(arm)
        parts["payload keys"] = (bool(po) and len(po) == len(pa_) and po != pa_ and all(
            set(c2["keys"]) == set(c1["keys"]) | {"arm", "target_construction"}
            and {k: v for k, v in c1.items() if k != "keys"} == {k: v for k, v in c2.items() if k != "keys"}
            for c1, c2 in zip(po, pa_)))
        ck_dirs = sorted(p for p in (work / "arm_e3").glob("ckpt_e3*") if p.is_dir())
        values = None
        if len(ck_dirs) == 1:
            proc = _subcommand("payload", ["--code-root", new_root, "--ckpt-dir", ck_dirs[0]], threads=threads,
                               cwd=work, data_root=data_root)
            _log(work / "payload.log", proc)
            values = _last_json(proc)
        parts["payload values"] = (isinstance(values, list) and len(values) == len(pa_) > 0 and all(
            v["arm"] == ARM_FIELDS["arm"] and v["target_construction"] == ARM_FIELDS["target_construction"]
            for v in values))
        steps = compare_step_dirs(work / "new_e3", work / "arm_e3")
        parts["steps as NEW E3"] = steps["identical"]
        details.append(f"differing={differing} last={list(a_row)[-1]} payload values={values} "
                       f"steps {steps['n_old']}/{steps['n_new']}, first mismatch {steps['mismatches'][:1]}")
    probes = {}
    for label, root, extra in (("old", old_root, []), ("new", new_root, ["--arm", ARM])):
        proc = _subcommand("main_probe", ["--code-root", root, *extra], threads=threads, cwd=work, data_root=data_root)
        _log(work / f"main_probe_{label}.log", proc)
        probes[label] = _last_json(proc)
    od, nd = (probes["old"] or {}).get("default"), (probes["new"] or {}).get("default")
    na = (probes["new"] or {}).get("arm")
    parts["main() without --arm"] = (
        od is not None and nd is not None and od["rc"] == nd["rc"] == 0 and od["run_calls"] == nd["run_calls"] == 1
        and "arm" not in nd["keys"] and od["keys"] == nd["keys"] and canonical(od["values"]) == canonical(nd["values"])
        and nd["registry_imported"] is False)
    parts["main() with --arm"] = (
        nd is not None and na is not None and na["rc"] == 0 and na["run_calls"] == 1
        and na["keys"] == sorted([*nd["keys"], "arm"]) and canonical(na["values"]) == canonical(nd["values"])
        and na["arm"] == {"run_stage": ARM_FIELDS["arm"], "target_construction": ARM_FIELDS["target_construction"]})
    details.append(f"main probes: old default={None if od is None else (od['rc'], od['keys'])} new default="
                   f"{None if nd is None else (nd['rc'], nd['registry_imported'])} new arm="
                   f"{None if na is None else (na['rc'], na['arm'], na['stderr_tail'][-120:])}")
    check("inv_negative_control_arm_run_differs", all(parts.values()) and len(parts) == 7,
          f"parts={parts}; " + "; ".join(details))


def section_static(work: Path, data_root: Path, old: str, old_root: Path, new_root: Path, threads: int) -> None:
    old_map = blob_map_dir(old_root)
    new_map = blob_map_checkout(REPO) if new_root == REPO else blob_map_dir(new_root)
    changed = sorted(p for p in old_map if p in new_map and new_map[p] != old_map[p])
    vanished = sorted(p for p in old_map if p not in new_map)
    added = sorted(p for p in new_map if p not in old_map)
    old_modules = {m for m in map(module_name, old_map) if m}
    shadowing = [p for p in added if module_name(p) in old_modules]
    outside = [p for p in changed if p not in SEAM_PATHS]
    missing = [p for p in C1_PATHS if p not in new_map]
    check("inv_blob_ids_unchanged", bool(old_map) and not outside and not vanished and not missing and not shadowing,
          f"{len(old_map)} OLD files; changed outside the seam: {outside[:12]}; vanished: {vanished[:12]}; C1 paths "
          f"missing: {missing}; added modules shadowing OLD's: {shadowing}; changed: {changed}; added "
          f"({len(added)}): {added[:12]}")
    lines = {}
    for label, root in (("old", old_root), ("new", new_root)):
        proc = _subcommand("launch_lines", ["--code-root", root, "--code-pin", old], threads=threads, cwd=work,
                           data_root=data_root)
        _log(work / f"launch_lines_{label}.log", proc)
        lines[label] = _last_json(proc)
    ok = isinstance(lines["old"], list) and len(lines["old"]) == len(LAUNCH_CASES) and lines["old"] == lines["new"]
    check("inv_launch_line_bytes_equal", ok,
          f"old={None if lines['old'] is None else len(lines['old'])} lines, new="
          f"{None if lines['new'] is None else len(lines['new'])} lines, equal={lines['old'] == lines['new']}")


# --------------------------------------------------------------------------------------------- main
def refused(text: str) -> int:
    print(f"REFUSED: {text}")
    print("RESULT: REFUSED -- nothing ran, nothing written")
    return 2


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--internal"] and len(argv) > 1 and argv[1] in INTERNAL:
        return INTERNAL[argv[1]](argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--old", default=OLD_DEFAULT)
    ap.add_argument("--new-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--section", action="append", choices=SECTIONS, default=None)
    a = ap.parse_args(argv)
    # SL-1 (plan 1.0): the DIAG guard, imported; this process never imports the trainer, so src may load here
    from src.eval.teacher_diag import Refused, refuse_test_path, require_outside_repo
    try:
        for value, what in ((a.new_root, "--new-root"), (a.work_dir, "--work-dir")):
            if value is not None:
                refuse_test_path(value, what)
        if a.work_dir is not None:
            work = require_outside_repo(a.work_dir, "--work-dir")
            if work.exists():
                return refused(f"--work-dir {work} exists; an output folder is never reused")
        else:
            for var in ("TMPDIR", "TEMP", "TMP"):      # before tempfile probes a candidate by writing to it
                if os.environ.get(var):
                    refuse_test_path(os.environ[var], f"${var} (set it elsewhere)")
            base = tempfile.gettempdir()
            n = 0
            while True:
                n += 1
                work = Path(base) / f"arms_smoke_{os.getpid()}_{n}"
                refuse_test_path(work, "the default work dir (set TMPDIR elsewhere)")
                work = require_outside_repo(work, "the default work dir")
                if not work.exists():
                    break
        new_root = REPO if a.new_root is None else Path(a.new_root).resolve()
        lacking = [rel for rel in ("src/training/train_distill.py", *BLOB_TOPS)
                   if not (new_root / rel).is_file() and not (new_root / rel).is_dir()]
        if lacking:
            return refused(f"--new-root {new_root} lacks {lacking}")
    except Refused as e:
        return refused(str(e))
    sections = a.section or list(SECTIONS)
    try:
        old = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", f"{a.old}^{{commit}}"],
                             capture_output=True, text=True, check=True).stdout.strip()
        archive = archive_old(old)
    except subprocess.CalledProcessError:
        return refused(f"--old {a.old!r} does not resolve to a commit in {REPO}")
    except RefusedExport as e:
        return refused(str(e))
    work.mkdir(parents=True)
    t0 = time.time()
    print("=" * 78)
    print(f"ARMS INVARIANCE SMOKE (L-TEACHER-ARMS C1) -- CPU, synthetic 16 + 4 set, stub teacher, {STEPS} steps")
    print(f"OLD {old} | NEW {'this checkout' if new_root == REPO else new_root} | sections {sections}")
    print(f"work dir: {work}")
    print("=" * 78, flush=True)
    old_root = work / "export_old"
    export = export_commit(REPO, old, old_root, paths=(*EXPORT_PATHS, TEST_EXCLUDE))
    export["checked_archive_bytes"] = len(archive)
    data_root = work / "data"
    make_synthetic_dataset(data_root)
    calls = {"child": lambda: section_child(work, old, old_root, new_root, a.threads),
             "workers": lambda: section_workers(work, data_root, old_root, new_root, a.threads),
             "static": lambda: section_static(work, data_root, old, old_root, new_root, a.threads)}
    for section in SECTIONS:
        if section not in sections:
            continue
        try:
            calls[section]()
        except Exception as e:                         # a run that raises fails its section's checks
            done = {n for n, _, _ in results}
            for name in SECTION_CHECKS[section]:
                if name not in done:
                    check(name, False, f"section {section} raised {type(e).__name__}: {e}")
    seconds = round(time.time() - t0, 1)
    (work / "arms_invariance_summary.json").write_text(json.dumps({
        "old": old, "new_root": str(new_root), "sections": sections, "seconds": seconds, "export": export,
        "export_exclude": [TEST_EXCLUDE, EXPORT_EXCLUDE],
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results]}, indent=1), encoding="utf-8")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:40}: {'PASS' if ok else 'FAIL'}  {detail[:400] if not ok else ''}".rstrip())
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nsummary: {work / 'arms_invariance_summary.json'} ({seconds} s)")
    print(f"RESULT: {'PASS' if results and passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if results and passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
