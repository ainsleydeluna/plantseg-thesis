#!/usr/bin/env python3
"""Smoke for the TRAIN/VAL-only E1 gate (scripts/preflight_e1_trainval.py) and the train_e1 real-run
M11 gate (DL-21; B66-prep S2). CPU only; no real data; synthetic roots under tempfile.mkdtemp().

Sections (default mode):
  A  constants, identity of the reused preflight_e1 stages, import hygiene (Q2, X5, X8, X9)
  B  data_isolation: every TEST surface refused by exact path, never listed; pairing; order
  C  arguments: refusals before any data-root access
  D  verdict logic and the launch block
  E  stage units (git and children injected)
  F  check-run-meta
  G  train_e1.main's real-run gate (S2b)
--full-rehearsal (pinned image only; the checkout must be a scratch git repository) adds
  R  the gate end to end in --rehearsal mode on a decodable 5367/846 synthetic root

Before any repository import, PLANTSEG_DATA_ROOT is pointed at a non-existent temp path, so nothing can
reach a real data root. Writes only under tempfile.mkdtemp() (removed in `finally`).
Run:  python -B scripts/smoke_preflight_e1_trainval.py [--full-rehearsal]
Exit 0 only when every check PASSes; a SKIP (only B05 on hosts without symlink rights, E17 on hosts with
CUDA) is never counted as a PASS.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse      # noqa: E402
import ast           # noqa: E402
import builtins      # noqa: E402
import contextlib    # noqa: E402
import hashlib       # noqa: E402
import io            # noqa: E402
import json          # noqa: E402
import os            # noqa: E402
import random        # noqa: E402
import shlex         # noqa: E402
import shutil        # noqa: E402
import stat          # noqa: E402
import subprocess    # noqa: E402
import tempfile      # noqa: E402
import textwrap      # noqa: E402
import traceback     # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="smoke_pf_e1tv_")).resolve()
os.environ["PLANTSEG_DATA_ROOT"] = str(TMP / "no_such_data_root")        # before any repo import
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import numpy as np   # noqa: E402
import torch         # noqa: E402

import preflight_e1_trainval as P                     # noqa: E402
import src.data.isolation as ISO                      # noqa: E402
import src.training.train_e1 as e1                    # noqa: E402
from configs.data import SPLIT_SIZES                  # noqa: E402
from src.eval.artifacts import GOVERNED_PREFIXES      # noqa: E402

CHECKS: list[tuple[str, str, str]] = []
DL21 = "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"
HEAD = "a" * 40
SEED42_RUN_META = {
    "event": "run_meta", "wall_clock": 1789131059.3474903, "mode": "real", "seed": 42,
    "git_head": "f77d05d7b35187bf0da7e7b94a629549fe2e1c05", "git_head_source": "git_checkout",
    "image_digest": DL21, "torch": "2.1.0+cu121", "numpy": "1.26.4", "device": "cuda",
    "cuda_available": True, "gpu_name": "NVIDIA A40", "num_workers": 12, "batch_size": 16,
    "max_iters": 80000, "val_interval": 4000, "max_val_batches": None, "ckpt_interval": 2000,
    "resumed_from": None, "num_classes": 116, "ignore_index": 255, "learning_rate": 0.01,
    "momentum": 0.9, "weight_decay": 0.0001, "lr_power": 0.9, "poly_horizon": 80000,
    "grad_clip_norm": None, "used_pretrained": True, "params": 2933688}
P4_REAL42 = {"batch_size": 16, "ckpt_dir_arg": None, "ckpt_interval": 2000, "device": "cuda",
             "grad_clip_norm": None, "jsonl_name": "e1_telemetry.jsonl", "keep_ckpts": 3,
             "log_every": 1, "max_iters": 80000, "max_val_batches": None, "mode": "real",
             "num_workers": 12, "pretrained": "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2",
             "resume": None, "seed": 42, "val_interval": 4000}
P4_DRY = {"batch_size": 2, "ckpt_dir_arg": None, "ckpt_interval": 2000, "device": "cpu",
          "grad_clip_norm": None, "jsonl_name": "e1_telemetry.jsonl", "keep_ckpts": 3, "log_every": 1,
          "max_iters": 4, "max_val_batches": 2, "mode": "dry", "num_workers": 0, "pretrained": False,
          "resume": None, "seed": 42, "val_interval": 2}
STAGE_ORDER = ("arguments", "data_isolation", "repo_state", "module_provenance", "class_weights",
               "smoke_loss", "image", "cuda", "imagenet_backbone", "smoke_loader_seed",
               "smoke_dataloader", "seed_sequence_R5", "dry_run", "repo_unchanged")


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, "PASS" if ok else "FAIL", " | ".join(str(detail).split("\n"))[:240]))


def skip(name: str, reason: str) -> None:
    CHECKS.append((name, "SKIP", reason))


def code_of(fn, *a, **kw):
    """The GateError / TrainValIsolationError code raised by fn, or None."""
    try:
        fn(*a, **kw)
        return None
    except (P.GateError, ISO.TrainValIsolationError) as e:
        return e.code


@contextlib.contextmanager
def env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def make_root(root: Path, counts=None, skip_dir=None) -> Path:
    counts = counts or {"train": 3, "val": 2}
    for split in ("train", "val"):
        for kind, suffix in (("images", ".jpg"), ("annotations", ".png")):
            d = root / kind / split
            if skip_dir == f"{kind}/{split}":
                continue
            d.mkdir(parents=True, exist_ok=True)
            for i in range(counts[split]):
                (d / f"{split}_{i:05d}{suffix}").touch()
    return root


def plant(root: Path, surface: str) -> Path:
    target = root.joinpath(*surface.split("/"))
    if surface.endswith(".json"):
        target.touch()
    else:
        target.mkdir(parents=True)
        (target / "trap_00000.jpg").touch()
    return target


def fresh(name: str) -> Path:
    p = TMP / name
    if p.exists():
        shutil.rmtree(p)
    return p


class Spies:
    """Record every listing/open while active."""

    def __init__(self):
        self.listed, self.opened = [], []
        self._saved = None

    def __enter__(self):
        self._saved = (os.scandir, os.listdir, Path.iterdir, builtins.open)
        rs, rl, ri, ro = self._saved

        def sc(p="."):
            self.listed.append(str(p))
            return rs(p)

        def ld(p="."):
            self.listed.append(str(p))
            return rl(p)

        def it(pth):
            self.listed.append(str(pth))
            return ri(pth)

        def op(file, *a, **kw):
            self.opened.append(str(file))
            return ro(file, *a, **kw)
        os.scandir, os.listdir, Path.iterdir, builtins.open = sc, ld, it, op
        return self

    def __exit__(self, *exc):
        os.scandir, os.listdir, Path.iterdir, builtins.open = self._saved


def run_main(argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = P.main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


# ------------------------------------------------------------------------------------------ A
def section_a() -> None:
    check("A01 TRAINVAL_COUNTS == {train:5367, val:846} == configs SPLIT_SIZES; no test; own object",
          P.TRAINVAL_COUNTS == {"train": 5367, "val": 846}
          == {k: SPLIT_SIZES[k] for k in ("train", "val")} and "test" not in P.TRAINVAL_COUNTS
          and P.TRAINVAL_COUNTS is not ISO.DEFAULT_EXPECTED_COUNTS)
    check("A02 E1_IMAGE_DIGEST == the DL-21 digest", P.E1_IMAGE_DIGEST == DL21)
    check("A03 ImageNet backbone constants == the seed-42 capture",
          P.IMAGENET_SHA256 == "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"
          and P.IMAGENET_BYTES == 22132113 and P.IMAGENET_FILE == "mobilenet_v3_large-5c1a4163.pth"
          and P.IMAGENET_SHA256[:8] in P.IMAGENET_FILE)
    check("A04 MAX_SM (9,0); OFFICIAL_SEEDS (42,43,44); E1_SAFETY_FLOOR 885523a",
          P.MAX_SM == (9, 0) and P.OFFICIAL_SEEDS == (42, 43, 44) and P.E1_SAFETY_FLOOR == "885523a")
    gov = ("src", "configs", "scripts", "requirements*", "docs/EVALUATION_CONTRACT.md",
           "docs/IMPLEMENTATION_CONTRACT.md")
    mapped = [g.rstrip("*") + ("/" if g in ("src", "configs", "scripts") else "") for g in gov]
    all_specs = set(P.GOVERNED_PATHSPECS) | set(P.STATUS_PATHSPECS) | set(P.UNCHANGED_PATHSPECS)
    check("A05 pathspecs: governed 1:1 with GOVERNED_PREFIXES; X5 adds the class-weight file; no "
          "repo-wide or docs/ spec",
          P.GOVERNED_PATHSPECS == gov and tuple(mapped) == GOVERNED_PREFIXES
          and P.STATUS_PATHSPECS == gov + ("reports/e1_class_weights.json",)
          and set(P.STATUS_PATHSPECS) <= set(P.UNCHANGED_PATHSPECS)
          and not any(s in (".", ":/", "docs", "docs/") or s.startswith("docs/reference") for s in all_specs))
    check("A06 SPARSE_PATTERNS == ('/*', '!/docs/reference/')",
          P.SPARSE_PATTERNS == ("/*", "!/docs/reference/"))
    pf1_file = Path(P.PF1.__file__).resolve()
    reused_ok = all(
        P.REUSED_PF1[k] is getattr(P.PF1, n) and P.REUSED_PF1[k].__globals__ is vars(P.PF1)
        for k, n in (("class_weights", "stage_class_weights"), ("smoke_loss", "stage_loss"),
                     ("smoke_dataloader", "stage_dataloader"), ("seed_sequence_R5", "stage_stochasticity")))
    calls = []
    recorders = {k: (lambda dev, k=k: calls.append((k, dev)) or (True, k, 0.0)) for k in P.REUSED_PF1}
    stage_fn = dict(P.STAGES)
    with patched(P, "REUSED_PF1", recorders), patched(P, "CLASS_WEIGHTS_SHA256",
                                                      P.sha256_file(REPO / "reports" / "e1_class_weights.json")):
        for stage, key in (("class_weights", "class_weights"), ("smoke_loss", "smoke_loss"),
                           ("smoke_dataloader", "smoke_dataloader"), ("seed_sequence_R5", "seed_sequence_R5")):
            stage_fn[stage]({})
    check("A07 the four reused stages ARE preflight_e1's functions, each STAGES entry calls its own one "
          "(with dev=False); R5 baseline and constants intact",
          reused_ok and pf1_file == REPO / "scripts" / "preflight_e1.py"
          and calls == [("class_weights", False), ("smoke_loss", False), ("smoke_dataloader", False),
                        ("seed_sequence_R5", False)]
          and P.PF1.B30_SEED_BASELINE == [[1608637542, 1273642419], [1935803228, 787846414]]
          and P.PF1.HARD_CHECKS_TOTAL == 6 and P.PF1.REPO == P.REPO, f"{pf1_file} {calls}")
    names = tuple(n for n, _ in P.STAGES)
    check("A08 14 stages in the fixed order; data_isolation is second (first data-root access)",
          names == STAGE_ORDER == P.STAGE_NAMES and names.index("data_isolation") == 1)
    tree = ast.parse((REPO / "scripts" / "preflight_e1_trainval.py").read_text(encoding="utf-8"))
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == "PF1"}
    spec_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "")
                  == "spec_from_file_location"]
    check("A09 AST: verify_env.py never named; preflight_e1.py named once (spec_from_file_location); "
          "PF1.main / stage_verify_env / stage_dryrun / STAGES never accessed",
          not any(c.endswith("verify_env.py") for c in consts)
          and sum(c == "preflight_e1.py" for c in consts) == 1 and len(spec_calls) == 1
          and not attrs & {"main", "stage_verify_env", "stage_dryrun", "STAGES"}, str(sorted(attrs)))
    iso_sha = hashlib.sha256((REPO / "src" / "data" / "isolation.py").read_bytes()).hexdigest()
    check("A10 sha256(src/data/isolation.py) == the frozen value (called, never edited)",
          iso_sha == "74a063313cc04c1e327dce22b0faa4ae3a9334a01e6cf7c58eb1bee13c2e5014", iso_sha[:16])
    body = tree.body
    idx_dwb = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "sys.dont_write_bytecode")
    idx_env = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                   and "PYTHONDONTWRITEBYTECODE" in ast.unparse(n.targets[0]))
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"importlib"}
    first_nonstd = next((i for i, n in enumerate(body) if isinstance(n, (ast.Import, ast.ImportFrom))
                         and not all((a.name.split(".")[0] if isinstance(n, ast.Import)
                                      else (n.module or "").split(".")[0]) in stdlib for a in n.names)),
                        len(body))
    idx_exec = next(i for i, n in enumerate(body) if "exec_module" in ast.unparse(n))
    check("A11 X8 ordering (AST): sys.dont_write_bytecode = True and PYTHONDONTWRITEBYTECODE are set "
          "before any non-stdlib import and before preflight_e1 is executed",
          idx_dwb < first_nonstd and idx_dwb < idx_exec and idx_env < idx_exec,
          f"dwb@{idx_dwb} env@{idx_env} exec@{idx_exec} first_nonstdlib@{first_nonstd}")
    child = textwrap.dedent(f'''
        import json, os, subprocess, sys
        spawned = []
        real_init = subprocess.Popen.__init__
        def spy(self, *a, **k):
            spawned.append(str(a[0] if a else k.get("args")))
            return real_init(self, *a, **k)
        subprocess.Popen.__init__ = spy
        dwb_before = sys.dont_write_bytecode
        before = list(sys.path)
        import importlib.util
        spec = importlib.util.spec_from_file_location("gate_under_test", {str(REPO / "scripts" / "preflight_e1_trainval.py")!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        res = {{"spawned": list(spawned), "verify_env": "verify_env" in sys.modules,
               "preflight_e1_registered": "preflight_e1" in sys.modules, "path_same": list(sys.path) == before,
               "dwb_before": dwb_before, "dwb_after": sys.dont_write_bytecode,
               "env_after": os.environ.get("PYTHONDONTWRITEBYTECODE")}}
        subprocess.run([sys.executable, "-c", "pass"])      # positive control: the spy does record
        res["spy_records"] = len(spawned) == len(res["spawned"]) + 1
        print("RESULT_JSON:" + json.dumps(res))
    ''')
    cenv = {k: v for k, v in os.environ.items() if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPATH")}
    # Without -B the loader may cache the gate's own bytecode before the gate turns caching off:
    # route any such cache to temp so the checkout is never written.
    cenv["PYTHONPYCACHEPREFIX"] = str(TMP / "pycache_a12")
    r = subprocess.run([sys.executable, "-c", child], cwd=str(TMP), capture_output=True, text=True,
                       env=cenv, timeout=300)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT_JSON:")]
    res = json.loads(lines[-1][12:]) if lines else {}
    check("A12 Q2/X8: importing the gate (no -B, no env flag, no sys.path help) spawns no process, loads no "
          "verify_env, does not register or run preflight_e1, leaves sys.path unchanged and turns bytecode "
          "writing off itself",
          bool(res) and res["spawned"] == [] and not res["verify_env"] and res["spy_records"]
          and not res["preflight_e1_registered"] and res["path_same"] and res["dwb_before"] is False
          and res["dwb_after"] is True and res["env_after"] == "1",
          json.dumps(res) if res else (r.stdout + r.stderr)[-300:])
    cw = REPO / "reports" / "e1_class_weights.json"
    import inspect
    check("A13 X5: preflight_e1's class-weight check pins no sha256, so the gate pins the 5325e32 blob",
          "sha256" not in inspect.getsource(P.PF1.check_class_weights)
          and P.CLASS_WEIGHTS_SHA256 == "fd78ba13d0bc06c0e806a5ab0b211f0044b448a494915b6135753c9701892dcf"
          == hashlib.sha256(cw.read_bytes()).hexdigest())
    check("A14 X9: run_meta has 29 keys, log_every is not one; exemptions exact",
          len(P.RUN_META_KEYS) == 29 == len(SEED42_RUN_META) and set(P.RUN_META_KEYS) == set(SEED42_RUN_META)
          and "log_every" not in P.RUN_META_KEYS
          and P.RUN_META_EXEMPT == ("seed", "git_head", "wall_clock", "gpu_name", "max_iters", "poly_horizon")
          and all(P.RUN_META_EXPECT[k] == SEED42_RUN_META[k] for k in P.RUN_META_EXPECT)
          and set(P.RUN_META_EXPECT) | set(P.RUN_META_EXEMPT) == set(SEED42_RUN_META))
    check("A15 Q4: no eval profile; launch-block constants 12 workers / log-every 50",
          set(P.PROFILES) == {"e1_80k", "e1_160k"} and "eval" not in P.PROFILES
          and P.PROFILES["e1_80k"] == {"extra_args": (), "max_iters": 80000, "poly_horizon": 80000}
          and P.PROFILES["e1_160k"] == {"extra_args": ("--iterations", "160000"),
                                        "dry_args": ("--iterations", "160000"),
                                        "max_iters": 160000, "poly_horizon": 160000}
          and P.E1_NUM_WORKERS == 12 and P.LOG_EVERY == 50)


# ------------------------------------------------------------------------------------------ B
def section_b() -> None:
    saved = P.TRAINVAL_COUNTS
    P.TRAINVAL_COUNTS = {"train": 3, "val": 2}
    all_listed, all_opened, details = [], [], []

    def iso(root, name, *, spy=True):
        ctx = {"root": root}
        sp = Spies()
        try:
            if spy:
                with sp:
                    res = P.stage_data_isolation(ctx)
            else:
                res = P.stage_data_isolation(ctx)
            code = None
        except P.GateError as e:
            res, code = None, e.code
            details.append(str(e))
        all_listed.extend(sp.listed)
        all_opened.extend(sp.opened)
        return code, res, ctx

    try:
        root = make_root(fresh("b_ok"))
        code, res, ctx = iso(root, "ok")
        check("B01 a clean TRAIN/VAL root PASSes with counts, absent surfaces and pairing",
              code is None and res[0] == "PASS"
              and ctx["isolation"]["counts"] == {"train": {"images": 3, "masks": 3}, "val": {"images": 2, "masks": 2}}
              and set(ctx["isolation"]["test_surfaces"].values()) == {"absent"}
              and ctx["isolation"]["pairing"] == {"train": 3, "val": 2})
        test_cases = []
        for tag, surface in (("B02", "images/test"), ("B03", "annotations/test"), ("B04", "annotation_test.json")):
            r = make_root(fresh(f"b_{tag}"))
            plant(r, surface)
            code, _, _ = iso(r, tag)
            test_cases.append(tag)
            check(f"{tag} {surface} present -> test_split_present", code == "test_split_present", str(code))
        r5 = make_root(fresh("b_B05"))
        try:
            os.symlink(str(r5 / "nowhere"), str(r5 / "images" / "test"), target_is_directory=True)
            made = True
        except (OSError, NotImplementedError):
            made = False
        if made:
            code, _, _ = iso(r5, "B05")
            check("B05 a dangling images/test symlink -> test_split_present", code == "test_split_present", str(code))
        else:
            skip("B05 a dangling images/test symlink -> test_split_present",
                 "os.symlink not permitted on this host (plan-named SKIP)")
        code, _, _ = iso(make_root(fresh("b_B06"), skip_dir="images/val"), "B06")
        check("B06 missing images/val -> split_dir_missing", code == "split_dir_missing", str(code))
        code, _, _ = iso(make_root(fresh("b_B07"), skip_dir="annotations/val"), "B07")
        check("B07 missing annotations/val -> split_dir_missing", code == "split_dir_missing", str(code))
        code, _, _ = iso(make_root(fresh("b_B08"), counts={"train": 3, "val": 1}), "B08")
        check("B08 val 1/1 -> split_count_mismatch", code == "split_count_mismatch", str(code))
        r9 = make_root(fresh("b_B09"))
        (r9 / "annotations" / "train" / "train_00002.png").unlink()
        code, _, _ = iso(r9, "B09")
        check("B09 train 3 images / 2 masks -> split_count_mismatch", code == "split_count_mismatch", str(code))
        r10 = make_root(fresh("b_B10"))
        (r10 / "annotations" / "val" / "orphan_99999.png").touch()
        code, _, _ = iso(r10, "B10")
        check("B10 an orphan extra val mask -> split_count_mismatch", code == "split_count_mismatch", str(code))
        r11 = make_root(fresh("b_B11"))
        (r11 / "annotations" / "val" / "val_00001.png").rename(r11 / "annotations" / "val" / "val_00001.PNG")
        code, _, _ = iso(r11, "B11")
        check("B11 an upper-case .PNG mask -> split_count_mismatch", code == "split_count_mismatch", str(code))
        r12 = make_root(fresh("b_B12"))
        for p in (r12 / "images" / "val").iterdir():
            p.unlink()
        for p in (r12 / "annotations" / "val").iterdir():
            p.unlink()
        for n in ("v0.jpg", "v1.jpg"):
            (r12 / "images" / "val" / n).touch()
        for n in ("v0.png", "vX.png"):
            (r12 / "annotations" / "val" / n).touch()
        code, _, _ = iso(r12, "B12")
        check("B12 a stem mismatch (counts equal) -> pairing_stem_mismatch naming v1.jpg",
              code == "pairing_stem_mismatch" and "v1.jpg" in details[-1], details[-1][:160])
        code, _, _ = iso(TMP / "b_absent_root", "B13")
        check("B13 a non-existent root -> data_root_missing", code == "data_root_missing", str(code))
        r14 = make_root(fresh("b_B14"), counts={"train": 3, "val": 1})
        plant(r14, "images/test")
        code, _, _ = iso(r14, "B14")
        check("B14 TEST present AND a wrong count -> test_split_present (TEST is checked first)",
              code == "test_split_present", str(code))
        check("B15 every refusal is '[<code>] …' and names DL-21",
              all(d.startswith("[") and "DL-21" in d for d in details), str(len(details)))
        bad = [p for p in all_listed + all_opened
               if "test" in Path(p).parts or Path(p).name == "annotation_test.json" or "trap_00000" in p]
        check("B16 no TEST path was ever listed or opened; the trap file never appears (the spies did record "
              "the TRAIN/VAL listings)",
              bad == [] and any(Path(p).parts[-1:] in (("train",), ("val",)) for p in all_listed),
              f"bad={bad[:3]} listed={len(all_listed)}")
    finally:
        P.TRAINVAL_COUNTS = saved
    calls = []

    def rec(root, expected=None):
        calls.append(expected)
        return {"counts": {}, "test_surfaces": {}, "method": "recorder"}
    with patched(ISO, "assert_trainval_only_root", rec), patched(P, "pairing_check", lambda r, s: {"pairs": 0}):
        P.stage_data_isolation({"root": TMP})
    check("B17 the stage passes the real counts explicitly (never None / the mutable default)",
          calls == [{"train": 5367, "val": 846}], str(calls))


# ------------------------------------------------------------------------------------------ C
def section_c() -> None:
    root = make_root(fresh("c_root"))
    runs, isos = [], []
    good_ck = str(TMP / "c_ckpt_absent")

    def ctx(**kw):
        base = {"ckpt_dir": good_ck, "record": str(TMP / "c_record.json"), "rehearsal": False}
        base.update(kw)
        return base

    def args_code(**kw):
        return code_of(P.stage_arguments, ctx(**kw))

    real_run = subprocess.run
    with patched(subprocess, "run", lambda *a, **k: runs.append(a) or real_run(*a, **k)), \
            patched(ISO, "assert_trainval_only_root", lambda *a, **k: isos.append(a)), Spies() as sp:
        with env(PLANTSEG_DATA_ROOT=None, PYTHONPATH=str(REPO)):
            c01 = args_code()
        with env(PLANTSEG_DATA_ROOT="relative/root", PYTHONPATH=str(REPO)):
            c02 = args_code()
        with env(PLANTSEG_DATA_ROOT=str(root), PYTHONPATH=str(REPO)):
            c04 = args_code(ckpt_dir="relative/ckpt")
            inside = REPO / "e1tv_should_not_exist"
            c05 = args_code(ckpt_dir=str(inside))
            nf = fresh("c_notfresh")
            nf.mkdir()
            (nf / "e1_telemetry.jsonl").write_text("{}\n", encoding="utf-8")
            c06 = args_code(ckpt_dir=str(nf))
            c07 = args_code(ckpt_dir=str(root / "images" / "train"))     # exists and is NOT empty
            c15 = args_code(record="relative/rec.json")
            c16 = args_code(record=str(root / "rec.json"))
            c17 = args_code(record=str(TMP / "no_such_dir" / "rec.json"))
            empty = fresh("c_empty")
            empty.mkdir()
            c08a, c08b = args_code(), args_code(ckpt_dir=str(empty))
            c10 = args_code(record=str(REPO / "rec.json"))
            existing = TMP / "c_existing.json"
            existing.write_text("{}", encoding="utf-8")
            c11 = args_code(record=str(existing))
            c12 = args_code(ckpt_dir=str(empty), record=str(empty / "rec.json"))
        with env(PLANTSEG_DATA_ROOT=str(root), PYTHONPATH=str(TMP)):
            c09a = args_code()
        with env(PLANTSEG_DATA_ROOT=str(root), PYTHONPATH=None):
            c09b = args_code()
    check("C01 PLANTSEG_DATA_ROOT unset -> data_root_unset", c01 == "data_root_unset", str(c01))
    check("C02 a relative data root -> data_root_relative", c02 == "data_root_relative", str(c02))
    rc, _, _ = run_main(["gate", "--seed", "41", "--ckpt-dir", good_ck, "--expect-head", HEAD, "--rehearsal"])
    check("C03 --seed 41 -> usage exit 2", rc == 2, f"rc={rc}")
    check("C04 a relative --ckpt-dir -> ckpt_dir_relative", c04 == "ckpt_dir_relative", str(c04))
    check("C05 --ckpt-dir inside the repo -> ckpt_dir_inside_repo; nothing created",
          c05 == "ckpt_dir_inside_repo" and not (REPO / "e1tv_should_not_exist").exists(), str(c05))
    check("C06 --ckpt-dir holding e1_telemetry.jsonl -> ckpt_dir_not_fresh", c06 == "ckpt_dir_not_fresh", str(c06))
    listed_ck = [p for p in sp.listed if str(root / "images" / "train") in p]
    check("C07 an existing, non-empty --ckpt-dir under the data root -> ckpt_dir_inside_data_root (not "
          "ckpt_dir_not_fresh), never listed",
          c07 == "ckpt_dir_inside_data_root" and listed_ck == [], str(c07))
    check("C15 a relative --record -> record_relative", c15 == "record_relative", str(c15))
    check("C16 --record under the data root -> record_inside_data_root", c16 == "record_inside_data_root", str(c16))
    check("C17 --record whose directory does not exist -> record_parent_missing",
          c17 == "record_parent_missing", str(c17))
    check("C08 absent and empty ckpt dirs PASS; the gate creates neither",
          c08a is None and c08b is None and not Path(good_ck).exists(), f"{c08a}/{c08b}")
    check("C09 PYTHONPATH not exactly the checkout (other dir, or unset) -> pythonpath_not_checkout (X8)",
          c09a == c09b == "pythonpath_not_checkout", f"{c09a}/{c09b}")
    check("C10 --record inside the repo -> record_inside_repo", c10 == "record_inside_repo", str(c10))
    check("C11 an existing --record -> record_exists", c11 == "record_exists", str(c11))
    check("C12 --record under --ckpt-dir -> record_inside_ckpt_dir", c12 == "record_inside_ckpt_dir", str(c12))
    root_listed = [p for p in sp.listed if p.startswith(str(root))]
    check("C00 the arguments stage made no subprocess call, no isolation call, no data-root listing",
          runs == [] and isos == [] and root_listed == [], f"{len(runs)}/{len(isos)}/{root_listed[:2]}")
    rc, _, err = run_main(["gate", "--seed", "43", "--ckpt-dir", good_ck, "--expect-head", HEAD])
    check("C13 a real gate run without --record -> exit 2", rc == 2 and "--record" in err, f"rc={rc}")
    rc, _, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", good_ck, "--rehearsal"])
    check("C14 a missing --expect-head -> usage exit 2", rc == 2, f"rc={rc}")


# ------------------------------------------------------------------------------------------ D
def stub_stages(verdicts: dict, called: list, *, hub=None):
    out = []
    for name in STAGE_ORDER:
        def fn(ctx, name=name):
            called.append(name)
            if name == "arguments":
                ctx["root"] = Path(ctx.get("root_override", "/staged/plantseg"))
            if name == "imagenet_backbone" and hub:
                ctx["hub_dir"] = hub
            v = verdicts.get(name, "PASS")
            if v == "FAIL":
                raise P.GateError(f"{name}_stub_fail", f"stub failure in {name}")
            return v, f"stub {name}", {}
        out.append((name, fn))
    return out


def golden_block(seed: int, d: str, root: str, extra=(), hub=None) -> str:
    q = shlex.quote
    env_ = [f"PYTHONPATH={q(str(REPO))}", f"PLANTSEG_DATA_ROOT={q(root)}", f"PLANTSEG_IMAGE_DIGEST={q(DL21)}"]
    if hub:
        env_.append(f"TORCH_HOME={q(str(Path(hub).parent))}")
    cmd = (f"env -u PLANTSEG_GIT_COMMIT {' '.join(env_)} nohup {q(sys.executable)} -B src/training/train_e1.py "
           f"--real-run --confirm-real-run --init imagenet --ckpt-dir {q(d)} --seed {seed} --num-workers 12 "
           f"--log-every 50" + (" " + " ".join(q(a) for a in extra) if extra else "")
           + f" > {q(d)}/e1_stdout.log 2>&1 &")
    return "\n".join(["set -eu", f"cd {q(str(REPO))}",
                      f"[ -z \"$(ls -A {q(d)} 2>/dev/null)\" ] || {{ printf 'STOP: %s is not empty\\n' {q(d)}; exit 1; }}",
                      f"mkdir -p {q(d)}", cmd, f"echo $! > {q(d)}/e1.pid"])


def block_of(stdout: str) -> str:
    return stdout.split("----- launch block -----\n", 1)[1].split("\n----- end launch block -----", 1)[0]


def section_d() -> dict:
    d = "/workspace/b66/e1_seed43"
    rec = TMP / "d_record.json"
    hub = "/root/.cache/torch/hub"
    called = []
    with patched(P, "STAGES", stub_stages({}, called, hub=hub)):
        rc, out, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--record", str(rec)])
    block = block_of(out) if "----- launch block -----" in out else ""
    record = json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else {}
    gold = golden_block(43, d, str(Path("/staged/plantseg")), hub=hub)
    check("D01 all PASS -> exit 0, VERDICT: GO, the golden launch block, record verdict GO",
          rc == 0 and "VERDICT: GO" in out and block == gold and record.get("verdict") == "GO"
          and record.get("launch_block") == gold, f"rc={rc}")
    called = []
    with patched(P, "STAGES", stub_stages({"cuda": "SKIPPED", "imagenet_backbone": "SKIPPED"}, called)):
        rc, out, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--rehearsal"])
    check("D02 rehearsal (cuda/imagenet SKIPPED) -> exit 1, NO-GO (rehearsal), no block, no --real-run",
          rc == 1 and "VERDICT: NO-GO (rehearsal)" in out and "VERDICT: GO" not in out
          and "--real-run" not in out and len(called) == 14, f"rc={rc}")
    rec3 = TMP / "d3.json"
    with patched(P, "STAGES", stub_stages({"cuda": "SKIPPED"}, [])):
        rc, out, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--record", str(rec3)])
    check("D03 a SKIPPED stage in a real run -> exit 1, NO-GO (SKIPPED is never a GO)",
          rc == 1 and "VERDICT: NO-GO" in out and "VERDICT: GO\n" not in out and "--real-run" not in out)
    called = []
    with patched(P, "STAGES", stub_stages({"smoke_loss": "FAIL"}, called)):
        rc, out, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--record",
                               str(TMP / "d4.json")])
    check("D04 a FAIL at smoke_loss stops the gate; later stages not run; first failing stage named",
          rc == 1 and called == list(STAGE_ORDER[:6]) and "first failing stage: smoke_loss" in out
          and out.count("not reached") == 8, f"called={called}")
    called = []
    with patched(P, "STAGES", stub_stages({"cuda": "FAIL"}, called)):
        rc, out, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--rehearsal"])
    check("D05 a rehearsal never waives a FAIL", rc == 1 and called[-1] == "cuda" and len(called) == 8
          and "first failing stage: cuda" in out)
    called, runs = [], []
    real_run = subprocess.run
    with patched(P, "STAGES", stub_stages({"data_isolation": "FAIL"}, called)), \
            patched(subprocess, "run", lambda *a, **k: runs.append(a) or real_run(*a, **k)):
        rc, out, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--rehearsal"])
    check("D06 a data_isolation FAIL runs no later stage and no subprocess",
          rc == 1 and called == ["arguments", "data_isolation"] and runs == [])
    must = ["env -u PLANTSEG_GIT_COMMIT", f"PYTHONPATH={shlex.quote(str(REPO))}",
            f"PLANTSEG_DATA_ROOT={shlex.quote(str(Path('/staged/plantseg')))}", DL21,
            "--real-run --confirm-real-run --init imagenet", "--seed 43", "--num-workers 12",
            "--log-every 50", f"--ckpt-dir {d}", "set -eu",
            f"TORCH_HOME={shlex.quote(str(Path(hub).parent))}"]
    never = ["--resume", "--max-iters", "--batch-size", "--val-interval", "--max-val-batches", "--device",
             "--grad-clip-norm"]
    check("D07 the PRINTED launch block carries every pin and none of the forbidden flags",
          block != "" and all(m in block for m in must) and not any(n in block for n in never))
    ex = TMP / "d8_existing.json"
    ex.write_text("KEEP", encoding="utf-8")

    def args_fail(ctx):
        raise P.GateError("record_exists", "stub")
    with patched(P, "STAGES", [("arguments", args_fail)] + stub_stages({}, [])[1:]):
        rc, _, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--record", str(ex)])
    check("D08 a record path refused by `arguments` is never written", rc == 1 and ex.read_text() == "KEEP")

    def crash(ctx):
        raise ValueError("unexpected")
    stages9 = stub_stages({}, [])
    stages9[6] = ("image", crash)
    with patched(P, "STAGES", stages9):
        rc, out, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--record",
                               str(TMP / "d9.json")])
    check("D09 a stage that crashes is a FAIL [stage_error], never a GO",
          rc == 1 and "first failing stage: image [stage_error]" in out and "VERDICT: GO" not in out, out[-160:])
    with patched(P, "STAGES", stub_stages({}, [])):
        rc, out, _ = run_main(["gate", "--seed", "43", "--ckpt-dir", d, "--expect-head", HEAD, "--record",
                               str(TMP / "d10_missing_dir" / "rec.json")])
    check("D10 an evidence record that cannot be written turns GO into NO-GO; no launch block is printed",
          rc == 1 and "VERDICT: NO-GO" in out and "VERDICT: GO" not in out and "launch block" not in out,
          out[-160:])
    rec11 = TMP / "d11.json"
    with patched(P, "STAGES", stub_stages({}, [], hub=hub)):
        rc, out, _ = run_main(["gate", "--seed", "42", "--ckpt-dir", d, "--expect-head", HEAD, "--record",
                               str(rec11), "--profile", "e1_160k"])
    blk11 = block_of(out) if "----- launch block -----" in out else ""
    check("D11 profile e1_160k: the GO block is the golden 160k block (--iterations 160000, never --max-iters)",
          rc == 0 and blk11 == golden_block(42, d, str(Path("/staged/plantseg")),
                                            extra=("--iterations", "160000"), hub=hub)
          and "--iterations 160000" in blk11 and "--max-iters" not in blk11, f"rc={rc}")
    return {"block": gold, "d": d}


# ------------------------------------------------------------------------------------------ E
class FakeGit:
    """Scriptable replacement for P._git. `answers` maps an argv prefix tuple to (rc, stdout bytes)."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, args, timeout=120):
        self.calls.append(list(args))
        for prefix, (rc, out) in self.answers.items():
            if tuple(args[:len(prefix)]) == prefix:
                return subprocess.CompletedProcess(args, rc, out, b"")
        raise AssertionError(f"unexpected git argv {args}")        # strict: nothing unscripted runs


def git_answers(head=HEAD, status=b"", status_rc=0, anc_rc=0, cat_rc=0, sparse=("true", "false"), gp=None,
                count="1"):
    return {("rev-parse", "HEAD"): (0, (head + "\n").encode()),
            ("cat-file",): (cat_rc, b""),
            ("rev-list", "--count", "HEAD"): (0, (count + "\n").encode()),
            ("merge-base",): (anc_rc, b""),
            ("-c", "core.quotepath=false", "status"): (status_rc, status),
            ("config", "--get", "core.sparseCheckout"): (0, (sparse[0] + "\n").encode()),
            ("config", "--get", "core.sparseCheckoutCone"): (0, (sparse[1] + "\n").encode()),
            ("rev-parse", "--git-path"): (0, (str(gp) + "\n").encode() if gp else b"/nonexistent\n"),
            ("config", "--get", "remote.origin.url"): (0, b"")}


def section_e() -> None:
    check("E01 scoped_dirty: governed prefixes plus the class-weight artifact only",
          P.scoped_dirty(b" M src/a.py\x00?? reports/x.md\x00 M docs/IMPLEMENTATION_CONTRACT.md\x00"
                         b"R  scripts/new.py\x00scripts/old.py\x00 M reports/e1_class_weights.json\x00")
          == ["docs/IMPLEMENTATION_CONTRACT.md", "reports/e1_class_weights.json", "scripts/new.py", "src/a.py"]
          and P.scoped_dirty(b"") == [])
    check("E02 the status argv is scoped (governed pathspecs + the class-weight file)",
          P.status_argv() == ["-c", "core.quotepath=false", "status", "--porcelain=v1", "-z",
                              "--untracked-files=all", "--", *P.STATUS_PATHSPECS])
    sparse_file = TMP / "e_sparse"
    sparse_file.write_text("/*\n!/docs/reference/\n", encoding="utf-8")

    def repo_code(answers, *, rehearsal=False, expect=HEAD, commit_env=None):
        fg = FakeGit(answers)
        with patched(P, "_git", fg), env(PLANTSEG_GIT_COMMIT=commit_env):
            c = code_of(P.stage_repo_state, {"expect_head": expect, "rehearsal": rehearsal})
        return c, fg
    c, fg3 = repo_code(git_answers(gp=sparse_file))
    status_calls = [a for a in fg3.calls if "status" in a]
    check("E03 HEAD == --expect-head with a clean scoped status and the DL-19 sparse config PASSes; the "
          "only status calls are the scoped stage-3 status and the stage-14 snapshot",
          c is None and status_calls == [P.status_argv(), P.status_argv(P.UNCHANGED_PATHSPECS, ignored=True)]
          and all(a[a.index("--") + 1:] and not set(a[a.index("--") + 1:]) & {".", ":/", "docs", "docs/"}
                  for a in status_calls), f"{c} {status_calls}")
    c, _ = repo_code(git_answers(head="b" * 40, gp=sparse_file))
    check("E04 HEAD differs -> head_mismatch", c == "head_mismatch", str(c))
    c, _ = repo_code(git_answers(gp=sparse_file), expect="5325e32")
    check("E05 a non-40-hex --expect-head -> expect_commit_malformed", c == "expect_commit_malformed", str(c))
    c, _ = repo_code(git_answers(anc_rc=1, gp=sparse_file))
    check("E06 885523a not an ancestor -> safety_floor_missing", c == "safety_floor_missing", str(c))
    c1, _ = repo_code(git_answers(status=b"?? src/x.py\x00", gp=sparse_file))
    c2, _ = repo_code(git_answers(status=b"?? notes.txt\x00", gp=sparse_file))
    check("E07 an untracked governed path -> governed_paths_dirty; a non-governed one PASSes",
          c1 == "governed_paths_dirty" and c2 is None, f"{c1}/{c2}")
    c, _ = repo_code(git_answers(status_rc=128, gp=sparse_file))
    check("E08 scoped status rc 128 -> governed_state_unprovable", c == "governed_state_unprovable", str(c))
    ca, _ = repo_code(git_answers(gp=sparse_file), commit_env="f77d05d7b35187bf0da7e7b94a629549fe2e1c05")
    cb, _ = repo_code(git_answers(gp=sparse_file), commit_env="")
    cc, _ = repo_code(git_answers(gp=sparse_file), commit_env="   ")
    check("E09 PLANTSEG_GIT_COMMIT set -> git_commit_env_set; empty or blank PASSes",
          ca == "git_commit_env_set" and cb is None and cc is None, f"{ca}/{cb}/{cc}")
    bad_file = TMP / "e_sparse_bad"
    bad_file.write_text("/*\n", encoding="utf-8")
    codes = [repo_code(git_answers(sparse=("", "false"), gp=sparse_file))[0],
             repo_code(git_answers(sparse=("true", "true"), gp=sparse_file))[0],
             repo_code(git_answers(gp=bad_file))[0]]
    check("E10 sparse unset / cone mode / patterns ['/*'] -> sparse_checkout_mismatch",
          codes == ["sparse_checkout_mismatch"] * 3, str(codes))
    fg11 = FakeGit(git_answers(cat_rc=1))
    with patched(P, "_git", fg11), env(PLANTSEG_GIT_COMMIT=None):
        res11 = P.stage_repo_state({"expect_head": HEAD, "rehearsal": True})
    queried = [a for a in fg11.calls if a[:2] == ["config", "--get"] and "sparse" in a[2].lower()]
    cg, _ = repo_code(git_answers(cat_rc=1, anc_rc=1, gp=sparse_file))
    ch, _ = repo_code(git_answers(cat_rc=1, anc_rc=1, count="7"), rehearsal=True)
    cp, _ = repo_code(git_answers(cat_rc=0, anc_rc=1), rehearsal=True)
    check("E11 rehearsal: sparse config not queried and the floor SKIPPED only in a history-less repo "
          "(stage PASS, both skips named); a real gate, a repo with history, or a present-but-not-ancestor "
          "floor -> safety_floor_missing",
          res11[0] == "PASS" and res11[1].count("SKIPPED (rehearsal") == 2 and queried == []
          and cg == ch == cp == "safety_floor_missing", f"{res11[1][:120]} | {cg}/{ch}/{cp}")
    with env(PLANTSEG_IMAGE_DIGEST=None):
        e12a = code_of(P.stage_image, {"rehearsal": True})
    with env(PLANTSEG_IMAGE_DIGEST="sha256:" + "0" * 64):
        e12b = code_of(P.stage_image, {"rehearsal": True})
    check("E12 digest unset -> image_digest_missing; wrong -> image_digest_mismatch",
          e12a == "image_digest_missing" and e12b == "image_digest_mismatch", f"{e12a}/{e12b}")
    seen = []

    def fake_child(stdout, rc=0):
        def f(argv, timeout=3600):
            seen.append([str(a) for a in argv])
            return subprocess.CompletedProcess(argv, rc, stdout, "")
        return f
    with env(PLANTSEG_IMAGE_DIGEST=DL21):
        with patched(P, "_child", fake_child("RESULT: FAIL (30/31)\n", 1)):
            e13a = code_of(P.stage_image, {"rehearsal": False})
        with patched(P, "_child", fake_child("RESULT: PASS (31/31)\n", 0)):
            e13b = code_of(P.stage_image, {"rehearsal": False})
        with patched(P, "_child", fake_child("no result line\n", 0)):
            e13c = code_of(P.stage_image, {"rehearsal": True})
    check("E13 preflight_environment must exit 0 with RESULT: PASS; --mode gpu for a real gate, image "
          "for a rehearsal",
          e13a == "environment_preflight_failed" and e13b is None and e13c == "environment_preflight_failed"
          and seen[0][-2:] == ["--mode", "gpu"] and seen[-1][-2:] == ["--mode", "image"], str(seen))
    good = TMP / "e_bb_good.bin"
    good.write_bytes(b"x" * 15)
    wrong = TMP / "e_bb_wrong.bin"
    wrong.write_bytes(b"y" * 15)
    check("E14 check_backbone: absent / size / sha codes; the right bytes pass",
          P.check_backbone(TMP / "absent.pth") == "imagenet_not_cached"
          and P.check_backbone(good) == "imagenet_size_mismatch"
          and P.check_backbone(wrong, sha=hashlib.sha256(b"x" * 15).hexdigest(), size=15) == "imagenet_sha_mismatch"
          and P.check_backbone(good, sha=hashlib.sha256(b"x" * 15).hexdigest(), size=15) is None)
    import src.models.student as ST
    builds = []
    saved_hub = torch.hub.get_dir()
    try:
        torch.hub.set_dir(str(fresh("e15") / "hub"))
        with patched(ST, "build_student", lambda *a, **k: builds.append(a) or None):
            e15a = code_of(P.stage_imagenet_backbone, {"rehearsal": False})
            res = P.stage_imagenet_backbone({"rehearsal": True})
    finally:
        torch.hub.set_dir(saved_hub)
    check("E15 absent backbone: a real gate FAILs imagenet_not_cached, a rehearsal SKIPs, no build attempted",
          e15a == "imagenet_not_cached" and res[0] == "SKIPPED" and builds == [], f"{e15a}/{res[0]}")
    fake = type(sys)("src._pf_fake")
    fake.__file__ = str(TMP / "elsewhere" / "fake.py")
    sys.modules["src._pf_fake"] = fake
    try:
        e16 = code_of(P.stage_module_provenance, {})
    finally:
        del sys.modules["src._pf_fake"]
    check("E16 a src.* module loaded from outside the checkout -> import_origin_foreign",
          e16 == "import_origin_foreign", str(e16))
    if torch.cuda.is_available():
        skip("E17 the CUDA child probe on a CPU-only host -> cuda_unavailable", "CUDA present (plan-named SKIP)")
    else:
        with env(PYTHONPATH=str(REPO)):
            e17 = code_of(P.stage_cuda, {"rehearsal": False})
        check("E17 the CUDA child probe on a CPU-only host -> cuda_unavailable", e17 == "cuda_unavailable", str(e17))
    dry_seen = []

    def dry_child(fmt, rc=0):
        def f(argv, timeout=3600):
            argv = [str(a) for a in argv]
            dry_seen.append(argv)
            tmpdir = argv[argv.index("--ckpt-dir") + 1]
            return subprocess.CompletedProcess(argv, rc, fmt.format(dir=tmpdir), "")
        return f
    good_out = "[ckpt] dir={dir} (verified OUTSIDE repo)\nRESULT: PASS (6/6 checks exercised, 0 skipped)\n"
    ctx18 = {"seed": 43, "root": TMP / "no_root", "profile": "e1_80k"}
    with patched(P, "_child", dry_child(good_out)):
        r18 = P.stage_dry_run(ctx18)
    tmp18 = dry_seen[-1][dry_seen[-1].index("--ckpt-dir") + 1]
    check("E18 dry_run: PASS on 6/6 with the temp [ckpt] dir; temp removed; --dry-run --seed 43 passed",
          r18[0] == "PASS" and not Path(tmp18).exists() and "--dry-run" in dry_seen[-1]
          and dry_seen[-1][dry_seen[-1].index("--seed") + 1] == "43")
    with patched(P, "_child", dry_child("[ckpt] dir=/elsewhere (verified OUTSIDE repo)\n"
                                        "RESULT: PASS (6/6 checks exercised, 0 skipped)\n")):
        e19 = code_of(P.stage_dry_run, dict(ctx18))
    tmp19 = dry_seen[-1][dry_seen[-1].index("--ckpt-dir") + 1]
    check("E19 a different [ckpt] dir -> dry_run_failed; temp removed",
          e19 == "dry_run_failed" and not Path(tmp19).exists(), str(e19))
    e20 = []
    for text in ("[ckpt] dir={dir} (verified OUTSIDE repo)\nRESULT: PASS (5/6 checks exercised, 1 skipped)\n",
                 "RESULT: NOOP ([CHECKS] 0/6)\n"):
        with patched(P, "_child", dry_child(text)):
            e20.append(code_of(P.stage_dry_run, dict(ctx18)))
    check("E20 reduced coverage or NOOP -> dry_run_failed", e20 == ["dry_run_failed"] * 2, str(e20))
    with patched(P, "_child", dry_child(good_out)):
        r25 = P.stage_dry_run(dict(ctx18, profile="e1_160k"))
    a25 = dry_seen[-1]
    ok25 = (r25[0] == "PASS" and "--iterations" in a25
            and a25[a25.index("--iterations") + 1] == "160000")
    check("E25 profile e1_160k: the gate's dry run exercises --iterations 160000", ok25, str(a25[-4:]))
    fg_same = FakeGit({("-c",): (0, b" M x\x00")})
    with patched(P, "_git", fg_same):
        a = code_of(P.stage_repo_unchanged, {"snapshot": b" M x\x00"})
        b = code_of(P.stage_repo_unchanged, {"snapshot": b""})
        c = code_of(P.stage_repo_unchanged, {"snapshot": None})
    check("E21 repo_unchanged: equal PASSes, changed FAILs, unreadable FAILs",
          (a, b, c) == (None, "repo_changed", "repo_state_unreadable"), str((a, b, c)))
    with patched(P, "_child", fake_child("RESULT: PASS (12/12)\n", 0)):
        e22a = code_of(P.stage_smoke_loader_seed, {})
    with patched(P, "_child", fake_child("RESULT: FAIL (11/12)\n", 1)):
        e22b = code_of(P.stage_smoke_loader_seed, {})
    check("E22 smoke_loader_seed must exit 0 with RESULT: PASS", e22a is None and e22b == "smoke_loader_seed_failed")
    with patched(P, "REUSED_PF1", dict(P.REUSED_PF1, class_weights=lambda dev: (True, "stub ok", 0.0))):
        e23a = P.stage_class_weights({})
        with patched(P, "CLASS_WEIGHTS_SHA256", "0" * 64):
            e23b = code_of(P.stage_class_weights, {})
    with patched(P, "REUSED_PF1", dict(P.REUSED_PF1, class_weights=lambda dev: (False, "stub fail", 0.0))):
        e23c = P.stage_class_weights({})
    check("E23 X5: class_weights = preflight_e1's check AND the pinned sha256",
          e23a[0] == "PASS" and e23b == "class_weights_sha_mismatch" and e23c[0] == "FAIL", str(e23a[1])[:80])
    hub24 = fresh("e24") / "hub"
    (hub24 / "checkpoints").mkdir(parents=True)
    (TMP / "e24" / "x").mkdir()
    hub24_unnormalised = TMP / "e24" / "x" / ".." / "hub"
    (hub24 / "checkpoints" / P.IMAGENET_FILE).write_bytes(b"z" * 15)
    built = []

    class _Stub:
        def __init__(self, used):
            self.used_pretrained = used
    saved_hub = torch.hub.get_dir()
    try:
        torch.hub.set_dir(str(hub24_unnormalised))
        with patched(P, "IMAGENET_BYTES", 15), patched(P, "IMAGENET_SHA256", hashlib.sha256(b"z" * 15).hexdigest()):
            with patched(ST, "build_student", lambda *a, **k: built.append(k.get("pretrained", a)) or _Stub(True)):
                ctx24 = {"rehearsal": False}
                res24 = P.stage_imagenet_backbone(ctx24)
            with patched(ST, "build_student", lambda *a, **k: _Stub(False)):
                e24b = code_of(P.stage_imagenet_backbone, {"rehearsal": False})
        torch.hub.set_dir(str(TMP / "e24" / "not_named_hub"))
        e24c = code_of(P.stage_imagenet_backbone, {"rehearsal": True})
    finally:
        torch.hub.set_dir(saved_hub)
    check("E24 backbone present and verified -> cached build_student(IMAGENET1K_V2) is used; the hub dir is "
          "made absolute (normalised, symlinks not followed), recorded raw and absolute (X8) and set for the "
          "launch block; used_pretrained False -> imagenet_not_loaded; a hub dir not named 'hub' -> "
          "hub_dir_nonstandard",
          res24[0] == "PASS" and built == [P.IMAGENET_ALIAS]
          and ctx24.get("hub_dir") == os.path.abspath(str(hub24_unnormalised)) == res24[2].get("hub_dir")
          and ctx24["hub_dir"] != str(hub24_unnormalised) and res24[2].get("hub_dir_raw") == str(hub24_unnormalised)
          and e24b == "imagenet_not_loaded" and e24c == "hub_dir_nonstandard",
          f"{res24[0]} {built} {ctx24.get('hub_dir')} {e24b} {e24c}")


# ------------------------------------------------------------------------------------------ F
def section_f() -> None:
    def crm(rows, seed=43, head=HEAD, name="f"):
        d = fresh(name)
        d.mkdir()
        with open(d / "e1_telemetry.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"event": "train", "iter": 1}) + "\n")
        return run_main(["check-run-meta", "--ckpt-dir", str(d), "--seed", str(seed), "--expect-head", head])

    base = dict(SEED42_RUN_META, seed=43, git_head=HEAD)
    rc, out, _ = crm([base])
    check("F01 the seed-42 row with seed 43 and the expected head -> PASS", rc == 0 and "RESULT: PASS" in out, out[-160:])
    muts = [("F02", "num_workers", 11), ("F03", "resumed_from", "x"), ("F04", "git_head_source", "image_env"),
            ("F05", "image_digest", None), ("F06", "poly_horizon", 160000), ("F07", "max_iters", 160000),
            ("F08", "torch", "2.9.1"), ("F11", "params", 2933687), ("F12", "used_pretrained", False),
            ("F13", "git_head", "c" * 40), ("F14", "seed", 44)]
    for tag, key, val in muts:
        rc, out, _ = crm([dict(base, **{key: val})], name=f"f_{tag}")
        check(f"{tag} run_meta {key}={val!r} -> FAIL naming {key}", rc == 1 and f"MISMATCH: {key}:" in out,
              out[-120:])
    rc, out, _ = crm([base, base], name="f_F09")
    check("F09 two run_meta rows (an appended relaunch) -> FAIL", rc == 1 and "run_meta rows: 2" in out)
    nonum = {k: v for k, v in base.items() if k != "numpy"}
    rc, out, _ = crm([nonum], name="f_F10")
    check("F10 a missing run_meta key -> FAIL", rc == 1 and "numpy" in out)
    rc, out, _ = crm([dict(base, foo=1)], name="f_F15")
    check("F15 an extra run_meta key -> FAIL", rc == 1 and "foo" in out)
    rc, out, _ = crm([dict(base, gpu_name="NVIDIA RTX A6000")], name="f_F16")
    check("F16 a non-A40 gpu_name is a WARN only", rc == 0 and "WARN" in out)
    empty = fresh("f_F17")
    empty.mkdir()
    rc, out, _ = run_main(["check-run-meta", "--ckpt-dir", str(empty), "--seed", "43", "--expect-head", HEAD])
    check("F17 no telemetry file -> FAIL", rc == 1)
    torn = fresh("f_F18")
    torn.mkdir()
    (torn / "e1_telemetry.jsonl").write_text(
        json.dumps(base) + "\n" + json.dumps({"event": "train", "iter": 1}) + "\n" + '{"event": "train", "it',
        encoding="utf-8")
    rc, out, _ = run_main(["check-run-meta", "--ckpt-dir", str(torn), "--seed", "43", "--expect-head", HEAD])
    check("F18 a torn final train row (run still appending) does not crash or fail check-run-meta",
          rc == 0 and "RESULT: PASS" in out, out[-120:])
    bad19 = fresh("f_F19")
    bad19.mkdir()
    (bad19 / "e1_telemetry.jsonl").write_text(
        json.dumps(base) + "\n" + '{"event": "run_meta", "se\n', encoding="utf-8")
    rc, out, _ = run_main(["check-run-meta", "--ckpt-dir", str(bad19), "--seed", "43", "--expect-head", HEAD])
    check("F19 an undecodable line that mentions run_meta -> FAIL",
          rc == 1 and "mentions run_meta but is not valid JSON" in out, out[-160:])

    def crm160(row, name):
        d = fresh(name)
        d.mkdir()
        (d / "e1_telemetry.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return run_main(["check-run-meta", "--ckpt-dir", str(d), "--seed", "42", "--expect-head", HEAD,
                         "--profile", "e1_160k"])
    rc, out, _ = crm160(dict(SEED42_RUN_META, git_head=HEAD, max_iters=160000, poly_horizon=160000), "f_F20")
    check("F20 profile e1_160k: a seed-42 row with max_iters == poly_horizon == 160000 -> PASS",
          rc == 0 and "RESULT: PASS" in out, out[-120:])
    rc, out, _ = crm160(dict(SEED42_RUN_META, git_head=HEAD), "f_F21")
    check("F21 profile e1_160k: an 80k row (the forgotten --iterations) -> FAIL naming max_iters and poly_horizon",
          rc == 1 and "MISMATCH: max_iters:" in out and "MISMATCH: poly_horizon:" in out, out[-160:])


# ------------------------------------------------------------------------------------------ G
def section_g(block: str, d: str) -> None:
    calls, isos = [], []

    def recorder(**kw):
        calls.append(kw)
        return 0

    real_iso = ISO.assert_trainval_only_root

    def iso_spy(*a, **k):
        isos.append(a)
        return real_iso(*a, **k)

    def e1_main(argv, root):
        out, err = io.StringIO(), io.StringIO()
        saved_root = e1.DATA["root"]
        e1.DATA["root"] = str(root)
        try:
            with patched(e1, "run", recorder), patched(torch.cuda, "is_available", lambda: True), \
                    patched(ISO, "assert_trainval_only_root", iso_spy), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = e1.main(argv)
        finally:
            e1.DATA["root"] = saved_root
        return rc, out.getvalue(), err.getvalue()

    real = ["--real-run", "--confirm-real-run", "--device", "cuda"]
    sp = Spies()
    for tag, surface in (("G01", "images/test"), ("G02", "annotations/test"), ("G03", "annotation_test.json")):
        r = make_root(fresh(f"g_{tag}"))
        plant(r, surface)
        n0 = len(calls)
        with sp:
            rc, _, err = e1_main(real, r)
        check(f"{tag} train_e1 real run with {surface} -> exit 2 [test_split_present] DL-21; run() not called",
              rc == 2 and "[test_split_present]" in err and "DL-21" in err and len(calls) == n0, err[-120:])
    rc, _, err = e1_main(real, TMP / "g_absent")
    check("G05 a missing root -> exit 2 [data_root_missing]", rc == 2 and "[data_root_missing]" in err)
    sp6 = Spies()
    with sp6:
        rc, _, err = e1_main(real, make_root(fresh("g_G06")))
    check("G06 wrong counts -> exit 2 [split_count_mismatch]", rc == 2 and "[split_count_mismatch]" in err)
    bad = [p for p in sp.listed + sp.opened if "test" in Path(p).parts or Path(p).name == "annotation_test.json"]
    check("G04 no TEST path was listed or opened by the train_e1 gate (positive control: the same spies "
          "record the TRAIN/VAL scans on the G06 root)",
          bad == [] and any(Path(p).parts[-1:] in (("train",), ("val",)) for p in sp6.listed),
          f"bad={bad[:3]} control={len(sp6.listed)}")
    full = make_root(fresh("g_full"), counts={"train": 5367, "val": 846})
    n0, n_iso0 = len(calls), len(isos)
    rc, out, _ = e1_main(real, full)
    check("G07 a full-count TRAIN/VAL root -> one isolation call, run() called once; the [data] line is printed",
          rc == 0 and len(calls) == n0 + 1 and len(isos) == n_iso0 + 1
          and "[data] TRAIN/VAL-only staged root verified (DL-21)" in out)
    n_iso = len(isos)
    rc, _, err = e1_main(["--real-run", "--confirm-real-run", "--device", "cpu"], full)
    check("G08 --device cpu -> exit 2 'on CPU' before any isolation call",
          rc == 2 and "on CPU" in err and len(isos) == n_iso)
    n0, n_iso = len(calls), len(isos)
    rc, _, _ = e1_main(["--dry-run"], full)
    check("G09 --dry-run kwargs == the P4 dry golden + poly_horizon 80000 (S3); no isolation call",
          rc == 0 and calls[-1] == dict(P4_DRY, poly_horizon=80000) and len(isos) == n_iso,
          json.dumps(calls[-1], default=str)[:160])

    def block_argv(blk):
        ln = next(x for x in blk.splitlines() if "train_e1.py" in x)
        tk = shlex.split(ln)
        return tk[tk.index("src/training/train_e1.py") + 1: tk.index(">")]
    argv = block_argv(block)
    for tag, seed in (("G10", 42), ("G11", 43), ("G12", 44)):
        a2 = [seed_tok if seed_tok != "43" else str(seed) for seed_tok in argv]
        rc, _, _ = e1_main(a2, full)
        want = dict(P4_REAL42, seed=seed, log_every=50, ckpt_dir_arg=d, poly_horizon=80000)
        got = calls[-1]
        diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
        check(f"{tag} the launch block's argv (seed {seed}) -> run() kwargs == P4 golden + "
              "{seed, log_every 50, ckpt_dir_arg, poly_horizon 80000} only", rc == 0 and diff == [], str(diff))
    block160 = golden_block(42, d, str(Path("/staged/plantseg")), extra=P.PROFILES["e1_160k"]["extra_args"])
    rc, _, _ = e1_main(block_argv(block160), full)
    want = dict(P4_REAL42, seed=42, log_every=50, ckpt_dir_arg=d, max_iters=160000, poly_horizon=160000)
    got = calls[-1]
    diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    check("G16 the e1_160k launch block's argv -> run() kwargs == P4 golden + {seed 42, log_every 50, "
          "ckpt_dir_arg, max_iters 160000, poly_horizon 160000} only", rc == 0 and diff == [], str(diff))
    st = (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().numpy().tobytes())
    real_iso(full, {"train": 5367, "val": 846})
    st2 = (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().numpy().tobytes())
    check("G13 assert_trainval_only_root draws no RNG (python/numpy/torch states identical)", st == st2)
    r = subprocess.run([sys.executable, "-B", "-c",
                        "import sys; import src.training.train_e1; print('ISO_LOADED', 'src.data.isolation' in sys.modules)"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1"))
    check("G14 importing train_e1 does not load src.data.isolation (lazy import)",
          "ISO_LOADED False" in r.stdout, (r.stdout + r.stderr)[-160:])
    ck = fresh("g_G15")
    ck.mkdir()
    (ck / "e1_telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    n0 = len(calls)
    rc, _, _ = e1_main(real + ["--ckpt-dir", str(ck)], full)
    check("G15 Q3 (S2c not adopted): an existing telemetry file is not refused by train_e1",
          rc == 0 and len(calls) == n0 + 1)


# ------------------------------------------------------------------------------------------ R
def mk_decodable_root(root: Path) -> None:
    from PIL import Image
    rng = np.random.RandomState(42)
    for split, n in (("train", 5367), ("val", 846)):
        idir, adir = root / "images" / split, root / "annotations" / split
        idir.mkdir(parents=True)
        adir.mkdir(parents=True)
        for i in range(n):
            h, w = 24 + (i % 5) * 4, 32 + (i % 7) * 4
            Image.fromarray(rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)).save(idir / f"syn_{split}_{i:05d}.jpg")
            m = np.zeros((h, w), dtype=np.uint8)
            m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1 + (i % 115)
            Image.fromarray(m).save(adir / f"syn_{split}_{i:05d}.png")


def section_r() -> None:
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(REPO), capture_output=True, text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != REPO:
        print("--full-rehearsal needs the checkout to be a (scratch) git repository", file=sys.stderr)
        raise SystemExit(2)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True).stdout.strip()
    root = TMP / "r_root"
    mk_decodable_root(root)
    snap_argv = ["git", "-c", "core.quotepath=false", "status", "--porcelain=v1", "-z", "--untracked-files=all",
                 "--ignored=matching", "--", *P.UNCHANGED_PATHSPECS]
    snap0p = subprocess.run(snap_argv, cwd=str(REPO), capture_output=True)
    snap0 = snap0p.stdout
    renv = {k: v for k, v in os.environ.items() if k != "PLANTSEG_GIT_COMMIT"}
    renv.update(PYTHONPATH=str(REPO), PLANTSEG_DATA_ROOT=str(root), PLANTSEG_IMAGE_DIGEST=DL21,
                PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")

    def gate(tag):
        rec = TMP / f"r_{tag}.json"
        r = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "preflight_e1_trainval.py"), "gate",
                            "--rehearsal", "--seed", "43", "--ckpt-dir", str(TMP / f"r_ckpt_{tag}"),
                            "--expect-head", head, "--record", str(rec)],
                           cwd=str(REPO), capture_output=True, text=True, env=renv, timeout=7200)
        return r, (json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else {})
    before_dirs = set(os.listdir(tempfile.gettempdir()))
    r, rec = gate("main")
    print(r.stdout)
    st = {s["stage"]: s["status"] for s in rec.get("stages", [])}
    skipped = sorted(k for k, v in st.items() if v == "SKIPPED")
    passed = sorted(k for k, v in st.items() if v == "PASS")
    check("R01 rehearsal: exit 1, NO-GO (rehearsal); only cuda (and an uncached backbone) SKIPPED, all else PASS",
          r.returncode == 1 and "VERDICT: NO-GO (rehearsal)" in r.stdout and len(st) == 14
          and skipped in (["cuda", "imagenet_backbone"], ["cuda"]) and len(passed) == 14 - len(skipped),
          f"skipped={skipped} passed={len(passed)}")
    check("R02 R5 seed sequence MATCH (the same preflight_e1 object)", "seed sequence MATCH" in r.stdout)
    after_dirs = set(os.listdir(tempfile.gettempdir()))
    left = [x for x in after_dirs - before_dirs if x.startswith("e1tv_dryrun_")]
    iso = next((s["record"].get("isolation") for s in rec.get("stages", []) if s["stage"] == "data_isolation"), {})
    check("R03 no e1tv_dryrun_* temp dir remains; the record carries 5367/846",
          left == [] and iso.get("counts") == {"train": {"images": 5367, "masks": 5367},
                                                "val": {"images": 846, "masks": 846}}, str(left))
    snap1p = subprocess.run(snap_argv, cwd=str(REPO), capture_output=True)
    check("R04 the scoped status (governed + reports, incl. ignored) is identical before and after "
          "(both status commands succeeded)",
          snap0p.returncode == 0 and snap1p.returncode == 0 and snap0 == snap1p.stdout,
          f"rc={snap0p.returncode}/{snap1p.returncode}")
    (root / "images" / "test").mkdir()
    r5, _ = gate("test")
    shutil.rmtree(root / "images" / "test")
    check("R05 a planted images/test -> first failing stage data_isolation [test_split_present]; no PF1 stage ran",
          r5.returncode == 1 and "first failing stage: data_isolation [test_split_present]" in r5.stdout
          and not any(ln.startswith("STAGE ") for ln in r5.stdout.splitlines()), r5.stdout[-200:])
    m = root / "annotations" / "val" / "syn_val_00000.png"
    m.rename(root / "annotations" / "val" / "syn_val_zzzzz.png")
    r6, _ = gate("pair")
    (root / "annotations" / "val" / "syn_val_zzzzz.png").rename(m)
    check("R06 one stem mismatch -> data_isolation [pairing_stem_mismatch]",
          r6.returncode == 1 and "first failing stage: data_isolation [pairing_stem_mismatch]" in r6.stdout,
          r6.stdout[-200:])


# ------------------------------------------------------------------------------------------ main
def rm_tree(path: Path) -> None:
    for p in path.rglob("*"):
        try:
            os.chmod(p, stat.S_IWRITE)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-rehearsal", action="store_true")
    args = ap.parse_args()
    print(f"torch {torch.__version__}  platform {sys.platform}  tmp={TMP}")
    try:
        gold = {"block": golden_block(43, "/workspace/b66/e1_seed43", str(Path("/staged/plantseg"))),
                "d": "/workspace/b66/e1_seed43"}
        sections = [("A", section_a), ("B", section_b), ("C", section_c), ("D", section_d),
                    ("E", section_e), ("F", section_f)]
        for label, fn in sections:
            try:
                res = fn()
                if label == "D" and res:
                    gold = res
            except Exception as e:                        # noqa: BLE001 -- a section crash is a FAIL
                check(f"{label} section completed", False,
                      f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-300:]}")
        try:
            section_g(gold["block"], gold["d"])
        except Exception as e:                            # noqa: BLE001
            check("G section completed", False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-300:]}")
        if args.full_rehearsal:
            try:
                section_r()
            except SystemExit:
                raise
            except Exception as e:                        # noqa: BLE001
                check("R section completed", False, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-300:]}")
    finally:
        rm_tree(TMP)
    width = max(len(n) for n, _, _ in CHECKS)
    for name, verdict, detail in CHECKS:
        print(f"  [{verdict}] {name:<{width}}  {detail}")
    n_pass = sum(1 for _, v, _ in CHECKS if v == "PASS")
    n_fail = sum(1 for _, v, _ in CHECKS if v == "FAIL")
    n_skip = sum(1 for _, v, _ in CHECKS if v == "SKIP")
    counted = n_pass + n_fail
    print(f"SUMMARY  {n_pass}/{counted} checks passed ({n_skip} skipped)")
    print(f"RESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({n_pass}/{counted})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
