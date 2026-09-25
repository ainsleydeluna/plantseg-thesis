#!/usr/bin/env python3
"""Smoke for scripts/probe_e1_grad_determinism.py (N10; B66-prep B-4). CPU only; no real data.

Default mode runs the probe with --synthetic --device cpu --repeats 3 and checks its refusals.
--full (pinned image) also drives the real path end to end: a synthetic checkpoint (random-init student,
its own sha256 passed as --expect-sha256), a decodable 5367/846 TRAIN/VAL-only root, --num-workers 12, and
a batch of 3 set through a driver (a CPU train step at the recipe's batch 16 exceeds a CPU container's memory).
Probe runs set PLANTSEG_DATA_ROOT to a missing temp path (unset falls back to configs.data's real default
root); S20 unsets it on purpose and overrides DATA['root'] in-process before the probe reads it.
Writes only under tempfile.mkdtemp() (removed in `finally`). Exit 0 only when every check PASSes.

Run:  python -B scripts/smoke_probe_e1_grad.py [--full]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse     # noqa: E402
import ast          # noqa: E402
import hashlib      # noqa: E402
import json         # noqa: E402
import os           # noqa: E402
import shutil       # noqa: E402
import subprocess   # noqa: E402
import tempfile     # noqa: E402
import textwrap     # noqa: E402
import traceback    # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "probe_e1_grad_determinism.py"
TMP = Path(tempfile.mkdtemp(prefix="smoke_probe_e1_grad_")).resolve()
NO_ROOT = TMP / "no_data_root"                  # never created
E1_CODE_PATHSPECS = ["src", "configs", "scripts", "requirements-e1.txt", "reports/e1_class_weights.json"]
CHECKS: list[tuple[str, str, str]] = []
B6_ON = {"deterministic_algorithms": True, "deterministic_algorithms_warn_only": True,
         "cudnn_deterministic": True, "cudnn_benchmark": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def check(name, ok, detail=""):
    CHECKS.append((name, "PASS" if ok else "FAIL", " | ".join(str(detail).split("\n"))[:240]))


def probe(args, env_extra=None, timeout=3600, script=PROBE, cwd=REPO):
    env = {k: v for k, v in os.environ.items() if k not in ("CUBLAS_WORKSPACE_CONFIG",)}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               PYTHONPATH=os.pathsep.join([str(REPO)] + ([os.environ["PYTHONPATH"]]
                                                         if os.environ.get("PYTHONPATH") else [])))
    # A missing temp root, never unset: unset falls back to configs.data's default (a real dataset path).
    env["PLANTSEG_DATA_ROOT"] = str(NO_ROOT)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-B", str(script), *args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)


def snapshot():
    """Scoped git status of the export's code dirs (incl. untracked/ignored) plus its top-level names;
    None unless REPO is itself the top of a git work tree."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(REPO), capture_output=True, text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != REPO:
        return None
    r = subprocess.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching",
                        "--", "src", "scripts", "configs", "reports"], cwd=str(REPO), capture_output=True)
    return (r.stdout, sorted(os.listdir(REPO))) if r.returncode == 0 else None


def section_default():
    out = TMP / "n10.json"
    r = probe(["--synthetic", "--device", "cpu", "--repeats", "3", "--out", str(out)])
    rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    check("S01 synthetic CPU run completes: exit 0, 'RESULT: COMPLETE', record written",
          r.returncode == 0 and "RESULT: COMPLETE" in r.stdout and bool(rec), (r.stdout + r.stderr)[-200:])
    check("S02 probes C, A and B each record exactly 3 passes; order C -> A -> B",
          all(len(rec.get(k, {}).get("passes", [])) == 3 for k in ("probe_C", "probe_A", "probe_B"))
          and rec.get("probe_order") == ["C", "A", "B"])
    lines = r.stdout.splitlines()
    idx = {k: next((i for i, ln in enumerate(lines) if ln.startswith(f"[{k}] pass 1")), -1) for k in "CAB"}
    check("S03 every pass is printed (3 x [C], [A], [B]) and C prints before A before B",
          all(sum(ln.startswith(f"[{k}] pass") for ln in lines) == 3 for k in "CAB")
          and -1 < idx["C"] < idx["A"] < idx["B"], str(idx))
    check("S04 P0: E1's determinism settings are live (set_seed(42)); versions, TF32, digest, HEAD and the "
          "E1-code-path git status (exactly the five scoped pathspecs; never a whole-repository status) recorded",
          rec.get("determinism") == B6_ON and all(k in rec for k in ("torch", "tf32", "image_digest",
                                                                     "fill_uninitialized_memory", "git_head"))
          and rec.get("git_status_e1_code_paths") == "" and rec.get("git_status_pathspecs") == E1_CODE_PATHSPECS,
          repr(rec.get("git_status_e1_code_paths"))[:80])
    a = rec.get("probe_A", {}).get("passes", [{}])
    check("S05 Probe A records ce_mean, ce_sum, the implied and the ATen normaliser (value or error) and the "
          "gradient hash per pass",
          all(set(p) >= {"ce_mean", "ce_sum", "implied_total_weight", "aten_total_weight", "aten_error",
                         "grad_sha256"} for p in a)
          and all((p["aten_total_weight"] is None) != (p["aten_error"] is None) for p in a), json.dumps(a[0])[:160])
    b = rec.get("probe_B", {}).get("passes", [{}])
    check("S06 Probe B records ce, dice and loss per pass (loss = ce + dice) and the all-parameter grad hash",
          all(set(p) >= {"ce", "dice", "loss", "grads_sha256"} for p in b)
          and all(abs(p["loss"] - (p["ce"] + p["dice"])) < 1e-4 for p in b), json.dumps(b[0])[:160])
    check("S07 distinct counts are recorded for every quantity", all(
        isinstance(rec.get("probe_A", {}).get("distinct", {}).get(k), int)
        for k in ("ce_mean", "ce_sum", "implied_total_weight", "aten_total_weight", "grad_sha256"))
        and isinstance(rec.get("probe_C", {}).get("distinct_logits"), int)
        and all(isinstance(rec.get("probe_B", {}).get("distinct", {}).get(k), int)
                for k in ("ce", "dice", "loss", "grads_sha256")))
    r2 = probe(["--synthetic", "--device", "cpu", "--repeats", "3", "--out", str(out)])
    check("S08 an existing --out -> exit 2 (out_exists); the record is untouched",
          r2.returncode == 2 and "[out_exists]" in r2.stderr and json.loads(out.read_text()) == rec)
    r3 = probe(["--synthetic", "--device", "cpu", "--out", str(REPO / "n10_should_not_exist.json")])
    check("S09 --out inside the repository -> exit 2 (out_inside_repo); nothing written",
          r3.returncode == 2 and "[out_inside_repo]" in r3.stderr and not (REPO / "n10_should_not_exist.json").exists())
    r4 = probe(["--synthetic", "--device", "cpu", "--out", "relative.json"])
    check("S10 a relative --out -> exit 2 (out_relative)", r4.returncode == 2 and "[out_relative]" in r4.stderr)
    root = TMP / "root"
    root.mkdir()
    r5 = probe(["--synthetic", "--device", "cpu", "--out", str(root / "n10.json")],
               env_extra={"PLANTSEG_DATA_ROOT": str(root)})
    check("S11 --out inside the data root -> exit 2 (out_inside_data_root)",
          r5.returncode == 2 and "[out_inside_data_root]" in r5.stderr)
    r6 = probe(["--synthetic", "--device", "cpu", "--out", str(TMP / "missing_dir" / "n10.json")])
    check("S12 --out in a missing directory -> exit 2 (out_parent_missing)",
          r6.returncode == 2 and "[out_parent_missing]" in r6.stderr)
    r7 = probe(["--device", "cpu", "--out", str(TMP / "n10_nock.json")])
    check("S13 no --checkpoint without --synthetic -> exit 2 (checkpoint_missing)",
          r7.returncode == 2 and "[checkpoint_missing]" in r7.stderr)
    fake_ck = TMP / "fake.pt"
    fake_ck.write_bytes(b"not a checkpoint")
    r8 = probe(["--device", "cpu", "--checkpoint", str(fake_ck), "--out", str(TMP / "n10_sha.json")])
    check("S14 a checkpoint whose sha256 is not the seed-42 one -> exit 2 before loading (checkpoint_sha_mismatch)",
          r8.returncode == 2 and "[checkpoint_sha_mismatch]" in r8.stderr and "UnpicklingError" not in r8.stderr)
    drv = TMP / "cuda_init_driver.py"
    drv.write_text(
        "import sys, torch\n"
        "torch.cuda.is_initialized = lambda: True\n"
        f"sys.argv = ['probe', '--synthetic', '--device', 'cuda', '--out', {str(TMP / 'n10_cuda.json')!r}]\n"
        f"sys.path.insert(0, {str(REPO / 'scripts')!r})\n"
        "import probe_e1_grad_determinism as P\n"
        "raise SystemExit(P.main(sys.argv[1:]))\n", encoding="utf-8")
    r9 = subprocess.run([sys.executable, "-B", str(drv)], cwd=str(REPO), capture_output=True, text=True,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO),
                                 PLANTSEG_DATA_ROOT=str(NO_ROOT)), timeout=600)
    check("S15 CUDA already initialised with --device cuda -> exit 2 (cuda_initialized), before set_seed",
          r9.returncode == 2 and "[cuda_initialized]" in r9.stderr, (r9.stdout + r9.stderr)[-160:])
    r10 = probe(["--synthetic", "--device", "cpu", "--repeats", "1", "--out", str(TMP / "n10_r1.json")])
    check("S16 --repeats 1 -> exit 2 (a determinism record needs >= 2 passes)",
          r10.returncode == 2 and "[repeats_too_few]" in r10.stderr)
    import importlib.util
    spec = importlib.util.spec_from_file_location("probe_under_test", str(PROBE))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    check("S17 Q10 constants: the E1 seed-42 checkpoint sha is the default, num_workers 12, batch size 16 (the "
          "E1 recipe), the batch-identity note says INFERRED, not MEASURED; E1_CODE_STATUS_PATHS is exactly the "
          "five E1-code pathspecs",
          m.E1_SEED42_CHECKPOINT_SHA256 == "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03"
          and m.RECIPE_NUM_WORKERS == 12 and m.BATCH_SIZE == 16 and "INFERRED" in m.BATCH_IDENTITY_NOTE
          and list(m.E1_CODE_STATUS_PATHS) == E1_CODE_PATHSPECS
          and "not MEASURED" in m.BATCH_IDENTITY_NOTE)
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    body = tree.body
    i_dwb = next((i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == "sys.dont_write_bytecode"
                  and isinstance(n.value, ast.Constant) and n.value.value is True), len(body))
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    first_nonstd = next((i for i, n in enumerate(body) if isinstance(n, (ast.Import, ast.ImportFrom))
                         and not all((a.name.split(".")[0] if isinstance(n, ast.Import)
                                      else (n.module or "").split(".")[0]) in stdlib for a in n.names)),
                        len(body))
    first_code = next((i for i, n in enumerate(body) if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.If))),
                      len(body))
    check("S18 X8: sys.dont_write_bytecode = True is set before any non-stdlib import and before the first "
          "function, class or `if` block, i.e. before main() can import torch or src (AST)",
          i_dwb < min(first_nonstd, first_code), f"dwb@{i_dwb} first_nonstdlib@{first_nonstd} first_code@{first_code}")
    ddrv = TMP / "default_root_driver.py"
    droot = TMP / "default_root"
    droot.mkdir()
    ddrv.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import configs.data as CD\n"
        f"CD.DATA['root'] = {str(droot)!r}\n"
        f"sys.path.insert(0, {str(REPO / 'scripts')!r})\n"
        "import probe_e1_grad_determinism as P\n"
        f"raise SystemExit(P.main(['--synthetic', '--device', 'cpu', '--out', {str(droot / 'n10.json')!r}]))\n",
        encoding="utf-8")
    env20 = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
    env20.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    r20 = subprocess.run([sys.executable, "-B", str(ddrv)], cwd=str(REPO), capture_output=True, text=True,
                         env=env20, timeout=600)
    check("S20 with PLANTSEG_DATA_ROOT unset, an --out inside the loader's root (configs.data DATA['root']) -> "
          "exit 2 (out_inside_data_root); nothing written",
          r20.returncode == 2 and "[out_inside_data_root]" in r20.stderr and not any(droot.iterdir()),
          (r20.stdout + r20.stderr)[-160:])
    cwd_root = TMP / "cwd_root"
    cwd_root.mkdir()
    r21 = probe(["--synthetic", "--device", "cpu", "--out", str(cwd_root / "n10.json")],
                env_extra={"PLANTSEG_DATA_ROOT": ""}, cwd=cwd_root)
    check("S21 an empty PLANTSEG_DATA_ROOT is the cwd for the loader (Path('')), so an --out under the cwd -> "
          "exit 2 (out_inside_data_root); nothing written",
          r21.returncode == 2 and "[out_inside_data_root]" in r21.stderr and not any(cwd_root.iterdir()),
          (r21.stdout + r21.stderr)[-160:])
    mods = set(rec.get("repo_modules", []))
    check("S22 modules_under_repo is computed late: on the synthetic run it covers src.training.train_e1 (and the "
          "losses module it imports), src.seeds and configs.data, all under the checkout (the old P0 placement "
          "fails this; F05 covers the real path's later imports)",
          rec.get("modules_under_repo") is True
          and {"src.training.losses", "src.training.train_e1", "src.seeds", "configs.data"} <= mods,
          f"{len(mods)} modules")
    sdrv = TMP / "step_spy_driver.py"
    sdrv.write_text(textwrap.dedent(f'''
        import json, sys
        sys.path.insert(0, {str(REPO)!r})
        import torch
        from src.training import losses as L
        events, crit_calls = [], []
        _ce, _dice, _bw = L.WeightedCrossEntropyLoss.forward, L.SoftDiceLoss.forward, torch.Tensor.backward
        _crit = L.CombinedCEDiceLoss.forward
        def ce_spy(self, *a, **k):
            out = _ce(self, *a, **k); events.append(("ce", out.grad_fn)); return out
        def dice_spy(self, *a, **k):
            out = _dice(self, *a, **k); events.append(("dice", out.grad_fn)); return out
        def bw_spy(self, *a, **k):
            events.append(("backward", self.grad_fn)); return _bw(self, *a, **k)
        def crit_spy(self, *a, **k):
            crit_calls.append(torch.is_grad_enabled()); return _crit(self, *a, **k)
        L.WeightedCrossEntropyLoss.forward, L.SoftDiceLoss.forward, torch.Tensor.backward = ce_spy, dice_spy, bw_spy
        L.CombinedCEDiceLoss.forward = crit_spy
        sys.path.insert(0, {str(REPO / "scripts")!r})
        import probe_e1_grad_determinism as P
        rc = P.main(["--synthetic", "--device", "cpu", "--repeats", "2", "--out", {str(TMP / "n10_spy.json")!r}])
        live = [e for e in events if e[1] is not None]          # grad-enabled ce/dice outputs and backward calls
        ok, n_bw, last = [], 0, {{}}
        for kind, fn in live:
            if kind == "backward":
                n_bw += 1
                nxt = [f for f, _ in fn.next_functions]
                ok.append(fn.name() == "AddBackward0" and len(nxt) == 2
                          and nxt[0] is last.get("ce") and nxt[1] is last.get("dice"))
            else:
                last[kind] = fn
        print("SPY_JSON:" + json.dumps({{"rc": rc, "n_backward": n_bw, "ok": ok,
                                        "live_kinds": [k for k, _ in live],
                                        "criterion_calls_grad_enabled": crit_calls}}))
    '''), encoding="utf-8")
    r23 = subprocess.run([sys.executable, "-B", str(sdrv)], cwd=str(REPO), capture_output=True, text=True,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO),
                                  PLANTSEG_DATA_ROOT=str(NO_ROOT)), timeout=600)
    spy = next((json.loads(ln[9:]) for ln in r23.stdout.splitlines() if ln.startswith("SPY_JSON:")), {})
    rec23 = json.loads((TMP / "n10_spy.json").read_text(encoding="utf-8")) if (TMP / "n10_spy.json").exists() else {}
    b23 = rec23.get("probe_B", {}).get("passes", [])
    check("S23 F4 spy: per pass exactly one grad-enabled criterion.ce, one criterion.dice and one backward, in "
          "that order, and the backward is on AddBackward0 of that pass's ce and dice outputs (E1's step, not "
          "recomputed); criterion(logits, mask) is only called under no_grad and equals ce + dice bitwise, as do "
          "the no_grad re-runs (CPU)",
          spy.get("rc") == 0 and spy.get("n_backward") == 2 and spy.get("ok") == [True, True]
          and spy.get("live_kinds") == ["ce", "dice", "backward"] * 2
          and spy.get("criterion_calls_grad_enabled") == [False, False] and len(b23) == 2
          and all(p["criterion_equals_ce_plus_dice"] is True and p["criterion_forward"] == p["loss"]
                  and p["rerun_ce_equals_ce"] is True and p["rerun_dice_equals_dice"] is True
                  and p["rerun_ce_plus_dice_equals_loss"] is True for p in b23)
          and rec23.get("probe_B", {}).get("ce_dice_source") == "the E1 step's own terms (not recomputed)",
          json.dumps(spy) if spy else (r23.stdout + r23.stderr)[-200:])
    e1_src = (REPO / "src" / "training" / "train_e1.py").read_text(encoding="utf-8")
    e1 = e1_src.splitlines()
    cited = [e1[i - 1].strip() for i in (437, 438, 439, 451)] if len(e1) >= 451 else []

    def _is_loss(t):
        return (isinstance(t, ast.Name) and t.id == "loss") or (
            isinstance(t, (ast.Tuple, ast.List)) and any(_is_loss(x) for x in t.elts))
    rebinds = []
    for node in ast.walk(ast.parse(e1_src)):
        if not 440 <= getattr(node, "lineno", 0) <= 450:
            continue
        if isinstance(node, ast.Assign) and any(_is_loss(t) for t in node.targets) \
                or isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)) and _is_loss(node.target) \
                or isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr.endswith("_") and _is_loss(node.func.value):
            rebinds.append(node.lineno)
    check("S24 the probe's citation holds: train_e1.py:437-439 are exactly ce = criterion.ce(logits, mask), dice = "
          "criterion.dice(logits, mask), loss = ce + dice (code part), :451 is loss.backward(), and :440-450 never "
          "rebind or modify loss in place (AST: assign/augassign/annassign/walrus/tuple targets, loss.<op>_() calls)",
          m.E1_STEP_LOSS_LINES == "src/training/train_e1.py:437-439"
          and m.E1_STEP_BACKWARD_LINE == "src/training/train_e1.py:451"
          and cited[:2] == ["ce = criterion.ce(logits, mask)", "dice = criterion.dice(logits, mask)"]
          and cited[2].split("#", 1)[0].strip() == "loss = ce + dice" and cited[3] == "loss.backward()"
          and rebinds == [], f"{cited} rebinds@{rebinds}")


def section_full():
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPO))
    from src.eval.model_loading import build_fp32_student, sha256_file
    root = TMP / "full_root"
    rng = np.random.RandomState(42)
    for split, n in (("train", 5367), ("val", 846)):
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i in range(n):
            h, w = 24 + (i % 5) * 4, 32 + (i % 7) * 4
            Image.fromarray(rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)).save(
                root / "images" / split / f"syn_{split}_{i:05d}.jpg")
            m = np.zeros((h, w), dtype=np.uint8)
            m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1 + (i % 115)
            Image.fromarray(m).save(root / "annotations" / split / f"syn_{split}_{i:05d}.png")
    torch.manual_seed(0)
    ck = TMP / "synthetic_e1.pt"
    torch.save({"iter": 4000, "model_state_dict": build_fp32_student().state_dict(), "num_classes": 116,
                "best_val_miou_all_class": 0.123}, ck)
    sha = sha256_file(ck)
    out = TMP / "n10_full.json"
    # A CPU train step on the recipe's 16 x 3 x 512 x 512 batch needs more memory than a CPU container has
    # (MEASURED: OOM-killed at 7.65 GB of 7.9 GB in the pinned image), so this driver runs the unchanged
    # probe with its module batch size set to 3 (a value used nowhere else, so F01's shape proves the loader
    # call reads BATCH_SIZE). S17 asserts the probe's own value (16).
    drv = TMP / "full_driver.py"
    drv.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'scripts')!r})\n"
        "import probe_e1_grad_determinism as P\n"
        "P.BATCH_SIZE = 3\n"
        "raise SystemExit(P.main(sys.argv[1:]))\n", encoding="utf-8")
    r = probe(["--device", "cpu", "--checkpoint", str(ck), "--expect-sha256", sha, "--repeats", "2",
               "--out", str(out)], env_extra={"PLANTSEG_DATA_ROOT": str(root)}, script=drv)
    rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    print(r.stdout[-2000:])
    bt = rec.get("batch", {})
    check("F01 real path: checkpoint weights, the first TRAIN batch (batch 3 via the smoke driver) from a "
          "5367/846 root with 12 workers; exit 0",
          r.returncode == 0 and rec.get("weights", {}).get("sha256") == sha
          and rec["weights"].get("is_e1_seed42") is False and bt.get("kind") == "first TRAIN batch"
          and bt.get("num_workers") == 12 and bt.get("shape") == [3, 3, 512, 512]
          and bt.get("isolation_counts") == {"train": {"images": 5367, "masks": 5367},
                                             "val": {"images": 846, "masks": 846}},
          (r.stdout + r.stderr)[-240:])
    a = rec.get("probe_A", {}).get("passes", [])
    check("F02 the ATen normaliser is available here and equals the float64 reference total_weight within "
          "float32 accumulation error (relative 1e-5)",
          bool(a) and all(p["aten_total_weight"] is not None for p in a)
          and abs(a[0]["aten_total_weight"] - bt.get("total_weight_float64", -1)) <= 1e-5 * max(1.0, bt.get(
              "total_weight_float64", 1)), f"{a[0].get('aten_total_weight') if a else None} vs "
                                             f"{bt.get('total_weight_float64')}")
    b = rec.get("probe_B", {}).get("passes", [])
    check("F04 real path: in every Probe B pass criterion(logits, mask) and the no_grad re-run equal the "
          "backpropagated ce + dice bitwise (recorded, CPU)",
          bool(b) and all(p["criterion_equals_ce_plus_dice"] is True and p["criterion_forward"] == p["loss"]
                          and p["rerun_ce_plus_dice_equals_loss"] is True for p in b),
          json.dumps([(p.get("loss"), p.get("criterion_forward")) for p in b])[:160])
    check("F05 real path: modules_under_repo covers the real path's later repo imports (src.eval.model_loading, "
          "src.data, src.data.isolation, configs.data)",
          rec.get("modules_under_repo") is True
          and {"src.eval.model_loading", "src.data", "src.data.isolation", "configs.data"}
          <= set(rec.get("repo_modules", [])), f"{len(rec.get('repo_modules', []))} modules")
    (root / "images" / "test").mkdir()
    r2 = probe(["--device", "cpu", "--checkpoint", str(ck), "--expect-sha256", sha, "--repeats", "2",
                "--out", str(TMP / "n10_test.json")], env_extra={"PLANTSEG_DATA_ROOT": str(root)})
    check("F03 a TEST surface in the root -> exit 2 [test_split_present]; no record",
          r2.returncode == 2 and "[test_split_present]" in r2.stderr and not (TMP / "n10_test.json").exists(),
          r2.stderr[-160:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    try:
        snap0 = snapshot()
        for label, fn in (("default", section_default),) + ((("full", section_full),) if args.full else ()):
            try:
                fn()
            except Exception as e:                        # noqa: BLE001 -- a section crash is a FAIL
                check(f"{label} section completed", False,
                      f"{type(e).__name__}: {e} | {traceback.format_exc(limit=4)[-300:]}")
        snap1 = snapshot()
        check("S19 nothing was written into the checkout by any section (code dirs' status incl. "
              "untracked/ignored, and the top-level names, unchanged)",
              snap0 is not None and snap0 == snap1, "REPO is not the top of a git work tree" if snap0 is None else "")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    width = max(len(n) for n, _, _ in CHECKS)
    for name, verdict, detail in CHECKS:
        print(f"  [{verdict}] {name:<{width}}  {detail}")
    n_pass = sum(1 for _, v, _ in CHECKS if v == "PASS")
    print(f"SUMMARY  {n_pass}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'PASS' if n_pass == len(CHECKS) else 'FAIL'} ({n_pass}/{len(CHECKS)})")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
