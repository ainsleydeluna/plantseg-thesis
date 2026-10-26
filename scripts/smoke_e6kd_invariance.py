#!/usr/bin/env python3
"""Lane L-AM3, B-2: E5's and E6's trainer paths are unchanged by B-2 (LB38's Q9; LB33, LB34). CPU only: the claims are
bitwise on the CPU.

The two commits are O (the lane's base, --old) and HEAD (--new). Each is exported with `git archive` of explicit paths
(src, configs, scripts, reports/e1_class_weights.json) in archive_old's form (scripts/smoke_arms_invariance.py
archive_old): the SL-1 exclude is in the pathspec and every member name is string-checked before anything is
extracted; a member whose name holds the token SL-1 forbids is never listed beyond that check, extracted, written or
hashed. Each run is a child process whose imports all come from its own export (checked module by module): the QAT
trainer of record (src/quant/qat.py run_qat, mode "smoke") on Q2's synthetic loaders, 6 steps (3 epochs: three VALs and
three epoch checkpoints), with hooks recording the state after every step.

Checks: the exports differ in the B-2 files (the guard); E5 and E6 give identical rows, states and checkpoints at O and
HEAD (and identical VAL rows for E6); their run_meta rows differ only in the declared keys; E6-KD at its real weights
(HEAD only) differs from E6 (the negative control).

    python -B scripts/smoke_e6kd_invariance.py [--old COMMIT] [--new COMMIT]

Ends with one RESULT line; exit 0 only when every check passes, 2 when a commit cannot be exported.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# standard-library-only imports: the children import the trainer from their own export
from scripts.invariance_harness import EXPORT_EXCLUDE, export_commit  # noqa: E402
from scripts.synthetic_ptq_fixtures import safe_tmpdir  # noqa: E402

OLD_DEFAULT = "3373139179e6e27ecbe25f280be7b2b7889c38f2"   # O: master at B-2's STEP 1 (M2, the merge of lane/am23-quant)
SL1_EXCLUDE = ":(exclude,icase)*" + "".join(("t", "e", "s", "t")) + "*"
EXPORT_PATHS = ("src", "configs", "scripts", "reports/e1_class_weights.json")
B2_FILES = ("src/quant/stages.py", "src/quant/runner.py", "src/quant/qat_select.py", "src/eval/artifacts.py",
            "src/eval/stage_artifacts.py")
NEW_ONLY = ("src/quant/qat_kd.py", "scripts/run_e6kd.py", "scripts/e6kd_eval.py", "configs/e6kd_selection_rules.json")
# QAT_CODE_PATHS' files that B-2 edits: their sha256 in run_meta's code_files_sha256 is all that may differ
DECLARED_CODE_FILES = ("src/quant/qat_select.py", "src/quant/runner.py", "src/quant/stages.py")
DECLARED_META_KEYS = ("wall_clock", "out_dir", "code_files_sha256")
VOLATILE = ("wall_clock", "iter_seconds", "val_seconds")
STEPS = 6
results: list[tuple[str, bool, str]] = []

CHILD = r'''
import hashlib, json, os, sys
code_root, kind, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, code_root)
os.environ["PLANTSEG_DATA_ROOT"] = os.path.join(os.path.dirname(out), "absent_data_root")
os.environ.pop("PLANTSEG_IMAGE_DIGEST", None)
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.student import build_student
from src.quant import qat as Q

def loaders(seed=42):
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(32, 3, 64, 64, generator=g)
    y = torch.randint(0, 116, (32, 64, 64), generator=g)
    y[:, :4, :] = 255
    vx = torch.randn(8, 3, 64, 64, generator=g)
    vy = torch.randint(0, 116, (8, 64, 64), generator=g)
    train = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(seed))
    return train, DataLoader(TensorDataset(vx, vy), batch_size=16, shuffle=False)

class Rec(Q.QATHooks):
    def __init__(self):
        self.states = []
    def after_step(self, step, prepared, row):
        self.states.append(Q.state_digest(prepared))

torch.manual_seed(0)
student = build_student(pretrained=False)
rec = Rec()
src = {"sha256": "0" * 64, "path": None, "bytes": None}
if kind in ("e5", "e6"):
    Q.run_qat(stage=kind, mode="smoke", model=student, source_meta=src, out_dir=out, seed=42, clip_norm=1.0,
              clip_source="smoke", device="cpu", num_workers=0, loaders=loaders(), hooks=rec, max_steps=STEPS,
              log=lambda *a: None)
else:
    import torch.nn as nn
    from scripts.invariance_harness import build_stub_segnext
    from src.distill.segnext_teacher import SegNeXtTeacherAdapter
    from src.distill.teacher import FrozenTeacher
    from src.quant import qat_kd as KD
    g = torch.Generator().manual_seed(7)
    proj = torch.nn.utils.skip_init(nn.Conv2d, 160, 320, 1, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.randn(320, 160, 1, 1, generator=g) * 0.05)
    KD.run_qat_kd(mode="smoke", model=student, teacher=FrozenTeacher(SegNeXtTeacherAdapter(build_stub_segnext())),
                  projection=proj, weights={"logit_kd": 0.25, "cwd_feat": 12.5, "cwd_logit": 1.5},
                  e3={"e3_weights": {"logit_kd": 0.5, "cwd_feat": 25.0, "cwd_logit": 3.0}, "grids": {}},
                  source_meta=src, out_dir=out, seed=42, clip_norm=1.0, clip_source="smoke", device="cpu",
                  num_workers=0, loaders=loaders(), hooks=rec, max_steps=STEPS, log=lambda *a: None)
root = os.path.realpath(code_root)
outside = sorted(n for n, m in list(sys.modules.items())
                 if n.split(".")[0] in ("src", "configs", "scripts") and getattr(m, "__file__", None)
                 and not os.path.realpath(m.__file__).startswith(root + os.sep))
print("CHILD_JSON:" + json.dumps({"states": rec.states, "outside": outside}))
'''.replace("STEPS", str(STEPS))


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def _clean(name: str) -> bool:
    return "".join(("t", "e", "s", "t")) not in name.lower()


class RefusedExport(RuntimeError):
    """A commit that cannot be exported under the SL-1 exclude (exit 2, nothing written)."""


def archive_checked(commit: str) -> bytes:
    """`git archive` of `commit` with the SL-1 exclude (archive_old's form); nothing is written. A failed archive, or a
    member name that still holds the forbidden token (each name compared as a string), is a refusal."""
    proc = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit, "--", *EXPORT_PATHS, SL1_EXCLUDE,
                           EXPORT_EXCLUDE], capture_output=True)
    if proc.returncode != 0:
        raise RefusedExport(f"git archive of {commit} failed: {proc.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
        named = sum(1 for n in tf.getnames() if not _clean(n))
    if named:
        raise RefusedExport(f"git archive of {commit} kept {named} member(s) whose name holds the forbidden token")
    return proc.stdout


def child(work: Path, code_root: Path, kind: str, tag: str) -> dict:
    out = work / tag / "run"
    out.parent.mkdir(parents=True)
    script = work / "invariance_child.py"
    if not script.is_file():
        script.write_text(CHILD, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PLANTSEG_IMAGE_DIGEST")}
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "CUDA_VISIBLE_DEVICES": ""})
    p = subprocess.run([sys.executable, "-B", str(script), str(code_root), kind, str(out)], capture_output=True,
                       text=True, env=env, cwd=str(out.parent), timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("CHILD_JSON:")]
    doc = json.loads(lines[-1][len("CHILD_JSON:"):]) if lines else {"states": [], "outside": ["no child json"]}
    tel = out / "qat_telemetry.jsonl"
    rows = [json.loads(ln) for ln in tel.read_text(encoding="utf-8").splitlines()] if tel.is_file() else []
    cks = {f"e{e:02d}": _sha256(out / "epoch_ckpts" / f"e{e:02d}.pt") for e in (1, 2, 3)}
    return {"rc": p.returncode, "err": p.stderr[-400:], "rows": rows, "states": doc["states"],
            "outside": doc["outside"], "checkpoints": cks}


def _sha256(p: Path) -> str | None:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def strip(rows: list[dict], event: str) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in rows if r.get("event") == event]


def meta_diff(a: dict, b: dict) -> tuple[list[str], list[str]]:
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    ca, cb = a.get("code_files_sha256") or {}, b.get("code_files_sha256") or {}
    files = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
    return keys, files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", default=OLD_DEFAULT, help="the lane's base commit O")
    ap.add_argument("--new", default="HEAD", help="the commit under check (HEAD)")
    a = ap.parse_args()
    t0 = time.time()
    commits = {}
    try:
        for side, ref in (("old", a.old), ("new", a.new)):
            commits[side] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", f"{ref}^{{commit}}"],
                                           capture_output=True, text=True, check=True).stdout.strip()
            archive_checked(commits[side])
    except subprocess.CalledProcessError as e:
        print(f"RESULT: REFUSED -- a commit does not resolve in {REPO}: {e}")
        return 2
    except RefusedExport as e:
        print(f"RESULT: REFUSED -- {e}")
        return 2
    work = safe_tmpdir("smoke_e6kd_invariance_")            # after the checks: a refusal writes nothing (DL-24 SM7)
    roots = {side: work / f"export_{side}" for side in commits}
    for side, root in roots.items():
        export_commit(REPO, commits[side], root, paths=(*EXPORT_PATHS, SL1_EXCLUDE))
    print("=" * 78)
    print(f"E6-KD INVARIANCE SMOKE (lane L-AM3, B-2) -- CPU, Q2's synthetic loaders, {STEPS} steps")
    print(f"OLD {commits['old']} | NEW {commits['new']} | work {work}")
    print("=" * 78, flush=True)

    differ = [rel for rel in B2_FILES
              if (roots["old"] / rel).read_bytes() != (roots["new"] / rel).read_bytes()]
    new_only = [rel for rel in NEW_ONLY if (roots["new"] / rel).is_file() and not (roots["old"] / rel).exists()]
    check("guard_exports_differ_in_the_b2_files", differ == list(B2_FILES) and new_only == list(NEW_ONLY),
          f"differ {differ}; new only {new_only}")

    runs = {}
    for side in ("old", "new"):
        for kind in ("e5", "e6"):
            runs[(side, kind)] = child(work, roots[side], kind, f"{side}_{kind}")
    runs[("new", "kd")] = child(work, roots["new"], "kd", "new_kd")
    ran = all(r["rc"] == 0 and not r["outside"] and len(r["states"]) == STEPS for r in runs.values())
    detail = {f"{s}_{k}": (r["rc"], r["outside"][:2], r["err"][-120:] if r["rc"] else "") for (s, k), r in runs.items()
              if r["rc"] or r["outside"]}
    for kind in ("e5", "e6"):
        o, n = runs[("old", kind)], runs[("new", kind)]
        check(f"{kind}_rows_identical", ran and strip(o["rows"], "train") == strip(n["rows"], "train")
              and len(strip(n["rows"], "train")) == STEPS, f"{detail}")
        check(f"{kind}_states_identical", ran and o["states"] == n["states"], f"{detail}")
        check(f"{kind}_checkpoints_identical", ran and o["checkpoints"] == n["checkpoints"]
              and all(o["checkpoints"].values()), f"{o['checkpoints']} {n['checkpoints']}")
    o6, n6 = runs[("old", "e6")], runs[("new", "e6")]
    check("e6_val_identical", ran and strip(o6["rows"], "val") == strip(n6["rows"], "val")
          and len(strip(n6["rows"], "val")) == 3, f"{detail}")
    bad = {}
    for kind in ("e5", "e6"):
        o, n = runs[("old", kind)], runs[("new", kind)]
        keys, files = meta_diff(o["rows"][0] if o["rows"] else {}, n["rows"][0] if n["rows"] else {})
        if not set(keys) <= set(DECLARED_META_KEYS) or files != sorted(DECLARED_CODE_FILES):
            bad[kind] = (keys, files)
    check("run_meta_differs_only_in_declared_keys", ran and not bad, f"{bad}")
    kd = runs[("new", "kd")]
    e6_loss = [r["loss"] for r in n6["rows"] if r.get("event") == "train"]
    kd_loss = [r["loss"] for r in kd["rows"] if r.get("event") == "train"]
    # the ramp is 0 at step 1, so step 1 is E6's; from step 2 the distillation terms move the loss and the state
    check("negative_control_e6kd_differs_from_e6", ran and len(kd_loss) == STEPS and e6_loss[:1] == kd_loss[:1]
          and all(x != y for x, y in zip(e6_loss[1:], kd_loss[1:])) and kd["states"][1:] != n6["states"][1:],
          f"E6 {e6_loss[:3]} E6-KD {kd_loss[:3]}")
    names = [n for n, _ok, _d in results]
    token_free = all(_clean(n) for n in names)
    created = [os.path.join(d, n) for d, ds, fs in os.walk(work) for n in ds + fs if not _clean(n) or ":" in n]
    check("no_forbidden_named_path_created", token_free and not created, f"{created[:2]}")
    print("\n[CHECKS]")
    for name, ok, det in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + det[:400]) if det and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n[time] {time.time() - t0:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
