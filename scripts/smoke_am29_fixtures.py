"""Synthetic fixtures shared by the L-AM29 smokes (scripts/smoke_am29_*.py). No PlantSeg data, no checkpoint of record.

Importing this module imports no repository code, so a smoke sets PLANTSEG_DATA_ROOT before configs/data.py is first
imported. Every file goes to a temp directory outside the repository whose path names neither TEST nor VAL (its VAL
tree is a child named images/val, as the evaluator reads it).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NC = 116
E3_RUN_META_BASE = {"event": "run_meta", "stage": "E3", "mode": "real", "lambda_logit": 0.5, "alpha_cwd": 25.0}


class Checks:
    """A smoke's checks: name, pass, detail; printed one per line, then one RESULT line."""

    def __init__(self, title: str):
        self.title, self.rows = title, []

    def __call__(self, name: str, ok, detail="") -> bool:
        self.rows.append((name, bool(ok), str(detail)[:600]))
        return bool(ok)

    def report(self) -> int:
        w = max((len(n) for n, _, _ in self.rows), default=10)
        for n, ok, d in self.rows:
            print(f"  {n:<{w}} : {'PASS' if ok else 'FAIL'}  {d}")
        bad = [n for n, ok, _ in self.rows if not ok]
        print(f"\nRESULT: {'PASS' if not bad else 'FAIL'} ({len(self.rows) - len(bad)}/{len(self.rows)}) {self.title}")
        return 0 if not bad else 1


def tmpdir(prefix: str) -> Path:
    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    return safe_tmpdir(prefix)


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p: Path, doc) -> str:
    Path(p).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return sha_file(p)


def git(repo: Path, *args) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@localhost",
           "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@localhost", "GIT_CONFIG_NOSYSTEM": "1"}
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"git {args}: {p.stderr}")
    return p.stdout


def loaders(seed: int = 42, n_train: int = 32, size: int = 64):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(n_train, 3, size, size, generator=g)
    y = torch.randint(0, NC, (n_train, size, size), generator=g)
    y[:, :4, :] = 255
    vx = torch.randn(8, 3, size, size, generator=g)
    vy = torch.randint(0, NC, (8, size, size), generator=g)
    train = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(vx, vy), batch_size=16, shuffle=False)
    return train, val


def student(seed: int = 0):
    import torch
    from src.models.student import build_student
    torch.manual_seed(seed)
    return build_student(pretrained=False)


def stub_teacher():
    """The E6-KD smokes' teacher: the invariance harness's stub SegNeXt in the thesis adapter, frozen."""
    from scripts.invariance_harness import build_stub_segnext
    from src.distill import teacher as TT
    from src.distill.segnext_teacher import SegNeXtTeacherAdapter
    return TT.FrozenTeacher(SegNeXtTeacherAdapter(build_stub_segnext()))


def projection(seed: int = 7):
    import torch
    import torch.nn as nn
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.utils.skip_init(nn.Conv2d, 160, 320, 1, bias=False)
    with torch.no_grad():
        p.weight.copy_(torch.randn(320, 160, 1, 1, generator=g) * 0.05)
    return p


def write_e3_parent(d: Path, *, seed: int, ckpt_name: str, it: int, arm: str | None = None,
                    proj_ckpt_name: str | None = None, proj_it: int | None = None, student_seed: int = 0,
                    with_projection: bool = True) -> tuple[str, str | None]:
    """An E3 (or E3-R2 when `arm` is given) run directory as train_distill keeps it: the best checkpoint (stage E3),
    best.json, e3_run_meta.jsonl with E3's terms and weights, e3_telemetry.jsonl with a run_end record, and
    projection.pt saved by save_projection (its checkpoint name and iter default to the checkpoint's)."""
    import torch

    from src.distill.cwd_projection import PROJECTION_FILE, PROJECTION_GROUP, save_projection, split_optimizer_state
    from src.quant import qat_kd as KD
    from src.training import train_distill as TD
    d.mkdir(parents=True)
    net = student(student_seed)
    payload = {"stage": "E3", "iter": it, "num_classes": NC, "model_state_dict": net.state_dict()}
    if arm is not None:
        payload["arm"] = arm
    torch.save(payload, d / ckpt_name)
    write_json(d / "best.json", {"best_ckpt": f"/workspace/kd/{d.name}/{ckpt_name}", "best_val_miou_all_class": 0.37})
    meta = {**E3_RUN_META_BASE, "seed": seed, "beta_cwd": TD.BETA_CWD_LOGIT, "terms": {t: True for t in KD.TERMS},
            "T_logit": TD.T_LOGIT, "T_cwd": TD.T_CWD, "cwd_C": TD.CWD_C_FEAT, "logit_kd_semantics": KD.LOGIT_KD_SEMANTICS,
            "ramp_iters": 335}
    if arm is not None:
        meta["arm"] = arm
    (d / "e3_run_meta.jsonl").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    (d / "e3_telemetry.jsonl").write_text(json.dumps({"event": "run_end", "iter": 80000, "checks_passed": True}) + "\n",
                                          encoding="utf-8")
    if not with_projection:
        return sha_file(d / ckpt_name), None
    proj = projection(11 + seed)
    opt = torch.optim.SGD([{"params": list(net.parameters()), "name": "student"},
                           {"params": list(proj.parameters()), "name": PROJECTION_GROUP}], lr=0.01, momentum=0.9)
    for prm in list(net.parameters())[:1] + list(proj.parameters()):
        prm.grad = torch.ones_like(prm)
    opt.step()
    _rest, part = split_optimizer_state(opt.state_dict())
    save_projection(d / PROJECTION_FILE, proj, part, stage="E3", it=proj_it if proj_it is not None else it,
                    checkpoint=proj_ckpt_name or ckpt_name)
    return sha_file(d / ckpt_name), sha_file(d / PROJECTION_FILE)


def selections_repo(root: Path, *, lam: float = 0.5, alpha: float = 25.0, alpha_run_id: str = "e3_s42_x",
                    alpha_ckpt_sha256: str = "0" * 64) -> dict:
    """A git repository holding the lambda and alpha selection files, committed (read_tracked_selection's repo_root)."""
    from src.quant import qat as Q
    root.mkdir(parents=True)
    (root / "sel").mkdir()
    lsha = write_json(root / "sel" / "lambda.json", {"format": Q.LAMBDA_SELECTION_FORMAT,
                                                     "winner": {"lambda": lam, "run_id": "e3_s42_lambda"}})
    asha = write_json(root / "sel" / "alpha.json", {"format": Q.ALPHA_SELECTION_FORMAT,
                                                    "winner": {"alpha": alpha, "run_id": alpha_run_id,
                                                               "ckpt_sha256": alpha_ckpt_sha256}})
    git(root, "init", "-q")
    git(root, "add", "--", "sel/lambda.json", "sel/alpha.json")
    git(root, "commit", "-q", "-m", "selections")
    return {"repo": root, "lambda": "sel/lambda.json", "lambda_sha256": lsha, "alpha": "sel/alpha.json",
            "alpha_sha256": asha}


def make_val_tree(root: Path, n_val: int = 846) -> dict:
    from scripts.synthetic_ptq_fixtures import make_tree
    return make_tree(root, n_train=0, n_val=n_val, seed=3)


def rows(run_dir: Path, event: str | None = None) -> list[dict]:
    out = [json.loads(ln) for ln in (Path(run_dir) / "qat_telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
    return [r for r in out if event is None or r.get("event") == event]


def quiet(*_a, **_k) -> None:
    return None
