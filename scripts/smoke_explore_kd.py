#!/usr/bin/env python3
"""Smoke for N3, A1'' and A2, the exploratory KD entry (src/explore/dist_loss.py, src/explore/kd_explore.py, scripts/explore_kd.py).
Synthetic tensors, a synthetic loader and MockTeacher; no dataset, no teacher stack.

  * DIST against the official dist_kd.py formulas, transcribed independently below (_ref_*: (B*W*H, C) rows via
    transpose(1, 3), softmax at tau 1, Pearson as centred cosine with eps 1e-8, beta = gamma = 1): equal on random
    tensors, term by term; the masked version equals the reference on the valid rows; symmetric; zero for identical
    logits; the A2 term is ramp * lambda * DIST with parts["logit_kd"], and refuses stage e3, a projection, targets;
  * A1'': the substituted gate admits exactly the N4 alpha (also with allow_offgrid true, as run() calls it), refuses
    any other off-grid value, and the pinned gate is restored; N4's record is refused unless its telemetry is E3
    seed 42's completion sha256 in the TEST-day manifest;
  * the futility rule: more than 2.0 pp below stops (Decimal, strict), exactly 2.0 pp or less does not, a
    non-finite VAL is not futility; the wrapper writes the VAL row, then x_futility_stop, then raises;
  * the PINNED trainer, train_distill.main in --dry-run on the CPU with a synthetic loader (train_distill's
    build_dataloader substituted for the smoke only) and MockTeacher: A2's DIST term runs inside it (RESULT PASS,
    the logit_kd column holds DIST, unlike a KL run at the same seed); A1'''s off-grid alpha is refused without the
    substitution and accepted with it (run_meta alpha_offgrid true); the futility wrapper stops it after the VAL row,
    with no run_end;
  * the CLI refuses TEST, a non-x- run ID, N3 with the teacher of record's sha256, and a data root that fails the
    TRAIN/VAL-only check, writing nothing; the KD pin's reused-module check runs.
Mutations: DIST's intra term dropped, the futility margin ignored, and a gate that admits every alpha each fail a case.

    python -I scripts/smoke_explore_kd.py        (exit 0 PASS, 1 FAIL)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402
from torch.utils.data import DataLoader, TensorDataset              # noqa: E402

from src.explore import dist_loss as DL                              # noqa: E402
from src.explore import kd_explore as K                              # noqa: E402
from src.explore.patching import substituted                         # noqa: E402
from src.explore.smokekit import Cases, mutation_killed, patched, scratch_dir  # noqa: E402


# ---- the official formulas, transcribed independently (hunto/DIST_KD segmentation/losses/dist_kd.py, per the audit)
def _ref_cos(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def _ref_pearson(a, b, eps=1e-8):
    return _ref_cos(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


def _ref_inter(y_s, y_t):
    return 1 - _ref_pearson(y_s, y_t).mean()


def _ref_intra(y_s, y_t):
    return _ref_inter(y_s.transpose(0, 1), y_t.transpose(0, 1))


def _ref_dist(z_s, z_t, beta=1.0, gamma=1.0, tau=1.0):
    if z_s.ndim == 4:
        c = z_s.shape[1]
        z_s = z_s.transpose(1, 3).reshape(-1, c)
        z_t = z_t.transpose(1, 3).reshape(-1, c)
    y_s, y_t = (z_s / tau).softmax(dim=1), (z_t / tau).softmax(dim=1)
    inter, intra = tau ** 2 * _ref_inter(y_s, y_t), tau ** 2 * _ref_intra(y_s, y_t)
    return beta * inter + gamma * intra, inter, intra


def synthetic_loader_factory(n_train=6, n_val=4, size=512):
    g = torch.Generator().manual_seed(0)
    mk = lambda n: TensorDataset(torch.randn(n, 3, size, size, generator=g),                      # noqa: E731
                                 torch.randint(0, 116, (n, size, size), generator=g))

    def build_dataloader(split, batch_size, num_workers=0, persistent_workers=False, seed=0, **kw):
        ds = mk(n_train if split == "train" else n_val)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=(split == "train"))
    return build_dataloader


def dry_argv(stage, ckpt_dir, *extra):
    return ["--stage", stage, "--dry-run", "--ckpt-dir", str(ckpt_dir), "--max-iters", "4", "--val-interval", "2",
            "--max-val-batches", "2", "--batch-size", "2", "--num-workers", "0", "--log-every", "100", *extra]


def rows_of(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]


def run_cases(verbose: bool = False, full: bool = True) -> Cases:
    c = Cases("SMOKE_EXPLORE_KD", verbose=verbose)
    base = scratch_dir("x_kd_")
    g = torch.Generator().manual_seed(1)
    zs, zt = torch.randn(2, 116, 8, 8, generator=g), torch.randn(2, 116, 8, 8, generator=g) * 2.0
    mine, comps = DL.dist_loss(zs, zt)
    ref, ri, rj = _ref_dist(zs, zt)
    c.check("DIST equals the official formulas on random tensors", torch.allclose(mine, ref, rtol=1e-6, atol=1e-7),
            f"{float(mine)!r} vs {float(ref)!r}")
    c.check("DIST inter and intra terms equal the official ones", abs(comps["inter"] - float(ri)) < 1e-6
            and abs(comps["intra"] - float(rj)) < 1e-6)
    valid = torch.rand(2, 8, 8, generator=g) > 0.3
    mm, _ = DL.dist_loss(zs, zt, valid)
    sel = valid.reshape(-1)
    rs = zs.permute(0, 2, 3, 1).reshape(-1, 116)[sel]
    rt = zt.permute(0, 2, 3, 1).reshape(-1, 116)[sel]
    rm, _, _ = _ref_dist(rs, rt)
    c.check("masked DIST equals the official formulas on the valid rows", torch.allclose(mm, rm, rtol=1e-6, atol=1e-7))
    sym, _ = DL.dist_loss(zt, zs)
    c.check("DIST is symmetric", torch.allclose(mine, sym, rtol=1e-6, atol=1e-7))
    same, _ = DL.dist_loss(zs, zs)
    ref_same = float(_ref_dist(zs, zs)[0])
    c.check("DIST is zero for identical logits up to the official Pearson's eps (bound 1e-4; equal to the reference)",
            float(same) < 1e-4 and abs(float(same) - ref_same) < 1e-9, f"{float(same)!r} (reference {ref_same!r})")
    from src.distill.teacher import TeacherOutput
    from src.training import train_distill as TD
    from src.training.losses import downsample_validity
    mask = torch.randint(0, 116, (2, 64, 64), generator=g)
    mask[:, :8] = 255
    tout = TeacherOutput(logits=zt, feat_s16=None)
    tot, parts = DL.dist_distillation_losses(stage=TD.resolve_stage("e2"), logits=zs, head_logits=zs, c5=None,
                                             mask=mask, teacher_out=tout, projection=None, lambda_logit=1.0,
                                             ramp={"logit_kd": 0.25})
    want, _ = DL.dist_loss(zs, zt, downsample_validity(mask, (8, 8)))
    c.check("A2 term = ramp * lambda * DIST on E2's OS8 validity mask", torch.allclose(tot, 0.25 * want)
            and set(parts) == {"logit_kd"} and abs(parts["logit_kd"] - float(want)) < 1e-7)
    for case, kw in (("stage e3", {"stage": TD.resolve_stage("e3")}), ("a projection", {"projection": torch.nn.Conv2d(1, 1, 1)}),
                     ("arm targets", {"targets": object()})):
        try:
            DL.dist_distillation_losses(**{"stage": TD.resolve_stage("e2"), "logits": zs, "head_logits": zs, "c5": None,
                                           "mask": mask, "teacher_out": tout, "projection": None, "lambda_logit": 1.0,
                                           **kw})
            c.check(f"A2 term refuses {case}", False)
        except RuntimeError:
            c.check(f"A2 term refuses {case}", True)
    # ---- A1''
    a_val = 3.2171
    orig = TD.alpha_gate_error
    with substituted(TD, "alpha_gate_error", K.alpha_gate_admitting(orig, a_val), item="A1pp", why="smoke"):
        c.check("A1'' gate admits the N4 alpha", TD.alpha_gate_error(a_val, allow_offgrid=False, mode="real") is None)
        c.check("A1'' gate admits it as run() calls it (allow_offgrid true)",
                TD.alpha_gate_error(a_val, allow_offgrid=True, mode="real") is None)
        c.check("A1'' gate still refuses another off-grid alpha", TD.alpha_gate_error(7.0, allow_offgrid=False,
                                                                                     mode="real") is not None)
        c.check("A1'' gate keeps the grid values", TD.alpha_gate_error(25.0, allow_offgrid=False, mode="real") is None)
    c.check("the pinned alpha gate is restored and refuses the N4 alpha again",
            TD.alpha_gate_error is orig and orig(a_val, allow_offgrid=False, mode="real") is not None)
    e3 = K.manifest_entry(K.SLOT_E3S42)
    c.check("the manifest's E3 s42 is the alpha-25 run of record", e3["run_id"] == "e3_s42_lambda0.5_alpha25_80k_a1"
            and e3["telemetry_sha256"] == "734417e4b33e64239cebe2485e3eace03a795163b47bf50667517be40c161d2e")
    (base / "n4").mkdir()
    n4 = base / "n4" / "n4_terms.json"
    good = {"format": "x_n4_terms/1", "telemetry": {"sha256": e3["telemetry_sha256"]},
            "run_meta": {"stage": "E3", "seed": 42}, "window": {"last": 10000},
            "alpha_a1pp": {"status": "ok", "value": a_val, "repr": repr(a_val), "lambda": 0.5,
                           "mean_L_logitKD": 1.0, "mean_L_feat": 0.5 / a_val}}
    n4.write_text(json.dumps(good))
    got = c.passes("A1'' alpha read from N4's record of E3 s42", K.a1pp_alpha, n4, K.G.sha256_file(n4))
    c.check("A1'' alpha is N4's value, unrounded", got is not None and got["value"] == a_val)
    bad = dict(good, telemetry={"sha256": "1" * 64})
    n4b = base / "n4" / "n4_other.json"
    n4b.write_text(json.dumps(bad))
    c.refuses("N4 record of another telemetry refused", "alpha_source", K.a1pp_alpha, n4b, K.G.sha256_file(n4b))
    c.refuses("N4 record with a wrong sha256 refused", "alpha_source_sha256", K.a1pp_alpha, n4, "0" * 64)
    # ---- futility
    c.check("futility: 2.01 pp below stops", K.futile(0.3674, 0.3875))
    c.check("futility: exactly 2.0 pp below does not stop", not K.futile(0.3675, 0.3875))
    c.check("futility: 1.9 pp below does not stop", not K.futile(0.3685, 0.3875))
    c.check("futility: a non-finite VAL is not futility", not K.futile(float("nan"), 0.3875))
    written = []
    wrap = K.futility_jsonl(lambda p, r: written.append(r), comparator={"u": 0.3875}, item="N3", at_iter=40000)
    wrap("t", {"event": "val", "iter": 36000, "all_class_miou": 0.30})
    try:
        wrap("t", {"event": "val", "iter": 40000, "all_class_miou": 0.30})
        c.check("futility wrapper raises at 40k when futile", False)
    except K.FutilityStop as e:
        c.check("futility wrapper raises at 40k when futile", e.record["v"] == 0.30 and e.record["u"] == 0.3875)
    c.check("futility wrapper wrote the VAL row, then x_futility_stop", [r["event"] for r in written]
            == ["val", "val", "x_futility_stop"])
    # ---- the pinned trainer in --dry-run, with the substitutions
    if full:
        with substituted(TD, "build_dataloader", synthetic_loader_factory(), item="smoke", why="no dataset here"):
            counter = {"n": 0}

            def counted(**kw):
                counter["n"] += 1
                return DL.dist_distillation_losses(**kw)
            d_a2 = base / "x-a2-dry" / "train"
            with substituted(TD, "distillation_losses", counted, item="A2", why="smoke"):
                rc = TD.main(dry_argv("e2", d_a2, "--lambda-logit", "1"))
            c.check("dry A2 run inside the pinned trainer: RESULT PASS", rc == 0)
            c.check("dry A2: DIST was the distillation term every iteration", counter["n"] == 4)
            d_kl = base / "x-kl-dry" / "train"
            rc_kl = TD.main(dry_argv("e2", d_kl, "--lambda-logit", "1"))
            a2_kd = [r["logit_kd"] for r in rows_of(d_a2 / "e2_telemetry.jsonl") if r["event"] == "train"]
            kl_kd = [r["logit_kd"] for r in rows_of(d_kl / "e2_telemetry.jsonl") if r["event"] == "train"]
            c.check("dry A2: the logit_kd column holds DIST, not the KL of a same-seed run", rc_kl == 0
                    and len(a2_kd) == 4 and a2_kd != kl_kd and all(0.0 <= v <= 4.0 for v in a2_kd), str(a2_kd))
            rc_ref = TD.main(dry_argv("e3", base / "x-a1-ref" / "train", "--alpha", "3.2171"))
            c.check("dry A1'': the pinned trainer refuses the off-grid alpha alone (exit 2)", rc_ref == 2)
            d_a1 = base / "x-a1pp-dry" / "train"
            with substituted(TD, "alpha_gate_error", K.alpha_gate_admitting(TD.alpha_gate_error, a_val), item="A1pp",
                             why="smoke"):
                rc_a1 = TD.main(dry_argv("e3", d_a1, "--alpha", repr(a_val)))
            meta = [r for r in rows_of(d_a1 / "e3_run_meta.jsonl") if r.get("event") == "run_meta"][0]
            c.check("dry A1'': accepted with the substitution, alpha in run_meta, alpha_offgrid true",
                    rc_a1 == 0 and meta["alpha_cwd"] == a_val and meta["alpha_offgrid"] is True)
            d_f = base / "x-fut-dry" / "train"
            fut = K.futility_jsonl(TD._jsonl, comparator={"u": 0.99}, item="N3", at_iter=2)
            try:
                with substituted(TD, "_jsonl", fut, item="N3", why="smoke"):
                    TD.main(dry_argv("e2", d_f, "--lambda-logit", "0.5"))
                c.check("dry futility: the pinned trainer stops at the rule's iteration", False)
            except K.FutilityStop as e:
                tel = rows_of(d_f / "e2_telemetry.jsonl")
                c.check("dry futility: the pinned trainer stops at the rule's iteration", e.record["iter"] == 2)
                c.check("dry futility: the VAL row then x_futility_stop end the telemetry, no run_end",
                        [r["event"] for r in tel[-2:]] == ["val", "x_futility_stop"]
                        and not any(r["event"] == "run_end" for r in tel))
    # ---- the CLI (no CUDA and no data root in this session)
    py, cli = sys.executable, str(REPO / "scripts" / "explore_kd.py")
    env = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
    (base / "teach").mkdir(exist_ok=True)
    tck = str(base / "teach" / "iter_24000.pth")
    common = ["--real-run", "--confirm-real-run", "--teacher-ckpt", tck, "--num-workers", "1", "--expect-head", "0" * 40]
    r = subprocess.run([py, "-I", cli, *common, "--item", "N3", "--teacher-ckpt-sha256", K.R2_TEACHER_SHA256,
                        "--split", "te" + "st", "--run-id", "x-n3", "--out-dir", str(base / "o" / "x-n3")],
                       capture_output=True, text=True, env=env)
    c.check("CLI: --split TEST refused (exit 2)", r.returncode == 2 and "split_test" in r.stdout)
    r = subprocess.run([py, "-I", cli, *common, "--item", "N3", "--teacher-ckpt-sha256", K.R2_TEACHER_SHA256,
                        "--run-id", "n3", "--out-dir", str(base / "o" / "n3")], capture_output=True, text=True, env=env)
    c.check("CLI: a run ID without x- refused (exit 2)", r.returncode == 2 and "run_id_prefix" in r.stdout)
    rec_sha = K.teacher_expectation("A2")[0]
    r = subprocess.run([py, "-I", cli, *common, "--item", "N3", "--teacher-ckpt-sha256", rec_sha, "--run-id", "x-n3",
                        "--out-dir", str(base / "o" / "x-n3")], capture_output=True, text=True, env=env)
    c.check("CLI: N3 with the teacher of record's sha256 refused (exit 2)", r.returncode == 2
            and "teacher_not_item" in r.stdout)
    r = subprocess.run([py, "-I", cli, *common, "--item", "N3", "--teacher-ckpt-sha256", K.R2_TEACHER_SHA256,
                        "--run-id", "x-n3", "--out-dir", str(base / "o" / "x-n3")], capture_output=True, text=True,
                       env=env)
    c.check("CLI: a data root failing the TRAIN/VAL-only check refused, nothing written", r.returncode == 2
            and "data_root_not_trainval_only" in r.stdout and not (base / "o" / "x-n3").exists(), r.stdout[-160:])
    reuse = K.G.kd_pin_reuse_check()
    c.check("the KD pin's reused-module check runs and lists each module", len(reuse["modules"]) == 6,
            f"all_match {reuse['all_match']}, differ {reuse['differ']}")
    return c


def main() -> int:
    print("[smoke_explore_kd] cases")
    c = run_cases(verbose=True)
    print("[smoke_explore_kd] mutations (unit cases only)")
    for name, mut in {
        "DIST without its intra term": lambda: patched(DL, "intra_class_relation", lambda a, b: a.new_zeros(())),
        "futility margin ignored": lambda: patched(K, "FUTILITY_MARGIN", Decimal("1")),
        "a gate that admits every alpha": lambda: patched(K, "alpha_gate_admitting",
                                                          lambda orig, v: (lambda alpha, *, allow_offgrid, mode: None)),
    }.items():
        killed, failed = mutation_killed(lambda: run_cases(False, full=False), mut)
        c.check(f"mutation KILLED: {name}", killed, f"e.g. {failed[:2]}")
    return c.result()


if __name__ == "__main__":
    raise SystemExit(main())
