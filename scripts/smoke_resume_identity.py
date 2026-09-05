#!/usr/bin/env python3
"""B36 — E1 checkpoint resume-identity drill. CPU only, read-only w.r.t. the repository.

Question Ch4 must answer: is a run RESUMED from a periodic checkpoint equivalent to an
UNINTERRUPTED one? E1 is 80,000 iterations on Community Cloud with a 4,000-iteration checkpoint
cadence, so an eviction mid-run is realistic.

train_e1.py:355-360 already WARNS that data-order continuity is not restored, so the verdict is not
in question. This drill produces the EVIDENCE for the attribution and the bounded/unbounded ruling.

FOUR runs, four temp dirs OUTSIDE the repository (train_e1.py hard-guards in-repo --ckpt-dir):

  P   --max-iters 10 --ckpt-interval 10                 produces the N/2 resume point
  A   --max-iters 20 --ckpt-interval 20                 uninterrupted reference
  B   --resume P/last.pt --max-iters 20                 the resumed run
  B2  identical to B, separate dir                      resume-to-resume determinism

Controls, both of which matter:
  * --val-interval 10 for EVERY run, so validation fires at iteration 10 in both P and A. The
    validation block precedes the checkpoint block in the loop (train_e1.py:437 vs :466), so P's
    saved RNG state at iter 10 includes validation's effect and matches A's state there. A
    mismatched val_interval would desynchronise the streams and produce a divergence that would
    then be misattributed to data order.
  * --num-workers 0. With workers, _seed_worker reseeds from torch.initial_seed() per worker per
    epoch (src/data/dataset.py:100-103) and the restored main-process state would not reproduce the
    worker streams.

PREMISE CHECK (strong form): P's first 10 iterations must equal A's first 10. Both runs validate at
iteration 10, so both write e1_student_best_iter10.pt -- this is compared ELEMENTWISE (model tensors
+ optimizer + scheduler), not via displayed losses. Full-precision JSONL losses are compared as a
secondary cross-check.

No GPU, no downloads, no commit. Writes nothing inside the repository.

Usage:
    python scripts/smoke_resume_identity.py
    python scripts/smoke_resume_identity.py --iters 20 --keep   # keep the temp dirs for inspection
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = REPO / "src" / "training" / "train_e1.py"
SEED = 42


# --------------------------------------------------------------------------------------------------
# run helpers
# --------------------------------------------------------------------------------------------------
def run_train(tag, ckpt_dir, max_iters, ckpt_interval, val_interval, batch_size, resume=None):
    cmd = [PY, str(TRAIN), "--dry-run",
           "--max-iters", str(max_iters), "--ckpt-interval", str(ckpt_interval),
           "--val-interval", str(val_interval), "--max-val-batches", "1",
           "--batch-size", str(batch_size), "--num-workers", "0",
           "--seed", str(SEED), "--keep-ckpts", "9",
           "--ckpt-dir", str(ckpt_dir)]
    if resume:
        cmd += ["--resume", str(resume)]
    print(f"\n  [{tag}] $ ...train_e1.py --dry-run --max-iters {max_iters}"
          + (f" --resume {Path(resume).parent.name}/last.pt" if resume else ""), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=3600)
    secs = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"[{tag}] train_e1.py exited {r.returncode}")
    print(f"  [{tag}] done in {secs:.1f}s", flush=True)
    return r


def losses(ckpt_dir):
    """iter -> full-precision loss/ce/dice/lr straight from the JSONL (not displayed values)."""
    out = {}
    p = Path(ckpt_dir) / "e1_telemetry.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("event") == "train":
            out[d["iter"]] = (d["loss"], d["ce"], d["dice"], d["lr"])
    return out


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------------------------------
def compare_models(a, b):
    """Elementwise. Returns (global_max, n_tensors, n_differing, worst_key, per_tensor list)."""
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    if set(sa) != set(sb):
        raise SystemExit(f"state_dict key mismatch: {set(sa) ^ set(sb)}")
    rows, gmax, worst, ndiff = [], 0.0, None, 0
    for k in sa:
        ta, tb = sa[k].float(), sb[k].float()
        d = float((ta - tb).abs().max().item()) if ta.numel() else 0.0
        rows.append((k, tuple(sa[k].shape), d))
        if d > 0:
            ndiff += 1
        if d > gmax:
            gmax, worst = d, k
    return gmax, len(rows), ndiff, worst, rows


def compare_optimizer(a, b):
    """SGD momentum buffers, elementwise, plus the param-group scalars."""
    oa, ob = a["optimizer_state_dict"], b["optimizer_state_dict"]
    ga = [{k: v for k, v in g.items() if k != "params"} for g in oa["param_groups"]]
    gb = [{k: v for k, v in g.items() if k != "params"} for g in ob["param_groups"]]
    gmax, nbuf, nmissing = 0.0, 0, 0
    for pid in sorted(set(oa["state"]) | set(ob["state"])):
        va, vb = oa["state"].get(pid, {}), ob["state"].get(pid, {})
        for key in sorted(set(va) | set(vb)):
            if key not in va or key not in vb:
                nmissing += 1
                continue
            xa, xb = va[key], vb[key]
            if torch.is_tensor(xa):
                nbuf += 1
                gmax = max(gmax, float((xa.float() - xb.float()).abs().max().item()))
    return gmax, nbuf, nmissing, ga == gb, ga, gb


def compare_scheduler(a, b):
    sa, sb = a["scheduler_state_dict"], b["scheduler_state_dict"]
    keys = sorted(set(sa) | set(sb))
    return {k: (sa.get(k), sb.get(k), sa.get(k) == sb.get(k)) for k in keys}


def report_pair(title, pa, pb):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    a, b = load(pa), load(pb)
    print(f"  A = {Path(pa).parent.name}/{Path(pa).name}   iter={a['iter']}")
    print(f"  B = {Path(pb).parent.name}/{Path(pb).name}   iter={b['iter']}")

    gmax, ntot, ndiff, worst, rows = compare_models(a, b)
    print(f"\n  MODEL   tensors={ntot}  differing={ndiff}  global_max_abs_dev={gmax:.6e}")
    if ndiff:
        top = sorted(rows, key=lambda r: -r[2])[:8]
        print(f"  {'tensor':<52}{'shape':<22}max|dev|")
        for k, shp, d in top:
            print(f"    {k:<50}{str(shp):<22}{d:.6e}")

    omax, nbuf, nmiss, groups_eq, ga, gb = compare_optimizer(a, b)
    print(f"\n  OPTIM   momentum_buffers={nbuf}  max_abs_dev={omax:.6e}  "
          f"param_groups_equal={groups_eq}  missing_keys={nmiss}")
    if not groups_eq:
        print(f"    A groups: {ga}")
        print(f"    B groups: {gb}")

    print("\n  SCHED")
    for k, (va, vb, eq) in compare_scheduler(a, b).items():
        mark = "==" if eq else "!="
        print(f"    {k:<22}{str(va):<24}{mark:<4}{vb}")

    identical = (gmax == 0.0 and omax == 0.0 and groups_eq and nmiss == 0
                 and all(eq for _, _, eq in compare_scheduler(a, b).values()))
    print(f"\n  -> {'IDENTICAL' if identical else 'NON-IDENTICAL'}")
    return identical, gmax, omax


def demo_sampler(n_samples, batch_size, restored_gen_state, upto_iter):
    """Show WHY a restored generator does not restore position, without training anything.

    DataLoader(shuffle=True) draws a fresh permutation from `generator` on EVERY __iter__ call.
    cycle() (train_e1.py:77-82) holds ONE iterator alive across the whole run, so an uninterrupted
    run stays inside permutation #1. A resumed run builds a NEW loader and a NEW cycle(), so its
    first __iter__ draws permutation #2 -- from the correctly restored generator state.
    """
    print("\n" + "=" * 78)
    print("MECHANISM: why restoring the generator does NOT restore epoch position")
    print("=" * 78)

    def draw_epoch(g):
        """One DataLoader.__iter__ worth of draws from the loader generator.

        _BaseDataLoaderIter.__init__ takes a base_seed from loader.generator BEFORE the sampler
        draws its permutation, and does so even at num_workers=0. Omitting it desynchronises any
        replay of the stream -- the first version of this drill did exactly that and produced a
        spurious 'different stream' result. The model below is verified against the checkpoint's
        own saved generator state by _assert_stream_model() before any conclusion is drawn.
        """
        torch.empty((), dtype=torch.int64).random_(generator=g).item()      # base_seed
        return torch.randperm(n_samples, generator=g)

    # Verify the consumption model reproduces the SAVED state before trusting anything built on it.
    g_chk = torch.Generator()
    g_chk.manual_seed(SEED)
    draw_epoch(g_chk)
    model_ok = torch.equal(g_chk.get_state(), restored_gen_state)
    print(f"  consumption model reproduces the checkpoint's saved generator state: {model_ok}")
    if not model_ok:
        print("  -> MODEL MISMATCH: the replay below would be meaningless. No conclusion drawn.")
        return None, None, None

    g1 = torch.Generator()
    g1.manual_seed(SEED)
    perm1 = draw_epoch(g1)                              # what run A drew at iteration 1
    perm2_from_A = draw_epoch(g1)                       # what run A would draw at its epoch 2
    lo = upto_iter * batch_size
    a_next = perm1[lo:lo + batch_size].tolist()

    g2 = torch.Generator()
    g2.set_state(restored_gen_state)
    perm2 = draw_epoch(g2)                              # what run B draws on its first __iter__
    b_next = perm2[:batch_size].tolist()

    print(f"  train samples={n_samples}  batch_size={batch_size}  "
          f"one epoch = {n_samples // batch_size} iterations")
    print(f"  iteration {upto_iter + 1} is deep inside epoch 1 -- no epoch boundary is reached.")
    print(f"\n  run A  iteration {upto_iter + 1} batch = perm1[{lo}:{lo + batch_size}] = {a_next}")
    print(f"  run B  iteration {upto_iter + 1} batch = perm2[0:{batch_size}]  = {b_next}")
    print(f"  same batch: {a_next == b_next}")

    same_stream = torch.equal(perm2, perm2_from_A)
    print(f"\n  B's permutation == A's epoch-2 permutation: {same_stream}")
    if same_stream:
        print("  -> B samples the SAME STREAM, consumed early: it starts A's NEXT epoch and")
        print("     discards the unconsumed tail of epoch 1. Distribution is unchanged.")
    else:
        print("  -> B's permutation is NOT A's next-epoch permutation. The resumed run draws from")
        print("     a position in the stream the uninterrupted run never occupies.")
    return a_next, b_next, same_stream


# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B36 resume-identity drill (CPU, out-of-repo).")
    ap.add_argument("--iters", type=int, default=20, help="N; checkpoint is taken at N/2")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--keep", action="store_true", help="keep the temp dirs")
    args = ap.parse_args(argv)

    N = args.iters
    H = N // 2
    root = Path(tempfile.mkdtemp(prefix="b36_resume_"))
    assert REPO not in root.parents and root != REPO, "temp root must be OUTSIDE the repo"
    dirs = {t: root / t for t in ("P", "A", "B", "B2")}
    for d in dirs.values():
        d.mkdir(parents=True)

    print("=" * 78)
    print("B36 — E1 CHECKPOINT RESUME-IDENTITY DRILL")
    print("=" * 78)
    print(f"  repo      : {REPO}")
    print(f"  temp root : {root}   (OUTSIDE the repo)")
    print(f"  N={N}  checkpoint at {H}  batch_size={args.batch_size}  seed={SEED}  num_workers=0")

    run_train("P ", dirs["P"], H, H, H, args.batch_size)
    run_train("A ", dirs["A"], N, N, H, args.batch_size)
    run_train("B ", dirs["B"], N, N, H, args.batch_size, resume=dirs["P"] / "last.pt")
    run_train("B2", dirs["B2"], N, N, H, args.batch_size, resume=dirs["P"] / "last.pt")

    # ---- premise (strong form): P@H == A@H, compared elementwise on real tensors ----
    pH, aH = dirs["P"] / f"e1_student_best_iter{H}.pt", dirs["A"] / f"e1_student_best_iter{H}.pt"
    premise_mode = "ELEMENTWISE state_dict comparison"
    if pH.exists() and aH.exists():
        premise_ok, _, _ = report_pair(
            f"PREMISE (strong): P@{H} vs A@{H} — do the first {H} iterations coincide?", pH, aH)
    else:
        premise_mode = "FULL-PRECISION JSONL loss comparison ONLY (EVIDENCED-NOT-PROVEN)"
        lp, la = losses(dirs["P"]), losses(dirs["A"])
        premise_ok = all(lp[i] == la[i] for i in range(1, H + 1))
        print(f"\n  [premise] iter-{H} best checkpoints absent; fell back to loss equality: "
              f"{premise_ok}")
    print(f"\n  premise check used: {premise_mode}")

    # ---- main comparison: uninterrupted vs resumed ----
    ident_AB, gmax_AB, omax_AB = report_pair(
        f"MAIN: A (uninterrupted {N}) vs B (resumed from {H})",
        dirs["A"] / "last.pt", dirs["B"] / "last.pt")

    # ---- resume-to-resume determinism ----
    ident_BB, gmax_BB, omax_BB = report_pair(
        "DETERMINISM: B vs B2 (two resumes from the SAME checkpoint)",
        dirs["B"] / "last.pt", dirs["B2"] / "last.pt")

    # ---- full-precision loss traces ----
    print("\n" + "=" * 78)
    print("FULL-PRECISION LOSS TRACES (from JSONL, not displayed values)")
    print("=" * 78)
    lp, la, lb, lb2 = (losses(dirs[t]) for t in ("P", "A", "B", "B2"))
    print(f"  P vs A, iters 1..{H} (the premise): "
          f"all equal = {all(lp[i] == la[i] for i in range(1, H + 1))}")
    print(f"\n  {'iter':<6}{'A loss':<22}{'B loss':<22}equal")
    for i in range(H + 1, N + 1):
        if i in la and i in lb:
            print(f"  {i:<6}{la[i][0]!r:<22}{lb[i][0]!r:<22}{la[i][0] == lb[i][0]}")
    first_div = next((i for i in range(H + 1, N + 1)
                      if i in la and i in lb and la[i][0] != lb[i][0]), None)
    print(f"\n  first iteration whose loss differs: {first_div}")
    print(f"  B vs B2, iters {H + 1}..{N}: all equal = "
          f"{all(lb[i] == lb2[i] for i in range(H + 1, N + 1) if i in lb and i in lb2)}")

    # ---- LR curve across the boundary (is the scheduler continuing or restarting?) ----
    print(f"\n  {'iter':<6}{'A lr':<24}{'B lr':<24}equal")
    for i in list(range(1, min(H, 3) + 1)) + list(range(H + 1, min(H + 3, N) + 1)) + [N]:
        if i in la and i in lb:
            print(f"  {i:<6}{la[i][3]!r:<24}{lb[i][3]!r:<24}{la[i][3] == lb[i][3]}")
        elif i in la:
            print(f"  {i:<6}{la[i][3]!r:<24}{'(not run in B)':<24}-")

    # ---- mechanism ----
    ck = load(dirs["P"] / "last.pt")
    gen_state = ck["rng_state"]["loader_generator"]
    sys.path.insert(0, str(REPO))
    from src.data import build_dataloader                                    # noqa: E402
    n_samples = len(build_dataloader("train", args.batch_size, num_workers=0).dataset)
    a_next, b_next, same_stream = demo_sampler(n_samples, args.batch_size, gen_state, H)

    # ---- summary ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  premise P@{H} == A@{H}          : {premise_ok}   ({premise_mode})")
    print(f"  A vs B  (uninterrupted/resumed) : "
          f"{'IDENTICAL' if ident_AB else 'NON-IDENTICAL'}  global_max={gmax_AB:.6e}")
    print(f"  B vs B2 (resume determinism)    : "
          f"{'IDENTICAL' if ident_BB else 'NON-IDENTICAL'}  global_max={gmax_BB:.6e}")
    print(f"  first diverging iteration       : {first_div}")
    print(f"  B's permutation == A's epoch 2  : {same_stream}")
    print(f"\n  temp root: {root}")
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
        print("  (temp root removed; pass --keep to retain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
