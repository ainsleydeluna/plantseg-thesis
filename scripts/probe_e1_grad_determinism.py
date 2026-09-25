#!/usr/bin/env python3
"""N10: E1 gradient-determinism probe (record-only; B52 N10, B66-prep B-4, ruling Q10).

Re-runs the B52 gradient comparison with every repeat recorded (B52 printed rows[:3] only) on the E1
seed-42 checkpoint (sha256 cf0879f7...; no download), under E1's exact determinism settings
(src/seeds.set_seed(42)), on the first seed-42 TRAIN batch. Probe order is C -> A -> B: Probe A's
train-mode forward updates BatchNorm running statistics, so the eval-mode Probe C runs first on the
freshly loaded checkpoint.

  Probe C  eval-mode forward of the batch, R times: sha256 of the logits per pass.
  Probe A  one train-mode forward, logits detached as a leaf; R times: weighted CE (mean and sum), the
           implied normaliser ce_sum/ce_mean, the actual ATen normaliser
           torch.ops.aten.nll_loss2d_forward(...)[1], and sha256 of d(ce_mean)/d(logits).
  Probe B  the full E1 step R times, exactly as train_e1.py:437-439 and :451 (zero_grad(set_to_none) ->
           forward -> ce = criterion.ce, dice = criterion.dice, loss = ce + dice -> backward; no optimizer
           step): ce, dice and loss (exact floats), sha256 over all parameter grads, and beside them
           criterion(logits, mask) under no_grad (never backpropagated) with its bitwise equality to ce + dice,
           plus a no_grad re-run of ce and dice that shows whether the kernels themselves are
           nondeterministic (how to read the flags: the record's criterion_compare_note).

Record-only: exit 0 whenever the probe completes, whatever it finds; 1 on an unexpected error; 2 on a
precondition refusal (--out exists, is relative, or lies inside the repository or the data root; its
directory is missing; CUDA is already initialised; checkpoint sha256 mismatch; TEST surface present).
The record is written atomically to --out (outside the repository).

Batch identity: with --num-workers 12 (the seed-42 recipe) a fresh build_dataloader('train', 16,
num_workers=12, seed=42) serves batch 0 from worker 0 with the loader generator's base seed, so the
batch is INFERRED from the loader code, not MEASURED, to be seed 42's first training batch.

  python -B scripts/probe_e1_grad_determinism.py --checkpoint CK --out /evidence/n10.json
  python -B scripts/probe_e1_grad_determinism.py --synthetic --device cpu --repeats 3 --out X   (smoke)
"""
import sys

sys.dont_write_bytecode = True                  # X8: before the first non-stdlib import

import os  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse     # noqa: E402
import hashlib      # noqa: E402
import json         # noqa: E402
import platform     # noqa: E402
import subprocess   # noqa: E402
import time         # noqa: E402
import traceback    # noqa: E402
import warnings     # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

E1_SEED42_CHECKPOINT_SHA256 = "cf0879f7007dfacbd0d510085ff28a4b47ee845ddb8fd599611d74109e1d6a03"
RECIPE_NUM_WORKERS = 12
BATCH_SIZE = 16
# Probe B mirrors E1's step exactly: train_e1 backpropagates ce + dice, never criterion(logits, mask).
E1_STEP_LOSS_LINES = "src/training/train_e1.py:437-439"
E1_STEP_BACKWARD_LINE = "src/training/train_e1.py:451"
IGNORE_INDEX = 255
EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
BATCH_IDENTITY_NOTE = ("batch 0 of a fresh build_dataloader('train', 16, num_workers=N, seed=42): with N = 12 "
                       "(chosen to match the seed-42 recipe) worker 0 serves batch 0 from the loader "
                       "generator's base seed, so identity with seed 42's first training batch is INFERRED "
                       "from the loader code, not MEASURED")


class Refused(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _under(path: Path, root: Path) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tensor_sha(t) -> str:
    return _sha(t.detach().cpu().contiguous().numpy().tobytes())


def check_out(out: str) -> Path:
    p = Path(out)
    if not p.is_absolute():
        raise Refused("out_relative", f"--out must be absolute: {out!r}")
    if p.exists():
        raise Refused("out_exists", f"--out {p} already exists")
    if _under(p, REPO):
        raise Refused("out_inside_repo", f"--out {p} is inside the repository")
    from configs.data import DATA                   # the root the loader uses (env var or the default)
    if _under(p, Path(DATA["root"])):               # Path(DATA["root"]) exactly as the loader builds it
        raise Refused("out_inside_data_root", f"--out {p} is inside the data root {DATA['root']!r}")
    if not p.parent.is_dir():
        raise Refused("out_parent_missing", f"--out's directory {p.parent} does not exist")
    return p


def git_head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


E1_CODE_STATUS_PATHS = ("src", "configs", "scripts", "requirements-e1.txt", "reports/e1_class_weights.json")


def git_status_e1_code():
    """Scoped status of the E1 code paths (not the section 7.1 governed set; never a whole-repository
    status); None outside a checkout."""
    r = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *E1_CODE_STATUS_PATHS],
                       cwd=str(REPO), capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def distinct(values) -> int:
    return len({json.dumps(v, sort_keys=True) for v in values})


def run_probe(args) -> dict:
    import torch
    import torch.nn.functional as F

    if args.device.startswith("cuda") and torch.cuda.is_initialized():
        raise Refused("cuda_initialized", "CUDA is already initialised; set_seed must run before the first "
                                          "CUDA op (CUBLAS_WORKSPACE_CONFIG)")
    from src.seeds import set_seed
    set_seed(42)                                        # E1's exact settings (called, not edited)
    from src.eval import eval_runtime as ER             # read-only state readers
    dev = torch.device(args.device)
    rec = {
        "probe": "N10 E1 gradient determinism (record-only)", "argv": sys.argv[1:],
        "repeats": args.repeats, "synthetic": args.synthetic, "probe_order": ["C", "A", "B"],
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if dev.type == "cuda" else None,
        "determinism": ER.determinism_state(), "fill_uninitialized_memory": ER.fill_uninitialized_memory_state(),
        "tf32": ER.tf32_state(), "image_digest": ER.image_digest(), "git_head": git_head(),
        "git_status_e1_code_paths": git_status_e1_code(), "git_status_pathspecs": list(E1_CODE_STATUS_PATHS),
    }
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        rec["gpu_name"] = torch.cuda.get_device_name(idx)
        rec["gpu_capability"] = "%d.%d" % torch.cuda.get_device_capability(idx)

    # ---- weights ----
    if args.synthetic:
        from src.models.student import build_student
        model = build_student(pretrained=False).to(dev).eval()
        rec["weights"] = {"kind": "synthetic random init (smoke only)", "sha256": None}
    else:
        from src.eval.model_loading import load_student_checkpoint, sha256_file
        ck = Path(args.checkpoint)
        digest = sha256_file(ck)
        if digest != args.expect_sha256:
            raise Refused("checkpoint_sha_mismatch", f"sha256({ck}) = {digest} != {args.expect_sha256}")
        model, info = load_student_checkpoint(ck, map_location=str(dev))
        rec["weights"] = {"kind": "E1 checkpoint", "path": str(ck), "sha256": digest,
                          "is_e1_seed42": digest == E1_SEED42_CHECKPOINT_SHA256, "iter": info.iteration,
                          "best_val_miou_all_class": info.best_val_miou_all_class}

    # ---- class weights + the batch ----
    from src.training.train_e1 import load_ce_weights
    w = load_ce_weights().to(dev)
    if args.synthetic:
        g = torch.Generator().manual_seed(42)
        img = torch.rand(2, 3, 64, 64, generator=g)
        mask = torch.randint(0, 116, (2, 64, 64), generator=g)
        mask[:, :4] = IGNORE_INDEX
        rec["batch"] = {"kind": "synthetic 2x3x64x64 (smoke only)", "num_workers": None}
    else:
        from configs.data import DATA, SPLIT_SIZES
        from src.data import build_dataloader
        from src.data.isolation import TrainValIsolationError, assert_trainval_only_root
        try:
            iso = assert_trainval_only_root(Path(DATA["root"]), {k: SPLIT_SIZES[k] for k in ("train", "val")})
        except TrainValIsolationError as e:
            raise Refused(e.code, f"{e} -- DL-21: the probe runs on a TRAIN/VAL-only staged root") from e
        loader = build_dataloader("train", BATCH_SIZE, num_workers=args.num_workers, seed=42)
        img, mask = next(iter(loader))
        del loader
        rec["batch"] = {"kind": "first TRAIN batch", "num_workers": args.num_workers,
                        "recipe_num_workers": RECIPE_NUM_WORKERS, "identity_note": BATCH_IDENTITY_NOTE,
                        "data_root": str(DATA["root"]), "isolation_counts": iso["counts"]}
    rec["batch"].update(shape=list(img.shape), mask_shape=list(mask.shape), img_sha256=_tensor_sha(img),
                        mask_sha256=_tensor_sha(mask))
    valid = mask[mask != IGNORE_INDEX]
    tw64 = float(w.detach().cpu().double()[valid.cpu()].sum())
    rec["batch"].update(total_weight_float64=tw64, total_weight_float32=float(torch.tensor(tw64, dtype=torch.float32)),
                        valid_pixels=int(valid.numel()))
    img, mask = img.to(dev), mask.to(dev)

    captured = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # ---- Probe C: eval-mode forward on the freshly loaded weights ----
        model.eval()
        c_passes = []
        for r in range(args.repeats):
            with torch.no_grad():
                logits = model(img)
            c_passes.append({"pass": r + 1, "logits_sha256": _tensor_sha(logits)})
            print(f"[C] pass {r + 1}: logits {c_passes[-1]['logits_sha256'][:16]}", flush=True)
        rec["probe_C"] = {"passes": c_passes, "distinct_logits": distinct([p["logits_sha256"] for p in c_passes])}

        # ---- Probe A: fixed logits, CE normaliser and gradient ----
        model.train()
        logits0 = model(img).detach().requires_grad_()
        a_passes = []
        for r in range(args.repeats):
            ce_mean = F.cross_entropy(logits0, mask, weight=w, ignore_index=IGNORE_INDEX)
            ce_sum = F.cross_entropy(logits0, mask, weight=w, ignore_index=IGNORE_INDEX, reduction="sum")
            try:
                out = torch.ops.aten.nll_loss2d_forward(F.log_softmax(logits0.detach(), 1), mask, w, 1,
                                                        IGNORE_INDEX)
                aten_tw, aten_err = float(out[1]), None
            except Exception as e:                        # noqa: BLE001 -- recorded, per Q10
                aten_tw, aten_err = None, f"{type(e).__name__}: {e}"
            grad = torch.autograd.grad(ce_mean, logits0)[0]
            a_passes.append({"pass": r + 1, "ce_mean": float(ce_mean), "ce_sum": float(ce_sum),
                             "implied_total_weight": float(ce_sum) / float(ce_mean),
                             "aten_total_weight": aten_tw, "aten_error": aten_err,
                             "grad_sha256": _tensor_sha(grad)})
            p = a_passes[-1]
            print(f"[A] pass {r + 1}: ce_mean={p['ce_mean']!r} ce_sum={p['ce_sum']!r} "
                  f"implied_tw={p['implied_total_weight']!r} aten_tw={p['aten_total_weight']!r} "
                  f"grad {p['grad_sha256'][:16]}", flush=True)
        rec["probe_A"] = {"passes": a_passes,
                          "distinct": {k: distinct([p[k] for p in a_passes])
                                       for k in ("ce_mean", "ce_sum", "implied_total_weight",
                                                 "aten_total_weight", "grad_sha256")}}

        # ---- Probe B: the full E1 step without the optimizer step ----
        from src.training.losses import CombinedCEDiceLoss
        criterion = CombinedCEDiceLoss(weight=w, ignore_index=IGNORE_INDEX).to(dev)
        params = [p for p in model.parameters() if p.requires_grad]
        b_passes = []
        for r in range(args.repeats):
            model.zero_grad(set_to_none=True)
            logits = model(img)
            ce = criterion.ce(logits, mask)             # the E1 step's own calls, E1_STEP_LOSS_LINES
            dice = criterion.dice(logits, mask)
            loss = ce + dice
            loss.backward()                             # as E1_STEP_BACKWARD_LINE
            with torch.no_grad():                       # recorded beside it, never backpropagated
                crit = criterion(logits, mask)
                ce2, dice2 = criterion.ce(logits, mask), criterion.dice(logits, mask)   # kernel re-run
            h = hashlib.sha256()
            for prm in params:
                if prm.grad is not None:
                    h.update(prm.grad.detach().cpu().contiguous().numpy().tobytes())
            b_passes.append({"pass": r + 1, "ce": float(ce), "dice": float(dice), "loss": float(loss),
                             "criterion_forward": float(crit),
                             "criterion_equals_ce_plus_dice": bool(torch.equal(crit, loss.detach())),
                             "rerun_ce_equals_ce": bool(torch.equal(ce2, ce.detach())),
                             "rerun_dice_equals_dice": bool(torch.equal(dice2, dice.detach())),
                             "rerun_ce_plus_dice_equals_loss": bool(torch.equal(ce2 + dice2, loss.detach())),
                             "grads_sha256": h.hexdigest()})
            p = b_passes[-1]
            print(f"[B] pass {r + 1}: ce={p['ce']!r} dice={p['dice']!r} loss={p['loss']!r} "
                  f"criterion={p['criterion_forward']!r} grads {p['grads_sha256'][:16]}", flush=True)
        rec["probe_B"] = {"passes": b_passes,
                          "loss_call": f"loss = criterion.ce(logits, mask) + criterion.dice(logits, mask), as "
                                       f"{E1_STEP_LOSS_LINES}; backward as {E1_STEP_BACKWARD_LINE}",
                          "ce_dice_source": "the E1 step's own terms (not recomputed)",
                          "criterion_forward_source": "criterion(logits, mask) under no_grad after backward; not "
                                                      "backpropagated (the E1 step never calls it)",
                          "criterion_compare_note": "criterion(...) and the rerun_* terms are independent no_grad "
                                                    "re-executions of the same CE and Dice kernels on the same "
                                                    "logits. Any rerun_ce_equals_ce or rerun_dice_equals_dice false, "
                                                    "or probe_A distinct ce_mean > 1, shows kernel nondeterminism "
                                                    "(on CUDA the CE forward, nll_loss2d_forward, is "
                                                    "nondeterministic), which alone can explain "
                                                    "criterion_equals_ce_plus_dice false. A genuine criterion-vs-"
                                                    "step difference is suggested only if criterion is unequal "
                                                    "while every rerun_* flag is true in all passes and probe_A's "
                                                    "ce_mean is single-valued",
                          "distinct": {k: distinct([p[k] for p in b_passes])
                                       for k in ("ce", "dice", "loss", "criterion_forward", "grads_sha256")}}
        # F3: after the last repo import (Probe B's losses), so every imported repo module is covered.
        repo_modules = sorted(n for n, m in list(sys.modules.items())
                              if n.split(".")[0] in ("src", "configs") and getattr(m, "__file__", None))
        rec["repo_modules"] = repo_modules
        rec["modules_under_repo"] = all(_under(Path(sys.modules[n].__file__), REPO) for n in repo_modules)
        for wmsg in caught:                             # an empty message must not lose the record
            first = (str(wmsg.message).splitlines() or [""])[0]
            captured.append(f"{wmsg.category.__name__}: {first[:300]}")
    counts = {}
    for line in captured:
        counts[line] = counts.get(line, 0) + 1
    rec["warnings"] = [{"warning": k, "count": v} for k, v in sorted(counts.items())]
    rec["peak_vram_bytes"] = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="N10 E1 gradient-determinism probe (record-only).")
    ap.add_argument("--out", required=True, help="record JSON (absolute; outside the repo and the data root)")
    ap.add_argument("--checkpoint", default=None, help="the E1 checkpoint (required unless --synthetic)")
    ap.add_argument("--expect-sha256", default=E1_SEED42_CHECKPOINT_SHA256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=RECIPE_NUM_WORKERS)
    ap.add_argument("--synthetic", action="store_true", help="CPU smoke mode: random init, random batch")
    args = ap.parse_args(argv)
    t0 = time.time()
    try:
        out = check_out(args.out)
        if not args.synthetic and not args.checkpoint:
            raise Refused("checkpoint_missing", "--checkpoint is required unless --synthetic")
        if args.repeats < 2:
            raise Refused("repeats_too_few", "--repeats must be >= 2 for a determinism record")
        rec = run_probe(args)
    except Refused as e:
        print(f"REFUSED {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:                                # noqa: BLE001 -- an incomplete run
        print(f"ERROR {type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}", file=sys.stderr)
        return EXIT_ERROR
    rec["wall_seconds"] = round(time.time() - t0, 1)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(f"N10 record written: {out}")
    print("RESULT: COMPLETE (record-only; no verdict) "
          f"C distinct logits={rec['probe_C']['distinct_logits']} "
          f"A distinct grads={rec['probe_A']['distinct']['grad_sha256']} "
          f"B distinct grads={rec['probe_B']['distinct']['grads_sha256']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
