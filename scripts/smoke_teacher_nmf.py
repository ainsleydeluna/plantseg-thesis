#!/usr/bin/env python3
"""B62 — M4 NMF control through the real MMEngine Runner / ValLoop (teacher image, CPU, synthetic data).

M4 (B61 §4): rand_init=True is kept; its randomness is isolated.
  * M4-T  training: the isolated head is bit-identical to the stock NMF2D and uses the global stream.
  * M4-V  every validation pass: private stream seeded 42, batch 1, frozen order, caller CPU RNG
          restored exactly, no torch.manual_seed / CUDA seeding, one basis draw per image.
  * M4-KD frozen E2/E3 teacher: private stream seeded once, advancing, caller RNG untouched.
Negative controls: seed 43, reversed order, the hook removed, batch size 2.
A synthetic TRAIN/VAL-only staged root in a temp dir; no PlantSeg data, no checkpoint download.
Exit codes: 0 pass · 1 a check fails · 2 the teacher stack is not importable.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
TMP = Path(tempfile.mkdtemp(prefix="smoke_teacher_nmf_"))
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def staged_root(n_train: int = 4, n_val: int = 3) -> Path:
    import numpy as np
    from PIL import Image
    rs = np.random.RandomState(7)
    root = TMP / "staged"
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i in range(n):
            h, w = int(rs.randint(200, 700)), int(rs.randint(200, 700))
            Image.fromarray((rs.rand(h, w, 3) * 255).astype(np.uint8)).save(root / "images" / split / f"{split}_{i}.jpg")
            m = rs.randint(0, 6, (h, w)).astype(np.uint8)
            Image.fromarray(m, mode="L").save(root / "annotations" / split / f"{split}_{i}.png")
    return root


def smoke_cfg(root: Path, work: Path, *, val_bs: int = 1, val_workers: int = 2):
    from mmengine.config import Config
    cfg = Config.fromfile(str(CONFIG))
    for dl in (cfg.train_dataloader, cfg.val_dataloader):
        dl.dataset.data_root = str(root)
    cfg.train_dataloader.update(batch_size=2, num_workers=0, persistent_workers=False)
    cfg.val_dataloader.update(batch_size=val_bs, num_workers=val_workers, persistent_workers=val_workers > 0)
    cfg.work_dir = str(work)
    cfg.load_from = None
    cfg.randomness = dict(seed=42)     # the G18 strict policy is verified by smoke_teacher_runner
    cfg.default_hooks.logger.interval = 1
    # These runners only VALIDATE (runner.val()). MMEngine's CheckpointHook sets up its file backend in
    # before_train, so save_best cannot run in a val-only runner; it is exercised inside real training
    # runs by smoke_teacher_selection.
    cfg.default_hooks.checkpoint.save_best = None
    return cfg


def main() -> int:
    try:
        import torch
        from mmengine.runner import Runner
    except ImportError as e:
        print(f"ENVIRONMENT: teacher stack not importable ({e}); run in the plantseg-teacher image.")
        return 2
    os.chdir(REPO)
    root = staged_root()
    runner = Runner.from_cfg(smoke_cfg(root, TMP / "work"))
    import src.training.teacher_components as tc
    from mmseg.models.decode_heads.ham_head import NMF2D
    model = runner.model
    ham = model.decode_head.hamburger.ham
    hook = [h for h in runner.hooks if isinstance(h, tc.TeacherNMFEvalStreamHook)][0]

    # ---- M4-T: the isolated head is the stock algorithm on the global stream ----
    stock = NMF2D(dict(runner.cfg.model.decode_head.ham_kwargs))
    x = torch.rand(2, 512, 16, 16)
    same = []
    for train_mode in (True, False):
        ham.train(train_mode)
        stock.train(train_mode)
        torch.manual_seed(5)
        a = ham(x)
        sa = torch.get_rng_state()
        torch.manual_seed(5)
        b = stock(x)
        same.append(torch.equal(a, b) and torch.equal(sa, torch.get_rng_state()))
    ham.train(False)
    check("m4t_isolated_head_equals_stock_on_global_stream", all(same), "train and eval mode")
    ham.train(True)
    ham.nmf_stream = tc.NMFStream(42, "M4-V")
    try:
        ham(x)
        guard = False
    except tc.NMFStreamError:
        guard = True
    ham.nmf_stream = None
    ham.train(False)
    check("m4t_stream_in_train_mode_fails_closed", guard)

    # ---- M4-V through the real ValLoop ----
    logits: list = []
    # BaseDecodeHead.predict calls self.forward() directly, so a hook on decode_head never fires in
    # validation; conv_seg runs through __call__ and its output IS the logits (dropout off in eval).
    handle = model.decode_head.conv_seg.register_forward_hook(
        lambda m, i, o: logits.append(o.detach().clone()))

    def val_pass():
        logits.clear()
        before = torch.get_rng_state()
        metrics = runner.val_loop.run()
        return [t.clone() for t in logits], torch.equal(before, torch.get_rng_state()), metrics

    torch.manual_seed(1234)
    before = torch.get_rng_state()
    logits.clear()
    runner.val()                                          # first pass: builds the loop and workers
    pass1 = [t.clone() for t in logits]
    first_restored = torch.equal(before, torch.get_rng_state())
    pass2, second_restored, _ = val_pass()
    check("m4v_first_pass_equals_repeated_pass_bitwise",
          len(pass1) == len(pass2) == 3 and all(torch.equal(a, b) for a, b in zip(pass1, pass2)))
    check("m4v_caller_cpu_rng_restored_first_pass", first_restored,
          "includes the persistent val DataLoader's first-iterator base-seed draw")
    check("m4v_caller_cpu_rng_restored_repeat_pass", second_restored)
    rec = hook.passes[-1]
    check("m4v_one_basis_draw_per_image", rec["draws"] == rec["n_images"] == 3, str(rec))
    check("m4v_stream_detached_after_pass", ham.nmf_stream is None)

    seeding = {n: getattr(torch, n) for n in ("manual_seed",)}
    cuda_seeding = {n: getattr(torch.cuda, n) for n in ("manual_seed", "manual_seed_all")}

    def forbidden(*_a, **_k):
        raise AssertionError("seeding call during an M4-V pass")
    torch.manual_seed = forbidden
    torch.cuda.manual_seed = forbidden
    torch.cuda.manual_seed_all = forbidden
    try:
        pass3, _, _ = val_pass()
        seeded = False
    except AssertionError:
        seeded, pass3 = True, []
    finally:
        torch.manual_seed = seeding["manual_seed"]
        torch.cuda.manual_seed, torch.cuda.manual_seed_all = cuda_seeding["manual_seed"], cuda_seeding["manual_seed_all"]
    check("m4v_no_torch_manual_seed_or_cuda_seeding", not seeded
          and all(torch.equal(a, b) for a, b in zip(pass1, pass3)))

    hook.seed = 43
    pass43, _, _ = val_pass()
    hook.seed = 42
    check("m4v_negative_control_seed_43_differs", any(not torch.equal(a, b) for a, b in zip(pass1, pass43)))

    # Order change: a second runner over the same images in reversed order (MMEngine `indices` keeps the
    # given order), same weights. Each image's basis depends on its position in the pass.
    manifest_42 = runner.message_hub.get_info("m4v/manifest_sha256")
    rev_cfg = smoke_cfg(root, TMP / "work_rev", val_workers=0)
    rev_cfg.val_dataloader.dataset.indices = [2, 1, 0]
    rev_runner = Runner.from_cfg(rev_cfg)
    rev_runner.model.load_state_dict(model.state_dict())
    rev_logits: list = []
    rev_handle = rev_runner.model.decode_head.conv_seg.register_forward_hook(
        lambda m, i, o: rev_logits.append(o.detach().clone()))
    rev_runner.val()
    rev_handle.remove()
    manifest_rev = rev_runner.message_hub.get_info("m4v/manifest_sha256")
    check("m4v_order_change_detected_by_manifest", manifest_rev != manifest_42)
    check("m4v_order_change_changes_draws",
          torch.equal(rev_logits[1], pass1[1])
          and (not torch.equal(rev_logits[0], pass1[2]) or not torch.equal(rev_logits[2], pass1[0])),
          "the middle image keeps its draw; the swapped images get other bases")

    runner._hooks.remove(hook)
    torch.manual_seed(99)
    before = torch.get_rng_state()
    logits.clear()
    runner.val_loop.run()
    unhooked = [t.clone() for t in logits]
    changed = not torch.equal(before, torch.get_rng_state())
    runner._hooks.append(hook)
    check("m4v_negative_control_without_hook_consumes_caller_rng", changed,
          "upstream behaviour draws the NMF bases from the caller's stream")
    check("m4v_negative_control_without_hook_differs",
          any(not torch.equal(a, b) for a, b in zip(pass1, unhooked)))
    handle.remove()

    # ---- M4-V fails closed on batch size 2 ----
    try:
        Runner.from_cfg(smoke_cfg(root, TMP / "work_bs2", val_bs=2, val_workers=0)).val()
        bs2 = False
    except RuntimeError as e:
        bs2 = "batch_size 1" in str(e)
    check("m4v_batch_size_2_fails_closed", bs2)

    # ---- M4-KD: the frozen E2/E3 teacher ----
    ckpt = TMP / "teacher_116.pth"
    torch.save({"meta": {}, "state_dict": model.state_dict()}, ckpt)
    from src.distill.teacher import load_frozen_teacher
    t_e2 = load_frozen_teacher(str(ckpt), config_path=str(CONFIG))
    t_e3 = load_frozen_teacher(str(ckpt), config_path=str(CONFIG))
    batch = torch.randn(2, 3, 128, 128)
    t_e2.begin_nmf_stream("M4-KD", 42)
    t_e3.begin_nmf_stream("M4-KD", 42)
    torch.manual_seed(2026)
    caller = torch.get_rng_state()
    e2 = [t_e2(batch).logits for _ in range(3)]
    check("m4kd_caller_cpu_rng_untouched", torch.equal(caller, torch.get_rng_state()))
    check("m4kd_private_state_advances", t_e2.nmf_stream_state()["draws"] == 3
          and not torch.equal(e2[0], e2[1]) and not torch.equal(e2[1], e2[2]), "no per-batch reset")
    e3 = [t_e3(batch).logits for _ in range(3)]
    check("m4kd_e2_e3_identical_sequences", all(torch.equal(a, b) for a, b in zip(e2, e3)),
          "same call order -> same NMF sequence")
    t_e2.begin_nmf_stream("M4-KD", 42)
    check("m4kd_replay_from_42_reproduces", torch.equal(t_e2(batch).logits, e2[0]))

    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:50}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
