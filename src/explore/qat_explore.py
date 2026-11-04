"""E3: the exploratory QAT entry for C1, N1, N2, N5 and D (audit EXPLORE-AUDIT-1 §6.2; addendum L-AM24-ADD-1).

Every run is E6-E2's recipe of record (AM-23 item 1 = AM-4a item 1 + clip 5.0, DL-139/DL-146), run by the pinned
src/quant/qat.py run_qat in mode "real", except the one named change of its item:

  C1   E6-E2 at seed 43 or 44 from E2 s43 / s44's best checkpoint of record; QAT seed = source seed. No change.
  N1   --kd-logit: logit KD in the QAT loss, lambda 0.5, T 4, E6-KD's one-epoch ramp and teacher stream (the teacher of
       record, one frozen forward per step on its private M4-KD NMF stream), no CWD term, no projection. run_qat has no
       loss hook, so N1 runs `run_qat_logit_kd` below: run_qat's loop with the teacher forward and
       train_distill.distillation_losses on stage e2 added, every other line E6-E2's (src/quant/qat.py constants,
       loaders, freezes, VAL bracket, state predicate, epoch checkpoint, telemetry).
  N2   --sites-on-input-grid --tf32-off: the ten sites (src/explore/sites.py) fake-quantized on their input's grid,
       installed on the prepared student run_qat builds, and TF32 off for cuDNN and matmul. The prepared student is
       built inside run_qat by the name prepare_qat_model of src/quant/qat.py; for this one run that name is bound to
       a wrapper that calls the pinned function and installs the sites (src/explore/patching.py), recorded in
       run_meta (x_substitution). The file is unchanged and its sha256 recorded.
  N5   --fp32-head: the LR-ASPP logit head in FP32 (src/explore/fp32_head.py): a MIXED-PRECISION model, never INT8.
  D    --parent-sha256: an exploratory FP32 winner (an x- run of N3, A1'' or A2) whose best VAL all-class mIoU is at
       least D_TRIGGER = 0.3875 (compared as Decimal of the value's repr); its own seed; E6-E2's recipe of record, not
       the corrected one (one change at a time). An E2-stage parent runs stage E6-E2, an E3-stage parent (A1'') stage
       E6 (E6's recipe is E6-E2's; only the parent loader differs).

The exploratory parent check accepts, for C1, N1, N2 and N5, exactly E2 s42 / s43 / s44's best checkpoints of record
by their recorded sha256 (E2_PARENTS_OF_RECORD), through the pinned resolve_parent (best.json, run_meta, a complete
run, the seed); D's parent through resolve_parent with --parent-sha256, which must match the file.
"""
from __future__ import annotations

import math
import time
from decimal import Decimal
from pathlib import Path

import torch
import torch.nn as nn

from . import guard as G

E2_PARENTS_OF_RECORD = {
    42: {"run_id": "e2_s42_lambda0.5_80k_a1", "checkpoint": "e2_student_best_iter64000.pt",
         "sha256": "73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06",
         "best_val": 0.38749080896377563, "records": "DL-109, DL-146; reports/derived/tday_manifest.json"},
    43: {"run_id": "e2_s43_lambda0.5_80k_a1", "checkpoint": "e2_student_best_iter72000.pt",
         "sha256": "e3c4e05ccfffa5c78d88a9309542108c7a974f03897d4564bd0644a032b4d385",
         "best_val": 0.3716566562652588, "records": "DL-120, DL-218; reports/derived/tday_manifest.json"},
    44: {"run_id": "e2_s44_lambda0.5_80k_a1", "checkpoint": "e2_student_best_iter48000.pt",
         "sha256": "be48da058a55a3efb049ead5f48409ab9061aad549708a89805fd7113ac2fa56",
         "best_val": 0.3703170716762543, "records": "DL-120, DL-218; reports/derived/tday_manifest.json"},
}
CLIP_OF_RECORD = 5.0                        # DL-139: the U4 winner; E6-E2's clip (DL-146)
CLIP_SELECTION_SHA256 = "c433145a6cfe0e133ebee8fcb70a76f05b60fc6462670abee34c5304704fe698"   # DL-139
D_TRIGGER = Decimal("0.3875")               # audit §2 D / §6.3: exploratory FP32 best VAL >= 0.3875
D_PARENT_ITEMS = ("N3", "A1pp", "A2")
N1_LAMBDA = 0.5
ITEMS = ("C1", "N1", "N2", "N5", "D")
ITEM_SEEDS = {"C1": (43, 44), "N1": (42,), "N2": (42,), "N5": (42,)}
ITEM_FLAGS = {"C1": frozenset(), "N1": frozenset({"kd_logit"}), "N2": frozenset({"sites_on_input_grid", "tf32_off"}),
              "N5": frozenset({"fp32_head"}), "D": frozenset()}
COMPARATORS = {"E6-E2 s42": 0.35132896900177, "E6 s42/43/44": [0.3308, 0.3490, 0.3197],
               "E5 s42/43/44": [0.3343, 0.3192, 0.3328], "E6-KD s42": 0.3321,
               "source": "audit §6.2 (values of record: DL-131, DL-139, DL-142 to DL-146)"}
RECIPE_OF_RECORD = ("E6-E2 (AM-23 item 1 = AM-4a item 1 + clip 5.0): SGD lr 3e-4, momentum 0.9, wd 1e-4, cosine "
                    "T_max 5,025, batch 16 drop_last, 15 epochs, clip 5.0, fake-quant from step 0, BN frozen top of "
                    "epoch 11, observers off top of epoch 13, the VAL bracket, the QNNPACK qconfig of record, TF32 at "
                    "torch defaults; block B: 15 converted epochs scored on CPU, AM-23's selection rules, finalize")
CHANGES = {"C1": "none (seeds 43/44, which AM-23 item 1 excludes)",
           "N1": "logit KD in the QAT loss: lambda 0.5, T 4, E6-KD's ramp and teacher stream; no CWD term",
           "N2": "the ten sites fake-quantized on their input's grid, and TF32 off (cuDNN and matmul)",
           "N5": "the LR-ASPP logit head in FP32 (mixed precision; never INT8)",
           "D": "none: the parent is an exploratory FP32 winner (best VAL >= 0.3875)"}


class ExploreQATError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def item_flag_error(item: str, flags: set, seed: int | None) -> tuple[str, str] | None:
    """The item's one change: exactly its flags; its seeds."""
    if item not in ITEMS:
        return "item_unknown", f"--item {item!r} is not one of {ITEMS}"
    want = ITEM_FLAGS[item]
    if set(flags) != set(want):
        return "item_flags", (f"--item {item} takes exactly the flags {sorted(want) or 'none'}, got {sorted(flags)} "
                              "(one named change per run)")
    if item in ITEM_SEEDS and seed not in ITEM_SEEDS[item]:
        return "item_seed", f"--item {item} runs seed {ITEM_SEEDS[item]}, not {seed!r} (QAT seed = source seed)"
    return None


def d_trigger_met(value) -> bool:
    return isinstance(value, float) and math.isfinite(value) and Decimal(repr(value)) >= D_TRIGGER


def exploratory_parent(item: str, run_dir, *, seed: int, expect_source_sha256: str | None = None,
                       parent_sha256: str | None = None) -> tuple[dict, dict]:
    """The exploratory parent check. Returns (the stage dict to run, the parent record); raises ExploreQATError or the
    pinned QATRefused."""
    from src.quant import qat as Q
    from src.quant.stages import resolve_quant_stage
    if item == "D":
        if parent_sha256 is None or expect_source_sha256 is not None:
            raise ExploreQATError("d_parent_flags", "D takes --parent-sha256 (the exploratory winner's checkpoint) "
                                                    "and no --expect-source-sha256")
        if parent_sha256 in {p["sha256"] for p in E2_PARENTS_OF_RECORD.values()}:
            raise ExploreQATError("d_parent_of_record", "D's parent is an exploratory winner, not an E2 of record")
        d = Path(run_dir).resolve()
        item_dir = d.parent
        if not item_dir.name.startswith(G.RUN_PREFIX) or not (item_dir / G.RUN_META_NAME).is_file():
            raise ExploreQATError("d_parent_not_exploratory", f"{d} is not the trainer folder of an x- item folder "
                                                              f"holding {G.RUN_META_NAME}")
        import json
        xm = json.loads((item_dir / G.RUN_META_NAME).read_text(encoding="utf-8"))
        if xm.get("item") not in D_PARENT_ITEMS or xm.get("amendment") != G.AMENDMENT or xm.get("exploratory") is not True:
            raise ExploreQATError("d_parent_item", f"D's parent must be an exploratory {D_PARENT_ITEMS} run; "
                                                   f"{item_dir.name} records item {xm.get('item')!r}")
        metas = sorted(p.name for p in d.glob("*_run_meta.jsonl"))
        if metas not in (["e2_run_meta.jsonl"], ["e3_run_meta.jsonl"]):
            raise ExploreQATError("d_parent_stage", f"{d} holds run_meta files {metas}; one of e2 or e3 is required")
        stage = resolve_quant_stage("e6-e2" if metas[0].startswith("e2") else "e6")
        parent = Q.resolve_parent(stage, d, parent_sha256, seed)
        if not d_trigger_met(parent["best_val_miou_all_class"]):
            raise ExploreQATError("d_trigger_not_met", f"the parent's best VAL {parent['best_val_miou_all_class']!r} "
                                                       f"is below D's trigger {D_TRIGGER} (a number, never a judgment)")
        return stage, {**parent, "x_item": xm.get("item"), "x_run_id": xm.get("run_id"),
                       "d_trigger": str(D_TRIGGER), "x_run_meta_sha256": G.sha256_file(item_dir / G.RUN_META_NAME)}
    if parent_sha256 is not None:
        raise ExploreQATError("parent_sha256_not_d", "--parent-sha256 is D's flag")
    rec = E2_PARENTS_OF_RECORD.get(seed)
    if rec is None or expect_source_sha256 != rec["sha256"]:
        raise ExploreQATError("parent_not_of_record", f"--expect-source-sha256 {expect_source_sha256!r} is not E2 "
                                                      f"seed {seed}'s best checkpoint of record "
                                                      f"({None if rec is None else rec['sha256']})")
    stage = resolve_quant_stage("e6-e2")
    parent = Q.resolve_parent(stage, run_dir, expect_source_sha256, seed)
    if parent["run_id"] != rec["run_id"] or parent["checkpoint"] != rec["checkpoint"]:
        raise ExploreQATError("parent_identity", f"the parent is {parent['run_id']}/{parent['checkpoint']}, the "
                                                 f"record's E2 seed {seed} is {rec['run_id']}/{rec['checkpoint']}")
    from src.distill.cwd_projection import PROJECTION_FILE
    if (Path(parent["run_dir"]) / PROJECTION_FILE).exists():
        raise ExploreQATError("parent_projection_present", "an E2 run directory holds no projection.pt")
    return stage, {**parent, "of_record": rec}


# ------------------------------------------------------------------ N1: run_qat's loop with logit KD
def run_qat_logit_kd(*, mode: str, model: nn.Module, teacher, source_meta: dict, out_dir, seed: int,
                     clip_norm: float, clip_source: str, device: str, num_workers: int, lambda_logit: float = N1_LAMBDA,
                     loaders=None, parent: dict | None = None, extra_meta: dict | None = None,
                     max_steps: int | None = None, max_val_batches: int | None = None, on_step=None,
                     log=print) -> dict:
    """E6-E2's QAT (src/quant/qat.py run_qat, line for line where nothing differs) plus lambda * L_LogitKD on stage
    e2's term switches, ramped over the first epoch as E6-KD ramps it, the teacher one frozen forward per step after
    the student's, on its private M4-KD stream begun once after set_seed. `mode` "smoke" accepts injected loaders, a
    step cap, a VAL cap and a step callback; "real" none of them."""
    from src.distill.features import StudentTaps
    from src.distill.nmf_stream import M4_NMF_SEED
    from src.quant import qat as Q
    from src.quant.prepare import disable_observers, freeze_bn_stats, prepare_qat_model, qat_freeze_steps, \
        unfused_batchnorm
    from src.quant.qconfig import QuantBackendUnavailable, select_qnnpack_backend
    from src.quant.stages import resolve_quant_stage
    from src.seeds import set_seed
    from src.training import train_distill as TD
    from src.training.losses import CombinedCEDiceLoss
    from src.training.train_e1 import CLASS_WEIGHTS_JSON, _image_digest, load_ce_weights, validate

    st = resolve_quant_stage("e6-e2")
    e2 = TD.resolve_stage("e2")
    if mode not in Q.MODES:
        raise ExploreQATError("mode_invalid", f"mode {mode!r} not in {Q.MODES}")
    if mode == "real" and (loaders is not None or max_steps is not None or max_val_batches is not None
                           or on_step is not None):
        raise ExploreQATError("real_run_overrides", "a real run takes no injected loaders, step cap, VAL cap or hook")
    if mode == "real" and clip_source != "clip_selection":
        raise ExploreQATError("clip_source", f"N1's clip is the U4 winner bound at launch, not {clip_source!r}")
    pins = Q.config_pins_error()
    if pins:
        raise ExploreQATError("config_pins", pins)
    if isinstance(clip_norm, bool) or not isinstance(clip_norm, float) or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ExploreQATError("grad_clip_norm_invalid", f"clip norm must be a positive finite float, got {clip_norm!r}")
    if mode == "real" and lambda_logit != N1_LAMBDA:
        raise ExploreQATError("lambda_not_n1", f"N1's lambda is {N1_LAMBDA}, not {lambda_logit!r}")
    from src.distill import teacher as TT
    if not isinstance(teacher, TT.FrozenTeacher) or teacher.trainable_parameters():
        raise ExploreQATError("teacher_not_frozen", "the teacher must be a FrozenTeacher with no trainable parameter")
    dev = torch.device(device)
    if mode == "real" and dev.type != "cuda":
        raise ExploreQATError("cuda_required", f"a real N1 run needs a CUDA device, got {device!r}")
    cuda_initialized_at_seed = bool(torch.cuda.is_initialized())
    if mode == "real" and cuda_initialized_at_seed:
        raise ExploreQATError("cuda_initialized_before_seed", "CUDA was initialised before set_seed")
    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ExploreQATError("out_dir_not_empty", f"{out} exists and is not an empty directory")
    run_id = out.name

    set_seed(seed)
    if loaders is None:
        train_loader, val_loader = Q.build_qat_loaders(seed, num_workers)
        loader_source = "E1 build_dataloader (train_e1.py:320-322), seed=--seed"
    else:
        train_loader, val_loader = loaders
        loader_source = "injected (smoke)"
    steps_per_epoch = len(train_loader)
    t_max = Q.EPOCHS * steps_per_epoch
    loader_batch = getattr(train_loader, "batch_size", None)
    loader_drop_last = getattr(train_loader, "drop_last", None)
    if mode == "real" and (steps_per_epoch != Q.REAL_STEPS_PER_EPOCH or t_max != Q.REAL_T_MAX):
        raise ExploreQATError("steps_per_epoch", f"steps_per_epoch {steps_per_epoch} / T_max {t_max}")
    if mode == "real" and (loader_batch != Q.BATCH_SIZE or loader_drop_last is not True):
        raise ExploreQATError("loader_of_record", f"batch_size {loader_batch!r}, drop_last {loader_drop_last!r}")
    if steps_per_epoch < 1:
        raise ExploreQATError("empty_loader", "the TRAIN loader yields no batch")
    bn_after_step, obs_after_step = qat_freeze_steps(steps_per_epoch, Q.BN_FREEZE_EPOCH, Q.OBS_FREEZE_EPOCH)
    ramp_iters = steps_per_epoch                                   # E6-KD's ramp: one epoch of the TRAIN loader
    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        raise ExploreQATError("backend_unavailable", str(e)) from e
    prepared = prepare_qat_model(model, select_backend=False)
    unfused = unfused_batchnorm(prepared)
    if unfused and mode == "real":
        raise ExploreQATError("unfused_batchnorm", f"{len(unfused)} BatchNorm module(s) outside a fused QAT ConvBn")
    census = Q.observer_census(prepared)
    fq_count = len(Q.fake_quant_modules(prepared))
    never = Q.never_observed_modules(prepared)
    kinds = Q.state_kinds(prepared)
    convbn = Q.fused_convbn_census(prepared)
    teacher.eval()
    teacher_nmf = teacher.begin_nmf_stream("M4-KD", M4_NMF_SEED)
    if mode == "real" and teacher_nmf is None:
        raise ExploreQATError("teacher_nmf_missing", "the teacher exposes no isolated NMF stream (M4-KD)")
    teacher_mock = bool(getattr(teacher.teacher, "is_mock", False))
    prepared = prepared.to(dev)
    prepared.train()
    teacher = teacher.to(dev)
    teacher.eval()
    weights = load_ce_weights().to(dev)
    criterion = CombinedCEDiceLoss(weight=weights, ignore_index=Q.IGNORE_INDEX).to(dev)
    optimizer = torch.optim.SGD(prepared.parameters(), lr=Q.LEARNING_RATE, momentum=Q.MOMENTUM,
                                weight_decay=Q.WEIGHT_DECAY, dampening=Q.DAMPENING, nesterov=Q.NESTEROV)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=Q.ETA_MIN)
    opt_rec = Q.optimizer_record(optimizer, prepared)
    opt_err = Q.optimizer_of_record_error(opt_rec)
    if opt_err:
        raise Q.QATStop("optimizer_of_record", opt_err)
    if {id(p) for p in prepared.parameters()} & teacher.parameter_ids():
        raise ExploreQATError("teacher_in_optimizer", "a teacher parameter reached the optimizer")
    ramps = TD.build_term_ramps(e2, ramp_iters)
    if tuple(ramps) != ("logit_kd",):
        raise ExploreQATError("terms_not_e2", f"stage e2 instantiates {tuple(ramps)}, not ('logit_kd',)")

    from src.quant.ptq import qconfig_summary
    from src.quant.qconfig import qat_qconfig
    qsum = qconfig_summary(qat_qconfig())
    identity = Q.code_identity()
    import numpy
    import sys
    import torchvision
    from configs.distill import LOGIT_KD_SEMANTICS
    tel_path = out / Q.TELEMETRY_NAME
    run_meta = {
        "event": "run_meta", "wall_clock": time.time(), "schema": Q.TELEMETRY_SCHEMA, "lane": Q.LANE,
        "mode": mode, "stage": st["name"], "source_stage": st["source_stage"], "seed": seed,
        "run_id": run_id, "out_dir": str(out.resolve()),
        "parent": parent, "source_checkpoint_sha256": source_meta["sha256"],
        "source_checkpoint": source_meta.get("path"), "source_checkpoint_bytes": source_meta.get("bytes"),
        "clip_norm": clip_norm, "clip_source": clip_source,
        "epochs": Q.EPOCHS, "batch_size": loader_batch, "drop_last": loader_drop_last,
        "gradient_accumulation": None, "ema": False, "distillation": True,
        "steps_per_epoch": steps_per_epoch, "total_steps": t_max,
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "loader": loader_source, "num_workers": num_workers, "val_batches_cap": max_val_batches,
        "persistent_workers": bool(getattr(train_loader, "persistent_workers", False)),
        "scheduler": {"class": type(scheduler).__name__, "T_max": scheduler.T_max,
                      "eta_min": scheduler.eta_min, "step": "once after every optimizer.step"},
        "optimizer": opt_rec, "lr": opt_rec["lr"], "momentum": opt_rec["momentum"],
        "weight_decay": opt_rec["weight_decay"], "nesterov": opt_rec["nesterov"], "dampening": opt_rec["dampening"],
        "bn_freeze_epoch": Q.BN_FREEZE_EPOCH, "obs_freeze_epoch": Q.OBS_FREEZE_EPOCH,
        "bn_frozen_from_epoch": Q.BN_FREEZE_EPOCH + 1, "obs_frozen_from_epoch": Q.OBS_FREEZE_EPOCH + 1,
        "bn_freeze_after_step": bn_after_step, "obs_freeze_after_step": obs_after_step,
        "objective": (f"supervised CE (E1 class weights) + Dice + r(step) * {lambda_logit} * L_LogitKD (stage e2's "
                      "term; T 4; no CWD term) -- lane L-AM24 N1"),
        "class_weights_sha256": Q.sha256_file(CLASS_WEIGHTS_JSON), "ignore_index": Q.IGNORE_INDEX,
        "engine": torch.backends.quantized.engine, "qconfig": qsum,
        "observers": {"activation": qsum["activation"]["observer_class"], "weight": qsum["weight"]["observer_class"]},
        "observer_census": census, "fake_quant_modules": fq_count, "never_observed_modules": never,
        "observer_freeze_scope": Q.OBSERVER_FREEZE_SCOPE, "fused_convbn": convbn, "unfused_batchnorm": len(unfused),
        "tf32": Q.tf32_state(), "determinism": Q.determinism_state(),
        "cuda_initialized_at_seed": cuda_initialized_at_seed, "device": str(dev),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": numpy.__version__,
        "python": sys.version.split()[0], "image_digest": _image_digest(), "host": Q.host_identity(),
        "git_head": identity["git_head"], "code_clean_at_head": identity["code_clean_at_head"],
        "code_files_sha256": identity["code_files_sha256"], "cwd_projection_loaded": False,
        "terms": {t: bool(e2[t]) for t in TD.TERMS}, "lambda_logit": lambda_logit, "T_logit": TD.T_LOGIT,
        "logit_kd_semantics": LOGIT_KD_SEMANTICS, "ramp_iters": ramp_iters,
        "ramp_rule": "r(step) = clamp((step - 1)/(ramp_iters - 1), 0, 1) over the global QAT step (E6-KD's)",
        "teacher_provenance": None if teacher.provenance is None else teacher.provenance.as_dict(),
        "teacher_nmf": teacher_nmf, "teacher_mock": teacher_mock,
    }
    if extra_meta:
        run_meta.update(extra_meta)
    out.mkdir(parents=True, exist_ok=True)
    tel = Q.Telemetry(tel_path)
    tel.write(run_meta)
    log(f"[n1] {st['name']}+logitKD seed {seed} clip {clip_norm} lambda {lambda_logit} | {steps_per_epoch} steps/epoch, "
        f"T_max {t_max}, ramp {ramp_iters} | {run_id}")

    step, epoch, nonfinite_since = 0, 0, None

    def _mark_nonfinite(at_step: int, fails: list[str]) -> None:
        nonlocal nonfinite_since
        nonfinite_since = at_step
        tel.write({"event": "state_nonfinite", "wall_clock": time.time(), "epoch": epoch, "step": at_step,
                   "nonfinite_since_step": at_step, "failing": fails})

    try:
        for epoch in range(1, Q.EPOCHS + 1):
            if epoch == Q.BN_FREEZE_EPOCH + 1:
                freeze_bn_stats(prepared)
            if epoch == Q.OBS_FREEZE_EPOCH + 1:
                disable_observers(prepared)
            ep = {"nonfinite_loss": 0, "nonfinite_grad": 0, "step_errors": 0, "clipped": 0,
                  "max_pre_clip_norm": None, "steps": 0}
            for img, mask in train_loader:
                if max_steps is not None and step >= max_steps:
                    break
                step += 1
                ep["steps"] += 1
                t0 = time.time()
                fp = Q.batch_sha256(mask)
                lr = optimizer.param_groups[0]["lr"]
                bn_frozen, obs_on = epoch > Q.BN_FREEZE_EPOCH, epoch <= Q.OBS_FREEZE_EPOCH
                try:
                    img_d, mask_d = img.to(dev), mask.to(dev)
                    optimizer.zero_grad(set_to_none=True)
                    with StudentTaps(prepared) as taps:
                        logits = prepared(img_d)
                        c5, head_logits = taps.require()
                    teacher_out = teacher(img_d)                 # one frozen forward per step, after the student's
                    ramp = ramps["logit_kd"](step)
                    ce = criterion.ce(logits, mask_d)
                    dice = criterion.dice(logits, mask_d)
                    sup = ce + dice
                    distill, parts = TD.distillation_losses(
                        stage=e2, logits=logits, head_logits=head_logits, c5=c5, mask=mask_d,
                        teacher_out=teacher_out, projection=None, lambda_logit=lambda_logit,
                        ramp={"logit_kd": ramp})
                    loss = sup + distill
                    if step == 1:
                        checks = {"terms_logit_kd_only": set(parts) == {"logit_kd"},
                                  "head_logits_os8": tuple(head_logits.shape[-2:]) == tuple(
                                      s // 8 for s in img_d.shape[-2:]),
                                  "teacher_logits_on_head_grid": tuple(teacher_out.logits.shape) == tuple(
                                      head_logits.shape),
                                  "distill_zero_at_step1": (float(distill) == 0.0) if ramp_iters > 1 else True,
                                  "teacher_frozen": not teacher.trainable_parameters()}
                        tel.write({"event": "step1_checks", "wall_clock": time.time(), "epoch": epoch, "step": step,
                                   "checks": checks, "passed": all(checks.values())})
                        if mode == "real" and not all(checks.values()):
                            raise Q.QATStop("step1_checks", f"N1 step-1 checks failed: {checks}")
                    if on_step is not None:
                        on_step(step, {"loss": loss, "ce": ce, "dice": dice, "distill": distill, "parts": parts,
                                       "ramp": ramp, "head_logits": head_logits, "teacher_out": teacher_out,
                                       "mask": mask_d, "logits": logits})
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,
                                                           norm_type=2.0, error_if_nonfinite=False)
                    optimizer.step()
                except Q.QATStop:
                    raise
                except Exception as e:                       # noqa: BLE001 -- AM-21 item 1(b), as run_qat
                    step_ok, step_fails = Q.state_predicate(Q.cpu_state(prepared), kinds, never)
                    if step_ok:
                        raise
                    if nonfinite_since is None:
                        _mark_nonfinite(step, step_fails)
                    scheduler.step()
                    ep["step_errors"] += 1
                    tel.write({"event": "step_error", "wall_clock": time.time(), "epoch": epoch, "step": step,
                               "epoch_step": ep["steps"], "lr": lr, "error": f"{type(e).__name__}: {str(e)[:300]}",
                               "batch_sha256": fp, "state_nonfinite": True})
                    continue
                scheduler.step()
                loss_v, ce_v, dice_v, gn = float(loss.item()), float(ce.item()), float(dice.item()), float(gnorm.item())
                if not math.isfinite(loss_v):
                    ep["nonfinite_loss"] += 1
                if not math.isfinite(gn):
                    ep["nonfinite_grad"] += 1
                clipped = bool(math.isfinite(gn) and gn > clip_norm)
                ep["clipped"] += int(clipped)
                if math.isfinite(gn):
                    prev = ep["max_pre_clip_norm"]
                    ep["max_pre_clip_norm"] = gn if prev is None else max(prev, gn)
                if nonfinite_since is None and (not math.isfinite(loss_v) or not math.isfinite(gn)):
                    ok, fails = Q.state_predicate(prepared.state_dict(), kinds, never)
                    if not ok:
                        _mark_nonfinite(step, fails)
                tel.write({"event": "train", "wall_clock": time.time(), "epoch": epoch, "step": step,
                           "epoch_step": ep["steps"], "lr": lr, "loss": loss_v, "ce": ce_v, "dice": dice_v,
                           "sup": float(sup.item()), "logit_kd": parts.get("logit_kd"), "ramp": ramp,
                           "grad_norm": gn, "clipped": clipped, "bn_frozen": bn_frozen, "observers_enabled": obs_on,
                           "batch_sha256": fp, "state_nonfinite": nonfinite_since is not None,
                           "iter_seconds": time.time() - t0})
            if max_steps is not None and step >= max_steps and ep["steps"] == 0:
                break
            state = Q.cpu_state(prepared)
            state_sha = Q.state_digest(state)
            finite, fails = Q.state_predicate(state, kinds, never)
            if not finite and nonfinite_since is None:
                _mark_nonfinite(step, fails)
            ck_path, ck_sha, ck_bytes = Q.save_epoch_checkpoint(
                out, stage_name=st["name"], source_stage=st["source_stage"], epoch=epoch, step=step, state=state,
                state_sha256=state_sha, seed=seed, clip_norm=clip_norm, run_id=run_id,
                source_sha256=source_meta["sha256"], state_finite=bool(finite), nonfinite_since_step=nonfinite_since)
            flags = Q.observer_flags(prepared)
            disable_observers(prepared)
            t_val = time.time()
            val_error = None
            try:
                all_miou, disease_miou, cm, n_val_batches = validate(prepared, val_loader, dev, Q.NUM_CLASSES,
                                                                     max_val_batches)
            except Exception as e:                           # noqa: BLE001
                val_ok, _ = Q.state_predicate(state, kinds, never)
                if val_ok:
                    Q.restore_observer_flags(prepared, flags)
                    raise
                prepared.train()
                all_miou = disease_miou = None
                cm, n_val_batches = None, 0
                val_error = f"{type(e).__name__}: {str(e)[:300]}"
            del state
            Q.restore_observer_flags(prepared, flags)
            val_seconds = time.time() - t_val
            if Q.state_digest(prepared) != state_sha:
                raise Q.QATStop("val_bracket_changed_state", f"epoch {epoch}: the VAL bracket changed the state")
            val_row = {"event": "val", "wall_clock": time.time(), "epoch": epoch, "step": step, "kind": "fake_quant",
                       "all_class_miou": all_miou, "disease_only_miou_PROVISIONAL": disease_miou,
                       "val_batches": n_val_batches, "val_total_px": int(cm.sum()) if cm is not None else None,
                       "val_seconds": val_seconds, "observers_disabled_during_val": True, "state_unchanged": True,
                       "selects": False, "state_nonfinite": nonfinite_since is not None}
            if val_error is not None:
                val_row["error"] = val_error
            tel.write(val_row)
            nmf = teacher.nmf_stream_state()
            tel.write({"event": "epoch_end", "wall_clock": time.time(), "epoch": epoch, "step": step,
                       "steps_in_epoch": ep["steps"], "checkpoint": f"{Q.EPOCH_DIR}/{ck_path.name}",
                       "checkpoint_sha256": ck_sha, "checkpoint_bytes": ck_bytes, "state_sha256": state_sha,
                       "fake_quant_val_all_class_miou": all_miou, "bn_frozen": epoch > Q.BN_FREEZE_EPOCH,
                       "observers_enabled": epoch <= Q.OBS_FREEZE_EPOCH, "nonfinite_loss_steps": ep["nonfinite_loss"],
                       "nonfinite_grad_steps": ep["nonfinite_grad"], "step_errors": ep["step_errors"],
                       "clipped_steps": ep["clipped"], "max_pre_clip_norm": ep["max_pre_clip_norm"],
                       "state_finite": bool(finite), "nonfinite_since_step": nonfinite_since,
                       "state_nonfinite": nonfinite_since is not None,
                       "teacher_nmf_draws": None if nmf is None else nmf["draws"],
                       "run_complete": epoch == Q.EPOCHS and max_steps is None})
            log(f"[n1] epoch {epoch:02d}/{Q.EPOCHS} step {step} fake-quant VAL {all_miou}")
            if max_steps is not None and step >= max_steps:
                break
    except Q.QATStop as e:
        tel.write({"event": "run_stop", "wall_clock": time.time(), "epoch": epoch, "step": step, "code": e.code,
                   "message": str(e)[:500], "state_nonfinite": nonfinite_since is not None})
        raise
    except BaseException as e:
        tel.write({"event": "run_abort", "wall_clock": time.time(), "epoch": epoch, "step": step,
                   "cause": "exception", "exception_type": type(e).__name__, "message": str(e)[:500],
                   "state_nonfinite": nonfinite_since is not None})
        raise
    complete = epoch == Q.EPOCHS and max_steps is None
    return {"run_id": run_id, "stage": st["name"], "seed": seed, "clip_norm": clip_norm, "telemetry": str(tel_path),
            "steps": step, "epochs_completed": epoch, "complete": complete, "nonfinite_since_step": nonfinite_since}
