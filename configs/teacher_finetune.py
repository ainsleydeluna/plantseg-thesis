# Teacher fine-tune config — Blocker B1
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B1 - Teacher fine-tune"
# Every value traced to ch3.pdf (method authority); versions corroborated by context.md.
# Analysis/config artifact only — contains NO training logic.

TEACHER_FINETUNE = {
    # Base / initialization
    "model": "SegNeXt-B / MSCAN-B",
    "init_checkpoint": "segnext_mscan-b_512x512_160k_ade20k",  # ADE20K-pretrained, MMSeg zoo (or equivalent)
    "framework": "MMSegmentation 1.2.2 + mmcv 2.1.0",

    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 6e-5,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "decode_head_lr_mult": 10,

    # Schedule / budget
    "lr_schedule": "poly",
    "iterations": 40000,
    "batch_size": 16,
    "crop": (512, 512),

    # Loss
    "loss": "cross_entropy",

    # Success criterion — METHODOLOGY DECISION OPEN (B59 B3, 2026-09-21). This recipe is a
    # THESIS-DERIVED SegNeXt-B teacher configuration (AdamW + ADE20K init). It is NOT the Wei et al.
    # (2026) protocol that produced 42.05 (Wei: SGD lr 1e-3, momentum 0.9, wd 5e-4), so ch3's
    # "recover 42.05% within +/-1.5-2.0 pp" is not a protocol-match test. Wei 42.05 mIoU / 56.30 mAcc /
    # ~28M params are contextual published values only.
    "success_criterion": "NEED_TO_CONFIRM - teacher acceptance band is a methodology decision open",
    "published_reference_values": "Wei 2026: 42.05 mIoU / 56.30 mAcc / ~28M params (contextual only)",

    # Role
    "role": "descriptive upper-bound reference only (not deployed, not an inferential comparator)",
}
