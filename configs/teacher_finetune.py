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

    # Success criterion (protocol match, not max accuracy)
    "success_criterion": "recover 42.05% mIoU within +/-1.5-2.0 pp",

    # Role
    "role": "descriptive upper-bound reference only (not deployed, not an inferential comparator)",
}
