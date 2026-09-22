# Teacher fine-tune config — Blocker B1
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B1 - Teacher fine-tune"
# Every value traced to ch3.pdf (method authority) and to the B60/B61 methodology locks; versions
# corroborated by context.md. The executable MMSeg config is
# configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py.
# Analysis/config artifact only — contains NO training logic.

TEACHER_FINETUNE = {
    # Base / initialization — operative MMSeg 1.x identity (G10, B62; readiness PASS, B61 §1)
    "model": "SegNeXt-B / MSCAN-B",
    "init_checkpoint": {
        "config": "segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512",   # MMSeg 1.x, ADE20K, 150 classes
        "filename": "segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth",
        "sha256": "647a0cda7678a35396689a4f8e9fddc33a088d8b539195d0dc97485ab8640ef1",
        "location": "outside the repository, supplied via SEGNEXT_ADE20K_CKPT",
        "classifier_only_mismatch": ("decode_head.conv_seg.weight", "decode_head.conv_seg.bias"),
    },
    "framework": "MMSegmentation 1.2.2 + mmcv 2.1.0",

    # Optimizer (M5)
    "optimizer": "AdamW",
    "learning_rate": 6e-5,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "decode_head_lr_mult": 10,

    # Schedule / budget (M5)
    "lr_schedule": "LinearLR warmup 1,500 iterations (start_factor 1e-6), then PolyLR power 1.0 to 40,000",
    "iterations": 40000,
    "batch_size": 16,
    "crop": (512, 512),
    "validation_interval": 4000,
    "checkpoint_interval": 4000,

    # Data (M2, M3, M11)
    "augmentation": "M2: E1-E3 recipe reused (src/data/transforms.train_preprocess + configs/augment.AUGMENT)",
    "train_scale": "M3: long side = round(512*r), r ~ U[0.75, 2.0]",
    "evaluation_canvas": "M3: long side 512 + symmetric pad to 512x512 (core_preprocess)",
    "data_root": "M11: staged with TRAIN + VAL only; images/test, annotations/test, annotation_test.json absent",

    # Loss (M5, M13)
    "loss": "cross_entropy, unweighted, ignore_index=255, avg_non_ignore=True",

    # NMF / checkpoint selection (M4, M12)
    "nmf_control": "M4: rand_init=True; M4-T upstream; M4-V seed 42 per eval pass; M4-KD seed 42 once per run",
    "checkpoint_selection": "M12: highest full-precision VAL all-class mIoU over 4k..40k; exact tie -> earliest",

    # Success criterion — LOCKED (M5, B60 §2.4): readiness rule R1-R4, not a published-value band.
    # R3: controlled deterministic re-evaluation of the selected checkpoint under M4-V, VAL all-class
    # mIoU strictly greater than E1's 0.36314016580581665, no margin; failure -> STOP.
    "success_criterion": "readiness rule R1-R4 (M5, B60); R3 floor 0.36314016580581665 on VAL",
    "published_reference_values": "Wei 2026: 42.05 mIoU / 56.30 mAcc / ~28M params (contextual only)",

    # Role
    "role": "descriptive upper-bound reference only (not deployed, not an inferential comparator)",
}
