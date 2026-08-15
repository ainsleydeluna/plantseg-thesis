# SegNeXt-B / MSCAN-B teacher fine-tune on PlantSeg (116 classes) — MMSegmentation 1.2.2 config.
#
# Authority: docs/teacher_prep_runbook.md §3 (derived-config spec) + docs/IMPLEMENTATION_CONTRACT.md
# B1/B6 + configs/teacher_finetune.py. This is the ONE framework-syntax config in the repository; the
# other configs/*.py files are plain analysis dicts and are unaffected.
#
# Written for the PINNED stack: MMSegmentation 1.2.2 + mmcv 2.1.0 + mmengine (MMSeg **1.x**
# conventions: `optim_wrapper`, `param_scheduler`, `train_dataloader`, `IterBasedTrainLoop`,
# `PackSegInputs`). No MMSeg 0.x keys (`data`, `optimizer_config`, `lr_config`, `runner`,
# `total_iters`, `samples_per_gpu`) appear anywhere below.
#
# TEACHER SCOPE — this file trains the descriptive upper-bound teacher only. It contains no KD, no
# CWD, no QAT and no PTQ; those belong to the student stages (configs/distill.py, configs/quant.py).
# The teacher objective is cross-entropy ONLY — the student's CE+Dice (B5) is deliberately NOT used.
#
# TEST SPLIT — `test_dataloader` below exists solely so the inherited ADE20K test config is replaced
# rather than left dangling, and for a later descriptive evaluation. It is NEVER touched by
# `runner.train()`: training reads images/train, and validation + checkpoint selection
# (`save_best='mIoU'`) read images/val only. The public PlantSeg repository points its
# `val_dataloader` at the TEST split; that behaviour is deliberately NOT copied.
#
# ============================================================================================
# PROVENANCE — THIS IS A **THESIS-DERIVED** CONFIGURATION, NOT A PUBLISHED-B REPRODUCTION
# ============================================================================================
# No MSCAN-B PlantSeg 40k config exists in the public PlantSeg repository at the pinned source
# commit 1a3dd4d9224bcc97a5850af7dd1c423abc24eae0 (it ships MSCAN-B ADE20K 160k, MSCAN-L PlantSeg
# 40k and MSCAN-T PlantSeg 40k only; run.sh launches the L config). This file is therefore composed:
#
#   * MSCAN-B architecture/base  <- public SegNeXt MSCAN-B **ADE20K** config (via `_base_`)
#   * SegNeXt-on-PlantSeg conventions <- public PlantSeg SegNeXt-family configs (train augmentation)
#   * optimization recipe        <- THESIS-locked B1 recipe (pre-registered; runbook §3)
#   * 116-class output space     <- THESIS (configs/plantseg_class_map.json)
#   * train/val/test discipline  <- THESIS (val for selection; test held out)
#   * core preprocessing         <- THESIS (runbook §11 teacher-student parity)
#
# The published PlantSeg benchmark for SegNeXt MSCAN-B is 42.05% mIoU / 56.30% mAcc / 28M params.
# That number is a CONTEXTUAL REFERENCE ONLY. The paper reports all benchmark methods trained with
# SGD (lr 1e-3, momentum 0.9, weight decay 5e-4, CE, batch 16); this configuration deliberately uses
# the thesis's pre-registered AdamW recipe instead, so a run of this file is NOT an exact
# reproduction of the published 42.05 protocol and must never be described as one.

import os

_base_ = ['mmseg::segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py']

# Machine-checkable provenance markers (asserted by scripts/smoke_teacher_config.py).
PROTOCOL_CLASSIFICATION = 'thesis-derived'
PUBLIC_PLANTSEG_SOURCE_COMMIT = '1a3dd4d9224bcc97a5850af7dd1c423abc24eae0'
PUBLISHED_MSCAN_B_REFERENCE = dict(miou=42.05, macc=56.30, params_m=28, role='contextual-reference-only')
PUBLISHED_BENCHMARK_OPTIMIZER = 'SGD lr=1e-3 momentum=0.9 wd=5e-4 (paper) — NOT used here'
AUGMENTATION_SOURCE = 'public-plantseg-segnext-family'   # teacher-specific; §11 governs preprocessing only
CORE_PREPROCESSING_SOURCE = 'thesis-parity-runbook-s11'

# --------------------------------------------------------------------------------------------
# Class space — verbatim from configs/plantseg_class_map.json (upstream
# PlantSeg115Dataset.METAINFO["classes"], commit 1a3dd4d9). Index 0 is the upstream background
# entry and its name is the empty string upstream; it is preserved exactly rather than renamed.
# --------------------------------------------------------------------------------------------
NUM_CLASSES = 116
IGNORE_INDEX = 255
BACKGROUND_INDEX = 0

PLANTSEG_CLASSES = (
    '',                                            #   0
    'apple black rot',                             #   1
    'apple mosaic virus',                          #   2
    'apple rust',                                  #   3
    'apple scab',                                  #   4
    'banana anthracnose',                          #   5
    'banana black leaf streak',                    #   6
    'banana bunchy top',                           #   7
    'banana cigar end rot',                        #   8
    'banana cordana leaf spot',                    #   9
    'banana panama disease',                       #  10
    'basil downy mildew',                          #  11
    'bean halo blight',                            #  12
    'bean mosaic virus',                           #  13
    'bean rust',                                   #  14
    'bell pepper bacterial spot',                  #  15
    'bell pepper blossom end rot',                 #  16
    'bell pepper frogeye leaf spot',               #  17
    'bell pepper powdery mildew',                  #  18
    'blueberry anthracnose',                       #  19
    'blueberry botrytis blight',                   #  20
    'blueberry mummy berry',                       #  21
    'blueberry rust',                              #  22
    'blueberry scorch',                            #  23
    'broccoli alternaria leaf spot',               #  24
    'broccoli downy mildew',                       #  25
    'broccoli ring spot',                          #  26
    'cabbage alternaria leaf spot',                #  27
    'cabbage black rot',                           #  28
    'cabbage downy mildew',                        #  29
    'carrot alternaria leaf blight',               #  30
    'carrot cavity spot',                          #  31
    'carrot cercospora leaf blight',               #  32
    'cauliflower alternaria leaf spot',            #  33
    'cauliflower bacterial soft rot',              #  34
    'celery anthracnose',                          #  35
    'celery early blight',                         #  36
    'cherry leaf spot',                            #  37
    'cherry powdery mildew',                       #  38
    'citrus canker',                               #  39
    'citrus greening disease',                     #  40
    'coffee berry blotch',                         #  41
    'coffee black rot',                            #  42
    'coffee brown eye spot',                       #  43
    'coffee leaf rust',                            #  44
    'corn gray leaf spot',                         #  45
    'corn northern leaf blight',                   #  46
    'corn rust',                                   #  47
    'corn smut',                                   #  48
    'cucumber angular leaf spot',                  #  49
    'cucumber bacterial wilt',                     #  50
    'cucumber powdery mildew',                     #  51
    'eggplant cercospora leaf spot',               #  52
    'eggplant phomopsis fruit rot',                #  53
    'eggplant phytophthora blight',                #  54
    'garlic leaf blight',                          #  55
    'garlic rust',                                 #  56
    'ginger leaf spot',                            #  57
    'ginger sheath blight',                        #  58
    'grape black rot',                             #  59
    'grape downy mildew',                          #  60
    'grape leaf spot',                             #  61
    'grapevine leafroll disease',                  #  62
    'lettuce downy mildew',                        #  63
    'lettuce mosaic virus',                        #  64
    'maple tar spot',                              #  65
    'peach anthracnose',                           #  66
    'peach brown rot',                             #  67
    'peach leaf curl',                             #  68
    'peach rust',                                  #  69
    'peach scab',                                  #  70
    'plum bacterial spot',                         #  71
    'plum brown rot',                              #  72
    'plum pocket disease',                         #  73
    'plum pox virus',                              #  74
    'plum rust',                                   #  75
    'potato early blight',                         #  76
    'potato late blight',                          #  77
    'raspberry fire blight',                       #  78
    'raspberry gray mold',                         #  79
    'raspberry leaf spot',                         #  80
    'raspberry yellow rust',                       #  81
    'rice blast',                                  #  82
    'rice sheath blight',                          #  83
    'soybean bacterial blight',                    #  84
    'soybean brown spot',                          #  85
    'soybean downy mildew',                        #  86
    'soybean frog eye leaf spot',                  #  87
    'soybean mosaic',                              #  88
    'soybean rust',                                #  89
    'squash powdery mildew',                       #  90
    'strawberry anthracnose',                      #  91
    'strawberry leaf scorch',                      #  92
    'tobacco blue mold',                           #  93
    'tobacco brown spot',                          #  94
    'tobacco frogeye leaf spot',                   #  95
    'tobacco mosaic virus',                        #  96
    'tomato bacterial leaf spot',                  #  97
    'tomato early blight',                         #  98
    'tomato late blight',                          #  99
    'tomato leaf mold',                            # 100
    'tomato mosaic virus',                         # 101
    'tomato septoria leaf spot',                   # 102
    'tomato yellow leaf curl virus',               # 103
    'wheat bacterial leaf streak (black chaff)',   # 104
    'wheat head scab',                             # 105
    'wheat leaf rust',                             # 106
    'wheat loose smut',                            # 107
    'wheat powdery mildew',                        # 108
    'wheat septoria blotch',                       # 109
    'wheat stem rust',                             # 110
    'wheat stripe rust',                           # 111
    'zucchini bacterial wilt',                     # 112
    'zucchini downy mildew',                       # 113
    'zucchini powdery mildew',                     # 114
    'zucchini yellow mosaic virus',                # 115
)

# --------------------------------------------------------------------------------------------
# Dataset — portable root, identical split layout to the student (configs/data.py)
# --------------------------------------------------------------------------------------------
PLANTSEG_DATA_ROOT_ENV = 'PLANTSEG_DATA_ROOT'
data_root = os.environ.get(PLANTSEG_DATA_ROOT_ENV, '')

dataset_type = 'BaseSegDataset'   # stock mmseg 1.x dataset; class space supplied via metainfo
crop_size = (512, 512)
metainfo = dict(classes=PLANTSEG_CLASSES)

# reduce_zero_label MUST stay False: PlantSeg masks are already 0-115 with background kept at 0.
_dataset_common = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    reduce_zero_label=False,
)

# TRAIN AUGMENTATION — teacher-specific, taken from the public PlantSeg SegNeXt-family configs.
# Runbook §11 mandates teacher/student parity for CORE PREPROCESSING only; it says nothing about
# augmentation, so the student's B2 recipe (vertical flip, +/-10 deg rotation, hue/saturation) is
# deliberately NOT imposed here. Every value below is source-backed, not invented.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

# VAL/TEST CORE PREPROCESSING — thesis parity (runbook §11: "512x512, aspect-preserving resize +
# pad, ImageNet mean/std normalization"). `scale=(512, 512)` with keep_ratio makes the LONG side 512
# (mmcv rescales by min(512/long, 512/short)), matching configs/data.py's "aspect-ratio preserving,
# long side -> 512". Padding to 512x512 is done by the data preprocessor below.
# NOTE the deliberate difference from the upstream PlantSeg/ADE20K eval scale (2048, 512), which
# would make the SHORT side 512 and does NOT match the thesis contract.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=16,                 # effective batch size 16, single GPU
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        **_dataset_common,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **_dataset_common,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline),
)

# Later descriptive evaluation ONLY — never read by training or checkpoint selection.
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **_dataset_common,
        data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# --------------------------------------------------------------------------------------------
# Model deltas — plain BN (never SyncBN), 116-class head, CE-only loss, no competing backbone init
# --------------------------------------------------------------------------------------------
norm_cfg = dict(type='BN', requires_grad=True)

# CORE PREPROCESSING — thesis parity (runbook §11).
# mean/std are configs/data.py's (0.485,0.456,0.406)/(0.229,0.224,0.225) expressed in 0-255 terms.
#
# IMAGE PADDING: `SegDataPreProcessor` NORMALISES FIRST and pads afterwards (`stack_batch` runs on
# already-normalised tensors), so `pad_val=0` fills the pad region with zero IN NORMALISED SPACE —
# which is exactly "padded with the ImageNet mean" in raw space, i.e. configs/data.py's
# image_pad_value (124,116,104). This is NOT a scalar-zero raw-pixel pad. The only difference from
# the student is the student's 8-bit rounding of the mean (124 vs 123.675, 116 vs 116.28, 104 vs
# 103.53), which lands the student's pad at ~(+0.006,-0.005,+0.008) normalised instead of exactly 0
# — and the pad region is masked out by seg_pad_val=255 anyway, so it enters neither loss nor metric.
PAD_MODE = 'imagenet-mean-equivalent-post-normalisation'
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=IGNORE_INDEX,
    size=crop_size,
)

model = dict(
    data_preprocessor=data_preprocessor,
    # Only the BACKBONE carries SyncBN in the stock recipe (the LightHamHead uses GroupNorm), so the
    # BN override targets the backbone. init_cfg=None so the IN-1K backbone is NOT re-pulled: the
    # full ADE20K segmentation model arrives via `load_from` (runbook §5).
    backbone=dict(norm_cfg=norm_cfg, init_cfg=None),
    decode_head=dict(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
)

# --------------------------------------------------------------------------------------------
# Optimizer / schedule — locked B1 recipe
# --------------------------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.),
        'head': dict(lr_mult=10.),          # decode-head lr_mult = 10
    }),
)

# Schedule. The runbook pins only "poly, 40,000 iters" (§3) — it fixes neither the power, the warmup
# nor the validation cadence, so each is resolved from the strongest source-backed teacher evidence
# and made explicit here rather than left implicit.
MAX_ITERS = 40000        # THESIS-locked (runbook §3); teacher is EXEMPT from the 80k student budget
WARMUP_ITERS = 1500      # source-backed: public PlantSeg SegNeXt-family LinearLR warmup 0 -> 1500
POLY_POWER = 1.0         # source-backed: public PlantSeg SegNeXt-family override (NOT the 0.9 of the
                         # generic schedule_40k.py base)
VAL_INTERVAL = 10000     # source-backed: public schedule_40k.py val/checkpoint interval

# HORIZON CORRECTION, stated openly: the public SegNeXt PlantSeg override leaves `end=160000`
# (carried over from the 160k ADE20K schedule) while its train loop stops at 40k, so the poly decay
# there only ever reaches 75% of base LR. This config sets end=MAX_ITERS so the decay actually
# completes over the run. The SHAPE is copied from the public family; the HORIZON is corrected.
POLY_END_SOURCE = 'horizon-corrected-to-max-iters'
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=WARMUP_ITERS),
    dict(type='PolyLR', eta_min=0.0, power=POLY_POWER, begin=WARMUP_ITERS, end=MAX_ITERS,
         by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    # Checkpoint selection is on VALIDATION mIoU. The test split never participates.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=VAL_INTERVAL,
                    save_best='mIoU', rule='greater'),
)

# --------------------------------------------------------------------------------------------
# Determinism, initialization and output paths
# --------------------------------------------------------------------------------------------
randomness = dict(seed=42, deterministic=True)

# NMF / Hamburger randomness: MMSeg's LightHamHead exposes NO seed key of its own. It is governed by
# the global torch RNG seeded above, inside mmseg/models/decode_heads/ham_head.py :: NMF2D._build_bases.
# A dedicated control remains NEED_TO_CONFIRM, to be settled against the pinned stack; it is
# deliberately NOT faked as a config key MMSeg would silently ignore.
NMF_SEED_CONTROL = 'NEED_TO_CONFIRM'

# REQUIRED external ADE20K initialization. No silent random init and no in-run download: if the env
# var is unset the sentinel below is not a real path, so the launch fails loudly.
SEGNEXT_ADE20K_CKPT_ENV = 'SEGNEXT_ADE20K_CKPT'
ADE20K_CKPT_UNSET_SENTINEL = 'NEED_TO_CONFIRM__SET_SEGNEXT_ADE20K_CKPT'
load_from = os.environ.get(SEGNEXT_ADE20K_CKPT_ENV, ADE20K_CKPT_UNSET_SENTINEL)
resume = False

# Trained teacher checkpoints live OUTSIDE the repository (runbook §10).
TEACHER_WORK_DIR_ENV = 'TEACHER_WORK_DIR'
WORK_DIR_UNSET_SENTINEL = 'NEED_TO_CONFIRM__SET_TEACHER_WORK_DIR'
work_dir = os.environ.get(TEACHER_WORK_DIR_ENV, WORK_DIR_UNSET_SENTINEL)
