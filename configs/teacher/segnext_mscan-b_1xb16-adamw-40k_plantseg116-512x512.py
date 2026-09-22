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
# METHODOLOGY LOCKS IMPLEMENTED HERE (B62): M2 augmentation, M3 scale, M4 NMF control, M5 schedule,
# M11 TRAIN/VAL-only isolation, M12 checkpoint selection, M13 CE ignore normalisation
# (reports/b60_teacher_methodology_lock.md, reports/b61_teacher_nmf_checkpoint_selection_lock.md).
#
# TEST SPLIT (M11) — there is no active TEST surface: `test_dataloader`, `test_evaluator` and
# `test_cfg` are None, and the staged data root holds TRAIN and VAL only. Training reads images/train;
# validation and checkpoint selection read images/val. The public PlantSeg repository points its
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
#   * train augmentation + scale <- THESIS M2/M3: the E1-E3 recipe, reused from src/data/transforms.py
#   * optimization recipe        <- THESIS-locked B1 recipe (M5)
#   * 116-class output space     <- THESIS (configs/plantseg_class_map.json)
#   * train/val discipline       <- THESIS M11 (TRAIN/VAL-only root; VAL for selection)
#   * core preprocessing         <- THESIS M3 (src/data/transforms.core_preprocess canvas)
#
# The published PlantSeg benchmark for SegNeXt MSCAN-B is 42.05% mIoU / 56.30% mAcc / 28M params.
# That number is a CONTEXTUAL REFERENCE ONLY. The paper reports all benchmark methods trained with
# SGD (lr 1e-3, momentum 0.9, weight decay 5e-4, CE, batch 16); this configuration deliberately uses
# the thesis's pre-registered AdamW recipe instead, so a run of this file is NOT an exact
# reproduction of the published 42.05 protocol and must never be described as one.

_base_ = ['mmseg::segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py']

import os

# B62 thesis components: M2/M3 transforms, the M4 isolated-NMF head, the M12 metric and hooks
# (src/training/teacher_components.py). The repository root must be importable; the launcher, the
# evaluator and the E2/E3 entry points put it on sys.path.
custom_imports = dict(imports=['src.training.teacher_components'], allow_failed_imports=False)

# Machine-checkable provenance markers (asserted by scripts/smoke_teacher_config.py).
PROTOCOL_CLASSIFICATION = 'thesis-derived'
PUBLIC_PLANTSEG_SOURCE_COMMIT = '1a3dd4d9224bcc97a5850af7dd1c423abc24eae0'
PUBLISHED_MSCAN_B_REFERENCE = dict(miou=42.05, macc=56.30, params_m=28, role='contextual-reference-only')
PUBLISHED_BENCHMARK_OPTIMIZER = 'SGD lr=1e-3 momentum=0.9 wd=5e-4 (paper) — NOT used here'
AUGMENTATION_SOURCE = 'M2:src/data/transforms.train_preprocess+configs/augment.AUGMENT'
CORE_PREPROCESSING_SOURCE = 'M3:src/data/transforms.core_preprocess'
METHODOLOGY_LOCKS = dict(M2='B60 §3', M3='B60 §4', M4='B61 §4', M5='B60 §2', M11='B60 §5',
                         M12='B61 §6', M13='B61 §7')

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

# TRAIN — M2 augmentation + M3 train scale (B60 §3-§4): SEMANTIC PARITY with the E1-E3 recipe, by
# reusing it. `ThesisTeacherTrainTransform` calls src/data/transforms.train_preprocess with
# configs/augment.AUGMENT: long side = round(512*r), r ~ U[0.75, 2.0] on the unpadded image ->
# rotation +/-10 deg (p 0.5, before the crop; image fill = ImageNet mean, mask fill = 255) ->
# 512x512 crop/pad with cat_max_ratio 0.95 -> independent H and V flips (p 0.5 each) -> image-only
# hue +/-0.015 and saturation [0.8, 1.2], jointly p 0.5. No brightness, contrast, blur, noise or
# JPEG, so no PhotoMetricDistortion. Bilinear image / nearest mask throughout.
train_pipeline = [
    dict(type='ThesisTeacherTrainTransform'),
    dict(type='PackSegInputs'),
]

# VAL — M3 clean evaluation: src/data/transforms.core_preprocess, the thesis evaluator's canvas.
# Long side 512 (bilinear image, nearest mask), symmetric pad to 512x512 (image 8-bit ImageNet mean,
# mask 255). The transform sets ori_shape = img_shape = (512, 512), so predictions are NOT resized
# back to the original image and the metric scores exactly the thesis canvas. (The previous
# Resize-before-LoadAnnotations pipeline scored at original resolution; it is gone.)
val_pipeline = [
    dict(type='ThesisTeacherEvalTransform'),
    dict(type='PackSegInputs'),
]
# No MMSeg test/inference pipeline for the teacher: the inherited upstream short-side (2048, 512)
# pipelines are removed. Teacher evaluation outside training uses the thesis evaluator (M4-V).
test_pipeline = None
tta_pipeline = None

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

# M4-V needs batch_size 1 and a frozen sequential order (BaseSegDataset sorts by img_path).
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **_dataset_common,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=val_pipeline),
)

# M11: no active TEST surface. MMEngine requires the three to be all None or all set
# (runner.py:351); None replaces the inherited ADE20K test config during the merge.
test_dataloader = None
test_evaluator = None
test_cfg = None

# M12: full-precision, same-pass, union-present all-class VAL mIoU (src/eval/metrics.py, E1's reducer).
# Keys: 'mIoU_full' (selection), 'mIoU_disease_full', 'mIoU' (percent, 2 decimals, display only).
SELECTION_KEY = 'mIoU_full'
val_evaluator = dict(type='ThesisConfusionMIoUMetric', num_classes=NUM_CLASSES,
                     ignore_index=IGNORE_INDEX, background_index=BACKGROUND_INDEX)

# --------------------------------------------------------------------------------------------
# Model deltas — plain BN (never SyncBN), 116-class head, CE-only loss, no competing backbone init
# --------------------------------------------------------------------------------------------
norm_cfg = dict(type='BN', requires_grad=True)

# CORE PREPROCESSING — normalisation only. mean/std are configs/data.py's
# (0.485,0.456,0.406)/(0.229,0.224,0.225) in 0-255 terms; the student's finalize() computes the same
# values (B59 §C: 2.4e-7). The thesis transforms already emit the padded 512x512 canvas — image
# padded with the 8-bit ImageNet mean (124,116,104), exactly as the student pads, and mask padded
# with 255 — so the preprocessor's own padding is a no-op in training and disabled at evaluation
# (test_cfg=None). The pad region carries label 255 and enters neither loss nor metric.
PAD_MODE = 'thesis-canvas-8bit-imagenet-mean-pad-applied-by-transform'
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=IGNORE_INDEX,
    size=crop_size,
    test_cfg=None,
)

model = dict(
    data_preprocessor=data_preprocessor,
    # Only the BACKBONE carries SyncBN in the stock recipe (the LightHamHead uses GroupNorm), so the
    # BN override targets the backbone. init_cfg=None so the IN-1K backbone is NOT re-pulled: the
    # full ADE20K segmentation model arrives via `load_from` (runbook §5).
    backbone=dict(norm_cfg=norm_cfg, init_cfg=None),
    decode_head=dict(
        # M4: the upstream LightHamHead with its NMF basis draw routed through IsolatedNMF2D. The
        # algorithm is unchanged (ham_kwargs.rand_init=True, inherited); only the RNG source of the
        # draw can be switched to a private stream for evaluation (M4-V) and KD (M4-KD).
        type='IsolatedNMFLightHamHead',
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        # M13: ignore/padded pixels enter neither the numerator nor the denominator of the CE mean.
        # Unweighted (M5): no class_weight.
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                         avg_non_ignore=True),
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

# Schedule — LOCKED (M5, B60 §2.2). The paramwise decay_mult=0 for pos_block/norm above is the
# upstream SegNeXt convention (identical in the inherited base); M5 does not conflict with it.
MAX_ITERS = 40000        # M5; the teacher is EXEMPT from the 80k student budget
WARMUP_ITERS = 1500      # M5: LinearLR 0 -> 1500, start_factor 1e-6
POLY_POWER = 1.0         # M5: PolyLR power 1.0 (NOT the 0.9 of the generic schedule_40k.py base)
VAL_INTERVAL = 4000      # M5: validation AND checkpoint every 4,000 iterations -> 10 validations (M12)

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
# (test_cfg is None — M11; see the dataloader section above.)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    # M12: select on the FULL-PRECISION VAL all-class mIoU ('mIoU_full'), never on the 2-decimal
    # display value. rule='greater' compares strictly (checkpoint_hook.py:123), so an exact tie keeps
    # the earliest iteration. max_keep_ckpts=-1 keeps all ten 4k checkpoints (the full selection
    # trail). The TEST split never participates.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=VAL_INTERVAL, max_keep_ckpts=-1,
                    save_last=True, save_best=SELECTION_KEY, rule='greater'),
)

# B62 hooks: the 150->116 classifier-only load rule, M4-V for every validation pass, and the M12
# selection record cross-checked against CheckpointHook (all in src/training/teacher_components.py).
M4_NMF_SEED = 42
custom_hooks = [
    dict(type='TeacherInitCompatibilityHook'),
    dict(type='TeacherNMFEvalStreamHook', seed=M4_NMF_SEED),
    dict(type='TeacherSelectionRecordHook', key=SELECTION_KEY, disease_key='mIoU_disease_full'),
]

# --------------------------------------------------------------------------------------------
# Determinism, initialization and output paths
# --------------------------------------------------------------------------------------------
randomness = dict(seed=42, deterministic=True)

# M4 (B61 §4) — NMF/Hamburger randomness is PRESERVED (rand_init=True, inherited) and ISOLATED:
#   M4-T  training: upstream fresh bases from the run's seeded global CPU stream (unchanged);
#   M4-V  every validation pass: private CPU stream seeded 42, batch 1, frozen order, caller RNG
#         restored (TeacherNMFEvalStreamHook);
#   M4-KD frozen E2/E3 teacher: private CPU stream seeded 42 once per run, advancing across calls.
# The private stream's only consumer is the NMF basis draw; it is seeded through a CPU
# torch.Generator, never torch.manual_seed (which also reseeds CUDA).
M4_NMF_POLICY = dict(algorithm='rand_init=True (upstream)', train='M4-T', val_and_final_eval='M4-V',
                     frozen_teacher_e2_e3='private stream seeded once per run', seed=M4_NMF_SEED)

# REQUIRED external ADE20K initialization, OUTSIDE the repository. No silent random init and no in-run
# download: if the env var is unset the sentinel below is not a real path, so the launch fails loudly.
# The launcher verifies the file's SHA-256 against EXPECTED_ADE20K_SHA256 before anything is built.
SEGNEXT_ADE20K_CKPT_ENV = 'SEGNEXT_ADE20K_CKPT'
ADE20K_CKPT_UNSET_SENTINEL = 'NEED_TO_CONFIRM__SET_SEGNEXT_ADE20K_CKPT'
EXPECTED_ADE20K_SHA256 = '647a0cda7678a35396689a4f8e9fddc33a088d8b539195d0dc97485ab8640ef1'
ADE20K_INIT_IDENTITY = dict(
    config='segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512',
    filename='segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth',
    sha256=EXPECTED_ADE20K_SHA256,
    classifier_only_mismatch=('decode_head.conv_seg.weight', 'decode_head.conv_seg.bias'),
    readiness='B61 §1')
load_from = os.environ.get(SEGNEXT_ADE20K_CKPT_ENV, ADE20K_CKPT_UNSET_SENTINEL)
resume = False

# Trained teacher checkpoints live OUTSIDE the repository (runbook §10).
TEACHER_WORK_DIR_ENV = 'TEACHER_WORK_DIR'
WORK_DIR_UNSET_SENTINEL = 'NEED_TO_CONFIRM__SET_TEACHER_WORK_DIR'
work_dir = os.environ.get(TEACHER_WORK_DIR_ENV, WORK_DIR_UNSET_SENTINEL)
