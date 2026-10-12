# R1 "B-EMA" — AM-18 item 2(a): the teacher-of-record recipe with exactly two registered changes
# (lane L-TEACHER-ARMS; ARMS plan decision (i); DL-24 audit F11, F12, F24).
#
#   1. MSCAN-B drop_path_rate 0.1 -> 0.2 (model.backbone.drop_path_rate);
#   2. mmengine 0.10.7's own EMAHook (ExponentialMovingAverage, momentum 0.0002, update_buffers True,
#      interval 1, begin_iter 0, strict_load False; priority: the class default 'NORMAL'), appended as a
#      fourth custom hook after the record's three, which are restated verbatim.
#
# Everything else is the configuration of record, inherited unchanged. The launcher
# (scripts/launch_teacher_arm.py) proves it: check_locked_config on the record config, the record's
# randomness/resume/work_dir/load_from/data_root rules on this one, and diff_cfg(record, R1) == REG["R1"].
# EMAHook swaps the averaged weights in for every validation (M12 selects on them) and saves them as each
# checkpoint's state_dict; the raw weights go to ema_state_dict, which no KD or evaluation path reads.
# Evaluation and distillation use the record config (eval-identical: DropPath is the identity in eval and
# holds no parameter).

_base_ = ['../segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py']

model = dict(backbone=dict(drop_path_rate=0.2))

custom_hooks = [
    dict(type='TeacherInitCompatibilityHook'),
    dict(type='TeacherNMFEvalStreamHook', seed=42),
    dict(type='TeacherSelectionRecordHook', key='mIoU_full', disease_key='mIoU_disease_full'),
    dict(type='EMAHook', ema_type='ExponentialMovingAverage', momentum=0.0002, update_buffers=True,
         interval=1, begin_iter=0, strict_load=False),
]
