# R2 "B-EMA + rare" — AM-18 item 2(b): R1's recipe plus the two TRAIN-data changes, and nothing else
# (lane L-TEACHER-ARMS; ARMS plan decision (ii); DL-24 audit F13 to F18; Phase B-2 ruling R6).
#
#   (i)  repeat-factor sampling: mmengine ClassBalancedDataset with oversample_thr = t = 20/5,367 over
#        RareClassTrainDataset, whose get_cat_ids gives an image's class only when it is rare (n_c < 20,
#        from the R2 table); background is never a category;
#   (ii) same-class copy-paste for rare classes: RareClassCopyPasteTrainTransform, on the EXIF-corrected
#        raw image and mask before M2/M3, with the record transform's per-sample RandomState.
#
# The wrapped dataset is the record's TRAIN dataset, key for key (taken from the base, not restated), with
# its type, its pipeline and the table path changed. The table path is the one host-dependent value: it
# comes from PLANTSEG_R2_TABLE, which the launcher sets from --r2-table (after checking the file's sha256
# against the table row), and the canonical diff records it by the variable's name ("$PLANTSEG_R2_TABLE").
# The batch size, sampler (InfiniteSampler, shuffle) and every VAL/TEST key are the record's.

_base_ = ['./r1_bema.py']

import os

custom_imports = dict(imports=['src.training.teacher_components', 'src.training.teacher_arm_components'],
                      allow_failed_imports=False)

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=20 / 5367,
        dataset=dict(
            _base_.train_dataloader.dataset,
            type='RareClassTrainDataset',
            table_path=os.environ.get('PLANTSEG_R2_TABLE', 'NEED_TO_CONFIRM__SET_PLANTSEG_R2_TABLE'),
            pipeline=[dict(type='RareClassCopyPasteTrainTransform'), dict(type='PackSegInputs')])))
