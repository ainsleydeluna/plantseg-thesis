# Metrics smoke test — B11b

**RESULT: PASS**  
_Generated: 2026-06-27 by `scripts/smoke_metrics.py` (synthetic; no training)._

## Output
```
num_classes=116  ignore_index=255
non-255 pixels=9  CM total=9
present GT classes=[0, 5]  all-class mIoU(perfect)=1.000000
all-class mIoU(partial)=0.791667
disease-only mIoU(perfect) = 1.000000  ** PROVISIONAL / NEED_TO_CONFIRM (excludes index 0) **
per-image disease-only (PROVISIONAL) = [1.0, 1.0]
[PASS] metrics smoke test
```

## Checks (gate PASS)
- [PASS] ignore 255 excluded (CM total == non-255 count)
- [PASS] ignore 255 excluded (mIoU invariant to 255-position preds)
- [PASS] absent classes excluded (perfect mIoU==1.0 over present [0, 5])
- [PASS] all-class mIoU sanity (partial in (0,1): 0.7917)

## NEED_TO_CONFIRM
- disease-only / background convention (reduce_zero_label) — disease-only mIoU is PROVISIONAL (excludes empirical background index 0; not yet settled).
