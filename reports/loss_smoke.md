# Loss smoke test — B11b

**RESULT: PASS**  
_Generated: 2026-06-27 by `scripts/smoke_loss.py` (synthetic; tiny backward smoke; no training)._

## Output
```
num_classes=116  ignore_index=255  (CE weight=None: real class weights NEED_TO_CONFIRM)
valid pixels=4  ignore(255) pixels=28
CE=4.9012  Dice=0.9754  CE+Dice=5.8766
grad finite=True  grad range=[-2.5608e-01, 2.5002e-02]
ignore-perturb: CE 4.9012->4.9012 (==)  Dice 0.9754->0.9754 (==)
[PASS] loss smoke test
```

## Checks (gate PASS)
- [PASS] CE finite
- [PASS] Dice finite
- [PASS] CE+Dice finite
- [PASS] gradients finite
- [PASS] ignore-255 perturbation leaves CE unchanged
- [PASS] ignore-255 perturbation leaves Dice unchanged

## NEED_TO_CONFIRM
- real CE class weights — batch/None here; full train-set weights NEED_TO_CONFIRM
- logit KD weight (lambda_logit) — NEED_TO_CONFIRM (validation sweep)
- reduce_zero_label — NEED_TO_CONFIRM
