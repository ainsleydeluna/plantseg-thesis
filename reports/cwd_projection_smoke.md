# CWD projection stub smoke test — B12

**RESULT: PASS**  
_Generated: 2026-06-27 by `scripts/smoke_cwd_projection.py` (no training, no CWD loss, no teacher code)._

Training-only CWD projection for E3 feature distillation: a single **bias-free 1×1 convolution**
mapping the student's stride-16 C5 feature to the teacher's stride-16 channel dimension.

| Item | Value |
|---|---|
| Module | `nn.Conv2d(160, 320, kernel_size=1, bias=False)` |
| Input shape | **`[2, 160, 32, 32]`** |
| Output shape | **`[2, 320, 32, 32]`** |
| `in_ch` source | **160** — B10-verified student OS16/C5 tap (features index 15); see [student_forward_smoke.md](student_forward_smoke.md) |
| `out_ch` source | **320 — LOCKED** (docs/IMPLEMENTATION_CONTRACT.md B3 / ch3: MSCAN-B stride-16 Stage-3, `C=320`) |
| Parameter count | **51,200** (= 160 × 320, bias-free) |
| Finite (no NaN/Inf) | yes |
| Result | **PASS** |

## `out_ch = 320` status: LOCKED
The authoritative contract (B3, tracing to ch3) fixes the teacher stride-16 channel count at **320**
(`T²/C` with `C=320`; "student 160-ch C5 → teacher 320-ch"; Table 3.1 forward-hook returns the
320-channel MSCAN-B Stage-3 feature). It is therefore **LOCKED**, not `NEED_TO_CONFIRM`.

_Deferred (not a blocker):_ the **empirical** teacher-side confirmation is the Table 3.1 forward-hook
check, which runs when the SegNeXt-B / MSCAN-B teacher is actually built. No teacher code exists yet
(out of scope for B12), so that empirical check is deferred — the spec value 320 stands as locked.

## Smoke-test output
```
projection: bias-free 1x1 conv 160 -> 320
[shapes] input (2, 160, 32, 32) -> output (2, 320, 32, 32)
[params] total = 51,200 (expect 51,200)
[checks] shape=True finite=True bias_free+params=True
[PASS] CWD projection stub smoke test
```

## Training-only — removal before E6/E7
**This projection is TRAINING-ONLY.** It exists only during E3 training and **must be removed from the
E3 checkpoint via `state_dict` editing before E6/E7** (quantization observer insertion / PTQ / QAT export),
so the evaluated and quantized models never contain it. This stub is **not** the CWD loss — full
channel-wise feature distillation is deferred to the E3 blocker.
