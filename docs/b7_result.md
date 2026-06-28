# B7 / B13 — QNNPACK eager-mode INT8 operator-support result (updated by B7b)

| Field | Value |
|---|---|
| Date | 2026-06-28 (B7b refactor) |
| PyTorch version | **2.9.1+cpu** |
| Quantization API | `torch.ao.quantization` (eager mode; no PT2E, no FX) |
| Requested engine | **QNNPACK** |
| Engine actually used | **onednn (PROXY — NOT QNNPACK)** |
| Smoke type | **QNNPACK operator-support + conversion smoke** — *not* an x86 latency benchmark, *not* training |

## ⚠ Environment blocker — QNNPACK unavailable on this machine (unchanged)
`torch.backends.quantized.supported_engines == ['onednn']`; setting `engine="qnnpack"` raises
`RuntimeError: quantized engine QNNPACK is not supported`. This build ships only onednn (no qnnpack/fbgemm).
**QNNPACK status remains `BLOCKED_ENV`.** The authoritative QNNPACK run (incl. the contract's INT8
**Sigmoid in the LR-ASPP global-pool** gate and quantized bilinear interpolate) **must be done in the
pinned env** (torch 2.1.0 + QNNPACK). Everything below is an explicitly-labeled **onednn x86 proxy**.

## B7b — student quantization-readiness refactor
**The student head was refactored to match the quantization-ready isolated head.** Changes in
`src/models/student.py` (FP32 behaviour preserved exactly):
- attention multiply `a * s` → **`FloatFunctional.mul`**; class-logit `+` → **`FloatFunctional.add`**;
- head now returns **OS8 logits**; the parent **dequants then upsamples to 512 in float** (one quantized
  interpolate, matching the isolated head — was two);
- added **`PlantSegStudent.fuse()`** (head 1×1 Conv-BN-ReLU + backbone blocks' `fuse_model()`).
- Unchanged: `build_student()` API, `num_classes=116`, output `[B,116,512,512]`, OS8 tap idx6/40ch,
  OS16 tap idx15/160ch, `features[0:16]`, param count **2,933,688** (FloatFunctional adds no parameters).

## Head smoke (`scripts/smoke_qnnpack_head.py`) — isolated LR-ASPP head
- **Outcome (onednn proxy): `CLEAN`** — converts + forwards, **no fallback**. Output **`(2,116,512,512)`**.
- Sigmoid fallback ladder (clean → FixedQParams(0–1, 1/256, 0) → DeQuant→Sigmoid→Quant float branch) is
  implemented and ready; **not triggered** on onednn. **QNNPACK: PENDING.**

## Full-student smoke (`scripts/smoke_qnnpack_full_student.py`) — existing student, fused
- **B7 (before refactor):** `FULL_STUDENT_FAILS` — `RuntimeError: empty_strided not supported on
  quantized tensors` at converted-forward (plain `*`/`+` in the head).
- **B7b (after refactor): `PASS_CLEAN` under the onednn PROXY** — `fusion_called=True`,
  output **`(1,116,512,512)`**, `fail_stage=n/a`. **No fallback used.**
- The script still does **not** print a clean QNNPACK pass (engine is a proxy); it prints `[DOCUMENTED]`.
- Benign: a `must run observer before calling calculate_qparams` warning remains (a non-calibrated
  observer defaulting its qparams); conversion + forward nonetheless succeeded under the proxy.

## Did the oneDNN proxy full-student smoke improve?
**Yes — decisively:** from a hard convert-forward failure (`empty_strided`) to a **clean
convert+forward** `(1,116,512,512)` under the onednn proxy, with fusion called. This confirms the
refactor made the student eager-INT8-convertible at the graph level.

## QNNPACK status
**`BLOCKED_ENV`** on this machine — **not** claimed to pass. The proxy result is strong evidence the
architecture is convertible, but it is **not** a QNNPACK result.

## Fallback used
- Head: **none** (clean on onednn). Full student: **none** (clean on onednn proxy after refactor).
- QNNPACK sigmoid ladder + a possible quantized-interpolate fallback remain wired for the pinned-env run.

## Exact output shapes
- Head (proxy, clean): **`[2,116,512,512]`**. Full student (proxy, clean): **`[1,116,512,512]`**.

## Does this still block E4 / E5 / E6 / E7?
The **as-built quantization-readiness blocker is resolved** (student converts cleanly on the onednn
proxy with FloatFunctional + fusion). **Remaining before E4–E7:**
1. **Authoritative QNNPACK verification** in the pinned env (torch 2.1.0 + QNNPACK): re-run both smokes;
   confirm INT8 Sigmoid in the global-pool branch and quantized bilinear interpolate (or record the
   fallback). Until then, no QNNPACK INT8 claim.
2. E4/E7 (PTQ) need the ~128-image calibration subset; E5/E6 (QAT) need the QAT fine-tune — separate
   blockers, out of B7/B7b scope (no training here).
3. Data-side `reduce_zero_label` / disease-only convention remains `NEED_TO_CONFIRM` (unrelated to quant).

## `NEED_TO_CONFIRM`
1. **QNNPACK operator support** — unavailable here; run both smokes in the pinned torch-2.1 + QNNPACK env.
2. **QNNPACK INT8 Sigmoid** in the LR-ASPP global-pool branch (contract Table 3.1 gate) — pending; ladder ready.
3. **QNNPACK quantized bilinear interpolate** — onednn OK; QNNPACK pending.
4. (Resolved by B7b) ~~student quantization-readiness refactor~~ — done: FloatFunctional + fuse(); onednn
   proxy converts cleanly.
