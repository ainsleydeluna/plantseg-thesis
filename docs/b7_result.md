# B7 / B13 — QNNPACK eager-mode INT8 operator-support result

| Field | Value |
|---|---|
| Date | 2026-06-28 |
| PyTorch version | **2.9.1+cpu** |
| Quantization API | `torch.ao.quantization` (eager mode; no PT2E, no FX) |
| Requested engine | **QNNPACK** |
| Engine actually used | **onednn (PROXY — NOT QNNPACK)** |
| Smoke type | **QNNPACK operator-support + conversion smoke** — *not* an x86 latency benchmark, *not* training |

## ⚠ Environment blocker — QNNPACK unavailable on this machine
`torch.backends.quantized.supported_engines == ['onednn']`. Setting `engine="qnnpack"` raises
`RuntimeError: quantized engine QNNPACK is not supported`. This torch build ships **only onednn** (no
qnnpack, no fbgemm). The **authoritative QNNPACK** operator-support result (esp. the contract's INT8
**Sigmoid in the LR-ASPP global-pool branch** gate) **must be produced in the pinned env**
(torch 2.1.0 + QNNPACK — the same RunPod/pinned environment the teacher-init step awaits).

Both smokes below therefore ran under an **onednn x86 proxy**, clearly labeled. The proxy validates the
eager-mode conversion *graph* (FloatFunctional mul/add, sigmoid, quantized bilinear interpolate) — which
is largely backend-independent — but it is **not** a QNNPACK pass.

## Head smoke (`scripts/smoke_qnnpack_head.py`) — isolated LR-ASPP head
- **Outcome (onednn proxy): `CLEAN`** — converted + forwarded with **no fallback** needed.
- Output shape: **`(2, 116, 512, 512)`**.
- Sigmoid fallback ladder (clean → FixedQParams(0–1, scale 1/256, zp 0) → DeQuant→Sigmoid→Quant float
  branch) is **implemented and ready**, but **not triggered** on onednn (clean default sufficed).
- **QNNPACK status: PENDING** (pinned env). The same script will, on QNNPACK, either pass clean or walk
  the ladder and record exactly which.

## Full-student smoke (`scripts/smoke_qnnpack_full_student.py`) — existing `src/models/student.py`
- **Outcome: `HEAD_PASS_FULL_STUDENT_FAILS_DOCUMENTED`** (onednn proxy).
- Fail stage: **`forward(converted)`**. Output shape: **none** (failed).
- **Exact error:** `RuntimeError: empty_strided not supported on quantized tensors yet`
  (pytorch/pytorch#74540).
- **Root cause:** the current student head uses **plain `*` / `+`** ops (not `FloatFunctional`) and the
  model is **not fused** — so it is **not quantization-ready**. The isolated head (which uses
  `FloatFunctional.mul`/`.add` + Conv-BN-ReLU fusion) converts cleanly, proving the *pattern* is fine.
- **No hacks** were applied to force a pass (per spec).

## Does this block E4 / E5 / E6 / E7?
**Yes, currently — pending two things:**
1. **Student quantization-readiness refactor** (out of B7 scope): rebuild the student head with
   `FloatFunctional` for the attention-multiply and logit-add, and apply Conv-BN-ReLU fusion (and restore
   the quantizable backbone's `fuse_model` path lost when `features` was truncated). The isolated head
   smoke is the reference implementation.
2. **Authoritative QNNPACK verification** in the pinned env (torch 2.1.0 + QNNPACK): re-run both smokes;
   confirm the INT8 Sigmoid / global-pool branch and quantized bilinear interpolate are supported (or
   record the fallback used).

Until both are done, E4/E5/E6/E7 (PTQ/QAT INT8 convert) cannot be claimed. The head result shows the
target architecture *is* convertible in eager INT8; the blockers are the student refactor + the QNNPACK
backend run.

## Exact output shapes
- Head (proxy, clean): **`[2, 116, 512, 512]`**.
- Full student: **none** (conversion-forward failed — documented above).

## Fallback used
- Head: **none** on onednn (clean). QNNPACK sigmoid ladder ready but untriggered here.
- Full student: **none attempted** (failure captured verbatim; no hacks).

## `NEED_TO_CONFIRM`
1. **QNNPACK operator support** — unavailable on this machine; run both smokes in the pinned torch-2.1 +
   QNNPACK env for the authoritative result.
2. **QNNPACK INT8 Sigmoid in the LR-ASPP global-pool branch** (contract Table 3.1 gate) — pending; the
   FixedQParams / DeQuant-Sigmoid-Quant fallback ladder is wired and will record which path QNNPACK needs.
3. **QNNPACK quantized bilinear interpolate** — onednn supports it; QNNPACK support pending (if absent, a
   dequant→interpolate→requant fallback would be added).
4. **Student quantization-readiness refactor** (FloatFunctional mul/add + fusion) — required before
   E4–E7; not done in B7.
