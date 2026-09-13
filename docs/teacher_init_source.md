# Teacher initialization weights — source record

ADE20K-pretrained **SegNeXt-B / MSCAN-B (512×512)** checkpoint used as the *initialization* for the
in-house teacher fine-tune (B1). **Not trained, not fine-tuned, not modified** by this step. See
[B8_checkpoint.md](B8_checkpoint.md) (no PlantSeg teacher checkpoint is publicly released) and
[IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §B1.

| Field | Value |
|---|---|
| MMSeg 1.x config name | `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512` |
| Exact `.pth` URL (resolved by `mim`) | `NEED_TO_CONFIRM` (paste from the `mim download` log) |
| SHA256 of the `.pth` | `NEED_TO_CONFIRM` (`sha256sum weights/<file>.pth`) |
| Download date | `NEED_TO_CONFIRM` |
| MMSeg version (download + test env) | **1.2.2** (pinned; mmcv 2.1.0, torch 2.1.0) |
| Reported ADE20K mIoU | **48.03 (SS) / 49.68 (MS)** |
| Source / license | **OpenMMLab** (MMSegmentation model zoo), **Apache-2.0** |
| Init-test result (`scripts/test_teacher_init.py`) | `NEED_TO_CONFIRM` (PASS/FAIL) — harness **not yet verified**, see [B54](../reports/b54_teacher_prerequisites.md) §2.4 |

> **Config-name reconciliation `[B54 2026-09-13]`.** The name in row 1 is the **operative** one: it is
> what `mim` resolves and what every executable site uses — `src/distill/segnext_teacher.py:35`,
> `scripts/test_teacher_init.py:32`, and `configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:44`
> (`_base_ = ['mmseg::segnext/…']`). Two sites still carry the older MMSeg 0.x spelling
> `segnext_mscan-b_512x512_160k_ade20k` — `docs/IMPLEMENTATION_CONTRACT.md:123` and
> `configs/teacher_finetune.py:9`. The contract's "(or equivalent)" admits the 1.x name, so this is a
> **wording lag, not a conflict**, and nothing runs against a forbidden name. The fix is queued as
> **G10, governed** — both sites sit on rule 8's path list, and harmlessness is not a route off it.
>
> **On row 4 (init-test result).** The harness was run on 2026-09-13 and could not be verified: this
> repository checkout has no numpy, torch or MM stack, so `scripts/test_teacher_init.py` aborted at
> its module-scope `import numpy` (`:28`) with exit 1 — before reaching the guarded MM-import check at
> `:54-60` that would have exited 2 with guidance. The field therefore stays `NEED_TO_CONFIRM`
> rather than being upgraded on an untested claim. It is closable in one CPU-only command inside the
> pinned environment, **before** any download: `exit 2` with "config/checkpoint not found" is itself
> proof the harness runs. B54 §2.4 records the import-guard defect as queue item G11.

> **Re-init note:** the stock checkpoint's classifier `decode_head.conv_seg` is sized for **150 ADE20K
> classes**. Before teacher fine-tuning it will be **re-initialized to the empirically verified PlantSeg
> class count = 116** (all-class: background `0` + 115 diseases `1–115`; see
> [reports/dataset_report.md](../reports/dataset_report.md)). The downloaded checkpoint itself is never
> edited — re-init happens in the training pipeline.

## How to fill the `NEED_TO_CONFIRM` fields (run in the pinned MMSeg env)

```bash
# 1) download config + checkpoint into weights/ (git-ignored); capture the printed URL
mim download mmsegmentation \
  --config segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512 --dest weights/

# 2) checksum + date
sha256sum weights/segnext_mscan-b_*ade20k*.pth
date -u +%Y-%m-%d

# 3) read-only init sanity test (must print "RESULT: PASS")
python scripts/test_teacher_init.py
```

Paste the resolved `.pth` URL, the SHA256, the date, and the test's PASS line back here and this record
will be finalized. `weights/` and `*.pth` are git-ignored — the checkpoint is never committed.
