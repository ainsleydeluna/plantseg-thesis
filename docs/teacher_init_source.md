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
| Init-test result (`scripts/test_teacher_init.py`) | `NEED_TO_CONFIRM` (PASS/FAIL) |

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
