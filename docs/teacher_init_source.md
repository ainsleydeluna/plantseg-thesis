# Teacher initialization weights — source record

ADE20K-pretrained **SegNeXt-B / MSCAN-B (512×512)** checkpoint used as the *initialization* for the
in-house teacher fine-tune (B1). **Not trained, not fine-tuned, not modified** by this step. See
[B8_checkpoint.md](B8_checkpoint.md) (no PlantSeg teacher checkpoint is publicly released) and
[IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §B1.

| Field | Value |
|---|---|
| MMSeg 1.x config name | `segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512` |
| Exact `.pth` URL | **`https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth`** — from the official `metafile.yaml` (pinned mmseg 1.2.2, byte-identical to upstream tag v1.2.2); HTTP 200, 0 redirects `[B61 §1]` |
| SHA256 of the `.pth` | **`647a0cda7678a35396689a4f8e9fddc33a088d8b539195d0dc97485ab8640ef1`** (110,977,141 bytes; MD5 `53b45828…7c79` = server Content-MD5). The filename suffix `b6f6c70c` is **not** the SHA-256 prefix; no official SHA-256 exists, so this is the first record (trust-on-first-use) `[B61 §1]` |
| Download date | **2026-09-22** (02:46:54–02:50:45 UTC), `curl` over HTTPS, stored **outside the repo** in `C:\Users\admin\plantseg_runs\teacher_checkpoint_readiness_20260922\checkpoint\` (evidence `SHA256SUMS` `3b80d584…d998`) |
| MMSeg version (download + test env) | **1.2.2** (pinned; mmcv 2.1.0, torch 2.1.0) |
| Reported ADE20K mIoU | **48.03 (SS) / 49.68 (MS)** |
| Source / license | **OpenMMLab** (MMSegmentation model zoo), **Apache-2.0** (code licence; no weights-specific licence is stated) |
| Init-test result (`scripts/test_teacher_init.py`) | ~~`NEED_TO_CONFIRM` (PASS/FAIL) — harness **not yet verified**~~ **PASS (2026-09-22)**: pinned image, CPU, `--network none`, explicit paths; missing 0 / unexpected 0; `conv_seg` 150; 27,636,310 params. 150 → 116 audit also PASS (only `conv_seg` re-initialised) `[B61 §1]` |

> **Config-name reconciliation `[B54 2026-09-13]`.** The name in row 1 is the **operative** one: it is
> what `mim` resolves and what every executable site uses — `src/distill/segnext_teacher.py:35`,
> `scripts/test_teacher_init.py:32`, and `configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py:44`
> (`_base_ = ['mmseg::segnext/…']`). Two sites still carry the older MMSeg 0.x spelling
> `segnext_mscan-b_512x512_160k_ade20k` — `docs/IMPLEMENTATION_CONTRACT.md:123` and
> `configs/teacher_finetune.py:9`. The contract's "(or equivalent)" admits the 1.x name, so this is a
> **wording lag, not a conflict**, and nothing runs against a forbidden name. The fix is queued as
> **G10, governed** — both sites sit on rule 8's path list, and harmlessness is not a route off it.
> **[UPDATED 2026-09-22 — B62] G10 closed:** `configs/teacher_finetune.py` now records the 1.x config name,
> exact filename and SHA-256, and the contract's B1 row carries the same identity.
>
> **On row 4 (init-test result).** The harness was run on 2026-09-13 and could not be verified: this
> repository checkout has no numpy, torch or MM stack, so `scripts/test_teacher_init.py` aborted at
> its module-scope `import numpy` (`:28`) with exit 1 — before reaching the guarded MM-import check at
> `:54-60` that would have exited 2 with guidance. The field therefore stays `NEED_TO_CONFIRM`
> rather than being upgraded on an untested claim. It is closable in one CPU-only command inside the
> pinned environment, **before** any download: `exit 2` with "config/checkpoint not found" is itself
> proof the harness runs. B54 §2.4 records the import-guard defect as queue item G11.
>
> **[UPDATED 2026-09-22 — B61]** Closed by measurement. In the pinned image numpy is present, so G11 does
> not affect the run, and the harness passed on the real checkpoint (the init-test row above). G11 itself stays queued as a
> governed fix.
> **[UPDATED 2026-09-22 — B62] G11 closed:** every third-party import is inside the guard (exit 2 with
> guidance, even without numpy); the checkpoint comes from explicit arguments or `SEGNEXT_ADE20K_CKPT`; there
> is no repository `weights/` fallback; an in-repo checkpoint and a SHA-256 mismatch are refused.

> **How the fields were actually filled `[B61 §1]`.** The procedure below was **not** used: its
> `--dest weights/` would put the checkpoint inside the repository, and it also downloads a config file,
> whereas only the one checkpoint was authorised (the stock config ships in the pinned package). The checkpoint was
> fetched once with `curl` from the official URL into the external evidence folder, hashed there, and
> tested with `python scripts/test_teacher_init.py <stock config> <checkpoint>` against the stock config
> shipped in the pinned mmseg package. The block below is kept as the original plan.

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
