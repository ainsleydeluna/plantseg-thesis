# B8 — Is a trained SegNeXt-B PlantSeg checkpoint publicly released?

**NONE found** (checked README / Releases / linked URLs / Zenodo on **2026-06-27**), plan = **in-house fine-tune Jul 13–26**.

The B1 teacher must be trained in-house per the contract: SegNeXt-B / MSCAN-B fine-tuned on PlantSeg,
target **42.05% mIoU within ±1.5–2.0 pp** (protocol match, descriptive upper-bound only). See
[IMPLEMENTATION_CONTRACT.md](IMPLEMENTATION_CONTRACT.md) §B1.

## What was checked (2026-06-27)

| Source | Result |
|---|---|
| `github.com/tqwei05/PlantSeg` README (raw, `main`) | No mention of checkpoint / weights / pretrained / model-zoo / `.pth` / Google Drive / Hugging Face. It is a **dataset + baseline-code** repo; instructions only say to place PlantSeg under `data/` and build the env via MMSegmentation. |
| GitHub **Releases** | **Empty** — "There aren't any releases here." No release assets (`.pth` / `.ckpt` / `.pt`). |
| Links in README | Only: the repo itself, `zenodo.org/records/14935094` (an earlier **dataset** record), MMSegmentation install docs + repo, and arXiv `2409.04038` (the paper). **None point to model weights.** |
| Zenodo `zenodo.org/records/17719108` | Single file **`plantseg.zip` (1.1 GB)** — the **dataset only** (images / annotations / split JSONs / `Metadata.csv`). Already extracted and verified in [reports/dataset_report.md](../reports/dataset_report.md); contains **no** model checkpoints. |
| Web search ("PlantSeg … SegNeXt … checkpoint / pretrained weights") | No released checkpoint surfaced — results are the dataset, the paper, the repo, and an unrelated Kaggle augmented-dataset. Reproduction is described as **train-it-yourself** via the MMSegmentation config. |

## Conclusion
No publicly released trained SegNeXt-B (MSCAN-B) PlantSeg checkpoint (~42.05% mIoU) was found. The
released artifacts are the **dataset** and **baseline code/configs**, not trained weights.

**Plan:** in-house fine-tune of the teacher (B1) during **Jul 13–26**, using the ADE20K-pretrained
SegNeXt-B / MSCAN-B init from the MMSegmentation model zoo (`segnext_mscan-b_512x512_160k_ade20k`),
following the locked B1 recipe; success = recover **42.05% mIoU ± 1.5–2.0 pp**.

_Re-check trigger:_ if a checkpoint is later published (repo Releases / Hugging Face / Zenodo), switch
the plan to **download + verify** instead of fine-tuning.
