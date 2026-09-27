# Chapter 4 correction queue

Corrections to the approved Chapter 3 text that Chapter 4 carries. Chapters 1–3 stay the approved plan
(DL-20); record corrections and methodology changes are documented in Chapter 4. Each row names its
source, and the source's wording governs. A row is added by the task that records its source and closed
when the Chapter 4 text covering it is drafted.

Created 2026-09-28 (CP-007a).

| # | Chapter 3 location | Correction | Source | Status |
|---|---|---|---|---|
| Q1 | Table 3.3, p. 111 (SegNeXt-L) | MSCAN-L's Stage-3 width equals MSCAN-B's (320 channels; Guo et al., 2022, Table 2; upstream configs). SegNeXt-L differs in depth ([3, 5, 27, 3] vs [3, 3, 12, 3]) and decoder width (1,024 vs 512), so the channel-width reason for rejecting it applies only to ConvNeXt-L (768 channels, different family). | AM-17 item 10(g) | open |
| Q2 | p. 102, teacher paragraph | B recipe provenance: tqwei05/PlantSeg at 1a3dd4d ships MSCAN-T and MSCAN-L PlantSeg configurations only. The MSCAN-B configuration of record (`configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py`) keeps the upstream optimizer and warm-up recipe but uses ADE20K full-model initialisation, scheduler end 40,000 and the repository training pipeline. p. 102's "the repository configuration that produced the published 42.05%" is qualified: it reads as "the repository's PlantSeg schedule with the MSCAN-B architecture", and these differences qualify the 42.05% comparison. | AM-17 item 10(f); B60 §2.1–§3 | open |
| Q3 | pp. 126, 133 and 146 | Split counts 5,367 TRAIN / 846 VAL / 1,561 TEST replace the approximations. | AM-17 item 10(a) | open |
| Q4 | p. 103; Table 3.3, p. 111 | 116 output classes (background 0 plus 115 diseases) replace "115 output channels"; 2,933,688 student parameters (training-only projection excluded) replace "5M-param". | AM-17 item 10(b) | open |
| Q5 | p. 91, summary sentence | Superseded by p. 139 and AM-14. 42.05% is the single-run SegNeXt-B result on the 7,774-image release (Wei et al., 2026); the 2024 preprint's 53.89% refers to the older release and is not a comparator. | AM-17 item 10(e) | open |
| Q6 | p. 102, teacher success criterion | The "recover 42.05% within ±1.5–2.0 pp" protocol-matching band has no methodological authority. Teacher acceptance is R3 on VAL (strictly > 0.36314016580581665); the 42.05% comparison is descriptive and made at TEST only, with the teacher also scored under the upstream protocol. | AM-13/DL-08; B60 §2.1; `docs/reference/context.md:92` marker | open |
