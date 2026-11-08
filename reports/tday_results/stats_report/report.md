# Statistics report -- tday-official-20261105


Analysis `plantseg_primary_analysis_v1`; contract sha256 `3b402cec476580b2c877219c13838190f7cdc3895aeaab28ff5974c1d2697382`; code commit `dc40dd2eb5b8be01882f53c37b7f6a3ea9365b71`; all 13 integrity checks re-established. k (AM-5 zero-disease images excluded from per-image analyses) = 0.

## Holm family (eight superiority contrasts)


| contrast | n | W | z | Wilcoxon p | Holm p | reject | r_rb | HL shift | dz | mean delta [BCa 95 %] | median delta | t (p) | dataset-level delta | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy_e1_e2 (E1 -> E2) | 1561 | 588915 | -0.459231 | 0.676966 | 1 | False | -0.013679 | 0 | -0.016341 | -0.0029337 [-0.0121486, 0.00571094] | 0 | -0.645623 (0.740691) | 0.0158483 | not rejected |
| accuracy_e2_e3 (E2 -> E3) | 1561 | 642913 | 2.89102 | 0.00191994 | 0.0134396 | True | 0.0868324 | 0.00222966 | 0.0541361 | 0.0101719 [0.000581027, 0.0192037] | 0 | 2.13889 (0.0162998) | 0.0159243 | sizable |
| accuracy_e4_e5 (E4 -> E5) | 1561 | 634850 | 2.09831 | 0.0179388 | 0.107633 | False | 0.0624651 | 0.00168023 | 0.0988764 | 0.0132829 [0.00662456, 0.020038] | 0 | 3.90655 (4.88111e-05) | 0.00960018 | not rejected |
| accuracy_e7_e6 (E7 -> E6) | 1561 | 645937 | 3.44393 | 0.000286664 | 0.00229331 | True | 0.104462 | 0.00246236 | 0.115762 | 0.0204781 [0.0120099, 0.0297319] | 0 | 4.57368 (2.58547e-06) | -0.00888417 | detected, below the smallest effect of interest |
| accuracy_e4_e7 (E4 -> E7) | 1561 | 560586 | -1.99466 | 0.97696 | 1 | False | -0.0595086 | -0.00204742 | -0.0267902 | -0.00594622 [-0.0165806, 0.00537917] | 0 | -1.05847 (0.854997) | 0.0191389 | not rejected |
| accuracy_e5_e6 (E5 -> E6) | 1561 | 592911 | -0.337206 | 0.632019 | 1 | False | -0.0100165 | 0 | 0.00578515 | 0.00124896 [-0.0094768, 0.0116473] | 0 | 0.228569 (0.409617) | 0.00065452 | not rejected |
| accuracy_e1_e6 (E1 -> E6) | 1561 | 513854 | -4.76916 | 0.999999 | 1 | False | -0.141721 | -0.00930706 | -0.0866144 | -0.0180224 [-0.028444, -0.00758945] | 0 | -3.42209 (0.999681) | -0.0199233 | not rejected |
| robustness_e1_e6 (E1 -> E6) | 1561 | 449863 | -8.878 | 1 | 1 | False | -0.260082 | -0.0211459 | -0.174824 | -0.0212894 [-0.0273333, -0.015295] | -0.0147731 | -6.90721 (1) | -0.0185271 | not rejected |

Labels: AM-17 item 2 (Holm decision and the dataset-level difference against 0.010, compared as stored fractions); item 3(f) from the committed MDE entry (power caveat: False).

## Bootstrap intervals (95 % BCa or the percentile fallback; z0 per interval)


| task | estimate | lower | upper | z0 | method | estimate inside |
|---|---|---|---|---|---|---|
| accuracy_e1_e2__mean_delta | -0.0029337 | -0.0121486 | 0.00571094 | -0.01855 | bca | True |
| accuracy_e1_e2__median_delta | 0 | 0 | 0 | 0 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e1_e2__hodges_lehmann_shift | 0 | -0.00269482 | 0.00138545 | 0.3639 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e1_e2__dataset_miou_delta | 0.0158483 | -0.000476273 | 0.0281222 | -0.3123 | bca | True |
| accuracy_e2_e3__mean_delta | 0.0101719 | 0.000581027 | 0.0192037 | -0.0188 | bca | True |
| accuracy_e2_e3__median_delta | 0 | 0 | 0 | 0 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e2_e3__hodges_lehmann_shift | 0.00222966 | 0 | 0.00511098 | -0.005891 | bca | True |
| accuracy_e2_e3__dataset_miou_delta | 0.0159243 | 0.000864859 | 0.0322175 | 0.07653 | bca | True |
| accuracy_e4_e5__mean_delta | 0.0132829 | 0.00662456 | 0.020038 | -0.002005 | bca | True |
| accuracy_e4_e5__median_delta | 0 | 0 | 0 | 0 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e4_e5__hodges_lehmann_shift | 0.00168023 | 0 | 0.00450672 | 0.006768 | bca | True |
| accuracy_e4_e5__dataset_miou_delta | 0.00960018 | 0.000399999 | 0.0183532 | -0.06698 | bca | True |
| accuracy_e7_e6__mean_delta | 0.0204781 | 0.0120099 | 0.0297319 | 0.0183 | bca | True |
| accuracy_e7_e6__median_delta | 0 | 0 | 0 | 0 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e7_e6__hodges_lehmann_shift | 0.00246236 | 0 | 0.00547396 | 0.002005 | bca | True |
| accuracy_e7_e6__dataset_miou_delta | -0.00888417 | -0.0271199 | -0.000874836 | -0.8352 | bca | True |
| accuracy_e4_e7__mean_delta | -0.00594622 | -0.0165806 | 0.00537917 | 0.01654 | bca | True |
| accuracy_e4_e7__median_delta | 0 | 0 | 0 | 0.0005013 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e4_e7__hodges_lehmann_shift | -0.00204742 | -0.00657483 | 0 | -0.01291 | bca | True |
| accuracy_e4_e7__dataset_miou_delta | 0.0191389 | -3.46171e-05 | 0.0337879 | -0.251 | bca | True |
| accuracy_e5_e6__mean_delta | 0.00124896 | -0.0094768 | 0.0116473 | -0.01128 | bca | True |
| accuracy_e5_e6__median_delta | 0 | 0 | 0 | 0 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e5_e6__hodges_lehmann_shift | 0 | -0.00337598 | 0.0021484 | 0.2771 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e5_e6__dataset_miou_delta | 0.00065452 | -0.020932 | 0.00886089 | -0.9641 | bca | True |
| accuracy_e1_e6__mean_delta | -0.0180224 | -0.028444 | -0.00758945 | 0.008272 | bca | True |
| accuracy_e1_e6__median_delta | 0 | -0.0014677 | 0 | 0.1903 | percentile_fallback_degenerate_acceleration | True |
| accuracy_e1_e6__hodges_lehmann_shift | -0.00930706 | -0.0139099 | -0.00457641 | 0.002256 | bca | True |
| accuracy_e1_e6__dataset_miou_delta | -0.0199233 | -0.0426528 | -0.0109961 | -0.8898 | bca | True |
| robustness_e1_e6__mean_delta | -0.0212894 | -0.0273333 | -0.015295 | 0.002005 | bca | True |
| robustness_e1_e6__median_delta | -0.0147731 | -0.022037 | -0.0101541 | -0.03347 | bca | True |
| robustness_e1_e6__hodges_lehmann_shift | -0.0211459 | -0.025999 | -0.0165153 | -0.008397 | bca | True |
| descriptive_e1_e3__mean_delta | 0.00723825 | -0.00283929 | 0.0168454 | -0.03134 | bca | True |
| descriptive_e1_e3__median_delta | 0 | 0 | 0 | -0.0008773 | percentile_fallback_degenerate_acceleration | True |
| descriptive_e1_e3__hodges_lehmann_shift | 0.00442176 | 0.000171512 | 0.0083819 | -0.009525 | bca | True |
| noninferiority_e3_e6__dataset_miou_delta | -0.0516959 | -0.0713649 | null | -0.7379 | bca | True |

## Non-inferiority E3 -> E6 (stored decisions)


| baseline mIoU | candidate mIoU | delta | drop | retention | lower bound | margin | passed |
|---|---|---|---|---|---|---|---|
| 0.40653 | 0.354834 | -0.0516959 | 0.0516959 | 0.872836 | -0.0713649 | 0.02 | False |


| sensitivity margin | passed (stored) |
|---|---|
| 0.01 | False |
| 0.015 | False |
| 0.02 | False |
| 0.025 | False |

## Accuracy on corrupted images (E1 -> E6, mIoU-C)


| quantity | E1 | E6 | difference |
|---|---|---|---|
| per-image mIoU-C test: mean delta (Holm p 1, label not rejected) | | | -0.0212894 |
| dataset-level mIoU-C (all-class, union-present (116 classes: background 0 and diseases 1..115); 15 cells per model, nested mean) | 0.248188 | 0.229661 | -0.0185271 |

## Descriptive E1 -> E3 (outside the Holm family)


| n | mean delta | mean delta (pp) | median delta | HL shift |
|---|---|---|---|---|
| 1561 | 0.00723825 | 0.723825 | 0 | 0.00442176 |

## E6-KD observation (descriptive; launches nothing)


| observed drop E3 - E6 | threshold | rule | triggered (stored) |
|---|---|---|---|
| 0.0516959 | 0.01 | strictly_greater | True |

## AM-6: dataset-level mIoU under the GT-present rule (descriptive)


| stage | union-present all-class | GT-present all-class | union-present disease-only | GT-present disease-only |
|---|---|---|---|---|
| E1 | 0.374757 | 0.378016 | 0.370352 | 0.373601 |
| E2 | 0.390605 | 0.390605 | 0.386283 | 0.386283 |
| E3 | 0.40653 | 0.40653 | 0.402318 | 0.402318 |
| E4 | 0.344579 | 0.347575 | 0.339938 | 0.34292 |
| E5 | 0.354179 | 0.357259 | 0.349689 | 0.352756 |
| E6 | 0.354834 | 0.357919 | 0.350309 | 0.353382 |
| E7 | 0.363718 | 0.363718 | 0.359234 | 0.359234 |

## Extension points (not populated by this report)


- AM-8 seed stability: seeds 43 and 44 reported per seed as descriptive tables of the eight contrasts (AM-8; AM-17 item 4)
- Rare-class strata TEST tables: per-class IoU changes by TRAIN pixel-share stratum (AM-17 item 8)
- Default candidates: E2 with lambda = 1 and E3 with alpha = 50 scored descriptively at TEST (AM-17 item 6)

## MDE entry

`/repo/reports/derived/mde_entry_20261009T034825Z.json` sha256 `7749c61eb7b82da6bc250bcce0d2239361460b110ef8b9ab615b44464a90c2ae`, tracked True, unmodified at HEAD True, last commit `8487551ca67bd50ffbe911d1f3e93244841cbc9f`; mde_w 0.01, tau_p 0.01, power_caveat False; problems []. Power caveat: False.
