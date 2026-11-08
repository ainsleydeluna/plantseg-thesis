# Robustness, descriptive outputs

- Status: official; policy official; split test.
- Class space: all-class, union-present (116 classes: background 0 and diseases 1..115).
- Metric: dataset-level all-class union-present pooled mIoU of each artifact's totals (pooled_miou_from_totals, float64; DL-73, P28).
- Descriptive only: no hypothesis test, no bootstrap and no significance value; never in family.json (D22).
- Cache: dc7185f0b5fc75e0e0fbd0608e5e30f621c189b6ed2d8ad8a4e57d4a3c26db47.
- rCD reference: {'file_id': 'E1_s42_80k', 'stage': 'E1', 'seed': 42}.

## Q3 mIoU-C and Q4 RPD

| model | stage | seed | clean | mIoU-C | RPD % |
|---|---|---|---|---|---|
| teacher_s42_40k | teacher | None | 0.399243 | 0.303786 | 23.909434 |
| E1_s42_80k | E1 | 42 | 0.374757 | 0.248188 | 33.773620 |
| E1_s43_80k | E1 | 43 | 0.371892 | 0.241668 | 35.016528 |
| E1_s44_80k | E1 | 44 | 0.372824 | 0.247019 | 33.743767 |
| E2_s42_lambda0.5_80k | E2 | 42 | 0.390605 | 0.271591 | 30.469087 |
| E3_s42_lambda0.5_alpha25_80k | E3 | 42 | 0.406530 | 0.266212 | 34.516011 |
| E5_s42_sel | E5 | 42 | 0.354179 | 0.219210 | 38.107620 |
| E6_s42_sel | E6 | 42 | 0.354834 | 0.229661 | 35.276486 |
| E6_s43_sel | E6 | 43 | 0.374334 | 0.241049 | 35.605737 |
| E6_s44_sel | E6 | 44 | 0.372353 | 0.227357 | 38.940590 |
| E4_s42_list42 | E4 | 42 | 0.344579 | 0.202382 | 41.266867 |
| E7_s42_list42 | E7 | 42 | 0.363718 | 0.213906 | 41.188949 |

## Q5 rCD per corruption and Q6 their mean (students at seed 42; E1 = 1)

| model | motion_blur | gaussian_noise | jpeg_compression | brightness | fog | mean |
|---|---|---|---|---|---|---|
| E1_s42_80k | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| E2_s42_lambda0.5_80k | 0.921464 | 0.931126 | 0.958401 | 0.932229 | 0.972621 | 0.943168 |
| E3_s42_lambda0.5_alpha25_80k | 1.055163 | 1.090622 | 1.073484 | 1.308250 | 1.167549 | 1.139014 |
| E5_s42_sel | 1.138807 | 0.914128 | 1.220259 | 0.952259 | 1.278542 | 1.100799 |
| E6_s42_sel | 1.076378 | 0.911200 | 0.962673 | 1.115900 | 1.046507 | 1.022532 |
| E4_s42_list42 | 1.121890 | 0.887457 | 1.279629 | 1.068620 | 1.564190 | 1.184357 |
| E7_s42_list42 | 1.091528 | 1.025451 | 1.385202 | 1.683309 | 1.424343 | 1.321967 |

## Q7 per corruption: the mean over severities 1-3

| model | motion_blur | gaussian_noise | jpeg_compression | brightness | fog |
|---|---|---|---|---|---|
| teacher_s42_40k | 0.274220 | 0.185074 | 0.340718 | 0.387319 | 0.331600 |
| E1_s42_80k | 0.241057 | 0.090538 | 0.327942 | 0.346290 | 0.235113 |
| E2_s42_lambda0.5_80k | 0.267405 | 0.125962 | 0.345738 | 0.364068 | 0.254784 |
| E3_s42_lambda0.5_alpha25_80k | 0.265454 | 0.096554 | 0.356274 | 0.369288 | 0.243488 |
| E5_s42_sel | 0.201920 | 0.094367 | 0.297052 | 0.327072 | 0.175638 |
| E6_s42_sel | 0.210922 | 0.095853 | 0.309766 | 0.323068 | 0.208695 |
| E4_s42_list42 | 0.194582 | 0.092347 | 0.284673 | 0.314159 | 0.126149 |
| E7_s42_list42 | 0.217780 | 0.072265 | 0.298869 | 0.315800 | 0.164817 |

## Q8 the Kamann-Rother average (AM-16 item 6: non-noise s1-5, gaussian_noise s1-3, nested) and the LE6 series means

| model | average | non_noise_s1_5 | all_five_s1_4 |
|---|---|---|---|
| teacher_s42_40k | 0.269044 | 0.290037 | 0.279758 |
| E1_s42_80k | 0.205227 | 0.233899 | 0.224147 |
| E2_s42_lambda0.5_80k | 0.227227 | 0.252544 | 0.246010 |
| E3_s42_lambda0.5_alpha25_80k | 0.221642 | 0.252914 | 0.241584 |
| E5_s42_sel | 0.178652 | 0.199724 | 0.195794 |
| E6_s42_sel | 0.190810 | 0.214549 | 0.207192 |
| E4_s42_list42 | 0.164990 | 0.183151 | 0.179937 |
| E7_s42_list42 | 0.174805 | 0.200439 | 0.191585 |

## Q2 per severity: non_noise_s1_5

| model | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| teacher_s42_40k | 0.364278 | 0.337424 | 0.298690 | 0.252933 | 0.196858 |
| E1_s42_80k | 0.332620 | 0.289779 | 0.240402 | 0.187303 | 0.119394 |
| E2_s42_lambda0.5_80k | 0.350054 | 0.313606 | 0.260336 | 0.206900 | 0.131822 |
| E3_s42_lambda0.5_alpha25_80k | 0.356710 | 0.310547 | 0.258621 | 0.205956 | 0.132737 |
| E5_s42_sel | 0.295110 | 0.252053 | 0.204099 | 0.153827 | 0.093529 |
| E6_s42_sel | 0.306601 | 0.263355 | 0.219381 | 0.171240 | 0.112165 |
| E4_s42_list42 | 0.274547 | 0.229059 | 0.186067 | 0.137681 | 0.088403 |
| E7_s42_list42 | 0.293919 | 0.251427 | 0.202603 | 0.153582 | 0.100665 |

## Q2 per severity: all_five_s1_4

| model | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| teacher_s42_40k | 0.350355 | 0.306860 | 0.254143 | 0.207673 |
| E1_s42_80k | 0.298337 | 0.247937 | 0.198290 | 0.152025 |
| E2_s42_lambda0.5_80k | 0.324374 | 0.272931 | 0.217469 | 0.169264 |
| E3_s42_lambda0.5_alpha25_80k | 0.320068 | 0.265512 | 0.213056 | 0.167699 |
| E5_s42_sel | 0.268813 | 0.218760 | 0.170057 | 0.125548 |
| E6_s42_sel | 0.281292 | 0.226367 | 0.181323 | 0.139784 |
| E4_s42_list42 | 0.251163 | 0.200437 | 0.155545 | 0.112603 |
| E7_s42_list42 | 0.263028 | 0.212033 | 0.166658 | 0.124623 |

## Q1 per cell: all-class (disease-only)

- teacher_s42_40k: motion_blur_s1 0.343929 (0.339074); motion_blur_s2 0.286795 (0.281583); motion_blur_s3 0.191935 (0.186050); motion_blur_s4 0.113215 (0.106827); motion_blur_s5 0.076688 (0.070109); gaussian_noise_s1 0.294663 (0.289525); gaussian_noise_s2 0.184602 (0.178797); gaussian_noise_s3 0.075957 (0.069883); gaussian_noise_s4 0.026633 (0.023106); jpeg_compression_s1 0.362717 (0.358118); jpeg_compression_s2 0.338386 (0.333630); jpeg_compression_s3 0.321049 (0.316176); jpeg_compression_s4 0.273812 (0.268647); jpeg_compression_s5 0.191542 (0.185816); brightness_s1 0.393256 (0.388828); brightness_s2 0.392740 (0.388291); brightness_s3 0.375961 (0.371402); brightness_s4 0.328485 (0.323581); brightness_s5 0.268670 (0.263350); fog_s1 0.357212 (0.352481); fog_s2 0.331775 (0.326869); fog_s3 0.305814 (0.300736); fog_s4 0.296219 (0.291090); fog_s5 0.250532 (0.245132)
- E1_s42_80k: motion_blur_s1 0.324011 (0.319255); motion_blur_s2 0.250392 (0.245109); motion_blur_s3 0.148767 (0.142861); motion_blur_s4 0.081652 (0.075539); motion_blur_s5 0.047462 (0.041267); gaussian_noise_s1 0.161204 (0.155418); gaussian_noise_s2 0.080570 (0.075175); gaussian_noise_s3 0.029841 (0.026330); gaussian_noise_s4 0.010916 (0.008720); jpeg_compression_s1 0.347218 (0.342655); jpeg_compression_s2 0.324990 (0.320274); jpeg_compression_s3 0.311617 (0.306824); jpeg_compression_s4 0.236144 (0.230842); jpeg_compression_s5 0.135214 (0.129217); brightness_s1 0.369647 (0.365226); brightness_s2 0.351199 (0.346651); brightness_s3 0.318025 (0.313250); brightness_s4 0.258112 (0.252933); brightness_s5 0.187998 (0.182358); fog_s1 0.289605 (0.284533); fog_s2 0.232535 (0.227047); fog_s3 0.183199 (0.177369); fog_s4 0.173303 (0.167481); fog_s5 0.106901 (0.100714)
- E1_s43_80k: motion_blur_s1 0.315065 (0.310241); motion_blur_s2 0.255922 (0.250716); motion_blur_s3 0.154589 (0.148804); gaussian_noise_s1 0.131288 (0.125134); gaussian_noise_s2 0.057956 (0.052419); gaussian_noise_s3 0.017437 (0.013953); jpeg_compression_s1 0.338512 (0.333892); jpeg_compression_s2 0.321817 (0.317099); jpeg_compression_s3 0.300502 (0.295630); brightness_s1 0.361712 (0.357214); brightness_s2 0.340369 (0.335712); brightness_s3 0.298850 (0.293899); fog_s1 0.293244 (0.288205); fog_s2 0.246234 (0.240885); fog_s3 0.191528 (0.185828)
- E1_s44_80k: motion_blur_s1 0.317107 (0.312276); motion_blur_s2 0.246839 (0.241497); motion_blur_s3 0.146401 (0.140470); gaussian_noise_s1 0.164392 (0.158485); gaussian_noise_s2 0.091787 (0.085690); gaussian_noise_s3 0.040388 (0.034786); jpeg_compression_s1 0.345858 (0.341292); jpeg_compression_s2 0.323691 (0.318977); jpeg_compression_s3 0.302675 (0.297821); brightness_s1 0.366581 (0.362105); brightness_s2 0.345183 (0.340550); brightness_s3 0.303764 (0.298854); fog_s1 0.288391 (0.283276); fog_s2 0.234713 (0.229216); fog_s3 0.187520 (0.181716)
- E2_s42_lambda0.5_80k: motion_blur_s1 0.341729 (0.337103); motion_blur_s2 0.285201 (0.280139); motion_blur_s3 0.175286 (0.169496); motion_blur_s4 0.089617 (0.083444); motion_blur_s5 0.054133 (0.047929); gaussian_noise_s1 0.221653 (0.216129); gaussian_noise_s2 0.110232 (0.104206); gaussian_noise_s3 0.046000 (0.041277); gaussian_noise_s4 0.018723 (0.014560); jpeg_compression_s1 0.365008 (0.360520); jpeg_compression_s2 0.347144 (0.342524); jpeg_compression_s3 0.325061 (0.320273); jpeg_compression_s4 0.255523 (0.250338); jpeg_compression_s5 0.138259 (0.132322); brightness_s1 0.384282 (0.379907); brightness_s2 0.371694 (0.367242); brightness_s3 0.336228 (0.331543); brightness_s4 0.273735 (0.268635); brightness_s5 0.203059 (0.197479); fog_s1 0.309197 (0.304279); fog_s2 0.250387 (0.245055); fog_s3 0.204769 (0.199143); fog_s4 0.208724 (0.203148); fog_s5 0.131839 (0.125804)
- E3_s42_lambda0.5_alpha25_80k: motion_blur_s1 0.353996 (0.349400); motion_blur_s2 0.270895 (0.265755); motion_blur_s3 0.171472 (0.165694); motion_blur_s4 0.105702 (0.099641); motion_blur_s5 0.057688 (0.051455); gaussian_noise_s1 0.173499 (0.167724); gaussian_noise_s2 0.085370 (0.079418); gaussian_noise_s3 0.030794 (0.025709); gaussian_noise_s4 0.014674 (0.009410); jpeg_compression_s1 0.370381 (0.365909); jpeg_compression_s2 0.358588 (0.354053); jpeg_compression_s3 0.339854 (0.335217); jpeg_compression_s4 0.267185 (0.261976); jpeg_compression_s5 0.145831 (0.139850); brightness_s1 0.400899 (0.396631); brightness_s2 0.376427 (0.371974); brightness_s3 0.330538 (0.325757); brightness_s4 0.268174 (0.262950); brightness_s5 0.199032 (0.193349); fog_s1 0.301564 (0.296532); fog_s2 0.236279 (0.230787); fog_s3 0.192622 (0.186863); fog_s4 0.182761 (0.176930); fog_s5 0.128395 (0.122307)
- E5_s42_sel: motion_blur_s1 0.283068 (0.278070); motion_blur_s2 0.202786 (0.197234); motion_blur_s3 0.119908 (0.113988); motion_blur_s4 0.070421 (0.064570); motion_blur_s5 0.038841 (0.032994); gaussian_noise_s1 0.163628 (0.157981); gaussian_noise_s2 0.085586 (0.080116); gaussian_noise_s3 0.033886 (0.029882); gaussian_noise_s4 0.012431 (0.009836); jpeg_compression_s1 0.318612 (0.313905); jpeg_compression_s2 0.295966 (0.291112); jpeg_compression_s3 0.276579 (0.271601); jpeg_compression_s4 0.199333 (0.193847); jpeg_compression_s5 0.108897 (0.102790); brightness_s1 0.354548 (0.350064); brightness_s2 0.336744 (0.332128); brightness_s3 0.289923 (0.284968); brightness_s4 0.222773 (0.217368); brightness_s5 0.154524 (0.148692); fog_s1 0.224210 (0.218769); fog_s2 0.172716 (0.166928); fog_s3 0.129988 (0.123935); fog_s4 0.122781 (0.116757); fog_s5 0.071853 (0.065848)
- E6_s42_sel: motion_blur_s1 0.287440 (0.282457); motion_blur_s2 0.210601 (0.205089); motion_blur_s3 0.134724 (0.128824); motion_blur_s4 0.083329 (0.077394); motion_blur_s5 0.049266 (0.043400); gaussian_noise_s1 0.180056 (0.174585); gaussian_noise_s2 0.078414 (0.072665); gaussian_noise_s3 0.029090 (0.024096); gaussian_noise_s4 0.013959 (0.009244); jpeg_compression_s1 0.330094 (0.325441); jpeg_compression_s2 0.309542 (0.304740); jpeg_compression_s3 0.289662 (0.284729); jpeg_compression_s4 0.223442 (0.218064); jpeg_compression_s5 0.129718 (0.123709); brightness_s1 0.350022 (0.345447); brightness_s2 0.330356 (0.325620); brightness_s3 0.288826 (0.283787); brightness_s4 0.221338 (0.215814); brightness_s5 0.165571 (0.159660); fog_s1 0.258848 (0.253627); fog_s2 0.202922 (0.197318); fog_s3 0.164315 (0.158480); fog_s4 0.156852 (0.151037); fog_s5 0.104106 (0.098433)
- E6_s43_sel: motion_blur_s1 0.303075 (0.298139); motion_blur_s2 0.211618 (0.206049); motion_blur_s3 0.122859 (0.116824); gaussian_noise_s1 0.200466 (0.194956); gaussian_noise_s2 0.105869 (0.099985); gaussian_noise_s3 0.035012 (0.030628); jpeg_compression_s1 0.348850 (0.344326); jpeg_compression_s2 0.326258 (0.321630); jpeg_compression_s3 0.302926 (0.298133); brightness_s1 0.370408 (0.365946); brightness_s2 0.352059 (0.347452); brightness_s3 0.305265 (0.300294); fog_s1 0.255560 (0.250216); fog_s2 0.208829 (0.203155); fog_s3 0.166687 (0.160734)
- E6_s44_sel: motion_blur_s1 0.305382 (0.300509); motion_blur_s2 0.232052 (0.226711); motion_blur_s3 0.131926 (0.125916); gaussian_noise_s1 0.159234 (0.153404); gaussian_noise_s2 0.073381 (0.067847); gaussian_noise_s3 0.019691 (0.017903); jpeg_compression_s1 0.338319 (0.333724); jpeg_compression_s2 0.314649 (0.309886); jpeg_compression_s3 0.290245 (0.285321); brightness_s1 0.361101 (0.356604); brightness_s2 0.327718 (0.322971); brightness_s3 0.262760 (0.257548); fog_s1 0.241530 (0.236137); fog_s2 0.198298 (0.192591); fog_s3 0.154066 (0.148098)
- E4_s42_list42: motion_blur_s1 0.272558 (0.267407); motion_blur_s2 0.193530 (0.187816); motion_blur_s3 0.117658 (0.111501); motion_blur_s4 0.070144 (0.063852); motion_blur_s5 0.042427 (0.036128); gaussian_noise_s1 0.157631 (0.151807); gaussian_noise_s2 0.085950 (0.080304); gaussian_noise_s3 0.033460 (0.029350); gaussian_noise_s4 0.012289 (0.009682); jpeg_compression_s1 0.310070 (0.305224); jpeg_compression_s2 0.284587 (0.279550); jpeg_compression_s3 0.259361 (0.254142); jpeg_compression_s4 0.188057 (0.182326); jpeg_compression_s5 0.109835 (0.103590); brightness_s1 0.345098 (0.340469); brightness_s2 0.320246 (0.315434); brightness_s3 0.277132 (0.272023); brightness_s4 0.206788 (0.201213); brightness_s5 0.143707 (0.137737); fog_s1 0.170460 (0.164501); fog_s2 0.117873 (0.111554); fog_s3 0.090114 (0.083645); fog_s4 0.085736 (0.079254); fog_s5 0.057643 (0.051054)
- E7_s42_list42: motion_blur_s1 0.290309 (0.285311); motion_blur_s2 0.220495 (0.214984); motion_blur_s3 0.142538 (0.136639); motion_blur_s4 0.088973 (0.082958); motion_blur_s5 0.057944 (0.051897); gaussian_noise_s1 0.139465 (0.133533); gaussian_noise_s2 0.054453 (0.048279); gaussian_noise_s3 0.022877 (0.017593); gaussian_noise_s4 0.008786 (0.004773); jpeg_compression_s1 0.324633 (0.319887); jpeg_compression_s2 0.299591 (0.294686); jpeg_compression_s3 0.272384 (0.267293); jpeg_compression_s4 0.199737 (0.194085); jpeg_compression_s5 0.109198 (0.102965); brightness_s1 0.347823 (0.343184); brightness_s2 0.323811 (0.318976); brightness_s3 0.275765 (0.270571); brightness_s4 0.211531 (0.205888); brightness_s5 0.161896 (0.155918); fog_s1 0.212912 (0.207310); fog_s2 0.161813 (0.155851); fog_s3 0.119725 (0.113500); fog_s4 0.114088 (0.107832); fog_s5 0.073623 (0.067313)

## Q9 provenance

- Inventory sha256 e4e3b90f13121fe863436f671b73d87ae2ea344bebf8b513a9851a9cf3a64650; evaluator commit dc40dd2eb5b8be01882f53c37b7f6a3ea9365b71.
- Code commit dc40dd2eb5b8be01882f53c37b7f6a3ea9365b71; files clean at that commit: True.
- Per-artifact file hashes, record hashes and runtime tuples: robustness_descriptive.json, `models`.
