# Follow-up findings

**Completed:** 66 training runs at 400 steps each, plus independent DIODE evaluation of 42 depth checkpoints. Every training configuration has three seeds. Encoder and decoder weights remained frozen in all runs.

The main new positive result is binary sky-mask extraction. Increasing edit complexity did not consistently improve depth transfer, and the independent measured-depth check limits the generality of the original depth result.

See the [five visual results](assets/README.md) for qualitative successes, failure cases, and aggregate comparison charts. The [original figures](../intrinsics/assets/README.md) remain available without duplication.

## Data and selection

The fresh ImageNet21k split retains 72 training and 24 validation images, with 96 new images from familiar categories and 96 images from 12 new categories. All 192 fresh evaluation paths and exact crops are disjoint from the original 144 images. Near duplicates and overlap with encoder pretraining were not audited.

Only validation loss selects checkpoints. The simple shared normalized-addition edit has the lowest mean depth validation loss across the tested editor configurations. Normal comparisons use the original 24-image test and 24-image unseen-category sets and remain exploratory. Sky edits are additionally audited on the fresh split without retraining.

See the [protocol](FOLLOWUP_PROTOCOL.md), [66-run metrics CSV](FOLLOWUP_METRICS.csv), and [environment/source hashes](FOLLOWUP_ENVIRONMENT.json).

## Depth: extra complexity gives no consistent transfer gain

Mean teacher-target RGB MSE over three seeds; lower is better. The fresh familiar-category and new-category sets each contain 96 images. Parameter counts include the direction dictionary and any conditioner.

| Method | Editor/readout | Trainable parameters | Validation MSE | Fresh familiar MSE | Fresh new-category MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| geodesic | shared | 98,305 | 0.02652 | 0.03246 | 0.05444 |
| geodesic | adaptive-size | 111,426 | 0.02606 | 0.03491 | 0.05738 |
| geodesic | subspace4 | 406,470 | 0.02733 | 0.03275 | 0.05692 |
| geodesic | subspace8 | 799,818 | 0.02661 | 0.03273 | 0.05602 |
| geodesic | budget4 | 110,534 | 0.02558 | 0.03038 | 0.05696 |
| geodesic | budget8 | 110,666 | 0.02889 | 0.03282 | 0.06024 |
| addition | shared | 98,305 | 0.02477 | 0.03153 | 0.05469 |
| addition | adaptive-size | 111,426 | 0.02714 | 0.03562 | 0.05874 |
| addition | subspace4 | 406,470 | 0.02661 | 0.03106 | 0.05756 |
| addition | subspace8 | 799,818 | 0.02903 | 0.03363 | 0.05952 |
| addition | budget4 | 110,534 | 0.02698 | 0.03235 | 0.05940 |
| addition | budget8 | 110,666 | 0.02829 | 0.03224 | 0.06399 |
| n/a | linear-probe | 1,155 | 0.06297 | 0.06105 | 0.10542 |
| n/a | nonlinear-probe | 49,667 | 0.03928 | 0.04060 | 0.07135 |

The compact 4-direction geodesic editor reduces familiar-category MSE from 0.03246 to 0.03038 (about 6.4%), but its paired category-bootstrap 95% interval for the difference is [-0.003989, 0.000077], including no improvement. Its new-category MSE is worse than the shared geodesic baseline. These exploratory intervals do not correct for multiple comparisons.

The factorized dictionaries have approximately one full direction’s parameter budget, plus a conditioner. Factorization also changes the spatial/channel structure, so this is not a pure parameter-count ablation. Neither the 4-/8-direction mixtures nor predicted strength establishes a robust improvement.

## Binary sky masks: positive extraction, limited category transfer

The task is teacher sky versus all other pixels (ADE20K label ID 2). A fixed threshold of 0.5 on the mean RGB decoder output yields the binary prediction. Raw edited images retain texture. This result does not replace the negative result for 150-class segmentation.

Pooled sky-class IoU, averaged over three seeds for trained/random methods:

| Method | Old unseen categories (24 images) | Fresh familiar categories (96 images) | Fresh new categories (96 images) |
| --- | ---: | ---: | ---: |
| zero edit | 0.2934 | 0.4241 | 0.1421 |
| training template | 0.1013 | 0.1133 | 0.0560 |
| geodesic: random direction at learned size | 0.2772 | 0.3822 | 0.1312 |
| geodesic: shared edit | 0.8011 | 0.7547 | 0.3705 |
| addition: random direction at learned size | 0.2218 | 0.3063 | 0.1073 |
| addition: shared edit | 0.8077 | 0.7547 | 0.4129 |
| linear-probe | 0.7337 | 0.7505 | 0.3562 |
| nonlinear-probe | 0.7691 | 0.7997 | 0.3720 |

Fresh new-category own-target IoU exceeds cyclically shuffled-target IoU: 0.3705 versus 0.1901 for geodesic edits and 0.4129 versus 0.2096 for addition. This supports image-specific agreement beyond the spatial template. The large drop from old to fresh categories limits any claim of robust category transfer.

The nonlinear probe uses 49,667 trainable parameters versus 98,305 for the shared edit; these are not exactly matched-capacity models. The probe also avoids the frozen decoder at inference.

## Normals: matched cosine-loss controls do not change the picture

Mean angular error in degrees, lower is better. These use the original teacher-labelled 24/24 evaluation images, not the fresh 192-image audit.

| Method | Original test | Original unseen categories |
| --- | ---: | ---: |
| Original shared geodesic, RGB loss (previous report) | 24.08 | 27.35 |
| Original shared addition, RGB loss (previous report) | 23.29 | 27.63 |
| geodesic/shared, cosine loss | 24.01 | 27.43 |
| addition/shared, cosine loss | 23.34 | 27.14 |
| n/a/linear-probe, cosine loss | 23.98 | 28.87 |
| n/a/nonlinear-probe, cosine loss | 22.79 | 26.93 |

The nonlinear head is competitive with or slightly better than shared decoder editing here. Changing the shared editor’s objective to cosine loss does not produce a major normal-estimation improvement. These remain teacher-agreement results without measured-normal ground truth.

## Measured depth: weak transfer to DIODE

The [official DIODE validation data](https://diode-dataset.org/) provide measured depth and validity masks. This pilot uses 80 fixed images across six scenes, four per scan, without adaptation or model selection on DIODE. It uses the documented 256px center-crop protocol and relative inverse-depth metrics; these are not calibrated metric-depth or standard full-benchmark scores.

All 42 supplied models are evaluated. Values below average three checkpoints per configuration. Lower normalized inverse-depth MSE and higher correlation are better. Forward timings are measured on the same RTX 2080 Ti, batch size one, with cached RGB latents; they exclude the common encoder.

| Method | Editor/readout | Normalized inverse MSE | Inverse Pearson | Forward ms |
| --- | --- | ---: | ---: | ---: |
| Fixed | Training template | 0.11185 | 0.1350 | — |
| geodesic | shared | 0.12104 | 0.2202 | 21.355 |
| geodesic | adaptive-size | 0.11792 | 0.2215 | 21.661 |
| geodesic | subspace4 | 0.11750 | 0.2159 | 21.567 |
| geodesic | subspace8 | 0.11454 | 0.2165 | 21.645 |
| geodesic | budget4 | 0.11633 | 0.2354 | 21.688 |
| geodesic | budget8 | 0.11824 | 0.2268 | 21.679 |
| addition | shared | 0.11831 | 0.2225 | 20.971 |
| addition | adaptive-size | 0.12351 | 0.2130 | 21.179 |
| addition | subspace4 | 0.12166 | 0.2113 | 21.277 |
| addition | subspace8 | 0.12119 | 0.2153 | 21.261 |
| addition | budget4 | 0.12135 | 0.2299 | 21.397 |
| addition | budget8 | 0.11678 | 0.2276 | 21.356 |
| n/a | linear-probe | 0.09504 | 0.3683 | 0.164 |
| n/a | nonlinear-probe | 0.08582 | 0.3955 | 0.261 |
| Reference | Pinned depth teacher | 0.08227 | 0.5926 | — |

Every editor has worse normalized inverse-depth MSE than the training template on this subset, despite somewhat better correlation. Direct probes transfer better than all tested editors, and the pinned teacher retains a substantial correlation advantage. The current evidence does not support broad measured-depth generalization.
