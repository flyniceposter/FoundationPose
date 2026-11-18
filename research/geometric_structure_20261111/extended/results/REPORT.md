# Extended BOP visible-geometry results

Development-only LINEMOD study. Objects held out from this adaptation: 12,13,14,15. No pose-estimation claim.

| Arm | Unseen-object MAE / diameter (mean ± seed SD) | Unseen-object MAE mm (mean ± seed SD) |
|---|---:|---:|
| frozen | 0.12038 ± 0.00105 | 27.05 ± 0.24 |
| depth | 0.12410 ± 0.00652 | 28.00 ± 1.54 |
| structure | 0.12341 ± 0.00605 | 27.84 ± 1.43 |
| probe_frozen | 0.11979 ± 0.00119 | 26.93 ± 0.29 |
| probe_depth | 0.12245 ± 0.00343 | 27.64 ± 0.85 |
| probe_structure | 0.12171 ± 0.00304 | 27.48 ± 0.76 |
| zero | 0.12314 | 27.77 |
| mean_template | 0.12705 | 28.63 |

Fresh probes freeze each encoder and reinitialize a common decoder. Seed SD is variability, not a confidence interval. See evaluation.json for seen-object, slope, per-object, and feature-shuffling results.

All checkpoint selection used development objects 10,11. The figure shows fixed first sampled frames of all four evaluation objects for seed 42; predictions are masked by GT for display. Uniform color is not used in this geometry experiment.

## Interpretation

This run did not demonstrate an improvement from geometry fine-tuning. On adaptation-held-out objects, the frozen encoder had lower mean normalized and millimeter depth error than either adapted arm. The structure arm was slightly better than depth-only adaptation but remained worse than the frozen baseline. Fresh-decoder probes also favored the original encoder, so the evidence does not support improved transferable geometry features.

On later frames of training objects, mean errors were 24.60 mm frozen, 24.45 mm depth-only and 24.43 mm structure. These small improvements do not transfer to held-out objects. The constant-zero relative-depth baseline reaches 27.77 mm on held-out objects; adapted reconstruction performance close to this baseline is particularly weak evidence for useful shape learning. Predictions capture some coarse structure but miss substantial local geometry.

All model variants become worse when feature locations are shuffled. This supports dependence on spatial arrangement, but it does not establish newly learned 3D understanding. The selected checkpoints were often early (200 or 600 steps), despite training all runs to 1000 steps. Simply keeping the final longer-trained checkpoint would ignore the development selection rule.

The experiment used 1560 sampled frame identities: 588 train, 84 temporal buffer (unused), 240 development, 168 seen-object temporal evaluation, and 480 adaptation-held-out evaluation. There were nine 1000-step adaptation/readout runs and nine 400-step fresh-decoder runs, totaling 12,600 optimizer steps. Seed variation is reported across three seeds, not treated as a confidence interval.

## What was verified

Numerical checks covered exact-target loss, invariance to invalid target pixels, constant-offset slope invariance, masked gradients, metric correctness and object-macro weighting. Artifact checks verified disjoint adaptation/development/evaluation object sets, unique frame identities, matching initialization, real encoder updates, unchanged BatchNorm running statistics, unchanged encoders during fresh probes, finite checkpoints, development-selected checkpoint minima and complete evaluation coverage. See verification.json.

## Practical next step

Do not replace the deployed encoder with these checkpoints. A better next experiment would use proper synthetic/real training scenes, supervision from rendered clean surfaces or reliable normals/correspondences, and a pose-preserving objective alongside geometry supervision. This run alone cannot distinguish data quality, objective choice, RGB-only input mismatch and adaptation overfitting as causes. Any new method should be selected on development data, with new final-evaluation objects/scenes rather than repeatedly selecting against these four objects.
