# BOP real RGB-D geometry completion

CAD-supervised missing-depth prediction, not RGB-only depth estimation. Primary reserved objects: 3,7. Previous evaluation objects are exploratory. Lower error is better.

| Arm | Reserved depth MAE mm (mean ± seed SD) | Reserved normal angle deg |
|---|---:|---:|
| frozen | 11.385 ± 0.171 | 22.00 |
| depth | 11.665 ± 0.568 | 112.88 |
| normal | 11.486 ± 0.302 | 20.38 |
| probe_frozen | 11.726 ± 0.234 | 22.69 |
| probe_depth | 12.069 ± 0.353 | 21.78 |
| probe_normal | 11.739 ± 0.456 | 21.00 |
| zero | 30.256 | — |
| nearest | 9.494 | — |

Normal angle is the object-macro average of per-image median angular errors. Depth-only decoder normal output is unsupervised and its normal score is not a meaningful learned-normal result.

All RGB images are real BOP LINEMOD photographs. Depth/normal labels come from the matching BOP CAD model rendered using annotated pose and camera calibration. Missing regions are artificial; real sensor missingness and deployed pose performance remain untested. GT masks/crops and known diameters make this a controlled diagnostic. Two reserved objects and three seeds are insufficient for a broad generalization claim.
