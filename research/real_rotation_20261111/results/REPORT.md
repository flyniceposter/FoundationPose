# Real BOP rotation, symmetry and texture

Real BOP RGB query photographs; references are CAD renders from training poses only. Lower symmetry-aware angle is better.

| Encoder | Reference appearance | Mean rotation error deg ± seed SD | Recall <=30deg | Oracle bank error deg |
|---|---|---:|---:|---:|
| pretrained | textured | 105.45 ± 0.00 | 0.272 | 30.61 |
| pretrained | uniform | 106.95 ± 0.00 | 0.256 | 30.61 |
| ordinary | textured | 104.08 ± 1.20 | 0.272 | 30.61 |
| ordinary | uniform | 104.45 ± 2.17 | 0.278 | 30.61 |
| symmetry | textured | 104.86 ± 1.03 | 0.274 | 30.61 |
| symmetry | uniform | 105.83 ± 1.94 | 0.265 | 30.61 |

Raw and symmetry-aware per-object results, seed results, oracle coverage and excess errors are in evaluation.json. Pretrained is identical across seeds; its zero seed SD does not indicate statistical certainty.

GT visible masks/crops and known identities are supplied. Evaluation uses later frames in the same sequences; no independent-scene or official BOP pose benchmark claim. Uniform references change appearance while real queries stay unchanged; this gap is not by itself proof of newly learned texture. Symmetry equivalence here is a rotation-only metric, not a full-pose loss handling symmetry translations.
