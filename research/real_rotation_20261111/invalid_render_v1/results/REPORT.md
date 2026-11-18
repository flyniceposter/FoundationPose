# Real BOP rotation, symmetry and texture

Real BOP RGB query photographs; references are CAD renders from training poses only. Lower symmetry-aware angle is better.

| Encoder | Reference appearance | Mean rotation error deg ± seed SD | Recall <=30deg | Oracle bank error deg |
|---|---|---:|---:|---:|
| pretrained | textured | 111.01 ± 0.00 | 0.206 | 30.61 |
| pretrained | uniform | 111.01 ± 0.00 | 0.206 | 30.61 |
| ordinary | textured | 104.37 ± 0.19 | 0.285 | 30.61 |
| ordinary | uniform | 104.37 ± 0.19 | 0.285 | 30.61 |
| symmetry | textured | 103.99 ± 0.49 | 0.285 | 30.61 |
| symmetry | uniform | 103.99 ± 0.49 | 0.285 | 30.61 |

Raw and symmetry-aware per-object results, seed results, oracle coverage and excess errors are in evaluation.json. Pretrained is identical across seeds; its zero seed SD does not indicate statistical certainty.

GT visible masks/crops and known identities are supplied. Evaluation uses later frames in the same sequences; no independent-scene or official BOP pose benchmark claim. Uniform references change appearance while real queries stay unchanged; this gap is not by itself proof of newly learned texture. Symmetry equivalence here is a rotation-only metric, not a full-pose loss handling symmetry translations.
