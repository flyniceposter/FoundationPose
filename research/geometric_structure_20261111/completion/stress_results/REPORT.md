# Geometry stress tests

No training or checkpoint selection occurred here. Original 40% missing, 2mm-noise checkpoints were evaluated under changed input corruption. Reused objects make this additional exploratory robustness evidence.

| Condition | Encoder/readout | Reserved depth mm | Reserved normal angle deg |
|---|---|---:|---:|
| lighter_holes | frozen | 10.485 | 21.07 |
| lighter_holes | normal | 10.521 | 19.53 |
| lighter_holes | probe_frozen | 10.732 | 21.52 |
| lighter_holes | probe_normal | 10.771 | 20.18 |
| lighter_holes | nearest observed depth | 8.103 | — |
| larger_holes | frozen | 13.689 | 25.93 |
| larger_holes | normal | 13.926 | 24.40 |
| larger_holes | probe_frozen | 14.178 | 26.72 |
| larger_holes | probe_normal | 14.164 | 24.83 |
| larger_holes | nearest observed depth | 12.394 | — |
| noisier_context | frozen | 11.339 | 22.18 |
| noisier_context | normal | 11.470 | 20.60 |
| noisier_context | probe_frozen | 11.720 | 22.87 |
| noisier_context | probe_normal | 11.735 | 21.24 |
| noisier_context | nearest observed depth | 10.497 | — |
