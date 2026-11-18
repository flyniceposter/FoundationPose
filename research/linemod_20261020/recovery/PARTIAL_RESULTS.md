# Partial recovery diagnostics — rounded log poses

The original evaluation did not complete. These matrices were recovered from rounded text in its preserved log, not from an original full-precision prediction file. The observed prefix is biased by object/frame order. Its pooled accuracy is not a full LINEMOD baseline or an official BOP score. Objects 1, 2, 4 and 5 have complete frame coverage; object 6 has only 339 frames; later objects have no recovered predictions.

| Object (0 = observed pooled) | Recovered valid | Mean translation (mm) | Mean raw rotation (degrees) | Observed ADD(-S) recall |
| --- | --- | --- | --- | --- |
| 1 | 1236 | 4.051 | 2.498 | 98.95% |
| 2 | 1214 | 3.319 | 1.676 | 100.00% |
| 4 | 1201 | 4.270 | 2.142 | 100.00% |
| 5 | 1196 | 4.668 | 1.215 | 100.00% |
| 6 | 339 | 3.102 | 2.131 | 100.00% |
| 0 | 5186 | 4.011 | 1.903 | 99.75% |

All expected-frame CSVs retain missing later poses as failures. Their pooled recall is a coverage-dependent lower bound, not measured full-dataset model performance. observed_subset_summary.csv reports recovered-valid-only accuracy with explicit denominators. Raw rotation is not symmetry corrected. Geometry diagnostics use the existing audited ADD/ADI and MSSD/MSPD implementations. Recall plots spanning all objects must be interpreted alongside missing counts. No GPU work was launched to produce these diagnostics.
