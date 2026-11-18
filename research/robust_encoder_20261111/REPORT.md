# Robust encoder experiment — November 11–12, 2026

Exploratory local BOP LINEMOD development experiments. Previously inspected data; GT object masks; no official benchmark claim. See PROTOCOL.md for the fixed design.

| Arm | Seed | Evaluation condition | Depth MAE mm | Normal error degrees |
|---|---:|---|---:|---:|
| combined | 123 | evaluation/standard | 7.236 | 18.66 |
| combined | 123 | evaluation/grayscale | 7.230 | 18.75 |
| combined | 123 | evaluation/occlusion | 10.658 | 24.05 |
| combined | 123 | evaluation/extra_dropout | 8.919 | 20.76 |
| combined | 123 | excluded/standard | 10.851 | 22.68 |
| combined | 123 | excluded/grayscale | 10.924 | 22.97 |
| combined | 123 | excluded/occlusion | 15.503 | 28.83 |
| combined | 123 | excluded/extra_dropout | 13.278 | 25.28 |
| combined | 2026 | evaluation/standard | 7.365 | 18.63 |
| combined | 2026 | evaluation/grayscale | 7.376 | 18.72 |
| combined | 2026 | evaluation/occlusion | 11.210 | 24.00 |
| combined | 2026 | evaluation/extra_dropout | 9.126 | 20.88 |
| combined | 2026 | excluded/standard | 10.866 | 22.22 |
| combined | 2026 | excluded/grayscale | 10.916 | 22.53 |
| combined | 2026 | excluded/occlusion | 15.817 | 27.26 |
| combined | 2026 | excluded/extra_dropout | 13.480 | 24.58 |
| combined | 42 | evaluation/standard | 7.266 | 18.64 |
| combined | 42 | evaluation/grayscale | 7.292 | 18.70 |
| combined | 42 | evaluation/occlusion | 10.878 | 24.06 |
| combined | 42 | evaluation/extra_dropout | 8.984 | 20.85 |
| combined | 42 | excluded/standard | 10.496 | 21.86 |
| combined | 42 | excluded/grayscale | 10.553 | 22.20 |
| combined | 42 | excluded/occlusion | 15.689 | 27.27 |
| combined | 42 | excluded/extra_dropout | 12.913 | 24.25 |
| frozen | 123 | evaluation/standard | 7.623 | 19.31 |
| frozen | 123 | evaluation/grayscale | 7.645 | 19.28 |
| frozen | 123 | evaluation/occlusion | 11.832 | 25.35 |
| frozen | 123 | evaluation/extra_dropout | 9.317 | 21.40 |
| frozen | 123 | excluded/standard | 10.231 | 22.09 |
| frozen | 123 | excluded/grayscale | 10.283 | 22.20 |
| frozen | 123 | excluded/occlusion | 15.349 | 27.62 |
| frozen | 123 | excluded/extra_dropout | 12.320 | 24.20 |
| frozen | 2026 | evaluation/standard | 7.312 | 19.09 |
| frozen | 2026 | evaluation/grayscale | 7.324 | 19.12 |
| frozen | 2026 | evaluation/occlusion | 11.318 | 24.97 |
| frozen | 2026 | evaluation/extra_dropout | 8.940 | 21.29 |
| frozen | 2026 | excluded/standard | 10.183 | 22.96 |
| frozen | 2026 | excluded/grayscale | 10.216 | 23.08 |
| frozen | 2026 | excluded/occlusion | 15.102 | 30.65 |
| frozen | 2026 | excluded/extra_dropout | 12.262 | 25.47 |
| frozen | 42 | evaluation/standard | 7.333 | 19.05 |
| frozen | 42 | evaluation/grayscale | 7.332 | 19.11 |
| frozen | 42 | evaluation/occlusion | 11.370 | 25.02 |
| frozen | 42 | evaluation/extra_dropout | 9.047 | 21.14 |
| frozen | 42 | excluded/standard | 10.656 | 23.39 |
| frozen | 42 | excluded/grayscale | 10.686 | 23.44 |
| frozen | 42 | excluded/occlusion | 15.629 | 28.43 |
| frozen | 42 | excluded/extra_dropout | 12.870 | 25.74 |
| geometry | 123 | evaluation/standard | 7.172 | 18.20 |
| geometry | 123 | evaluation/grayscale | 7.202 | 18.24 |
| geometry | 123 | evaluation/occlusion | 10.337 | 23.22 |
| geometry | 123 | evaluation/extra_dropout | 8.808 | 20.31 |
| geometry | 123 | excluded/standard | 10.718 | 19.98 |
| geometry | 123 | excluded/grayscale | 10.781 | 20.22 |
| geometry | 123 | excluded/occlusion | 15.696 | 25.49 |
| geometry | 123 | excluded/extra_dropout | 12.887 | 21.74 |
| geometry | 2026 | evaluation/standard | 7.032 | 18.21 |
| geometry | 2026 | evaluation/grayscale | 7.124 | 18.27 |
| geometry | 2026 | evaluation/occlusion | 10.117 | 23.06 |
| geometry | 2026 | evaluation/extra_dropout | 8.622 | 20.17 |
| geometry | 2026 | excluded/standard | 10.648 | 21.80 |
| geometry | 2026 | excluded/grayscale | 10.664 | 21.93 |
| geometry | 2026 | excluded/occlusion | 15.716 | 27.77 |
| geometry | 2026 | excluded/extra_dropout | 13.107 | 23.55 |
| geometry | 42 | evaluation/standard | 7.050 | 18.27 |
| geometry | 42 | evaluation/grayscale | 7.123 | 18.35 |
| geometry | 42 | evaluation/occlusion | 10.138 | 23.32 |
| geometry | 42 | evaluation/extra_dropout | 8.615 | 20.31 |
| geometry | 42 | excluded/standard | 10.595 | 22.05 |
| geometry | 42 | excluded/grayscale | 10.629 | 22.26 |
| geometry | 42 | excluded/occlusion | 15.892 | 25.74 |
| geometry | 42 | excluded/extra_dropout | 13.044 | 23.69 |
| geometry_symmetry | 123 | evaluation/standard | 7.254 | 18.65 |
| geometry_symmetry | 123 | evaluation/grayscale | 7.237 | 18.74 |
| geometry_symmetry | 123 | evaluation/occlusion | 10.766 | 24.09 |
| geometry_symmetry | 123 | evaluation/extra_dropout | 8.970 | 20.77 |
| geometry_symmetry | 123 | excluded/standard | 10.699 | 22.63 |
| geometry_symmetry | 123 | excluded/grayscale | 10.767 | 22.95 |
| geometry_symmetry | 123 | excluded/occlusion | 15.382 | 28.50 |
| geometry_symmetry | 123 | excluded/extra_dropout | 13.098 | 25.19 |
| geometry_symmetry | 2026 | evaluation/standard | 7.327 | 18.39 |
| geometry_symmetry | 2026 | evaluation/grayscale | 7.364 | 18.42 |
| geometry_symmetry | 2026 | evaluation/occlusion | 10.718 | 23.88 |
| geometry_symmetry | 2026 | evaluation/extra_dropout | 8.995 | 20.58 |
| geometry_symmetry | 2026 | excluded/standard | 11.248 | 21.48 |
| geometry_symmetry | 2026 | excluded/grayscale | 11.329 | 21.89 |
| geometry_symmetry | 2026 | excluded/occlusion | 15.730 | 26.91 |
| geometry_symmetry | 2026 | excluded/extra_dropout | 13.705 | 23.47 |
| geometry_symmetry | 42 | evaluation/standard | 7.260 | 18.60 |
| geometry_symmetry | 42 | evaluation/grayscale | 7.285 | 18.69 |
| geometry_symmetry | 42 | evaluation/occlusion | 10.875 | 24.05 |
| geometry_symmetry | 42 | evaluation/extra_dropout | 8.990 | 20.79 |
| geometry_symmetry | 42 | excluded/standard | 10.413 | 21.99 |
| geometry_symmetry | 42 | excluded/grayscale | 10.490 | 22.33 |
| geometry_symmetry | 42 | excluded/occlusion | 15.768 | 27.34 |
| geometry_symmetry | 42 | excluded/extra_dropout | 12.871 | 24.28 |
| geometry_texture | 123 | evaluation/standard | 7.177 | 18.20 |
| geometry_texture | 123 | evaluation/grayscale | 7.209 | 18.24 |
| geometry_texture | 123 | evaluation/occlusion | 10.336 | 23.23 |
| geometry_texture | 123 | evaluation/extra_dropout | 8.810 | 20.31 |
| geometry_texture | 123 | excluded/standard | 10.738 | 19.99 |
| geometry_texture | 123 | excluded/grayscale | 10.804 | 20.21 |
| geometry_texture | 123 | excluded/occlusion | 15.702 | 25.49 |
| geometry_texture | 123 | excluded/extra_dropout | 12.902 | 21.71 |
| geometry_texture | 2026 | evaluation/standard | 7.008 | 18.23 |
| geometry_texture | 2026 | evaluation/grayscale | 7.094 | 18.31 |
| geometry_texture | 2026 | evaluation/occlusion | 10.099 | 23.06 |
| geometry_texture | 2026 | evaluation/extra_dropout | 8.592 | 20.18 |
| geometry_texture | 2026 | excluded/standard | 10.654 | 21.74 |
| geometry_texture | 2026 | excluded/grayscale | 10.676 | 21.90 |
| geometry_texture | 2026 | excluded/occlusion | 15.721 | 27.60 |
| geometry_texture | 2026 | excluded/extra_dropout | 13.015 | 23.40 |
| geometry_texture | 42 | evaluation/standard | 6.986 | 18.05 |
| geometry_texture | 42 | evaluation/grayscale | 7.050 | 18.18 |
| geometry_texture | 42 | evaluation/occlusion | 10.065 | 23.22 |
| geometry_texture | 42 | evaluation/extra_dropout | 8.566 | 20.23 |
| geometry_texture | 42 | excluded/standard | 10.649 | 22.23 |
| geometry_texture | 42 | excluded/grayscale | 10.679 | 22.40 |
| geometry_texture | 42 | excluded/occlusion | 15.843 | 26.18 |
| geometry_texture | 42 | excluded/extra_dropout | 13.069 | 23.86 |

Pose registration results: ADD measures corresponding-point error; ADD-S uses nearest points and can be permissive for asymmetric objects. Report both.

| Arm | Condition | Cases | ADD mm | ADD-S mm | ADD recall <10% diameter | ADD-S recall <10% diameter |
|---|---|---:|---:|---:|---:|---:|
| combined | dropout40 | 60 | 5.91 | 3.32 | 100.0% | 100.0% |
| combined | grayscale | 60 | 5.13 | 2.96 | 100.0% | 100.0% |
| combined | noise5mm | 60 | 4.98 | 2.92 | 100.0% | 100.0% |
| combined | observed | 60 | 5.06 | 2.94 | 100.0% | 100.0% |
| geometry | dropout40 | 60 | 5.35 | 3.13 | 95.0% | 100.0% |
| geometry | grayscale | 60 | 4.71 | 2.79 | 100.0% | 100.0% |
| geometry | noise5mm | 60 | 4.60 | 2.79 | 100.0% | 100.0% |
| geometry | observed | 60 | 4.66 | 2.79 | 100.0% | 100.0% |
| geometry_symmetry | dropout40 | 60 | 5.87 | 3.30 | 100.0% | 100.0% |
| geometry_symmetry | grayscale | 60 | 5.07 | 2.94 | 100.0% | 100.0% |
| geometry_symmetry | noise5mm | 60 | 5.00 | 2.94 | 100.0% | 100.0% |
| geometry_symmetry | observed | 60 | 5.03 | 2.93 | 100.0% | 100.0% |
| geometry_texture | dropout40 | 60 | 5.24 | 3.06 | 96.7% | 100.0% |
| geometry_texture | grayscale | 60 | 4.72 | 2.80 | 100.0% | 100.0% |
| geometry_texture | noise5mm | 60 | 4.61 | 2.79 | 100.0% | 100.0% |
| geometry_texture | observed | 60 | 4.65 | 2.79 | 100.0% | 100.0% |
| heads_only | dropout40 | 60 | 4.38 | 2.68 | 96.7% | 100.0% |
| heads_only | grayscale | 60 | 3.56 | 2.31 | 100.0% | 100.0% |
| heads_only | noise5mm | 60 | 3.61 | 2.33 | 100.0% | 100.0% |
| heads_only | observed | 60 | 3.44 | 2.29 | 100.0% | 100.0% |
| joint_combined | dropout40 | 60 | 4.06 | 2.50 | 100.0% | 100.0% |
| joint_combined | grayscale | 60 | 3.46 | 2.25 | 100.0% | 100.0% |
| joint_combined | noise5mm | 60 | 3.44 | 2.23 | 100.0% | 100.0% |
| joint_combined | observed | 60 | 3.46 | 2.25 | 100.0% | 100.0% |
| joint_original | dropout40 | 60 | 4.22 | 2.64 | 96.7% | 100.0% |
| joint_original | grayscale | 60 | 3.59 | 2.34 | 100.0% | 100.0% |
| joint_original | noise5mm | 60 | 3.69 | 2.40 | 100.0% | 100.0% |
| joint_original | observed | 60 | 3.62 | 2.32 | 100.0% | 100.0% |
| original | dropout40 | 20 | 4.87 | 2.96 | 100.0% | 100.0% |
| original | grayscale | 20 | 4.47 | 2.75 | 100.0% | 100.0% |
| original | noise5mm | 20 | 4.46 | 2.78 | 100.0% | 100.0% |
| original | observed | 20 | 4.56 | 2.77 | 100.0% | 100.0% |

The original model has 20 cases per condition; each adapted arm has the same 20 cases repeated across three training seeds (60). These different counts are expected. All adapted arms/seeds were checked against the same baseline frame/condition set.

Pose-head development training (not final evaluation):
heads_only_123_complete: best development error / diameter = 0.020160807776846923
heads_only_2026_complete: best development error / diameter = 0.02013154969608877
heads_only_42_complete: best development error / diameter = 0.020012338759988778
joint_combined_123_complete: best development error / diameter = 0.0212333290692186
joint_combined_2026_complete: best development error / diameter = 0.021083417511545122
joint_combined_42_complete: best development error / diameter = 0.021168495289020938
joint_original_123_complete: best development error / diameter = 0.02002674313262105
joint_original_2026_complete: best development error / diameter = 0.020067718192876782
joint_original_42_complete: best development error / diameter = 0.019954212439188268

Input examples are saved in pose/input_*.png. Training histories and per-frame metrics are in results/. A completed run requires pose/complete.json and all 15 encoder-training completion files and all 9 pose-head-training completion files. Texture reconstruction is an auxiliary diagnostic, not evidence of semantic texture understanding. No improvement is assumed.
