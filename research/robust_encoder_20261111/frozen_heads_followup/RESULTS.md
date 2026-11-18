# Frozen adapted encoder + trained fusion/pose heads

Exploratory fixed evaluation: five adaptation objects, four frames per object, four conditions. GT masks; no untouched-object or official benchmark claim. Training seeds are repeats of the same frames, not independent images.

| Arm | Condition | Seed | Cases | ADD mm | ADD-S mm | Paired ADD change vs original mm |
|---|---|---:|---:|---:|---:|---:|
| original | observed | 0 | 20 | 4.564 | 2.774 | +0.000 |
| original | noise5mm | 0 | 20 | 4.456 | 2.779 | +0.000 |
| original | dropout40 | 0 | 20 | 4.868 | 2.964 | +0.000 |
| original | grayscale | 0 | 20 | 4.468 | 2.753 | +0.000 |
| combined | observed | 42 | 20 | 4.952 | 2.878 | +0.388 |
| combined | noise5mm | 42 | 20 | 4.910 | 2.895 | +0.454 |
| combined | dropout40 | 42 | 20 | 5.981 | 3.331 | +1.113 |
| combined | grayscale | 42 | 20 | 5.088 | 2.895 | +0.620 |
| combined | observed | 123 | 20 | 5.242 | 3.015 | +0.678 |
| combined | noise5mm | 123 | 20 | 5.006 | 2.928 | +0.550 |
| combined | dropout40 | 123 | 20 | 5.592 | 3.184 | +0.724 |
| combined | grayscale | 123 | 20 | 5.336 | 3.055 | +0.868 |
| combined | observed | 2026 | 20 | 4.974 | 2.926 | +0.410 |
| combined | noise5mm | 2026 | 20 | 5.017 | 2.949 | +0.561 |
| combined | dropout40 | 2026 | 20 | 6.164 | 3.436 | +1.296 |
| combined | grayscale | 2026 | 20 | 4.967 | 2.925 | +0.499 |
| heads_only | observed | 42 | 20 | 3.386 | 2.294 | -1.178 |
| heads_only | noise5mm | 42 | 20 | 3.832 | 2.396 | -0.624 |
| heads_only | dropout40 | 42 | 20 | 4.534 | 2.723 | -0.334 |
| heads_only | grayscale | 42 | 20 | 3.730 | 2.389 | -0.738 |
| heads_only | observed | 123 | 20 | 3.218 | 2.170 | -1.346 |
| heads_only | noise5mm | 123 | 20 | 3.118 | 2.144 | -1.338 |
| heads_only | dropout40 | 123 | 20 | 4.095 | 2.570 | -0.773 |
| heads_only | grayscale | 123 | 20 | 3.476 | 2.253 | -0.992 |
| heads_only | observed | 2026 | 20 | 3.722 | 2.395 | -0.842 |
| heads_only | noise5mm | 2026 | 20 | 3.879 | 2.437 | -0.577 |
| heads_only | dropout40 | 2026 | 20 | 4.516 | 2.754 | -0.352 |
| heads_only | grayscale | 2026 | 20 | 3.462 | 2.302 | -1.007 |
| frozen_combined_heads | observed | 42 | 20 | 3.577 | 2.315 | -0.987 |
| frozen_combined_heads | noise5mm | 42 | 20 | 3.703 | 2.343 | -0.753 |
| frozen_combined_heads | dropout40 | 42 | 20 | 3.986 | 2.511 | -0.882 |
| frozen_combined_heads | grayscale | 42 | 20 | 3.818 | 2.416 | -0.651 |
| frozen_combined_heads | observed | 123 | 20 | 3.597 | 2.366 | -0.967 |
| frozen_combined_heads | noise5mm | 123 | 20 | 3.413 | 2.257 | -1.043 |
| frozen_combined_heads | dropout40 | 123 | 20 | 8.940 | 2.528 | +4.072 |
| frozen_combined_heads | grayscale | 123 | 20 | 3.494 | 2.298 | -0.974 |
| frozen_combined_heads | observed | 2026 | 20 | 3.650 | 2.322 | -0.914 |
| frozen_combined_heads | noise5mm | 2026 | 20 | 3.519 | 2.282 | -0.937 |
| frozen_combined_heads | dropout40 | 2026 | 20 | 4.205 | 2.586 | -0.663 |
| frozen_combined_heads | grayscale | 2026 | 20 | 3.550 | 2.321 | -0.918 |

Negative paired ADD change means improvement. Per-object results below expose inconsistent gains.

| Object | Condition | Seed | Combined encoder ADD mm | Frozen combined + trained downstream ADD mm | Change mm |
|---:|---|---:|---:|---:|---:|
| 5 | observed | 42 | 6.094 | 3.602 | -2.491 |
| 5 | noise5mm | 42 | 5.938 | 3.485 | -2.453 |
| 5 | dropout40 | 42 | 6.230 | 3.983 | -2.248 |
| 5 | grayscale | 42 | 6.234 | 3.742 | -2.492 |
| 8 | observed | 42 | 5.786 | 3.994 | -1.792 |
| 8 | noise5mm | 42 | 5.754 | 4.614 | -1.140 |
| 8 | dropout40 | 42 | 8.221 | 4.765 | -3.457 |
| 8 | grayscale | 42 | 5.915 | 3.901 | -2.013 |
| 9 | observed | 42 | 4.335 | 3.739 | -0.597 |
| 9 | noise5mm | 42 | 4.402 | 3.530 | -0.873 |
| 9 | dropout40 | 42 | 5.635 | 3.793 | -1.842 |
| 9 | grayscale | 42 | 4.843 | 4.263 | -0.579 |
| 10 | observed | 42 | 3.159 | 3.089 | -0.070 |
| 10 | noise5mm | 42 | 3.040 | 3.429 | +0.389 |
| 10 | dropout40 | 42 | 4.273 | 4.023 | -0.249 |
| 10 | grayscale | 42 | 3.193 | 3.256 | +0.064 |
| 11 | observed | 42 | 5.388 | 3.464 | -1.924 |
| 11 | noise5mm | 42 | 5.417 | 3.456 | -1.961 |
| 11 | dropout40 | 42 | 5.547 | 3.366 | -2.181 |
| 11 | grayscale | 42 | 5.257 | 3.926 | -1.331 |
| 5 | observed | 123 | 6.290 | 3.384 | -2.906 |
| 5 | noise5mm | 123 | 6.191 | 3.189 | -3.002 |
| 5 | dropout40 | 123 | 6.578 | 4.129 | -2.449 |
| 5 | grayscale | 123 | 6.380 | 3.580 | -2.800 |
| 8 | observed | 123 | 6.554 | 4.053 | -2.501 |
| 8 | noise5mm | 123 | 5.858 | 4.058 | -1.800 |
| 8 | dropout40 | 123 | 6.553 | 5.285 | -1.268 |
| 8 | grayscale | 123 | 6.398 | 3.779 | -2.618 |
| 9 | observed | 123 | 4.350 | 3.589 | -0.760 |
| 9 | noise5mm | 123 | 4.070 | 3.930 | -0.140 |
| 9 | dropout40 | 123 | 5.361 | 3.769 | -1.592 |
| 9 | grayscale | 123 | 4.906 | 3.799 | -1.106 |
| 10 | observed | 123 | 3.147 | 3.589 | +0.442 |
| 10 | noise5mm | 123 | 3.266 | 3.162 | -0.103 |
| 10 | dropout40 | 123 | 4.126 | 28.216 | +24.090 |
| 10 | grayscale | 123 | 3.239 | 3.223 | -0.016 |
| 11 | observed | 123 | 5.870 | 3.371 | -2.498 |
| 11 | noise5mm | 123 | 5.646 | 2.729 | -2.917 |
| 11 | dropout40 | 123 | 5.340 | 3.300 | -2.040 |
| 11 | grayscale | 123 | 5.758 | 3.088 | -2.670 |
| 5 | observed | 2026 | 6.050 | 3.742 | -2.308 |
| 5 | noise5mm | 2026 | 5.998 | 3.512 | -2.486 |
| 5 | dropout40 | 2026 | 6.600 | 3.768 | -2.832 |
| 5 | grayscale | 2026 | 6.110 | 3.412 | -2.698 |
| 8 | observed | 2026 | 5.516 | 3.916 | -1.599 |
| 8 | noise5mm | 2026 | 5.751 | 3.818 | -1.933 |
| 8 | dropout40 | 2026 | 8.036 | 5.184 | -2.852 |
| 8 | grayscale | 2026 | 5.167 | 3.920 | -1.247 |
| 9 | observed | 2026 | 4.468 | 3.542 | -0.926 |
| 9 | noise5mm | 2026 | 4.360 | 3.832 | -0.528 |
| 9 | dropout40 | 2026 | 5.740 | 3.754 | -1.986 |
| 9 | grayscale | 2026 | 4.608 | 3.783 | -0.825 |
| 10 | observed | 2026 | 3.110 | 3.887 | +0.777 |
| 10 | noise5mm | 2026 | 3.078 | 3.036 | -0.042 |
| 10 | dropout40 | 2026 | 4.669 | 4.079 | -0.590 |
| 10 | grayscale | 2026 | 3.220 | 3.119 | -0.101 |
| 11 | observed | 2026 | 5.727 | 3.164 | -2.563 |
| 11 | noise5mm | 2026 | 5.899 | 3.395 | -2.504 |
| 11 | dropout40 | 2026 | 5.775 | 4.240 | -1.535 |
| 11 | grayscale | 2026 | 5.730 | 3.515 | -2.214 |
