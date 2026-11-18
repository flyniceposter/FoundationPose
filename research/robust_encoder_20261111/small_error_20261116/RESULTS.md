# Small-error training results

No convincing improvement under the predeclared criterion; inspect mixed effects and uncertainty.

Primary endpoint: ADD-S / diameter at iteration 5 on observed input. Relative improvement: 1.00%. Seed wins: 2/3; object wins: 3/5. Paired object-bootstrap interval for mixture minus standard: [-0.0007327674704432939, 0.0006785886253669694]. Only five correlated object sequences: this interval is descriptive.

| Condition | Arm | Iteration | ADD mm | ADD-S mm |
|---|---|---:|---:|---:|
| observed | original | 1 | 17.859 | 3.349 |
| observed | original | 2 | 4.189 | 2.623 |
| observed | original | 3 | 4.056 | 2.580 |
| observed | original | 4 | 4.097 | 2.585 |
| observed | original | 5 | 4.078 | 2.590 |
| observed | original | 10 | 4.026 | 2.561 |
| observed | standard | 1 | 19.038 | 3.611 |
| observed | standard | 2 | 3.559 | 2.308 |
| observed | standard | 3 | 3.302 | 2.235 |
| observed | standard | 4 | 3.388 | 2.274 |
| observed | standard | 5 | 3.308 | 2.234 |
| observed | standard | 10 | 3.210 | 2.187 |
| observed | mixture | 1 | 23.932 | 4.260 |
| observed | mixture | 2 | 4.114 | 2.541 |
| observed | mixture | 3 | 3.380 | 2.265 |
| observed | mixture | 4 | 3.415 | 2.275 |
| observed | mixture | 5 | 3.224 | 2.192 |
| observed | mixture | 10 | 3.133 | 2.128 |
| dropout40 | original | 1 | 46.885 | 30.941 |
| dropout40 | original | 2 | 31.800 | 28.436 |
| dropout40 | original | 3 | 31.349 | 28.158 |
| dropout40 | original | 4 | 30.893 | 28.030 |
| dropout40 | original | 5 | 30.909 | 28.026 |
| dropout40 | original | 10 | 30.896 | 28.032 |
| dropout40 | standard | 1 | 48.543 | 30.719 |
| dropout40 | standard | 2 | 33.699 | 29.203 |
| dropout40 | standard | 3 | 30.395 | 27.799 |
| dropout40 | standard | 4 | 30.406 | 27.798 |
| dropout40 | standard | 5 | 30.248 | 27.738 |
| dropout40 | standard | 10 | 30.236 | 27.718 |
| dropout40 | mixture | 1 | 54.147 | 31.538 |
| dropout40 | mixture | 2 | 35.125 | 29.826 |
| dropout40 | mixture | 3 | 30.629 | 27.860 |
| dropout40 | mixture | 4 | 30.365 | 27.726 |
| dropout40 | mixture | 5 | 30.254 | 27.701 |
| dropout40 | mixture | 10 | 30.085 | 27.631 |

## Primary differences by seed

```json
{
  "42": {
    "standard": 0.01305264425778328,
    "mixture": 0.013238089989429916,
    "delta": 0.00018544573164663673
  },
  "123": {
    "standard": 0.01266309998304141,
    "mixture": 0.012634309785373343,
    "delta": -2.8790197668066703e-05
  },
  "2026": {
    "standard": 0.013123830032314519,
    "mixture": 0.012578201139351,
    "delta": -0.0005456288929635185
  }
}
```

## Primary differences by object

```json
{
  "5": {
    "standard": 0.011589547760015903,
    "mixture": 0.010741947842334449,
    "delta": -0.0008475999176814544
  },
  "8": {
    "standard": 0.009972724074207004,
    "mixture": 0.009396024482554203,
    "delta": -0.0005766995916528014
  },
  "9": {
    "standard": 0.018459781249728708,
    "mixture": 0.018547479037910142,
    "delta": 8.769778818143448e-05
  },
  "10": {
    "standard": 0.011678780385428633,
    "mixture": 0.013063062137540256,
    "delta": 0.0013842817521116223
  },
  "11": {
    "standard": 0.013031790319185096,
    "mixture": 0.012335821356584717,
    "delta": -0.0006959689626003795
  }
}
```

## Limits and prior work

Static small/exact-pose mixture; not unrolled iterative training. Original encoder and scoring network fixed. Three matched seeds, 40 locally unused frames on five already-inspected objects, original dataset GT masks. No official benchmark or unseen-pretraining claim. No selection of a test iteration by GT accuracy. Evaluation frames can be temporally close to previously inspected frames. Any improvement is local evidence, not definitive transfer.

DeepIM (ECCV 2018), section 4.1 / Table 1, already identifies insufficient close-to-GT training poses and demonstrates iterative training. This ablation does not establish method novelty.

https://openaccess.thecvf.com/content_ECCV_2018/papers/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.pdf

## Accounting for empty-depth inputs

In object 9 frame 001092, block dropout removed all usable masked depth. Production registration returned its fallback pose before refinement. The same fallback is retained for all seven model variants at all six requested stopping points (42 records, actual refinement count zero). None is omitted; the observed-input primary endpoint is unaffected.

## Iteration contribution

On observed input, iterations 2 to 5 reduced mean ADD-S by 0.074 mm for standard training and 0.349 mm for the mixture. The mixture had worse error at iteration 2, so the larger correction does not establish a large final accuracy advantage: at iteration 5, mean ADD-S was 2.234 versus 2.192 mm. The diameter-normalized primary gain was only 1.00%, with mixed seed/object effects and an interval spanning zero.

Checks: 3,360 unique expected registration records, finite metrics, disjoint local holdout, identical frozen encoder/BN state, matched initial candidates wherever refinement ran. The only code fix after training added recording for production early-return fallback cases; it did not change model training or non-fallback predictions.
