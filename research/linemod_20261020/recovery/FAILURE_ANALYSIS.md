## Observed failure patterns and follow-up experiment

Final baseline coverage and validity passed: **False**. Treat this as an incomplete/invalid-prediction diagnostic; repair the experiment before claiming a complete baseline. All expected poses remain in recall denominators. Error means describe valid poses only.

| Lowest ADD(-S) recall objects | Recall | Mean translation (mm) | Mean raw rotation (°) |
| --- | --- | --- | --- |
| 8 | 0.00% | N/A | N/A |
| 9 | 0.00% | N/A | N/A |
| 10 | 0.00% | N/A | N/A |

10614 missing and 0 invalid poses. 10627 total ADD(-S) failures. 0 valid poses had raw rotation >90° while passing MSSD <0.1 diameter; these need symmetry-aware interpretation rather than an automatic rotation-failure label.

Pooled descriptive strata:

| Dimension | Bin | Poses | ADD(-S) recall |
| --- | --- | --- | --- |
| visibility | 0.5–0.8 | 95 | 22.11% |
| visibility | ≥0.8 | 15705 | 32.80% |
| camera_distance_mm | <800 | 4870 | 33.08% |
| camera_distance_mm | 800–1000 | 6685 | 33.43% |
| camera_distance_mm | ≥1000 | 4245 | 31.26% |

These are associations, not demonstrated causes. Pooled bins can differ in object composition; failure_strata.csv also supplies within-object bins. See [stratified plot](failure_strata.png) and [selected success/failure examples](qualitative_examples.csv). The examples are selected extremes, not a random sample. No depth-noise or mask-quality failure mechanism is established by pose errors alone.

The recommended next controlled experiment is **segmentation-mask robustness**. This baseline uses ideal GT visible masks, so performance under imperfect segmentation remains unmeasured. Hypothesis: modest boundary errors or localization shifts reduce pose recall, especially for small visible object regions; a carefully bounded mask treatment may improve robustness. The baseline's hardest objects and visibility strata identify where to inspect this hypothesis but do not prove it.

Use the deterministic proposed_mask_study_split.csv: 20 development and 20 held-out frames per object, disjoint within each object. The full baseline has already been examined, so these are development-study splits, not a fresh benchmark test set. Preserve RGB, depth, CAD model, camera calibration, inference settings and seed schedule. Compare the unmodified GT-mask control against erosion/dilation radii 2 and 5 pixels and mask shifts of 2 and 5 pixels in each cardinal direction, clipping at image boundaries. These magnitudes are specified before running; do not select the best severity after inspecting held-out results. Never overwrite dataset masks: generate variants in a separate experiment directory. Use seeds 0, 1 and 2 paired across conditions. Empty perturbed masks count as failures and are logged.

Measure paired per-frame changes in ADD(-S), MSSD and MSPD recall, translation/rotation distributions, mask IoU versus GT, and warmed inference latency. Report per-object and object-macro differences with uncertainty from a paired bootstrap stratified by object. Compare within the same frames and seeds to limit pose difficulty and stochastic confounding. Tune any mitigation on development frames only, then report all preregistered conditions on held-out frames. Include segmentation-only measurements as well as pose outcomes. Introduce depth-noise and refinement ablations separately after this controlled test, so changes are attributable to one factor.

No follow-up GPU inference has been started. Running this study or changing core algorithms requires the researcher's approval. This report contains diagnostic scores, not official BOP AR/VSD or a benchmark submission.
