# Rutgers ARC Lab — FoundationPose LINEMOD research record

## Status

Inference was active at inspection on October 20, 2026 (America/New_York), host PID 3257117. No duplicate inference was launched and the process was not interrupted. Numerical accuracy is pending final predictions. Consult status.json and analysis.log for the current analysis state; RESULTS.md is written only after analysis succeeds. A stopped process alone does not establish successful inference: coverage and pose validity must also be checked.

## Existing experiment

Repository: /home/Sahil/FoundationPose; Git commit a1b694b83e633c2cb6115b9063d940a687759392. Initial tracked working tree was clean.

Existing command:

```bash
python -u run_linemod.py --linemod_dir /home/Sahil/datasets/bop/lm --use_reconstructed_mesh 0 --debug 0 --debug_dir /home/Sahil/FoundationPose/debug/bop_full
```

Log: debug/bop_full/run.log. Final output: debug/bop_full/linemod_res.yml. The runner keeps results in memory and writes YAML once after the complete loop; there is no per-object result checkpoint in this script. Do not infer completed dataset accuracy from log messages or mix earlier demo/sample outputs into this experiment.

The 13 objects are 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15; bowl (3) and mug (7) are excluded by LinemodReader. split=None selects every RGB image in each scene, not the historical LINEMOD train/test list or a BOP target-filtered subset. dataset_inventory.csv records expected counts and RGB availability.

Dataset root: /home/Sahil/datasets/bop/lm. Images and annotations: lm_test_all/test/<scene>/{rgb,depth,mask_visib,scene_gt.json,scene_camera.json}. CAD models: models/obj_<id>.ply, reached by the reader through lm_models/models -> ../models. models_eval is a separate mesh directory and is not used for these diagnostics.

This is RGB-D pose registration with GT CAD models and GT masks (detect_type='mask'), with each frame registered independently. It is not an end-to-end object detection evaluation. The runner assigns reader GT poses to est.gt_pose; inspection found compute_add_err_to_gt_pose returns constant -1 tensors, so logged `add_errs min:-1.0` is a placeholder, not a measured negative error or an accuracy result. No claim about absence of all possible data leakage is implied by this observation. The reader uses the first scene camera intrinsics for inference; analysis uses 3D pose distances and does not rely on camera intrinsics.

## Diagnostic methodology

analyze.py is a separate CPU-only script; it does not import or invoke the GPU estimator. It waits for the original process identity to disappear or terminate, then reads the final YAML. No packages are installed. The existing foundationpose Python environment supplies numpy, scipy, trimesh, PyYAML, and matplotlib.

Expected keys are scene ID, image ID, and object ID matched to scene_gt.json. Missing predictions and invalid poses count as failures in all recall denominators. Unexpected keys are listed in manifest.json. Matrices must be finite 4x4 homogeneous transforms with orthonormal rotation, determinant near +1, and positive camera Z. The YAML translation is in metres (the reader scales CAD vertices and GT translations by 1e-3); diagnostics multiply predicted translation by 1000 to compare to the original BOP millimetre coordinates. Pose convention is model-to-camera, with no inverse transform or manual centering correction.

Full CAD vertex sets are loaded with process=False. ADD is mean corresponding transformed-vertex Euclidean distance. ADI/ADD-S is mean nearest-neighbour distance from GT transformed vertices to predicted transformed vertices, using a CPU KD-tree. ADD(-S) uses ADI for eggbox (10) and glue (11), and ADD for other objects, reflecting a conventional LINEMOD diagnostic rather than BOP's general symmetry treatment. Correctness uses strict error < 0.1 times models_info.json diameter. Full-vertex weights depend on CAD tessellation; these are not uniform surface samples. Translation error is Euclidean distance in millimetres; rotation error is the geodesic angle in degrees without symmetry correction and can penalize equivalent symmetric poses.

summary.csv contains per-object and pooled (obj_id=0) recalls and valid-prediction medians. RESULTS.md adds the unweighted object macro ADD(-S) recall. Per-pose metrics and status are in per_pose.csv. Missing/invalid rows contain infinity for undefined errors; valid-only medians exclude these rows, while recalls include them as failures. recall.png shows per-object ADD(-S) recall, and curves.png shows pooled ADD and ADI recalls against diameter-normalized thresholds. No confidence scores or inference runtimes are invented.

## Official benchmark distinction

No official BOP benchmark score is calculated or claimed here. Official BOP19 localization AR combines threshold-averaged VSD, MSSD, and MSPD recalls, with visibility, symmetry, targets, model sampling and matching defined by the official toolkit/protocol. These diagnostics do not implement that pipeline. Ground-truth masks and evaluation over all images further limit comparisons to published methods with different inputs and splits. A future official evaluation requires an explicitly selected benchmark protocol, verified target coverage, BOP-format results and the official toolkit; it must disclose GT-mask input.

References: https://bop.felk.cvut.cz/tasks/ and https://github.com/thodan/bop_toolkit/blob/master/scripts/eval_bop19_pose.py .

## Reproduction and provenance

After the existing inference has ended, run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/Sahil/miniforge3/envs/foundationpose/bin/python research/linemod_20261020/analyze.py
```

To inspect dataset inventory without calculating metrics, append --inventory-only. The waiting workflow appends --wait-pid 3257117; it does not signal that PID. Do not launch a second analysis while the watcher is running. The script writes only inside this new research directory (and a matplotlib cache under /tmp). It preserves original code, weights, datasets, logs and predictions. Re-running replaces only its own generated analysis outputs; archive this directory first if retaining multiple analyses.

manifest.json records package versions and SHA-256 hashes of predictions, analysis and relevant core scripts, GT, cameras, models, models_info and weight files. The inference log has no reliable per-frame timestamps; no measured throughput or completion time is claimed. research/linemod_20261020/process_snapshot.txt captures host process inspection.

## User-confirmed dataset source

The user supplied https://bop.felk.cvut.cz/datasets/ on October 20, 2026. This is the BOP dataset catalog; following the original LINEMOD research scope, the selected entry remains LM (LINEMOD), listed under BOP-Classic-Extra. LM-O is a separate dataset. The catalog distributes the LM base, models, and all-test archives; matching archive names and BOP-format extracted files are present under /home/Sahil/datasets/bop/lm. Archive identity has not been verified against remote checksums. No re-download, dataset replacement, or new inference was initiated in response to this clarification.

## Extended analysis

extended.py waits for the primary analysis to complete, then writes extended_summary.csv with mean, median and 90th-percentile translation/rotation/ADD/ADI errors for valid predictions. It writes bop_diagnostic_per_pose.csv with MSSD (millimetres) and MSPD (pixels), using models_eval vertices, per-image K and identity plus the discrete symmetries in models_eval/models_info.json. Included objects have no annotated continuous symmetries; the script refuses unsupported continuous symmetries. MSSD minimizes maximum 3D corresponding-point distance over GT symmetry transforms; MSPD minimizes maximum projected pixel distance. Recalls at MSSD < 0.1 diameter and MSPD < 5 px are diagnostics over all expected poses, not official threshold-averaged target-filtered BOP scores. The official toolkit is absent from the installed foundationpose environment; it has not been installed. VSD and official AR remain uncomputed.

error_distributions.png shows valid-only translation and rotation distributions with outliers hidden. examples/ contains up to one best successful and one worst failed valid prediction per object, selected using normalized ADD(-S), with GT boxes green and predicted boxes red over the RGB image. qualitative_examples.csv records selection and errors. If no valid success or failure exists for an object, that category has no example; missing/invalid predictions are counted in the CSVs rather than drawn as boxes. These selected extremes are illustrative, not representative frequency estimates. extended_manifest.json hashes evaluation meshes, their metadata and the extension script.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/Sahil/miniforge3/envs/foundationpose/bin/python research/linemod_20261020/extended.py
```

Add --wait to wait for primary completion. extended_status.json and extended.log record its state. Formula reference: https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py .

## Weaknesses and proposed experiments

These are hypotheses and design limitations, not measured failure conclusions while inference is pending. The three-frame object-2 result supplied by the researcher is encouraging but is not representative of full-dataset accuracy; it is not independently recomputed in this report.

1. **Mask dependence:** the baseline uses GT visible masks. Compare controlled erosion/dilation and synthetic mask shifts on a fixed, object-balanced validation subset; measure recall change against mask perturbation magnitude. Then evaluate detector-produced masks with the same pose estimator to isolate segmentation failures. Keep final evaluation frames out of parameter selection.
2. **Symmetric ambiguity:** compare raw angular error with MSSD/MSPD and ADD-S on eggbox/glue. Large raw angular error alone need not indicate a physically distinct pose. Inspect the projected overlays before attributing errors to refinement.
3. **Initialization versus refinement:** on a fixed held-out development subset, vary candidate rotation density and refinement iteration count separately; report recall and measured per-image GPU latency after warm-up. Record seeds and configurations. Do not launch these GPU experiments without permission.
4. **Depth quality and occlusion:** stratify failures by valid depth fraction inside the visible mask, range, object size in pixels and visibility fraction where GT metadata permits. Test depth filtering and modest bounded noise on a development subset; keep one factor varied at a time. These stratified statistics are proposed follow-up analyses, not existing output files.
5. **Generalization:** consider a separately approved LM-O experiment to test clutter/occlusion robustness; LM and LM-O are distinct datasets, and this run establishes no LM-O result.
6. **Experiment resilience:** this core runner saves results only at the end and has verbose untimestamped logs. For a future approved code change, add atomic per-scene checkpoints, timestamps, configuration capture and resume logic. No such core-script change has been made.

Only CPU post-processing has been launched. Package installation, core-code changes and additional expensive GPU inference require permission under the researcher's standing instructions.


## Methodology audit and automatic completion workflow

The audit corrected the directional ADI calculation before any real predictions were scored. It now matches the official toolkit direction: construct a KD-tree on predicted transformed CAD vertices and query GT transformed vertices. The reverse direction is generally different. Regression checks include an asymmetric point-cloud example that detects this reversal, metre-to-millimetre conversion, malformed/missing poses, rotation validity, non-commuting GT rotation plus symmetry offsets, known MSSD/MSPD values, CSV denominators and actual success/failure image output.

The estimator returns `poses[0] @ get_tf_to_centered_mesh()`, which maps the original CAD model to the camera. Applying a further model-center correction in analysis would be an error; none is applied. MSSD/MSPD symmetry composition is GT-pose times model-symmetry transform, including the symmetry translation in millimetres. ADD-S does not substitute for BOP's globally annotated symmetry handling. Raw angular error remains explicitly uncorrected for symmetries.

The expanded dataset inventory verifies all 15,800 expected images have depth, visible masks, camera entries and an unambiguous same-object GT instance. Each included scene has one unique intrinsics matrix and one depth scale. Thus the reader's first-intrinsics/last-depth-scale assumption is consistent within these scenes. The save-loop and YAML conversion have been inspected: scene/image/object keys and 4x4 lists match the analysis loader. The runner writes once at `debug/bop_full/linemod_res.yml` only after all objects; saved coverage cannot be guaranteed before that file is produced. Disk space was approximately 1.2 TB available at audit. There is no checkpoint/resume protection in the unmodified core runner.

The two earlier CPU waiters were replaced with a single `supervise.py` process so corrected source is loaded at completion. It uses an exclusive lock, monitors the original process identity, records 45-second progress samples, and never imports or launches FoundationPose inference. After the original PID exits, it requires a stable final YAML and no traceback/runtime error in the inference log, runs the primary and extended analyses sequentially, and updates this report with actual measurements. It cannot recover the original shell's process exit status; successful completion is assessed from the terminated original process, stable output, log checks and exact prediction coverage/validity. Missing or invalid poses are reported explicitly rather than silently excluded from recall.

Follow `PROGRESS.md`, `progress.json`, `progress_samples.csv`, `status.json` and `supervisor.log`. Runtime estimates use logged completed poses, host process elapsed time and recent measured throughput; they exclude post-processing and are projections rather than guarantees. `prediction_validation.json` will list exact missing/unexpected keys and validity. Actual measurements will appear in `MEASURED_RESULTS.md` and in the automated section of this report after both analysis stages finish. No placeholder full-run accuracy is inserted while inference is running.

Reproduce regression checks:

```bash
OPENBLAS_NUM_THREADS=1 /home/Sahil/miniforge3/envs/foundationpose/bin/python research/linemod_20261020/test_analysis.py
```

Do not launch a second supervisor or analysis alongside the configured process. The supervisor command, if recovery is needed after confirming no supervisor remains, is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/Sahil/miniforge3/envs/foundationpose/bin/python -u research/linemod_20261020/supervise.py --pid 3257117
```


## Measured runtime snapshot

At 2026-10-20T23:35:37-04:00, the original inference was active at object 2, frame 888. The log contained 2,124 completed poses of 15,800 expected (13.44%). Overall measured throughput was 1.218 poses/s; recent measured throughput was 0.711 poses/s. Projection from the overall mean: 3.12 hours remaining; projection from the recent interval: 5.34 hours. Early recent-rate measurements span only 45 seconds; use the continuously refreshed PROGRESS.md for a longer observation window. Future objects and setup overhead can change this substantially. These estimates describe inference only.

A CPU timing probe used one synthetic 2 mm translated GT pose per object with the full CAD/evaluation vertex sets. Extrapolated metric-kernel time was 2.22 minutes for 15,800 poses, excluding YAML parsing, data IO, plots and hashing. This probe measures CPU post-processing cost; it is not measured inference performance or research accuracy. See cpu_timing_probe.csv and preflight.json.

## Completion and failure-analysis safeguards

The supervisor remains CPU-only and does not restart or signal inference. Its validation establishes whether the final prediction artifact is complete and valid; it does not claim to have recovered the original shell's exit status. The original PID must terminate, the final YAML must exist and remain stable, no traceback/runtime error may be logged, and exact scene/image/object coverage and matrix validity must be checked. A terminated process with no final predictions is an analysis failure, not a successful baseline. Missing/invalid poses remain explicit failures; the final report flags an incomplete baseline.

An additional CPU finalizer waits for both analysis stages and the measured-results report. It joins BOP scene_gt_info.json by the same GT-instance index, writes failure_metadata.csv, failure_strata.csv and failure_strata.png, identifies the lowest-recall objects, and documents a controlled mask-quality follow-up. Bins show descriptive associations rather than inferred causes. No claim that noise or segmentation caused an error is made without an intervention. finalization_status.json and finalization.log track this stage. It adds an actual-results section only after predictions are scored. A deterministic proposed_mask_study_split.csv will identify disjoint development/held-out frames without running any GPU experiment.

Reproduce the final CPU stage only after upstream analysis has completed:

```bash
OPENBLAS_NUM_THREADS=1 /home/Sahil/miniforge3/envs/foundationpose/bin/python research/linemod_20261020/finalize.py
```

Add --wait to wait for upstream completion. The regression fixture covers the entire pipeline, including a deliberately incomplete prediction set; it verifies that the generated report identifies incompleteness rather than calling it a complete baseline. Synthetic test artifacts live under /tmp and are not research results.


## Interruption detected — 2026-10-21T16:21:31-04:00

Host checks found inference PID 3257117 and supervisor PID 3269392 absent. The log stopped at object 6, frame 339, at approximately 00:32 Eastern on October 21; the last monitor recorded 5,184 completed poses (32.8%). No final linemod_res.yml exists, so full-run metrics and final plots could not be produced. The cause is undetermined; no traceback or out-of-memory message was found in the log. The log ends mid-inference and includes trailing NUL bytes. Earlier waiting statuses and runtime estimates are stale. No GPU restart has been launched. A new GPU run requires approval.


## Recovered partial results

See recovery/PARTIAL_RESULTS.md for the standalone table.

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


## Restart preparation

See recovery/RESTART_READY.md for the tested checkpointed wrapper and explicit GPU approval options. No core algorithms were edited and no new GPU inference was launched during recovery. All 5,186 recovered matrices passed the existing pose-validity checks. Their observed-subset errors are reported above with rounded-log precision disclosed. The remaining 10,614 frames require GPU inference to complete a recovered baseline; recomputing all 15,800 gives a native-precision retry. Both variants write new output directories and chain CPU analysis only after successful child exit and validated completion artifacts.

## Approved checkpointed continuation — October 21, 2026

The researcher explicitly approved computing the remaining 10,614 poses and disclosing the rounded-log recovered prefix. The job was launched as user service foundationpose-bop-recovery-20261021.service, independent of this terminal. It uses debug/bop_recovery_20261021 for all new checkpoints, logs, predictions and analysis. It restored 5,186 recovered matrices and was verified to have saved native-precision predictions beyond the interrupted prefix (object 6, frame 350). No core scripts, original weights, datasets or interrupted-run artifacts were changed. Automatic restart is disabled; a server reboot still stops execution, but atomic per-frame checkpoints enable a later resume.

The job owns its inference child and requires normal exit, a completion marker and final-file hash verification before CPU analysis. The separate CPU monitor refreshes PROGRESS.md and status.json and appends the new measured tables and failure analysis to this report when the continuation and analyses finish. Do not use the historical stopped-run estimates as current progress. The final new-job report is debug/bop_recovery_20261021/analysis/REPORT.md. Mixed precision and recovered-prefix provenance must accompany any published diagnostic result.

Startup validation confirmed unchanged core-script hashes and an unchanged original interrupted-run log. Recovered frame 338 is stored from rounded text, while newly inferred frame 339 is native float32 and passed finite, homogeneous-transform, orthonormal-rotation and positive-depth checks. See debug/bop_recovery_20261021/startup_validation.json. The original dead workflow status files were archived and marked superseded; PROGRESS.md/status.json now refer to the approved continuation.

<!-- CHECKPOINT_RECOVERY_FINAL_RESULTS -->
## Approved recovery: completed measurements

The first 5,186 poses were recovered from rounded log text; the remaining 10,614 were computed by the approved continuation. This is a mixed-precision recovered baseline, not a pristine full-precision rerun or official BOP score.

## Measured results — 2026-10-21T19:43:01-04:00

Analysis completed on the final saved predictions. Expected: 15800; valid: 15800; missing: 0; invalid: 0. Exact key coverage: True. All poses valid: True.

Pooled valid-prediction translation error: mean 4.467 mm, median 4.131 mm, p90 7.293 mm. Raw rotation error: mean 2.094°, median 1.809°, p90 3.171°. Raw rotation is not symmetry corrected.

| Object | Valid / expected | Mean translation (mm) | Mean raw rotation (°) | ADD(-S) recall < 0.1d | MSSD recall < 0.1d | MSPD recall < 5px |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1236 / 1236 | 4.051 | 2.498 | 98.95% | 96.20% | 97.25% |
| 2 | 1214 / 1214 | 3.319 | 1.676 | 100.00% | 100.00% | 91.35% |
| 4 | 1201 / 1201 | 4.270 | 2.142 | 100.00% | 99.67% | 92.51% |
| 5 | 1196 / 1196 | 4.668 | 1.215 | 100.00% | 99.75% | 94.40% |
| 6 | 1179 / 1179 | 2.825 | 1.841 | 100.00% | 99.49% | 97.96% |
| 8 | 1188 / 1188 | 4.515 | 1.304 | 100.00% | 100.00% | 93.27% |
| 9 | 1254 / 1254 | 3.579 | 2.254 | 99.04% | 97.21% | 96.01% |
| 10 | 1253 / 1253 | 3.967 | 1.735 | 100.00% | 99.84% | 98.40% |
| 11 | 1220 / 1220 | 5.520 | 2.903 | 100.00% | 99.67% | 81.48% |
| 12 | 1237 / 1237 | 4.843 | 2.602 | 99.03% | 98.79% | 90.22% |
| 13 | 1152 / 1152 | 6.111 | 2.362 | 100.00% | 100.00% | 30.21% |
| 14 | 1227 / 1227 | 4.771 | 2.646 | 98.70% | 98.53% | 57.62% |
| 15 | 1243 / 1243 | 5.679 | 2.002 | 100.00% | 100.00% | 78.04% |
| Pooled | 15800 / 15800 | 4.467 | 2.094 | 99.66% | 99.15% | 84.72% |

Object macro ADD(-S) recall: 99.67%. Lowest observed per-object recalls: object 14 (98.70%), object 1 (98.95%), object 12 (99.03%). These rank observed performance; the cause requires inspecting failure images and controlled experiments.

These are diagnostic scores on the full 13-object split with GT visible masks. No official BOP AR, VSD score, or benchmark submission is claimed. See [recall](recall.png), [error curves](curves.png), [error distributions](error_distributions.png), and [selected examples](qualitative_examples.csv).


## Observed failure patterns and follow-up experiment

Final baseline coverage and validity passed: **True**. All expected poses remain in recall denominators. Error means describe valid poses only.

| Lowest ADD(-S) recall objects | Recall | Mean translation (mm) | Mean raw rotation (°) |
| --- | --- | --- | --- |
| 14 | 98.70% | 4.771 | 2.646 |
| 1 | 98.95% | 4.051 | 2.498 |
| 12 | 99.03% | 4.843 | 2.602 |

0 missing and 0 invalid poses. 53 total ADD(-S) failures. 5 valid poses had raw rotation >90° while passing MSSD <0.1 diameter; these need symmetry-aware interpretation rather than an automatic rotation-failure label.

Pooled descriptive strata:

| Dimension | Bin | Poses | ADD(-S) recall |
| --- | --- | --- | --- |
| visibility | 0.5–0.8 | 95 | 97.89% |
| visibility | ≥0.8 | 15705 | 99.68% |
| camera_distance_mm | <800 | 4870 | 99.90% |
| camera_distance_mm | 800–1000 | 6685 | 99.69% |
| camera_distance_mm | ≥1000 | 4245 | 99.36% |

These are associations, not demonstrated causes. Pooled bins can differ in object composition; failure_strata.csv also supplies within-object bins. See [stratified plot](failure_strata.png) and [selected success/failure examples](qualitative_examples.csv). The examples are selected extremes, not a random sample. No depth-noise or mask-quality failure mechanism is established by pose errors alone.

The recommended next controlled experiment is **segmentation-mask robustness**. This baseline uses ideal GT visible masks, so performance under imperfect segmentation remains unmeasured. Hypothesis: modest boundary errors or localization shifts reduce pose recall, especially for small visible object regions; a carefully bounded mask treatment may improve robustness. The baseline's hardest objects and visibility strata identify where to inspect this hypothesis but do not prove it.

Use the deterministic proposed_mask_study_split.csv: 20 development and 20 held-out frames per object, disjoint within each object. The full baseline has already been examined, so these are development-study splits, not a fresh benchmark test set. Preserve RGB, depth, CAD model, camera calibration, inference settings and seed schedule. Compare the unmodified GT-mask control against erosion/dilation radii 2 and 5 pixels and mask shifts of 2 and 5 pixels in each cardinal direction, clipping at image boundaries. These magnitudes are specified before running; do not select the best severity after inspecting held-out results. Never overwrite dataset masks: generate variants in a separate experiment directory. Use seeds 0, 1 and 2 paired across conditions. Empty perturbed masks count as failures and are logged.

Measure paired per-frame changes in ADD(-S), MSSD and MSPD recall, translation/rotation distributions, mask IoU versus GT, and warmed inference latency. Report per-object and object-macro differences with uncertainty from a paired bootstrap stratified by object. Compare within the same frames and seeds to limit pose difficulty and stochastic confounding. Tune any mitigation on development frames only, then report all preregistered conditions on held-out frames. Include segmentation-only measurements as well as pose outcomes. Introduce depth-noise and refinement ablations separately after this controlled test, so changes are attributable to one factor.

No follow-up GPU inference has been started. Running this study or changing core algorithms requires the researcher's approval. This report contains diagnostic scores, not official BOP AR/VSD or a benchmark submission.
