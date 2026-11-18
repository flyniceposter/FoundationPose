# GPU encoder investigation — Final budget report

Updated 2026-10-28T17:05:41-04:00. Process state: **pilot_complete**.

Started: 2026-10-28T17:01:00.134538-04:00. Scheduled deadline: 2026-10-29T05:01:00.134538-04:00.
Completed units: 7; recorded failed units: 0. Distinct frames: 1; objects represented: 1; real pose candidates evaluated per unit: 37. Elapsed wall time at status update: 0.001 hours.

This is the extension to the earlier CPU-only report. CUDA is accessible outside the sandbox; the earlier claim of a driver problem was incorrect. These new experiments use the installed renderer, source depth filtering, real rendered/observed inputs, frozen pretrained CNNs, and the actual refinement/ranking heads.

## What was measured

The queue changes input disturbances and then intervenes on observed RGB, XYZ, shared-CNN spatial structure, or fusion-CNN spatial structure. Replacing a map by its spatial mean preserves channel means but destroys its layout. The same initial 37 candidates and candidate ordering are retained across matched units. One-step refinement and ranking on the original candidate set are separate endpoints. Ranking includes a GT candidate and does not measure blind global registration.

Every unit has per-candidate known initial errors, refined errors, scores, feature distances, and poses in units/*.json. Compact per-unit summaries are transactionally stored in summary.sqlite. No pretrained weights are changed. Feature interventions are deliberately out of distribution; task degradation establishes sensitivity to those interventions, not a validated encoder replacement.

## Coverage and descriptive means

The tables reflect completed units, not assumed completion of the full oversized queue. They must not be compared as though every group covered identical frames. Summary means first average repeated units within a frame and then macro-average represented objects. Missing groups are not silently treated as completed successes. See analysis/coverage.csv for exact object/condition/intervention coverage. Ranking selection errors average over all exactly tied maximum-logit candidates, rather than letting the first GT candidate win a constant-score tie. Raw argmax indices remain in the raw records. Constant-score Spearman correlations are undefined, not perfect: ranking_spearman_defined reports their availability and ranking_tie_count reports degeneracy.

| condition | ablation | units | objects | refined_rotation_deg | selected_adds_mm |
| --- | --- | --- | --- | --- | --- |
| clean | intact | 1 | 1 | 4.6894 | 0.0001 |
| clean | pooled_shared | 1 | 1 | 9.5295 | 15.3114 |
| clean | shuffle_fusion | 1 | 1 | 5.3716 | 0.0001 |
| clean | shuffle_shared | 1 | 1 | 9.7227 | 8.6481 |
| clean | zero_observed_rgb | 1 | 1 | 4.7205 | 2.6854 |
| clean | zero_observed_xyz | 1 | 1 | 6.2946 | 2.5 |
| clean | zero_shared | 1 | 1 | 13.0364 | 9.85 |

## Exact paired effects

analysis/paired_encoder_interventions.csv compares each intervention with intact computation only on matching frame, condition, seed and candidate set. Different interventions can have different denominators. Positive differences mean larger errors. At final reporting, 2,000 within-object frame bootstrap resamples give conditional intervals; candidate/seed repeats are not counted as independent frames. These intervals do not model correlated sequences or retraining uncertainty, and no multiple-testing significance claim is made.

![Encoder effects](analysis/encoder_effects.png)

## Limitations

GT centers the local candidate neighborhood; initial errors are restricted to the declared grid. The prior 520-frame diagnostic split is reused and additional frames are from the same LM sequences. No new model selection is performed. Per-image camera intrinsics and source-generated mesh diameter are used. CAD geometry and annotations are fixed input information. Sampled ADD/ADI uses 512 deterministic CAD vertices and is not full-mesh ADD, official BOP AR, VSD, MSSD or MSPD. Rotation errors include raw and annotated discrete-symmetry-aware versions. Ranking quality is conditional on this favorable candidate set. Correlations between feature distance and pose error are descriptive; the separate feature-map interventions test the installed computation under artificial changes.

## Reproduction and next interpretation

Read ../PROTOCOL.md for the predeclared design. run_gpu.py resumes matching atomic checkpoints and refuses source mismatches; summary.sqlite supports live reporting. GPU execution requires running outside the restricted tool sandbox on this server. Source hashes are in manifest.json; object geometry hashes are in objects/*.json. Raw data are preserved in units/*.json. Live reports are partial and the final report is valid only with its explicit process state and coverage.

For Friday, present the paired effects with their frame counts, then relate them to spatial-versus-pooled feature distances. Do not present all low-drift features as robust, or these local tests as a global pose benchmark. The earlier CPU slides provide architecture and sanity-check context; this phase supplies real-pair/task evidence.
