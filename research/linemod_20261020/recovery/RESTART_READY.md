# Recovery and restart handoff

The original run is not active. The host has rebooted; the immediate cause of the earlier inference stop remains undetermined. Source log, weights, datasets and core scripts are preserved.

5,186 complete matrices were recovered from the original log: all frames for objects 1, 2, 4, 5 and 339 frames for object 6. They were checked against actual sorted RGB filenames and matching scene/image/object IDs. recovery_manifest.json records the original log SHA-256, counts and rounded-text precision. CPU diagnostics, tables and real overlay images are complete in this directory. PARTIAL_RESULTS.md distinguishes observed-prefix statistics from full expected-set recall with missing predictions counted as failures.

The isolated checkpoint_runner.py imports and delegates to the existing run_linemod.py. It does not edit core algorithms. It saves each inferred pose atomically at native precision in a new directory. It rejects conflicting manifests, duplicate inference, invalid checkpoint shapes and completed output overwrites. A shared lock prevents two adapters from launching inference simultaneously. The existing register() resets seed 0 per frame and initializes candidates afresh; skipping completed frames does not intentionally change frame-independent registration settings. Numerical determinism across devices/software is not guaranteed.

launch_checkpoint_job.py owns the inference child process. It starts CPU scoring only after exit code zero plus an explicit completion marker with a matching final prediction hash. It then runs the audited primary, extended and failure analyses and writes the new report under the new job's analysis/ directory. CPU-only planning is the default; GPU work requires --run.

CPU plan, no GPU:

```bash
OPENBLAS_NUM_THREADS=1 /home/Sahil/miniforge3/envs/foundationpose/bin/python research/linemod_20261020/launch_checkpoint_job.py --output debug/bop_recovery_20261021 --reuse-recovered research/linemod_20261020/recovery/recovered_predictions.yml
```

After explicit approval, add --run to compute 10,614 remaining registrations while reusing the rounded prefix. The report must disclose the mixed precision and interrupted/recovered provenance. Alternatively omit --reuse-recovered and its path, select a new output such as debug/bop_native_retry_20261021, and add --run to recompute all 15,800 poses at native precision. Reusing rounded poses reduces computation but does not reconstruct a bit-identical native output file from the interrupted run. Neither option is an official BOP benchmark submission.

A fake-estimator test simulated an interruption after one frame, verified that no success marker was written, resumed only the missing frame, and rejected overwrite of a completed run. Tests import no GPU estimator. See checkpoint_tests.log. The recovered matrices also have CPU regression tests for frame-index-to-image-ID mapping and truncated matrices.

Runtime cannot be guaranteed: the former observed rate varied from roughly 0.7 to 1.4 poses/s. At those rates, 10,614 registrations would take roughly 2.1–4.2 hours plus initialization and analysis; 15,800 would take roughly 3.1–6.3 hours. A GPU probe from another workload was present during recovery inspection. A host-wide GPU lock cannot exclude jobs owned by other researchers, so GPU availability must be rechecked immediately before an approved launch. The checkpointed runner has not yet been exercised on the real GPU; an approved launch may reveal environment issues that CPU tests cannot detect.

A host reboot terminates these processes. Re-running the same approved checkpointed job with an unchanged manifest resumes its per-frame checkpoints; no automatic GPU relaunch on reboot is configured.
