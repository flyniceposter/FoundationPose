# FoundationPose encoder study

This is an isolated, CPU-based research study of the installed frozen CNN encoders. No original weights, data, or earlier studies are changed.

When the workflow is complete, open **PRESENTATION.pdf** or **PRESENTATION.pptx** for the talk, **REPORT.md** for methods/results, **ONE_PAGE_SUMMARY.md** for the meeting brief, and **SPEAKER_NOTES.md** / **PRESENTATION_QA.md** for preparation. The PowerPoint uses rendered slide images with speaker notes. The optional offline PRESENTATION.html supports arrow-key navigation; keep its slides/ directory alongside it.

## Status

- results/status.json: main 520-frame measurement run.
- secondary_results/status.json: common-input and BatchNorm/fusion controls on 130 development frames.
- workflow_status.json: orientation probe, analysis, and artifact generation.
- DELIVERABLES.json: generated after report and presentation creation.
- final_verification.json: artifact/provenance checks after generation.

The early 8–12-hour estimate was not a guaranteed runtime. Measured runtimes are reported; no deliberate waiting is used to inflate research duration. No claim of GPU pose validation is made.

## Experiments

Read PROTOCOL.md, SECONDARY_PROTOCOL.md, and PROBE_PROTOCOL.md for the primary and supporting designs and their limits. The main experiment has 23 named conditions (39 condition/seed combinations), four shared stages, both pretrained checkpoints, and matched random-weight controls. Supporting work probes all five fusion stages with explicitly artificial self-pairs and separates convolution from BatchNorm effects. A supervised ridge readout evaluates orientation decodability from frozen pooled features; it is not the installed pose head.

## Reproduction

Use the existing environment; no new packages are required:

```bash
cd /home/Sahil/FoundationPose/research/encoder_study_20261028
/home/Sahil/miniforge3/envs/foundationpose/bin/python validate.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python validate_analysis.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python audit.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python study.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python secondary.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python probe.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python analyze.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python illustrate.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python build_report.py
/home/Sahil/miniforge3/envs/foundationpose/bin/python verify_artifacts.py
```

Run each measurement only once at a time. The scripts lock their output directories and resume matching per-frame checkpoints. Source/configuration mismatches need a new --output directory. finish.py waits for measurement completion, then runs the orientation probe, analysis, and report generation; it assumes the measurement jobs have already been started and does not fix failed measurement jobs.

The code uses this workspace's existing data and weight paths. A presentation archive does not include the dataset, original FoundationPose source, or checkpoint weights. Porting the experiment to another machine requires adjusting these paths and installing compatible dependencies. Original input provenance is captured in results/manifest.json.

## Presentation guidance

Lead with the actual input (RGB + XYZ), explain the shared and fusion CNNs, show paired disturbances, and finish with the orientation probe and its controls. Distinguish feature distance from pose error. A low-drift encoder can be uninformative; a spatially sensitive encoder can be useful for pose alignment. GT-conditioned crops, reused splits, known object identity in the probe, and sequence correlations limit generalization claims. The artificial fusion study never establishes real rendered/observed matching.
