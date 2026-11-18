# Active twelve-hour GPU encoder run

Started Wednesday, October 28, 2026, at 5:06 PM Eastern. Scheduled measurement deadline: Thursday, October 29, at 5:06 AM Eastern. Friday's presentation date is October 30.

The authoritative live status is [CURRENT_RUN.json](CURRENT_RUN.json), a link to gpu_phase2/production/status.json. This note records the launch; it is not a substitute for checking the status file after the deadline.

This is a persistent local GPU process, launched outside the restricted execution sandbox, PID 261241 at launch. It continues independently of chat sampling while the server and process remain running. The job does real experiment work through the twelve-hour budget, checkpointing every unit. It does not intentionally repeat finished work or sleep to fill time. The queue is larger than the budget; actual coverage is reported.

## What is running

- Real rendered/observed inputs through both pretrained encoders and installed task heads.
- Thirty-seven fixed local pose candidates per unit.
- Five input conditions and seven encoder interventions.
- Observed RGB/XYZ removal, pooled/zeroed/shuffled shared features, and shuffled fusion features.
- One-step pose refinement, fixed-set ranking, per-layer feature diagnostics, and exact paired comparisons.
- Object-balanced cohorts, existing diagnostic frames first, then additional LM frames and further perturbation seeds.

The full design is [gpu_phase2/PROTOCOL.md](gpu_phase2/PROTOCOL.md). Candidates are GT-conditioned; this is an encoder diagnostic, not an official pose benchmark. Ranking ties are handled explicitly to avoid giving a constant scorer a perfect result merely because GT is listed first.

## Results

- [Live status](gpu_phase2/production/status.json)
- [Partial report, updated every 15 minutes](gpu_phase2/production/LIVE_REPORT.md)
- [Progress log](gpu_phase2/production.log)
- Raw checkpoints: gpu_phase2/production/units/
- At the end: gpu_phase2/production/GPU_REPORT.md, GPU_PRESENTATION.pdf, GPU_PRESENTATION.pptx, and analysis/*.csv.

The seven-intervention pilot and numerical checks passed before production launch. The production job's first 5,566 units completed without failures. Future failures, if any, are recorded explicitly; no claim of guaranteed successful completion is made merely from launching it.

Correction to the first-phase environment diagnosis: the RTX 5090 and driver work outside the sandbox. The earlier CUDA-unavailable observation reflected restricted device access, not a broken driver. The earlier CPU results remain valid within their stated scope; the new run supplies real-pair and downstream task evidence.
