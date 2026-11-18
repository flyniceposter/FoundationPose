# Dataset audit and queued filtering validation

Authorized by the user: compare included mustard/driller demos against BOP and investigate removable/unused points, including hours of validation.

Completed CPU audit: debug/point_audit_v1/REPORT.md, per_frame.csv, summary.csv and examples/. It covers 520 BOP study frames and the one available initialization mask per demo. BOP invalid masked depth: 4.36%; mustard 18.60%; driller 20.55%. Existing invalid-depth rejection already addresses these. Of valid BOP masked depth, 1.71% lies outside the largest connected mask component. These are candidates, not proven bad points. Surviving depth tails can be real object geometry. Inspect overlays before assigning a mechanism.

Existing code filters invalid/near-zero depth and performs erosion and bilateral smoothing. The audit reproduces only the erosion stage on CPU; it does not claim these are final neural-network inputs. Full scene depth remains part of refinement/ranking, so background cannot simply be called unused.

Queue: foundationpose-point-filter-v1.service. It waits for debug/mask_study_v1/status.json to say complete, then starts the isolated point_filter_study.py run. It does not contend with the current GPU job. CPU geometry and depth-gating tests passed (3 tests); a real GPU smoke for this wrapper has not been run because the existing study owns the GPU lock. An exception records failure and stops; inspect status.json and service.log rather than assuming completion.

Protocol: 520 existing study frames; three paired seeds; six conditions (control, dilation5, shifts ±5 on x/y). Four development methods: raw, largest component, median-depth gates of half and one CAD diameter. Uses depth and CAD diameter available at inference; filtering never uses GT poses. Only initialization masks change; full-scene depth stays unchanged. Development macro ADD(-S) chooses a method, ties favor raw; frozen selection precedes held-out inference. Maximum registrations: 28,080. If raw wins, redundant held-out work is skipped. Actual throughput/remaining time must be measured from live checkpoints.

Input audit examined held-out mask/depth statistics, so disclose this when presenting the exploratory study. Do not tune on held-out pose outcomes. This does not establish novelty or guarantee a beneficial filter. No source data or original FoundationPose files are changed.

Reports and outputs: debug/point_filter_study_v1/REPORT.md, per_pose.csv, per_object_summary.csv, summary.csv, paired_intervals.json, perturbation_intervals.json, heldout_recall.png. Filter configuration and source hashes are in filter_manifest.json and manifest.json. The frozen source/configuration must match for resume. Stop/inspect the user service before attempting manual resume with the same command and output.
