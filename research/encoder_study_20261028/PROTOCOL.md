# FoundationPose encoder study

Requested scope: approximately twelve hours of substantive research, focused on encoders. Runtime is measured, not padded to meet a duration. This directory preserves the previous studies, weights, and dataset.

## Questions

1. What architecture and actual input representation do the installed encoders use?
2. How do intermediate features respond to appearance disturbances, depth disturbances, missing modalities, and spatial shifts?
3. How do trained refinement and ranking encoders differ from their own randomly initialized controls?
4. Does feature sensitivity change with layer depth and object identity?

## Primary experiment

Use the existing 520-frame object-balanced LINEMOD diagnostic split (13 objects, 40 frames each). Preserve its development/held-out labels; this split has already been used and is not an untouched benchmark. No hyperparameters are fitted to these data. Analyze both separately and disclose prior exposure.

Load both full pretrained networks strictly on CPU. Extract original class definitions with Python AST to avoid importing GPU-dependent rendering utilities; preserve class bodies verbatim and record source hashes. Use eval mode and inference mode. Random controls have the identical architecture, fixed seed, and initial BatchNorm statistics; this comparison combines effects of learned weights and normalization statistics.

Construct 160x160 observation crops from projected GT translation of the bounding-box-centered CAD model and metadata CAD diameter, following the source crop-window geometry, with model-specific crop ratios. Centered translation is t + R times the CAD bounding-box center, matching the estimator's centered mesh convention. GT is used to fix diagnostic alignment, not to claim blind registration performance. RGB is scaled to [0,1]. Depth is backprojected into camera XYZ, centered on the pose translation, and divided by CAD radius. Apply source-equivalent componentwise out-of-bounds invalidation. Reuse Kornia's perspective warp and interpolation settings. Source-specific depth validity thresholds are retained: refinement inherits PairH5Dataset (0.001 m); scoring inherits TripletH5Dataset (0.1 m). This CPU observation-branch experiment has no rendered branch and makes no claims about rendered/observed matching or full pose accuracy.

The experiment deliberately uses raw dataset depth and fixed metadata diameter. Production registration additionally erodes/bilaterally filters depth and estimates mesh diameter from sampled vertices. Therefore these are controlled diagnostic inputs, not exact replay of production preprocessing. Crop ratios also differ (refine 1.2; score 1.1), so cross-network differences cannot be attributed to learned weights alone. Trained-versus-random comparisons within each architecture use identical inputs.

Disturbances: brightness factors, blur, RGB noise, depth noise in millimetres, depth dropout, zero RGB, zero XYZ, RGB-only shifts, XYZ-only shifts, joint shifts, and rectangular occlusion. Stochastic conditions use three deterministic seeds paired across network variants. Disturbances are diagnostic interventions, not calibrated sensor models. Shifts are crop pixels and do not emulate the previous initialization-mask shifts.

Measure every shared-branch stage: spatial cosine distance to clean features, relative feature change, pooled cosine distance, activation zero fraction, activation RMS, and finite-value checks. Use pooled feature object retrieval across development/held-out splits as a secondary representation diagnostic, not a pose metric or proof of generalization to unseen objects. Aggregate per frame before computing uncertainty; seeds and layers are not independent sample units. Bootstrap frames within object and macro-average objects; confidence intervals apply to this fixed object set and not to unseen objects. Report every condition rather than selecting favorable ones.

## Deliverables

Architecture/weight audit, checkpointed per-frame measurements, summary tables, figures, reproducible scripts, report, and presentation with speaker notes. Clearly separate measured results from hypotheses and proposed GPU follow-up.

## Environment constraint

Initial checks: nvidia-smi cannot communicate with the driver; torch.cuda.is_available() is false. Do not change drivers or system settings. CPU encoder experiments remain feasible. Actual GPU pose refinement/ranking is deferred until CUDA is available.

## Primary references

- Wen et al., FoundationPose, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf
- Official project: https://nvlabs.github.io/FoundationPose/
- Installed sources: learning/models/refine_network.py, score_network.py, network_modules.py; learning/datasets/h5_dataset.py; Utils.py.

The paper describes a shared CNN applied to rendering and observation within each network. The refinement and ranking networks have the same backbone design but separate checkpoints; architectural similarity does not imply identical learned weights.
