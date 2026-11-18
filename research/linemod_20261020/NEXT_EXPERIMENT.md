# Proposed experiment: segmentation-mask robustness

Status: preregistered proposal only. No additional GPU inference is authorized or running. The baseline is still pending; inspect its measured results and failure patterns before committing compute.

The current baseline uses ground-truth visible masks. The next controlled question is whether boundary and localization errors in those masks materially reduce pose accuracy. Hypothesis: modest mask perturbations reduce ADD(-S), MSSD and MSPD recall, particularly when the visible object region is small. Visibility associations in the baseline will guide interpretation; they cannot establish this causal effect.

Use a fixed object-balanced set: 20 development plus 20 held-out frames for each of the 13 objects (520 frames total), ranked by SHA-256 of `mask-study-v1/<obj_id>/<im_id>`. The finalizer will write the exact manifest. The sets are disjoint. Because the full baseline is examined, they are study splits rather than an untouched official benchmark test.

Hold RGB, depth, model, calibration, rotation candidates, refinement iterations and scoring fixed. Pair seeds 0, 1 and 2 across every condition. Compare 13 conditions: original mask; erosion by radius 2 or 5 pixels; dilation by radius 2 or 5 pixels; shifts by ±2 or ±5 pixels along each image axis. Fix a circular morphology kernel and clip shifted masks at image boundaries. Save variants under a new experiment directory; never alter dataset masks. An empty perturbed mask is a recorded failure.

Measure per-frame mask IoU against the original mask, translation error in mm, raw rotation error in degrees, ADD/ADI and symmetry-aware MSSD/MSPD. Report paired recall changes, object macro averages, error percentiles, and warmed per-frame latency. Keep missing/invalid outputs in recall denominators. Use a paired frame bootstrap stratified by object, keeping conditions and seed replicates together when resampling. Develop any mitigation using only development frames. Evaluate all preregistered conditions on held-out frames without selecting the best severity from their results.

The full proposed factorial design requires 520 × 13 × 3 = **20,280 pose registrations**. This is substantial GPU work and requires explicit approval before launch. A small approved smoke test should validate variant generation and manifest matching before committing the full budget. No throughput for this future experiment has been measured.

Mask perturbations isolate sensitivity; they do not establish performance with a real segmentation model. A detector-produced-mask comparison can follow using the same frames, with detector/model/configuration recorded. Depth-noise and refinement ablations should be separate experiments so each effect is attributable to one intervention. Scores remain diagnostic unless an appropriate official BOP evaluation protocol is run.
