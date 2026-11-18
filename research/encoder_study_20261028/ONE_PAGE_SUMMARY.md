# Encoder study: meeting brief

The shared encoders load correctly and use RGB plus XYZ, with 684,480 shared-CNN parameters each.

The largest measured final-layer spatial change was zero_xyz for refinement (0.3442) and joint_shift_5 for ranking (0.4080).

Under identical clean inputs, cross-network linear CKA is 0.924 for the shared CNN and 0.816 for the artificial self-pair fusion CNN. This describes representation geometry, not task accuracy.

On the object-specific orientation readout, final pretrained pooled features give 42.67° (refinement) and 43.14° (ranking) mean symmetry-aware error, versus 81.92° for the constant baseline. Random-feature controls give 51.17° and 48.34°. These are GT-conditioned diagnostic probes, not deployed pose errors.

For a joint 5-pixel shift, spatial versus pooled cosine distances are 0.3292 versus 0.0008 in refinement and 0.4080 versus 0.0008 in ranking. Pooling can hide spatial changes relevant to alignment.

| Network | RGB_removed | XYZ_removed | Joint_shift_5px | Depth_noise_5mm |
| --- | --- | --- | --- | --- |
| refine | 0.2296 | 0.3442 | 0.3292 | 0.0055 |
| score | 0.3044 | 0.2740 | 0.4080 | 0.0024 |

520 primary frames across 13 objects; paired random controls. Supporting common-input controls cover 130 development frames and both shared/fusion CNNs. These are feature diagnostics, not pose accuracy. Native preprocessing differs across networks; common-input controls are provided.

See REPORT.md and PRESENTATION.pdf for details.
