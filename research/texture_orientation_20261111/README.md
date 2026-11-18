# Texture and reference-rotation matching

This is the original texture/orientation experiment, separate from the depth-supervision pilot.

## What runs

1. Render the local FoundationPose demo mustard bottle at 12 known yaw rotations, 30 degrees apart.
2. Render each view again with a uniform surface color. Geometry, normals, camera, light and background are held fixed. Automatically verify that paired depth buffers match.
3. Use the actual pretrained FoundationPose refinement shared CNN (`encodeA`) on RGB with XYZ withheld (three zero channels).
4. Pool to a 4x4 spatial descriptor and L2-normalize. Match a query against the references by cosine similarity.
5. Fine-tune encoder weights using same-object reference rotations as negatives. Cross-entropy over cosine similarities / 0.07 encourages the correct reference, rather than just recognizing the object. Frozen BatchNorm statistics preserve the original normalization state. No new projection head is introduced.
6. Compare pretrained and adapted encoders, each on textured-query/textured-reference and uniform-query/uniform-reference pairs. Same weights are used for the two appearance conditions. The adapted encoder trains on textured images only.

Training query offsets are +/-6 degrees at elevations 4/20 degrees. Evaluation query offsets are +/-8 degrees at elevations 15/8 degrees and a different light intensity. The shared reference bank is deliberately used in both stages, as in a reference retrieval system. Report nearest-reference accuracy and circular yaw error. Correct retrieval has 8-degree error due to reference spacing; a uniformly random reference has expected nearest-reference accuracy 1/12.

This creates explicit orientation supervision that encourages use of any informative appearance or shape cues. It does NOT guarantee texture learning. A textured-versus-uniform accuracy difference supports appearance dependence in this controlled setting; material color also changes, and this alone cannot isolate label semantics or prove real-world improvement. Lack of a difference may mean geometry already suffices or training failed. Inspect similarities as well as top-1 results.

## Run

From `/home/Sahil/FoundationPose`:

```bash
/home/Sahil/miniforge3/envs/foundationpose/bin/python research/texture_orientation_20261111/experiment.py --steps 150
```

Defaults to CPU encoder training and an EGL offscreen renderer (works without CUDA in this environment). `--device cuda` is optional when available. Outputs under `pilot/`:

- `reference_views.png`: actual paired rendered references.
- `matching_example.png`: a fixed held-out synthetic query, true nearest reference, predicted reference and measured similarities.
- `metrics.json`: metrics, pose schedules, asset/checkpoint hashes, provenance and limitations.
- `render_pairs.npz`, `similarities.npz`: reproducible rendered inputs and match scores.
- `texture_orientation_encoder.pth`: separately saved adapted encoder; not installed in the deployed pose pipeline.

## Scope

The mesh and texture come from `demo_data/mustard0/mesh` in the local FoundationPose checkout. This is a demo asset, not the BOP-format YCB-V mesh. The locally downloaded BOP dataset is LINEMOD; do not label these renders as BOP training images or mix coordinate frames with BOP annotations. A subsequent real-image BOP experiment needs the matching YCB-V mesh, its own camera parameters and poses, and official training images with separately held-out scenes.

This pilot estimates one yaw degree of freedom for one known object using synthetic images. It is not full 6D pose estimation, unseen-object evaluation or a benchmark result. Geometry remains imperfectly rotationally symmetric after texture removal: do not declare all uniform-object rotations equivalent or override BOP symmetry metadata. Neither renderer-conditioned held-out images nor the training loss establish real-image generalization.
