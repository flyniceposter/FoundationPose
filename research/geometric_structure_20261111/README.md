# Visible geometry supervision for the FoundationPose shared encoder

This isolated development experiment uses the actual pretrained refinement `encodeA` and adds a small convolutional depth decoder. It updates encoder weights, rather than only probing frozen features. Existing inference code and pretrained weights are unchanged.

Input is cropped RGB in [0,1] concatenated with three zero channels. Withholding XYZ prevents copying depth from the original RGB-XYZ input. The target is measured visible-object depth minus that frame's median valid object depth, divided by object diameter. This teaches relative visible shape; it does not recover absolute camera distance or the invisible back surface. Centering uses target depth only. Ground-truth masks/bounds provide controlled diagnostic crops.

Loss is masked smooth-L1 depth error plus 0.2 times adjacent-pixel depth-gradient error. Invalid/missing depth and background are excluded. BatchNorm running statistics remain frozen. Two arms start from identical encoder/decoder weights and see identical minibatches: decoder training with a frozen encoder, and decoder training with encoder adaptation. A constant-zero relative-depth baseline is also reported. Validation MAE is averaged per frame, in millimeters after undoing diameter scaling.

Run from the FoundationPose directory:

```bash
/home/Sahil/miniforge3/envs/foundationpose/bin/python research/geometric_structure_20261111/train_geometry.py --steps 60
```

Use `--device cuda` when available. Outputs are separate checkpoints, exact train/validation frame manifests and metrics in `pilot/metrics.json`, and a two-example `pilot/geometry_example.png`.

## Interpretation and limits

Local availability is LINEMOD test sequences, not YCB-V training data. All frames used here are explicitly reclassified as development evidence. The first 75% of sampled frames train the models; the last 25% are a temporal validation block. This is a small within-sequence pilot, not a clean benchmark, independent-scene evaluation, or evidence of unseen-object generalization. These frames must not later be reported as untouched test data. Some were already inspected by earlier local studies.

A short pilot checks that data, gradients, checkpoints and visualizations work. Compare the frozen-encoder baseline before claiming an adaptation benefit. Neither lower reconstruction loss nor an auxiliary decoder proves improved pose estimation. The adapted RGB-only encoder also needs joint RGB-D/pose training and downstream pose validation before replacing the deployed encoder. This experiment does not silently install its checkpoint.

Next scientific step: acquire proper training images or synthetic rendered training scenes, define independent held-out scenes/objects before selection, and compare pose-supervised training with versus without this auxiliary geometry loss under the same budget. When RGB-D is supplied, mask/corrupt geometric inputs where reconstruction is supervised to avoid a copying shortcut.
