# Speaker notes

Suggested duration: 12–18 minutes plus questions.

## Slide 1: What does FoundationPose’s encoder encode?

Open with the research question: before proposing an encoder replacement, understand its actual inputs, feature behavior, and what evidence would connect those features to pose quality. This study is diagnostic and does not claim a new method.

## Slide 2: The research question

Explain that feature stability alone is not success. A pose encoder needs to retain alignment information, so some sensitivity is useful. The goal is to characterize the network before attributing failure to it.

## Slide 3: Six channels, two CNN stages

The constructors default to four channels, but both installed configs specify six. Separate refinement and ranking checkpoints use the same architecture. The output here is before positional embedding and task-specific attention/heads. Architectural receptive field reaches 43 input pixels in the shared CNN and 147 in the fusion CNN.

## Slide 4: Where the CNN encoders sit

The observed and rendered branches share the first CNN within each task network. Their concatenated features enter the second CNN. Our primary run probes the observed branch; the supporting fusion run deliberately uses observation/self-pairs and stops before the task heads.

## Slide 5: A controlled dataset experiment

The dataset objects are 1,2,4,5,6,8,9,10,11,12,13,14,15. The split has been used previously; disclose that it is not untouched benchmark data. Multiple measurements of a frame do not create independent samples. Native crop ratios differ between tasks.

## Slide 6: What was changed—and measured

Noise is applied to valid crop depth before reconstruction of normalized XYZ. These are custom interventions, not calibrated physical sensor models. A shift moves input data, unlike the earlier study which shifted the initialization mask. Confidence intervals condition on these objects and can be optimistic under sequence correlation.

## Slide 7: Implementation checks passed

Describe why this matters: wrong depth units, crop centers, or validity thresholds could produce convincing but invalid results. The source normalization was executed on CPU with only CUDA transfers replaced by no-ops for validation. The full study does not use production rendering or depth filtering.

## Slide 8: Which disturbances change the features?

Read this as feature distance, not error rate. Bars compare pretrained and random encoders using each task’s native input construction. Intervals bootstrap frames within objects. Different interventions are not calibrated to equal severity. Refer to the report table for precise values.

## Slide 9: Three concrete comparisons

Removing XYZ entirely is a much stronger intervention than adding five millimetres of noise. These numbers quantify representation sensitivity; they do not prove which modality contributes more to task accuracy. Spatial sensitivity can be useful for alignment.

## Slide 10: Sensitivity changes with depth

The two strided layers and two residual blocks do not have identical response profiles. Compare each line within a task and a metric. Do not read increasing feature distance as a monotonic decline in pose quality.

## Slide 11: A pooled vector can hide spatial changes

Spatial cosine compares activations at their fixed grid positions. Global pooling discards that layout, so a translated feature map can have a similar pooled vector even when spatial distance is substantial. This supports retaining spatial probes for pose analysis; it does not prove a downstream pose effect or establish that all pooling is harmful.

## Slide 12: A fixed example of feature responses

This is the first development row in the existing split, not a handpicked success. The bottom maps show activation RMS across channels with the same color scale within each network. They are not attention or saliency. RGB may remain visually unchanged when only geometry is disturbed.

## Slide 13: Separate weights from BatchNorm

This supporting experiment uses the same crops and validity threshold for both networks, 130 development frames, and three initializations per control. Random-plus-trained-BatchNorm and trained-plus-reset-BatchNorm isolate parameter groups. Hybrids are deliberately out of distribution and are not deployable model replacements.

## Slide 14: Low drift alone can be misleading

The random-convolution/trained-BatchNorm hybrid can barely change even after a severe intervention because input-dependent variation is small relative to a common activation offset. Resetting trained BatchNorm can greatly inflate scale. Both are artificial mismatches, not deployed model failures. This shows why stable features need information and scale checks.

## Slide 15: Probe the pair-fusion encoder

Important disclosure: the clean reference branch is an observation, not a rendering. This artificial observation/self-pair test measures perturbation propagation through the actual fusion blocks. It does not establish rendered/observed matching or ranking ability.

## Slide 16: Similar architecture, different representation geometry

Linear CKA on pooled common-input clean features is 0.9237 at the final shared block and 0.8161 at the final self-pair fusion block. CKA compares centered relationships between samples and is not a pose metric. Cite Kornblith et al., ICML 2019. Pooling loses spatial information.

## Slide 17: A representation sanity check: object retrieval

A clean development gallery has 260 samples; held-out queries have 260. Cosine nearest-neighbor retrieval predicts one of the same 13 object identities. No classifier is trained. Random features can also retrieve images, and background or sequence similarity can help. This is not unseen-object generalization or pose accuracy.

## Slide 18: Is orientation decodable from pooled features?

A separate ridge readout is trained for each object from only 20 development frames. Five-fold development-only selection chooses regularization; every fold standardizes from its own training data. We project predictions to rotations, evaluate on held-out frames, and compare to a constant rotation and three shuffled-label controls. The encoders stay frozen. GT-conditioned crops, known object identity, and temporal correlation limit the result. This is not the installed pose estimator.

## Slide 19: The learned-feature advantage has limits

These are descriptive comparisons on a reused diagnostic split. They do not establish significance, unseen-object transfer, or an optimal encoder layer. The earliest stage has 64 pooled channels while subsequent stages have 128, so cross-stage comparisons also change feature dimension. Within-stage trained/random comparisons match dimension.

## Slide 20: What we can—and cannot—conclude

This is the key scientific boundary. Explicitly say no ADD(-S), BOP AR, or speedup improvement is claimed. The work provides a reproducible basis for the next causal or downstream study. CUDA was unavailable; no GPU experiment is being implied.

## Slide 21: The next encoder experiment

Keep the same candidate poses and crops within each controlled comparison. Include symmetry-aware errors and disclose GT use. A learned relationship between encoder drift and downstream failure should be validated on new sequences, not on the same data used to form the hypothesis.

## Slide 22: Discussion and references

References provide architecture context, CKA methodology, corruption-testing context, and motivation for model-dependent sanity checks. We did not reproduce ImageNet-C or saliency-map experiments. Suggested question for the professor: should the next step prioritize real-pair pose sensitivity, learned feature inspection, or encoder adaptation?
