# Prior work and scope of this experiment

The broad idea is already published. DeepIM (ECCV 2018), section 3 on iterative training and section 4.1 / Table 1, reports that training only one-step corrections can fail to benefit from repeated inference. It trains on poses updated by its own predictions. Its discussion attributes the poor single-step result partly to insufficient rendered poses close to ground truth. Thus neither training for repeated refinement nor supplying near-correct examples should be claimed as a new general idea.

Primary source: [DeepIM: Deep Iterative Matching for 6D Pose Estimation](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.pdf), Yi Li et al., ECCV 2018.

FoundationPose already predicts camera-frame translation and rotation updates and supports repeatedly rendering and refining the latest pose. The present experiment does not introduce iterative inference or a new head architecture.

Primary source: [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf), Bowen Wen et al., CVPR 2024, section 3.3.

What is tested here is narrower: a fixed mixture of standard, small-error and exact-pose supervised examples while freezing the original FoundationPose encoder and training its fusion/rotation/translation modules. It does not feed model-generated poses back into training, differentiate through multiple renders, or implement DeepIM's iterative-training algorithm. A positive ablation would establish a useful adaptation result for this configuration, not establish method novelty. No exhaustive claim that this exact mixture has or has not appeared in prior literature is made.
