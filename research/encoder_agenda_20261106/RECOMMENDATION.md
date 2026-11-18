# Three-month encoder research recommendation

Prepared November 6, 2026. Goal supplied by Sahil: making encoders better.

**Recommended question: Can lightweight adaptation of FoundationPose's shared CNN improve pose refinement under realistic RGB-D corruption, while retaining clean accuracy and sensitivity to pose differences, with the fusion CNN and pose heads frozen?**

This is a recommendation to investigate, not a demonstrated improvement or a claim of new methodology. The dependable team deliverable is a reproducible encoder adaptation/evaluation pipeline plus the strongest validated checkpoint, even if ordinary fine-tuning wins. Publication novelty remains unestablished.

## What actually ran

Read the existing CPU orientation-probe implementation/report, GPU intervention code, paired intervention CSV and final GPU report. Extracted existing clean paired effects to `local_evidence.json`. Checked current GPU visibility outside the sandbox: RTX 5090, 32,607 MiB total, 19,861 MiB free at the check; future allocation is unknown. The sandbox-only GPU check failed, so that result was not taken as evidence that the host lacks CUDA.

Screened 36 explicit research questions in `questions.csv`. Ran 10,000 Monte Carlo decision scenarios for each of four priority settings, 40,000 total. These perturb subjective 1–5 ratings and priority weights. They do **not** simulate learned encoders, estimate expected accuracy gains, establish novelty, or give probabilities of research success. Source, random seed, assumptions and outputs are in `screen.py`, `simulation_metadata.json` and `decision_sensitivity.csv`. No new GPU inference or model training was run. Existing experiment results were read, not independently reproduced.

The strongest alternatives are ordinary corruption-augmented fine-tuning, small corruption-robust adapters, depth-validity feature gating, anti-aliased downsampling, and dense correspondence supervision. Ordinary fine-tuning has the best mean rank in delivery/balanced/limited-compute scenarios. The adapter has the best mean rank in the publication-oriented scenario. These rankings reflect the explicitly chosen ratings; they cannot scientifically select a winning architecture. The two leading approaches therefore belong in the same experiment, not competing three-month commitments.

## Why this direction follows from local evidence

The recorded orientation probe fits object-specific ridge regression from frozen pooled features to rotation entries: 20 development and 20 held-out images per object, 520 images overall. Final pretrained refinement/ranking features have recorded symmetry-aware mean errors 42.67/43.14 degrees versus a constant baseline of 81.92 degrees. Random features achieve 51.17/48.34 degrees. This shows orientation decodability in that GT-conditioned setup, not an encoder improvement or deployed pose accuracy.

The GPU study's paired clean-condition changes are:

| Intervention | Refined rotation error increase | Selected candidate sampled ADD(-S) error increase |
| --- | ---: | ---: |
| Replace observed shared features by spatial mean | 7.47 degrees | 15.87 mm |
| Shuffle observed shared-feature locations | 7.66 degrees | 13.76 mm |
| Shuffle final fusion-feature locations | 0.217 degrees | 0.022 mm |

Source: `../encoder_study_20261028/gpu_phase2/production/analysis/paired_encoder_interventions.csv`. Recorded matched clean comparisons cover 15,800 frames from 13 objects; repeats are not additional independent frames.

Interpretation: the installed computation is sensitive to the spatial organization/cross-branch alignment of shared features. Because only the observed branch is shuffled, this intervention also breaks rendered/observed correspondence. It does not isolate a shortage of spatial information or prove that stronger spatial encoding helps. Final fusion shuffling happens at a different location and has different semantics; small damage there does not establish that the fusion CNN is unnecessary. Destructive out-of-distribution interventions are motivation, not a diagnosis of the best training objective.

The mild corruptions tested so far produce comparatively small changes. A large robustness gain on those exact conditions may have little headroom. First find a realistic, reproducible team-relevant failure regime, including actual sensor frames, rather than selecting an arbitrary severe synthetic corruption because it makes the proposed method look good.

## Prior art and what a defensible contribution would require

1. [FoundationPose, CVPR 2024](https://arxiv.org/abs/2312.08344) already uses synthetic training and a pose-conditioned contrastive formulation. Generic pose contrastive training is not a new contribution.
2. [Kim and Cho, ICIP 2025, Adapting Foundation Features via Cross-View Contrastive Learning for Unseen Object Pose Estimation](https://snu.elsevierpure.com/en/publications/adapting-foundation-features-via-cross-view-contrastive-learning-/) already adapts features using real/rendered and cross-view correspondences for FoundPose. FoundPose and FoundationPose are distinct systems; novelty cannot be claimed simply by changing the base model.
3. [SCFlow2, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_SCFlow2_Plug-and-Play_Object_Pose_Refiner_with_Shape-Constraint_Scene_Flow_CVPR_2025_paper.html) uses geometry-constrained RGB-D refinement and novel-object generalization. Dense geometry supervision is established territory.
4. [WAPR, authors' ECCV 2026 project](https://wangyulin-seu.github.io/WAPR-site/index.html) explicitly addresses wide-angle unseen-object refinement. A broad proposal to widen the convergence range needs direct comparison with this work.
5. [BOP Challenge 2025](https://bop.felk.cvut.cz/challenges/bop-challenge-2025/) defines practical unseen-object and industrial evaluation settings; choose the relevant task before claiming benchmark comparability.

The possible contribution here is a demonstrated accuracy/robustness/compute advantage from encoder-only adaptation that preserves the installed downstream interface, with a clear account of when nuisance consistency helps or destroys pose sensitivity. The search above is targeted, not an exhaustive novelty review. Read full closest papers and their training recipes in week one before choosing the contribution wording. If the objective/adapter matches prior work, position this as a careful application and benchmark contribution unless a genuine distinction emerges.

## Minimal intervention and hypothesis

Start with the refinement network only. Put a small zero-initialized residual bottleneck after the final shared-CNN block, apply the same adapter to both rendered and observed branches, retain the 128-channel spatial output, and freeze original encoder/fusion/head weights and normalization statistics. Backpropagate through frozen downstream modules into adapter parameters; freezing parameters must not disable autograd through those modules. Rendering/cropping can be cached outside autograd. The existing prediction method uses inference mode and is not itself a training loop.

Hypothesis: task-supervised adaptation plus consistency across geometry-preserving sensor/appearance disturbances improves real-corruption pose success more than the same adapter trained on task loss alone, at essentially unchanged interface and small runtime overhead.

Begin with three loss terms at most: installed-style pose residual supervision with explicit symmetry handling; masked clean/corrupted feature consistency for the same candidate pose; and a modest clean-feature anchor to limit drift. Select their weights only on development data. Do not force two different candidate poses to have equal features. Do not impose pixelwise feature equality after crop shifts without coordinate warping, after invalid depth without validity masks, or on newly occluded/unobservable regions. Sensor RGB/XYZ misalignment can be calibration failure rather than nuisance invariance. For symmetric objects, use full annotated transforms and one consistent equivalent target, not incompatible independent rotation/translation minimizations.

If consistency hurts hard pose distinctions, that is a falsification of this objective, not a reason to weaken evaluation. Keep the minimal successful method; auxiliary pose decoders, feature Jacobians and new fusion architectures are optional future projects, not simultaneous requirements.

## Required comparisons

| Arm | Trainable component | Purpose |
| --- | --- | --- |
| A | None | Original pretrained baseline |
| B | Same small adapter; task loss only | Controls parameter count and adapter optimization |
| C | Same adapter; same augmented data; task loss only | Isolates augmentation from consistency loss |
| D | Same adapter; same augmented data; task plus consistency/anchor | Proposed method |
| E | Ordinary shared-CNN fine-tuning; same augmented data and frozen heads | Strong simple practical baseline |
| F, if feasible | Head-only fine-tuning on same data | Tests whether task mismatch is better fixed downstream |

B uses clean task training; C adds the exact corruption exposure used by D. E must have the same data, hyperparameter-selection opportunities and reported training budget; report its larger trainable parameter count. Also report a compute-matched E if compute differs materially. Compare separate loss components only if D has a reproducible benefit. Identity adapter must reproduce A before any training. Freeze other pipeline settings and random initialization schedules. Do not hide a stronger E because D looks more novel.

## Evaluation that protects the conclusion

- Treat every previously inspected LM frame as development evidence. LM-O overlaps LM objects/scenes and is an occlusion stress test, not independent unseen-object evidence.
- Establish train/development/final-test manifests before training. Use object-disjoint adaptation evaluation plus scene/sequence separation; hold out a second dataset or fresh team captures for the final test. State whether objects are unseen to adaptation only or also verified unseen to original pretraining. Audit mesh near-duplicates when using synthetic CAD training collections.
- Training may use GT labels. Final inference uses masks/candidates obtainable by the deployed pipeline, with no GT-centered crops and no inserted perfect GT candidate. Run the local controlled-candidate experiment separately as a diagnostic.
- Primary endpoint: object-macro symmetry-aware MSSD recall at a predeclared threshold, e.g. 0.1 object diameter, under a predefined realistic corruption mixture. This is one diagnostic recall, not official BOP AR. Report VSD/MSSD/MSPD via the BOP toolkit if claiming official benchmark metrics; specify visibility tolerances and task setting.
- Also report clean recall, each corruption separately, per-object results, actual sensor captures, warmed end-to-end median/p95 latency, peak memory, trainable parameters and training/rendering costs. Keep ranking frozen to isolate refinement; report candidate-level refinement and the full unchanged pipeline to detect whether selection hides a gain.
- Use at least three training seeds for finalists, pairing inference settings. Report uncertainty across seeds and bootstrap independent sequence/scene blocks rather than pretending temporally adjacent frames or candidates are independent. Estimate sample needs from pilot effect and block variance; thousands of repeated candidates are not a substitute for independent scenes.
- A proposed engineering target, to finalize before final-test access: at least +3 percentage points corrupted-condition macro recall versus the strongest simple baseline, no worse than -1 point clean recall, and <=10% end-to-end latency overhead. These numbers are planning targets, not predicted effects. Require a positive paired uncertainty interval for the primary difference, and enough independent test blocks to make that meaningful.

## Twelve-week schedule, approximately Nov 9, 2026–Jan 31, 2027

| Weeks | Work | Decision/deliverable |
| --- | --- | --- |
| 1–2 | Reproduce a complete small baseline; audit closest prior art; obtain sensor examples; define splits, primary endpoint and corruption magnitudes; verify identity adapter; implement/cache training pairs | Reliable benchmark, differentiable training path, pilot resource measurement; do not scale an invalid baseline |
| 3–4 | Run A–E on a small object-balanced development set, at least two pilot seeds; inspect pose-sensitive failures and ordinary fine-tuning | Continue adapter objective only if benefit is reproducible beyond augmentation alone with acceptable clean accuracy. If E wins, deliver E and redirect analysis to the reason |
| 5–6 | Train finalists with realistic missing-depth/appearance disturbances; tune only on development data; ablate consistency/anchor only when justified | Freeze candidate method and selection rule; avoid expanding to a new backbone |
| 7–8 | Adaptation-unseen objects and separate-scene development tests; real sensor validation; strongest-baseline comparison | Freeze method, checkpoint-selection procedure and final-test protocol |
| 9–10 | One final held-out evaluation campaign, three seeds, per-object and per-condition reporting, latency/memory and failure analysis | Reproducible evidence for either benefit or limitation; no held-out retuning |
| 11–12 | Team integration, versioned checkpoints/configs, reproduction commands, report and demonstration; paper draft only if evidence/novelty justify it | Useful team handoff with quantified supported claims |

Compute planning assumes recurring access to the observed GPU, not exclusive access. In weeks 1–2 measure rendering, I/O, forward/backward costs and cache size before promising run counts. Adapters reduce optimizer state but frozen downstream layers still require activation gradients. Lack of a ready training loop or a fresh independent evaluation set is a real schedule dependency. A 32-GB GPU alone does not establish that a particular batch size or entire training budget fits.

## Stop rules and fallback value

Stop scaling if baseline coverage is incomplete, corruption effects disappear under realistic initialization, identity adaptation changes predictions, or test data leak into method selection. By week four, stop the extra consistency objective if it cannot beat same-augmentation task-only training or damages clean pose performance. If the new method merely improves the pooled rotation probe, do not call it a pose-estimation improvement.

The fallback is the strongest simple encoder fine-tuning recipe, a reliable held-out benchmark, realistic sensor failure taxonomy and a reproducible adaptation harness. Those directly help the team even if the novel loss fails. A claim supported by that result is more defensible than promising a new best encoder in advance.

## Code entry points

- Existing orientation readout: `../encoder_study_20261028/probe.py:21`.
- Shared encoder/interface: `../../learning/models/refine_network.py:38`.
- Candidate generation and intervention semantics: `../encoder_study_20261028/gpu_phase2/run_gpu.py:74` and `:86`.
- Existing real-pair prediction calls: `../encoder_study_20261028/gpu_phase2/run_gpu.py:216`.
- Full inference preprocessing: `../../learning/training/predict_pose_refine.py`.
- Decision analysis reproduction: `python3 research/encoder_agenda_20261106/screen.py` from the FoundationPose directory.
