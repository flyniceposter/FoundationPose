# Twelve-hour GPU encoder investigation

User authorized a further twelve continuous hours on October 28, 2026; Friday presentation date is October 30, 2026, America/New_York. The server GPU is accessible outside the execution sandbox. The phase-one report's GPU limitation described sandbox access, not a failed NVIDIA driver. Phase-one CPU measurements remain valid within their stated scope.

## Questions and endpoints

1. Do feature distances on real rendered/observed pairs track known pose misalignment?
2. Does removing observed RGB or XYZ alter one-step refinement or fixed-set ranking?
3. Does destroying observed spatial structure while preserving channel means change task behavior?
4. How do input disturbances and these encoder interventions interact across objects and initial error sizes?

Use the installed frozen refinement and ranking checkpoints, upstream rendering/cropping/normalization, source depth erosion and bilateral filtering, CAD model centering, and per-frame calibration. Fix a 37-pose candidate set: GT plus ±5/10/20 mm translations on three axes and ±5/15/30 degree rotations on three camera axes. Seed zero uses canonical axes; later seeds rotate the perturbation axes by a deterministic random rotation. This is local GT-conditioned pose diagnosis, not global registration or an official benchmark.

Conditions: clean, RGB intensity ×0.5, RGB Gaussian blur sigma 3 full-image pixels, valid raw-depth Gaussian noise sigma 5 mm, raw-depth dropout 30%. Apply image/depth interventions before source preprocessing; they are artificial diagnostic noise models. Candidates remain fixed within a frame/seed across interventions. Depth intervention RNG is paired across encoder ablations.

Encoder interventions: intact; zero observed RGB; zero observed XYZ; replace final observed shared-CNN map by its channelwise spatial mean broadcast to every location; zero the same observed map; permute spatial positions of that observed map; permute spatial positions of the final fusion-CNN map. The rendered branch remains unchanged for observed-branch interventions. Hooks operate on clones and are removed after each unit. Pretrained weights and BatchNorm statistics remain unchanged. These are out-of-distribution causal interventions on the installed computation, not trained replacement encoders or proofs of deployable improvements.

Capture per-candidate rendered/observed spatial and pooled cosine distances at all four shared stages and all five fusion stage pooled norms. Record one-step refined poses and fixed-candidate ranking logits. Ranking is measured on the original candidate set; it is not ranking-after-refinement. Main task outcomes: change in translation error (mm), raw and discrete-symmetry-aware rotation error (degrees), and sampled-vertex ADD/ADI diagnostic change; top-scored candidate error and within-set Spearman association with initial sampled ADD(-S). Pose errors retain GT candidate in ranking; disclose this favorable local setting. Annotated symmetries use centered-model transforms. Sample 512 CAD vertices deterministically; sampled ADD/ADI is not exact full-mesh or official BOP evaluation.

## Schedule and stopping

Ranking tie rule: report expected selection error averaged over all exactly maximum-logit candidates. Preserve raw argmax indices separately. This prevents a constant-scoring intervention from appearing perfect merely because GT is the first candidate. Record tie counts and the fraction of defined Spearman correlations; undefined constant-score correlations are never imputed as perfect.

Twelve-hour wall-clock budget starts when the persistent production job starts, after pilot verification. Schedule balanced passes across 13 objects; previously selected 520 frames first, followed by the remaining LM frames. For each cohort, loop input condition and encoder intervention before revisiting frames under another seed. The queue is intentionally larger than twelve hours. Stop launching new units at the deadline, finish the current unit, and generate a final report and slide deck. Do not repeat finished units or sleep to inflate runtime. Report completed and failed units, object counts, condition/intervention coverage, deadline and elapsed work time. Never compare an incomplete condition group as though it had full coverage. Paired differences use only exact matching frame/seed/condition units; disclose their denominators. No held-out model selection is performed. Input GT and prior diagnostic-split exposure limit generalization claims.

Atomic per-unit JSON checkpoints and progress JSON preserve work. Periodic summaries are descriptive and explicitly partial. Final analyses average repeated candidates/seeds within frame, then macro-average objects. Frame-stratified paired bootstrap intervals are conditional on these objects, selected frames, candidates and fixed models, and may understate sequence-correlated uncertainty. No claim of unseen-object transfer, official BOP accuracy, or a new SOTA method is made. A persistent local process performs the work; chat sampling is not represented as continuously running.
