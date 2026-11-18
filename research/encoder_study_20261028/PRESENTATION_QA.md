# Questions your professor may ask

**Did you test the encoder or the whole pose estimator?** Both shared and fusion CNN encoders were probed. Full task heads, renderer, candidate ranking, and pose refinement accuracy were not evaluated in this study.

**Why did you use GT?** To hold alignment and crop construction fixed while changing the input. That isolates a feature response, but limits claims about blind registration.

**Does lower cosine distance mean a better encoder?** No. It may mean robustness, or loss of useful information. Spatial sensitivity may be needed for pose correction. We report retrieval and random controls, but downstream validation remains necessary.

**Why compare to random weights?** To see which patterns are present without learning. The first comparison includes BatchNorm changes; the supporting factorial controls separate convolution and normalization effects.

**Do weight norms tell us which modality matters?** No. Input scaling, normalization, and downstream computation prevent that interpretation. Removal experiments also represent extreme out-of-distribution inputs.

**Is the self-pair experiment real pose matching?** No. It is an explicit artificial control for the fusion CNN. A renderer-based experiment is required to test actual pose-conditioned matching.

**Why are orientation-probe errors much larger than earlier FoundationPose pose errors?** This readout uses globally pooled features from a single observation and only 20 training examples per object. The installed estimator compares rendered/observed spatial features, uses fusion and task heads, and refines candidates iteratively. These are different systems and evaluation protocols. The probe's degree errors must not be reported as FoundationPose's pose accuracy.

**Is this a new method or publishable performance improvement?** It is a reproducible diagnostic study that supports hypotheses and next experiments. It does not establish an improved method or benchmark result.

**Why did it not run for twelve hours?** That was an early effort estimate. Measured CPU experiments completed faster. Research quality comes from valid controls and interpretation, not running the same work longer.
