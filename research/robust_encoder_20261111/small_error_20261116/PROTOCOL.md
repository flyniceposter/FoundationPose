# Small-error training ablation — November 16, 2026

Question: does mixing small-error and exact-pose examples into supervised downstream refinement training improve the normal five-iteration registration output?

This is an exploratory adaptation experiment on five previously inspected LINEMOD objects, not a novel-method or official benchmark claim. DeepIM (ECCV 2018), section 4.1 and Table 1, already motivates close-to-ground-truth examples and iterative training. We test a static mixture, not unrolled or on-policy training.

Matched arms, seeds 42/123/2026: `standard` uses the existing 5–18 degree / up-to-4%-diameter per-axis perturbations. `mixture` uses two standard examples, one small-error example (0.1–3 degrees / up-to-0.5%-diameter per axis), and one exact-pose example per batch of four. Frame/perturbation indices, original pretrained initialization, corruption schedule, optimizer, steps, and checkpoint-selection criterion are matched. No changes to the encoder, scoring network, or production inference code. Fusion and translation/rotation heads train; all BN buffers stay fixed. AdamW 1e-5, weight decay 1e-4, clipping 1, 1500 steps.

Use the existing 24 train / 8 development frames per object and production RGB+XYZ crop conventions. Add small/exact starts for those same frames only. Development criterion: 0.5 standard + 0.25 small + 0.25 exact normalized symmetry-aware pose loss, each category equally weighting objects and observed/corrupted inputs. Select every 100 steps, identically for both arms.

Before training, lock eight unused evaluation frames per object (40 total), spread evenly over available last-15%-temporal-partition frames. Exclude all frames in the robust-encoder manifest, head-training manifest, earlier registration records, and real-rotation RGB-D/reference cache. Assert disjointness. These are unused by the listed local experiments; not proven unseen to model pretraining, not independent objects, and not a new dataset. Save manifests and hashes before training.

Full registration on all 40 fixed frames, observed and 40% block depth dropout; no GT pose initialization or GT candidate selection. GT visible masks are supplied as in existing evaluation. Record scorer-selected poses after iterations 1/2/3/4/5/10 from a continuous ten-step run; same initial candidates across arms/seeds. Primary endpoint: mean ADD-S / object diameter at iteration 5, observed input, paired mixture minus standard, with equal object weights. Secondary: ADD, dropout, and changes between early/late iterations. Report original pretrained model as a reference, not the matched training control. Never choose a test iteration by evaluation error.

Define convincing evidence within this limited experiment as >=5% primary relative improvement, improvement in all three seed means and >=4/5 object means, and a paired object-cluster bootstrap 95% interval entirely below zero. Bootstrap intervals with only five objects are descriptive and do not establish transfer. Failure to satisfy this rule is mixed/inconclusive evidence, unless consistent worsening is observed. Report all outcomes; do not promise definitive generalization. No inference of independent sample significance from correlated frames.

Preserve original experiments. Run resumably under a persistent user service, after >=20GiB free and <40% GPU utilization in three checks spaced 30 seconds apart. Write status/logs, cache, matched checkpoints, per-case records, report and plot within this directory. No external publication.

Prior work: https://openaccess.thecvf.com/content_ECCV_2018/papers/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.pdf
