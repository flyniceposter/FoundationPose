"""Decision sensitivity analysis, NOT simulated pose-estimation performance.

Scores are explicit analyst judgments on 1..5 scales, not learned estimates.
Ranking frequency measures sensitivity to these assumptions, not success odds.
Run: python3 screen.py
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRITERIA = ['team_value', 'local_evidence', 'three_month_feasibility',
            'clean_attribution', 'novelty_potential']
# Each row: question, five subjective scores, main limitation.
ROWS = [
('Pose-sensitive corruption-robust shared-CNN adapters with frozen heads',5,4,4,5,3,'Benefit over matched ordinary fine-tuning is unproven; adaptation is established prior art.'),
('Ordinary shared-CNN fine-tuning with realistic corruption augmentation',5,4,5,5,1,'Essential baseline; likely engineering contribution rather than method novelty.'),
('Visibility-weighted dense correspondence supervision for shared features',5,4,3,4,3,'Correspondence supervision and geometry-guided refinement already exist.'),
('Depth-validity-conditioned RGB/XYZ feature gating',5,3,4,4,3,'Must beat input validity masking and modality dropout at matched budget.'),
('Anti-aliased downsampling for crop-shift stability',4,4,4,5,2,'Image shifts can change pose evidence; require coordinate-consistent targets.'),
('Multi-scale shared-feature fusion for fine pose details',4,4,3,4,2,'Adds latency and changes head interface unless carefully constrained.'),
('Separate RGB and XYZ stems with late fusion',5,3,3,4,2,'More parameters confound architecture with capacity.'),
('Hard-negative pose-pair mining for encoder training',4,3,4,4,2,'FoundationPose already uses pose-conditioned contrastive ranking.'),
('Symmetry-aware encoder supervision',4,3,4,4,2,'Well-established idea; labels need complete symmetry handling.'),
('Joint refinement/ranking shared encoder',4,3,3,3,3,'Negative transfer and multi-task balancing increase scope.'),
('DINO feature distillation into existing CNN',4,2,3,3,2,'Semantic teacher may suppress fine pose distinctions.'),
('Replace CNN with a pretrained vision transformer',4,2,2,2,1,'Large compute/interface changes obscure why performance changes.'),
('Rotation-equivariant 3D encoder replacement',4,2,2,3,3,'2D projected RGBD is not globally SO(3)-equivariant under occlusion.'),
('SE(3)-equivariant point-cloud side branch',4,2,2,3,3,'Integration, missing depth and runtime risk.'),
('Synthetic-to-real feature alignment',5,3,3,3,1,'Crowded literature; generic alignment can remove useful geometry.'),
('Cross-view contrastive feature adaptation',4,3,4,4,1,'Directly overlaps published ICIP 2025 work on FoundPose.'),
('Occlusion-aware feature training',5,3,4,4,2,'Need realistic occluders and visibility-conditioned supervision.'),
('Mask-boundary robustness through encoder training',5,3,4,4,2,'Crop generation and preprocessing may dominate any encoder change.'),
('Calibrated pose-quality prediction from encoder features',4,3,4,4,2,'Useful diagnostic but not directly better encoder representations.'),
('Encoder-driven adaptive refinement stopping',3,2,4,3,2,'Optimizes scheduling more directly than encoder quality.'),
('Encoder-driven candidate pruning',4,2,4,3,2,'May lose rare correct hypotheses; not primarily encoder improvement.'),
('Temporal feature consistency for tracking',4,2,3,3,2,'Temporal losses can suppress real fast motion.'),
('Self-supervised online encoder adaptation',5,2,2,2,3,'Drift, false pseudo-labels and reproducibility risk.'),
('Geometry-filtered pseudo-label training',4,2,3,3,2,'Confirmation bias and additional verification pipeline.'),
('Object-coordinate auxiliary prediction',4,3,3,4,2,'Canonical coordinates are ambiguous for symmetric objects.'),
('Surface-normal auxiliary prediction',3,2,4,4,1,'Auxiliary accuracy may not transfer to final pose accuracy.'),
('Pooled orientation probe improvement',2,4,5,5,1,'Can improve a probe without helping the actual pose estimator.'),
('Spatial rather than pooled orientation probe',3,4,5,5,2,'Valuable diagnostic, limited direct team deployment benefit.'),
('Feature-rank/channel pruning',3,2,4,4,1,'Speed contribution; may weaken geometric information.'),
('Quantization-aware encoder training',3,2,4,4,1,'Deployment engineering; depends on actual latency bottleneck.'),
('Teacher-student encoder compression',3,2,3,4,1,'Need measured team latency requirement.'),
('Learned depth denoising before CNN',4,2,3,3,1,'Can erase sharp geometry; not an encoder-only intervention.'),
('Sensor-noise calibrated training distribution',5,3,4,4,2,'Needs team sensor captures to establish relevant corruptions.'),
('Wide-angle refinement curriculum',5,3,2,3,1,'WAPR already directly addresses this; large training scope.'),
('Feature Jacobian regularization for local pose updates',4,3,2,4,4,'Interesting but second-order computation and objective design risk.'),
('Training-data selection by pose/visibility coverage',4,3,4,4,2,'Data contribution; must compare equal training/rendering budgets.'),
]
SCENARIOS = {
    'balanced_team_delivery': [0.30,0.20,0.25,0.15,0.10],
    'engineering_delivery': [0.40,0.15,0.30,0.10,0.05],
    'publication_oriented': [0.20,0.10,0.20,0.20,0.30],
    'limited_compute': [0.25,0.15,0.40,0.15,0.05],
}


def main():
    rng = random.Random(20261106)
    rows = [dict(question=r[0], **dict(zip(CRITERIA,r[1:6])), limitation=r[6]) for r in ROWS]
    outputs = []
    for scenario, center in SCENARIOS.items():
        wins = [0]*len(rows); tops = [0]*len(rows); rank_sum = [0]*len(rows)
        n = 10000
        for _ in range(n):
            # Perturb priority weights and ALL subjective ratings. A single
            # draw is a hypothetical judgment scenario, not a training run.
            weights = [rng.gammavariate(v*40,1) for v in center]
            total = sum(weights); weights = [v/total for v in weights]
            utilities = [sum(w*max(1,min(5,rng.triangular(row[c]-1,row[c]+1,row[c])))
                             for c,w in zip(CRITERIA,weights)) for row in rows]
            order = sorted(range(len(rows)), key=lambda i:utilities[i], reverse=True)
            wins[order[0]] += 1
            for rank,i in enumerate(order,1):
                rank_sum[i] += rank
                tops[i] += rank<=5
        for i,row in enumerate(rows):
            outputs.append(dict(scenario=scenario,question=row['question'],
                nominal_utility=sum(w*row[c] for w,c in zip(center,CRITERIA)),
                mean_rank=rank_sum[i]/n, top1_fraction=wins[i]/n,top5_fraction=tops[i]/n))
    for name,data in [('questions.csv',rows),('decision_sensitivity.csv',outputs)]:
        with (HERE/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    paired=HERE.parent/'encoder_study_20261028/gpu_phase2/production/analysis/paired_encoder_interventions.csv'
    evidence=[r for r in csv.DictReader(paired.open()) if r['condition']=='clean']
    (HERE/'local_evidence.json').write_text(json.dumps(dict(
        source=str(paired),note='Existing recorded outputs; not independently rerun inference.',
        clean_paired_effects=evidence),indent=2)+'\n')
    (HERE/'simulation_metadata.json').write_text(json.dumps(dict(
        seed=20261106,questions=len(rows),scenarios=SCENARIOS,
        draws_per_scenario=10000,total_draws=40000,
        meaning='Sensitivity of analyst priorities/ratings; NOT scientific effect prediction or success probability.',
        scores='Subjective 1..5; see questions.csv. Scores independently perturbed +/-1 with clipping.',
        limitations='Assumed ratings are correlated in reality; numerical precision is not evidential confidence.'),indent=2)+'\n')
    for scenario in SCENARIOS:
        print(scenario)
        for r in sorted((r for r in outputs if r['scenario']==scenario),key=lambda r:r['mean_rank'])[:5]:
            print(f"  {r['mean_rank']:.2f} mean rank; {r['top5_fraction']:.1%} top-five frequency: {r['question']}")


if __name__=='__main__':
    main()
