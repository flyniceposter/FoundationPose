"""Build a measured-results report and a presentation with speaker notes."""
import html
import json
import os
import textwrap
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('MPLCONFIGDIR','/tmp/encoder-study-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from study import HERE,atomic,digest

A=HERE/'analysis'
PAPER='https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf'
CKA='https://proceedings.mlr.press/v97/kornblith19a.html'
SALIENCY='https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html'
CORRUPTION='https://arxiv.org/abs/1903.12261'


def table(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    coverage=json.loads((A/'coverage.json').read_text());audit=json.loads((HERE/'encoder_audit.json').read_text())
    assert json.loads((A/'status.json').read_text())['state']=='analysis_complete'
    summary=pd.read_csv(A/'primary_summary.csv');intervals=pd.read_csv(A/'primary_intervals.csv');ret=pd.read_csv(A/'retrieval.csv')
    sec=pd.read_csv(A/'secondary_summary.csv');cka=pd.read_csv(A/'cka_common_inputs.csv');rank=pd.read_csv(A/'effective_rank.csv')
    split=coverage['heldout_label'];held=summary[(summary.split==split)&(summary.stage==3)]
    def metric(kind,condition,which='cosine'):
        return float(held[(held.variant==kind+'_pretrained')&(held.condition==condition)][which].iloc[0])
    def ci(kind,condition):
        s=intervals[(intervals.variant==kind+'_pretrained')&(intervals.split==split)&(intervals.condition==condition)&(intervals.metric=='cosine')].iloc[0]
        return f'{s["mean"]:.4f} [{s.ci_low:.4f}, {s.ci_high:.4f}]'
    names=['brightness_0.5','blur_3','zero_rgb','zero_xyz','rgb_shift_5','xyz_shift_5','joint_shift_5','rgb_noise_0.1','depth_noise_5','depth_dropout_0.3','occlusion_0.3']
    result_table=pd.DataFrame([{'Condition':n,'Refinement, mean [95% CI]':ci('refine',n),'Ranking, mean [95% CI]':ci('score',n)} for n in names])
    retrieval_table=ret[(ret.stage==3)&(ret.condition=='clean')][['variant','object_retrieval_accuracy']].copy()
    retrieval_table['object_retrieval_accuracy']=retrieval_table.object_retrieval_accuracy.map(lambda v:f'{100*v:.2f}%')
    modality_table=pd.DataFrame([dict(Network=k,RGB_removed=f'{metric(k,"zero_rgb"):.4f}',XYZ_removed=f'{metric(k,"zero_xyz"):.4f}',Joint_shift_5px=f'{metric(k,"joint_shift_5"):.4f}',Depth_noise_5mm=f'{metric(k,"depth_noise_5"):.4f}') for k in ['refine','score']])
    final_cka=cka[(cka.kind=='refine_vs_score')&cka.stage.isin(['shared_3','fusion_4'])]
    cka_shared=float(final_cka[final_cka.stage=='shared_3'].linear_cka.iloc[0]);cka_fusion=float(final_cka[final_cka.stage=='fusion_4'].linear_cka.iloc[0])
    runtime_primary=coverage['primary_status']['elapsed_seconds']/60
    runtime_secondary=coverage['secondary_status']['elapsed_seconds']/60
    secondary_final=sec[(sec.family=='shared')&(sec.stage==3)&sec.condition.isin(['zero_rgb','zero_xyz','depth_noise_5','depth_dropout_0.3'])]
    bn_table=secondary_final.pivot(index=['kind','condition'],columns='variant',values='cosine').reset_index().round(4)
    spatial_pooled=pd.DataFrame([dict(kind=k,condition=n,spatial=f'{metric(k,n):.4f}',pooled=f'{metric(k,n,"pooled_cosine"):.4f}') for k in ['refine','score'] for n in ['joint_shift_5','depth_noise_5','zero_xyz']])
    arch_table=pd.DataFrame([dict(Network=k,Shared_CNN_parameters=f'{audit[k]["shared_parameters"]:,}',Fusion_CNN_parameters=f'{audit[k]["fusion_parameters"]:,}',Full_network_parameters=f'{audit[k]["total_parameters"]:,}',Crop_ratio=audit[k]['crop_ratio']) for k in ['refine','score']])
    probe=pd.read_csv(A/'orientation_probe_summary.csv')
    probe_rows=probe[(probe.stage==3)|(probe.variant=='constant')].copy()
    for column in ['raw_deg','symmetry_deg']:probe_rows[column]=probe_rows[column].map(lambda v:f'{v:.2f}')
    fig,ax=plt.subplots(figsize=(12,5))
    chosen=probe[(probe.stage==3)|(probe.variant=='constant')]
    ax.barh(chosen.variant.str.replace('_',' '),chosen.symmetry_deg,color=['#116b78' if 'pretrained' in v and 'shuffled' not in v else '#aab9bd' for v in chosen.variant])
    ax.set_xlabel('Object-macro mean symmetry-aware orientation error (degrees)')
    ax.set_title('Frozen pooled-feature orientation probe\n20 development / 20 held-out frames per object; object-specific ridge readout')
    fig.tight_layout();fig.savefig(A/'figures/orientation_probe.png',dpi=160);plt.close(fig)
    strongest={k:held[(held.variant==k+'_pretrained')&(held.condition!='clean')].sort_values('cosine',ascending=False).iloc[0] for k in ['refine','score']}
    takeaways=[f"The shared encoders load correctly and use RGB plus XYZ, with 684,480 shared-CNN parameters each.",
               f"The largest measured final-layer spatial change was {strongest['refine'].condition} for refinement ({strongest['refine'].cosine:.4f}) and {strongest['score'].condition} for ranking ({strongest['score'].cosine:.4f}).",
               f"Under identical clean inputs, cross-network linear CKA is {cka_shared:.3f} for the shared CNN and {cka_fusion:.3f} for the artificial self-pair fusion CNN. This describes representation geometry, not task accuracy."]
    def probe_error(variant):
        return float(probe[(probe.variant==variant)&((probe.stage==3)|(probe.variant=='constant'))].symmetry_deg.iloc[0])
    takeaways.append(f'On the object-specific orientation readout, final pretrained pooled features give {probe_error("refine_pretrained"):.2f}° (refinement) and {probe_error("score_pretrained"):.2f}° (ranking) mean symmetry-aware error, versus {probe_error("constant"):.2f}° for the constant baseline. Random-feature controls give {probe_error("refine_random"):.2f}° and {probe_error("score_random"):.2f}°. These are GT-conditioned diagnostic probes, not deployed pose errors.')
    perobj_probe=pd.read_csv(A/'orientation_probe_per_object.csv')
    perobj_probe=perobj_probe[(perobj_probe.stage==3)&~perobj_probe.variant.str.contains('shuffled')].pivot(index='obj_id',columns='variant',values='symmetry_deg').reset_index().round(2)
    wins={k:int((perobj_probe[k+'_pretrained']<perobj_probe[k+'_random']).sum()) for k in ['refine','score']}
    takeaways.append(f'For a joint 5-pixel shift, spatial versus pooled cosine distances are {metric("refine","joint_shift_5"):.4f} versus {metric("refine","joint_shift_5","pooled_cosine"):.4f} in refinement and {metric("score","joint_shift_5"):.4f} versus {metric("score","joint_shift_5","pooled_cosine"):.4f} in ranking. Pooling can hide spatial changes relevant to alignment.')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,k in zip(axs,['refine','score']):
        cond=['joint_shift_5','zero_xyz','depth_noise_5']
        x=np.arange(3)
        for offset,column,label,color in [(-.18,'cosine','Spatial feature map','#116b78'),(.18,'pooled_cosine','Globally pooled vector','#d47b30')]:
            values=[metric(k,n,column) for n in cond]
            bars=ax.bar(x+offset,values,width=.34,label=label,color=color)
            ax.bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=3,fontsize=9)
        ax.set_xticks(x,['Joint shift 5 px','XYZ removed','Depth noise 5 mm']);ax.set_ylim(0,.5);ax.set_ylabel('Cosine distance');ax.set_title(k.capitalize());ax.legend(fontsize=8)
    fig.suptitle('Pooling hides most of the measured shift response\nFinal pretrained shared block; 260 held-out diagnostic frames')
    fig.tight_layout();fig.savefig(A/'figures/spatial_vs_pooled.png',dpi=160);plt.close(fig)
    geometry=pd.read_csv(A/'secondary_representation_stats.csv')
    geometry=geometry[geometry.stage=='shared_3'].groupby(['kind','variant'],as_index=False)[['pooled_rms','centered_variation_fraction','effective_rank']].mean()
    scale_table=sec[(sec.family=='shared')&(sec.stage==3)&(sec.condition=='clean')][['kind','variant','rms']].merge(geometry,on=['kind','variant'])
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,k in zip(axs,['refine','score']):
        g=scale_table[scale_table.kind==k]
        ax.scatter(g.centered_variation_fraction,g.rms,s=70,color='#116b78')
        offsets={'pretrained':(6,-16),'random':(6,4),'random_conv_trained_bn':(6,10),'trained_conv_reset_bn':(6,-5)}
        for r in g.itertuples():ax.annotate(r.variant.replace('_',' '),(r.centered_variation_fraction,r.rms),fontsize=8,xytext=offsets[r.variant],textcoords='offset points')
        ax.set_yscale('log');ax.set_xlim(-.03,1.1);ax.set_xlabel('Centered pooled variation / total pooled norm');ax.set_ylabel('Clean activation RMS (log scale)');ax.set_title(k.capitalize())
    fig.suptitle('BatchNorm mismatches can create misleading feature stability');fig.tight_layout();fig.savefig(A/'figures/bn_scale.png',dpi=160);plt.close(fig)
    stamp=datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
    report=f'''# FoundationPose encoder diagnostics

Completed {stamp}. Scope: frozen CNN encoders in the installed FoundationPose refinement and ranking networks.

## Findings you can present

{chr(10).join('- '+s for s in takeaways)}

The contribution is an empirical encoder audit and controlled sensitivity study. It is not a new pose-estimation method, a retrained model, or an official benchmark result. The encoder measurements finished substantially faster than the original 8–12-hour estimate; no time was padded and no twelve-hour run is claimed.

## What the encoder does

The architecture uses the same CNN weights for the rendered and observed branch **within** each network. Their features are concatenated and processed by a second CNN. Refinement and ranking are separate networks with separate checkpoint weights. The [FoundationPose paper]({PAPER}) places these CNNs before the pose-update and candidate-comparison modules. All architecture counts and shapes below were checked directly against installed source and checkpoints.

Input: 6 × 160 × 160, consisting of RGB in [0,1] and normalized XYZ. The shared CNN produces 64 × 80 × 80, then 128 × 40 × 40, then two residual-block outputs of 128 × 40 × 40. Pair concatenation produces 256 × 40 × 40. The fusion CNN has two 256-channel residual blocks, a stride-2 convolution to 512 × 20 × 20, and two 512-channel residual blocks. The shared CNN's theoretical maximum receptive fields are 7, 11, 27, and 43 input pixels; the final fusion block reaches 147 pixels. These are architectural receptive fields, not attention or measured pixel importance.

{table(arch_table)}

![Encoder architecture](analysis/figures/architecture.png)

This resolves a practical source of confusion: the model constructors default to four channels, but the installed pretrained configurations explicitly request six. Using constructor defaults would not reproduce these checkpoints.

## Design and coverage

- Primary: 520 unique frames, 40 per object across 13 LINEMOD objects; 260 development and 260 held-out diagnostic frames. Both pretrained networks and one fixed random-weight control each. 23 named conditions, expanded to 39 condition/seed combinations by using three seeds for stochastic disturbances.
- Primary coverage: 2,080 frame/variant records, 81,120 perturbed-or-clean encoder inputs, and 324,480 stage/condition/seed records. These are repeated measurements of 520 frames, not independent data points.
- Supporting controls: 130 development frames (10 per object), 2 networks, 1 pretrained variant plus 3 initialization seeds for each of 3 controls. 2,600 frame/variant records; 31,200 encoder inputs; 280,800 stage/condition records across 4 shared and 5 fusion blocks.
- Control variants: random convolution plus reset BatchNorm; random convolution plus trained BatchNorm; trained convolution plus reset BatchNorm. The supporting study gives both networks identical crops and validity thresholds.
- Exact key coverage, stage coverage, finite values, and clean identity comparisons passed. CPU normalization agrees with the original normalization functions within {coverage['validation']['refine']['original_normalization_max_error']:.2g}. Random networks under common inputs agreed across both architectures.

Measurements used torch inference/eval mode on CPU. Full checkpoints were loaded strictly. Original class and helper-function bodies were extracted using AST to bypass GPU-specific imports; CNN definitions and forward logic were not rewritten. Source, configuration, checkpoint, dataset-image, and annotation hashes are recorded in the primary manifest.

## Input construction and interventions

Crop center is projected GT translation of the bounding-box-centered CAD mesh: t_centered = t_original + R × center_CAD. Size is derived from the metadata CAD diameter and crop ratio. Native ratios are 1.2 for refinement and 1.1 for ranking. RGB uses bilinear perspective warping and XYZ uses nearest-neighbor warping, matching the inspected crop interpolation settings. XYZ is camera-space backprojection, minus centered translation, divided by CAD radius. Invalid depth and individual XYZ components outside [-2,2] are zeroed following the installed normalization code. The source refinement path uses 0.001 m depth validity; ranking uses 0.1 m.

These diagnostics use raw dataset depth, metadata diameter, per-image intrinsics, and GT alignment. Production registration additionally filters depth and estimates diameter from sampled vertices. The study therefore tests controlled inputs to the encoder, not exact production replay. Native cross-network contrasts combine learned weights and preprocessing; within-network trained/random comparisons are paired on identical tensors. The common-input supporting experiment removes this preprocessing difference.

Brightness factors 0.5 and 1.5; Gaussian blur sigma 1 and 3 crop pixels; RGB Gaussian noise sigma 0.02 and 0.1 on [0,1]; Gaussian depth noise sigma 1, 5, and 10 mm at valid crop pixels; independent depth dropout 10%, 30%, 50%; RGB/XYZ removal; +2/+5 horizontal crop-pixel shifts of RGB, XYZ, or both; central square masks of approximately 10%/30% crop area. Shift fill and modality removal use zero in normalized input space. Occlusion is an artificial missing-data intervention, not physically rendered occlusion. These sensor-noise settings are not calibrated to a physical camera. Different perturbation magnitudes are not equal-severity measurements.

## Metrics and uncertainty

For clean activation F and disturbed activation G, spatial cosine distance is 1 − dot(vec(F),vec(G))/(norm(F) norm(G)). Relative L2 drift is norm(G−F)/norm(F). Pooled cosine distance computes cosine after global average pooling. Also recorded: activation zero fraction and RMS. A smaller drift means a more stable representation for this intervention; a constant, uninformative representation can also have low drift. Spatial shift sensitivity can reflect useful localization rather than a defect.

Seeds are averaged within each frame before aggregation. Per-object means are macro-averaged. Intervals use 2,000 paired object-stratified frame bootstrap samples with fixed seed; uncertainty is conditional on these objects, checkpoint weights, and selected frames. Frames from the same object sequence may be temporally correlated, so these intervals can be optimistic and are not unseen-object generalization intervals. The primary held-out partition was already used in previous diagnostics; it is not pristine test data. No model or condition was selected to optimize these held-out numbers. No multiple-comparison significance claims are made.

## Primary held-out results

Final shared-CNN spatial cosine distance, mean and 95% bootstrap interval. Distances are dimensionless and are **not** pose-error percentages. Complete results for all conditions, layers, splits, and objects are in the CSV tables.

{table(result_table)}

![Feature sensitivity](analysis/figures/feature_sensitivity.png)

![Layer profiles](analysis/figures/layer_profiles.png)

### Spatial versus pooled features

{table(spatial_pooled)}

![Spatial and pooled response](analysis/figures/spatial_vs_pooled.png)

Pooling discards spatial arrangement and may conceal changes that matter to pose estimation. That is why both metrics are retained. Do not replace the full pose-conditioned comparison with a pooled feature cosine based on these diagnostics alone.

## Learned weights and normalization controls

The primary random baseline changes both learned convolution weights and BatchNorm parameters/statistics. This is a useful architecture control but cannot isolate their causes. The supporting common-input study separates those components. Values below are final shared-block spatial distances, averaged across frames and control initializations.

{table(bn_table)}

Clean activation scale and pooled-feature spread help interpret these distances. The centered variation fraction is ||X − mean_frame(X)||_F / ||X||_F for pooled clean features across frames; control seeds are summarized by their mean. Values near zero indicate that sample-to-sample variation is small relative to the shared activation offset. This statistic is descriptive, not a task-performance score.

{table(scale_table.round(5))}

![Activation scale and variation](analysis/figures/bn_scale.png)

For example, mismatching random convolution weights with trained BatchNorm can yield very small disturbance drift alongside little relative variation across inputs. Conversely, resetting trained normalization can inflate activation scale. Neither is evidence of a better encoder. This is a concrete reason to inspect feature spread and controls alongside stability. The trained networks themselves retain their original normalization throughout the primary study.

![Normalization controls](analysis/figures/bn_controls_shared.png)

Hybrids combine parameters that were never trained together. They diagnose sensitivity to parameter groups; high drift or unusual activation scale in a hybrid is not evidence that the deployed encoder is broken. First-convolution RGB/XYZ weight norms are in weight_audit.csv; input scale and later layers prevent interpreting those norms directly as modality importance.

## Pair-fusion encoder

The fusion diagnostic fixes A to a clean observation and perturbs B. Its baseline is an artificial observation/observation self-pair. It measures how the actual fusion CNN propagates a disturbance, without needing a GPU renderer. It does not test real rendered/observed alignment. The final fusion features have 512 × 20 × 20 elements before positional embedding and task modules.

![Fusion controls](analysis/figures/bn_controls_fusion.png)

## A fixed feature-map illustration

![Activation example](analysis/figures/activation_example.png)

The first pre-existing development row, object 1/frame 958, was used without selecting for a favorable outcome. Per-location RMS across feature channels is shown with a fixed scale within each network. These maps visualize activation magnitude, not attention or causal importance. The two networks receive identical diagnostic inputs in this illustration.

## Representation geometry and object retrieval

Linear CKA on pooled clean features compares centered relationships between frames. Under common inputs, refinement-versus-ranking CKA is {cka_shared:.4f} at the last shared block and {cka_fusion:.4f} at the final artificial self-pair fusion block. CKA's use for representation comparison follows [Kornblith et al., ICML 2019]({CKA}); high CKA does not mean identical weights or identical task behavior. Effective ranks and all layer comparisons are included in the CSVs.

![Common-input CKA](analysis/figures/cka.png)

Nearest-neighbor object retrieval uses 260 development gallery frames and 260 held-out queries, cosine similarity of pooled features, no trained classifier, and object-macro accuracy. Clean final-block values:

{table(retrieval_table)}

![Object retrieval](analysis/figures/retrieval.png)

This is object identity among the same 13 objects in the same dataset, not pose estimation or recognition of unseen objects. Background, lighting, sequence similarity, and crop scale can support retrieval. Random features may also retain useful image similarity. The random controls and effective-rank tables are essential to interpretation. There is no independently established chance distribution; 1/13 is merely a uniform-label reference.

## Frozen-feature orientation probe

To test more than stability or object identity, an object-specific ridge readout predicts nine orientation-matrix entries from clean pooled features. Encoder weights remain frozen. Each object has 20 development training frames and 20 held-out frames. Five-fold development-only selection chooses alpha from [0.01, 0.1, 1, 10, 100, 1000]; standardization is fitted separately inside every training fold. Predicted matrices are projected to SO(3). Three label-permutation controls repeat the same selection procedure. Their errors are averaged per held-out frame.

The constant baseline is the projected development mean rotation for each object. Raw error is geodesic rotation angle; symmetry-aware error minimizes over identity and annotated discrete rotation symmetries. This orientation-only metric ignores symmetry translations and is not ADD/MSSD or official BOP evaluation. Complete per-stage/per-object results, selected alphas, per-frame errors, and paired bootstrap changes are in orientation_probe_*.csv. Final shared-block and constant-baseline values, in degrees:

{table(probe_rows)}

The advantage is not uniform: the pretrained final block beats its random-feature control on {wins['refine']}/13 objects for refinement and {wins['score']}/13 for ranking. At the first shared stage, the trained pooled-feature probe is worse than the random control on the object-macro average; the pretrained advantage appears later. This is conditional on this readout and input construction and does not establish superiority of every trained layer. Final per-object symmetry-aware errors:

{table(perobj_probe)}

![Orientation readout](analysis/figures/orientation_probe.png)

This measures decodability in GT-conditioned crops with known object identity, not blind pose accuracy or unseen-object generalization. GT affects centering, crop scale, and normalization; the rotated CAD center also influences crop alignment. Pooling discards information used by the actual spatial pair-comparison network. The first pooled stage has 64 channels and later shared stages have 128, so cross-stage comparisons also change readout dimension. Probe bootstrap intervals condition on fitted readouts and development frames; they do not resample and refit training sets. Primary random-feature probes use one initialization; the three-initialization supporting controls address feature sensitivity, not orientation-probe initialization uncertainty. Poor probe performance would not establish that the full encoder lacks pose information, and good performance would not validate the installed pose heads.

## What this establishes

The installed encoders can be loaded and probed reproducibly on CPU. They encode both RGB appearance and centered XYZ geometry, and their feature sensitivity can be quantified across layer depth and interventions. The study provides paired comparisons, normalization controls, and measured representation comparisons to support discussion with your professor.

It does not establish downstream ADD(-S), MSSD/MSPD, actual candidate ordering, rendered/observed matching, robustness to real sensor failure, training-set novelty, causal feature importance, or a better encoder architecture. The study does not modify pretrained weights or claim to improve the prior point-filter results.

## Recommended next experiment

Once CUDA is available, save real rendered/observed tensors at fixed candidate poses, validate exact production preprocessing, and repeat the encoder probes on those tensors. Start from a predefined grid of translation and symmetry-aware rotation perturbations around GT poses. Measure encoder drift alongside change in pose error after refinement and within-set ranking quality. Keep the rendered inputs, pose-conditioned crops, and candidate sets fixed for each controlled input intervention. Then test whether the feature diagnostics predict downstream failure on a separately selected object/sequence set. A correlation would still not show that a particular feature caused the failure.

## Relation to the earlier filtering study

The prior point-filter study changes initialization masks while retaining full-image RGB/depth inputs. This study applies interventions directly to normalized encoder inputs with fixed GT-aligned crops. Consequently, a five-pixel input shift here is a different experiment from a five-pixel initialization-mask shift. No cross-study numerical equivalence is claimed.

## Research context

The [FoundationPose paper]({PAPER}) motivates learned rendered/observed comparison and separate refinement/ranking modules. [Kornblith et al.]({CKA}) motivate representation comparison with CKA. [Hendrycks and Dietterich]({CORRUPTION}) motivate testing several common corruption families; the current RGB-XYZ interventions are a custom diagnostic suite, not ImageNet-C. [Adebayo et al.]({SALIENCY}) show why visually appealing explanations need model-dependent sanity checks; our random and normalization controls follow that general caution, without claiming to reproduce their saliency experiments.

## Execution and reproduction

Primary measurement wall time: {runtime_primary:.2f} minutes. Supporting measurement wall time: {runtime_secondary:.2f} minutes for its completion invocation, which reused 260 already completed supporting pilot frame/variant records; pilot time is separately logged. Runs overlapped, so their times must not be added to claim elapsed study duration. No twelve-hour experiment is claimed. Analysis and artifact preparation are separate from encoder measurement time.

Use /home/Sahil/miniforge3/envs/foundationpose/bin/python from this directory:

```bash
python validate.py
python audit.py
python study.py
python secondary.py
python probe.py
python analyze.py
python illustrate.py
python build_report.py
```

Existing matching checkpoints resume; source/configuration mismatches require a fresh output directory. Primary and secondary processes use independent advisory locks. Numerical validation must pass before interpreting results. Results reflect these installed checkpoints and Python/package versions. GPU validation is unperformed because CUDA was unavailable, not silently replaced with a benchmark claim.

Files: results/ and secondary_results/ contain checkpointed measurements and manifests; analysis/ contains complete tables and figures; validation.json, analysis_validation.json, and encoder_audit.json contain verification evidence. PRESENTATION.pdf and PRESENTATION.pptx contain the talk; SPEAKER_NOTES.md includes narration. PRESENTATION_QA.md covers likely questions. The PowerPoint file uses rendered slide images plus speaker notes, preserving appearance; the source text lives in build_report.py.
'''
    (HERE/'REPORT.md').write_text(report)
    (HERE/'ONE_PAGE_SUMMARY.md').write_text('# Encoder study: meeting brief\n\n'+'\n\n'.join(takeaways)+'\n\n'+table(modality_table)+'\n\n520 primary frames across 13 objects; paired random controls. Supporting common-input controls cover 130 development frames and both shared/fusion CNNs. These are feature diagnostics, not pose accuracy. Native preprocessing differs across networks; common-input controls are provided.\n\nSee REPORT.md and PRESENTATION.pdf for details.\n')
    slides=[]
    def add(title,bullets=None,image=None,notes='',subtitle=None):slides.append(dict(title=title,bullets=bullets or [],image=image,notes=notes,subtitle=subtitle))
    add('What does FoundationPose’s encoder encode?',[
        'A controlled study of pretrained RGB–XYZ CNN features',
        '13 LINEMOD objects · 520 primary frames · frozen checkpoints',
        'Architecture audit, disturbances, random controls, and pair-fusion probes'],notes='Open with the research question: before proposing an encoder replacement, understand its actual inputs, feature behavior, and what evidence would connect those features to pose quality. This study is diagnostic and does not claim a new method.',subtitle='Encoder research study · October 28, 2026')
    add('The research question',[
        'What enters the encoder, and what leaves each layer?',
        'How do appearance, geometry, missing data, and shifts change features?',
        'Which effects remain when learned weights and normalization are controlled?',
        'What can these measurements tell us—and what still needs pose validation?'],notes='Explain that feature stability alone is not success. A pose encoder needs to retain alignment information, so some sensitivity is useful. The goal is to characterize the network before attributing failure to it.')
    add('Six channels, two CNN stages',[
        'Input: RGB [0,1] + centered XYZ / CAD radius → 6 × 160 × 160',
        'Shared CNN: 64 × 80 × 80 → 128 × 40 × 40 → two residual blocks',
        'Concatenate the two branches → fusion CNN → 512 × 20 × 20',
        f'Shared CNN: {audit["refine"]["shared_parameters"]:,} parameters; same design in both networks',
        'Weights are shared across branches within a network; checkpoints differ across tasks'],notes='The constructors default to four channels, but both installed configs specify six. Separate refinement and ranking checkpoints use the same architecture. The output here is before positional embedding and task-specific attention/heads. Architectural receptive field reaches 43 input pixels in the shared CNN and 147 in the fusion CNN.')
    add('Where the CNN encoders sit',image='architecture.png',notes='The observed and rendered branches share the first CNN within each task network. Their concatenated features enter the second CNN. Our primary run probes the observed branch; the supporting fusion run deliberately uses observation/self-pairs and stops before the task heads.')
    add('A controlled dataset experiment',[
        '520 unique frames: 40 per object; existing 260 / 260 diagnostic split',
        '23 named conditions → 39 condition/seed combinations',
        'Both pretrained shared CNNs + matched random-weight controls',
        '81,120 encoder inputs; 324,480 layer records',
        'GT fixes crop alignment; this is not blind pose registration'],notes='The dataset objects are 1,2,4,5,6,8,9,10,11,12,13,14,15. The split has been used previously; disclose that it is not untouched benchmark data. Multiple measurements of a frame do not create independent samples. Native crop ratios differ between tasks.')
    add('What was changed—and measured',[
        'Appearance: brightness, blur, Gaussian RGB noise, RGB removal',
        'Geometry: 1–10 mm noise, 10–50% dropout, XYZ removal',
        'Alignment: 2 / 5 crop-pixel shifts; separate and joint modalities',
        'Metrics: spatial cosine, relative L2, pooled cosine, sparsity, RMS',
        'Seeds averaged per frame; object-macro means and paired bootstrap intervals'],notes='Noise is applied to valid crop depth before reconstruction of normalized XYZ. These are custom interventions, not calibrated physical sensor models. A shift moves input data, unlike the earlier study which shifted the initialization mask. Confidence intervals condition on these objects and can be optimistic under sequence correlation.')
    add('Implementation checks passed',[
        'Both full checkpoints loaded with exact parameter matching',
        f'CPU XYZ normalization versus source: max error ≈ {coverage["validation"]["refine"]["original_normalization_max_error"]:.1e}',
        'Clean identity, finite features, modality isolation, deterministic seeds',
        'Eval-mode batch checks; exact frame and stage coverage',
        'Common-input random architectures agree across tasks'],notes='Describe why this matters: wrong depth units, crop centers, or validity thresholds could produce convincing but invalid results. The source normalization was executed on CPU with only CUDA transfers replaced by no-ops for validation. The full study does not use production rendering or depth filtering.')
    add('Which disturbances change the features?',image='feature_sensitivity.png',notes='Read this as feature distance, not error rate. Bars compare pretrained and random encoders using each task’s native input construction. Intervals bootstrap frames within objects. Different interventions are not calibrated to equal severity. Refer to the report table for precise values.')
    add('Three concrete comparisons',[
        f'Refinement: remove XYZ {metric("refine","zero_xyz"):.3f}; remove RGB {metric("refine","zero_rgb"):.3f}',
        f'Ranking: remove XYZ {metric("score","zero_xyz"):.3f}; remove RGB {metric("score","zero_rgb"):.3f}',
        f'5 mm noise: refinement {metric("refine","depth_noise_5"):.4f}; ranking {metric("score","depth_noise_5"):.4f}',
        f'Joint 5 px shift: refinement {metric("refine","joint_shift_5"):.3f}; ranking {metric("score","joint_shift_5"):.3f}',
        'All values: final shared-layer spatial cosine distance'],notes='Removing XYZ entirely is a much stronger intervention than adding five millimetres of noise. These numbers quantify representation sensitivity; they do not prove which modality contributes more to task accuracy. Spatial sensitivity can be useful for alignment.')
    add('Sensitivity changes with depth',image='layer_profiles.png',notes='The two strided layers and two residual blocks do not have identical response profiles. Compare each line within a task and a metric. Do not read increasing feature distance as a monotonic decline in pose quality.')
    add('A pooled vector can hide spatial changes',image='spatial_vs_pooled.png',notes='Spatial cosine compares activations at their fixed grid positions. Global pooling discards that layout, so a translated feature map can have a similar pooled vector even when spatial distance is substantial. This supports retaining spatial probes for pose analysis; it does not prove a downstream pose effect or establish that all pooling is harmful.')
    add('A fixed example of feature responses',image='activation_example.png',notes='This is the first development row in the existing split, not a handpicked success. The bottom maps show activation RMS across channels with the same color scale within each network. They are not attention or saliency. RGB may remain visually unchanged when only geometry is disturbed.')
    add('Separate weights from BatchNorm',image='bn_controls_shared.png',notes='This supporting experiment uses the same crops and validity threshold for both networks, 130 development frames, and three initializations per control. Random-plus-trained-BatchNorm and trained-plus-reset-BatchNorm isolate parameter groups. Hybrids are deliberately out of distribution and are not deployable model replacements.')
    add('Low drift alone can be misleading',image='bn_scale.png',notes='The random-convolution/trained-BatchNorm hybrid can barely change even after a severe intervention because input-dependent variation is small relative to a common activation offset. Resetting trained BatchNorm can greatly inflate scale. Both are artificial mismatches, not deployed model failures. This shows why stable features need information and scale checks.')
    add('Probe the pair-fusion encoder',image='bn_controls_fusion.png',notes='Important disclosure: the clean reference branch is an observation, not a rendering. This artificial observation/self-pair test measures perturbation propagation through the actual fusion blocks. It does not establish rendered/observed matching or ranking ability.')
    add('Similar architecture, different representation geometry',image='cka.png',notes=f'Linear CKA on pooled common-input clean features is {cka_shared:.4f} at the final shared block and {cka_fusion:.4f} at the final self-pair fusion block. CKA compares centered relationships between samples and is not a pose metric. Cite Kornblith et al., ICML 2019. Pooling loses spatial information.')
    add('A representation sanity check: object retrieval',image='retrieval.png',notes='A clean development gallery has 260 samples; held-out queries have 260. Cosine nearest-neighbor retrieval predicts one of the same 13 object identities. No classifier is trained. Random features can also retrieve images, and background or sequence similarity can help. This is not unseen-object generalization or pose accuracy.')
    add('Is orientation decodable from pooled features?',image='orientation_probe.png',notes='A separate ridge readout is trained for each object from only 20 development frames. Five-fold development-only selection chooses regularization; every fold standardizes from its own training data. We project predictions to rotations, evaluate on held-out frames, and compare to a constant rotation and three shuffled-label controls. The encoders stay frozen. GT-conditioned crops, known object identity, and temporal correlation limit the result. This is not the installed pose estimator.')
    add('The learned-feature advantage has limits',[
        f'Final pretrained features beat random controls on {wins["refine"]}/13 objects (refinement)',
        f'Final pretrained features beat random controls on {wins["score"]}/13 objects (ranking)',
        'The first pooled stage is worse than random on the macro orientation probe',
        'Random features retain useful image information; learning is not the only explanation',
        'Interpret the complete layer profile and per-object table'],notes='These are descriptive comparisons on a reused diagnostic split. They do not establish significance, unseen-object transfer, or an optimal encoder layer. The earliest stage has 64 pooled channels while subsequent stages have 128, so cross-stage comparisons also change feature dimension. Within-stage trained/random comparisons match dimension.')
    add('What we can—and cannot—conclude',[
        'Established: actual architecture, valid loading, and measured feature responses',
        'Established: paired controls for weights, BatchNorm, layers, and common inputs',
        'Still unmeasured: real rendered/observed comparison and pose outcomes',
        'Limits: GT-aligned raw-depth crops, reused split, sequence correlation',
        'Feature drift is a diagnostic—not a failure label'],notes='This is the key scientific boundary. Explicitly say no ADD(-S), BOP AR, or speedup improvement is claimed. The work provides a reproducible basis for the next causal or downstream study. CUDA was unavailable; no GPU experiment is being implied.')
    add('The next encoder experiment',[
        'Capture real rendered / observed tensors at fixed candidate poses',
        'Repeat controlled interventions with exact production preprocessing',
        'Relate feature drift to refinement improvement and candidate ranking',
        'Evaluate on separately selected sequences or objects',
        'Use those results to justify any encoder adaptation or replacement'],notes='Keep the same candidate poses and crops within each controlled comparison. Include symmetry-aware errors and disclose GT use. A learned relationship between encoder drift and downstream failure should be validated on new sequences, not on the same data used to form the hypothesis.')
    add('Discussion and references',[
        'Wen et al. · FoundationPose · CVPR 2024',
        'Kornblith et al. · Similarity of Neural Network Representations Revisited · ICML 2019',
        'Hendrycks & Dietterich · Common Corruptions and Perturbations · ICLR 2019',
        'Adebayo et al. · Sanity Checks for Saliency Maps · NeurIPS 2018',
        'Full tables, code, hashes, limits, and reproduction steps: REPORT.md'],notes='References provide architecture context, CKA methodology, corruption-testing context, and motivation for model-dependent sanity checks. We did not reproduce ImageNet-C or saliency-map experiments. Suggested question for the professor: should the next step prioritize real-pair pose sensitivity, learned feature inspection, or encoder adaptation?')
    presentation(slides)
    qna='''# Questions your professor may ask

**Did you test the encoder or the whole pose estimator?** Both shared and fusion CNN encoders were probed. Full task heads, renderer, candidate ranking, and pose refinement accuracy were not evaluated in this study.

**Why did you use GT?** To hold alignment and crop construction fixed while changing the input. That isolates a feature response, but limits claims about blind registration.

**Does lower cosine distance mean a better encoder?** No. It may mean robustness, or loss of useful information. Spatial sensitivity may be needed for pose correction. We report retrieval and random controls, but downstream validation remains necessary.

**Why compare to random weights?** To see which patterns are present without learning. The first comparison includes BatchNorm changes; the supporting factorial controls separate convolution and normalization effects.

**Do weight norms tell us which modality matters?** No. Input scaling, normalization, and downstream computation prevent that interpretation. Removal experiments also represent extreme out-of-distribution inputs.

**Is the self-pair experiment real pose matching?** No. It is an explicit artificial control for the fusion CNN. A renderer-based experiment is required to test actual pose-conditioned matching.

**Why are orientation-probe errors much larger than earlier FoundationPose pose errors?** This readout uses globally pooled features from a single observation and only 20 training examples per object. The installed estimator compares rendered/observed spatial features, uses fusion and task heads, and refines candidates iteratively. These are different systems and evaluation protocols. The probe's degree errors must not be reported as FoundationPose's pose accuracy.

**Is this a new method or publishable performance improvement?** It is a reproducible diagnostic study that supports hypotheses and next experiments. It does not establish an improved method or benchmark result.

**Why did it not run for twelve hours?** That was an early effort estimate. Measured CPU experiments completed faster. Research quality comes from valid controls and interpretation, not running the same work longer.
'''
    (HERE/'PRESENTATION_QA.md').write_text(qna)
    atomic(HERE/'DELIVERABLES.json',dict(state='complete',completed_at=stamp,scope='CPU frozen CNN encoder diagnostics; no GPU pose claims',
                                     files={str(p.relative_to(HERE)):digest(p) for p in [HERE/'REPORT.md',HERE/'PRESENTATION.pdf',HERE/'PRESENTATION.pptx',HERE/'SPEAKER_NOTES.md',HERE/'encoder_audit.json',A/'coverage.json']},
                                     primary_wall_minutes=runtime_primary,secondary_completion_invocation_wall_minutes=runtime_secondary))
    print('Report and presentation complete:',HERE)


def presentation(slides):
    out=HERE/'slides';out.mkdir(exist_ok=True)
    notes=[]
    with PdfPages(HERE/'PRESENTATION.pdf') as pdf:
        for i,s in enumerate(slides,1):
            fig=plt.figure(figsize=(16,9),facecolor='#f5f3ec')
            fig.text(.055,.92,s['title'],fontsize=27,fontweight='bold',color='#173540',va='top')
            fig.text(.055,.045,'FOUNDATIONPOSE  /  ENCODER DIAGNOSTICS',fontsize=10,color='#48616a')
            fig.text(.945,.045,f'{i:02d} / {len(slides):02d}',fontsize=10,color='#48616a',ha='right')
            if s.get('image'):
                ax=fig.add_axes([.05,.12,.9,.72]);ax.imshow(plt.imread(A/'figures'/s['image']));ax.axis('off')
            else:
                if s.get('subtitle'):fig.text(.058,.79,s['subtitle'],fontsize=18,color='#b46527')
                top=.70 if s.get('subtitle') else .76
                gap=min(.135,.58/max(1,len(s['bullets'])))
                for j,b in enumerate(s['bullets']):
                    wrapped=textwrap.fill(b,width=86)
                    fig.text(.065,top-j*gap,'•',fontsize=24,color='#b46527',va='top')
                    fig.text(.095,top-j*gap,wrapped,fontsize=21,color='#173540',va='top',linespacing=1.4)
            pdf.savefig(fig,facecolor=fig.get_facecolor())
            fig.savefig(out/f'{i:02d}.png',dpi=120,facecolor=fig.get_facecolor())
            plt.close(fig)
            notes.append(f'## Slide {i}: {s["title"]}\n\n{s["notes"]}\n')
    (HERE/'SPEAKER_NOTES.md').write_text('# Speaker notes\n\nSuggested duration: 12–18 minutes plus questions.\n\n'+'\n'.join(notes))
    pages=''.join(f'<section class="slide" id="s{i}" style="display:{"block" if i==1 else "none"}"><img alt="{html.escape(s["title"])}" src="slides/{i:02d}.png"><details><summary>Speaker notes</summary><p>{html.escape(s["notes"])}</p></details></section>' for i,s in enumerate(slides,1))
    (HERE/'PRESENTATION.html').write_text('<!doctype html><meta charset="utf-8"><title>FoundationPose encoder study</title><style>body{margin:0;background:#142b34;color:#fff;font:18px system-ui}.slide{max-width:1440px;margin:auto}.slide img{width:100%;display:block}details{padding:16px 30px}nav{text-align:center;padding:10px}button{font:inherit;margin:0 12px;padding:6px 20px}</style>'+pages+'<nav><button onclick="step(-1)">Previous</button><span id="count"></span><button onclick="step(1)">Next</button></nav><script>let current=1;const total='+str(len(slides))+';function step(d){document.getElementById("s"+current).style.display="none";current=Math.max(1,Math.min(total,current+d));document.getElementById("s"+current).style.display="block";document.getElementById("count").textContent=current+" / "+total}document.addEventListener("keydown",e=>{if(e.key==="ArrowRight"||e.key===" "){e.preventDefault();step(1)}if(e.key==="ArrowLeft")step(-1)});step(0)</script>')
    from pptx_export import export
    export(HERE,slides)


if __name__=='__main__':main()
