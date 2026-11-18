"""Descriptive live and final reports from compact transactional unit summaries."""
import json
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('MPLCONFIGDIR','/tmp/encoder-gpu-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
METRICS=['refined_translation_mm','refined_rotation_deg','refined_adds_mm','improvement_fraction','selected_adds_mm','selected_gt','ranking_tie_count','ranking_spearman','ranking_spearman_defined','refine_spatial_error_spearman','refine_pooled_error_spearman','score_spatial_error_spearman','score_pooled_error_spearman']


def md_table(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for r in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(map(str,r))+' |')
    return '\n'.join(lines)


def report(out,final=False):
    out=Path(out);status=json.loads((out/'status.json').read_text())
    db=sqlite3.connect(out/'summary.sqlite');rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM units')];db.close()
    if not rows:return
    df=pd.DataFrame(rows);df['frame_key']=df.obj_id.astype(str)+'/'+df.im_id.astype(str);target=out/'analysis';target.mkdir(exist_ok=True)
    df.to_csv(target/'per_unit_summary.csv',index=False)
    frames=df.groupby(['obj_id','im_id','condition','ablation'],as_index=False)[METRICS].mean()
    obj=frames.groupby(['obj_id','condition','ablation'],as_index=False)[METRICS].mean()
    summary=obj.groupby(['condition','ablation'],as_index=False)[METRICS].mean()
    counts=df.groupby(['condition','ablation'],as_index=False).agg(units=('key','size'),objects=('obj_id','nunique'),frames=('frame_key','nunique'))
    summary=summary.merge(counts,on=['condition','ablation']);summary.to_csv(target/'summary.csv',index=False);obj.to_csv(target/'per_object.csv',index=False)
    pairs=[];baseline=df[df.ablation=='intact']
    for ablation in sorted(set(df.ablation)-{'intact'}):
        g=df[df.ablation==ablation].merge(baseline,on=['obj_id','im_id','condition','seed'],suffixes=('_ablated','_intact'),validate='one_to_one')
        if g.empty:continue
        for condition,q in g.groupby('condition'):
            result=dict(condition=condition,ablation=ablation,paired_units=len(q),paired_frames=len(q[['obj_id','im_id']].drop_duplicates()),objects=q.obj_id.nunique())
            for metric in ['refined_translation_mm','refined_rotation_deg','refined_adds_mm','selected_adds_mm']:
                q=q.copy();q['delta']=q[metric+'_ablated']-q[metric+'_intact']
                f=q.groupby(['obj_id','im_id']).delta.mean().reset_index();result[metric+'_delta']=float(f.groupby('obj_id').delta.mean().mean())
                if final:
                    rng=np.random.default_rng(20261028);draws=np.zeros(2000);labels=sorted(f.obj_id.unique())
                    for objid in labels:
                        v=f[f.obj_id==objid].delta.to_numpy();draws+=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)/len(labels)
                    result[metric+'_ci_low'],result[metric+'_ci_high']=np.quantile(draws,[.025,.975]).tolist()
            pairs.append(result)
    paired=pd.DataFrame(pairs);paired.to_csv(target/'paired_encoder_interventions.csv',index=False)
    coverage=df.groupby(['obj_id','condition','ablation']).size().rename('completed_units').reset_index();coverage.to_csv(target/'coverage.csv',index=False)
    figure=target/'encoder_effects.png';fig,axs=plt.subplots(1,2,figsize=(13,6))
    if not paired.empty:
        clean=paired[paired.condition=='clean']
        for ax,metric,title in zip(axs,['refined_rotation_deg_delta','selected_adds_mm_delta'],['One-step symmetry-aware rotation error change (°)','Selected candidate sampled ADD(-S) change (mm)']):
            ax.barh(clean.ablation.str.replace('_',' '),clean[metric],color='#116b78');ax.axvline(0,color='gray',lw=1);ax.set_xlabel(title)
    fig.suptitle('Encoder interventions relative to intact computation\nExact paired units only; positive error change is worse; GT-conditioned candidate sets')
    fig.tight_layout();fig.savefig(figure,dpi=160);plt.close(fig)
    display=summary[['condition','ablation','units','objects','refined_rotation_deg','selected_adds_mm']].round(4)
    headline='Final budget report' if final else 'Live partial report'
    text=f'''# GPU encoder investigation — {headline}

Updated {datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')}. Process state: **{status['state']}**.

Started: {status['started_at']}. Scheduled deadline: {status['deadline_at']}.
Completed units: {len(df):,}; recorded failed units: {status.get('failed_units',0)}. Distinct frames: {len(df[['obj_id','im_id']].drop_duplicates()):,}; objects represented: {df.obj_id.nunique()}; real pose candidates evaluated per unit: 37. Elapsed wall time at status update: {status['elapsed_seconds']/3600:.3f} hours.

This is the extension to the earlier CPU-only report. CUDA is accessible outside the sandbox; the earlier claim of a driver problem was incorrect. These new experiments use the installed renderer, source depth filtering, real rendered/observed inputs, frozen pretrained CNNs, and the actual refinement/ranking heads.

## What was measured

The queue changes input disturbances and then intervenes on observed RGB, XYZ, shared-CNN spatial structure, or fusion-CNN spatial structure. Replacing a map by its spatial mean preserves channel means but destroys its layout. The same initial 37 candidates and candidate ordering are retained across matched units. One-step refinement and ranking on the original candidate set are separate endpoints. Ranking includes a GT candidate and does not measure blind global registration.

Every unit has per-candidate known initial errors, refined errors, scores, feature distances, and poses in units/*.json. Compact per-unit summaries are transactionally stored in summary.sqlite. No pretrained weights are changed. Feature interventions are deliberately out of distribution; task degradation establishes sensitivity to those interventions, not a validated encoder replacement.

## Coverage and descriptive means

The tables reflect completed units, not assumed completion of the full oversized queue. They must not be compared as though every group covered identical frames. Summary means first average repeated units within a frame and then macro-average represented objects. Missing groups are not silently treated as completed successes. See analysis/coverage.csv for exact object/condition/intervention coverage. Ranking selection errors average over all exactly tied maximum-logit candidates, rather than letting the first GT candidate win a constant-score tie. Raw argmax indices remain in the raw records. Constant-score Spearman correlations are undefined, not perfect: ranking_spearman_defined reports their availability and ranking_tie_count reports degeneracy.

{md_table(display)}

## Exact paired effects

analysis/paired_encoder_interventions.csv compares each intervention with intact computation only on matching frame, condition, seed and candidate set. Different interventions can have different denominators. Positive differences mean larger errors. At final reporting, 2,000 within-object frame bootstrap resamples give conditional intervals; candidate/seed repeats are not counted as independent frames. These intervals do not model correlated sequences or retraining uncertainty, and no multiple-testing significance claim is made.

![Encoder effects](analysis/encoder_effects.png)

## Limitations

GT centers the local candidate neighborhood; initial errors are restricted to the declared grid. The prior 520-frame diagnostic split is reused and additional frames are from the same LM sequences. No new model selection is performed. Per-image camera intrinsics and source-generated mesh diameter are used. CAD geometry and annotations are fixed input information. Sampled ADD/ADI uses 512 deterministic CAD vertices and is not full-mesh ADD, official BOP AR, VSD, MSSD or MSPD. Rotation errors include raw and annotated discrete-symmetry-aware versions. Ranking quality is conditional on this favorable candidate set. Correlations between feature distance and pose error are descriptive; the separate feature-map interventions test the installed computation under artificial changes.

## Reproduction and next interpretation

Read ../PROTOCOL.md for the predeclared design. run_gpu.py resumes matching atomic checkpoints and refuses source mismatches; summary.sqlite supports live reporting. GPU execution requires running outside the restricted tool sandbox on this server. Source hashes are in manifest.json; object geometry hashes are in objects/*.json. Raw data are preserved in units/*.json. Live reports are partial and the final report is valid only with its explicit process state and coverage.

For Friday, present the paired effects with their frame counts, then relate them to spatial-versus-pooled feature distances. Do not present all low-drift features as robust, or these local tests as a global pose benchmark. The earlier CPU slides provide architecture and sanity-check context; this phase supplies real-pair/task evidence.
'''
    name='GPU_REPORT.md' if final else 'LIVE_REPORT.md';(out/name).write_text(text)
    if final:deck(out,status,summary,paired,figure)


def deck(out,status,summary,paired,figure):
    slides=[('Twelve-hour encoder investigation',[
        f'Run state: {status["state"]}',f'Completed units: {status["completed_units"]:,}; failures: {status["failed_units"]}',
        'Frozen FoundationPose encoders · real rendered/observed pairs',
        'Local GT-conditioned candidate sets; not a blind registration benchmark']),
        ('What makes this an encoder experiment',[
        'Capture all shared CNN stages and fusion-stage descriptors',
        'Keep candidates fixed; alter observed modalities or spatial feature structure',
        'Measure actual one-step refinement and within-set ranking behavior',
        'Compare only exact paired frames, seeds and conditions']),
        ('Candidate and input controls',[
        'GT + 36 translation/rotation perturbations per unit',
        'Clean, brightness, blur, raw-depth noise and dropout',
        'Upstream crop, render, depth filter, centering and normalization',
        'Observed branch changes preserve the rendered branch']),
        ('Spatial structure interventions',[
        'Observed RGB removed / XYZ removed',
        'Observed shared feature map pooled and broadcast',
        'Observed shared feature map zeroed or spatially shuffled',
        'Final fusion map spatially shuffled; weights remain frozen']),
        ('Measured paired encoder effects',None),
        ('Interpretation boundaries',[
        'Report exact coverage and paired denominators for every effect',
        'OOD interventions test sensitivity, not a deployable replacement model',
        'A GT candidate is included; this is a favorable local ranking setting',
        'Sampled geometry errors and conditional uncertainty are diagnostics']),
        ('Evidence to present on Friday',[
        'Architecture / input audit from the first phase',
        'Real-pair feature sensitivity and downstream paired effects',
        'Failure examples and per-object heterogeneity from raw checkpoints',
        'A precise next hypothesis rather than an unsupported benchmark claim'])]
    slide_dir=out/'slides';slide_dir.mkdir(exist_ok=True);export_slides=[]
    with PdfPages(out/'GPU_PRESENTATION.pdf') as pdf:
        for i,(title,bullets) in enumerate(slides,1):
            fig=plt.figure(figsize=(16,9),facecolor='#f5f3ec');fig.text(.06,.91,title,fontsize=28,fontweight='bold',color='#173540')
            if bullets is None:
                ax=fig.add_axes([.04,.13,.92,.68]);ax.imshow(plt.imread(figure));ax.axis('off')
            else:
                import textwrap
                for j,line in enumerate(bullets):fig.text(.075,.74-j*.14,'• '+textwrap.fill(line,85),fontsize=21,color='#173540',va='top')
            fig.text(.06,.045,'GPU ENCODER STUDY — '+status['state'],fontsize=10,color='#48616a');fig.text(.94,.045,str(i),ha='right',fontsize=10)
            pdf.savefig(fig);fig.savefig(slide_dir/f'{i:02d}.png',dpi=120);plt.close(fig)
            export_slides.append(dict(title=title,notes='See GPU_REPORT.md and analysis/paired_encoder_interventions.csv for exact coverage, uncertainty and scope.'))
    sys.path.insert(0,str(HERE.parent));from pptx_export import export
    # Exporter's expected image folder is out/slides.
    path=export(out,export_slides);path.rename(out/'GPU_PRESENTATION.pptx')


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--final',action='store_true');a=ap.parse_args();report(a.output,a.final)
