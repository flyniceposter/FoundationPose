"""Aggregate complete encoder measurements; refuse partial-coverage reports."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/encoder-study-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import HERE, SPLIT, conditions, atomic, digest

METRICS=['cosine','relative_l2','pooled_cosine','zero_fraction','rms']
COLORS={'refine_pretrained':'#116b78','score_pretrained':'#d47b30','refine_random':'#69b9c0','score_random':'#d7b896'}


def cka(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    x=x-x.mean(0);y=y-y.mean(0)
    return float(np.square(x.T@y).sum()/np.sqrt(np.square(x.T@x).sum()*np.square(y.T@y).sum()))


def effective_rank(x):
    x=x-x.mean(0)
    s=np.linalg.svd(x,compute_uv=False)**2
    p=s/max(s.sum(),1e-30);p=p[p>0]
    return float(np.exp(-(p*np.log(p)).sum()))


def bootstrap(values,objects,n=2000):
    values=np.asarray(values,dtype=np.float64);objects=np.asarray(objects)
    rng=np.random.default_rng(20261028);draws=np.zeros(n)
    labels=np.unique(objects)
    for obj in labels:
        v=values[objects==obj]
        draws+=v[rng.integers(0,len(v),size=(n,len(v)))].mean(1)/len(labels)
    return float(np.mean([values[objects==obj].mean() for obj in labels])),*np.quantile(draws,[.025,.975]).tolist()


def load_primary():
    root=HERE/'results'
    status=json.loads((root/'status.json').read_text())
    assert status['state']=='measurements_complete',status
    frames=list(csv.DictReader(SPLIT.open()));expected={(int(r['obj_id']),int(r['im_id'])) for r in frames}
    data=[];embeddings={};sample_rows=None
    for variant in COLORS:
        paths=sorted((root/variant).glob('*.json'))
        observed=set();e={f'stage_{s}':[] for s in range(4)};metadata=[]
        for p in paths:
            item=json.loads(p.read_text());r=item['frame'];obj,im=int(r['obj_id']),int(r['im_id'])
            assert (obj,im) not in observed;observed.add((obj,im))
            assert len(item['records'])==len(conditions())*4
            assert {(m['stage'],m['condition'],m['seed']) for m in item['records']}=={(s,n,k) for s in range(4) for n,k in conditions()}
            for m in item['records']:data.append(dict(variant=variant,obj_id=obj,im_id=im,split=r['split'],**m))
            with np.load(p.with_suffix('.npz')) as z:
                for stage in e:e[stage].append(z[stage])
            metadata.append(dict(obj_id=obj,im_id=im,split=r['split']))
        assert observed==expected,(variant,len(observed))
        embeddings[variant]={k:np.stack(v) for k,v in e.items()}
        if sample_rows is not None:assert sample_rows==metadata
        sample_rows=metadata
    df=pd.DataFrame(data)
    assert np.isfinite(df[METRICS].to_numpy()).all()
    assert df[df.condition=='clean'].relative_l2.max()<1e-4
    return df,embeddings,pd.DataFrame(sample_rows),status


def primary_summary(df,out):
    frame=df.groupby(['variant','split','obj_id','im_id','stage','condition'],as_index=False)[METRICS].mean()
    obj=frame.groupby(['variant','split','obj_id','stage','condition'],as_index=False)[METRICS].mean()
    summary=obj.groupby(['variant','split','stage','condition'],as_index=False)[METRICS].mean()
    summary.to_csv(out/'primary_summary.csv',index=False)
    obj.to_csv(out/'per_object_summary.csv',index=False)
    df.to_csv(out/'primary_measurements.csv',index=False)
    intervals=[]
    for keys,g in frame[frame.stage==3].groupby(['variant','split','condition']):
        for metric in ['cosine','relative_l2']:
            mean,lo,hi=bootstrap(g[metric],g.obj_id)
            intervals.append(dict(variant=keys[0],split=keys[1],condition=keys[2],metric=metric,mean=mean,ci_low=lo,ci_high=hi,frames=len(g)))
    pd.DataFrame(intervals).to_csv(out/'primary_intervals.csv',index=False)
    deltas=[]
    final=frame[frame.stage==3]
    for kind in ['refine','score']:
        a=final[final.variant==kind+'_pretrained'];b=final[final.variant==kind+'_random']
        paired=a.merge(b,on=['split','obj_id','im_id','stage','condition'],suffixes=('_trained','_random'),validate='one_to_one')
        for (split,condition),g in paired.groupby(['split','condition']):
            for metric in ['cosine','relative_l2']:
                mean,lo,hi=bootstrap(g[metric+'_trained']-g[metric+'_random'],g.obj_id)
                deltas.append(dict(kind=kind,split=split,condition=condition,metric=metric,trained_minus_random=mean,ci_low=lo,ci_high=hi))
    pd.DataFrame(deltas).to_csv(out/'paired_trained_random.csv',index=False)
    return frame,summary,pd.DataFrame(intervals)


def retrieval(emb,meta,out):
    gallery=meta.split.eq('development').to_numpy();query=~gallery
    labels=meta.obj_id.to_numpy();records=[];ranks=[]
    specs=conditions()
    for variant,stages in emb.items():
        for stage,x in stages.items():
            clean=x[:,0,:];clean=clean/np.linalg.norm(clean,axis=1,keepdims=True).clip(1e-12)
            g=clean[gallery]
            for name in dict(specs):
                indices=[i for i,(n,s) in enumerate(specs) if n==name]
                accuracies=[]
                for i in indices:
                    q=x[query,i,:];q=q/np.linalg.norm(q,axis=1,keepdims=True).clip(1e-12)
                    sim=q@g.T;pred=labels[gallery][sim.argmax(1)];truth=labels[query]
                    accuracies.append(np.mean([np.mean(pred[truth==o]==o) for o in np.unique(truth)]))
                records.append(dict(variant=variant,stage=int(stage[-1]),condition=name,object_retrieval_accuracy=float(np.mean(accuracies)),seed_min=float(min(accuracies)),seed_max=float(max(accuracies)),query_frames=int(query.sum()),gallery_frames=int(gallery.sum())))
            centered=x[:,0,:]-x[:,0,:].mean(0)
            ranks.append(dict(variant=variant,stage=int(stage[-1]),effective_rank=effective_rank(x[:,0,:]),channels=x.shape[-1],median_embedding_norm=float(np.median(np.linalg.norm(x[:,0,:],axis=1))),centered_rms=float(np.sqrt(np.mean(centered**2)))))
    pd.DataFrame(records).to_csv(out/'retrieval.csv',index=False)
    pd.DataFrame(ranks).to_csv(out/'effective_rank.csv',index=False)
    comparisons=[]
    for va,stages in emb.items():
        for vb,stages_b in emb.items():
            for sa,a in stages.items():
                for sb,b in stages_b.items():
                    comparisons.append(dict(variant_a=va,variant_b=vb,stage_a=int(sa[-1]),stage_b=int(sb[-1]),linear_cka=cka(a[:,0],b[:,0])))
    pd.DataFrame(comparisons).to_csv(out/'cka_native_inputs.csv',index=False)
    return pd.DataFrame(records),pd.DataFrame(ranks)


def secondary(out):
    root=HERE/'secondary_results';status=json.loads((root/'status.json').read_text())
    assert status['state']=='measurements_complete',status
    manifest=json.loads((root/'manifest.json').read_text());expected={(int(r['obj_id']),int(r['im_id'])) for r in manifest['frames']}
    records=[];emb={};weight_audit=[]
    from secondary import variants,setup
    for kind in ['refine','score']:
        for variant,seed in variants():
            folder=root/f'{kind}_{variant}_{seed}';paths=sorted(folder.glob('*.json'));observed=set()
            pooled={};metadata=[]
            for p in paths:
                item=json.loads(p.read_text());r=item['frame'];obj,im=int(r['obj_id']),int(r['im_id']);observed.add((obj,im))
                assert len(item['records'])==len(manifest['conditions'])*9
                assert {(m['family'],m['stage'],m['condition']) for m in item['records']}=={(family,stage,name) for family,count in [('shared',4),('fusion',5)] for stage in range(count) for name in manifest['conditions']}
                for m in item['records']:records.append(dict(kind=kind,variant=variant,seed=seed,obj_id=obj,im_id=im,**m))
                with np.load(p.with_suffix('.npz')) as z:
                    for k in z.files:pooled.setdefault(k,[]).append(z[k][0])
                metadata.append((obj,im))
            assert observed==expected and len(paths)==len(expected)
            emb[(kind,variant,seed)]={k:np.stack(v) for k,v in pooled.items()}
            model,_=setup(kind,variant,seed);encoder=model.encodeA if kind=='refine' else model.encoderA
            w=encoder[0].net[0].weight.detach().numpy()
            bn=encoder[0].net[1]
            weight_audit.append(dict(kind=kind,variant=variant,seed=seed,first_conv_rgb_norm=float(np.linalg.norm(w[:,:3])),first_conv_xyz_norm=float(np.linalg.norm(w[:,3:])),bn_scale_mean=float(bn.weight.detach().mean()),bn_running_var_median=float(bn.running_var.median())))
    df=pd.DataFrame(records);assert np.isfinite(df[METRICS].to_numpy()).all()
    df.to_csv(out/'secondary_measurements.csv',index=False)
    frame=df.groupby(['kind','variant','obj_id','im_id','family','stage','condition'],as_index=False)[METRICS].mean()
    perobj=frame.groupby(['kind','variant','obj_id','family','stage','condition'],as_index=False)[METRICS].mean()
    summary=perobj.groupby(['kind','variant','family','stage','condition'],as_index=False)[METRICS].mean()
    summary.to_csv(out/'secondary_summary.csv',index=False)
    perobj.to_csv(out/'secondary_per_object.csv',index=False)
    seeds=df.groupby(['kind','variant','seed','family','stage','condition'],as_index=False)[METRICS].mean()
    seeds.to_csv(out/'secondary_seed_summary.csv',index=False)
    pd.DataFrame(weight_audit).to_csv(out/'weight_audit.csv',index=False)
    comparison=[]
    for kind in ['refine','score']:
        for family,stage in [('shared',3),('fusion',4)]:
            a=frame[(frame.kind==kind)&(frame.variant=='pretrained')&(frame.family==family)&(frame.stage==stage)]
            for variant in ['random','random_conv_trained_bn','trained_conv_reset_bn']:
                b=frame[(frame.kind==kind)&(frame.variant==variant)&(frame.family==family)&(frame.stage==stage)]
                paired=a.merge(b,on=['kind','obj_id','im_id','family','stage','condition'],suffixes=('_trained','_control'),validate='one_to_one')
                for condition,g in paired.groupby('condition'):
                    mean,lo,hi=bootstrap(g.cosine_trained-g.cosine_control,g.obj_id)
                    comparison.append(dict(kind=kind,family=family,condition=condition,control=variant,trained_minus_control=mean,ci_low=lo,ci_high=hi,frames=len(g)))
    pd.DataFrame(comparison).to_csv(out/'secondary_paired_differences.csv',index=False)
    ckas=[]
    for kind in ['refine','score']:
        a=emb[(kind,'pretrained',0)]
        for (k,v,s),b in emb.items():
            if k!=kind:continue
            for stage in a:ckas.append(dict(kind=kind,control=v,seed=s,stage=stage,linear_cka=cka(a[stage],b[stage])))
    a=emb[('refine','pretrained',0)];b=emb[('score','pretrained',0)]
    for stage in a:ckas.append(dict(kind='refine_vs_score',control='pretrained',seed=0,stage=stage,linear_cka=cka(a[stage],b[stage])))
    # Identical random networks and identical common inputs must match exactly.
    for seed in [101,202,303]:
        for stage in a:
            assert np.allclose(emb[('refine','random',seed)][stage],emb[('score','random',seed)][stage],rtol=1e-5,atol=1e-6)
    pd.DataFrame(ckas).to_csv(out/'cka_common_inputs.csv',index=False)
    geometry=[]
    for (kind,variant,seed),stages in emb.items():
        for stage,x in stages.items():
            centered=x-x.mean(0)
            geometry.append(dict(kind=kind,variant=variant,seed=seed,stage=stage,
                                 effective_rank=effective_rank(x),pooled_rms=float(np.sqrt(np.mean(x**2))),
                                 centered_variation_fraction=float(np.linalg.norm(centered)/max(np.linalg.norm(x),1e-12))))
    pd.DataFrame(geometry).to_csv(out/'secondary_representation_stats.csv',index=False)
    return summary,status


def plots(summary,intervals,ret,ranks,sec,out):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    split=next(s for s in summary.split.unique() if s!='development')
    selected=['brightness_0.5','blur_3','zero_rgb','zero_xyz','rgb_shift_5','xyz_shift_5','joint_shift_5','rgb_noise_0.1','depth_noise_5','depth_dropout_0.3','occlusion_0.3']
    labels=['Brightness ×0.5','Blur σ=3 px','RGB removed','XYZ removed','RGB shift 5 px','XYZ shift 5 px','Joint shift 5 px','RGB noise σ=0.1','Depth noise σ=5 mm','Depth dropout 30%','Central occlusion 30%']
    fig,axs=plt.subplots(1,2,figsize=(14,7),sharey=True)
    for ax,kind in zip(axs,['refine','score']):
        for offset,v in [(-.18,kind+'_pretrained'),(.18,kind+'_random')]:
            s=intervals[(intervals.variant==v)&(intervals.split==split)&(intervals.metric=='cosine')].set_index('condition').loc[selected]
            ax.barh(np.arange(len(selected))+offset,s['mean'],height=.33,color=COLORS[v],label=v.split('_')[-1])
            ax.errorbar(s['mean'],np.arange(len(selected))+offset,xerr=np.array([s['mean']-s.ci_low,s.ci_high-s['mean']]),fmt='none',ecolor='#333333',capsize=2,lw=.7)
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('Spatial cosine distance from clean features');ax.set_title(kind.capitalize()+' shared encoder, final block');ax.legend()
    fig.suptitle('Feature sensitivity on 260 held-out diagnostic frames\nFrame-stratified bootstrap intervals; native preprocessing per network',fontsize=14)
    fig.tight_layout();fig.savefig(figdir/'feature_sensitivity.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,kind in zip(axs,['refine','score']):
        s=summary[(summary.variant==kind+'_pretrained')&(summary.split==split)]
        for name in ['zero_rgb','zero_xyz','depth_noise_5','depth_dropout_0.3','rgb_shift_5','joint_shift_5']:
            d=s[s.condition==name].sort_values('stage');ax.plot(d.stage+1,d.cosine,'o-',label=name.replace('_',' '))
        ax.set_xticks([1,2,3,4]);ax.set_xlabel('Shared-encoder stage');ax.set_title(kind.capitalize());ax.set_ylabel('Spatial cosine distance');ax.legend(fontsize=8)
    fig.suptitle('Sensitivity evolves through the shared CNN');fig.tight_layout();fig.savefig(figdir/'layer_profiles.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    for v in COLORS:
        d=ret[(ret.variant==v)&(ret.condition=='clean')];ax.plot(d.stage+1,d.object_retrieval_accuracy*100,'o-',color=COLORS[v],label=v.replace('_',' '))
    ax.axhline(100/13,color='gray',ls='--',label='Uniform-label chance (7.69%)');ax.set_xticks([1,2,3,4]);ax.set_ylim(0,105);ax.set_xlabel('Shared-encoder stage');ax.set_ylabel('Nearest-neighbor object retrieval (%)');ax.legend(fontsize=9);ax.set_title('Clean held-out queries → development gallery; descriptive object diagnostic')
    fig.tight_layout();fig.savefig(figdir/'retrieval.png',dpi=160);plt.close(fig)
    for family,last in [('shared',3),('fusion',4)]:
        fig,axs=plt.subplots(1,2,figsize=(14,6),sharey=True)
        names=['zero_rgb','zero_xyz','depth_noise_5','depth_dropout_0.3','rgb_shift_5','joint_shift_5']
        for ax,kind in zip(axs,['refine','score']):
            for j,(v,color) in enumerate(zip(['pretrained','random','random_conv_trained_bn','trained_conv_reset_bn'],['#116b78','#a7bbc3','#d47b30','#8566a4'])):
                s=sec[(sec.kind==kind)&(sec.variant==v)&(sec.family==family)&(sec.stage==last)].set_index('condition').loc[names]
                ax.barh(np.arange(len(names))+(j-1.5)*.19,s.cosine,height=.18,color=color,label=v.replace('_',' '))
            ax.set_yticks(range(len(names)),[n.replace('_',' ') for n in names]);ax.invert_yaxis();ax.set_xlabel('Spatial cosine distance');ax.set_title(kind.capitalize());ax.legend(fontsize=8)
        fig.suptitle(('Common-input shared encoder' if family=='shared' else 'Artificial observation/self-pair fusion encoder')+'\n130 development frames; three initializations per control, averaged per frame')
        fig.tight_layout();fig.savefig(figdir/f'bn_controls_{family}.png',dpi=160);plt.close(fig)
    c=pd.read_csv(out/'cka_common_inputs.csv');c=c[c.kind=='refine_vs_score']
    fig,ax=plt.subplots(figsize=(11,4));ax.bar(c.stage,c.linear_cka,color='#116b78');ax.set_ylim(0,1);ax.set_ylabel('Linear CKA');ax.tick_params(axis='x',rotation=35);ax.set_title('Refinement vs ranking representations on identical clean inputs\nPooled activations; CKA measures representation geometry, not pose quality')
    fig.tight_layout();fig.savefig(figdir/'cka.png',dpi=160);plt.close(fig)
    return split


def main():
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    df,emb,meta,status=load_primary()
    frame,summary,intervals=primary_summary(df,out)
    ret,ranks=retrieval(emb,meta,out)
    sec,secondary_status=secondary(out)
    split=plots(summary,intervals,ret,ranks,sec,out)
    atomic(out/'coverage.json',dict(primary_frame_variants=len(meta)*4,primary_records=len(df),primary_status=status,
                                   secondary_status=secondary_status,validation=json.loads((HERE/'validation.json').read_text()),heldout_label=split,
                                   primary_finite=True,primary_clean_identity=True,common_input_random_network_identity=True,
                                   sources={str(p):digest(p) for p in [Path(__file__),HERE/'results/manifest.json',HERE/'secondary_results/manifest.json']}))
    atomic(out/'status.json',dict(state='analysis_complete'))
    print('Analysis complete:',out)


if __name__=='__main__':main()
