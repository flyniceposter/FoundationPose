"""Development-only ridge selection for an explicitly limited orientation probe."""
import csv
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from study import HERE,DATA,atomic,digest
from analyze import load_primary,bootstrap

ALPHAS=[.01,.1,1,10,100,1000]


def project(matrix):
    u,_,v=np.linalg.svd(np.asarray(matrix).reshape(-1,3,3))
    d=np.ones((len(u),3));d[:,-1]=np.linalg.det(u@v)
    return (u*d[:,None,:])@v


def fit_predict(x,y,q,alpha):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    x=(x-mean)/scale;q=(q-mean)/scale;target_mean=y.mean(0)
    weights=np.linalg.solve(x@x.T+alpha*np.eye(len(x)),y-target_mean)
    return q@x.T@weights+target_mean


def predict(x,y,q):
    order=np.random.default_rng(42).permutation(len(x));folds=np.array_split(order,5)
    losses=[]
    for alpha in ALPHAS:
        err=[]
        for val in folds:
            train=np.setdiff1d(np.arange(len(x)),val)
            out=fit_predict(x[train],y[train],x[val],alpha)
            err.append(np.mean((out-y[val])**2))
        losses.append(np.mean(err))
    alpha=ALPHAS[int(np.argmin(losses))]
    return project(fit_predict(x,y,q,alpha)),alpha,float(min(losses))


def angles(pred,truth,symmetries):
    relatives=pred[:,None]@np.transpose(truth[:,None]@symmetries[None],(0,1,3,2))
    values=np.degrees(np.arccos(np.clip((np.trace(relatives,axis1=-2,axis2=-1)-1)/2,-1,1)))
    return values[:,0],values.min(1)


def main():
    assert np.allclose(project(np.eye(3))[0],np.eye(3))
    assert np.allclose(angles(np.eye(3)[None],np.eye(3)[None],np.eye(3)[None])[0],0)
    _,emb,meta,_=load_primary()
    out=HERE/'analysis';info=json.loads((DATA/'models/models_info.json').read_text())
    rotations=[]
    for r in meta.itertuples():
        gt=json.loads((DATA/f'lm_test_all/test/{r.obj_id:06d}/scene_gt.json').read_text())[str(r.im_id)]
        rotations.append(np.array(next(g for g in gt if g['obj_id']==r.obj_id)['cam_R_m2c']).reshape(3,3))
    rotations=np.stack(rotations);records=[];choices=[]
    for obj in sorted(meta.obj_id.unique()):
        obj_info=info[str(obj)]
        assert not obj_info.get('symmetries_continuous')
        syms=[np.eye(3)]+[np.array(s).reshape(4,4)[:3,:3] for s in obj_info.get('symmetries_discrete',[])]
        syms=project(np.stack(syms))
        train=((meta.obj_id==obj)&(meta.split=='development')).to_numpy()
        query=((meta.obj_id==obj)&(meta.split!='development')).to_numpy()
        y=rotations[train].reshape(-1,9);truth=rotations[query]
        constant=np.repeat(project(y.mean(0)),len(truth),axis=0)
        raw,corrected=angles(constant,truth,syms)
        for i,r in enumerate(meta[query].itertuples()):records.append(dict(obj_id=obj,im_id=r.im_id,variant='constant',stage=-1,control_seed=0,raw_deg=raw[i],symmetry_deg=corrected[i]))
        for variant,stages in emb.items():
            for stage,values in stages.items():
                x=values[train,0];q=values[query,0]
                for seed in [0,101,202,303]:
                    label=y if seed==0 else y[np.random.default_rng(seed+int(obj)).permutation(len(y))]
                    pred,alpha,cv=predict(x,label,q)
                    raw,corrected=angles(pred,truth,syms)
                    choices.append(dict(obj_id=obj,variant=variant,stage=int(stage[-1]),control_seed=seed,alpha=alpha,development_cv_entry_mse=cv))
                    for i,r in enumerate(meta[query].itertuples()):
                        records.append(dict(obj_id=obj,im_id=r.im_id,variant=variant if seed==0 else variant+'_shuffled',stage=int(stage[-1]),control_seed=seed,raw_deg=raw[i],symmetry_deg=corrected[i]))
    df=pd.DataFrame(records);assert np.isfinite(df[['raw_deg','symmetry_deg']].to_numpy()).all()
    df.to_csv(out/'orientation_probe_per_frame.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'orientation_probe_selection.csv',index=False)
    frame=df.groupby(['obj_id','im_id','variant','stage'],as_index=False)[['raw_deg','symmetry_deg']].mean()
    perobj=frame.groupby(['obj_id','variant','stage'],as_index=False)[['raw_deg','symmetry_deg']].mean()
    perobj.to_csv(out/'orientation_probe_per_object.csv',index=False)
    result=perobj.groupby(['variant','stage'],as_index=False)[['raw_deg','symmetry_deg']].mean()
    baseline=frame[frame.variant=='constant']
    comparisons=[]
    for (variant,stage),g in frame[frame.variant!='constant'].groupby(['variant','stage']):
        paired=g.merge(baseline,on=['obj_id','im_id'],suffixes=('_probe','_constant'),validate='one_to_one')
        for metric in ['raw_deg','symmetry_deg']:
            mean,lo,hi=bootstrap(paired[metric+'_probe']-paired[metric+'_constant'],paired.obj_id)
            comparisons.append(dict(variant=variant,stage=stage,metric=metric,probe_minus_constant_deg=mean,ci_low=lo,ci_high=hi))
    result.to_csv(out/'orientation_probe_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'orientation_probe_comparisons.csv',index=False)
    atomic(out/'orientation_probe_status.json',dict(state='complete',sources={str(p):digest(p) for p in [HERE/'probe.py',HERE/'PROBE_PROTOCOL.md']},
                                                  scope='Object-specific pooled-feature probe on GT-conditioned crops; not blind pose inference'))
    print(result.to_string(index=False))


if __name__=='__main__':main()
