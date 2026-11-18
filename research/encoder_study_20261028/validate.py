"""Check diagnostic preprocessing against the original functions, on CPU."""
import ast
import copy
import csv
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import kornia
from study import HERE, ROOT, SPLIT, load_model, frame_inputs, make_input, measure, conditions, atomic


def source_function(path, name, cls=None):
    tree=ast.parse(path.read_text())
    nodes=tree.body
    if cls: nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls).body
    node=copy.deepcopy(next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name))
    node.returns=None
    for a in node.args.args: a.annotation=None
    ns=dict(torch=torch,np=np,kornia=kornia)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]


def run():
    torch.set_num_threads(2)
    row=next(csv.DictReader(SPLIT.open()))
    results={}
    # Original normalization code executes unmodified except .cuda() is a CPU no-op.
    for kind, parent in [('refine','PairH5Dataset'),('score','TripletH5Dataset')]:
        model,cfg=load_model(kind,True)
        base=frame_inputs(row,cfg,kind)
        actual=make_input(base,'clean',0,kind,row)
        rgb,depth,rays,t,radius=base
        raw_xyz=torch.from_numpy(rays*depth[None])[None]
        batch=SimpleNamespace(rgbAs=torch.zeros(1,3,160,160),poseA=torch.eye(4)[None],Ks=torch.eye(3)[None],
                              tf_to_crops=torch.eye(3)[None],mesh_diameters=torch.tensor([2*radius]),
                              xyz_mapAs=raw_xyz.clone(),xyz_mapBs=raw_xyz.clone())
        batch.poseA[0,:3,3]=torch.from_numpy(t)
        fn=source_function(ROOT/'learning/datasets/h5_dataset.py','transform_depth_to_xyzmap',parent)
        with patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self):
            result=fn(SimpleNamespace(cfg=cfg),batch,160,160)
        err=float((result.xyz_mapBs[0]-actual[3:]).abs().max())
        assert err<2e-6,(kind,err)
        encoder=model.encodeA if kind=='refine' else model.encoderA
        with torch.inference_mode():
            solo=encoder(actual[None]); paired=encoder(torch.stack([actual,actual]))
        assert torch.allclose(solo,paired[:1],atol=3e-5,rtol=1e-4)
        stats=measure(solo,solo)
        assert abs(stats['relative_l2'][0])<1e-7
        assert abs(stats['cosine'][0])<1e-5
        assert torch.equal(actual,make_input(base,'clean',0,kind,row))
        assert torch.equal(make_input(base,'rgb_noise_0.1',1,kind,row),make_input(base,'rgb_noise_0.1',1,kind,row))
        assert not torch.equal(make_input(base,'rgb_noise_0.1',1,kind,row),make_input(base,'rgb_noise_0.1',2,kind,row))
        no_rgb=make_input(base,'zero_rgb',0,kind,row)
        assert no_rgb[:3].count_nonzero()==0 and torch.equal(no_rgb[3:],actual[3:])
        no_xyz=make_input(base,'zero_xyz',0,kind,row)
        assert no_xyz[3:].count_nonzero()==0 and torch.equal(no_xyz[:3],actual[:3])
        shift=make_input(base,'rgb_shift_5',0,kind,row)
        assert torch.equal(shift[3:],actual[3:]) and torch.equal(shift[:3,:,5:],actual[:3,:,:-5])
        for n,s in conditions(): assert torch.isfinite(make_input(base,n,s,kind,row)).all()
        results[kind]=dict(original_normalization_max_error=err,feature_shape=list(solo.shape),
                           strict_checkpoint_load=True,eval_batch_independence=True,deterministic_perturbations=True,
                           modality_isolation=True,finite_inputs=True)
    synthetic=(np.ones((3,160,160),dtype=np.float32)*.5,np.ones((160,160),dtype=np.float32)*.05,
               np.ones((3,160,160),dtype=np.float32),np.array([.01,.02,.04],dtype=np.float32),.1)
    assert make_input(synthetic,'clean',0,'refine',row)[3:].count_nonzero()>0
    assert make_input(synthetic,'clean',0,'score',row)[3:].count_nonzero()==0
    results['source_specific_depth_threshold_check']=True
    # Relative drift distinguishes scaling, whereas cosine should not.
    f=torch.arange(1,13,dtype=torch.float32).reshape(1,3,2,2)
    scale=measure(2*f,f)
    assert abs(scale['cosine'][0])<1e-6 and abs(scale['relative_l2'][0]-1)<1e-6
    results['metric_scale_check']=True
    results['status']='passed'
    atomic(HERE/'validation.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':run()
