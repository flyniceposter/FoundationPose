import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import sys, json, argparse, logging
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
import train as t
from estimater import FoundationPose
from datareader import LinemodReader
from Utils import set_seed
import nvdiffrast.torch as dr

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-only',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.smoke:args.baseline_only=True
    torch.set_num_threads(4);logging.getLogger().setLevel(logging.WARNING)
    out=HERE/'pose';out.mkdir(exist_ok=True)
    records=json.loads((out/'records.json').read_text()) if (out/'records.json').exists() else []
    completed={(r['arm'],r['seed'],r['object'],r['frame'],r['condition']) for r in records}
    cache=torch.load(HERE.parent/'real_rotation_20261111/cache_vertexcolor_v2_240_96.pt',map_location='cpu',weights_only=False)
    context=dr.RasterizeCudaContext()
    est=None;original=None;original_full=None
    head_arms=['heads_only','joint_original','joint_combined','frozen_combined_heads']
    variants=[('original',0)]+([] if args.baseline_only else [(arm,seed) for seed in t.SEEDS for arm in t.ARMS if arm!='frozen']+[(arm,seed) for seed in t.SEEDS for arm in head_arms])
    for obj in (t.r.OBJECTS[:1] if args.smoke else t.r.OBJECTS):
        reader=LinemodReader(str(t.DATA/'lm_test_all/test'/f'{obj:06d}'),split=None)
        reader.K=reader.K.astype(np.float32)
        mesh=reader.get_gt_mesh(obj)
        if est is None:
            est=FoundationPose(model_pts=mesh.vertices.copy(),model_normals=mesh.vertex_normals.copy(),symmetry_tfs=reader.symmetry_tfs[obj],mesh=mesh,glctx=context,debug=0,debug_dir=str(out/'debug'))
            original={k:v.cpu().clone() for k,v in est.refiner.model.encodeA.state_dict().items()}
            original_full={k:v.cpu().clone() for k,v in est.refiner.model.state_dict().items()}
        else:
            est.reset_object(model_pts=mesh.vertices.copy(),model_normals=mesh.vertex_normals.copy(),symmetry_tfs=reader.symmetry_tfs[obj],mesh=mesh)
            est.make_rotation_grid(min_n_views=40,inplane_step=60)
        est.diameter=float(est.diameter)
        d=cache['objects'][obj]
        samples=[s for group in d['splits'].values() for s in group if t.partition(obj,s['frame'])=='evaluation']
        samples=sorted(samples,key=lambda s:int(s['frame']))
        samples=[samples[i] for i in np.linspace(0,len(samples)-1,4,dtype=int)]
        vertices=np.asarray(mesh.vertices);vertices=vertices[np.linspace(0,len(vertices)-1,min(5000,len(vertices)),dtype=int)]
        diameter=reader.get_model_diameter(obj)
        for arm,seed in variants:
            est.refiner.model.load_state_dict(original_full)
            if arm in head_arms:
                state=torch.load(HERE/'heads'/f'{arm}_{seed}.pth',map_location='cpu',weights_only=False)['model']
                est.refiner.model.load_state_dict(state)
                weights=est.refiner.model.encodeA.state_dict()
            elif arm=='original':weights=original
            else:
                state=torch.load(HERE/'results'/f'{arm}_{seed}.pth',map_location='cpu',weights_only=False)['model']
                weights={k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}
            est.refiner.model.encodeA.load_state_dict(weights);est.refiner.model.eval()
            for s in (samples[:1] if args.smoke else samples):
                i=reader.id_strs.index(s['frame']);rgb=reader.get_color(i);depth=reader.get_depth(i);mask=reader.get_mask(i,obj);gt=reader.get_gt_pose(i,obj)
                for condition in (['observed'] if args.smoke else ['observed','noise5mm','dropout40','grayscale']):
                    key=(arm,seed,obj,s['frame'],condition)
                    if key in completed:continue
                    rng=np.random.default_rng(obj*100000+int(s['frame']));image=rgb.copy();z=depth.copy()
                    if condition=='noise5mm':
                        valid=z>0;z[valid]+=rng.normal(0,.005,valid.sum());z=np.maximum(z,0)
                    if condition=='dropout40':
                        blocks=rng.random((8,8))<.4
                        holes=np.asarray(Image.fromarray(blocks).resize((z.shape[1],z.shape[0]),Image.Resampling.NEAREST));z[holes]=0
                    if condition=='grayscale':image=np.repeat(np.round(image.mean(2)).astype(np.uint8)[...,None],3,2)
                    set_seed(42);est.gt_pose=None
                    pred=np.asarray(est.register(K=reader.K,rgb=image,depth=z,ob_mask=mask,ob_id=obj,iteration=5))
                    assert np.isfinite(pred).all()
                    a=vertices@pred[:3,:3].T+pred[:3,3];b=vertices@gt[:3,:3].T+gt[:3,3]
                    add=float(np.linalg.norm(a-b,axis=1).mean());adds=float(cKDTree(b).query(a)[0].mean())
                    row=dict(arm=arm,seed=seed,object=obj,frame=s['frame'],condition=condition,add_mm=add*1000,adds_mm=adds*1000,add_recall_10diam=float(add<.1*diameter),adds_recall_10diam=float(adds<.1*diameter),prediction=pred.tolist(),ground_truth=gt.tolist())
                    records.append(row);t.write(out/'records.json',records);completed.add(key)
                    if arm=='original':Image.fromarray(image).save(out/f'input_{obj}_{s["frame"]}_{condition}.png')
                    t.write(HERE/'progress.json',dict(stage='pose_baseline' if args.baseline_only else 'pose_evaluation',completed_cases=len(records),last_case=list(key)))
                    print('POSE',key,add*1000,adds*1000,flush=True)
    t.write(out/('registration_smoke.json' if args.smoke else 'baseline_complete.json' if args.baseline_only else 'complete.json'),dict(cases=len(records)))
if __name__=='__main__':main()
