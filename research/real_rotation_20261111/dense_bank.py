"""Exploratory coverage control: add a fixed random SO(3) CAD reference bank."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import argparse
import json
import copy
import numpy as np
import torch
import trimesh
import pyrender
from scipy.spatial.transform import Rotation
import run as r

HERE=Path(__file__).resolve().parent
OUT=HERE/'dense_results'


def build(prepared):
    path=HERE/'dense_reference_bank_v1.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    generator=np.random.default_rng(20261111)
    rotations=Rotation.random(512,random_state=generator).as_matrix()
    renderer=r.Renderer(96);banks={}
    for obj,d in prepared['objects'].items():
        mesh=trimesh.load(Path('/home/Sahil/datasets/bop/lm/models')/f'obj_{obj:06d}.ply',process=False)
        diameter=float(np.linalg.norm(mesh.bounds[1]-mesh.bounds[0]));mesh.vertices-=mesh.bounds.mean(0)
        mesh.vertices/=1000
        original=pyrender.Mesh.from_trimesh(mesh,smooth=False)
        plain=mesh.copy();mean=np.mean(mesh.visual.vertex_colors[:,:3],0).astype(np.uint8)
        plain.visual=trimesh.visual.ColorVisuals(mesh=plain,vertex_colors=np.tile(np.r_[mean,255],(len(plain.vertices),1)))
        uniform=pyrender.Mesh.from_trimesh(plain,smooth=False)
        for m in [original,uniform]:
            for p in m.primitives:
                p.material.metallicFactor=0;p.material.roughnessFactor=1;p.material.doubleSided=True;p.material.alphaMode='OPAQUE'
        renders=[]
        translation=np.array([0.,0.,3*diameter])
        for R in rotations:
            v=np.asarray(mesh.vertices)@R.T+translation/1000
            uv=v[:,:2]/v[:,2:]
            lo,hi=uv.min(0),uv.max(0);f=80/max(hi-lo);center=(lo+hi)/2
            K=np.array([[f,0,47.5-f*center[0]],[0,f,47.5-f*center[1]],[0,0,1.]])
            renders.append(renderer.pair((original,uniform),R,translation,K))
        renders=np.concatenate([d['renders'],np.array(renders)],0)
        assert abs(renders[:,0].astype(float)-renders[:,1].astype(float)).mean()>.1
        banks[obj]={'renders':renders,'rotations':np.concatenate([np.array([s['rotation'] for s in d['references']]),rotations],0)}
        print('dense bank rendered',obj,len(renders),flush=True)
    renderer.renderer.delete();torch.save(banks,path);return banks


@torch.no_grad()
def evaluate(model,prepared,banks,condition):
    rows=[];predictions={}
    for obj,d in prepared['objects'].items():
        query=r.inputs([s['rgb'] for s in d['splits']['evaluation']],'cuda')
        q=torch.cat([model(query[i:i+32]) for i in range(0,len(query),32)])
        renders=banks[obj]['renders'][:,0 if condition=='textured' else 1]
        descriptors=[]
        for start in range(0,len(renders),32):descriptors.append(model(r.inputs(renders[start:start+32],'cuda')))
        sim=(q@torch.cat(descriptors).T).cpu().numpy();indices=sim.argmax(1);predictions[obj]=indices
        R=np.array([s['rotation'] for s in d['splits']['evaluation']]);bank=banks[obj]['rotations']
        raw=r.distances(R,bank,[np.eye(3)]);sym=r.distances(R,bank,d['symmetries'])
        for i,s in enumerate(d['splits']['evaluation']):
            e=float(sym[i,indices[i]]);oracle=float(sym[i].min())
            rows.append({'object':obj,'frame':s['frame'],'rotation_error_deg':e,'raw_error_deg':float(raw[i,indices[i]]),'oracle_error_deg':oracle,'excess_over_oracle_deg':e-oracle,'recall_30deg':float(e<=30),'predicted_reference':int(indices[i])})
    return r.summary(rows),predictions


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
    prepared=torch.load(HERE/'cache_vertexcolor_v2_240_96.pt',map_location='cpu',weights_only=False)
    # No fitting here. Fixed coverage intervention after main evaluation; report as exploratory.
    banks=build(prepared)
    cfg=r.pilot.OmegaConf.load(r.pilot.ROOT/'weights/2023-10-28-18-33-37/config.yml')
    base=r.pilot.RefineNet(cfg,c_in=cfg.c_in);base.load_state_dict(torch.load(r.pilot.ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    results={};selected={}
    for seed in r.SEEDS:
        for arm in ['pretrained','ordinary','symmetry']:
            model=r.Matcher(copy.deepcopy(base.encodeA)).cuda().eval()
            if arm!='pretrained':model.load_state_dict(torch.load(HERE/'results'/f'{arm}_{seed}.pth',map_location='cpu',weights_only=False)['model'])
            for condition in ['textured','uniform']:
                metric,pred=evaluate(model,prepared,banks,condition);results[f'{seed}/{arm}/{condition}']=metric
                if seed==42:selected[arm,condition]=pred
    r.ext.write(OUT/'evaluation.json',results)
    lines=['# Exploratory reference-bank coverage control','','No training or checkpoint reselection. Add 512 SO(3) rotations sampled with seed 20261111, independent of evaluation labels, to each original 48-view bank. Canonical render crops normalize object extent. The fixed bank now has 560 views/object. This tests reference coverage and distractor effects; it is not a new encoder or a fresh final test.','','| Encoder | References | Mean symmetry-aware error deg | Recall <=30deg | Oracle coverage error deg |','|---|---|---:|---:|---:|']
    for arm in ['pretrained','ordinary','symmetry']:
        for cond in ['textured','uniform']:
            vals=[results[f'{s}/{arm}/{cond}']['macro'] for s in r.SEEDS]
            lines.append(f"| {arm} | {cond} | {np.mean([v['rotation_error_deg'] for v in vals]):.2f} | {np.mean([v['recall_30deg'] for v in vals]):.3f} | {vals[0]['oracle_error_deg']:.2f} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,3,figsize=(9,14))
    for row,obj in enumerate(r.OBJECTS):
        d=prepared['objects'][obj];sample=d['splits']['evaluation'][0]
        error=r.distances(sample['rotation'][None],banks[obj]['rotations'],d['symmetries'])[0]
        a=int(selected['symmetry','textured'][obj][0]);b=int(selected['symmetry','uniform'][obj][0])
        for col,(im,title) in enumerate([(sample['original'],'Real BOP image'),(banks[obj]['renders'][a,0],f'Textured dense-bank match: {error[a]:.1f}°'),(banks[obj]['renders'][b,1],f'Uniform dense-bank match: {error[b]:.1f}°')]):
            axes[row,col].imshow(im);axes[row,col].set_title(title,fontsize=9);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[row,0].set_ylabel(f'LM object {obj}')
    fig.suptitle('Exploratory dense reference bank; same fixed real evaluation queries, seed 42')
    fig.tight_layout();fig.savefig(OUT/'dense_rotation_examples.png',dpi=150)
    r.ext.write(OUT/'status.json',{'status':'complete','reference_count':560,'random_rotation_seed':20261111,'training':'none'})
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
