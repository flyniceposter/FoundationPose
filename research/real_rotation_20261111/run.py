"""Real RGB queries against training-only BOP CAD pose references."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import sys
import importlib.util
import copy
import json
import hashlib
import csv
import argparse
import time
import numpy as np
from PIL import Image
import trimesh
import pyrender
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('completion',HERE.parent/'geometric_structure_20261111/completion/run.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
ext=c.ext;pilot=c.pilot
OBJECTS=[5,8,9,10,11];SEEDS=[42,123,2026]


def project(R):
    u,_,v=np.linalg.svd(R);return (u*np.array([1,1,np.linalg.det(u@v)]))@v


def distances(query,reference,syms):
    values=[]
    for S in syms:
        q=query@S
        traces=np.einsum('nij,mij->nm',q,reference)
        values.append(np.degrees(np.arccos(np.clip((traces-1)/2,-1,1))))
    return np.min(values,axis=0)


class Renderer(c.CADRenderer):
    def pair(self,meshes,R,t,K):
        outputs=[];depths=[]
        for mesh in meshes:
            scene=pyrender.Scene(bg_color=[40/255]*3+[1],ambient_light=[.6]*3)
            scene.add(pyrender.IntrinsicsCamera(fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],znear=.01,zfar=5))
            scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.5))
            tf=np.eye(4);tf[:3,:3]=R;tf[:3,3]=np.asarray(t)/1000;tf=np.diag([1,-1,-1,1])@tf
            scene.add(mesh,pose=tf)
            rgb,depth=self.renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            rgb=rgb[:,:,:3];rgb[depth<=0]=40
            outputs.append(rgb);depths.append(depth)
        assert np.allclose(depths[0],depths[1],atol=1e-6),'Appearance ablation changed geometry'
        return outputs


def prepare(args):
    path=HERE/f'cache_vertexcolor_v2_{args.frames}_{args.size}.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    info=json.loads((args.data/'models/models_info.json').read_text());renderer=Renderer(args.size)
    objects={};manifest=[]
    for obj in OBJECTS:
        assert not info[str(obj)].get('symmetries_continuous')
        syms=[np.eye(3)]+[project(np.array(S).reshape(4,4)[:3,:3]) for S in info[str(obj)].get('symmetries_discrete',[])]
        scene=args.data/'lm_test_all/test'/f'{obj:06d}'
        gt=json.loads((scene/'scene_gt.json').read_text());cameras=json.loads((scene/'scene_camera.json').read_text())
        frames=sorted(gt,key=int);frames=[frames[i] for i in np.linspace(0,len(frames)-1,args.frames,dtype=int)]
        splits={'train':[],'development':[],'evaluation':[]}
        for i,frame in enumerate(frames):
            split='train' if i<int(.6*len(frames)) else ('development' if int(.65*len(frames))<=i<int(.8*len(frames)) else ('evaluation' if i>=int(.85*len(frames)) else 'buffer'))
            if split=='buffer':continue
            gi=next(k for k,g in enumerate(gt[frame]) if g['obj_id']==obj);g=gt[frame][gi];stem=f'{int(frame):06d}'
            mask=np.asarray(Image.open(scene/'mask_visib'/f'{stem}_{gi:06d}.png'))>0
            if mask.sum()<100:continue
            box=c.crop_box(mask);scale=args.size/(box[2]-box[0]);K=np.array(cameras[frame]['cam_K']).reshape(3,3).astype(float)
            K[0,0]*=scale;K[1,1]*=scale;K[0,2]=(K[0,2]-box[0]+.5)*scale-.5;K[1,2]=(K[1,2]-box[1]+.5)*scale-.5
            original=c.resized(np.asarray(Image.open(scene/'rgb'/f'{stem}.png').convert('RGB')),box,args.size,False)
            visible=c.resized(mask.astype(np.uint8),box,args.size)>0
            masked=original.copy();masked[~visible]=40
            sample={'original':original,'rgb':masked,'rotation':project(np.array(g['cam_R_m2c']).reshape(3,3)),'translation':g['cam_t_m2c'],'K':K,'frame':stem,'object':obj}
            splits[split].append(sample);manifest.append({'object':obj,'frame':stem,'split':split})
        mesh=trimesh.load(args.data/'models'/f'obj_{obj:06d}.ply',process=False);mesh.vertices/=1000
        material=lambda:pyrender.MetallicRoughnessMaterial(baseColorFactor=[1,1,1,1],metallicFactor=0,roughnessFactor=1,doubleSided=True)
        colored=pyrender.Mesh.from_trimesh(mesh,smooth=False)
        plain=mesh.copy();mean=np.mean(mesh.visual.vertex_colors[:,:3],axis=0).astype(np.uint8)
        plain.visual=trimesh.visual.ColorVisuals(mesh=plain,vertex_colors=np.tile(np.r_[mean,255],(len(plain.vertices),1)))
        uniform=pyrender.Mesh.from_trimesh(plain,smooth=False)
        for render_mesh in (colored,uniform):
            for primitive in render_mesh.primitives:
                primitive.material.metallicFactor=0
                primitive.material.roughnessFactor=1
                primitive.material.doubleSided=True
                primitive.material.alphaMode='OPAQUE'
        ids=np.linspace(0,len(splits['train'])-1,48,dtype=int)
        references=[splits['train'][i] for i in ids]
        renders=[renderer.pair((colored,uniform),s['rotation'],s['translation'],s['K']) for s in references]
        assert np.abs(np.asarray(renders)[:,0].astype(float)-np.asarray(renders)[:,1].astype(float)).mean()>.1,'Texture ablation has no appearance effect'
        objects[obj]={'splits':splits,'references':references,'renders':np.array(renders),'symmetries':np.array(syms)}
        print('rotation real object',obj,{k:len(v) for k,v in splits.items()},flush=True)
    renderer.renderer.delete();result={'objects':objects,'manifest':manifest};torch.save(result,path);return result


def inputs(images,device):
    rgb=torch.from_numpy(np.stack(images).copy()).permute(0,3,1,2).float().to(device)/255
    return torch.cat([rgb,torch.zeros_like(rgb)],1)


class Matcher(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
    def forward(self,x):return F.normalize(F.adaptive_avg_pool2d(self.encoder(x),(4,4)).flatten(1),dim=1)


@torch.no_grad()
def evaluate(model,prepared,tensors,split,condition='textured'):
    rows=[];scores={}
    for obj,d in prepared['objects'].items():
        reference=model(tensors[obj][condition])
        queries=tensors[obj][split];features=torch.cat([model(queries[i:i+32]) for i in range(0,len(queries),32)])
        similarity=(features@reference.T).cpu().numpy();scores[obj]=similarity
        R=np.array([s['rotation'] for s in d['splits'][split]]);bank=np.array([s['rotation'] for s in d['references']])
        raw=distances(R,bank,[np.eye(3)]);sym=distances(R,bank,d['symmetries']);prediction=similarity.argmax(1)
        for i,s in enumerate(d['splits'][split]):
            e=float(sym[i,prediction[i]]);oracle=float(sym[i].min())
            rows.append({'object':obj,'frame':s['frame'],'rotation_error_deg':e,'raw_error_deg':float(raw[i,prediction[i]]),'oracle_error_deg':oracle,'excess_over_oracle_deg':e-oracle,'recall_30deg':float(e<=30),'predicted_reference':int(prediction[i])})
    return rows,scores


def summary(rows):
    keys=['rotation_error_deg','raw_error_deg','oracle_error_deg','excess_over_oracle_deg','recall_30deg']
    per={str(obj):{k:float(np.mean([r[k] for r in rows if r['object']==obj])) for k in keys} for obj in OBJECTS}
    return {'macro':{k:float(np.mean([v[k] for v in per.values()])) for k in keys},'per_object':per,'frames':len(rows)}


def train(model,teacher,initial,arm,seed,args,prepared,tensors):
    path=args.out/f'{arm}_{seed}.pth'
    if path.exists():model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);return
    model.load_state_dict(initial);model.eval()
    for p in model.parameters():p.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    rng=np.random.default_rng(seed);generator=torch.Generator(device=args.device).manual_seed(seed+19)
    labels={}
    for obj,d in prepared['objects'].items():
        R=np.array([s['rotation'] for s in d['splits']['train']]);bank=np.array([s['rotation'] for s in d['references']])
        angle=distances(R,bank,d['symmetries'] if arm=='symmetry' else [np.eye(3)])
        # Stable row-wise softmax, avoiding underflow when bank coverage is poor.
        logits=-angle**2/(2*15**2);logits-=logits.max(1,keepdims=True)
        target=np.exp(logits);target/=target.sum(1,keepdims=True)
        labels[obj]=torch.tensor(target,dtype=torch.float32,device=args.device)
    best=float('inf');best_state=None;history=[];best_step=0;start=time.monotonic()
    for step in range(args.steps+1):
        if step%250==0 or step==args.steps:
            rows,_=evaluate(model,prepared,tensors,'development');score=summary(rows)['macro']
            history.append(dict(step=step,seconds=time.monotonic()-start,**score))
            if score['rotation_error_deg']<best:
                best=score['rotation_error_deg'];best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('rotation',arm,seed,step,score,flush=True);ext.write(args.out/f'{arm}_{seed}_history.json',history)
        if step==args.steps:break
        obj=int(rng.choice(OBJECTS));indices=rng.choice(len(tensors[obj]['train']),8,replace=False)
        x=tensors[obj]['train'][indices].clone()
        brightness=.8+.4*torch.rand((len(x),1,1,1),device=args.device,generator=generator)
        contrast=.8+.4*torch.rand((len(x),1,1,1),device=args.device,generator=generator)
        rgb=x[:,:3];mean=rgb.mean((2,3),keepdim=True);x[:,:3]=((rgb-mean)*contrast+mean).mul(brightness).clamp(0,1)
        optimizer.zero_grad(set_to_none=True)
        q=model(x);ref=model(tensors[obj]['textured'])
        loss=-(labels[obj][indices]*F.log_softmax(q@ref.T/.07,dim=1)).sum(1).mean()
        with torch.no_grad():original=teacher(x)
        loss=loss+.01*F.mse_loss(q,original)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite rotation loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
    model.load_state_dict(best_state)
    torch.save({'model':best_state,'best_step':best_step,'history':history,'seed':seed,'arm':arm,'seconds':time.monotonic()-start},path)


def figure(prepared,scores,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(OBJECTS),5,figsize=(14,14))
    for row,obj in enumerate(OBJECTS):
        d=prepared['objects'][obj];query=d['splits']['evaluation'][0]
        bank=np.array([s['rotation'] for s in d['references']]);error=distances(query['rotation'][None],bank,d['symmetries'])[0]
        pt=int(scores['symmetry','textured'][obj][0].argmax());pu=int(scores['symmetry','uniform'][obj][0].argmax());oracle=int(error.argmin())
        pictures=[query['original'],query['rgb'],d['renders'][oracle,0],d['renders'][pt,0],d['renders'][pu,1]]
        titles=['Real BOP crop','Masked encoder input',f'Best bank view: {error[oracle]:.1f}°',f'Textured match: {error[pt]:.1f}°',f'Uniform match: {error[pu]:.1f}°']
        for col,(im,title) in enumerate(zip(pictures,titles)):
            axes[row,col].imshow(im);axes[row,col].set_title(title,fontsize=9);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[row,0].set_ylabel(f'LM object {obj}')
    fig.suptitle('Real BOP photographs matched to training-only CAD references\nFixed first evaluation query per object; seed 42, symmetry-aware encoder. Angles are symmetry-aware errors.')
    fig.tight_layout();fig.savefig(out/'real_rotation_examples.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/home/Sahil/datasets/bop/lm'));p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--frames',type=int,default=240);p.add_argument('--size',type=int,default=96);p.add_argument('--steps',type=int,default=1500);p.add_argument('--device',default='cuda')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    prepared=prepare(args)
    ext.write(args.out/'manifest.json',{'frames':prepared['manifest'],'reference_frames':{obj:[s['frame'] for s in d['references']] for obj,d in prepared['objects'].items()},'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),'args':{k:str(v) for k,v in vars(args).items()},'real_queries':True,'reference_depth_equality_checked':True})
    tensors={obj:{**{split:inputs([s['rgb'] for s in d['splits'][split]],args.device) for split in d['splits']},'textured':inputs(d['renders'][:,0],args.device),'uniform':inputs(d['renders'][:,1],args.device)} for obj,d in prepared['objects'].items()}
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(pilot.ROOT/'weights/2023-10-28-18-33-37/config.yml');base=pilot.RefineNet(cfg,c_in=cfg.c_in)
    base.load_state_dict(torch.load(pilot.ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    teacher=Matcher(copy.deepcopy(base.encodeA)).to(args.device).eval();del base
    for param in teacher.parameters():param.requires_grad_(False)
    initial={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()};selected={}
    for seed in SEEDS:
        for arm in ['ordinary','symmetry']:
            model=Matcher(copy.deepcopy(teacher.encoder)).to(args.device).eval()
            train(model,teacher,initial,arm,seed,args,prepared,tensors)
            selected[seed,arm]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    metrics={};allrows=[];visualscores={}
    for seed in SEEDS:
        for arm in ['pretrained','ordinary','symmetry']:
            model=Matcher(copy.deepcopy(teacher.encoder)).to(args.device).eval();model.load_state_dict(initial if arm=='pretrained' else selected[seed,arm])
            for condition in ['textured','uniform']:
                rows,scores=evaluate(model,prepared,tensors,'evaluation',condition)
                metrics[f'{seed}/{arm}/{condition}']=summary(rows)
                for row in rows:allrows.append(dict(row,seed=seed,arm=arm,condition=condition))
                if seed==42:visualscores[arm,condition]=scores
    ext.write(args.out/'evaluation.json',metrics)
    with (args.out/'per_frame.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    np.savez_compressed(args.out/'scores_seed42.npz',**{f'{a}_{b}_{obj}':v for (a,b),ds in visualscores.items() for obj,v in ds.items()})
    figure(prepared,visualscores,args.out)
    lines=['# Real BOP rotation, symmetry and texture','','Real BOP RGB query photographs; references are CAD renders from training poses only. Lower symmetry-aware angle is better.','','| Encoder | Reference appearance | Mean rotation error deg ± seed SD | Recall <=30deg | Oracle bank error deg |','|---|---|---:|---:|---:|']
    for arm in ['pretrained','ordinary','symmetry']:
        for condition in ['textured','uniform']:
            vals=[metrics[f'{seed}/{arm}/{condition}']['macro'] for seed in SEEDS]
            e=[v['rotation_error_deg'] for v in vals]
            lines.append(f"| {arm} | {condition} | {np.mean(e):.2f} ± {np.std(e,ddof=1):.2f} | {np.mean([v['recall_30deg'] for v in vals]):.3f} | {vals[0]['oracle_error_deg']:.2f} |")
    lines+=['','Raw and symmetry-aware per-object results, seed results, oracle coverage and excess errors are in evaluation.json. Pretrained is identical across seeds; its zero seed SD does not indicate statistical certainty.','', 'GT visible masks/crops and known identities are supplied. Evaluation uses later frames in the same sequences; no independent-scene or official BOP pose benchmark claim. Uniform references change appearance while real queries stay unchanged; this gap is not by itself proof of newly learned texture. Symmetry equivalence here is a rotation-only metric, not a full-pose loss handling symmetry translations.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n');ext.write(args.out/'status.json',{'status':'complete','optimizer_steps':len(SEEDS)*2*args.steps})
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
