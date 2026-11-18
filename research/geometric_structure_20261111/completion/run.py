"""Real BOP RGB-D geometry completion supervised by CAD depth and normals."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import sys
import json
import copy
import time
import hashlib
import csv
import argparse
import importlib.util
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
import torch.nn.functional as F
import trimesh
import pyrender

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('extended',HERE.parent/'extended/run.py')
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
pilot=ext.pilot
TRAIN=[1,2,4,5,6,8,9];DEV=[10,11];EXPLORATORY=[12,13,14,15];RESERVED=[3,7]
SEEDS=[42,123,2026]


def crop_box(mask):
    yy,xx=np.where(mask)
    side=int(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)*1.2)
    cx,cy=(xx.max()+xx.min())//2,(yy.max()+yy.min())//2
    return (int(cx-side//2),int(cy-side//2),int(cx-side//2+side),int(cy-side//2+side))


def resized(array,box,size,nearest=True):
    return np.asarray(Image.fromarray(array).crop(box).resize((size,size),Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)).copy()


class CADRenderer:
    def __init__(self,size):
        self.size=size;self.renderer=pyrender.OffscreenRenderer(size,size)
    def render(self,mesh,R,t,K):
        scene=pyrender.Scene(bg_color=[0,0,0,0])
        scene.add(pyrender.IntrinsicsCamera(fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],znear=.01,zfar=5))
        transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=np.asarray(t)/1000
        transform=np.diag([1,-1,-1,1])@transform
        scene.add(mesh,pose=transform)
        _,depth=self.renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        return depth*1000


def normal_target(depth,K,valid,diameter):
    yy,xx=np.indices(depth.shape)
    xyz=np.stack([(xx-K[0,2])*depth/K[0,0],(yy-K[1,2])*depth/K[1,1],depth],axis=-1)
    dx=xyz[1:-1,2:]-xyz[1:-1,:-2];dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
    n=-np.cross(dx,dy);length=np.linalg.norm(n,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    ok&=(np.linalg.norm(dx,axis=-1)<.15*diameter)&(np.linalg.norm(dy,axis=-1)<.15*diameter)&(length>1e-6)
    normals=np.zeros((*depth.shape,3),dtype=np.float32)
    normals[1:-1,1:-1]=n/np.maximum(length[:,:,None],1e-6)
    nvalid=np.zeros(depth.shape,dtype=bool);nvalid[1:-1,1:-1]=ok
    return normals,nvalid


def prepare(args):
    cache=HERE/f'cache_{args.frames}_{args.size}_{hashlib.sha256(str(args.data.resolve()).encode()).hexdigest()[:8]}.pt'
    hole_prob=getattr(args,'hole_prob',.4);noise_mm=getattr(args,'noise_mm',2.)
    if hole_prob!=.4 or noise_mm!=2.:
        cache=cache.with_name(cache.stem+f'_holes{hole_prob}_noise{noise_mm}'+cache.suffix)
    if cache.exists():return torch.load(cache,map_location='cpu',weights_only=False)
    renderer=CADRenderer(args.size)
    info=json.loads((args.data/'models/models_info.json').read_text())
    splits={k:[] for k in ['train','development','seen','exploratory','reserved']};manifest=[]
    for obj in TRAIN+DEV+EXPLORATORY+RESERVED:
        mesh=trimesh.load(args.data/'models'/f'obj_{obj:06d}.ply',process=False)
        mesh.vertices/=1000
        render_mesh=pyrender.Mesh.from_trimesh(mesh,smooth=False)
        scene=args.data/'lm_test_all/test'/f'{obj:06d}'
        gt=json.loads((scene/'scene_gt.json').read_text());cams=json.loads((scene/'scene_camera.json').read_text())
        frames=sorted(gt,key=int);frames=[frames[i] for i in np.linspace(0,len(frames)-1,args.frames,dtype=int)]
        diameter=info[str(obj)]['diameter']
        accepted=0
        for index,frame in enumerate(frames):
            if obj in TRAIN:split='train' if index<int(.7*len(frames)) else ('seen' if index>=int(.8*len(frames)) else 'buffer')
            else:split='development' if obj in DEV else ('reserved' if obj in RESERVED else 'exploratory')
            if split=='buffer':continue
            gi=next(i for i,g in enumerate(gt[frame]) if g['obj_id']==obj)
            stem=f'{int(frame):06d}';g=gt[frame][gi]
            mask=np.asarray(Image.open(scene/'mask_visib'/f'{stem}_{gi:06d}.png'))>0
            if mask.sum()<100:continue
            box=crop_box(mask);scale=args.size/(box[2]-box[0])
            K=np.array(cams[frame]['cam_K']).reshape(3,3).astype(float)
            K[0,0]*=scale;K[1,1]*=scale
            K[0,2]=(K[0,2]-box[0]+.5)*scale-.5;K[1,2]=(K[1,2]-box[1]+.5)*scale-.5
            rgb=resized(np.asarray(Image.open(scene/'rgb'/f'{stem}.png').convert('RGB')),box,args.size,False)
            observed=resized(np.asarray(Image.open(scene/'depth'/f'{stem}.png')).astype(np.float32)*cams[frame]['depth_scale'],box,args.size)
            visible=resized(mask.astype(np.uint8),box,args.size)>0
            clean=renderer.render(render_mesh,np.array(g['cam_R_m2c']).reshape(3,3),g['cam_t_m2c'],K)
            valid=visible&(observed>0)&(clean>0)&(abs(observed-clean)<max(15,.05*diameter))
            seed=int(hashlib.sha256(f'{obj}/{stem}/holes-v1'.encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed)
            blocks=rng.random((8,8))<hole_prob
            holes=np.repeat(np.repeat(blocks,args.size//8,0),args.size//8,1)
            context=visible&(observed>0)&~holes
            if (valid&holes).sum()<30 or context.sum()<30:continue
            yy,xx=np.indices(clean.shape)
            noisy=observed+rng.normal(0,noise_mm,observed.shape)
            xyz=np.stack([(xx-K[0,2])*noisy/K[0,0],(yy-K[1,2])*noisy/K[1,1],noisy],-1).astype(np.float32)
            center=np.median(xyz[context],axis=0)
            normalized=(xyz-center)/(diameter/2);normalized[~context]=0
            normals,nvalid=normal_target(clean,K,valid,diameter)
            depthmask=valid&holes;normalmask=nvalid&holes
            if normalmask.sum()<8:continue
            target=(clean-center[2])/diameter;target[~valid]=0
            rgbtensor=torch.from_numpy(rgb).permute(2,0,1).float()/255
            x=torch.cat([rgbtensor,torch.from_numpy(normalized).permute(2,0,1)])
            # Nearest measured context is a strong observed-only, nonlearned completion baseline.
            nearest=distance_transform_edt(~context,return_distances=False,return_indices=True)
            filled=(noisy[tuple(nearest)]-center[2])/diameter
            sample={'x':x,'target':torch.from_numpy(target)[None].float(),'normals':torch.from_numpy(normals).permute(2,0,1),'valid':torch.from_numpy(depthmask)[None],'normalvalid':torch.from_numpy(normalmask)[None],'object':obj,'frame':stem,'diameter':diameter,'nearest':torch.from_numpy(filled)[None].float(),'context':torch.from_numpy(context)[None],'normal_fullvalid':torch.from_numpy(nvalid)[None],'cad_depth':torch.from_numpy(clean)[None],'intrinsics':torch.from_numpy(K),'center':torch.from_numpy(center)}
            splits[split].append(sample);accepted+=1
            manifest.append({'object':obj,'frame':stem,'split':split,'depth_pixels':int(depthmask.sum()),'normal_pixels':int(normalmask.sum()),'seed':seed})
        print(f'CAD data object={obj}: {accepted} accepted',flush=True)
    renderer.renderer.delete()
    result={'splits':splits,'manifest':manifest}
    torch.save(result,cache);return result


class Model(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.decoder=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,4,1))
    def forward(self,x):
        feat=self.encoder(x)
        y=F.interpolate(self.decoder(feat),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return y[:,:1],F.normalize(y[:,1:],dim=1),feat


def tensor_data(samples,device):
    return tuple(torch.stack([s[k] for s in samples]).to(device) for k in ['x','target','normals','valid','normalvalid'])


def loss_values(depth,normals,target,ntrue,mask,nmask,nweight):
    d=ext.image_mean(F.smooth_l1_loss(depth,target,beta=.025,reduction='none'),mask).mean()
    n=ext.image_mean(1-(normals*ntrue).sum(1,keepdim=True).clamp(-1,1),nmask).mean()
    return d+nweight*n


@torch.no_grad()
def evaluate(model,data,samples,baseline=None):
    rows=[];preds=[]
    for start in range(0,len(samples),32):
        x,y,n,m,nm=[v[start:start+32] for v in data]
        if baseline is None:
            depth,normal,_=model(x)
            angles=torch.rad2deg(torch.acos((normal*n).sum(1,keepdim=True).clamp(-1,1)))
        else:
            if baseline=='zero':depth=torch.zeros_like(y)
            else:depth=torch.stack([s['nearest'] for s in samples[start:start+32]]).to(x.device)
            angles=None
        error=ext.image_mean(abs(depth-y),m)
        for i,s in enumerate(samples[start:start+32]):
            row={'object':s['object'],'frame':s['frame'],'mae_diameter':float(error[i]),'mae_mm':float(error[i])*s['diameter']}
            if angles is not None:row['median_normal_angle_deg']=float(angles[i][nm[i]].median())
            rows.append(row)
        preds.append(torch.cat([depth.cpu(),normal.cpu()],1) if baseline is None else depth.cpu())
    return rows,torch.cat(preds)


def summary(rows):
    keys=[k for k in rows[0] if k not in ['object','frame']]
    per={str(obj):{k:float(np.mean([r[k] for r in rows if r['object']==obj])) for k in keys} for obj in sorted(set(r['object'] for r in rows))}
    return {'macro':{k:float(np.mean([v[k] for v in per.values()])) for k in keys},'per_object':per,'frames':len(rows)}


def train(model,teacher,state,arm,seed,steps,data,devdata,devsamples,out,probe=False):
    path=out/f'{arm}_{seed}.pth'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);return
    model.load_state_dict(state)
    tune=not probe and arm!='frozen'
    for p in model.encoder.parameters():p.requires_grad_(tune)
    model.eval()
    groups=[{'params':model.decoder.parameters(),'lr':1e-3}]+([{'params':model.encoder.parameters(),'lr':3e-5}] if tune else [])
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed);aug=torch.Generator(device=data[0].device).manual_seed(seed+19)
    interval=100 if probe else 250
    best=float('inf');best_state=None;best_step=0;history=[];start=time.monotonic()
    nweight=.2 if arm in ['frozen','normal'] or probe else 0
    for step in range(steps+1):
        if step%interval==0 or step==steps:
            rows,_=evaluate(model,devdata,devsamples);score=summary(rows)['macro']
            history.append(dict(step=step,seconds=time.monotonic()-start,**score))
            if score['mae_diameter']<best:
                best=score['mae_diameter'];best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(arm,seed,step,score,flush=True)
            ext.write(out/f'{arm}_{seed}_history.json',history)
        if step==steps:break
        idx=torch.randperm(len(data[0]),generator=rng)[:16].to(data[0].device)
        x,y,n,m,nm=[v[idx] for v in data];x=x.clone()
        brightness=.8+.4*torch.rand((len(x),1,1,1),device=x.device,generator=aug)
        contrast=.8+.4*torch.rand((len(x),1,1,1),device=x.device,generator=aug)
        rgb=x[:,:3];mean=rgb.mean((2,3),keepdim=True)
        x[:,:3]=((rgb-mean)*contrast+mean).mul(brightness).clamp(0,1)
        optimizer.zero_grad(set_to_none=True)
        d,pn,feat=model(x);loss=loss_values(d,pn,y,n,m,nm,nweight)
        if tune:
            with torch.no_grad():original=teacher(x)
            loss=loss+.01*F.mse_loss(F.normalize(feat,dim=1),F.normalize(original,dim=1))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
    model.load_state_dict(best_state)
    torch.save({'model':best_state,'best_step':best_step,'seed':seed,'arm':arm,'history':history,'probe':probe,'seconds':time.monotonic()-start},path)


def figure(samples,predictions,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    indices=[next(i for i,s in enumerate(samples) if s['object']==obj) for obj in RESERVED]
    fig,axs=plt.subplots(2,6,figsize=(17,6))
    for row,idx in enumerate(indices):
        sample=samples[idx];mask=sample['valid'][0].numpy();d=sample['diameter']
        true=sample['target'][0].numpy()*d;limit=max(float(np.quantile(abs(true[mask]),.98)),1)
        axs[row,0].imshow(sample['x'][:3].permute(1,2,0));axs[row,0].set_ylabel(f"LM object {sample['object']}")
        axs[row,1].imshow(sample['x'][5],cmap='coolwarm');axs[row,1].contour(mask,levels=[.5],colors=['black'],linewidths=.3)
        for col,value in [(2,true),(3,predictions['frozen'][idx,0].numpy()*d),(4,predictions['normal'][idx,0].numpy()*d)]:
            axs[row,col].imshow(np.ma.masked_where(~mask,value),cmap='coolwarm',vmin=-limit,vmax=limit)
        normals=predictions['normal'][idx,1:].permute(1,2,0).numpy()
        display=(normals+1)/2;display[~sample['normalvalid'][0].numpy()]=1
        axs[row,5].imshow(display)
        for col,title in enumerate(['Real BOP RGB','Observed XYZ Z (holes)','CAD depth in holes','Frozen encoder','Adapted + normals','Predicted normals in holes']):
            if row==0:axs[row,col].set_title(title,fontsize=10)
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
    fig.suptitle('Real-image geometry completion: fixed first frames of reserved adaptation objects\nDepth panels show only supervised holes; CAD is a target, not an encoder input')
    fig.tight_layout();fig.savefig(out/'completion_examples.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/home/Sahil/datasets/bop/lm'))
    p.add_argument('--out',type=Path,default=HERE/'results');p.add_argument('--frames',type=int,default=120);p.add_argument('--size',type=int,default=96)
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--probe-steps',type=int,default=500);p.add_argument('--device',default='cuda')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    assert args.size%8==0
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    prepared=prepare(args);splits=prepared['splits']
    ext.write(args.out/'manifest.json',{'frames':prepared['manifest'],'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),'args':{k:str(v) for k,v in vars(args).items()},'data':'Real BOP LINEMOD RGB-D with CAD-rendered targets','scope':'development; reserved objects excluded from previous adaptation run'})
    # Fail early on bad supervision rather than training on misregistered CAD targets.
    for split,samples in splits.items():
        assert samples,split
        for s in samples:
            assert torch.isfinite(s['x']).all() and torch.isfinite(s['target']).all()
            assert (s['x'][3:,s['valid'][0]]==0).all(),'Target depth leaked into input holes'
            assert (s['valid']&s['context']).sum()==0
            assert s['normalvalid'].sum()>0
    tensors={k:tensor_data(v,args.device) for k,v in splits.items()}
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(pilot.ROOT/'weights/2023-10-28-18-33-37/config.yml')
    base=pilot.RefineNet(cfg,c_in=cfg.c_in);base.load_state_dict(torch.load(pilot.ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    teacher=copy.deepcopy(base.encodeA).to(args.device).eval();del base
    for param in teacher.parameters():param.requires_grad_(False)
    selected={};initials={}
    for seed in SEEDS:
        torch.manual_seed(seed);model=Model(copy.deepcopy(teacher)).to(args.device).eval()
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initials[seed]=state
        for arm in ['frozen','depth','normal']:
            train(model,teacher,state,arm,seed,args.steps,tensors['train'],tensors['development'],splits['development'],args.out)
            selected[(seed,arm)]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for seed in SEEDS:
        for arm in ['frozen','depth','normal']:
            state=copy.deepcopy(initials[seed]);source=selected[(seed,arm)]
            for k,v in source.items():
                if k.startswith('encoder.'):state[k]=v
            model=Model(copy.deepcopy(teacher)).to(args.device).eval()
            train(model,teacher,state,'probe_'+arm,seed,args.probe_steps,tensors['train'],tensors['development'],splits['development'],args.out,probe=True)
            selected[(seed,'probe_'+arm)]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    metrics={};allrows=[];visual={}
    for seed in SEEDS:
        for arm in ['frozen','depth','normal','probe_frozen','probe_depth','probe_normal']:
            model=Model(copy.deepcopy(teacher)).to(args.device).eval();model.load_state_dict(selected[(seed,arm)])
            for split in ['seen','exploratory','reserved']:
                rows,pred=evaluate(model,tensors[split],splits[split]);metrics[f'{seed}/{arm}/{split}']=summary(rows)
                for row in rows:allrows.append(dict(row,seed=seed,arm=arm,split=split))
                if seed==42 and split=='reserved' and not arm.startswith('probe_'):visual[arm]=pred
    for split in ['seen','exploratory','reserved']:
        for baseline in ['zero','nearest']:
            rows,_=evaluate(None,tensors[split],splits[split],baseline=baseline);metrics[f'baseline/{baseline}/{split}']=summary(rows)
    ext.write(args.out/'evaluation.json',metrics)
    with (args.out/'per_frame.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    torch.save(visual,args.out/'visual_predictions.pth');figure(splits['reserved'],visual,args.out)
    lines=['# BOP real RGB-D geometry completion','','CAD-supervised missing-depth prediction, not RGB-only depth estimation. Primary reserved objects: 3,7. Previous evaluation objects are exploratory. Lower error is better.','','| Arm | Reserved depth MAE mm (mean ± seed SD) | Reserved normal angle deg |','|---|---:|---:|']
    for arm in ['frozen','depth','normal','probe_frozen','probe_depth','probe_normal']:
        errors=[metrics[f'{s}/{arm}/reserved']['macro']['mae_mm'] for s in SEEDS]
        angles=[metrics[f'{s}/{arm}/reserved']['macro']['median_normal_angle_deg'] for s in SEEDS]
        lines.append(f'| {arm} | {np.mean(errors):.3f} ± {np.std(errors,ddof=1):.3f} | {np.mean(angles):.2f} |')
    for baseline in ['zero','nearest']:
        lines.append(f"| {baseline} | {metrics[f'baseline/{baseline}/reserved']['macro']['mae_mm']:.3f} | — |")
    lines+=['','Normal angle is the object-macro average of per-image median angular errors. Depth-only decoder normal output is unsupervised and its normal score is not a meaningful learned-normal result.','', 'All RGB images are real BOP LINEMOD photographs. Depth/normal labels come from the matching BOP CAD model rendered using annotated pose and camera calibration. Missing regions are artificial; real sensor missingness and deployed pose performance remain untested. GT masks/crops and known diameters make this a controlled diagnostic. Two reserved objects and three seeds are insufficient for a broad generalization claim.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    ext.write(args.out/'status.json',{'status':'complete','optimizer_steps':len(SEEDS)*3*(args.steps+args.probe_steps),'seed_count':len(SEEDS)})
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
