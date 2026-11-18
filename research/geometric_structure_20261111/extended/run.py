"""Multi-object, matched-seed BOP visible-geometry adaptation experiment."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('geometry_pilot',HERE.parent/'train_geometry.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
TRAIN=[1,2,4,5,6,8,9];DEV=[10,11];TEST=[12,13,14,15]
SEEDS=[42,123,2026]


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def image_mean(value,mask):
    return (value*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)


def geometry_loss(pred,target,valid,weight):
    depth=image_mean(F.smooth_l1_loss(pred,target,beta=.05,reduction='none'),valid).mean()
    slopes=[]
    for stride in (1,2,4):
        for axis in (-1,-2):
            before=[slice(None)]*4;after=[slice(None)]*4
            before[axis]=slice(None,-stride);after[axis]=slice(stride,None)
            a,b=tuple(before),tuple(after)
            mask=valid[a]&valid[b]
            delta=((pred[b]-pred[a])-(target[b]-target[a])).abs()/stride
            slopes.append(image_mean(delta,mask).mean())
    return depth+weight*torch.stack(slopes).mean()


def metric_vectors(pred,target,valid):
    mae=image_mean((pred-target).abs(),valid)
    slope=[]
    for axis in (-1,-2):
        before=[slice(None)]*4;after=[slice(None)]*4
        before[axis]=slice(None,-1);after[axis]=slice(1,None)
        a,b=tuple(before),tuple(after)
        slope.append(image_mean(((pred[b]-pred[a])-(target[b]-target[a])).abs(),valid[a]&valid[b]))
    return mae,torch.stack(slope).mean(0)


def macro(rows):
    keys=['mae_diameter','mae_mm','slope_diameter_per_pixel']
    objects=sorted(set(r['object'] for r in rows))
    per={str(obj):{k:float(np.mean([r[k] for r in rows if r['object']==obj])) for k in keys} for obj in objects}
    return {'macro':{k:float(np.mean([v[k] for v in per.values()])) for k in keys},'per_object':per,'frames':len(rows)}


class Model(pilot.GeometryModel):
    def forward(self,x,shuffle=False):
        features=self.encoder(x)
        if shuffle:
            # Fixed, repeatable permutation at common spatial resolution.
            generator=torch.Generator(device=features.device).manual_seed(991)
            idx=torch.randperm(features.shape[-2]*features.shape[-1],generator=generator,device=features.device)
            features=features.flatten(2)[:,:,idx].reshape_as(features)
        return F.interpolate(self.decoder(features),size=x.shape[-2:],mode='bilinear',align_corners=False)


def prepare(args):
    cache=HERE/f'cache_{args.frames}_{args.size}.pt'
    if cache.exists():
        return torch.load(cache,map_location='cpu',weights_only=False)
    splits={'train':[],'development':[],'seen':[],'unseen':[]}
    manifest=[];source_hashes={}
    for obj in TRAIN+DEV+TEST:
        samples,diameter=pilot.load_samples(args.data,obj,args.frames,args.size)
        for i,sample in enumerate(samples):
            sample['object']=obj;sample['diameter']=diameter
            if obj in TRAIN:
                split='train' if i<int(.7*len(samples)) else ('seen' if i>=int(.8*len(samples)) else 'buffer')
            else:split='development' if obj in DEV else 'unseen'
            manifest.append({'object':obj,'frame':sample['frame'],'split':split})
            if split!='buffer':splits[split].append(sample)
        scene=args.data/'lm_test_all/test'/f'{obj:06d}'
        for name in ('scene_gt.json','scene_camera.json'):
            file=scene/name;source_hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        print('loaded object',obj,flush=True)
    result={'splits':splits,'manifest':manifest,'source_hashes':source_hashes}
    torch.save(result,cache)
    return result


def tensors(samples,device):
    x=torch.stack([s['x'] for s in samples]).to(device)
    y=torch.stack([s['target'] for s in samples]).to(device)
    m=torch.stack([s['valid'] for s in samples]).to(device)
    return x,y,m


@torch.no_grad()
def evaluate(model,data,samples,shuffle=False,constant=None):
    rows=[];predictions=[]
    for start in range(0,len(samples),32):
        x,y,m=[t[start:start+32] for t in data]
        if constant is None:p=model(x,shuffle=shuffle)
        else:p=constant.to(x.device).expand(len(x),-1,-1,-1)
        mae,slope=metric_vectors(p,y,m)
        predictions.append(p.cpu())
        for i,s in enumerate(samples[start:start+32]):
            rows.append({'object':s['object'],'frame':s['frame'],'mae_diameter':float(mae[i]),'mae_mm':float(mae[i])*s['diameter'],'slope_diameter_per_pixel':float(slope[i])})
    return rows,torch.cat(predictions)


def train(model,initial,arm,seed,steps,interval,data,devdata,devsamples,out,probe=False):
    checkpoint=out/f'{arm}_seed{seed}.pth'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);return model,saved['history']
    model.load_state_dict(initial)
    tune=arm!='frozen' and not probe
    for p in model.encoder.parameters():p.requires_grad_(tune)
    model.eval()
    groups=[{'params':model.decoder.parameters(),'lr':1e-3}]
    if tune:groups.append({'params':model.encoder.parameters(),'lr':1e-4})
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed)
    augmentation_rng=torch.Generator(device=data[0].device).manual_seed(seed+19)
    history=[];best=float('inf');best_state=None;best_step=0;start=time.monotonic()
    weight=.2 if arm=='structure' and not probe else 0
    for step in range(steps+1):
        if step%interval==0 or step==steps:
            rows,_=evaluate(model,devdata,devsamples)
            score=macro(rows)['macro']['mae_diameter']
            history.append({'step':step,'development_mae_diameter':score,'elapsed_seconds':time.monotonic()-start})
            if score<best:
                best=score;best_step=step
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f'{arm} seed={seed} step={step}/{steps} dev MAE/diameter={score:.5f}',flush=True)
            write(out/f'{arm}_seed{seed}_history.json',history)
        if step==steps:break
        idx=torch.randperm(len(data[0]),generator=rng)[:16].to(data[0].device)
        x,y,m=[v[idx] for v in data]
        x=x.clone()
        brightness=.8+.4*torch.rand((len(x),1,1,1),generator=augmentation_rng,device=x.device)
        contrast=.8+.4*torch.rand((len(x),1,1,1),generator=augmentation_rng,device=x.device)
        rgb=x[:,:3];mean=rgb.mean((2,3),keepdim=True)
        x[:,:3]=((rgb-mean)*contrast+mean).mul(brightness).clamp(0,1)
        optimizer.zero_grad(set_to_none=True)
        loss=geometry_loss(model(x),y,m,weight)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
    model.load_state_dict(best_state)
    torch.save({'model':best_state,'best_step':best_step,'development_mae_diameter':best,'history':history,'seed':seed,'arm':arm,'fresh_probe':probe,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'seconds':time.monotonic()-start},checkpoint)
    return model,history


def make_figure(samples,predictions,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[next(i for i,s in enumerate(samples) if s['object']==obj) for obj in TEST]
    fig,axs=plt.subplots(4,5,figsize=(15,12))
    for row,idx in enumerate(chosen):
        s=samples[idx];valid=s['valid'][0].numpy();truth=s['target'][0].numpy()*s['diameter']
        axs[row,0].imshow(s['x'][:3].permute(1,2,0));axs[row,0].set_ylabel(f"LM object {s['object']}")
        values=[truth]+[predictions[name][idx,0].numpy()*s['diameter'] for name in ('frozen','depth','structure')]
        limit=max(float(np.quantile(np.abs(truth[valid]),.98)),1)
        for col,value in enumerate(values,1):
            im=axs[row,col].imshow(np.ma.masked_where(~valid,value),cmap='coolwarm',vmin=-limit,vmax=limit)
        fig.colorbar(im,ax=list(axs[row,1:]),label='Relative depth (mm)',fraction=.015,pad=.01)
        for col,title in enumerate(['RGB input','Measured depth','Frozen encoder','Depth fine-tuning','Depth + surface slopes']):
            if row==0:axs[row,col].set_title(title)
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
    fig.suptitle('BOP LINEMOD: fixed adaptation-held-out object examples, seed 42\nDevelopment-only; ground-truth crops, RGB-only input, masked relative-depth display',fontsize=14)
    fig.subplots_adjust(top=.90,wspace=.08,hspace=.12)
    fig.savefig(out/'geometry_comparison.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('/home/Sahil/datasets/bop/lm'))
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--device',default='cuda')
    p.add_argument('--frames',type=int,default=120);p.add_argument('--size',type=int,default=96)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--probe-steps',type=int,default=400)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.device=='cuda':
        assert torch.cuda.is_available(),'CUDA unavailable; run with host GPU access'
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    prepared=prepare(args);splits=prepared['splits']
    write(args.out/'manifest.json',{'scope':'development-only; object-disjoint to adaptation, not original pretraining','frames':prepared['manifest'],'source_hashes':prepared['source_hashes'],'args':{k:str(v) for k,v in vars(args).items()},'torch':torch.__version__,'device':torch.cuda.get_device_name() if args.device=='cuda' else 'cpu','protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()})
    data={k:tensors(v,args.device) for k,v in splits.items()}
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(pilot.ROOT/'weights/2023-10-28-18-33-37/config.yml')
    original=pilot.RefineNet(cfg,c_in=cfg.c_in)
    original.load_state_dict(torch.load(pilot.ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    encoder=copy.deepcopy(original.encodeA);del original
    selected={};initials={}
    # All adaptations and all fresh probes complete before evaluation-set access.
    for seed in SEEDS:
        torch.manual_seed(seed)
        model=Model(copy.deepcopy(encoder)).to(args.device).eval()
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initials[seed]=initial
        for arm in ('frozen','depth','structure'):
            model,_=train(model,initial,arm,seed,args.steps,200,data['train'],data['development'],splits['development'],args.out)
            selected[(seed,arm)]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for seed in SEEDS:
        for arm in ('frozen','depth','structure'):
            state=copy.deepcopy(initials[seed])
            source=selected[(seed,arm)] if arm!='frozen' else initials[seed]
            for key,value in source.items():
                if key.startswith('encoder.'):state[key]=value
            model=Model(copy.deepcopy(encoder)).to(args.device).eval()
            model,_=train(model,state,'probe_'+arm,seed,args.probe_steps,100,data['train'],data['development'],splits['development'],args.out,probe=True)
            selected[(seed,'probe_'+arm)]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    allrows=[];summary={};visual={}
    for seed in SEEDS:
        for arm in ('frozen','depth','structure','probe_frozen','probe_depth','probe_structure'):
            model=Model(copy.deepcopy(encoder)).to(args.device).eval();model.load_state_dict(selected[(seed,arm)])
            for split in ('seen','unseen'):
                rows,preds=evaluate(model,data[split],splits[split])
                summary[f'{seed}/{arm}/{split}']=macro(rows)
                for r in rows:allrows.append(dict(r,seed=seed,arm=arm,split=split))
                if seed==42 and split=='unseen' and not arm.startswith('probe_'):visual[arm]=preds
                if split=='unseen':
                    rows,_=evaluate(model,data[split],splits[split],shuffle=True)
                    summary[f'{seed}/{arm}/unseen_shuffled']=macro(rows)
            write(args.out/'evaluation.json',summary)
    # Training-only template: average each location over valid training observations.
    _,y,m=data['train'];template=(y*m).sum(0,keepdim=True)/m.sum(0,keepdim=True).clamp_min(1)
    for split in ('seen','unseen'):
        for name,value in [('zero',torch.zeros_like(template)),('mean_template',template)]:
            rows,_=evaluate(None,data[split],splits[split],constant=value)
            summary[f'baseline/{name}/{split}']=macro(rows)
            for r in rows:allrows.append(dict(r,seed=-1,arm=name,split=split))
    write(args.out/'evaluation.json',summary)
    with (args.out/'per_frame.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    torch.save({'predictions':visual,'samples':[{'object':s['object'],'frame':s['frame']} for s in splits['unseen']]},args.out/'visual_predictions.pth')
    make_figure(splits['unseen'],visual,args.out)
    lines=['# Extended BOP visible-geometry results','', 'Development-only LINEMOD study. Objects held out from this adaptation: 12,13,14,15. No pose-estimation claim.','', '| Arm | Unseen-object MAE / diameter (mean ± seed SD) | Unseen-object MAE mm (mean ± seed SD) |','|---|---:|---:|']
    for arm in ('frozen','depth','structure','probe_frozen','probe_depth','probe_structure'):
        a=[summary[f'{seed}/{arm}/unseen']['macro']['mae_diameter'] for seed in SEEDS]
        b=[summary[f'{seed}/{arm}/unseen']['macro']['mae_mm'] for seed in SEEDS]
        lines.append(f'| {arm} | {np.mean(a):.5f} ± {np.std(a,ddof=1):.5f} | {np.mean(b):.2f} ± {np.std(b,ddof=1):.2f} |')
    for name in ('zero','mean_template'):
        v=summary[f'baseline/{name}/unseen']['macro'];lines.append(f"| {name} | {v['mae_diameter']:.5f} | {v['mae_mm']:.2f} |")
    lines+=['','Fresh probes freeze each encoder and reinitialize a common decoder. Seed SD is variability, not a confidence interval. See evaluation.json for seen-object, slope, per-object, and feature-shuffling results.','', 'All checkpoint selection used development objects 10,11. The figure shows fixed first sampled frames of all four evaluation objects for seed 42; predictions are masked by GT for display. Uniform color is not used in this geometry experiment.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(args.out/'status.json',{'status':'complete','seeds':SEEDS,'arms':3,'adaptation_steps':args.steps,'probe_steps':args.probe_steps})
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
