import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('rotation',HERE.parent/'real_rotation_20261111/run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=r.c
DATA=Path('/home/Sahil/datasets/bop/lm')
ARMS=['frozen','geometry','geometry_symmetry','geometry_texture','combined']
SEEDS=[42,123,2026]

def write(path,value):
    c.ext.write(path,value)

def save(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def partition(obj,frame):
    scene=DATA/'lm_test_all/test'/f'{obj:06d}'
    if obj not in partition.cache:
        keys=sorted(json.loads((scene/'scene_gt.json').read_text()),key=int)
        partition.cache[obj]={int(k):i/len(keys) for i,k in enumerate(keys)}
    p=partition.cache[obj][int(frame)]
    return 'train' if p<.6 else 'development' if .65<=p<.8 else 'evaluation' if p>=.85 else 'buffer'
partition.cache={}

def prepare():
    g=torch.load(HERE.parent/'geometric_structure_20261111/completion/cache_120_96_f4eb1016.pt',map_location='cpu',weights_only=False)
    splits={s:[] for s in ['train','development','evaluation','excluded']}
    for samples in g['splits'].values():
        for s in samples:
            part=partition(s['object'],s['frame'])
            if s['object'] in [3,7]:
                if part=='evaluation':splits['excluded'].append(s)
            elif part!='buffer':splits[part].append(s)
    p=torch.load(HERE.parent/'real_rotation_20261111/cache_vertexcolor_v2_240_96.pt',map_location='cpu',weights_only=False)
    manifest=[]
    for split,samples in splits.items():
        manifest.extend(dict(task='geometry',split=split,object=s['object'],frame=s['frame']) for s in samples)
    for obj,d in p['objects'].items():
        allsamples=[s for samples in d['splits'].values() for s in samples]
        d['splits']={split:[s for s in allsamples if partition(obj,s['frame'])==split] for split in ['train','development','evaluation']}
        ids=[i for i,s in enumerate(d['references']) if partition(obj,s['frame'])=='train']
        d['references']=[d['references'][i] for i in ids];d['renders']=d['renders'][ids]
        assert len(ids)>1
        for split,samples in d['splits'].items():
            manifest.extend(dict(task='rotation',split=split,object=obj,frame=s['frame']) for s in samples)
    keys={split:{(s['object'],s['frame']) for s in manifest if s['split']==split} for split in ['train','development','evaluation','excluded']}
    assert not keys['train'] & (keys['development']|keys['evaluation']|keys['excluded'])
    assert not keys['development'] & keys['evaluation']
    for samples in splits.values():
        assert samples
        for s in samples:
            assert not (s['context'] & s['valid']).any()
            assert (s['x'][3:,s['valid'][0]]==0).all()
    write(HERE/'manifest.json',{'samples':manifest,'protocol_sha256':hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),'reference_frames':{obj:[s['frame'] for s in d['references']] for obj,d in p['objects'].items()}})
    return splits,p

class Model(c.Model):
    def __init__(self,encoder):
        super().__init__(encoder)
        self.texture=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,3,1))
    def forward(self,x):
        d,n,f=super().forward(x)
        return d,n,F.interpolate(self.texture(f),x.shape[-2:],mode='bilinear',align_corners=False),f

def texture_target(rgb):
    return rgb-F.avg_pool2d(rgb,9,1,4)

def corrupt(x,seed):
    gen=torch.Generator(device=x.device).manual_seed(seed)
    holes=F.interpolate((torch.rand((len(x),1,8,8),device=x.device,generator=gen)<.25).float(),x.shape[-2:],mode='nearest').bool()
    out=x.clone();out[:,:3]=out[:,:3].masked_fill(holes,0.5);out[:,3:]=out[:,3:].masked_fill(holes,0)
    return out,holes

def embed(model,x):
    return F.normalize(F.adaptive_avg_pool2d(model.encoder(x),(4,4)).flatten(1),dim=1)

@torch.no_grad()
def evaluate(model,samples,device,condition='standard'):
    rows=[]
    for start in range(0,len(samples),16):
        batch=samples[start:start+16];x,y,n,m,nm=c.tensor_data(batch,device)
        if condition=='grayscale':x[:,:3]=x[:,:3].mean(1,keepdim=True).expand(-1,3,-1,-1)
        if condition=='occlusion':x[:,:3,24:60,24:60]=.5;x[:,3:,24:60,24:60]=0
        if condition=='extra_dropout':
            gen=torch.Generator(device=device).manual_seed(777+start)
            mask=F.interpolate((torch.rand((len(x),1,8,8),generator=gen,device=device)<.3).float(),x.shape[-2:],mode='nearest').bool()
            x[:,3:]=x[:,3:].masked_fill(mask,0)
        d,pn,_,_=model(x)
        errors=c.ext.image_mean(abs(d-y),m)
        angles=torch.rad2deg(torch.acos((pn*n).sum(1,keepdim=True).clamp(-1,1)))
        tx,tm=corrupt(x,8100+start);_,_,pred,_=model(tx)
        target=texture_target(torch.stack([s['x'][:3] for s in batch]).to(device))
        # Exclude crop background using observed and supervised object pixels.
        valid=torch.stack([s['context']|s['valid'] for s in batch]).to(device)&tm
        tex=c.ext.image_mean((pred-target).abs().mean(1,keepdim=True),valid)
        for i,s in enumerate(batch):rows.append(dict(object=s['object'],frame=s['frame'],mae_diameter=float(errors[i]),mae_mm=float(errors[i])*s['diameter'],median_normal_angle_deg=float(angles[i][nm[i]].median()),texture_residual_mae=float(tex[i])))
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);device='cpu' if args.smoke else 'cuda'
    splits,p=prepare()
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(ROOT/'weights/2023-10-28-18-33-37/config.yml')
    base=c.pilot.RefineNet(cfg,c_in=cfg.c_in)
    base.load_state_dict(torch.load(ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    teacher=copy.deepcopy(base.encodeA).to(device).eval();del base
    for v in teacher.parameters():v.requires_grad_(False)
    tensors={obj:{'train':r.inputs([s['rgb'] for s in d['splits']['train']],device),'refs':r.inputs(d['renders'][:,0],device)} for obj,d in p['objects'].items()}
    labels={}
    for obj,d in p['objects'].items():
        a=r.distances(np.array([s['rotation'] for s in d['splits']['train']]),np.array([s['rotation'] for s in d['references']]),d['symmetries'])
        logits=-a*a/450;logits-=logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
        labels[obj]=torch.tensor(prob,dtype=torch.float32,device=device)
    out=HERE/('smoke' if args.smoke else 'results');out.mkdir(exist_ok=True)
    for seed in ([42] if args.smoke else SEEDS):
        torch.manual_seed(seed);initial=Model(copy.deepcopy(teacher)).state_dict();initial={k:v.cpu().clone() for k,v in initial.items()}
        for arm in ARMS:
            bestpath=out/f'{arm}_{seed}.pth';lastpath=out/f'{arm}_{seed}_last.pth';done=out/f'{arm}_{seed}_complete.json'
            if done.exists():continue
            model=Model(copy.deepcopy(teacher)).to(device).eval();model.load_state_dict(initial)
            tune=arm!='frozen';sym=arm in ['geometry_symmetry','combined'];tex=arm in ['geometry_texture','combined']
            for v in model.encoder.parameters():v.requires_grad_(tune)
            groups=[dict(params=list(model.decoder.parameters())+list(model.texture.parameters()),lr=1e-3)]
            if tune:groups.append(dict(params=model.encoder.parameters(),lr=1e-5))
            opt=torch.optim.AdamW(groups,weight_decay=1e-4)
            begin=0;best=float('inf');history=[]
            if lastpath.exists():
                ck=torch.load(lastpath,map_location=device,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);begin=ck['step'];best=ck['best'];history=ck['history']
            steps=1 if args.smoke else 3000
            for step in range(begin,steps):
                rng=np.random.default_rng(seed*100000+step);idx=rng.choice(len(splits['train']),min(8,len(splits['train'])),replace=False)
                batch=[splits['train'][i] for i in idx];x,y,n,m,nm=c.tensor_data(batch,device)
                corrupted,holes=corrupt(x,seed*100000+step)
                opt.zero_grad(set_to_none=True);d,pn,pt,f=model(corrupted)
                loss=c.loss_values(d,pn,y,n,m,nm,.2)
                if tex or not tune:
                    mask=holes&torch.stack([s['context']|s['valid'] for s in batch]).to(device)
                    loss=loss+.2*c.ext.image_mean((pt-texture_target(x[:,:3])).abs().mean(1,keepdim=True),mask).mean()
                if tune:
                    with torch.no_grad():tf=teacher(corrupted)
                    loss=loss+.01*F.mse_loss(F.normalize(f,dim=1),F.normalize(tf,dim=1))
                if sym:
                    obj=r.OBJECTS[step%len(r.OBJECTS)];t=tensors[obj];ids=rng.choice(len(t['train']),8,replace=False)
                    query=t['train'][ids].clone();query,unused=corrupt(query,seed*100000+step+1)
                    q=embed(model,query);ref=embed(model,t['refs'])
                    loss=loss-.1*(labels[obj][ids]*F.log_softmax(q@ref.T/.07,dim=1)).sum(1).mean()
                assert torch.isfinite(loss),'Nonfinite objective'
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if (step+1)%100==0 or step+1==steps:
                    dev=splits['development'][:8] if args.smoke else splits['development']
                    rows=evaluate(model,dev,device);score=c.summary(rows)['macro'];criterion=score['mae_diameter']+.2*score['median_normal_angle_deg']/180
                    history.append(dict(step=step+1,loss=float(loss.detach()),**score))
                    if criterion<best:
                        best=criterion;save(bestpath,dict(model=model.state_dict(),seed=seed,arm=arm,step=step+1,score=score))
                    save(lastpath,dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,best=best,history=history))
                    write(out/f'{arm}_{seed}_history.json',history)
                    write(HERE/'progress.json',dict(stage='smoke' if args.smoke else 'training',arm=arm,seed=seed,step=step+1,total_steps=steps,updated_unix=time.time(),development=score))
                    print(seed,arm,step+1,score,flush=True)
            selected=torch.load(bestpath,map_location=device,weights_only=False);model.load_state_dict(selected['model'])
            assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items() if ('running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k))
            if tune:assert any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items() if k.startswith('encoder.'))
            else:assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items() if k.startswith('encoder.'))
            results={}
            for split in ['evaluation','excluded']:
                for condition in (['standard'] if args.smoke else ['standard','grayscale','occlusion','extra_dropout']):
                    samples=splits[split][:4] if args.smoke else splits[split]
                    rows=evaluate(model,samples,device,condition);results[f'{split}/{condition}']={'summary':c.summary(rows),'rows':rows}
            write(done,results)
    write(out/'status.json',dict(state='complete',smoke=args.smoke,time=time.time()))
if __name__=='__main__':main()
