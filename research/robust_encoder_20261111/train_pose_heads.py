"""Supervised refinement training using production crop and pose-update conventions."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import argparse, copy, json, logging
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
import train as t
from learning.training.predict_pose_refine import PoseRefinePredictor, make_crop_data_batch
from datareader import LinemodReader
from Utils import depth2xyzmap, make_mesh_tensors, so3_exp_map, egocentric_delta_pose_to_pose
import nvdiffrast.torch as dr
HERE=Path(__file__).resolve().parent
ARMS=['heads_only','joint_original','joint_combined','frozen_combined_heads']

def predicted_pose(output,initial,diameter,cfg):
    assert cfg['normalize_xyz'] and cfg['trans_rep']=='tracknet' and cfg['rot_rep']=='axis_angle'
    delta=output['trans']*(diameter[:,None]/2)
    rot=so3_exp_map(torch.tanh(output['rot'])*cfg['rot_normalizer']).permute(0,2,1)
    return egocentric_delta_pose_to_pose(initial,delta,rot)

def pose_loss(pred,targets,vertices,diameter):
    # Each target is GT @ one full BOP symmetry in original CAD coordinates.
    p=torch.einsum('bij,bnj->bni',pred[:,:3,:3],vertices)+pred[:,None,:3,3]
    q=torch.einsum('bsij,bnj->bsni',targets[:,:,:3,:3],vertices)+targets[:,:,None,:3,3]
    error=(p[:,None]-q).norm(dim=-1).mean(-1)/diameter[:,None]
    return error.min(1).values

@torch.inference_mode()
def prepare(predictor,smoke=False):
    path=HERE/'pose_pairs.pt'
    if path.exists() and not smoke:return torch.load(path,map_location='cpu',weights_only=False)
    context=dr.RasterizeCudaContext();splits={'train':[],'development':[]};manifest=[]
    source=torch.load(HERE.parent/'real_rotation_20261111/cache_vertexcolor_v2_240_96.pt',map_location='cpu',weights_only=False)
    for obj in (t.r.OBJECTS[:1] if smoke else t.r.OBJECTS):
        reader=LinemodReader(str(t.DATA/'lm_test_all/test'/f'{obj:06d}'),split=None)
        reader.K=reader.K.astype(np.float32)
        mesh=reader.get_gt_mesh(obj);mesh_tensors=make_mesh_tensors(mesh)
        diameter=reader.get_model_diameter(obj)
        vertices=np.asarray(mesh.vertices);vertices=vertices[np.linspace(0,len(vertices)-1,256,dtype=int)].astype(np.float32)
        syms=np.asarray(reader.symmetry_tfs[obj]);assert len(syms)<=64
        allsamples=[s for ss in source['objects'][obj]['splits'].values() for s in ss]
        for split in splits:
            samples=sorted([s for s in allsamples if t.partition(obj,s['frame'])==split],key=lambda s:int(s['frame']))
            n=(1 if smoke else 24 if split=='train' else 8)
            samples=[samples[i] for i in np.linspace(0,len(samples)-1,n,dtype=int)]
            for sample in samples:
                frame=sample['frame'];index=reader.id_strs.index(frame)
                rgb=reader.get_color(index);depth=reader.get_depth(index);gt=reader.get_gt_pose(index,obj)
                rng=np.random.default_rng(obj*100000+int(frame))
                # Near-pose refinement starts: rotation errors within the model's 20 degree update range.
                initial=np.repeat(gt[None],2 if smoke else 4,axis=0)
                for p in initial:
                    axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
                    p[:3,:3]=Rotation.from_rotvec(axis*np.deg2rad(rng.uniform(5,18))).as_matrix()@p[:3,:3]
                    p[:3,3]+=rng.uniform(-.04,.04,3)*diameter
                # Production crop processing, including pose-relative XYZ normalization.
                batch=make_crop_data_batch(predictor.cfg.input_resize,torch.as_tensor(initial,device="cuda",dtype=torch.float32),mesh,rgb,depth,reader.K,crop_ratio=predictor.cfg.crop_ratio,xyz_map=depth2xyzmap(depth,reader.K),cfg=predictor.cfg,glctx=context,mesh_tensors=mesh_tensors,dataset=predictor.dataset,mesh_diameter=diameter)
                A=torch.cat([batch.rgbAs,batch.xyz_mapAs],1).cpu().half()
                B=torch.cat([batch.rgbBs,batch.xyz_mapBs],1).cpu().half()
                for i in range(len(initial)):
                    splits[split].append(dict(A=A[i],B=B[i],initial=torch.tensor(initial[i],dtype=torch.float32),targets=torch.tensor(gt[None]@syms,dtype=torch.float32),vertices=torch.from_numpy(vertices),diameter=diameter,object=obj,frame=frame))
                manifest.append(dict(object=obj,frame=frame,split=split,perturbations=len(initial)))
                print('PAIRS',obj,split,frame,flush=True)
    assert not {(r['object'],r['frame']) for r in manifest if r['split']=='train'}&{(r['object'],r['frame']) for r in manifest if r['split']=='development'}
    result=dict(splits=splits,manifest=manifest)
    if not smoke:t.save(path,result);t.write(HERE/'pose_pair_manifest.json',manifest)
    return result

def batch(samples,device):
    maxsym=max(len(s['targets']) for s in samples)
    targets=[torch.cat([s['targets'],s['targets'][-1:].expand(maxsym-len(s['targets']),-1,-1)]) for s in samples]
    return (torch.stack([s['A'] for s in samples]).to(device).float(),torch.stack([s['B'] for s in samples]).to(device).float(),torch.stack([s['initial'] for s in samples]).to(device),torch.stack(targets).to(device),torch.stack([s['vertices'] for s in samples]).to(device),torch.tensor([s['diameter'] for s in samples],device=device))

def corrupt(B,seed):
    g=torch.Generator(device=B.device).manual_seed(seed);B=B.clone()
    missing=F.interpolate((torch.rand((len(B),1,8,8),device=B.device,generator=g)<.4).float(),B.shape[-2:],mode='nearest').bool()
    B[:,3:]=B[:,3:].masked_fill(missing,0)
    noise=torch.randn(B[:,3:].shape,device=B.device,generator=g)*.02
    B[:,3:]+=noise*(B[:,3:]!=0)
    if seed%3==0:B[:,:3,40:90,40:90]=0;B[:,3:,40:90,40:90]=0
    if seed%5==0:B[:,:3]=B[:,:3].mean(1,keepdim=True).expand(-1,3,-1,-1)
    return B

@torch.no_grad()
def evaluate(model,samples,cfg,device):
    values=[]
    for start in range(0,len(samples),4):
        A,B,initial,targets,vertices,diameter=batch(samples[start:start+4],device)
        for condition in ['observed','corrupted']:
            observed=B if condition=='observed' else corrupt(B,99000+start)
            pred=predicted_pose(model(A,observed),initial,diameter,cfg)
            error=pose_loss(pred,targets,vertices,diameter)
            for i,s in enumerate(samples[start:start+4]):values.append(dict(object=s['object'],frame=s['frame'],condition=condition,error_diameter=float(error[i])))
    # Equal object and condition weights, irrespective of valid-pair counts.
    return float(np.mean([np.mean([v['error_diameter'] for v in values if v['object']==obj and v['condition']==condition]) for obj in sorted({v['object'] for v in values}) for condition in ['observed','corrupted']])),values

def smoke():
    torch.set_num_threads(4);torch.manual_seed(42)
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(t.ROOT/'weights/2023-10-28-18-33-37/config.yml')
    model=t.c.pilot.RefineNet(cfg,c_in=cfg.c_in).eval()
    model.load_state_dict(torch.load(t.ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    A=torch.rand(1,6,160,160);B=corrupt(A,42)
    initial=torch.eye(4)[None];diameter=torch.tensor([.2]);vertices=torch.randn(1,32,3)*.1
    targets=initial[:,None].clone();targets[:,:,0,3]=.01
    pred=predicted_pose(model(A,B),initial,diameter,cfg);loss=pose_loss(pred,targets,vertices,diameter).mean();loss.backward()
    for prefix in ['encodeA','encodeAB','trans_head','rot_head']:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(prefix)),prefix
    assert pose_loss(targets[:,0],targets,vertices,diameter).max()<1e-6
    # Full rigid symmetry target, including a translated symmetry axis.
    symmetry=torch.eye(4);symmetry[0,0]=-1;symmetry[1,1]=-1;symmetry[0,3]=.02
    equivalent=targets[:,0]@symmetry
    both=torch.cat([targets,equivalent[:,None]],1)
    assert pose_loss(equivalent,both,vertices,diameter).max()<1e-6
    print('PASS: real-size forward/backward; finite nonzero encoder/fusion/translation/rotation gradients; exact-pose and full symmetry loss',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--pair-smoke',action='store_true');parser.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS);args=parser.parse_args()
    if args.smoke:smoke();return
    torch.set_num_threads(4);logging.getLogger().setLevel(logging.WARNING)
    predictor=PoseRefinePredictor();predictor.model.eval();prepared=prepare(predictor,args.pair_smoke)
    if args.pair_smoke:
        model=predictor.model;A,B,initial,targets,vertices,diameter=batch(prepared['splits']['train'][:2],'cuda')
        model.zero_grad(set_to_none=True);loss=pose_loss(predicted_pose(model(A,corrupt(B,42)),initial,diameter,predictor.cfg),targets,vertices,diameter).mean();loss.backward()
        assert torch.isfinite(loss);assert model.trans_head[-1].weight.grad.abs().sum()>0;assert model.rot_head[-1].weight.grad.abs().sum()>0
        t.write(HERE/'pose_pair_smoke.json',dict(passed=True,loss=float(loss.detach()),pairs=len(A)))
        print('PASS real BOP pair training',flush=True);return
    original={k:v.cpu().clone() for k,v in predictor.model.state_dict().items()};out=HERE/'heads';out.mkdir(exist_ok=True)
    for seed in t.SEEDS:
        for arm in args.arms:
            done=out/f'{arm}_{seed}_complete.json'
            if done.exists():continue
            model=predictor.model;model.load_state_dict(original);model.eval()
            if arm in ['joint_combined','frozen_combined_heads']:
                checkpoint=torch.load(HERE/'results'/f'combined_{seed}.pth',map_location='cpu',weights_only=False)['model']
                model.encodeA.load_state_dict({k.removeprefix('encoder.'):v for k,v in checkpoint.items() if k.startswith('encoder.')})
            torch.manual_seed(seed)
            frozen_encoder={k:v.cpu().clone() for k,v in model.encodeA.state_dict().items()}
            for name,param in model.named_parameters():param.requires_grad_(arm not in ['heads_only','frozen_combined_heads'] or not name.startswith('encodeA.'))
            model.zero_grad(set_to_none=True)
            optimizer=torch.optim.AdamW([dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encodeA.')],lr=3e-6),dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encodeA.')],lr=1e-5)],weight_decay=1e-4)
            best=float('inf');start=0;history=[];last=out/f'{arm}_{seed}_last.pth';bestpath=out/f'{arm}_{seed}.pth'
            if last.exists():
                state=torch.load(last,map_location='cuda',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];best=state['best'];history=state['history']
            for step in range(start,1500):
                rng=np.random.default_rng(seed*100000+step);ids=rng.choice(len(prepared['splits']['train']),4,replace=False)
                samples=[prepared['splits']['train'][i] for i in ids];A,B,initial,targets,vertices,diameter=batch(samples,'cuda')
                # Alternate observed and corrupted input; matching across all arms.
                if step%2:B=corrupt(B,seed*100000+step)
                optimizer.zero_grad(set_to_none=True)
                pred=predicted_pose(model(A,B),initial,diameter,predictor.cfg)
                loss=pose_loss(pred,targets,vertices,diameter).mean()
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                if (step+1)%100==0:
                    score,rows=evaluate(model,prepared['splits']['development'],predictor.cfg,'cuda')
                    history.append(dict(step=step+1,development_pose_error_diameter=score,train_loss=float(loss.detach())))
                    if score<best:
                        best=score;t.save(bestpath,dict(model=model.state_dict(),seed=seed,arm=arm,step=step+1,development=score))
                    t.save(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,best=best,history=history))
                    t.write(out/f'{arm}_{seed}_history.json',history);t.write(HERE/'progress.json',dict(stage='pose_head_training',arm=arm,seed=seed,step=step+1,total_steps=1500,development=score))
                    print('HEADS',arm,seed,step+1,score,flush=True)
            chosen=torch.load(bestpath,map_location='cpu',weights_only=False)['model']
            encoder_unchanged=all(torch.equal(chosen['encodeA.'+k].cpu(),v) for k,v in frozen_encoder.items())
            if arm in ['heads_only','frozen_combined_heads']:
                assert encoder_unchanged, 'Frozen encoder changed'
            for prefix in ['trans_head','rot_head']:
                assert any(not torch.equal(v.cpu(),original[k]) for k,v in chosen.items() if k.startswith(prefix))
            assert all(torch.equal(v.cpu(),original[k]) for k,v in chosen.items() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k)
            t.write(done,dict(best_development_error_diameter=best,history=history,encoder_unchanged=encoder_unchanged))
    t.write(out/'status.json',dict(state='complete'))
if __name__=='__main__':main()
