"""Persistent, checkpointed real-pair encoder study with a twelve-hour deadline."""
import argparse
import csv
import fcntl
import hashlib
import json
import logging
import sqlite3
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import estimater as estimator_module
from estimater import FoundationPose
from datareader import LinemodReader
from learning.training.predict_pose_refine import PoseRefinePredictor
from learning.training.predict_score import ScorePredictor
from Utils import dr,erode_depth,bilateral_filter_depth,depth2xyzmap,set_seed

DATA=Path('/home/Sahil/datasets/bop/lm')
OBJECTS=[1,2,4,5,6,8,9,10,11,12,13,14,15]
CONDITIONS=['clean','rgb_half','rgb_blur3','depth_noise5mm','depth_dropout30']
ABLATIONS=['intact','zero_observed_xyz','zero_observed_rgb','pooled_shared','zero_shared','shuffle_shared','shuffle_fusion']


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def stamp():return datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_summary(db,key,result):
    before=result['initial_errors'];after=result['refined_errors'];idx=result['top_index'];frame=result['frame']
    ties=np.flatnonzero(np.asarray(result['ranking_logits'])==max(result['ranking_logits']))
    summary=dict(key=key,obj_id=frame['obj_id'],im_id=frame['im_id'],split=frame['split'],seed=result['seed'],condition=result['condition'],ablation=result['ablation'],
                 initial_translation_mm=float(np.mean(before['translation_mm'][1:])),
                 refined_translation_mm=float(np.mean(after['translation_mm'][1:])),
                 initial_rotation_deg=float(np.mean(before['symmetry_rotation_deg'][1:])),
                 refined_rotation_deg=float(np.mean(after['symmetry_rotation_deg'][1:])),
                 initial_adds_mm=float(np.mean(before['sampled_adds_mm'][1:])),
                 refined_adds_mm=float(np.mean(after['sampled_adds_mm'][1:])),
                 improvement_fraction=float(np.mean(np.asarray(after['sampled_adds_mm'][1:])<np.asarray(before['sampled_adds_mm'][1:]))),
                 selected_adds_mm=float(np.mean(np.asarray(before['sampled_adds_mm'])[ties])),
                 selected_rotation_deg=float(np.mean(np.asarray(before['symmetry_rotation_deg'])[ties])),
                 selected_gt=float(0 in ties)/len(ties),ranking_tie_count=len(ties),
                 ranking_spearman=result['ranking_quality_spearman'],ranking_spearman_defined=int(result['ranking_quality_spearman'] is not None))
    for kind in ['refine','score']:
        feature=result[kind+'_features']['shared_3']
        for metric in ['spatial','pooled']:
            x=np.asarray(feature[metric]);y=np.asarray(before['sampled_adds_mm'])
            value=float(spearmanr(x,y).statistic) if np.ptp(x)>1e-10 else None
            summary[f'{kind}_{metric}_error_spearman']=value if value is not None and np.isfinite(value) else None
    db.execute('INSERT OR REPLACE INTO units (key, payload) VALUES (?, ?)',(key,json.dumps(summary,allow_nan=False)));db.commit()


def candidates(gt,seed):
    basis=np.eye(3) if seed==0 else Rotation.random(random_state=seed).as_matrix()
    poses=[gt.copy()];labels=['gt']
    for axis in range(3):
        for mm in [5,10,20]:
            for sign in [-1,1]:
                p=gt.copy();p[:3,3]+=basis[:,axis]*(sign*mm/1000);poses.append(p);labels.append(f't{axis}_{sign*mm}mm')
    for axis in range(3):
        for deg in [5,15,30]:
            for sign in [-1,1]:
                p=gt.copy();p[:3,:3]=Rotation.from_rotvec(basis[:,axis]*np.radians(sign*deg)).as_matrix()@gt[:3,:3];poses.append(p);labels.append(f'r{axis}_{sign*deg}deg')
    return np.asarray(poses,dtype=np.float32),labels


class Probe:
    def __init__(self,model,kind,ablation,seed):
        self.model=model;self.kind=kind;self.ablation=ablation;self.seed=seed;self.handles=[];self.records={}
    def __enter__(self):
        self.handles.append(self.model.register_forward_pre_hook(self.input_hook))
        shared=self.model.encodeA if self.kind=='refine' else self.model.encoderA
        fusion=self.model.encodeAB if self.kind=='refine' else self.model.encoderAB
        for i,layer in enumerate(shared):self.handles.append(layer.register_forward_hook(self.shared_hook(i)))
        for i,layer in enumerate(fusion):self.handles.append(layer.register_forward_hook(self.fusion_hook(i)))
        return self
    def input_hook(self,module,args):
        if self.ablation not in ['zero_observed_rgb','zero_observed_xyz']:return None
        args=list(args);b=args[1].clone()
        if self.ablation=='zero_observed_rgb':b[:,:3]=0
        if self.ablation=='zero_observed_xyz':b[:,3:6]=0
        args[1]=b;return tuple(args)
    def permute(self,x):
        generator=torch.Generator(device=x.device);generator.manual_seed(7300+self.seed)
        order=torch.randperm(x.shape[-2]*x.shape[-1],device=x.device,generator=generator)
        return x.flatten(2)[:,:,order].reshape_as(x)
    def shared_hook(self,stage):
        def hook(module,args,out):
            if stage==3 and self.ablation in ['pooled_shared','zero_shared','shuffle_shared']:
                out=out.clone();n=len(out)//2;b=out[n:]
                if self.ablation=='pooled_shared':out[n:]=b.mean((2,3),keepdim=True).expand_as(b)
                elif self.ablation=='zero_shared':out[n:]=0
                else:out[n:]=self.permute(b)
            n=len(out)//2;a,b=out[:n].float(),out[n:].float()
            spatial=1-torch.nn.functional.cosine_similarity(a.flatten(1),b.flatten(1),dim=1)
            pooled=1-torch.nn.functional.cosine_similarity(a.mean((2,3)),b.mean((2,3)),dim=1)
            self.records[f'shared_{stage}']=dict(spatial=spatial.detach().cpu().tolist(),pooled=pooled.detach().cpu().tolist())
            return out
        return hook
    def fusion_hook(self,stage):
        def hook(module,args,out):
            if stage==4 and self.ablation=='shuffle_fusion':out=self.permute(out)
            self.records[f'fusion_{stage}']=dict(pooled_norm=out.float().mean((2,3)).norm(dim=1).detach().cpu().tolist())
            return out
        return hook
    def __exit__(self,*args):
        for h in self.handles:h.remove()


def errors(poses,gt,points,symmetries,obj):
    poses=np.asarray(poses);assert np.isfinite(poses).all()
    assert np.allclose(poses[:,3,:],np.array([0,0,0,1]),atol=1e-4)
    assert np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=2e-3)
    trans=np.linalg.norm(poses[:,:3,3]-gt[:3,3],axis=1)*1000
    raw=np.degrees(Rotation.from_matrix(poses[:,:3,:3]@gt[:3,:3].T).magnitude())
    symrot=np.stack([np.degrees(Rotation.from_matrix(poses[:,:3,:3]@(gt@s)[:3,:3].T).magnitude()) for s in symmetries]).min(0)
    gtpts=points@gt[:3,:3].T+gt[:3,3]
    adds=[];adi=[]
    for p in poses:
        pred=points@p[:3,:3].T+p[:3,3]
        adds.append(float(np.linalg.norm(pred-gtpts,axis=1).mean()*1000))
        adi.append(float(cKDTree(pred).query(gtpts)[0].mean()*1000))
    return dict(translation_mm=trans.tolist(),raw_rotation_deg=raw.tolist(),symmetry_rotation_deg=symrot.tolist(),
                sampled_add_mm=adds,sampled_adi_mm=adi,sampled_adds_mm=adi if obj in [10,11] else adds)


def frames():
    split=ROOT/'debug/bop_recovery_20261021/analysis/proposed_mask_study_split.csv'
    selected=list(csv.DictReader(split.open()));seen={(int(r['obj_id']),int(r['im_id'])) for r in selected}
    preferred={o:[dict(obj_id=o,im_id=int(r['im_id']),split=r['split']) for r in selected if int(r['obj_id'])==o] for o in OBJECTS}
    remaining={}
    for obj in OBJECTS:
        scene=DATA/f'lm_test_all/test/{obj:06d}'
        ids=sorted(int(p.stem) for p in (scene/'rgb').glob('*') if p.suffix.lower() in ['.png','.jpg'])
        remaining[obj]=[dict(obj_id=obj,im_id=i,split='additional_diagnostic') for i in ids if (obj,i) not in seen]
    def balanced(groups):
        return [groups[obj][i] for i in range(max(map(len,groups.values()))) for obj in OBJECTS if i<len(groups[obj])]
    # Complete 13-object microcohorts before expanding to more frames.
    first=balanced(preferred);rest=balanced(remaining)
    return [first[i:i+13] for i in range(0,len(first),13)]+[rest[i:i+13] for i in range(0,len(rest),13)]


class Experiment:
    def __init__(self,out):
        self.out=out;self.refiner=PoseRefinePredictor();self.scorer=ScorePredictor();self.glctx=dr.RasterizeCudaContext()
        self.cache={};self.frame_cache={}
    def context(self,obj):
        if obj in self.cache:return self.cache[obj]
        set_seed(12345+obj)
        reader=LinemodReader(str(DATA/f'lm_test_all/test/{obj:06d}'),split=None)
        mesh=reader.get_gt_mesh(obj)
        est=FoundationPose(model_pts=mesh.vertices.copy(),model_normals=mesh.vertex_normals.copy(),mesh=mesh,
                           symmetry_tfs=reader.symmetry_tfs[obj],scorer=self.scorer,refiner=self.refiner,glctx=self.glctx,
                           debug=0,debug_dir=str(self.out/'object_debug'/str(obj)))
        # Torch infers float64 from NumPy scalar offsets in upstream crop code.
        est.diameter=float(est.diameter)
        center=np.eye(4);center[:3,3]=est.model_center
        syms=np.asarray(reader.symmetry_tfs[obj]);syms=np.linalg.inv(center)[None]@syms@center[None]
        rng=np.random.default_rng(987+obj);points=est.mesh.vertices[rng.choice(len(est.mesh.vertices),min(512,len(est.mesh.vertices)),replace=False)]
        value=(est,center,syms,points)
        atomic(self.out/'objects'/f'{obj:06d}.json',dict(obj_id=obj,mesh_center=est.model_center.tolist(),inference_diameter_m=est.diameter,
                                                      mesh_sha256=digest(reader.get_gt_mesh_file(obj)),sampled_vertices=len(points)))
        self.cache[obj]=value;return value
    def image(self,row,condition,seed):
        obj,im=row['obj_id'],row['im_id'];key=(obj,im,condition,seed)
        if key in self.frame_cache:return self.frame_cache[key]
        scene=DATA/f'lm_test_all/test/{obj:06d}'
        cam=json.loads((scene/'scene_camera.json').read_text())[str(im)]
        g=next(g for g in json.loads((scene/'scene_gt.json').read_text())[str(im)] if g['obj_id']==obj)
        path=next((scene/'rgb').glob(f'{im:06d}.*'))
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        depth=cv2.imread(str(scene/f'depth/{im:06d}.png'),-1).astype(np.float32)*cam['depth_scale']/1000
        depth[(depth<.001)|(depth>100)]=0
        rng=np.random.default_rng(seed*10000000+obj*100000+im)
        if condition=='rgb_half':rgb=(rgb.astype(float)*.5).astype(np.uint8)
        elif condition=='rgb_blur3':rgb=cv2.GaussianBlur(rgb,(0,0),3)
        elif condition=='depth_noise5mm':
            valid=depth>=.001;depth[valid]+=rng.normal(0,.005,int(valid.sum()));depth[depth<.001]=0
        elif condition=='depth_dropout30':depth[rng.random(depth.shape)<.3]=0
        depth=bilateral_filter_depth(erode_depth(depth,radius=2,device='cuda'),radius=2,device='cuda')
        K=np.asarray(cam['cam_K'],dtype=np.float32).reshape(3,3)
        gt=np.eye(4);gt[:3,:3]=np.asarray(g['cam_R_m2c']).reshape(3,3);gt[:3,3]=np.asarray(g['cam_t_m2c'])/1000
        value=(rgb,depth,K,gt,depth2xyzmap(depth,K))
        if len(self.frame_cache)>26:self.frame_cache.clear()
        self.frame_cache[key]=value;return value
    @torch.inference_mode()
    def unit(self,row,condition,ablation,seed): 
        started=time.time();obj=row['obj_id'];set_seed(seed)
        est,center,syms,points=self.context(obj)
        rgb,depth,K,gt_original,xyz=self.image(row,condition,seed)
        gt=gt_original@center;poses,labels=candidates(gt,seed)
        initial=errors(poses,gt,points,syms,obj)
        args=dict(rgb=rgb,depth=depth,K=K,ob_in_cams=poses,mesh=est.mesh,mesh_tensors=est.mesh_tensors,
                  glctx=self.glctx,mesh_diameter=est.diameter,get_vis=False)
        with Probe(self.refiner.model,'refine',ablation,seed) as refine_probe:
            refined,_=self.refiner.predict(**args,xyz_map=xyz,iteration=1)
        with Probe(self.scorer.model,'score',ablation,seed) as score_probe:
            scores,_=self.scorer.predict(**args)
        refined=refined.detach().cpu().numpy();scores=scores.detach().cpu().numpy()
        if not np.isfinite(scores).all():raise RuntimeError('Nonfinite ranking logits')
        after=errors(refined,gt,points,syms,obj)
        correlation=float(spearmanr(scores,-np.asarray(initial['sampled_adds_mm'])).statistic)
        return dict(frame=row,condition=condition,ablation=ablation,seed=seed,candidates=labels,
                    initial_errors=initial,refined_errors=after,ranking_logits=scores.tolist(),top_index=int(np.argmax(scores)),
                    ranking_quality_spearman=correlation if np.isfinite(correlation) else None,
                    refine_features=refine_probe.records,score_features=score_probe.records,
                    candidate_poses_centered=poses.tolist(),refined_poses_centered=refined.tolist(),
                    finite_validated=True,elapsed_seconds=time.time()-started,completed_at=stamp())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=12);ap.add_argument('--pilot-units',type=int,default=0)
    ap.add_argument('--output',type=Path,default=HERE/'production');args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=(ROOT/'research/.foundationpose_inference.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logging.getLogger().setLevel(logging.ERROR);torch.set_num_threads(4)
    assert torch.cuda.is_available(),'CUDA is not available in this execution environment'
    sources=[Path(__file__),HERE/'PROTOCOL.md',HERE/'report_gpu.py']+list((ROOT/'learning/models').glob('*.py'))+list((ROOT/'weights').glob('*/*'))
    sources += [ROOT/'estimater.py',ROOT/'Utils.py',ROOT/'datareader.py',ROOT/'learning/training/predict_pose_refine.py',ROOT/'learning/training/predict_score.py',ROOT/'learning/datasets/h5_dataset.py']
    manifest=dict(hashes={str(p):digest(p) for p in sources if p.is_file()},conditions=CONDITIONS,ablations=ABLATIONS,
                  gpu=torch.cuda.get_device_name(),torch=torch.__version__,scope='Local GT-conditioned real-pair encoder interventions')
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Source mismatch; use a new directory')
    atomic(mp,manifest)
    state_path=out/'status.json';old=json.loads(state_path.read_text()) if state_path.exists() else {}
    started=old.get('started_unix',time.time());deadline=old.get('deadline_unix',started+args.hours*3600)
    db=sqlite3.connect(out/'summary.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS units (key TEXT PRIMARY KEY, payload TEXT NOT NULL)');db.commit()
    completed=old.get('completed_units',0);failed=old.get('failed_units',0);consecutive_failures=0
    def status(state,**kw):
        atomic(state_path,dict(state=state,started_unix=started,deadline_unix=deadline,
               started_at=datetime.fromtimestamp(started,ZoneInfo('America/New_York')).isoformat(),
               deadline_at=datetime.fromtimestamp(deadline,ZoneInfo('America/New_York')).isoformat(),
               updated_at=stamp(),completed_units=completed,failed_units=failed,elapsed_seconds=time.time()-started,**kw))
    status('initializing');exp=Experiment(out);cohorts=frames();last_report=0
    if args.pilot_units:cohorts=[cohorts[0][:1]]
    try:
        for seed in range(100000):
            for cohort_idx,cohort in enumerate(cohorts):
                for condition in CONDITIONS:
                    for ablation in ABLATIONS:
                        for row in cohort:
                            if time.time()>=deadline or (args.pilot_units and completed>=args.pilot_units):
                                status('pilot_complete' if args.pilot_units else 'measurement_budget_complete');return
                            key=f'{row["obj_id"]:06d}_{row["im_id"]:06d}_{seed}_{condition}_{ablation}'
                            path=out/'units'/f'{key}.json'
                            if path.exists():
                                if db.execute('SELECT 1 FROM units WHERE key=?',(key,)).fetchone() is None:save_summary(db,key,json.loads(path.read_text()))
                                continue
                            try:
                                result=exp.unit(row,condition,ablation,seed);atomic(path,result);save_summary(db,key,result);completed+=1;consecutive_failures=0
                            except Exception as exc:
                                failed+=1;consecutive_failures+=1;atomic(out/'failures'/f'{key}.json',dict(frame=row,condition=condition,ablation=ablation,seed=seed,error=repr(exc),traceback=traceback.format_exc()))
                                if consecutive_failures>=3:raise RuntimeError('Three consecutive experiment units failed; inspect failures before continuation') from exc
                            status('running',current=dict(cohort=cohort_idx,seed=seed,condition=condition,ablation=ablation,frame=row))
                            print(f'{stamp()} complete={completed} failed={failed} {key}',flush=True)
                            if not args.pilot_units and time.time()-last_report>900:
                                from report_gpu import report
                                report(out,final=False);last_report=time.time()
    except Exception as exc:
        status('failed',error=repr(exc));raise
    finally:
        if not args.pilot_units:
            from report_gpu import report
            report(out,final=True)


if __name__=='__main__':main()
