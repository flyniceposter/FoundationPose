"""CPU observation-encoder diagnostics; see PROTOCOL.md for claim boundaries."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import fcntl

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path('/home/Sahil/datasets/bop/lm')
SPLIT = ROOT/'debug/bop_recovery_20261021/analysis/proposed_mask_study_split.csv'
RUNS = {'refine': '2023-10-28-18-33-37', 'score': '2024-01-11-20-02-45'}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def classes(path, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(tree, str(path), 'exec'), namespace)


def load_model(kind, trained):
    ns = dict(torch=torch, nn=nn, math=math, np=np)
    classes(ROOT/'learning/models/network_modules.py', ns)
    classes(ROOT/f'learning/models/{"refine" if kind=="refine" else "score"}_network.py', ns)
    cfg = OmegaConf.load(ROOT/'weights'/RUNS[kind]/'config.yml')
    torch.manual_seed(20261028)
    model = ns['RefineNet' if kind=='refine' else 'ScoreNetMultiPair'](cfg=cfg, c_in=cfg.c_in)
    if trained:
        ckpt = torch.load(ROOT/'weights'/RUNS[kind]/'model_best.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt.get('model', ckpt), strict=True)
    return model.eval(), cfg


def conditions():
    c = [('clean', 0)]
    for name in ['brightness_0.5', 'brightness_1.5', 'blur_1', 'blur_3', 'zero_rgb', 'zero_xyz',
                 'rgb_shift_2', 'rgb_shift_5', 'xyz_shift_2', 'xyz_shift_5', 'joint_shift_2', 'joint_shift_5', 'occlusion_0.1', 'occlusion_0.3']:
        c.append((name, 0))
    for name in ['rgb_noise_0.02', 'rgb_noise_0.1', 'depth_noise_1', 'depth_noise_5', 'depth_noise_10',
                 'depth_dropout_0.1', 'depth_dropout_0.3', 'depth_dropout_0.5']:
        c += [(name, seed) for seed in range(3)]
    return c


def frame_inputs(row, cfg, kind):
    obj, im = int(row['obj_id']), int(row['im_id'])
    scene = DATA/f'lm_test_all/test/{obj:06d}'
    gt = json.loads((scene/'scene_gt.json').read_text())[str(im)]
    gt = next(x for x in gt if x['obj_id']==obj)
    cam = json.loads((scene/'scene_camera.json').read_text())[str(im)]
    rgb_path = next((scene/'rgb').glob(f'{im:06d}.*'))
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB).astype(np.float32)/255
    depth = cv2.imread(str(scene/f'depth/{im:06d}.png'), -1).astype(np.float32)*cam['depth_scale']/1000
    K = np.array(cam['cam_K'], dtype=np.float32).reshape(3,3)
    t = np.array(gt['cam_t_m2c'], dtype=np.float32)/1000
    info = json.loads((DATA/'models/models_info.json').read_text())[str(obj)]
    model_center = np.array([info['min_'+a]+info['size_'+a]/2 for a in 'xyz'], dtype=np.float32)/1000
    t = t + np.array(gt['cam_R_m2c'], dtype=np.float32).reshape(3,3) @ model_center
    diameter = info['diameter']/1000
    center = (K@t)[:2]/t[2]
    radius_px = max(K[0,0], K[1,1])*diameter*cfg.crop_ratio/(2*t[2])
    left, top = np.round(center-radius_px)
    right, bottom = np.round(center+radius_px)
    tf = torch.tensor([[160/(right-left),0,-left*160/(right-left)], [0,160/(bottom-top),-top*160/(bottom-top)], [0,0,1]], dtype=torch.float32)[None]
    def warp(x, mode):
        return kornia.geometry.transform.warp_perspective(x, tf, (160,160), mode=mode, align_corners=False)[0].numpy()
    crop_rgb = warp(torch.from_numpy(rgb).permute(2,0,1)[None], 'bilinear')
    h,w = depth.shape
    y,x = np.mgrid[:h,:w].astype(np.float32)
    rays = np.stack(((x-K[0,2])/K[0,0], (y-K[1,2])/K[1,1], np.ones_like(x)))
    crop_rays = warp(torch.from_numpy(rays)[None], 'nearest')
    crop_depth = warp(torch.from_numpy(depth)[None,None], 'nearest')[0]
    return crop_rgb, crop_depth, crop_rays, t, diameter/2


def make_input(base, name, seed, kind, row):
    rgb, depth, rays, t, radius = base
    rgb, depth = rgb.copy(), depth.copy()
    key = f"{row['obj_id']}/{row['im_id']}/{name}/{seed}"
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16],16))
    value = float(name.rsplit('_',1)[-1]) if name not in ['clean', 'zero_rgb', 'zero_xyz'] else 0
    if name.startswith('brightness'): rgb = np.clip(rgb*value, 0, 1)
    if name.startswith('blur'): rgb = cv2.GaussianBlur(rgb.transpose(1,2,0), (0,0), value).transpose(2,0,1)
    if name.startswith('rgb_noise'): rgb = np.clip(rgb+rng.normal(0,value,rgb.shape),0,1)
    if name.startswith('depth_noise'):
        valid = depth >= (0.001 if kind=='refine' else 0.1)
        depth[valid] += rng.normal(0,value/1000,int(valid.sum()))
    if name.startswith('depth_dropout'): depth[rng.random(depth.shape)<value]=0
    xyz = rays*depth[None]
    invalid = xyz[2:3] < (0.001 if kind=='refine' else 0.1)
    xyz = (xyz-t[:,None,None])/radius
    xyz[np.broadcast_to(invalid, xyz.shape) | (np.abs(xyz)>=2)] = 0
    if name=='zero_rgb': rgb[:] = 0
    if name=='zero_xyz': xyz[:] = 0
    def shift(a):
        b = np.zeros_like(a); d = int(value); b[:,:,d:] = a[:,:,:-d]; return b
    if name.startswith(('rgb_shift','joint_shift')): rgb = shift(rgb)
    if name.startswith(('xyz_shift','joint_shift')): xyz = shift(xyz)
    if name.startswith('occlusion'):
        size = round(160*np.sqrt(value)); lo = (160-size)//2
        rgb[:,lo:lo+size,lo:lo+size] = 0
        xyz[:,lo:lo+size,lo:lo+size] = 0
    return torch.from_numpy(np.concatenate((rgb,xyz)).astype(np.float32))


def measure(features, baseline):
    f, b = features.flatten(1), baseline.flatten(1).expand(features.shape[0],-1)
    cosine = 1-torch.nn.functional.cosine_similarity(f,b,dim=1)
    relative = (f-b).norm(dim=1)/b.norm(dim=1).clamp_min(1e-8)
    pooled = features.mean((2,3))
    pool_cosine = 1-torch.nn.functional.cosine_similarity(pooled,baseline.mean((2,3)).expand_as(pooled),dim=1)
    return dict(cosine=cosine.tolist(), relative_l2=relative.tolist(), pooled_cosine=pool_cosine.tolist(),
                zero_fraction=(f==0).float().mean(1).tolist(), rms=f.square().mean(1).sqrt().tolist())


@torch.inference_mode()
def run(args):
    out = args.output; out.mkdir(parents=True,exist_ok=True)
    lock = (out/'.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)
    rows = list(csv.DictReader(SPLIT.open()))
    if args.limit: rows = rows[:args.limit]
    sources = [Path(__file__), HERE/'PROTOCOL.md', SPLIT]
    sources += list((ROOT/'learning/models').glob('*.py'))
    sources += list((ROOT/'weights').glob('*/*'))
    sources += [ROOT/'learning/datasets/h5_dataset.py', ROOT/'Utils.py']
    sources += [DATA/'models/models_info.json']
    for row in rows:
        scene = DATA/f"lm_test_all/test/{int(row['obj_id']):06d}"
        sources += [scene/'scene_gt.json', scene/'scene_camera.json']
        sources += list((scene/'rgb').glob(f"{int(row['im_id']):06d}.*"))
        sources += list((scene/'depth').glob(f"{int(row['im_id']):06d}.*"))
    sources = sorted(set(sources))
    manifest = dict(hashes={str(p):digest(p) for p in sources if p.is_file()}, torch=torch.__version__,
                    kornia=kornia.__version__, device='cpu', threads=args.threads, frames=len(rows), conditions=conditions(),
                    scope='Observation shared CNN only; GT-aligned diagnostic crops; not full pose evaluation')
    mp = out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != json.loads(json.dumps(manifest)): raise RuntimeError('Manifest changed; use a fresh output directory')
    atomic(mp,manifest)
    started = time.time(); audit = {}; done=0
    for kind in RUNS:
        for trained in [True,False]:
            variant = kind+('_pretrained' if trained else '_random')
            model,cfg = load_model(kind,trained)
            encoder = model.encodeA if kind=='refine' else model.encoderA
            audit[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                  shared_encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                                  cfg=OmegaConf.to_container(cfg), architecture=str(encoder))
            atomic(out/'architecture.json',audit)
            target=out/variant; target.mkdir(exist_ok=True)
            for row in rows:
                path=target/f"{int(row['obj_id']):06d}_{int(row['im_id']):06d}.json"
                if path.exists() and path.with_suffix('.npz').exists(): done+=1; continue
                base = frame_inputs(row,cfg,kind)
                specs=conditions()
                inputs=torch.stack([make_input(base,n,s,kind,row) for n,s in specs])
                clean=inputs[:1].clone()
                records=[]; embedding=None; pooled_stages={}
                # Small batches limit CPU RAM; clean activation recomputed per stage.
                for stage,layer in enumerate(encoder):
                    clean=layer(clean)
                    output=[]
                    for offset in range(0,len(inputs),args.batch): output.append(layer(inputs[offset:offset+args.batch]))
                    inputs=torch.cat(output)
                    if not torch.isfinite(inputs).all(): raise RuntimeError('Nonfinite features')
                    metrics=measure(inputs,clean)
                    if metrics['relative_l2'][0] > 1e-4: raise RuntimeError('Clean self-comparison failed')
                    pooled_stages[f'stage_{stage}']=inputs.mean((2,3)).numpy()
                    for i,(name,seed) in enumerate(specs):
                        records.append(dict(stage=stage,condition=name,seed=seed,**{k:v[i] for k,v in metrics.items()}))
                    if stage==len(encoder)-1: embedding=clean.mean((2,3))[0].tolist()
                np.savez_compressed(path.with_suffix('.npz'),**pooled_stages)
                atomic(path,dict(frame=row,variant=variant,records=records,clean_embedding=embedding))
                done+=1
                atomic(out/'status.json',dict(state='running',completed_frame_variants=done,total_frame_variants=len(rows)*4,
                                             variant=variant,elapsed_seconds=time.time()-started))
                print(f'{done}/{len(rows)*4} {variant} object={row["obj_id"]} frame={row["im_id"]} elapsed={time.time()-started:.1f}s',flush=True)
    atomic(out/'status.json',dict(state='measurements_complete',completed_frame_variants=done,total_frame_variants=len(rows)*4,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=HERE/'results')
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--batch',type=int,default=4)
    run(ap.parse_args())
