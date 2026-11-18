"""Common-input, BatchNorm-factorial, and controlled pair-fusion diagnostics."""
import argparse
import csv
import json
import time
import fcntl
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from study import HERE, ROOT, SPLIT, RUNS, load_model, frame_inputs, make_input, measure, atomic, digest

CONDITIONS=['clean','brightness_0.5','blur_3','zero_rgb','zero_xyz','rgb_shift_5','xyz_shift_5',
            'joint_shift_5','rgb_noise_0.1','depth_noise_5','depth_dropout_0.3','occlusion_0.3']


def variants():
    yield 'pretrained',0
    for name in ['random','random_conv_trained_bn','trained_conv_reset_bn']:
        for seed in [101,202,303]: yield name,seed


def setup(kind,variant,seed):
    model,cfg=load_model(kind,True)
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module,nn.Conv2d) and variant in ['random','random_conv_trained_bn']:
            module.reset_parameters()
        if isinstance(module,nn.BatchNorm2d) and variant in ['random','trained_conv_reset_bn']:
            module.reset_parameters()
    cfg.crop_ratio=1.2
    return model.eval(),cfg


@torch.inference_mode()
def run(args):
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    lock=(out/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)
    rows=list(csv.DictReader(SPLIT.open()))
    rows=[r for obj in sorted({int(r['obj_id']) for r in rows}) for r in [r for r in rows if int(r['obj_id'])==obj and r['split']=='development'][:args.per_object]]
    sources=[Path(__file__),HERE/'study.py',HERE/'SECONDARY_PROTOCOL.md',SPLIT]
    sources+=list((ROOT/'learning/models').glob('*.py'))+list((ROOT/'weights').glob('*/*'))
    manifest=dict(hashes={str(p):digest(p) for p in sources if p.is_file()},frames=rows,conditions=CONDITIONS,variants=list(variants()),
                  scope='Common-input CNN and artificial observation/self-pair fusion; no pose metrics',threads=args.threads)
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=json.loads(json.dumps(manifest)):raise RuntimeError('Manifest mismatch')
    atomic(mp,manifest)
    total=len(rows)*len(list(variants()))*2;done=0;started=time.time()
    for kind in RUNS:
        for variant,seed in variants():
            model,cfg=setup(kind,variant,seed)
            encoder=model.encodeA if kind=='refine' else model.encoderA
            fusion=model.encodeAB if kind=='refine' else model.encoderAB
            target=out/f'{kind}_{variant}_{seed}';target.mkdir(exist_ok=True)
            for row in rows:
                path=target/f"{int(row['obj_id']):06d}_{int(row['im_id']):06d}.json"
                if path.exists() and path.with_suffix('.npz').exists():done+=1;continue
                # Same explicit validity threshold for both architectures.
                base=frame_inputs(row,cfg,'refine')
                x=torch.stack([make_input(base,n,0,'refine',row) for n in CONDITIONS])
                records=[];pooled={};fixed=None
                for family,layers in [('shared',encoder),('fusion',fusion)]:
                    if family=='fusion':
                        fixed=x[:1].expand_as(x)
                        x=torch.cat((fixed,x),dim=1)
                    clean=x[:1].clone()
                    for stage,layer in enumerate(layers):
                        clean=layer(clean)
                        x=torch.cat([layer(x[i:i+args.batch]) for i in range(0,len(x),args.batch)])
                        if not torch.isfinite(x).all():raise RuntimeError('Nonfinite features')
                        metrics=measure(x,clean)
                        if metrics['relative_l2'][0]>1e-4:raise RuntimeError('Self baseline failed')
                        pooled[f'{family}_{stage}']=x.mean((2,3)).numpy()
                        for i,name in enumerate(CONDITIONS):
                            records.append(dict(family=family,stage=stage,condition=name,**{k:v[i] for k,v in metrics.items()}))
                np.savez_compressed(path.with_suffix('.npz'),**pooled)
                atomic(path,dict(frame=row,kind=kind,variant=variant,initialization_seed=seed,records=records))
                done+=1
                atomic(out/'status.json',dict(state='running',completed=done,total=total,variant=target.name,elapsed_seconds=time.time()-started))
                print(f'{done}/{total} {target.name} elapsed={time.time()-started:.1f}s',flush=True)
    atomic(out/'status.json',dict(state='measurements_complete',completed=done,total=total,elapsed_seconds=time.time()-started))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'secondary_results')
    ap.add_argument('--per-object',type=int,default=10);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--batch',type=int,default=4)
    run(ap.parse_args())
