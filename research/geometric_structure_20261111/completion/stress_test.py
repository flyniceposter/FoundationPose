"""Evaluate already-selected real-RGB-D checkpoints at new missingness/noise levels."""
from pathlib import Path
import argparse
import json
import copy
import numpy as np
import torch
import run as c

HERE=Path(__file__).resolve().parent
OUT=HERE/'stress_results'


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    conditions=[('lighter_holes',.2,2.),('larger_holes',.6,2.),('noisier_context',.4,5.)]
    cfg=c.pilot.OmegaConf.load(c.pilot.ROOT/'weights/2023-10-28-18-33-37/config.yml')
    original=c.pilot.RefineNet(cfg,c_in=cfg.c_in)
    original.load_state_dict(torch.load(c.pilot.ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    encoder=copy.deepcopy(original.encodeA);del original
    results={}
    for name,hole,noise in conditions:
        args=argparse.Namespace(data=Path('/home/Sahil/datasets/bop/lm'),frames=120,size=96,hole_prob=hole,noise_mm=noise)
        prepared=c.prepare(args)
        for split in ['development','exploratory','reserved']:
            samples=prepared['splits'][split];tensors=c.tensor_data(samples,'cuda')
            for sample in samples:
                assert (sample['x'][3:,sample['valid'][0]]==0).all()
                assert not (sample['valid']&sample['context']).any()
            for seed in c.SEEDS:
                for arm in ['frozen','normal','probe_frozen','probe_normal']:
                    model=c.Model(copy.deepcopy(encoder)).cuda().eval()
                    state=torch.load(HERE/'results'/f'{arm}_{seed}.pth',map_location='cpu',weights_only=False)
                    model.load_state_dict(state['model'])
                    rows,_=c.evaluate(model,tensors,samples)
                    results[f'{name}/{seed}/{arm}/{split}']=c.summary(rows)
            for baseline in ['zero','nearest']:
                rows,_=c.evaluate(None,tensors,samples,baseline=baseline)
                results[f'{name}/baseline/{baseline}/{split}']=c.summary(rows)
            del tensors
        c.ext.write(OUT/'evaluation.json',results)
        print('stress condition complete:',name,flush=True)
    lines=['# Geometry stress tests','','No training or checkpoint selection occurred here. Original 40% missing, 2mm-noise checkpoints were evaluated under changed input corruption. Reused objects make this additional exploratory robustness evidence.','','| Condition | Encoder/readout | Reserved depth mm | Reserved normal angle deg |','|---|---|---:|---:|']
    for name,_,_ in conditions:
        for arm in ['frozen','normal','probe_frozen','probe_normal']:
            vals=[results[f'{name}/{seed}/{arm}/reserved']['macro'] for seed in c.SEEDS]
            lines.append(f"| {name} | {arm} | {np.mean([v['mae_mm'] for v in vals]):.3f} | {np.mean([v['median_normal_angle_deg'] for v in vals]):.2f} |")
        near=results[f'{name}/baseline/nearest/reserved']['macro']['mae_mm'];lines.append(f'| {name} | nearest observed depth | {near:.3f} | — |')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');c.ext.write(OUT/'status.json',{'status':'complete','conditions':conditions,'checkpoint_selection':'none; reused previously selected models'})
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
