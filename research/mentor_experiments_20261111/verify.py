"""Validate experiment invariants before generating a mentor-facing report."""
from pathlib import Path
import json
import numpy as np
import torch
import importlib.util

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    rotation=module('rotation',ROOT/'research/real_rotation_20261111/run.py')
    c=rotation.c
    d=torch.load(ROOT/'research/real_rotation_20261111/cache_vertexcolor_v2_240_96.pt',map_location='cpu',weights_only=False)
    checks=[]
    for obj,v in d['objects'].items():
        sets={k:{s['frame'] for s in samples} for k,samples in v['splits'].items()}
        assert sets['train'].isdisjoint(sets['development']) and sets['train'].isdisjoint(sets['evaluation']) and sets['development'].isdisjoint(sets['evaluation'])
        assert {s['frame'] for s in v['references']}<=sets['train']
        assert np.abs(v['renders'][:,0].astype(float)-v['renders'][:,1].astype(float)).mean()>.1
        assert all((np.array(s['rgb'])>=0).all() for s in v['splits']['evaluation'])
    Q=np.eye(3);S=np.diag([-1.,-1.,1.])
    assert rotation.distances(Q[None],S[None],[Q,S])[0,0]<1e-5
    assert rotation.distances(Q[None],S[None],[Q])[0,0]>179
    checks+=['real-query split disjointness','references drawn only from training','appearance ablation changes pixels','equivalent rotations accepted by symmetry metric']
    rotation_out=ROOT/'research/real_rotation_20261111/results'
    for seed in [42,123,2026]:
        plain=torch.load(rotation_out/f'ordinary_{seed}.pth',map_location='cpu',weights_only=False)
        sym=torch.load(rotation_out/f'symmetry_{seed}.pth',map_location='cpu',weights_only=False)
        assert abs(plain['history'][0]['rotation_error_deg']-sym['history'][0]['rotation_error_deg'])<1e-8
        for ckpt in [plain,sym]:
            assert ckpt['history'][ckpt['best_step']//250]['rotation_error_deg']==min(h['rotation_error_deg'] for h in ckpt['history'])
            assert all(torch.isfinite(v).all() for v in ckpt['model'].values())
    checks+=['matched rotation initialization','development-selected rotation checkpoint','finite rotation weights']
    comp=ROOT/'research/geometric_structure_20261111/completion'
    cache=next(p for p in comp.glob('cache_120_96_*.pt') if '_holes' not in p.name)
    data=torch.load(cache,map_location='cpu',weights_only=False)
    for split,samples in data['splits'].items():
        for sample in samples:
            assert not (sample['context']&sample['valid']).any()
            assert not sample['x'][3:,sample['valid'][0]].any()
            normals=sample['normals'].permute(1,2,0)[sample['normalvalid'][0]]
            assert torch.allclose(normals.norm(dim=1),torch.ones(len(normals)),atol=1e-4)
    for seed in [42,123,2026]:
        base=torch.load(comp/'results'/f'frozen_{seed}.pth',map_location='cpu',weights_only=False)
        adapt=torch.load(comp/'results'/f'normal_{seed}.pth',map_location='cpu',weights_only=False)
        probe=torch.load(comp/'results'/f'probe_normal_{seed}.pth',map_location='cpu',weights_only=False)
        changed=False
        for key,value in adapt['model'].items():
            if key.startswith('encoder.'):
                assert torch.equal(value,probe['model'][key])
                if key.endswith('weight') and not torch.equal(value,base['model'][key]):changed=True
                if 'running_' in key or 'num_batches_tracked' in key:assert torch.equal(value,base['model'][key])
        assert changed
    checks+=['supervised geometry excluded from XYZ inputs','normal targets unit length','adapted encoder actually changed','BN statistics fixed','fresh-decoder probes preserve adapted encoder']
    result={'passed':True,'checks':checks,'rotation_scope':'real BOP photographs; rendered training references','geometry_scope':'real BOP RGB-D; rendered supervision; artificial holes','excluded_run':'real_rotation_20261111/invalid_render_v1'}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
