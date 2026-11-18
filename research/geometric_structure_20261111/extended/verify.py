"""Audit completed run artifacts and matched-comparison invariants."""
from pathlib import Path
import json
import torch

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())['frames']
    identities={}
    for r in manifest:
        key=(r['object'],r['frame'])
        assert key not in identities
        identities[key]=r['split']
    objects={name:{r['object'] for r in manifest if r['split']==name} for name in ('train','development','unseen')}
    assert objects['train'].isdisjoint(objects['development'])
    assert objects['train'].isdisjoint(objects['unseen'])
    assert objects['development'].isdisjoint(objects['unseen'])
    assert json.loads((OUT/'status.json').read_text())['status']=='complete'
    audit=[]
    for seed in (42,123,2026):
        checkpoints={a:torch.load(OUT/f'{a}_seed{seed}.pth',map_location='cpu',weights_only=False) for a in ('frozen','depth','structure','probe_frozen','probe_depth','probe_structure')}
        baseline=checkpoints['frozen']['model']
        start=[checkpoints[a]['history'][0]['development_mae_diameter'] for a in ('frozen','depth','structure')]
        assert max(start)-min(start)<1e-8,'Arms must have same initialization'
        for arm in ('depth','structure'):
            state=checkpoints[arm]['model']
            changed=sum(not torch.equal(state[k],baseline[k]) for k in state if k.startswith('encoder.') and k.endswith('weight'))
            assert changed>0,'Adapted encoder did not change'
            assert all(torch.equal(v,state[k]) for k,v in baseline.items() if k.startswith('encoder.') and ('running_' in k or 'num_batches_tracked' in k))
            for k,v in state.items():
                if k.startswith('encoder.'):
                    assert torch.equal(v,checkpoints['probe_'+arm]['model'][k]),'Fresh probe changed frozen encoder'
            audit.append({'seed':seed,'arm':arm,'encoder_weight_tensors_changed':changed,'best_step':checkpoints[arm]['best_step']})
        for k,v in baseline.items():
            if k.startswith('encoder.'):
                assert torch.equal(v,checkpoints['probe_frozen']['model'][k])
        for arm,ckpt in checkpoints.items():
            assert ckpt['development_mae_diameter']==min(h['development_mae_diameter'] for h in ckpt['history'])
            assert all(torch.isfinite(v).all() for v in ckpt['model'].values())
    evals=json.loads((OUT/'evaluation.json').read_text())
    assert len(evals)==58
    for name,result in evals.items():
        assert result['frames']==(480 if 'unseen' in name else 168)
        assert all(v>=0 for v in result['macro'].values())
    result={'passed':True,'checks':['disjoint adaptation objects','unique frames','identical arm initialization','encoder weights changed','BN stats fixed','fresh probes preserve encoders','development-only checkpoint selection','finite checkpoints','evaluation coverage'],'encoder_audit':audit}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
