"""Verify candidate geometry, interventions, and hook transparency on the GPU."""
import json
from pathlib import Path
import numpy as np
import torch
import run_gpu as r


def main():
    torch.set_num_threads(4)
    gt=np.eye(4);gt[2,3]=1
    poses,labels=r.candidates(gt,0)
    assert len(poses)==37 and np.allclose(poses[0],gt)
    assert abs(np.linalg.norm(poses[1,:3,3]-gt[:3,3])-.005)<1e-6
    assert np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-6)
    # Pure hook operation: pooling preserves per-channel means, shuffling preserves values.
    x=torch.randn(6,8,5,5,device='cuda');probe=r.Probe(None,'refine','pooled_shared',0)
    y=probe.shared_hook(3)(None,None,x)
    assert torch.equal(y[:3],x[:3])
    assert torch.allclose(y[3:].mean((2,3)),x[3:].mean((2,3)),atol=1e-6)
    probe=r.Probe(None,'refine','shuffle_shared',0);z=probe.shared_hook(3)(None,None,x)
    assert torch.equal(z[:3],x[:3])
    assert torch.allclose(z[3:].mean((2,3)),x[3:].mean((2,3)),atol=1e-6)
    assert not torch.equal(z[3:],x[3:])
    out=r.HERE/'validation_run';out.mkdir(exist_ok=True)
    exp=r.Experiment(out);row=r.frames()[0][0]
    a=exp.unit(row,'clean','intact',0)
    class NoHooks:
        def __init__(self,*args):self.records={}
        def __enter__(self):return self
        def __exit__(self,*args):return None
    original=r.Probe;r.Probe=NoHooks
    try:b=exp.unit(row,'clean','intact',0)
    finally:r.Probe=original
    assert np.allclose(a['ranking_logits'],b['ranking_logits'],atol=1e-3,rtol=1e-3)
    assert np.allclose(a['refined_poses_centered'],b['refined_poses_centered'],atol=1e-5,rtol=1e-4)
    assert a['initial_errors']['translation_mm'][0]<.001
    assert a['initial_errors']['sampled_adds_mm'][0]<.001
    result=dict(state='passed',gpu=torch.cuda.get_device_name(),candidates=37,gt_candidate_error_below_0_001mm=True,
                pooling_preserves_channel_means=True,shuffle_preserves_channel_means=True,
                rendered_branch_unchanged=True,intact_hook_transparency=True)
    r.atomic(r.HERE/'validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
