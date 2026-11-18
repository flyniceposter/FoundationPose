"""Inspect the installed encoders without changing their state."""
import json
import numpy as np
import torch
import torch.nn as nn
from study import HERE,load_model,atomic,digest,ROOT


@torch.inference_mode()
def main():
    torch.set_num_threads(2);audit={};models={}
    for kind in ['refine','score']:
        model,cfg=load_model(kind,True);models[kind]=model
        shared=model.encodeA if kind=='refine' else model.encoderA
        fusion=model.encodeAB if kind=='refine' else model.encoderAB
        x=torch.zeros(1,6,160,160);stages=[]
        for i,layer in enumerate(shared):
            x=layer(x);stages.append(dict(family='shared',stage=i,shape=list(x.shape),parameters=sum(p.numel() for p in layer.parameters())))
        x=torch.cat([x,x],dim=1)
        for i,layer in enumerate(fusion):
            x=layer(x);stages.append(dict(family='fusion',stage=i,shape=list(x.shape),parameters=sum(p.numel() for p in layer.parameters())))
        w=shared[0].net[0].weight.detach().numpy()
        bn=[]
        for name,module in model.named_modules():
            if isinstance(module,nn.BatchNorm2d):
                bn.append(dict(name=name,channels=module.num_features,variance_min=float(module.running_var.min()),variance_median=float(module.running_var.median()),scale_abs_mean=float(module.weight.abs().mean())))
        audit[kind]=dict(c_in=cfg.c_in,crop_ratio=cfg.crop_ratio,strict_load=True,input_shape=[1,6,160,160],
                         total_parameters=sum(p.numel() for p in model.parameters()),shared_parameters=sum(p.numel() for p in shared.parameters()),
                         fusion_parameters=sum(p.numel() for p in fusion.parameters()),stages=stages,batchnorm=bn,
                         first_conv_rgb_norm=float(np.linalg.norm(w[:,:3])),first_conv_xyz_norm=float(np.linalg.norm(w[:,3:])))
    comparison=[]
    for family in ['shared','fusion']:
        a=getattr(models['refine'],'encodeA' if family=='shared' else 'encodeAB').state_dict()
        b=getattr(models['score'],'encoderA' if family=='shared' else 'encoderAB').state_dict()
        assert a.keys()==b.keys()
        for name,v in a.items():
            if v.numel()<2:continue
            x=v.numpy().flatten().astype(float);y=b[name].numpy().flatten().astype(float)
            comparison.append(dict(family=family,tensor=name,shape=list(v.shape),identical=bool(np.array_equal(x,y)),
                                   cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))) if np.linalg.norm(x)*np.linalg.norm(y)>0 else None))
    audit['cross_network_tensors']=comparison
    audit['source_hashes']={str(p):digest(p) for p in [ROOT/'estimater.py',ROOT/'learning/models/refine_network.py',ROOT/'learning/models/score_network.py',ROOT/'learning/models/network_modules.py']}
    audit['shared_receptive_field_px']=[7,11,27,43]
    audit['fusion_receptive_field_px']=[59,75,83,115,147]
    audit['receptive_field_note']='Theoretical maximum CNN receptive field through longest residual path; not empirical attribution.'
    atomic(HERE/'encoder_audit.json',audit)
    print('Encoder architecture/weight audit complete.')


if __name__=='__main__':main()
