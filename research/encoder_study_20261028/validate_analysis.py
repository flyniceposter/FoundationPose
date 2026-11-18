"""Synthetic checks for scientific metrics, probe geometry, and model controls."""
import numpy as np
import torch
from analyze import cka,effective_rank,bootstrap
from secondary import setup
from study import HERE,load_model,atomic
from probe import project,angles,fit_predict


def main():
    torch.set_num_threads(2)
    x=np.random.default_rng(11).normal(size=(30,8));q,_=np.linalg.qr(np.random.default_rng(12).normal(size=(8,8)))
    assert abs(cka(x,x)-1)<1e-12
    assert abs(cka(x,3*x@q)-1)<1e-12
    assert 1<=effective_rank(x)<=8
    mean,lo,hi=bootstrap([1,1,4,4],[1,1,2,2]);assert mean==2.5 and lo==2.5 and hi==2.5
    for kind in ['refine','score']:
        name='encodeA' if kind=='refine' else 'encoderA'
        p,_=load_model(kind,True);p=getattr(p,name)
        h,_=setup(kind,'random_conv_trained_bn',101);h=getattr(h,name)
        assert torch.equal(p[0].net[1].running_var,h[0].net[1].running_var)
        assert torch.equal(p[0].net[1].weight,h[0].net[1].weight)
        assert not torch.equal(p[0].net[0].weight,h[0].net[0].weight)
        z,_=setup(kind,'trained_conv_reset_bn',101);z=getattr(z,name)
        assert torch.equal(p[0].net[0].weight,z[0].net[0].weight)
        assert torch.all(z[0].net[1].running_var==1) and torch.all(z[0].net[1].weight==1)
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    raw,sym=angles(rotation[None],np.eye(3)[None],np.stack([np.eye(3),rotation]))
    assert np.allclose(raw,90) and np.allclose(sym,0)
    matrices=np.random.default_rng(4).normal(size=(50,3,3))
    so3=project(matrices)
    assert np.allclose(np.linalg.det(so3),1)
    assert np.allclose(so3@so3.transpose(0,2,1),np.eye(3))
    # Query changes must not alter predictions for an unchanged query sample.
    y=np.random.default_rng(5).normal(size=(30,9));query=x[:2]
    a=fit_predict(x,y,query,1);b=fit_predict(x,y,np.concatenate([query,np.ones((1,8))*1e5]),1)
    assert np.allclose(a,b[:2])
    atomic(HERE/'analysis_validation.json',dict(status='passed',cka_scale_orthogonal_invariance=True,
           effective_rank_bounds=True,object_macro_bootstrap=True,batchnorm_factorial_controls=True,
           symmetry_angle_metric=True,so3_projection=True,probe_query_independence=True))
    print('Scientific metric and control checks passed.')


if __name__=='__main__':main()
