"""Numerical checks for geometry masks, slopes and object-macro aggregation."""
import torch
from run import geometry_loss, metric_vectors, macro


def main():
    torch.set_num_threads(2)
    target=torch.zeros(2,1,8,8)
    target[0,0]=torch.arange(8).float()[None]/10
    valid=torch.zeros_like(target,dtype=torch.bool);valid[:,:,1:7,1:7]=True
    pred=target.clone().requires_grad_()
    assert geometry_loss(pred,target,valid,.2).item()==0
    changed=target.clone();changed[~valid]=900
    assert geometry_loss(pred,changed,valid,.2).item()==0
    offset=(target+.1).requires_grad_()
    depth_only=geometry_loss(offset,target,valid,0)
    with_slopes=geometry_loss(offset,target,valid,.2)
    assert torch.allclose(depth_only,with_slopes,atol=1e-6),'Constant offsets should not change slopes'
    bad=torch.zeros_like(target,requires_grad=True)
    loss=geometry_loss(bad,target,valid,.2);loss.backward()
    assert bad.grad[valid].abs().sum()>0
    assert bad.grad[~valid].abs().sum()==0,'Invalid pixels must not receive geometry gradients'
    mae,slope=metric_vectors(target,target,valid)
    assert not mae.any() and not slope.any()
    rows=[dict(object=1,mae_diameter=1.,mae_mm=10.,slope_diameter_per_pixel=2.)]*10+[dict(object=2,mae_diameter=3.,mae_mm=30.,slope_diameter_per_pixel=4.)]
    assert macro(rows)['macro']['mae_diameter']==2.,'Object macro must not weight by frame count'
    print('PASS: exact-target loss, invalid-value invariance, slope invariance, masked gradients, metrics and object-macro weighting')


if __name__=='__main__':main()
