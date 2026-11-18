"""Development-only geometry adaptation of FoundationPose's actual shared encoder."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet


def load_samples(root, object_id, count, size):
    scene = root / 'lm_test_all/test' / f'{object_id:06d}'
    gt = json.loads((scene / 'scene_gt.json').read_text())
    cameras = json.loads((scene / 'scene_camera.json').read_text())
    diameter = json.loads((root / 'models/models_info.json').read_text())[str(object_id)]['diameter']
    frames = sorted(gt, key=int)
    frames = [frames[i] for i in np.linspace(0, len(frames)-1, min(count, len(frames)), dtype=int)]
    samples = []
    for frame in frames:
        index = next((i for i, g in enumerate(gt[frame]) if g['obj_id'] == object_id), None)
        if index is None:
            continue
        stem = f'{int(frame):06d}'
        mask = np.asarray(Image.open(scene / 'mask_visib' / f'{stem}_{index:06d}.png')) > 0
        depth = np.asarray(Image.open(scene / 'depth' / f'{stem}.png')).astype(np.float32) * cameras[frame]['depth_scale']
        valid = mask & np.isfinite(depth) & (depth > 0)
        if valid.sum() < 100:
            continue
        ys, xs = np.where(mask)
        side = int(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)*1.2)
        cx, cy = (xs.max()+xs.min())//2, (ys.max()+ys.min())//2
        box = (int(cx-side//2), int(cy-side//2), int(cx-side//2+side), int(cy-side//2+side))
        rgb = Image.open(scene / 'rgb' / f'{stem}.png').convert('RGB').crop(box).resize((size, size), Image.Resampling.BILINEAR)
        # Relative depth measures visible shape, not absolute distance. Target-only centering.
        relative = (depth - np.median(depth[valid])) / diameter
        relative[~valid] = 0
        target = np.asarray(Image.fromarray(relative).crop(box).resize((size,size), Image.Resampling.NEAREST)).copy()
        weight = np.asarray(Image.fromarray(valid.astype(np.uint8)).crop(box).resize((size,size), Image.Resampling.NEAREST)).copy()
        if weight.sum() < 20:
            continue
        rgb = torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).float()/255
        # Original six-channel interface retained, but geometric input is deliberately withheld.
        x = torch.cat([rgb, torch.zeros_like(rgb)])
        samples.append({'x':x, 'target':torch.from_numpy(target)[None], 'valid':torch.from_numpy(weight.astype(bool))[None], 'frame':stem})
    if len(samples) < 8:
        raise ValueError('Need at least eight valid frames')
    return samples, diameter


class GeometryModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(nn.Conv2d(128,64,3,padding=1), nn.ReLU(), nn.Conv2d(64,1,1))

    def forward(self, x):
        return F.interpolate(self.decoder(self.encoder(x)), size=x.shape[-2:], mode='bilinear', align_corners=False)


def losses(pred, target, valid):
    if not valid.any():
        raise ValueError('No valid depth pixels')
    depth = F.smooth_l1_loss(pred[valid], target[valid], beta=.05)
    terms = []
    for axis in (-1,-2):
        p, q = pred.diff(dim=axis), target.diff(dim=axis)
        adjacent = valid.narrow(axis,0,valid.shape[axis]-1) & valid.narrow(axis,1,valid.shape[axis]-1)
        if adjacent.any():
            terms.append((p[adjacent]-q[adjacent]).abs().mean())
    slope = torch.stack(terms).mean() if terms else pred.sum()*0
    return depth + .2*slope


def batch(samples, device):
    return [torch.stack([s[k] for s in samples]).to(device) for k in ('x','target','valid')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('/home/Sahil/datasets/bop/lm'))
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'pilot')
    parser.add_argument('--object-id', type=int, default=1)
    parser.add_argument('--frames', type=int, default=48)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--size', type=int, default=80)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    args.out.mkdir(parents=True, exist_ok=True)
    samples, diameter = load_samples(args.data,args.object_id,args.frames,args.size)
    # Contiguous temporal blocks; this is NOT an independent scene or official test split.
    boundary = int(len(samples)*.75)
    train, val = samples[:boundary], samples[boundary:]
    cfg = OmegaConf.load(ROOT/'weights/2023-10-28-18-33-37/config.yml')
    net = RefineNet(cfg, c_in=cfg.c_in)
    net.load_state_dict(torch.load(ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    initial = GeometryModel(copy.deepcopy(net.encodeA))
    del net
    metrics, predictions = {}, {}
    for arm, tune in [('frozen_encoder',False), ('geometry_adapted',True)]:
        model = copy.deepcopy(initial).to(args.device)
        for p in model.encoder.parameters():
            p.requires_grad_(tune)
        # Keep all pretrained BN statistics fixed, including during encoder fine-tuning.
        model.eval()
        optimizer = torch.optim.Adam([{'params':model.decoder.parameters(),'lr':1e-3}] + ([{'params':model.encoder.parameters(),'lr':1e-5}] if tune else []))
        generator = torch.Generator().manual_seed(42)
        history = []
        for step in range(args.steps):
            indices = torch.randperm(len(train),generator=generator)[:4].tolist()
            x,y,m = batch([train[i] for i in indices], args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = losses(model(x),y,m)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        errors, zero_errors, outputs = [], [], []
        with torch.no_grad():
            for sample in val:
                x,y,m = batch([sample],args.device)
                pred = model(x)
                errors.append(float((pred[m]-y[m]).abs().mean())*diameter)
                zero_errors.append(float(y[m].abs().mean())*diameter)
                outputs.append(pred.cpu()[0,0].numpy())
        metrics[arm] = {'relative_depth_mae_mm':float(np.mean(errors)), 'constant_zero_mae_mm':float(np.mean(zero_errors)), 'training_loss_first':history[0], 'training_loss_last':history[-1]}
        predictions[arm] = outputs
        torch.save({'encoder':model.encoder.state_dict(), 'decoder':model.decoder.state_dict(), 'args':{k:str(v) for k,v in vars(args).items()}, 'target':'median-centered visible depth / object diameter; RGB only, XYZ zero'},args.out/f'{arm}.pth')
        print(arm, metrics[arm],flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,5,figsize=(15,6))
    for row, idx in enumerate([0,len(val)//2]):
        sample = val[idx]
        mask = sample['valid'][0].numpy()
        truth = sample['target'][0].numpy()*diameter
        baseline = predictions['frozen_encoder'][idx]*diameter
        pred = predictions['geometry_adapted'][idx]*diameter
        limit = max(float(np.abs(truth[mask]).max()),1)
        axes[row,0].imshow(sample['x'][:3].permute(1,2,0))
        for col, array in [(1,truth),(2,baseline),(3,pred)]:
            axes[row,col].imshow(np.ma.masked_where(~mask,array),cmap='coolwarm',vmin=-limit,vmax=limit)
        error = axes[row,4].imshow(np.ma.masked_where(~mask,abs(pred-truth)),cmap='magma',vmin=0)
        fig.colorbar(error,ax=axes[row,4],label='mm')
        for col, title in enumerate(['RGB input','Measured relative depth','Frozen encoder + trained head','Geometry-adapted encoder','Adapted absolute error']):
            axes[row,col].set_title(title,fontsize=9)
            axes[row,col].axis('off')
    fig.suptitle('LM development pilot: RGB-only visible geometry prediction (GT crops; not pose evaluation)')
    fig.tight_layout()
    fig.savefig(args.out/'geometry_example.png',dpi=150)
    report = {'warning':'LM test images reused for DEVELOPMENT ONLY; contiguous within-sequence split, no unseen-scene or pose improvement claim.', 'train_frames':[s['frame'] for s in train], 'validation_frames':[s['frame'] for s in val], 'object_id':args.object_id,'steps':args.steps,'metrics':metrics}
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
