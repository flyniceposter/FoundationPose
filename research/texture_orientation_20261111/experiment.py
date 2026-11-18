"""Controlled textured/uniform reference-view retrieval with FoundationPose encodeA."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import trimesh
import pyrender

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet


def rotation(angle):
    a=np.radians(angle)
    return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])


def camera_pose(elevation):
    eye=np.array([.46*np.cos(np.radians(elevation)),0,.46*np.sin(np.radians(elevation))])
    z=eye/np.linalg.norm(eye)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    pose=np.eye(4);pose[:3,:3]=np.stack([x,y,z],axis=1);pose[:3,3]=eye
    return pose


class Renderer:
    def __init__(self,size):
        self.path=ROOT/'demo_data/mustard0/mesh/textured_simple.obj'
        mesh=trimesh.load(self.path,force='mesh',process=False)
        mesh.vertices-=mesh.bounds.mean(axis=0)
        self.mesh=mesh
        self.renderer=pyrender.OffscreenRenderer(size,size)
        self.scene=pyrender.Scene(bg_color=[.16,.16,.16,1],ambient_light=[.5,.5,.5])
        self.camera=self.scene.add(pyrender.PerspectiveCamera(yfov=np.radians(34),znear=.01,zfar=2),pose=camera_pose(12))
        self.light=self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=camera_pose(12))
        self.textured=pyrender.Mesh.from_trimesh(mesh,smooth=False)
        uniform=mesh.copy()
        uniform.visual=trimesh.visual.ColorVisuals(mesh=uniform,vertex_colors=np.tile([180,150,55,255],(len(uniform.vertices),1)))
        material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[180/255,150/255,55/255,1],metallicFactor=0,roughnessFactor=1,doubleSided=True)
        # Same material response in both conditions; only base-color texture differs.
        for primitive in self.textured.primitives:
            primitive.material.metallicFactor=0
            primitive.material.roughnessFactor=1
            primitive.material.doubleSided=True
        self.uniform=pyrender.Mesh.from_trimesh(uniform,material=material,smooth=False)

    def render(self,angle,elevation=12,intensity=2):
        cp=camera_pose(elevation)
        self.scene.set_pose(self.camera,cp);self.scene.set_pose(self.light,cp)
        self.light.light.intensity=intensity
        pose=np.eye(4);pose[:3,:3]=rotation(angle)
        images,depths=[],[]
        for mesh in (self.textured,self.uniform):
            node=self.scene.add(mesh,pose=pose)
            color,depth=self.renderer.render(self.scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            self.scene.remove_node(node)
            images.append(color[:,:,:3]);depths.append(depth)
        if not np.allclose(depths[0],depths[1],atol=1e-6):
            raise RuntimeError('Geometry changed between appearance conditions')
        return images


def as_input(images,device):
    rgb=torch.from_numpy(np.stack(images).copy()).permute(0,3,1,2).float().to(device)/255
    return torch.cat([rgb,torch.zeros_like(rgb)],dim=1)


class Matcher(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.pool=nn.AdaptiveAvgPool2d((4,4))
    def forward(self,x):
        return F.normalize(self.pool(self.encoder(x)).flatten(1),dim=1)


def angular_error(a,b):
    return np.abs((np.asarray(a)-np.asarray(b)+180)%360-180)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'pilot')
    parser.add_argument('--steps',type=int,default=150)
    parser.add_argument('--size',type=int,default=96)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    renderer=Renderer(args.size)
    angles=np.arange(0,360,30)
    refs=[renderer.render(float(a)) for a in angles]
    train_pairs=[];labels=[]
    for elevation in (4,20):
        for offset in (-6,6):
            for i,angle in enumerate(angles):
                train_pairs.append(renderer.render(float(angle+offset),elevation,1.6 if offset<0 else 2.4));labels.append(i)
    # Distinct rendered query angles, elevations and light intensity; synthetic within-object evaluation.
    query_angles=np.concatenate([angles-8,angles+8])
    query_pairs=[renderer.render(float(a),elevation=15 if i<12 else 8,intensity=1.9) for i,a in enumerate(query_angles)]
    renderer.renderer.delete()
    cfg=OmegaConf.load(ROOT/'weights/2023-10-28-18-33-37/config.yml')
    net=RefineNet(cfg,c_in=cfg.c_in)
    net.load_state_dict(torch.load(ROOT/'weights/2023-10-28-18-33-37/model_best.pth',map_location='cpu',weights_only=True))
    initial=Matcher(copy.deepcopy(net.encodeA));del net
    model=copy.deepcopy(initial).to(args.device).eval()
    reference_inputs=as_input([p[0] for p in refs],args.device)
    train_inputs=as_input([p[0] for p in train_pairs],args.device)
    targets=torch.tensor(labels,device=args.device)
    optimizer=torch.optim.Adam(model.encoder.parameters(),lr=1e-5)
    generator=torch.Generator().manual_seed(42)
    history=[]
    # Same-object incorrect rotations are negatives; correct reference is the positive.
    for step in range(args.steps):
        idx=torch.randperm(len(train_inputs),generator=generator)[:6].to(args.device)
        optimizer.zero_grad(set_to_none=True)
        q=model(train_inputs[idx]);r=model(reference_inputs)
        loss=F.cross_entropy(q@r.T/.07,targets[idx])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();optimizer.step();history.append(float(loss.detach()))
        if (step+1)%50==0:print(f'step {step+1}: contrastive loss={history[-1]:.4f}',flush=True)
    scores={};metrics={}
    with torch.no_grad():
        for name,matcher in [('pretrained',initial.to(args.device).eval()),('adapted',model)]:
            metrics[name]={};scores[name]={}
            for condition,j in [('textured',0),('uniform',1)]:
                r=matcher(as_input([p[j] for p in refs],args.device))
                q=matcher(as_input([p[j] for p in query_pairs],args.device))
                sim=(q@r.T).cpu().numpy();scores[name][condition]=sim
                predicted=angles[sim.argmax(1)]
                truth=np.tile(np.arange(12),2)
                metrics[name][condition]={'nearest_reference_accuracy':float(np.mean(sim.argmax(1)==truth)),'mean_yaw_error_deg':float(angular_error(predicted,query_angles).mean()),'predicted_yaw_deg':predicted.tolist()}
    torch.save({'encoder':model.encoder.state_dict(),'pool':'adaptive_avg_4x4; L2 normalization','input':'RGB / 255 + zero XYZ','seed':42,'steps':args.steps},args.out/'texture_orientation_encoder.pth')
    np.savez_compressed(args.out/'render_pairs.npz',reference_angles=angles,references=np.array(refs),query_angles=query_angles,queries=np.array(query_pairs),training=np.array(train_pairs),training_labels=np.array(labels))
    np.savez_compressed(args.out/'similarities.npz',**{f'{name}_{condition}':v for name,cs in scores.items() for condition,v in cs.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,6,figsize=(13,5))
    for row,j in enumerate([0,1]):
        for col,idx in enumerate([0,2,4,6,8,10]):
            axs[row,col].imshow(refs[idx][j]);axs[row,col].axis('off');axs[row,col].set_title(f'{angles[idx]}°')
        axs[row,0].text(-.06,.5,['Original texture','Uniform color'][row],transform=axs[row,0].transAxes,rotation=90,ha='right',va='center')
    fig.suptitle('Same mustard-bottle geometry, poses and lighting; surface appearance changed')
    fig.tight_layout();fig.savefig(args.out/'reference_views.png',dpi=160);plt.close(fig)
    # Fixed query selected by index before inspecting matching results.
    qi=4;fig,axs=plt.subplots(2,4,figsize=(13,7))
    for row,(condition,j) in enumerate([('textured',0),('uniform',1)]):
        sim=scores['adapted'][condition][qi];best=int(sim.argmax());correct=qi%12
        for col,(im,title) in enumerate([(query_pairs[qi][j],f'{condition} query: {query_angles[qi]}°'),(refs[correct][j],f'Nearest true reference: {angles[correct]}°'),(refs[best][j],f'Encoder match: {angles[best]}°')]):
            axs[row,col].imshow(im);axs[row,col].axis('off');axs[row,col].set_title(title,fontsize=10)
        axs[row,3].plot(angles,sim,'o-');axs[row,3].axvline(angles[correct],color='green',linestyle='--');axs[row,3].set(xlabel='Reference yaw (degrees)',ylabel='Cosine similarity',ylim=(0,1.02))
    fig.suptitle('Fixed held-out synthetic query: actual adapted-encoder predictions (RGB only)')
    fig.tight_layout();fig.savefig(args.out/'matching_example.png',dpi=160);plt.close(fig)
    checkpoint=ROOT/'weights/2023-10-28-18-33-37/model_best.pth'
    report={'scope':'One-object synthetic development pilot; not BOP-format mesh, real-image evaluation or pose benchmark. Uniform condition is an ablation, not certified symmetry.', 'asset':str(renderer.path),'asset_sha256':hashlib.sha256(renderer.path.read_bytes()).hexdigest(),'texture_sha256':hashlib.sha256((renderer.path.parent/'texture_map.png').read_bytes()).hexdigest(),'pretrained_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'seed':42,'steps':args.steps,'size':args.size,'reference_angles':angles.tolist(),'train_elevations':[4,20],'train_angle_offsets':[-6,6],'query_angles':query_angles.tolist(),'query_elevations':[15,8],'pair_depth_equality_checked':True,'fixed_figure_query_index':qi,'training_loss_first':history[0] if history else None,'training_loss_last':history[-1] if history else None,'metrics':metrics}
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
