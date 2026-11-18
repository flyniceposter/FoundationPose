"""Fixed development-frame illustrations, not selected best/worst results."""
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/encoder-study-mpl')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from study import HERE,SPLIT,load_model,frame_inputs,make_input,atomic


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    out=HERE/'analysis/figures';out.mkdir(parents=True,exist_ok=True)
    row=next(csv.DictReader(SPLIT.open()))
    names=['clean','depth_noise_5','depth_dropout_0.3','rgb_shift_5','joint_shift_5','zero_xyz']
    model,cfg=load_model('refine',True);cfg.crop_ratio=1.2
    base=frame_inputs(row,cfg,'refine')
    inputs=torch.stack([make_input(base,n,0,'refine',row) for n in names])
    maps={}
    for kind in ['refine','score']:
        model,_=load_model(kind,True)
        layer=model.encodeA if kind=='refine' else model.encoderA
        features=layer(inputs)
        maps[kind]=np.sqrt(features.square().mean(1).numpy())
    fig,axs=plt.subplots(4,len(names),figsize=(17,10))
    for j,n in enumerate(names):
        axs[0,j].imshow(inputs[j,:3].permute(1,2,0).numpy().clip(0,1));axs[0,j].set_title(n.replace('_',' '),fontsize=10)
        axs[1,j].imshow(inputs[j,5],vmin=-2,vmax=2,cmap='coolwarm')
        for i,k in enumerate(['refine','score'],2):
            axs[i,j].imshow(maps[k][j],vmin=0,vmax=np.quantile(maps[k],.99),cmap='viridis')
        for i in range(4):axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
    for i,label in enumerate(['RGB input','Normalized Z','Refinement RMS','Ranking RMS']):axs[i,0].set_ylabel(label,fontsize=12)
    fig.suptitle(f'Fixed development example: object {row["obj_id"]}, frame {row["im_id"]}\nCommon inputs; activation RMS is neither attention nor feature importance',fontsize=16)
    fig.tight_layout();fig.savefig(out/'activation_example.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,5));ax.set_xlim(0,15);ax.set_ylim(0,5);ax.axis('off')
    boxes=[(.1,1.6,2.8,1.8,'RGB + XYZ\n6 × 160 × 160\n(render / observation)'),(3.4,1.6,2.8,1.8,'Shared CNN\n128 × 40 × 40\n684,480 parameters'),(6.7,1.6,2.6,1.8,'Concatenate\n256 × 40 × 40\n(two branches)'),(9.8,1.6,2.8,1.8,'Fusion CNN\n512 × 20 × 20\nthen tokens / heads')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor='#e5f0ef',edgecolor='#116b78',linewidth=2))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=13)
    for x1,x2 in [(2.9,3.3),(6.2,6.6),(9.3,9.7)]:ax.annotate('',(x2,2.5),(x1,2.5),arrowprops=dict(arrowstyle='->',lw=2,color='#b46527'))
    ax.text(6.4,4.2,'CNN encoders inspected and measured',ha='center',fontsize=20,fontweight='bold',color='#173540')
    ax.text(6.4,.65,'The primary experiment probes the observed branch. Supporting fusion tests use artificial observation/self-pairs.',ha='center',fontsize=12,color='#48616a')
    fig.tight_layout();fig.savefig(out/'architecture.png',dpi=150);plt.close(fig)
    atomic(HERE/'illustration_manifest.json',dict(frame=row,selection='First development row in pre-existing split, fixed before figure generation',conditions=names,
                                                scope='Common-input visual illustrations; RMS maps are not attribution'))
    print('Encoder illustrations complete.')


if __name__=='__main__':main()
