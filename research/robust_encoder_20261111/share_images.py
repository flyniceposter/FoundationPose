from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
HERE=Path(__file__).resolve().parent
OUT=HERE/'share';OUT.mkdir(exist_ok=True)
DATA=Path('/home/Sahil/datasets/bop/lm')
rows=json.loads((HERE/'pose/records.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
arms=['original','combined','heads_only','joint_combined']
labels=['Original','Encoder only\n(shape + symmetry + texture)','Fusion + pose heads\n(encoder frozen)','Combined encoder\n+ pose heads']
colors=['#718096','#db7048','#3986ad','#22866c']
conditions=['observed','noise5mm','dropout40','grayscale']
fig,axs=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1,1]})
for ax,condition,title in zip(axs,['observed','dropout40'],['Observed RGB-D','40% artificial depth dropout']):
 vals=[]
 for arm in arms:
  rr=[r for r in rows if r['arm']==arm and r['condition']==condition]
  vals.append(np.mean([r['add_mm'] for r in rr]))
 ax.barh(np.arange(4),vals,color=colors,height=.6)
 ax.set_yticks(np.arange(4),labels if ax==axs[0] else ['']*4);ax.invert_yaxis();ax.set_xlim(0,7);ax.set_title(title,pad=14,weight='bold');ax.set_xlabel('Mean pose error (ADD, mm) — lower is better')
 for i,v in enumerate(vals):ax.text(v+.12,i,f'{v:.2f}',va='center',weight='bold')
 ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Pose-head training helped; encoder-only adaptation did not',fontsize=17,weight='bold',y=.97)
fig.text(.5,.055,'Exploratory BOP LINEMOD: 5 objects, 20 evaluation images; adapted models averaged over 3 training seeds.\nKnown objects and ground-truth masks. Improved precision here does not establish broad robustness or novelty.',ha='center',fontsize=10,color='#444444')
fig.subplots_adjust(left=.23,right=.96,top=.83,bottom=.20,wspace=.12)
fig.savefig(OUT/'results_comparison.png',dpi=180,facecolor='white');plt.close(fig)
# Fixed first evaluation frame per object, fixed seed; no selection by model success.
objects=[5,8,9];names={5:'Can',8:'Driller',9:'Duck'}
fig,axes=plt.subplots(3,3,figsize=(12,11))
for ri,obj in enumerate(objects):
 scene=DATA/'lm_test_all/test'/f'{obj:06d}'
 originals=sorted([r for r in rows if r['arm']=='original' and r['object']==obj and r['condition']=='observed'],key=lambda r:int(r['frame']))
 base=originals[0];frame=base['frame'];rgb=np.asarray(Image.open(scene/'rgb'/f'{frame}.png').convert('RGB'))
 cams=json.loads((scene/'scene_camera.json').read_text());K=np.array(cams[str(int(frame))]['cam_K']).reshape(3,3)
 gtdata=json.loads((scene/'scene_gt.json').read_text());gi=next(i for i,g in enumerate(gtdata[str(int(frame))]) if g['obj_id']==obj)
 mask=np.asarray(Image.open(scene/'mask_visib'/f'{frame}_{gi:06d}.png'))>0
 yy,xx=np.where(mask);margin=35;x0=max(0,xx.min()-margin);x1=min(rgb.shape[1],xx.max()+margin);y0=max(0,yy.min()-margin);y1=min(rgb.shape[0],yy.max()+margin)
 mesh=trimesh.load(DATA/'models'/f'obj_{obj:06d}.ply',process=False);v=np.asarray(mesh.vertices)/1000
 lo=v.min(0);hi=v.max(0);corners=np.array([[hi[k] if (i>>k)&1 else lo[k] for k in range(3)] for i in range(8)])
 edges=[(i,i^(1<<k)) for i in range(8) for k in range(3) if i<(i^(1<<k))]
 def draw(ax,pose,color,style):
  pose=np.array(pose);p=corners@pose[:3,:3].T+pose[:3,3];uv=p@K.T;uv=uv[:,:2]/uv[:,2:3]
  for a,b in edges:ax.plot(uv[[a,b],0]-x0,uv[[a,b],1]-y0,color=color,ls=style,lw=1.5)
 for ci,arm in enumerate(['original','combined','joint_combined']):
  r=next(r for r in rows if r['arm']==arm and r['object']==obj and r['frame']==frame and r['condition']=='observed' and r['seed']==(0 if arm=='original' else 42))
  ax=axes[ri,ci];ax.imshow(rgb[y0:y1,x0:x1],interpolation='nearest');draw(ax,r['ground_truth'],'#27d16c','--');draw(ax,r['prediction'],'#e843c4','-')
  ax.set_xlim(0,x1-x0);ax.set_ylim(y1-y0,0);ax.set_xticks([]);ax.set_yticks([])
  title=['Original','Adapted encoder only','Adapted encoder + pose heads'][ci]
  ax.set_title(f'{title}\nADD error: {r["add_mm"]:.2f} mm',fontsize=11)
  if ci==0:ax.set_ylabel(f'{names[obj]} — frame {frame}',fontsize=10)
fig.suptitle('Actual BOP photos with saved pose predictions',fontsize=17,weight='bold',y=.985)
fig.text(.5,.028,'Green dashed: ground-truth 3D box. Magenta solid: predicted 3D box. Boxes are projections of the CAD bounding box.\nFixed first evaluation image per object; adapted seed 42. Examples are not selected by improvement; aggregate results are in the chart.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.065,1,.95]);fig.savefig(OUT/'pose_prediction_examples.png',dpi=300,facecolor='white');fig.savefig(OUT/'pose_prediction_examples.pdf',facecolor='white');plt.close(fig)
print(OUT)
