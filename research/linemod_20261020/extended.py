"""Supplemental CPU BOP-formula diagnostics and qualitative examples.
Independent of inference; --wait waits for analyze.py to finish.
"""
import argparse,csv,json,time
from pathlib import Path
import analyze as a
import numpy as np
import trimesh,yaml,cv2
import matplotlib.pyplot as plt

def project(x,K):
    q=x@K.T
    if np.any(q[:,2]<=0):return None
    return q[:,:2]/q[:,2,None]

def distances(pts,R,t,Rg,tg,K,symmetries):
    predicted=pts@R.T+t;uv=project(predicted,K)
    mssd=mspd=float('inf')
    for s in symmetries:
        target=(pts@s[:3,:3].T+s[:3,3])@Rg.T+tg
        mssd=min(mssd,float(np.linalg.norm(predicted-target,axis=1).max()))
        vg=project(target,K)
        if uv is not None and vg is not None:mspd=min(mspd,float(np.linalg.norm(uv-vg,axis=1).max()))
    return mssd,mspd

def run():
    with open(a.OUT/'per_pose.csv') as f:rows=list(csv.DictReader(f))
    pred=yaml.safe_load(a.PRED.read_text());info=json.loads((a.DATA/'models_eval/models_info.json').read_text())
    meshes={};ground={};cams={};records=[];provenance={}
    for obj in a.IDS:
        if info[str(obj)].get('symmetries_continuous'):raise RuntimeError('Continuous symmetries require discretization; unsupported by this diagnostic')
        model=a.DATA/f'models_eval/obj_{obj:06d}.ply';meshes[obj]=np.asarray(trimesh.load(model,process=False).vertices)
        scene=a.DATA/f'lm_test_all/test/{obj:06d}';ground[obj]=json.loads((scene/'scene_gt.json').read_text());cams[obj]=json.loads((scene/'scene_camera.json').read_text());provenance[str(model)]=a.digest(model)
    for r in rows:
        obj=int(r['obj_id']);im=int(r['im_id']);meta=info[str(obj)]
        g=next(g for g in ground[obj][str(im)] if g['obj_id']==obj);K=np.array(cams[obj][str(im)]['cam_K']).reshape(3,3)
        mssd=mspd=float('inf')
        if r['status']=='valid':
            p=np.asarray(a.lookup(a.lookup(a.lookup(pred,obj),im),obj));syms=[np.eye(4)]+[np.array(s).reshape(4,4) for s in meta.get('symmetries_discrete',[])]
            mssd,mspd=distances(meshes[obj],p[:3,:3],p[:3,3]*1000,np.array(g['cam_R_m2c']).reshape(3,3),np.array(g['cam_t_m2c']),K,syms)
        records.append(dict(scene_id=obj,im_id=im,obj_id=obj,status=r['status'],mssd_mm=mssd,mspd_px=mspd,mssd_normalized=mssd/meta['diameter'],mssd_pass_0_1d=int(mssd<.1*meta['diameter']),mspd_pass_5px=int(mspd<5)))
    a.csvwrite('bop_diagnostic_per_pose.csv',records)
    summaries=[]
    for obj in a.IDS+[0]:
        ix=[i for i,r in enumerate(rows) if obj==0 or int(r['obj_id'])==obj];valid=[rows[i] for i in ix if rows[i]['status']=='valid']
        d=dict(obj_id=obj,expected=len(ix),valid=len(valid))
        for metric in ['translation_mm','rotation_deg','add_mm','adi_mm']:
            values=[float(r[metric]) for r in valid]
            for label,func in [('mean',np.mean),('median',np.median),('p90',lambda v:np.percentile(v,90))]:d[f'{label}_{metric}']=float(func(values)) if values else None
        d['mssd_recall_0_1d']=np.mean([records[i]['mssd_pass_0_1d'] for i in ix]);d['mspd_recall_5px']=np.mean([records[i]['mspd_pass_5px'] for i in ix]);summaries.append(d)
    a.csvwrite('extended_summary.csv',summaries)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['translation_mm','rotation_deg']):
        values=[[float(r[metric]) for r in rows if int(r['obj_id'])==obj and r['status']=='valid'] for obj in a.IDS]
        ax.boxplot(values,tick_labels=[str(obj) for obj in a.IDS],showfliers=False);ax.set(xlabel='Object ID',ylabel=metric,title='Valid predictions; outliers hidden')
    fig.tight_layout();fig.savefig(a.OUT/'error_distributions.png',dpi=160);plt.close(fig)
    selected=[];image_dir=a.OUT/'examples';image_dir.mkdir(exist_ok=True)
    for obj in a.IDS:
        candidates=[r for r in rows if int(r['obj_id'])==obj and r['status']=='valid']
        def score(r):return float(r['adi_mm'] if obj in [10,11] else r['add_mm'])/float(r['diameter_mm'])
        for label,group in [('success',[r for r in candidates if int(r['adds_pass'])]),('failure',[r for r in candidates if not int(r['adds_pass'])])]:
            if not group:continue
            r=(min if label=='success' else max)(group,key=score);im=int(r['im_id']);scene=a.DATA/f'lm_test_all/test/{obj:06d}'
            image_path=next(iter(sorted((scene/'rgb').glob(f'{im:06d}.*'))));image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
            p=np.asarray(a.lookup(a.lookup(a.lookup(pred,obj),im),obj));g=next(g for g in ground[obj][str(im)] if g['obj_id']==obj);K=np.array(cams[obj][str(im)]['cam_K']).reshape(3,3)
            pts=meshes[obj];lo=pts.min(0);hi=pts.max(0);corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
            fig,ax=plt.subplots(figsize=(8,6));ax.imshow(image)
            for R,t,color,name in [(p[:3,:3],p[:3,3]*1000,'red','prediction'),(np.array(g['cam_R_m2c']).reshape(3,3),np.array(g['cam_t_m2c']),'lime','GT')]:
                uv=project(corners@R.T+t,K)
                if uv is None:continue
                for j in range(8):
                    for bit in [1,2,4]:
                        k=j^bit
                        if j<k:ax.plot(uv[[j,k],0],uv[[j,k],1],color=color,linewidth=1.5,label=name if j==0 and bit==1 else None)
            ax.set(xlim=(0,image.shape[1]),ylim=(image.shape[0],0),title=f'Object {obj}, frame {im}: {label}; ADD(-S)/d={score(r):.3f}');ax.legend();ax.axis('off');fig.tight_layout()
            filename=f'examples/obj_{obj:02d}_{label}_{im:06d}.png';fig.savefig(a.OUT/filename,dpi=140);plt.close(fig)
            selected.append(dict(obj_id=obj,im_id=im,classification=label,normalized_adds=score(r),file=filename))
    if selected:a.csvwrite('qualitative_examples.csv',selected)
    provenance[str(a.DATA/'models_eval/models_info.json')]=a.digest(a.DATA/'models_eval/models_info.json');provenance[str(Path(__file__))]=a.digest(Path(__file__))
    (a.OUT/'extended_manifest.json').write_text(json.dumps(provenance,indent=2))
    (a.OUT/'extended_status.json').write_text(json.dumps(dict(state='complete',finished_unix=time.time(),examples=len(selected))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    if args.wait:
        while True:
            status=json.loads((a.OUT/'status.json').read_text())
            if status['state']=='analysis_complete':break
            if status['state']=='analysis_failed':raise RuntimeError('Primary analysis failed: '+str(status))
            (a.OUT/'extended_status.json').write_text(json.dumps(dict(state='waiting_for_primary_analysis',checked_unix=time.time())))
            time.sleep(45)
    run()
if __name__=='__main__':
    try:main()
    except Exception as e:
        (a.OUT/'extended_status.json').write_text(json.dumps(dict(state='failed',error=str(e))));raise
