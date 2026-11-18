"""Read-only CPU diagnostics for the existing FoundationPose run."""
import argparse, csv, hashlib, json, os, platform, time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/foundationpose-research-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import scipy
from scipy.spatial import cKDTree
import trimesh, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections.abc import Mapping
ROOT = Path('/home/Sahil/FoundationPose')
DATA = Path('/home/Sahil/datasets/bop/lm')
OUT = Path(__file__).resolve().parent
IDS = [1,2,4,5,6,8,9,10,11,12,13,14,15]
PRED = ROOT/'debug/bop_full/linemod_res.yml'
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def csvwrite(name, rows):
    with open(OUT/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def lookup(d,k):
    if not isinstance(d, Mapping):return {}
    for key in (k,str(k),f'{k:06d}'):
        if key in d:return d[key]
    return {}
def vertex_errors(predicted, ground_truth):
    """BOP ADD and directional ADI, both in the input point units."""
    return (float(np.linalg.norm(predicted-ground_truth,axis=1).mean()),
            float(cKDTree(predicted).query(ground_truth,workers=1)[0].mean()))
def inventory():
    rows=[]
    for obj in IDS:
        scene=DATA/f'lm_test_all/test/{obj:06d}'
        gt=json.loads((scene/'scene_gt.json').read_text())
        rgb={int(p.stem) for p in (scene/'rgb').iterdir() if p.is_file()}
        expected=[int(i) for i,gs in gt.items() if any(g['obj_id']==obj for g in gs)]
        cameras=json.loads((scene/'scene_camera.json').read_text())
        missing_masks=missing_depth=multi=0
        for im in expected:
            matches=[j for j,g in enumerate(gt[str(im)]) if g['obj_id']==obj]
            multi+=int(len(matches)!=1)
            missing_masks+=sum(not (scene/f'mask_visib/{im:06d}_{j:06d}.png').is_file() for j in matches)
            missing_depth+=int(not (scene/f'depth/{im:06d}.png').is_file())
        rows.append(dict(obj_id=obj,gt_images=len(gt),rgb_images=len(rgb),expected_poses=len(expected),missing_rgb=len(set(expected)-rgb),missing_masks=missing_masks,missing_depth=missing_depth,missing_camera=sum(str(im) not in cameras for im in expected),ambiguous_images=multi,unique_intrinsics=len({tuple(c['cam_K']) for c in cameras.values()}),unique_depth_scales=len({c['depth_scale'] for c in cameras.values()})))
    csvwrite('dataset_inventory.csv',rows)
    return rows

def analyze():
    inv=inventory()
    before=(PRED.stat().st_size,PRED.stat().st_mtime_ns)
    predictions=yaml.safe_load(PRED.read_text())
    if not isinstance(predictions,Mapping):raise RuntimeError('Prediction YAML is not a mapping')
    if before!=(PRED.stat().st_size,PRED.stat().st_mtime_ns):raise RuntimeError('Predictions changed while loading')
    info=json.loads((DATA/'models/models_info.json').read_text())
    rows=[]; errors=[]; expected_keys=set(); actual_keys=set()
    for s,ims in predictions.items():
        for i,objs in ims.items():
            for o in objs:actual_keys.add((int(s),int(i),int(o)))
    files=[PRED,ROOT/'run_linemod.py',ROOT/'datareader.py',ROOT/'estimater.py',Path(__file__)]
    for obj in IDS:
        scene=DATA/f'lm_test_all/test/{obj:06d}'
        gtpath=scene/'scene_gt.json';files.extend([gtpath,scene/'scene_camera.json',DATA/f'models/obj_{obj:06d}.ply'])
        gt=json.loads(gtpath.read_text())
        # Full CAD vertex set, millimetres; no stochastic surface sampling.
        pts=np.asarray(trimesh.load(DATA/f'models/obj_{obj:06d}.ply',process=False).vertices)
        diameter=float(info[str(obj)]['diameter'])
        for im,gs in sorted(gt.items(),key=lambda p:int(p[0])):
            matches=[g for g in gs if g['obj_id']==obj]
            if not matches:continue
            if len(matches)!=1:raise RuntimeError('Ambiguous same-object instances')
            expected_keys.add((obj,int(im),obj))
            g=matches[0];R=np.array(g['cam_R_m2c']).reshape(3,3);t=np.array(g['cam_t_m2c'])
            value=lookup(lookup(lookup(predictions,obj),int(im)),obj)
            row=dict(scene_id=obj,im_id=int(im),obj_id=obj,status='missing',add_mm=float('inf'),adi_mm=float('inf'),translation_mm=float('inf'),rotation_deg=float('inf'),diameter_mm=diameter,add_pass=0,adi_pass=0,adds_pass=0)
            if isinstance(value, (list,tuple,np.ndarray)) and len(value):
                try:p=np.asarray(value,dtype=float)
                except (TypeError,ValueError):p=np.array([])
                valid=p.shape==(4,4) and np.isfinite(p).all()
                if valid:valid=np.allclose(p[3],[0,0,0,1],atol=1e-5) and np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-3) and abs(np.linalg.det(p[:3,:3])-1)<1e-3 and p[2,3]>0
                if valid:
                    a=pts@p[:3,:3].T+p[:3,3]*1000;b=pts@R.T+t
                    add,adi=vertex_errors(a,b)
                    rot=float(np.degrees(np.arccos(np.clip((np.trace(p[:3,:3]@R.T)-1)/2,-1,1))))
                    row.update(status='valid',add_mm=add,adi_mm=adi,translation_mm=float(np.linalg.norm(p[:3,3]*1000-t)),rotation_deg=rot,add_pass=int(add<.1*diameter),adi_pass=int(adi<.1*diameter),adds_pass=int((adi if obj in [10,11] else add)<.1*diameter))
                else:row['status']='invalid';errors.append([obj,int(im),'invalid pose'])
            elif (obj,int(im),obj) in actual_keys:
                row['status']='invalid';errors.append([obj,int(im),'malformed pose'])
            rows.append(row)
        print(f'Analyzed object {obj}',flush=True)
    csvwrite('per_pose.csv',rows)
    summaries=[]
    for obj in IDS+[0]:
        group=[r for r in rows if obj==0 or r['obj_id']==obj];valid=[r for r in group if r['status']=='valid']
        summary=dict(obj_id=obj,expected=len(group),valid=len(valid),missing=sum(r['status']=='missing' for r in group),invalid=sum(r['status']=='invalid' for r in group))
        for metric in ['add','adi','adds']:summary[metric+'_recall_0.1d']=sum(r[metric+'_pass'] for r in group)/len(group)
        for metric in ['add_mm','adi_mm','translation_mm','rotation_deg']:summary['median_'+metric]=float(np.median([r[metric] for r in valid])) if valid else None
        summaries.append(summary)
    csvwrite('summary.csv',summaries)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar([str(r['obj_id']) for r in summaries[:-1]],[100*r['adds_recall_0.1d'] for r in summaries[:-1]])
    ax.set(xlabel='Object ID',ylabel='Recall (%)',ylim=(0,100),title='Diagnostic ADD(-S) < 0.1 diameter; all expected poses')
    fig.tight_layout();fig.savefig(OUT/'recall.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots()
    for key in ['add_mm','adi_mm']:
        x=np.linspace(0,.2,101);y=[np.mean([r[key]/r['diameter_mm']<v for r in rows]) for v in x];ax.plot(x,y,label=key)
    ax.set(xlabel='Error / object diameter',ylabel='Recall',title='Diagnostic CAD vertex distance curves');ax.legend();fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=180);plt.close(fig)
    files.append(DATA/'models/models_info.json')
    files.extend(p for p in (ROOT/'weights').rglob('*') if p.is_file())
    manifest=dict(versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,trimesh=trimesh.__version__,matplotlib=matplotlib.__version__),sha256={str(p):digest(p) for p in files},unexpected_keys=sorted(actual_keys-expected_keys),errors=errors)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    total=summaries[-1];macro=float(np.mean([s['adds_recall_0.1d'] for s in summaries[:-1]]))
    verification=dict(expected=len(expected_keys),saved=len(actual_keys),missing_keys=sorted(expected_keys-actual_keys),unexpected_keys=sorted(actual_keys-expected_keys),invalid=total['invalid'],coverage_complete=actual_keys==expected_keys,all_poses_valid=total['valid']==len(expected_keys))
    (OUT/'prediction_validation.json').write_text(json.dumps(verification,indent=2))
    (OUT/'RESULTS.md').write_text(f'''# Completed diagnostic analysis

Expected poses: {total['expected']}; valid: {total['valid']}; missing: {total['missing']}; invalid: {total['invalid']}.
Unexpected prediction keys: {len(actual_keys-expected_keys)}.
Micro ADD(-S) recall at 0.1 diameter: {total['adds_recall_0.1d']:.6f}; object macro recall: {macro:.6f}.

These are diagnostic scores, not official BOP scores. Read REPORT.md for methods and limitations.
''')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);ap.add_argument('--inventory-only',action='store_true');a=ap.parse_args()
    inventory()
    if a.inventory_only:return
    if a.wait_pid:
        proc=Path(f'/proc/{a.wait_pid}')
        identity=(proc/'stat').read_text().split()[21] if proc.exists() else None
        while proc.exists():
            try:
                stat=(proc/'stat').read_text().split()
                if stat[21]!=identity or stat[2]=='Z':break
            except FileNotFoundError:break
            (OUT/'status.json').write_text(json.dumps(dict(state='waiting_for_inference',pid=a.wait_pid,checked_unix=time.time(),prediction_exists=PRED.exists())))
            time.sleep(45)
    if not PRED.exists():raise RuntimeError('Inference ended without final predictions; no accuracy claims possible. Inspect run.log.')
    analyze()
    (OUT/'status.json').write_text(json.dumps(dict(state='analysis_complete',finished_unix=time.time())))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (OUT/'status.json').write_text(json.dumps(dict(state='analysis_failed',error=str(e),checked_unix=time.time())))
        raise
