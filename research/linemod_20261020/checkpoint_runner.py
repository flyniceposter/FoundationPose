"""Isolated adapter for the unchanged upstream runner; GPU use requires --run.
Default is a CPU-only plan. Writes only to a fresh output directory.
"""
import argparse,fcntl,hashlib,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/home/Sahil/datasets/bop/lm')
IDS=[1,2,4,5,6,8,9,10,11,12,13,14,15]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',delete=False) as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno());temporary=Path(f.name)
    temporary.replace(path)
def atomic_pose(path,pose):
    with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix=path.name+'.',delete=False) as f:
        np.save(f,pose,allow_pickle=False);f.flush();os.fsync(f.fileno());temporary=Path(f.name)
    temporary.replace(path)
def valid_pose(p):return p.shape==(4,4) and np.isfinite(p).all() and np.allclose(p[3],[0,0,0,1],atol=1e-6)
def expected():
    keys=[]
    for obj in IDS:
        scene=DATA/f'lm_test_all/test/{obj:06d}';gt=json.loads((scene/'scene_gt.json').read_text())
        for image in sorted((scene/'rgb').iterdir()):
            if not image.is_file():continue
            im=int(image.stem)
            if sum(g['obj_id']==obj for g in gt[str(im)])!=1:raise RuntimeError('Ambiguous GT matching')
            keys.append((obj,im,obj))
    return keys

def checkpoint_name(key):return f'{key[0]:06d}_{key[1]:06d}_{key[2]:06d}.npy'
def gpu_conflict():
    # Read-only host scan. Exclude this adapter and require no existing baseline process.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=path.read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        if int(path.parent.name)==os.getpid():continue
        if any(v.decode(errors='replace').endswith('/run_linemod.py') or v==b'run_linemod.py' for v in args):raise RuntimeError('An existing run_linemod.py is active; refusing duplicate inference')

def run(args):
    keys=expected();keyset=set(keys);out=args.output.resolve();recovered={}
    if args.reuse_recovered:
        raw=yaml.safe_load(args.reuse_recovered.read_text())
        for s,images in raw.items():
            for im,objects in images.items():
                for obj,value in objects.items():
                    key=(int(s),int(im),int(obj));p=np.asarray(value)
                    if key not in keyset or not valid_pose(p):raise RuntimeError('Invalid recovery key or pose')
                    recovered[key]=p
    planned=dict(expected=len(keys),reused_rounded_log_poses=len(recovered),new_gpu_registrations=len(keys)-len(recovered),output=str(out),core_modified=False)
    print(json.dumps(planned,indent=2),flush=True)
    if not args.run:return
    out.mkdir(parents=True,exist_ok=True);cp=out/'checkpoints';cp.mkdir(exist_ok=True)
    with open(ROOT/'research/.foundationpose_inference.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_conflict()
        manifest=dict(plan=planned,core_sha256={str(ROOT/name):sha(ROOT/name) for name in ['run_linemod.py','estimater.py','datareader.py','Utils.py']},weights_sha256={str(p):sha(p) for p in (ROOT/'weights').rglob('*') if p.is_file()},recovery_sha256=sha(args.reuse_recovered) if args.reuse_recovered else None,adapter_sha256=sha(__file__))
        meta=out/'run_manifest.json'
        if meta.exists():
            if json.loads(meta.read_text())!=manifest:raise RuntimeError('Resume manifest mismatch; refusing to mix configurations')
        else:
            if list(cp.iterdir()) or (out/'linemod_res.yml').exists():raise RuntimeError('Untracked output artifacts exist')
            atomic_json(meta,manifest)
        if (out/'completion.json').exists():raise RuntimeError('Job already completed; refusing to overwrite completed results')
        for key,p in recovered.items():
            path=cp/checkpoint_name(key)
            if not path.exists():atomic_pose(path,p)
        # Heavy imports only after explicit --run, never during planning or testing.
        sys.path.insert(0,str(ROOT));import run_linemod as upstream
        upstream.opt=SimpleNamespace(linemod_dir=str(DATA),use_reconstructed_mesh=0,debug=0,debug_dir=str(out),ref_view_dir='unused')
        upstream.detect_type='mask';upstream.set_seed(0)
        original=upstream.run_pose_estimation_worker
        completed_count=sum(1 for _ in cp.glob('*.npy'))
        def worker(reader,i_frames,*rest,**kwargs):
            nonlocal completed_count
            if len(i_frames)!=1:raise RuntimeError('Checkpoint adapter expects one frame per worker call')
            obj=int(kwargs['ob_id'] if 'ob_id' in kwargs else rest[2]);im=str(reader.id_strs[i_frames[0]]);scene=int(reader.get_video_id());key=(scene,int(im),obj);path=cp/checkpoint_name(key)
            if path.exists():
                p=np.load(path,allow_pickle=False)
                if not valid_pose(p):raise RuntimeError('Invalid saved checkpoint')
                result={scene:{im:{obj:p}}}
            else:
                result=original(reader,i_frames,*rest,**kwargs);p=np.asarray(result[scene][im][obj])
                if not valid_pose(p):raise RuntimeError('Malformed inferred pose')
                atomic_pose(path,p)
                completed_count+=1
            atomic_json(out/'checkpoint_status.json',dict(last_key=key,checkpoints=completed_count,expected=len(keys)))
            return result
        upstream.run_pose_estimation_worker=worker
        upstream.run_pose_estimation()
        saved=yaml.safe_load((out/'linemod_res.yml').read_text());actual={(int(s),int(im),int(o)) for s,images in saved.items() for im,objects in images.items() for o in objects}
        if actual!=set(keys):raise RuntimeError('Final prediction keys differ from expected set')
        atomic_json(out/'completion.json',dict(inference_returned_normally=True,expected=len(keys),saved=len(actual),prediction_sha256=sha(out/'linemod_res.yml'),reused_rounded_log_poses=len(recovered),precision_note='Recovered prefix is rounded log text; remaining poses are full precision' if recovered else 'All poses recomputed and saved at native precision'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reuse-recovered',type=Path);ap.add_argument('--run',action='store_true');run(ap.parse_args())
