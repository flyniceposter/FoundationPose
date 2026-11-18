"""Recover logged matrices on CPU; never treat rounded logs as exact predictions."""
import re,json,hashlib,csv
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'recovery'
DATA=Path('/home/Sahil/datasets/bop/lm')
def parse(text,frame_ids):
    current=None;recover={};records=[];lines=text.splitlines();i=0
    while i<len(lines):
        match=re.search(r'\[run_pose_estimation_worker\(\)\].*i_frame:(\d+), ob_id:(\d+)',lines[i])
        if match:current=(int(match[2]),int(match[1]))
        if lines[i]=='[run_pose_estimation_worker()] pose:' and current:
            block='\n'.join(lines[i+1:i+5]);numbers=re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',block)
            if len(numbers)!=16 or not block.rstrip().endswith(']]'):raise ValueError('Incomplete logged matrix')
            p=np.asarray([float(x) for x in numbers]).reshape(4,4);obj,index=current;im=frame_ids[obj][index]
            if not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1],atol=1e-7):raise ValueError('Malformed homogeneous pose')
            k=(obj,im,obj)
            if k in recover:raise ValueError('Duplicate logged frame')
            recover[k]=p;records.append(dict(scene_id=obj,im_id=im,obj_id=obj,frame_index=index,source_line=i+1,source='rounded_original_run_log'))
            current=None;i+=4
        i+=1
    return recover,records

def main():
    log=ROOT/'debug/bop_full/run.log';raw=log.read_bytes();text=raw.decode();ids={}
    for scene in (DATA/'lm_test_all/test').iterdir():
        if scene.is_dir():ids[int(scene.name)]=[int(p.stem) for p in sorted((scene/'rgb').iterdir()) if p.is_file()]
    poses,records=parse(text,ids);result={}
    for (scene,im,obj),p in poses.items():result.setdefault(scene,{}).setdefault(f'{im:06d}',{})[obj]=p.tolist()
    with (OUT/'recovered_predictions.yml').open('x') as f:yaml.safe_dump(result,f)
    with (OUT/'recovered_index.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    manifest=dict(source=str(log),sha256=hashlib.sha256(raw).hexdigest(),recovered=len(poses),null_bytes=raw.count(b'\0'),precision='Rounded text matrices; not guaranteed bit-identical to original float32 predictions',counts={str(obj):sum(k[2]==obj for k in poses) for obj in sorted(ids)})
    (OUT/'recovery_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
