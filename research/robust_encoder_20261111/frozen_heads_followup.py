"""Persistent runner for the frozen adapted encoder control; no previous jobs rerun."""
from pathlib import Path
import json, subprocess, time
import supervisor as s
HERE=Path(__file__).resolve().parent
OUT=HERE/'frozen_heads_followup'
OUT.mkdir(exist_ok=True)
def status(**value):
    value['updated_unix']=time.time()
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(OUT/'status.json')
s.status=status

def main():
    for stage,args in [('training',['train_pose_heads.py','--arms','frozen_combined_heads']),('registration',['pose_eval.py'])]:
        marker=OUT/(stage+'_complete.json')
        if marker.exists():continue
        s.wait_gpu();status(state='running',stage=stage)
        with (OUT/(stage+'.log')).open('a',buffering=1) as log:
            result=subprocess.run([s.PYTHON,'-u',str(HERE/args[0]),*args[1:]],cwd=HERE.parents[1],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status(state='failed',stage=stage,returncode=result.returncode)
            raise RuntimeError(stage+' failed')
        marker.write_text(json.dumps({'completed_unix':time.time()}))
    subprocess.run([s.PYTHON,str(HERE/'frozen_heads_summary.py')],check=True)
    status(state='complete',report=str(OUT/'RESULTS.md'))
if __name__=='__main__':main()
