from pathlib import Path
import json, subprocess, sys, time, traceback
HERE=Path(__file__).resolve().parent
PYTHON='/home/Sahil/miniforge3/envs/foundationpose/bin/python'

def status(**value):
    value['updated_unix']=time.time();tmp=HERE/'service_status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(HERE/'service_status.json')

def wait_gpu():
    ready=0
    while ready<3:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        free,util=[int(v.strip()) for v in output.splitlines()[0].split(',')]
        ready=ready+1 if free>=20480 and util<40 else 0
        status(state='waiting_for_gpu',free_mib=free,utilization_percent=util,ready_checks=ready,required_ready_checks=3)
        print('Waiting for GPU:',free,'MiB free,',util,'percent busy',flush=True)
        if ready<3:time.sleep(30)

def main():
    stages=[('baseline', ['pose_eval.py','--baseline-only']),('training',['train.py']),('pose_heads',['train_pose_heads.py']),('pose',['pose_eval.py']),('report',['report.py'])]
    for stage,args in stages:
        marker=HERE/f'.{stage}_done'
        if marker.exists():continue
        if stage!='report':wait_gpu()
        status(state='running',stage=stage)
        with (HERE/f'{stage}.log').open('a',buffering=1) as log:
            result=subprocess.run([PYTHON,'-u',str(HERE/args[0]),*args[1:]],cwd=str(HERE.parents[1]),stdout=log,stderr=subprocess.STDOUT)
        subprocess.run([PYTHON,str(HERE/'report.py')],check=False)
        if result.returncode:
            status(state='failed',stage=stage,returncode=result.returncode,log=str(HERE/f'{stage}.log'))
            raise RuntimeError(f'{stage} failed; see {stage}.log')
        marker.write_text(str(time.time()))
    status(state='complete',report=str(HERE/'REPORT.html'))
if __name__=='__main__':main()
