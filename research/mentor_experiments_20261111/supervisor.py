"""Finite remote experiment queue, detached from SSH with persistent logs/status."""
import os
import signal
import sys
import subprocess
import json
import time
import fcntl
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    lock=(HERE/'queue.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another experiment supervisor is already active')
    tasks=[('real_rotation',ROOT/'research/real_rotation_20261111/run.py'),('geometry_stress',ROOT/'research/geometric_structure_20261111/completion/stress_test.py'),('verification',HERE/'verify.py'),('report',HERE/'report.py')]
    status={'pid':os.getpid(),'session_id':os.getsid(0),'state':'running','started_unix':time.time(),'jobs':[]}
    for name,script in tasks:
        entry={'name':name,'state':'running','script':str(script),'started_unix':time.time()};status['jobs'].append(entry)
        (HERE/'queue_status.json').write_text(json.dumps(status,indent=2))
        with (HERE/f'{name}.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u',str(script)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        entry.update(state='complete' if result.returncode==0 else 'failed',returncode=result.returncode,finished_unix=time.time())
        (HERE/'queue_status.json').write_text(json.dumps(status,indent=2))
        if result.returncode:
            status['state']='failed';(HERE/'queue_status.json').write_text(json.dumps(status,indent=2));return
    status['state']='complete';status['finished_unix']=time.time();(HERE/'queue_status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
