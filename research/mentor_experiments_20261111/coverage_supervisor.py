"""Finite follow-up after the initial remote queue, with independent logs."""
from pathlib import Path
import signal
import time
import subprocess
import sys
import json
import os
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
signal.signal(signal.SIGHUP,signal.SIG_IGN)
status={'pid':os.getpid(),'session_id':os.getsid(0),'state':'waiting_for_main_queue'}
def save(): (HERE/'coverage_status.json').write_text(json.dumps(status,indent=2))
save()
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
    state=json.loads((HERE/'queue_status.json').read_text())['state']
    if state=='complete':break
    if state=='failed':
        status['state']='blocked_by_main_queue_failure';save();raise SystemExit(1)
    time.sleep(2)
else:raise SystemExit('Main queue did not finish within two hours')
for name,script in [('dense_bank',ROOT/'research/real_rotation_20261111/dense_bank.py'),('report',HERE/'report.py')]:
    status.update(state='running',job=name);save()
    with (HERE/f'{name}_coverage.log').open('a') as log:
        result=subprocess.run([sys.executable,'-u',str(script)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status.update(state='failed',returncode=result.returncode);save();raise SystemExit(result.returncode)
status.update(state='complete');save()
