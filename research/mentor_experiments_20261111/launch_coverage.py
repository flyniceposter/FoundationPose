from pathlib import Path
import os
import subprocess
import sys
import json
HERE=Path(__file__).resolve().parent
status=HERE/'coverage_status.json'
if status.exists():
    saved=json.loads(status.read_text())
    if saved.get('state') in ['running','waiting_for_main_queue']:
        try:os.kill(saved['pid'],0)
        except ProcessLookupError:pass
        else:raise SystemExit('Coverage supervisor already active')
with (HERE/'coverage_supervisor.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'-u',str(HERE/'coverage_supervisor.py')],cwd=HERE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print('Detached coverage follow-up:',p.pid,'session:',os.getsid(p.pid))
