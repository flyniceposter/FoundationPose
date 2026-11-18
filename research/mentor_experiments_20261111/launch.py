"""Start the finite experiment queue in its own remote session."""
import os
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
status=HERE/'queue_status.json'
if status.exists():
    saved=json.loads(status.read_text())
    if saved.get('state')=='running':
        try:os.kill(saved['pid'],0)
        except ProcessLookupError:pass
        else:raise SystemExit('Existing supervisor is active; refusing duplicate GPU jobs')
with (HERE/'supervisor.log').open('a') as output:
    child=subprocess.Popen([sys.executable,'-u',str(HERE/'supervisor.py')],cwd=HERE.parents[1],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(HERE/'supervisor_pid.json').write_text(json.dumps({'pid':child.pid,'session_id':os.getsid(child.pid),'detached':True},indent=2))
print('Detached remote supervisor:',child.pid,'session:',os.getsid(child.pid),flush=True)
