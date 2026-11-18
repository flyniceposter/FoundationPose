"""Finish analysis and artifacts after checkpointed measurement jobs complete."""
import json
import subprocess
import sys
import time
from pathlib import Path
from study import HERE,atomic


def wait_for(folder):
    path=HERE/folder/'status.json'
    while True:
        if path.exists():
            status=json.loads(path.read_text())
            if status.get('state')=='measurements_complete':return
            if status.get('state')=='failed':raise RuntimeError(status)
        time.sleep(5)


def execute(name):
    atomic(HERE/'workflow_status.json',dict(state='running',step=name))
    print('Starting',name,flush=True)
    subprocess.run([sys.executable,str(HERE/name)],cwd=HERE,check=True)


def main():
    try:
        atomic(HERE/'workflow_status.json',dict(state='running',step='waiting_for_primary_measurements'))
        wait_for('results')
        execute('probe.py')
        atomic(HERE/'workflow_status.json',dict(state='running',step='waiting_for_secondary_measurements'))
        wait_for('secondary_results')
        execute('analyze.py')
        execute('build_report.py')
        atomic(HERE/'workflow_status.json',dict(state='complete',report=str(HERE/'REPORT.md'),presentation=str(HERE/'PRESENTATION.pdf')))
    except Exception as exc:
        atomic(HERE/'workflow_status.json',dict(state='failed',error=repr(exc)))
        raise


if __name__=='__main__':main()
