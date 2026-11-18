"""Server-side hourly health audit with bounded recovery of interrupted services."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'debug/research_health'
JOBS = [('mask_study_v1', 'foundationpose-mask-study-v1.service'),
        ('point_filter_study_v1', 'foundationpose-point-filter-v1.service'),
        ('dataset_comparison_v1', 'foundationpose-comparison-report-v1.service')]


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=45)
    return dict(code=result.returncode, stdout=result.stdout[-6000:], stderr=result.stderr[-2000:])


def main():
    OUT.mkdir(exist_ok=True)
    previous = json.loads((OUT/'latest.json').read_text()) if (OUT/'latest.json').exists() else {}
    retries = previous.get('retries', {})
    report = dict(checked_local=datetime.now(ZoneInfo('America/New_York')).isoformat(),
        checked_unix=time.time(), jobs=[], alerts=[], retries=retries,
        disk_free_gb=round(shutil.disk_usage(ROOT).free/1024**3, 1))
    report['gpu'] = command(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader'])
    if report['gpu']['code']:
        report['alerts'].append('GPU query failed; inspect GPU state before further inference')
    if report['disk_free_gb'] < 20:
        report['alerts'].append('Less than 20GB free disk space')
    for folder, unit in JOBS:
        path = ROOT/'debug'/folder/'status.json'
        status = json.loads(path.read_text()) if path.exists() else dict(state='not_started')
        service = command(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'ExecMainStatus'])
        properties = dict(line.split('=',1) for line in service['stdout'].splitlines() if '=' in line)
        item = dict(folder=folder, unit=unit, status=status, service=properties)
        age = time.time()-status.get('updated_unix', time.time())
        if status['state']=='running' and age>900:
            report['alerts'].append(f'{folder}: progress has not updated for {age/60:.1f} minutes')
        active = properties.get('ActiveState') in ['active', 'activating']
        complete = status['state']=='complete'
        if not active and not complete:
            logpath = ROOT/'debug'/folder/'service.log'
            if logpath.exists():
                with logpath.open('rb') as f:
                    f.seek(max(0, logpath.stat().st_size-4000)); tail=f.read().decode(errors='replace')
            else:
                tail=''
            item['log_tail']=tail
            # Never change code, manifests, inputs or selection after failure.
            # Retry interrupted processes or explicitly transient CUDA failures;
            # data/config errors require investigation, not blind retries.
            transient = any(s in (tail+str(status)).lower() for s in ['out of memory', 'driver shutting down', 'cuda-capable device(s) is/are busy'])
            interrupted = status['state'] in ['running', 'queued', 'waiting_for_experiments']
            if (interrupted or transient) and retries.get(unit, 0)<2 and report['gpu']['code']==0:
                item['recovery'] = command(['systemctl', '--user', 'restart', unit])
                retries[unit]=retries.get(unit,0)+1
            else:
                report['alerts'].append(f'{folder}: service stopped with state {status["state"]}; inspect logs')
        report['jobs'].append(item)
    report['all_experiments_complete'] = all(j['status']['state']=='complete' for j in report['jobs'])
    temporary=OUT/'latest.tmp'; temporary.write_text(json.dumps(report,indent=2)); temporary.replace(OUT/'latest.json')
    with (OUT/'history.jsonl').open('a') as f:
        f.write(json.dumps(report)+'\n')
    print(json.dumps(report,indent=2))
    if report['all_experiments_complete']:
        command(['systemctl','--user','stop','foundationpose-hourly-health.timer'])


if __name__=='__main__':
    main()
