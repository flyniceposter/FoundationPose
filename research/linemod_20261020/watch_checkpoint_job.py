"""Read-only CPU monitoring of the approved user service and checkpoint progress."""
import csv,json,subprocess,time,fcntl
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
BASE=Path(__file__).resolve().parent
RUN=BASE.parents[1]/'debug/bop_recovery_20261021'
UNIT='foundationpose-bop-recovery-20261021.service'
def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.monitor-tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def now():return datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
def main():
    with (RUN/'.monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        history=deque();samples=RUN/'progress_samples.csv'
        if not samples.exists():samples.write_text('unix_time,local_time,checkpoints,new_predictions,rate_per_s,remaining_s\n')
        while True:
            stamp=time.time();manifest=json.loads((RUN/'run_manifest.json').read_text());reused=manifest['plan']['reused_rounded_log_poses'];expected=manifest['plan']['expected']
            path=RUN/'checkpoint_status.json';progress=json.loads(path.read_text()) if path.exists() else dict(checkpoints=reused,last_key=None)
            n=progress['checkpoints'];history.append((stamp,n))
            while len(history)>2 and stamp-history[0][0]>300:history.popleft()
            rate=(n-history[0][1])/(stamp-history[0][0]) if stamp>history[0][0] else None
            eta=(expected-n)/rate if rate and rate>0 else None
            state=subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],capture_output=True,text=True)
            properties=dict(line.split('=',1) for line in state.stdout.splitlines() if '=' in line)
            job=RUN/'job_status.json';jobstatus=json.loads(job.read_text()) if job.exists() else {}
            phase=jobstatus.get('state') or ('cpu_analysis' if (RUN/'completion.json').exists() else 'inference_running')
            if properties.get('ActiveState') not in ['active','activating'] and not phase.startswith('analysis_complete'):
                phase='stopped_without_completed_analysis'
            data=dict(state=phase,checked_local=now(),expected=expected,checkpoints=n,completed_percent=100*n/expected,reused_rounded_log_poses=reused,new_predictions=n-reused,last_key=progress['last_key'],rolling_poses_per_s=rate,eta_s=eta,service=properties,run_directory=str(RUN),precision='5186 rounded-log poses plus native-precision continuation')
            atomic(RUN/'live_status.json',data);atomic(BASE/'status.json',data)
            (BASE/'PROGRESS.md').write_text(f"Checked {now()}. Approved recovery service: {phase}. Saved poses: {n}/{expected} ({100*n/expected:.2f}%); rounded recovered: {reused}; new native-precision: {n-reused}. Last scene/image/object: {progress['last_key']}. Recent speed: {rate or 0:.3f} new poses/s. Estimated inference remaining: {eta/3600 if eta is not None else float('nan'):.2f} hours (excludes CPU analysis). Files: debug/bop_recovery_20261021/.\n")
            with samples.open('a',newline='') as f:csv.writer(f).writerow([stamp,now(),n,n-reused,rate,eta])
            if phase.startswith('analysis_complete'):
                marker='\n<!-- CHECKPOINT_RECOVERY_FINAL_RESULTS -->\n';report=BASE/'REPORT.md'
                measured=(RUN/'analysis/MEASURED_RESULTS.md').read_text();failure=(RUN/'analysis/FAILURE_ANALYSIS.md').read_text()
                report.write_text(report.read_text().split(marker)[0]+marker+'## Approved recovery: completed measurements\n\nThe first 5,186 poses were recovered from rounded log text; the remaining 10,614 were computed by the approved continuation. This is a mixed-precision recovered baseline, not a pristine full-precision rerun or official BOP score.\n\n'+measured+'\n\n'+failure)
                break
            if phase=='stopped_without_completed_analysis':break
            time.sleep(45)
if __name__=='__main__':main()
