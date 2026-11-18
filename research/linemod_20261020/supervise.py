"""Single CPU supervisor. Observe existing GPU inference; never launch inference.
Run in host PID namespace. Uses an exclusive lock to prevent duplicate supervisors.
"""
import argparse,csv,fcntl,json,os,re,subprocess,sys,time,traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
LOG=ROOT/'debug/bop_full/run.log'
PRED=ROOT/'debug/bop_full/linemod_res.yml'

def atomic(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def stamp():return datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
def proc(pid):
    try:
        # Skip comm (which can contain spaces); fields below start at process state.
        value=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(state=value[0],start_ticks=int(value[19]))
    except FileNotFoundError:return None

def completed_report():
    def read(name):
        with open(OUT/name) as f:return list(csv.DictReader(f))
    primary=read('summary.csv');extended=read('extended_summary.csv');validation=json.loads((OUT/'prediction_validation.json').read_text());total=primary[-1];extra=extended[-1]
    by_obj={r['obj_id']:r for r in extended}
    def number(key):return f"{float(extra[key]):.3f}" if extra[key] else 'N/A'
    text=f'''## Measured results — {stamp()}

Analysis completed on the final saved predictions. Expected: {total['expected']}; valid: {total['valid']}; missing: {total['missing']}; invalid: {total['invalid']}. Exact key coverage: {validation['coverage_complete']}. All poses valid: {validation['all_poses_valid']}.

Pooled valid-prediction translation error: mean {number('mean_translation_mm')} mm, median {number('median_translation_mm')} mm, p90 {number('p90_translation_mm')} mm. Raw rotation error: mean {number('mean_rotation_deg')}°, median {number('median_rotation_deg')}°, p90 {number('p90_rotation_deg')}°. Raw rotation is not symmetry corrected.

| Object | Valid / expected | Mean translation (mm) | Mean raw rotation (°) | ADD(-S) recall < 0.1d | MSSD recall < 0.1d | MSPD recall < 5px |
| --- | --- | --- | --- | --- | --- | --- |
'''
    for r in primary:
        e=by_obj[r['obj_id']];label='Pooled' if r['obj_id']=='0' else r['obj_id']
        def fmt(k):return f"{float(e[k]):.3f}" if e[k] else 'N/A'
        text+=f"| {label} | {r['valid']} / {r['expected']} | {fmt('mean_translation_mm')} | {fmt('mean_rotation_deg')} | {100*float(r['adds_recall_0.1d']):.2f}% | {100*float(e['mssd_recall_0_1d']):.2f}% | {100*float(e['mspd_recall_5px']):.2f}% |\n"
    macro=sum(float(r['adds_recall_0.1d']) for r in primary[:-1])/len(primary[:-1])
    worst=sorted(primary[:-1],key=lambda r:float(r['adds_recall_0.1d']))[:3]
    text+=f"\nObject macro ADD(-S) recall: {100*macro:.2f}%. Lowest observed per-object recalls: "+', '.join(f"object {r['obj_id']} ({100*float(r['adds_recall_0.1d']):.2f}%)" for r in worst)+'. These rank observed performance; the cause requires inspecting failure images and controlled experiments.\n'
    text+='\nThese are diagnostic scores on the full 13-object split with GT visible masks. No official BOP AR, VSD score, or benchmark submission is claimed. See [recall](recall.png), [error curves](curves.png), [error distributions](error_distributions.png), and [selected examples](qualitative_examples.csv).\n'
    (OUT/'MEASURED_RESULTS.md').write_text(text)
    report=OUT/'REPORT.md';content=report.read_text();marker='\n<!-- AUTO_MEASURED_RESULTS -->\n';content=content.split(marker)[0];report.write_text(content+marker+text)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,default=3257117);args=ap.parse_args()
    lock=open(OUT/'.supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=proc(args.pid)
    if identity is not None:
        cmd=Path(f'/proc/{args.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'run_linemod.py' not in cmd or str(ROOT/'debug/bop_full') not in cmd:raise RuntimeError('Unexpected process command; refusing to monitor this PID')
    elif not PRED.exists():raise RuntimeError('Inference process absent and no prediction file')
    with open(OUT/'dataset_inventory.csv') as f:inventory=list(csv.DictReader(f))
    expected=sum(int(r['expected_poses']) for r in inventory)
    offset=0;pending='';started=completed=0;current=None;errors=[];recent=deque();first=time.time();observations=OUT/'progress_samples.csv'
    if not observations.exists():observations.write_text('unix_time,local_time,started,completed,obj_id,frame,elapsed_s,rolling_poses_per_s,eta_s\n')
    while True:
        with open(LOG) as f:f.seek(offset);chunk=f.read();offset=f.tell()
        lines=(pending+chunk).split('\n');pending=lines.pop()
        for line in lines:
            match=re.search(r'i_frame:(\d+), ob_id:(\d+)',line)
            if match:started+=1;current=(int(match[2]),int(match[1]))
            if '[run_pose_estimation_worker()] pose:' in line:completed+=1
            if any(x in line for x in ['Traceback (most recent call last)','RuntimeError:','CUDA out of memory','ob_mask not found','valid too small, return']):errors.append(line)
        now=time.time();recent.append((now,completed))
        while len(recent)>2 and now-recent[0][0]>300:recent.popleft()
        rolling=(completed-recent[0][1])/(now-recent[0][0]) if now>recent[0][0] else None
        live=proc(args.pid);running=live is not None and identity is not None and live['start_ticks']==identity['start_ticks'] and live['state']!='Z'
        elapsed=None
        if identity is not None:
            uptime=float(Path('/proc/uptime').read_text().split()[0]);elapsed=uptime-identity['start_ticks']/os.sysconf('SC_CLK_TCK')
        overall=completed/elapsed if elapsed and elapsed>0 else None
        rate=rolling if rolling and len(recent)>1 else overall
        eta=(expected-completed)/rate if rate and rate>0 else None
        state=dict(state='waiting_for_inference' if running else 'checking_completion',pid=args.pid,supervisor_pid=os.getpid(),checked_local=stamp(),checked_unix=now,expected=expected,started=started,completed_logged=completed,current_object=current[0] if current else None,current_frame=current[1] if current else None,elapsed_s=elapsed,rolling_poses_per_s=rolling,overall_poses_per_s=overall,eta_s=eta,eta_note='Projection of measured speed; object/setup costs and future slowdowns may vary.',log_flags=errors,prediction_exists=PRED.exists())
        atomic('progress.json',state);atomic('status.json',state)
        with open(observations,'a',newline='') as f:csv.writer(f).writerow([now,stamp(),started,completed,*(current or ('','')),elapsed,rolling,eta])
        (OUT/'PROGRESS.md').write_text(f"Checked {stamp()}. PID {args.pid} active: {running}. Logged complete: {completed}/{expected}; started: {started}. Current object/frame: {current}. Measured overall speed: {overall or 0:.3f} poses/s; rolling speed: {rolling or 0:.3f} poses/s. Estimated inference remaining: {eta/3600 if eta is not None else float('nan'):.2f} hours. Post-processing time is additional. This estimate changes with measured processing speed.\n")
        if not running:break
        time.sleep(45)
    if not PRED.is_file():raise RuntimeError('Inference ended without final YAML; no accuracy computed')
    if any('Traceback' in e or 'RuntimeError:' in e or 'CUDA out of memory' in e for e in errors):raise RuntimeError('Inference error found in log; inspect before scoring')
    # A file can exist while being written; process termination and stability are both required.
    stat=PRED.stat();time.sleep(3)
    if (stat.st_size,stat.st_mtime_ns)!=(PRED.stat().st_size,PRED.stat().st_mtime_ns):raise RuntimeError('Prediction file still changing')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/foundationpose-research-mpl')
    for filename,log in [('analyze.py','analysis.log'),('extended.py','extended.log')]:
        atomic('status.json',dict(state='cpu_analysis',stage=filename,checked_local=stamp()))
        with open(OUT/log,'a') as f:subprocess.run([sys.executable,str(OUT/filename)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    completed_report()
    validation=json.loads((OUT/'prediction_validation.json').read_text())
    atomic('status.json',dict(state='analysis_complete' if validation['coverage_complete'] and validation['all_poses_valid'] else 'analysis_complete_with_prediction_issues',finished_local=stamp(),validation=validation))
    print('Analysis and measured report complete',stamp(),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic('status.json',dict(state='analysis_failed',error=str(e),checked_local=stamp()));traceback.print_exc();sys.exit(1)
