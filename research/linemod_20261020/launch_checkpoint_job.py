"""Own the inference child, require exit 0 and completion marker, then run CPU stages."""
import argparse,fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reuse-recovered',type=Path);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    cmd=[sys.executable,str(HERE/'checkpoint_runner.py'),'--output',str(args.output)]
    if args.reuse_recovered:cmd+=['--reuse-recovered',str(args.reuse_recovered)]
    if not args.run:subprocess.run(cmd,check=True);return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with open(out/'.job.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=out/'job_status.json'
        if status.exists() and json.loads(status.read_text()).get('state','').startswith('analysis_complete'):
            raise RuntimeError('Analysis already complete; refusing to overwrite results')
        # Separate from original outputs, logs, and report.
        marker=out/'completion.json'
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/foundationpose-research-mpl')
        if not marker.exists():
            with (out/'inference.log').open('a') as log:
                result=subprocess.run(cmd+['--run'],env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode!=0:
                (out/'job_status.json').write_text(json.dumps(dict(state='inference_failed',exit_code=result.returncode)))
                raise RuntimeError('Inference did not exit normally; checkpoints retained; CPU scoring not started')
        import analyze as a
        completion=json.loads(marker.read_text())
        if not completion['inference_returned_normally'] or a.digest(out/'linemod_res.yml')!=completion['prediction_sha256']:raise RuntimeError('Completion marker/hash mismatch')
        a.OUT=out/'analysis';a.OUT.mkdir(exist_ok=True);a.PRED=out/'linemod_res.yml'
        report=a.OUT/'REPORT.md'
        if not report.exists():report.write_text('# Checkpointed LINEMOD recovery experiment\n\n'+completion['precision_note']+'.\n\nDiagnostic evaluation with GT visible masks; no official BOP benchmark score is claimed. Original interrupted artifacts are preserved separately.\n')
        a.analyze()
        import extended as e
        e.run()
        import supervise as supervisor
        supervisor.OUT=a.OUT;supervisor.completed_report()
        import finalize as finalizer
        finalizer.run()
        validation=json.loads((a.OUT/'prediction_validation.json').read_text())
        (out/'job_status.json').write_text(json.dumps(dict(state='analysis_complete' if validation['coverage_complete'] and validation['all_poses_valid'] else 'analysis_complete_with_issues',recovered_rounded_poses=completion['reused_rounded_log_poses'],analysis=str(a.OUT)),indent=2))

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();sys.exit(1)
