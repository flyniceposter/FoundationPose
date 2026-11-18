"""CPU-only failure stratification and final report, after validated baseline analysis."""
import argparse,csv,json,time,hashlib
from pathlib import Path
import analyze as a
import numpy as np
import matplotlib.pyplot as plt
MARKER='\n<!-- AUTO_FAILURE_ANALYSIS -->\n'

def load(name):
    with open(a.OUT/name) as f:return list(csv.DictReader(f))
def key(r):return tuple(int(r[k]) for k in ['scene_id','im_id','obj_id'])
def write_status(state,**kwargs):
    path=a.OUT/'finalization_status.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(dict(state=state,checked_unix=time.time(),**kwargs),indent=2));tmp.replace(path)

def run():
    rows=load('per_pose.csv');bop={key(r):r for r in load('bop_diagnostic_per_pose.csv')};validation=json.loads((a.OUT/'prediction_validation.json').read_text())
    metadata={};provenance={};split=[]
    for obj in a.IDS:
        scene=a.DATA/f'lm_test_all/test/{obj:06d}'
        gt=json.loads((scene/'scene_gt.json').read_text());info_path=scene/'scene_gt_info.json';info=json.loads(info_path.read_text());provenance[str(info_path)]=a.digest(info_path)
        for im,gs in gt.items():
            j=next(j for j,g in enumerate(gs) if g['obj_id']==obj);g=gs[j];entry=info[im][j]
            t=np.array(g['cam_t_m2c']);metadata[(obj,int(im),obj)]=dict(gt_z_mm=float(t[2]),gt_distance_mm=float(np.linalg.norm(t)),visib_fract=float(entry['visib_fract']),visible_pixels=int(entry['px_count_visib']))
        # Fixed, deterministic per-object allocation; no GPU jobs launched.
        ranked=sorted((r for r in rows if int(r['obj_id'])==obj),key=lambda r:hashlib.sha256(f"mask-study-v1/{obj}/{r['im_id']}".encode()).hexdigest())
        for index,r in enumerate(ranked[:40]):split.append(dict(obj_id=obj,im_id=int(r['im_id']),split='development' if index<20 else 'held_out',selection='sha256(mask-study-v1/obj_id/im_id)'))
    joined=[]
    for r in rows:
        b=bop[key(r)];m=metadata[key(r)];d=dict(r,**m)
        for metric in ['mssd_mm','mspd_px','mssd_normalized','mssd_pass_0_1d','mspd_pass_5px']:d[metric]=b[metric]
        d['failure']=int(not int(r['adds_pass']))
        d['high_raw_rotation_good_geometry']=int(r['status']=='valid' and float(r['rotation_deg'])>90 and int(b['mssd_pass_0_1d'])==1)
        joined.append(d)
    a.csvwrite('failure_metadata.csv',joined);a.csvwrite('proposed_mask_study_split.csv',split)
    strata=[]
    # Fixed diagnostic bins. Counts and pooled object mix must accompany interpretation.
    dimensions=[('visibility',[(0,.5,'<0.5'),(.5,.8,'0.5–0.8'),(.8,float('inf'),'≥0.8')], 'visib_fract'),('camera_distance_mm',[(0,800,'<800'),(800,1000,'800–1000'),(1000,float('inf'),'≥1000')],'gt_distance_mm')]
    for obj in [0]+a.IDS:
        for dimension,bins,column in dimensions:
            for low,high,label in bins:
                group=[r for r in joined if (obj==0 or int(r['obj_id'])==obj) and low<=float(r[column])<high]
                if not group:continue
                valid=[r for r in group if r['status']=='valid']
                strata.append(dict(obj_id=obj,dimension=dimension,bin=label,n=len(group),valid=len(valid),adds_recall=sum(int(r['adds_pass']) for r in group)/len(group),mssd_recall=sum(int(r['mssd_pass_0_1d']) for r in group)/len(group),mean_translation_mm=float(np.mean([float(r['translation_mm']) for r in valid])) if valid else None,mean_rotation_deg=float(np.mean([float(r['rotation_deg']) for r in valid])) if valid else None))
    a.csvwrite('failure_strata.csv',strata)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(dimension,_,_) in zip(axes,dimensions):
        group=[r for r in strata if r['obj_id']==0 and r['dimension']==dimension]
        bars=ax.bar([r['bin'] for r in group],[100*r['adds_recall'] for r in group])
        for bar,r in zip(bars,group):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"n={r['n']}",ha='center',fontsize=8)
        ax.set(xlabel=dimension,ylabel='Diagnostic ADD(-S) recall (%)',ylim=(0,110))
    fig.suptitle('Pooled descriptive strata; object mix may confound comparisons');fig.tight_layout();fig.savefig(a.OUT/'failure_strata.png',dpi=170);plt.close(fig)
    summary=load('summary.csv');extended={r['obj_id']:r for r in load('extended_summary.csv')};hardest=sorted(summary[:-1],key=lambda r:float(r['adds_recall_0.1d']))[:3]
    symmetric_flags=sum(int(r['high_raw_rotation_good_geometry']) for r in joined)
    complete=validation['coverage_complete'] and validation['all_poses_valid']
    report='## Observed failure patterns and follow-up experiment\n\n'
    report+=f"Final baseline coverage and validity passed: **{complete}**. "
    if not complete:report+='Treat this as an incomplete/invalid-prediction diagnostic; repair the experiment before claiming a complete baseline. '
    report+='All expected poses remain in recall denominators. Error means describe valid poses only.\n\n'
    report+='| Lowest ADD(-S) recall objects | Recall | Mean translation (mm) | Mean raw rotation (°) |\n| --- | --- | --- | --- |\n'
    for r in hardest:
        ex=extended[r['obj_id']]
        def fmt(k):return f"{float(ex[k]):.3f}" if ex[k] else 'N/A'
        report+=f"| {r['obj_id']} | {100*float(r['adds_recall_0.1d']):.2f}% | {fmt('mean_translation_mm')} | {fmt('mean_rotation_deg')} |\n"
    report+=f"\n{sum(r['status']=='missing' for r in joined)} missing and {sum(r['status']=='invalid' for r in joined)} invalid poses. {sum(int(r['failure']) for r in joined)} total ADD(-S) failures. {symmetric_flags} valid poses had raw rotation >90° while passing MSSD <0.1 diameter; these need symmetry-aware interpretation rather than an automatic rotation-failure label.\n\n"
    report+='Pooled descriptive strata:\n\n| Dimension | Bin | Poses | ADD(-S) recall |\n| --- | --- | --- | --- |\n'
    for r in strata:
        if r['obj_id']==0:report+=f"| {r['dimension']} | {r['bin']} | {r['n']} | {100*r['adds_recall']:.2f}% |\n"
    report+='\nThese are associations, not demonstrated causes. Pooled bins can differ in object composition; failure_strata.csv also supplies within-object bins. See [stratified plot](failure_strata.png) and [selected success/failure examples](qualitative_examples.csv). The examples are selected extremes, not a random sample. No depth-noise or mask-quality failure mechanism is established by pose errors alone.\n\n'
    report+='''The recommended next controlled experiment is **segmentation-mask robustness**. This baseline uses ideal GT visible masks, so performance under imperfect segmentation remains unmeasured. Hypothesis: modest boundary errors or localization shifts reduce pose recall, especially for small visible object regions; a carefully bounded mask treatment may improve robustness. The baseline's hardest objects and visibility strata identify where to inspect this hypothesis but do not prove it.

Use the deterministic proposed_mask_study_split.csv: 20 development and 20 held-out frames per object, disjoint within each object. The full baseline has already been examined, so these are development-study splits, not a fresh benchmark test set. Preserve RGB, depth, CAD model, camera calibration, inference settings and seed schedule. Compare the unmodified GT-mask control against erosion/dilation radii 2 and 5 pixels and mask shifts of 2 and 5 pixels in each cardinal direction, clipping at image boundaries. These magnitudes are specified before running; do not select the best severity after inspecting held-out results. Never overwrite dataset masks: generate variants in a separate experiment directory. Use seeds 0, 1 and 2 paired across conditions. Empty perturbed masks count as failures and are logged.

Measure paired per-frame changes in ADD(-S), MSSD and MSPD recall, translation/rotation distributions, mask IoU versus GT, and warmed inference latency. Report per-object and object-macro differences with uncertainty from a paired bootstrap stratified by object. Compare within the same frames and seeds to limit pose difficulty and stochastic confounding. Tune any mitigation on development frames only, then report all preregistered conditions on held-out frames. Include segmentation-only measurements as well as pose outcomes. Introduce depth-noise and refinement ablations separately after this controlled test, so changes are attributable to one factor.

No follow-up GPU inference has been started. Running this study or changing core algorithms requires the researcher's approval. This report contains diagnostic scores, not official BOP AR/VSD or a benchmark submission.
'''
    (a.OUT/'FAILURE_ANALYSIS.md').write_text(report)
    p=a.OUT/'REPORT.md';content=p.read_text().split(MARKER)[0];p.write_text(content+MARKER+report)
    provenance[str(Path(__file__))]=a.digest(Path(__file__));(a.OUT/'failure_manifest.json').write_text(json.dumps(provenance,indent=2))
    write_status('complete',baseline_complete=complete,rows=len(rows))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    if args.wait:
        while True:
            status=json.loads((a.OUT/'status.json').read_text())
            if status['state'] in ['analysis_complete','analysis_complete_with_prediction_issues'] and (a.OUT/'MEASURED_RESULTS.md').is_file():break
            if status['state']=='analysis_failed':raise RuntimeError('Upstream analysis failed: '+str(status))
            write_status('waiting_for_validated_analysis');time.sleep(45)
    run()
if __name__=='__main__':
    try:main()
    except Exception as error:write_status('failed',error=str(error));raise
