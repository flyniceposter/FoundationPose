"""Partial, rounded-log diagnostics; not a completed baseline."""
import csv,json
import analyze as a
import extended as e
import finalize as f

def main(finish_existing=False):
    a.OUT=a.OUT/'recovery';a.PRED=a.OUT/'recovered_predictions.yml'
    base_report=a.OUT/'REPORT.md'
    if not base_report.exists():base_report.write_text('# Partial rounded-log recovery\n\nThis interrupted prefix is not a complete baseline. Missing-pose recalls in full expected-set CSVs are coverage-dependent lower bounds. Observed-prefix accuracy is reported separately in PARTIAL_RESULTS.md.\n')
    if finish_existing:
        manifest=json.loads((a.OUT/'manifest.json').read_text())
        if manifest['sha256'][str(a.PRED)]!=a.digest(a.PRED):raise RuntimeError('Recovered predictions changed since analysis')
    else:a.analyze();e.run()
    f.run()
    with open(a.OUT/'per_pose.csv') as stream:rows=list(csv.DictReader(stream))
    summaries=[]
    for obj in a.IDS+[0]:
        group=[r for r in rows if (obj==0 or int(r['obj_id'])==obj) and r['status']=='valid']
        if not group:continue
        summaries.append(dict(obj_id=obj,recovered_valid=len(group),mean_translation_mm=sum(float(r['translation_mm']) for r in group)/len(group),mean_rotation_deg=sum(float(r['rotation_deg']) for r in group)/len(group),observed_adds_recall=sum(int(r['adds_pass']) for r in group)/len(group)))
    a.csvwrite('observed_subset_summary.csv',summaries)
    report='''# Partial recovery diagnostics — rounded log poses

The original evaluation did not complete. These matrices were recovered from rounded text in its preserved log, not from an original full-precision prediction file. The observed prefix is biased by object/frame order. Its pooled accuracy is not a full LINEMOD baseline or an official BOP score. Objects 1, 2, 4 and 5 have complete frame coverage; object 6 has only 339 frames; later objects have no recovered predictions.

| Object (0 = observed pooled) | Recovered valid | Mean translation (mm) | Mean raw rotation (degrees) | Observed ADD(-S) recall |
| --- | --- | --- | --- | --- |
'''
    for r in summaries:report+=f"| {r['obj_id']} | {r['recovered_valid']} | {r['mean_translation_mm']:.3f} | {r['mean_rotation_deg']:.3f} | {100*r['observed_adds_recall']:.2f}% |\n"
    report+='\nAll expected-frame CSVs retain missing later poses as failures. Their pooled recall is a coverage-dependent lower bound, not measured full-dataset model performance. observed_subset_summary.csv reports recovered-valid-only accuracy with explicit denominators. Raw rotation is not symmetry corrected. Geometry diagnostics use the existing audited ADD/ADI and MSSD/MSPD implementations. Recall plots spanning all objects must be interpreted alongside missing counts. No GPU work was launched to produce these diagnostics.\n'
    (a.OUT/'PARTIAL_RESULTS.md').write_text(report)
    parent=a.OUT.parent/'REPORT.md'
    with parent.open('a') as stream:stream.write('\n\n## Recovered partial results\n\n'+report.replace('# Partial recovery diagnostics — rounded log poses','See recovery/PARTIAL_RESULTS.md for the standalone table.'))
    (a.OUT/'recovery_analysis_status.json').write_text(json.dumps(dict(state='partial_log_recovery_analyzed',recovered_valid=sum(r['recovered_valid'] for r in summaries if r['obj_id']!=0),full_baseline_complete=False),indent=2))
    print(report,flush=True)
if __name__=='__main__':main()
