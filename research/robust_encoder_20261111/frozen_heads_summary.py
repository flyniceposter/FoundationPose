"""Paired four-arm comparison on the existing fixed registration cases."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'frozen_heads_followup'
ARMS=['original','combined','heads_only','frozen_combined_heads']
def main():
    records=json.loads((HERE/'pose/records.json').read_text())
    baseline={(r['object'],r['frame'],r['condition']):r for r in records if r['arm']=='original'}
    lines=['# Frozen adapted encoder + trained fusion/pose heads','',
      'Exploratory fixed evaluation: five adaptation objects, four frames per object, four conditions. GT masks; no untouched-object or official benchmark claim. Training seeds are repeats of the same frames, not independent images.','',
      '| Arm | Condition | Seed | Cases | ADD mm | ADD-S mm | Paired ADD change vs original mm |',
      '|---|---|---:|---:|---:|---:|---:|']
    summary=[]
    for arm in ARMS:
      for seed in ([0] if arm=='original' else [42,123,2026]):
        rows=[r for r in records if r['arm']==arm and r['seed']==seed]
        assert {(r['object'],r['frame'],r['condition']) for r in rows}==set(baseline),(arm,seed,'unmatched cases')
        for condition in ['observed','noise5mm','dropout40','grayscale']:
          cases=[r for r in rows if r['condition']==condition]
          def macro(key):
            return float(np.mean([np.mean([r[key] for r in cases if r['object']==obj]) for obj in sorted({r['object'] for r in cases})]))
          for r in cases:r['delta']=r['add_mm']-baseline[(r['object'],r['frame'],r['condition'])]['add_mm']
          row=dict(arm=arm,seed=seed,condition=condition,cases=len(cases),add_mm=macro('add_mm'),adds_mm=macro('adds_mm'),paired_add_change_mm=macro('delta'))
          summary.append(row)
          lines.append(f"| {arm} | {condition} | {seed} | {len(cases)} | {row['add_mm']:.3f} | {row['adds_mm']:.3f} | {row['paired_add_change_mm']:+.3f} |")
    lines+=['','Negative paired ADD change means improvement. Per-object results below expose inconsistent gains.','', '| Object | Condition | Seed | Combined encoder ADD mm | Frozen combined + trained downstream ADD mm | Change mm |','|---:|---|---:|---:|---:|---:|']
    for seed in [42,123,2026]:
      for obj in sorted({r['object'] for r in baseline.values()}):
        for condition in ['observed','noise5mm','dropout40','grayscale']:
          def avg(arm):return float(np.mean([r['add_mm'] for r in records if r['arm']==arm and r['seed']==seed and r['object']==obj and r['condition']==condition]))
          a,b=avg('combined'),avg('frozen_combined_heads')
          lines.append(f'| {obj} | {condition} | {seed} | {a:.3f} | {b:.3f} | {b-a:+.3f} |')
    OUT.mkdir(exist_ok=True)
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
