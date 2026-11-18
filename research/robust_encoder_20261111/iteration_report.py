"""Summarize a completed iteration diagnostic without changing its records."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
records = json.loads((args.directory / 'records.json').read_text())
summary = json.loads((args.directory / 'summary.json').read_text())
frames = sorted({(r['object'], r['frame']) for r in records})
steps = len(summary)
assert len(records) == len(frames) * steps
lookup = {(r['object'], r['frame'], r['iterations']): r for r in records}
assert len(lookup) == len(records)
best = {}
for metric in ['add_mm', 'adds_mm']:
    mean_key = 'mean_' + metric
    best[metric] = min(summary, key=lambda r: r[mean_key])
    winners = [min(range(1, steps+1), key=lambda i: lookup[(*frame, i)][metric]) for frame in frames]
    best[metric + '_frame_winners'] = dict(sorted(Counter(winners).items()))
    best[metric + '_100_vs_2'] = dict(
        improved=sum(lookup[(*f, steps)][metric] < lookup[(*f, 2)][metric] for f in frames),
        worsened=sum(lookup[(*f, steps)][metric] > lookup[(*f, 2)][metric] for f in frames))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, metric, label in zip(axes, ['add_mm', 'adds_mm'], ['ADD (mm)', 'ADD-S (mm)']):
    for obj in sorted({f[0] for f in frames}):
        values = [np.mean([r[metric] for r in records if r['object']==obj and r['iterations']==i]) for i in range(1, steps+1)]
        ax.plot(range(1, steps+1), values, alpha=.5, linewidth=1, label=f'Object {obj}')
    ax.plot([r['iterations'] for r in summary], [r['mean_'+metric] for r in summary], color='black', linewidth=2, label='All frames')
    ax.set(xlabel='Refinement iteration', ylabel=label, xscale='log')
    ax.grid(alpha=.2)
axes[1].legend(fontsize=8)
fig.suptitle(f'Original FoundationPose · {len(frames)} evaluation frames · observed inputs · lower is better')
fig.tight_layout()
fig.savefig(args.directory / 'iteration_errors.png', dpi=170)
(args.directory / 'analysis.json').write_text(json.dumps(best, indent=2))
print(json.dumps(best, indent=2))
for r in summary:
    if r['iterations'] in [1, 2, 3, 5, 10, 20, 50, 100]:
        print(r)
