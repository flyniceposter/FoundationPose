"""Independent completeness check and iteration-gain summary for the finished run."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT / 'manifest.json').read_text())
records = json.loads((OUT / 'registration.json').read_text())
variants = [('original', 0)] + [(arm, seed) for seed in manifest['seeds'] for arm in manifest['arms']]
expected = {(arm, seed, f['object'], f['frame'], condition, i)
            for arm, seed in variants for f in manifest['frames']
            for condition in ['observed', 'dropout40'] for i in [1, 2, 3, 4, 5, 10]}
lookup = {(r['arm'], r['seed'], r['object'], r['frame'], r['condition'], r['iterations']): r for r in records}
assert len(records) == len(lookup) == len(expected) == 3360
assert set(lookup) == expected
assert all(np.isfinite(r[k]) for r in records for k in ['add_mm', 'adds_mm', 'add_diameter', 'adds_diameter'])
gains = []
for arm in ['original'] + manifest['arms']:
    for condition in ['observed', 'dropout40']:
        for start, stop in [(1, 5), (2, 5), (5, 10)]:
            first = [r for r in records if r['arm'] == arm and r['condition'] == condition and r['iterations'] == start]
            later = [lookup[(r['arm'], r['seed'], r['object'], r['frame'], condition, stop)] for r in first]
            gains.append(dict(arm=arm, condition=condition, start=start, stop=stop,
                              mean_adds_reduction_mm=float(np.mean([a['adds_mm']-b['adds_mm'] for a, b in zip(first, later)])),
                              mean_adds_diameter_reduction=float(np.mean([a['adds_diameter']-b['adds_diameter'] for a, b in zip(first, later)])),
                              improved_cases=sum(b['adds_mm'] < a['adds_mm'] for a, b in zip(first, later)),
                              total_cases=len(first)))
development = []
for seed in manifest['seeds']:
    for arm in manifest['arms']:
        history = json.loads((OUT / f'{arm}_{seed}_history.json').read_text())
        chosen = min(history, key=lambda r: r['development'])
        development.append(dict(arm=arm, seed=seed, **chosen))
result = dict(completeness_passed=True, records=len(records), iteration_gains=gains, selected_development=development)
(OUT / 'verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
