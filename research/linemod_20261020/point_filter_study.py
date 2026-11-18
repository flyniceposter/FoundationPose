"""Queued point-filter ablation; isolated from mask_study_v1.

Uses development-only selection, then frozen held-out evaluation. Filters only
the registration mask; full-image RGB/depth retain the pretrained input format.
"""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import mask_study as m

METHODS = ['raw', 'largest_component', 'depth_half_diameter', 'depth_diameter']
CONDITIONS = ['control', 'dilate5', 'shiftx-5', 'shiftx+5', 'shifty-5', 'shifty+5']


def filter_mask(mask, method, depth=None, diameter_m=None):
    if method == 'raw':
        return mask.copy()
    if method == 'largest_component':
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        return labels == (1+np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else mask.copy()
    if method not in ['depth_half_diameter', 'depth_diameter']:
        raise ValueError(method)
    valid = mask & np.isfinite(depth) & (depth >= .001) & (depth < 100)
    if not valid.any():
        return np.zeros_like(mask)
    factor = .5 if method == 'depth_half_diameter' else 1.
    return valid & (np.abs(depth-np.median(depth[valid])) <= factor*diameter_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--max-new', type=int, default=0)
    ap.add_argument('--wait-for-mask-study', action='store_true')
    ap.add_argument('--output', type=Path, default=m.ROOT/'debug/point_filter_study_v1')
    args = ap.parse_args()
    m.METHODS = METHODS
    m.CONDITIONS = CONDITIONS
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        records = m.prepare(out)
        config = dict(wrapper_sha256=m.digest(__file__), methods=METHODS, conditions=CONDITIONS,
            maximum_registrations=28080, raw_registrations=9360,
            selection='Highest development macro ADD(-S); ties prefer raw, largest component, half diameter, full diameter.',
            scope='Filters object initialization mask; keeps full scene depth unchanged. CAD diameter is inference-available; no GT pose used in filter.',
            disclosure='Development and held-out input diagnostics already inspected; held-out pose outcomes must not be used for selection.')
        path = out/'filter_manifest.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise RuntimeError('Filter source/configuration mismatch')
        m.atomic_json(path, config)
        # Replace generic planner counts, which were fixed to the original morphology study.
        m.atomic_json(out/'plan.json', dict(frames=520, conditions=len(CONDITIONS), seeds=3,
            raw_registrations=9360, maximum_total_registrations=28080, development_registrations=18720,
            heldout_registrations=9360, prepared_masks=520*len(CONDITIONS)))
        if not args.run:
            print((out/'plan.json').read_text()); return
        while args.wait_for_mask_study:
            state = json.loads((m.ROOT/'debug/mask_study_v1/status.json').read_text())['state']
            if state == 'complete':
                break
            if state == 'failed':
                raise RuntimeError('Prior mask study failed; investigate before allocating more GPU work')
            m.atomic_json(out/'status.json', dict(state='queued', waiting_for='mask_study_v1', updated_unix=time.time()))
            time.sleep(45)
        sys.path.insert(0, str(m.ROOT))
        import run_linemod as upstream
        original_depth = upstream.LinemodReader.get_depth
        metadata = json.loads((m.DATA/'models/models_info.json').read_text())
        current = {}

        def get_depth(reader, index, *a, **kw):
            depth = original_depth(reader, index, *a, **kw)
            current['depth'] = depth
            current['diameter_m'] = metadata[str(int(Path(reader.base_dir).name))]['diameter']/1000
            return depth

        upstream.LinemodReader.get_depth = get_depth
        m.repair = lambda mask, method: filter_mask(mask, method, current.get('depth'), current.get('diameter_m'))
        original_report = m.report

        def report(out, records):
            original_report(out, records)
            p = out/'REPORT.md'
            text = p.read_text().replace('# Mask robustness study', '# Point filtering study')
            text = text.replace('Candidates were raw, opening radius 2, and closing radius 2.',
                'Candidates were raw, largest mask component, and median-depth gates of half/full CAD diameter. Filtering changes only the initialization mask; full-scene depth remains unchanged.')
            p.write_text(text)

        def select(out, records):
            scores = {method:m.macro(m.load_group(out, records, 'development', method)) for method in METHODS}
            best = max(METHODS, key=lambda method:scores[method])
            choice = dict(method=best, development_macro_adds=scores, policy=config['selection'])
            path = out/'frozen_selection.json'
            if path.exists():
                assert json.loads(path.read_text()) == choice
            else:
                m.atomic_json(path, choice)
            return best

        m.select = select
        m.report = report
        try:
            with (m.ROOT/'research/.foundationpose_inference.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                m.execute(out, records, args.max_new)
        except Exception as exc:
            m.atomic_json(out/'status.json', dict(state='failed', error=repr(exc), updated_unix=time.time()))
            raise
        finally:
            upstream.LinemodReader.get_depth = original_depth


if __name__ == '__main__':
    main()
