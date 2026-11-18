"""Checkpointed, isolated mask study. Default prepares CPU artifacts; --run uses CUDA.

Development chooses among raw, circular opening radius 2, and closing radius 2.
Held-out masks are evaluated only after the choice is frozen. No GT-based repair.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from checkpoint_runner import atomic_json, atomic_pose, valid_pose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path('/home/Sahil/datasets/bop/lm')
SPLIT = ROOT/'debug/bop_recovery_20261021/analysis/proposed_mask_study_split.csv'
CONDITIONS = ['control', 'erode2', 'erode5', 'dilate2', 'dilate5'] + [f'shift{x}{d:+d}' for x in 'xy' for d in (-5, -2, 2, 5)]
METHODS = ['raw', 'open2', 'close2']
SEEDS = [0, 1, 2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def kernel(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return ((x*x+y*y) <= radius*radius).astype(np.uint8)


def perturb(mask, condition):
    m = mask.astype(np.uint8)
    if condition == 'control':
        return m.astype(bool)
    if condition.startswith(('erode', 'dilate')):
        fn = cv2.erode if condition.startswith('erode') else cv2.dilate
        return fn(m, kernel(int(condition[-1])), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    axis, delta = condition[5], int(condition[6:])
    transform = np.float32([[1, 0, delta if axis == 'x' else 0], [0, 1, delta if axis == 'y' else 0]])
    return cv2.warpAffine(m, transform, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)


def repair(mask, method):
    if method == 'raw':
        return mask.copy()
    operation = {'open2': cv2.MORPH_OPEN, 'close2': cv2.MORPH_CLOSE}[method]
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel(2), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)/union) if union else 1.0


def rows():
    with SPLIT.open() as f:
        result = list(csv.DictReader(f))
    assert len(result) == 520
    assert len({(r['obj_id'], r['im_id']) for r in result}) == 520
    for obj in {r['obj_id'] for r in result}:
        assert sum(r['obj_id'] == obj for r in result) == 40
    return result


def mask_path(out, obj, im, condition):
    return out/'masks'/f'{obj:06d}_{im:06d}_{condition}.png'


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    (out/'masks').mkdir(exist_ok=True)
    records = rows()
    labels = sorted({r['split'] for r in records})
    assert len(labels) == 2 and 'development' in labels
    for obj in {r['obj_id'] for r in records}:
        assert all(sum(r['obj_id'] == obj and r['split'] == label for r in records) == 20 for label in labels)
    sources = [SPLIT, Path(__file__)] + [ROOT/n for n in ['estimater.py', 'run_linemod.py', 'datareader.py', 'Utils.py']]
    sources += sorted(p for p in (ROOT/'weights').rglob('*') if p.is_file())
    for obj in sorted({int(r['obj_id']) for r in records}):
        sources += [DATA/f'lm_test_all/test/{obj:06d}'/n for n in ['scene_gt.json', 'scene_camera.json']]
        sources += [DATA/f'{directory}/obj_{obj:06d}.ply' for directory in ['models', 'models_eval']]
    sources += [DATA/'models/models_info.json', DATA/'models_eval/models_info.json']
    for row in records:
        scene = DATA/f"lm_test_all/test/{int(row['obj_id']):06d}"
        im = int(row['im_id'])
        sources += sorted((scene/'rgb').glob(f'{im:06d}.*'))
        sources += sorted((scene/'depth').glob(f'{im:06d}.*'))
        sources += sorted((scene/'mask_visib').glob(f'{im:06d}_*.png'))
    manifest = dict(version=1, conditions=CONDITIONS, seeds=SEEDS, development_methods=METHODS,
        selection='Highest development object-macro ADD(-S) recall; ties prefer raw then open2 then close2. Freeze before held-out inference.',
        seed_adapter='Replace estimater module set_seed in isolated process; original register seed call delegates to paired seed.',
        stages='Development raw/open2/close2; freeze best method; held-out raw and selected method.',
        hashes={str(p): digest(p) for p in sources})
    destination = out/'manifest.json'
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise RuntimeError('Manifest mismatch; use a new output directory')
    atomic_json(destination, manifest)
    metadata = []
    for row in records:
        obj, im = int(row['obj_id']), int(row['im_id'])
        scene = DATA/f'lm_test_all/test/{obj:06d}'
        gt = json.loads((scene/'scene_gt.json').read_text())[str(im)]
        indices = [i for i, g in enumerate(gt) if g['obj_id'] == obj]
        assert len(indices) == 1
        source = scene/f'mask_visib/{im:06d}_{indices[0]:06d}.png'
        m = cv2.imread(str(source), -1)
        if m is None:
            raise RuntimeError(f'Missing mask {source}')
        m = m > 0
        for condition in CONDITIONS:
            variant = perturb(m, condition)
            path = mask_path(out, obj, im, condition)
            if path.exists():
                assert np.array_equal(cv2.imread(str(path), -1) > 0, variant)
            elif not cv2.imwrite(str(path), variant.astype(np.uint8)*255):
                raise RuntimeError(f'Cannot save {path}')
            metadata.append(dict(obj_id=obj, im_id=im, split=row['split'], condition=condition,
                pixels=int(variant.sum()), iou=iou(m, variant), source_sha256=digest(source), variant_sha256=digest(path)))
    write_csv(out/'mask_metadata.csv', metadata)
    atomic_json(out/'plan.json', dict(frames=520, conditions=13, seeds=3, raw_registrations=20280,
        maximum_total_registrations=50700, prepared_masks=len(metadata), heldout_label=next(x for x in labels if x != 'development')))
    return records


def checkpoint(out, row, condition, seed, method):
    return out/'checkpoints'/f"{row['split']}_{int(row['obj_id']):06d}_{int(row['im_id']):06d}_{condition}_{seed}_{method}.json"


def score(pose, reader, index, obj, assets):
    import analyze as a
    import extended as e
    pts, evalpts, diameter, syms = assets
    g = reader.get_gt_pose(index, obj)
    Rg, tg = g[:3, :3], g[:3, 3]*1000
    R, t = pose[:3, :3], pose[:3, 3]*1000
    add, adi = a.vertex_errors(pts@R.T+t, pts@Rg.T+tg)
    mssd, mspd = e.distances(evalpts, R, t, Rg, tg, reader.K, syms)
    return dict(add_mm=add, adi_mm=adi, translation_mm=float(np.linalg.norm(t-tg)),
        rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(R@Rg.T)-1)/2, -1, 1)))),
        adds_pass=int((adi if obj in [10, 11] else add) < .1*diameter),
        mssd_mm=mssd, mspd_px=mspd, mssd_pass=int(mssd < .1*diameter), mspd_pass=int(mspd < 5))


def execute(out, records, max_new=0):
    sys.path.insert(0, str(ROOT))
    import run_linemod as u
    import estimater as estimator_module
    import torch
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run outside device-restricted sandbox')
    torch.cuda.set_device(0)
    u.wp.force_load(device='cuda')
    glctx = u.dr.RasterizeCudaContext()
    mesh = trimesh.primitives.Box(extents=np.ones(3)).to_mesh()
    est = u.FoundationPose(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
        mesh=mesh, symmetry_tfs=None, glctx=glctx, debug_dir=str(out), debug=0)
    original_seed = estimator_module.set_seed
    current_seed = [0]
    estimator_module.set_seed = lambda _ignored: original_seed(current_seed[0])
    (out/'checkpoints').mkdir(exist_ok=True)
    info = json.loads((DATA/'models/models_info.json').read_text())
    evalinfo = json.loads((DATA/'models_eval/models_info.json').read_text())
    new_count = 0
    completed_count = len(list((out/'checkpoints').glob('*.json')))
    try:
        for stage in ['development', 'heldout']:
            if stage == 'heldout':
                selected = select(out, records)
                methods = list(dict.fromkeys(['raw', selected]))
                selected_rows = [r for r in records if r['split'] != 'development']
            else:
                methods = METHODS
                selected_rows = [r for r in records if r['split'] == 'development']
            for obj in sorted({int(r['obj_id']) for r in selected_rows}):
                reader = u.LinemodReader(str(DATA/f'lm_test_all/test/{obj:06d}'), split=None)
                index_by_im = {int(s): i for i, s in enumerate(reader.id_strs)}
                mesh = reader.get_gt_mesh(obj)
                est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
                    symmetry_tfs=reader.symmetry_tfs[obj], mesh=mesh)
                assert not evalinfo[str(obj)].get('symmetries_continuous')
                assets = (np.asarray(trimesh.load(DATA/f'models/obj_{obj:06d}.ply', process=False).vertices),
                    np.asarray(trimesh.load(DATA/f'models_eval/obj_{obj:06d}.ply', process=False).vertices),
                    info[str(obj)]['diameter'], [np.eye(4)]+[np.array(s).reshape(4, 4) for s in evalinfo[str(obj)].get('symmetries_discrete', [])])
                warmed = False
                for row in [r for r in selected_rows if int(r['obj_id']) == obj]:
                    im = int(row['im_id']); index = index_by_im[im]
                    rgb, depth = reader.get_color(index), reader.get_depth(index)
                    gt_mask = cv2.imread(str(mask_path(out, obj, im, 'control')), -1) > 0
                    for condition in CONDITIONS:
                        variant = cv2.imread(str(mask_path(out, obj, im, condition)), -1) > 0
                        for method in methods:
                            t0 = time.perf_counter(); mask = repair(variant, method)
                            preprocess_s = time.perf_counter()-t0
                            for seed in SEEDS:
                                path = checkpoint(out, row, condition, seed, method)
                                if path.exists():
                                    saved = json.loads(path.read_text())
                                    if saved['status'] not in ['valid', 'empty_mask', 'invalid']:
                                        raise RuntimeError(f'Invalid checkpoint {path}')
                                    if saved['status'] == 'valid' and not valid_pose(np.load(path.with_suffix('.npy'), allow_pickle=False)):
                                        raise RuntimeError(f'Invalid pose checkpoint {path}')
                                    continue
                                current_seed[0] = seed
                                result = dict(obj_id=obj, im_id=im, split=row['split'], condition=condition,
                                    method=method, seed=seed, mask_iou=iou(mask, gt_mask), mask_pixels=int(mask.sum()),
                                    preprocessing_s=preprocess_s, warmed=warmed, status='empty_mask',
                                    adds_pass=0, mssd_pass=0, mspd_pass=0, latency_s=None)
                                if mask.any():
                                    torch.cuda.synchronize(); started = time.perf_counter()
                                    pose = est.register(K=reader.K, rgb=rgb, depth=depth, ob_mask=mask, ob_id=obj)
                                    torch.cuda.synchronize(); result['latency_s'] = time.perf_counter()-started
                                    warmed = True
                                    if valid_pose(pose) and np.allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-3) and abs(np.linalg.det(pose[:3,:3])-1) < 1e-3 and pose[2,3] > 0:
                                        result.update(status='valid', **score(pose, reader, index, obj, assets))
                                        atomic_pose(path.with_suffix('.npy'), pose)
                                    else:
                                        result['status'] = 'invalid'
                                atomic_json(path, result)
                                new_count += 1
                                completed_count += 1
                                atomic_json(out/'status.json', dict(state='running', stage=stage,
                                    checkpoints=completed_count, last=str(path.name), updated_unix=time.time()))
                                if max_new and new_count >= max_new:
                                    atomic_json(out/'smoke_test.json', dict(state='passed', checkpoints_created=new_count))
                                    return
        report(out, records)
    finally:
        estimator_module.set_seed = original_seed


def load_group(out, records, split, method):
    result = []
    for row in records:
        if (row['split'] == 'development') != (split == 'development'):
            continue
        for condition in CONDITIONS:
            for seed in SEEDS:
                path = checkpoint(out, row, condition, seed, method)
                if not path.exists():
                    raise RuntimeError(f'Incomplete stage: {path}')
                result.append(json.loads(path.read_text()))
    return result


def macro(records, metric='adds_pass'):
    return float(np.mean([np.mean([r[metric] for r in records if r['obj_id'] == obj]) for obj in sorted({r['obj_id'] for r in records})]))


def select(out, records):
    scores = {m: macro(load_group(out, records, 'development', m)) for m in METHODS}
    best = max(METHODS, key=lambda m: scores[m])
    choice = dict(method=best, development_macro_adds=scores,
        policy='Highest development object macro ADD(-S), ties raw/open2/close2; no held-out data used.')
    path = out/'frozen_selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == choice
    else:
        atomic_json(path, choice)
    return best


def paired_interval(raw, treated, metric):
    """Stratified paired frame bootstrap, preserving all seed/condition replicates."""
    key = lambda r: (r['obj_id'], r['im_id'], r['condition'], r['seed'])
    reference = {key(r): r[metric] for r in raw}
    frames = {}
    for r in treated:
        frames.setdefault((r['obj_id'], r['im_id']), []).append(r[metric]-reference[key(r)])
    objects = sorted({k[0] for k in frames})
    groups = [np.array([np.mean(v) for k, v in frames.items() if k[0] == obj]) for obj in objects]
    rng = np.random.default_rng(20261022)
    boot = np.mean([np.mean(g[rng.integers(0, len(g), size=(2000, len(g)))], axis=1) for g in groups], axis=0)
    return float(np.mean([g.mean() for g in groups])), np.percentile(boot, [2.5, 97.5]).tolist()


def report(out, records):
    best = select(out, records)
    all_rows, summaries = [], []
    for split in ['development', 'heldout']:
        for method in (METHODS if split == 'development' else list(dict.fromkeys(['raw', best]))):
            group = load_group(out, records, split, method); all_rows.extend(group)
            for condition in CONDITIONS:
                subset = [r for r in group if r['condition'] == condition]
                latencies = [r['latency_s'] for r in subset if r['warmed'] and r['latency_s'] is not None]
                summaries.append(dict(split=split, method=method, condition=condition, expected=len(subset),
                    valid=sum(r['status'] == 'valid' for r in subset), adds_recall=macro(subset),
                    mssd_recall=macro(subset, 'mssd_pass'), mspd_recall=macro(subset, 'mspd_pass'),
                    mean_iou=float(np.mean([r['mask_iou'] for r in subset])),
                    warmed_median_latency_s=float(np.median(latencies)) if latencies else None))
    # All records have matching columns even when a mask is empty or a pose invalid.
    fields = sorted(set().union(*(r.keys() for r in all_rows)))
    write_csv(out/'per_pose.csv', [{k:r.get(k) for k in fields} for r in all_rows])
    write_csv(out/'summary.csv', summaries)
    raw = load_group(out, records, 'heldout', 'raw')
    treated = load_group(out, records, 'heldout', best)
    intervals = {metric: paired_interval(raw, treated, metric) for metric in ['adds_pass', 'mssd_pass', 'mspd_pass']}
    atomic_json(out/'paired_intervals.json', intervals)
    perturb_intervals = {}
    for condition in CONDITIONS:
        reference = [dict(r, condition=condition) for r in raw if r['condition']=='control']
        perturbed = [r for r in raw if r['condition']==condition]
        perturb_intervals[condition] = {metric: paired_interval(reference, perturbed, metric) for metric in ['adds_pass', 'mssd_pass', 'mspd_pass']}
    atomic_json(out/'perturbation_intervals.json', perturb_intervals)
    by_object = []
    for split in ['development', 'heldout']:
        for method in (METHODS if split=='development' else list(dict.fromkeys(['raw', best]))):
            group = [r for r in all_rows if r['method']==method and (r['split']=='development')==(split=='development')]
            for obj in sorted({r['obj_id'] for r in group}):
                for condition in CONDITIONS:
                    subset = [r for r in group if r['obj_id']==obj and r['condition']==condition]
                    summary = dict(split=split, method=method, obj_id=obj, condition=condition,
                        adds_recall=macro(subset), mssd_recall=macro(subset, 'mssd_pass'), mspd_recall=macro(subset, 'mspd_pass'))
                    for metric in ['translation_mm', 'rotation_deg', 'add_mm', 'adi_mm']:
                        values = [r[metric] for r in subset if r['status']=='valid']
                        for label, fn in [('mean', np.mean), ('median', np.median), ('p90', lambda v: np.percentile(v, 90))]:
                            summary[f'{label}_{metric}'] = float(fn(values)) if values else None
                    by_object.append(summary)
    write_csv(out/'per_object_summary.csv', by_object)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for method in list(dict.fromkeys(['raw', best])):
        values = [next(r['adds_recall'] for r in summaries if r['split']=='heldout' and r['method']==method and r['condition']==c) for c in CONDITIONS]
        ax.plot(CONDITIONS, np.array(values)*100, marker='o', label=method)
    ax.set(ylabel='Object-macro ADD(-S) recall (%)', title='Held-out mask robustness')
    ax.tick_params(axis='x', rotation=45); ax.legend(); fig.tight_layout()
    fig.savefig(out/'heldout_recall.png', dpi=160); plt.close(fig)
    text = '# Mask robustness study\n\nCompleted diagnostic experiment with GT-mask perturbations. These study splits were selected after inspecting the full baseline; this is not an official BOP benchmark.\n\n'
    text += f'Development selected **{best}** before held-out inference. Candidates were raw, opening radius 2, and closing radius 2. Identity may win; improvement is not assumed.\n\n'
    text += '| Held-out condition | Raw ADD(-S) recall | Selected ADD(-S) recall |\n| --- | --- | --- |\n'
    for c in CONDITIONS:
        r = next(r for r in summaries if r['split']=='heldout' and r['method']=='raw' and r['condition']==c)
        t = next(r for r in summaries if r['split']=='heldout' and r['method']==best and r['condition']==c)
        text += f"| {c} | {r['adds_recall']:.2%} | {t['adds_recall']:.2%} |\n"
    text += '\nPaired object-stratified frame bootstrap, 2,000 samples; seeds and conditions stay paired:\n\n'
    for metric, (delta, bounds) in intervals.items():
        text += f'- {metric}: {delta*100:.3f} percentage points; 95% interval [{bounds[0]*100:.3f}, {bounds[1]*100:.3f}].\n'
    text += '\nSee per_pose.csv for translation/rotation and ADD/ADI/MSSD/MSPD errors, summary.csv for recall, mask IoU and warmed latency, manifest.json for source hashes. Empty and invalid poses remain in recall denominators. Latency excludes CPU metric evaluation; preprocessing time is recorded separately.\n'
    (out/'REPORT.md').write_text(text)
    atomic_json(out/'status.json', dict(state='complete', records=len(all_rows), selected_method=best, finished_unix=time.time()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT/'debug/mask_study_v1')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--max-new', type=int, default=0, help='Stop after N checkpoints for smoke validation')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        records = prepare(args.output)
        if args.run:
            with (ROOT/'research/.foundationpose_inference.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    execute(args.output, records, args.max_new)
                except Exception as exc:
                    atomic_json(args.output/'status.json', dict(state='failed', error=repr(exc), updated_unix=time.time()))
                    raise
        print((args.output/'plan.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
