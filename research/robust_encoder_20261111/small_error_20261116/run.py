"""Matched static small/exact-pose mixture ablation; preserves existing experiments."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import copy
import hashlib
import importlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
HERE = OUT.parent
sys.path.insert(0, str(HERE))
import train_pose_heads as h
from pose_eval import FoundationPose, LinemodReader, dr, t
from Utils import depth2xyzmap, make_mesh_tensors

SEEDS = [42, 123, 2026]
ARMS = ['standard', 'mixture']


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def status(**kwargs):
    write(OUT / 'status.json', dict(updated_unix=time.time(), **kwargs))


def wait_gpu():
    ready = 0
    while ready < 3:
        line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()[0]
        free, util = map(int, line.split(','))
        ready = ready + 1 if free >= 20480 and util < 40 else 0
        status(state='waiting_for_gpu', ready=ready, free_mib=free, utilization=util)
        print('GPU', ready, free, util, flush=True)
        if ready < 3:
            time.sleep(30)


def lock_manifest():
    path = OUT / 'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    used = set()
    def walk(value):
        if isinstance(value, dict):
            if 'object' in value and 'frame' in value:
                used.add((int(value['object']), int(value['frame'])))
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
    files = [HERE / 'manifest.json', HERE / 'pose_pair_manifest.json', HERE / 'pose/records.json']
    for path in files:
        walk(json.loads(path.read_text()))
    cache_path = HERE.parent / 'real_rotation_20261111/cache_vertexcolor_v2_240_96.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    for obj, data in cache['objects'].items():
        for samples in data['splits'].values():
            for s in samples:
                used.add((int(obj), int(s['frame'])))
        for s in data['references']:
            used.add((int(obj), int(s['frame'])))
    frames = []
    for obj in t.r.OBJECTS:
        keys = sorted(json.loads((t.DATA / 'lm_test_all/test' / f'{obj:06d}/scene_gt.json').read_text()), key=int)
        available = [k for k in keys if t.partition(obj, k) == 'evaluation' and (obj, int(k)) not in used]
        assert len(available) >= 8
        for i in np.linspace(0, len(available)-1, 8, dtype=int):
            frames.append(dict(object=obj, frame=f'{int(available[i]):06d}'))
    assert len({(r['object'], r['frame']) for r in frames}) == 40
    assert not {(r['object'], int(r['frame'])) for r in frames} & used
    manifest = dict(frames=frames, seeds=SEEDS, arms=ARMS, excluded_local_frames=len(used),
                    protocol_sha256=hashlib.sha256((OUT / 'PROTOCOL.md').read_bytes()).hexdigest(),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    write(OUT / 'manifest.json', manifest)
    return manifest


@torch.inference_mode()
def prepare(predictor):
    path = OUT / 'pairs.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    base = torch.load(HERE / 'pose_pairs.pt', map_location='cpu', weights_only=False)
    result = {split: dict(standard=samples, small=[], exact=[]) for split, samples in base['splits'].items()}
    context = dr.RasterizeCudaContext()
    for split, samples in base['splits'].items():
        groups = {}
        for s in samples:
            groups.setdefault((s['object'], s['frame']), []).append(s)
        extras = {}
        for (obj, frame), group in groups.items():
            reader = LinemodReader(str(t.DATA / 'lm_test_all/test' / f'{obj:06d}'), split=None)
            reader.K = reader.K.astype(np.float32)
            mesh = reader.get_gt_mesh(obj)
            index = reader.id_strs.index(frame)
            rgb, depth, gt = reader.get_color(index), reader.get_depth(index), reader.get_gt_pose(index, obj)
            diameter = reader.get_model_diameter(obj)
            rng = np.random.default_rng(16000000 + obj*100000 + int(frame))
            initial = np.repeat(gt[None], len(group)+1, axis=0)
            for pose in initial[:-1]:
                axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
                pose[:3, :3] = Rotation.from_rotvec(axis*np.deg2rad(rng.uniform(.1, 3))).as_matrix() @ gt[:3, :3]
                pose[:3, 3] += rng.uniform(-.005, .005, 3)*diameter
            crops = h.make_crop_data_batch(predictor.cfg.input_resize, torch.as_tensor(initial, device='cuda', dtype=torch.float32), mesh, rgb, depth, reader.K, crop_ratio=predictor.cfg.crop_ratio, xyz_map=depth2xyzmap(depth, reader.K), cfg=predictor.cfg, glctx=context, mesh_tensors=make_mesh_tensors(mesh), dataset=predictor.dataset, mesh_diameter=diameter)
            A = torch.cat([crops.rgbAs, crops.xyz_mapAs], 1).cpu().half()
            B = torch.cat([crops.rgbBs, crops.xyz_mapBs], 1).cpu().half()
            for j, sample in enumerate(group):
                small = dict(sample, A=A[j], B=B[j], initial=torch.tensor(initial[j], dtype=torch.float32, device='cpu'))
                exact = dict(sample, A=A[-1], B=B[-1], initial=torch.tensor(gt, dtype=torch.float32, device='cpu'))
                assert torch.allclose(exact['initial'], exact['targets'][0], atol=1e-6)
                extras[id(sample)] = (small, exact)
            print('CACHE', split, obj, frame, flush=True)
        result[split]['small'] = [extras[id(s)][0] for s in samples]
        result[split]['exact'] = [extras[id(s)][1] for s in samples]
    # Ensure all categories have exactly matching frame/perturbation indices.
    for split, cats in result.items():
        for category in ['small', 'exact']:
            assert [(s['object'], s['frame']) for s in cats[category]] == [(s['object'], s['frame']) for s in cats['standard']]
    t.save(path, result)
    return result


def development(model, data, cfg):
    scores = {category: h.evaluate(model, samples, cfg, 'cuda')[0] for category, samples in data.items()}
    return .5*scores['standard'] + .25*scores['small'] + .25*scores['exact'], scores


def train(predictor, data):
    original = {k: v.cpu().clone() for k, v in predictor.model.state_dict().items()}
    for seed in SEEDS:
        for arm in ARMS:
            done = OUT / f'{arm}_{seed}_complete.json'
            if done.exists():
                continue
            model = predictor.model
            model.load_state_dict(original); model.eval()
            torch.manual_seed(seed)
            for name, param in model.named_parameters():
                param.requires_grad_(not name.startswith('encodeA.'))
            model.zero_grad(set_to_none=True)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-4)
            best = float('inf'); start = 0; history = []
            last = OUT / f'{arm}_{seed}_last.pth'
            bestpath = OUT / f'{arm}_{seed}.pth'
            if last.exists():
                saved = torch.load(last, map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
                start, best, history = saved['step'], saved['best'], saved['history']
            for step in range(start, 1500):
                # Frame indices, augmentation and budget are exactly matched between arms.
                rng = np.random.default_rng(seed*100000+step)
                ids = rng.choice(len(data['train']['standard']), 4, replace=False)
                categories = ['standard']*4 if arm == 'standard' else ['standard', 'standard', 'small', 'exact']
                samples = [data['train'][category][i] for category, i in zip(categories, ids)]
                A, B, initial, targets, vertices, diameter = h.batch(samples, 'cuda')
                if step % 2:
                    B = h.corrupt(B, seed*100000+step)
                opt.zero_grad(set_to_none=True)
                loss = h.pose_loss(h.predicted_pose(model(A, B), initial, diameter, predictor.cfg), targets, vertices, diameter).mean()
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
                if (step+1) % 100 == 0:
                    score, categories_score = development(model, data['development'], predictor.cfg)
                    history.append(dict(step=step+1, development=score, categories=categories_score, training_loss=float(loss.detach())))
                    if score < best:
                        best = score
                        t.save(bestpath, dict(model=model.state_dict(), seed=seed, arm=arm, step=step+1, development=score))
                    t.save(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step+1, best=best, history=history))
                    write(OUT / f'{arm}_{seed}_history.json', history)
                    status(state='training', seed=seed, arm=arm, step=step+1, development=score)
                    print('TRAIN', arm, seed, step+1, score, flush=True)
            chosen = torch.load(bestpath, map_location='cpu', weights_only=False)['model']
            assert all(torch.equal(value, original[name]) for name, value in chosen.items() if name.startswith('encodeA.') or 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name)
            assert all(any(not torch.equal(value, original[name]) for name, value in chosen.items() if name.startswith(prefix)) for prefix in ['trans_head', 'rot_head'])
            write(done, dict(best_development=best, encoder_and_bn_unchanged=True, history=history))


def registration(predictor, manifest):
    path = OUT / 'registration.json'
    records = json.loads(path.read_text()) if path.exists() else []
    completed = {(r['arm'], r['seed'], r['object'], r['frame'], r['condition'], r['iterations']) for r in records}
    context = dr.RasterizeCudaContext()
    estimator = None
    initial_by_case = {}
    pretrained = None
    variants = [('original', 0)] + [(arm, seed) for seed in SEEDS for arm in ARMS]
    for sample in manifest['frames']:
        obj, frame = sample['object'], sample['frame']
        reader = LinemodReader(str(t.DATA / 'lm_test_all/test' / f'{obj:06d}'), split=None)
        reader.K = reader.K.astype(np.float32)
        mesh = reader.get_gt_mesh(obj)
        if estimator is None:
            estimator = FoundationPose(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=reader.symmetry_tfs[obj], mesh=mesh, glctx=context, debug=0, debug_dir=str(OUT / 'debug'))
            pretrained = {k: v.cpu().clone() for k, v in estimator.refiner.model.state_dict().items()}
        else:
            estimator.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=reader.symmetry_tfs[obj], mesh=mesh)
            estimator.make_rotation_grid(min_n_views=40, inplane_step=60)
        estimator.diameter = float(estimator.diameter)
        diameter = reader.get_model_diameter(obj)
        index = reader.id_strs.index(frame)
        rgb, depth, mask = reader.get_color(index), reader.get_depth(index), reader.get_mask(index, obj)
        gt = reader.get_gt_pose(index, obj)
        vertices = np.asarray(mesh.vertices)
        vertices = vertices[np.linspace(0, len(vertices)-1, min(5000, len(vertices)), dtype=int)]
        target = vertices @ gt[:3, :3].T + gt[:3, 3]
        tree = cKDTree(target)
        for arm, seed in variants:
            estimator.refiner.model.load_state_dict(pretrained if arm == 'original' else torch.load(OUT / f'{arm}_{seed}.pth', map_location='cpu', weights_only=False)['model'])
            estimator.refiner.model.eval()
            for condition in ['observed', 'dropout40']:
                prefix = (arm, seed, obj, frame, condition)
                if all((*prefix, i) in completed for i in [1, 2, 3, 4, 5, 10]):
                    continue
                z = depth.copy()
                if condition == 'dropout40':
                    from PIL import Image
                    rng = np.random.default_rng(obj*100000+int(frame))
                    blocks = rng.random((8, 8)) < .4
                    holes = np.asarray(Image.fromarray(blocks).resize((z.shape[1], z.shape[0]), Image.Resampling.NEAREST))
                    z[holes] = 0
                module = importlib.import_module(estimator.refiner.__class__.__module__)
                original_update = module.egocentric_delta_pose_to_pose
                original_score = estimator.scorer.predict
                original_generate = estimator.generate_random_pose_hypo
                history = []
                def generate(*a, **kw):
                    poses = original_generate(*a, **kw)
                    initial = poses.detach().cpu().numpy()
                    case = (obj, frame, condition)
                    if case in initial_by_case:
                        assert np.array_equal(initial_by_case[case], initial)
                    else:
                        initial_by_case[case] = initial.copy()
                    return poses
                def update(*a, **kw):
                    poses = original_update(*a, **kw)
                    history.append(poses.detach().cpu().numpy().copy())
                    return poses
                def score_history(**kwargs):
                    assert len(history) == 10
                    assert all(len(poses) == len(kwargs['ob_in_cams']) for poses in history)
                    for iterations in [1, 2, 3, 4, 5, 10]:
                        poses = history[iterations-1]
                        scores, vis = original_score(**dict(kwargs, ob_in_cams=poses))
                        best = int(torch.as_tensor(scores).argmax().item())
                        pred = poses[best] @ estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
                        assert np.isfinite(pred).all()
                        points = vertices @ pred[:3, :3].T + pred[:3, 3]
                        add = float(np.linalg.norm(points-target, axis=1).mean())
                        adds = float(tree.query(points)[0].mean())
                        row = dict(arm=arm, seed=seed, object=obj, frame=frame, condition=condition, iterations=iterations, add_mm=add*1000, adds_mm=adds*1000, add_diameter=add/diameter, adds_diameter=adds/diameter, diameter_m=diameter, prediction=pred.tolist(), ground_truth=gt.tolist())
                        if (*prefix, iterations) not in completed:
                            records.append(row); completed.add((*prefix, iterations))
                    write(path, records)
                    return scores, vis
                module.egocentric_delta_pose_to_pose = update
                estimator.scorer.predict = score_history
                estimator.generate_random_pose_hypo = generate
                try:
                    estimator.gt_pose = None
                    returned = estimator.register(K=reader.K, rgb=rgb.copy(), depth=z, ob_mask=mask.copy(), ob_id=obj, iteration=10)
                    if not history:
                        # Production register returns early when fewer than four
                        # valid masked depth pixels remain. Keep this failure,
                        # rather than dropping the frame or claiming refinement.
                        pred = np.asarray(returned)
                        assert np.isfinite(pred).all()
                        points = vertices @ pred[:3, :3].T + pred[:3, 3]
                        add = float(np.linalg.norm(points-target, axis=1).mean())
                        adds = float(tree.query(points)[0].mean())
                        for iterations in [1, 2, 3, 4, 5, 10]:
                            if (*prefix, iterations) not in completed:
                                records.append(dict(arm=arm, seed=seed, object=obj, frame=frame, condition=condition, iterations=iterations, actual_refinement_iterations=0, fallback='insufficient_valid_depth', add_mm=add*1000, adds_mm=adds*1000, add_diameter=add/diameter, adds_diameter=adds/diameter, diameter_m=diameter, prediction=pred.tolist(), ground_truth=gt.tolist()))
                                completed.add((*prefix, iterations))
                        write(path, records)
                finally:
                    module.egocentric_delta_pose_to_pose = original_update
                    estimator.scorer.predict = original_score
                    estimator.generate_random_pose_hypo = original_generate
                status(state='registration', cases=len(records), expected=40*7*2*6, last_case=list(prefix))
                print('REGISTER', prefix, len(records), flush=True)
    assert len(records) == 40*7*2*6
    return records


def report(records):
    objects = sorted({r['object'] for r in records})
    summary = []
    for condition in ['observed', 'dropout40']:
        for arm in ['original'] + ARMS:
            for iteration in [1, 2, 3, 4, 5, 10]:
                rows = [r for r in records if r['condition'] == condition and r['arm'] == arm and r['iterations'] == iteration]
                summary.append(dict(condition=condition, arm=arm, iterations=iteration, **{metric: float(np.mean([r[metric] for r in rows])) for metric in ['add_mm', 'adds_mm', 'add_diameter', 'adds_diameter']}))
    primary = [r for r in records if r['condition'] == 'observed' and r['iterations'] == 5 and r['arm'] != 'original']
    seed_deltas, object_deltas = {}, {}
    for seed in SEEDS:
        means = {arm: float(np.mean([r['adds_diameter'] for r in primary if r['seed'] == seed and r['arm'] == arm])) for arm in ARMS}
        seed_deltas[seed] = dict(**means, delta=means['mixture']-means['standard'])
    for obj in objects:
        means = {arm: float(np.mean([r['adds_diameter'] for r in primary if r['object'] == obj and r['arm'] == arm])) for arm in ARMS}
        object_deltas[obj] = dict(**means, delta=means['mixture']-means['standard'])
    means = {arm: float(np.mean([r['adds_diameter'] for r in primary if r['arm'] == arm])) for arm in ARMS}
    delta = means['mixture']-means['standard']
    relative = -delta/means['standard']
    values = np.array([r['delta'] for r in object_deltas.values()])
    rng = np.random.default_rng(20261116)
    ci = np.quantile(rng.choice(values, (20000, len(values)), replace=True).mean(1), [.025, .975]).tolist()
    seed_wins = sum(r['delta'] < 0 for r in seed_deltas.values())
    object_wins = sum(r['delta'] < 0 for r in object_deltas.values())
    convincing = bool(relative >= .05 and seed_wins == 3 and object_wins >= 4 and ci[1] < 0)
    conclusion = 'Convincing improvement within this limited local experiment' if convincing else 'No convincing improvement under the predeclared criterion; inspect mixed effects and uncertainty'
    analysis = dict(primary_metric='ADD-S / diameter at iteration 5 on observed input', means=means, mixture_minus_standard=delta, relative_improvement=relative, descriptive_object_cluster_ci95=ci, seed_deltas=seed_deltas, object_deltas=object_deltas, convincing_local_improvement=convincing, conclusion=conclusion)
    write(OUT / 'summary.json', summary); write(OUT / 'analysis.json', analysis)
    lines = ['# Small-error training results', '', conclusion + '.', '', f'Primary endpoint: {analysis["primary_metric"]}. Relative improvement: {100*relative:.2f}%. Seed wins: {seed_wins}/3; object wins: {object_wins}/5. Paired object-bootstrap interval for mixture minus standard: {ci}. Only five correlated object sequences: this interval is descriptive.', '', '| Condition | Arm | Iteration | ADD mm | ADD-S mm |', '|---|---|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["condition"]} | {row["arm"]} | {row["iterations"]} | {row["add_mm"]:.3f} | {row["adds_mm"]:.3f} |')
    lines += ['', '## Primary differences by seed', '', '```json', json.dumps(seed_deltas, indent=2), '```', '', '## Primary differences by object', '', '```json', json.dumps(object_deltas, indent=2), '```', '', '## Limits and prior work', '', 'Static small/exact-pose mixture; not unrolled iterative training. Original encoder and scoring network fixed. Three matched seeds, 40 locally unused frames on five already-inspected objects, original dataset GT masks. No official benchmark or unseen-pretraining claim. No selection of a test iteration by GT accuracy. Evaluation frames can be temporally close to previously inspected frames. Any improvement is local evidence, not definitive transfer.', '', 'DeepIM (ECCV 2018), section 4.1 / Table 1, already identifies insufficient close-to-GT training poses and demonstrates iterative training. This ablation does not establish method novelty.', '', 'https://openaccess.thecvf.com/content_ECCV_2018/papers/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.pdf']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, condition in zip(axes, ['observed', 'dropout40']):
        for arm in ['original'] + ARMS:
            rows = [r for r in summary if r['condition'] == condition and r['arm'] == arm]
            ax.plot([r['iterations'] for r in rows], [r['adds_mm'] for r in rows], marker='o', label=arm)
        ax.set(title=condition, xlabel='Refinement iterations', ylabel='Mean ADD-S (mm)')
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Static small/exact training mixture · 40 fixed frames · three training seeds')
    fig.tight_layout(); fig.savefig(OUT / 'iteration_errors.png', dpi=160)
    status(state='complete', conclusion=conclusion, report=str(OUT / 'RESULTS.md'))
    print('FINAL', json.dumps(analysis), flush=True)


def main():
    torch.set_num_threads(4)
    logging.getLogger().setLevel(logging.WARNING)
    manifest = lock_manifest()
    if (OUT / 'analysis.json').exists():
        report(json.loads((OUT / 'registration.json').read_text())); return
    wait_gpu()
    status(state='preparing')
    predictor = h.PoseRefinePredictor()
    data = prepare(predictor)
    train(predictor, data)
    # Release training caches and model before full multi-candidate registration.
    del data, predictor
    import gc
    gc.collect(); torch.cuda.empty_cache()
    records = registration(None, manifest)
    report(records)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status(state='failed', traceback=traceback.format_exc())
        raise
