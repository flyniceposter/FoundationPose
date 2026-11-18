"""Original-model diagnostic recording each step of continuous refinement."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import importlib
import json
import logging
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from pose_eval import FoundationPose, LinemodReader, dr, t

HERE = Path(__file__).resolve().parent
OUT = HERE / 'iteration_diagnostic'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--all-frames', action='store_true')
    args = parser.parse_args()
    assert args.iterations > 0
    global OUT
    OUT = HERE / f'iteration_diagnostic_{args.iterations}'
    if args.all_frames:
        OUT = HERE / f'iteration_diagnostic_{args.iterations}_20frames'
    OUT.mkdir(exist_ok=True)
    # Respect the existing research job's GPU availability rule.
    ready = 0
    while ready < 3:
        line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()[0]
        free, util = map(int, line.split(','))
        ready = ready + 1 if free >= 20480 and util < 40 else 0
        print('GPU readiness', ready, free, util, flush=True)
        if ready < 3:
            time.sleep(30)
    torch.set_num_threads(4)
    logging.getLogger().setLevel(logging.WARNING)
    baseline = json.loads((HERE / 'pose/records.json').read_text())
    selected = {}
    for row in baseline:
        if row['arm'] == 'original' and row['condition'] == 'observed':
            selected.setdefault(row['object'], [])
            if row['frame'] not in selected[row['object']]:
                selected[row['object']].append(row['frame'])
    selected = [(obj, frame) for obj, frames in sorted(selected.items()) for frame in (frames if args.all_frames else frames[:1])]
    assert len(selected) == (20 if args.all_frames else 5), selected
    (OUT / 'manifest.json').write_text(json.dumps(dict(model='original', condition='observed', iterations=args.iterations, frames=selected, selection='Existing evaluation frames; no selection by iteration error', ground_truth_used_for='Error measurement only; existing GT object masks supplied to registration'), indent=2))
    context = dr.RasterizeCudaContext()
    estimator = None
    records = []
    for obj, frame in selected:
        reader = LinemodReader(str(t.DATA / 'lm_test_all/test' / f'{obj:06d}'), split=None)
        reader.K = reader.K.astype(np.float32)
        mesh = reader.get_gt_mesh(obj)
        if estimator is None:
            estimator = FoundationPose(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=reader.symmetry_tfs[obj], mesh=mesh, glctx=context, debug=0, debug_dir=str(OUT / 'debug'))
        else:
            estimator.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=reader.symmetry_tfs[obj], mesh=mesh)
            estimator.make_rotation_grid(min_n_views=40, inplane_step=60)
        estimator.diameter = float(estimator.diameter)
        index = reader.id_strs.index(frame)
        rgb, depth, mask = reader.get_color(index), reader.get_depth(index), reader.get_mask(index, obj)
        gt = reader.get_gt_pose(index, obj)
        vertices = np.asarray(mesh.vertices)
        vertices = vertices[np.linspace(0, len(vertices)-1, min(5000, len(vertices)), dtype=int)]
        target = vertices @ gt[:3, :3].T + gt[:3, 3]
        # Capture the actual updates from one continuous production refinement run.
        # Score each saved candidate set afterwards so scoring cannot affect updates.
        module = importlib.import_module(estimator.refiner.__class__.__module__)
        original_update = module.egocentric_delta_pose_to_pose
        original_score = estimator.scorer.predict
        history = []
        def capture_update(*a, **kw):
            updated = original_update(*a, **kw)
            history.append(updated.detach().cpu().numpy().copy())
            if len(history) % 10 == 0:
                print('REFINEMENT', obj, len(history), flush=True)
            return updated
        def score_history(**kwargs):
            # Current diagnostic has <1024 candidates: one update batch per step.
            assert len(history) == args.iterations
            assert all(len(poses) == len(kwargs['ob_in_cams']) for poses in history)
            for iterations, poses in enumerate(history, 1):
                score_kwargs = dict(kwargs, ob_in_cams=poses)
                scores, vis = original_score(**score_kwargs)
                best = int(torch.as_tensor(scores).argmax().item())
                pred = poses[best] @ estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
                assert np.isfinite(pred).all()
                points = vertices @ pred[:3, :3].T + pred[:3, 3]
                row = dict(object=obj, frame=frame, condition='observed', arm='original', iterations=iterations,
                           add_mm=float(np.linalg.norm(points-target, axis=1).mean()*1000),
                           adds_mm=float(cKDTree(target).query(points)[0].mean()*1000),
                           prediction=pred.tolist(), ground_truth=gt.tolist())
                records.append(row)
                (OUT / 'records.json').write_text(json.dumps(records, indent=2))
                print('RESULT', obj, frame, iterations, row['add_mm'], row['adds_mm'], flush=True)
            return scores, vis
        module.egocentric_delta_pose_to_pose = capture_update
        estimator.scorer.predict = score_history
        try:
            estimator.gt_pose = None
            pred = estimator.register(K=reader.K, rgb=rgb.copy(), depth=depth.copy(), ob_mask=mask.copy(), ob_id=obj, iteration=args.iterations)
            assert np.allclose(pred, records[-1]['prediction'])
        finally:
            module.egocentric_delta_pose_to_pose = original_update
            estimator.scorer.predict = original_score
    summary = [dict(iterations=i, mean_add_mm=float(np.mean([r['add_mm'] for r in records if r['iterations']==i])), mean_adds_mm=float(np.mean([r['adds_mm'] for r in records if r['iterations']==i]))) for i in range(1, args.iterations+1)]
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
