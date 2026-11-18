"""Validate local formats and assemble evidence without treating predictions as GT."""
import argparse
import csv
import json
from pathlib import Path
import time
import cv2
import numpy as np
import trimesh
from mask_study import DATA, ROOT, SPLIT, atomic_json, write_csv

OUT = ROOT/'debug/dataset_comparison_v1'


def pose_valid(pose):
    return bool(pose.shape==(4,4) and np.isfinite(pose).all()
        and np.allclose(pose[3], [0,0,0,1], atol=1e-5)
        and np.allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-3)
        and abs(np.linalg.det(pose[:3,:3])-1)<1e-3 and pose[2,3]>0)


def validate():
    OUT.mkdir(exist_ok=True)
    findings, issues = [], []
    for name, meshname in [('mustard0','textured_simple.obj'), ('kinect_driller_seq','textured_mesh.obj')]:
        base = ROOT/'demo_data'/name
        rgbs = sorted((base/'rgb').glob('*.png'))
        depths = {p.stem:p for p in (base/'depth').glob('*.png')}
        masks = {p.stem:p for p in (base/'masks').glob('*.png')}
        K = np.loadtxt(base/'cam_K.txt').reshape(3,3)
        assert np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and np.allclose(K[2], [0,0,1])
        sizes, types, mismatches = set(), set(), 0
        for p in rgbs:
            rgb = cv2.imread(str(p)); d = cv2.imread(str(depths.get(p.stem, '')), -1)
            if rgb is None or d is None:
                issues.append(dict(dataset=name, frame=p.stem, problem='missing RGB/depth')); continue
            sizes.add(str(d.shape)); types.add(str(d.dtype))
            mismatches += int(rgb.shape[:2]!=d.shape)
            if p.stem in masks:
                m = cv2.imread(str(masks[p.stem]), -1)
                mismatches += int(m is None or m.shape[:2]!=d.shape)
        mesh = trimesh.load(base/'mesh'/meshname, process=False)
        poses = sorted((base/'ob_in_cam').glob('*.txt'))
        invalid_poses = sum(not pose_valid(np.loadtxt(p)) for p in poses)
        annotated = sorted((base/'annotated_poses').glob('*'))
        findings.append(dict(dataset=name, rgb_frames=len(rgbs), depth_frames=len(depths), mask_frames=len(masks),
            dimensions=';'.join(sorted(sizes)), depth_dtype=';'.join(sorted(types)), dimension_mismatches=mismatches,
            depth_scale_to_m=.001, independent_gt_pose_files=len(annotated), existing_output_pose_files=len(poses),
            invalid_existing_output_poses=invalid_poses, mesh_extents_m=str(mesh.extents.tolist()),
            task='first-frame registration then temporal tracking', pose_ground_truth_available=bool(annotated)))
    with SPLIT.open() as f:
        samples = list(csv.DictReader(f))
    grouped = {}
    for sample in samples:
        grouped.setdefault(int(sample['obj_id']), []).append(int(sample['im_id']))
    for obj, images in sorted(grouped.items()):
        scene = DATA/f'lm_test_all/test/{obj:06d}'
        cameras = json.loads((scene/'scene_camera.json').read_text())
        gt = json.loads((scene/'scene_gt.json').read_text())
        scales = {float(c['depth_scale']) for c in cameras.values()}
        # Upstream stores only the last camera depth scale for a scene. Verify
        # constancy here instead of silently accepting a frame-dependent scale.
        if len(scales)!=1:
            issues.append(dict(dataset=f'bop_{obj}', frame='all', problem='per-frame depth scale varies; upstream reader assumes scene-constant scale'))
        bad_gt = 0
        for im, annotations in gt.items():
            for g in annotations:
                if g['obj_id']!=obj:
                    continue
                p = np.eye(4); p[:3,:3] = np.array(g['cam_R_m2c']).reshape(3,3); p[:3,3] = np.array(g['cam_t_m2c'])/1000
                bad_gt += int(not pose_valid(p))
        sizes, types, mismatches = set(), set(), 0
        for im in images:
            cam = cameras[str(im)]; K = np.array(cam['cam_K']).reshape(3,3)
            assert np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and np.allclose(K[2], [0,0,1])
            rgbfile = next((scene/'rgb').glob(f'{im:06d}.*'))
            rgb = cv2.imread(str(rgbfile)); depth = cv2.imread(str(scene/f'depth/{im:06d}.png'), -1)
            indices = [j for j,g in enumerate(gt[str(im)]) if g['obj_id']==obj]
            assert len(indices)==1
            mask = cv2.imread(str(scene/f'mask_visib/{im:06d}_{indices[0]:06d}.png'), -1)
            assert rgb is not None and depth is not None and mask is not None
            mismatches += int(rgb.shape[:2]!=depth.shape or mask.shape[:2]!=depth.shape)
            sizes.add(str(depth.shape)); types.add(str(depth.dtype))
        mesh = trimesh.load(DATA/f'models/obj_{obj:06d}.ply', process=False)
        findings.append(dict(dataset=f'bop_linemod_{obj:02d}', rgb_frames=len(list((scene/'rgb').glob('*'))),
            depth_frames=len(list((scene/'depth').glob('*'))), mask_frames=len(list((scene/'mask_visib').glob('*'))),
            dimensions=';'.join(sorted(sizes)), depth_dtype=';'.join(sorted(types)), dimension_mismatches=mismatches,
            depth_scale_to_m=str(sorted(s/1000 for s in scales)), independent_gt_pose_files=len(gt), existing_output_pose_files=0,
            invalid_existing_output_poses=bad_gt, mesh_extents_m=str((mesh.extents/1000).tolist()),
            task='independent per-frame registration', pose_ground_truth_available=True))
    write_csv(OUT/'format_validation.csv', findings)
    atomic_json(OUT/'validation.json', dict(state='complete', demo_depth_checks='all RGB frames',
        bop_image_checks=520, bop_gt_checks='all annotations for all 13 objects', findings=findings, issues=issues,
        note='Existing ob_in_cam files are output artifacts, not independent GT. Alignment checks validate dimensions, not sensor calibration accuracy.'))
    assemble()


def assemble():
    validation = json.loads((OUT/'validation.json').read_text())
    states = {}
    for name in ['mask_study_v1', 'point_filter_study_v1']:
        p = ROOT/'debug'/name/'status.json'
        states[name] = json.loads(p.read_text()) if p.exists() else dict(state='not_started')
    text = '# FoundationPose: dataset comparison and point filtering\n\n'
    text += '## Main conclusion\n\nThe included NVIDIA demos and BOP LINEMOD are not directly interchangeable accuracy benchmarks. The demo script registers an object once and then tracks subsequent frames; the BOP runner registers every frame independently. The local demo sequences do not supply verified independent pose ground truth. A difference in visible tracking quality versus BOP recall would mix tasks, objects, sensor/scene conditions and annotation quality. It must not be presented as a measured dataset-specific accuracy difference.\n\n'
    text += '## Verified baseline\n\nThe completed LINEMOD run has 15,800/15,800 valid poses and 99.66% diagnostic ADD(-S) recall with GT visible masks. The first 5,186 poses were recovered from rounded log text; the rest are native precision. This is a recovered diagnostic baseline, not an official BOP AR/VSD result. See ../bop_recovery_20261021/analysis/REPORT.md.\n\n'
    text += '## Format validation\n\nSee format_validation.csv and validation.json for local measurements. Demo depth is converted from millimetres to metres. BOP uses depth_scale times raw depth to obtain millimetres, then divides by 1,000; BOP model geometry and pose translations also require millimetre-to-metre conversion. Both runners use object-to-camera poses. Model meshes are re-centered internally and returned poses refer to the original model frame. Calibration checks cover finite positive focal lengths and image dimensions; they do not certify physical RGB/depth registration.\n\n'
    text += f"Detected explicit validation issues: {len(validation['issues'])}. BOP image/mask checks cover the 520 study frames; GT matrix checks cover all annotations of the 13 objects. Demo RGB/depth checks cover every available frame.\n\n"
    text += 'The BOP reader stores one depth scale per scene. The validator checks that the local scene scales are constant; a format with varying per-frame scales would need a reader fix before comparisons.\n\n'
    text += '## Where points disappear\n\n1. Raw invalid/near-zero depth is rejected by the readers.\n2. Depth erosion rejects neighborhoods with too many invalid or discontinuous depth samples, then bilateral smoothing modifies surviving depth.\n3. The object mask selects depth for translation initialization. Smaller or shifted masks change the inferred center.\n4. CAD vertices are voxel-downsampled for internal model representation; this intentionally reduces redundant geometry.\n5. Refinement/ranking create pose-conditioned crops, so points outside a hypothesis crop do not reach that hypothesis input.\n6. Both local weight configs enable normalize_xyz. The dataset transform subtracts the candidate center, divides XYZ by mesh radius, and zeros coordinate entries whose absolute normalized values are at least 2. Thus even valid depth can be suppressed because of distance from a candidate pose; masking is coordinate-wise, so partially zeroed vectors are possible. See learning/datasets/h5_dataset.py. Counts for this stage require per-hypothesis instrumentation and have not been measured here.\n7. Full-image scene depth is the source for those crops; background inside crops is not automatically unused.\n\n'
    text += 'Read-only point audit: ../point_audit_v1/REPORT.md. It compares two demo initialization masks against 520 BOP masks. Invalid masked depth was 18.60% (mustard), 20.55% (driller), and 4.36% (BOP). These describe this sample, not a controlled sensor or dataset effect. Smaller BOP mask components contain about 1.71% of valid masked depth. Component and depth-tail flags are candidates, not proof of erroneous points.\n\n'
    text += '## Experiments and current status\n\n'
    for name, status in states.items():
        text += f"- {name}: **{status['state']}**, {status.get('checkpoints', status.get('records', 0))} saved results. See ../{name}/REPORT.md when complete, status.json and service.log while running.\n"
        report = ROOT/'debug'/name/'REPORT.md'
        if status['state']=='complete' and report.exists():
            text += '\n'+report.read_text().replace('# ', '### ')+'\n\n'
    text += '\n## What can be presented\n\nPresent the validated baseline, format/task differences, point-rejection map, audit overlays, and completed controlled ablations. Frame mitigation results as exploratory and include failure cases and paired uncertainty intervals. Input statistics and the full baseline were inspected before this study; these are study holdouts, not an untouched official benchmark. No claim of novelty is established. If filtering does not improve held-out performance, report that negative result rather than choosing a more favorable condition afterward.\n\n'
    text += '## Evidence and protocol sources\n\n- [Official BOP dataset format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md): model/camera coordinate conventions, pose translations, and depth scale.\n- [Official FoundationPose repository](https://github.com/NVlabs/FoundationPose): demo tracking and public-dataset evaluation.\n- Local source evidence: datareader.py, estimater.py, Utils.py, run_demo.py and run_linemod.py. Experiment source hashes are frozen in each manifest; core source has not been modified by these studies.\n'
    (OUT/'REPORT.md').write_text(text)
    done = all(s['state']=='complete' for s in states.values())
    failed = any(s['state']=='failed' for s in states.values())
    atomic_json(OUT/'status.json', dict(state='complete' if done else 'experiment_failed' if failed else 'waiting_for_experiments',
        updated_unix=time.time(), experiments=states, validation_issues=validation['issues']))
    return done or failed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--watch', action='store_true'); args = ap.parse_args()
    if not args.watch or not (OUT/'validation.json').exists():
        validate()
    while args.watch and not assemble():
        time.sleep(45)


if __name__=='__main__':
    main()
