"""CPU-only audit of demo and BOP depth support. Never changes source data."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from mask_study import DATA, ROOT, SPLIT, write_csv

OUT = ROOT/'debug/point_audit_v1'


def depth_erosion(depth):
    """CPU equivalent of Utils.erode_depth default neighborhood rejection.

    Uses float32 metres, 5x5 support, 1mm difference and >80% rejection.
    This is before bilateral smoothing, not a claim about final network inputs.
    """
    d = depth.astype(np.float32)
    h, w = d.shape
    padded = np.pad(d, 2, constant_values=np.nan)
    bad = np.zeros_like(d)
    total = np.zeros_like(d)
    for y in range(5):
        for x in range(5):
            neighbor = padded[y:y+h, x:x+w]
            inside = np.isfinite(neighbor)
            total += inside
            bad += inside & ((neighbor < .001) | (neighbor >= 100) | (np.abs(neighbor-d) > .001))
    return np.where(bad/total > .8, 0, d)


def inspect(name, frame, split, depth, mask, image, metadata):
    if mask.ndim == 3:
        mask = next(mask[...,c] for c in range(mask.shape[2]) if mask[...,c].any())
    mask = mask > 0
    if mask.shape != depth.shape:
        raise ValueError('Mask/depth alignment mismatch')
    valid = np.isfinite(depth) & (depth >= .001) & (depth < 100)
    cleaned = depth_erosion(depth)
    support = mask & valid
    survives = mask & (cleaned >= .001)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    largest = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else -1
    islands = support & (labels != largest)
    median = float(np.median(depth[support])) if support.any() else 0.
    # Conservative diagnostic flags, never ground-truth labels for bad points.
    tail = survives & (np.abs(depth-median) > .05)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5,5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    boundary = mask & ~inner
    row = dict(dataset=name, frame=frame, split=split, image_pixels=int(mask.size),
        mask_pixels=int(mask.sum()), full_valid_depth=int(valid.sum()), background_valid_depth=int((valid & ~mask).sum()),
        masked_valid_depth=int(support.sum()), masked_invalid_depth=int((mask & ~valid).sum()),
        masked_after_existing_erosion=int(survives.sum()), connected_mask_components=count-1,
        valid_depth_outside_largest_component=int(islands.sum()), median_mask_depth_m=median,
        surviving_depth_gt_50mm_from_median=int(tail.sum()), boundary_valid_depth=int((boundary & support).sum()), **metadata)
    if image is not None:
        overlay = image.copy()
        overlay[boundary & support] = [0,255,255]
        overlay[islands] = [255,0,255]
        overlay[tail] = [0,0,255]
        cv2.imwrite(str(OUT/'examples'/f'{name}_{frame}.jpg'), overlay)
    return row


def main():
    OUT.mkdir(exist_ok=True); (OUT/'examples').mkdir(exist_ok=True)
    records = []
    for name in ['mustard0', 'kinect_driller_seq']:
        base = ROOT/'demo_data'/name
        masks = sorted((base/'masks').glob('*.png'))
        for path in masks:
            d = cv2.imread(str(base/'depth'/path.name), -1)
            if d is None:
                continue
            records.append(inspect(name, path.stem, 'demo_initialization', d.astype(np.float32)/1000,
                cv2.imread(str(path), -1), cv2.imread(str(base/'rgb'/path.name)), dict(depth_scale_to_m=.001)))
    with SPLIT.open() as f:
        samples = list(csv.DictReader(f))
    for sample in samples:
        obj, im = int(sample['obj_id']), int(sample['im_id'])
        scene = DATA/f'lm_test_all/test/{obj:06d}'
        gt = json.loads((scene/'scene_gt.json').read_text())[str(im)]
        indices = [i for i,g in enumerate(gt) if g['obj_id']==obj]
        assert len(indices)==1
        cam = json.loads((scene/'scene_camera.json').read_text())[str(im)]
        scale = float(cam['depth_scale'])/1000
        depthpath = scene/f'depth/{im:06d}.png'
        maskpath = scene/f'mask_visib/{im:06d}_{indices[0]:06d}.png'
        d = cv2.imread(str(depthpath), -1).astype(np.float32)*scale
        imagepath = next((scene/'rgb').glob(f'{im:06d}.*'))
        # Inspect held-out inputs descriptively; do not use them to tune any filter.
        example = cv2.imread(str(imagepath)) if sample['split']=='development' and im==int(next(r['im_id'] for r in samples if r['obj_id']==str(obj) and r['split']=='development')) else None
        records.append(inspect('bop_linemod', f'{obj:06d}_{im:06d}', sample['split'], d,
            cv2.imread(str(maskpath), -1), example, dict(depth_scale_to_m=scale)))
    write_csv(OUT/'per_frame.csv', records)
    summary = []
    for dataset in sorted({r['dataset'] for r in records}):
        subset = [r for r in records if r['dataset']==dataset]
        total = lambda name: sum(r[name] for r in subset)
        denom = total('masked_valid_depth')
        summary.append(dict(dataset=dataset, masked_frames=len(subset),
            masked_valid_depth=denom, invalid_mask_fraction=total('masked_invalid_depth')/max(1,total('mask_pixels')),
            existing_erosion_removal_fraction=1-total('masked_after_existing_erosion')/max(1,denom),
            disconnected_valid_fraction=total('valid_depth_outside_largest_component')/max(1,denom),
            surviving_50mm_tail_fraction=total('surviving_depth_gt_50mm_from_median')/max(1,total('masked_after_existing_erosion'))))
    write_csv(OUT/'summary.csv', summary)
    text = '# Demo versus BOP point audit\n\nRead-only CPU audit of raw depth, mask connectivity and a CPU reproduction of existing depth erosion (before bilateral smoothing). No dataset points were deleted.\n\n'
    text += 'The included demos are sequences used for initialization and tracking. Their masks cover initialization frames only. The BOP sample contains 520 object-balanced frames with GT visible masks and pose annotations. These differ in objects, sensor, scenes and annotation coverage; aggregate differences are descriptive and cannot isolate a dataset effect.\n\n'
    text += '| Input | Masked frames | Invalid depth inside mask | Removed by existing erosion | Valid depth in other mask components | Surviving depth >50mm from median |\n| --- | --- | --- | --- | --- | --- |\n'
    for r in summary:
        text += f"| {r['dataset']} | {r['masked_frames']} | {r['invalid_mask_fraction']:.2%} | {r['existing_erosion_removal_fraction']:.2%} | {r['disconnected_valid_fraction']:.2%} | {r['surviving_50mm_tail_fraction']:.2%} |\n"
    text += '\n## What can be filtered?\n\nInvalid and near-zero depth is already rejected. Background outside the mask is excluded from translation initialization, but full-image depth is still passed into pose refinement and ranking; removing it globally could change learned inputs and hide useful occluders. Existing code also applies depth erosion and bilateral smoothing. It is incorrect to call all background points unused.\n\nDisconnected mask components and unusually separated depth layers are candidate filters. Neither is automatically an outlier: disconnected visible surfaces can belong to an occluded object, and object geometry can legitimately extend more than 50mm in depth. The 50mm threshold is a diagnostic flag, not a tuned removal rule. Overlay colors: yellow boundary, magenta smaller components, red surviving depth tails.\n\n## Validation needed\n\nUse development frames only to choose filter thresholds. Compare unchanged input against largest-component masks and geometry-scaled depth gating in an isolated ablation; keep the ongoing morphology study unchanged. Count empty filtered masks as failures. Freeze the choice before examining held-out pose outcomes, and disclose that these study images and input diagnostics were previously inspected. For the demos, tracking outputs in ob_in_cam are not established independent GT; use them for qualitative stability only. BOP GT poses support accuracy measurements. Improvements must be evaluated, not inferred from fewer points.\n'
    (OUT/'REPORT.md').write_text(text)
    (OUT/'status.json').write_text(json.dumps(dict(state='complete',frames=len(records),source_data_modified=False),indent=2))
    print(text)


if __name__=='__main__':
    main()
