"""Create a self-contained report with actual real-image results and inline figures."""
from pathlib import Path
import json
import html
import base64
import shutil
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):return json.loads(path.read_text())


def main():
    geom=ROOT/'research/geometric_structure_20261111/completion'
    rot=ROOT/'research/real_rotation_20261111'
    g=read(geom/'results/evaluation.json');r=read(rot/'results/evaluation.json');verification=read(HERE/'verification.json')
    assert verification['passed']
    geo_rows=[]
    for arm in ['frozen','depth','normal','probe_frozen','probe_depth','probe_normal']:
        vals=[g[f'{s}/{arm}/reserved']['macro'] for s in [42,123,2026]]
        geo_rows.append([arm,f"{np.mean([v['mae_mm'] for v in vals]):.2f}",f"{np.mean([v['median_normal_angle_deg'] for v in vals]):.2f}" if arm!='depth' else 'unsupervised'])
    rotation_rows=[]
    for arm in ['pretrained','ordinary','symmetry']:
        for cond in ['textured','uniform']:
            vals=[r[f'{s}/{arm}/{cond}']['macro'] for s in [42,123,2026]]
            rotation_rows.append([arm,cond,f"{np.mean([v['rotation_error_deg'] for v in vals]):.2f}",f"{100*np.mean([v['recall_30deg'] for v in vals]):.1f}%",f"{vals[0]['oracle_error_deg']:.2f}"])
    normal_base=np.mean([g[f'{s}/frozen/reserved']['macro']['median_normal_angle_deg'] for s in [42,123,2026]])
    normal_new=np.mean([g[f'{s}/normal/reserved']['macro']['median_normal_angle_deg'] for s in [42,123,2026]])
    fresh_base=np.mean([g[f'{s}/probe_frozen/reserved']['macro']['median_normal_angle_deg'] for s in [42,123,2026]])
    fresh_new=np.mean([g[f'{s}/probe_normal/reserved']['macro']['median_normal_angle_deg'] for s in [42,123,2026]])
    ordinary=np.mean([r[f'{s}/ordinary/textured']['macro']['rotation_error_deg'] for s in [42,123,2026]])
    sym=np.mean([r[f'{s}/symmetry/textured']['macro']['rotation_error_deg'] for s in [42,123,2026]])
    orig=r['42/pretrained/textured']['macro']['rotation_error_deg']
    lines=['# Encoder experiments for the mentor meeting — November 11, 2026','',
      '## What the mentor’s three notes became','',
      '- **Fine-tune symmetry:** compare ordinary and symmetry-aware reference-rotation supervision, using BOP discrete symmetry annotations for the eggbox and glue bottle.',
      '- **Add structure with a pipeline:** real RGB-D crop → hide XYZ patches → shared encoder → depth/normal decoder, with reproducible manifests, matched seeds and independent readout checks.',
      '- **Make texture useful:** match real RGB object crops to differently oriented CAD references, then replace only reference appearance with uniform color while holding geometry fixed. This encourages and tests appearance use; it does not guarantee texture learning.','',
      '## Main findings','',
      f'- Surface-normal supervision reduced reserved-object surface-direction error from **{normal_base:.2f}° to {normal_new:.2f}°** (three-seed averages). The fresh-decoder comparison was **{fresh_base:.2f}° to {fresh_new:.2f}°**, supporting a modest improvement in normal decodability in this controlled completion task.',
      '- Depth completion did not improve consistently over the original frozen encoder. A simple nearest-observed-depth baseline was better at depth filling. Do not claim general geometric reconstruction success.',
      f'- Real-image textured-reference rotation error: pretrained **{orig:.2f}°**, ordinary adaptation **{ordinary:.2f}°**, symmetry-aware adaptation **{sym:.2f}°**. These are high errors; judge adaptation relative to the baseline and bank coverage, not as deployment-ready accuracy.',
      '- All queries in the two main experiments are real BOP photographs. CAD renders are reference views or geometry labels. The mustard illustration is separate and uses the FoundationPose demo asset.','',
      '## Geometry completion: real BOP RGB-D','',
      'Original encoder: FoundationPose refinement shared CNN, original weights untouched. Keep measured XYZ context but hide 40% of an 8×8 block grid; add 2 mm noise to retained depth. Predict missing CAD depth and surface normals. Pose and camera labels generate targets, never fill the hidden XYZ input. Context center is computed from retained observations only.',
      '', 'Seven training objects; development objects 10/11 select checkpoints. Objects 3/7 were excluded from the preceding adaptation run and reserved for this follow-up. Earlier evaluation objects 12–15 are exploratory. Train 1500 steps per arm/seed and 500 steps per fresh-decoder probe; three seeds, 18,000 optimizer steps. Both adapted arms use the same small feature anchor.','',
      '| Encoder/readout | Reserved depth MAE mm ↓ | Reserved normal angle deg ↓ |','|---|---:|---:|']
    lines+=['| '+' | '.join(row)+' |' for row in geo_rows]
    lines+=['',f"Observed-only nearest-fill depth baseline: **{g['baseline/nearest/reserved']['macro']['mae_mm']:.2f} mm**. Normal metric is object-macro average of per-image median angular error. The depth-only model’s normal branch is unsupervised. Fresh probes freeze the encoder and reinitialize an identical decoder.", '',
      '## Rotation and texture: real BOP RGB queries','',
      'Objects: can (5), driller (8), duck (9), eggbox (10), glue (11). Per object: 144 training, 36 development and 36 evaluation queries, with two temporal buffer blocks. The 48-reference bank comes only from training poses. No evaluation GT pose is inserted into the bank. Train ordinary and symmetry-aware encoders for 1500 steps, three seeds, 9000 optimizer steps. Report all objects, including difficult failures.','',
      '| Encoder | References | Rotation error deg ↓ | Recall ≤30° ↑ | Oracle bank error deg ↓ |','|---|---|---:|---:|---:|']
    lines+=['| '+' | '.join(row)+' |' for row in rotation_rows]
    lines+=['','The oracle is the closest orientation available in the fixed bank, selected using labels only for diagnostic coverage measurement. Texture ablation keeps the real query unchanged and changes CAD-reference appearance. A gap therefore measures sensitivity to reference appearance, not label understanding alone. Rotation equivalence uses the rotational part of BOP symmetries; this is not a full-pose loss handling symmetry translations.','',
      '## Real images and predictions','',
      'The comparison images show the first evaluation query of each object, fixed before results. Background masking and crop boxes use GT visible masks. Rendered references are labeled as such.','',
      '![Real-image rotation matching](real_rotation_examples.png)','',
      '![Real-image geometry completion](completion_examples.png)','',
      '## Additional robustness checks','']
    stress=geom/'stress_results/REPORT.md'
    if stress.exists():lines.append(stress.read_text().replace('# Geometry stress tests','Changed missingness and noise; no retraining:'))
    dense=rot/'dense_results/REPORT.md'
    if dense.exists():
        lines+=['','## Reference coverage follow-up','',dense.read_text().replace('# Exploratory reference-bank coverage control','Fixed generic orientation bank:'),'','![Dense-bank real-image matching](dense_rotation_examples.png)']
    lines+=['','## What can be said at the meeting','',
      f'“We tested the mentor’s notes with real BOP images. Normal supervision gave a modest improvement in surface-normal prediction, including with a fresh decoder, but depth completion did not beat a simple interpolation baseline. We also measured rotation matching, symmetry-aware training and a controlled reference-texture ablation. This is evidence about specific encoder features, not proof of better full-pipeline 6D pose.”','',
      '## Limits and audit','',
      '- Official LINEMOD test sequences were repurposed for development. Several objects/frames had prior local inspection. Held out here means excluded from the current adaptation, not verified unseen to pretrained FoundationPose.',
      '- Temporal blocks share scenes. Two reserved geometry objects and three seeds provide limited generalization evidence. Seed standard deviation is not a confidence interval.',
      '- GT masks, known object identity, known CAD diameter, artificial missingness and rendered labels simplify these diagnostics. No detector, deployed sensor missingness or end-to-end pose improvement was demonstrated.',
      '- The earlier RGB-only depth experiment was negative (27.05 mm frozen versus 27.84 mm depth-plus-slope adaptation on its four adaptation-held-out objects). The geometry-completion task has different inputs, masks and targets: do not compare its mm numbers directly against RGB-only results.',
      '- A first real-rotation rendering run was excluded: an explicit material override suppressed vertex colors, making textured and uniform references identical. Its files remain in `real_rotation_20261111/invalid_render_v1`. Corrected references must differ in pixels while paired depth remains identical.',
      '- Verification covered real-data split separation, training-only reference banks, appearance differences, symmetry-metric equivalence, geometry target exclusion, unit normals, actual encoder updates, frozen BN statistics, fresh-probe encoder preservation and development-only checkpoint selection.',
      '- Deployed pretrained weights were not modified. All candidate checkpoints remain separate.','',
      '## Reproduction','',
      'From `/home/Sahil/FoundationPose`, using `/home/Sahil/miniforge3/envs/foundationpose/bin/python`: run `research/geometric_structure_20261111/completion/run.py`, `research/geometric_structure_20261111/completion/stress_test.py`, and `research/real_rotation_20261111/run.py`. Protocols, exact manifests, per-frame CSVs, histories and JSON evaluations are in those experiment directories.','',
      '## Separate mustard-bottle illustration','',
      'The earlier image below uses the local FoundationPose `demo_data/mustard0/mesh` asset; it is not a BOP photograph or evidence from the real-BOP evaluation.','',
      '![Mustard demo illustration](mustard_reference_views.png)']
    assets={'real_rotation_examples.png':rot/'results/real_rotation_examples.png','completion_examples.png':geom/'results/completion_examples.png','mustard_reference_views.png':ROOT/'research/texture_orientation_20261111/pilot/reference_views.png'}
    if (rot/'dense_results/dense_rotation_examples.png').exists():
        assets['dense_rotation_examples.png']=rot/'dense_results/dense_rotation_examples.png'
    for name,src in assets.items():shutil.copyfile(src,HERE/name)
    markdown='\n'.join(lines)+'\n';(HERE/'REPORT.md').write_text(markdown)
    # Simple self-contained HTML: images remain usable after copying this one file over SSH.
    body=[];in_table=False;in_list=False
    for line in lines:
        if line.startswith('|'):
            if set(line.replace('|','').replace('-','').replace(':','').strip())==set():continue
            if not in_table:body.append('<table>');in_table=True
            cells=[x.strip() for x in line.strip('|').split('|')]
            body.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>');continue
        if in_table:body.append('</table>');in_table=False
        if line.startswith('!['):
            name=line.split('](')[1].rstrip(')');data=base64.b64encode((HERE/name).read_bytes()).decode()
            body.append(f'<img src="data:image/png;base64,{data}" alt="{html.escape(name)}">');continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));body.append(f'<h{level}>'+html.escape(line[level:].strip())+f'</h{level}>')
        elif line.startswith('- '):body.append('<p>• '+html.escape(line[2:])+'</p>')
        elif line:body.append('<p>'+html.escape(line)+'</p>')
    (HERE/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>BOP encoder experiments</title><style>body{font:16px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:20px;color:#182838}img{width:100%;margin:20px 0}table{border-collapse:collapse;width:100%}td{border:1px solid #bbc;padding:8px}h1,h2{color:#17475a}p{white-space:normal}</style>'+''.join(body))
    bundle=HERE/'BOP_encoder_experiments.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['REPORT.md','REPORT.html','verification.json',*assets]:archive.write(HERE/name,name)
        for source in [rot/'results/evaluation.json',geom/'results/evaluation.json',geom/'stress_results/evaluation.json',rot/'dense_results/evaluation.json']:
            if source.exists():archive.write(source,str(source.relative_to(ROOT/'research')))
    desktop=Path('/home/Sahil/Desktop')
    for name in ['REPORT.html','BOP_encoder_experiments.zip']:
        shutil.copyfile(HERE/name,desktop/('BOP_mentor_report.html' if name=='REPORT.html' else name))
    print('Self-contained report and ZIP saved:',HERE,flush=True)


if __name__=='__main__':main()
