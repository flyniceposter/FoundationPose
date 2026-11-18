from pathlib import Path
import json, html, base64
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    lines=['# Robust encoder experiment — November 11–12, 2026','','Exploratory local BOP LINEMOD development experiments. Previously inspected data; GT object masks; no official benchmark claim. See PROTOCOL.md for the fixed design.','']
    lines+=['| Arm | Seed | Evaluation condition | Depth MAE mm | Normal error degrees |','|---|---:|---|---:|---:|']
    for file in sorted((HERE/'results').glob('*_complete.json')):
        arm,seed=file.stem.removesuffix('_complete').rsplit('_',1)
        for key,value in json.loads(file.read_text()).items():
            m=value['summary']['macro'];lines.append(f"| {arm} | {seed} | {key} | {m['mae_mm']:.3f} | {m['median_normal_angle_deg']:.2f} |")
    path=HERE/'pose/records.json'
    if path.exists():
        records=json.loads(path.read_text());lines+=['','Pose registration results: ADD measures corresponding-point error; ADD-S uses nearest points and can be permissive for asymmetric objects. Report both.','','| Arm | Condition | Cases | ADD mm | ADD-S mm | ADD recall <10% diameter | ADD-S recall <10% diameter |','|---|---|---:|---:|---:|---:|---:|']
        for arm in sorted({r['arm'] for r in records}):
            for condition in sorted({r['condition'] for r in records}):
                rows=[r for r in records if r['arm']==arm and r['condition']==condition]
                # Macro average by object, then by seed. Balanced complete runs have equal cases.
                def mean(key):
                    return float(np.mean([np.mean([r[key] for r in rows if r['object']==obj]) for obj in sorted({r['object'] for r in rows})]))
                lines.append(f"| {arm} | {condition} | {len(rows)} | {mean('add_mm'):.2f} | {mean('adds_mm'):.2f} | {mean('add_recall_10diam'):.1%} | {mean('adds_recall_10diam'):.1%} |")
        lines+=['','The original model has 20 cases per condition; each adapted arm has the same 20 cases repeated across three training seeds (60). These different counts are expected. All adapted arms/seeds were checked against the same baseline frame/condition set.']
    else:lines+=['','Pose evaluation has not produced results yet.']
    lines+=['','Pose-head development training (not final evaluation):']
    for file in sorted((HERE/'heads').glob('*_complete.json')):
        value=json.loads(file.read_text());lines.append(file.stem+': best development error / diameter = '+str(value['best_development_error_diameter']))
    lines+=['','Input examples are saved in pose/input_*.png. Training histories and per-frame metrics are in results/. A completed run requires pose/complete.json and all 15 encoder-training completion files and all 9 pose-head-training completion files. Texture reconstruction is an auxiliary diagnostic, not evidence of semantic texture understanding. No improvement is assumed.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    body='<html><meta charset="utf-8"><title>Robust encoder results</title><style>body{max-width:1200px;margin:40px auto;font:16px sans-serif}pre{white-space:pre-wrap}img{width:30%;margin:1%}</style><h1>Robust encoder experiment</h1><pre>'+html.escape('\n'.join(lines))+'</pre>'
    for image in sorted((HERE/'pose').glob('input_*.png'))[:12]:
        body+='<figure><figcaption>'+html.escape(image.name)+'</figcaption><img src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'"></figure>'
    (HERE/'REPORT.html').write_text(body+'</html>')
if __name__=='__main__':main()
