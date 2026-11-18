"""Validate generated package structure, data coverage, and provenance; archive talk."""
import json
import re
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from study import HERE,atomic,digest


def main():
    deliverables=json.loads((HERE/'DELIVERABLES.json').read_text())
    assert deliverables['state']=='complete'
    checks={}
    for folder in ['results','secondary_results']:
        manifest=json.loads((HERE/folder/'manifest.json').read_text())
        for path,value in manifest['hashes'].items():assert digest(path)==value,path
        status=json.loads((HERE/folder/'status.json').read_text())
        assert status['state']=='measurements_complete'
        checks[folder+'_source_hashes']=len(manifest['hashes'])
    for filename in ['validation.json','analysis_validation.json','cad_center_validation.json']:
        assert json.loads((HERE/filename).read_text())['status']=='passed'
    for path,value in deliverables['files'].items():assert digest(HERE/path)==value,path
    images=sorted((HERE/'slides').glob('*.png'));assert len(images)>=16
    pdf=subprocess.check_output(['pdfinfo',str(HERE/'PRESENTATION.pdf')],text=True)
    pages=int(re.search(r'Pages:\s+(\d+)',pdf).group(1));assert pages==len(images)
    with zipfile.ZipFile(HERE/'PRESENTATION.pptx') as z:
        assert z.testzip() is None
        slides=[p for p in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',p)]
        notes=[p for p in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',p)]
        assert len(slides)==len(notes)==pages
        for part in z.namelist():
            if part.endswith(('.xml','.rels')):ET.fromstring(z.read(part))
    report=(HERE/'REPORT.md').read_text()
    for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report):assert (HERE/target).exists(),target
    checks.update(pdf_pages=pages,powerpoint_slides=len(slides),powerpoint_notes=len(notes),report_images_resolve=True,
                  scientific_validation=True,manifest_hashes_consistent=True,report_and_deck_hashes_match=True,
                  powerpoint_validation='ZIP/XML/relationship structure and slide counts; not rendered in Microsoft PowerPoint')
    atomic(HERE/'final_verification.json',dict(state='passed',checks=checks,scripts={p.name:digest(p) for p in HERE.glob('*.py')}))
    selected=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.md','.py','.pdf','.pptx','.html','.json']]
    selected+=list((HERE/'slides').glob('*.png'))+list((HERE/'analysis/figures').glob('*.png'))
    selected+=[p for p in (HERE/'analysis').glob('*.csv') if p.name not in ['primary_measurements.csv','secondary_measurements.csv']]
    selected+=list((HERE/'analysis').glob('*.json'))
    selected+=[HERE/'results/manifest.json',HERE/'secondary_results/manifest.json']
    with zipfile.ZipFile(HERE/'ENCODER_STUDY_PRESENTATION.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(selected)):z.write(p,p.relative_to(HERE))
    print(json.dumps(checks,indent=2))
    print('Presentation archive:',HERE/'ENCODER_STUDY_PRESENTATION.zip')


if __name__=='__main__':main()
