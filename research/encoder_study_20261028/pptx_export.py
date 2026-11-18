"""Export rendered slides and notes to standards-based OOXML without dependencies."""
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
HEADER='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def xml(tag,body,extra=''):
    return HEADER+f'<p:{tag} xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}" {extra}>{body}</p:{tag}>'


def rels(items):
    body=''.join(f'<Relationship Id="{rid}" Type="{R}/{kind}" Target="{target}"/>' for rid,kind,target in items)
    return HEADER+f'<Relationships xmlns="{PKG}">{body}</Relationships>'


def group():
    return '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'


def export(root,slides):
    root=Path(root);n=len(slides);output=root/'PRESENTATION.pptx'
    overrides=[('/ppt/presentation.xml','presentation.main'),('/ppt/slideMasters/slideMaster1.xml','slideMaster'),('/ppt/slideLayouts/slideLayout1.xml','slideLayout'),('/ppt/notesMasters/notesMaster1.xml','notesMaster')]
    overrides += [(f'/ppt/slides/slide{i}.xml','slide') for i in range(1,n+1)]
    overrides += [(f'/ppt/notesSlides/notesSlide{i}.xml','notesSlide') for i in range(1,n+1)]
    ct=HEADER+'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
    for part,kind in overrides:ct+=f'<Override PartName="{part}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml"/>'
    ct+='<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/></Types>'
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml',ct)
        z.writestr('_rels/.rels',rels([('rId1','officeDocument','ppt/presentation.xml')]))
        ids=''.join(f'<p:sldId id="{255+i}" r:id="rId{i+2}"/>' for i in range(1,n+1))
        z.writestr('ppt/presentation.xml',xml('presentation','<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:notesMasterIdLst><p:notesMasterId r:id="rId2"/></p:notesMasterIdLst><p:sldIdLst>'+ids+'</p:sldIdLst><p:sldSz cx="14630400" cy="8229600" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/>'))
        z.writestr('ppt/_rels/presentation.xml.rels',rels([('rId1','slideMaster','slideMasters/slideMaster1.xml'),('rId2','notesMaster','notesMasters/notesMaster1.xml')]+[(f'rId{i+2}','slide',f'slides/slide{i}.xml') for i in range(1,n+1)]))
        cmap='<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
        z.writestr('ppt/slideMasters/slideMaster1.xml',xml('sldMaster','<p:cSld><p:spTree>'+group()+'</p:spTree></p:cSld>'+cmap+'<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>'))
        z.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels',rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')]))
        z.writestr('ppt/slideLayouts/slideLayout1.xml',xml('sldLayout','<p:cSld name="Blank"><p:spTree>'+group()+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>','type="blank" preserve="1"'))
        z.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels',rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')]))
        z.writestr('ppt/notesMasters/notesMaster1.xml',xml('notesMaster','<p:cSld><p:spTree>'+group()+'</p:spTree></p:cSld>'+cmap+'<p:notesStyle/>'))
        z.writestr('ppt/notesMasters/_rels/notesMaster1.xml.rels',rels([('rId1','theme','../theme/theme1.xml')]))
        colors={'dk1':'173540','lt1':'FFFFFF','dk2':'48616A','lt2':'F5F3EC','accent1':'116B78','accent2':'D47B30','accent3':'8566A4','accent4':'69B9C0','accent5':'AAB9BD','accent6':'D7B896','hlink':'116B78','folHlink':'8566A4'}
        scheme=''.join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k,v in colors.items())
        font=''.join(f'<a:{tag}><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:{tag}>' for tag in ['majorFont','minorFont'])
        fill='<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        lines=''.join(f'<a:ln w="{w}">{fill}<a:prstDash val="solid"/><a:miter lim="800000"/></a:ln>' for w in [9525,25400,38100])
        fmt='<a:fmtScheme name="Simple"><a:fillStyleLst>'+fill*3+'</a:fillStyleLst><a:lnStyleLst>'+lines+'</a:lnStyleLst><a:effectStyleLst>'+('<a:effectStyle><a:effectLst/></a:effectStyle>'*3)+'</a:effectStyleLst><a:bgFillStyleLst>'+fill*3+'</a:bgFillStyleLst></a:fmtScheme>'
        z.writestr('ppt/theme/theme1.xml',HEADER+f'<a:theme xmlns:a="{A}" name="Encoder study"><a:themeElements><a:clrScheme name="Encoder study">{scheme}</a:clrScheme><a:fontScheme name="Arial">{font}</a:fontScheme>{fmt}</a:themeElements></a:theme>')
        for i,s in enumerate(slides,1):
            img=f'../media/slide{i}.png'
            picture='<p:pic><p:nvPicPr><p:cNvPr id="2" name="'+escape(s['title'],{'"':'&quot;'})+'"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="14630400" cy="8229600"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
            z.writestr(f'ppt/slides/slide{i}.xml',xml('sld','<p:cSld><p:spTree>'+group()+picture+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>'))
            z.writestr(f'ppt/slides/_rels/slide{i}.xml.rels',rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image',img),('rId3','notesSlide',f'../notesSlides/notesSlide{i}.xml')]))
            text=escape(s['notes'])
            shape='<p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes"/><p:cNvSpPr txBox="1"/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>'+text+'</a:t></a:r></a:p></p:txBody></p:sp>'
            z.writestr(f'ppt/notesSlides/notesSlide{i}.xml',xml('notes','<p:cSld><p:spTree>'+group()+shape+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>'))
            z.writestr(f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels',rels([('rId1','notesMaster','../notesMasters/notesMaster1.xml'),('rId2','slide',f'../slides/slide{i}.xml')]))
            z.write(root/'slides'/f'{i:02d}.png',f'ppt/media/slide{i}.png')
    with ZipFile(output) as z:
        assert z.testzip() is None
        for p in z.namelist():
            if p.endswith(('.xml','.rels')):ET.fromstring(z.read(p))
        assert len([p for p in z.namelist() if p.startswith('ppt/slides/slide') and p.endswith('.xml')])==n
        import posixpath
        for p in z.namelist():
            if not p.endswith('.rels'):continue
            folder='' if p=='_rels/.rels' else posixpath.dirname(posixpath.dirname(p))
            for relation in ET.fromstring(z.read(p)):
                target=posixpath.normpath(posixpath.join(folder,relation.attrib['Target']))
                assert target in z.namelist(),(p,target)
    return output
