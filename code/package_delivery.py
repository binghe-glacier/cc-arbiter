"""Package explicit deliverables; omit raw images, weights, caches and system fonts."""
from pathlib import Path
import hashlib,json,zipfile
import fitz,markdown
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    body=markdown.markdown((ROOT/'作者必读_修订与投稿建议.md').read_text(encoding='utf8'),extensions=['tables','fenced_code'])
    css='body{font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.75;color:#233746;max-width:1050px;margin:40px auto;padding:0 30px}h1,h2{color:#28639a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #bcc9d2;padding:8px}code{overflow-wrap:anywhere}a{color:#28639a}'
    (ROOT/'START_HERE.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CC-Arbiter 加强版</title><style>'+css+'</style>'+body+'</html>',encoding='utf8')
    gallery=fitz.open()
    stems=['fig1_architecture','fig2_coverage','fig3_selective_performance','fig4_risk_certificates','fig5_positive_retention','fig6_correlated_failures','fig7_discrimination','fig8_matched_replication','fig9_positive_review']
    for i,stem in enumerate(stems,1):
        with fitz.open(ROOT/f'paper/figures/{stem}.pdf') as src:
            page=gallery.new_page(width=842,height=595)
            page.insert_text((36,32),f'CC-Arbiter | Figure {i}',fontname='hebo',fontsize=13,color=(.14,.22,.28))
            page.show_pdf_page(fitz.Rect(32,48,810,555),src,0,keep_proportion=True)
            page.insert_text((36,577),'Vector original; generated from fixed primary and secondary analyses.',fontsize=8,color=(.4,.46,.5))
    gallery.save(ROOT/'output/pdf/figures.pdf',deflate=True)
    files=[]
    files.extend(ROOT.glob('*.md'));files.extend(ROOT.glob('requirements*.txt'));files.append(ROOT/'START_HERE.html');files.append(ROOT/'submission_fields.txt');files.append(ROOT/'LICENSE')
    files.extend((ROOT/'code').glob('*.py'))
    for name in ['data','results']:
        for ext in ['*.json','*.jsonl','*.csv']:files.extend((ROOT/name).glob(ext))
    files.extend((ROOT/'paper').glob('*.tex'));files.extend((ROOT/'paper/figures').glob('*'))
    files.extend(ROOT/'output/pdf'/x for x in ['main.pdf','main_review.pdf','supplement.pdf','figures.pdf'])
    for ext in ['*.json','*.txt','*.log']:files.extend((ROOT/'evidence').glob(ext))
    files=sorted(set(files),key=lambda p:p.relative_to(ROOT).as_posix())
    assert all(p.is_file() for p in files)
    assert not any(p.suffix in ['.parquet','.ttf','.pt','.tgz','.jpg'] or p.name=='.env' for p in files)
    manifest={'version':3,'status':'Revision of 2026-10-12, update of 2026-10-13: journal formatting, declarations, cohort/subgroup tables, figure fonts and pdfLaTeX compatibility applied. Ethics approval not required (confirmed by supervisor). Code under the MIT license; public archive at https://github.com/binghe-glacier/cc-arbiter; archived at Zenodo under DOI 10.5281/zenodo.23329663. Reference list expanded to 31 citations, each checked against publisher records; generative-AI disclosure covers reference-list expansion. Final author approval pending. Not submitted.',
              'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
    (ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    archive=ROOT/'CC-Arbiter_v3_revision_20261012.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in files+[ROOT/'MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in manifest['files']:assert hashlib.sha256(z.read(p['path'])).hexdigest()==p['sha256']
    print(json.dumps({'archive':str(archive),'files':len(files)+1,'bytes':archive.stat().st_size,'sha256':sha(archive)},indent=2))

if __name__=='__main__':main()
