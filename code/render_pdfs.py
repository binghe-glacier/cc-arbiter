from pathlib import Path
import json
import fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
out={}
for name in ['main','supplement']:
    doc=fitz.open(ROOT/f'output/pdf/{name}.pdf'); info=[]; thumbs=[]
    for i,page in enumerate(doc):
        png=ROOT/f'tmp/pdfs/{name}_{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(png)
        image=Image.open(png).convert('RGB');image.thumbnail((425,601));thumbs.append(image.copy())
        text=page.get_text(); outside=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for s in line['spans']:
                    x0,y0,x1,y1=s['bbox']
                    if x0<12 or x1>page.rect.width-12 or y0<10 or y1>page.rect.height-10:outside.append(s['text'])
        info.append({'page':i+1,'chars':len(text),'unresolved_reference':'??' in text,'outside_page_margin':outside})
    for start in range(0,len(thumbs),6):
        canvas=Image.new('RGB',(3*450,2*635),'#DEE4E9');draw=ImageDraw.Draw(canvas)
        for k,img in enumerate(thumbs[start:start+6]):
            x=(k%3)*450+12;y=(k//3)*635+25;canvas.paste(img,(x,y));draw.text((x,y-18),f'{name} page {start+k+1}',fill='black')
        canvas.save(ROOT/f'tmp/pdfs/{name}_contact_{start//6+1}.png')
    out[name]={'pages':len(doc),'details':info}
(ROOT/'evidence/pdf_layout_check.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
