"""Independent single-image reruns and image/preprocessing QA."""
from pathlib import Path
import hashlib,io,json
import numpy as np
from PIL import Image,ImageDraw
import pyarrow.parquet as pq
import torch,torchxrayvision as xrv
ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    manifest=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8')); labels=manifest['labels']
    models=[xrv.models.DenseNet(weights=m,cache_dir=str(ROOT/'weights')).eval().cuda() for m in ['densenet121-res224-chex','densenet121-res224-mimic_ch']]
    rows=[]; imgs=[]; cache={}
    for ds in ['nih','iu']:
        pred={r['image_key']:r for r in map(json.loads,(ROOT/f'data/{ds}_image_scores.jsonl').read_text().splitlines())}
        units=manifest[ds]; chosen=np.random.default_rng(239).choice(len(units),8,replace=False)
        for idx in chosen:
            u=units[idx]; im=u['images'][0]
            if 'row' in im:
                if im['source'] not in cache: cache[im['source']]=pq.read_table(im['source'],columns=['image'])['image'].to_pylist()
                raw=cache[im['source']][im['row']]['bytes']
            else: raw=Path(im['source']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==im['key']
            pil=Image.open(io.BytesIO(raw)); arr=np.asarray(pil.convert('L')); imgs.append((ds+' '+u['id'],pil.convert('L')))
            xx=xrv.datasets.XRayResizer(224)(xrv.datasets.XRayCenterCrop()(xrv.datasets.normalize(arr,255)[None,:,:]))
            with torch.inference_mode():
                pp=np.stack([m(torch.as_tensor(xx[None],device='cuda'))[0,[m.pathologies.index(k) for k in labels]].cpu().numpy() for m in models])
            error=float(np.max(np.abs(pp-np.array(pred[im['key']]['scores']))))
            rows.append({'dataset':ds,'id':u['id'],'mode':pil.mode,'original_size':pil.size,'grayscale_range':[int(arr.min()),int(arr.max())],
                         'normalized_range':[float(xx.min()),float(xx.max())],'max_single_vs_batch_difference':error})
            print(ds,u['id'],'single/batch max abs difference',error,flush=True)
            assert error<1e-4
    canvas=Image.new('RGB',(4*300,4*325),'white'); draw=ImageDraw.Draw(canvas)
    for k,(name,pil) in enumerate(imgs):
        pil.thumbnail((290,290)); x=(k%4)*300+(300-pil.width)//2; y=(k//4)*325+25
        canvas.paste(pil,(x,y));draw.text(((k%4)*300+5,(k//4)*325+5),name,fill='black')
    canvas.save(ROOT/'tmp/pdfs/input_qa.png')
    (ROOT/'evidence/inference_verification.json').write_text(json.dumps({'images_checked':len(rows),'records':rows,'tolerance_review_required':max(r['max_single_vs_batch_difference'] for r in rows)>1e-4},indent=2),encoding='utf8')
    print('Verified',len(rows),'images; max difference',max(r['max_single_vs_batch_difference'] for r in rows))
if __name__=='__main__': main()
