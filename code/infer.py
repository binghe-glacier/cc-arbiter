"""Fresh two-expert inference; no outcome selection or fitting."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, io, json, time
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch, torchxrayvision as xrv
ROOT=Path(__file__).resolve().parents[1]
MODELS=['densenet121-res224-chex','densenet121-res224-mimic_ch']
def main(data_root=None):
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    manifest=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8')); labels=manifest['labels']
    if data_root is not None:
        from pathlib import PureWindowsPath
        for ds in ['nih','iu']:
            for unit in manifest[ds]:
                for im in unit['images']:
                    parts=PureWindowsPath(im['source']).parts
                    sub='chestxray14' if ds=='nih' else 'iuxray'
                    start=parts.index(sub)
                    im['source']=str(data_root.joinpath(*parts[start:]))
    models=[xrv.models.DenseNet(weights=m,cache_dir=str(ROOT/'weights')).eval().cuda() for m in MODELS]
    indices=[[m.pathologies.index(k) for k in labels] for m in models]
    meta={'models':[],'torch':torch.__version__,'torchxrayvision':getattr(xrv,'__version__','unknown'),
          'device':torch.cuda.get_device_name(),'preprocessing':'PIL grayscale; normalize(maxval=255); XRayCenterCrop; XRayResizer(224,skimage); float32',
          'operating_point_normalization':True,'label_order':labels,'precision':'FP32; TF32 disabled; deterministic algorithms'}
    for name,m in zip(MODELS,models):
        fp=Path(m.weights_filename_local)
        meta['models'].append({'name':name,'url':xrv.models.model_urls[name]['weights_url'], 'sha256':hashlib.sha256(fp.read_bytes()).hexdigest(), 'bytes':fp.stat().st_size})
    (ROOT/'evidence/inference_environment.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    crop=xrv.datasets.XRayCenterCrop(); resize=xrv.datasets.XRayResizer(224,engine='skimage')
    all_records=[]; start=time.perf_counter(); timing={}; failures=[]
    for dataset in ['nih','iu']:
        source_cache={}
        if dataset=='nih':
            for source in sorted({im['source'] for u in manifest[dataset] for im in u['images']}):
                tbl=pq.read_table(source,columns=['image']); source_cache[source]=tbl['image'].to_pylist()
        jobs=[(u,im) for u in manifest[dataset] for im in u['images']]
        def load(job):
            u,im=job
            try:
                raw=source_cache[im['source']][im['row']]['bytes'] if 'row' in im else Path(im['source']).read_bytes()
                img=np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
                arr=resize(crop(xrv.datasets.normalize(img,255)[None,:,:]))
                return job,arr,None
            except Exception as e: return job,None,type(e).__name__+': '+str(e)
        t=time.perf_counter(); batch=[]; data=[]; predictions=[]
        def flush():
            if not data: return
            tensor=torch.as_tensor(np.stack(data),device='cuda',dtype=torch.float32)
            with torch.inference_mode():
                p=np.stack([m(tensor)[:,idx].cpu().numpy() for m,idx in zip(models,indices)],axis=1)
            if not np.isfinite(p).all(): raise RuntimeError('nonfinite expert outputs')
            for (u,im),pred in zip(batch,p):
                predictions.append({'dataset':dataset,'id':u['id'],'image_key':im['key'],'scores':pred.tolist()})
            batch.clear(); data.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for j,(job,arr,err) in enumerate(pool.map(load,jobs)):
                if err: failures.append({'dataset':dataset,'id':job[0]['id'],'image':job[1]['key'],'error':err}); continue
                batch.append(job); data.append(arr)
                if len(batch)==32: flush()
                if (j+1)%500==0: print(dataset,j+1,'/',len(jobs),'seconds',round(time.perf_counter()-t,1),flush=True)
            flush()
        lookup={}
        for r in predictions: lookup.setdefault(r['id'],[]).append(r['scores'])
        # A failed view invalidates its whole unit; never silently change aggregation.
        failed={x['id'] for x in failures if x['dataset']==dataset}
        for u in manifest[dataset]:
            if u['id'] in failed: continue
            p=np.max(np.asarray(lookup[u['id']]),axis=0)
            all_records.append({'dataset':dataset,'id':u['id'],'split':u['split'],'y':u['y'],'scores':p.tolist(),'n_images':len(u['images'])})
        with (ROOT/f'data/{dataset}_image_scores.jsonl').open('w',encoding='utf8') as f:
            for r in predictions: f.write(json.dumps(r)+'\n')
        timing[dataset]={'images':len(jobs),'units':len(lookup)-len(failed),'seconds_including_preprocessing':time.perf_counter()-t}
    with (ROOT/'data/unit_scores.jsonl').open('w',encoding='utf8') as f:
        for r in all_records: f.write(json.dumps(r)+'\n')
    (ROOT/'evidence/inference_run.json').write_text(json.dumps({'timing':timing,'failures':failures,'total_seconds':time.perf_counter()-start},indent=2),encoding='utf8')
    print(json.dumps(timing,indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path);main(ap.parse_args().data_root)
