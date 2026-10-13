"""Select previously unseen NIH patients and run unchanged frozen experts."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,io,json,os,time
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.write_text(json.dumps(o,indent=2),encoding='utf8')

def prepare():
    prior=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8'))
    ids={u['id'] for u in prior['nih']};oldhash={im['key'] for u in prior['nih'] for im in u['images']}
    downloads=json.loads((ROOT/'evidence/replication_download.json').read_text())
    cache={};units={};seenids=set();overlap=set();excluded=0;hashdrop=0;rows=0
    for d in downloads:
        fp=ROOT/d['file'];assert sha(fp)==d['sha256']
        table=pq.read_table(fp).to_pylist();cache[str(fp)]=table
        for i,r in enumerate(table):
            rows+=1;key='pid'+str(r['Patient ID']);seenids.add(key)
            if key in ids:excluded+=1;overlap.add(key);continue
            h=hashlib.sha256(r['image']['bytes']).hexdigest()
            if h in oldhash:hashdrop+=1;continue
            im={'source':str(fp.relative_to(ROOT)),'row':i,'key':h,'original_path':r['image']['path']}
            u={'id':key,'images':[im],'y':[int(l in r['label']) for l in prior['labels']],
               'view':r['View Position'],'age':r['Patient Age'],'sex':r['Patient Gender']}
            if key not in units or h<units[key]['images'][0]['key']:units[key]=u
    seen={};drop=[]
    for key,u in sorted(units.items()):
        h=u['images'][0]['key']
        if h in seen:drop.append(key)
        else:seen[h]=key
    for key in drop:del units[key]
    selected=[units[k] for k in sorted(units)]
    assert not ids.intersection(units)
    manifest={'labels':prior['labels'],'units':selected}
    save(ROOT/'data/nih_replication_cohort.json',manifest)
    audit={'source_rows':rows,'source_unique_patients':len(seenids),'prior_patients':len(ids),
        'overlapping_patients_excluded':len(overlap),'overlapping_patient_rows_excluded':excluded,
        'prior_image_hash_rows_excluded':hashdrop,'cross_patient_duplicate_images_excluded':len(drop),
        'selected_patients':len(selected),'discarded_additional_views':rows-excluded-hashdrop-len(drop)-len(selected),
        'positive_targets':np.array([r['y'] for r in selected]).sum(0).tolist(),'labels':prior['labels'],
        'patient_overlap_after_exclusion':0,'selected_image_overlap_after_exclusion':0,
        'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'cohort_sha256':sha(ROOT/'data/nih_replication_cohort.json'),
        'enhancement_protocol_sha256':sha(ROOT/'ENHANCEMENT_PROTOCOL.md')}
    save(ROOT/'evidence/replication_cohort_audit.json',audit);print(json.dumps(audit,indent=2),flush=True)
    return manifest,cache

def main(weights):
    manifest,cache=prepare()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch,torchxrayvision as xrv
    from infer import MODELS
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    original=json.loads((ROOT/'evidence/inference_environment.json').read_text())
    models=[xrv.models.DenseNet(weights=m,cache_dir=str(weights)).eval().cuda() for m in MODELS]
    for m,meta in zip(models,original['models']):assert sha(Path(m.weights_filename_local))==meta['sha256']
    idx=[[m.pathologies.index(l) for l in manifest['labels']] for m in models]
    crop=xrv.datasets.XRayCenterCrop();resize=xrv.datasets.XRayResizer(224,engine='skimage')
    def load(u):
        im=u['images'][0];raw=cache[str(ROOT/im['source'])][im['row']]['image']['bytes']
        try:
            a=np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
            return u,resize(crop(xrv.datasets.normalize(a,255)[None,:,:])),None
        except Exception as e:return u,None,type(e).__name__
    batch=[];arrays=[];records=[];failures=[];start=time.perf_counter()
    def flush():
        if not arrays:return
        tensor=torch.as_tensor(np.stack(arrays),device='cuda',dtype=torch.float32)
        with torch.inference_mode():p=np.stack([m(tensor)[:,j].cpu().numpy() for m,j in zip(models,idx)],axis=1)
        assert np.isfinite(p).all()
        for u,s in zip(batch,p):records.append({'dataset':'nih_replication','id':u['id'],'split':'test','y':u['y'],'scores':s.tolist(),'n_images':1})
        batch.clear();arrays.clear()
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for k,(u,arr,error) in enumerate(pool.map(load,manifest['units'])):
            if error:failures.append({'id':u['id'],'error':error});continue
            batch.append(u);arrays.append(arr)
            if len(batch)==32:flush()
            if (k+1)%400==0:print('replication inference',k+1,'elapsed',round(time.perf_counter()-start,1),flush=True)
        flush()
    # All retained units must be present; failures require an explicit protocol deviation.
    save(ROOT/'evidence/replication_inference.json',{'units':len(records),'failures':failures,'seconds':time.perf_counter()-start,
         'weights_verified_against_original':True,'precision':'FP32; TF32 disabled; deterministic algorithms','torch':torch.__version__})
    if failures:raise RuntimeError('Replication had image failures; do not silently alter the cohort.')
    with (ROOT/'data/nih_replication_scores.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    print('completed',len(records),round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,default=ROOT/'weights');main(ap.parse_args().weights)
