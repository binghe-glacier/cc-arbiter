"""Unseen patient construction and fixed-score inference for review routing."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,io,json,os,time
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,o):(ROOT/name).write_text(json.dumps(o,indent=2),encoding='utf8')

def main(weights):
    lock=json.loads((ROOT/'evidence/review_routing_lock.json').read_text())
    for name,h in lock['sha256'].items():assert sha(ROOT/name)==h,name
    orig=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8'))
    old=orig['nih']+json.loads((ROOT/'data/nih_replication_cohort.json').read_text())['units']
    oldids={x['id'] for x in old};oldhash={i['key'] for x in old for i in x['images']}
    downloads=json.loads((ROOT/'evidence/review_download.json').read_text());assert [d['shard'] for d in downloads]==[2,3,4,5]
    units={};cache={};rows=0;seen=set();overlap=set();excluded=0;hashdrops=0
    for d in downloads:
        fp=ROOT/d['file'];assert sha(fp)==d['sha256']
        table=pq.read_table(fp).to_pylist();cache[str(fp)]=table
        for i,r in enumerate(table):
            rows+=1;key='pid'+str(r['Patient ID']);seen.add(key)
            if key in oldids:overlap.add(key);excluded+=1;continue
            h=hashlib.sha256(r['image']['bytes']).hexdigest()
            if h in oldhash:hashdrops+=1;continue
            u={'id':key,'images':[{'source':str(fp.relative_to(ROOT)),'row':i,'key':h}],
               'y':[int(l in r['label']) for l in orig['labels']]}
            if key not in units or h<units[key]['images'][0]['key']:units[key]=u
    hashes=set();duplicate=[]
    for key,u in sorted(units.items()):
        h=u['images'][0]['key']
        if h in hashes:duplicate.append(key)
        else:hashes.add(h)
    for key in duplicate:del units[key]
    ids=sorted(units);np.random.default_rng(20261011).shuffle(ids)
    for i,key in enumerate(ids):units[key]['split']='risk' if i<len(ids)//2 else 'test'
    selected=[units[k] for k in sorted(units)]
    save('data/review_cohort.json',{'labels':orig['labels'],'units':selected})
    audit={'source_rows':rows,'source_patients':len(seen),'prior_patient_ids':len(oldids),'overlapping_patients_excluded':len(overlap),
           'overlapping_rows_excluded':excluded,'prior_hash_rows_excluded':hashdrops,'cross_patient_duplicates_removed':len(duplicate),
           'selected_patients':len(selected),'split_counts':{s:sum(u['split']==s for u in selected) for s in ['risk','test']},
           'patient_overlap_remaining':len(oldids&set(units)),'manifest_sha256':sha(ROOT/'data/review_cohort.json'),
           'locked_before_inference_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    assert audit['patient_overlap_remaining']==0
    save('evidence/review_cohort_audit.json',audit);print(json.dumps(audit,indent=2),flush=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch,torchxrayvision as xrv
    from infer import MODELS
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    models=[xrv.models.DenseNet(weights=m,cache_dir=str(weights)).eval().cuda() for m in MODELS]
    env=json.loads((ROOT/'evidence/inference_environment.json').read_text())
    for m,d in zip(models,env['models']):assert sha(Path(m.weights_filename_local))==d['sha256']
    idx=[[m.pathologies.index(l) for l in orig['labels']] for m in models]
    crop=xrv.datasets.XRayCenterCrop();resize=xrv.datasets.XRayResizer(224,engine='skimage')
    def load(u):
        im=u['images'][0];raw=cache[str(ROOT/im['source'])][im['row']]['image']['bytes']
        img=np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
        return u,resize(crop(xrv.datasets.normalize(img,255)[None,:,:]))
    batch=[];arrays=[];records=[];start=time.perf_counter()
    def flush():
        if not arrays:return
        tensor=torch.as_tensor(np.stack(arrays),device='cuda',dtype=torch.float32)
        with torch.inference_mode():p=np.stack([m(tensor)[:,j].cpu().numpy() for m,j in zip(models,idx)],axis=1)
        assert np.isfinite(p).all()
        for u,s in zip(batch,p):records.append({'id':u['id'],'dataset':'nih_review','split':u['split'],'y':u['y'],'scores':s.tolist(),'n_images':1})
        batch.clear();arrays.clear()
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for k,(u,arr) in enumerate(pool.map(load,selected)):
            batch.append(u);arrays.append(arr)
            if len(batch)==32:flush()
            if (k+1)%500==0:print('review inference',k+1,round(time.perf_counter()-start,1),flush=True)
        flush()
    with (ROOT/'data/review_scores.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    save('evidence/review_inference.json',{'units':len(records),'seconds':time.perf_counter()-start,'failures':[],
          'weights_match_original':True,'precision':'FP32; TF32 disabled; deterministic algorithms','score_sha256':sha(ROOT/'data/review_scores.jsonl')})
    print('review inference completed',len(records),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,default=ROOT/'weights');main(ap.parse_args().weights)
