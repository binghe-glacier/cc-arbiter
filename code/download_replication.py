"""Fetch the two fixed, public NIH replication shards; never select by outcome."""
from pathlib import Path
import concurrent.futures, hashlib, json, time
import requests
ROOT=Path(__file__).resolve().parents[1]
COMMIT='efdd60f009509e6309d306f3681ac863b0c2fabb'
REPO='BahaaEldin0/NIH-Chest-Xray-14'

def fetch(i):
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{COMMIT}/default/test/{i:04d}.parquet'
    dest=ROOT/f'raw/nih_replication/test-{i:04d}.parquet'
    dest.parent.mkdir(parents=True,exist_ok=True)
    log=[]
    for attempt in range(5):
        part=dest.with_suffix('.part'); size=part.stat().st_size if part.exists() else 0
        try:
            import pyarrow.parquet as pq
            # A previous complete transfer can remain as .part after a Windows rename failure.
            existing=dest if dest.exists() else part
            if existing.exists():
                try:
                    with pq.ParquetFile(existing) as table: rows=table.metadata.num_rows
                    digest=hashlib.sha256(existing.read_bytes()).hexdigest()
                    if existing!=dest:existing.replace(dest)
                    return {'shard':i,'url':url,'conversion_commit':COMMIT,'source_main_commit':'932bcdba9d7d9590704d4f20bc70fc2c3a1bbad7','file':str(dest.relative_to(ROOT)),'bytes':dest.stat().st_size,'sha256':digest,'rows':rows,'failed_attempts':log,'recovered_complete_transfer':True}
                except (OSError,ValueError):pass
            headers={'Range':f'bytes={size}-'} if size else {}
            with requests.get(url,stream=True,headers=headers,timeout=(25,45)) as r:
                r.raise_for_status()
                if r.status_code==206 and not r.headers.get('Content-Range','').startswith(f'bytes {size}-'):
                    raise RuntimeError('Unexpected range response')
                mode='ab' if size and r.status_code==206 else 'wb'
                with part.open(mode) as f:
                    last=time.monotonic()
                    for chunk in r.iter_content(2**20):
                        f.write(chunk)
                        if time.monotonic()-last>20:
                            print('download',i,round(f.tell()/2**20),'MiB',flush=True);last=time.monotonic()
            with pq.ParquetFile(part) as table: rows=table.metadata.num_rows
            digest=hashlib.sha256(part.read_bytes()).hexdigest()
            part.replace(dest)
            return {'shard':i,'url':url,'conversion_commit':COMMIT,'source_main_commit':'932bcdba9d7d9590704d4f20bc70fc2c3a1bbad7','file':str(dest.relative_to(ROOT)),'bytes':dest.stat().st_size,'sha256':digest,'rows':rows,'failed_attempts':log}
        except Exception as e:
            log.append({'attempt':attempt+1,'error':type(e).__name__})
            print('retry',i,attempt+1,type(e).__name__,flush=True)
            time.sleep(2)
    return {'shard':i,'url':url,'failed_attempts':log,'status':'unavailable'}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as pool: result=list(pool.map(fetch,[0,1]))
    (ROOT/'evidence/replication_download.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
