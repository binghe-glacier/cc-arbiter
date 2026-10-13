"""Download the four protocol-fixed public review-routing shards."""
from pathlib import Path
import concurrent.futures,json
from download_replication import fetch
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as pool:records=list(pool.map(fetch,[2,3,4,5]))
    (ROOT/'evidence/review_download.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print(json.dumps(records,indent=2))
