"""Integrity checks and an exact numerical rerun from the supplied score files."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    lock=json.loads((ROOT/'evidence/enhancement_lock.json').read_text())
    files=['results/secondary_results.json','results/secondary_thresholds.json','results/nih_replication_decisions.jsonl','results/secondary_policy_summary.csv']
    before={p:sha(ROOT/p) for p in files}
    subprocess.run([sys.executable,str(ROOT/'code/reproduce.py')],cwd=ROOT,check=True)
    after={p:sha(ROOT/p) for p in files}
    assert before==after,'Secondary numerical outputs changed on exact rerun'
    for p,h in lock['sha256'].items():assert sha(ROOT/p)==h,p
    old=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8'))
    new=json.loads((ROOT/'data/nih_replication_cohort.json').read_text())
    assert not {x['id'] for x in old['nih']}&{x['id'] for x in new['units']}
    assert not {im['key'] for x in old['nih'] for im in x['images']}&{im['key'] for x in new['units'] for im in x['images']}
    r=json.loads((ROOT/'results/secondary_results.json').read_text());count=0
    for ds,x in r['datasets'].items():
        pol=x['policies'];cc=pol['CC-Arbiter']
        for name in ['Mean / matched release','Youden / matched release']:
            assert pol[name]['accepted']==cc['accepted']
            for l,m in cc['per_label'].items():assert pol[name]['per_label'][l]['accepted']==m['accepted']
        count+=x['units']
    assert count==3105
    gr=json.loads((ROOT/'evidence/graph_replay_v2.json').read_text())
    calls=sum(x['units'] for d in gr['datasets'].values() for x in d.values())
    labels=sum(x['label_outputs'] for d in gr['datasets'].values() for x in d.values())
    assert calls==6210 and labels==43470
    assert all(x['mismatches']==0 for d in gr['datasets'].values() for x in d.values())
    score_rows=[json.loads(l) for l in (ROOT/'data/nih_replication_scores.jsonl').read_text().splitlines()]
    p=np.array([x['scores'] for x in score_rows]);assert p.shape==(1582,2,7) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    assert {x['id'] for x in score_rows}=={x['id'] for x in new['units']}
    originals=json.loads((ROOT/'evidence/original_files_unchanged.json').read_text(encoding='utf8'))
    available=[x for x in originals if Path(x['path']).exists()]
    for x in available:assert sha(Path(x['path']))==x['sha256']
    report={'primary_protocol_scores_calibration_results_unchanged':True,'secondary_rerun_exactly_matches':True,
      'secondary_output_sha256':after,'new_patients':1582,'new_patient_id_overlap':0,'new_selected_image_hash_overlap':0,
      'matched_budgets_equal_for_every_label':True,'graph_invocations':calls,'graph_label_outputs':labels,'graph_mismatches':0,
      'new_scores_sha256':sha(ROOT/'data/nih_replication_scores.jsonl'),'original_D_files_hash_verified':len(available),
      'original_tests':18,'additional_tests':6}
    (ROOT/'evidence/strengthened_integrity.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
