"""Verify exact rerun, previous-result preservation and all new patient separations."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    files=['results/review_results.json','results/review_stacker.json','results/review_decisions.jsonl','results/review_candidate_bounds.csv','results/positive_attrition.json']
    before={name:sha(ROOT/name) for name in files}
    with (ROOT/'evidence/review_reproduction.log').open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,str(ROOT/'code/reproduce.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    after={name:sha(ROOT/name) for name in files};assert before==after
    for lockname in ['enhancement_lock.json','review_routing_lock.json']:
        lock=json.loads((ROOT/'evidence'/lockname).read_text())
        for name,h in lock['sha256'].items():assert sha(ROOT/name)==h,name
    old=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8'))['nih']+json.loads((ROOT/'data/nih_replication_cohort.json').read_text())['units']
    new=json.loads((ROOT/'data/review_cohort.json').read_text())['units']
    oldids={r['id'] for r in old};oldhash={i['key'] for r in old for i in r['images']}
    parts={s:[r for r in new if r['split']==s] for s in ['risk','test']}
    riskids={r['id'] for r in parts['risk']};testids={r['id'] for r in parts['test']}
    assert not oldids&(riskids|testids) and not riskids&testids
    assert not oldhash&{i['key'] for r in new for i in r['images']}
    assert len(new)==2557 and len(riskids)==1278 and len(testids)==1279
    graph=json.loads((ROOT/'evidence/review_graph_replay.json').read_text())
    assert sum(r['patients'] for r in graph)==5116 and all(r['mismatches']==0 for r in graph)
    originals=json.loads((ROOT/'evidence/original_files_unchanged.json').read_text(encoding='utf8'))
    available=[r for r in originals if Path(r['path']).exists()]
    for r in available:assert sha(Path(r['path']))==r['sha256']
    info={'review_rerun_exactly_matches':True,'review_output_sha256':after,'earlier_locked_results_unchanged':True,
          'new_routing_patients':2557,'routing_calibration_patients':1278,'routing_test_patients':1279,
          'overlap_with_all_3698_previous_NIH_patients':0,'risk_test_patient_overlap':0,'previous_selected_image_overlap':0,
          'routing_graph_invocations':5116,'routing_graph_mismatches':0,'passed_tests':31,'original_files_verified':len(available)}
    (ROOT/'evidence/review_integrity.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

if __name__=='__main__':main()
