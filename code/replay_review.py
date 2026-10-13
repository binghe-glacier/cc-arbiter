"""Execute all new routing test patients under two schedulers and both rules."""
from pathlib import Path
import json
from langchain_core.messages import HumanMessage
from replay_graph import Scheduler,ScoreTool
from review_agent import ReviewRoutingAgent
ROOT=Path(__file__).resolve().parents[1]

def main():
    r=json.loads((ROOT/'results/review_results.json').read_text());st=json.loads((ROOT/'results/review_stacker.json').read_text())
    rows={x['id']:x for x in map(json.loads,(ROOT/'data/review_scores.jsonl').read_text().splitlines()) if x['split']=='test'}
    expected={x['id']:x for x in map(json.loads,(ROOT/'results/review_decisions.jsonl').read_text().splitlines())}
    labels=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8'))['labels'];result=[]
    for method in ['stacker','max_mean']:
        for scheduler in ['batched','sequential']:
            agent=ReviewRoutingAgent(Scheduler(scheduler),[ScoreTool('chex',0,rows,labels),ScoreTool('mimic',1,rows,labels)],['chex','mimic'],labels,method,st,r['calibration'][method])
            bad=0;flagged=0
            for key in rows:
                out,_=agent.invoke([HumanMessage(content=key)],method+':'+key,key)
                bad+=out['action']!=('review' if expected[key]['flags'][method] else 'no_flag')
                assert out['diagnosis_value'] is None and not out['human_review_performed'] and not out['no_flag_means_clinical_normality']
                flagged+=out['action']=='review'
            assert bad==0
            result.append({'method':method,'scheduler':scheduler,'patients':len(rows),'flagged':flagged,'mismatches':bad})
    (ROOT/'evidence/review_graph_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
