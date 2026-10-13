"""Calibrate once on new routing-risk patients, then evaluate untouched patients."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from evaluate import save,ci_binomial,LABELS
from review_core import fit_stacker,score,fit_route,flag
ROOT=Path(__file__).resolve().parents[1]

def fit():
    old=[json.loads(l) for l in (ROOT/'data/unit_scores.jsonl').read_text().splitlines()]
    train=[r for r in old if r['dataset']=='nih' and r['split']=='conformal']
    p=np.array([r['scores'] for r in train]);y=np.array([r['y'] for r in train]);state=fit_stacker(p,y.any(1).astype(int))
    state['training_id_sha256']=hashlib.sha256('\n'.join(r['id'] for r in train).encode()).hexdigest()
    save(ROOT/'results/review_stacker.json',state)
    return state

def metrics(flags,y,boot):
    z=y.any(1);low=~flags;pos=int(z.sum());n=len(z);tp=int((flags&z).sum());miss=int((low&z).sum());nl=int(low.sum())
    def ci(a,b):
        a=np.asarray(a);b=np.asarray(b);x=np.divide(a,b,out=np.full(a.shape,np.nan,dtype=float),where=b>0)
        return np.nanquantile(x,[.025,.975]).tolist() if np.isfinite(x).any() else [None,None]
    out={'patients':n,'positive_patients':pos,'flagged':int(flags.sum()),'unflagged':nl,'flagged_positive':tp,'unflagged_positive':miss,
       'sensitivity':tp/pos if pos else None,'flag_fraction':float(flags.mean()),'unflagged_fraction':float(low.mean()),
       'unflagged_positive_fraction':miss/nl if nl else None,'exact_sensitivity_ci':ci_binomial(tp,pos),'exact_unflagged_positive_ci':ci_binomial(miss,nl),
       'ci':{'sensitivity':ci((flags&z)[boot].sum(1),z[boot].sum(1)),
             'flag_fraction':ci(flags[boot].sum(1),np.full(len(boot),n)),
             'unflagged_positive_fraction':ci((low&z)[boot].sum(1),low[boot].sum(1))},'per_finding':{}}
    for j,label in enumerate(LABELS):
        m=y[:,j]==1;den=int(m.sum());num=int((m&flags).sum())
        out['per_finding'][label]={'positive':den,'flagged_positive':num,'sensitivity':num/den if den else None,'exact_ci':ci_binomial(num,den),'guaranteed_separately':False}
    return out

def main(fit_only=False):
    locked=json.loads((ROOT/'evidence/review_routing_lock.json').read_text())
    for name,h in locked['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    st=fit()
    if fit_only:print('Fitted stacker on',st['fitting_units'],'original conformal patients;',st['fitting_positives'],'positive');return
    rows=[json.loads(l) for l in (ROOT/'data/review_scores.jsonl').read_text().splitlines()]
    splits={s:[r for r in rows if r['split']==s] for s in ['risk','test']}
    ids={s:{r['id'] for r in rr} for s,rr in splits.items()}
    old=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8'))['nih']+json.loads((ROOT/'data/nih_replication_cohort.json').read_text())['units']
    assert not ids['risk']&ids['test'] and not (ids['risk']|ids['test'])&{r['id'] for r in old}
    p={s:np.array([r['scores'] for r in rr]) for s,rr in splits.items()};y={s:np.array([r['y'] for r in rr]) for s,rr in splits.items()}
    n=len(y['test']);boot=np.random.default_rng(20261012).integers(0,n,(2000,n))
    result={'endpoint':'Patient has >=1 of seven NIH annotations; review flag is not a diagnosis.',
      'protocol':locked,'split_patients':{s:len(rr) for s,rr in splits.items()},'split_positive_patients':{s:int(y[s].any(1).sum()) for s in y},
      'policies':{},'calibration':{}}
    masks={}
    for method in ['stacker','max_mean']:
        gate=fit_route(score(p['risk'],method,st),y['risk'].any(1).astype(int));result['calibration'][method]=gate
        s=score(p['test'],method,st);masks[method]=flag(s,gate);masks[method+'_fixed_0.5']=s>=.5
    masks['always_flag']=np.ones(n,bool)
    for name,mask in masks.items():result['policies'][name]=metrics(mask,y['test'],boot)
    save(ROOT/'results/review_results.json',result)
    with (ROOT/'results/review_decisions.jsonl').open('w') as f:
        for i,r in enumerate(splits['test']):f.write(json.dumps({'id':r['id'],'y':r['y'],'flags':{k:bool(v[i]) for k,v in masks.items()}})+'\n')
    for method in ['stacker','max_mean']:
        print(method,'CALIBRATION',result['calibration'][method]['selected']);print('TEST',json.dumps(result['policies'][method],indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fit-only',action='store_true');main(ap.parse_args().fit_only)
