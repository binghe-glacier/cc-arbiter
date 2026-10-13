"""Execute locked protocol. No test outcomes enter calibration or policy fitting."""
from pathlib import Path
import hashlib, json, time
import numpy as np
from scipy.stats import beta
from cc_arbiter import ConformalSets, ConditionalRiskGate, proposals
ROOT=Path(__file__).resolve().parents[1]
LABELS=json.loads((ROOT/'data/cohorts.json').read_text(encoding='utf8'))['labels']
def clean(obj):
    if isinstance(obj,dict): return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean(x) for x in obj]
    if isinstance(obj,np.ndarray): return clean(obj.tolist())
    if isinstance(obj,(float,np.floating)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.integer): return int(obj)
    return obj
def save(path,obj): path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False),encoding='utf8')
def ci_binomial(k,n):
    if n==0: return [None,None]
    return [0. if k==0 else beta.ppf(.025,k,n-k+1),1. if k==n else beta.ppf(.975,k+1,n-k)]
def div(a,b): return float(a/b) if b else None
def metrics(pred,keep,y,boot=None):
    n,j=y.shape; wrong=keep & (pred!=y); tp=keep & (pred==1)&(y==1); tn=keep&(pred==0)&(y==0)
    accepted=int(keep.sum()); errors=int(wrong.sum()); total=y.size
    out={'units':n,'opportunities':total,'accepted':accepted,'errors':errors,'release':accepted/total,'selective_risk':div(errors,accepted),
         'erroneous_release':errors/total,'correct_release':float((keep&~wrong).sum()/total),
         'positive_targets':int(y.sum()),'negative_targets':int((y==0).sum()),'true_positive_released':int(tp.sum()),
         'true_negative_released':int(tn.sum()),'positive_target_release':div((keep&(y==1)).sum(),y.sum()),
         'positive_recall_released':div(tp.sum(),y.sum()),'positive_predictive_value':div(tp.sum(),(keep&(pred==1)).sum()),
         'per_label':{}}
    for k,label in enumerate(LABELS):
        ac=int(keep[:,k].sum()); er=int(wrong[:,k].sum()); pos=int(y[:,k].sum())
        out['per_label'][label]={'n':n,'positive_targets':pos,'accepted':ac,'errors':er,'release':ac/n,'selective_risk':div(er,ac),
                  'risk_ci':ci_binomial(er,ac),'true_positive_released':int(tp[:,k].sum()),'positive_recall_released':div(tp[:,k].sum(),pos),
                  'accepted_predicted_positive':int((keep[:,k]&(pred[:,k]==1)).sum())}
    if boot is not None:
        a=keep.sum(axis=1)[boot].sum(axis=1); e=wrong.sum(axis=1)[boot].sum(axis=1)
        p=tp.sum(axis=1)[boot].sum(axis=1); yy=y.sum(axis=1)[boot].sum(axis=1)
        def percentile(num,den):
            v=np.divide(num,den,out=np.full(len(num),np.nan),where=np.asarray(den)>0)
            return np.nanquantile(v,[.025,.975]).tolist() if np.isfinite(v).any() else [None,None]
        out['ci']={'release':percentile(a,np.full(len(a),total)),'selective_risk':percentile(e,a),
                   'erroneous_release':percentile(e,np.full(len(e),total)), 'positive_recall_released':percentile(p,yy)}
    return out
def policies(p,cp,gates,target):
    sets=cp.predict(p); avg=p.mean(axis=1); zeros=np.zeros(avg.shape,int); allkeep=np.ones(avg.shape,bool)
    pred=(p>=.5).astype(int)
    result={'Always negative':(zeros,allkeep),'CheXpert expert':(pred[:,0],allkeep),'MIMIC expert':(pred[:,1],allkeep),
            'Mean score':((avg>=.5).astype(int),allkeep),'Expert agreement':(pred[:,0],pred[:,0]==pred[:,1])}
    pm,em,sm=proposals(p,sets,'mean'); pc,ec,sc=proposals(p,sets,'intersection')
    result['CP intersection']=(pc,ec)
    result['Risk gate (mean)']=(pm,gates['mean'].accept(em,sm,target))
    result['CC-Arbiter']=(pc,gates['intersection'].accept(ec,sc,target))
    return result
def coverage(p,y,cp):
    sets=cp.predict(p); out={}
    for e,name in enumerate(['CheXpert expert','MIMIC expert']):
        covered=np.take_along_axis(sets[:,e],y[:,:,None],axis=2)[:,:,0]; size=sets[:,e].sum(axis=2)
        out[name]={'coverage':covered.mean(),'mean_size':size.mean(),'singleton':(size==1).mean(),'empty':(size==0).mean(),'per_label':{}}
        for j,label in enumerate(LABELS):
            st={'coverage':covered[:,j].mean(),'mean_size':size[:,j].mean(),'singleton':(size[:,j]==1).mean()}
            for value,key in [(0,'negative'),(1,'positive')]:
                mask=y[:,j]==value; n=int(mask.sum()); k=int(covered[mask,j].sum())
                st[key]={'n':n,'covered':k,'coverage':div(k,n),'ci':ci_binomial(k,n),'n_cal':int(cp.counts[e,j,value]),'q':cp.q[e,j,value]}
            out[name]['per_label'][label]=st
    return out
def main():
    locked=json.loads((ROOT/'evidence/protocol_lock.json').read_text())
    assert locked['manifest_sha256']==hashlib.sha256((ROOT/'data/cohorts.json').read_bytes()).hexdigest()
    assert locked['protocol_sha256']==hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()
    records=[json.loads(l) for l in (ROOT/'data/unit_scores.jsonl').read_text().splitlines()]
    results={'protocol':locked,'labels':LABELS,'primary_target':.1,'datasets':{}}
    for ds in ['nih','iu']:
        rows={s:[r for r in records if r['dataset']==ds and r['split']==s] for s in ['conformal','risk','test']}
        groups=[{r['id'] for r in rows[s]} for s in rows]
        assert not any(groups[a]&groups[b] for a in range(3) for b in range(a+1,3))
        p={s:np.array([r['scores'] for r in rr]) for s,rr in rows.items()}; y={s:np.array([r['y'] for r in rr]) for s,rr in rows.items()}
        cp=ConformalSets().fit(p['conformal'],y['conformal'])
        sets=cp.predict(p['risk']); gates={}
        for mode in ['mean','intersection']:
            pred,eligible,score=proposals(p['risk'],sets,mode)
            gates[mode]=ConditionalRiskGate().fit(pred,eligible,score,y['risk'])
        save(ROOT/f'results/{ds}_calibration.json',{'labels':LABELS,'alpha':cp.alpha,'q':cp.q,'counts':cp.counts,'gates':{k:g.state() for k,g in gates.items()}})
        n=len(y['test']); boot=np.random.default_rng(718).integers(0,n,size=(2000,n))
        res={'split_units':{s:len(r) for s,r in rows.items()},'coverage':coverage(p['test'],y['test'],cp),'targets':{},'stress':[]}
        for target in [.05,.1,.2]:
            pol=policies(p['test'],cp,gates,target)
            res['targets'][str(target)]={'policies':{name:metrics(pred,keep,y['test'],boot) for name,(pred,keep) in pol.items()},
                    'certificates':{mode:dict(zip(LABELS,g.select(target))) for mode,g in gates.items()}}
        # Shared masks, no filtering by model correctness, calibration stays fixed.
        u=np.random.default_rng(1309).random(n)
        for mode in ['single_expert','correlated_experts']:
            for rate in [.1,.25,.5]:
                corrupt=p['test'].copy(); mask=u<rate
                if mode=='single_expert': corrupt[mask,1]=1-corrupt[mask,1]
                else: corrupt[mask]=1-corrupt[mask]
                pol=policies(corrupt,cp,gates,.1)
                res['stress'].append({'mode':mode,'requested_rate':rate,'corrupted_units':int(mask.sum()),'units':n,
                                      'policies':{name:metrics(pred,keep,y['test']) for name,(pred,keep) in pol.items()}})
        # Paired error/release differences for the chief added component.
        pol=policies(p['test'],cp,gates,.1)
        pc,kc=pol['CC-Arbiter']; pm,km=pol['Risk gate (mean)']; yy=y['test']
        d=((kc&(pc!=yy)).sum(1)-(km&(pm!=yy)).sum(1))/len(LABELS)
        dr=(kc.sum(1)-km.sum(1))/len(LABELS)
        res['paired_cc_minus_risk_mean']={'erroneous_release_difference':d.mean(),'ci':np.quantile(d[boot].mean(1),[.025,.975]),
                    'release_difference':dr.mean(),'release_ci':np.quantile(dr[boot].mean(1),[.025,.975])}
        results['datasets'][ds]=res
        with (ROOT/f'results/{ds}_test_decisions.jsonl').open('w') as f:
            for i,r in enumerate(rows['test']):
                f.write(json.dumps({'id':r['id'],'y':r['y'],'policies':{k:{'pred':v[0][i].tolist(),'accepted':v[1][i].tolist()} for k,v in pol.items()}})+'\n')
        print('\nDATASET',ds,'test units',n)
        for name,m in res['targets']['0.1']['policies'].items():
            print(name,'release',round(m['release'],4),'risk',m['selective_risk'],'positive recalled',m['positive_recall_released'])
    save(ROOT/'results/results.json',results)
    save(ROOT/'evidence/evaluation_integrity.json',{'no_split_overlap':True,'label_rows_not_treated_as_independent':True,'test_hash':hashlib.sha256((ROOT/'data/unit_scores.jsonl').read_bytes()).hexdigest(),
          'family_size':308,'delta':.05,'test_outcomes_used_for_fitting':False,'source_protocol_verified':True})
if __name__=='__main__': main()
