"""Protocol-locked secondary baselines, discrimination and frozen replication.

No fitting routine accepts test outcomes. Batch-matched masks use scores and IDs.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
from cc_arbiter import ConformalSets,ConditionalRiskGate
from evaluate import policies,metrics,save,LABELS
ROOT=Path(__file__).resolve().parents[1]
B=2000

def frozen_policy(ds):
    state=json.loads((ROOT/f'results/{ds}_calibration.json').read_text())
    cp=ConformalSets(state['alpha']);cp.q=np.array(state['q'],dtype=float)
    cp.q=np.where(np.isnan(cp.q),np.inf,cp.q);cp.counts=np.array(state['counts'])
    return cp,{k:ConditionalRiskGate.from_state(v) for k,v in state['gates'].items()}

def fit_thresholds(scores,targets):
    """Use only the supplied calibration set. Exact candidates; high-tie rule."""
    out=[]
    for j in range(scores.shape[1]):
        s=scores[:,j];y=targets[:,j];cand=np.r_[np.unique(s),np.nextafter(s.max(),np.inf)]
        pred=s[:,None]>=cand[None,:];err=(pred!=y[:,None]).sum(0)
        eidx=np.flatnonzero(err==err.min())[-1]
        fallback=not (y.any() and (y==0).any())
        if fallback: yidx=eidx
        else:
            # Integer objective is proportional to sensitivity + specificity - 1.
            objective=(pred[y==1].sum(0)*(y==0).sum() - pred[y==0].sum(0)*(y==1).sum())
            yidx=np.flatnonzero(objective==objective.max())[-1]
        out.append({'label':LABELS[j],'youden':float(cand[yidx]),'minimum_error':float(cand[eidx]),
                    'positive_calibration':int(y.sum()),'negative_calibration':int((y==0).sum()),'missing_class_fallback':fallback})
    return out

def matched_mask(scores,thresholds,budget,ids):
    """Exact label-specific budgets; no target argument or correctness sorting."""
    tie=np.array([hashlib.sha256(x.encode()).hexdigest() for x in ids])
    keep=np.zeros(scores.shape,bool)
    for j,n in enumerate(budget):
        rank=np.lexsort((tie,-np.abs(scores[:,j]-thresholds[j])))
        keep[rank[:int(n)],j]=True
    return keep

def rates(pred,keep,y,boot):
    out=metrics(pred,keep,y,boot)
    out['sensitivity_after_deferral']=out['positive_recall_released']
    out['specificity_after_deferral']=out['true_negative_released']/out['negative_targets'] if out['negative_targets'] else None
    for j,l in enumerate(LABELS):
        q=out['per_label'][l];neg=int((y[:,j]==0).sum())
        q['specificity_after_deferral']=float(((pred[:,j]==0)&keep[:,j]&(y[:,j]==0)).sum()/neg) if neg else None
    return out

def weighted_discrimination(y,s,w):
    """AUROC/AP with score ties grouped; rows of w are unit multiplicities."""
    order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order];ww=w[:,order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.cumsum(ww*yy[None,:],axis=1)[:,ends]
    fp=np.cumsum(ww*(1-yy)[None,:],axis=1)[:,ends]
    pt=np.c_[np.zeros(len(w)),tp[:,:-1]];pf=np.c_[np.zeros(len(w)),fp[:,:-1]]
    pos=tp[:,-1];neg=fp[:,-1];valid=(pos>0)&(neg>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        auc=(((tp+pt)/2)*(fp-pf)).sum(1)/(pos*neg)
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp>0)
        ap=((tp-pt)*precision).sum(1)/pos
    auc[~valid]=np.nan;ap[~valid]=np.nan
    return auc,ap

def discrimination(p,y,boot):
    n=len(y);ws=np.zeros((len(boot),n),dtype=np.int32)
    for b,idx in enumerate(boot):ws[b]=np.bincount(idx,minlength=n)
    out={}
    for name,s in [('CheXpert',p[:,0]),('MIMIC',p[:,1]),('Mean',p.mean(1))]:
        out[name]={}
        for j,label in enumerate(LABELS):
            a,ap=weighted_discrimination(y[:,j],s[:,j],np.ones((1,n),int))
            aa=[];pp=[]
            for start in range(0,len(ws),100):
                q,r=weighted_discrimination(y[:,j],s[:,j],ws[start:start+100]);aa.extend(q);pp.extend(r)
            valid=int(np.isfinite(aa).sum())
            out[name][label]={'positive':int(y[:,j].sum()),'n':n,'prevalence':float(y[:,j].mean()),
                'auroc':a[0],'ap':ap[0],'auroc_ci':np.nanquantile(aa,[.025,.975]) if valid else [None,None],
                'ap_ci':np.nanquantile(pp,[.025,.975]) if valid else [None,None],'valid_bootstrap':valid}
    return out

def paired_difference(pc,kc,pb,kb,y,boot):
    ac=kc.sum(1);ab=kb.sum(1);ec=(kc&(pc!=y)).sum(1);eb=(kb&(pb!=y)).sum(1)
    tc=(kc&(pc==1)&(y==1)).sum(1);tb=(kb&(pb==1)&(y==1)).sum(1)
    def interval(x):
        x=np.asarray(x);return np.nanquantile(x,[.025,.975]).tolist() if np.isfinite(x).any() else [None,None]
    with np.errstate(invalid='ignore',divide='ignore'):
        diff=ec[boot].sum(1)/ac[boot].sum(1)-eb[boot].sum(1)/ab[boot].sum(1)
    return {'cc_minus_comparator_error_count':int(ec.sum()-eb.sum()),
            'conditional_error_difference':float(ec.sum()/ac.sum()-eb.sum()/ab.sum()) if ac.sum() and ab.sum() else None,
            'conditional_error_difference_ci':interval(diff),
            'erroneous_release_difference':float((ec-eb).mean()/y.shape[1]),
            'erroneous_release_difference_ci':interval((ec-eb)[boot].mean(1)/y.shape[1]),
            'true_positive_count_difference':int(tc.sum()-tb.sum()),
            'interval_scope':'paired unit bootstrap conditional on fitted thresholds and original batch selections'}

def main():
    lock=json.loads((ROOT/'evidence/enhancement_lock.json').read_text())
    for p,h in lock['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    records=[json.loads(x) for x in (ROOT/'data/unit_scores.jsonl').read_text().splitlines()]
    fitted={}
    for ds in ['nih','iu']:
        cal=[r for r in records if r['dataset']==ds and r['split']=='conformal']
        fitted[ds]=fit_thresholds(np.array([r['scores'] for r in cal]).mean(1),np.array([r['y'] for r in cal]))
    save(ROOT/'results/secondary_thresholds.json',fitted)
    cohorts={ds:[r for r in records if r['dataset']==ds and r['split']=='test'] for ds in ['nih','iu']}
    fp=ROOT/'data/nih_replication_scores.jsonl'
    if fp.exists():cohorts['nih_replication']=[json.loads(x) for x in fp.read_text().splitlines()]
    results={'analysis_status':'secondary; original test outcomes known before enhancement plan','bootstrap':{'n':B,'seed':20261010,'unit':'patient/report; all seven labels together'},'datasets':{}}
    for cohort,rows in cohorts.items():
        ds='nih' if cohort.startswith('nih') else 'iu';n=len(rows)
        p=np.array([r['scores'] for r in rows]);y=np.array([r['y'] for r in rows]);ids=[r['id'] for r in rows]
        boot=np.random.default_rng(20261010).integers(0,n,(B,n))
        cp,gates=frozen_policy(ds);pol=policies(p,cp,gates,.1);avg=p.mean(1)
        tt=np.array([r['youden'] for r in fitted[ds]]);te=np.array([r['minimum_error'] for r in fitted[ds]])
        pol['Mean / calibration Youden']=((avg>=tt).astype(int),np.ones_like(y,bool))
        pol['Mean / calibration minimum error']=((avg>=te).astype(int),np.ones_like(y,bool))
        pc,kc=pol['CC-Arbiter'];budget=kc.sum(0)
        pol['Mean / matched release']=((avg>=.5).astype(int),matched_mask(avg,np.full(7,.5),budget,ids))
        pol['Youden / matched release']=((avg>=tt).astype(int),matched_mask(avg,tt,budget,ids))
        res={'units':n,'positive_targets':int(y.sum()),'policies':{name:rates(pred,keep,y,boot) for name,(pred,keep) in pol.items()},
             'discrimination':discrimination(p,y,boot),'matched_comparisons':{name:paired_difference(pc,kc,*pol[name],y,boot) for name in ['Mean / matched release','Youden / matched release']}}
        if cohort=='nih_replication':
            res['targets']={str(t):{k:rates(*v,y,boot) for k,v in policies(p,cp,gates,t).items()} for t in [.05,.1,.2]}
            with (ROOT/'results/nih_replication_decisions.jsonl').open('w') as f:
                for i,r in enumerate(rows):f.write(json.dumps({'id':r['id'],'y':r['y'],'policies':{k:{'pred':v[0][i].tolist(),'accepted':v[1][i].tolist()} for k,v in pol.items()}})+'\n')
        results['datasets'][cohort]=res
        print(cohort,'n',n,flush=True)
        for name in ['Mean / calibration Youden','Mean / calibration minimum error','CC-Arbiter','Mean / matched release','Youden / matched release']:
            m=res['policies'][name];print(name,m['accepted'],m['errors'],m['true_positive_released'],flush=True)
    feasibility={}
    for target in [.05,.1,.2]:
        n0=math.ceil(math.log(.05/308)/math.log(1-target));detail={}
        for ds in ['nih','iu']:
            _,gates=frozen_policy(ds);g=gates['intersection'];detail[ds]={}
            for j,label in enumerate(LABELS):
                enough=g.n[j]>=n0;valid=g.bounds[j]<=target
                detail[ds][label]={'max_candidate_n':int(g.n[j].max()),'minimum_candidate_bound':float(g.bounds[j].min()),
                    'count_sufficient_candidates':int(enough.sum()),'certified_candidates':int(valid.sum()),
                    'barrier':'certified' if valid.any() else 'insufficient accepted count even at zero errors' if not enough.any() else 'observed errors prevent certification'}
        feasibility[str(target)]={'zero_error_minimum_n':n0,'per_dataset':detail}
    results['certification_feasibility']=feasibility
    save(ROOT/'results/secondary_results.json',results)

if __name__=='__main__':main()
