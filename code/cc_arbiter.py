"""Auditable binary evidence gating with finite-family conditional-risk bounds.

Statistical unit: one independent unit for each label. Labels are not treated as
independent observations. Family correction includes datasets, policies, labels
and thresholds. No guarantee covers clinical truth, arbitrary attacks or prose.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta

GRID = np.linspace(0.0, 0.5, 11)
FAMILY_SIZE = 2 * 2 * 7 * len(GRID)

def finite_quantile(scores, alpha):
    if not 0 < alpha < 1: raise ValueError('alpha must be in (0,1)')
    x=np.asarray(scores,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('finite score vector required')
    rank=int(np.ceil((len(x)+1)*(1-alpha)))
    return float('inf') if rank>len(x) else float(np.sort(x)[rank-1])

def validate_scores(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=3 or p.shape[1]!=2 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('expected finite n x 2 x labels scores in [0,1]')
    return p

def validate_targets(y,shape):
    y=np.asarray(y)
    if y.shape!=shape or not np.isin(y,[0,1]).all(): raise ValueError('binary targets with matching shape required')
    return y.astype(int)

@dataclass
class ConformalSets:
    alpha: float = 0.10
    q: object = None
    counts: object = None

    def fit(self,p,y):
        p=validate_scores(p); y=validate_targets(y,(len(p),p.shape[2]))
        if not len(p): raise ValueError('empty calibration set')
        self.q=np.empty((2,p.shape[2],2)); self.counts=np.zeros_like(self.q,dtype=int)
        for e in range(2):
            for j in range(p.shape[2]):
                for c in [0,1]:
                    scores=np.abs(c-p[y[:,j]==c,e,j]); self.counts[e,j,c]=len(scores)
                    self.q[e,j,c]=finite_quantile(scores,self.alpha)
        return self

    def predict(self,p):
        if self.q is None: raise RuntimeError('not calibrated')
        p=validate_scores(p)
        if p.shape[1:]!=self.q.shape[:2]: raise ValueError('label dimensions differ')
        return np.abs(np.arange(2)-p[:,:,:,None])<=self.q[None,:,:,:]

def proposals(p,sets=None,mode='mean'):
    p=validate_scores(p); avg=p.mean(axis=1)
    if mode=='mean':
        pred=(avg>=.5).astype(int); eligible=np.ones_like(pred,dtype=bool)
    elif mode=='intersection':
        if sets is None or np.shape(sets)!=p.shape+(2,): raise ValueError('matching expert sets required')
        common=np.all(sets,axis=1); eligible=common.sum(axis=-1)==1
        pred=common[:,:,1].astype(int)
    else: raise ValueError('unknown proposal mode')
    uncertainty=np.where(pred==1,1-avg,avg)
    return pred,eligible,uncertainty

def binomial_upper(errors,n,tail):
    if not 0 < tail < 1 or not 0 <= errors <= n: raise ValueError('invalid binomial inputs')
    if n==0 or errors==n: return 1.0
    return float(beta.ppf(1-tail,errors+1,n-errors))

@dataclass
class ConditionalRiskGate:
    grid: object = None
    family_size: int = FAMILY_SIZE
    delta: float = .05
    bounds: object = None
    n: object = None
    errors: object = None

    def __post_init__(self):
        self.grid=GRID.copy() if self.grid is None else np.array(self.grid,dtype=float)
        if self.grid.ndim!=1 or not len(self.grid) or not np.isfinite(self.grid).all() or np.any(np.diff(self.grid)<=0):
            raise ValueError('grid must be finite, increasing and fixed before calibration')
        if not 0<self.delta<1 or self.family_size<len(self.grid): raise ValueError('invalid family correction')

    def fit(self,pred,eligible,score,y):
        pred=np.asarray(pred); eligible=np.asarray(eligible,dtype=bool); score=np.asarray(score,dtype=float)
        if pred.ndim!=2 or pred.shape!=score.shape or pred.shape!=eligible.shape or not np.isfinite(score).all(): raise ValueError('invalid policy outputs')
        y=validate_targets(y,pred.shape)
        if not np.isin(pred,[0,1]).all() or not len(pred): raise ValueError('binary predictions on nonempty data required')
        if self.family_size<pred.shape[1]*len(self.grid): raise ValueError('family_size too small for labels x thresholds')
        shape=(pred.shape[1],len(self.grid)); self.bounds=np.ones(shape); self.n=np.zeros(shape,dtype=int); self.errors=np.zeros(shape,dtype=int)
        for j in range(shape[0]):
            for k,t in enumerate(self.grid):
                keep=eligible[:,j] & (score[:,j]<=t); n=int(keep.sum()); err=int(((pred[:,j]!=y[:,j]) & keep).sum())
                self.n[j,k]=n; self.errors[j,k]=err; self.bounds[j,k]=binomial_upper(err,n,self.delta/self.family_size)
        return self

    def select(self,target=.1):
        if self.bounds is None: raise RuntimeError('not calibrated')
        if not 0<target<1: raise ValueError('risk target must be in (0,1)')
        chosen=[]
        for j in range(len(self.bounds)):
            valid=np.flatnonzero((self.bounds[j]<=target)&(self.n[j]>0))
            if len(valid):
                k=int(valid[-1]); chosen.append({'threshold':float(self.grid[k]),'bound':float(self.bounds[j,k]),'n':int(self.n[j,k]),'errors':int(self.errors[j,k])})
            else: chosen.append({'threshold':None,'bound':None,'n':0,'errors':0})
        return chosen

    def accept(self,eligible,score,target=.1):
        eligible=np.asarray(eligible,dtype=bool); score=np.asarray(score,dtype=float)
        chosen=self.select(target)
        if eligible.shape!=score.shape or eligible.ndim!=2 or score.shape[1]!=len(chosen): raise ValueError('label dimensions differ')
        thresholds=np.array([x['threshold'] if x['threshold'] is not None else -np.inf for x in chosen])
        return eligible & np.isfinite(score) & (score<=thresholds)

    def state(self):
        return {'grid':self.grid.tolist(),'family_size':self.family_size,'delta':self.delta,'bounds':self.bounds.tolist(),'n':self.n.tolist(),'errors':self.errors.tolist()}

    @classmethod
    def from_state(cls,obj):
        gate=cls(grid=obj['grid'],family_size=obj['family_size'],delta=obj['delta'])
        for k in ['bounds','n','errors']: setattr(gate,k,np.asarray(obj[k]))
        return gate
