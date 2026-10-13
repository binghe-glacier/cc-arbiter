"""A separate patient-level review flag; never a released positive diagnosis."""
import numpy as np
from scipy.special import expit,logit
from cc_arbiter import binomial_upper,validate_scores
GRID=np.linspace(0,1,41)
FAMILY=2*41*2

def features(p):
    p=validate_scores(p)
    if p.shape[2]!=7:raise ValueError('exactly seven labels required')
    return logit(np.clip(p.reshape(len(p),14),1e-6,1-1e-6))

def fit_stacker(p,z):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import sklearn
    x=features(p);z=np.asarray(z)
    if z.shape!=(len(x),) or not np.isin(z,[0,1]).all() or len(np.unique(z))!=2:raise ValueError('binary two-class fitting target required')
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,random_state=0).fit(scaler.transform(x),z)
    state={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),
           'sklearn':sklearn.__version__,'fitting_units':len(z),'fitting_positives':int(z.sum()),'iterations':model.n_iter_.tolist()}
    np.testing.assert_allclose(score(p,'stacker',state),model.predict_proba(scaler.transform(x))[:,1],atol=1e-12,rtol=0)
    return state

def score(p,method,stacker):
    p=validate_scores(p)
    if method=='max_mean':return p.mean(1).max(1)
    if method=='stacker':
        x=(features(p)-np.array(stacker['mean']))/np.array(stacker['scale'])
        return expit(x@np.array(stacker['coef'])+stacker['intercept'])
    raise ValueError('unknown review score')

def fit_route(s,z):
    s=np.asarray(s);z=np.asarray(z)
    if s.ndim!=1 or s.shape!=z.shape or not np.isfinite(s).all() or not np.isin(z,[0,1]).all() or ((s<0)|(s>1)).any():raise ValueError('invalid routing calibration data')
    candidates=[];npos=int(z.sum())
    for t in GRID:
        low=s<t;n=int(low.sum());miss=int((low&(z==1)).sum())
        candidates.append({'threshold':float(t),'positive_calibration':npos,'missed_positive':miss,'unflagged_calibration':n,
            'miss_fraction_upper':binomial_upper(miss,npos,.05/FAMILY),'unflagged_positive_upper':binomial_upper(miss,n,.05/FAMILY)})
    valid=[r for r in candidates if r['miss_fraction_upper']<=.1 and r['unflagged_positive_upper']<=.1 and r['unflagged_calibration']>0]
    return {'family':FAMILY,'delta':.05,'target':.1,'candidates':candidates,'selected':valid[-1] if valid else None}

def flag(s,gate):
    s=np.asarray(s,dtype=float)
    if s.ndim!=1 or not np.isfinite(s).all() or ((s<0)|(s>1)).any():raise ValueError('finite score vector required')
    return np.ones(len(s),bool) if gate['selected'] is None else s>=gate['selected']['threshold']
