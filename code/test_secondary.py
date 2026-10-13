"""Independent checks for the new estimators and label-free comparisons."""
import unittest
import numpy as np
from scipy.stats import binom
from secondary_analysis import weighted_discrimination,fit_thresholds,matched_mask,frozen_policy

class SecondaryTests(unittest.TestCase):
    def test_tied_auc_ap_against_pairwise_definition(self):
        y=np.array([0,1,0,1,1]);s=np.array([.1,.6,.6,.6,.9]);w=np.array([[1,2,1,1,1],[2,1,0,3,2]])
        a,p=weighted_discrimination(y,s,w)
        for k,ww in enumerate(w):
            yy=np.repeat(y,ww);ss=np.repeat(s,ww);pos=ss[yy==1];neg=ss[yy==0]
            auc=np.mean((pos[:,None]>neg)+.5*(pos[:,None]==neg))
            ap=sum(np.sum((ss>=v)&(yy==1))/np.sum(ss>=v) for v in pos)/len(pos)
            self.assertAlmostEqual(a[k],auc);self.assertAlmostEqual(p[k],ap)
    def test_missing_bootstrap_class_undefined(self):
        a,p=weighted_discrimination(np.array([0,1]),np.array([.2,.7]),np.array([[2,0]]))
        self.assertTrue(np.isnan(a[0]) and np.isnan(p[0]))
    def test_youden_and_error_fit_have_different_objectives(self):
        s=np.array([.1,.2,.3,.4,.5,.6])[:,None];y=np.array([0,0,0,0,1,0])[:,None]
        t=fit_thresholds(s,y)[0];self.assertEqual(t['youden'],.5);self.assertGreater(t['minimum_error'],.6)
    def test_match_counts_and_permutation_invariance_with_ties(self):
        s=np.array([[.1,.8],[.9,.2],[.1,.4],[.9,.6]]);ids=['p1','p2','p3','p4'];t=np.array([.5,.5]);b=[2,1]
        m=matched_mask(s,t,b,ids);np.testing.assert_array_equal(m.sum(0),b)
        perm=np.array([3,1,0,2]);q=matched_mask(s[perm],t,b,[ids[i] for i in perm])
        np.testing.assert_array_equal(q,m[perm])
    def test_zero_error_sample_formula_is_sharp(self):
        import math
        for r in [.05,.1,.2]:
            n=math.ceil(math.log(.05/308)/math.log(1-r))
            self.assertLessEqual(binom.cdf(0,n,r),.05/308)
            self.assertGreater(binom.cdf(0,n-1,r),.05/308)
    def test_frozen_infinite_quantiles_and_all_defer(self):
        cp,g=frozen_policy('iu');self.assertTrue(np.isinf(cp.q).any())
        self.assertTrue(all(x['threshold'] is None for x in g['intersection'].select(.1)))

if __name__=='__main__':unittest.main()
