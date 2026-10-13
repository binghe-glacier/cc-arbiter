import unittest
import numpy as np
from scipy.stats import binom
from cc_arbiter import ConformalSets, ConditionalRiskGate, finite_quantile, binomial_upper, proposals

class CalibrationTests(unittest.TestCase):
    def test_quantile_rank_and_small_class(self):
        self.assertTrue(np.isinf(finite_quantile([], .1)))
        self.assertTrue(np.isinf(finite_quantile([.2]*8,.1)))
        self.assertAlmostEqual(finite_quantile(range(9),.1),8)
        for alpha in [0,1,-.1]:
            with self.assertRaises(ValueError): finite_quantile([1],alpha)
    def test_exact_upper_tail(self):
        # Independent analytic checks, including zero-error closed form.
        self.assertAlmostEqual(binomial_upper(0,100,.01),1-.01**.01,places=12)
        q=binomial_upper(4,100,.005)
        self.assertAlmostEqual(binom.cdf(4,100,q),.005,places=12)
        self.assertEqual(binomial_upper(0,0,.01),1.)
    def test_reject_all_not_zero_threshold(self):
        y=np.zeros((20,1),int); pred=np.ones_like(y); eligible=np.ones_like(y,bool); score=np.zeros_like(y,float)
        gate=ConditionalRiskGate().fit(pred,eligible,score,y)
        self.assertIsNone(gate.select(.1)[0]['threshold'])
        self.assertFalse(gate.accept(eligible,score,.1).any())
    def test_finite_family_selection_and_serialization(self):
        y=np.zeros((1000,1),int); e=np.ones_like(y,bool); s=np.linspace(0,.5,1000)[:,None]
        gate=ConditionalRiskGate().fit(y,e,s,y)
        self.assertEqual(gate.select(.1)[0]['threshold'],.5)
        clone=ConditionalRiskGate.from_state(gate.state())
        np.testing.assert_array_equal(clone.accept(e,s),gate.accept(e,s))
    def test_intersection_conflict_and_broad_sets(self):
        p=np.array([[[.2],[.8]],[[.1],[.2]],[[.7],[.8]]])
        sets=np.array([[[[1,0]],[[0,1]]],[[[1,0]],[[1,1]]],[[[1,1]],[[1,1]]]],bool)
        pred,eligible,s=proposals(p,sets,'intersection')
        self.assertEqual(eligible[:,0].tolist(),[False,True,False]); self.assertEqual(pred[1,0],0)
    def test_missing_class_and_bad_inputs(self):
        p=np.full((100,2,1),.05); y=np.zeros((100,1),int)
        cp=ConformalSets().fit(p,y)
        self.assertTrue(np.isinf(cp.q[:,:,1]).all())
        self.assertTrue(cp.predict(p)[:,:,:,1].all())
        with self.assertRaises(ValueError): cp.predict(np.full((1,2,1),np.nan))
        with self.assertRaises(ValueError): ConformalSets().fit(p,np.full((100,1),2))
    def test_threshold_ties_kept_together(self):
        y=np.zeros((1000,1),int); e=np.ones_like(y,bool); s=np.full((1000,1),.1)
        gate=ConditionalRiskGate().fit(y,e,s,y)
        self.assertEqual(gate.n[0,1],0); self.assertEqual(gate.n[0,2],1000)
if __name__=='__main__': unittest.main(verbosity=2)
