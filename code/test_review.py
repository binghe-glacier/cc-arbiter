"""Tests for the distinct routing objective, denominators and final action."""
import unittest
import numpy as np
from scipy.stats import binom
from langchain_core.messages import HumanMessage
from test_integration import ScriptedPlanner,call
from review_core import fit_route,flag,score,FAMILY
from review_agent import ReviewRoutingAgent
LABELS=[f'L{i}' for i in range(7)]

class Tool:
    def __init__(self,name,bad=False):self.name=name;self.bad=bad
    def invoke(self,args):return {'image_key':'wrong' if self.bad else args['image_key'],'scores':{l:.05 for l in LABELS}}

def agent(steps,bad=False):
    return ReviewRoutingAgent(ScriptedPlanner(steps),[Tool('A'),Tool('B',bad)],['A','B'],LABELS,'max_mean',{}, {'selected':{'threshold':.5}})

class ReviewTests(unittest.TestCase):
    def test_both_denominators_and_exact_tail(self):
        z=np.r_[np.ones(200),np.zeros(800)];s=np.r_[np.full(200,.9),np.full(800,.1)]
        g=fit_route(s,z);r=g['selected'];self.assertAlmostEqual(r['threshold'],.9)
        self.assertEqual(r['positive_calibration'],200);self.assertEqual(r['unflagged_calibration'],800)
        for endpoint,n in [('miss_fraction_upper',200),('unflagged_positive_upper',800)]:self.assertAlmostEqual(binom.cdf(0,n,r[endpoint]),.05/FAMILY,places=10)
        np.testing.assert_array_equal(flag(s,g),z.astype(bool))
    def test_class_imbalance_does_not_allow_all_missed_positives(self):
        z=np.r_[np.ones(10),np.zeros(990)];s=np.zeros(1000)
        self.assertIsNone(fit_route(s,z)['selected'])
    def test_missing_positive_calibration_fails_closed(self):
        g=fit_route(np.linspace(0,1,100),np.zeros(100));self.assertIsNone(g['selected'])
        self.assertTrue(flag(np.array([0,.5,1]),g).all())
    def test_exported_stacker_known_logit(self):
        st={'mean':[0]*14,'scale':[1]*14,'coef':[1]+[0]*13,'intercept':0}
        p=np.full((2,2,7),.3);p[1,0,0]=.8
        np.testing.assert_allclose(score(p,'stacker',st),[.3,.8],atol=1e-12)
    def test_final_no_flag_does_not_claim_normality_or_diagnosis(self):
        a=agent([call('A','B')]);out,_=a.invoke([HumanMessage(content='imageA')],'r1','imageA')
        self.assertEqual(out['action'],'no_flag');self.assertIsNone(out['diagnosis_value'])
        self.assertFalse(out['no_flag_means_clinical_normality']);self.assertFalse(out['human_review_performed'])
    def test_mismatched_or_missing_evidence_routes_to_review(self):
        for a in [agent([]),agent([call('A','B')],True),agent([call('A','B'),call('B',key='wrong')])]:
            out,_=a.invoke([HumanMessage(content='imageA')],'r1','imageA');self.assertEqual(out['action'],'review')
    def test_no_routing_state_leak_across_requests(self):
        a=agent([call('A','B')]);a.invoke([HumanMessage(content='imageA')],'r1','imageA')
        out,_=a.invoke([HumanMessage(content='imageB')],'r2','imageB');self.assertEqual(out['action'],'review')

if __name__=='__main__':unittest.main()
