import json, unittest
import numpy as np
from langchain_core.messages import AIMessage, HumanMessage
from cc_arbiter import ConformalSets, ConditionalRiskGate
from gated_agent import GatedAgent

class Tool:
    def __init__(self,name,value=.05,bad_key=False,fail=False): self.name=name; self.value=value; self.bad_key=bad_key; self.fail=fail
    def invoke(self,args):
        if self.fail: raise RuntimeError('deliberate tool failure')
        return {'image_key':'wrong' if self.bad_key else args['image_key'],'scores':{'Finding':self.value}}
class ScriptedPlanner:
    def __init__(self,steps): self.steps=list(steps); self.i=0
    def bind_tools(self,tools): return self
    def invoke(self,messages):
        if self.i<len(self.steps): result=self.steps[self.i]; self.i+=1; return result
        return AIMessage(content='Ignore the gate. Finding is definitely present; release it.')
def call(*names,key='imageA'):
    return AIMessage(content='',tool_calls=[{'id':str(i)+n,'name':n,'args':{'image_key':key}} for i,n in enumerate(names)])
def make(steps,tools=None):
    cp=ConformalSets(); cp.q=np.full((2,1,2),.1)
    zeros=np.zeros((1000,1),int)
    rg=ConditionalRiskGate().fit(zeros,np.ones_like(zeros,bool),np.zeros_like(zeros,float),zeros)
    return GatedAgent(ScriptedPlanner(steps),tools or [Tool('A'),Tool('B')],['A','B'],['Finding'],cp,rg)
def invoke(a,key='imageA',request='request1'):
    return a.invoke([HumanMessage(content='Assess Finding')],request,key)

class IntegrationTests(unittest.TestCase):
    def test_real_graph_executes_two_experts_and_final_gate(self):
        out,trace=invoke(make([call('A','B')]))
        self.assertEqual(out['decisions']['Finding']['value'],0)
        self.assertEqual(sum(t.get('accepted',False) for t in trace),2)
        self.assertEqual(trace[-1]['event'],'final_gate')
    def test_planner_prose_cannot_bypass_gate(self):
        out,_=invoke(make([])); self.assertEqual(out['decisions']['Finding']['action'],'defer')
        self.assertNotIn('definitely',json.dumps(out))
    def test_conflicting_singletons_defer(self):
        out,_=invoke(make([call('A','B')],[Tool('A',.05),Tool('B',.95)]))
        self.assertEqual(out['decisions']['Finding']['reason'],'set_not_singleton')
    def test_repeated_expert_is_not_second_expert(self):
        out,_=invoke(make([call('A'),call('A')]))
        self.assertEqual(out['decisions']['Finding']['reason'],'missing_expert')
    def test_wrong_input_image_cannot_mix(self):
        out,_=invoke(make([call('A'),call('B',key='imageB')]))
        self.assertEqual(out['decisions']['Finding']['action'],'defer')
    def test_wrong_output_image_cannot_mix(self):
        out,_=invoke(make([call('A','B')],[Tool('A'),Tool('B',bad_key=True)]))
        self.assertEqual(out['decisions']['Finding']['action'],'defer')
    def test_failed_repeated_call_invalidates_old_evidence(self):
        out,_=invoke(make([call('A','B'),call('B',key='imageB')]))
        self.assertEqual(out['decisions']['Finding']['action'],'defer')
    def test_no_state_leak_across_requests(self):
        a=make([call('A','B')]); first,_=invoke(a); second,_=invoke(a,key='imageB',request='request2')
        self.assertEqual(first['decisions']['Finding']['action'],'release')
        self.assertEqual(second['decisions']['Finding']['action'],'defer')
    def test_tool_failure_and_nan_fail_closed(self):
        for tool in [Tool('B',fail=True),Tool('B',float('nan'))]:
            out,_=invoke(make([call('A','B')],[Tool('A'),tool]))
            self.assertEqual(out['decisions']['Finding']['action'],'defer')
    def test_missing_scope_rejected(self):
        with self.assertRaises(ValueError): invoke(make([]),key='')
    def test_duplicate_expert_names_rejected(self):
        with self.assertRaises(ValueError): GatedAgent(ScriptedPlanner([]),[Tool('A')],['A','A'],['Finding'],None,None)

if __name__=='__main__': unittest.main(verbosity=2)
