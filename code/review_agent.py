"""Distinct final review action sharing the proven request-scoped tool boundary."""
import json
import numpy as np
from langchain_core.messages import AIMessage
from gated_agent import GatedAgent
from review_core import score,flag

class ReviewRoutingAgent(GatedAgent):
    def __init__(self,model,tools,expert_names,labels,method,stacker,review_gate):
        self.method=method;self.stacker=stacker;self.review_gate=review_gate
        super().__init__(model,tools,expert_names,labels,None,None)

    def final_gate(self,state):
        complete=all(e in state['evidence'] and state['evidence'][e]['request_id']==state['request_id']
                     and state['evidence'][e]['image_key']==state['image_key'] for e in self.experts)
        needs_review=True;reason='incomplete_evidence'
        if complete:
            p=np.array([[state['evidence'][e]['scores'] for e in self.experts]])
            needs_review=bool(flag(score(p,self.method,self.stacker),self.review_gate)[0])
            reason='calibrated_routing_rule' if self.review_gate['selected'] is not None else 'no_certified_threshold'
        payload={'request_id':state['request_id'],'image_key':state['image_key'],'action':'review' if needs_review else 'no_flag',
                 'diagnosis_value':None,'target':'at_least_one_of_seven_NIH_annotations','reason':reason,
                 'human_review_performed':False,'no_flag_means_clinical_normality':False}
        return {'messages':[AIMessage(content=json.dumps(payload),additional_kwargs={'final_gate':True})],
                'trace':[{'event':'final_review_gate','complete_evidence':complete,'flagged':needs_review}]}
