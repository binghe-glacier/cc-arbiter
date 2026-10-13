"""LangGraph integration with request-scoped evidence and an enforced final gate.

The planning model never decides the final binary outputs. Only supported
classifier tools are accepted. No clinical free-text answer is certified.
"""
import json, operator
from typing import Annotated, TypedDict
import numpy as np
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from cc_arbiter import proposals

class GateState(TypedDict):
    messages: Annotated[list,operator.add]
    request_id: str
    image_key: str
    evidence: dict
    trace: Annotated[list,operator.add]
    rounds: int

class GatedAgent:
    def __init__(self,model,tools,expert_names,labels,conformal,risk_gate,target=.1,max_rounds=3):
        if len(expert_names)!=2 or len(set(expert_names))!=2: raise ValueError('two distinct registered experts required')
        if len(set(labels))!=len(labels) or not labels: raise ValueError('unique labels required')
        self.tools={t.name:t for t in tools}; self.experts=tuple(expert_names); self.labels=list(labels)
        if any(x not in self.tools for x in self.experts): raise ValueError('missing registered expert')
        self.model=model.bind_tools(tools); self.conformal=conformal; self.risk_gate=risk_gate
        self.target=target; self.max_rounds=max_rounds
        # Bound methods are captured only after this fully initialized class exists.
        graph=StateGraph(GateState)
        graph.add_node('plan',self.plan); graph.add_node('execute',self.execute); graph.add_node('final_gate',self.final_gate)
        graph.set_entry_point('plan')
        graph.add_conditional_edges('plan',self.route,{'execute':'execute','final_gate':'final_gate'})
        graph.add_edge('execute','plan'); graph.add_edge('final_gate',END)
        self.workflow=graph.compile()

    def plan(self,state):
        return {'messages':[self.model.invoke(state['messages'])]}

    def route(self,state):
        return 'execute' if state['messages'][-1].tool_calls and state['rounds']<self.max_rounds else 'final_gate'

    def execute(self,state):
        evidence=dict(state['evidence']); messages=[]; trace=[]
        for call in state['messages'][-1].tool_calls:
            name=call['name']; args=call['args']; accepted=False; reason='unsupported_tool'
            if name in self.experts:
                # An invalid/repeated expert call invalidates that expert's previous
                # result for the current request, preventing stale-value reuse.
                evidence.pop(name,None)
                if args.get('image_key')!=state['image_key']:
                    reason='image_key_mismatch'
                else:
                    try:
                        output=self.tools[name].invoke(args)
                        if isinstance(output,str): output=json.loads(output)
                        if output.get('image_key')!=state['image_key']: raise ValueError('output key mismatch')
                        vals=[float(output['scores'][k]) for k in self.labels]
                        if not np.isfinite(vals).all() or np.any(np.asarray(vals)<0) or np.any(np.asarray(vals)>1): raise ValueError('invalid scores')
                        evidence[name]={'request_id':state['request_id'],'image_key':state['image_key'],'scores':vals}
                        accepted=True; reason='evidence_recorded'
                    except Exception as e: reason='tool_failure:'+type(e).__name__
            status={'expert':name,'accepted':accepted,'reason':reason,'image_key':state['image_key']}
            messages.append(ToolMessage(tool_call_id=call['id'],name=name,content=json.dumps(status)))
            trace.append(status)
        return {'messages':messages,'evidence':evidence,'rounds':state['rounds']+1,'trace':trace}

    def final_gate(self,state):
        complete=all(e in state['evidence'] and state['evidence'][e]['request_id']==state['request_id']
                     and state['evidence'][e]['image_key']==state['image_key'] for e in self.experts)
        decision={k:{'action':'defer','value':None,'reason':'missing_expert','route':'request_second_expert'} for k in self.labels}
        if complete:
            p=np.array([[state['evidence'][e]['scores'] for e in self.experts]])
            sets=self.conformal.predict(p); pred,eligible,score=proposals(p,sets,'intersection')
            accept=self.risk_gate.accept(eligible,score,self.target)[0]
            for j,k in enumerate(self.labels):
                decision[k]={'action':'release' if accept[j] else 'defer','value':int(pred[0,j]) if accept[j] else None,
                             'reason':'certified_policy' if accept[j] else ('set_not_singleton' if not eligible[0,j] else 'uncertified_threshold'),
                             'route':'none' if accept[j] else 'review',
                             'expert_sets':[np.flatnonzero(sets[0,e,j]).tolist() for e in range(2)]}
        payload={'request_id':state['request_id'],'image_key':state['image_key'],'target':'dataset_annotation',
                 'risk_target':self.target,'decisions':decision}
        return {'messages':[AIMessage(content=json.dumps(payload),additional_kwargs={'final_gate':True})],
                'trace':[{'event':'final_gate','complete_evidence':complete,'released':sum(v['action']=='release' for v in decision.values())}]}

    def invoke(self,messages,request_id,image_key):
        if not request_id or not image_key: raise ValueError('explicit nonempty request and image keys required')
        # Fresh invocation state: evidence is never a mutable object on the agent.
        state={'messages':messages,'request_id':request_id,'image_key':image_key,'evidence':{},'trace':[],'rounds':0}
        result=self.workflow.invoke(state,config={'recursion_limit':4*self.max_rounds+6})
        final=result['messages'][-1]
        if not final.additional_kwargs.get('final_gate'): raise RuntimeError('final gate not executed')
        return json.loads(final.content),result['trace']
