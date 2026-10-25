"""Restorable weighted route/source draws and shuffled cyclic queues."""
import torch
from .config import digest

def derived_seed(seed,key):return int(digest([seed,key])[:15],16)

class DeterministicSampler:
    def __init__(self,config,pools):
        self.config=config;self.pools={str(k[0])+'\0'+str(k[1]):list(v) for k,v in pools.items()}
        self.route_names=[r.name for r in config.stage.routes];self.routes={r.name:r for r in config.stage.routes}
        self.route_weights=torch.tensor([r.weight or 1. for r in config.stage.routes],dtype=torch.float64)
        self.draw_generator=torch.Generator().manual_seed(derived_seed(config.seed,config.stage.stage_id))
        self.generators={k:torch.Generator().manual_seed(derived_seed(config.seed,config.stage.stage_id+k)) for k in sorted(self.pools)}
        self.queues={};self.cursors={k:0 for k in self.pools};self.epochs={k:0 for k in self.pools};self.attempts=0
        self.source_weights={s.source_id:s.weight or 1. for s in config.sources if s.role=='train'}
        for k in self.pools:self._shuffle(k)
    def _shuffle(self,key):
        order=torch.randperm(len(self.pools[key]),generator=self.generators[key]).tolist()
        self.queues[key]=[self.pools[key][i] for i in order];self.cursors[key]=0
    def draw(self):
        ri=int(torch.multinomial(self.route_weights,1,generator=self.draw_generator));route=self.route_names[ri]
        sources=sorted(k.split('\0')[1] for k in self.pools if k.split('\0')[0]==route)
        weights=torch.tensor([self.source_weights[s] for s in sources],dtype=torch.float64)
        source=sources[int(torch.multinomial(weights,1,generator=self.draw_generator))];key=route+'\0'+source
        if self.cursors[key]==len(self.queues[key]):self.epochs[key]+=1;self._shuffle(key)
        index=self.queues[key][self.cursors[key]];self.cursors[key]+=1;self.attempts+=1
        return route,source,index
    def state_dict(self):
        return {'signature':digest({'pools':self.pools,'config':self.config.signature()}),'attempts':self.attempts,
            'draw_state':self.draw_generator.get_state(),'queue_states':{k:g.get_state() for k,g in self.generators.items()},
            'queues':{k:list(v) for k,v in self.queues.items()},'cursors':dict(self.cursors),'epochs':dict(self.epochs)}
    def load_state_dict(self,state):
        if state['signature']!=self.state_dict()['signature']:raise ValueError('Sampler pool/config signature differs')
        for k,q in state['queues'].items():
            if sorted(q)!=sorted(self.pools[k]) or not 0<=state['cursors'][k]<=len(q):raise ValueError('Corrupt sampler queue/cursor')
        self.draw_generator.set_state(state['draw_state'])
        for k,v in state['queue_states'].items():self.generators[k].set_state(v)
        self.queues={k:list(v) for k,v in state['queues'].items()};self.cursors=dict(state['cursors']);self.epochs=dict(state['epochs']);self.attempts=state['attempts']
