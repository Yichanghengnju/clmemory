import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from agent_memory.estar_submission import build_app, SKILL_SHA256, VERSION
from test_submission import SyntheticEmbedder as BaseSyntheticEmbedder, HEADERS


class SyntheticEmbedder(BaseSyntheticEmbedder):
    dimensions=1024
    identity={'name':'synthetic1024','dimensions':1024,'formal_use':False}


class Harness:
    enable_evidence_chain=False
    identity={'harness':'@deepseek-ai/dsh@0.2.0-rc.1','skill_sha256':SKILL_SHA256,
              'model':'gpt-4o-mini','max_steps':6,'max_tokens':1536,'timeout_s':150}
    def __init__(self):self.calls=0
    def run(self,tools):
        self.calls+=1
        observed=tools.execute('hybrid_search',{'query':tools.request.query})
        ids=[s['source_id'] for s in observed['sources']]
        tools.execute('finish_evidence',{'source_ids':ids[:2]})
        return {'selected':tools.selected,'run_id':'scripted-native-shape','root':'synthetic-only'}


def test_real_event_backend_Estar_assembly_and_cache_http_contract(tmp_path):
    h=Harness();app=build_app(db_path=tmp_path/'db',api_key='synthetic-key',
        embedder=SyntheticEmbedder(),run_dir=tmp_path/'run',harness=h)
    b={'user_id':'synthetic-E','request_id':'r1','session_id':'s1','messages':[
        {'role':'user','content':'The fictional E* inventory stores42 marbles in the green box.'},
        {'role':'assistant','content':'The fictional inventory note has been received.'}]}
    with TestClient(app) as c:
        assert c.post('/add',headers=HEADERS,json=b).status_code==200
        assert c.post('/add',headers=HEADERS,json=b).status_code==200
        assert c.post('/add',headers=HEADERS,json={**b,'session_id':'conflict'}).status_code==409
        q={'user_id':b['user_id'],'query':'How many marbles are in the fictional green box?','top_k':100}
        a=c.post('/search',headers=HEADERS,json=q)
        assert a.status_code==200 and '42 marbles' in a.text and h.calls==1
        assert c.post('/search',headers=HEADERS,json=q).json()==a.json() and h.calls==1
        assert c.post('/search',json=q).status_code==401
        assert c.post('/search',headers={'Authorization':'Token wrong'},json=q).status_code==401
        assert c.post('/search',headers=HEADERS,json={**q,'user_id':'foreign'}).json()=={'data':[]}
        with ThreadPoolExecutor(max_workers=16) as pool:
            replies=list(pool.map(lambda _:c.post('/search',headers=HEADERS,json=q),range(16)))
        assert all(v.status_code==200 and v.json()==a.json() for v in replies) and h.calls==1
    m=json.loads((tmp_path/'run/version-manifest.json').read_text())
    assert m['version']==VERSION and m['H'] and m['F'] and m['S14']
    assert not m['KD'] and not m['AgentDistill'] and m['S14_sha256']==SKILL_SHA256


def test_published_S14_text_has_exact_frozen_identity():
    p=Path(__file__).resolve().parents[1]/'skills/S14.txt'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==SKILL_SHA256
