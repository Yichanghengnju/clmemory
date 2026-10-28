from concurrent.futures import ThreadPoolExecutor
import json
from fastapi.testclient import TestClient
from agent_memory.memmachine_submission import build_app,VERSION
from agent_memory.method_memmachine import MemMachineMemory
from agent_memory.domain import SearchRequest
from test_submission import SyntheticEmbedder as BaseSyntheticEmbedder,HEADERS


class SyntheticEmbedder(BaseSyntheticEmbedder):
    dimensions=1024
    identity={'name':'test-hash-only','dimensions':1024,'formal_use':False}


def fixture():
    return {'user_id':'synthetic-memmachine','request_id':'long-100','session_id':'session',
        'messages':[{'role':'user','content':f'Synthetic source {i}. '+('A fictional inventory note. '*110)} for i in range(120)]}


def test_native_backend_only_with_long_top100_and_direct_method_equivalence(tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('clmemory delivery/filter must not be instantiated')
    monkeypatch.setattr('agent_memory.delivery.SubmissionMemory',forbidden)
    monkeypatch.setattr('agent_memory.final_filter.FinalEvidenceFilter',forbidden)
    app=build_app(db_path=tmp_path/'db',api_key='synthetic-key',embedder=SyntheticEmbedder(),manifest_dir=tmp_path/'run')
    assert type(app.state.memory_service) is MemMachineMemory
    with TestClient(app) as c:
        b=fixture();assert c.post('/add',headers=HEADERS,json=b).status_code==200
        q={'user_id':b['user_id'],'query':'Synthetic inventory','top_k':100}
        response=c.post('/search',headers=HEADERS,json=q);rows=response.json()['data']
        # top_k is a cap. Native retrieval may return fewer hits; never add
        # clmemory backfill to make a baseline test mechanically return 100.
        assert response.status_code==200 and 1<=len(rows)<=100 and len({x['id'] for x in rows})==len(rows)
        assert len(response.content)>131184
        direct=app.state.memory_service.search(SearchRequest(**q))
        assert rows==[{'id':h.id,'content':h.content,'score':h.score} for h in direct]
        short=c.post('/search',headers=HEADERS,json={**q,'top_k':8})
        assert short.status_code==200 and len(short.json()['data'])==8
    m=json.loads((tmp_path/'run/version-manifest.json').read_text())
    assert m['version']==VERSION and m['method']['name']=='memmachine' and m['method']['max_results']==100
    assert m['final_filter_enabled'] is False and m['clmemory_neighbors'] is False and m['clmemory_anchors'] is False
    assert m['llm_calls_in_add']==m['llm_calls_in_search']==0


def test_auth_scope_idempotency_conflict_and_16_concurrent_http_searches(tmp_path):
    app=build_app(db_path=tmp_path/'db',api_key='synthetic-key',embedder=SyntheticEmbedder(),manifest_dir=tmp_path/'run')
    with TestClient(app) as c:
        b=fixture();b['messages']=b['messages'][:12]
        assert c.post('/add',headers=HEADERS,json=b).status_code==200
        assert c.post('/add',headers=HEADERS,json=b).status_code==200
        assert c.post('/add',headers=HEADERS,json={**b,'session_id':'changed'}).status_code==409
        q={'user_id':b['user_id'],'query':'Synthetic inventory','top_k':100}
        assert c.post('/search',json=q).status_code==401
        assert c.post('/search',headers={'Authorization':'Token wrong'},json=q).status_code==401
        assert c.post('/search',headers=HEADERS,json={**q,'user_id':'different-user'}).json()=={'data':[]}
        with ThreadPoolExecutor(max_workers=16) as pool:
            responses=list(pool.map(lambda _:c.post('/search',headers=HEADERS,json=q),range(16)))
        assert all(x.status_code==200 and len(x.json()['data'])==12 for x in responses)
