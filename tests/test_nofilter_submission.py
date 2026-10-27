from concurrent.futures import ThreadPoolExecutor
import json
from fastapi.testclient import TestClient
from agent_memory.nofilter_submission import build_app, VERSION
from agent_memory.domain import SearchRequest
from test_submission import SyntheticEmbedder, HEADERS


def fixture():
    return {'user_id':'synthetic-nofilter','request_id':'long-100','session_id':'session',
        'messages':[{'role':'user','content':f'Synthetic source {i}. '+('A fictional inventory note. '*55)} for i in range(120)]}


def test_exact_original_delivery_long_top100_no_filter_or_model_initialization(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No final filter may be instantiated')
    monkeypatch.setattr('agent_memory.final_filter.FinalEvidenceFilter',forbidden)
    app=build_app(db_path=tmp_path/'db.sqlite3',api_key='synthetic-key',embedder=SyntheticEmbedder(),manifest_dir=tmp_path/'run')
    with TestClient(app) as client:
        body=fixture();assert client.post('/add',headers=HEADERS,json=body).status_code==200
        query={'user_id':body['user_id'],'query':'Synthetic inventory','top_k':100}
        response=client.post('/search',headers=HEADERS,json=query)
        assert response.status_code==200
        rows=response.json()['data'];assert len(rows)==100 and len({r['id'] for r in rows})==100
        assert sum(len(r['content']) for r in rows)>131184
        direct=app.state.memory_service.search(SearchRequest(**query))
        assert rows==[{'id':h.id,'content':h.content,'score':h.score} for h in direct]
        head=client.post('/search',headers=HEADERS,json={**query,'top_k':16}).json()['data']
        assert rows[:16]==head
    manifest=json.loads((tmp_path/'run/version-manifest.json').read_text())
    assert manifest['version']==VERSION and manifest['method']['max_results']==100
    assert 'final_filter' not in manifest['method']
    assert manifest['llm_calls_in_add']==manifest['llm_calls_in_search']==0
    assert manifest['final_filter_enabled'] is False


def test_http_auth_scope_conflict_and_concurrent_source_identity(tmp_path):
    app=build_app(db_path=tmp_path/'db.sqlite3',api_key='synthetic-key',embedder=SyntheticEmbedder(),manifest_dir=tmp_path/'run')
    with TestClient(app) as client:
        body=fixture();body['messages']=body['messages'][:8]
        assert client.post('/add',headers=HEADERS,json=body).status_code==200
        assert client.post('/add',headers=HEADERS,json=body).status_code==200
        changed={**body,'messages':[{'role':'user','content':'changed'}]}
        assert client.post('/add',headers=HEADERS,json=changed).status_code==409
        query={'user_id':body['user_id'],'query':'Synthetic inventory','top_k':100}
        assert client.post('/search',json=query).status_code==401
        assert client.post('/search',headers=HEADERS,json={**query,'user_id':'different-user'}).json()=={'data':[]}
        with ThreadPoolExecutor(max_workers=16) as pool:
            results=list(pool.map(lambda _:client.post('/search',headers=HEADERS,json=query),range(16)))
        assert all(r.status_code==200 and len(r.json()['data'])==8 for r in results)
        assert all(r.json()==results[0].json() for r in results)


def test_whole_stage_removal_includes_the_synthetic_credential_pattern(tmp_path):
    app=build_app(db_path=tmp_path/'db.sqlite3',api_key='synthetic-key',embedder=SyntheticEmbedder(),manifest_dir=tmp_path/'run')
    fake='sk-'+'synthetic'*6
    with TestClient(app) as client:
        body={'user_id':'fixture','request_id':'fixture','session_id':'s',
            'messages':[{'role':'user','content':'Synthetic fake API token: '+fake}]}
        assert client.post('/add',headers=HEADERS,json=body).status_code==200
        result=client.post('/search',headers=HEADERS,json={'user_id':'fixture','query':'fake API token','top_k':100})
        assert result.status_code==200 and fake in result.json()['data'][0]['content']
