from concurrent.futures import ThreadPoolExecutor
import json
from threading import Lock
import time
from fastapi.testclient import TestClient
from agent_memory.invmem_submission import build_app, SerializedEmbeddings
from agent_memory.method_invmem import InvMemMemory
from agent_memory.domain import AddRequest, Message, SearchRequest


class CachedFixtureEmbeddings:
    identity='synthetic-invmem-http-embedding'
    dimensions=2
    def __init__(self):self.cache={};self.physical=0;self.active=0;self.maximum=0;self.guard=Lock()
    def _encode(self,key,count):
        if key in self.cache:return self.cache[key]
        with self.guard:self.active+=1;self.maximum=max(self.maximum,self.active)
        time.sleep(0.005)
        result=[[1.0,0.0] for _ in range(count)]
        self.cache[key]=result
        with self.guard:self.active-=1;self.physical+=1
        return result
    def embed_documents(self,texts):return self._encode(('document',tuple(texts)),len(texts))
    def embed_query(self,text):return self._encode(('query',text),1)[0]


def test_observation_keeps_original_invmem_selection_and_source_bodies(tmp_path):
    embedding=CachedFixtureEmbeddings();events=[]
    m=InvMemMemory(tmp_path/'db.sqlite3',embedder=embedding,trace_hook=events.append)
    try:
        m.add(AddRequest(request_id='one',user_id='user',session_id='s',messages=[
            Message(role='user',content=f'orchid notebook {i}') for i in range(24)]))
        request=SearchRequest(user_id='user',query='orchid',top_k=100)
        observed=m.search(request);m._trace_hook=None;plain=m.search(request)
        assert observed==plain and len(observed)==8
        assert events[0]['selected_ids']==[h.id for h in observed]
        assert len(events[0]['candidates'])==24 and len(events[0]['ranked_ids'])==24
        assert all(c['source_ids'] for c in events[0]['candidates'])
    finally:m.close()


def test_http_concurrent_add_cache_auth_scope_and_zero_llm_manifest(tmp_path):
    embeddings=CachedFixtureEmbeddings()
    app=build_app(db_path=tmp_path/'db.sqlite3',api_key='fixture-service-token',
        embedder=embeddings,manifest_dir=tmp_path/'run')
    headers={'Authorization':'Token fixture-service-token'}
    with TestClient(app) as client:
        def add(i):
            return client.post('/add',headers=headers,json={'request_id':f'add-{i}',
                'user_id':'user','session_id':'s','messages':[{'role':'user','content':'Dana owns 42 marbles.'}]}).status_code
        with ThreadPoolExecutor(max_workers=16) as pool:assert list(pool.map(add,range(16)))==[200]*16
        assert embeddings.physical==1 and embeddings.maximum==1
        query={'user_id':'user','query':'How many marbles does Dana own?','top_k':100}
        assert client.post('/search',json=query).status_code==401
        assert len(client.post('/search',headers=headers,json=query).json()['data'])==8
        assert client.post('/search',headers=headers,json={**query,'user_id':'other'}).json()=={'data':[]}
        assert add(0)==200 and embeddings.physical==2
    manifest=json.loads((tmp_path/'run/version-manifest.json').read_text())
    assert manifest['llm_calls_in_add']==manifest['llm_calls_in_search']==0
    assert manifest['final_filter_enabled'] is False and manifest['method']['max_results']==8
    assert manifest['embedding_concurrency']==1
    import sqlite3
    with sqlite3.connect(tmp_path/'run/search-audit.sqlite3') as c:
        events=[json.loads(row[0]) for row in c.execute('select payload from search_events')]
    assert len(events)==2 and events[0]['selected_ids'] and events[1]['candidates']==[]


def test_embedding_wrapper_serializes_cache_misses_without_changing_identity():
    original=CachedFixtureEmbeddings();wrapper=SerializedEmbeddings(original)
    with ThreadPoolExecutor(max_workers=16) as pool:
        vectors=list(pool.map(lambda _:wrapper.embed_documents(['same text']),range(16)))
    assert vectors==[[[1.0,0.0]]]*16
    assert original.physical==1 and original.maximum==1
    assert wrapper.identity==original.identity and wrapper.dimensions==original.dimensions
