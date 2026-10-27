from concurrent.futures import ThreadPoolExecutor
import json
from threading import Lock
import time
from agent_memory.domain import Evidence, SearchRequest
from agent_memory.final_filter import FinalEvidenceFilter

def test_concurrent_same_request_waits_then_reuses_one_physical_result(tmp_path):
    key=tmp_path/'key';key.write_text('synthetic-key')
    calls=[]
    def send(payload):
        calls.append(1);time.sleep(0.04)
        return 200,json.dumps({'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':{
            'content':json.dumps({'query_decision':'allow','items':{'e000':{'decision':'keep','reason':'relevant'}}})}}],
            'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
    f=FinalEvidenceFilter({'model':'gpt-4o-mini','endpoint':'https://example.com/v1/chat/completions',
        'api_key_file':str(key),'cache_dir':str(tmp_path/'cache'),'shared_lock':str(tmp_path/'slot'),
        'max_calls':1},track='textual',transport=send)
    request=SearchRequest(user_id='synthetic',query='Which color?',top_k=100)
    hit=Evidence(id='original-id',source_ids=('original-id',),content='Blue notebook.')
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            results=list(pool.map(lambda _:f.apply(request,[hit]),range(16)))
        assert results==[[hit]]*16 and calls==[1]
    finally:f.close()

def test_distinct_requests_share_one_model_slot_and_keep_the_cumulative_cap(tmp_path):
    key=tmp_path/'key';key.write_text('synthetic-key')
    guard=Lock();active=maximum=calls=0
    def send(payload):
        nonlocal active,maximum,calls
        with guard:
            active+=1;maximum=max(maximum,active);calls+=1
        time.sleep(0.03)
        with guard:active-=1
        return 200,json.dumps({'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':{
            'content':json.dumps({'query_decision':'allow','items':{'e000':{'decision':'keep','reason':'relevant'}}})}}],
            'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
    f=FinalEvidenceFilter({'model':'gpt-4o-mini','endpoint':'https://example.com/v1/chat/completions',
        'api_key_file':str(key),'cache_dir':str(tmp_path/'cache'),'shared_lock':str(tmp_path/'slot'),
        'max_calls':2},track='textual',transport=send)
    hit=Evidence(id='original-id',source_ids=('original-id',),content='Blue notebook.')
    requests=[SearchRequest(user_id=f'synthetic-{i}',query='Which color?',top_k=100) for i in range(2)]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(lambda request:f.apply(request,[hit]),requests))==[[hit],[hit]]
        assert calls==2 and maximum==1
    finally:f.close()
