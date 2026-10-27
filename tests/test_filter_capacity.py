import json
import logging
import pytest
from fastapi.testclient import TestClient
from agent_memory.api import create_app
from agent_memory.domain import Evidence, SearchRequest
from agent_memory.final_filter import FilterInputBoundError
from test_alias_contract import build, response

REQUEST=SearchRequest(user_id='synthetic-long',query='Which notebook did Dana choose?',top_k=100)


def test_one_hundred_long_sources_are_sent_in_full_and_returned_in_original_order(tmp_path):
    calls=[]
    hits=[Evidence(id=f'source-{i}',source_ids=(f'source-{i}',),
        content=f'Synthetic note {i}. '+('A complete source sentence. '*50)) for i in range(100)]
    def send(payload):
        rows=json.loads(payload['messages'][-1]['content'])['evidence']
        assert [r['content'] for r in rows]==[h.content for h in hits]
        assert len(json.dumps(payload,ensure_ascii=False).encode())>131184
        calls.append(payload)
        return response({r['id']:{'decision':'keep' if i%20==0 else 'drop',
            'reason':'relevant' if i%20==0 else 'irrelevant'} for i,r in enumerate(rows)})
    f=build(tmp_path,send)
    try:
        assert f.apply(REQUEST,hits)==hits[::20]
        assert f.apply(REQUEST,hits)==hits[::20]
        assert len(calls)==1
    finally:f.close()


def test_actual_token_limit_rejects_before_model_call_without_truncating(tmp_path):
    calls=[]
    f=build(tmp_path,lambda payload:calls.append(payload))
    try:
        with pytest.raises(FilterInputBoundError) as result:
            f.apply(REQUEST,[Evidence(id='long',source_ids=('long',),content=' a'*110001)])
        assert result.value.diagnostic['limit']=='max_input_tokens'
        assert result.value.diagnostic['input_tokens']>110000
        assert calls==[]
        assert f.store.list_tasks(stage='final-relevance-safety-v3')==[]
    finally:f.close()


def test_multibyte_and_special_token_literals_are_untrusted_plain_text(tmp_path):
    text='汉'*30000+' literal <|endoftext|> belongs to the source.'
    calls=[]
    def send(payload):
        assert json.loads(payload['messages'][-1]['content'])['evidence'][0]['content']==text
        calls.append(payload)
        return response({'e000':{'decision':'keep','reason':'relevant'}})
    f=build(tmp_path,send)
    hit=Evidence(id='multibyte',source_ids=('multibyte',),content=text)
    try:
        assert f.apply(REQUEST,[hit])==[hit]
        assert len(calls)==1
    finally:f.close()


def test_error_log_records_guard_details_without_headers_text_or_exception_args(tmp_path,caplog):
    secret='sk-'+'z'*32
    query='private synthetic question '+secret
    class Memory:
        def search(self,request):
            error=FilterInputBoundError(payload_bytes=131184,input_tokens=30000,candidates=100,limit='synthetic')
            error.args=(secret,)
            raise error
    app=create_app(tmp_path/'unused.sqlite3',service=Memory(),api_key=secret)
    with caplog.at_level(logging.ERROR,logger='clmemory.service'),TestClient(app) as client:
        result=client.post('/search',headers={'Authorization':'Token '+secret},
            json={'query':query,'user_id':'private-synthetic-user','top_k':100})
    assert result.status_code==503 and result.json()=={'detail':'Memory service unavailable'}
    entry=json.loads(next(r.message for r in caplog.records if r.name=='clmemory.service'))
    assert entry['diagnostic']['payload_bytes']==131184
    assert entry['error_type']=='FilterInputBoundError' and entry['frames']
    assert 'query_sha256' in entry and 'user_id_sha256' in entry
    assert secret not in caplog.text and query not in caplog.text and 'private-synthetic-user' not in caplog.text
