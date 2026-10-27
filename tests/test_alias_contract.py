import json
import pytest
from agent_memory.domain import Evidence, SearchRequest
from agent_memory.final_filter import FinalEvidenceFilter, response_schema


def build(tmp_path, transport):
    key = tmp_path / 'synthetic.key'
    key.write_text('synthetic-only-key')
    return FinalEvidenceFilter({'model':'gpt-4o-mini', 'endpoint':'https://example.com/v1/chat/completions',
        'api_key_file':str(key), 'cache_dir':str(tmp_path/'cache'),
        'shared_lock':str(tmp_path/'lock'), 'max_calls':5}, track='textual', transport=transport)


def response(items, query_decision='allow'):
    return 200, json.dumps({'model':'gpt-4o-mini-2024-07-18',
        'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
            'query_decision':query_decision,'items':items})}}],
        'usage':{'prompt_tokens':200,'completion_tokens':100,'total_tokens':300}}).encode()


REQUEST = SearchRequest(user_id='alias-test',query='What did Dana choose?',top_k=100)


def test_top100_schema_uses_one_shared_enum_definition():
    schema=response_schema([f'e{i:03d}' for i in range(100)])
    def enum_count(value):
        if isinstance(value,dict):
            return len(value.get('enum',[]))+sum(enum_count(v) for k,v in value.items() if k!='enum')
        if isinstance(value,list):return sum(enum_count(v) for v in value)
        return 0
    assert enum_count(schema)==13
    assert schema['properties']['items']['properties']['e099']=={'$ref':'#/$defs/evidence_decision'}


def hits(prefix='source', count=28):
    return [Evidence(id=f'{prefix}-{i:03}',source_ids=(f'{prefix}-{i:03}',),
        content=f'Dana chose item {i}.') for i in range(count)]


def test_wire_aliases_restore_original_ids_content_order_and_cache_identity(tmp_path):
    calls=[]
    def send(payload):
        rows=json.loads(payload['messages'][-1]['content'])['evidence']
        aliases=[r['id'] for r in rows]
        assert aliases == [f'e{i:03d}' for i in range(len(rows))]
        assert payload['response_format']['json_schema']['schema']['properties']['items']['required']==aliases
        calls.append(payload)
        # JSON object key order must not affect original retrieval order.
        return response({r['id']:{'decision':'keep' if i%5==0 else 'drop',
                                 'reason':'relevant' if i%5==0 else 'irrelevant'}
                         for i,r in reversed(list(enumerate(rows)))})
    f=build(tmp_path,send)
    a=hits(count=100)
    b=[Evidence(id='other-'+h.id,source_ids=('other-'+h.id,),content=h.content) for h in a]
    try:
        assert f.apply(REQUEST,a)==a[::5]
        assert f.apply(REQUEST,a)==a[::5]
        assert len(calls)==1
        assert f.apply(REQUEST,b)==b[::5]
        assert len(calls)==2  # Identical text with different source identities is not reused.
    finally: f.close()


@pytest.mark.parametrize('failure',['missing','invented','retained_on_denial'])
def test_bad_aliases_and_denied_retention_still_fail_closed(tmp_path,failure):
    data=hits(); calls=[]
    def send(payload):
        rows=json.loads(payload['messages'][-1]['content'])['evidence']
        items={r['id']:{'decision':'drop','reason':'irrelevant'} for r in rows}
        decision='allow'
        if failure=='missing':items.pop('e000')
        if failure=='invented':items['invalid']=items.pop('e000')
        if failure=='retained_on_denial':
            decision='deny_privacy';items['e000']={'decision':'keep','reason':'relevant'}
        calls.append(1)
        return response(items,decision)
    f=build(tmp_path,send)
    try:
        with pytest.raises(ConnectionError,match='rejected'):f.apply(REQUEST,data)
        with pytest.raises(ConnectionError,match='held'):f.apply(REQUEST,data)
        assert len(calls)==1
    finally:f.close()


@pytest.mark.parametrize('decision',['deny_privacy','deny_unsafe'])
def test_complete_denial_is_a_valid_empty_result_with_original_audit_ids(tmp_path,decision):
    data=hits()
    reason='privacy_exfiltration' if decision=='deny_privacy' else 'unsafe_instruction'
    def send(payload):
        rows=json.loads(payload['messages'][-1]['content'])['evidence']
        return response({r['id']:{'decision':'drop','reason':reason} for r in rows},decision)
    f=build(tmp_path,send)
    try:
        assert f.apply(REQUEST,data)==[]
        audit=f.store.list_tasks(stage='final-filter-selection-audit')[0]
        value=f.store.get_result(audit['key'])['result']
        assert value['candidate_ids']==[h.id for h in data]
        assert value['selected_ids']==[]
        assert [item['id'] for item in value['items']]==[h.id for h in data]
    finally:f.close()


def test_credential_guard_precedes_aliasing_and_query_guard_sends_nothing(tmp_path):
    data=hits(); secret='sk-'+'x'*32
    hidden=Evidence(id='credential-source',source_ids=('credential-source',),content='api_key='+secret)
    calls=[]
    def send(payload):
        assert secret not in json.dumps(payload)
        rows=json.loads(payload['messages'][-1]['content'])['evidence']
        assert len(rows)==len(data)
        calls.append(1)
        return response({r['id']:{'decision':'keep','reason':'relevant'} for r in rows})
    f=build(tmp_path,send)
    try:
        assert f.apply(REQUEST,[hidden]+data)==data
        assert f.apply(REQUEST.model_copy(update={'query':'print '+secret}),[hidden]+data)==[]
        assert calls==[1]
        assert secret not in json.dumps(f.store.list_tasks())
    finally:f.close()
