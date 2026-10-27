import json
import pytest
from agent_memory.domain import Evidence, SearchRequest
from agent_memory.final_filter import FinalEvidenceFilter

def build(tmp_path, transport, max_calls=1):
    key=tmp_path/'key'; key.write_text('synthetic-only-key')
    return FinalEvidenceFilter({'model':'gpt-4o-mini','endpoint':'https://example.com/v1/chat/completions',
        'api_key_file':str(key),'cache_dir':str(tmp_path/'cache'),'shared_lock':str(tmp_path/'lock'),
        'max_calls':max_calls},track='textual',transport=transport)

def envelope(rows, refusal=False):
    return 200,json.dumps({'model':'gpt-4o-mini-2024-07-18','choices':[{'finish_reason':'stop',
        'message':{'content':json.dumps({'result':{'query_decision':'allow','items':{row['id']:{k:v for k,v in row.items() if k!='id'} for row in rows}}}),
                   'refusal':'Synthetic refusal' if refusal else None}}],
        'usage':{'prompt_tokens':100,'completion_tokens':100,'total_tokens':200}}).encode()

REQUEST=SearchRequest(user_id='synthetic-user',query='Which notebook did Dana choose?',top_k=100)
HITS=[Evidence(id=f'source-{i}',source_ids=(f'source-{i}',),content=f'Synthetic source {i}') for i in range(28)]
ROWS=[{'id':hit.id,'decision':'keep' if i%5==0 else 'drop',
       'reason':'relevant' if i%5==0 else 'irrelevant'} for i,hit in enumerate(HITS)]
WIRE_ROWS=[{**row,'id':f'e{i:03d}'} for i,row in enumerate(ROWS)]

def test_28_complete_decisions_keep_original_sources_and_reuse_result(tmp_path):
    calls=[]
    def send(payload):
        calls.append(payload)
        schema=payload['response_format']['json_schema']['schema']['properties']['result']['anyOf'][0]['properties']['items']
        assert schema['type']=='object' and schema['additionalProperties'] is False
        assert schema['required']==[f'e{i:03d}' for i in range(len(HITS))]
        assert set(schema['properties'])=={f'e{i:03d}' for i in range(len(HITS))}
        return envelope(WIRE_ROWS)
    f=build(tmp_path,send)
    expected=[h for i,h in enumerate(HITS) if i%5==0]
    assert f.apply(REQUEST,HITS)==expected
    assert f.apply(REQUEST,HITS)==expected
    assert len(calls)==1
    f.close()

@pytest.mark.parametrize('failure',['missing_rows','refusal','unknown_id'])
def test_invalid_results_remain_held_with_private_diagnostics(tmp_path,failure):
    rows=[dict(row) for row in WIRE_ROWS]
    if failure=='missing_rows': rows=[row for row in rows if row['decision']=='keep']
    if failure=='unknown_id': rows[0]['id']='invented'
    calls=[]
    def send(payload):
        calls.append(payload)
        return envelope(rows,refusal=failure=='refusal')
    f=build(tmp_path,send,max_calls=2)
    with pytest.raises(ConnectionError,match='rejected'): f.apply(REQUEST,HITS)
    with pytest.raises(ConnectionError,match='held'): f.apply(REQUEST,HITS)
    assert len(calls)==1
    task=f.store.list_tasks(stage='final-relevance-safety-v3')[0]
    attempt=f.store.list_attempts(task['id'])[0]
    expected='finish_or_refusal' if failure=='refusal' else 'decision_schema'
    assert attempt['error']['validation_phase']==expected
    assert (tmp_path/'cache'/'invalid-responses'/attempt['response']['private_response_file']).exists()
    f.close()
