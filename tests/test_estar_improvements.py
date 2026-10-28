import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from agent_memory.deepseek_memory import QueryTools
from agent_memory.domain import SearchRequest, Evidence
from agent_memory.evidence_safety import Visibility
from agent_memory.native_chat import NativeChatProvider
from agent_memory.mini_proxy_provider import MiniProxyError
from agent_memory.persistent_native_harness import NativeHarnessPool, PersistentNativeHarness

RUNTIME=Path(os.environ.get('AM_TEST_RUNTIME','data/deepseek-runtime'))

def source(sid,text,role='user',seq=1):
    return dict(id=sid,user_id='alice',session_id='s',request_id=sid,position=seq,seq=seq,
                timestamp=None,content=text,role=role)

def view(sources):
    store=SimpleNamespace(all_sources=lambda uid:sources if uid=='alice' else [])
    return QueryTools(SimpleNamespace(_store=store),SearchRequest(user_id='alice',query='orchid',top_k=8),
                      Visibility('alice',sources))

def hit(s):return Evidence(s['id'],s['content'],(s['id'],))

@pytest.mark.parametrize('old,control',[
    ('I enjoy Middle Eastern cuisine.','Forget that I enjoy Middle Eastern food.'),
    ('I prefer using email and Slack.','Please forget that I prefer using email and Slack.'),
    ('I enjoy Middle Eastern cuisine.','Please forget that I enjoy Middle Eastern cuisine such as shawarma and falafel.'),
    ('I prefer email and Slack.','Forget that I prefer using email and Slack more than WhatsApp for main conversations.'),
    ('我喜欢乌龙茶','忘记我喜欢乌龙茶。'),
])
def test_withdrawal_retains_exact_update_and_disables_old_value(old,control):
    sources=[source('old',old),source('withdraw',control,seq=2)]
    v=Visibility('alice',sources)
    assert not v.visible('old') and v.visible('withdraw')
    assert v.state_events[0]['linked_source_ids']==['old']
    hits=v.state_evidence(control,[hit(sources[0])])
    assert len(hits)==1 and control in hits[0].content
    assert 'dislike' not in hits[0].content

def test_explicit_attribute_correction_binds_subject_attribute_time_source():
    sources=[source('old','My preferred communication channel is email.'),
             source('new','My preferred communication channel is Slack.',seq=2),
             source('other','My preferred beverage is tea.',seq=3)]
    v=Visibility('alice',sources)
    assert not v.visible('old') and v.visible('new') and v.visible('other')
    event=v.state_events[0]
    assert event['attribute']=='communication channel' and event['subject']=='alice'
    assert event['source_id']=='new' and event['sequence']==2

@pytest.mark.parametrize('control,role',[
    ('Forget that I enjoy tea.','assistant'),
    ('She said "forget that I enjoy tea".','user'),
    ('For this task, pretend I prefer coffee.','user'),
    ('Could you forget something about tea?','user'),
])
def test_quoted_assistant_hypothetical_or_unknown_is_not_state_update(control,role):
    v=Visibility('alice',[source('old','I enjoy tea.'),source('not-control',control,role,seq=2)])
    assert v.visible('old') and not v.state_events

def test_secret_control_remains_hidden_and_no_cross_user_scope():
    v=Visibility('alice',[source('old','My password is CEDAR-447.'),source('secret','Forget my password.',seq=2)])
    assert not v.visible('old') and not v.visible('secret')
    with pytest.raises(ValueError):Visibility('bob',list(v.sources.values()))

def test_third_person_same_value_is_not_subject_update():
    sources=[source('other','Your friend Bob enjoys tea.',role='assistant'),
             source('mine','I enjoy tea.'),source('withdraw','Forget that I enjoy tea.',seq=2)]
    v=Visibility('alice',sources)
    assert v.visible('other') and not v.visible('mine')

def test_reaffirmation_does_not_permanently_blacklist_value():
    sources=[source('old','I enjoy tea.'),source('withdraw','Forget that I enjoy tea.',seq=2),
             source('stale','You enjoy tea.',role='assistant',seq=3),
             source('new','I enjoy tea.',seq=4)]
    v=Visibility('alice',sources)
    assert not v.visible('old') and not v.visible('stale') and v.visible('new')

def test_correction_history_retains_update_context_but_withdrawal_still_private():
    sources=[source('old','My preferred beverage is coffee.'),
             source('new','My preferred beverage is tea.',seq=2)]
    v=Visibility('alice',sources,query='What was my preferred beverage before the change?')
    assert v.visible('old') and v.visible('new')
    assert {s for h in v.state_evidence('beverage',[hit(sources[0])]) for s in h.source_ids}=={'old','new'}
    withdrawn=Visibility('alice',[source('old','I enjoy tea.'),source('withdraw','Forget that I enjoy tea.',seq=2)],
                         query='What did I previously enjoy?')
    assert not withdrawn.visible('old')

def test_prefill_is_whole_legal_counted_observation_with_no_dispatch():
    sources=[source('s1','orchid one'),source('s2','x'*10000),source('s3','orchid three')]
    tools=view(sources);tools.prefill([hit(s) for s in sources])
    assert [s['source_id'] for s in tools.prefilled]==['s1','s3']
    assert tools.used_chars==len('orchid oneorchid three') and not tools.actions
    assert all(s['span']==[0,s['source_length']] for s in tools.prefilled)
    tools.execute('finish_evidence',{'source_ids':['s1']})
    after=tools.execute('read_source',{'source_id':'s3'})
    assert after['status']=='skipped_after_finish' and len(tools.actions)==1

def test_finish_reservation_does_not_execute_over_limit_or_invent_selection():
    tools=view([source('s1','orchid one')]);tools.max_tools=2
    tools.execute('grep',{'pattern':'orchid'})
    assert tools.execute('read_source',{'source_id':'s1'})['status']=='finish_required'
    assert tools.selected is None and len(tools.actions)==1
    tools.execute('finish_evidence',{'source_ids':['s1']})
    assert len(tools.actions)==2

def test_foreign_and_unobserved_sources_still_rejected():
    tools=view([source('s1','orchid one')])
    with pytest.raises(ValueError,match='not available'):tools.execute('read_source',{'source_id':'foreign'})
    with pytest.raises(ValueError,match='unobserved'):tools.execute('finish_evidence',{'source_ids':['s1']})

def body(actions=(),text=None):
    message={'role':'assistant','content':text}
    if actions:message['tool_calls']=[{'id':'call-'+str(i),'type':'function',
        'function':{'name':name,'arguments':json.dumps(args)}} for i,(name,args) in enumerate(actions)]
    return {'id':'fixture','object':'chat.completion','created':0,'model':'gpt-4o-mini',
        'choices':[{'index':0,'finish_reason':'tool_calls' if actions else 'stop','message':message}],
        'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}

class Provider:
    def __init__(self,mode):self.calls=[];self.mode=mode
    def chat(self,payload,*,scope,finalized=False):
        assert not finalized, 'No model request is allowed after validated finish'
        self.calls.append(payload);n=len(self.calls)
        if self.mode=='early_stop' and n==1:return body(text='Evidence found.')
        if self.mode=='never_finish':return body(text='Evidence found.')
        if self.mode=='last_step' and n<6:return body([('read_source',{'source_id':'s1'})])
        if self.mode=='foreign':return body([('read_source',{'source_id':'foreign'})])
        actions=[('finish_evidence',{'source_ids':['s1']})]
        if self.mode=='same_batch':actions.append(('read_source',{'source_id':'foreign'}))
        return body(actions)

@pytest.mark.skipif(not RUNTIME.exists(),reason='pinned SDK required')
@pytest.mark.parametrize('mode,expected_calls',[('finish',1),('early_stop',2),('same_batch',1),('last_step',6)])
def test_official_sdk_terminal_reminder_and_final_slot(tmp_path,mode,expected_calls):
    provider=Provider(mode);pool=NativeHarnessPool(max_workers=1)
    harness=PersistentNativeHarness(provider=provider,pool=pool,runtime_dir=RUNTIME,artifact_dir=tmp_path,timeout_s=30)
    tools=view([source('s1','orchid one')]);tools.prefill([hit(source('s1','orchid one'))])
    try:
        result=harness.run(tools)
        assert result['selected']==['s1'] and len(provider.calls)==expected_calls
        if mode=='same_batch':assert tools.control_events[0]['status']=='skipped_after_finish'
        if mode=='last_step':assert provider.calls[-1]['tool_choice']['function']['name']=='finish_evidence'
    finally:pool.close()

@pytest.mark.skipif(not RUNTIME.exists(),reason='pinned SDK required')
@pytest.mark.parametrize('mode',['never_finish','foreign'])
def test_official_sdk_does_not_forge_finish_or_ignore_scope_failure(tmp_path,mode):
    provider=Provider(mode);pool=NativeHarnessPool(max_workers=1)
    harness=PersistentNativeHarness(provider=provider,pool=pool,runtime_dir=RUNTIME,artifact_dir=tmp_path,timeout_s=30)
    tools=view([source('s1','orchid one')]);tools.prefill([hit(source('s1','orchid one'))])
    try:
        with pytest.raises(RuntimeError):harness.run(tools)
        assert tools.selected is None and len(provider.calls)<=6
    finally:pool.close()

def test_no_total_cap_means_accounting_continues_and_pause_still_applies(tmp_path):
    key=tmp_path/'key';key.write_text('fixture')
    provider=NativeChatProvider(tmp_path/'calls.sqlite3',key_file=key,max_calls=None,max_output_tokens=None,
        max_input_estimate=None,transport=lambda *a:(200,json.dumps(body(text='done')).encode()))
    payload={'model':'gpt-4o-mini','max_tokens':100,'messages':[{'role':'user','content':'fixture'}]}
    try:
        for i in range(4):provider.chat(payload,scope={'user_id':str(i)})
        assert provider.admission_totals()=={'calls':4,'output_reservation':400,'input_estimate':4*len(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())}
        (tmp_path/'PAUSE').write_text('manual control')
        with pytest.raises(MiniProxyError,match='control'):provider.chat(payload,scope={'user_id':'new'})
        assert provider.admission_totals()['calls']==4
    finally:provider.close()
