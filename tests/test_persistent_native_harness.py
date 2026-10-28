from pathlib import Path
from types import SimpleNamespace
import json
import threading
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import os

from agent_memory.deepseek_memory import QueryTools
from agent_memory.domain import SearchRequest
from agent_memory.evidence_safety import Visibility
from agent_memory.persistent_native_harness import NativeHarnessPool, PersistentNativeHarness

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path(os.environ.get('AM_TEST_RUNTIME', ROOT/'data/deepseek-runtime'))
pytestmark=pytest.mark.skipif(not (RUNTIME/'node_modules/@deepseek-ai/dsh-sdk-client/package.json').exists(),reason='pinned official SDK not installed')


class ScriptedProvider:
    def __init__(self):self.calls=[];self.lock=threading.Lock()
    def chat(self,payload,*,scope,finalized=False):
        with self.lock:
            count=sum(c['scope']['user_id']==scope['user_id'] for c in self.calls)
            self.calls.append({'scope':scope,'payload':payload,'finalized':finalized})
        message={'role':'assistant','content':None}
        if not finalized:
            name='grep' if count==0 else 'finish_evidence'
            args={'pattern':'orchid'} if count==0 else {'source_ids':[scope['user_id']+'-source']}
            message['tool_calls']=[{'id':'tool-'+str(count),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]
        return {'id':'fixture-'+str(len(self.calls)),'object':'chat.completion','created':0,'model':'gpt-4o-mini',
            'choices':[{'index':0,'finish_reason':'stop' if finalized else 'tool_calls','message':message}],
            'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}


def tools(user):
    sources=[{'id':user+'-source','user_id':user,'session_id':'one','request_id':'r1',
              'position':0,'seq':1,'role':'user','timestamp':None,'content':user+' private orchid marker'}]
    store=SimpleNamespace(all_sources=lambda uid:[dict(s) for s in sources if s['user_id']==uid])
    return QueryTools(SimpleNamespace(_store=store),SearchRequest(user_id=user,query='orchid',top_k=8),Visibility(user,sources))


def test_official_runtime_reuses_process_but_has_fresh_user_sessions(tmp_path):
    provider=ScriptedProvider();pool=NativeHarnessPool(max_workers=1,max_sessions=4)
    harness=PersistentNativeHarness(pool=pool,provider=provider,runtime_dir=RUNTIME,artifact_dir=tmp_path,timeout_s=30)
    try:
        first=harness.run(tools('alice'))
        first_receipt=json.loads((Path(first['root'])/'receipt.json').read_text('utf-8'))
        worker=pool.workers[0];original_chat=provider.chat;stale_rejected=[]
        def probe_stale_capability(payload,*,scope,finalized=False):
            if scope['user_id']=='bob' and not stale_rejected:
                request=urllib.request.Request(worker.url+'/tools',data=b'{"name":"grep","args":{"pattern":"orchid"}}',
                    headers={'Authorization':'Bearer '+worker.token,'X-AM-Session':first_receipt['execution']['session_id'],'Content-Type':'application/json'})
                with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request,timeout=3)
                stale_rejected.append(error.value.code)
            return original_chat(payload,scope=scope,finalized=finalized)
        provider.chat=probe_stale_capability
        second=harness.run(tools('bob'))
        a=json.loads((Path(first['root'])/'receipt.json').read_text('utf-8'))
        b=json.loads((Path(second['root'])/'receipt.json').read_text('utf-8'))
        assert a['execution']['worker_pid']==b['execution']['worker_pid']
        assert not a['execution']['process_reused'] and b['execution']['process_reused']
        assert a['execution']['session_id']!=b['execution']['session_id']
        assert first['selected']==['alice-source'] and second['selected']==['bob-source']
        assert stale_rejected==[403]
        bob=[c for c in provider.calls if c['scope']['user_id']=='bob']
        assert len(bob)==2 and all('alice private orchid marker' not in json.dumps(c['payload']) for c in bob)
        worker=pool.workers[0]
        # An idle/stale capability cannot access the last user's source pool.
        request=urllib.request.Request(worker.url+'/tools',data=b'{"name":"grep","args":{"pattern":"orchid"}}',
            headers={'Authorization':'Bearer '+worker.token,'X-AM-Session':a['execution']['session_id'],'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request,timeout=3)
        assert error.value.code==403
    finally:pool.close()


def test_official_sdk_batches_multiple_reads_in_one_model_step(tmp_path):
    class BatchReads(ScriptedProvider):
        def chat(self,payload,*,scope,finalized=False):
            count=len(self.calls);self.calls.append({'scope':scope,'payload':payload,'finalized':finalized})
            message={'role':'assistant','content':None}
            ids=['alice-source','alice-source2']
            actions=([('grep',{'pattern':'orchid'})] if count==0 else
                     [('read_source',{'source_id':sid}) for sid in ids] if count==1 else
                     [('finish_evidence',{'source_ids':ids})] if count==2 else [])
            if actions:message['tool_calls']=[{'id':'tool-'+str(count)+'-'+str(i),'type':'function',
                'function':{'name':name,'arguments':json.dumps(args)}} for i,(name,args) in enumerate(actions)]
            return {'id':'batch-fixture-'+str(count),'object':'chat.completion','created':0,'model':'gpt-4o-mini',
                'choices':[{'index':0,'finish_reason':'tool_calls' if actions else 'stop','message':message}],
                'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}
    view=tools('alice');original=list(view.visibility.sources.values())[0]
    second={**original,'id':'alice-source2','seq':2,'content':'Another private orchid detail'}
    sources=[original,second];view.visibility=Visibility('alice',sources)
    view.baseline._store.all_sources=lambda uid:[dict(s) for s in sources if s['user_id']==uid]
    provider=BatchReads();pool=NativeHarnessPool(max_workers=1)
    harness=PersistentNativeHarness(pool=pool,provider=provider,runtime_dir=RUNTIME,artifact_dir=tmp_path,timeout_s=30)
    try:
        result=harness.run(view)
        assert result['selected']==['alice-source','alice-source2'] and len(provider.calls)==3
        assert [a['name'] for a in view.actions]==['grep','read_source','read_source','finish_evidence']
    finally:pool.close()


def test_parallel_official_workers_do_not_share_user_conversations(tmp_path):
    provider=ScriptedProvider();pool=NativeHarnessPool(max_workers=2,max_sessions=4)
    harness=PersistentNativeHarness(pool=pool,provider=provider,runtime_dir=RUNTIME,artifact_dir=tmp_path,timeout_s=30)
    try:
        with ThreadPoolExecutor(2) as executor:
            jobs=[executor.submit(harness.run,tools(user)) for user in ['alice','bob']]
            values=[j.result() for j in jobs]
        assert {tuple(v['selected']) for v in values}=={('alice-source',),('bob-source',)}
        assert len(pool.workers)==2
        for call in provider.calls:
            other='bob' if call['scope']['user_id']=='alice' else 'alice'
            assert other+' private orchid marker' not in json.dumps(call['payload'])
    finally:pool.close()


def test_failed_runtime_is_reaped_and_next_user_gets_fresh_process(tmp_path):
    class Failure(ScriptedProvider):
        def chat(self,payload,*,scope,finalized=False):
            if scope['user_id']=='alice':raise RuntimeError('fixture provider failure; no retry')
            return super().chat(payload,scope=scope,finalized=finalized)
    provider=Failure();pool=NativeHarnessPool(max_workers=1,max_sessions=4)
    harness=PersistentNativeHarness(pool=pool,provider=provider,runtime_dir=RUNTIME,artifact_dir=tmp_path,timeout_s=30)
    try:
        with pytest.raises(RuntimeError):harness.run(tools('alice'))
        assert pool.workers==[]
        value=harness.run(tools('bob'))
        receipt=json.loads((Path(value['root'])/'receipt.json').read_text('utf-8'))
        assert value['selected']==['bob-source'] and not receipt['execution']['process_reused']
    finally:pool.close()


def test_timed_out_session_is_not_reused_or_replayed(tmp_path):
    class Slow(ScriptedProvider):
        attempts=0
        def chat(self,payload,*,scope,finalized=False):
            if scope['user_id']=='bob':
                self.attempts+=1;time.sleep(4);raise TimeoutError('fixture held in-flight timeout')
            return super().chat(payload,scope=scope,finalized=finalized)
    provider=Slow();pool=NativeHarnessPool(max_workers=1,max_sessions=4)
    harness=PersistentNativeHarness(pool=pool,provider=provider,runtime_dir=RUNTIME,artifact_dir=tmp_path,timeout_s=2.5)
    try:
        value=harness.run(tools('alice'))
        pid=json.loads((Path(value['root'])/'receipt.json').read_text('utf-8'))['execution']['worker_pid']
        with pytest.raises((RuntimeError,TimeoutError)):harness.run(tools('bob'))
        assert pool.workers==[] and provider.attempts==1
        value=harness.run(tools('charlie'))
        receipt=json.loads((Path(value['root'])/'receipt.json').read_text('utf-8'))
        assert receipt['execution']['worker_pid']!=pid
        assert value['selected']==['charlie-source']
    finally:pool.close()
