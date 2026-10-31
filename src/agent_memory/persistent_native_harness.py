"""Bounded reusable official SDK processes, with a new source capability/session per query.

The official SDK continues to own the agent loop. Failed/deadline runtimes are
retired and never replayed. A worker serves one query at a time, and both model
and tool transports must prove the currently bound opaque session capability.
"""
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import queue
import secrets
import subprocess
import threading
import time
from uuid import uuid4

from .deepseek_memory import NativeHarness, _json, _sse
from .native_chat import native_request_window


@dataclass
class Binding:
    tools: object
    harness: object
    session: str
    root: Path
    deadline: float
    scope: dict
    calls: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    tool_seconds: float = 0.0


class _Worker:
    def __init__(self, harness, deadline):
        self.harness = harness
        self.key = (str(harness.runtime_dir), harness.skill, harness.max_tokens, harness.max_steps, harness.timeout_s,
                    harness.enable_evidence_chain)
        self.root = harness.artifact_dir / ('worker-' + uuid4().hex)
        self.root.mkdir(parents=True)
        self.token = secrets.token_urlsafe(32)
        self.active = None
        self.lock = threading.RLock()
        self.used = 0
        self.retired = False
        self.busy = True
        self.lines = queue.Queue()
        self.proc = None
        worker = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                with worker.lock:
                    binding = worker.active
                    authorized = (binding is not None and not worker.retired and
                        self.headers.get('Authorization') == 'Bearer ' + worker.token and
                        self.headers.get('X-AM-Session') == binding.session)
                if not authorized:
                    self.send_error(403); return
                if time.monotonic() >= binding.deadline:
                    self.send_error(504); return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size < 600000: raise ValueError('request body budget')
                    value = json.loads(self.rfile.read(size))
                    if self.path == '/tools':
                        started = time.monotonic()
                        try: data = _json(binding.tools.execute(value['name'], value['args'])).encode()
                        finally: binding.tool_seconds += time.monotonic()-started
                        kind = 'application/json'
                    elif self.path == '/v1/chat/completions':
                        if binding.tools.current_snapshot() != binding.scope['snapshot']:
                            raise RuntimeError('source visibility changed during native query')
                        if len(binding.calls) >= binding.harness.max_steps:
                            raise RuntimeError('query model step budget exhausted')
                        if len(_json(value).encode()) > 180000: raise ValueError('query context budget exhausted')
                        call = {'request': value}; binding.calls.append(call)
                        with native_request_window(binding.deadline, call.setdefault('timing', {})):
                            body = binding.harness.provider.chat(value, scope=binding.scope,
                                finalized=binding.tools.selected is not None)
                        call['response'] = body
                        data, kind = _sse(body), 'text/event-stream'
                    else:
                        self.send_error(404); return
                    with worker.lock:
                        if worker.active is not binding or worker.retired or time.monotonic() >= binding.deadline:
                            raise RuntimeError('expired source capability; response not delivered')
                    self.send_response(200); self.send_header('Content-Type', kind)
                    self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
                except Exception as exc:
                    binding.errors.append({'path': self.path, 'type': type(exc).__name__, 'reason': str(exc)[:300]})
                    try: self.send_error(502, 'memory capability failed')
                    except OSError: pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.url = url
        code = Path(__file__).resolve().parents[2] / 'tools/deepseek_harness'
        # The same official profile and tools/loop/LLM settings as NativeHarness.
        disabled = ['llm-deepseek', 'subprocess', 'sandbox', 'sandbox-policy', 'pty', 'terminal-bash',
                    'terminal-pwsh', 'persistent-bash', 'persistent-pwsh', 'mcp-resources', 'llm-retry']
        patch = ''.join(f'- id: {name}\n  disabled: true\n' for name in disabled)
        patch += '- id: agent-loop\n  config:\n    maxParallelToolCalls: 1\n- insert:\n'
        patch += "    - id: credentials\n      name: '@deepseek-ai/dsh-credentials-local'\n      config:\n        watch: false\n"
        patch += "    - id: agentmemory-model\n      name: '@deepseek-ai/dsh-llm-pi-ai'\n      config:\n        providers:\n          agentmemory-mini:\n            apiKeyEnv: AM_CAPABILITY\n            api: openai-completions\n"
        patch += f'            baseURL: {url}/v1\n            retryPolicy:\n              mode: normal\n              maxRetries: 0\n'
        patch += '            compat:\n              supportsDeveloperRole: false\n              supportsStore: false\n              maxTokensField: max_tokens\n            models:\n              - id: gpt-4o-mini\n                contextWindow: 128000\n                maxTokens: 4096\n                reasoning: false\n'
        patch += '    - id: memory-tools\n      name: ' + _json((code / 'memory-plugin.mjs').as_posix()) + '\n'
        (self.root/'memory.patch.yml').write_text(patch, 'utf-8')
        (self.root/'home').mkdir()
        config = {'patch':str(self.root/'memory.patch.yml'), 'home':str(self.root/'home'),
                  'max_tokens':harness.max_tokens, 'timeout_ms':int(harness.timeout_s*1000)}
        (self.root/'worker.json').write_text(_json(config), 'utf-8')
        env = {k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('API_KEY','AUTH_TOKEN','ANTHROPIC','OPENAI'))}
        env.update(AM_DSH_RUNTIME=str(harness.runtime_dir), AM_CAPABILITY=self.token, AM_TOOL_URL=url+'/tools',
            AM_GATEWAY_PREFIX=url, AM_POOLED='1', DSH_SYSTEM_PROMPT=harness.skill, DSH_DISABLE_TELEMETRY='1')
        env['AM_CHAIN_TOOL'] = '1' if harness.enable_evidence_chain else '0'
        self.stderr = (self.root/'stderr.log').open('wb')
        started = time.monotonic()
        try:
            self.proc = subprocess.Popen(['node',str(code/'run_worker.mjs'),str(self.root/'worker.json')],
                env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            def reader():
                try:
                    for line in self.proc.stdout:
                        try:self.lines.put(json.loads(line))
                        except ValueError:self.lines.put({'error':'non-JSON worker output'})
                finally:self.lines.put({'error':'official SDK worker exited'})
            threading.Thread(target=reader,daemon=True).start()
            ready=self.receive(deadline)
            if not ready.get('ready'):raise RuntimeError('official SDK worker failed: '+str(ready))
            self.startup_s=time.monotonic()-started
        except BaseException:
            self.close();raise

    def receive(self, deadline):
        return self.lines.get(timeout=max(.001,deadline-time.monotonic()))

    def send(self, value):
        self.proc.stdin.write(_json(value)+'\n');self.proc.stdin.flush()

    def close(self):
        with self.lock:self.retired=True;self.active=None
        if self.proc is not None and self.proc.poll() is None:
            try:self.send({'command':'close'});self.proc.stdin.close()
            except (OSError,ValueError):pass
            try:self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(self.proc.pid),'/T','/F'],capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,timeout=10)
                else:self.proc.kill();self.proc.wait(timeout=10)
        self.server.shutdown();self.server.server_close()
        if hasattr(self,'stderr'):self.stderr.close()


class NativeHarnessPool:
    def __init__(self, max_workers=12, max_sessions=16):
        self.max_workers=max_workers;self.max_sessions=max_sessions
        self.condition=threading.Condition();self.workers=[];self.creating=0;self.closed=False

    def acquire(self, harness, deadline):
        key=(str(harness.runtime_dir),harness.skill,harness.max_tokens,harness.max_steps,harness.timeout_s,
             harness.enable_evidence_chain)
        started=time.monotonic();evicted=None
        with self.condition:
            while True:
                if self.closed:raise RuntimeError('native worker pool closed')
                for worker in self.workers:
                    if not worker.busy and not worker.retired and worker.key==key:
                        worker.busy=True;return worker,time.monotonic()-started,False
                idle=next((w for w in self.workers if not w.busy),None)
                if len(self.workers)+self.creating < self.max_workers or idle is not None:
                    if idle is not None:self.workers.remove(idle);evicted=idle
                    self.creating+=1;break
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('native worker allocation deadline')
                self.condition.wait(remaining)
        try:
            if evicted:evicted.close()
            worker=_Worker(harness,deadline)
            with self.condition:self.workers.append(worker)
            return worker,time.monotonic()-started,True
        finally:
            with self.condition:self.creating-=1;self.condition.notify_all()

    def release(self, worker, healthy):
        with self.condition:
            retire=not healthy or worker.used>=self.max_sessions or self.closed
            if not retire:
                worker.busy=False;self.condition.notify_all()
        if retire:
            # Reap before making the allocation available to another query.
            worker.close()
            with self.condition:
                if worker in self.workers:self.workers.remove(worker)
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed=True;workers=list(self.workers);self.workers.clear();self.condition.notify_all()
        for worker in workers:worker.close()


class PersistentNativeHarness(NativeHarness):
    def __init__(self, *, pool, **kwargs):
        super().__init__(**kwargs);self.pool=pool

    def run(self, tools):
        if self.model!='gpt-4o-mini':raise ValueError('persistent native transport is pinned to the Mini profile; use standalone teacher harness')
        started=time.monotonic();deadline=started+self.timeout_s
        session='am-'+uuid4().hex;root=self.artifact_dir/session;root.mkdir()
        scope={'track':'textual','user_id':tools.request.user_id,'snapshot':tools.current_snapshot(),**self.identity}
        binding=Binding(tools,self,session,root,deadline,scope)
        worker=None;healthy=False;execution={}
        try:
            worker,wait,new=self.pool.acquire(self,deadline)
            execution={'worker_pid':worker.proc.pid,'worker_home':str(worker.root/'home'),
                'queue_and_startup_s':wait,'worker_startup_s':worker.startup_s if new else 0.,
                'process_reused':not new,'session_id':session}
            with worker.lock:
                if worker.active is not None:raise RuntimeError('native worker already has an active capability')
                worker.active=binding
            worker.send({'command':'run','prompt':_json({'query':tools.request.query,'options':tools.request.options,'top_k':tools.request.top_k}),
                'session_id':session,'result':str(root/'result.json'),'timeout_ms':max(1,int((deadline-time.monotonic())*1000))})
            result=worker.receive(deadline+1)
            if result.get('session_id')!=session or not result.get('complete'):
                raise RuntimeError('official native session failed: '+str(result))
            execution.update(result)
            if tools.selected is None:raise RuntimeError('native harness ended without finish_evidence')
            if binding.errors:raise RuntimeError('native capability failed: '+_json(binding.errors))
            healthy=True;worker.used+=1
            return {'selected':tools.selected,'run_id':session,'root':str(root)}
        finally:
            if worker:
                with worker.lock:worker.active=None
                self.pool.release(worker,healthy)
            (root/'receipt.json').write_text(_json({'scope':scope,'calls':binding.calls,'tools':tools.actions,
                'errors':binding.errors,'selected':tools.selected,'latency_s':time.monotonic()-started,
                'transport':'official-sdk-persistent-fresh-session-v1','execution':execution,
                'tool_execute_s':binding.tool_seconds}), 'utf-8')
