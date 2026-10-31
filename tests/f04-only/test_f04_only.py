"""New f04-only contract tests: real native backend, synthetic1024, no network."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, Lock
import hashlib
import http.client
import importlib
import json
import math
import re
import socket
import sys
import urllib.request

import httpx
import pytest
from fastapi.testclient import TestClient

CANDIDATE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CANDIDATE / 'src'))
from agent_memory.f04_submission import build_app, ExactQuerySingleFlight
from agent_memory.bailian_embeddings import BailianEmbeddings
from agent_memory.domain import SearchRequest
from agent_memory.method_memmachine import _native
from agent_memory.method_memmachine_w1 import MemMachineW1Memory
from agent_memory.complementary_tail import select_tail

HEADERS = {'Authorization': 'Bearer synthetic-f04-only-token'}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []
    def denied(*args, **kwargs):
        attempts.append(True)
        raise AssertionError('offline f04-only test attempted network')
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', denied)
    monkeypatch.setattr(http.client.HTTPConnection, 'connect', denied)
    monkeypatch.setattr(http.client.HTTPSConnection, 'connect', denied)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', denied)
    yield
    assert not attempts


class Synthetic1024Transport:
    def __init__(self):
        self.calls = []
        self.lock = Lock()
    def __call__(self, request, timeout):
        payload = json.loads(request.data)
        with self.lock:
            self.calls.append(payload)
        rows = []
        for i, text in enumerate(payload['input']['texts']):
            vector = [0.0] * 1024
            for token in re.findall(r'\w+', text.lower()):
                vector[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 1024] += 1.0
            vector[0] += .01
            rows.append({'text_index': i, 'embedding': vector})
        return 200, json.dumps({'output': {'embeddings': rows}, 'usage': {'total_tokens': 1},
                                'request_id': 'synthetic-f04-only'}).encode()


def make_app(root, transport=None):
    root.mkdir(parents=True, exist_ok=True)
    key = root / 'synthetic-embedding.key'
    key.write_text('synthetic-only-no-real-provider', encoding='utf-8')
    profile = {'provider': 'bailian', 'model': 'text-embedding-v4', 'dimensions': 1024,
        'endpoint': 'https://synthetic.maas.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding',
        'api_key_file': str(key), 'cache_dir': str(root / 'embedding-cache')}
    transport = transport or Synthetic1024Transport()
    embedder = BailianEmbeddings(profile, user_confirmed=True, max_attempts=1, transport=transport)
    app = build_app(db_path=root / 'memory.sqlite3', api_key='synthetic-f04-only-token',
                    embedder=embedder, run_dir=root / 'run')
    return app, app.state.memory_service.memory, embedder, transport


def payload(user='f04-alice', rid='r1', count=124):
    return {'user_id': user, 'request_id': rid, 'session_id': 'fictional-s1', 'messages': [
        {'role': 'user' if i % 2 == 0 else 'assistant',
         'timestamp': 1704067200000 + i * 60000 if i % 3 == 0 else None,
         'content': f'Fictional cobalt archive entry {i:03d} holds {400+i} shells.\n  原文  exact.'}
        for i in range(count)]}


def query(user='f04-alice', k=100, text='fictional cobalt archive shells'):
    return {'user_id': user, 'query': text, 'top_k': k}


def check_original(hit, sources):
    match = re.fullmatch(r'\[source=([^;]+); role=([^;]+); timestamp=([^\]]+)\]\n(.*)',
                         hit['content'], re.DOTALL)
    assert match
    sid, role, date, body = match.groups()
    row = sources[sid]
    expected = ('unknown' if row['timestamp'] is None else
                datetime.fromtimestamp(row['timestamp']/1000, timezone.utc).isoformat())
    assert role == row['role'] and date == expected and body == row['content']


def test_candidate_imports_real_native_and_direct_assembly(tmp_path):
    assert Path.cwd().resolve() == CANDIDATE
    assert _native().EventMemory.__module__.startswith('memmachine_server.')
    app, service, _, _ = make_app(tmp_path)
    with TestClient(app):
        assert type(app.state.memory_service) is ExactQuerySingleFlight
        assert type(service) is MemMachineW1Memory
        assert service.config['max_results'] == 16
        manifest = app.state.version_manifest
        assert manifest['f04'] and manifest['static_score']['coverage_weight'] == .14
        assert manifest['static_score']['redundancy_weight'] == .12
        assert manifest['llm_calls_in_add'] == manifest['llm_calls_in_search'] == 0
        assert manifest['evidence_char_budget'] is None
        assert all(manifest[x] is False for x in ('H', 'F', 'S14', 'KD', 'AgentDistill'))
        flags = manifest['method']['w1_screening']
        assert flags['complementary_tail']
        assert not any(flags[x] for x in ('coverage_rescue', 'combined_tail', 'rule_quota',
                                         'numeric_anchor_boost', 'date_annotations'))
        assert not manifest['method']['r_recipe']['session_rrf']
        assert not manifest['method']['r_recipe']['temporal_soft_boost']
        assert not (tmp_path / 'run/native-calls.sqlite3').exists()
        for name, module in tuple(sys.modules.items()):
            if name.startswith('agent_memory') and getattr(module, '__file__', None):
                assert Path(module.__file__).resolve().is_relative_to(CANDIDATE / 'src')
        assert not any(name in sys.modules for name in (
            'agent_memory.estar_submission', 'agent_memory.native_chat',
            'agent_memory.persistent_native_harness', 'agent_memory.hybrid_retrieval',
            'agent_memory.deepseek_memory', 'agent_memory.evidence_safety'))


def test_http_native_top_k_core_tail_and_verbatim(tmp_path):
    app, service, embedder, transport = make_app(tmp_path)
    with TestClient(app) as client:
        assert client.post('/add', headers=HEADERS, json=payload()).status_code == 200
        sources = {s['id']: s for s in service._store.all_sources('f04-alice')}
        for k in (1, 16, 32, 100):
            request = SearchRequest.model_validate(query(k=k))
            core, rows = service._search_core(request)
            full = service._with_expand(request, core, rows)
            assert len(core) == min(k, 16)
            assert len(full) == k and full[:len(core)] == core
            assert all(h.kind == 'complementary_tail' for h in full[len(core):])
            assert all(h.id != h.source_ids[0] for h in core)
            response = client.post('/search', headers=HEADERS, json=query(k=k))
            assert response.status_code == 200
            hits = response.json()['data']
            assert len(hits) == k and len({h['id'] for h in hits}) == k
            assert hits == [{'id': h.id, 'content': h.content, 'score': h.score} for h in full]
            for hit in hits:
                check_original(hit, sources)
        assert len([p for p in transport.calls if p['parameters']['text_type'] == 'query']) == 1
        for bad in (0, 101, True):
            assert client.post('/search', headers=HEADERS, json=query(k=bad)).status_code == 422


def test_auth_idempotence_conflicts_and_isolation(tmp_path):
    app, service, _, transport = make_app(tmp_path)
    original = payload(count=2)
    with TestClient(app) as client:
        for endpoint, body in (('/add', original), ('/search', query())):
            assert client.post(endpoint, json=body).status_code == 401
            assert client.post(endpoint, headers={'x-api-key': 'wrong'}, json=body).status_code == 401
        assert transport.calls == []
        receipt = client.post('/add', headers=HEADERS, json=original)
        assert receipt.status_code == 200
        count = len(transport.calls)
        assert client.post('/add', headers=HEADERS, json=original).json() == receipt.json()
        variants = [{**original, 'session_id': 'changed'}]
        for field, value in [('content', 'changed body'), ('role', 'assistant'), ('timestamp', 1)]:
            variants.append({**original, 'messages': [{**original['messages'][0], field: value},
                                                     original['messages'][1]]})
        for body in variants:
            assert client.post('/add', headers=HEADERS, json=body).status_code == 409
        assert len(transport.calls) == count
        bob = payload(user='f04-alice-suffix', count=1)
        bob['messages'][0]['content'] = 'Fictional BOB-ONLY-913 archive holds 91 stones.'
        assert client.post('/add', headers=HEADERS, json=bob).status_code == 200
        alice = client.post('/search', headers=HEADERS, json=query()).json()
        b = client.post('/search', headers=HEADERS, json=query(user='f04-alice-suffix')).json()
        assert 'BOB-ONLY-913' not in json.dumps(alice) and 'BOB-ONLY-913' in json.dumps(b)
        assert '400 shells' not in json.dumps(b)
        count = len(transport.calls)
        assert client.post('/search', headers=HEADERS, json=query(user='empty-user')).json() == {'data': []}
        assert len(transport.calls) == count


def test_repeat_reload_and_options_use_durable_exact_embedding_cache(tmp_path):
    app, _, _, transport = make_app(tmp_path)
    request = payload(count=4)
    with TestClient(app) as client:
        receipt = client.post('/add', headers=HEADERS, json=request).json()
        first = client.post('/search', headers=HEADERS, json=query()).json()
        count = len(transport.calls)
        assert client.post('/search', headers=HEADERS, json=query()).json() == first
        assert client.post('/add', headers=HEADERS, json=request).json() == receipt
        assert len(transport.calls) == count
        for options in (['blue', 'green'], ['green', 'blue']):
            q = {**query(), 'options': options}
            assert client.post('/search', headers=HEADERS, json=q).status_code == 200
        assert len(transport.calls) == count + 2
    reopened, _, _, second = make_app(tmp_path)
    with TestClient(reopened) as client:
        assert client.post('/add', headers=HEADERS, json=request).json() == receipt
        assert client.post('/search', headers=HEADERS, json=query()).json() == first
        assert second.calls == []


def test_cold_16_concurrent_searches_one_embedding_attempt(tmp_path):
    app, _, embedder, transport = make_app(tmp_path)
    barrier = Barrier(16)
    with TestClient(app) as client:
        assert client.post('/add', headers=HEADERS, json=payload(count=20)).status_code == 200
        before = len(embedder.store.list_attempts())
        def run(_):
            barrier.wait(timeout=20)
            return client.post('/search', headers=HEADERS, json=query())
        with ThreadPoolExecutor(max_workers=16) as pool:
            replies = list(pool.map(run, range(16)))
        assert all(r.status_code == 200 for r in replies)
        assert all(r.json() == replies[0].json() for r in replies)
        assert len(embedder.store.list_attempts()) == before + 1
        before = len(transport.calls)
        with ThreadPoolExecutor(max_workers=16) as pool:
            replies = list(pool.map(lambda _: client.post('/search', headers=HEADERS, json=query()), range(16)))
        assert all(r.status_code == 200 for r in replies) and len(transport.calls) == before


def test_add_invalidates_source_view_without_reembedding_query(tmp_path):
    app, _, _, transport = make_app(tmp_path)
    with TestClient(app) as client:
        assert client.post('/add', headers=HEADERS, json=payload(count=1)).status_code == 200
        assert len(client.post('/search', headers=HEADERS, json=query()).json()['data']) == 1
        extra = payload(rid='r2', count=1)
        extra['messages'][0]['content'] = 'Fictional cobalt archive NEW-SOURCE immutable body.'
        assert client.post('/add', headers=HEADERS, json=extra).status_code == 200
        count = len(transport.calls)
        reply = client.post('/search', headers=HEADERS, json=query())
        assert reply.status_code == 200 and 'NEW-SOURCE' in reply.text
        assert len(reply.json()['data']) == 2 and len(transport.calls) == count


def test_static_initial_context_scores_and_original_tie_order():
    def row(sid, body):
        return {'id': sid, 'content': body, 'role': 'user', 'timestamp': None}
    rows = [row('a', 'cobalt amber'), row('b', 'cobalt amber cobalt'), row('c', 'cobalt')]
    actual = select_tail(rows, 'cobalt amber', ['seed'], limit=3)
    assert [h.source_ids[0] for h in actual] == ['a', 'b', 'c']
    assert [h.score for h in actual] == pytest.approx([.14, .14, .07])
    assert select_tail([row('r', 'cobalt amber cobalt')], 'cobalt amber', ['cobalt amber'], limit=1)[0].score == pytest.approx(-.12)


def test_unknown_embedding_outcome_held_without_replay(tmp_path):
    def unknown(request, timeout):
        raise TimeoutError('synthetic transport outcome unknown')
    app, _, embedder, _ = make_app(tmp_path, transport=unknown)
    with TestClient(app, raise_server_exceptions=False) as client:
        body = payload(count=1)
        assert client.post('/add', headers=HEADERS, json=body).status_code == 500
        assert client.post('/add', headers=HEADERS, json=body).status_code == 500
        attempts = embedder.store.list_attempts()
        assert len(attempts) == 1
        assert attempts[0]['response']['category'] == 'transport_unknown'
        assert client.post('/search', headers=HEADERS, json=query()).json() == {'data': []}
