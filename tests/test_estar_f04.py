"""Offline f04 compatibility checks, using the candidate and real event storage."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import http.client
import importlib
import json
from pathlib import Path
import socket
import sys
from threading import Barrier, Event
import urllib.request

import httpx
import pytest
from fastapi.testclient import TestClient

# Resolve from this candidate, even when the parent workspace has an installed
# agent_memory package. Reject already-imported parent code instead of mixing it.
CANDIDATE = Path(__file__).resolve().parents[1]
CANDIDATE_SRC = CANDIDATE / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
for _name, _module in tuple(sys.modules.items()):
    if _name == "agent_memory" or _name.startswith("agent_memory."):
        _file = getattr(_module, "__file__", None)
        if _file is not None and not Path(_file).resolve().is_relative_to(CANDIDATE_SRC):
            raise AssertionError(f"non-candidate import: {_name}: {_file}")

from agent_memory import estar_submission  # noqa: E402
from agent_memory.anchor_channel import _render_source  # noqa: E402
from agent_memory.bailian_embeddings import BailianEmbeddings  # noqa: E402
from agent_memory.bounded_expand import expand_source_ids  # noqa: E402
from agent_memory.complementary_tail import select_tail  # noqa: E402
from agent_memory.deepseek_memory import EnsembleMemory, fuse_sources  # noqa: E402
from agent_memory.domain import AddRequest, Evidence, Message, SearchRequest  # noqa: E402
from agent_memory.evidence_safety import Visibility  # noqa: E402
from agent_memory.method_memmachine import MemMachineMemory, _native  # noqa: E402
from agent_memory.method_memmachine_w1 import MemMachineW1Memory  # noqa: E402
from test_estar_submission import Harness, SyntheticEmbedder as BaseEmbedder  # noqa: E402
from test_submission import HEADERS  # noqa: E402


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    """Fail even if a branch catches an attempted provider/network operation."""
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("unexpected provider, transport, or native worker operation")
        raise AssertionError("offline fixture attempted a real provider/network operation")

    monkeypatch.setattr(estar_submission.NativeChatProvider, "__init__", forbidden)
    monkeypatch.setattr(estar_submission.NativeHarnessPool, "__init__", forbidden)
    monkeypatch.setattr(estar_submission.PersistentNativeHarness, "run", forbidden)
    monkeypatch.setattr(BailianEmbeddings, "embed_documents", forbidden)
    monkeypatch.setattr(BailianEmbeddings, "embed_query", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", forbidden)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    yield
    assert attempts == [], "a failed network attempt must not become an accepted fallback"


class CountingEmbedder(BaseEmbedder):
    def __init__(self):
        self.document_batches = []
        self.query_texts = []

    def embed_documents(self, texts):
        texts = list(texts)
        self.document_batches.append(texts)
        return super().embed_documents(texts)

    def embed_query(self, text):
        self.query_texts.append(text)
        return super().embed_query(text)


class RecordingHarness(Harness):
    def __init__(self):
        super().__init__()
        self.requests = []

    def run(self, tools):
        self.requests.append(tools.request.model_dump())
        return super().run(tools)


class EmptyHarness(Harness):
    def run(self, tools):
        self.calls += 1
        tools.execute("finish_evidence", {"source_ids": []})
        return {"selected": tools.selected, "run_id": "offline-empty-selection"}


class FailedHarness(Harness):
    def run(self, tools):
        self.calls += 1
        raise TimeoutError("synthetic H timeout; preserve R/F fallback")


def make_app(tmp_path, harness=None, embedder=None):
    harness = harness if harness is not None else RecordingHarness()
    embedder = embedder if embedder is not None else CountingEmbedder()
    app = estar_submission.build_app(
        db_path=tmp_path / "db", api_key="synthetic-key", embedder=embedder,
        run_dir=tmp_path / "run", harness=harness,
    )
    return app, app.state.memory_service.memory, harness, embedder


def payload(user="alice", rid="r1", messages=None):
    return {
        "user_id": user, "request_id": rid, "session_id": "s1",
        "messages": messages if messages is not None else [
            {"role": "user", "content": "fictional cobalt archive holds 37 shells."},
        ],
    }


def query(user="alice", k=100, text="fictional cobalt archive shells"):
    return {"user_id": user, "query": text, "top_k": k}


def records(service):
    return [service.ledger.get_result(task["key"])["result"]
            for task in service.ledger.list_tasks(status="succeeded")]


def source(sid, text, role="user"):
    return {"id": sid, "content": text, "user_id": "alice", "role": role,
            "timestamp": None, "session_id": "s1", "request_id": "r1", "position": 0}


def assert_original_hit(hit, sources):
    sid = hit["id"].removeprefix("ensemble_")
    row = sources[sid]
    date = ("unknown" if row["timestamp"] is None else
            datetime.fromtimestamp(row["timestamp"] / 1000, timezone.utc).isoformat())
    assert hit["id"] == "ensemble_" + sid
    assert hit["content"] == (
        f"[source={sid}; role={row['role']}; timestamp={date}]\n" + row["content"]
    )


def test_candidate_import_provenance_and_native_backend():
    assert Path.cwd().resolve() == CANDIDATE
    for name in ("agent_memory", "agent_memory.estar_submission", "agent_memory.api",
                 "agent_memory.domain", "agent_memory.method_store", "agent_memory.deepseek_memory",
                 "agent_memory.method_memmachine", "agent_memory.method_memmachine_w1",
                 "agent_memory.complementary_tail", "agent_memory.evidence_safety"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(CANDIDATE_SRC), name
    for name in ("test_estar_submission", "test_submission"):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(CANDIDATE / "tests")
    assert _native().EventMemory.__module__.startswith("memmachine_server.")
    assert "agent_memory.hybrid_retrieval" not in sys.modules
    assert "agent_memory.r_source_packing" not in sys.modules


def test_direct_w1_assembly_manifest_frozen_s14_and_no_extra_tuning(tmp_path):
    app, service, harness, _ = make_app(tmp_path)
    with TestClient(app):
        baseline = service.baseline
        assert type(service) is EnsembleMemory
        assert type(baseline) is MemMachineW1Memory
        assert isinstance(baseline, MemMachineMemory)
        assert not hasattr(baseline, "native")
        assert service.harness is harness
        assert service.evidence_char_budget == 60000
        m = json.loads((tmp_path / "run/version-manifest.json").read_text(encoding="utf-8"))
        assert m == app.state.version_manifest
        assert m["version"] == estar_submission.VERSION == "textual-f04-estar-s14-v1.0"
        assert all(m[key] is True for key in ("H", "F", "S14", "f04"))
        assert all(m[key] is False for key in ("KD", "AgentDistill", "deep_tail",
                                             "extra_raw_search_audit", "final_core_protection",
                                             "final_100_item_guarantee"))
        assert m["method"] == service.identity
        assert m["method"]["R"]["r_recipe"] == {
            "anchor_append_m": 8, "neighbor_budget": 8, "neighbor_score_ratio": 0.3,
            "session_rrf": False, "temporal_soft_boost": False,
        }
        flags = m["method"]["R"]["w1_screening"]
        assert flags["complementary_tail"] is True
        assert all(flags[key] is False for key in (
            "coverage_rescue", "combined_tail", "rule_quota", "numeric_anchor_boost", "date_annotations"))
        assert flags["complementary_weights"] == [0.14, 0.12]
        assert m["native_core_cap"] == baseline.config["max_results"] == 16
        assert m["requested_top_k_max"] == 100 and baseline.config["expand_context"] == 0
        assert m["method"]["fusion"] == "source-rrf60-v1"
        assert m["f04_scoring"] == "static coverage and redundancy against initial R context"
        assert m["final_delivery"] == (
            "unchanged source RRF60, visibility rehydration, 60000-char whole-source budget")
        assert m["f04_disabled_tuning"] == [
            "rule_quota", "numeric_anchor_boost", "coverage_rescue", "date_annotations"]
        skill = CANDIDATE / "skills/S14.txt"
        assert hashlib.sha256(skill.read_bytes()).hexdigest() == m["S14_sha256"] == (
            "cb745515471e0a5828e99c390ffb3c9fb5f22b66809a0f67044b1e45af764f10")
        assert not harness.enable_evidence_chain and harness.calls == 0
        assert not (tmp_path / "run/native-calls.sqlite3").exists()


@pytest.mark.parametrize("name,digest", [
    ("method_memmachine_w1.py", "820c061f578fd543c609adb54bfdfd559d08a748373f33236fa48aa9ef8c3ed0"),
    ("complementary_tail.py", "307b70f21ca800f0bd94f80efb62dc9986c6082f04741345e5531bb45f7fc3ce"),
    ("bounded_expand.py", "742aff118095d98fe3e2a4fa67e13ccd1eab977c40d0669984689911bc722614"),
    ("coverage_rescue.py", "217ac1ec8b5c70ceec5c865648cfc92a3c58079a212a55ddb7a724a6a5dc49a9"),
    ("date_annotations.py", "7a68bb52042428b9189366ca64d83cdcd9396936377d61de8c809ad62e7326b8"),
])
def test_tested_r_modules_keep_frozen_provenance(name, digest):
    assert hashlib.sha256((CANDIDATE_SRC / "agent_memory" / name).read_bytes()).hexdigest() == digest


def test_real_event_low_top_k_original_roles_timestamps_and_f04_tail(tmp_path):
    app, service, harness, _ = make_app(tmp_path)
    messages = [{
        "role": "user" if i % 2 == 0 else "assistant",
        "timestamp": 1704067200000 + i * 60000 if i % 3 == 0 else None,
        "content": f"fictional cobalt archive entry {i:03d} holds {400 + i} shells.\n  原文  stays exact.",
    } for i in range(112)]
    with TestClient(app) as client:
        assert client.post("/add", headers=HEADERS, json=payload(messages=messages)).status_code == 200
        rows = service.baseline._store.all_sources("alice")
        by_id = {s["id"]: s for s in rows}
        assert len(rows) == 112
        for k in (1, 16, 32, 100):
            request = SearchRequest.model_validate(query(k=k))
            core, bound_sources = service.baseline._search_core(request)
            r_hits = service.baseline._with_expand(request, core, bound_sources)
            assert len(core) == min(k, 16)
            assert len(r_hits) == k and r_hits[:len(core)] == core
            assert all(h.id != h.source_ids[0] for h in core)
            assert all(h.kind == "complementary_tail" for h in r_hits[len(core):])
            reply = client.post("/search", headers=HEADERS, json=query(k=k))
            assert reply.status_code == 200
            hits = reply.json()["data"]
            assert len(hits) == k and len({h["id"] for h in hits}) == k
            for hit in hits:
                assert_original_hit(hit, by_id)
            assert client.post("/search", headers=HEADERS, json=query(k=k)).json() == reply.json()
        full = client.post("/search", headers=HEADERS, json=query()).json()["data"]
        assert any("role=user;" in h["content"] for h in full)
        assert any("role=assistant;" in h["content"] for h in full)
        assert any("timestamp=unknown]" in h["content"] for h in full)
        assert any("timestamp=2024-01-01T" in h["content"] for h in full)
        for k in (0, 101):
            assert client.post("/search", headers=HEADERS, json=query(k=k)).status_code == 422
        assert harness.calls == 4
        assert all(row["failures"] == {} for row in records(service))


def test_add_idempotence_conflicts_cross_user_and_durable_exact_cache(tmp_path):
    app, service, harness, embedder = make_app(tmp_path)
    original = payload(messages=[
        {"role": "user", "content": "fictional cobalt user keeps 37 shells.", "timestamp": 1704067200000},
        {"role": "assistant", "content": "fictional cobalt assistant suggests 91 shells.", "timestamp": None},
    ])
    with TestClient(app) as client:
        receipt = client.post("/add", headers=HEADERS, json=original)
        assert receipt.status_code == 200
        batches = len(embedder.document_batches)
        assert client.post("/add", headers=HEADERS, json=original).json() == receipt.json()
        assert len(embedder.document_batches) == batches
        variants = [{**original, "session_id": "changed"}]
        for field, value in (("content", "changed"), ("role", "assistant"), ("timestamp", 1704067200001)):
            variants.append({**original, "messages": [
                {**original["messages"][0], field: value}, original["messages"][1]]})
        for variant in variants:
            assert client.post("/add", headers=HEADERS, json=variant).status_code == 409
        assert len(embedder.document_batches) == batches
        bob = payload(user="alice-suffix", messages=[{
            "role": "user", "content": "fictional cobalt BOB-ONLY-913 note.",
        }])
        assert client.post("/add", headers=HEADERS, json=bob).status_code == 200
        a_sources = service.baseline._store.all_sources("alice")
        b_sources = service.baseline._store.all_sources("alice-suffix")
        assert len(a_sources) == 2 and len(b_sources) == 1
        assert {s["id"] for s in a_sources}.isdisjoint(s["id"] for s in b_sources)
        a = client.post("/search", headers=HEADERS, json=query()).json()
        b = client.post("/search", headers=HEADERS, json=query(user="alice-suffix")).json()
        assert "BOB-ONLY" not in json.dumps(a) and "BOB-ONLY" in json.dumps(b)
        assert "37 shells" not in json.dumps(b)
        before = len(embedder.query_texts)
        assert client.post("/search", headers=HEADERS, json=query()).json() == a
        assert len(embedder.query_texts) == before and harness.calls == 2
        assert client.post("/search", headers=HEADERS, json=query(user="empty")).json() == {"data": []}
        assert harness.calls == 2
    reopened, service2, harness2, embedder2 = make_app(tmp_path)
    with TestClient(reopened) as client:
        assert client.post("/add", headers=HEADERS, json=original).json() == receipt.json()
        assert client.post("/search", headers=HEADERS, json=query()).json() == a
        assert client.post("/search", headers=HEADERS, json=query(user="alice-suffix")).json() == b
        assert embedder2.document_batches == [] and embedder2.query_texts == []
        assert harness2.calls == 0 and len(service2.ledger.list_attempts()) == 3


def test_cache_binds_exact_query_options_top_k_and_source_snapshot(tmp_path):
    app, service, harness, embedder = make_app(tmp_path)
    with TestClient(app) as client:
        assert client.post("/add", headers=HEADERS, json=payload()).status_code == 200
        requests = [query(), query(text="fictional cobalt archive shells "),
                    {**query(), "options": ["blue", "green"]},
                    {**query(), "options": ["green", "blue"]}, query(k=32)]
        for i, request in enumerate(requests, 1):
            first = client.post("/search", headers=HEADERS, json=request)
            assert first.status_code == 200 and first.json()["data"]
            assert harness.calls == i
            count = len(embedder.query_texts)
            assert client.post("/search", headers=HEADERS, json=request).json() == first.json()
            assert len(embedder.query_texts) == count and harness.calls == i
        extra = payload(rid="r2", messages=[{"role": "user", "content": "fictional cobalt NEW-SOURCE note."}])
        assert client.post("/add", headers=HEADERS, json=extra).status_code == 200
        later = client.post("/search", headers=HEADERS, json=query())
        assert later.status_code == 200 and "NEW-SOURCE" in later.text and harness.calls == 6
        assert client.post("/search", headers=HEADERS, json=query()).json() == later.json()
        assert harness.calls == len(service.ledger.list_attempts()) == 6
        assert all(row["failures"] == {} for row in records(service))


def test_cold_identical_search_16_concurrent_http_requests_share_one_execution(tmp_path):
    class BlockingHarness(RecordingHarness):
        def __init__(self):
            super().__init__()
            self.entered, self.release = Event(), Event()

        def run(self, tools):
            self.entered.set()
            assert self.release.wait(20), "test did not release the cold in-flight request"
            return super().run(tools)

    harness = BlockingHarness()
    app, service, _, embedder = make_app(tmp_path, harness=harness)
    barrier = Barrier(17)
    with TestClient(app) as client:
        assert client.post("/add", headers=HEADERS, json=payload()).status_code == 200

        def search(_):
            barrier.wait(timeout=20)
            return client.post("/search", headers=HEADERS, json=query())

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(search, i) for i in range(16)]
            try:
                barrier.wait(timeout=20)
                assert harness.entered.wait(20)
            finally:
                harness.release.set()
            replies = [future.result(timeout=45) for future in futures]
        assert all(reply.status_code == 200 for reply in replies)
        assert replies[0].json()["data"]
        assert all(reply.json() == replies[0].json() for reply in replies)
        assert harness.calls == 1 and len(embedder.query_texts) == 1
        assert len(service.ledger.list_attempts()) == 1
        assert len(records(service)) == 1 and records(service)[0]["failures"] == {}


def test_f04_scores_remain_static_against_initial_context_and_stable_ties():
    rows = [source("a", "cobalt amber"), source("b", "cobalt amber cobalt"),
            source("c", "cobalt"), source("initial-duplicate", "seed")]
    actual = select_tail(rows, "cobalt amber", ["seed"], limit=3)
    assert [h.source_ids[0] for h in actual] == ["a", "b", "c"]
    assert [h.score for h in actual] == pytest.approx([0.14, 0.14, 0.07])
    assert [h.content for h in actual] == [_render_source(s) for s in rows[:3]]
    # A dynamic recomputation after choosing a would prefer c over b. The tested
    # W1 selector deliberately retains its initial scores and chooses b second.
    recomputed = select_tail(rows[1:3], "cobalt amber", ["seed", "cobalt amber"], limit=2)
    assert [h.source_ids[0] for h in recomputed] == ["c", "b"]
    assert [h.score for h in recomputed] == pytest.approx([-0.06, -0.12])


def test_f04_anchor_hit_ratio_and_initial_redundancy_have_original_scale():
    rows = [source("one", "37 record"), source("both", "37 41 record")]
    actual = select_tail(rows, "37 41", ["seed"], limit=2)
    assert [h.source_ids[0] for h in actual] == ["both", "one"]
    assert [h.score for h in actual] == pytest.approx([1.14, 0.57])
    redundant = select_tail([source("x", "cobalt amber cobalt")], "cobalt amber",
                            ["cobalt amber"], limit=1)
    assert redundant[0].score == pytest.approx(-0.12)


def test_w1_neighbor_evidence_id_mismatch_is_preserved_with_positive_control(tmp_path):
    app, service, _, _ = make_app(tmp_path)
    with TestClient(app):
        rows = [source("left", "left exact body"), source("core", "core exact body"),
                source("right", "right exact body")]
        by_id = {s["id"]: s for s in rows}
        core = Evidence("memmachine_legacy_hash", _render_source(rows[1]), ("core",), 1.0, "memmachine_event")
        request = SearchRequest(user_id="alice", query="unrelated", top_k=3)
        assert expand_source_ids([core.id], rows, budget=8) == []
        assert expand_source_ids(["core"], rows, budget=8) == ["left", "right"]
        actual = service.baseline._with_expand(request, [core], by_id)
        assert actual[0] == core and core.id != core.source_ids[0]
        assert [h.source_ids[0] for h in actual] == ["core", "left", "right"]
        assert [h.kind for h in actual[1:]] == ["complementary_tail", "complementary_tail"]
        control = service.baseline._with_expand(request, [replace(core, id="core")], by_id)
        assert [h.kind for h in control[1:]] == ["source", "source"]
        assert "Evidence.id differs from source_id" in app.state.version_manifest["f04_known_limit"]


@pytest.mark.parametrize("k", [16, 100])
def test_old_fusion_rrf60_rehydrates_deduplicates_filters_and_can_change_head16(k):
    rows = [source(f"r{i}", f"exact R body {i}\n  原文") for i in range(16)]
    rows += [source("new", "exact H-only original"), source("hidden", "secret note"),
             source("withdrawal", "Forget source:hidden.")]
    visibility = Visibility("alice", rows)
    r_hits = [Evidence(f"legacy-{i}", "fabricated branch prose", (f"r{i}",), float(1000 - i))
              for i in range(16)]
    r_hits += [r_hits[0]]
    actual = fuse_sources(r_hits, ["new", "new", "hidden", "foreign"], visibility, top_k=k)
    expected_ids = (["r0", "new"] + [f"r{i}" for i in range(1, 16)])[:k]
    assert [h.source_ids[0] for h in actual] == expected_ids
    assert [h.id for h in actual] == ["ensemble_" + sid for sid in expected_ids]
    for hit in actual:
        assert hit.content == _render_source(visibility.read(hit.source_ids[0]))
        assert "fabricated" not in hit.content
        rank = int(hit.source_ids[0][1:]) + 1 if hit.source_ids[0].startswith("r") else 1
        assert hit.score == pytest.approx(1 / (60 + rank))
    assert [h.source_ids[0] for h in actual[:16]] != [f"r{i}" for i in range(16)]
    if k == 16:
        assert "r15" not in {h.source_ids[0] for h in actual}
    both = fuse_sources(r_hits, ["r15", "r15"], visibility, top_k=100)
    assert both[0].source_ids == ("r15",)
    assert both[0].score == pytest.approx(1 / 76 + 1 / 61)


def add_sized_sources(service, rendered_lengths, *, h_only_last=False):
    request = AddRequest(user_id="alice", request_id="sized", session_id="s1",
                         messages=[Message(role="user", content="placeholder") for _ in rendered_lengths])
    stable = service.baseline._store.sources_for(request)
    for i, (row, length) in enumerate(zip(stable, rendered_lengths)):
        header_length = len(_render_source({**asdict(row), "content": ""}))
        prefix = "h-only " if h_only_last and i == len(stable) - 1 else "cobalt "
        assert length >= header_length + len(prefix)
        request.messages[i].content = prefix + "界" * (length - header_length - len(prefix))
    service.add(request)  # Actual upstream EventMemory encoding and MethodStore commit.
    rows = service.baseline._store.all_sources("alice")
    assert [len(_render_source(row)) for row in rows] == rendered_lengths
    return rows


def forced_rank(rows):
    # Force only branch ordering; storage, visibility, fusion, budget and cache
    # remain the actual candidate implementation. Fake prose must be rehydrated.
    return [Evidence(f"forced-{i}", "fabricated short branch view", (row["id"],), 1000.0 - i)
            for i, row in enumerate(rows)]


@pytest.mark.parametrize("lengths,kept", [
    ([60000], [0]),
    ([30000, 30000], [0, 1]),
    ([40000, 25000, 19000], [0, 2]),
])
def test_final_60000_character_budget_keeps_whole_sources_and_skips_without_truncation(
        tmp_path, monkeypatch, lengths, kept):
    app, service, harness, _ = make_app(tmp_path, harness=EmptyHarness())
    with TestClient(app) as client:
        rows = add_sized_sources(service, lengths)
        monkeypatch.setattr(service.baseline, "search", lambda request: forced_rank(rows))
        reply = client.post("/search", headers=HEADERS, json=query())
        assert reply.status_code == 200
        hits = reply.json()["data"]
        assert [h["id"] for h in hits] == ["ensemble_" + rows[i]["id"] for i in kept]
        assert [len(h["content"]) for h in hits] == [lengths[i] for i in kept]
        assert sum(len(h["content"]) for h in hits) <= 60000
        assert sum(len(h["content"].encode("utf-8")) for h in hits) > 60000
        for hit in hits:
            assert_original_hit(hit, {s["id"]: s for s in rows})
        assert client.post("/search", headers=HEADERS, json=query()).json() == reply.json()
        assert harness.calls == 1 and records(service)[0]["failures"] == {}


def test_final_budget_can_drop_first_r_core_and_return_99_without_backfill(tmp_path, monkeypatch):
    class NewSourceHarness(Harness):
        def run(self, tools):
            self.calls += 1
            found = tools.execute("grep", {"pattern": "h-only"})["sources"]
            assert len(found) == 1
            tools.execute("finish_evidence", {"source_ids": [found[0]["source_id"]]})
            return {"selected": tools.selected, "run_id": "offline-new-source"}

    app, service, _, _ = make_app(tmp_path, harness=NewSourceHarness())
    with TestClient(app) as client:
        rows = add_sized_sources(service, [60001] + [180] * 100, h_only_last=True)
        monkeypatch.setattr(service.baseline, "search", lambda request: forced_rank(rows[:100]))
        reply = client.post("/search", headers=HEADERS, json=query())
        assert reply.status_code == 200
        ids = [h["id"] for h in reply.json()["data"]]
        assert len(ids) == 99
        assert ids == ["ensemble_" + rows[-1]["id"]] + ["ensemble_" + s["id"] for s in rows[1:99]]
        assert "ensemble_" + rows[0]["id"] not in ids  # First R core exceeds the whole-source budget.
        assert "ensemble_" + rows[99]["id"] not in ids  # RRF top_k truncation precedes budget; no refill.
        assert sum(len(h["content"]) for h in reply.json()["data"]) + 180 < 60000
        record = records(service)[0]
        assert len(record["R_source_ids"]) == 100 and record["H"]["selected"] == [rows[-1]["id"]]
        assert record["failures"] == {}


def test_single_whole_source_over_final_budget_is_failed_not_empty_success(tmp_path, monkeypatch):
    app, service, harness, _ = make_app(tmp_path, harness=EmptyHarness())
    with TestClient(app, raise_server_exceptions=False) as client:
        rows = add_sized_sources(service, [60001])
        monkeypatch.setattr(service.baseline, "search", lambda request: forced_rank(rows))
        reply = client.post("/search", headers=HEADERS, json=query())
        assert reply.status_code == 500 and reply.json() == {"detail": "Memory service failed"}
        tasks = service.ledger.list_tasks(status="failed")
        assert len(tasks) == 1 and records(service) == [] and harness.calls == 1
        error = service.ledger.list_attempts()[0]["error"]
        assert error == {"kind": "RuntimeError", "reason": "deliverable source exceeds final evidence budget"}


def test_h_timeout_keeps_real_f04_r_fallback_and_success_cache(tmp_path):
    app, service, harness, embedder = make_app(tmp_path, harness=FailedHarness())
    with TestClient(app) as client:
        messages = [{"role": "user", "content": f"fictional cobalt archive entry {i} shells."}
                    for i in range(40)]
        assert client.post("/add", headers=HEADERS, json=payload(messages=messages)).status_code == 200
        reply = client.post("/search", headers=HEADERS, json=query(k=32))
        assert reply.status_code == 200 and len(reply.json()["data"]) == 32
        failures = records(service)[0]["failures"]
        assert set(failures) == {"H"} and failures["H"]["type"] == "TimeoutError"
        count = len(embedder.query_texts)
        assert client.post("/search", headers=HEADERS, json=query(k=32)).json() == reply.json()
        assert harness.calls == 1 and len(embedder.query_texts) == count
        assert len(service.ledger.list_attempts()) == 1
