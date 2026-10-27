"""Unit contract checks with synthetic embeddings, not a real-model acceptance."""

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from agent_memory.domain import AddRequest, Message, RequestConflict, SearchRequest
from agent_memory.method_invmem import InvMemMemory, UPSTREAM_COMMIT, _bm25, _chunks, _tokens


class FakeEmbeddings:
    identity = "unit-test-synthetic-embedding-v1"

    def __init__(self, vectors=None, query_vector=None):
        self.vectors = vectors or {}
        self.query_vector = query_vector or [1.0, 0.0]
        self.document_calls = []
        self.query_calls = []
        self.guard = threading.Lock()

    def embed_documents(self, texts):
        with self.guard:
            self.document_calls.append(list(texts))
        return [self.vectors.get(text, [1.0, 0.0]) for text in texts]

    def embed_query(self, query):
        self.query_calls.append(query)
        return self.query_vector


class NeverProvider:
    def complete(self, **kwargs):
        raise AssertionError("InvMem must not call a query/Add LLM")


def add_request(request_id="request", user_id="alice", session_id="session", texts=None):
    return AddRequest(
        request_id=request_id, user_id=user_id, session_id=session_id,
        messages=[Message(role="user", content=text) for text in (texts or ["orchid garden"])],
    )


def search_request(query="orchid", user_id="alice", top_k=8, options=None):
    return SearchRequest(query=query, user_id=user_id, top_k=top_k, options=options)


@pytest.fixture
def memory(tmp_path):
    service = InvMemMemory(tmp_path / "invmem.sqlite", NeverProvider(), FakeEmbeddings())
    yield service
    service.close()


def text_of(evidence):
    return evidence.content.rsplit("\n[chunk ", 1)[0]


def test_chunk_exact_original_slices_and_40_token_overlap():
    text = " \t" + "  \n".join(f"word{i}" for i in range(700)) + "\n "
    chunks = _chunks(text, 320, 40)
    assert len(chunks) == 3
    assert [(start, end) for _, _, _, start, end in chunks] == [(0, 320), (280, 600), (560, 700)]
    for chunk, start, end, _, _ in chunks:
        assert chunk == text[start:end]
    assert chunks[0][1] == 0 and chunks[-1][2] == len(text)
    assert _tokens(chunks[0][0])[-40:] == _tokens(chunks[1][0])[:40]
    assert _tokens(chunks[1][0])[-40:] == _tokens(chunks[2][0])[:40]
    assert _tokens("兰花 apple,中文!") == ["兰", "花", "apple", ",", "中", "文", "!"]
    assert len(_chunks("x " * 320, 320, 40)) == 1
    assert len(_chunks("x " * 321, 320, 40)) == 2


def test_add_sources_metadata_restart_and_no_llm(tmp_path):
    path = tmp_path / "memory.sqlite"
    embedder = FakeEmbeddings()
    service = InvMemMemory(path, NeverProvider(), embedder)
    request = AddRequest(
        request_id="request", user_id="alice", session_id="session",
        messages=[Message(role="user", content="orchid first", timestamp=0),
                  Message(role="assistant", content="orchid suggestion")],
    )
    try:
        receipt = service.add(request)
        results = service.search(search_request())
        assert len(results) == 2 and all(item.kind == "chunk" for item in results)
        sources = service._store.sources_for(request)
        assert [item.source_ids for item in results] == [(source.id,) for source in sources]
        assert '"role": "assistant"' in results[1].content
        assert '"timestamp": null' in results[1].content
        assert '"timestamp": 0' in results[0].content
        assert embedder.document_calls == [["orchid first", "orchid suggestion"]]
    finally:
        service.close()
    reopened = InvMemMemory(path, NeverProvider(), FakeEmbeddings())
    try:
        assert reopened.add(request) == receipt
        assert reopened.embedder.document_calls == []
        assert reopened.search(search_request()) == results
    finally:
        reopened.close()


def test_chunk_records_keep_only_single_true_source(memory):
    request = add_request(texts=["word " * 650, "second message"])
    sources = memory._store.sources_for(request)
    memory.add(request)
    documents = memory._store.documents("alice", kinds=["chunk"])
    assert len(documents) == 4
    assert [item["metadata"]["ordinal"] for item in documents] == [0, 1, 2, 3]
    assert [item["metadata"]["chunk_ordinal"] for item in documents] == [0, 1, 2, 0]
    for document in documents:
        source = sources[document["metadata"]["message_position"]]
        assert document["source_ids"] == [source.id] or tuple(document["source_ids"]) == (source.id,)
        metadata = document["metadata"]
        assert document["text"] == source.content[metadata["char_start"]:metadata["char_end"]]


def test_bm25_standard_formula_and_no_metadata():
    scores = _bm25(["apple apple apple", "apple pear plum kiwi mango", "pear"], "apple")
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    assert scores == pytest.approx([
        idf * 3 * 2.5 / (3 + 1.5),
        idf * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 5 / 3)), 0.0,
    ])
    assert _bm25(["apple"], "role session timestamp") == [0.0]


def test_dense_lexical_rrf_uses_ranks_only(tmp_path):
    embedding = FakeEmbeddings({"unrelated": [1.0, 0.0], "needle": [0.6, 0.8]})
    service = InvMemMemory(tmp_path / "rrf.sqlite", embedder=embedding, configdict={"window": 0})
    try:
        service.add(add_request(texts=["unrelated", "needle"]))
        results = service.search(search_request("needle"))
        assert [text_of(item) for item in results] == ["needle", "unrelated"]
        assert results[0].score == pytest.approx(1 / 62 + 0.5 / 61)
        assert results[1].score == pytest.approx(1 / 61)
    finally:
        service.close()


def test_dense_all_scores_rank_but_lexical_requires_positive(tmp_path):
    embedding = FakeEmbeddings({"negative": [-1.0, 0.0], "zero": [0.0, 1.0], "positive": [1.0, 0.0]})
    service = InvMemMemory(tmp_path / "positive.sqlite", embedder=embedding, configdict={"window": 0})
    try:
        service.add(add_request(texts=["negative", "zero", "positive"]))
        results = service.search(search_request("absent"))
        assert [text_of(item) for item in results] == ["positive", "zero", "negative"]
        assert [item.score for item in results] == pytest.approx([1 / 61, 1 / 62, 1 / 63])
        lexical_results = service.search(search_request("negative"))
        assert [text_of(item) for item in lexical_results] == ["negative", "positive", "zero"]
        assert lexical_results[0].score == pytest.approx(1 / 63 + 0.5 / 61)
    finally:
        service.close()


def test_window_cross_request_stored_order_not_timestamp_or_request_id(tmp_path):
    embedding = FakeEmbeddings({"before": [-1.0, 0.0], "target": [1.0, 0.0], "after": [-1.0, 0.0], "other session": [-1.0, 0.0], "lasttarget": [1.0, 0.0]})
    service = InvMemMemory(tmp_path / "window.sqlite", embedder=embedding)
    try:
        for request_id, text, session in [("z", "before", "s"), ("m", "target", "s"), ("a", "after", "s")]:
            service.add(add_request(request_id, session_id=session, texts=[text]))
        results = service.search(search_request("target"))
        assert [text_of(item) for item in results] == ["target", "before", "after"]
        assert [item.score for item in results] == pytest.approx([1 / 61 + 0.5 / 61, 1 / 62, 1 / 63])
        assert len({item.id for item in results}) == len(results)
        assert [text_of(item) for item in service.search(search_request("target", top_k=1))] == ["target"]
        # A different session immediately before the next seed blocks expansion;
        # stored adjacency does not skip that row to reach an older same-session one.
        service.add(add_request("other", session_id="other", texts=["other session"]))
        service.add(add_request("last", texts=["lasttarget"], session_id="s"))
        blocked_results = service.search(search_request("lasttarget", top_k=2))
        assert [text_of(item) for item in blocked_results] == ["lasttarget", "target"]
    finally:
        service.close()


def test_window_dedup_and_max_eight(memory):
    memory.add(add_request(texts=[f"orchid {index}" for index in range(30)]))
    results = memory.search(search_request(top_k=100))
    assert len(results) == len({item.id for item in results}) == 8
    assert len(memory.search(search_request(top_k=3))) == 3


def test_rank_fill_after_seed_window(tmp_path):
    service = InvMemMemory(tmp_path / "fill.sqlite", embedder=FakeEmbeddings(), configdict={"window": 0, "seed_count": 1})
    try:
        service.add(add_request(texts=["first", "second", "third"]))
        assert [text_of(item) for item in service.search(search_request("absent"))] == ["first", "second", "third"]
    finally:
        service.close()


def test_exact_user_scope_and_payload_conflicts(memory):
    for user in ("alice", "alice%", "ALICE", "bob"):
        memory.add(add_request(user_id=user))
    assert len(memory.search(search_request())) == 1
    assert len(memory.search(search_request(user_id="alice%"))) == 1
    assert memory.search(search_request(user_id="ali")) == []
    assert memory.search(search_request(user_id="alice' OR 1=1 --")) == []
    assert memory.search(search_request())[0].source_ids != memory.search(search_request(user_id="bob"))[0].source_ids
    memory.add(add_request())
    calls = len(memory.embedder.document_calls)
    for changed in (add_request(texts=["changed"]), add_request(session_id="other")):
        with pytest.raises(RequestConflict):
            memory.add(changed)
    assert len(memory.embedder.document_calls) == calls


def test_options_are_concatenated_without_prefix(memory):
    memory.add(add_request())
    memory.search(search_request("what?", options=["orchid", "rose"]))
    assert memory.embedder.query_calls == ["what? orchid rose"]
    assert memory.identity["options_weight"] == 0


def test_empty_query_and_top_k_limits(memory):
    for query in ("", "  \n"):
        with pytest.raises(ValidationError):
            memory.search(search_request(query))
    for top_k in (0, -1, 101, True):
        with pytest.raises(ValidationError):
            search_request(top_k=top_k)
    assert memory.search(search_request()) == []
    assert memory.embedder.query_calls == []


def test_same_request_concurrent_instances_embed_once(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    embedder = FakeEmbeddings()
    entered, release = threading.Event(), threading.Event()
    original = embedder.embed_documents

    def blocked(texts):
        entered.set()
        assert release.wait(5)
        return original(texts)

    embedder.embed_documents = blocked
    services = [InvMemMemory(path, embedder=embedder) for _ in range(3)]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(service.add, add_request()) for service in services]
            try:
                assert entered.wait(5)
            finally:
                release.set()
            receipts = [future.result(timeout=5) for future in futures]
        assert all(item == receipts[0] for item in receipts)
        assert len(embedder.document_calls) == 1
        assert len(services[0]._store.documents("alice", kinds=["chunk"])) == 1
    finally:
        release.set()
        for service in services:
            service.close()


def test_identity_freezes_embedding_and_algorithm(tmp_path):
    path = tmp_path / "identity.sqlite"
    embedder = FakeEmbeddings()
    service = InvMemMemory(path, embedder=embedder)
    try:
        identity = service.identity
        assert identity["upstream_commit"] == UPSTREAM_COMMIT
        assert "unknown" in identity["original_license"]
        assert "independent" in identity["adaptation"]
        assert identity["chunk_size"] == 320 and identity["chunk_overlap"] == 40
        assert identity["rrf_k"] == 60 and identity["lexical_weight"] == 0.5
        assert identity["window"] == 1 and identity["seed_count"] == 20
        assert identity["embedding_identity"] == embedder.identity
        other = FakeEmbeddings()
        other.identity = "other-model"
        with pytest.raises(ValueError):
            InvMemMemory(path, embedder=other)
        with pytest.raises(ValueError):
            InvMemMemory(path, embedder=FakeEmbeddings(), configdict={"chunk_size": 300})
        embedder.identity = "changed-model"
        with pytest.raises(ValueError, match="changed"):
            service.add(add_request())
        with pytest.raises(ValueError, match="changed"):
            service.search(search_request())
    finally:
        service.close()


@pytest.mark.parametrize("vectors", [[[]], [[float("nan"), 1]], [[0, 0]], [], [[1, 0], [1, 0]]])
def test_bad_embeddings_do_not_complete_add(tmp_path, vectors):
    embedder = FakeEmbeddings()
    embedder.embed_documents = lambda texts: vectors
    service = InvMemMemory(tmp_path / "bad.sqlite", embedder=embedder)
    try:
        with pytest.raises(ValueError):
            service.add(add_request())
        assert not service._store.completed(add_request())
        assert service._store.documents("alice", kinds=["chunk"]) == []
    finally:
        service.close()


@pytest.mark.parametrize("config", [
    {"chunk_overlap": 320}, {"chunk_size": 0}, {"chunk_size": True},
    {"window": -1}, {"seed_count": 0}, {"rrf_k": 0}, {"lexical_weight": float("inf")},
    {"options_weight": 0.5}, {"reranker": True}, {"query_llm": True},
])
def test_invalid_configuration_rejected(tmp_path, config):
    with pytest.raises(ValueError):
        InvMemMemory(tmp_path / "invalid.sqlite", embedder=FakeEmbeddings(), configdict=config)


def test_missing_embedder_rejected(tmp_path):
    with pytest.raises(ValueError, match="embedder"):
        InvMemMemory(tmp_path / "missing.sqlite")


def test_failed_embedding_retry_keeps_source_ids_and_complete_atomicity(memory):
    request = add_request()
    expected_source = memory._store.sources_for(request)[0].id
    original = memory.embedder.embed_documents

    def failed(texts):
        raise RuntimeError("embedding unavailable")

    memory.embedder.embed_documents = failed
    with pytest.raises(RuntimeError, match="unavailable"):
        memory.add(request)
    assert memory._store.all_sources("alice") == []
    assert not memory._store.completed(request)
    memory.embedder.embed_documents = original
    assert memory.add(request).success
    assert memory.search(search_request())[0].source_ids == (expected_source,)


def test_commit_race_rechecks_payload(memory):
    request = add_request()
    original = memory._store.commit

    def winner_then_conflict(req, records):
        original(req, records)
        raise RequestConflict("simulated concurrent commit")

    memory._store.commit = winner_then_conflict
    assert memory.add(request).success
    assert len(memory._store.documents("alice", kinds=["chunk"])) == 1


def test_commit_race_different_payload_remains_conflict(memory):
    original = memory._store.commit

    def other_payload_wins(req, records):
        changed = add_request(texts=["different winner"])
        original(changed, [])
        raise RequestConflict("different payload won")

    memory._store.commit = other_payload_wins
    with pytest.raises(RequestConflict):
        memory.add(add_request())
    assert memory._store.documents("alice", kinds=["chunk"]) == []


def test_query_embedding_dimension_errors_are_not_silent_fallback(memory):
    memory.add(add_request())
    memory.embedder.query_vector = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="dimension"):
        memory.search(search_request())
    memory.embedder.query_vector = [float("inf"), 0.0]
    with pytest.raises(ValueError, match="finite"):
        memory.search(search_request())
