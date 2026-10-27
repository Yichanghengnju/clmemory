"""Independent adaptation of InvMem #1 (vanilla-rag-memory).

Algorithm reference: upstream commit 31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca.
This implementation was independently written from the verified public algorithm;
upstream source is not copied. Original license: unknown / not declared.
Add and Search are retrieval only: no query LLM, no Qwen2.5 invocation, and no
reranker. That absence is a method design difference, not a disconnected call.
The supplied embedding implementation owns BGE CLS pooling, L2 normalization,
and the query-only instruction prefix; this module never adds that prefix.
"""

import json
import math
import os
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Iterator
from weakref import WeakValueDictionary

from .domain import AddReceipt, AddRequest, Evidence, RequestConflict, SearchRequest

UPSTREAM_COMMIT = "31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca"
# One CJK character, one English word/number, or one punctuation mark.
_TOKEN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)
_DEFAULTS = {
    "chunk_size": 320,
    "chunk_overlap": 40,
    "rrf_k": 60,
    "lexical_weight": 0.5,
    "window": 1,
    "seed_count": 20,
    "max_results": 8,
    "options_weight": 0.0,
    "reranker": False,
}
_GUARD = Lock()
_LOCKS: WeakValueDictionary[tuple[str, str, str], RLock] = WeakValueDictionary()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@contextmanager
def _request_lock(key: tuple[str, str, str]) -> Iterator[None]:
    with _GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _LOCKS[key] = lock
    with lock:
        yield


def _chunks(text: str, size: int, overlap: int) -> list[tuple[str, int, int, int, int]]:
    """Return exact source slices and character/token offsets, without rewriting."""
    tokens = list(_TOKEN.finditer(text))
    if not tokens:
        # Preserve non-ASCII letters not covered by the reference tokenizer too.
        return [(text, 0, len(text), 0, 0)] if text.strip() else []
    result = []
    for start in range(0, len(tokens), size - overlap):
        end = min(start + size, len(tokens))
        char_start = 0 if start == 0 else tokens[start].start()
        char_end = len(text) if end == len(tokens) else tokens[end - 1].end()
        result.append((text[char_start:char_end], char_start, char_end, start, end))
        if end == len(tokens):
            break
    return result


def _tokens(text: str) -> list[str]:
    return [match.group().lower() for match in _TOKEN.finditer(text)]


def _bm25(texts: list[str], query: str) -> list[float]:
    counts = [Counter(_tokens(text)) for text in texts]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    if not average:
        return [0.0] * len(texts)
    frequencies: Counter[str] = Counter()
    for count in counts:
        frequencies.update(count.keys())
    scores = [0.0] * len(texts)
    for term in set(_tokens(query)):
        df = frequencies[term]
        if not df:
            continue
        idf = math.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
        for index, count in enumerate(counts):
            tf = count[term]
            if tf:
                scores[index] += idf * tf * 2.5 / (
                    tf + 1.5 * (0.25 + 0.75 * lengths[index] / average)
                )
    return scores


def _vector(value: object, dimension: int | None = None) -> list[float]:
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError("embedding must be a finite nonzero vector") from error
    if not vector or any(not math.isfinite(item) for item in vector) or not any(vector):
        raise ValueError("embedding must be a finite nonzero vector")
    if dimension is not None and len(vector) != dimension:
        raise ValueError("embedding dimension mismatch")
    return vector


class InvMemMemory:
    """domain.Memory backed by shared MethodStore and an explicit embedder.

    MethodStore returns chunk dictionaries in persisted insertion order. The
    order must include messages from earlier requests in the same session;
    source.position alone resets on every Add and cannot supply that order.
    """

    def __init__(self, db_path: str | Path, provider=None, embedder=None, configdict=None, *, config=None, trace_hook=None):
        if configdict is not None and config is not None:
            raise ValueError("Specify only config or configdict")
        if config is not None:
            configdict = config
        if embedder is None:
            raise ValueError("InvMem requires an explicit BGE embedder")
        self.embedder = embedder
        self._trace_hook = trace_hook
        # provider is accepted for the common caller contract, never invoked.
        supplied = dict(configdict or {})
        unknown = set(supplied) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown InvMem configuration: {sorted(unknown)}")
        self.config = {**_DEFAULTS, **supplied}
        for key in ("chunk_size", "rrf_k", "seed_count", "max_results"):
            if type(self.config[key]) is not int or self.config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("chunk_overlap", "window"):
            if type(self.config[key]) is not int or self.config[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.config["chunk_overlap"] >= self.config["chunk_size"]:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        for key in ("lexical_weight", "options_weight"):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.config["options_weight"] != 0:
            raise ValueError("options_weight must be 0; options concatenate into the query")
        if self.config["reranker"] is not False:
            raise ValueError("this InvMem adaptation supports reranker=False only")
        self._identity = _canonical(self.identity)
        # Imported lazily so the algorithm can be loaded while the shared store
        # is assembled independently. Production never falls back to a fake store.
        from .method_store import MethodStore
        self._store = MethodStore(db_path, self.identity)
        self._database_key = (
            f":memory:{id(self)}" if str(db_path) == ":memory:"
            else os.path.normcase(str(Path(db_path).resolve()))
        )

    @property
    def identity(self) -> dict[str, object]:
        embedding_identity = getattr(self.embedder, "identity", None)
        if embedding_identity is None or embedding_identity == "" or embedding_identity == {}:
            raise ValueError("embedder identity must be nonempty finite JSON")
        identity = {
            "name": "invmem", "version": 1,
            "upstream_repository": "vanilla-rag-memory",
            "upstream_commit": UPSTREAM_COMMIT,
            "adaptation": "independent implementation of verified public algorithm",
            "original_license": "unknown (not declared)",
            "embedding_identity": embedding_identity,
            "tokenizer": "cjk-single-english-word-punctuation-v1",
            "bm25": {"k1": 1.5, "b": 0.75},
            "dense_rank": "all similarities; normalized-vector dot product",
            "lexical_rank": "positive scores only",
            "window_order": "seed then stored predecessor then successor; session must match",
            **self.config,
        }
        try:
            return json.loads(_canonical(identity))
        except (TypeError, ValueError) as error:
            raise ValueError("embedder identity must be nonempty finite JSON") from error

    def _check_identity(self) -> None:
        if _canonical(self.identity) != self._identity:
            raise ValueError("InvMem configuration or embedding identity changed")

    def add(self, request: AddRequest) -> AddReceipt:
        self._check_identity()
        request = AddRequest.model_validate(request.model_dump())
        receipt = AddReceipt(request_id=request.request_id, user_id=request.user_id, session_id=request.session_id)
        with _request_lock((self._database_key, request.user_id, request.request_id)):
            if self._store.completed(request):
                return receipt
            records = []
            for source in self._store.sources_for(request):
                for chunk_ordinal, (text, start, end, token_start, token_end) in enumerate(
                    _chunks(source.content, self.config["chunk_size"], self.config["chunk_overlap"])
                ):
                    records.append({
                        "text": text, "source_ids": [source.id], "kind": "chunk",
                        "metadata": {
                            "ordinal": len(records), "chunk_ordinal": chunk_ordinal,
                            "message_position": source.position,
                            "session_id": source.session_id, "request_id": source.request_id,
                            "role": source.role, "timestamp": source.timestamp,
                            "char_start": start, "char_end": end,
                            "token_start": token_start, "token_end": token_end,
                        },
                    })
            if records:
                vectors = list(self.embedder.embed_documents([record["text"] for record in records]))
                if len(vectors) != len(records):
                    raise ValueError("embed_documents returned the wrong number of vectors")
                dimension = None
                for record, value in zip(records, vectors):
                    vector = _vector(value, dimension)
                    dimension = len(vector)
                    record["vector"] = vector
            self._check_identity()
            try:
                self._store.commit(request, records)
            except RequestConflict:
                # A separate process can win the commit race after embedding.
                # completed checks payload equality and rethrows real conflicts.
                if not self._store.completed(request):
                    raise
        return receipt

    def search(self, request: SearchRequest) -> list[Evidence]:
        self._check_identity()
        request = SearchRequest.model_validate(request.model_dump())
        documents = self._store.documents(request.user_id, kinds=["chunk"])
        if not documents:
            if self._trace_hook is not None:
                self._trace_hook({'request':request.model_dump(),'candidates':[],
                    'ranked_ids':[],'selected_ids':[]})
            return []
        query = request.query
        if request.options:
            query += " " + " ".join(request.options)
        query_vector = _vector(self.embedder.embed_query(query))
        dense = []
        for document in documents:
            vector = _vector(document["vector"], len(query_vector))
            dense.append(sum(left * right for left, right in zip(vector, query_vector)))
        lexical = _bm25([document["text"] for document in documents], query)
        scores = [0.0] * len(documents)
        for values, weight in ((dense, 1.0), (lexical, self.config["lexical_weight"])):
            # Dense ranks include zero/negative similarities; only BM25 omits
            # nonpositive scores. RRF uses ranks, not raw score magnitudes.
            eligible = range(len(documents)) if values is dense else (
                i for i, score in enumerate(values) if score > 0
            )
            ranked = sorted(eligible, key=lambda i: (-values[i], i))
            for rank, index in enumerate(ranked, 1):
                scores[index] += weight / (self.config["rrf_k"] + rank)
        ranked = sorted((i for i, score in enumerate(scores) if score > 0), key=lambda i: (-scores[i], i))
        limit = min(request.top_k, self.config["max_results"])
        selected, seen = [], set()
        for index in ranked[:self.config["seed_count"]]:
            # Verified upstream memory_store.py:555-583: seed first, then
            # predecessor/successor by stored row order, same-session only.
            # Do not jump past another session to find a more distant chunk.
            candidates = [index]
            for distance in range(1, self.config["window"] + 1):
                candidates.extend((index - distance, index + distance))
            for neighbor in candidates:
                if (
                    0 <= neighbor < len(documents)
                    and documents[neighbor]["metadata"]["session_id"] == documents[index]["metadata"]["session_id"]
                    and neighbor not in seen
                ):
                    selected.append(neighbor)
                    seen.add(neighbor)
                    if len(selected) == limit:
                        break
            if len(selected) == limit:
                break
        if len(selected) < limit:
            for index in ranked:
                if index not in seen:
                    selected.append(index)
                    seen.add(index)
                    if len(selected) == limit:
                        break
        self._check_identity()
        if self._trace_hook is not None:
            self._trace_hook({'request':request.model_dump(),'embedding_query':query,
                'candidates':[{'id':document['id'],'source_ids':list(document['source_ids']),
                    'dense':dense[i],'bm25':lexical[i],'rrf':scores[i]}
                    for i,document in enumerate(documents)],
                'ranked_ids':[documents[i]['id'] for i in ranked],
                'selected_ids':[documents[i]['id'] for i in selected]})
        return [
            Evidence(
                id=documents[index]["id"],
                content=documents[index]["text"] + "\n[chunk " + json.dumps(
                    documents[index]["metadata"], ensure_ascii=False, sort_keys=True
                ) + "]",
                source_ids=tuple(documents[index]["source_ids"]),
                score=scores[index], kind="chunk",
            )
            for index in selected
        ]

    def close(self) -> None:
        self._store.close()
