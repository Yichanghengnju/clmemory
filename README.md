# clmemory — Textual wide verbatim delivery (textual-f04-only-v1.0)

Cycle-2 Agent Memory Leaderboard (Textual) submission: **zero-generative-LLM
memory with wide verbatim delivery**. Add stores every message verbatim with
role/timestamp/source identity. Search runs MemMachine-native BM25 + dense
retrieval fused with RRF (16 protected core hits), then a deterministic
complementary tail selector fills the remaining slots up to the requested
`top_k` (up to 100) with the most complementary undelivered sources — coverage
of query content words first, redundancy (token-Jaccard) penalized. The
platform runs Answer and evaluation; this service never generates answers.

- **Add**: attributed MemMachine EventMemory, whole-text passthrough, zero
  generative LLM calls. Embeddings: official `text-embedding-v4`, 1024 dims.
- **Search**: BM25 + dense RRF (k=60), 16 native core results protected,
  complementary tail (coverage 0.14 / redundancy 0.12 static weights) to the
  requested `top_k`; every delivered item is the original source text with its
  role and timestamp header. Visibility/retraction safety filter applied.
  Zero generative LLM calls in Search.
- **Identity**: `method = memmachine-neighbor-grep-append` lineage,
  version `textual-f04-only-v1.0`, entry `agent_memory.f04_submission`.
- **Official smoke**: `teval_3e3e55fa6e5b3442` / `version_1fcc37e2e921`,
  46/46 completed, score 72.80, smoke passed (2026-10-30).
- **Contract**: unauthenticated `GET /health`; authenticated `POST /add`,
  `POST /search` (`Authorization: Token|Bearer` or `X-Api-Key`). Add is
  idempotent per (user, request, payload); changed payload under the same id
  returns 409. Search returns `{data: [{id, content, score, source_ids}]}`,
  never a final answer. Max `top_k` 100; Add/Search concurrency 16/16.
- **Privacy**: no benchmark data, labels, or platform payloads are stored in
  this repository. Service receipts are private, evaluation-only, retained at
  most 30 days.

## Provenance and attribution

The native event-memory layer is MemMachine (Apache-2.0), pinned at upstream
commit `d5a39a8e774e977fb1e6c22efb03555c4e51e05e`; see
`licenses/MemMachine-Apache-2.0.txt`, `THIRD_PARTY_NOTICES.md` and
`docs/provenance.md`. Our changes on top: the W1 delivery layer
(`method_memmachine_w1.py`, `complementary_tail.py` — the static tail
selector), the frozen official service wrapper (`f04_submission.py`), exact
input single-flight for searches, and the Bailian embedding adapter. All
retrieval scoring is deterministic; there is no learned model in this path.

## Install

Python 3.12+ with the pinned native dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]' -r requirements-native.txt
.venv/bin/python -m pytest -q
```

## Configure and run

Create a private `embedding.local.json` from `embedding.example.json`
(DashScope workspace endpoint + external `api_key_file` + private cache dir),
and a private service token file for the platform (random secret; not the
DashScope key, not the platform-issued Eval Key).

```sh
.venv/bin/python -m agent_memory.f04_submission \
  --db run/textual/memory.sqlite3 \
  --api-key-file run/textual/service.key \
  --embedding-profile embedding.local.json \
  --run-dir run/textual \
  --host 127.0.0.1 --port 18131
```

Put the service behind HTTPS (reverse proxy/funnel) for public evaluation.
Identical concurrent searches wait on one flight instead of duplicating work;
successful Add is persisted and searchable before acknowledgement.

## License

Apache-2.0; see `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md`.
Contact: Chang-Heng Yi, Nanjing University / Cllab.
