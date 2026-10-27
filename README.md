# clmemory v1.0.5

**A source-preserving, event-only MemMachine adaptation for the Agent Memory
Leaderboard Textual and Coding tracks.** It directly uses MemMachine's native
event storage and BM25/dense/RRF components, with separate delivery policies.
Read [the attribution](THIRD_PARTY_NOTICES.md) and [exact changes and limits](docs/provenance.md).

Add stores original messages and their source roles/timestamps. Search returns
source-bound evidence, never a final answer or code patch. Add uses zero
generative LLM calls. Search applies a final `gpt-4o-mini` selection pass for
relevance, entity/time/state and safety/privacy after every retrieval channel.
Embeddings are `text-embedding-v4`, 1024 dimensions.
No official score is claimed.

This release repairs the final-filter response protocol for the Textual service;
see [v1.0.5 changes and validation](docs/release-v1.0.5.md). The separately frozen
Coding repository and its v1.0 deployment are not upgraded by this release.

## Install

Use a dedicated Python 3.12+ environment on Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]' -r requirements-native.txt
.venv/bin/python -m pytest -q
```

The native dependency is pinned to upstream MemMachine commit
`d5a39a8e774e977fb1e6c22efb03555c4e51e05e` (Apache-2.0).
Synthetic tests use a deterministic test embedder and make no external calls.
The deployment command always requires the real embedding profile; it has no
synthetic or alternate-model fallback.

## Configure and run

Create a private `embedding.local.json` from `embedding.example.json`; set the
official DashScope workspace endpoint, external `api_key_file`, and a private
cache directory. Create a separate random service-key file for AML; it is not
the DashScope key or the platform-issued Eval Key. Never commit any key or local
profile. The `.example` file is a template and does not contain a usable endpoint.
Create a private filter profile from `filter.example.json` with the model-provider
key, private cache and an OS lock path shared between both track services. The
CLI requires the final filter; it never silently switches to unfiltered results.
The selection guard uses the `o200k_base` tokenizer. Preload it while online with
`python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"`. For an offline
execution host, copy that verified cache and set `TIKTOKEN_CACHE_DIR` for this
service process; see the release notes for the vocabulary hash.

```sh
.venv/bin/python -m agent_memory.submission --track textual \
  --db run/textual/memory.sqlite3 --api-key-file run/textual/api.key \
  --embedding-profile embedding.local.json --filter-profile filter.local.json \
  --manifest-dir run/textual \
  --host 0.0.0.0 --port 8000 --http-concurrency 128
```

Run Coding separately with `--track coding`, a different port, database,
service-key file, manifest directory and private embedding-cache directory.
The filter cache is also track-specific. Filter model calls serialize through
the shared OS lock. The initial review configuration permits 200 physical mini
requests per track, including failures; successful exact inputs are cached.
That is a smoke/access-review budget, not a Full-evaluation capacity claim.
Use one worker per track; native index and source delivery use per-user
cross-process locks. Current AML minimum settings: Add concurrency=16, Search concurrency=16.
HTTP admission at 128 does not increase the one shared selection-model slot.
Maximum accepted top_k is 100; fewer unique evidence items may be returned.
An unknown user returns an empty data array.

Routes: unauthenticated `GET /health`, authenticated `POST /add` and `POST /search`.
Authentication accepts `Authorization: Token`, `Authorization: Bearer`, or
`X-Api-Key`. An identical successful Add is reusable; changed payload under the
same user/request ID returns 409. Successful Add is persisted and searchable
before acknowledgement. A version manifest is written at startup, without keys.

Deploy behind HTTPS for encrypted transport where available. If direct HTTP is
used for initial access review, disclose that in the application; the public
listener still requires a dedicated service key. Capacity is constrained by
host storage and embedding quota; no Full-scale throughput is claimed from
synthetic checks. Known transient embedding responses have at most three
physical attempts; uncertain transport is held and requires operator review.
Do not erase its cache or retry under a fresh identity to bypass that hold.

The final filter preserves the order and exact body of each retained item. It
records all keep/drop reasons in its private ledger. Common credential patterns
are blocked before the selection-model request. The model separately checks
relevance and unauthorized disclosure/injection, with explicit controls for
actual facts, historical questions and questions about assistant suggestions.
It is a fallible classifier, not a general privacy or safety guarantee. Detection
of common secrets does not detect every personal datum and does not retroactively
change what was stored or embedded during Add. Benign medical/personal facts and
ordinary coding commands are not blanket-denied. Uncertain relevance may be kept.
Classifier failures, invalid decisions, quota exhaustion and oversized inputs
return service errors; they never become a successful empty retrieval. The
single selection request has a 2,000,000 UTF-8-byte transport bound and a
110,000-token estimate bound over canonical request JSON, using `o200k_base`.
The output bound remains 8,192 tokens; the remaining 9,808 tokens of the 128,000
context window are reserved for accounting/formatting differences. Inputs are
kept whole. This is a conservative preflight estimate, not a billing-token claim.
Actual token/byte overflow is an explicit error, without truncation or fallback.
The response schema constrains both per-item pairs and query-level denial.
Error logs contain hashes, numeric guard limits and code locations; headers,
source text and exception arguments are excluded.

The platform controls source chunking, answering and scoring. There are no
benchmark data, labels, judge prompts, research runners, or private logs here.

Contact: Chang-Heng Yi, Nanjing University / Cllab. Apache-2.0; see LICENSE and NOTICE.
