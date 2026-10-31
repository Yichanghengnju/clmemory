# clmemory E*: frozen native harness + source retrieval

This is the user-selected Textual Smoke candidate `textual-Estar-S14-v1.0`. It keeps the existing R+H+F composition with the unchanged S14 retrieval rule library. It has not been shown to be a universal winner; the local paired confirmation has gains, regressions, missingness and source/score limitations.

- Add: attributed MemMachine native EventMemory, whole-text/passthrough; no generative model.
- R: dense/BM25/RRF16, at most8 same-session neighbors and8 anchors, unchanged16000-o200k whole-source packing.
- H: unmodified official DeepSeek dsh0.2.0-rc.1 and SDK client0.2.0-rc.1 own the model/tool loop. Persistent workers use a fresh source capability/session per query; six workers, at most16 sessions per healthy worker.
- Search model: GPT-4o-mini, at most6 responses,1536 output tokens/request,12 primitive calls,150 seconds/query. Tools: hybrid_search, grep, read_source, finish_evidence. There are no shell/filesystem/subprocess capabilities exposed to the retrieval model.
- F: the unchanged source RRF fusion, source visibility/retraction checks and60000-character whole-source delivery budget. Original failure/fallback behavior is retained and reported separately from H success.
- S14: `skills/S14.txt`, exact SHA256 `cb745515471e0a5828e99c390ffb3c9fb5f22b66809a0f67044b1e45af764f10`. No post-confirmation rule editing, KD, AgentDistill or deep-tail expansion.
- Embedding: text-embedding-v4 /1024 /L2. Official Answer/Eval are run by AML, not by this service.
- HTTP128 admission, one process; identical requests wait for their existing task rather than colliding on its lock. This adapter changes admission only, with no query/prompt/evidence changes.

Install the Python and pinned native MemMachine dependencies as described in the retained base documentation. Install the pinned dsh packages in a separate runtime with `npm ci --prefix tools/deepseek_harness`, or use an independently verified runtime containing both exact pinned packages. Node22 is the tested host runtime.

Run `python -m agent_memory.estar_submission --db PRIVATE_DB --api-key-file PRIVATE_MEMORY_TOKEN --embedding-profile PRIVATE_V4_PROFILE --mini-key-file PRIVATE_MINI_KEY --runtime-dir SDK_RUNTIME --run-dir PRIVATE_RUN --port 18119`.

Credentials and run databases are external to this source checkout. Model/embedding sends are single-attempt transports; saved exact success is reused, failed/unknown stays held. The independent Smoke has an operational ceiling of384 native model sends including preflight, not a newly claimed upstream hard limit. Source/model receipts are private evaluation-only operational data, kept for at most30 days and never used for training, tuning or public case analysis.

This tag is an independent Smoke release. Main and the previous Textual/Coding backends remain unchanged. Run at most one Smoke and no automatic resume or Full. The public Textual root is temporarily routed here only for the authorized test and restored afterward.

Original upstream and base release attribution are retained below.

---

# clmemory method profiles

This repository includes two independently frozen Textual profiles:

- `clmemory` v1.0.5: attributed MemMachine event retrieval and a final evidence filter.
- `invmem-v4-baseline8-v1.0`: independent InvMem algorithm adaptation, original
  top8 recipe, text-embedding-v4 / 1024 dimensions, no generative LLM or final filter.
  See [InvMem configuration, provenance and validation](docs/release-invmem-v4-v1.0.md).
- `v1.0.5-nofilter-textual`: the original clmemory v1.0.5 delivery with the entire
  final filter explicitly removed. It retains top100 and v4/1024 embeddings.
  See [the ablation contract and checks](docs/release-v1.0.5-nofilter.md).

The InvMem profile is a separate evaluation version, database and embedding cache.
The previously published v1.0.5 tag and its deployment sources remain frozen.
The following sections document the clmemory profile.

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

## Native MemMachine baseline

The separate `memmachine-event-v4-v1.0` profile uses native EventMemory with a 100-result cap and no clmemory append/filter. See [configuration and limitations](docs/release-memmachine-event-v4-v1.0.md).
