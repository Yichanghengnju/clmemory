# InvMem with text-embedding-v4: original top8 baseline

This profile independently implements the public InvMem/vanilla-rag-memory
algorithm referenced at commit `31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca`.
It is our existing independent adaptation, not a redistribution of upstream
source or a claim to reproduce an identified leaderboard deployment.
The pinned-source audit did not identify an upstream code license; no upstream
code is copied. Our adapter's committed-source hash and export provenance are
recorded in `invmem-export-provenance.json`.

The user selected this baseline explicitly:

| Setting | Frozen value |
| --- | --- |
| Chunk size / overlap | 320 / 40 local regex tokens |
| Dense / positive-BM25 RRF weights | 1.0 / 0.5 |
| RRF constant / seeds | 60 / 20 |
| Same-session neighbor window | 1 |
| Maximum returned evidence | 8, including when the caller requests top_k=100 |
| Embedding | text-embedding-v4, 1024 dimensions, L2 normalized |
| Embedding document/query prefixes | None |
| Generative Add/Search calls | Zero |
| Final LLM filter / reranker | None |

Chunking, ranking, neighbor order and source slicing retain the original adapter
behavior. The optional observation callback records actual candidate scores,
ranked IDs and selected IDs in a private SQLite audit, together with the exact
request. The trace has no effect on selection. The original sources, chunk
offsets, roles, timestamps and vectors stay in the private MethodStore.
Failure to save a trace is reported separately rather than fabricating a record.
No benchmark questions, source data, labels, keys or private profiles are public.

The HTTP assembly serializes embedding-provider access to one slot, retaining
the underlying embedding identity and exact-input persistent cache. Concurrent
identical cache requests wait for the successful result. This is resource
coordination and does not add a reranking or generation stage. Existing cache
successes may be reused only under the same text/model/dimension/input-type
identity; previous failed or unknown attempts are preserved. Each exact provider
input retains the embedding client's finite attempt limit.

37 offline method/HTTP checks pass, including an observation/no-observation
equivalence check, concurrent Add/cache reuse, user isolation, authentication,
chunk provenance and top8 limits. A synthetic real-v4 Add/Search engineering
probe also passes: 8 returned items, required source present, repeated Add
reused, unknown user empty, missing authentication rejected. It made two new
embedding requests and zero generative requests; successful compatible query
embedding was reused. These are engineering checks, not benchmark scores.

Use the existing package's base installation; this profile does not require
the optional native MemMachine runtime. Create a private v4 embedding profile
and a private service token file, then run:

```sh
python -m agent_memory.invmem_submission \
  --db run/invmem/memory.sqlite3 --api-key-file /private/service-token.key \
  --embedding-profile /private/embedding.json --manifest-dir run/invmem \
  --host 127.0.0.1 --port 18106
```

The CLI requires the real v4/1024 profile. Synthetic embedders are test fixtures
only. HTTP admission is 128, with one service worker. Keep this recipe, its
storage and evidence separate from clmemory v1.0.5 and Coding. Comparing its
official Smoke with clmemory evaluates different complete methods: clmemory
uses a larger candidate delivery set and a final LLM filter. It is not an
embedding-only controlled comparison between those two methods.

Algorithm reference:
[vanilla-rag-memory pinned source](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory/tree/31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca).
