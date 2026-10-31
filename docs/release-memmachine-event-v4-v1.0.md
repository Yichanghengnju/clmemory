# MemMachine native event-only baseline for Textual Smoke

Version: `memmachine-event-v4-top100-v1.0`; tag: `memmachine-event-v4-v1.0`.
This is an attributed integration of MemMachine's native EventMemory pinned to
`d5a39a8e774e977fb1e6c22efb03555c4e51e05e` under Apache-2.0. It is not a
reproduction of the unidentified MemMachine deployment on the AML leaderboard.

The existing `method_memmachine.py`, source store, embedding and HTTP adapter
remain byte-identical to the published clmemory v1.0.5-nofilter release. The new
entry point constructs `MemMachineMemory` directly, without clmemory's delivery
subclass, neighbor/anchor append, final filter, or generative model calls.

Native configuration: whole-text derivation, passthrough segmentation,
SQLite/USearch, dense candidates with identity/BM25 RRF (`k=60`), overfetch 4,
context expansion 0, and maximum results 100. Semantic/profile memory,
short-term summary and retrieval agent are disabled. Embedding is
`text-embedding-v4`, 1024-dimensional, L2 normalized. Query options concatenate
without gold labels. Immutable source hydration retains original text, roles,
timestamps and stable IDs.

The 100-result cap matches the clmemory no-filter comparison arm; it is not
claimed to be an upstream default. This experiment compares complete delivery
recipes and does not isolate each neighbor/anchor/ranking effect separately.

Run with the existing pinned native dependencies on PYTHONPATH:

```text
clmemory-memmachine --db /private/run/memory.sqlite3 --api-key-file /private/service.key --embedding-profile /private/embedding.json --manifest-dir /private/run --host 127.0.0.1 --port 18108
```

The service admits up to 128 HTTP connections, serializes provider embedding
cache accesses, and records no extra raw Search audit. Official options remain
Add concurrency 16, Search concurrency 16, top_k 100. Only one new Smoke is
authorized; no automatic resume or Full. Storage and cache are independent.
Original clmemory, InvMem and Coding backends are retained. Keys and evaluation
data must never be published. Preflight uses independent synthetic data only.
