# clmemory v1.0.5 without the final filter

This is a separately identified, user-requested method ablation. It is not a
fallback for an error and does not alter the frozen filtered v1.0.5 release.
The 14 base implementation files named in `nofilter-base-provenance.json`
match tag `v1.0.5` / commit `2fd89a0e12b71e194961a872d3ffd1a1648eebfb`
byte for byte. The new entry point uses the original `build_app` with
`final_filter=None` and labels the resulting manifest `v1.0.5-nofilter-textual`.

Unchanged: whole-text Add, native MemMachine event backend pinned to
`d5a39a8e774e977fb1e6c22efb03555c4e51e05e`, BM25/dense RRF, top16 head,
up to8 neighbors and8 anchors, deeper source delivery up to100, source bodies,
embedding text-embedding-v4 /1024/L2, user isolation, API authentication,
generic error diagnostics and 128-request HTTP admission with one worker.

Removed as one complete stage: the final relevance/entity/time/state and
safety/privacy selection, including its common-credential regex guard and
generative model call. All valid candidate evidence from original delivery
passes through. It no longer has the filtered method's evidence-denial behavior.
Authentication and tenant/source isolation remain in force. This distinction
is intentional and disclosed; it is not claimed to be a safety improvement.

Eight relevant method/HTTP checks pass: source-preserving top100 long output,
exact equality to original delivery, retained top16 head, no filter construction,
manifest identity, concurrent requests, auth/isolation/409 and complete stage
removal using synthetic data. A real-v4 synthetic HTTP probe returns100 items
and183466 response bytes, finds the required source, reuses a repeated Add,
rejects missing authentication and isolates an unknown user. It uses12 new
embedding requests and zero generative requests. These checks are not official
scores. Compatible embedding cache records may be copied with all failures and
unknown outcomes preserved; source caches and historical runs are not modified.

The ablation does not add raw Search-content audit logging. Runtime storage
needed for evaluation remains private and subject to the official evaluation
data-use and retention rules. Only independent synthetic inputs are used for
preflight; no official question content is used to tune this ablation.

Run with the same native dependencies as the original clmemory method:

```sh
python -m agent_memory.nofilter_submission \
  --db run/nofilter/memory.sqlite3 --api-key-file /private/service-token.key \
  --embedding-profile /private/embedding.json --manifest-dir run/nofilter \
  --host 127.0.0.1 --port 18107
```

The CLI requires real text-embedding-v4 /1024 and accepts no final-filter
profile. Add and Search make zero generative LLM calls. This release does not
modify the independent InvMem or Coding recipes, sources, records or results.
