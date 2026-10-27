# v1.0.5: long-input admission, complete decision constraints and diagnostics

The previous 90,000-byte guard could reject a full source-preserving top100
request before any model task existed. The long-source synthetic check for this
release sends 191,417 bytes, estimated at 35,894 tokens, as one complete request.
The real model returns all required decisions: all six distinct supporting
facts survive, zero unrelated sources survive, and source bodies/order remain
unchanged. This is a synthetic engineering check, not an official score.

The final filter now estimates canonical-JSON tokens using `tiktoken` 0.14 and
`o200k_base`, with a 110,000-token input guard, the existing 8,192-token output
bound, and 9,808 tokens reserved within the documented 128,000-token context.
A separate 2,000,000-byte guard bounds transport memory. No evidence is cut,
summarized or split into independently judged batches. Genuine overflow stays
an explicit service error. The count is a conservative estimate, not exact API
billing usage. Model, retrieval, relevance/safety policy and physical-call budget
remain unchanged. Prior source snapshots and failures are retained.

The model result is wrapped in a required `result` object. Its nested `anyOf`
distinguishes an allowed query from a denied query: denied queries can only
return drop decisions for every required evidence ID. Shared definitions retain
13 enum values at top100. Exhaustive validation covers all 48 one-item
query-decision/reason combinations, with 18 accepted and 30 rejected, matching
the existing strict runtime policy exactly. This closes the six denial/keep
combinations that the prior schema accepted while the runtime rejected.

Safe HTTP error logging records request hashes, exception class, code locations
and numeric guard diagnostics, while retaining generic response bodies. It does
not write authentication headers, raw questions/sources or exception arguments.
The engineering suite contains 30 passing checks, including full long-source
delivery, real token overflow, multibyte/special-token input, exact-result cache,
concurrency and the logging boundary.

For an offline execution host, prepare the tokenizer cache on a connected
machine and copy it to the private service directory. Set `TIKTOKEN_CACHE_DIR`
only for that service. The official `o200k_base.tiktoken` vocabulary has SHA256
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`;
its cache filename is `fb374d419588a4632f3f557e76b4b70aebbca790`.

Sources: [OpenAI model specification](https://developers.openai.com/api/docs/models/gpt-4o-mini),
[official tokenizer](https://github.com/openai/tiktoken),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
The initial `source-export.json` remains the original import provenance;
`release-manifest.json` identifies the current release files.

This is a separately identified Textual release. Coding's independently frozen
repository/deployment and previous official evaluations are unchanged.
