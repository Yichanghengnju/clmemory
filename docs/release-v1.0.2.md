# v1.0.2: complete final-filter decisions

Date: 2026-10-27. This release is intended for the Textual deployment. The
separate Coding repository and frozen deployment remain unchanged.

With a larger candidate batch, the v1.0.1 selector could return only its kept
items. The response still conformed to the old JSON schema, but the local
validator required one decision per candidate and correctly rejected the
incomplete result as a service error. A synthetic 28-candidate reproduction
returned six keep rows and omitted every drop row. This reproduces a concrete
protocol defect; the earlier official failure did not retain its complete
provider response, so its precise validation failure cannot be established.

The output schema now requires an object property for every input evidence ID.
Each property contains its keep/drop decision and reason. The adapter verifies
the exact ID set and reconstructs decisions in input order before applying the
existing validator. It retains the exact source bodies and relative order of
kept evidence. Missing/unknown IDs, refusals and other malformed results remain
service errors, with no unfiltered fallback or automatic retry.

Invalid responses now retain the validation phase and a private response file
reference and digest. These diagnostic files belong only in the private run
directory, never the public repository. Operators must apply the evaluation
data-retention policy to these files as well as the existing stores and ledgers.

The model remains GPT-4o-mini and embeddings remain text-embedding-v4 / 1024.
Native retrieval, delivery, credential guards, private per-item audit and the
shared model lock are unchanged. The existing physical-call ledger and finite
call cap continue across this protocol revision; no budget or failure reset is
introduced. Old source tags, attempts and results are retained.

Validation of this protocol candidate: 15 engineering tests passed, including
complete 28-item output, cached success reuse and held missing-ID/refusal cases.
Six real-model Textual synthetic role/history/privacy/injection checks passed.
A real-model 28-item case returned all 28 decisions, retaining four of six
equivalent relevant copies and zero wrong-entity items. It preserved the fact
needed to answer the synthetic question, but did not retain every relevant copy.
These are bounded engineering checks, not an official Smoke pass, a benchmark
improvement or a general safety guarantee. No official evaluation content was
used to develop this repair.

An earlier fixed-length-array candidate passed shape validation but removed all
relevant evidence in that synthetic case. It was rejected and not deployed.
