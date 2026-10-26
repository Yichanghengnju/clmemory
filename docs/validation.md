# Engineering validation of the first release

Date: 2026-10-26. This document reports engineering checks, not a benchmark or
an official smoke score. The two track profiles are `v1.0-textual` and
`v1.0-coding`, from the same source commit with separate runtime state.

The synthetic tests exercise the real native MemMachine storage/retrieval
stack, using a test-only hash embedder. Eleven tests cover authentication,
idempotency/conflict, restart persistence, user/track isolation, response-count
bounds, concurrent Add, head preservation, Coding tail behavior, final-filter
selection/audit, credential guards, malformed decisions, held transport,
oversized input, and the actual HTTP exit-filter assembly. A fixture test
checks an individual behavior; this count is not a scored-task denominator.

Twelve real `gpt-4o-mini` engineering fixture checks (six per track) passed for:

- actual current state versus an unaccepted proposal;
- historical state retained when the question asks about history;
- explicit unauthorized disclosure requests;
- malicious injected instructions versus a legitimate historical test command;
- retaining an unaccepted proposal when the question asks what was suggested;
- an additional actual-purchase example with a rejected alternative.

Earlier filter candidates failed two role/intent checks. Those failures were
retained and the policy was revised before this release; the final checks are
engineering regression checks, not independent evidence of method improvement.
The returned model identity was `gpt-4o-mini-2024-07-18`; that is provider-reported
identity, not an independent attestation of the serving provider's internals.

Each track's real public HTTP path was checked with `text-embedding-v4 / 1024`
and the mini filter: unauthenticated health, rejected unauthenticated Add,
synchronous Add, exact duplicate replay, changed-payload 409, immediate Search,
top_k=1/16/100, cross-user empty results and invalid-top_k rejection.

No claim is made about Full-scale throughput, broad safety effectiveness,
overall QA improvement, or an official evaluation result. The filter can make
false keep/drop decisions outside these fixtures. The access-review services
start with a finite 200-call mini budget per track and single-track concurrency
settings of Add=1 and Search=1; Full must be capacity-planned separately.
