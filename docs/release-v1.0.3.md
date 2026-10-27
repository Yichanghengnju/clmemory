# v1.0.3: bounded top-100 filter protocol

Date: 2026-10-27. This follow-up retains the published, undeployed v1.0.2 tag.
It completes the same Textual protocol repair; Coding's separate frozen source
and live service remain unchanged.

The model now uses short request-scoped IDs such as e000. The service validates
every required alias and maps its decision back to the original evidence ID.
Original evidence IDs are also included in the cache identity, so identical
content under different source identities cannot reuse an incorrectly bound
selection. Returned source bodies and relative order remain unchanged.

Each per-ID schema references one shared decision definition using $defs/$ref.
At 100 candidates this has 13 enum values, compared with 1003 when the ten
per-item decision/reason enums are repeated. OpenAI documents a maximum of
1000 enum values across a schema and supports shared definitions:
https://developers.openai.com/api/docs/guides/structured-outputs

Eight additional offline checks cover top-100 schema bounds, alias remapping,
cache binding, missing/invented keys, denial handling and credential guards.
The short-ID protocol source was also checked in a real-model synthetic top-100
fixture with six distinct required facts and 94 unrelated items: all six facts
were retained, no unrelated item was retained, and original source IDs and
content were preserved. That protocol source is identified by SHA256
fe04b52d9a9a2c6eba0c534d90b6e169b5c40aff223f4a5d9300a84753574f39.

Integration additionally fixes admission under concurrent Search requests.
Threads queue for the existing single shared model slot, with a bounded wait,
rather than receiving a coordinator-lock conflict immediately. Same-input audit
commits also serialize through a separate track-local lock. Two offline
concurrency checks verify same-input result reuse and maximum model concurrency
of one. No successful model fixture was replayed for this scheduling change.
The CLI also accepts --http-concurrency for independent HTTP admission; it does
not change model concurrency or budgets. The Textual smoke deployment uses 128
HTTP tasks so the platform's minimum Add/Search concurrency does not collide
with a 16-connection server ceiling. This does not establish Full throughput.

Earlier duplicate-fact and pre-shared-definition receipts remain historical;
they are not counted as final-source checks or rewritten into full successes.
The successful synthetic model calls are reused without replay. These checks
do not establish an official Smoke pass, general filtering accuracy or Full
throughput. Existing model identities, finite budgets, held attempts, safety
rules and failure-without-unfiltered-fallback behavior remain in force.
