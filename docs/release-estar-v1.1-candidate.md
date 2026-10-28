# E* v1.1 implementation and validation boundary

The three changes were directly requested after the v1.0 Smoke had completed.
Official benchmark messages were not read for implementation. The 7 previous
H degradations were classified using saved operational failure metadata only.

The native plugin uses the installed pinned SDK's public ToolRunContext
concludeTurn method and agent/turn-stopping steering hook. The SDK and its loop
are unchanged. Python remains the source capability and single-send transport.
The last model slot selects only already observed, currently visible sources;
the server never invents a finish list. Tool action reservation cannot turn a
foreign read or unresolved model response into success.

Attribute updates retain original user text and immutable source IDs. A
superseded current preference is not inferred to be its opposite. A later
explicit user assertion can establish a new preference; an assistant copy
cannot. Quoted stories, assistant controls, third-person preferences and secrets
have negative controls. A whole source that mixes invalid and valid attributes
may be withheld; no arbitrary subspan rewrite or general semantic guarantee is
claimed. Explicit historical correction queries retain history with the update.

R prefill is bounded by complete source count and character allowance. It uses
the ordinary visibility snapshot; no foreign, revoked, clipped or invented
source is pre-observed. Its initial payload is exact untrusted source data.

The saved comparison has 16 independent synthetic cases per arm, the actual
Mini model, actual v4 embeddings and exact query-vector warm-up before Search.
It tests required current evidence and stale-source absence by source ID. It
does not measure official answer accuracy or prove three separate causal
effects. Combined token/latency changes are useful development evidence only;
the small contexts cannot establish Full-scale throughput or accuracy.

The Full service profile explicitly has cumulative calls/output/input=None.
Accounting and single-query bounds remain. The old finite 384 Smoke guard and
the old teacher budgets are preserved in their original snapshots.
