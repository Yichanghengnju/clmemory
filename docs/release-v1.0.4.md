# v1.0.4: pair decision and reason constraints

The previous schema independently allowed keep/drop and every reason code,
while the existing runtime validator required keep to use a keep reason and
drop to use a drop reason. A complete model response could therefore conform
to the schema but still be rejected, for example keep with explicitly_superseded.

The shared per-item definition now uses nested anyOf branches for the existing
valid pairs. This changes the wire constraint, not the relevance/safety policy,
the source contents, model, retrieval method, budgets or held historical inputs.
The local validator remains strict; no failed response is edited into a result.

An offline Draft202012Validator check enumerates all sixteen decision/reason
combinations and establishes exact agreement with the existing per-item runtime
policy: eight valid pairs and eight rejected pairs. The shared schema still has
13 enum values at top_k=100. Denial and complete-ID checks continue in the
existing validator. A previously failed input's three attempts remain preserved;
the old official task is not resumed under this changed implementation.

The v1.0.3 release and its official attempts remain frozen. This release must be
identified separately before another official Smoke is started.
