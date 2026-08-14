Composition rules are now registered where they are defined, and rules
about a particular codec or chunk grid live with that entity.

- `@document_rule` / `@entity_rule` register a rule as a side effect of
  defining it, so `ZARR_V3_ARRAY_RULES` and friends are assembled from
  the registry instead of hand-written tuples: a rule can no longer be
  defined and left out of the set it belongs to.
- Both decorators check a rule's declared `requires` against the document
  type's known keys at import time, and `@entity_rule` additionally
  requires the entity to have a shape validator. A rule whose dependency
  is misspelled, or that targets an entity the package does not model,
  can never fire — and a rule that never fires is indistinguishable from
  one that always passes. Both now raise at import.
- Entity rules live in `zarr_metadata.rules._entities`, one module per
  codec or chunk grid, auto-discovered on import. Adding a third chunk
  grid means adding its types, its shape validator, and its rules module;
  it means no edit to the array-rules module, which now carries only
  generic dispatchers for the `chunk_grid` and `codecs` fields.
- `Rule.keys` is renamed `Rule.requires`, and `inapplicable()` joins
  `applicable()` so a caller can ask which rules did *not* run rather
  than confusing "never evaluated" with "passed".

The engine's docstring now cites its prior art rather than presenting the
design as novel: the fire-only-when-dependencies-are-present gate is
Ecto's `validate_change/3`, Clojure spec's two-phase `s/keys`, and
Valibot's `partialCheck`; presence-conditional rules-as-data are JSON
Schema's `dependentSchemas`. It also records the two properties that
follow — the gate is order-free (no topological sort, so unlike Yup
cyclic dependencies are expressible), and absence is deliberately
inexpressible, being negation-as-failure over an open world.

Sharding's recursion improved in passing: inner pipelines are now judged
against a synthetic document whose `shape` and `chunk_grid` describe the
inner chunk, so a rule written against "the shape" is correct at any
nesting depth. A transpose inside a shard is now checked against the
inner chunk's rank, which it previously escaped entirely.
