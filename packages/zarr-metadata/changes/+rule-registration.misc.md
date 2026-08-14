Composition rules are now registered where they are defined, and rules
about a particular codec or chunk grid live with that entity.

- `@document_rule` and `@entity_rule` add rules to their registry.
- Both validate `requires` against the document type at import time;
  entity rules also require a modeled identifier and shape validator.
- Entity modules are discovered automatically and dispatched by field
  and canonical name.
- `Rule.keys` is renamed `Rule.requires`, and `inapplicable()` joins
  `applicable()` to expose rules skipped for missing dependencies.

Sharding inner pipelines are now judged
against a synthetic document whose `shape` and `chunk_grid` describe the
inner chunk. Nested transpose and sharding rules therefore use the
correct rank and chunk geometry.
