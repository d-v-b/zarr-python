Added `zarr_metadata.rules`: composition rules for full metadata
documents, promoting the rule engine out of the builder into a layer of
its own. The package now models metadata in three layers with one
contract each — `model` checks structure element by element,
`rules` judges composition across the document, and `builder` constructs
while applying both.

- **Rule sets**: `ZARR_V3_ARRAY_RULES` grows to twelve rules — fill
  value vs. data type, codec pipeline kind ordering, known-name shapes,
  dimension-name counts, chunk-grid values (positive extents) and
  geometry (regular rank; rectilinear rank and per-dimension chunk-size
  sums, RLE pairs included), transpose orders (self-permutation at any
  depth, rank agreement with `shape`), and sharding (inner `codecs` and
  `index_codecs` judged as pipelines recursively at every nesting depth;
  inner chunk shapes positive, rank-matched, and evenly dividing the
  enclosing chunk, recursively). New `ZARR_V2_ARRAY_RULES`
  (chunks/shape rank agreement) and `ZARR_V3_GROUP_RULES` (inline
  consolidated metadata recurses, judging each embedded child document
  by its own rules at its path).
- **Read-side trios**: `validate_*` / `is_*` / `parse_*` for array and
  group documents in both format versions mirror the model layer's
  grammar with a stronger judgment — structure *and* composition, every
  problem reported together, JSON arrays normalized to tuples before
  judgment. The `is_*` functions deliberately return `bool` rather than
  `TypeIs`: a composition-invalid document is still an instance of the
  TypedDict, so only the structural layer can narrow honestly.
- **Boundary change**: two composition checks that lived in the
  structural validator moved here — v3 `dimension_names` vs `shape` and
  v2 `chunks` vs `shape` rank agreement. `zarr_metadata.model`'s
  validators, parsers, and dataclasses now accept those documents (they
  are lossless, structurally well-formed representations of what a store
  may contain); use the `rules` trios to judge them. This also removes
  the double report the overlap used to produce.
- **Strictness stance**, now documented on the package: `zarr_metadata`
  models canonical documents and is deliberately stricter than any given
  implementation; implementations coerce ambiguous input as they see fit
  and then validate the canonical result.

Known follow-up: v2 fill-value/dtype consistency (NumPy dtype grammar)
has no rule yet.
