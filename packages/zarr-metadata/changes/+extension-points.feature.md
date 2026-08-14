Added `zarr_metadata.v3._extension_points`: a table of the v3 extension
points — `data_type`, `chunk_grid`, `chunk_key_encoding`, `codecs`,
`storage_transformers` — recording, per identifier, where it was
standardized and where its definition lives, and per point, whether
`must_understand: false` is permitted there and whether the field holds
one entity or a sequence.

Keyed by `(field, name)`, because names are unique only *within* an
extension point: `bytes` is both a core codec and a registered extension
data type, so a flat name-keyed table would have to assign it one
provenance and be wrong about the other.

Provenance has three states, not two. `Provenance.PROPOSED` exists
because the `zstd` codec's cited specification is an open pull request
rather than merged text — anything typed against `BytesBytesCodecMetadata`
today is partly typed against a draft, which the table now says out loud.

**Name canonicalization.** `canonical_name(field, name)` is identity
except for the parameterized raw-bytes data type family, where `r8`,
`r16`, `r24` all reduce to `RAW_BYTES_FAMILY` (`"r<N>"`, the spec's own
notation). Canonicalization is by grammar *shape*, not validity: `r12`
and `r0` canonicalize into the family too, so a malformed member of a
family the package models is reported as a misspelling rather than
mistaken for an unknown third-party extension. Canonical names are
lookup keys only — never emitted into metadata, never shown to a user.

This gives the `^r(\d+)$` grammar a single owner
(`RAW_BYTES_NAME_PATTERN` in `zarr_metadata.v3.data_type.raw`); it was
previously duplicated in the rules layer.

Entity rules are re-keyed by `(field, name)` to match, closing a latent
bug: `numpy.datetime64`, `numpy.timedelta64`, and `struct` all have
configurations and are the obvious next candidates for entity rules, so
a rule written for the `bytes` data type would have fired on the `bytes`
codec. `entity_rule` now takes the extension point, and rejects an
identifier the table does not list at that point.
