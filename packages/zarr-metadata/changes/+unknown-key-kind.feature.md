Unknown members inside a *known* entity's `configuration` (e.g. an extra
key in a `blosc` configuration) now report as their own `unknown_key`
problem kind, and no longer suppress the other rules about that entity.

The Zarr v3 spec constrains a `configuration` only to "be an object" and
never says whether it is closed; `must_understand` is defined over
metadata *document fields*, so it cannot reach inside a configuration.
The question has been open since 2023
([zarr-specs#270](https://github.com/zarr-developers/zarr-specs/issues/270)),
filed after a real interop break — jzarr emitted
`blosc.configuration.numThreads` and zarr-python refused to open the
array. This package keeps the strict reading (matching most registered
extension schemas and the zarrs, tensorstore, and zarr-java
implementations), but makes it survivable:

- the dedicated `unknown_key` kind lets a caller filter for tolerance
  instead of string-matching messages;
- rules that read a known entity's fields now ignore `unknown_key` when
  deciding whether the entity is interpretable, so a cosmetic extra key
  can no longer silently hide a genuine chunk-geometry or codec error
  on the same entity — previously it suppressed every other check there;
- documents carrying unmodelled members round-trip byte-faithfully, now
  covered by a regression test using the jzarr case as its fixture.

The `TypeIs` guards in `zarr_metadata.v3.codec.kind` deliberately stay
exact: narrowing is two-sided, and a value with an extra member is not an
instance of a closed TypedDict.
