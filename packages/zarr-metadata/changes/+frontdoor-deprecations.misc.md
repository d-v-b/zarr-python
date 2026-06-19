Deprecated the `RoundingMode` and `OutOfRangeMode` names in
`zarr_metadata.v3.codec.cast_value`. They are retained as aliases for the
renamed `CastRoundingMode` and `CastOutOfRangeMode` so existing imports keep
working, and will be removed in a future release. Update imports to the
`Cast`-prefixed names.
