**Breaking:** `to_key_value` takes a `JSONWriterConfig` instead of an
    `indent` keyword. The config is a frozen dataclass, exported from
    `zarr_metadata` and `zarr_metadata.model`, that holds every choice about
    how the JSON bytes are spelled: `indent`, `sort_keys`, `ensure_ascii`
    and `trailing_newline`. Its defaults are what the writers already did,
    which is how zarr-python writes metadata, so a bare `to_key_value()`
    writes the same bytes as before. Replace `to_key_value(indent=2)` with
    `to_key_value(JSONWriterConfig(indent=2))`.
