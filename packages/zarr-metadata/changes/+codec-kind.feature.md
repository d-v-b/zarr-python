Added `zarr_metadata.v3.codec.kind`: codec kind classification. One
branded union per spec pipeline kind over the concrete codec types
(`ArrayArrayCodecMetadata`, `ArrayBytesCodecMetadata`,
`BytesBytesCodecMetadata`, plus `KnownCodecMetadata` and the paired
`*_CODEC_NAMES` constants), and `TypeIs` guards
(`is_array_array_codec`, `is_array_bytes_codec`, `is_bytes_bytes_codec`,
`is_known_codec`) that narrow a codec entry to the matching union. The
guards are **shape-exact**: `TypeIs` narrowing is two-sided, so a guard
answers `True` exactly when the value is an instance of a canonical
codec type — bare spellings only where the codec's spec permits them,
object forms deep-checked against their TypedDicts (key sets and
configuration value types, judged at the canonical data level with JSON
arrays as tuples and `int` meaning JSON integer, not boolean). Unknown
codec names answer `False` to every guard, so extension codecs classify
as "unknown kind" rather than being misassigned. The companion
`codec_kind_of_name` classifies by name alone, returning a `CodecKind`
literal or `None`, for ordering semantics where a known codec in an
invalid spelling must still rank as its kind. All names are re-exported
from `zarr_metadata.v3.codec`.
