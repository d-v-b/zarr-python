Added `ChunkKey`, a `typing.NewType` brand over `str` returned by `encode` —
the static analogue of the validated `StoreKey` newtype in the `zarrs` Rust
implementation. `decode` and membership testing still accept plain `str`,
since their job is judging untrusted input; code that has validated a string
by other means may brand it directly with `ChunkKey(candidate)`.
