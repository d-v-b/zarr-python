Added `BoundedChunkKeyEncoding` and `ChunkKeyEncoding.bind`: an encoding
restricted to a known chunk grid shape. Bounded `encode` and `decode` reject
out-of-grid coordinates and keys (`ChunkCoordsOutOfBoundsError`,
`ChunkKeyOutOfBoundsError`), bounded `decode` is a total inverse of `encode`
(resolving the `v2` encoding's rank-zero ambiguity), and the valid key set is
a finite collection supporting membership testing, iteration, and `len`.
