# zarr-storage design (v0)

Status: draft, 2026-11-17. This is the new storage API from the storage plan
(`proposals/store-hierarchy-layering.md` in the planning worktree). It is built in iterations; each
iteration ends with every backend passing the shared conformance suite.

Sources, and what each contributed:

- **zng_storage** (Davis Bennett's earlier prototype): capability protocols instead of a class
  hierarchy, a sync floor with an async family beside it, generations and result types, `store / "sub"`.
- **TensorStore**: a store is a driver plus a path (`KvStore{driver, path}`); generations with
  "unconditional" and "absent" meanings; reserved file-name suffixes for the file driver's own files;
  in-flight operations outlive the last reference.
- **zarrs**: `list_dir` returning keys and prefixes; the `StoreKey`/`StorePrefix` grammar; a byte-range
  type with suffix ranges; a missing key is a value (`Option<Bytes>`), not an error.
- **zarr-python-planning store proposals**: `ReadResult`/`PutResult`/`ObjectMetadata`, conditional
  writes that report `applied=False` instead of raising, the conformance-suite layout.
- **The storage plan**: directory prefixes, `Key | Prefix | Entry` listing, `delete` vs `delete_dir`,
  views with the root applied once, the constructed store owns the backend, no value equality.
- **The roadmap**: separate sync and async surfaces, reads into caller-provided buffers, typed explicit
  configuration, transactions later.

## Shape

```
Backend   capability protocols over backend paths; owns shared state and lifecycle
   ▲
Store     one concrete class: (backend, root, read_only, owner); validates keys, applies the root,
          relativizes listings, enforces read-only; `store / "a"` is a view on the same backend
```

A **backend** is an object that implements some capability protocols (`Read`, `Write`, `List`, and their
async twins) on *backend paths*: `/`-separated strings, already joined with the root. Backends never
see relative keys and never join or strip paths. They hold everything that must exist once: the dict,
the filesystem root, a client, locks, concurrency limits.

A **`Store`** is the only class users hold. It is a view: a backend plus a root. All key validation,
root joining, listing relativization and read-only enforcement happen in `Store`, once, so no backend can
get them wrong on its own (the source of most bugs in the legacy API: #2190, #2467, #3639, #3922, #4204,
#4452).

## Keys and directories

- **Key:** a string with no leading `/`, no trailing `/`, no empty segment, no `.` or `..` segment, no
  NUL character, and not empty. Validated by `Store` on every key argument; `InvalidKeyError`
  otherwise. A backend may refuse more with `InvalidKeyError`: the local backend refuses its own
  reserved suffix (as TensorStore's file driver does), names too long for the filesystem, and on
  Windows the names Windows reads as something else (`:`, `\`, device names like `NUL`, a trailing `.`
  or space).
- **Directory:** every prefix argument names a directory. `""` is the root; `"a"` and `"a/"` are the same
  directory; otherwise it follows the key grammar. `list("lat")` never sees `lat_bnds/…`.
- **Root:** a store's root is a directory in the backend. `store / "a/b"` appends to it.

## Operations (sync; each has an `_async` twin)

| Method | Returns | Notes |
|---|---|---|
| `get(key, *, byte_range=None)` | `ReadResult \| None` | `None` when the key is missing (zarrs, TensorStore). Conditional reads (`if_not_match`) arrive with caching in iteration 3 |
| `get_ranges(key, ranges)` | `RangesResult \| None` | all parts from one generation (object stores pin later reads to the first read's generation, and fall back to one whole read if it changed) |
| `get_into(key, out, *, byte_range=None)` | `int \| None` | bytes written into `out`, any writable C-contiguous buffer; `ValueError` if it is too small, read-only or not contiguous |
| `head(key)` | `ObjectMetadata \| None` | size, generation, last-modified |
| `put(key, value, *, if_match=None, if_none_match=False)` | `PutResult` | `value` is any C-contiguous buffer, copied; a failed precondition returns `applied=False`, it doesn't raise; `KeyConflictError` where the backend can't hold a key and a directory with one name |
| `delete(key, *, if_match=None)` | `DeleteResult` | one key; deleting a directory name deletes nothing |
| `delete_dir(directory)` | `None` | everything under the directory |
| `list(directory="")` | `Iterator[str]` | every key under the directory, relative to the store's root |
| `list_dir(directory="")` | `Iterator[Key \| Prefix \| Entry]` | direct children, names relative to the directory; `Entry` when the backend can't tell; a `Prefix` may hold no keys (an empty directory on a filesystem) |

Listing arguments are checked when the method is called; the items are produced lazily, and taking
one after the store closes raises `StoreClosedError`.

**Byte ranges** (zarrs `ByteRange`, TensorStore `OptionalByteRangeRequest`): `RangeRequest(start, end)`,
`OffsetRequest(start)`, `SuffixRequest(length)`. An end past the value is clamped; a start past the value
raises `InvalidByteRangeError`; a suffix longer than the value returns the whole value.

**Generations** are opaque and compared only by equality. Backends mint them: a counter (memory),
`(device, inode, mtime_ns, size)` (local, as TensorStore's file driver), an ETag (object stores).

## Views, lifecycle, identity

- `store / "a"` and `store.with_read_only(True)` return views on the same backend. A view of a
  read-only store is read-only; `with_read_only(False)` on one raises `ReadOnlyError`.
- The store the user constructed owns the backend. `close()` on it closes the backend; on a derived
  view it does nothing. After the backend closes, every view raises `StoreClosedError`.
- No value equality: `==` is identity. `shares_backend(other)` and `overlaps(other)` answer the questions
  callers actually ask.
- Pickling carries the backend's configuration and the view's root. A pickled memory store carries a
  copy of its data.

## Errors

`StorageError` is the base. `InvalidKeyError` and `InvalidByteRangeError` also subclass `ValueError`;
`UnsupportedOperationError` also subclasses `NotImplementedError`. `KeyConflictError`, `ReadOnlyError`
and `StoreClosedError` complete the set. A missing key is not an error, and neither is reading under a
key (`a/b` when `a` is a key): that reads as absent.

## Backends

| Backend | Generations | Conditional `put` / `delete` | Key and directory with one name | Pickles |
|---|---|---|---|---|
| `MemoryBackend` | counter | yes / yes | yes | a copy of the data |
| `LocalBackend` | `(dev, inode, mtime_ns, size)`, with each write stamping a unique ns mtime | yes / yes, on filesystems that keep ns times | no (`KeyConflictError`) | the root |
| `ObjectStoreBackend` | ETag and version (obstore's `LocalStore` ETag can repeat on ext4) | if the store supports it / no | as the store does | as the store does |

`LocalBackend` stages each write in a reserved directory under the root and renames it into place,
holding the lock of the key's directory; the lock lives in that directory, so overlapping roots and
other processes share it. Locks are always taken ancestor before descendant. `delete_dir` removes a
tree bottom-up under each directory's lock, so a write in progress finishes before its directory
goes. Deletes, and writes whose precondition fails, can leave directories behind (pruning raced
every concurrent write). A later write of a key with such a directory's name reclaims it with
`rmdir`, only if no key lies under it, so a reclaim can never remove a key; otherwise the write
raises `KeyConflictError`, whatever its preconditions. A symlink in a key's place is that key's
file. Each write stamps its file with a nanosecond mtime unique within the process, because
filesystems such as ext4 reuse inodes at once and advance mtimes only per clock tick; on a
filesystem that rounds file times, conditional writes raise `UnsupportedOperationError`.
`LocalBackend` doesn't sync to disk, and needs a root that doesn't span filesystems.

`ObjectStoreBackend.get_ranges` reads the first range unpinned, takes the generation and size from
that response, and reads the rest pinned to it (`If-Match`, and checked against each response);
on a mismatch it reads the value once, whole, and slices it. It does not use obstore's own
`get_ranges`, which can panic when a value shrinks mid-read. Only obstore's `LocalStore` reads a
refusal under a file as an absent key, and only there does a write reclaim an empty directory in
its way; for every other store, an error is raised. obstore's `LocalStore` has no locking, so
writes that concurrently create and remove the same directories can surface its errors.

## Conformance suite

`zarr_storage.testing.StoreSuite` holds the contract every backend meets; optional capabilities come
as mixins a backend's test class adds (`ConditionalPutSuite`, `KeyAndDirectorySuite`,
`PickleSuite`, ...) or their refusing counterparts (`NoConditionalPutSuite`,
`KeyDirectoryConflictSuite`, `ConditionalRaceSuite`, ...). Every test runs twice: through the sync
methods, and through the async methods on one event loop, so the two surfaces meet one contract.

`StoreSuite.test_state_machine` drives random sequences of puts, conditional puts, deletes,
conditional deletes, `delete_dir` and listings through random views, and checks every step against a
dict model (`zarr_storage.testing._stateful.StoreMachine`), again through both surfaces. The obstore
tests also fail any async run that makes a blocking obstore call.

## Configuration

No global configuration. Every knob (local I/O concurrency) is a constructor parameter with its
default in the signature.

## Open

- Durability: no backend syncs to disk; a `sync` option on `LocalBackend` would.
- Staged files left by a crash are never read but are never removed either.
- Windows rules are implemented and unit-tested, not yet run on Windows.

## Iterations

1. **Core:** types, `Store`, the memory and local backends, the conformance suite in
   `zarr_storage.testing`, pickling.
2. **Remote and archive backends:** obstore (S3, GCS, Azure, HTTP; async-native), fsspec, zip.
3. **Composition:** range coalescing, caching, retry, a sync bridge for async-only backends, URL specs.
4. **Transactions:** staged writes committed with generations, memory first.
