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
- **zarr-python-planning store proposals**: result types and `ObjectMetadata`, conditional writes that
  report a refusal instead of raising, the conformance-suite layout.
- **HTTP and Rust `object_store`**: a missing key and an unchanged value are outcomes (`404`, `304`),
  not errors; conditional operations are their own methods (`put` vs `put_opts`).
- **The storage plan**: directory prefixes, `Key | Prefix | Entry` listing, `delete` vs `delete_dir`,
  views with the root applied once, the constructed store owns the backend, no value equality.
- **The roadmap**: separate sync and async surfaces, reads into caller-provided buffers, typed explicit
  configuration, transactions later.

## Shape

```
Backend      capability protocols over backend paths; owns shared state and lifecycle
   ▲
_View        (backend, root, read_only, context, owner): validates keys, applies the root,
   ▲         relativizes listings, enforces read-only, limits and schedules every operation with
             its context's resources; `store / "a"` is a view on the same backend
Store        the operations as sync methods; `close()`, `with`; `as_async()`
AsyncStore   the same names as coroutines; `aclose()`, `async with`; `as_sync()`
```

A **backend** is an object that implements some capability protocols (`Readable`, `Writable`,
`Listable`, `WritableIfMatch`, `DeletableIfMatch`, and their async twins) on *backend paths*:
`/`-separated strings, already joined with the root. Backends never see relative keys and never
join or strip paths. They hold everything that must exist once per backend: the dict, the
filesystem root, a client, locks. They hold no thread pools or limits: a backend names the
`resource_key` its operations are limited under, and the store draws the resources from its
context (see "Contexts"). Backends must be thread-safe.

A **`Store`** or **`AsyncStore`** is the only object users hold. Each is a view: a backend plus a
root, under a context. All key validation, root joining, listing relativization, read-only
enforcement and resource limiting happen in their shared base, once, so no backend can get them
wrong on its own (the source of most bugs in the legacy API: #2190, #2467, #3639, #3922, #4204,
#4452).

**Sync and async are two classes** (as zarrs has separate sync and async storage traits): `Store`
calls a backend's sync protocols, and `AsyncStore` its async ones where the backend has them and
its sync ones on the context's thread pool otherwise (zarrs's `SyncToAsyncStorageAdapter`, with
the pool in place of its `spawn_blocking`), under the same method names, so `store.get(k)` and
`await astore.get(k)` read the same way and neither class carries the other's methods. A backend
that only blocks, such as a filesystem, needs no async code. Every backend serves both; `store.as_async()` and `astore.as_sync()` are
views on the same backend that don't own it. Lifecycle differs too: `Store` has `close()` and
`with`, `AsyncStore` has `await aclose()` and `async with`, which awaits a `ClosableAsync`
backend's `aclose` (a backend whose cleanup awaits a client session) and calls `close` on any other.

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

## Operations (`Store`; `AsyncStore` has each under the same name, as a coroutine or async iterator)

| Method | Returns | Notes |
|---|---|---|
| `get(key, *, byte_range=None)` | `Found \| Missing` | |
| `get_if_changed(key, generation, *, byte_range=None)` | `Found \| Missing \| Unchanged` | `Unchanged` without reading bytes when the value still has `generation` (HTTP `If-None-Match`), to revalidate a cached copy |
| `get_ranges(key, ranges)` | `FoundRanges \| Missing` | all parts from one generation (object stores pin later reads to the first read's generation, and fall back to one whole read if it changed) |
| `get_into(key, out, *, byte_range=None)` | `Copied \| Missing` | `ReadableInto[O]` capability; `Copied.size` bytes written into `out`, a buffer the caller allocated, of the type the backend reads into (see below); `ValueError` if it is too small or can't be written |
| `head(key)` | `ObjectMetadata \| Missing` | size, generation, last-modified |
| `exists(key)` | `bool` | |
| `put(key, value)` | `Written` | `value` is any C-contiguous buffer, copied; `KeyConflictError` where the backend can't hold a key and a directory with one name |
| `put_if_absent(key, value)` | `Written \| Rejected` | `Rejected` carries the generation of the value already there |
| `put_if_match(key, value, generation)` | `Written \| Rejected \| Missing` | `WritableIfMatch` capability; compare-and-swap |
| `delete(key)` | `Deleted \| Missing` | one key; deleting a directory name finds it `Missing` |
| `delete_if_match(key, generation)` | `Deleted \| Rejected \| Missing` | `DeletableIfMatch` capability |
| `delete_dir(directory)` | `None` | everything under the directory |
| `get_many(requests)` | `Iterator[(request, Found \| Missing)]` | a request is a key or `(key, byte_range)`, echoed back; lazy, in the order reads finish |
| `get_into_many(requests)` | `Iterator[(request, Copied \| Missing)]` | `(key, out)` or `(key, out, byte_range)`; `out` typed as for `get_into` |
| `put_many(items)` | `list[(key, Written)]` | `(key, value)` pairs; eager, results in the order given; a key named twice raises `ValueError` |
| `delete_many(keys)` | `list[(key, Deleted \| Missing)]` | as `put_many` |
| `list(directory="")` | `Iterator[str]` | every key under the directory, relative to the store's root |
| `list_dir(directory="")` | `Iterator[Key \| Prefix \| Entry]` | direct children, names relative to the directory; `Entry` when the backend can't tell; a `Prefix` may hold no keys (an empty directory on a filesystem) |

**Results are closed unions of small frozen types**, one per outcome, so a `match` handles each and a
type checker sees what's left. A method returns only the outcomes it can produce: `get` can't be
`Unchanged`, `put` can't be `Rejected`. That is why conditional operations are separate methods
rather than keyword arguments, and it lets a capability a backend lacks be a type error
(`delete_if_match` on an object store) instead of a runtime refusal. A refusal some stores can only
discover at runtime (`put_if_match` on obstore's `LocalStore`, or on a filesystem that rounds file
times) still raises `UnsupportedOperationError`. `Missing` and the read results carry `read_at`, when
the backend read, so a cache can age an absence like a value.

**Reads into caller buffers, on any device.** `ReadableInto` is generic in the buffer type the
backend writes into, and `Store.get_into` accepts exactly that type. The built-in backends are
`ReadableInto[Buffer]`: any writable C-contiguous host buffer. A GPU direct storage backend (cuFile,
kvikio) would be `ReadableInto[DeviceBuffer]`, reading from disk straight into device memory, and
passing it a host buffer is a type error (and the reverse). The caller allocates, because the caller
knows the size and the device; the store never needs an allocator, so it stays free of Zarr's buffer
classes (zarr-python's `prototype` argument puts allocation in the store instead). `get` keeps
returning host bytes, which is what metadata documents need. Not yet: `get_ranges_into` (several
ranges into several buffers, from one generation, for sharded chunks) and the write-side twin
`put_from`.

**Batches are for throughput, not atomicity.** `get_many`, `get_into_many`, `put_many` and
`delete_many` each do what a loop of the single operation would, and promise nothing more: each
read finds whatever generation is current when it runs, and a batch of writes can land partly
(`get_ranges`, which reads several parts of one value from one generation, is a consistency
guarantee, not a batch). Each result is paired with its request, so `dict(store.get_many(keys))`
works and order doesn't matter. Reads are lazy and come in the order they finish, so a caller can
decode chunks as they arrive. Writes are eager, because a lazy write nobody iterates would silently
write nothing; they return results in the order given. Every key is checked, and every value viewed,
before any operation starts, and a write batch that names a key twice raises, since which write
wins would be a race. If operations raise, the batch starts no new ones, lets those running finish,
and raises `BatchError`, an `ExceptionGroup` of the failures whose `completed` holds the writes that
landed with their results and whose `remaining` holds the requests that failed or never started
(`except*` keeps both). Conditional operations have no batch form: compare-and-swap coordinates
single documents, and a batch of it belongs to transactions.

`Store` provides every batch method for any backend that can read or write, running single
operations concurrently under the view's context: `Store` on the context's thread pool, `AsyncStore`
as tasks, at most the context limiter's `limit` at once, each holding its own slot. A backend with
a batch capability (`ReadableMany`, `ReadableIntoMany[O]`, `WritableMany`, and async twins) is
used instead, for batching the store itself can do (one request for many keys, one GPU
submission); it receives the context so it can draw on the same resources. The rule for any
fallback: `Store` may fall back when only speed differs, never when behavior does. A batch of single
operations returns what a native batch would; a `get_into` that read and then copied would not put
the bytes where the backend's buffer type says, which is why `get_into` has no fallback. The batch
runners in `zarr_storage._batch` implement the windowing and failure rule once.

Listing arguments are checked when the method is called; the items are produced lazily, and taking
one after the store closes raises `StoreClosedError`.

**Byte ranges** (zarrs `ByteRange`, TensorStore `OptionalByteRangeRequest`): `RangeRequest(start, end)`,
`OffsetRequest(start)`, `SuffixRequest(length)`. An end past the value is clamped; a start past the value
raises `InvalidByteRangeError`; a suffix longer than the value returns the whole value.

**Generations** are opaque and compared only by equality. Backends mint them: a counter (memory),
`(device, inode, mtime_ns, size)` (local, as TensorStore's file driver), an ETag (object stores).

## Views, lifecycle, identity

- `store / "a"`, `store.with_read_only(True)`, `store.with_context(ctx)`, `store.limited(n)` and
  `store.as_async()` (or `as_sync()`) return views on the same backend. A view keeps its context
  unless it is given another. A view of a
  read-only store is read-only; `with_read_only(False)` on one raises `ReadOnlyError`.
- The store the user constructed owns the backend. `close()` (`aclose()` on an `AsyncStore`) on it
  closes the backend; on a derived view it does nothing. After the backend closes, every view raises `StoreClosedError`.
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

| Backend | Generations | `put_if_match` / `delete_if_match` | Key and directory with one name | Pickles |
|---|---|---|---|---|
| `MemoryBackend` | counter | yes / yes | yes | a copy of the data |
| `LocalBackend` | `(dev, inode, mtime_ns, size)`, with each write stamping a unique ns mtime | yes / yes, on filesystems that keep ns times | no (`KeyConflictError`) | the root |
| `ObjectStoreBackend` | ETag and version (obstore's `LocalStore` ETag can repeat on ext4) | if the store supports it / not implemented | as the store does | as the store does |

Every backend has `put_if_absent` and `get_if_changed`. `LocalBackend` is sync only; `AsyncStore`
runs its operations on the context's pool. `MemoryBackend` and `ObjectStoreBackend` are also
natively async. Their resource keys are `"local"`, `"memory"`, and `"obstore."` plus the obstore
store's class name, so every S3 backend in a process shares one limit by default.

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
filesystem that rounds file times, `put_if_match` and `delete_if_match` raise
`UnsupportedOperationError`.
`LocalBackend` doesn't sync to disk, and needs a root that doesn't span filesystems.

`ObjectStoreBackend.get_ranges` reads the first range unpinned, takes the generation and size from
that response, and reads the rest pinned to it (`If-Match`, and checked against each response);
on a mismatch it reads the value once, whole, and slices it. It does not use obstore's own
`get_ranges`, which can panic when a value shrinks mid-read. Only obstore's `LocalStore` reads a
refusal under a file as an absent key (judged from the refusal's own `ENOTDIR`, since looking at
the parents afterwards races a concurrent delete of the parent), and only there does a write reclaim an empty directory in
its way; for every other store, an error is raised. obstore's `LocalStore` has no locking, so
writes that concurrently create and remove the same directories can surface its errors.

## Conformance suite

`zarr_storage.testing.StoreSuite` holds the contract every backend meets; optional capabilities come
as mixins a backend's test class adds (`ConditionalPutSuite`, `KeyAndDirectorySuite`,
`PickleSuite`, ...) or their refusing counterparts (`NoConditionalPutSuite`,
`KeyDirectoryConflictSuite`, `ConditionalRaceSuite`, ...). Every test runs twice: through `Store`,
and through an `AsyncStore` over the same backend on one event loop, so the two classes meet one
contract.

`StoreSuite.test_state_machine` drives random sequences of `put`, `put_if_absent`, `put_if_match`,
`put_many`, `delete`, `delete_if_match`, `delete_many`, `delete_dir` and listings through random views, and checks every step against a
dict model (`zarr_storage.testing._stateful.StoreMachine`), again through both surfaces. After every
step, each stored key reads back as the model says, through `get` and `get_many`, and
`get_if_changed` with its generation is `Unchanged`. The obstore
tests also fail any async run that makes a blocking obstore call.

## Contexts

Resource limits are part of a store's context, as in TensorStore. zarrs has no I/O limits of its
own (its per-call `Resources` carry only a compute target, from Rayon's global pool); it bounds
object-store requests by wrapping the store in object_store's `LimitStore` and sharing that store.

- A **`Context`** holds a `ThreadPool` and a `Limiter` per resource key. It is immutable;
  `ctx.child(threads=..., limits={...})` overrides some resources and takes the rest from its
  parent, and a context with no parent takes the rest from the process-wide defaults
  (`Context.default()`): `ThreadPool.default()` of `max(4, cpu_count)` threads (TensorStore's
  `file_io_concurrency` default) and one `Limiter(32)` per resource key (its
  `s3_request_concurrency` default). A default is shared state, but it changes only speed, never
  what an operation does.
- **Every view carries a context**, `Context.default()` unless given one (`Store(backend,
  context=...)`, `store.with_context(ctx)`). Every operation through the view holds a slot of
  `context.limiter(backend.resource_key)`; blocking work (a sync backend called through an
  `AsyncStore`, and the operations of a `Store` batch) runs on `context.threads`. Stores sharing a
  context share its limits, whatever their backends. Like TensorStore's, a context is not part of a
  store's identity: `shares_backend` and `overlaps` ignore it.
- **`store.limited(n)`** is a view whose operations, together, run at most `n` at once, for a burst
  of work: its context is a child whose limiter for the backend's key is `Limiter(n, within=...)`
  the parent's. A limiter `within` another holds a slot of both, so a scoped limit never exceeds the
  limit it narrows. Here the design departs from TensorStore, where a child context's resource
  replaces the parent's: replacing would let a scoped limit escape the process-wide one.
- **Slots are taken innermost first**, so work queued on a narrow limit holds none of the wide one,
  and **where the work runs**: a sync backend's operation called through an `AsyncStore` takes its
  slot on the pool thread that runs it. No operation holds a slot while it waits for a thread, so
  pool threads waiting for slots and slot holders waiting for threads can't block each other.
- `limiter.slot()` is a sync or async context manager, and a slot is not re-entrant, as with a lock:
  stores take exactly one slot per operation, and never call each other while holding one. An
  earlier draft made slots re-entrant through a context variable; that leaked into every task and
  thread inheriting the context (`asyncio.gather`, and new threads on free-threaded Python 3.14),
  letting a fan-out run unbounded under one slot.
- A batch started on one of its pool's own threads runs on that thread, one operation at a time:
  waiting there for work queued on the same pool could wait forever once every thread does it.
- Whoever creates a resource owns it; the default pool can't be shut down. Pickling a context
  pickles its resources' settings, and pickle stores each object once, so stores that shared a
  context or a resource share one after a round trip; a default unpickles as the receiving
  process's default.
- Pools for I/O and for compute should stay separate (TensorStore's `file_io_concurrency` and
  `data_copy_concurrency`): compute waiting on I/O in one pool starves it. A context holds the I/O
  pool; zarr-python's codec pipeline would take its own.

## Caching

Planned for iteration 3; not built yet. A cache is a resource of the store's context, like
TensorStore's `cache_pool`. It is neither a store class that wraps another store (zarr-python's
`CacheStore`) nor a hook on store methods:

- **A wrapper can't get its capabilities right.** A Python class can't have exactly the capabilities
  of the backend it wraps. It would need one class per combination, or it would claim every
  capability and raise at runtime, losing the typed capabilities.
- **A hook that caches is a wrapper.** To serve a hit, a hook has to skip the read and supply the
  result, and then it is a wrapper.

The rule: wrapping is for layers that change what keys mean (zip, sharding). Policies that leave
the meaning alone (limits, caching, retries) are context resources.

- **A `CachePool(tier, total_bytes)` is a context resource**, set with `Context(cache=...)` or
  `ctx.child(cache=...)`. The default pool holds 0 bytes, so caching is off until a context gives it
  a budget. Stores sharing a context share its pool: one byte budget, one eviction order.
- **The cached bytes live in a store, the tier.** The tier is a memory store by default, a local
  store for a disk cache, or any other writable store. Each cached value is a value in the tier, so
  the cache gets the following from the tier without new code:
  - byte-range reads and `get_ranges`, served from one cached whole value;
  - batches, for the hits of a batch;
  - thread safety and both surfaces (a memory tier answers an `AsyncStore` inline);
  - the conformance suite, as the tier's tests.

  zarr-python's `CacheStore` keeps its bytes in a store too; what changes is where the cache sits.
- **A cache is more than a store.** Beside the tier, the pool keeps an index in memory for what a
  store can't hold. Per entry, the index records:
  - the source's generation (the tier mints its own);
  - the source's `read_at`;
  - the entry's size and recency;
  - absences: a `Missing` from the source has no bytes to store.

  The index also runs eviction, and it combines concurrent misses on one key into one source read.
- **Tier keys are opaque and written once.** The index names each entry's bytes with a fresh key
  under a random per-pool prefix. Tier keys don't come from source keys, because the key grammar
  has no way to name the source backend. A refresh writes a new key and points the index at it.
  Eviction removes the index entry, then deletes the key. So no reader sees a tier value change
  under it, and a tier read that returns `Missing` (evicted, or a disk tier someone else cleaned) is
  a miss. Pools sharing a disk tier don't collide.
- **Entries are keyed by backend instance and path**, the identity `shares_backend` uses. Closing a
  backend drops its entries.
- **Each view decides how stale a read may be**, with `store.with_max_age(seconds)`, default `0`.
  TensorStore makes the same split (`recheck_cached_metadata`, `recheck_cached_data`), because
  metadata and chunks need different answers.
  - A hit younger than `max_age` returns without touching the source.
  - An older hit revalidates with `get_if_changed(key, generation)`. On object stores that's a
    conditional GET: one round trip, no bytes.
  - `math.inf` trusts the cache forever, for values that are never rewritten.
  - An absence ages like a value, but only a full read can revalidate it, so caching absences pays
    only when `max_age > 0`: zarr's metadata probes for format versions, for example.
- **Writes keep a pool coherent.** A write or delete through a view whose context holds the pool
  drops that key's entry, and `delete_dir` drops every entry under its prefix. Writes also bump a
  per-key epoch, and a read that started before a write to its key returns its result but
  doesn't fill the cache. So a view
  sees its pool's own writes whatever its `max_age`. Writes from other processes, other pools, or
  straight to the backend show up only through revalidation, as in TensorStore.
- **Hits hold no source slot.** The tier runs under a context of the pool's own, so a hit waits on
  neither the source's limiter nor the process-wide `memory` limiter. A revalidation or a miss holds
  a source slot like any other read.
- Listings are never cached, and neither is a value larger than the budget.
- Pickling a pool pickles its budget and its tier's configuration, never its contents. The
  receiver starts with an empty index and an empty tier.
- `get_into` into a buffer type the tier can't write (device memory, with a host tier) bypasses
  the pool.
- Every `StoreSuite` runs again through a cached view (a pool with a budget, `max_age=0`), where the
  cache must be invisible. The state machine adds writes made straight to the backend, which only
  revalidation can catch.

## Configuration

No global configuration. Every knob is a constructor parameter with its default in the signature;
resources are passed in a store's context.

## Open

- Durability: no backend syncs to disk; a `sync` option on `LocalBackend` would.
- Staged files left by a crash are never read but are never removed either.
- Windows rules are implemented and unit-tested, not yet run on Windows.
- Listings hold no slot of the context's limiter.
- `get_ranges_into`, `put_from`, and a GPU direct storage backend whose `get_into_many` submits one
  cuFile batch.
- Caching: range reads of values the pool doesn't hold whole go to the source. Shard indexes are the
  case that matters, and they belong to the codec pipeline, not here: the sharding codec knows where
  an index is and can cache it decoded, keyed by store, key and generation. The storage layer gives
  it what it needs: the generation of every read, and `get_if_changed` with a byte range to
  revalidate one.
- Caching: a disk tier starts cold in every process, because the index lives in memory.
- Caching: a pool over a memory backend copies values for no gain; nothing stops one yet.

## Iterations

1. **Core:** types, `Store`, the memory and local backends, the conformance suite in
   `zarr_storage.testing`, pickling.
2. **Remote and archive backends:** obstore (S3, GCS, Azure, HTTP; async-native), fsspec, zip.
3. **Composition:** range coalescing, caching (see Caching), retry, a sync bridge for async-only backends, URL specs.
4. **Transactions:** staged writes committed with generations, memory first.
