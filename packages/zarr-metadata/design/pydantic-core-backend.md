# A pydantic-core backend for reading v3 array documents

An experiment, on the branch `claude/zarr-metadata-pydantic-core-backend`,
cut from fork PR d-v-b/zarr-python#407 (`claude/zarr-metadata-read-perf-52cd33`,
head c777c9b3). The question: if pydantic-core does the whole structural
walk of a document, what does a read cost, what of the package's
semantics has to be rebuilt around pydantic's, and is it worth pursuing?

The prototype is `src/zarr_metadata/v3/_fast.py`, behind the optional
extra `zarr-metadata[fast]`; `read_array_v3` in `model/_validation.py`
takes it where it is installed, and `Context` holds its compiled reader.
`design/differential_corpus.py` is the acceptance test, and
`tests/v3/test_fast.py` keeps the two readers to one another in the suite.

## The design

One pydantic-core schema per scope. It performs, in Rust, everything the
Python reader does before it asks a definition's rules:

- refining the document to JSON -- arrays as tuples, string keys, finite
  numbers, integers the interpreter writes;
- judging each extension point's envelope (`name`, `configuration`,
  `must_understand`), and each nested field's;
- picking the definition a name denotes, through a tagged union per kind
  whose discriminator is a Python callable that files a name as the scope
  does (`r16` under `r*`) and sends a name nothing in scope claims to a
  branch that accepts any well-named extension;
- type-checking each configuration against its TypedDict, with the bounds
  its members carry, compiled from the same `typeddict_keys` reading the
  Python checker uses;
- nested fields (a shard's inner codecs, a struct's field types), through
  the same per-kind unions.

What the validator accepts is handed to Python, which builds the
`AcceptedField`/`UnclaimedField` objects, asks `Definition.rules`, the
fill value rules, the chunk grid's shape rules and `chunk_lengths`, reads
the pipeline (`read_pipeline`: order, chunk rules, transitions, inner
pipelines), and hands `read_array_v3` what a reading with no problem is
built of. `canonical`, keys, claims and the models are untouched.

**The validator is an oracle of acceptance, not of problems.** The
package's problems -- `loc`, `kind`, message, `input`, `ctx` -- are the
Python reader's, and the fast path renders none: any document pydantic
refuses, and any document whose rules find a problem, is read again by
the Python walk, which reports exactly as before. So the fast path can
only change what a read costs, never what it says, provided pydantic
never accepts what Python refuses (and never hands back different
values). That is the property the differential run checks.

Problem rendering from pydantic's `type`/`ctx`/`input` was not built,
on purpose. The task asked for identical problems, and pydantic cannot
produce several of the package's problem shapes at all: a key a closed
TypedDict does not declare is `unknown_key` *and the field still reads*,
while pydantic's `extra_forbidden` fails the whole typed dict; a stray
envelope member or a `must_understand: false` is reported beside a
field that is read; a non-string key is reported at the parent with
the key shown; pydantic stops each typed-dict at its first failed field
set, where the Python walk reports the fields a configuration holds as
their own problems and then the rules'. Rendering would have been a
third reader of problems, held to the second by yet another
differential; the fallback makes the Python walk the one reader of
problems, at the price of a second (slower) read of a document that has
one. For zarr's use -- opening arrays that are overwhelmingly valid --
that is the right trade.

## Numbers

Apple Silicon (this machine), Python 3.12.13, pydantic-core 2.46.5,
`PYTHONHASHSEED=0`, best of fifteen `timeit` repeats, the machine otherwise
idle. The benchmark is `tests/benchmarks/read_documents.py`; the
zarr-python number is `ArrayV3Metadata.from_dict` from this checkout's
`src/zarr` on the same document, in the repository's own environment.

| read (best of 15 repeats, the script's batch sizes) | base (#407) | prototype, fast path | prototype, `ZARR_METADATA_READER=python` | ratio |
| --------------------------------------------------- | ----------- | -------------------- | ---------------------------------------- | ----- |
| `ZarrV3ArrayMetadata.from_json`                     | 72.2 us     | 18.4 us              | 71.6 us                                  | 3.9x  |
| `ZarrV2ArrayMetadata.from_json` (untouched)         | 18.0 us     | 18.3 us              | 24.1 us (noise)                          | --    |
| `ZarrV3GroupMetadata.from_json`, 200 consolidated arrays | 14.3 ms | 3.96 ms              | 14.5 ms                                  | 3.6x  |
| zarr-python `ArrayV3Metadata.from_dict`             | 48.5 us     |                      |                                          |       |

Two cautions about these numbers. The machine is noisy: a second round
gave 85.5 and 28.8 us for the first row, and zarr-python's `from_dict`
ranged from 48.5 to 78 us across runs; the ratios held. And the
benchmark's batch size matters: `timeit` disables the garbage collector,
a model and its reading form a reference cycle, and a batch of 6000
reads in a row is slower per read than the script's 2000 for both
readers alike (base 96 us, prototype 35 us, zarr-python 67 us), so the
absolute times are the script's and the ratios are the stable finding.
With the collector on, the prototype reads a v3 array in 20-25 us.

Where the prototype's time goes, for the v3 array document: `cProfile`
over 4000 reads in a row, each function's share scaled to the 32 us a
read costs in that regime (`validate_python` alone is 3.7 us by
`timeit`).

| step                                                       | about | share |
| ---------------------------------------------------------- | ----- | ----- |
| `SchemaValidator.validate_python`: the whole structural walk, of which five discriminator calls back into Python are 2.5 us | 4.2 us | 13% |
| building five fields: order checks, `trusted_field` (5.5 us, of which `frozen` views 2.6 us) | 8.0 us | 25% |
| the rules in Python: the fill value's checker and rules 2.1 us, `chunk_grid_lengths` 2.8 us, `read_pipeline` 7.4 us (order, `bytes` chunk rules, `Chunk`s, `ruled`/`asked` wrappers) | 12.3 us | 38% |
| the model: `FastArrayRead`, `ArrayMembersV3`, `ZarrV3ArrayMetadataReading`, `_adopt` (`frozen` ×3, `dataclasses.replace`) | 3.0 us | 9% |
| the rest: `read_array_v3` glue, `reader_for`, `from_json` | ~4 us | 15% |

Fixed costs:

| cost                                                | measured |
| --------------------------------------------------- | -------- |
| `import pydantic_core`                              | 35 ms in a fresh interpreter (the package's own import is 170 ms; with the fast path on it measured 165 ms, so the extra is within the noise of that) |
| compiling `CORE_AND_EXTENSIONS` (33 definitions, 91 schema definitions) | 4.0 ms, of which building the `SchemaValidator` itself 0.5 ms; the rest is the Python schema compiler |
| compiling `CORE` (24 definitions, 65 schema definitions) | 2.5 ms (0.34 ms in Rust) |
| a validator for a document at another depth (a consolidated child) | about 0.4 ms, once per depth |

A scope compiles once; a scope built again -- unpickled with a model,
`Context.of` of the same definitions -- finds the reader of its equal
through a dictionary keyed by `Context`'s equality (`_READERS` in
`_fast.py`), so unpickling a model costs a hash of the scope, not a
compile. The first differential run showed why: `pickle.loads` of every
model built a fresh `Context`, and the reader compiled 49 times in 600
cases.

## The semantic gaps, and how each was handled

Every one of these was found either by a probe before the design or by
the differential run. "Rebuilt" means the schema expresses it; "fallback"
means pydantic refuses more than Python and the Python walk reports it,
which is correct and only slower.

| the package says                                          | pydantic says                                            | handled |
| --------------------------------------------------------- | -------------------------------------------------------- | ------- |
| a list is read as a tuple; a `range`, a `deque`, any `Sequence` too | strict `tuple` refuses a list; lax takes a `set`, a generator | rebuilt: `is_instance((list, tuple))` gate, then a lax tuple; a `range`/`deque` is a fallback |
| `True` is not an integer; `1.0` and `"1"` are not either  | lax int takes all three                                  | rebuilt: every scalar strict |
| `Literal[3]` is `3` alone                                 | pydantic's literal takes `True` for `1`, `3.0` for `3`, even under `strict` config | rebuilt: each literal behind a strict scalar of its type (`chain`), union of chains for mixed literals |
| a `float` member keeps an integer as an integer           | strict float returns `1.0` for `1`                       | rebuilt: `int \| float` union, int first |
| a `Decimal` or a `Fraction` is not JSON                   | strict float accepts both, as floats                     | rebuilt: `is_instance(float)` gate before the float (found by the differential run) |
| a `StrEnum` member reads as its string; an `IntEnum` as its int; subclasses likewise | strict str/int accept subclasses and return the plain value | same behaviour, verified |
| NaN and infinities are not JSON in metadata, and are the floats they are in attributes | `allow_inf_nan` per schema                               | rebuilt: two JSON unions, finite and user |
| an integer of more digits than `sys.get_int_max_str_digits` writes is refused | any integer is accepted (a 5000-digit one too)           | rebuilt: `ge`/`le` of `±(10**digits - 1)` on every integer schema, read at compile time; the reader recompiles if the interpreter's limit changed. Found by the suite (`test_refine_json`). Costs ~70 ns per integer |
| `-0.0` stays `-0.0`                                       | same                                                     | verified |
| a non-string key is a problem at the parent               | `invalid_key` / `string_type` at the key                 | fallback |
| a `MappingProxyType`, `UserDict`, `ChainMap`, dict subclass is an object | lax typed dict and dict accept any `Mapping`           | same behaviour, verified; strict would refuse them |
| a closed TypedDict's unknown key is `unknown_key` and the field still reads | `extra_forbidden` fails the typed dict                 | fallback |
| the 256-level depth cap, counted from the outermost document's root, a container at level 256 refused | a recursion guard that counts *definition references*, 255 deep, not nesting | rebuilt: every container schema (typed dict, tuple, dict) is behind a definition of its own, so references count levels exactly; a consolidated child read `at` depth *k* gets a validator whose root is wrapped in *k* bare references. Verified at the boundary (255 ok, 256 refused; with three wrappers 252 ok, 253 refused) and by the corpus's "lists N deep" mutations at every path |
| `r16` is filed under `r*`, with `{"bits": 16}` carried by the name; `r12` is refused by the rules; `r*` itself is no name | a tagged union keyed on `name` cannot default or pattern-match | rebuilt: a callable discriminator (measured *faster* than pydantic's key lookup, 0.14 vs 0.24 us) that files names as `filed_under` does; carried names read through `Definition.read_configuration` in Python |
| a name nothing in scope claims is `UnclaimedField`, its configuration any JSON object | no default branch in a tagged union                      | rebuilt: the discriminator returns an "unclaimed" tag; its branch is a well-named string (the spec's two patterns, Rust-regex anchored with `$`, which there is end-of-text) or an envelope of any JSON configuration |
| a codec of dynamic size in a static-only member is a problem | --                                                       | rebuilt: the static union lists static codecs only, with the full table's names in the discriminator, so a dynamic codec's tag is not a choice and pydantic refuses |
| `must_understand: false` at an extension point or nested field is `invalid_value`; absent or `true` is fine | --                                                       | rebuilt: `Literal[True]` behind a strict bool; `false` is a fallback. At a top-level extension *field* it is plain JSON, as the package has it |
| a field requiring a configuration written bare is `missing_key` | --                                                       | rebuilt: no bare-string choice for such a definition; the tag is not a choice, so pydantic refuses |
| nested fields are read in the document's order, and `nested` lists them so; keys stay in the document's order in `to_json`/`to_key_value` | typed-dict output is in the schema's order               | rebuilt: each typed-dict output is compared with the input's key order (`list(src) != list(out)`, ~0.1 us) and rebuilt when it differs; nested fields are walked in the output's order. The second half was found by the differential run (two key-reversal cases on a shard) |
| a nested field is put back into the configuration as a document writes it (`{"name": "crc32c"}` for a bare `"crc32c"`), and `json` keeps it as written | --                                                       | rebuilt: the walk returns (as written, as read) pairs and copies only along paths that changed |
| a model instance or an `AcceptedField` inside a document is not JSON | not a `Mapping`, refused                                 | fallback, verified |
| `Context` is the scope; `_Claimed` and other `Scope`s exist | --                                                       | the fast path engages only for `type(context) is Context` |
| an error in a definition's own function carries a note naming it | --                                                       | kept: rules run through `ruled`/`asked` as before |
| definitions whose configuration holds a field alias inside a union, or a TypedDict that holds itself | --                                                       | not compiled: `Unsupported`, and the scope reads by the Python walk alone (none of the package's definitions do this) |

Things a reviewer should know that are not gaps:

- pydantic-core builds new dicts and tuples for everything it validates
  (checked: no aliasing of the caller's containers), as the Python walk
  does.
- `typing.cast` is a runtime call. The package's hot paths call it about
  60 times per read (`_pipeline`, `chunk_grid_lengths`, `ruled`, ...),
  roughly 3-4 us. The fast path avoids it by letting pydantic's untyped
  output flow as `Any` (with `noqa: ANN401` where the package's rule asks
  for a reason).
- `fill_value_problems` finds the fill value's checker through a
  `functools.cache` keyed by the annotation, and hashing a union
  annotation costs ~3 us per read; the reader caches the checker per
  definition identity instead. The Python walk still pays it.

## The differential run

`design/differential_corpus.py`, run from the package directory once
with `PYTHONPATH` at a `git archive` of c777c9b3's `src` and once as is.
The corpus: the three benchmark documents, every JSON fixture under
`tests/` (node documents as documents, field cases as fields of their
kind), and a v3 array document for each data type's fill-value fixture;
then, for every path into a document (to depth 8 for v3 arrays), the
member deleted, replaced by each of 127 junk values, objects given an
extra key, a non-string key, their keys reversed, `must_understand` of
four kinds, fifteen bad names, and values nested to the depth cap at
that path (two below, at, one above, as lists and as dicts). Each case
records every problem (loc, kind, message, input, ctx), every field with
its canonical spelling, the pipeline, the chunk, and, where there is a
model, its store bytes, the key order of its document, equality and
hash against a second read, pickling, deep copy, claims and
`refines(self)`.

99,959 cases. The final run differs from the base in **0 cases**
(`diff base.tsv head2.tsv` is empty). Over the run, 740,196 array reads
took the fast path and 141,010 fell back to the Python walk (the count
includes each consolidated child of each group case; among the mutated
array documents themselves, most are invalid and fall back, as
intended).

Two earlier runs found the two gaps listed above (`Decimal`/`Fraction`
accepted as floats, 94 cases; nested fields in the plan's order rather
than the document's, 2 cases); both are fixed and covered by
`tests/v3/test_fast.py`, which also reads every fixture and a set of
awkward documents both ways in the suite.

The suite: `just test` passes (2586 tests, 43 of them new), on 3.12 and
3.11; `just lint`, `just typecheck` (pyright strict, no `type: ignore`)
and `just docs-check` pass.

## What remains in Python, and what it costs

Per v3 array read, pydantic is about an eighth of the prototype's time;
the rest is Python:

- field construction (about a quarter): the order check, `trusted_field`
  -- an `AcceptedField` built without the hand-built checks but with the
  same read-only views -- and `frozen`. The views are the public API's
  promise and the largest single Python cost; a lazy view would help
  both readers.
- the rules (about two fifths): the fill value's checker and rules, the
  grid's shape rules and lengths, `read_pipeline` with its `Chunk`s and
  the `ruled`/`asked` wrappers. These are the definitions' own code and
  the reason the package exists; they are not expressible in a schema.
  `read_pipeline` alone is 7 us for a two-codec pipeline: `_order_problems`,
  a `Chunk` per stage, `_chunk_problems`, `_inner_pipelines`, each through
  wrappers that `cast` and build lambdas.
- the model (about a tenth): the reading and members dataclasses, three
  more `frozen` views, `dataclasses.replace`.

So the floor of this design, with the current field, pipeline and model
objects, is about 15 us: the structural walk is no longer where the time
is. Going further means cheaper `AcceptedField`/`Stage`/model
construction (lazy views, fewer dataclass layers, no `cast` on hot
paths), which would benefit the Python walk equally.

## Recommendation

**Keep the fast path as an optional, differential-tested accelerator,
not as the reader.** Concretely:

1. It buys about 3.9x on a v3 array read and 3.6x on a consolidated
   group, with the problems, fields, keys and bytes identical over
   100k mutated documents. For zarr-python integration, where
   pydantic-core is a dependency of pydantic and widely installed, the
   `[fast]` extra more than closes the gap to `ArrayV3Metadata.from_dict`
   (48-78 us here against 18-29 us).
2. It cannot replace the Python walk. Problems are the product, and
   pydantic's error model cannot produce the package's (partial reads
   with `unknown_key`, envelope problems beside a read field, per-field
   loc conventions). The Python walk stays the one reader of problems;
   the fast path is an oracle of acceptance whose correctness is a
   theorem about "pydantic never accepts what Python refuses, and hands
   back the same values" -- which is exactly what the corpus checks and
   what the suite now pins. Any change to `_typed_json`, `_json` or a
   definition's TypedDict must keep both readers in step; the
   differential script is the tool for that, and should run in CI if
   this lands (about five minutes per build).
3. The gaps above are the real cost of the design: each is a
   place where pydantic's notion of a value differs from JSON's, and
   each had to be rediscovered by a probe or a diff. They are all closed
   now, but they are the kind of thing that drifts with pydantic-core
   releases (the literal leniency and the `Decimal` acceptance are
   pydantic's own choices, not bugs). Pinning `pydantic-core>=2.46` and
   running the corpus against each new minor is the maintenance bill.
4. If the aim is to make the *default* read fast with no extra, the
   profile says where the remaining Python time is -- field views,
   model dataclasses, `cast` calls, the fill-value cache key -- and
   those improvements are worth more than they look because they apply
   to both readers and to v2 and groups, which this prototype does not
   touch beyond consolidated array children.

Not done: v2 documents (their reader is already ~20 us), group documents
themselves (only their consolidated array children take the fast path,
through the per-depth validators), and problem rendering from pydantic's
errors, for the reasons above.
