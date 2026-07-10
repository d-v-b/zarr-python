# Contributing

Zarr is a community maintained project. We welcome contributions in the form of bug reports, bug fixes, documentation, enhancement proposals and more. This page provides information on how best to contribute.

These guidelines apply to everyone: new contributors, core developers, and agentic tools acting on behalf of either. If you are an agent reading this page, the guidelines describe obligations you must satisfy on your operator's behalf, and one you cannot satisfy for them (see [Own your changes](#own-your-changes)).

## Asking for help

Please post questions about Zarr usage to our [Zulip chat](https://ossci.zulipchat.com/#narrow/channel/423692-Zarr-Python/). If you don't get a response within a day or two, feel free to raise a [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/new) including a link to your question.

## Bug reports

Bugs should be reported in [GitHub issues](https://github.com/zarr-developers/zarr-python/issues/new). Please include the following items in a bug report:

1. A minimal, self-contained snippet of Python code reproducing the problem. You can format the code nicely using markdown, e.g.:

```python exec="false" reason="illustrative pseudocode with a '# etc.' placeholder, not runnable"
import zarr
g = zarr.group()
# etc.
```

2. An explanation of why the current behavior is wrong/not desired, and what you expect instead.

3. Information about the version of Zarr, along with versions of dependencies and the Python interpreter, and installation information. The version of Zarr can be obtained from the `zarr.__version__` attribute. Indicate how Zarr was installed, e.g., "installed via pip into a virtual environment", or "installed using conda". Information about other packages installed can be obtained by executing `pip freeze` (if using pip to install packages) or `conda env export` (if using conda to install packages) from the operating system command prompt.

## Enhancement proposals

If you have an idea about a new feature or some other improvement to Zarr, please start a conversation in our [Zulip chat](https://ossci.zulipchat.com/#narrow/channel/423692-Zarr-Python/) or open a [GitHub issue](https://github.com/zarr-developers/zarr-python/issues/new).

## Contribution guidelines

Two distinct things can be true of any contribution: someone is *accountable* for it, and someone *composed* it. Accountability can never be delegated to a tool. Composition can, as long as you say so. Every guideline below follows from that distinction.

### Own your changes

Every contribution — code, PR description, issue comment, review response — has exactly one human author, meaning the person who is answerable for it. If a maintainer asks why a line is correct, that person answers, from their own understanding, without going back to the tool that produced it.

Tools do not change this. If you used one to generate a change, read the change critically and test it before requesting review. If you cannot explain why each part of it is correct and how it fits into the project, it is not ready.

Agents cannot attest on a contributor's behalf. Our [pull request template](https://github.com/zarr-developers/zarr-python/blob/main/.github/PULL_REQUEST_TEMPLATE.md) asks the author to confirm that they have reviewed and understood the changes. An agent must leave that box for its operator to check, and must tell its operator that the expectation exists.

### Attribute your sources

Unlabeled text is a claim that you composed it. Where a block of text was composed by a tool, mark where the block begins and name the tool. A line such as the following, immediately before the machine-composed text, is enough:

```markdown
:robot: _AI-generated text below, from <tool name>_ :robot:
```

Labeling makes machine-composed text welcome rather than suspect. Once you have read a tool's summary, agreed with it, and can defend it, posting it under a label costs a reviewer nothing and often communicates more thoroughly than prose written in a hurry.

Two limits on this:

**A pull request opens with a sentence you wrote yourself.** Say why the change exists and why it deserves a reviewer's attention. Machine-composed detail can follow, labeled. This one sentence is the part that cannot be produced without having thought about the change, and it is what tells a maintainer whether to spend their afternoon on your diff.

Automated dependency updates are exempt from this, and the reason is worth stating, because it explains the rule. A bot's identity fully supplies its motivation: "a dependency released a new version" is the whole story, and the bot's name tells you so. An agent does not supply the motivation for the work you asked it to do. That judgment is yours, a reader cannot infer it from the fact that an agent ran, and so you have to write it down.

**Review responses are owed to a person.** When a reviewer asks you a question, they are asking *you*. Answer in your own words. Do not paste an unread machine reply into a review thread.

### Keep PRs reviewable

Generating a change is fast; reviewing it is not. A large diff shifts the burden from you to the reviewer, and a reviewer's time is the scarcest resource this project has. Pull requests that cannot be reviewed in reasonable time with reasonable effort may be closed, however useful or correct they might be.

Tools make it cheap to produce a broad, flat set of unrelated fixes. Such a pull request is hard to review and, because its title cannot describe its contents, it is invisible to the people who would have wanted to weigh in on any one part of it. Prefer to split by topic, even when each individual change is safe. Use tools not only to write code but to prepare a better pull request: coherent commits, a clear description, and minimal scope.

If you plan a large contribution — a significant refactor, a new subsystem — discuss the scope and approach with maintainers first, on our [Zulip chat](https://ossci.zulipchat.com/#narrow/channel/423692-Zarr-Python/) or in an issue. Maintainers may ask that a large change be broken into smaller pieces.

### Write commit messages for the reader

A pull request is merged by squashing, so its title becomes a permanent line in the history of `main`. Write it as [a Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) — `fix: handle 0-d arrays in save_array` — describing what changed, not which files you touched. Someone reading [the commit list](https://github.com/zarr-developers/zarr-python/commits/main/) should come away with an accurate picture of how the project is developing.

### Documentation is held to the same standard

Zarr has domain-specific semantics — chunked storage, codec pipelines, the differences between Zarr formats 2 and 3 — that language models frequently get wrong, confidently and plausibly. Do not submit documentation you have not read and verified against the behavior of the code.

## How to contribute

### Forking the repository

The Zarr source code is hosted on GitHub at the following location:

* [https://github.com/zarr-developers/zarr-python](https://github.com/zarr-developers/zarr-python)

You will need your own fork to work on the code. Go to the link above and hit the ["Fork"](https://github.com/zarr-developers/zarr-python/fork) button. Then clone your fork to your local machine:

```bash
git clone git@github.com:your-user-name/zarr-python.git
cd zarr-python
git remote add upstream git@github.com:zarr-developers/zarr-python.git
```

### Creating a development environment

To work with the Zarr source code, it is recommended to use [hatch](https://hatch.pypa.io/latest/index.html) to create and manage development environments. Hatch automatically installs the correct Zarr dependencies. Assuming you have a Python 3 interpreter already installed, and you have cloned the Zarr source code and your current working directory is the root of the repository, you can do something like the following:

```bash
pip install hatch
hatch env show  # list all available environments
```

To verify that your development environment is working, you can run the unit tests for one of the test environments, e.g.:

```bash
hatch env run --env test.py3.12-optional run
```

### Creating a branch

Before you do any new work or submit a pull request, please open an issue on GitHub to report the bug or propose the feature you'd like to add.

It's best to synchronize your fork with the upstream repository, then create a new, separate branch for each piece of work you want to do. E.g.:

```bash
git checkout main
git fetch upstream
git checkout -b shiny-new-feature upstream/main
git push -u origin shiny-new-feature
```

This changes your working directory to the 'shiny-new-feature' branch. Keep any changes in this branch specific to one bug or feature so it is clear what the branch brings to Zarr.

To update this branch with latest code from Zarr, you can retrieve the changes from the main branch and perform a rebase:

```bash
git fetch upstream
git rebase upstream/main
```

This will replay your commits on top of the latest Zarr git main. If this leads to merge conflicts, these need to be resolved before submitting a pull request. Alternatively, you can merge the changes in from upstream/main instead of rebasing, which can be simpler:

```bash
git pull upstream main
```

Again, any conflicts need to be resolved before submitting a pull request.

### Running the test suite

Zarr's tests are in the `tests` directory. We use [`pytest`](https://docs.pytest.org/en/stable/). To run the test suite, initialize your development environment (see [creating a development environment](#creating-a-development-environment) above) and invoke:

```bash
hatch env run --env test.py3.12-optional run
```

All tests are automatically run via GitHub Actions for every pull request. Automated checks must pass before changes are accepted. Test coverage is also collected automatically via the Codecov service.

### Code standards - using prek

`Zarr` uses a set of git hooks managed by [`prek`](https://github.com/j178/prek), a fast, Rust-based pre-commit hook manager that is fully compatible with `.pre-commit-config.yaml` files. `prek` can be installed locally by running:

```bash
uv tool install prek
```

or:

```bash
pip install prek
```

The hooks can be installed locally by running:

```bash
prek install
```

This will run the checks every time a commit is created locally. The checks will by default only run on the files modified by a commit, but the checks can be triggered for all the files by running:

```bash
prek run --all-files
```

You can also run hooks only for files in a specific directory:

```bash
prek run --directory src/zarr
```

Or run hooks for files changed in the last commit:

```bash
prek run --last-commit
```

To list all available hooks:

```bash
prek list
```

To skip failing checks and push the code for further discussion, use the `--no-verify` option with `git commit`.

### Test coverage

Zarr strives to maintain high test coverage. Running

```bash
hatch env run --env test.py3.12-optional run-coverage
```

will automatically run the test suite with coverage and produce an XML coverage report.

You can also generate an HTML coverage report by running:

```bash
hatch env run --env test.py3.12-optional run-coverage-html
```

When submitting a pull request, coverage will also be collected across all supported Python versions via the Codecov service, and will be reported back within the pull request.

### Documentation

Docstrings for user-facing classes and functions should follow the [numpydoc](https://numpydoc.readthedocs.io/en/stable/format.html#docstring-standard) standard, including sections for Parameters and Examples. All examples should run and pass as doctests under Python 3.12.

Zarr uses mkdocs for documentation, hosted on readthedocs.org. Documentation is written in Markdown (.md files) in the `docs` folder. The documentation consists both of prose and API documentation. All user-facing classes and functions are included in the API documentation, under the `docs/api` folder using the [mkdocstrings](https://mkdocstrings.github.io/) extension. Add new public functions or classes to the relevant markdown file in `docs/api/*.md`. New features or important usage information should be included in the user-guide (`docs/user-guide`). Any changes should also be included as a new file in the `changes` directory.

The documentation can be built locally by running:

```bash
hatch --env docs run build
```

The resulting built documentation will be available in the `site` folder.

Hatch can also be used to serve continuously updating version of the documentation during development at [http://127.0.0.1:8000/](http://127.0.0.1:8000/). This can be done by running:

```bash
hatch --env docs run serve
```

#### Adding executable code blocks in the documentation

Zarr uses [Markdown Exec](https://pawamoy.github.io/markdown-exec/usage/) to execute code blocks in Markdown files. Add `exec="true"` to a code block header for it to be executed when the docs are built. For example:

````md
```python exec="true"
print("Hello world")
```
````

Below are other useful options that can be added to the code block. See [Markdown Exec's documentation](https://pawamoy.github.io/markdown-exec/usage/#options-summary) for a full list:

  - `source="above"` makes sure the code within the code block is also rendered in the documentation (rather than just the output).
  - `session="<name-of-docs-page>"` executes code blocks in a named session reusing previously defined variables.
  - `result="ansi"` or `result="html"` to render the output. If the code does not produce output, you should leave off the `result` option to prevent an empty cell from rendering in the docs.

For example:

````md
```python exec="true" session="contributing" source="above" result="ansi"
print("Hello world")
```
````

renders as:

```python exec="true" session="contributing" source="above" result="ansi"
print("Hello world")
```

#### Validating code blocks: `exec` vs `test`

Python code blocks in the documentation are checked by tests
(`tests/test_docs.py`). A block declares *how* it is
validated using one of two independent attributes:

  - **`exec="true"`** — Markdown Exec runs the block **at docs-build time to render its
    output** into the page. This is the attribute described above; it is also what the
    test suite executes. Use it for ordinary examples whose output should appear in the
    docs.
  - **`test="true"`** — the block is **run by the test suite only**, *not* at build time.
    Use this for an example that should be validated but cannot run in the docs-build
    environment — for example one that needs a GPU or a cloud backend. Markdown Exec
    leaves a `test="true"` block as a static, syntax-highlighted snippet (it never
    executes it), while the test suite still runs it (see the marker note below).

A block may carry both (`exec="true" test="true"`), though in practice `exec="true"`
already implies it is tested, so you rarely need `test="true"` alongside it.

The two attributes are kept separate on purpose: `exec=` controls *build-time rendering*
and `test=` controls *test-time validation*. Tagging a GPU/cloud example `exec="true"`
would make `mkdocs build` try to run it on a machine without that infrastructure and fail
the build; `test="true"` lets it be validated without being built.

##### Opting a block out of validation

A handful of blocks genuinely cannot run and are not executable Python — a REPL
transcript, a deliberately-incorrect "before" snippet, a `--8<--` file include. Mark
these explicitly by opening the fence with
`exec="false" reason="REPL output transcript, not executable source"` (supply a reason
that fits the block).

`exec="false"` with a non-empty `reason` is an explicit, greppable opt-out. A test
(`test_no_unvalidated_blocks`) requires **every** Python block to be either `exec="true"`,
`test="true"`, or `exec="false"` with a reason — so a block can never silently skip
validation. A bare ` ```python ` fence, or a typo like `exec="on"`, fails that test.

Markdown Exec only renders `exec="true"` fences; the `mkdocs_hooks.py` hook at the
repository root makes `test="true"` and `exec="false"` fences render as ordinary
highlighted code blocks. Without it, these fences would fail superfences parsing and
their contents would spill into the page as raw markdown.

##### Marker-bound blocks (GPU, S3)

A `test="true"` block that needs special infrastructure declares a pytest marker with
`markers="..."`, which binds it to that infrastructure in the test suite:

  - `markers="gpu"` — run only under `pytest -m gpu` (the GPU CI environment); skipped
    elsewhere via `importorskip("cupy")`.
  - `markers="s3"` — run against a mock S3 (moto) backend supplied by a test fixture, so
    the example can use a bare `s3://…` URL with no test-only connection details on show.

##### Placement of `test="true"` blocks

Because Markdown Exec does not execute a `test="true"` (or `exec="false"`) block, placing
one *before* an `exec="true"` block on the same page can disrupt the build-time execution
of that later block. Put `test="true"` blocks **after** all `exec="true"` blocks on the
page (or on a page where they are the only Python block). The `test_test_only_blocks_come_last`
test enforces this, and the CI docs build runs with `--strict` so any such breakage fails
the build rather than passing as a warning.

#### Building documentation without executing code blocks

Sometimes, you may want the documentation to build quicker. You can disable code block execution by commenting out the [markdown-exec plugin](https://github.com/zarr-developers/zarr-python/blob/884a8c91afcc3efe28b3da952be3b85125c453cb/mkdocs.yml#L132) in the mkdocs configuration file. This will make code blocks and cross references render incorrectly (i.e., expect build warnings), but also reduces build time by ~3x. Be sure to undo the commenting out before opening your pull request.

### Changelog

Zarr Python uses [towncrier](https://towncrier.readthedocs.io/en/stable/tutorial.html) to manage release notes. Most pull requests should include at least one news fragment describing the changes. To add a release note, you'll need the GitHub issue or pull request number and the type of your change (`feature`, `bugfix`, `doc`, `removal`, `misc`). With that, run `towncrier create` with your development environment, which will prompt you for the issue number, change type, and the news text:

```bash
towncrier create
```

Alternatively, you can manually create the files in the `changes` directory using the naming convention `{issue-number}.{change-type}.md`.

See the [towncrier](https://towncrier.readthedocs.io/en/stable/tutorial.html) docs for more.

## Project governance

This section documents the processes that core developers follow to maintain the project. The current core developers are listed in [`TEAM.md`](https://github.com/zarr-developers/zarr-python/blob/main/TEAM.md).

### Merging pull requests

Pull requests submitted by an external contributor should be reviewed and approved by at least one core developer before being merged. Ideally, pull requests submitted by a core developer should be reviewed and approved by at least one other core developer before being merged.

Pull requests should not be merged until all CI checks have passed (GitHub Actions, Codecov) against code that has had the latest main merged in.

Before merging, the milestone must be set to decide whether a PR will be in the next patch, minor, or major release. The next section explains which types of changes go in each release.

### Self-merging pull requests

The default is that a pull request opened by a core developer is reviewed and approved by at least one other core developer before it is merged. We trust core developers to use their judgment, though, and we would rather bias toward action than make routine changes wait on review they do not really need.

So a core developer may merge their own pull request whenever they judge the change to be low-risk, provided the standard merge requirements are met — CI is green against code that has had the latest `main` merged in, a changelog fragment has been added, and the milestone is set — and other core developers have had a fair chance to weigh in. As a rule of thumb, leave the pull request open for a few days before self-merging, unless it is genuinely trivial or time-sensitive. If you are confident a change is fine, merge it; if you have real doubts, ask for a review. It is generally advisable to ping another developer in the PR description for awareness about the direction, even if you choose not to request a formal review.

Some changes warrant more caution, and a second reviewer is usually worth seeking even when you could self-merge: changes to the public API, anything touching data-format or on-disk compatibility, and performance-sensitive code. These are the most expensive to get wrong and the hardest to reverse. Reverts, by contrast, are cheap — if a self-merged change turns out to be a mistake, reverting it is itself a low-risk change that any core developer can make, and the reworked version can go through normal review. When something recently merged is actively causing harm — a broken `main`, a release blocker, or data corruption — fix it fast and request review after the fact rather than waiting.

This policy exists to lower the cost of routine work and to help newer core developers grow comfortable merging changes. It is not a license to merge past an unresolved objection: if another core developer asks to review a change, give them that chance.

### Release procedure

Open an issue on GitHub announcing the release using the release checklist template:
[https://github.com/zarr-developers/zarr-python/issues/new?template=release-checklist.md](https://github.com/zarr-developers/zarr-python/issues/new?template=release-checklist.md). The release checklist includes all steps necessary for the release.

#### Preparing a release

Releases are prepared using the ["Prepare release notes"](https://github.com/zarr-developers/zarr-python/actions/workflows/prepare_release.yml) workflow. To run it:

1. Go to the [workflow page](https://github.com/zarr-developers/zarr-python/actions/workflows/prepare_release.yml) and click "Run workflow".
2. Enter the release version (e.g. `3.2.0`) and the target branch (defaults to `main`).
3. The workflow will run `towncrier build` to render the changelog, remove consumed fragments from `changes/`, and open a pull request on the `release/v<version>` branch.
4. The release PR is automatically labeled `run-downstream`, which triggers the [downstream test workflow](https://github.com/zarr-developers/zarr-python/actions/workflows/downstream.yml) to run Xarray and numcodecs integration tests against the release branch.
5. Review the rendered changelog in `docs/release-notes.md` and verify downstream tests pass before merging.

## Compatibility and versioning policies

### Versioning

Versions of this library are identified by a triplet of integers with the form `<major>.<minor>.<patch>`, for example `3.0.4`. A release of `zarr-python` is associated with a new version identifier. That new identifier is generated by incrementing exactly one of the components of the previous version identifier by 1. When incrementing the `major` component of the version identifier, the `minor` and `patch` components are reset to 0. When incrementing the minor component, the patch component is reset to 0.

Releases are classified by the library changes contained in that release. This classification determines which component of the version identifier is incremented on release.

* **major** releases (for example, `2.18.0` -> `3.0.0`) are for changes that will require extensive adaptation efforts from many users and downstream projects. For example, breaking changes to widely-used user-facing APIs should only be applied in a major release.

  Users and downstream projects should carefully consider the impact of a major release before adopting it. In advance of a major release, developers should communicate the scope of the upcoming changes, and help users prepare for them.

* **minor** releases (for example, `3.0.0` -> `3.1.0`) are for changes that do not require significant effort from most users or downstream projects to respond to. API changes are possible in minor releases if the burden on users imposed by those changes is sufficiently small.

  For example, a recently released API may need fixes or refinements that are breaking, but low impact due to the recency of the feature. Such API changes are permitted in a minor release.

  Minor releases are safe for most users and downstream projects to adopt.

* **patch** releases (for example, `3.1.0` -> `3.1.1`) are for changes that contain no breaking or behavior changes for downstream projects or users. Examples of changes suitable for a patch release are bugfixes and documentation improvements.

  Users should always feel safe upgrading to the latest patch release.

Note that this versioning scheme is not consistent with [Semantic Versioning](https://semver.org/). Contrary to SemVer, the Zarr library may release breaking changes in `minor` releases, or even `patch` releases under exceptional circumstances. But we should strive to avoid doing so.

A better model for our versioning scheme is [Intended Effort Versioning](https://jacobtomlinson.dev/effver/), or "EffVer". The guiding principle of EffVer is to categorize releases based on the *expected effort required to upgrade to that release*.

Zarr developers should make changes as smooth as possible for users. This means making backwards-compatible changes wherever possible. When a backwards-incompatible change is necessary, users should be notified well in advance, e.g. via informative deprecation warnings.

### Data format compatibility

The Zarr library is an implementation of a file format standard defined externally -- see the [Zarr specifications website](https://zarr-specs.readthedocs.io) for the list of Zarr file format specifications.

If an existing Zarr format version changes, or a new version of the Zarr format is released, then the Zarr library will generally require changes. It is very likely that a new Zarr format will require extensive breaking changes to the Zarr library, and so support for a new Zarr format in the Zarr library will almost certainly come in a new `major` release. When the Zarr library adds support for a new Zarr format, there may be a period of accelerated changes as developers refine newly added APIs and deprecate old APIs. In such a transitional phase breaking changes may be more frequent than usual.


## Experimental API policy

The `zarr.experimental` namespace contains features that are under active development and may change without notice. When contributing to or depending on experimental features, please keep the following in mind:

### For contributors

When adding a new feature to `zarr.experimental`:

1. Place the feature under `src/zarr/experimental/` and export it from `src/zarr/experimental/__init__.py`.
2. Document the feature in `docs/user-guide/experimental.md` and note clearly that it is experimental.
3. Add a changelog entry categorized as `feature`.

We aim to either **promote** or **remove** experimental features within **6 months** of their addition. To promote a feature to stable:

1. Move it from `zarr.experimental` to the appropriate stable module.
2. Keep a deprecated re-export in `zarr.experimental` for one minor release.
3. Update the documentation to reflect the stable location.

### For users

Features in `zarr.experimental` carry no stability guarantees. They may be changed or removed in any release, including patch releases. If you depend on an experimental feature, pin your `zarr-python` version accordingly.

## Benchmarks

Zarr uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) for running
performance benchmarks as part of our test suite. The benchmarks are found in `tests/benchmarks`.
By default pytest is configured to run these benchmarks as plain tests (i.e., no benchmarking). To run
a benchmark with timing measurements, use the `--benchmark-enable` when invoking `pytest`.

The benchmarks are run as part of the continuous integration suite through [codspeed](https://codspeed.io/zarr-developers/zarr-python).
