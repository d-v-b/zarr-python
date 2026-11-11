"""
Check that this checkout behaves as a baseline zarr does, wherever code today can look.

    python tools/compat/compat.py --baseline REF [--seeds N] [--work DIR]

Exports `src` at `REF` (a commit, branch or tag), runs the scenarios in
`compat_scenarios.py` under it and under this checkout's `src`, carries pickles each way
between the two, and prints the differences (see `diff.py` for how they are classed).
Run it with an interpreter that has this checkout's dependencies, such as the test
environment's.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def export(ref: str, into: Path) -> Path:
    """The `src` directory of `ref`, with the version file the build would write."""
    shutil.rmtree(into, ignore_errors=True)
    into.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", ref, "src"], check=True, capture_output=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(into)], input=archive, check=True)
    described = subprocess.run(
        ["git", "-C", str(ROOT), "describe", "--tags", "--match", "v[0-9]*", ref],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # zarr refuses to import as version 0.0.0, so name the baseline after its tag.
    tag, _, rest = described.removeprefix("v").partition("-")
    distance, _, commit = rest.partition("-")
    version = f"{tag}.post{distance}+{commit}" if rest else (tag or "999.0.0")
    (into / "src" / "zarr" / "_version.py").write_text(
        f"__version__ = version = {version!r}\n"
        f"__version_tuple__ = version_tuple = (0, 0, 0, {version!r})\n"
        "__commit_id__ = commit_id = None\n"
    )
    return into / "src"


def run(src: Path, out: Path, work: Path, *args: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(HERE / "run.py"),
            str(src),
            str(out),
            "--tmp",
            str(work / "tmp"),
            *args,
        ],
        check=True,
    )


def diff(old: Path, new: Path, out: Path, title: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    subprocess.run(
        [sys.executable, str(HERE / "diff.py"), str(old), str(new), "--out", str(out)], check=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    work = args.work or Path(tempfile.mkdtemp(prefix="zarr-compat-"))
    work.mkdir(parents=True, exist_ok=True)
    sources = {"base": export(args.baseline, work / "base"), "head": ROOT / "src"}

    for name, src in sources.items():
        run(src, work / f"scenarios-{name}.json", work, "--seeds", str(args.seeds))
    diff(
        work / "scenarios-base.json",
        work / "scenarios-head.json",
        work / "diff-scenarios.json",
        "scenarios",
    )

    # Pickles: each version dumps over its own stores, kept pristine, and each version loads
    # every dump over a fresh copy of the stores it was made over.
    for name, src in sources.items():
        stores = work / f"stores-{name}"
        shutil.rmtree(stores, ignore_errors=True)
        run(
            src,
            work / "unused.json",
            work,
            "--dump-pickles",
            str(work / f"pickles-{name}"),
            "--stores",
            str(stores),
        )
        shutil.rmtree(work / f"pristine-{name}", ignore_errors=True)
        shutil.copytree(stores, work / f"pristine-{name}")
    for dumped in sources:
        for loader, src in sources.items():
            stores = work / f"stores-{dumped}"
            shutil.rmtree(stores)
            shutil.copytree(work / f"pristine-{dumped}", stores)
            run(
                src,
                work / f"pickles-{dumped}-in-{loader}.json",
                work,
                "--load-pickles",
                str(work / f"pickles-{dumped}"),
            )
    reference = work / "pickles-base-in-base.json"
    for dumped, loader in (("base", "head"), ("head", "base"), ("head", "head")):
        diff(
            reference,
            work / f"pickles-{dumped}-in-{loader}.json",
            work / f"diff-pickles-{dumped}-in-{loader}.json",
            f"pickles dumped by {dumped}, loaded by {loader} (vs base in base)",
        )
    print(f"\nwork directory: {work}")


if __name__ == "__main__":
    main()
