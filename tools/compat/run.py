"""
Run the compatibility scenarios under one zarr source tree.

    PYTHONPATH=SRC python run.py SRC OUT.json --tmp DIR [--seeds N] [--dump-pickles DIR --stores DIR]
    PYTHONPATH=SRC python run.py SRC OUT.json --tmp DIR --load-pickles DIR

`SRC` is the directory that holds the `zarr` package. Put it on `PYTHONPATH`, which
comes before an installed zarr; the run stops unless zarr was imported from `SRC`.
`--tmp` is wiped first; give both runs the same path, so the paths they record agree.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import compat_scenarios as scenarios

import zarr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tmp", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--dump-pickles", type=Path)
    parser.add_argument("--stores", type=Path)
    parser.add_argument("--load-pickles", type=Path)
    args = parser.parse_args()

    if not Path(zarr.__file__).resolve().is_relative_to(args.src.resolve()):
        raise SystemExit(f"imported {zarr.__file__}, not zarr from {args.src}")

    shutil.rmtree(args.tmp, ignore_errors=True)
    args.tmp.mkdir(parents=True)
    rec = scenarios.Recorder(args.tmp)
    if args.dump_pickles is not None:
        scenarios.dump_pickles(args.dump_pickles, args.stores)
    elif args.load_pickles is not None:
        scenarios.load_pickles(rec, args.load_pickles)
    else:
        scenarios.run_all(rec, args.tmp, range(args.seeds))
    args.out.write_text(json.dumps({"zarr": zarr.__version__, "events": rec.events}, indent=1))


if __name__ == "__main__":
    main()
