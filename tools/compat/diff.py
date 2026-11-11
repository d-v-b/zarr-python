"""
Compare two runs of the compatibility scenarios.

    python diff.py OLD.json NEW.json [--out DETAILS.json] [--examples N]

Each difference is classed:

- `value`: a step returned something else, or raised where it returned (or the reverse,
  unless it is an `added` one);
- `added`: a step raised `AttributeError` or `TypeError` in the old run and returned in the
  new one, as a call of something the old version lacks does;
- `message`: a step raised the same exception with another message;
- `warnings`: a step behaved the same but warned differently;
- `internal`: a step that looks at private state (its label holds `internal:`);
- `steps`: a scenario ran other steps (the first one that differs is reported).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_KIND = re.compile(r"\d+|\(.*|\[.*|\{.*")


def classify(old: dict[str, Any], new: dict[str, Any]) -> str | None:
    if "internal:" in old["label"]:
        same = {k: v for k, v in old.items() if k != "warnings"} == {
            k: v for k, v in new.items() if k != "warnings"
        }
        return None if same else "internal"
    if "raises" in old and "raises" in new:
        if old["raises"] != new["raises"]:
            return "value"
        if old["message"] != new["message"]:
            return "message"
    elif "raises" in old:
        added = old["raises"].rsplit(".", 1)[-1] in ("AttributeError", "TypeError")
        return "added" if added else "value"
    elif "raises" in new or old["value"] != new["value"]:
        return "value"
    if old.get("warnings") != new.get("warnings"):
        return "warnings"
    return None


def kind_of(label: str) -> str:
    return ":".join(p for p in _KIND.sub("", label).split(":") if p and p != "h")


def compare(
    old_events: list[dict[str, Any]], new_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[list[dict[str, Any]]]] = defaultdict(lambda: [[], []])
    for side, events in enumerate((old_events, new_events)):
        for event in events:
            by_scenario[event["scenario"]][side].append(event)
    found = []
    for scenario, (old, new) in by_scenario.items():
        for i in range(max(len(old), len(new))):
            if i >= len(old) or i >= len(new) or old[i]["label"] != new[i]["label"]:
                found.append(
                    {
                        "class": "steps",
                        "scenario": scenario,
                        "kind": "steps",
                        "old": old[i] if i < len(old) else None,
                        "new": new[i] if i < len(new) else None,
                    }
                )
                break
            cls = classify(old[i], new[i])
            if cls is not None:
                found.append(
                    {
                        "class": cls,
                        "scenario": scenario,
                        "kind": f"{scenario.split(':', 1)[0]}:{kind_of(old[i]['label'])}",
                        "old": old[i],
                        "new": new[i],
                    }
                )
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--examples", type=int, default=1)
    args = parser.parse_args()
    old = json.loads(args.old.read_text())
    new = json.loads(args.new.read_text())
    found = compare(old["events"], new["events"])
    print(f"{old['zarr']} -> {new['zarr']}: {len(old['events'])} steps, {len(found)} differ")
    for cls, n in Counter(f["class"] for f in found).most_common():
        print(f"  {cls}: {n}")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for f in found:
        groups[f["class"], f["kind"]].append(f)
    for (cls, kind), items in sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        print(f"\n[{cls}] {kind} x{len(items)} in {len({f['scenario'] for f in items})} scenarios")
        for f in items[: args.examples]:
            print(f"  {f['scenario']}")
            for side in ("old", "new"):
                event = f[side]
                text = (
                    "-"
                    if event is None
                    else json.dumps({k: v for k, v in event.items() if k != "scenario"})
                )
                print(f"    {side}: {text[:600]}")
    if args.out is not None:
        args.out.write_text(json.dumps(found, indent=1))


if __name__ == "__main__":
    main()
