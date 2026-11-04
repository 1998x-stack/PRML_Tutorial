#!/usr/bin/env python3
"""Inventory PRML lesson pairs without executing or modifying lesson files.

Byte length is only a candidate-stub heuristic, never an academic quality score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def chapter_key(path: Path) -> tuple[int, str]:
    number = path.name.split("_", 1)[0]
    return (int(number), path.name)


def inspect_curriculum(root: Path, min_bytes: int = 250) -> dict:
    """Return a deterministic, read-only chapter inventory."""
    if min_bytes < 0:
        raise ValueError("min_bytes must be >= 0")
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    chapter_dirs = sorted(
        (
            item for item in root.iterdir()
            if item.is_dir()
            and item.name.split("_", 1)[0].isdigit()
        ),
        key=chapter_key,
    )
    chapters: list[dict] = []
    unpaired: list[str] = []
    totals = {"markdown": 0, "python": 0, "pairs": 0, "small_markdown": 0, "small_python": 0}

    for directory in chapter_dirs:
        md = {p.relative_to(directory).with_suffix("").as_posix(): p
              for p in directory.rglob("*.md")}
        py = {p.relative_to(directory).with_suffix("").as_posix(): p
              for p in directory.rglob("*.py")}
        all_keys = set(md) | set(py)
        both = set(md) & set(py)
        small_md = [p.relative_to(root).as_posix()
                    for p in md.values() if p.stat().st_size < min_bytes]
        small_py = [p.relative_to(root).as_posix()
                    for p in py.values() if p.stat().st_size < min_bytes]
        unpaired.extend(f"{directory.name}/{name} ({'missing .py' if name in md else 'missing .md'})"
                        for name in sorted(all_keys - both))
        entry = {
            "chapter": directory.name,
            "markdown": len(md),
            "python": len(py),
            "pairs": len(both),
            "small_markdown": len(small_md),
            "small_python": len(small_py),
            "candidate_stub_paths": sorted(set(small_md) | set(small_py)),
        }
        chapters.append(entry)
        for field in totals:
            totals[field] += entry[field]

    return {
        "root": str(root),
        "chapter_count": len(chapters),
        "totals": totals,
        "chapters": chapters,
        "unpaired": sorted(unpaired),
        "note": "Small files are stub candidates, not proof of inadequate quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--min-bytes", type=int, default=250)
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON")
    parser.add_argument("--strict", action="store_true", help="Fail on candidate stubs as well as missing pairs")
    args = parser.parse_args()
    try:
        result = inspect_curriculum(args.root.resolve(), args.min_bytes)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Chapter directories: {result['chapter_count']}")
        for c in result["chapters"]:
            print(f"{c['chapter']}: {c['pairs']} pairs, "
                  f"{c['small_markdown']} small Markdown, {c['small_python']} small Python")
        print("Totals:", result["totals"])
        if result["unpaired"]:
            print("Unpaired:", *result["unpaired"], sep="\n  - ")
    if result["unpaired"]:
        return 1
    if args.strict and (result["totals"]["small_markdown"] or result["totals"]["small_python"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
