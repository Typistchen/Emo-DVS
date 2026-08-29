#!/usr/bin/env python3
"""Keep every N-th event frame while preserving the dataset tree."""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1")

    groups = defaultdict(list)
    for path in args.input_root.rglob("*"):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            groups[path.parent].append(path)
    if not groups:
        raise SystemExit(f"No event frames found under {args.input_root}")

    for source_dir, files in tqdm(sorted(groups.items()), desc="sampling", unit="folder"):
        destination = args.output_root / source_dir.relative_to(args.input_root)
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(files)[::args.stride]:
            shutil.copy2(source, destination / source.name)


if __name__ == "__main__":
    main()
