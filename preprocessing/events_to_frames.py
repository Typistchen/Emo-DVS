#!/usr/bin/env python3
"""Render v2e HDF5 event streams as red/blue event frames."""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--window-seconds", type=float, default=0.3)
    parser.add_argument("--timestamp-unit", choices=("auto", "seconds", "microseconds"), default="auto")
    return parser.parse_args()


def load_events(path):
    with h5py.File(path, "r") as handle:
        if "events" not in handle:
            raise KeyError(f"'events' dataset not found in {path}")
        events = handle["events"][:]
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"Expected events shaped [N,4], got {events.shape}")
    return events[:, 0], events[:, 1].astype(np.int64), events[:, 2].astype(np.int64), events[:, 3]


def render(path, output_dir, args):
    timestamps, xs, ys, polarities = load_events(path)
    if not len(timestamps):
        raise ValueError("Empty event stream")
    if np.any(np.diff(timestamps) < 0):
        order = np.argsort(timestamps)
        timestamps, xs, ys, polarities = timestamps[order], xs[order], ys[order], polarities[order]

    unit = args.timestamp_unit
    if unit == "auto":
        unit = "seconds" if np.issubdtype(timestamps.dtype, np.floating) else "microseconds"
    window = args.window_seconds if unit == "seconds" else int(args.window_seconds * 1_000_000)
    output_dir.mkdir(parents=True, exist_ok=True)

    start, end = timestamps[0], timestamps[-1]
    frame_index = 0
    current = start
    while current <= end:
        following = current + window
        left = np.searchsorted(timestamps, current, side="left")
        right = np.searchsorted(timestamps, following, side="left")
        frame = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
        x, y, p = xs[left:right], ys[left:right], polarities[left:right]
        valid = (x >= 0) & (x < args.width) & (y >= 0) & (y < args.height)
        x, y, p = x[valid], y[valid], p[valid]
        frame[y[p <= 0], x[p <= 0]] = (255, 0, 0)
        frame[y[p > 0], x[p > 0]] = (0, 0, 255)
        cv2.imwrite(str(output_dir / f"frame_{frame_index:06d}.png"), frame)
        frame_index += 1
        current = following
    return frame_index


def main():
    args = parse_args()
    files = sorted(args.input_root.rglob("events.h5"))
    if not files:
        raise SystemExit(f"No events.h5 files found under {args.input_root}")
    failures = []
    for path in tqdm(files, desc="event frames", unit="file"):
        try:
            render(path, args.output_root / path.parent.relative_to(args.input_root), args)
        except Exception as exc:
            failures.append(f"{path}\t{exc!r}")
    if failures:
        logs = args.output_root / "_logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "failed.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"Failed to process {len(failures)} file(s); see {logs / 'failed.txt'}")


if __name__ == "__main__":
    main()
