#!/usr/bin/env python3
"""Batch-convert videos to v2e ``events.h5`` files."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a video tree with SensorsINI/v2e.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True, help="Cloned SensorsINI/v2e directory")
    parser.add_argument("--python", default=sys.executable, help="Python used to execute v2e.py")
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--save-error-output", action="store_true")
    return parser.parse_known_args()


def main():
    args, v2e_args = parse_args()
    input_root = args.input_root.absolute()
    output_root = args.output_root.absolute()
    v2e_py = args.v2e_root.absolute() / "v2e.py"

    if not input_root.is_dir():
        raise SystemExit(f"Input directory not found: {input_root}")
    if not v2e_py.is_file():
        raise SystemExit(f"v2e.py not found: {v2e_py}")

    videos = sorted(path for path in input_root.rglob("*") if path.suffix.lower() == ".mp4")
    if not videos:
        raise SystemExit(f"No .mp4 files found under {input_root}")

    logs_dir = output_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    success, skipped, failed, commands = [], [], [], []

    for video in tqdm(videos, desc="v2e", unit="video"):
        output_dir = output_root / video.relative_to(input_root).with_suffix("")
        h5_path = output_dir / "events.h5"
        if h5_path.exists() and not args.overwrite_existing:
            skipped.append(str(video))
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.python,
            str(v2e_py),
            "-i", str(video),
            "-o", str(output_dir),
            "--overwrite",
            "--no_preview",
            "--skip_video_output",
            "--disable_slomo",
            "--output_width", str(args.width),
            "--output_height", str(args.height),
            "--dvs_h5", "events.h5",
            *v2e_args,
        ]
        commands.append(shlex.join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and h5_path.exists():
            success.append(str(video))
        else:
            failed.append(f"{video}\treturn_code={result.returncode}")
            if args.save_error_output:
                safe_name = str(video.relative_to(input_root)).replace("/", "__").replace("\\", "__")
                (logs_dir / f"{safe_name}.stdout.txt").write_text(result.stdout or "", encoding="utf-8")
                (logs_dir / f"{safe_name}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")

    for name, lines in {
        "success.txt": success,
        "skipped.txt": skipped,
        "failed.txt": failed,
        "commands.txt": commands,
    }.items():
        (logs_dir / name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Total={len(videos)} Success={len(success)} Skipped={len(skipped)} Failed={len(failed)}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
