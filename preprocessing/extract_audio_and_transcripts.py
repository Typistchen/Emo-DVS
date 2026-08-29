#!/usr/bin/env python3
"""Extract 16-kHz mono WAV files and optional Qwen3-ASR transcripts."""

import argparse
import subprocess
from pathlib import Path

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--text-root", type=Path)
    parser.add_argument("--asr-model", help="Qwen3-ASR model ID/path; omit to extract audio only")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if bool(args.text_root) != bool(args.asr_model):
        parser.error("--text-root and --asr-model must be provided together")

    videos = sorted(args.input_root.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found under {args.input_root}")
    pending = []
    for video in tqdm(videos, desc="audio extraction", unit="video"):
        relative = video.relative_to(args.input_root).with_suffix(".wav")
        wav = args.audio_root / relative
        wav.parent.mkdir(parents=True, exist_ok=True)
        if not wav.exists():
            command = ["ffmpeg", "-i", str(video), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav), "-y"]
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                continue
        if args.text_root:
            transcript = (args.text_root / relative).with_suffix(".txt")
            transcript.parent.mkdir(parents=True, exist_ok=True)
            if not transcript.exists():
                pending.append((wav, transcript))

    if not args.asr_model or not pending:
        return
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(args.asr_model, dtype=torch.float16, device_map=args.device)
    for start in tqdm(range(0, len(pending), args.batch_size), desc="ASR", unit="batch"):
        batch = pending[start:start + args.batch_size]
        results = model.transcribe(audio=[str(item[0]) for item in batch])
        if not isinstance(results, list):
            results = [results]
        for (_, output), result in zip(batch, results):
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            output.write_text(text.strip(), encoding="utf-8")


if __name__ == "__main__":
    main()
