#!/usr/bin/env python3
"""Extract 1024-D HuBERT-Large sequences from a tree of WAV files."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-seconds", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    files = sorted(args.input_root.rglob("*.wav"))
    if not files:
        raise SystemExit(f"No .wav files found under {args.input_root}")

    for source in tqdm(files, desc="audio features", unit="file"):
        output = (args.output_root / source.relative_to(args.input_root)).with_suffix(".pt")
        if output.exists() and not args.overwrite:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            waveform, sample_rate = torchaudio.load(source)
            waveform = waveform.mean(dim=0)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            chunks = []
            size = args.chunk_seconds * 16000
            with torch.no_grad():
                for start in range(0, waveform.numel(), size):
                    chunk = waveform[start:start + size]
                    if chunk.numel() < 400:
                        continue
                    inputs = processor(chunk.numpy(), sampling_rate=16000, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    chunks.append(model(**inputs).last_hidden_state.cpu())
            if chunks:
                feature = torch.cat(chunks, dim=1).transpose(1, 2)
                feature = F.avg_pool1d(feature, kernel_size=5, stride=5).transpose(1, 2)
            else:
                feature = torch.zeros(1, 1, 1024)
            torch.save(feature, output)
        except Exception as exc:
            print(f"Failed {source}: {exc}")
            torch.save(torch.zeros(1, 1, 1024), output)


if __name__ == "__main__":
    main()
