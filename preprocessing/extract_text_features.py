#!/usr/bin/env python3
"""Extract 1024-D RoBERTa-Large token sequences from transcript files."""

import argparse
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_transcript(raw):
    raw = raw.strip()
    if raw.startswith("ASRTranscription("):
        match = re.search(r"text=(['\"])(.*?)\1, time_stamps=", raw, re.DOTALL)
        return match.group(2) if match else ""
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="FacebookAI/roberta-large")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    files = sorted(args.input_root.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {args.input_root}")

    for source in tqdm(files, desc="text features", unit="file"):
        output = (args.output_root / source.relative_to(args.input_root)).with_suffix(".pt")
        if output.exists() and not args.overwrite:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        text = parse_transcript(source.read_text(encoding="utf-8"))
        if not text:
            feature = torch.zeros(1, 1, 1024)
        else:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                feature = model(**inputs).last_hidden_state.cpu()
        torch.save(feature, output)


if __name__ == "__main__":
    main()
