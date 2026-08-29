#!/usr/bin/env python3
"""Extract 768-D pre-projection CLIP frame features from event-frame folders."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "fau_pretraining"))
from models.model_dual_clip import DualCLIPFAUModel  # noqa: E402


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 336)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


class FrameFolder(Dataset):
    def __init__(self, folder):
        self.files = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return TRANSFORM(image.convert("RGB"))


@torch.no_grad()
def encode_preprojection_cls(visual: nn.Module, images: torch.Tensor):
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([class_token, x], dim=1)
    positional = visual.positional_embedding.to(x.dtype)
    x = x + (positional.unsqueeze(0) if positional.ndim == 2 else positional)
    x = visual.ln_pre(x).permute(1, 0, 2)
    x = visual.transformer(x).permute(1, 0, 2)
    return visual.ln_post(x[:, 0, :]).float()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing event-frame folders")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="FAU pretraining checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    folders = sorted({path.parent for path in args.input_root.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}})
    if not folders:
        raise SystemExit(f"No event-frame folders found under {args.input_root}")

    device = torch.device(args.device)
    model = DualCLIPFAUModel(
        clip_model_name="ViT-B/16",
        num_frames=16,
        use_learnable_prompt=False,
        freeze_spatial=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    for folder in tqdm(folders, desc="visual features", unit="video"):
        relative = folder.relative_to(args.input_root)
        output = args.output_root / relative.parent / f"{relative.name}.pt"
        if output.exists() and not args.overwrite:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(
            FrameFolder(folder),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        features = [encode_preprojection_cls(model.clip_model.visual, batch.to(device)).cpu() for batch in loader]
        torch.save(torch.cat(features, dim=0) if features else torch.zeros(1, 768), output)


if __name__ == "__main__":
    main()
