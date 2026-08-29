#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


# =========================
# 环境配置
# =========================
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
project_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, project_root)

from data.dataset_videoemotion8 import (
    MultimodalEmotionDataset,
    multimodal_collate_fn,
    IMAGE_DIR,
    AUDIO_DIR,
    TEXT_DIR,
    EMOTION_TO_IDX
)


# =========================================================
# 1. 工具函数
# =========================================================

def print_text_presence_stats(subset, name="Dataset"):
    total = len(subset)
    has_text_count = 0
    no_text_count = 0

    for real_idx in subset.indices:
        sample = subset.dataset[real_idx]
        item = sample if isinstance(sample, dict) else subset.dataset[real_idx]
        has_text = item["has_text"]
        if torch.is_tensor(has_text):
            has_text = float(has_text.item())

        if has_text > 0.5:
            has_text_count += 1
        else:
            no_text_count += 1

    print(f"\n📝 {name} 文本存在情况")
    print("-" * 50)
    print(f"Total      : {total}")
    print(f"With text  : {has_text_count} ({100.0 * has_text_count / max(total,1):.2f}%)")
    print(f"No text    : {no_text_count} ({100.0 * no_text_count / max(total,1):.2f}%)")
    print("-" * 50)


def get_idx_to_emotion():
    return {v: k for k, v in EMOTION_TO_IDX.items()}


def print_text_presence_by_class(subset, name="Dataset"):
    idx_to_emotion = get_idx_to_emotion()

    stats = {
        class_idx: {"with_text": 0, "no_text": 0, "total": 0}
        for class_idx in idx_to_emotion.keys()
    }

    for real_idx in subset.indices:
        sample = subset.dataset[real_idx]

        label = sample["label"]
        has_text = sample["has_text"]

        if torch.is_tensor(label):
            label = int(label.item())
        if torch.is_tensor(has_text):
            has_text = float(has_text.item())

        stats[label]["total"] += 1
        if has_text > 0.5:
            stats[label]["with_text"] += 1
        else:
            stats[label]["no_text"] += 1

    print(f"\n📚 {name} 各类别文本存在情况")
    print("-" * 80)
    print(f"{'Cls':>3s} | {'Emotion':<15s} | {'Total':>5s} | {'WithText':>8s} | {'NoText':>7s} | {'NoTextRatio':>10s}")
    print("-" * 80)

    for class_idx in sorted(idx_to_emotion.keys()):
        class_name = idx_to_emotion[class_idx]
        total = stats[class_idx]["total"]
        with_text = stats[class_idx]["with_text"]
        no_text = stats[class_idx]["no_text"]
        no_text_ratio = 100.0 * no_text / total if total > 0 else 0.0

        print(
            f"{class_idx:>3d} | {class_name:<15s} | {total:>5d} | "
            f"{with_text:>8d} | {no_text:>7d} | {no_text_ratio:>9.2f}%"
        )
    print("-" * 80)


def print_split_distribution(subset, name="Dataset"):
    idx_to_emotion = get_idx_to_emotion()
    counter = Counter()

    for real_idx in subset.indices:
        sample = subset.dataset[real_idx]
        label = sample["label"]
        if torch.is_tensor(label):
            label = label.item()
        counter[int(label)] += 1

    total = sum(counter.values())

    print(f"\n📊 {name} 类别分布 (Total: {total})")
    print("-" * 60)
    for class_idx in sorted(idx_to_emotion.keys()):
        class_name = idx_to_emotion[class_idx]
        count = counter[class_idx]
        ratio = 100.0 * count / total if total > 0 else 0.0
        print(f"{class_idx:>2d} | {class_name:<15s} | {count:>4d} samples | {ratio:>6.2f}%")
    print("-" * 60)


def compute_class_weights(subset, num_classes):
    counter = Counter()

    for real_idx in subset.indices:
        sample = subset.dataset[real_idx]
        label = sample["label"]
        if torch.is_tensor(label):
            label = label.item()
        counter[int(label)] += 1

    total = sum(counter.values())
    weights = []

    for class_idx in range(num_classes):
        count = counter[class_idx]
        weight = total / (num_classes * count) if count > 0 else 0.0
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights


def build_confusion_matrix(preds, labels, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for p, t in zip(preds, labels):
        cm[t.long(), p.long()] += 1
    return cm


def print_confusion_matrix_and_per_class_acc(cm, idx_to_emotion):
    num_classes = cm.size(0)

    print("\n🧾 Validation Confusion Matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(14)
    for i in range(num_classes):
        header += f"{i:>6d}"
    print(header)
    print("-" * len(header))

    for i in range(num_classes):
        row_name = idx_to_emotion[i][:12].ljust(14)
        row_vals = "".join([f"{cm[i, j].item():>6d}" for j in range(num_classes)])
        print(row_name + row_vals)

    print("\n📌 Per-class Accuracy:")
    for i in range(num_classes):
        total_i = cm[i].sum().item()
        correct_i = cm[i, i].item()
        acc_i = 100.0 * correct_i / total_i if total_i > 0 else 0.0
        print(f"{i:>2d} | {idx_to_emotion[i]:<15s} | {correct_i:>3d}/{total_i:<3d} | {acc_i:>6.2f}%")


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


class ProfileWrapper(nn.Module):
    """
    给 thop 用，只返回 logits，避免多输出结构影响 profile
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img_feat, img_mask, aud_feat, aud_mask, txt_feat, txt_mask, has_text):
        logits, _, _ = self.model(
            img_feat=img_feat,
            img_mask=img_mask,
            aud_feat=aud_feat,
            aud_mask=aud_mask,
            txt_feat=txt_feat,
            txt_mask=txt_mask,
            has_text=has_text
        )
        return logits


def profile_model_flops_and_params(model, batch, device):
    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)

    result = {
        "flops": "N/A",
        "params": "N/A",
        "trainable_params": trainable_params,
        "total_params": total_params
    }

    if not THOP_AVAILABLE:
        print("\n⚠️ thop 未安装，无法统计 FLOPs。")
        print("请先安装: pip install thop")
        return result

    try:
        wrapper = ProfileWrapper(model).to(device)
        wrapper.eval()

        v_f = batch["image_feat"].to(device)
        v_m = batch["image_mask"].to(device)
        a_f = batch["audio_feat"].to(device)
        a_m = batch["audio_mask"].to(device)
        t_f = batch["text_feat"].to(device)
        t_m = batch["text_mask"].to(device)
        has_text = batch["has_text"].to(device)

        with torch.no_grad():
            flops, params = profile(
                wrapper,
                inputs=(v_f, v_m, a_f, a_m, t_f, t_m, has_text),
                verbose=False
            )

        flops_str, params_str = clever_format([flops, params], "%.3f")
        result["flops"] = flops_str
        result["params"] = params_str
        return result

    except Exception as e:
        print(f"\n⚠️ FLOPs 统计失败: {e}")
        return result


# =========================================================
# 2. 模型定义（信息论版：InfoNCE / MI surrogate）
# =========================================================

class TemporalAttentionPooling(nn.Module):
    """
    安全版单 token 时序注意力池化
    输入:
        x_padded: [B, T, C]
        valid_mask: [B, T], True 表示有效位置
    输出:
        pooled: [B, D]
    """
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super().__init__()

        if out_dim % 2 != 0:
            raise ValueError(f"out_dim must be even for sinusoidal positional encoding, got {out_dim}")

        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.global_query = nn.Parameter(torch.randn(1, 1, out_dim))
        nn.init.trunc_normal_(self.global_query, std=0.02)

        self.mha = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_padded, valid_mask):
        B, T, _ = x_padded.shape
        valid_mask = valid_mask.bool()

        x = self.proj(x_padded)
        D = x.shape[-1]

        position = torch.arange(T, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, D, 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / D)
        )
        pos_embed = torch.zeros(T, D, device=x.device)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        x = x + pos_embed.unsqueeze(0)

        q = self.global_query.expand(B, -1, -1)

        empty_mask = ~valid_mask.any(dim=1)
        safe_x = x.clone()
        safe_valid_mask = valid_mask.clone()

        if empty_mask.any():
            safe_x[empty_mask] = 0.0
            safe_valid_mask[empty_mask, 0] = True

        padding_mask = ~safe_valid_mask

        out, _ = self.mha(
            query=q,
            key=safe_x,
            value=safe_x,
            key_padding_mask=padding_mask
        )

        out = out.squeeze(1)

        if empty_mask.any():
            out[empty_mask] = 0.0

        out = self.norm(out)
        out = self.dropout(out)
        return out


class ModalityGate(nn.Module):
    """
    softmax 模态权重分配
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 32)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(mid_dim, 3)
        )

    def forward(self, h_v, h_a, h_t, flag_emb):
        gate_input = torch.cat([h_v, h_a, h_t, flag_emb], dim=-1)
        gate_logits = self.gate_net(gate_input)
        gates = torch.softmax(gate_logits, dim=-1)
        g_v = gates[:, 0:1]
        g_a = gates[:, 1:2]
        g_t = gates[:, 2:3]
        return g_v, g_a, g_t


class InformationGuidedTransformerFusion(nn.Module):
    """
    信息论版融合：
        1) 三模态池化
        2) 缺失文本建模
        3) softmax 门控
        4) 分类主干融合
        5) 用 InfoNCE 作为互信息下界代理，约束跨模态共享信息
    """
    def __init__(
        self,
        img_dim=768,
        aud_dim=1024,
        txt_dim=1024,
        hidden_dim=128,
        fusion_dim=128,
        num_classes=8,
        num_heads=8,
        num_layers=1,
        dropout=0.4,
        image_res_scale=0.05,
        mi_dim=128,
        mi_temperature=0.07
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.mi_dim = mi_dim
        self.mi_temperature = mi_temperature

        self.img_in_norm = nn.LayerNorm(img_dim)
        self.aud_in_norm = nn.LayerNorm(aud_dim)
        self.txt_in_norm = nn.LayerNorm(txt_dim)

        self.img_pool = TemporalAttentionPooling(img_dim, hidden_dim, num_heads=num_heads, dropout=0.1)
        self.aud_pool = TemporalAttentionPooling(aud_dim, hidden_dim, num_heads=num_heads, dropout=0.1)
        self.txt_pool = TemporalAttentionPooling(txt_dim, hidden_dim, num_heads=num_heads, dropout=0.1)

        self.missing_text_emb = nn.Parameter(torch.randn(1, hidden_dim))
        nn.init.trunc_normal_(self.missing_text_emb, std=0.02)

        self.text_flag_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.modality_gate = ModalityGate(hidden_dim, dropout=dropout)

        self.v_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.a_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.t_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.flag_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        total_blocks = 6
        self.final_fusion = nn.Sequential(
            nn.Linear(total_blocks * fusion_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_dim, num_classes)

        self.v_mi_proj = nn.Sequential(
            nn.Linear(hidden_dim, mi_dim),
            nn.LayerNorm(mi_dim),
            nn.GELU(),
            nn.Linear(mi_dim, mi_dim)
        )
        self.a_mi_proj = nn.Sequential(
            nn.Linear(hidden_dim, mi_dim),
            nn.LayerNorm(mi_dim),
            nn.GELU(),
            nn.Linear(mi_dim, mi_dim)
        )
        self.t_mi_proj = nn.Sequential(
            nn.Linear(hidden_dim, mi_dim),
            nn.LayerNorm(mi_dim),
            nn.GELU(),
            nn.Linear(mi_dim, mi_dim)
        )

    @staticmethod
    def _cosine_mean(x, y, eps=1e-8):
        x = F.normalize(x, dim=-1, eps=eps)
        y = F.normalize(y, dim=-1, eps=eps)
        return (x * y).sum(dim=-1).mean()

    @staticmethod
    def _cosine_squared_mean(x, y, eps=1e-8):
        x = F.normalize(x, dim=-1, eps=eps)
        y = F.normalize(y, dim=-1, eps=eps)
        cos = (x * y).sum(dim=-1)
        return (cos ** 2).mean()

    def _info_nce_loss(self, z1, z2):
        if z1.size(0) < 2 or z2.size(0) < 2:
            return z1.new_tensor(0.0)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        logits = z1 @ z2.transpose(0, 1) / self.mi_temperature
        labels = torch.arange(z1.size(0), device=z1.device)

        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_12 + loss_21)

    def forward(self, img_feat, img_mask, aud_feat, aud_mask, txt_feat, txt_mask, has_text=None):
        img_feat = self.img_in_norm(img_feat)
        aud_feat = self.aud_in_norm(aud_feat)
        txt_feat = self.txt_in_norm(txt_feat)

        h_v = self.img_pool(img_feat, img_mask)
        h_a = self.aud_pool(aud_feat, aud_mask)
        h_t_raw = self.txt_pool(txt_feat, txt_mask)

        if has_text is None:
            has_text = torch.ones(h_t_raw.size(0), device=h_t_raw.device, dtype=h_t_raw.dtype)
        else:
            has_text = has_text.to(h_t_raw.device, dtype=h_t_raw.dtype)

        has_text = has_text.view(-1, 1)

        missing_text = self.missing_text_emb.expand(h_t_raw.size(0), -1)
        text_exists_mask = has_text > 0.5
        h_t = torch.where(text_exists_mask, h_t_raw, missing_text)

        flag_emb = self.text_flag_proj(has_text)

        g_v, g_a, g_t = self.modality_gate(h_v, h_a, h_t, flag_emb)

        h_v_g = g_v * h_v
        h_a_g = g_a * h_a
        h_t_g = g_t * h_t

        v_feat = self.v_proj(h_v_g)
        a_feat = self.a_proj(h_a_g)
        t_feat = self.t_proj(h_t_g)

        h_shared = h_v_g + h_a_g + h_t_g
        shared_feat = self.shared_proj(h_shared)

        flag_feat = self.flag_proj(flag_emb)
        gate_feat = self.gate_proj(torch.cat([g_v, g_a, g_t], dim=-1))

        fusion_input = torch.cat(
            [v_feat, a_feat, t_feat, shared_feat, flag_feat, gate_feat],
            dim=-1
        )
        fused_feat = self.final_fusion(fusion_input)
        logits = self.classifier(fused_feat)

        z_v = self.v_mi_proj(h_v_g)
        z_a = self.a_mi_proj(h_a_g)
        z_t = self.t_mi_proj(h_t_g)

        mi_va = self._info_nce_loss(z_v, z_a)

        text_valid = (has_text.squeeze(-1) > 0.5)

        if text_valid.sum().item() >= 2:
            mi_vt = self._info_nce_loss(z_v[text_valid], z_t[text_valid])
            mi_at = self._info_nce_loss(z_a[text_valid], z_t[text_valid])
            mi_loss = (mi_va + mi_vt + mi_at) / 3.0
        else:
            mi_vt = z_v.new_tensor(0.0)
            mi_at = z_v.new_tensor(0.0)
            mi_loss = mi_va

        h_shared_aux = self.shared_proj(h_shared)
        align_loss = 1.0 - self._cosine_mean(fused_feat, h_shared_aux)

        h_raw_shared = (h_v + h_a + h_t) / 3.0
        r_v = h_v - h_raw_shared
        r_a = h_a - h_raw_shared
        r_t = h_t - h_raw_shared

        red_loss = (
            self._cosine_squared_mean(r_v, r_a) +
            self._cosine_squared_mean(r_v, r_t) +
            self._cosine_squared_mean(r_a, r_t)
        ) / 3.0

        aux_losses = {
            "align_loss": align_loss,
            "red_loss": red_loss,
            "mi_loss": mi_loss
        }

        aux_outputs = {
            "h_v": h_v,
            "h_a": h_a,
            "h_t_raw": h_t_raw,
            "h_t": h_t,
            "missing_text_emb": missing_text,
            "has_text": has_text,
            "flag_emb": flag_emb,

            "g_v": g_v.detach(),
            "g_a": g_a.detach(),
            "g_t": g_t.detach(),

            "h_v_g": h_v_g,
            "h_a_g": h_a_g,
            "h_t_g": h_t_g,

            "v_feat": v_feat,
            "a_feat": a_feat,
            "t_feat": t_feat,
            "shared_feat": shared_feat,
            "gate_feat": gate_feat,
            "fused_feat": fused_feat,

            "mi_va": mi_va.detach(),
            "mi_vt": mi_vt.detach(),
            "mi_at": mi_at.detach(),

            "h_shared": h_shared,
            "h_raw_shared": h_raw_shared,
            "r_v": r_v,
            "r_a": r_a,
            "r_t": r_t,
        }

        return logits, aux_losses, aux_outputs


# =========================================================
# 3. Trainer
# =========================================================

class IGTFTrainer:
    def __init__(self, args, class_weights):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(EMOTION_TO_IDX)
        self.idx_to_emotion = get_idx_to_emotion()

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        self.model = InformationGuidedTransformerFusion(
            img_dim=args.img_dim,
            aud_dim=args.aud_dim,
            txt_dim=args.txt_dim,
            hidden_dim=args.hidden_dim,
            fusion_dim=args.fusion_dim,
            num_classes=self.num_classes,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            image_res_scale=args.image_res_scale,
            mi_dim=args.mi_dim,
            mi_temperature=args.mi_temperature
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )

        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        class_weights = class_weights.to(self.device)
        print("\n⚖️ 使用类别权重:")
        for i, w in enumerate(class_weights.tolist()):
            print(f"{i:>2d} | {self.idx_to_emotion[i]:<15s} | weight = {w:.4f}")
        print()

        self.ce_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing
        )
        self.val_ce_criterion = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(str(self.output_dir / "logs"))

        self.best_val_acc = 0.0
        self.best_val_macro_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def compute_total_loss(self, logits, labels, aux_losses):
        ce_loss = self.ce_criterion(logits, labels)

        total_loss = (
            ce_loss
            + self.args.lambda_mi * aux_losses["mi_loss"]
            + self.args.lambda_align * aux_losses["align_loss"]
            + self.args.lambda_red * aux_losses["red_loss"]
        )

        loss_dict = {
            "ce_loss": ce_loss,
            "mi_loss": aux_losses["mi_loss"],
            "align_loss": aux_losses["align_loss"],
            "red_loss": aux_losses["red_loss"],
            "total_loss": total_loss
        }
        return total_loss, loss_dict

    def train_epoch(self, dataloader, epoch):
        self.model.train()

        total_total = 0.0
        total_ce = 0.0
        total_mi = 0.0
        total_align = 0.0
        total_red = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            v_f = batch["image_feat"].to(self.device, non_blocking=True)
            v_m = batch["image_mask"].to(self.device, non_blocking=True)
            a_f = batch["audio_feat"].to(self.device, non_blocking=True)
            a_m = batch["audio_mask"].to(self.device, non_blocking=True)
            t_f = batch["text_feat"].to(self.device, non_blocking=True)
            t_m = batch["text_mask"].to(self.device, non_blocking=True)
            has_text = batch["has_text"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    logits, aux_losses, _ = self.model(
                        img_feat=v_f,
                        img_mask=v_m,
                        aud_feat=a_f,
                        aud_mask=a_m,
                        txt_feat=t_f,
                        txt_mask=t_m,
                        has_text=has_text
                    )
                    total_loss, loss_dict = self.compute_total_loss(logits, labels, aux_losses)

                self.scaler.scale(total_loss).backward()

                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, aux_losses, _ = self.model(
                    img_feat=v_f,
                    img_mask=v_m,
                    aud_feat=a_f,
                    aud_mask=a_m,
                    txt_feat=t_f,
                    txt_mask=t_m,
                    has_text=has_text
                )
                total_loss, loss_dict = self.compute_total_loss(logits, labels, aux_losses)

                total_loss.backward()

                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                self.optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            total_total += loss_dict["total_loss"].item()
            total_ce += loss_dict["ce_loss"].item()
            total_mi += loss_dict["mi_loss"].item()
            total_align += loss_dict["align_loss"].item()
            total_red += loss_dict["red_loss"].item()

            if (batch_idx + 1) % 10 == 0:
                acc = total_correct / max(total_samples, 1)
                print(
                    f"  Epoch {epoch+1} [{batch_idx+1}/{len(dataloader)}] "
                    f"Total: {loss_dict['total_loss'].item():.4f} | "
                    f"CE: {loss_dict['ce_loss'].item():.4f} | "
                    f"MI: {loss_dict['mi_loss'].item():.4f} | "
                    f"Align: {loss_dict['align_loss'].item():.4f} | "
                    f"Red: {loss_dict['red_loss'].item():.4f} | "
                    f"Acc: {acc*100:.2f}%"
                )

        return {
            "total_loss": total_total / len(dataloader),
            "ce_loss": total_ce / len(dataloader),
            "mi_loss": total_mi / len(dataloader),
            "align_loss": total_align / len(dataloader),
            "red_loss": total_red / len(dataloader),
            "acc": total_correct / max(total_samples, 1)
        }

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()

        total_ce = 0.0
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with_text_correct = 0
        with_text_total = 0
        without_text_correct = 0
        without_text_total = 0

        gv_sum = 0.0
        ga_sum = 0.0
        gt_sum = 0.0
        gate_batches = 0

        for batch in dataloader:
            v_f = batch["image_feat"].to(self.device)
            v_m = batch["image_mask"].to(self.device)
            a_f = batch["audio_feat"].to(self.device)
            a_m = batch["audio_mask"].to(self.device)
            t_f = batch["text_feat"].to(self.device)
            t_m = batch["text_mask"].to(self.device)
            has_text = batch["has_text"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    logits, _, aux_outputs = self.model(
                        img_feat=v_f,
                        img_mask=v_m,
                        aud_feat=a_f,
                        aud_mask=a_m,
                        txt_feat=t_f,
                        txt_mask=t_m,
                        has_text=has_text
                    )
                    ce_loss = self.val_ce_criterion(logits, labels)
            else:
                logits, _, aux_outputs = self.model(
                    img_feat=v_f,
                    img_mask=v_m,
                    aud_feat=a_f,
                    aud_mask=a_m,
                    txt_feat=t_f,
                    txt_mask=t_m,
                    has_text=has_text
                )
                ce_loss = self.val_ce_criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_ce += ce_loss.item()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            has_text_bool = has_text > 0.5
            with_text_mask = has_text_bool
            without_text_mask = ~has_text_bool

            if with_text_mask.any():
                with_text_correct += (preds[with_text_mask] == labels[with_text_mask]).sum().item()
                with_text_total += with_text_mask.sum().item()

            if without_text_mask.any():
                without_text_correct += (preds[without_text_mask] == labels[without_text_mask]).sum().item()
                without_text_total += without_text_mask.sum().item()

            gv_sum += aux_outputs["g_v"].mean().item()
            ga_sum += aux_outputs["g_a"].mean().item()
            gt_sum += aux_outputs["g_t"].mean().item()
            gate_batches += 1

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        cm = build_confusion_matrix(all_preds, all_labels, self.num_classes)

        # Macro-ACC：各类别准确率平均
        per_class_acc = []
        for i in range(self.num_classes):
            total_i = cm[i].sum().item()
            correct_i = cm[i, i].item()
            acc_i = correct_i / total_i if total_i > 0 else 0.0
            per_class_acc.append(acc_i)

        macro_acc = sum(per_class_acc) / len(per_class_acc)

        # F1：这里使用 weighted F1
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="weighted")

        with_text_acc = with_text_correct / max(with_text_total, 1)
        without_text_acc = without_text_correct / max(without_text_total, 1)

        return {
            "ce_loss": total_ce / len(dataloader),
            "acc": total_correct / max(total_samples, 1),
            "macro_acc": macro_acc,
            "f1": f1,
            "cm": cm,
            "with_text_acc": with_text_acc,
            "without_text_acc": without_text_acc,
            "with_text_total": with_text_total,
            "without_text_total": without_text_total,
            "g_v_mean": gv_sum / max(gate_batches, 1),
            "g_a_mean": ga_sum / max(gate_batches, 1),
            "g_t_mean": gt_sum / max(gate_batches, 1),
        }

    def train(self, train_loader, val_loader):
        print("=" * 90)
        print("🚀 InfoNCE / 信息论风格多模态融合训练启动")
        print("=" * 90)

        sample_batch = next(iter(train_loader))
        model_profile = profile_model_flops_and_params(self.model, sample_batch, self.device)

        print("\n📦 Model Complexity")
        print("-" * 60)
        print(f"FLOPs              : {model_profile['flops']}")
        print(f"Params (profile)   : {model_profile['params']}")
        print(f"Trainable Params   : {model_profile['trainable_params']:,}")
        print(f"Total Params       : {model_profile['total_params']:,}")
        print("-" * 60)

        for epoch in range(self.args.epochs):
            train_stats = self.train_epoch(train_loader, epoch)
            val_stats = self.validate(val_loader)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            improved_acc = False
            improved_f1 = False

            if val_stats["acc"] > self.best_val_acc:
                self.best_val_acc = val_stats["acc"]
                improved_acc = True

            if val_stats["macro_acc"] > self.best_val_macro_acc:
                self.best_val_macro_acc = val_stats["macro_acc"]

            if val_stats["f1"] > self.best_val_f1:
                self.best_val_f1 = val_stats["f1"]
                improved_f1 = True

            state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_val_acc": self.best_val_acc,
                "best_val_macro_acc": self.best_val_macro_acc,
                "best_val_f1": self.best_val_f1,
                "args": vars(self.args)
            }

            print(f"\n{'='*90}")
            print(f"Epoch {epoch+1}/{self.args.epochs} Summary (LR: {current_lr:.6f})")
            print(
                f"Train -> Total: {train_stats['total_loss']:.4f} | "
                f"CE: {train_stats['ce_loss']:.4f} | "
                f"MI: {train_stats['mi_loss']:.4f} | "
                f"Align: {train_stats['align_loss']:.4f} | "
                f"Red: {train_stats['red_loss']:.4f} | "
                f"Acc: {train_stats['acc']*100:.2f}%"
            )
            print(
                f"Val   -> CE: {val_stats['ce_loss']:.4f} | "
                f"Acc: {val_stats['acc']*100:.2f}% | "
                f"Macro-ACC: {val_stats['macro_acc']*100:.2f}% | "
                f"F1: {val_stats['f1']*100:.2f}%"
            )
            print(
                f"Val Text Split -> "
                f"WithText: {val_stats['with_text_acc']*100:.2f}% "
                f"(N={val_stats['with_text_total']}) | "
                f"NoText: {val_stats['without_text_acc']*100:.2f}% "
                f"(N={val_stats['without_text_total']})"
            )
            print(
                f"Val Gate Mean -> "
                f"g_v: {val_stats['g_v_mean']:.4f} | "
                f"g_a: {val_stats['g_a_mean']:.4f} | "
                f"g_t: {val_stats['g_t_mean']:.4f}"
            )
            print(f"{'='*90}\n")

            print_confusion_matrix_and_per_class_acc(val_stats["cm"], self.idx_to_emotion)

            self.writer.add_scalar("Loss/Train_Total", train_stats["total_loss"], epoch)
            self.writer.add_scalar("Loss/Train_CE", train_stats["ce_loss"], epoch)
            self.writer.add_scalar("Loss/Train_MI", train_stats["mi_loss"], epoch)
            self.writer.add_scalar("Loss/Train_Align", train_stats["align_loss"], epoch)
            self.writer.add_scalar("Loss/Train_Red", train_stats["red_loss"], epoch)
            self.writer.add_scalar("Acc/Train", train_stats["acc"], epoch)

            self.writer.add_scalar("Loss/Val_CE", val_stats["ce_loss"], epoch)
            self.writer.add_scalar("Acc/Val", val_stats["acc"], epoch)
            self.writer.add_scalar("Acc/Val_MacroACC", val_stats["macro_acc"], epoch)
            self.writer.add_scalar("F1/Val", val_stats["f1"], epoch)
            self.writer.add_scalar("Acc/Val_WithText", val_stats["with_text_acc"], epoch)
            self.writer.add_scalar("Acc/Val_NoText", val_stats["without_text_acc"], epoch)
            self.writer.add_scalar("Gate/Val_gv_mean", val_stats["g_v_mean"], epoch)
            self.writer.add_scalar("Gate/Val_ga_mean", val_stats["g_a_mean"], epoch)
            self.writer.add_scalar("Gate/Val_gt_mean", val_stats["g_t_mean"], epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            # 每个 epoch 都保存
            torch.save(state, self.output_dir / "checkpoints" / f"epoch_{epoch+1:03d}.pth")

            # 保存最后一个
            torch.save(state, self.output_dir / "checkpoints" / "last.pth")

            # 最佳 acc
            if improved_acc:
                self.patience_counter = 0
                torch.save(state, self.output_dir / "checkpoints" / "best.pth")
                print(f"✨ New best val acc: {self.best_val_acc*100:.2f}%")
            else:
                self.patience_counter += 1
                print(f"⏳ No improvement. Patience: {self.patience_counter}/{self.args.early_stop_patience}")

            # 最佳 F1
            if improved_f1:
                torch.save(state, self.output_dir / "checkpoints" / "best_f1.pth")
                print(f"🏆 New best F1: {self.best_val_f1*100:.2f}%")

            if self.patience_counter >= self.args.early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

        self.writer.close()


# =========================================================
# 4. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True, help="VideoEmotion8 visual feature directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="VideoEmotion8 audio feature directory")
    parser.add_argument("--text_dir", type=str, required=True, help="VideoEmotion8 text feature directory")
    parser.add_argument("--output_dir", type=str, default="outputs/videoemotion8")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)

    parser.add_argument("--img_dim", type=int, default=768)
    parser.add_argument("--aud_dim", type=int, default=1024)
    parser.add_argument("--txt_dim", type=int, default=1024)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--mi_dim", type=int, default=128)
    parser.add_argument("--mi_temperature", type=float, default=0.07)

    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--image_res_scale", type=float, default=0.05)

    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.15)

    parser.add_argument("--lambda_mi", type=float, default=0.05)
    parser.add_argument("--lambda_align", type=float, default=0.005)
    parser.add_argument("--lambda_red", type=float, default=0.0)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")

    full_dataset = MultimodalEmotionDataset(
        image_dir=args.image_dir,
        audio_dir=args.audio_dir,
        text_dir=args.text_dir
    )

    train_size = int(0.66 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print_split_distribution(train_dataset, "Train")
    print_split_distribution(val_dataset, "Val")

    print_text_presence_stats(train_dataset, "Train")
    print_text_presence_stats(val_dataset, "Val")

    print_text_presence_by_class(train_dataset, "Train")
    print_text_presence_by_class(val_dataset, "Val")

    class_weights = compute_class_weights(train_dataset, num_classes=len(EMOTION_TO_IDX))
    print("Class weights:", class_weights.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multimodal_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multimodal_collate_fn
    )

    first_batch = next(iter(train_loader))
    print("\n🔎 Batch shape check:")
    print("image_feat:", tuple(first_batch["image_feat"].shape))
    print("audio_feat:", tuple(first_batch["audio_feat"].shape))
    print("text_feat :", tuple(first_batch["text_feat"].shape))
    print("has_text  :", tuple(first_batch["has_text"].shape))

    trainer = IGTFTrainer(args, class_weights=class_weights)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
