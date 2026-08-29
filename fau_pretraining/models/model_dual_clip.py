#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-CLS VideoCLIP 完整模型
✅ 核心架构：分离式时空注意力 (Spatial ViT + Temporal ViT)
✅ 适配能力：224x336 非正方形输入 (Bicubic 位置编码插值)
✅ 接口兼容：完美平替原有的 CLIPFAUModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import List, Optional, Tuple, Dict

from .learnable_prompt import LearnablePrompt

class DualCLIPFAUModel(nn.Module):
    """
    基于分离式时空注意力的全端到端模型
    完全兼容原本的 Text-Anchored 分类逻辑
    """
    def __init__(self, 
                 clip_model_name: str = 'ViT-B/16',
                 num_frames: int = 16,
                 use_learnable_prompt: bool = False,
                 num_context: int = 16,
                 temporal_layers: int = 4,
                 temporal_heads: int = 8,
                 freeze_spatial: bool = True):
        super().__init__()
        
        self.num_frames = num_frames
        self.freeze_spatial = freeze_spatial
        
        # ================= 1. 加载并魔改 CLIP 视觉主干 =================
        print(f"📥 [Dual-CLS] 加载 CLIP 模型：{clip_model_name}")
        self.clip_model, _ = clip.load(clip_model_name, device='cpu', jit=False)
        self.clip_dim = 512 # ViT-B/16 最终投影维度
        
        # 🌟 核心外科手术：位置编码插值，让 CLIP 完美吃下 224x336
        self._interpolate_clip_positional_embedding()
        
        if freeze_spatial:
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False
            print("❄️ [Dual-CLS] CLIP 空间主干已冻结，仅微调时序网络")

        # ================= 2. 纯净的时序 Transformer (Temporal ViT) =================
        print("🎯 [Dual-CLS] 初始化 Temporal ViT (17 Tokens)")
        # 时序级别的 [CLS] Token
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, self.clip_dim))
        nn.init.trunc_normal_(self.temporal_cls_token, std=0.02)
        
        # 时序绝对位置编码 (16帧 + 1个时序CLS = 17)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames + 1, self.clip_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.clip_dim,
            nhead=temporal_heads,
            dim_feedforward=self.clip_dim * 4,
            dropout=0.1,
            activation=nn.GELU(),
            batch_first=True,
            norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
            enable_nested_tensor=False
        )

        # ================= 3. 文本可学习 Prompt =================
        self.use_learnable_prompt = use_learnable_prompt
        if use_learnable_prompt:
            self.learnable_prompt = LearnablePrompt(
                self.clip_model,
                num_context=num_context,
                ctx_dim=self.clip_dim,
                class_specific=True
            )
            self.prompt_initialized = False
        else:
            self.learnable_prompt = None

        # ================= 4. 可学习温度系数 =================
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1. / 0.07)))
        self._count_trainable_params()

    def _interpolate_clip_positional_embedding(self):
        """将 CLIP 的 197 维位置编码拉伸到 295 维 (适配 224x336)"""
        visual_model = self.clip_model.visual
        old_pos_embed = visual_model.positional_embedding # [197, 768]
        
        cls_pos = old_pos_embed[0:1, :]  # [1, 768]
        patch_pos = old_pos_embed[1:, :] # [196, 768]
        embed_dim = old_pos_embed.shape[-1]
        
        # 196 -> 14x14
        patch_pos_2d = patch_pos.reshape(1, 14, 14, embed_dim).permute(0, 3, 1, 2)
        # 双三次插值 -> 14x21
        patch_pos_interp = F.interpolate(patch_pos_2d, size=(14, 21), mode='bicubic', align_corners=False)
        # 重新展平 -> 294
        patch_pos_new = patch_pos_interp.permute(0, 2, 3, 1).reshape(294, embed_dim)
        
        new_pos_embed = torch.cat([cls_pos, patch_pos_new], dim=0) # [295, 768]
        visual_model.positional_embedding = nn.Parameter(new_pos_embed)
        print("🔧 [Dual-CLS] CLIP 位置编码已扩容至 295维 (14x21 patches)")

    def _count_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 总参数量：{total_params:,}")
        print(f"📊 可训练参数量：{trainable_params:,}")
        print(f"📊 可训练比例：{trainable_params/total_params*100:.2f}%")

    def get_trainable_params(self) -> List[Dict]:
        """为优化器提供可训练参数"""
        params = []
        # 时序网络参数
        params.append({'params': self.temporal_transformer.parameters(), 'weight_decay': 0.01})
        params.append({'params': [self.temporal_cls_token, self.temporal_pos_embed], 'weight_decay': 0.0})
        # 文本 Prompt
        if self.use_learnable_prompt and self.learnable_prompt:
            params.append({'params': self.learnable_prompt.parameters(), 'weight_decay': 0.01})
        # 温度系数
        params.append({'params': [self.logit_scale], 'lr': 1e-5, 'weight_decay': 0.0})
        return params

    def encode_event(self, frames: torch.Tensor) -> torch.Tensor:
        """
        核心时空流：提取 16 帧画面的全局视频特征
        frames: [B, 16, 3, 224, 336]
        """
        B, T, C, H, W = frames.shape
        
        # Step 1: 空间层提取 -> [B*16, 512]
        x = frames.view(B * T, C, H, W)
        spatial_cls = self.clip_model.encode_image(x).float()
        
        # Step 2: 时序层提取
        temporal_x = spatial_cls.view(B, T, -1) # [B, 16, 512]
        t_cls_tokens = self.temporal_cls_token.expand(B, -1, -1) # [B, 1, 512]
        temporal_x = torch.cat([t_cls_tokens, temporal_x], dim=1) # [B, 17, 512]
        
        temporal_x = temporal_x + self.temporal_pos_embed
        temporal_out = self.temporal_transformer(temporal_x) # [B, 17, 512]
        
        video_cls = temporal_out[:, 0, :] # 取出时序 CLS [B, 512]
        video_cls = video_cls / video_cls.norm(dim=-1, keepdim=True)
        return video_cls

    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        """编码文本描述"""
        device = next(self.parameters()).device
        if self.use_learnable_prompt and self.learnable_prompt:
            text_features = self.learnable_prompt(class_names, None)
        else:
            prompts = [f"A face showing {name}" for name in class_names]
            tokens = clip.tokenize(prompts).to(device)
            text_features = self.clip_model.encode_text(tokens)
            
        text_features = text_features.to(device, non_blocking=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, frames: torch.Tensor, all_class_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """完美平替的 Forward 接口"""
        device = next(self.parameters()).device
        
        visual_features = self.encode_event(frames)
        text_features = self.encode_text(all_class_names)
        
        visual_features = visual_features.to(device, non_blocking=True)
        text_features = text_features.to(device, non_blocking=True)
        
        logit_scale = self.logit_scale.to(device, non_blocking=True).exp().clamp(max=100.0)
        logits = logit_scale * visual_features @ text_features.t()
        
        return logits, visual_features, text_features