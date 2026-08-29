#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learnable Prompt: 可学习文本提示
"""

# ✅ 关键修复 1: 导入类型提示
from typing import List, Optional

import torch
import torch.nn as nn
import clip


class LearnablePrompt(nn.Module):
    """
    可学习上下文 Prompt（简化版）
    为每个 FAU 类别学习特定的上下文向量
    """
    def __init__(self, clip_model, num_context=16, ctx_dim=None,
                 class_specific=True):
        super().__init__()
        self.clip_model = clip_model
        self.num_context = num_context
        
        # 获取 CLIP 文本编码器维度
        if ctx_dim is None:
            ctx_dim = clip_model.text_projection.shape[1] if hasattr(clip_model, 'text_projection') else 512
        self.ctx_dim = ctx_dim
        
        self.class_specific = class_specific
        
        if class_specific:
            # 为每个类别学习独立的上下文向量（在 init_prompts 时初始化）
            self.ctx_vectors = None
        else:
            # 共享上下文向量
            self.ctx_vectors = nn.Parameter(torch.randn(num_context, ctx_dim))
    
    def init_class_specific_prompts(self, num_classes):
        """为每个类别初始化独立的 Prompt"""
        self.ctx_vectors = nn.Parameter(
            torch.randn(num_classes, self.num_context, self.ctx_dim)
        )
    
    def forward(self, class_names: List[str], 
                class_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码文本描述（LearnablePrompt 只负责文本编码！）
        
        Args:
            class_names: list of FAU 类别名称，例如 ['Inner Brow Raiser', 'Smile']
            class_indices: [B] 每个样本的类别索引（用于 class_specific，可选）
        
        Returns:
            [B, D] 归一化后的文本特征
        """
        # ✅ 获取当前设备并确保组件同步
        device = next(self.parameters()).device
        
        # 确保 clip_model 也在同一设备
        if next(self.clip_model.parameters()).device != device:
            self.clip_model = self.clip_model.to(device)
        
        # 构建 Prompt 模板
        if self.class_specific and self.ctx_vectors is not None and class_indices is not None:
            # 类别特定 Prompt（简化：仍使用固定模板，可扩展）
            prompts = [f"A face showing {name}" for name in class_names]
        else:
            # 共享 Prompt
            prompts = [f"A face showing {name}" for name in class_names]
        
        # ✅ 确保 tokens 在正确设备
        tokens = clip.tokenize(prompts).to(device)
        
        # 编码文本
        text_features = self.clip_model.encode_text(tokens)
        
        # 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def _build_prompts_with_ctx(self, class_names: List[str], 
                               ctx_vectors: torch.Tensor) -> List[str]:
        """
        构建带可学习上下文的 Prompt（简化版本）
        完整实现需要修改 CLIP 的 token embedding 过程
        """
        # 简化：返回固定模板
        return [f"A face showing {name}" for name in class_names]