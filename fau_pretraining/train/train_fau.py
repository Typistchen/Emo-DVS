#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP-FAU 预训练脚本 (Dual-CLS 架构)
✅ 架构更新：完美适配 DualCLIPFAUModel
✅ 保存机制：每一轮生成 epoch_x.pth，破除 interval 限制，必定捕捉历史最佳
✅ 评估指标：加入并打印 Train Accuracy 和 Val Accuracy
"""

import os
import sys
import argparse
import yaml
import json
import time
from datetime import datetime
from pathlib import Path

# ✅ 强行注入国内镜像源
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import torch
import torch.nn as nn
import torch.optim as optim
# 使用最新的 torch.amp 接口
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
project_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, project_root)

from models.model_dual_clip import DualCLIPFAUModel
from data.dataset import create_dataloader


# ================= 分布式训练初始化 =================
def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    torch.distributed.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式环境"""
    torch.distributed.destroy_process_group()


# ================= 训练器类 =================
class PretrainTrainer:
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # 输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # 初始化模型
        self.model = self._init_model()
        
        # 优化器
        self.optimizer = self._init_optimizer()
        
        # 学习率调度器
        self.scheduler = self._init_scheduler()
        
        # 混合精度
        self.scaler = GradScaler('cuda') if config.get('mixed_precision', True) else None
        
        # 日志
        if rank == 0:
            self.writer = SummaryWriter(str(self.output_dir / 'logs' / 'tensorboard'))
        else:
            self.writer = None
        
        # 统计
        self.best_val_loss = float('inf')
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],    # ✅ 记录 Acc
            'val_acc': [],      # ✅ 记录 Acc
            'learning_rate': []
        }
    
    def _init_model(self):
        """初始化模型：使用全新的 DualCLIPFAUModel 参数签名"""
        model = DualCLIPFAUModel(
            clip_model_name=self.config.get('clip_model', 'ViT-B/16'),
            num_frames=self.config.get('num_frames', 16),
            use_learnable_prompt=self.config.get('use_learnable_prompt', False),
            num_context=self.config.get('num_context', 16),
            temporal_layers=self.config.get('num_transformer_layers', 4),
            temporal_heads=self.config.get('num_heads', 8),
            freeze_spatial=self.config.get('freeze_clip', True)
        )
        
        model = model.to(self.device)
        
        # 分布式包装
        if self.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                find_unused_parameters=True
            )
        
        return model
    
    def _init_optimizer(self):
        """初始化优化器"""
        if hasattr(self.model, 'module'):
            trainable_params = self.model.module.get_trainable_params()
        else:
            trainable_params = self.model.get_trainable_params()
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        return optimizer
    
    def _init_scheduler(self):
        """初始化学习率调度器（带 Warmup）"""
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        
        warmup_epochs = self.config.get('warmup_epochs', 10)
        
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'] - warmup_epochs,
            eta_min=float(self.config.get('min_lr', 1e-6))
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )
        return scheduler

    def train_epoch(self, dataloader, epoch):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0  # 记录正确预测数
        total_samples = 0  # 记录总样本数
        num_batches = len(dataloader)
        
        # 分布式 sampler
        if self.world_size > 1:
            dataloader.sampler.set_epoch(epoch)
            
        # 🎯 获取全量的 FAU 类别名称（作为固定的文本锚点）
        all_class_names = dataloader.dataset.unique_fau_names
        
        for batch_idx, batch in enumerate(dataloader):
            frames = batch['frames'].to(self.device, non_blocking=True)  # [B, T, C, H, W]
            labels = batch['class_index'].to(self.device, non_blocking=True) # [B]
            
            self.optimizer.zero_grad()
            
            # 使用 torch.amp.autocast
            with autocast('cuda', enabled=(self.scaler is not None)):
                # 🎯 前向传播
                logits, visual_feat, text_feat = self.model(frames, all_class_names)
                
                # 🎯 交叉熵损失
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            # ✅ 计算当前 Batch 的准确率
            with torch.no_grad():
                _, preds = torch.max(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            
            # 统计 Loss
            total_loss += loss.item()
            
            # 打印阶段性进度
            if self.rank == 0 and (batch_idx + 1) % self.config.get('log_interval', 10) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                current_acc = total_correct / total_samples
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{num_batches}] "
                      f"Loss: {avg_loss:.4f} | Acc: {current_acc*100:.2f}%")
        
        avg_loss = total_loss / num_batches
        epoch_acc = total_correct / total_samples
        return avg_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, dataloader, epoch):
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0  # 记录正确预测数
        total_samples = 0  # 记录总样本数
        num_batches = len(dataloader)
        
        # 🎯 获取全量类别名
        all_class_names = dataloader.dataset.unique_fau_names
        
        for batch in dataloader:
            frames = batch['frames'].to(self.device, non_blocking=True)
            labels = batch['class_index'].to(self.device, non_blocking=True)
            
            with autocast('cuda', enabled=(self.scaler is not None)):
                logits, visual_feat, text_feat = self.model(frames, all_class_names)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            total_loss += loss.item()
            
            # ✅ 统计验证集准确率
            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / num_batches
        val_acc = total_correct / total_samples if total_samples > 0 else 0
        return avg_loss, val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点 (每一轮必定触发)"""
        if self.rank != 0:
            return
        
        # 获取模型状态（处理 DDP 包装）
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_state = actual_model.state_dict()
        
        # 1. 构建全量 checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_history': self.train_history
        }
        
        # 🌟 每一轮都单独保存一个专属的 epoch 文件
        epoch_model_path = self.output_dir / 'checkpoints' / f'epoch_{epoch+1}.pth'
        torch.save(checkpoint, epoch_model_path)
        
        # 永远保持 latest.pth 更新
        torch.save(checkpoint, self.output_dir / 'checkpoints' / 'latest.pth')
        
        # 2. 捕捉到最佳模型时，保存 best 和 纯净特征提取器
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoints' / 'best.pth')
            print(f"  ✨ 保存最佳全量模型 (Epoch {epoch+1}, Loss: {self.best_val_loss:.4f})")
            
            # 剥离并保存纯净版时序层，用于后续 MIB 融合网络！
            pretrain_checkpoint = {
                'temporal_transformer_state_dict': actual_model.temporal_transformer.state_dict(),
                'temporal_cls_token': actual_model.temporal_cls_token.data.clone(),
                'temporal_pos_embed': actual_model.temporal_pos_embed.data.clone(),
                'logit_scale': actual_model.logit_scale.data.clone(),
                'config': self.config,
                'epoch': epoch,
                'pretrain_type': 'Dual_CLS_TextAnchored_Classification'
            }
            
            if self.config.get('use_learnable_prompt', False) and actual_model.learnable_prompt is not None:
                pretrain_checkpoint['learnable_prompt_state_dict'] = actual_model.learnable_prompt.state_dict()
            
            torch.save(pretrain_checkpoint, self.output_dir / 'checkpoints' / 'pretrain_for_emotion.pth')
            print(f"  💾 已同步剥离最佳时序网络：pretrain_for_emotion.pth")
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        if self.rank == 0:
            print("=" * 70)
            print("🚀 CLIP-FAU 预训练开始 (Dual-CLS 架构 | 每一轮全量存档)")
            print("=" * 70)
            print(f"📁 输出目录：{self.output_dir}")
            print(f"📊 训练样本：{len(train_loader.dataset)}")
            print(f"📊 验证样本：{len(val_loader.dataset)}")
            print(f"🎯 损失函数：CrossEntropy (文本锚点分类)")
            print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # 训练 
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证 
            val_loss, val_acc = self.validate(val_loader, epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 记录历史
            if self.rank == 0:
                self.train_history['train_loss'].append(train_loss)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['train_acc'].append(train_acc) 
                self.train_history['val_acc'].append(val_acc)   
                self.train_history['learning_rate'].append(current_lr)
                
                # TensorBoard 日志
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('LR', current_lr, epoch)
            
            # 判断并记录当前轮次是否打破了最佳记录
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # 🌟 强制要求每一轮走完必定存档，拆除了 interval 限制！
            self.save_checkpoint(epoch, is_best)
            
            # 打印 epoch 总结 
            if self.rank == 0:
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{self.config['epochs']}")
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
                print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
                print(f"Learning Rate: {current_lr:.6f}")
                print(f"Best Val Loss: {self.best_val_loss:.4f}")
                print(f"{'='*70}\n")
        
        # 保存训练历史字典
        if self.rank == 0:
            with open(self.output_dir / 'logs' / 'training_history.json', 'w') as f:
                json.dump(self.train_history, f, indent=2)
            
            total_time = time.time() - start_time
            print(f"\n🎉 预训练完成！")
            print(f"⏱️  总耗时：{total_time/3600:.2f} 小时")
            print(f"📁 模型保存在：{self.output_dir / 'checkpoints'}")
            
            self.writer.close()
        
        # 清理分布式
        if self.world_size > 1:
            cleanup_distributed()


# ================= 主函数 =================
def main_worker(rank, world_size, config):
    """单卡训练函数"""
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    trainer = PretrainTrainer(config, rank=rank, world_size=world_size)
    
    # DataLoader 创建 (原封不动)
    train_loader, train_dataset = create_dataloader(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        num_frames=config['num_frames'],
        frame_size_h=config['frame_size_h'],
        frame_size_w=config['frame_size_w'],
        split='train',
        num_workers=config.get('num_workers', 4),
        fau_config_path=config.get('fau_config_path'),
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size
    )
    
    val_loader, val_dataset = create_dataloader(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        num_frames=config['num_frames'],
        frame_size_h=config['frame_size_h'],
        frame_size_w=config['frame_size_w'],
        split='val',
        num_workers=config.get('num_workers', 4),
        fau_config_path=config.get('fau_config_path'),
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size
    )
    
    # 开始训练
    trainer.train(train_loader, val_loader)


def main():
    parser = argparse.ArgumentParser(description='CLIP-FAU 预训练')
    parser.add_argument('--config', type=str,
                        default=str(Path(__file__).resolve().parents[1] / 'configs' / 'train_fau.yaml'),
                        help='配置文件路径')
    parser.add_argument('--world_size', type=int, default=1,
                        help='GPU 数量')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 强制类型转换，防止 yaml 解析把科学计数法解析成字符串
    numeric_keys = ['learning_rate', 'min_lr', 'weight_decay', 'dropout', 
                    'batch_size', 'epochs', 'num_frames', 'frame_size',
                    'embed_dim', 'num_transformer_layers', 'num_heads',
                    'num_context', 'save_interval', 'log_interval', 'world_size']
    
    for key in numeric_keys:
        if key in config and isinstance(config[key], str):
            try:
                config[key] = float(config[key])
                if key in ['batch_size', 'epochs', 'num_frames', 'frame_size', 
                          'embed_dim', 'num_transformer_layers', 'num_heads',
                          'num_context', 'save_interval', 'log_interval', 'world_size']:
                    config[key] = int(config[key])
            except (ValueError, TypeError):
                print(f"⚠️  无法转换配置项 {key} = {config[key]}")
    config['world_size'] = args.world_size
    
    # 启动多进程训练
    if config.get('distributed', True) and args.world_size > 1:
        torch.multiprocessing.spawn(
            main_worker,
            args=(args.world_size, config),
            nprocs=args.world_size,
            join=True
        )
    else:
        # 单卡训练
        main_worker(0, 1, config)

if __name__ == '__main__':
    main()
