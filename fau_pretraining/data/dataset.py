
"""
FAU Event Frames 数据集加载器
用于 CLIP-FAU 预训练
✅ 修复 1：使用自然语言文本描述（而非 FAU 代码）
✅ 修复 2：提取全量类别并返回 class_index，用于有监督的文本锚点分类
"""

import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


# ================= FAU 语义名称映射表 =================
FAU_SEMANTIC_NAMES = {
    'FAU1': 'Inner Brow Raiser',
    'FAU2': 'Outer Brow Raiser',
    'FAU4': 'Brow Lowerer',
    'FAU5': 'Upper Lid Raiser',
    'FAU6': 'Cheek Raiser',
    'FAU7': 'Lid Tightener',
    'FAU9': 'Nose Wrinkler',
    'FAU10': 'Upper Lip Raiser',
    'FAU11': 'Nasolabial Deepener',
    'FAU12': 'Lip Corner Puller',
    'FAU13': 'Cheek Puffer',
    'FAU14': 'Dimpler',
    'FAU15': 'Lip Corner Depressor',
    'FAU16': 'Lower Lip Depressor',
    'FAU17': 'Chin Raiser',
    'FAU18': 'Lip Puckerer',
    'FAU20': 'Lip Stretcher',
    'FAU22': 'Lip Funneler',
    'FAU23': 'Lip Tightener',
    'FAU24': 'Lip Pressor',
    'FAU25': 'Lips Part',
    'FAU26': 'Jaw Drop',
    'FAU27': 'Mouth Stretch',
    'FAU28': 'Lip Suck',
    'FAU41': 'Lid Droop',
    'FAU43': 'Eyes Closed',
    'FAU45': 'Blink',
    'FAU51': 'Head Turn Left',
    'FAU52': 'Head Turn Right',
    'FAU53': 'Head Up',
    'FAU54': 'Head Down',
    'FAU55': 'Head Tilt Left',
    'FAU56': 'Head Tilt Right',
    'FAU57': 'Head Forward',
    'FAU58': 'Head Back',
}

# FAU 区域映射（用于层次化描述）
FAU_REGIONS = {
    'FAU1': 'upper face', 'FAU2': 'upper face', 'FAU4': 'upper face',
    'FAU5': 'eye region', 'FAU7': 'eye region', 'FAU41': 'eye region',
    'FAU43': 'eye region', 'FAU45': 'eye region',
    'FAU6': 'lower face', 'FAU9': 'mid face', 'FAU10': 'lower face',
    'FAU11': 'mid face', 'FAU12': 'lower face', 'FAU13': 'lower face',
    'FAU14': 'lower face', 'FAU15': 'lower face', 'FAU16': 'lower face',
    'FAU17': 'lower face', 'FAU18': 'lower face', 'FAU20': 'lower face',
    'FAU22': 'lower face', 'FAU23': 'lower face', 'FAU24': 'lower face',
    'FAU25': 'lower face', 'FAU26': 'lower face', 'FAU27': 'lower face',
    'FAU28': 'lower face',
    'FAU51': 'head', 'FAU52': 'head', 'FAU53': 'head', 'FAU54': 'head',
    'FAU55': 'head', 'FAU56': 'head', 'FAU57': 'head', 'FAU58': 'head',
}


def _generate_text_prompts(fau_id: str, fau_name: str) -> List[str]:
    """
    为 FAU 生成多模板自然语言描述
    """
    region = FAU_REGIONS.get(fau_id, 'facial')
    
    # 多模板库
    templates = [
        # 基础模板
        f"A face showing {fau_name}",
        f"A person with {fau_name.lower()}",
        f"Facial expression of {fau_name}",
        f"Someone displaying {fau_name}",
        
        # 层次化模板
        f"Facial movement in the {region}: {fau_name}",
        f"{region.title()} action: {fau_name}",
        
        # 微表情强调
        f"A subtle facial movement: {fau_name}",
        f"Brief muscle contraction: {fau_name}",
        f"Localized facial motion: {fau_name}",
    ]
    
    return templates


class FAUPretrainDataset(Dataset):
    """
    FAU 预训练数据集
    每个样本：16 帧 event frames + FAU 标签 + 全局类别索引
    """
    def __init__(self, 
                 root_dir: str,
                 num_frames: int = 16,
                 frame_size_h: int = 224,
                 frame_size_w: int = 336,
                 split: str = 'train',
                 fau_config_path: Optional[str] = None,
                 use_multi_template: bool = True):
        """
        Args:
            root_dir: EVK4_Frames_Sampled 根目录
            num_frames: 每个样本的帧数（固定 16）
            frame_size_h: 输入图像高度
            frame_size_w: 输入图像宽度
            split: train/val/test
            fau_config_path: FAU 配置文件路径（可选）
            use_multi_template: 是否使用多模板文本增强（推荐 True）
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.split = split
        self.use_multi_template = use_multi_template
        
        # CLIP 标准预处理
        self.transform = transforms.Compose([
            transforms.Resize((frame_size_h, frame_size_w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        
        # 加载 FAU 配置（如果有）
        self.fau_config = {}
        if fau_config_path and os.path.exists(fau_config_path):
            with open(fau_config_path, 'r', encoding='utf-8') as f:
                self.fau_config = json.load(f)
        
        # 1. 收集所有样本
        self.samples = self._collect_samples()
        random.seed(42)
        random.shuffle(self.samples)
        
        # 2. 🎯 核心修复：提取所有唯一的 FAU 语义名称，并建立全局索引映射
        # 注意：要在 split 之前进行，保证 train 和 val 看到的类别总数和顺序是一致的
        self.unique_fau_names = sorted(list(set([s['fau_name'] for s in self.samples])))
        self.class_to_idx = {name: i for i, name in enumerate(self.unique_fau_names)}
        self.num_classes = len(self.unique_fau_names)
        
        # 3. 划分数据集
        self._split_dataset()
        
        print(f"📊 {split} 集：{len(self.samples)} 个样本")
        print(f"✅ 文本描述：{'多模板增强' if use_multi_template else '单模板'}")
        print(f"🏷️  总类别数 (Num Classes)：{self.num_classes}")
    
    def _collect_samples(self) -> List[Dict]:
        """收集所有样本路径和标签"""
        samples = []
        
        fau_dirs = sorted([d for d in self.root_dir.iterdir() 
                          if d.is_dir() and d.name.upper().startswith('FAU')])
        
        for fau_dir in fau_dirs:
            fau_id = fau_dir.name.upper()
            
            # 优先使用语义映射表，其次使用配置文件，最后用 FAU 代码
            if fau_id in FAU_SEMANTIC_NAMES:
                fau_name = FAU_SEMANTIC_NAMES[fau_id]
            elif fau_id in self.fau_config:
                fau_name = self.fau_config[fau_id].get('name', fau_id)
            else:
                fau_name = fau_id
            
            # 生成多模板文本描述
            text_prompts = _generate_text_prompts(fau_id, fau_name)
            
            for sample_dir in fau_dir.iterdir():
                if sample_dir.is_dir():
                    frame_paths = sorted(sample_dir.glob('sampled_frame_*.png'))
                    
                    if len(frame_paths) >= self.num_frames:
                        samples.append({
                            'frame_paths': frame_paths[:self.num_frames],
                            'fau_id': fau_id,
                            'fau_name': fau_name,
                            'sample_path': str(sample_dir),
                            'text_prompts': text_prompts,
                            'text_prompt': text_prompts[0] if not self.use_multi_template else text_prompts
                        })
        
        return samples
    
    def _split_dataset(self):
        """划分训练/验证/测试集"""
        total = len(self.samples)
        
        if self.split == 'train':
            self.samples = self.samples[:int(total * 0.7)]
        elif self.split == 'val':
            self.samples = self.samples[int(total * 0.7):int(total * 0.8)]
        elif self.split == 'test':
            self.samples = self.samples[int(total * 0.8):]
        elif self.split == 'trainval':
            self.samples = self.samples[:int(total * 0.8)]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载 16 帧
        frames = []
        for frame_path in sample['frame_paths']:
            img = Image.open(frame_path).convert('RGB')
            img = self.transform(img)
            frames.append(img)
        
        frames = torch.stack(frames)
        
        # 训练时随机选择一个文本模板，验证/测试时固定使用第一个
        if self.split == 'train' and self.use_multi_template:
            text_prompt = random.choice(sample['text_prompts'])
        else:
            text_prompt = sample['text_prompts'][0]
        
        return {
            'frames': frames,                      # [T, C, H, W]
            'fau_id': sample['fau_id'],            # e.g., 'FAU18'
            'fau_name': sample['fau_name'],        # e.g., 'Lip Puckerer'
            'text_prompt': text_prompt,            # e.g., 'A face showing Lip Puckerer'
            'sample_path': sample['sample_path'],
            'class_index': self.class_to_idx[sample['fau_name']] # 🎯 核心修复：返回类别索引
        }


def create_dataloader(root_dir: str,
                      batch_size: int = 16,
                      num_frames: int = 16,
                      frame_size_h: int = 224,
                      frame_size_w: int = 336,
                      split: str = 'train',
                      num_workers: int = 4,
                      fau_config_path: Optional[str] = None,
                      use_multi_template: bool = True,
                      distributed: bool = False,
                      rank: int = 0,
                      world_size: int = 1) -> Tuple[DataLoader, FAUPretrainDataset]:
    """创建数据加载器"""
    dataset = FAUPretrainDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        frame_size_h = frame_size_h,
        frame_size_w = frame_size_w,
        split=split,
        fau_config_path=fau_config_path,
        use_multi_template=use_multi_template
    )
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train')
        )
        shuffle = False
    else:
        sampler = None
        shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader, dataset