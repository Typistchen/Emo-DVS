import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

# ================= 1. 核心路径配置 =================
BASE_DIR = Path("data/features")
IMAGE_DIR = BASE_DIR / "VideoEmotion8_visual"
AUDIO_DIR = BASE_DIR / "VideoEmotion8_audio"
TEXT_DIR = BASE_DIR / "VideoEmotion8_text"

# ================= 2. 标签映射表 =================
EMOTION_TO_IDX = {
    "Anger": 0,
    "Anticipation": 1,
    "Disgust": 2,
    "Fear": 3,
    "Joy": 4,
    "Sadness": 5,
    "Surprise": 6,
    "Trust": 7
}


class MultimodalEmotionDataset(Dataset):
    def __init__(
        self,
        image_dir,
        audio_dir,
        text_dir,
        text_dim=1024,
        max_image_len=40000,
        max_audio_len=50000,
        max_text_len=40000
    ):
        """
        三模态变长序列数据集
        允许文本缺失：
          - image/audio 必须存在
          - text 可以不存在或为空
          - 若 image 长度超过 max_image_len，则丢弃
          - 若 audio 长度超过 max_audio_len，则丢弃
          - 若 text 存在且长度超过 max_text_len，则丢弃
        """
        self.image_dir = Path(image_dir)
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)

        self.text_dim = text_dim
        self.max_image_len = max_image_len
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        self.samples = []
        self._build_dataset()

    def _build_dataset(self):
        print("🔍 正在扫描数据并执行模态对齐...")
        print("   规则: image/audio 必须存在, text 允许缺失")
        print(f"   图像长度上限: {self.max_image_len}")
        print(f"   音频长度上限: {self.max_audio_len}")
        print(f"   文本长度上限: {self.max_text_len}")

        image_files = list(self.image_dir.rglob("*.pt"))

        valid_count = 0
        missing_count = 0
        missing_text_count = 0

        long_image_count = 0
        long_audio_count = 0
        long_text_count = 0
        bad_text_count = 0
        bad_image_count = 0
        bad_audio_count = 0

        for img_path in image_files:
            rel_path = img_path.relative_to(self.image_dir)

            aud_path = self.audio_dir / rel_path
            txt_path = self.text_dir / rel_path

            # image / audio 必须有
            if not aud_path.exists():
                missing_count += 1
                continue

            # -------------------------
            # 1) 图像长度过滤
            # -------------------------
            try:
                img_check = torch.load(img_path, map_location="cpu")
                if img_check.dim() == 3 and img_check.size(0) == 1:
                    img_check = img_check.squeeze(0)

                if img_check.dim() != 2:
                    print(f"⚠️ [拦截-图像形状异常] 放弃样本: {rel_path.stem} (shape={tuple(img_check.shape)})")
                    missing_count += 1
                    bad_image_count += 1
                    continue

                img_len = img_check.size(0)
                if img_len > self.max_image_len:
                    print(f"⚠️ [拦截-超长图像] 放弃样本: {rel_path.stem} (图像 {img_len} 帧)")
                    missing_count += 1
                    long_image_count += 1
                    continue

            except Exception:
                print(f"⚠️ [拦截-图像加载失败] 放弃样本: {rel_path.stem}")
                missing_count += 1
                bad_image_count += 1
                continue

            # -------------------------
            # 2) 音频长度过滤
            # -------------------------
            try:
                aud_check = torch.load(aud_path, map_location="cpu")
                if aud_check.dim() == 3 and aud_check.size(0) == 1:
                    aud_check = aud_check.squeeze(0)

                if aud_check.dim() != 2:
                    print(f"⚠️ [拦截-音频形状异常] 放弃样本: {rel_path.stem} (shape={tuple(aud_check.shape)})")
                    missing_count += 1
                    bad_audio_count += 1
                    continue

                aud_len = aud_check.size(0)
                if aud_len > self.max_audio_len:
                    print(f"⚠️ [拦截-超长音频] 放弃样本: {rel_path.stem} (音频 {aud_len} 帧)")
                    missing_count += 1
                    long_audio_count += 1
                    continue

            except Exception:
                print(f"⚠️ [拦截-音频加载失败] 放弃样本: {rel_path.stem}")
                missing_count += 1
                bad_audio_count += 1
                continue

            # -------------------------
            # 3) 文本长度过滤
            # 文本允许缺失，但如果存在且长度 > max_text_len，则丢弃
            # -------------------------
            text_exists = txt_path.exists()
            if not text_exists:
                missing_text_count += 1
            else:
                try:
                    txt_check = torch.load(txt_path, map_location="cpu")
                    if txt_check.dim() == 3 and txt_check.size(0) == 1:
                        txt_check = txt_check.squeeze(0)

                    if txt_check.dim() == 2:
                        txt_len = txt_check.size(0)
                        txt_dim = txt_check.size(1)

                        if txt_dim != self.text_dim:
                            print(f"⚠️ [拦截-文本维度异常] 放弃样本: {rel_path.stem} (shape={tuple(txt_check.shape)})")
                            missing_count += 1
                            bad_text_count += 1
                            continue

                        if txt_len > self.max_text_len:
                            print(f"⚠️ [拦截-超长文本] 放弃样本: {rel_path.stem} (文本 {txt_len} 帧)")
                            missing_count += 1
                            long_text_count += 1
                            continue

                    elif txt_check.numel() == 0:
                        # 空文本允许保留，后续按无文本处理
                        pass
                    else:
                        print(f"⚠️ [拦截-文本形状异常] 放弃样本: {rel_path.stem}")
                        missing_count += 1
                        bad_text_count += 1
                        continue

                except Exception:
                    print(f"⚠️ [拦截-文本加载失败] 放弃样本: {rel_path.stem}")
                    missing_count += 1
                    bad_text_count += 1
                    continue

            emotion_folder = rel_path.parts[0]
            emotion_name = emotion_folder.split("-")[-1]
            label_idx = EMOTION_TO_IDX[emotion_name]

            self.samples.append({
                "image_path": img_path,
                "audio_path": aud_path,
                "text_path": txt_path,
                "text_exists": text_exists,
                "label": label_idx,
                "video_id": rel_path.stem
            })
            valid_count += 1

        print(f"🎬 数据集构建完毕！")
        print(f"   有效样本数: {valid_count}")
        print(f"   丢弃数: {missing_count}")
        print(f"   文本缺失样本数: {missing_text_count}")
        print(f"   超长图像拦截数: {long_image_count}")
        print(f"   超长音频拦截数: {long_audio_count}")
        print(f"   超长文本拦截数: {long_text_count}")
        print(f"   异常图像拦截数: {bad_image_count}")
        print(f"   异常音频拦截数: {bad_audio_count}")
        print(f"   异常文本拦截数: {bad_text_count}")

    def __len__(self):
        return len(self.samples)

    def _load_feature(self, path):
        feat = torch.load(path, map_location="cpu")
        if feat.dim() == 3 and feat.size(0) == 1:
            feat = feat.squeeze(0)
        return feat

    def _load_text_or_dummy(self, sample):
        """
        返回:
          txt_feat: [T, D]
          has_text: float tensor, 1.0 / 0.0
        """
        txt_path = sample["text_path"]

        # 情况 1：文本文件不存在
        if not sample["text_exists"]:
            txt_feat = torch.zeros(1, self.text_dim, dtype=torch.float32)
            has_text = torch.tensor(0.0, dtype=torch.float32)
            return txt_feat, has_text

        # 情况 2：文本文件存在，但可能为空 / 异常 / 全0
        try:
            txt_feat = self._load_feature(txt_path)

            # 空张量或非法形状
            if txt_feat.numel() == 0 or txt_feat.dim() != 2 or txt_feat.size(-1) != self.text_dim:
                txt_feat = torch.zeros(1, self.text_dim, dtype=torch.float32)
                has_text = torch.tensor(0.0, dtype=torch.float32)
                return txt_feat, has_text

            # 双保险：理论上前面已经拦过
            if txt_feat.size(0) > self.max_text_len:
                txt_feat = txt_feat[:self.max_text_len]

            # 如果整体接近全0，则视为无文本
            if txt_feat.abs().sum().item() < 1e-8:
                txt_feat = torch.zeros(1, self.text_dim, dtype=torch.float32)
                has_text = torch.tensor(0.0, dtype=torch.float32)
                return txt_feat, has_text

            has_text = torch.tensor(1.0, dtype=torch.float32)
            return txt_feat, has_text

        except Exception:
            txt_feat = torch.zeros(1, self.text_dim, dtype=torch.float32)
            has_text = torch.tensor(0.0, dtype=torch.float32)
            return txt_feat, has_text

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_feat = self._load_feature(sample["image_path"])
        aud_feat = self._load_feature(sample["audio_path"])
        txt_feat, has_text = self._load_text_or_dummy(sample)

        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "image": img_feat,
            "audio": aud_feat,
            "text": txt_feat,
            "has_text": has_text,
            "label": label,
            "video_id": sample["video_id"]
        }


# ================= 3. Collate Function =================
def multimodal_collate_fn(batch):
    images = [item["image"] for item in batch]
    audios = [item["audio"] for item in batch]
    texts  = [item["text"] for item in batch]
    has_text = torch.stack([item["has_text"] for item in batch], dim=0)  # [B]
    labels = torch.stack([item["label"] for item in batch])
    video_ids = [item["video_id"] for item in batch]

    img_lengths = torch.tensor([x.size(0) for x in images], dtype=torch.long)
    aud_lengths = torch.tensor([x.size(0) for x in audios], dtype=torch.long)
    txt_lengths = torch.tensor([x.size(0) for x in texts], dtype=torch.long)

    images_padded = pad_sequence(images, batch_first=True, padding_value=0.0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    texts_padded  = pad_sequence(texts, batch_first=True, padding_value=0.0)

    B, T_img_max, _ = images_padded.shape
    _, T_aud_max, _ = audios_padded.shape
    _, T_txt_max, _ = texts_padded.shape

    img_mask = torch.arange(T_img_max).expand(B, T_img_max) < img_lengths.unsqueeze(1)
    aud_mask = torch.arange(T_aud_max).expand(B, T_aud_max) < aud_lengths.unsqueeze(1)
    txt_mask = torch.arange(T_txt_max).expand(B, T_txt_max) < txt_lengths.unsqueeze(1)

    return {
        "image_feat": images_padded,
        "audio_feat": audios_padded,
        "text_feat": texts_padded,
        "image_mask": img_mask,
        "audio_mask": aud_mask,
        "text_mask": txt_mask,
        "has_text": has_text,
        "label": labels,
        "video_id": video_ids
    }


# ================= 4. 测试运行代码 =================
if __name__ == "__main__":
    dataset = MultimodalEmotionDataset(
        image_dir=IMAGE_DIR,
        audio_dir=AUDIO_DIR,
        text_dir=TEXT_DIR,
        text_dim=1024,
        max_image_len=10000,
        max_audio_len=50000,
        max_text_len=51300
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=multimodal_collate_fn,
        pin_memory=True
    )

    for batch in dataloader:
        print("\n🔥 成功拉取一个 Batch 的数据！")
        print("has_text:", batch["has_text"])
        print("image_feat shape:", batch["image_feat"].shape)
        print("audio_feat shape:", batch["audio_feat"].shape)
        print("text_feat shape:", batch["text_feat"].shape)
        break
