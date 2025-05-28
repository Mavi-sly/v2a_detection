from slowfast.datasets.build import DATASET_REGISTRY
from torch.utils.data import Dataset
import os
import torch
import cv2
import random

@DATASET_REGISTRY.register()
class Greathits(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.video_dir = cfg.DATA.PATH_TO_DATA_DIR
        self.transform = None  # 可选
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.sampling_rate = cfg.DATA.SAMPLING_RATE
        self.total_stride = self.num_frames * self.sampling_rate
        self.alpha = cfg.SLOWFAST.ALPHA
        self.crop_size = cfg.DATA.TRAIN_CROP_SIZE

        # 加载 split 文件
        split_file = os.path.join(self.video_dir, f"{split}.txt")
        with open(split_file, 'r') as f:
            base_names = [line.strip() for line in f]

        self.samples = []
        for base in base_names:
            video_path = os.path.join(self.video_dir, f"{base}_denoised.mp4")
            label_path = video_path.replace("_denoised.mp4", "_times.txt")

            with open(label_path, 'r') as f:
                event_times = [float(line.split()[0]) for line in f if line.strip()]

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for start_frame in range(0, total_frames - self.total_stride + 1, self.total_stride):
                self.samples.append({
                    "video_path": video_path,
                    "label_times": event_times,
                    "fps": fps,
                    "start_frame": start_frame
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        video_path = sample_info["video_path"]
        label_times = sample_info["label_times"]
        fps = sample_info["fps"]
        start_frame = sample_info["start_frame"]

        cap = cv2.VideoCapture(video_path)
        frames = []
        labels = []

        for i in range(self.num_frames):
            frame_index = start_frame + i * self.sampling_rate
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            # 居中裁剪
            frame_size = self.crop_size
            h, w = frame.shape[:2]
            y = (h - frame_size) // 2
            x = (w - frame_size) // 2
            frame = frame[y:y + frame_size, x:x + frame_size]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

            current_time = frame_index / fps
            label = 1.0 if any(abs(t - current_time) < 0.05 for t in label_times) else 0.0

            frames.append(frame)
            labels.append(label)

        cap.release()

        if len(frames) < self.num_frames:
            return self.__getitem__((idx + 1) % len(self))
        
        # 丢弃一些样本
        # abandon_rate=0
        #
        # if self.num_frames == 16: abandon_rate=0.
        # elif self.num_frames == 32: abandon_rate=0.
        # abandon_rate
        # if self.num_frames == 16 and sum(labels) == 0.0:
        #     if random.random() < 0.5:  # 50% 概率丢弃
        #         return self.__getitem__((idx + 1) % len(self))
        
        
        
        slow_indices = range(0, self.num_frames, self.alpha)
        slow_pathway = torch.stack([frames[i] for i in slow_indices]).permute(1, 0, 2, 3)
        fast_pathway = torch.stack(frames).permute(1, 0, 2, 3)
        inputs = [slow_pathway, fast_pathway]

        labels = torch.tensor(labels)
        video_idx = torch.tensor([idx])
        time = torch.arange(0, self.num_frames)
        meta = {"dummy": torch.tensor(0)}  # 注意此处为 Tensor
        return inputs, labels, video_idx, time, meta


# ✅ 模块级 collate_fn 函数，供 loader.py 引用
def greathits_collate_fn(batch):
    all_inputs_slow, all_inputs_fast = [], []
    all_labels, all_video_idx, all_time = [], [], []
    all_meta = {}

    for inputs, labels, video_idx, time, meta in batch:
        slow_path, fast_path = inputs
        all_inputs_slow.append(slow_path)
        all_inputs_fast.append(fast_path)
        all_labels.append(labels)
        all_video_idx.append(video_idx)
        all_time.append(time)

        for k, v in meta.items():
            if k not in all_meta:
                all_meta[k] = []
            all_meta[k].append(v if isinstance(v, torch.Tensor) else torch.tensor(v))

    for k in all_meta:
        all_meta[k] = torch.stack(all_meta[k])

    inputs = [
        torch.stack(all_inputs_slow),  # [B, C, T_slow, H, W]
        torch.stack(all_inputs_fast),  # [B, C, T_fast, H, W]
    ]

    return (
        inputs,
        torch.stack(all_labels),
        torch.cat(all_video_idx),
        torch.cat(all_time),
        all_meta
    )
