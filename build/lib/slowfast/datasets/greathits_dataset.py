from slowfast.datasets.build import DATASET_REGISTRY
from torch.utils.data import Dataset
import os
import torch
import cv2


@DATASET_REGISTRY.register()
class Greathits(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.video_dir = cfg.DATA.PATH_TO_DATA_DIR
        self.transform = None  # 可添加 transforms

        # 根据 split 加载视频列表
        split_file = os.path.join(self.video_dir, f"{split}.txt")
        with open(split_file, 'r') as f:
            base_names = [line.strip() for line in f]
        self.video_files = [f"{name}_denoised.mp4" for name in base_names]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):

        print(f"读取样本: {idx}")

        video_path = os.path.join(self.video_dir, self.video_files[idx])
        label_path = video_path.replace('_denoised.mp4', '_times.txt')

        # 读取事件时间点
        with open(label_path, 'r') as f:
            event_times = [float(line.split()[0]) for line in f if line.strip()]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = self.cfg.DATA.NUM_FRAMES  # 32
        sampling_rate = self.cfg.DATA.SAMPLING_RATE  # 2
        total_stride = num_frames * sampling_rate  # 64

        samples = []
        start_frame = 0
        while start_frame + total_stride <= total_frames:
            frame_indices = range(start_frame, start_frame + total_stride, sampling_rate)
            frames, labels = [], []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # 居中裁剪
                h, w = frame.shape[:2]
                y = (h - 224) // 2
                x = (w - 224) // 2
                frame = frame[y:y + 224, x:x + 224]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                else:
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

                # 生成标签
                current_time = idx / fps
                label = 1.0 if any(abs(t - current_time) < 0.1 for t in event_times) else 0.0
                labels.append(label)
                frames.append(frame)

            if len(frames) == num_frames:
                alpha = self.cfg.SLOWFAST.ALPHA  # 8
                # 关键修改：生成位置索引，而非直接使用帧号
                slow_position_indices = range(0, len(frames), alpha)
                slow_frames = [frames[i] for i in slow_position_indices]

                # 构造双路径输入
                slow_frames = torch.stack(slow_frames).permute(1, 0, 2, 3)  # [C, T_slow, H, W]
                fast_frames = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T_fast, H, W]
                inputs = [slow_frames, fast_frames]

                # 标签、video_idx、time、meta 逻辑不变
                labels = torch.tensor(labels)
                video_idx = torch.tensor([idx])
                time = torch.arange(0, num_frames)
                meta = {"dummy": 0}
                samples.append((inputs, labels, video_idx, time, meta))

            start_frame += total_stride

        cap.release()

        print("inputs shape:", inputs[0].shape, inputs[1].shape)
        print("labels shape:", labels.shape)

        return samples

    @staticmethod
    def collate_fn(batch):
        all_inputs_slow, all_inputs_fast = [], []
        all_labels, all_video_idx, all_time, all_meta = [], [], [], []

        for video_samples in batch:
            for sample in video_samples:
                inputs, labels, video_idx, time, meta = sample
                slow_path, fast_path = inputs
                all_inputs_slow.append(slow_path)
                all_inputs_fast.append(fast_path)
                all_labels.append(labels)
                all_video_idx.append(video_idx)
                all_time.append(time)
                all_meta.append(meta)

        # 合并为 List[Tensor]
        inputs = [
            torch.stack(all_inputs_slow),  # [B, C, T_slow, H, W]
            torch.stack(all_inputs_fast)  # [B, C, T_fast, H, W]
        ]

        return (
            inputs,
            torch.stack(all_labels),
            torch.cat(all_video_idx),
            torch.cat(all_time),
            all_meta
        )