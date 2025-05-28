import os
import torch
from tqdm import tqdm
from slowfast.config.defaults import get_cfg
from slowfast.datasets.build import build_dataset
from slowfast.utils.parser import load_config
from argparse import ArgumentParser
import argparse  # 别忘了导入 argparse


def count_labels(dataset, threshold=0.05):
    pos_count = 0
    neg_count = 0
    total_clips = 0

    for sample in tqdm(dataset.samples, desc="Analyzing"):
        fps = sample["fps"]
        label_times = sample["label_times"]
        start_frame = sample["start_frame"]
        num_frames = dataset.num_frames
        sampling_rate = dataset.sampling_rate

        for i in range(num_frames):
            frame_index = start_frame + i * sampling_rate
            time_sec = frame_index / fps
            label = 1.0 if any(abs(t - time_sec) < threshold for t in label_times) else 0.0

            if label == 1.0:
                pos_count += 1
            else:
                neg_count += 1

        total_clips += 1

    return pos_count, neg_count, total_clips


def main():
    parser = ArgumentParser()
    parser.add_argument("cfg_path", help="Path to .yaml config file")
    parser.add_argument("--split", default="train", help="Dataset split (train/val)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Time threshold in seconds")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = get_cfg()
    cfg = load_config(args, args.cfg_path)

    dataset = build_dataset(cfg.TRAIN.DATASET if args.split == "train" else cfg.TEST.DATASET, cfg, args.split)

    pos, neg, total = count_labels(dataset, threshold=args.threshold)
    total_labels = pos + neg
    print("\n=== Label Distribution ===")
    print(f"Total clips:         {total}")
    print(f"Total frames:        {total_labels}")
    print(f"Positive frames:     {pos}")
    print(f"Negative frames:     {neg}")
    print(f"Positive ratio:      {pos / total_labels:.4f}")
    print(f"Negative ratio:      {neg / total_labels:.4f}")
    print(f"Pos/Neg ratio:       {pos / neg:.4f}" if neg > 0 else "Neg = 0")
    print(f"[建议 pos_weight 设置] pos_weight = {neg / pos:.2f}")


if __name__ == "__main__":
    main()
