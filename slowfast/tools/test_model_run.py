import torch
from slowfast.models.build import build_model
from slowfast.datasets.greathits_dataset import Greathits
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from slowfast.config.defaults import get_cfg

from slowfast.datasets import build_dataset


def test_model():
    cfg = get_cfg()
    cfg.merge_from_file("../configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml")
    cfg.NUM_GPUS = 0

    cfg.DETECTION.ENABLE = False
    cfg.MODEL.NUM_CLASSES = 400
    cfg.DATA.NUM_FRAMES = 32
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 224
    cfg.DATA.INPUT_CHANNEL_NUM = [3, 3]
    cfg.SLOWFAST.ALPHA = 8

    model = build_model(cfg)
    model.eval()

    slow_pathway = torch.randn(1, 3, 4, 224, 224)
    fast_pathway = torch.randn(1, 3, 32, 224, 224)
    inputs = [slow_pathway, fast_pathway]

    with torch.no_grad():
        output = model(inputs)

    print("✅ 模型输出 shape:", output.shape)

def toy_training_step():
    # Step 1: 配置模型
    cfg = get_cfg()
    cfg.merge_from_file("../configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml")
    cfg.DETECTION.ENABLE = False
    cfg.MODEL.NUM_CLASSES = 400
    cfg.DATA.NUM_FRAMES = 32
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.INPUT_CHANNEL_NUM = [3, 3]
    cfg.SLOWFAST.ALPHA = 8
    cfg.NUM_GPUS = 0  # CPU 模式
    cfg.TRAIN.MIXED_PRECISION = False  # 禁用 AMP
    cfg.MODEL.LOSS_FUNC = "cross_entropy"  # 默认分类损失

    # Step 2: 创建模型和损失函数
    model = build_model(cfg)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Step 3: 创建模拟输入
    slow = torch.randn(2, 3, 4, 224, 224)  # [B, C, T_slow, H, W]
    fast = torch.randn(2, 3, 32, 224, 224) # [B, C, T_fast, H, W]
    inputs = [slow, fast]
    labels = torch.randint(0, cfg.MODEL.NUM_CLASSES, (2,))  # 两个样本的标签

    # Step 4: 前向传播、反向传播、优化器 step
    outputs = model(inputs)  # 输出为 [2, 400]
    loss = loss_fn(outputs, labels)
    print("Loss:", loss.item())

    loss.backward()
    optimizer.step()
    print("✅ 成功完成一个训练 step")

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file("../configs/GreatHits/greathits.yaml")
    dataset = build_dataset("greathits", cfg, "train")
    frames, labels = dataset[0]
    print("📦 Frame shape:", frames.shape)  # 应该是 [T, C, H, W]
    print("🏷️  Label shape:", labels.shape)  # 应该是 [T]

