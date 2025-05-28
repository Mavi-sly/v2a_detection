import torch
import torch.nn as nn
import torch.nn.functional as F


class GreatHitsHead(nn.Module):
    def __init__(self, dim_slow, dim_fast, num_classes, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.total_dim = dim_slow + dim_fast

        self.proj = nn.Conv3d(self.total_dim, self.total_dim, kernel_size=1)
        self.attention_fc = nn.Linear(self.total_dim, 1)
        self.classifier = nn.Linear(self.total_dim, num_classes)

    def forward(self, x):
        slow, fast = x[0], x[1]  # slow: [B, C1, T1, H, W], fast: [B, C2, T2, H, W]

        # 对 slow 做时间插值，变成 fast 的时间长度
        B, C1, T1, H, W = slow.shape
        T2 = fast.shape[2]
        slow_up = F.interpolate(slow, size=(T2, H, W), mode="trilinear", align_corners=False)

        # 拼接 slow + fast → [B, C1+C2, T2, H, W]
        fused = torch.cat([slow_up, fast], dim=1)

        # 空间池化（只保留时间）
        x = F.adaptive_avg_pool3d(fused, (T2, 1, 1)).squeeze(-1).squeeze(-1)  # [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, C]

        # proj & attention
        x = self.proj(x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1))  # → [B, C, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # → [B, T, C]

        attn = torch.softmax(self.attention_fc(x), dim=1)  # [B, T, 1]
        x = x * attn  # 加权帧特征

        x = self.classifier(self.dropout(x))  # [B, T, num_classes]
        return x.squeeze(-1)

