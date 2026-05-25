import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DAB_Block(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # GroupNorm thay thế BatchNorm giúp mô hình cực kỳ ổn định với kích thước Batch nhỏ
        self.norm1 = nn.GroupNorm(8, d_model)
        self.norm2 = nn.GroupNorm(8, d_model)
        
        # Nhánh Gating Mechanism (Disfluency-Aware Bottleneck) lọc nhiễu ngập ngừng, nói lắp
        self.gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # 1. Multi-head Self-Attention
        x_norm = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # 2. Gated Purification (Màng lọc tinh lọc đặc trưng lỗi phát âm)
        g = self.gate(x.transpose(1, 2)).transpose(1, 2)
        x = x * g 
        
        # 3. Position-wise Feed Forward Network
        x_norm = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.ffn(x_norm)
        return x, g

class DAB_Transformer(nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_layers):
        super().__init__()
        # CNN Feature Extractor với tổng bước nhảy Stride đạt chuẩn 32 (8 x 4)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=8, padding=5),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(8, d_model), nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([DAB_Block(d_model, nhead) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.encoder(x.unsqueeze(1)).transpose(1, 2)
        x = self.pos_encoder(x)
        
        all_gates = []
        for b in self.blocks:
            # Hàm bọc custom_forward an toàn để tránh lỗi tuple khi sử dụng checkpoint
            def custom_forward(input_x):
                return b(input_x)
            x, g = checkpoint(custom_forward, x, use_reentrant=False)
            all_gates.append(g)
            
        logits = self.classifier(x)
        return logits, all_gates