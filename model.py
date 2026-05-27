"""
model.py — DAB_Transformer

  [M1] CNN encoder
  [M-aug] SpecAugment (train only)
  [M2] Positional encoding
  [M3] DAB blocks
  [M4] Classifier → logits CTC
"""
import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from config import Config
from augment import FeatureSpecAugment


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
        self.norm1 = nn.GroupNorm(8, d_model)
        self.norm2 = nn.GroupNorm(8, d_model)
        self.gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        x_norm = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        g = self.gate(x.transpose(1, 2)).transpose(1, 2)
        x = x * g
        x_norm = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.ffn(x_norm)
        return x, g


class DAB_Transformer(nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_layers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=8, padding=5),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(8, d_model), nn.ReLU(),
        )
        self.spec_augment = FeatureSpecAugment()
        self.pos_encoder = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([DAB_Block(d_model, nhead) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.encoder(x.unsqueeze(1)).transpose(1, 2)
        x = self.spec_augment(x)
        x = self.pos_encoder(x)

        all_gates = []
        for b in self.blocks:
            if Config.USE_GRADIENT_CHECKPOINT and self.training:
                x, g = checkpoint(b, x, use_reentrant=False)
            else:
                x, g = b(x)
            all_gates.append(g)

        logits = self.classifier(x)
        return logits, all_gates
