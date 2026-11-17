"""Selected v5 architectures. State-dict field names are part of the model format."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constants import ALPHABET, LETTERS


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
        )


class PlateDetector(nn.Module):
    """Small U-Net: one plate probability per pixel."""

    def __init__(self):
        super().__init__()
        self.a = Block(3, 8)
        self.b = Block(8, 16)
        self.c = Block(16, 32)
        self.d = Block(32, 64)
        self.up_c = Block(96, 32)
        self.up_b = Block(48, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.a(F.avg_pool2d(x, 2))
        b = self.b(F.max_pool2d(a, 2))
        c = self.c(F.max_pool2d(b, 2))
        d = self.d(F.max_pool2d(c, 2))
        c = self.up_c(
            torch.cat(
                (F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False), c), 1
            )
        )
        b = self.up_b(
            torch.cat(
                (F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1
            )
        )
        return F.interpolate(self.out(b), size=x.shape[-2:], mode="bilinear", align_corners=False)


class AttentionReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 40)),
        )
        self.project = nn.Linear(96, 128)
        self.positions = nn.Parameter(torch.randn(1, 40, 128) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, 9, 128) * 0.02)
        block = nn.TransformerDecoderLayer(128, 4, 256, 0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(block, 2, norm=nn.LayerNorm(128))
        self.classifier = nn.Linear(128, len(ALPHABET) + 1)
        allowed = torch.zeros(9, len(ALPHABET) + 1, dtype=torch.bool)
        for i in range(9):
            chars = LETTERS if i in (0, 4, 5) else "0123456789"
            for c in chars:
                allowed[i, ALPHABET.index(c) + 1] = True
        allowed[8, 0] = True
        self.register_buffer("allowed", allowed)

    def forward(self, x):
        memory = self.project(self.features(x).squeeze(2).transpose(1, 2)) + self.positions
        output = self.decoder(self.queries.expand(len(x), -1, -1), memory)
        return self.classifier(output).masked_fill(~self.allowed, -1e4)


def targets(texts):
    target = torch.zeros(len(texts), 9, dtype=torch.long)
    for i, text in enumerate(texts):
        target[i, : len(text)] = torch.tensor([ALPHABET.index(c) + 1 for c in text])
    return target


def decode_attention(logits):
    return ["".join(ALPHABET[i - 1] for i in row if i) for row in logits.argmax(-1).tolist()]


class CharacterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 7, 96),
            nn.ReLU(),
            nn.Linear(96, len(ALPHABET)),
        )

    def forward(self, x):
        return self.layers(x)


class CharacterEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        values = torch.stack([model(x).log_softmax(1) for model in self.models])
        return values.logsumexp(0) - np.log(len(self.models))


class AttentionEnsemble(nn.Module):
    def __init__(self, models, temperatures=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.temperatures = temperatures or [1.0] * len(models)
        if len(self.temperatures) != len(models) or min(self.temperatures) <= 0:
            raise ValueError("Invalid temperatures")

    def forward(self, x):
        return torch.stack(
            [
                (m(x) / temperature).log_softmax(-1)
                for m, temperature in zip(self.models, self.temperatures)
            ]
        ).logsumexp(0) - math.log(len(self.models))
