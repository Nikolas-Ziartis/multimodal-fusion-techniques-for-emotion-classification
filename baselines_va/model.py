import torch
import torch.nn as nn


# average over time but ignore padded frames
def masked_mean(x, lengths):
    B, T, D = x.shape
    ar = torch.arange(T, device=x.device).unsqueeze(0)
    mask = (ar < lengths.clamp(min=1).unsqueeze(1)).float()
    mask = mask.unsqueeze(-1)
    return (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)


# plain average over time
def mean_pool(x):
    return x.mean(1)


# small classifier used for each single modality
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class UnimodalNet(nn.Module):
    def __init__(self, modality, feature_dim=768, hidden_dim=256,
                 num_classes=8, dropout=0.5):
        super().__init__()
        assert modality in ('audio', 'video')
        self.modality = modality
        self.head = MLPHead(feature_dim, hidden_dim, num_classes, dropout)

    def forward(self, vision, audio, audio_len):
        if self.modality == 'audio':
            pooled = masked_mean(audio, audio_len)
        else:
            pooled = mean_pool(vision)
        return self.head(pooled)


class ConcatNet(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=256,
                 num_classes=8, dropout=0.5):
        super().__init__()
        self.head = MLPHead(2 * feature_dim, hidden_dim, num_classes, dropout)

    def forward(self, vision, audio, audio_len):
        pv = mean_pool(vision)
        pa = masked_mean(audio, audio_len)
        return self.head(torch.cat([pv, pa], dim=1))


def build_model(name, feature_dim=768, hidden_dim=256, num_classes=8, dropout=0.5):
    if name in ('audio', 'video'):
        return UnimodalNet(name, feature_dim, hidden_dim, num_classes, dropout)
    if name == 'concat':
        return ConcatNet(feature_dim, hidden_dim, num_classes, dropout)
    raise ValueError(f"unknown trained baseline '{name}'")
