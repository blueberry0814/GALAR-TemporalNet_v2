"""
Loss functions.
- FocalLoss: for rare, multi-label lesion classes — focuses on hard examples
- AsymmetricLoss: asymmetric focusing for positive/negative samples
- ClassWeightedBCE: BCE with inverse-frequency class weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al., 2017)
    L = -alpha * (1 - p)^gamma * log(p)

    gamma > 0 down-weights easy examples (high p), focusing on hard rare classes.
    """

    def __init__(self, gamma: float = 3.0, alpha: float = 0.75, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [*, n_class]  (raw, before sigmoid)
        targets : [*, n_class]  (0 or 1)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al., 2021) — optimized for multi-label rare-class settings.

    Applies different focusing parameters to positives and negatives:
      - gamma_pos: positive focusing (typically 0~2, kept low)
      - gamma_neg: negative focusing (typically 2~4, kept high)
      - clip: probability floor for negatives (suppresses easy-negative false negatives)

    More effective than Focal Loss in multi-label settings where negatives dominate:
      - Aggressively down-weights easy negative samples
      - Preserves positive gradients with mild gamma_pos
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(-20.0, 20.0)          # numerical stability
        p = torch.sigmoid(logits)
        p = p.clamp(1e-7, 1.0 - 1e-7)

        # clip negative probabilities to strengthen easy-negative down-weighting
        p_neg = (p + self.clip).clamp(max=1.0 - 1e-7)

        loss_pos = targets       * (1 - p)     ** self.gamma_pos * torch.log(p)
        loss_neg = (1 - targets) * p_neg       ** self.gamma_neg * torch.log(1.0 - p_neg)
        loss = -(loss_pos + loss_neg)
        return loss  # reduction applied in train.py after mask weighting


class ClassWeightedBCE(nn.Module):
    """
    BCE weighted by the inverse positive frequency per class.
    pos_weight is precomputed from training data statistics.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.pos_weight = pos_weight  # [n_class]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
