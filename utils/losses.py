"""
손실 함수 모음
- FocalLoss: 병변 클래스 (희귀, 다중 레이블) → 어려운 예제에 더 집중
- ClassWeightedBCE: 클래스 빈도 역수 가중치 BCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al., 2017)
    L = -alpha * (1 - p)^gamma * log(p)

    gamma > 0 이면 쉬운 예제(p 높은)의 loss 기여를 줄여
    어려운 희귀 클래스에 집중하게 됨.
    """

    def __init__(self, gamma: float = 3.0, alpha: float = 0.75, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [*, n_class]  (raw, sigmoid 전)
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
    Asymmetric Loss (Ridnik et al., 2021) — 다중 레이블 희귀 클래스에 최적화.

    양성/음성에 다른 focusing parameter 적용:
      - gamma_pos: 양성 예제 focusing (보통 0~2, 낮게)
      - gamma_neg: 음성 예제 focusing (보통 2~4, 높게)
      - clip: 확률 하한 (음성 false negative 억제)

    Focal보다 효과적인 이유:
      - 음성 샘플이 압도적으로 많은 다중 레이블에서
        음성 easy sample만 더 강하게 down-weight
      - 양성 예제는 gamma_pos=1로 mild하게 → gradient 보존
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits → p via numerically stable path
        logits = logits.clamp(-20.0, 20.0)          # pre-clip: Mamba 폭발 방지
        p = torch.sigmoid(logits)
        p = p.clamp(1e-7, 1.0 - 1e-7)              # log(0) 방지

        # 음성 확률에 clip 적용 (easy negative down-weight 강화)
        p_neg = (p + self.clip).clamp(max=1.0 - 1e-7)

        loss_pos = targets       * (1 - p)     ** self.gamma_pos * torch.log(p)
        loss_neg = (1 - targets) * p_neg       ** self.gamma_neg * torch.log(1.0 - p_neg)
        loss = -(loss_pos + loss_neg)
        return loss  # reduction은 train.py에서 mask 적용 후 처리


class ClassWeightedBCE(nn.Module):
    """
    클래스별 양성 빈도의 역수로 가중치를 준 BCE.
    학습 데이터 통계에서 pos_weight를 미리 계산해 넘겨줌.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.pos_weight = pos_weight  # [n_class]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
