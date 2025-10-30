import torch
from torch import nn
import PIL.Image as Image
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true.float())

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, y_pred, y_true):
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
            y_true = y_true.float()

        # Flatten per-sample
        y_pred = y_pred.view(y_pred.size(0), -1)
        y_true = y_true.view(y_true.size(0), -1)

        # Dice = (2 * sum(p*t) + smooth) / (sum(p)+sum(t) + smooth)
        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Loss = 1 - dice, averaged over batch
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: logits; y_true: {0,1}
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), reduction="none")
        p_t = torch.exp(-bce)  # = sigmoid(logits) for y=1 and 1-sigmoid for y=0
        focal = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            focal = alpha_t * focal
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true.float())

        # TV regularizer on probabilities
        p = torch.sigmoid(y_pred)
        dx = p[..., 1:, :] - p[..., :-1, :]
        dy = p[..., :, 1:] - p[..., :, :-1]
        tv = dx.abs().mean() + dy.abs().mean()
        return bce + 0.1 * tv

