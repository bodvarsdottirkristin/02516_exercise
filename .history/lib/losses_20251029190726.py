import torch
from torch import nn
import PIL.Image as Image

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss

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
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        raise Exception("Implement this!")

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        regularization = ...
        return loss + 0.1*regularization

