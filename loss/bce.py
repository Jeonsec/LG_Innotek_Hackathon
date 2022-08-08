import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(x, target)


class BCELosswithLS(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=4):
        super(BCELosswithLS, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        off_value = self.smoothing / self.num_classes
        on_value = 1.0 - self.smoothing + off_value
        target = target * (on_value - off_value) + off_value
        return F.binary_cross_entropy(x, target)
