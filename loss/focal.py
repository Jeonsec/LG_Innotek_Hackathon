import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """
    Focal Loss
    Reference:
        https://amaarora.github.io/2020/06/29/FocalLoss.html
    """

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-bce_loss)
        focal_loss = at * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
