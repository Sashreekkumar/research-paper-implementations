import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        return 1 - (2 * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)