import torch
import torch.nn as nn

class NSNLLLoss(nn.Module):
    def __init__(self):
        super(NSNLLLoss, self).__init__()
    def forward(self, probs):
        probs = probs.clamp(min=1e-4, max=1.0-1e-4)
        loss = torch.sum(torch.log(probs[:, 0])) + torch.sum(torch.log(1 - probs[:, 1:]))
        return loss.neg()
