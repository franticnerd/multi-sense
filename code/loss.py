import torch
import torch.nn as nn

class NSNLLLoss(nn.Module):
    def __init__(self):
        super(NSNLLLoss, self).__init__()
    def forward(self, probs):
        loss = torch.sum(torch.log(probs[:, 0])) + torch.sum(torch.log(1 - probs[:, 1:]))
        return loss.neg()
