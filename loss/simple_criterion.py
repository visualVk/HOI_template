from torch import nn


class SimpleCriterion(nn.Module):
    def __init__(self):
        super(SimpleCriterion, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, s, t):
        loss = self.criterion(s, t)
        return loss
