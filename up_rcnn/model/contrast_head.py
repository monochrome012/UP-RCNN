import torch
import torch.nn.functional as F
from torch import nn


class ContrastHead(nn.Module):

    def __init__(self, in_channel=256, fc_channel=1024, out_channel=128):
        super(ContrastHead, self).__init__()

        self.fc1 = nn.Linear(in_channel * 7 * 7, fc_channel)
        self.fc2 = nn.Linear(fc_channel, fc_channel)
        self.fc3 = nn.Linear(fc_channel, out_channel)

    def forward(self, x):
        x = torch.cat(x, dim=0)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.normalize(self.fc3(x), dim=1)
        return x
