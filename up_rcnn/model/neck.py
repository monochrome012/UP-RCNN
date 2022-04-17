import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):

    def __init__(self, in_channel=[256, 512, 1024, 2048]):
        super(FPN, self).__init__()

        self.toplayer = nn.Conv2d(in_channel[-1], in_channel[0], 1, 1, 0)

        self.smooth1 = nn.Conv2d(in_channel[0], in_channel[0], 3, 1, 1)
        self.smooth2 = nn.Conv2d(in_channel[0], in_channel[0], 3, 1, 1)
        self.smooth3 = nn.Conv2d(in_channel[0], in_channel[0], 3, 1, 1)

        self.latlayer1 = nn.Conv2d(in_channel[-2], in_channel[0], 1, 1, 0)
        self.latlayer2 = nn.Conv2d(in_channel[-3], in_channel[0], 1, 1, 0)
        self.latlayer3 = nn.Conv2d(in_channel[-4], in_channel[0], 1, 1, 0)

    def forward(self, c2, c3, c4, c5):

        p5 = self.toplayer(c5)
        p4 = F.interpolate(p5, size=c4.shape[-2:], mode='bilinear') + self.latlayer1(c4)
        p3 = F.interpolate(p4, size=c3.shape[-2:], mode='bilinear') + self.latlayer2(c3)
        p2 = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear') + self.latlayer3(c2)

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return [p2, p3, p4, p5]
