import torch
import torch.nn as nn
import torch.nn.functional as F


class TowerNetwork(nn.Module):
    def __init__(self, channels_r=256):
        super(TowerNetwork, self).__init__()
        # initialW=HeNormal(0.1)

        self.conv1_1   = nn.Conv2d(3, channels_r, kernel_size=2, padding=0, stride=2)
        self.conv1_2   = nn.Conv2d(channels_r, channels_r//2, kernel_size=3, padding=1, stride=1)
        self.conv1_res = nn.Conv2d(channels_r, channels_r, kernel_size=2, padding=0, stride=2)
        self.conv1_3   = nn.Conv2d(channels_r//2, channels_r, kernel_size=2, padding=0, stride=2)
        self.conv2_1   = nn.Conv2d(channels_r + 7, channels_r//2, kernel_size=3, padding=1, stride=1)
        self.conv2_2   = nn.Conv2d(channels_r//2, channels_r, kernel_size=3, padding=1, stride=1)
        self.conv2_res = nn.Conv2d(channels_r + 7, channels_r, kernel_size=3, padding=1, stride=1)
        self.conv2_3   = nn.Conv2d(channels_r, channels_r, kernel_size=1, padding=0, stride=1)

    def forward(self, x, v):
        # resnet
        resnet_in = F.relu(self.conv1_1(x))
        residual = F.relu(self.conv1_res(resnet_in))
        out = F.relu(self.conv1_2(resnet_in))
        out = F.relu(self.conv1_3(out)) + residual

        # vector
        v = v.view(v.shape[0], v.shape[1], 1, 1)
        v = v.repeat(1,1,out.shape[2],out.shape[3]) # broadcast_to

        # resnet
        resnet_in = torch.cat((out, v), dim=1)
        residual = F.relu(self.conv2_res(resnet_in))
        out = F.relu(self.conv2_1(resnet_in))
        out = F.relu(self.conv2_2(out)) + residual
        out = self.conv2_3(out)
        return out