import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            # input shape [None, 4, 84, 84]
            # output shape: [None, 32, 20,20 ]

            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # output shape: [None, 64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # output shape: [None, 64, 4, 4]
            nn.ReLU()
        )

        conv_net_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_net_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv_net(torch.zeros(1, *shape))

        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv_net(x).view(x.size()[0], -1)
        return self.fc(conv_out)