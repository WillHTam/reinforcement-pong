"""
The DeepMind model has three convolution layers followed by two dense layers.
All layers have the ReLU activation.  
The output is Q-values for every action available in the environment, with no activation. 
"""
import torch
import torch.nn as nn

import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        """
        Don't know the exact number of values in the output from the convolution layer.
        Instead of hard coding the shape, accept a shape argument and apply the convolution layer to a tensor of that shape.
        The result will be the number of parameters returned by this function.   
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Since PyTorch doesn't have a flatten function, do the flatten here.
        Accepts the 4D tensor of batch size, color channels, and image dimensions.
        Apply the convolution layer to the input, then flatten with .view()
        Finally pass the flattened 2D tensor to the dense layers to obtain the Q-values for every batch input.
        """
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
