#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-08-20 15:17
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

"""WaveNet implemented with PyTorch
Environments:
pytorch>=1.6.0
"""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


class WaveNetLayer(nn.Module):
    """Single dilated conv layer in WaveNet
    # Arguments:
        x: input passed to this layer.
        out_channels: number of out_channels used for dilated convolution.
        kernel_size: the kernel size of the dilated convolution.
        dilation: the dilation rate for the dilated convolution.

    # Returns:
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Dilated Conv
        # We use 'same' padding to keep the sequence length consistent
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding,
                               dilation=dilation)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        
        # --- THE FIX IS HERE ---
        # Change out_channels=1 to out_channels=out_channels
        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels, 
                               kernel_size=1)

    def forward(self, x):
        # x shape: [Batch, 32, Length]
        conv_out = self.conv1(x)

        # Gated Activation Unit
        tanh_out = self.tanh(conv_out)
        sigm_out = self.sigm(conv_out)
        x_mul = torch.mul(tanh_out, sigm_out)
        
        # 1x1 Conv to produce skip and residual
        x_skip_connection = self.conv2(x_mul) # Now stays at 32 channels
        
        # Residual connection
        x_residual = torch.add(x, x_skip_connection)

        return x_residual, x_skip_connection


class WaveNetBlock(nn.Module):
    """wavenet_block, serveral wavenet layers which's dilation_rates are 2-based exponentially ascending, form a wavenet_block.
    # Arguments:
        out_channels: number of out_channels used for dilated convolution.
        kernel_size: the kernel size of the dilated convolution.
        n: number of the dilated convolution layers.

    # Returns:
    """
    def __init__(self, in_channels, out_channels, kernel_size, n):
        super().__init__()
        self.n = n
        
        # Use nn.ModuleList instead of Sequential if you need to 
        # manually loop and extract intermediate skip connections.
        self.wavenet_layers = nn.ModuleList([
            WaveNetLayer(in_channels, out_channels, kernel_size, 2**i) 
            for i in range(n)
        ])

    def forward(self, x):
        x_skip_connections = []

        # Iterate through ModuleList correctly
        for layer in self.wavenet_layers:
            # x is the residual path, x_skip is the side path
            x, x_skip = layer(x)
            x_skip_connections.append(x_skip)

        # Return the final residual and the list of skip connections
        return x, x_skip_connections


class WaveNet(nn.Module):
    """WaveNet model. In this configuration, we follow the origin paper, extract skip_connection layers' output to produce predictions.
    # Arguments:
        input_size:
        out_channels:
        kernel_size:
        n:

    # Returns:
    """

    def __init__(self, input_size, out_channels, kernel_size, n):
        super(WaveNet, self).__init__() # Make sure we can call model
        self.in_channels = 1

        self.conv1 = nn.Conv1d(input_size, out_channels, 1) # Change 1 to input_size
        self.wavenet_block = WaveNetBlock(
            out_channels, out_channels, kernel_size, n) # Change self.in_channels to out_channels
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, 1)
        self.conv4 = nn.Conv1d(out_channels, out_channels, 1)
        self.conv5 = nn.Conv1d(out_channels, input_size, 1)
        self.fc = nn.Linear(input_size, 256)
    

    def forward(self, x):
        """
        docstring
        """
        class Sin(nn.Module):
            def forward(self, input):
                return torch.sin(input)
        # Apply causal conv to the input
        x = self.conv1(x)

        # Note that the x_residual output port is not used. It may be used to form multi wavenet_block in a cascading configuration.
        # 2. Block: Returns list of skip tensors, each [Batch, 32, 17672]
        x_residual, x_skip_list = self.wavenet_block(x)

        # 3. Sum the SKIP connections (the list), NOT the channels (dim 1)
        # Using Python's sum() on a list of tensors keeps the shape [Batch, 32, 17672]
        x = sum(x_skip_list)

        # 4. Final Convs
        x = Sin(x)
        x = F.relu(self.conv2(x)) # Expects 32 in, gives 32 out
        x = Sin(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    pass


if __name__ == "__main__":
    main()
