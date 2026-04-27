#!/usr/bin/env python
"""
DCNv2 compatibility shim using torchvision.ops.deform_conv2d (functional)
=========================================================================
This shim preserves the exact parameter names that YOLACT++ checkpoints
use (conv2.weight, conv2.bias) while using torchvision's deformable
convolution under the hood.

Key: The original DCNv2 stores weight/bias as direct nn.Parameters
(not inside a sub-module), so state_dict keys are:
  backbone.layers.X.Y.conv2.weight
  backbone.layers.X.Y.conv2.bias
  backbone.layers.X.Y.conv2.conv_offset_mask.weight
  backbone.layers.X.Y.conv2.conv_offset_mask.bias
"""

import math
import torch
from torch import nn
from torch.nn import init
from torchvision.ops import deform_conv2d


class DCN(nn.Module):
    """
    Drop-in replacement for DCNv2's DCN module.
    Uses torchvision.ops.deform_conv2d (functional API) to avoid
    sub-module naming issues with state_dict loading.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, deformable_groups=1, bias=True):
        super(DCN, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        
        # Direct weight + bias parameters (matching original DCNv2 state_dict keys)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Offset + mask prediction conv
        # backbone.py accesses: self.conv_offset_mask.weight and .bias
        channels_per_group = 3 * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            deformable_groups * channels_per_group,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        init.zeros_(self.conv_offset_mask.weight)
        init.zeros_(self.conv_offset_mask.bias)
    
    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        
        kh, kw = self.kernel_size
        n = self.deformable_groups * kh * kw
        
        offset = offset_mask[:, :2*n, :, :]
        mask = offset_mask[:, 2*n:, :, :]
        mask = torch.sigmoid(mask)
        
        return deform_conv2d(
            x, offset, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
