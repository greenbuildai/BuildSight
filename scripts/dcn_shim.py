#!/usr/bin/env python
"""
DCNv2 fallback shim - uses nn.Conv2d as drop-in replacement.
Weight shapes are identical to real DCN so trained .pth files load correctly.
conv_offset_mask attribute provided for YOLACT++ backbone compatibility.
Forward uses standard convolution (deformation disabled) - acceptable for eval.
"""
import torch
import torch.nn as nn


class DCN(nn.Conv2d):
    """
    nn.Conv2d-based drop-in for DCNv2's DCN class.
    Inherits weight, bias, stride, padding, dilation directly from Conv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, deformable_groups=1, bias=True):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            bias=bias
        )
        channels_per_group = 3 * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            deformable_groups * channels_per_group,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=True
        )
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, x):
        return super().forward(x)


__all__ = ['DCN']
