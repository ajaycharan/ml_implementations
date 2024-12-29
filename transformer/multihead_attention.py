"""
---
title: PyTorch implementation of MHA
---
Inspired from the [annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
Some [training code](autoregressive_example.ipynb) that uses a basic Transformer with MHA for NLP auto-regression 
"""

import math
from typing import Optional, List

import torch
from torch import nn

class PrepareForMHA(nn.Module):
    """
    # Prep for MHA
    Does linear transform and splits the vector into given number of heads
    This is used to transform **key**, **query**, and **value** vectors
    """
    def __init__(self, d_model: int, d_k: int, heads:int, bias: bool):
        """
        :param d_model: input dimension
        :param d_k: dimensions in key vector
        :param heads: number of heads
        :param bias: whether to include a bias term in the linear tfm layer
        """
        super().__init__()
        # linear transform layer
        self.linear = nn.Linear(d_model, heads*d_k, bias)
        self.heads = heads
        # num of dimensions in vectors in each head
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor):
        """
        ## forward pass: Apply the linear transform to the last dimension and split into heads
        :param x: has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`
        """
        head_shape = x.shape[:-1]
        x = self.linear(x)
        # split the last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)
        # output has shape `[seq_len, batch_size, heads, d_model]` or `[batch_size, heads, d_model]`
        return x


