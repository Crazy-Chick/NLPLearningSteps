#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
class Highway(torch.nn.Module):
    def __init__(self, word_count):
        super(Highway, self).__init__()
        self.conv_to_pro = nn.Linear(word_count, word_count, bias=True)
        self.conv_to_gate = nn.Linear(word_count, word_count, bias=True)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
    
    def forward(self, con_v):
        x_proj = self.relu(self.conv_to_pro(con_v))
        x_gate = self.sigmod(self.conv_to_gate(con_v))
        x_highway = x_gate * x_proj + (1 - x_gate) * con_v
        return x_highway


### END YOUR CODE 

