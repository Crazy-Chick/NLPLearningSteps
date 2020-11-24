#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn

class CNN(torch.nn.Module):
    def __init__(self, char_vec_len, word_count, out_channels, kernel_size):
        super(CNN, self).__init__()
        self.char_vec_len = char_vec_len
        self.word_count = word_count
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1d = torch.nn.Conv1d(in_channels=char_vec_len, out_channels=out_channels, kernel_size=kernel_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.word_count-self.kernel_size+1)
    
    def forward(self, x_reshape):
        x_conv = self.conv1d(x_reshape)
        x_conv = self.relu(x_conv)
        x_conv = self.max_pool(x_conv)
        flag = False
        if (x_conv.shape[0] == 1):
            flag = True
        x_conv = torch.squeeze(x_conv)
        if (flag):
            x_conv = torch.unsqueeze(x_conv, 0)
        return x_conv

### END YOUR CODE

