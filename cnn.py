#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, e_char, e_word):
        super(CNN, self).__init__()
        kernel_size = 5
        self.e_word = e_word
        self.conv = nn.Conv1d(e_char, e_word, kernel_size)
        # self.conv.weight.data.fill_(0.01)
        # self.conv.bias.data.fill_(0.03)
    
    def forward(self, x_reshaped):
        # input == x_reshaped: batch_size * e_char * m_word
        # x_conv: batch_size * e_word * (m_word - k + 1)
        # output == x_conv_out: batch_size * e_word
        x_conv = nn.functional.relu(self.conv(x_reshaped))
        x_conv_out = torch.squeeze(nn.MaxPool1d(x_conv.shape[-1])(x_conv), -1)
        # print(x_conv)
        # print(x_conv_out)
        return x_conv_out

### END YOUR CODE

