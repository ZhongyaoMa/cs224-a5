#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        # self.linear1.weight.data.fill_(0.01)
        # self.linear1.bias.data.fill_(0.02)
        # self.linear2.weight.data.fill_(0.03)
        # self.linear2.bias.data.fill_(0.04)
    
    def forward(self, input):
        # input: batch_size * input_size
        # output: batch_size * input_size
        proj = nn.functional.relu(self.linear1(input))
        gate = torch.nn.Sigmoid()(self.linear2(input))
        # print(proj)
        # print(gate)
        # print(output)
        return gate * proj + (1 - gate) * input

### END YOUR CODE 

