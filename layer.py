# -*- coding: utf-8 -*-

import numpy as np
from tools import conv2, get_height_after_conv
#from torch import nn
#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)[source]

class ConvLayer:
    def __init__(self, input_channel, output_channel, kernel_size, stride):
        self.weights = np.random.randn(input_channel, output_channel, 
                                       kernel_size, kernel_size)
        self.bias = np.zeros((output_channel))
        self.stride = stride
        
    def forward(self, input_data):
        
        in_row, in_col, in_channel  = input_data.shape
        out_channel, kernel_size = self.weights.shape[1], self.weights.shape[2]
        out_row = get_height_after_conv(in_row, kernel_size, self.stride)
        self.top_val = np.zeros((out_row , out_row , out_channel))
                
        for o in range(out_channel):
            for i in range(in_channel):
                self.top_val[:,:,o] += conv2(input_data[:,:,i], 
                            self.weights[i, o], 
                            self.stride)
            self.top_val[:,:,o] += self.bias[o]
        return self.top_val
    
        #return nn.conv2d(input_data,  self.weights, 
        #                 strides = [1, self.stride, self.stride, 1], 
        #                 padding='VALID' ) + self.bias
    
        
class FCLayer:
    def __init__(self, input_num, output_num):
        self.weights = np.random.randn(input_num, output_num)
        self.bias = np.zeros((output_num, 1))
        
        
    def forward(self, input_data):
        return np.dot(self.weights.T, input_data) + self.bias


class FlattenLayer:
    def __init__(self):
        pass
    def forward(self, in_data):
        self.r, self.c, self.in_channel  = in_data.shape
        return in_data.reshape( self.in_channel * self.r * self.c, 1)