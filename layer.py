# -*- coding: utf-8 -*-

import numpy as np
    
class ConvLayer:
    def __init__(self, input_channel, output_channel, kernel_size):
        self.weights = np.random.randn(input_channel, output_channel, 
                                       kernel_size, kernel_size)
        self.bias = np.zeros((output_channel))
        
    def forward(self, input_data):
        raise NotImplementedError()
        
        
class FCLayer:
    def __init__(self, input_num, output_num):
        self.weights = np.random.randn(input_num, output_num)
        self.bias = np.zeros((output_num, 1))
        
        
    def forward(self, input_data):
        return np.dot(self.weights.T, input_data) + self.bias