# -*- coding: utf-8 -*-

import numpy as np
from tools import conv2, get_height_after_conv, rot180, padding
#from torch import nn
#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)[source]

class ConvLayer:
    def __init__(self, input_channel, output_channel, kernel_size, stride):
        self.in_val = 0
        self.weights = np.random.randn(input_channel, output_channel, 
                                       kernel_size, kernel_size)
        self.bias = np.zeros((output_channel))
        self.stride = stride
        self.db = np.zeros_like(self.bias)
        self.dw = np.zeros_like(self.weights)   
        
    def update_val(self, val):
        self.in_val = val
        
    def forward(self, input_data):
        self.in_val = input_data
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
    
    def backward(self, residuals):
        in_channel, out_channel, kernel_size, a = self.weights.shape
        dw = np.zeros_like(self.weights)        
        
        for i in range(in_channel):
            for o in range(out_channel):
                dw[i, o] += conv2(self.in_val, residuals[o])
        
        self.db = residuals.sum(axis=3).sum(axis=2).sum(axis=0) 
        self.dw = dw
        
        # gradient_x
        gradient_x = np.zeros_like(self.in_val)
        
        for i in range(in_channel):
            for o in range(out_channel):
                gradient_x[i] += conv2(padding(residuals, kernel_size - 1), 
                          rot180(self.weights[i, o]))
        # IMPORtANT !!!
        # gradient_x /= self.batch_size
        # update
        
        return gradient_x
    
    def get_diff_weights_bias(self):
        return self.dw, self.db
        
        
class FCLayer:
    def __init__(self, input_num, output_num):
        self.in_val = 0
        self.weights = np.random.randn(input_num, output_num)
        self.bias = np.zeros((output_num, 1))
        self.db = np.zeros_like(self.bias)
        self.dw = np.zeros_like(self.weights)       
        
    
    def update_val(self, val):
        self.in_val = val
        
    def forward(self, input_data):
        self.in_val = input_data
        return np.dot(self.weights.T, input_data) + self.bias
    
    def backward(self, loss):
        self.dw = np.dot(self.in_val, loss.T)
        self.db = np.sum(loss) 
        residual_x = np.dot(self.weights, loss)
        return residual_x
    
    def get_diff_weights_bias(self):
        return self.dw, self.db

class FlattenLayer:
    def __init__(self):
        self.in_val = 0
        pass
    
    def update_val(self, val):
        self.in_val = val
        
    def forward(self, in_data):
        self.r, self.c, self.in_channel  = in_data.shape
        return in_data.reshape(self.in_channel * self.r * self.c, 1)

    def backward(self, residual):
        # return residual.reshape(self.in_channel, self.r, self.c)
        return residual.reshape(self.r, self.c, self.in_channel)
    
    def get_diff_weights_bias(self):
        return None, None


class SoftmaxLayer:
    def __init__(self):
        self.in_val = 0
        pass
    
    def update_val(self, val):
        self.in_val = val
    
    def forward(self, x):        
        e_x = np.exp(x)
        temp = e_x / sum(e_x)
        self.in_val = temp
        return temp

    def backward(self, residuals):
        return (self.in_val-residuals)

    def get_diff_weights_bias(self):
        pass

class ReLULayer:
    def __init__(self):
        self.in_val = 0
        pass
    
    def update_val(self, val):
        self.in_val = val
    
    def forward(self, in_data):
        self.in_val = in_data
        ret = in_data.copy()
        ret[ret < 0] = 0
        return ret
    
    def backward(self, residual):
        gradient_x = residual.copy()
        gradient_x[self.in_val < 0] = 0
        return gradient_x
    
    def get_diff_weights_bias(self):
        pass