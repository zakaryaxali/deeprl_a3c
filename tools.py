# -*- coding: utf-8 -*-

import numpy as np
import math

"""
Useful functions
"""




def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)


def conv_delta(out_data, weights, stride, in_data_size):
    """
    Computes gradient deltas in backpropagation for convolution layer
    Could be optimized : too much loops !!
    Equation (20) in : 
    http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    ret = np.empty((in_data_size, in_data_size))
    out_temp_size = int(weights.shape[0]/stride)
    
    for row in range(in_data_size):
        for col in range(in_data_size):
            row_even = row % 2
            col_even = col % 2
            i = math.ceil(row/stride)
            j = math.ceil(col/stride)
            for m in range(out_temp_size):
                for n in range(out_temp_size):                                         
                    if i-m >=0 and j-n >= 0 and i-m <= out_temp_size and j-n <= out_temp_size:
                        ret[row, col] += out_data[i-m, j-n] * weights[row_even + stride * m, col_even + stride * n]
    
    return ret
    
    
def inv_conv2(in_data, out_data, stride):
    """
    Computes gradient weights in backpropagation for convolution layer
    """
    in_row, in_col = in_data.shape
    out_row, out_col = out_data.shape
    
    kernel_size = get_kernel(in_row, out_row, stride)
    ret = np.empty((kernel_size, kernel_size))
    
    for y in range(0, out_row):
        for x in range(0, out_row):
            sub = in_data[stride*y : stride*y + kernel_size, 
                          stride*x : stride*x + kernel_size]
            ret += np.sum(sub * out_data[y,x])
    return ret


# https://gist.githubusercontent.com/JiaxiangZheng/a60cc8fe1bf6e20c1a41abc98131d518/raw/3630ae57e2e6c5669868a173b763f00fc6ddfb76/CNN.py
def conv2(X, k, stride):
    # as a demo code, here we ignore the shape check
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    
    ret_row = get_height_after_conv(x_row, k_row, stride)            
    ret = np.empty((ret_row, ret_row))
    
    for y in range(0, ret_row):
        for x in range(0, ret_row):
            sub = X[stride*y : stride*y + k_row, stride*x : stride*x + k_col]
            ret[y,x] = np.sum(sub * k)
    return ret

def get_kernel(init_height, output_size, stride):
    return int(init_height-(output_size-1)*stride)
 
def get_height_after_conv(init_height, filter_size, stride):
    return int(((init_height-filter_size)/stride+1))

def rot180(in_data):
    ret = in_data.copy()
    yEnd = ret.shape[0] - 1
    xEnd = ret.shape[1] - 1
    for y in range(ret.shape[0] / 2):
        for x in range(ret.shape[1]):
            ret[yEnd - y][x] = ret[y][x]
    for y in range(ret.shape[0]):
        for x in range(ret.shape[1] / 2):
            ret[y][xEnd - x] = ret[y][x]
    return ret

def padding(in_data, size):
    cur_r, cur_w = in_data.shape[0], in_data.shape[1]
    new_r = cur_r + size * 2
    new_w = cur_w + size * 2
    ret = np.zeros((new_r, new_w))
    ret[size:cur_r + size, size:cur_w+size] = in_data
    return ret