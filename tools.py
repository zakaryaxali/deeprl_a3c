# -*- coding: utf-8 -*-

import numpy as np
import math

"""
Useful functions
"""

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

# https://gist.githubusercontent.com/JiaxiangZheng/a60cc8fe1bf6e20c1a41abc98131d518/raw/3630ae57e2e6c5669868a173b763f00fc6ddfb76/CNN.py
def conv2(X, k, stride):
    # as a demo code, here we ignore the shape check
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    
    ret_row = get_height_after_conv(x_row, k_row, stride)            
    ret = np.empty((ret_row, ret_row))
    
    for y in range(ret_row):
        for x in range(ret_row):
            sub = X[y : y + k_row, x : x + k_col]
            ret[y,x] = np.sum(sub * k)
    return ret


def relu(in_data):
        ret = in_data.copy()
        ret[ret < 0] = 0
        return ret

    
def get_height_after_conv(init_height, filter_size, stride):
    return int(((init_height-filter_size)/stride+1))