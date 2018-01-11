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