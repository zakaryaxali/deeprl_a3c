# -*- coding: utf-8 -*-

from multiprocessing.sharedctypes import Array
from ctypes import c_double
import numpy as np

class SharedWeights:
    """
    Class used to share weights between processes
    """
    def __init__(self, learning_rate, inital_theta):
        self.learning_rate = learning_rate
        self.coef_theta = Array(c_double, (inital_theta).flat, lock=False) 
        self.shared_theta = np.frombuffer(self.coef_theta)
        
    def gradient_descent(self, d_theta):
        """
        Updates the shared weights with gradients from a process
        """
        self.shared_theta -= self.learning_rate * d_theta