# -*- coding: utf-8 -*-

from tensorflow import nn #change tensorflow => not to use run()
from tools import relu
import numpy as np
from tools import softmax

class NNetwork():
    """
    Deep Neural Network that takes in 
    Inputs : images of a game (84,84,4)
    Outputs : 
        - probabilities for each actions 
        - value of the current state V(s) = E[Rt\s_t=s]
    
    """
    def __init__(self):        
        self.pi = []
        self.value = 0
        self.layers = {}
    
    
    def add_layer(self, layer, position):
        self.layers[position] = layer
    
    
    def get_lstm(self, s_t, lstm_pos):
        """
        Returns lstm values
        """
        out_data = s_t        
        for layer_pos in range(1, lstm_pos+1):
            layer = self.layers[layer_pos]            
            out_data = relu(layer.forward(out_data))
        return out_data
        
        
    def get_value(self, lstm_outputs, pos_layer):
        """
        Returns the value of the current state
        """
        layer = self.layers[pos_layer]
        return np.dot(layer.weights.T, lstm_outputs) + layer.bias
    
    
    def get_pi(self, lstm_outputs, pos_layer):
        """
        Returns the probabilities of selecting each action
        """
        layer = self.layers[pos_layer]
        return softmax(np.dot(layer.weights.T, lstm_outputs) + layer.bias)
        
