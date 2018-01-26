# -*- coding: utf-8 -*-



import numpy as np

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
            out_data = layer.forward(out_data)
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
        # ADD SOFTMAX LAYER
        return np.dot(layer.weights.T, lstm_outputs) + layer.bias
    
    
    
    def get_intermediate_values(self, pos_layer):
        """
        Returns values at each neurons 
        of layers which are useful for the backpropagation
        """
        intermediate_values = []
        for layer_pos in range(1, pos_layer+1):
            layer = self.layers[layer_pos]            
            intermediate_values.append(layer.in_val)
        return intermediate_values 
