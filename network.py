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
        return layer.forward(lstm_outputs)
    
    
    def get_pi(self, lstm_outputs, pos_layer1, pos_layer2):
        """
        Returns the probabilities of selecting each action
        """
        layer = self.layers[pos_layer1]
        temp = layer.forward(lstm_outputs)
        layer = self.layers[pos_layer2]
        pi = layer.forward(temp)        
        return pi
    
    
    def get_intermediate_values(self):
        """
        Returns values at each neurons 
        of layers which are useful for the backpropagation
        """
        intermediate_values = []
        for layer_pos in range(1, len(self.layers) + 1):
            layer = self.layers[layer_pos]               
            intermediate_values.append(layer.in_val)
        return intermediate_values 

    
    def get_loss_pi(self, R, V, pi):
        """
        Returns the loss for the policy
        """
        loss_pi = np.log(pi)*(R-V)
        return loss_pi
    
    
    def get_loss_value(self, R, V):
        """
        Returns the loss for the value
        """
        loss_value = (R-V)**2
        return loss_value
    
    
    def backpropag_pi(self, loss, values):
        """
        Makes the backpropagation on all the convolutional network 
        using the probability final layer
        Returns the weigts difference
        """
        out_data = loss        
        layer_pos = len(self.layers) - 1
        
        while layer_pos >=1:
            layer = self.layers[layer_pos]            
            layer.update_val(values[layer_pos-1])
            out_data = layer.backward(out_data)
            layer_pos -= 1        
    
    def get_all_diff_weights_bias(self):
        dw = []
        db = []
        layer_pos = len(self.layers)
        
        while layer_pos >=1:
            layer = self.layers[layer_pos]            
            curt_dw_db = layer.get_diff_weights_bias()
            if not curt_dw_db is None:
                dw.append(curt_dw_db[0])
                db.append(curt_dw_db[1])
                layer.clear_weights_bias()
                
            layer_pos -= 1
            
        return dw, db 
    
    def backpropag_value(self, loss, values):
        """
        Makes the backpropagation on the last value layer
        Returns the weigts difference of that layer
        """
        out_data = loss        
        layer_pos = len(self.layers)
                
        layer = self.layers[layer_pos]            
        layer.update_val(values[layer_pos-1])
        layer.backward(out_data)
         