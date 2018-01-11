# -*- coding: utf-8 -*-

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
    
    def get_pi(self):
        """
        Returns the probabilities of selecting each action
        """
        raise NotImplementedError()
        
    def get_value(self):
        """
        Returns the value of the current state
        """
        raise NotImplementedError()
        
