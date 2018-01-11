# -*- coding: utf-8 -*-

from ale import ALEGame
from network import NNetwork
import numpy as np

class ActorA3C():
    """
    Class used for one thread. 
    One thread will play and update the deep neural network 
    """    
    def __init__(self, game_name, rand_seed, gamma, thread_index=0):
        self.thread_index = thread_index
        self.game_state = ALEGame(rand_seed * thread_index)
        self.gamma = gamma
        self.local_network = NNetwork()
        self.T = 0 #need to change this maybe
        
    def process(self, T_MAX, t_max):
        """
        A3C - pseudocode for each actor-learner thread
        The actor plays the Atari Game
        """
                
        while self.T<T_MAX:
            t = 0
            # todo : Reset gradients : d_theta
            # todo : Synchronize thread-specific parameters 
            # done : Get state s_t
            s_t = self.game_state.s_t
            rewards = []
            
            while t<t_max or self.game_state.is_game_over==False:
                # todo : Perform a_t according to policy pi
                probas_pi = self.local_network.get_pi(s_t)
                action = self.get_action_from_pi(probas_pi) # Find best action
                self.game_state.process_to_next_image(action)
                
                # done : Receive reward r_t and new state s_t1
                rewards.append(self.game_state.reward)
                self.game_state.update()
                s_t = self.game_state.s_t
                t += 1
                self.T += 1 
      
            if self.game_state.is_game_over:
                R = 0.
            else:
                # todo : V(s_t,theta)
                R = self.local_network.get_value(s_t)
            
            i = t-1
            while i >= 0:                
                R = rewards[i] + self.gamma * R
                # todo : compute and accmulate gradients
                
                i -= 1
                
            # todo : Perform Asynchronous update
                
            
    def get_action_from_pi(self, prob_policies):
        """
        Choose an action considering the probabilities of selecting one
        in particular
        """
        return np.random.choice(range(len(prob_policies)), p=prob_policies)