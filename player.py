# -*- coding: utf-8 -*-

from ale import ALEGame
from network import NNetwork
import numpy as np
from constants import FC_PI_POS, FC_LSTM_POS, SM_POS, FC_V_POS

class ActorA3C():
    """
    Class used for one thread. 
    One thread will play and update the deep neural network 
    """    
    def __init__(self, game_name, rand_seed, gamma, thread_index=0):
        self.thread_index = thread_index
        self.game_state = ALEGame(rand_seed * thread_index, game_name)
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
            states = []
            actions = [] # seems unused
            pis = []
            rewards = []
            values = []
            intermediate_values = []
            
            while t<t_max and self.game_state.is_game_over==False:
                # todo : Perform a_t according to policy pi
                lstm_outpus = self.local_network.get_lstm(s_t, FC_LSTM_POS)
                probas_pi = self.local_network.get_pi(lstm_outpus
                                                      , FC_PI_POS
                                                      , SM_POS)
                action = self.get_action_from_pi(probas_pi) # Find best action
                
                self.game_state.process_to_next_image(action)
                
                # done : Receive reward r_t and new state s_t1
                rewards.append(self.game_state.reward)                
                states.append(s_t)
                actions.append(action)
                pis.append(probas_pi[action])
                values.append(self.local_network.get_value(lstm_outpus
                                                           , FC_V_POS))
                
                self.game_state.update()
                s_t = self.game_state.s_t
                
                #Retrieve values NNet for next gradient descent
                intermediate_values.append(
                        self.local_network.get_intermediate_values())
                
                t += 1
                self.T += 1 
      
            if self.game_state.is_game_over:
                R = 0.
            else:
                # todo : V(s_t,theta)
                lstm_outpus = self.local_network.get_lstm(s_t, FC_LSTM_POS)
                R = self.local_network.get_value(lstm_outpus, FC_V_POS)
            
            i = t-1
            d_theta = 0
            d_theta_v = 0
            while i >= 0:                
                R = rewards[i] + self.gamma * R
                
                # todo : compute and accmulate gradients   
                loss_pi = self.local_network.get_loss_pi(R, values[i], pis[i])
                loss_value = self.local_network.get_loss_value(R, values[i])
                d_theta += self.local_network.backpropag_pi(loss_pi
                                                            , intermediate_values[i])
                d_theta_v += self.local_network.backpropag_value(loss_value
                                                                 , intermediate_values[i])
                i -= 1
                
            # todo : Perform Asynchronous update
                
            
    def get_action_from_pi(self, prob_policies):
        """
        Choose an action considering the probabilities of selecting one
        in particular
        """
        return np.random.choice(range(len(prob_policies)), 
                                p=prob_policies.reshape(len(prob_policies)))