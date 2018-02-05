# -*- coding: utf-8 -*-

from ale import ALEGame
from network import NNetwork
import numpy as np
import constants 
from layer import ConvLayer, FCLayer, FlattenLayer, ReLULayer, SoftmaxLayer
from tools import get_list_from_vect

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
        self.T = 0
        self.episode_reward = 0
        
    def process(self, T_MAX, t_max, sw):
        """
        A3C - pseudocode for each actor-learner thread
        The actor plays the Atari Game
        """        
        while self.T<T_MAX and sw.stop_process.value==False:
            print('thread'+ str(self.thread_index) + ' T:' + str(self.T))            
            t = 0            
            d_theta = None
            weights_bias = get_list_from_vect(sw.shared_theta
                                               , self.local_network.get_all_shapes())
            self.local_network.update_weights_bias(weights_bias)
            weights_bias = None
            s_t = self.game_state.s_t
            pis = []
            rewards = []
            values = []
            intermediate_values = []
            
            while t<t_max and self.game_state.is_game_over==False:
                lstm_outpus = self.local_network.get_lstm(s_t, constants.RELU3_POS)
                probas_pi = self.local_network.get_pi(lstm_outpus
                                                      , constants.FC_PI_POS
                                                      , constants.SM_POS)
                action = self.get_action_from_pi(probas_pi) # Find best action
                
                self.game_state.process_to_next_image(action)
                rewards.append(self.game_state.reward)    
                self.episode_reward += self.game_state.reward
                pis.append(probas_pi[action])
                values.append(self.local_network.get_value(lstm_outpus
                                                           , constants.FC_V_POS))
                
                self.game_state.update()
                s_t = self.game_state.s_t
                intermediate_values.append(
                        self.local_network.get_intermediate_values())
                
                t += 1
                self.T += 1 
      
            if self.game_state.is_game_over:
                R = 0.
                print('thread '
                      + str(self.thread_index) 
                      + ' finished a game. Reward: ' 
                      + str(self.episode_reward))                
                self.episode_reward = 0                
                self.game_state.reset()
            else:
                lstm_outpus = self.local_network.get_lstm(s_t, constants.RELU3_POS)
                R = self.local_network.get_value(lstm_outpus, constants.FC_V_POS)
            
            i = t-1
            while i >= 0:                
                R = rewards[i] + self.gamma * R                 
                loss_pi = self.local_network.get_loss_pi(R, values[i], pis[i])
                loss_value = self.local_network.get_loss_value(R, values[i])
                self.local_network.backpropag_pi(loss_pi
                                                 , intermediate_values[i])
                self.local_network.backpropag_value(loss_value
                                                    , intermediate_values[i])
                i -= 1
            
            d_theta = self.local_network.get_all_diff_weights_bias()
            sw.gradient_descent(d_theta)            
        
    def test_play(self, T_MAX, t_max, weights):
        """
        A3C - pseudocode for testing a game and getting scores        
        """        
        while self.T<T_MAX:                      
            print('T:' + str(self.T)) 
            t = 0            
            weights_bias = get_list_from_vect(weights
                                               , self.local_network.get_all_shapes())
            self.local_network.update_weights_bias(weights_bias)
            weights_bias = None
            s_t = self.game_state.s_t
            pis = []
            rewards = []
            values = []
            intermediate_values = []
            
            while t<t_max and self.game_state.is_game_over==False:
                lstm_outpus = self.local_network.get_lstm(s_t, constants.RELU3_POS)
                probas_pi = self.local_network.get_pi(lstm_outpus
                                                      , constants.FC_PI_POS
                                                      , constants.SM_POS)
                action = self.get_action_from_pi(probas_pi) 
                
                self.game_state.process_to_next_image(action)
                rewards.append(self.game_state.reward)    
                self.episode_reward += self.game_state.reward
                pis.append(probas_pi[action])
                values.append(self.local_network.get_value(lstm_outpus
                                                           , constants.FC_V_POS))
                
                self.game_state.update()
                s_t = self.game_state.s_t
                intermediate_values.append(
                        self.local_network.get_intermediate_values())
                
                t += 1
                self.T += 1 
      
            if self.game_state.is_game_over:
                print('GameOver. Reward : ' + str(self.episode_reward))
                self.episode_reward = 0                
                self.game_state.reset()                     
            
        
    def get_action_from_pi(self, prob_policies):
        """
        Choose an action considering the probabilities of selecting one
        in particular
        """
        return np.random.choice(range(len(prob_policies)), 
                                p=prob_policies.reshape(len(prob_policies)))
        
        
def create_player_atari(thread_idx, is_theta=True):
    """
    Creates a player, its NeuralNet and all the layers
    Returns the player
    """
    player = ActorA3C(game_name=constants.ROM, 
                         rand_seed=constants.INITIAL_SEED, 
                         gamma=constants.GAMMA,
                         thread_index=thread_idx)
    
    lay_conv1 = ConvLayer(input_channel=constants.SKIPED_FRAMES, 
                      output_channel=constants.CONV1_FILTERS, 
                      kernel_size=constants.CONV1_SIZE, 
                      stride=constants.CONV1_STRIDE, 
                      is_weights_init=is_theta)
    
    lay_conv2 = ConvLayer(input_channel=constants.CONV1_FILTERS, 
                          output_channel=constants.CONV2_FILTERS, 
                          kernel_size=constants.CONV2_SIZE, 
                          stride=constants.CONV2_STRIDE, 
                          is_weights_init=is_theta)
    
    lay_fc3 = FCLayer(constants.FC_LSTM_UNITS, constants.FC_LSTM_OUTPUTS
                      , is_weights_init=is_theta)
    
    lay_fc4 = FCLayer(constants.FC_PI_UNITS, 
                      len(player.game_state.real_actions)
                      ,is_weights_init=is_theta)
    
    lay_fc5 = FCLayer(constants.FC_V_UNITS, constants.FC_V_OUTPUTS
                      ,is_weights_init=is_theta)
    
    player.local_network.add_layer(lay_conv1, constants.CONV1_POS)
    player.local_network.add_layer(ReLULayer(), constants.RELU1_POS)    
    player.local_network.add_layer(lay_conv2, constants.CONV2_POS)
    player.local_network.add_layer(ReLULayer(), constants.RELU2_POS)    
    player.local_network.add_layer(FlattenLayer(), constants.FLATTEN_POS)    
    player.local_network.add_layer(lay_fc3, constants.FC_LSTM_POS)
    player.local_network.add_layer(ReLULayer(), constants.RELU3_POS)
    player.local_network.add_layer(lay_fc4, constants.FC_PI_POS)
    player.local_network.add_layer(SoftmaxLayer(), constants.SM_POS)    
    player.local_network.add_layer(lay_fc5, constants.FC_V_POS)
    
    return player