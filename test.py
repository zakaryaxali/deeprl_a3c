# -*- coding: utf-8 -*-


"""
Test File
"""

from layer import ConvLayer, FCLayer, FlattenLayer, ReLULayer, SoftmaxLayer
from player import ActorA3C
import constants 

player_thread = ActorA3C(game_name=constants.ROM, 
                         rand_seed=constants.INITIAL_SEED, 
                         gamma=constants.GAMMA)

lay_conv1 = ConvLayer(input_channel=constants.SKIPED_FRAMES, 
                      output_channel=constants.CONV1_FILTERS, 
                      kernel_size=constants.CONV1_SIZE, 
                      stride=constants.CONV1_STRIDE)

lay_conv2 = ConvLayer(input_channel=constants.CONV1_FILTERS, 
                      output_channel=constants.CONV2_FILTERS, 
                      kernel_size=constants.CONV2_SIZE, 
                      stride=constants.CONV2_STRIDE)

lay_fc3 = FCLayer(constants.FC_LSTM_UNITS, constants.FC_LSTM_OUTPUTS)

lay_fc4 = FCLayer(constants.FC_PI_UNITS, 
                  len(player_thread.game_state.real_actions))

lay_fc5 = FCLayer(constants.FC_V_UNITS, constants.FC_V_OUTPUTS)

player_thread.local_network.add_layer(lay_conv1, constants.CONV1_POS)
player_thread.local_network.add_layer(ReLULayer(), constants.RELU1_POS)

player_thread.local_network.add_layer(lay_conv2, constants.CONV2_POS)
player_thread.local_network.add_layer(ReLULayer(), constants.RELU2_POS)

player_thread.local_network.add_layer(FlattenLayer(), constants.FLATTEN_POS)

player_thread.local_network.add_layer(lay_fc3, constants.FC_LSTM_POS)
player_thread.local_network.add_layer(ReLULayer(), constants.RELU3_POS)

player_thread.local_network.add_layer(lay_fc4, constants.FC_PI_POS)
player_thread.local_network.add_layer(SoftmaxLayer(), constants.SM_POS)

player_thread.local_network.add_layer(lay_fc5, constants.FC_V_POS)


player_thread.process(constants.T_MAX, constants.NBR_STEPS)
