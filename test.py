# -*- coding: utf-8 -*-

"""
Test File
"""


import pickle
import constants 
from player import create_player_atari
  
all_players = []
  
all_players.append(create_player_atari(0,  False))
pkl_file = open(constants.INPUT_FILE, 'rb')
pkl = pickle.load(pkl_file)
vect_weights_bias = pkl["weights"]
pkl_file.close()
   
all_players[0].test_play(constants.T_MAX, constants.NBR_STEPS, vect_weights_bias)
    
print('finish')
