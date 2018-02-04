# -*- coding: utf-8 -*-


"""
Test File
"""

import constants 
from async import SharedWeights
import multiprocessing as mp
from player import create_player_atari

all_players = []
train_threads = []

all_players.append(create_player_atari(0))
vect_weights_bias= all_players[0].local_network.get_all_weights_bias()
sw = SharedWeights(constants.LEARNING_RATE, vect_weights_bias)

# all_players[0].process(constants.T_MAX, constants.NBR_STEPS, sw)



for idx_thread in range(constants.PARALLEL_THREADS):
    if idx_thread != 0:
        all_players.append(create_player_atari(idx_thread, False))
        
    train_threads.append(mp.Process(target=all_players[idx_thread].process 
                                    ,args=(constants.T_MAX, constants.NBR_STEPS, sw)))
    
for p in train_threads:
    p.start()
        
for p in train_threads:
    p.join()

print('finish')
