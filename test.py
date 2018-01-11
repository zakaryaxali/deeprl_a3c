# -*- coding: utf-8 -*-
"""
Test File
"""

from ale import ALEState
from constants import INITIAL_SEED

game_atari = ALEState(INITIAL_SEED)
print(game_atari.reward)
print(game_atari.s_t)

game_atari.process_to_next_image(0)
print(game_atari.s_t)
