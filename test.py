# -*- coding: utf-8 -*-
"""
Test File
"""
import tensorflow as tf
from game_lib import GameState

gs = GameState(123)
print(gs.real_actions)
x_t = gs.preprocess_image()
print(x_t)
print(x_t.shape)
print(tf.reshape(x_t,[-1,84,84,1]))
