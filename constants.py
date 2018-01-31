# -*- coding: utf-8 -*-

# ARCADE LEARNING ENVIRONMENT
ROM = "breakout.bin"
INITIAL_SEED = 123
SKIPED_FRAMES = 4
IMAGE_HEIGHT = 210 
IMAGE_WIDTH = 160


# A3C Algorithm
T_MAX = 1e7
NBR_STEPS = 20      
GAMMA = 0.99

# 1ST CONVOLUTIONAL LAYER
CONV1_POS = 1
CONV1_FILTERS = 16
CONV1_SIZE = 8
CONV1_STRIDE = 4

# 2ND CONVOLUTIONAL LAYER
CONV2_POS = 3
CONV2_FILTERS = 32
CONV2_SIZE = 4
CONV2_STRIDE = 2

# FLATTEN LAYER
FLATTEN_POS = 5

# RELU LAYER
RELU1_POS = 2
RELU2_POS = 4
RELU3_POS = 7

# FULLY CONNECTED LAYER
FC_LSTM_POS = 6
FC_LSTM_UNITS = 2592 #9*9*32
FC_LSTM_OUTPUTS = 256

# PROBABILITIES FULLY CONNECTED LAYER
FC_PI_POS = 8
FC_PI_UNITS = 256
SM_POS = 9
# FC_PI_OUTPUTS = 18 => ONLY real actions !!

# VALUE FULLY CONNECTED LAYER
FC_V_POS = 10
FC_V_UNITS = 256
FC_V_OUTPUTS = 1
