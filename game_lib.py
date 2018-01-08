# -*- coding: utf-8 -*-
import numpy as np

from skimage.transform import resize
from ale_python_interface import ALEInterface

from constants import ROM, IMAGE_HEIGHT, IMAGE_WIDTH


class GameState(object):
    def __init__(self, rand_seed, display=False, no_op_max=7):
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', rand_seed)
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setBool(b'color_averaging', True)
        self.ale.setInt(b'frame_skip', 4)
        self._no_op_max = no_op_max
        
        if display:
            self._setup_display()
        
        self.ale.loadROM(ROM.encode('ascii'))
        
        # collect minimal action set
        self.real_actions = self.ale.getMinimalActionSet()
        
        # height=210, width=160
        self._screen = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.uint8)
        
        self.reset()
        
    def preprocess_image(self):
        # screen shape is (210, 160, 1)
        self.ale.getScreenGrayscale(self._screen)
        
        # reshape it into (210, 160)
        reshaped_screen = np.reshape(self._screen, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # resize to height=110, width=84
        resized_screen = resize(reshaped_screen, (110, 84))
        
        x_t = resized_screen[18:102,:]        
        x_t = np.reshape(x_t, (84, 84, 1))
        x_t = x_t.astype(np.float32)
        x_t *= (1.0/255.0)
        return x_t
    
    def reset(self):
        self.ale.reset_game()
        
    def get_reward(self):
        return self.reward