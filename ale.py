# -*- coding: utf-8 -*-
import numpy as np
import skimage.transform
from ale_python_interface import ALEInterface

from constants import IMAGE_HEIGHT, IMAGE_WIDTH, SKIPED_FRAMES

class ALEGame(object):
    """
    Class linked to the Arcade Learning Environment
    """
    def __init__(self, rand_seed, game_name):
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', rand_seed)
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setBool(b'color_averaging', True)
        self.ale.setInt(b'frame_skip', SKIPED_FRAMES)
        self.ale.loadROM(game_name.encode('ascii'))
                
        self.real_actions = self.ale.getMinimalActionSet()
        self.screen = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.uint8)
        self.reset()

        
    def preprocess_image(self, is_to_reshape=False):
        """
        Get image from the game and reshape it
        """
        self.ale.getScreenGrayscale(self.screen)                
        reshaped_screen = np.reshape(self.screen, (IMAGE_HEIGHT, IMAGE_WIDTH))    
        x_t = skimage.transform.resize(reshaped_screen
                                       , (110,84)
                                       , preserve_range=True)
        
        x_t = x_t[18:102,:]        
        
        if is_to_reshape:
            x_t = np.reshape(x_t, (84, 84, 1))
            
        x_t = x_t.astype(np.float32)
        x_t *= (1.0/255.0)
        return x_t

    
    def reset(self):        
        """
        Resets the game and create the first state
        """
        self.ale.reset_game()
        self.act(0)
        x_t = self.preprocess_image()
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    
    
    def act(self, action):
        self.reward = self.ale.act(action)
        self.is_game_over = self.ale.game_over()
    
    
    def process_to_next_image(self, action):
        """
        Acts and get new state        
        """
        real_action = self.real_actions[action]
        self.act(real_action)
        x_t1 = self.preprocess_image(True)    
        self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)    


    def update(self):
        self.s_t = self.s_t1

