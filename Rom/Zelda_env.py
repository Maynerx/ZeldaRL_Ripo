from typing import Any
import gymnasium as gym
from pyboy.utils import WindowEvent
import pyboy
import numpy as np
from memory_adress import *
from skimage.transform import resize
import json
from pathlib import Path
import mediapy as media


with open('/'.join(__file__.split('\\')[:-1]) + '/reward_table.json', 'r+') as f:
    reward_table = json.load(f)

PATH = '/'.join(__file__.split('\\')[:-1]) + '/bin/zelda.gbc'

# TODO : Implement a negatif reward system when the model return in an already known area

class ZeldaEnv(gym.Env):
    def __init__(self, pos, show = False):
        super().__init__()
        self.init_state = 'init.state'
        self.frame_stacks = 3
        self.step_count = 0
        self.max_step = 10000


        self.got_shield = False
        self.visited_location = []
        self.visited_worlds = []


        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]


        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

        self.statut = False
        self.pyboy = pyboy.PyBoy(
            PATH,
            window='SDL2' if show else'null'
                )
        self.screen = self.pyboy.screen
        self.old_info = self._get_info()
        self.pyboy.set_emulation_speed(25)
        self.reset_number = 0
        self.reset()
        self.initialize(pos=pos)




    def initialize(self, pos):
        self.full_frame_writer = media.VideoWriter(f'vid/full_{pos}.mp4', (144, 160), fps=60)
        self.full_frame_writer.__enter__()

    def reset(self, seed = None, options = None):
        # TODO : implement the reset method
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)

        
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)
        
        self.step_count = 0
        self.statut = False
        self.old_info = self._get_info()
        self.clear()
        self.reset_number += 1
        return self.render(), self._get_info()
    
    
    def render(self, is_resize = True):
        # TODO : implement the render method
        game_pixels_render = self.screen.ndarray
        if is_resize:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
        return game_pixels_render
    
    def get_image(self):
        img_np_array = self.render(is_resize=False)
        image = np.delete(arr=img_np_array, obj=3, axis=2)
        return image
    
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.get_image())
    
    def step(self, action: Any):
        # TODO : implement the step method
        self._action_on_emulator(action)
        obs = self.render()
        info = self._get_info()
        rewards = self._get_rewards(info)
        if self.step_count >= self.max_step:
            self.statut = True
        else:
            self.step_count += 1

        
        return obs, rewards, self.statut, self.statut, info
    
    def close(self):
        self.pyboy.stop()
        self.full_frame_writer.close()


    def check_change(self):
        return {i:self.old_info[i]!=self._get_info()[i] for i in self.old_info.keys()}
    
    def update(self):
        self.old_info = self._get_info()

    def clear(self):
        self.got_shield = False
        self.visited_location = [(self._get_player_x, self._get_player_y)]
        self.visited_worlds = [self._get_current_world]


    
    def _get_rewards(self, info):
        '''The reward method'''
        # TODO : implement the reward and create a reward table file
        changes = self.check_change()
        reward = 0
        if changes['shield_level'] and self.got_shield == False:
            reward += reward_table['shield']


            
        if changes['map_statut']:#changes['player_location'] and (self._get_player_x, self._get_player_y) not in self.visited_location:
            reward += reward_table['explore']
            self.visited_location.append((self._get_player_x, self._get_player_y))



        if changes['current_world'] and self._get_current_world not in self.visited_worlds:
            reward += reward_table['explore_house']
            self.visited_worlds.append(self._get_current_world)

        self.update()
            
        return reward
    
    def _action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        for i in range(9):
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            self.add_video_frame()
            self.pyboy.tick()
        #self.pyboy.send_input(self.release_actions[action])
    
    

    # MARK: memory readers
    
    @property
    def _get_current_world(self):
        '''Get the current world (to determine)'''
        return self.pyboy.memory[CURRENT_WORLD]
    
    @property
    def _get_player_x(self):
        '''Get the player's x position'''
        return self.pyboy.memory[PLAYER_X]

    @property
    def _get_player_y(self):
        '''Get the player's y position'''
        return self.pyboy.memory[PLAYER_Y]
    
    @property
    def _get_maps_statuts(self):
        '''Get maps statuts'''
        return self.pyboy.memory[WORLD_STATUT[0]:WORLD_STATUT[1]]
    
    @property
    def _get_held_items(self):
        '''Get both hield items'''
        return self.pyboy.memory[HELD_ITEM_1:HELD_ITEM_2]
    
    
    @property
    def _get_player_inventory(self):
        '''Get the player's inventory'''
        return self.pyboy.memory[INVENTORY[0]:INVENTORY[1]]
    
    @property
    def _get_number_bombs(self):
        '''Get the player's numbers of bombs'''
        return self.pyboy.memory[NUMBER_BOMBS]
    
    @property
    def _get_number_arrows(self):
        '''Get the player's numbers of arrows'''
        return self.pyboy.memory[NUMBER_ARROWS]
    
    @property
    def _get_shield_level(self):
        '''Get the shield statut'''
        return self.pyboy.memory[SHIELD_LEVEL]
    
    @property
    def _get_sword_level(self):
        '''Get the sword statut'''
        return self.pyboy.memory[SWORD_LEVEL]
    
    @property
    def _get_health_level(self):
        '''Refer to the memory_adress file'''
        return self.pyboy.memory[CURRENT_HEALTH]
    
    @property
    def _get_max_bombs(self):
        '''Get the maximum number of bombs'''
        return self.pyboy.memory[MAX_BOMB]
    
    @property
    def _get_max_arrows(self):
        '''Get the maximum number of arrows'''
        return self.pyboy.memory[MAX_ARROWS]
    
    @property
    def _get_nbr_killed_monster(self):
        '''Get the number of killed monsters'''
        return self.pyboy.memory[KILLED_MONSTERS[0]]
    
    @property
    def _get_items_list(self):
        '''Refer to the items table'''
        item_list = {'01' : False, '02' : False, '03' : False, '04' : False, '05' : False, '06' : False, '07' : False, '08' : False, '09' : False, '0A' : False, '0B' : False, '0C' : False, '0D' : False}
        for item_slot in self._get_player_inventory:
            if item_slot in item_list.keys():
                item_list[item_slot] = True
        return item_list
    
    
    def _get_info(self):
        # TODO : add the bolean gets
        '''Return the info after a step occurs'''
        return dict(
            current_world=self._get_current_world,
            player_x=self._get_player_x,
            player_y=self._get_player_y,
            player_location=(self._get_player_x, self._get_player_y),
            map_statut=self._get_maps_statuts,
            held_items=self._get_held_items,
            inventory=self._get_player_inventory,
            nbr_bombs=self._get_number_bombs,
            nbr_arrows=self._get_number_arrows,
            shield_level=self._get_shield_level,
            sword_level=self._get_sword_level,
            health=self._get_health_level,
            max_bombs=self._get_max_bombs,
            maw_arrows=self._get_max_arrows,
            item_list=self._get_items_list,
            killed_monster=self._get_nbr_killed_monster,
            visited_location=self.visited_location,
            visited_world=self.visited_worlds
        )
    
    
    

    

    

