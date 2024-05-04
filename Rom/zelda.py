import pyboy as pb
from pyboy import PyBoy
import keyboard
from memory_adress import *
import numpy as np


pyboy = PyBoy('rom/bin/zelda.gbc')


curent = None

def create():
    arr = []
    for i in range(0, 256):
        arr += pyboy.memory_scanner.scan_memory(i, start_addr=0x0000, end_addr=0xFFFF)
    return {i:pyboy.memory[i] for i in arr}

def choose_algo(dic, changes):
    for ele in dic.copy().keys():
        if pyboy.memory[ele] != dic[ele]:
            changes[ele] += 1
            dic[ele] = pyboy.memory[ele]
    return dic, changes

def choose_algo_2(dic):
    for ele in dic.copy().keys():
        if pyboy.memory[ele] != dic[ele]:
            del dic[ele]
    return dic

def choose_algo_3(dic, changes):
    for ele in dic.copy().keys():
        if pyboy.memory[ele] != dic[ele]:
            changes[ele] += 1
            dic[ele] = pyboy.memory[ele]
        else:
            del dic[ele]
    return dic, changes

def final(dic, change, i):
    for ele in dic.copy().keys():
        if change[ele] != i:
            del dic[ele]
    return dic

def save_change(dic, changes):
    with open('memory3.txt', 'w+') as f:
        print('''
---------------------------------------
|   index   |   values   |   change   |
|-----------|------------|------------|''', file=f)
        for ele in dic.keys():
            print(f'|   {hex(ele)}   |     {f"0{dic[ele]}" if dic[ele] < 10 else dic[ele]}     |     {f"0{changes[ele]}" if changes[ele] < 10 else changes[ele]}     |' ,file=f)

def print_game_data():
    """
    print({'world' : pyboy.memory[CURRENT_WORLD],
        'Player_pos' : (pyboy.memory[PLAYER_X], pyboy.memory[PLAYER_Y]),
        'Shield_level' : pyboy.memory[SHIELD_LEVEL],
        'Max_arrow' : pyboy.memory[MAX_ARROWS],
        'Current_health' : pyboy.memory[CURRENT_HEALTH],
        'Number_of_bomb' : pyboy.memory[NUMBER_BOMBS],
        'Number_of_arrow' : pyboy.memory[NUMBER_ARROWS],
        'Item_held' : (pyboy.memory[HELD_ITEM_1], pyboy.memory[HELD_ITEM_2])
              })"""
    
    #print(pyboy.memory_scanner.scan_memory(10, start_addr=0x0000, end_addr=0xFFFF))
    print(pyboy.memory[0xDBB5], pyboy.memory[0xF415])
    '''
    with open('Monster2.state', 'wb') as f:
        pyboy.save_state(f)
    '''

# 0xDB55 - 0xDB9F : areas in the open world / related to the chunk
# 0xD900:0xDAFF - 0xDB3E:0xDB42 - 0xDB59 - 0xDB5F:0xDB60 - 0xDB6D:0xDB6E - 0xDB72:0xDB75 - 0xDB79:0xDB80 - 0xDB85:0xDB95 - 0xDBA0:0xDBA7 - 0xDBAB:0xDBAD : Nothing
# 0xDB4F:0xDB54 - 0xDB61:0xDB64 - 0xDB6F:0xDB71 - 0xDB80:0xDB84 - 0xDB96:0xDB9E - 0xDBA9:0xDAA - : IDK


# 0xd32e - 0xf32e - 0xe450 - 0xe120 - 0xe5ac - 0xc5ac - 0xc120 - 


def inp():
    curent = None
    if keyboard.is_pressed('z'):
        pyboy.button_press('up')
        curent = 'up'

    elif keyboard.is_pressed('s'):
        pyboy.button_press('down')
        curent = 'down'

    elif keyboard.is_pressed('q'):
        pyboy.button_press('left')
        curent = 'left'

    elif keyboard.is_pressed('d'):
        pyboy.button_press('right')
        curent = 'right'


    elif keyboard.is_pressed('a'):
        pyboy.button_press('a')
        curent = 'a'


    elif keyboard.is_pressed('e'):
        pyboy.button_press('b')
        curent = 'b'

    elif keyboard.is_pressed('v'):
        pyboy.button_press('start')
        curent = 'start'

    if keyboard.is_pressed('m'):
        print_game_data()

    return curent


with open('Monster.state', 'rb') as f:
    pyboy.load_state(f)

def get_levels_sum():
        poke_levels = [max(pyboy.memory[a] - 2, 0) for a in range(WORLD_STATUT[0], WORLD_STATUT[1]+1)]
        return max(sum(i > 0 for i in poke_levels) - 4, 0) 

def get_levels_reward(max_level_rew):
        explore_thresh = 22
        scale_factor = 4
        level_sum = get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        max_level_rew = max(max_level_rew, scaled)
        return max_level_rew


#dic = create()
#change = [0]*(2**16)
while True: # Use 'while True:' for infinite
    curent = inp()
    pyboy.tick()
    #dic, change = choose_algo(dic, change)
    #save_change(dic, change)
    print(pyboy.memory[0xff98], pyboy.memory[0xff99])# pyboy.memory[0xff98], pyboy.memory[0xffdb])
    if curent != None:
        pyboy.button_release(curent)
        curent = None


"""
dic = create()
change = [0]*(2**16)
for i in range(2):
    #pyboy.button_press('right')
    pyboy.tick()
    dic = choose_algo_2(dic)
    #save_change(dic, change)


pyboy.button_press('up')
pyboy.tick()
dic, change = choose_algo_3(dic, change)

pyboy.button_press('up')
pyboy.tick()
dic, change = choose_algo_3(dic, change)

pyboy.button_press('up')
pyboy.tick()
dic, change = choose_algo_3(dic, change)


dic = final(dic, change, 3)

save_change(dic, change)

"""





