CURRENT_WORLD = 0xD401 # if in houses : 0 else : 1 or more
PLAYER_X = 0xD404 # must check
PLAYER_Y = 0xD405 # must check
GREAT_PLAYER_X = 0xFF98 
GREAT_PLAYER_Y = 0xFF99 
LOCATION = [0xD404, 0xD405]
CURRENT_LOADED_MAP = [0xD700, 0xD79B]
WORLD_STATUT = [0xD800, 0xD8FF] # * min - max In reality idk, It seems that when you discoverd a new location it's setting it to 128
HELD_ITEM_1 = 0xDB00 # Refered to the item's index list
HELD_ITEM_2 = 0xDB01 # Refered to the item's index list
INVENTORY = [0xDB02, 0xDB0B]
NUMBER_BOMBS = 0xDB4D # Seems to work
SHIELD_LEVEL = 0xDB44 # 0 if we don't take it, 1 else and 2 if hold.
SWORD_LEVEL = 0xDB4E # Same as the shield
NUMBER_ARROWS = 0xDB45 # Seems to work
MAX_BOMB = 0xDB77 # Seems to work
MAX_ARROWS = 0xDB78 # Seems to work
CURRENT_HEALTH = 0xDB5A # Works in a scale of 24, so each heart represent 6 hp
KILLED_MONSTERS = [0xDBB5, 0xF415] # Finally found!!!!!
NUMBERR_DEATH = 0xDB56 # ! TBD


"""
Items list:

------------------------------
|   item           |    id    |
|------------------|----------|
|  Sword           |    01    |
|  Bombs           |    02    |
|  Power Bracelet  |    03    |
|  Shield          |    04    |
|  Bow             |    05    |
|  HookShot        |    06    |
|  Fire Rod        |    07    |
|  Pegasus boots   |    08    |
|  Ocarina         |    09    |
|  Feather         |    0A    |
|  Shovel          |    0B    |
|  Magic Powder    |    0C    |
|  Boomrang        |    0D    |
------------------------------

"""