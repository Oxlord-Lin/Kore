"""权重硬编码"""




"""权重硬编码转换"""






"""所有的辅助函数都在这里"""

"""
extra-helper
"""
from kaggle_environments.envs.kore_fleets.helpers import *
from random import choice, randint, randrange, sample, seed, random

def get_col_row(size, pos):
    return pos % size, pos // size

def get_to_pos(size, pos, direction):
    col, row = get_col_row(size, pos)
    if direction == "NORTH":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "SOUTH":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "EAST":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "WEST":
        return pos - 1 if col > 0 else (row + 1) * size - 1
    
def get_shortest_flight_path_between(position_a, position_b, size, trailing_digits=False):
    mag_x = 1 if position_b.x > position_a.x else -1
    abs_x = abs(position_b.x - position_a.x)
    dir_x = mag_x if abs_x < size/2 else -mag_x
    mag_y = 1 if position_b.y > position_a.y else -1
    abs_y = abs(position_b.y - position_a.y)
    dir_y = mag_y if abs_y < size/2 else -mag_y
    flight_path_x = ""
    if abs_x > 0:
        flight_path_x += "E" if dir_x == 1 else "W"
        flight_path_x += str(abs_x - 1) if (abs_x - 1) > 0 else ""
    flight_path_y = ""
    if abs_y > 0:
        flight_path_y += "N" if dir_y == 1 else "S"
        flight_path_y += str(abs_y - 1) if (abs_y - 1) > 0 else ""
    if not len(flight_path_x) == len(flight_path_y):
        if len(flight_path_x) < len(flight_path_y):
            return flight_path_x + (flight_path_y if trailing_digits else flight_path_y[0])
        else:
            return flight_path_y + (flight_path_x if trailing_digits else flight_path_x[0])
    return flight_path_y + (flight_path_x if trailing_digits or not flight_path_x else flight_path_x[0]) if random() < .5 else flight_path_x + (flight_path_y if trailing_digits or not flight_path_y else flight_path_y[0])

def get_total_ships(board, player):
    ships = 0
    for fleet in board.fleets.values():
        if fleet.player_id == player:
            ships += fleet.ship_count
    for shipyard in board.shipyards.values():
        if shipyard.player_id == player:
            ships += shipyard.ship_count
    return ships    

# ref @egrehbbt 
def max_flight_plan_len_for_ship_count(ship_count):
    return math.floor(2 * math.log(ship_count)) + 1

# ref @egrehbbt 
def min_ship_count_for_flight_plan_len(flight_plan_len):
    return math.ceil(math.exp((flight_plan_len - 1) / 2))

# ref @egrehbbt 
def collection_rate_for_ship_count(ship_count):
    return min(math.log(ship_count) / 20, 0.99)

def spawn_ships(shipyard, remaining_kore, spawn_cost):
    return ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))


"""
attack
"""
# from kaggle_environments.envs.kore_fleets.helpers import *

def should_attack(board, shipyard, remaining_kore, spawn_cost, invading_fleet_size):
            closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, board.current_player)
            dist_to_closest_enemy_shipyard = 100 if not closest_enemy_shipyard else shipyard.position.distance_to(closest_enemy_shipyard.position, board.configuration.size)
            if (closest_enemy_shipyard 
                and (closest_enemy_shipyard.ship_count < 20 or dist_to_closest_enemy_shipyard < 15) 
                and (remaining_kore >= spawn_cost or shipyard.ship_count >= invading_fleet_size) 
                and (board.step > 300 or dist_to_closest_enemy_shipyard < 12)):
                return True
            return False

def get_closest_enemy_shipyard(board, position, me):
    min_dist = 1000000
    enemy_shipyard = None
    for shipyard in board.shipyards.values():
        if shipyard.player_id == me.id:
            continue
        dist = position.distance_to(shipyard.position, board.configuration.size)
        if dist < min_dist:
            min_dist = dist
            enemy_shipyard = shipyard
    return enemy_shipyard


"""
build
"""

# from kaggle_environments.envs.kore_fleets.helpers import *

def should_build(shipyard, remaining_kore):
    if remaining_kore > 500 and shipyard.max_spawn > 5:
        return True
    return False

def check_location(board, loc, me):
    if board.cells.get(loc).shipyard and board.cells.get(loc).shipyard.player.id == me.id:
        return 0
    kore = 0
    for i in range(-6, 7):
        for j in range(-6, 7):
            pos = loc.translate(Point(i, j), board.configuration.size)
            kore += board.cells.get(pos).kore or 0
    return kore

def build_new_shipyard(shipyard, board, me, convert_cost, search_radius=3):
    best_dir = 0
    best_kore = 0
    best_gap1 = 3
    best_gap2 = 3
    for i in range(4):
        next_dir = (i + 1) % 4
        for gap1 in range(0, search_radius, 1):
            for gap2 in range(0, search_radius, 1):
                enemy_shipyard_close = False
                diff1 = Direction.from_index(i).to_point() * gap1
                diff2 = Direction.from_index(next_dir).to_point() * gap2
                diff = diff1 + diff2
                pos = shipyard.position.translate(diff, board.configuration.size)
                for shipyard in board.shipyards.values():
                    if ((shipyard.player_id != me.id)
                        and (pos.distance_to(shipyard.position, board.configuration.size) < 6)):
                        enemy_shipyard_close = True
                if enemy_shipyard_close:
                    continue
                h = check_location(board, pos, me)
                if h > best_kore:
                    best_kore = h
                    best_gap1 = gap1
                    best_gap2 = gap2
                    best_dir = i
    gap1 = str(best_gap1)
    gap2 = str(best_gap2)
    next_dir = (best_dir + 1) % 4
    flight_plan = Direction.list_directions()[best_dir].to_char() + gap1
    flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
    flight_plan += "C"
    return ShipyardAction.launch_fleet_with_flight_plan(max(convert_cost + 50, int(shipyard.ship_count/2)), flight_plan)

"""
defend
"""


# from kaggle_environments.envs.kore_fleets.helpers import *

def should_defend(board, me, shipyard, radius=5):
    loc = shipyard.position
    for i in range(1-radius, radius):
        for j in range(1-radius, radius):
            pos = loc.translate(Point(i, j), board.configuration.size)
            if ((board.cells.get(pos).fleet is not None) 
                and (board.cells.get(pos).fleet.ship_count > 50)
                and (board.cells.get(pos).fleet.player_id!=me.id)
                and ((board.cells.get(pos).fleet.ship_count) > shipyard.ship_count)):
                return True               
    return False


"""
mine
"""
# from kaggle_environments.envs.kore_fleets.helpers import *

def should_mine(shipyard, best_fleet_size):
    if shipyard.ship_count >= best_fleet_size:
        return True
    return False

def check_path(board, start, dirs, dist_a, dist_b, collection_rate, L=False):
    kore = 0
    npv = .99
    current = start
    steps = 2 * (dist_a + dist_b + 2)
    for idx, d in enumerate(dirs):
        if L and idx==2:
            break
        for _ in range((dist_a if idx % 2 == 0 else dist_b) + 1):
            current = current.translate(d.to_point(), board.configuration.size)
            kore += int((board.cells.get(current).kore or 0) * collection_rate)
            final_kore = int((board.cells.get(current).kore or 0) * collection_rate)
    if L: kore = (kore) + (kore*(1-collection_rate)) - final_kore
    return math.pow(npv, steps) * kore / steps

def get_circular_flight_plan(gap1, gap2, start_dir):
    flight_plan = Direction.list_directions()[start_dir].to_char()
    if int(gap1):
        flight_plan += gap1
    next_dir = (start_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap2):
        flight_plan += gap2
    next_dir = (next_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap1):
        flight_plan += gap1
    next_dir = (next_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    return flight_plan

def get_L_flight_plan(gap1, gap2, start_dir):
    flight_plan = Direction.list_directions()[start_dir].to_char()
    if int(gap1):
        flight_plan += gap1
    next_dir = (start_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap2):
        flight_plan += gap2
    next_dir = (next_dir + 2) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap2):
        flight_plan += gap2
    next_dir = (next_dir - 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    return flight_plan

def get_rectangle_flight_plan(gap, start_dir):
    flight_plan = Direction.list_directions()[start_dir].to_char()
    next_dir = (start_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    if int(gap):
        flight_plan += gap
    next_dir = (next_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    next_dir = (next_dir + 1) % 4
    flight_plan += Direction.list_directions()[next_dir].to_char()
    return flight_plan

def check_flight_paths(board, shipyard, search_radius):
    best_h = 0
    best_gap1 = 1
    best_gap2 = 1
    best_dir = board.step % 4
    for i in range(4):
        dirs = Direction.list_directions()[i:] + Direction.list_directions()[:i]
        for gap1 in range(0, search_radius):
            for gap2 in range(0, search_radius):
                fleet_size = min_ship_count_for_flight_plan_len(7)
                h = check_path(board, shipyard.position, dirs, gap1, gap2, collection_rate_for_ship_count(fleet_size), L=False)
                if h/fleet_size > best_h:
                    best_h = h/fleet_size
                    best_flight_plan = get_circular_flight_plan(str(gap1), str(gap2), i)
                    best_fleet_size = fleet_size
                h = check_path(board, shipyard.position, dirs, gap1, gap2, collection_rate_for_ship_count(collection_rate_for_ship_count(fleet_size)), L=True)
                if h/fleet_size > best_h:
                    best_h = h/fleet_size
                    best_flight_plan = get_L_flight_plan(str(gap1), str(gap2), i)
                    best_fleet_size = fleet_size
                if gap1!=0:
                    continue
                fleet_size = min_ship_count_for_flight_plan_len(5)
                h = check_path(board, shipyard.position, dirs, gap1, gap2, collection_rate_for_ship_count(fleet_size), L=False)
                if h/fleet_size > best_h:
                    best_h = h/fleet_size
                    best_flight_plan = get_rectangle_flight_plan(str(gap2), i)
                    best_fleet_size = fleet_size    
    return best_fleet_size, best_flight_plan                   


"""attacker"""

from kaggle_environments.envs.kore_fleets.helpers import *
# from action_helper.extra_helpers import *
# from action_helper.defend import *
# from action_helper.attack import *
# from action_helper.build import *
# from action_helper.mine import *

# from action_helper.all_helper import *
# from all_helper import *

from random import randint
# import itertools
import numpy as np
from random import choice, randint, randrange, sample, seed, random
import math

def attacker_agent(obs, config):
    
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    turn = board.step
    
    invading_fleet_size = 50
    convert_cost_buffer = 80
    mining_search_radius = 10
    defence_radius = 7
    
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        
        best_fleet_size, best_flight_plan = check_flight_paths(board, shipyard, mining_search_radius) 
        
        # 优先考虑进攻

        if should_attack(board, shipyard, remaining_kore, spawn_cost, invading_fleet_size):
            if shipyard.ship_count >= invading_fleet_size:
                    closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, board.current_player)
                    flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(invading_fleet_size, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)

        elif should_defend(board, me, shipyard, defence_radius):
            if remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)

        elif should_build(shipyard, remaining_kore):
            if shipyard.ship_count >= convert_cost + convert_cost_buffer:
                shipyard.next_action = build_new_shipyard(shipyard, board, me, convert_cost)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                
        elif should_mine(shipyard, best_fleet_size):
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(best_fleet_size, best_flight_plan)
        
        elif (remaining_kore > spawn_cost):
            shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
        
        elif (len(me.fleet_ids) == 0 and shipyard.ship_count <= 22) and len(shipyards)==1:
            if remaining_kore > 11:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
            else:
                direction = Direction.NORTH
                if shipyard.ship_count > 0:
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, direction.to_char())
                
    return me.next_actions

"""balanced"""

import math
from random import random, sample, randint

# from kaggle_environments import utils
from kaggle_environments.helpers import Point, Direction
from kaggle_environments.envs.kore_fleets.helpers import Board, ShipyardAction

# checks a path to see how profitable it is, using net present value to discount 
# the return time
# def check_path(board, start, dirs, dist_a, dist_b, collection_rate):
#     kore = 0
#     npv = .98
#     current = start
#     steps = 2 * (dist_a + dist_b + 2)
#     for idx, d in enumerate(dirs):
#         for _ in range((dist_a if idx % 2 == 0 else dist_b) + 1):
#             current = current.translate(d.to_point(), board.configuration.size)
#             kore += int((board.cells.get(current).kore or 0) * collection_rate)
#     return math.pow(npv, steps) * kore / (2 * (dist_a + dist_b + 2))

# used to see how much kore is around a spot to potentially put a new shipyard
def check_location(board, loc, me):
    if board.cells.get(loc).shipyard and board.cells.get(loc).shipyard.player.id == me.id:
        return 0
    kore = 0
    for i in range(-3, 4):
        for j in range(-3, 4):
            pos = loc.translate(Point(i, j), board.configuration.size)
            kore += board.cells.get(pos).kore or 0
    return kore

def get_closest_enemy_shipyard(board, position, me):
    min_dist = 1000000
    enemy_shipyard = None
    for shipyard in board.shipyards.values():
        if shipyard.player_id == me.id:
            continue
        dist = position.distance_to(shipyard.position, board.configuration.size)
        if dist < min_dist:
            min_dist = dist
            enemy_shipyard = shipyard
    return enemy_shipyard
    
def get_shortest_flight_path_between(position_a, position_b, size, trailing_digits=False):
    mag_x = 1 if position_b.x > position_a.x else -1
    abs_x = abs(position_b.x - position_a.x)
    dir_x = mag_x if abs_x < size/2 else -mag_x
    mag_y = 1 if position_b.y > position_a.y else -1
    abs_y = abs(position_b.y - position_a.y)
    dir_y = mag_y if abs_y < size/2 else -mag_y
    flight_path_x = ""
    if abs_x > 0:
        flight_path_x += "E" if dir_x == 1 else "W"
        flight_path_x += str(abs_x - 1) if (abs_x - 1) > 0 else ""
    flight_path_y = ""
    if abs_y > 0:
        flight_path_y += "N" if dir_y == 1 else "S"
        flight_path_y += str(abs_y - 1) if (abs_y - 1) > 0 else ""
    if not len(flight_path_x) == len(flight_path_y):
        if len(flight_path_x) < len(flight_path_y):
            return flight_path_x + (flight_path_y if trailing_digits else flight_path_y[0])
        else:
            return flight_path_y + (flight_path_x if trailing_digits else flight_path_x[0])
    return flight_path_y + (flight_path_x if trailing_digits or not flight_path_x else flight_path_x[0]) if random() < .5 else flight_path_x + (flight_path_y if trailing_digits or not flight_path_y else flight_path_y[0])


def balanced_agent(obs, config):
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost

    # randomize shipyard order
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)
        invading_fleet_size = 100
        dist_to_closest_enemy_shipyard = 100 if not closest_enemy_shipyard else shipyard.position.distance_to(closest_enemy_shipyard.position, size)
        if closest_enemy_shipyard and (closest_enemy_shipyard.ship_count < 20 or dist_to_closest_enemy_shipyard < 15) and (remaining_kore >= spawn_cost or shipyard.ship_count >= invading_fleet_size):
            if shipyard.ship_count >= invading_fleet_size:
                flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(invading_fleet_size, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))

        elif remaining_kore > 500 and shipyard.max_spawn > 5:
            if shipyard.ship_count >= convert_cost + 20:
                start_dir = randint(0, 3)
                next_dir = (start_dir + 1) % 4
                best_kore = 0
                best_gap1 = 0
                best_gap2 = 0
                for gap1 in range(2, 4):
                    for gap2 in range(2, 4):
                        # gap2 = randint(3, 9)
                        diff1 = Direction.from_index(start_dir).to_point() * gap1
                        diff2 = Direction.from_index(next_dir).to_point() * gap2
                        diff = diff1 + diff2
                        pos = shipyard.position.translate(diff, board.configuration.size)
                        h = check_location(board, pos, me)
                        if h > best_kore:
                            best_kore = h
                            best_gap1 = gap1
                            best_gap2 = gap2
                gap1 = str(best_gap1)
                gap2 = str(best_gap2)
                flight_plan = Direction.list_directions()[start_dir].to_char() + gap1
                flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
                flight_plan += "C"
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(max(convert_cost + 20, int(shipyard.ship_count/2)), flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))

        # launch a large fleet if able
        elif shipyard.ship_count >= 21:
            best_h = 0
            best_gap1 = 5
            best_gap2 = 5
            start_dir = board.step % 4
            dirs = Direction.list_directions()[start_dir:] + Direction.list_directions()[:start_dir]
            for gap1 in range(0, 10):
                for gap2 in range(0, 10):
                    h = check_path(board, shipyard.position, dirs, gap1, gap2, .2)
                    if h > best_h:
                        best_h = h
                        best_gap1 = gap1
                        best_gap2 = gap2
            gap1 = str(best_gap1)
            gap2 = str(best_gap2)
            flight_plan = Direction.list_directions()[start_dir].to_char()
            if int(gap1):
                flight_plan += gap1
            next_dir = (start_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            if int(gap2):
                flight_plan += gap2
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            if int(gap1):
                flight_plan += gap1
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(21, flight_plan)
        # else spawn if possible
        elif remaining_kore > board.configuration.spawn_cost * shipyard.max_spawn:
            remaining_kore -= board.configuration.spawn_cost
            if remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))
    return me.next_actions

"""defenser"""

from kaggle_environments.envs.kore_fleets.helpers import *
# from action_helper.extra_helpers import *
# from action_helper.defend import *
# from action_helper.attack import *
# from action_helper.build import *
# from action_helper.mine import *

# from all_helper import *
# from all_helper import *


from random import randint
# import itertools
import numpy as np
from random import choice, randint, randrange, sample, seed, random
import math

def defense_agent(obs, config):
    
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    turn = board.step
    
    invading_fleet_size = 50
    convert_cost_buffer = 80
    mining_search_radius = 10
    defence_radius = 7
    
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        
        best_fleet_size, best_flight_plan = check_flight_paths(board, shipyard, mining_search_radius) 
        
        if should_defend(board, me, shipyard, defence_radius):
            if remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                
        elif should_attack(board, shipyard, remaining_kore, spawn_cost, invading_fleet_size):
            if shipyard.ship_count >= invading_fleet_size:
                    closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, board.current_player)
                    flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(invading_fleet_size, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)

        elif should_build(shipyard, remaining_kore):
            if shipyard.ship_count >= convert_cost + convert_cost_buffer:
                shipyard.next_action = build_new_shipyard(shipyard, board, me, convert_cost)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                
        elif should_mine(shipyard, best_fleet_size):
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(best_fleet_size, best_flight_plan)
        
        elif (remaining_kore > spawn_cost):
            shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
        
        elif (len(me.fleet_ids) == 0 and shipyard.ship_count <= 22) and len(shipyards)==1:
            if remaining_kore > 11:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
            else:
                direction = Direction.NORTH
                if shipyard.ship_count > 0:
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, direction.to_char())
                
    return me.next_actions

"""expander"""

from kaggle_environments.envs.kore_fleets.helpers import *
# from action_helper.extra_helpers import *
# from action_helper.defend import *
# from action_helper.attack import *
# from action_helper.build import *
# from action_helper.mine import *


# from all_helper import *
# from action_helper.all_helper import *
# from all_helper import *


from random import randint
# import itertools
import numpy as np
from random import choice, randint, randrange, sample, seed, random
import math

def expand_agent(obs, config):
    
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    turn = board.step
    
    invading_fleet_size = 50
    convert_cost_buffer = 80
    mining_search_radius = 10
    defence_radius = 7
    
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        
        best_fleet_size, best_flight_plan = check_flight_paths(board, shipyard, mining_search_radius) 
        
        # 优先考虑新建船厂

        if should_build(shipyard, remaining_kore):
            if shipyard.ship_count >= convert_cost + convert_cost_buffer:
                shipyard.next_action = build_new_shipyard(shipyard, board, me, convert_cost)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)

        elif should_defend(board, me, shipyard, defence_radius):
            if remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                
        elif should_attack(board, shipyard, remaining_kore, spawn_cost, invading_fleet_size):
            if shipyard.ship_count >= invading_fleet_size:
                    closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, board.current_player)
                    flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(invading_fleet_size, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)

        elif should_mine(shipyard, best_fleet_size):
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(best_fleet_size, best_flight_plan)
        
        elif (remaining_kore > spawn_cost):
            shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
        
        elif (len(me.fleet_ids) == 0 and shipyard.ship_count <= 22) and len(shipyards)==1:
            if remaining_kore > 11:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
            else:
                direction = Direction.NORTH
                if shipyard.ship_count > 0:
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, direction.to_char())
                
    return me.next_actions

"""miner"""

from kaggle_environments.envs.kore_fleets.helpers import *
# from action_helper.extra_helpers import *
# from action_helper.defend import *
# from action_helper.attack import *
# from action_helper.build import *
# from action_helper.mine import *


# from all_helper import *
# from action_helper.all_helper import *
# from all_helper import *

from random import randint
# import itertools
import numpy as np
from random import choice, randint, randrange, sample, seed, random
import math

def miner_agent(obs, config):
    
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    turn = board.step
    
    invading_fleet_size = 50
    convert_cost_buffer = 80
    mining_search_radius = 10
    defence_radius = 7
    
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        
        best_fleet_size, best_flight_plan = check_flight_paths(board, shipyard, mining_search_radius) 

        # 优先考虑挖矿

        if should_mine(shipyard, best_fleet_size):
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(best_fleet_size, best_flight_plan)      

        elif should_defend(board, me, shipyard, defence_radius):
            if remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                
        elif should_attack(board, shipyard, remaining_kore, spawn_cost, invading_fleet_size):
            if shipyard.ship_count >= invading_fleet_size:
                    closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, board.current_player)
                    flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(invading_fleet_size, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)

        elif should_build(shipyard, remaining_kore):
            if shipyard.ship_count >= convert_cost + convert_cost_buffer:
                shipyard.next_action = build_new_shipyard(shipyard, board, me, convert_cost)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                

        
        elif (remaining_kore > spawn_cost):
            shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
        
        elif (len(me.fleet_ids) == 0 and shipyard.ship_count <= 22) and len(shipyards)==1:
            if remaining_kore > 11:
                shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
            else:
                direction = Direction.NORTH
                if shipyard.ship_count > 0:
                    shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, direction.to_char())
                
    return me.next_actions






# Imports
# from typing import Union
from kaggle_environments.envs.kore_fleets.helpers import (
    Board,
    Configuration,
    Observation,
    ShipyardAction,
    Direction,
    Point,
    Fleet
)

import numpy as np
import tensorflow as tf
import keras
from keras import layers

num_actions = 5
def create_q_model():
    global num_actions
    return keras.models.Sequential(
        [
            layers.Input(shape=(21, 21, 7)),
            layers.Conv2D(64, 8),
            layers.Activation("linear"),
            layers.Conv2D(128, 10),
            layers.Activation("linear"),
            layers.Flatten(),
            layers.Dense(64),
            layers.Activation("sigmoid"),
            layers.Dense(num_actions),
            layers.Activation("linear")
        ]
    )


global model
model = create_q_model()
model.load_weights('my_weights_v3.h5') # 这个相对路径不知道该咋写
# path_prefix = '/kaggle_simulations/agent/'
# model.load_weights('/kaggle_simulations/agent/my_weights_v3.h5')
print('成功导入权重！')






def map_value(value, enemy: bool = False,) -> float:
    """
    Helper function for build_observation. For this to work, we must
    assume the following:
        - The maximum value of Kore in a single square is expected to be
            500.
        - The maximum fleet size is expected to be 1000.
        - The maximum amount of kore a fleet can carry is expected to be
            5000.
    Maps a value to a range of [-1, 1] if enemy, or [0, 1] otherwise.
    """
    MAX_NATURAL_KORE_IN_SQUARE = 500
    MAX_ASSUMED_FLEET_SIZE = 1000
    MAX_ASSUMED_KORE_IN_FLEET = 5000
    max_value = float(
        max(
            MAX_NATURAL_KORE_IN_SQUARE,
            MAX_ASSUMED_FLEET_SIZE,
            MAX_ASSUMED_KORE_IN_FLEET,
        )
    )
    val = value / max_value
    if enemy:
        return -val
    return val


def build_observation(raw_observation: Observation, env_configuration) -> np.ndarray:
    """
    一共七个特征：

    1. 所有位置的天然矿石含量
    2. 每个位置上fleet的规模（即包含船只的数量），用正数表示我方fleet，用负数表示敌方fleet
    3. 所有舰队当前位置，用+1表示我方舰队所在位置，用-1表示敌方舰队所在位置
    4. 每个舰队所携带的kore矿石数量，全部按正数记
    5. 所有船厂位置，用+1表示我方船厂所在位置，用-1表示敌方船厂所在位置
    6. 敌方所有fleet下一步的位置，用-1表示
    7. 我方所有fleet下一步的位置，用+1表示

    以上七个打包成(21,21,7)的张量输入神经网络
    """

    # Build the Board object that will help us build the layers
    board = Board(raw_observation, env_configuration) 

    """
    # 第一层：记录所有位置的矿石的数量（不包含fleet上的矿石），并映射到[0,1]区间
    """
    kore_layer = np.array(raw_observation.kore).reshape(
        env_configuration.size, env_configuration.size   
    )
    for i in range(kore_layer.shape[0]):
        for j in range(kore_layer.shape[1]):
            kore_layer[i,j] = map_value(kore_layer[i,j])


    """
    # 第二层：记录双方所有位置上fleet的数量（如果是敌人则是负数），包含还在船厂里没放出来的fleet
    """
    fleet_num_layer = np.zeros((env_configuration.size, env_configuration.size)
    )
    # - Get fleets and shipyards on the map
    fleets = [fleet for _, fleet in board.fleets.items()]
    shipyards = [shipyard for _, shipyard in board.shipyards.items()]
    # - Iterate over fleets, getting its position and size
    for fleet in fleets:
        # - Get the position of the fleet
        position = fleet.position
        x, y = position.x, position.y
        # - Get the size of the fleet
        size = fleet.ship_count  # 舰队规模
        # - Check if the fleet is an enemy fleet
        if fleet.player != board.current_player:
            multilpier = -1
        else:
            multilpier = 1
        # - Set the fleet layer to the size of the fleet
        fleet_num_layer[x, y] = multilpier * map_value(size)
    # - Iterate over shipyards, getting its position and size
    for shipyard in shipyards:
        # - Get the position of the shipyard
        position = shipyard.position
        x, y = position.x, position.y
        # - Get the size of the shipyard
        size = shipyard.ship_count  # 船厂里的船只数量
        # - Check if the shipyard is an enemy shipyard
        if shipyard.player != board.current_player:
            multilpier = -1
        else:
            multilpier = 1
        # - Set the fleet layer to the size of the shipyard
        fleet_num_layer[x, y] = multilpier * map_value(size)

    
    """
    # 第三层：记录所有舰队当前的位置，区分敌(-1)我(+1)
    """
    # Building the enemy positions layer
    fleet_position_layer = np.zeros(
        (env_configuration.size, env_configuration.size)
    )
    # - Iterate over fleets
    for fleet in fleets:
        if fleet.player == board.current_player:
            position = fleet.position  # Get the position of the fleet
            x, y = position.x, position.y
            fleet_position_layer[x, y] = 1
        else:
            position = fleet.position
            x, y = position.x, position.y
            fleet_position_layer[x, y] = -1


    """
    # 第四层：建立所有位置上由fleet携带的矿石数量（不区分是否是敌人的舰队），并映射到[0,1]区间
    """
    # Building the layer of the kore carried by fleet
    kore_carried_by_fleet_layer = np.zeros(
        (env_configuration.size, env_configuration.size)
    )
    # - Iterate over fleets
    for fleet in fleets:
        # - Get the position of the fleet
        position = fleet.position
        x, y = position.x, position.y
        # - Get the amount of kore the fleet is carrying
        kore = fleet.kore
        # - Set the kore layer to the amount of kore
        kore_carried_by_fleet_layer[x, y] = map_value(kore)

    """
    第五层：所有船厂的位置以及每一回合最多能生产的ship数量（越大则说明这个船厂被控制了越久），区分敌(-1)我(+1)
    """
    shipyard_position_layer = np.zeros(
        (env_configuration.size, env_configuration.size)
    )
    for shipyard in shipyards:
        # - Get the position of the shipyard
        position = shipyard.position
        x, y = position.x, position.y
        # - Get the size of the shipyard
        # - Check if the shipyard is an enemy shipyard
        if shipyard.player != board.current_player:
            shipyard_position_layer[x,y] = -1 * shipyard.max_spawn
        else:
            shipyard_position_layer[x,y] = +1 * shipyard.max_spawn

    

    def next_positioin(fleet:Fleet,size:int = 21):
        """计算fleet的下一个位置"""
        position :Point = fleet.position
        # x,y = position.x, position.y
        direction = fleet.direction.to_char()
        flight_plan = fleet.flight_plan
        if len(flight_plan) == 0 :
            move = direction
        elif flight_plan[0] not in ('N','S','W','E'):
            move = direction
        else:
            move = flight_plan[0]
        move = Direction.from_char(move)
        move = move.to_point()
        next_positioin = position.translate(move,size=size)
        return next_positioin
    
    
    """
    第六层：对手所有fleet下一步的位置，用 -1 表示
    """
    enemy_fleet_next_postion_layer = np.zeros(
        (env_configuration.size, env_configuration.size)
    )
    for fleet in fleets:
        if fleet.player != board.current_player:  # 如果是敌方的fleet
            next_pos = next_positioin(fleet, size=env_configuration.size)
            x,y = next_pos.x, next_pos.y
            enemy_fleet_next_postion_layer[x,y] = -1

    """
    第七层：我方所有fleet下一步的位置，用 +1 表示
    """
    my_fleet_next_postion_layer = np.zeros(
        (env_configuration.size, env_configuration.size)
    )
    for fleet in fleets:
        if fleet.player == board.current_player:  # 如果是我方的fleet
            next_pos = next_positioin(fleet, size=env_configuration.size)
            x,y = next_pos.x, next_pos.y
            my_fleet_next_postion_layer[x,y] = +1


    # Building our observation box
    observation = np.zeros(
        (env_configuration.size, env_configuration.size, 7)
    )
    observation[:, :, 0] = kore_layer
    observation[:, :, 1] = fleet_num_layer
    observation[:, :, 2] = fleet_position_layer
    observation[:, :, 3] = kore_carried_by_fleet_layer
    observation[:, :, 4] = shipyard_position_layer
    observation[:, :, 5] = enemy_fleet_next_postion_layer
    observation[:, :, 6] = my_fleet_next_postion_layer

    with open('states.txt','a+') as f:
        f.write('回合数：'+str(board.step)+'\n')
        f.write(str(list(observation.flatten())))
        f.write('\n')


    return observation

# from  balanced import balanced_agent
# from  attacker import attacker_agent
# from  miner import miner_agent
# from  expander import expand_agent
# from  defenser import defense_agent


def DQN_agent(raw_observation,config):
    """第二阶段AI"""
    global model
    observation = build_observation(raw_observation,config)
    observation_tensor = tf.convert_to_tensor(observation)
    observation_tensor = tf.expand_dims(observation_tensor,0)
    actions = model(observation_tensor,training=False) 
    action_type = int(np.argmax(actions))
    # print(action_type)
    
    if action_type == 0:
        action = balanced_agent(raw_observation,config)
    elif action_type == 1:
        action = attacker_agent(raw_observation,config)
    elif action_type == 2:
        action = miner_agent(raw_observation,config)
    elif action_type == 3:
        action = expand_agent(raw_observation,config)
    elif action_type == 4:
        action = defense_agent(raw_observation,config)

    return action