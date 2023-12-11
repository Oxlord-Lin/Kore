"""抢劫对方无家可归的舰队"""

from kaggle_environments.envs.kore_fleets.helpers import *
# from action_helper.extra_helpers import *
# from action_helper.defend import *
# from action_helper.attack import *
# from action_helper.build import *
# from action_helper.mine import *

# from all_helper import *
from .all_helper import *


from random import randint
import itertools
import numpy as np
from random import choice, randint, randrange, sample, seed, random
import math

def rob_agent(obs, config):
    
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

    fleets = [fleet for _, fleet in board.fleets.items()]

    homeless_opp_fleet = []

    for fleet in fleets:
        if fleet.player != board.current_player and len(fleet.flight_plan) == 0: # 是敌方的舰队，且有可能无家可归
            position :Point = fleet.position
            x,y = position.x, position.y
            direction = fleet.direction.to_index()
            if direction % 2 == 0: # 如果沿南-北方向飞行
                homeless = True
                for shipyard in shipyards:
                    if shipyard.position.y == y:
                        homeless = False
                        break
                if homeless:
                    homeless_opp_fleet.append(fleet)
            if direction %2 == 1: # 如果沿东-西方向飞行
                homeless = True
                for shipyard in shipyards:
                    if shipyard.position.x == x:
                        homeless = False
                        break
                if homeless:
                    homeless_opp_fleet.append(fleet)

    ID = None

    if len(homeless_opp_fleet) > 0 and len(shipyards) > 0:
        homeless_opp_fleet = list(sorted(homeless_opp_fleet,key = lambda fleet: fleet.kore))
        fleet_to_rob :Fleet = homeless_opp_fleet[-1] # 携带最多矿石的fleet最值得抢劫

        fleet_size = fleet_to_rob.ship_count
        
        shipyards = list(sorted(shipyards,key = lambda shipyard: shipyard.ship_count))
        max_ship_shipyard = shipyards[-1] # 拥有最多舰队的船厂

        if max_ship_shipyard.ship_count > fleet_size + 40: # 如果适合抢劫
            mag_x = 1 if fleet_to_rob.position.x > max_ship_shipyard.position.x else -1
            abs_x = abs(fleet_to_rob.position.x - max_ship_shipyard.position.x)
            dir_x = mag_x if abs_x < size/2 else -mag_x
            mag_y = 1 if fleet_to_rob.position.y > max_ship_shipyard.position.y else -1
            abs_y = abs(fleet_to_rob.position.y - max_ship_shipyard.position.y)
            dir_y = mag_y if abs_y < size/2 else -mag_y
            flight_path_x = ""
            if abs_x > 0:
                flight_path_x += "E" if dir_x == 1 else "W"
                flight_path_x += str(abs_x - 1) if (abs_x - 1) > 0 else ""
            flight_path_y = ""
            if abs_y > 0:
                flight_path_y += "N" if dir_y == 1 else "S"
                flight_path_y += str(abs_y - 1) if (abs_y - 1) > 0 else ""
            
            if fleet_to_rob.direction.to_char() == 'N':
                flight_plan = flight_path_x + 'S9N9'
                flight_plan += 'W' if dir_x == 1 else 'E'
            
            elif fleet_to_rob.direction.to_char() == 'S':
                flight_plan = flight_path_x + 'N9S9'
                flight_plan += 'W' if dir_x == 1 else 'E'

            elif fleet_to_rob.direction.to_char() == 'W':
                flight_plan = flight_path_y + 'E9W9'
                flight_plan += 'S' if dir_y == 1 else 'N'

            elif fleet_to_rob.direction.to_char() == 'E':
                flight_plan = flight_path_y + 'W9E9'
                flight_plan += 'S' if dir_y == 1 else 'N'

            max_ship_shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(fleet_size + 20, flight_plan)

            ID = max_ship_shipyard.id

        else: # 如果不适合抢劫
            ID = None

    for shipyard in shipyards:

        if shipyard.id == ID:
            continue
        
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
