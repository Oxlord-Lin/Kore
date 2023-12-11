"""群起而攻之"""

from kaggle_environments.envs.kore_fleets.helpers import *
from .all_helper import *

from random import randint
import itertools
import numpy as np
from random import choice, randint, randrange, sample, seed, random
import math

def siege_agent(obs, config):
    
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    turn = board.step
    opponent = board.opponents[0]
    opp_shipyards = opponent.shipyards
    opp_fleets = opponent.fleets
    
    invading_fleet_size = 50
    convert_cost_buffer = 80
    mining_search_radius = 10
    defence_radius = 7


    """优先考虑围攻对方最弱的船厂
    最弱的定义方式：找出敌方造船能力最低的2个船厂（也即控制的时间最短），
    在其中选择拥有船只数量最少的船厂作为攻击对象"""
    my_shipyards = me.shipyards
    if len(my_shipyards) >= 2 and len(opp_shipyards) > 0 and len(my_shipyards) >= len(opp_shipyards):
        opp_shipyards = list(sorted(opp_shipyards, key = lambda shipyard : shipyard.max_spawn))
        weakest_opp_shipyard = list(sorted(opp_shipyards[:2], key = lambda shipyard : shipyard.ship_count))[0] # 对手最弱的船厂
        
        for shipyard in my_shipyards:
            flight_plan = get_shortest_flight_path_between(shipyard.position, weakest_opp_shipyard.position, size)
            if shipyard.ship_count >= 35:
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(35, flight_plan)
            
            else: # 如果该船厂不满足围攻条件，则该船厂转入defenser策略
        
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
    
    else: # 如果不满足围攻条件，则转入defenser策略
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
