"""主动拦截对我方有高威胁性的敌方舰队"""
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

def defend_dangerous_fleet_agent(obs, config):
    
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

    opponent = board.opponents[0]
    opp_shipyards = opponent.shipyards
    opp_fleets = opponent.fleets

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


    # for fleet in opp_fleets:
    #     pos = fleet.position # 敌方舰队当前位置
    #     opp_x, opp_y = pos.x, pos.y
    #     next_pos = next_positioin(fleet, size) # 敌方舰队下一步位置
    #     for my_pos in my_shipyard_position:
    #         if next_pos.distance_to(my_pos,size) <=1:


    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:

        x, y = shipyard.position.x, shipyard.position.y

        for fleet in opp_fleets:
            pos = fleet.position # 敌方舰队当前位置
            opp_x, opp_y = pos.x, pos.y
            next_pos = next_positioin(fleet, size) # 敌方舰队下一步位置

            if next_pos.distance_to(shipyard.position,size) <= 1: # 说明是一个高威胁性的fleet
                if shipyard.ship_count >= fleet.ship_count - 1 : # 我方船厂很强壮，不会被对方轻易攻占
                    shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                else: # 我方船厂比较弱，很可能被对方fleet攻占
                    if shipyard.max_spawn > 4: 
                        # 说明这个船厂被我方占领比较久，现在没有船，可能是因为都派出去采矿了，
                        # 等那些船回来之后就有可能重新夺回这个船厂，因此当前的策略是尽可能地再制造一些船，消耗对方的fleet
                        # 等待援兵回来
                        shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                    else: # 说明这是一个控制不是很久的船厂，可以暂时放弃，君子报仇十年不晚
                        if shipyard.ship_count > 0: # 如果还有船
                            if len(shipyards) >= 2: # 如果我方还有其他船厂，就转移到另一个最近的船厂
                                closest_my_shipyard  = get_closest_my_shipyard(board,shipyard.position,me)
                                flight_plan = get_shortest_flight_path_between(shipyard.position, closest_my_shipyard.position, size)
                                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, flight_plan)
                            else: # 我方没有其他船厂了！那就攻击敌方最弱的船厂！
                                opp_shipyards = list(sorted(opp_shipyards, key = lambda shipyard : shipyard.max_spawn))
                                weakest_opp_shipyard = list(sorted(opp_shipyards[:3], key = lambda shipyard : shipyard.ship_count))[0] # 对手最弱的船厂
                                flight_plan = get_shortest_flight_path_between(shipyard.position, weakest_opp_shipyard.position, size)
                                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, flight_plan)
                        else:
                            shipyard.next_action = spawn_ships(shipyard, remaining_kore, spawn_cost)
                            
        # 如果没有高威胁性的敌方舰队，则转入attacker的战略
        best_fleet_size, best_flight_plan = check_flight_paths(board, shipyard, mining_search_radius) 
        
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
