
from random import random, sample
import math
from kaggle_environments import utils
from kaggle_environments.helpers import Point, Direction
from kaggle_environments.envs.kore_fleets.helpers import Board, ShipyardAction

def check_path(board, start, dirs, dist_a, dist_b, collection_rate):
    kore = 0
    npv = .98
    current = start
    steps = 2 * (dist_a + dist_b + 2)
    for idx, d in enumerate(dirs):
        for _ in range((dist_a if idx % 2 == 0 else dist_b) + 1):
            current = current.translate(d.to_point(), board.configuration.size)
            kore += int((board.cells.get(current).kore or 0) * collection_rate)
    return math.pow(npv, steps) * kore / (2 * (dist_a + dist_b + 2))

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

def attacker_agent(obs, config):
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    # randomize shipyard order
    shipyards = sample(shipyards, len(shipyards))
    for idx, shipyard in enumerate(shipyards):
        closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)
        # if we see an enemy shipyard, build ships and attack it!
        if closest_enemy_shipyard and (remaining_kore >= spawn_cost or shipyard.ship_count >= 50):
            if shipyard.ship_count >= 50:
                flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(50, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))
        # if we have run out of kore, do some mining
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
        # if we have multiple shipyards, send our ships back and forth between them
        elif shipyard.ship_count > 0 and len(shipyards) > 1:
            flight_plan = get_shortest_flight_path_between(shipyard.position, shipyards[(idx + 1) % len(shipyards)].position, size)
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, flight_plan)
            
    return me.next_actions

