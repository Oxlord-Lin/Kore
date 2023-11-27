
import math
from random import random, sample, randint

from kaggle_environments import utils
from kaggle_environments.helpers import Point, Direction
from kaggle_environments.envs.kore_fleets.helpers import Board, ShipyardAction

# checks a path to see how profitable it is, using net present value to discount 
# the return time
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
