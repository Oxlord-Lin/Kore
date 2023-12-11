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


def get_closest_my_shipyard(board:Board, position, me):
    min_dist = 1000000
    my_shipyard = None
    for shipyard in board.shipyards.values():
        if shipyard.player_id != me.id: # 如果是敌方船厂，则跳过
            continue
        dist = position.distance_to(shipyard.position, board.configuration.size)
        if 0 < dist and dist < min_dist: # 0 < dist 不能是当前船厂
            min_dist = dist
            my_shipyard = shipyard
    return my_shipyard


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
