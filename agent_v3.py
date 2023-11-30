# Imports
from typing import Union
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
# model.load_weights('my_weights_v3') # 这个相对路径不知道该咋写
model.load_weights('my_weights_v3.h5')
# model.load_weights('./weight/my_weights_v3.h5')


def map_value(value: Union[int, float], enemy: bool = False,) -> float:
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

    return observation

from  utils.balanced import balanced_agent
from  utils.attacker import attacker_agent
from  utils.miner import miner_agent
from  utils.expander import expand_agent
from  utils.defenser import defense_agent


def DQN_agent(raw_observation,config):
    """第三个训练出来的AI，还不知道水平如何"""
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