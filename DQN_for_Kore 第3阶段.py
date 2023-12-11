# Imports
import random
from typing import Union

from gym import Env, spaces
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import (
    Board,
    Configuration,
    Observation,
    ShipyardAction,
    Direction,
    Point,
    Fleet,
    Player
)

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils_v3.reward_utils import get_board_value


"""
# 参数区（可以忽略）
"""

# Configuration paramaters 
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = epsilon  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 410 # 这个参数不用管，没啥用，因为我们设置了遇到done就自动结束
# Number of frames to take random action and observe output
epsilon_random_frames = 8000
# Number of frames for exploration
epsilon_greedy_frames = 60000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 200000
# Train the model after 4 actions
update_after_actions = 4 # 更新Q网络的频率
# How often to update the target network
update_target_network = 1000 # 更新Q-target网络的频率
# 动作空间维度
num_actions = 10


"""
# build the custom environments where we can train our agent
"""

class CustomKoreEnv(Env): # 继承游戏环境，增加一些适用于Qlearning的方法
    """
    This is a custom Kore environment, adapted for use with
    OpenAI Gym. We don't actually need to use OpenAI Gym in
    this example, but I've chosen to do it for the sake of
    compatibility for those who already have code for it.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, agent2="random"):
        super().__init__()
        # Initialize the actual environment
        kore_env = make("kore_fleets", debug=True)
        
        from utils_v3.real_balanced import balanced_agent
        from agent_v1 import DQN_agent as DQN_agent_1 # 喜欢扩张领地和采矿
        from utils_v3.attacker import attacker_agent
        from utils_v3.miner import miner_agent
        from utils_v3.expander import expand_agent
        from utils_v3.defenser import defense_agent
        from agent_v2 import DQN_agent as DQN_agent_2  # 喜欢主动攻击别人的船厂
        from agent_v3 import DQN_agent as DQN_agent_3 # 第二阶段训练成果，kaggle得分较高

        # opps = [balanced_agent,DQN_agent,attacker_agent,miner_agent,expand_agent,defense_agent,DQN_agent_2]
        # opps = [balanced_agent,DQN_agent,DQN_agent_2]
        
        # opp_index = np.random.choice(3, p=[0.8, 0.1, 0.1])
        # opp_index = np.random.choice(7, p=[0.4, 0.04, 0.04, 0.04, 0.04, 0.4, 0.04])

        # opps = [balanced_agent, defense_agent, DQN_agent_1, DQN_agent_2, DQN_agent_3]
        opps = [balanced_agent, defense_agent, DQN_agent_3]

        # opp_index = np.random.choice(5,p=[0.4,0.3,0.05,0.05,0.2])
        opp_index = np.random.choice(3,p=[0.35,0.3,0.35])
        
        agent2 =opps[opp_index]
        
        self.env = kore_env.train([None, agent2])
        # self.env = kore_env.train([None, balanced_agent])

        # if random.random() < 0.5:
        #     self.env = kore_env.train([None, agent2])
        # else:
        #     self.env = kore_env.train([agent2,None])

        self.env_configuration: Configuration = kore_env.configuration
        map_size = self.env_configuration.size
        self.board: Board = None

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(map_size, map_size, 7), dtype=np.float64
        )

        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float64)

    def map_value(self, value: Union[int, float], enemy: bool = False,) -> float:
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

    def build_observation(self, raw_observation: Observation) -> np.ndarray:
        """
        一共12个特征：

        1. 所有位置的天然矿石含量
        2. 每个位置上fleet的规模（即包含船只的数量），用正数表示我方fleet，用负数表示敌方fleet
        3. 所有舰队当前位置，用+1表示我方舰队所在位置，用-1表示敌方舰队所在位置
        4. 每个舰队所携带的kore矿石数量，全部按正数记
        5. 所有船厂位置，用+1表示我方船厂所在位置，用-1表示敌方船厂所在位置
        6. 敌方所有fleet一步后的位置，用-1表示
        7. 我方所有fleet一步后的位置，用+1表示
        8. 敌方高威胁性船厂
        9. 敌方高威胁性舰队
        10.我方无家可归的舰队
        11.敌方无家可归的舰队
        12.敌方试图攻击我方船厂的舰队

        以上12个特征，打包成(21,21,12)的张量输入神经网络
        """

        # Build the Board object that will help us build the layers
        board = Board(raw_observation, self.env_configuration) 

        """
        # 第一层：记录所有位置的矿石的数量（不包含fleet上的矿石），并映射到[0,1]区间
        """
        kore_layer = np.array(raw_observation.kore).reshape(
            self.env_configuration.size, self.env_configuration.size   
        )
        for i in range(kore_layer.shape[0]):
            for j in range(kore_layer.shape[1]):
                kore_layer[i,j] = self.map_value(kore_layer[i,j])


        """
        # 第二层：记录双方所有位置上fleet的数量（如果是敌人则是负数），包含还在船厂里没放出来的fleet
        """
        fleet_num_layer = np.zeros((self.env_configuration.size, self.env_configuration.size)
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
            fleet_num_layer[x, y] = multilpier * self.map_value(size)
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
            fleet_num_layer[x, y] = multilpier * self.map_value(size)

        
        """
        # 第三层：记录所有舰队当前的位置，区分敌(-1)我(+1)
        """
        # Building the enemy positions layer
        fleet_position_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size)
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
            (self.env_configuration.size, self.env_configuration.size)
        )
        # - Iterate over fleets
        for fleet in fleets:
            # - Get the position of the fleet
            position = fleet.position
            x, y = position.x, position.y
            # - Get the amount of kore the fleet is carrying
            kore = fleet.kore
            # - Set the kore layer to the amount of kore
            kore_carried_by_fleet_layer[x, y] = self.map_value(kore)

        """
        第五层：所有船厂的位置以及每一回合最多能生产的ship数量（越大则说明这个船厂被控制了越久），区分敌(-)我(+)
        """
        shipyard_position_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size)
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
            (self.env_configuration.size, self.env_configuration.size)
        )
        for fleet in fleets:
            if fleet.player != board.current_player:  # 如果是敌方的fleet
                next_pos = next_positioin(fleet, size=self.env_configuration.size)
                x,y = next_pos.x, next_pos.y
                enemy_fleet_next_postion_layer[x,y] = -1

        """
        第七层：我方所有fleet下一步的位置，用 +1 表示
        """
        my_fleet_next_postion_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size)
        )
        for fleet in fleets:
            if fleet.player == board.current_player:  # 如果是我方的fleet
                next_pos = next_positioin(fleet, size=self.env_configuration.size)
                x,y = next_pos.x, next_pos.y
                my_fleet_next_postion_layer[x,y] = +1

        
        """第8层：高威胁性的敌方船厂，用(-1*shipyard.max_spawn)表示"""
        dangerous_opponent_shipyard_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size)
            )
        me :Player= board.current_player
        my_shipyard_position = [shipyard.position for shipyard in me.shipyards]
        for shipyard in shipyards:
            # - Get the position of the shipyard
            position = shipyard.position
            x, y = position.x, position.y
            # - Check if the shipyard is an enemy shipyard
            if shipyard.player != board.current_player: # 敌方船厂
                opp_shipyard_pos :Point = shipyard.position # 敌方船厂位置
                for my_pos in my_shipyard_position:
                    if opp_shipyard_pos.distance_to(my_pos,self.env_configuration.size) <= 2: # 建在我家门口
                        dangerous_opponent_shipyard_layer[x,y] = -1 * shipyard.max_spawn


        """第9层：高威胁性敌方舰队，用(-1*ship_count)表示"""
        dangerous_opponent_fleet_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size) )
            
        # me :Player= board.current_player
        # my_shipyard_position = [shipyard.position for shipyard in me.shipyards]
        for fleet in fleets:
            if fleet.player != board.current_player:  # 如果是敌方的fleet
                pos = fleet.position # 敌方舰队当前位置
                x,y = pos.x, pos.y
                next_pos = next_positioin(fleet, size=self.env_configuration.size) # 敌方舰队下一步位置
                for my_pos in my_shipyard_position:
                    if next_pos.distance_to(my_pos,self.env_configuration.size) <=1:
                        dangerous_opponent_fleet_layer[x,y] = -1*fleet.ship_count
        
        """第10层：无家可归的【我方】舰队，将其所在的行或列用+1*ship_count表示"""
        homeless_my_fleet_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size) )
        
        for fleet in fleets:
            if fleet.player == board.current_player and len(fleet.flight_plan) == 0: # 是我方的舰队，且有可能无家可归
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
                        homeless_my_fleet_layer[:,y] = +1*fleet.ship_count
                if direction %2 == 1: # 如果沿东-西方向飞行
                    homeless = True
                    for shipyard in shipyards:
                        if shipyard.position.x == x:
                            homeless = False
                            break
                    if homeless:
                        homeless_my_fleet_layer[x,:] = +1*fleet.ship_count

        """第11层：无家可归的敌方舰队，将其所在的行或列用-1*ship_count表示"""
        homeless_opponent_fleet_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size) )
        
        opponent = board.opponents[0]

        opponent_shipyard_position = [shipyard.position for shipyard in opponent.shipyards]

        for fleet in fleets:
            if fleet.player != board.current_player and len(fleet.flight_plan) == 0: # 是敌方的舰队，且可能无家可归
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
                        homeless_opponent_fleet_layer[:,y] = -1*fleet.ship_count
                if direction %2 == 1: # 如果沿东-西方向飞行
                    homeless = True
                    for shipyard in shipyards:
                        if shipyard.position.x == x:
                            homeless = False
                            break
                    if homeless:
                        homeless_opponent_fleet_layer[x,:] = -1*fleet.ship_count               

        """第12层：想要攻击我方船厂的敌方舰队，用-1*ship_count表示"""
        opponent_offensive_fleet_layer = np.zeros(
            (self.env_configuration.size, self.env_configuration.size) )
        
        for fleet in fleets:
            if fleet.player != board.current_player and len(fleet.flight_plan) == 0: # 是敌方的舰队，且已经没有飞行计划，常出现在攻击我方船厂的情况
                position :Point = fleet.position
                x,y = position.x, position.y
                direction = fleet.direction.to_index()
                if direction % 2 == 0: # 如果沿南-北方向飞行
                    offensive = False
                    for my_pos in my_shipyard_position:
                        if my_pos.y == y:
                            offensive == True
                            break
                    if offensive:
                        opponent_offensive_fleet_layer[x,y] = -1*fleet.ship_count
                if direction %2 == 1: # 如果沿东-西方向飞行
                    offensive = False
                    for my_pos in my_shipyard_position:
                        if my_pos.x == x:
                            offensive == True
                            break
                    if offensive:
                        opponent_offensive_fleet_layer[x,y] = -1*fleet.ship_count

        # Building our observation box
        observation = np.zeros(
            (self.env_configuration.size, self.env_configuration.size, 12)
        )
        observation[:, :, 0] = kore_layer
        observation[:, :, 1] = fleet_num_layer
        observation[:, :, 2] = fleet_position_layer
        observation[:, :, 3] = kore_carried_by_fleet_layer
        observation[:, :, 4] = shipyard_position_layer
        observation[:, :, 5] = enemy_fleet_next_postion_layer
        observation[:, :, 6] = my_fleet_next_postion_layer
        observation[:, :, 7] = dangerous_opponent_shipyard_layer
        observation[:, :, 8] = dangerous_opponent_fleet_layer
        observation[:, :, 9] = homeless_my_fleet_layer
        observation[:, :, 10] = homeless_opponent_fleet_layer
        observation[:, :, 11] = opponent_offensive_fleet_layer

        return observation


    def reset(self):
        """
        Resets the environment.
        """
        self.raw_observation = self.env.reset()
        obs = self.build_observation(self.raw_observation)
        return obs

    def step(self, action_type : int):
        from utils_v3.balanced import balanced_agent
        from utils_v3.attacker import attacker_agent
        from utils_v3.miner import miner_agent
        from utils_v3.expander import expand_agent
        from utils_v3.defenser import defense_agent
        from utils_v3.rescue import rescue_agent
        from utils_v3.rob import rob_agent
        from utils_v3.defend_dangerous_fleet import defend_dangerous_fleet_agent
        from utils_v3.defend_dangerous_shipyard import defend_dangerous_shipyard_agent
        from utils_v3.siege import siege_agent
        """
        Performs an action in the environment.
        """
        # Get the Board object and update it
        self.board = Board(self.raw_observation, self.env_configuration)
        previous_board = self.board

        # Sets done if no shipyards are left
        # if len(self.board.current_player.shipyards) == 0:
        #     return np.zeros((21, 21, 7)), 0, True, {}
        
        
        # Get the action for the shipyard

        if action_type == 0:
            action = balanced_agent(self.raw_observation,self.env_configuration)
        elif action_type == 1:
            action = attacker_agent(self.raw_observation,self.env_configuration)
        elif action_type == 2:
            action = miner_agent(self.raw_observation,self.env_configuration)
        elif action_type == 3:
            action = expand_agent(self.raw_observation,self.env_configuration)
        elif action_type == 4:
            action = defense_agent(self.raw_observation,self.env_configuration)
        elif action_type == 5: 
            action = rescue_agent(self.raw_observation,self.env_configuration)
        elif action_type == 6:
            action = rob_agent(self.raw_observation,self.env_configuration)
        elif action_type == 7:
            action = defend_dangerous_fleet_agent(self.raw_observation,self.env_configuration)
        elif action_type == 8:
            action = defend_dangerous_shipyard_agent(self.raw_observation,self.env_configuration)
        elif action_type == 9:
            action = siege_agent(self.raw_observation,self.env_configuration)

        # self.board.current_player.shipyards[0].next_action = action
        # self.raw_observation, old_reward, done, info = self.env.step(
        #     self.board.current_player.next_actions
        # )
        self.raw_observation, old_reward, done, info = self.env.step(
            action
        )
        observation = self.build_observation(self.raw_observation)
        # reward = self.map_reward(old_reward)

        # from reward_utils import get_board_value
        new_board = Board(self.raw_observation,self.env_configuration)
        modified_reward = get_board_value(new_board) - get_board_value(previous_board)
        if done:
            # we_won = 0
            # Who won?
            # if agent_reward is None or opponent_reward is None:
            #     we_won = -1
            me = new_board.current_player
            op = new_board.opponents[0]

            if len(me.fleets) == 0 or len(me.shipyards) == 0: # 我们被对手消灭了
                we_won = -1
            elif len(op.fleets) == 0 or len(op.shipyards) == 0: # 我们把对手消灭了
                we_won = 1
            else: # 如果双方都没有被消灭，则此时是到达400轮而结束，比较kore的数量
                agent_reward = me.kore
                opponent_reward = me.kore
                we_won = 1 if agent_reward > opponent_reward else -1 # 看看谁的kore更多

            if we_won == 1:
                print('本次战斗胜利！')
                win_reward = we_won * (8000  +  10 * (400 - new_board.step))
            if we_won == -1:
                print('本次战斗失败！')
                win_reward = old_reward + we_won * (2000 + 10 * (400 - new_board.step))

            print(win_reward)

            # win_reward = we_won * (1000 + 10 * (400 - new_board.step))
        else:
            win_reward = 0

        modified_reward += win_reward


        ##  对侵略性的行为进行一些奖励（但不能太高）

        # if action_type == 1 : # 中后期再进攻，前期努力发育，做大做强更重要，不要随意进攻
        #     if previous_board.step > 200: 
        #         modified_reward += 1
        #     else:
        #         modified_reward -= 1

        # if action_type == 3:
        #     if ( 125 < previous_board.step and previous_board.step < 350 ) : # 游戏的中途比较适合扩张领土，鼓励AI建造船厂
        #         modified_reward += 1 + min(max(len(new_board.current_player.shipyards) - len(new_board.opponents[0].shipyards), 0) , 2)
        #     else:
        #         modified_reward -= 1

        # if action_type == 2 :  # 如果矿石很多了还采矿，就要惩罚；如果矿石不够而去采矿，则奖励
        #     modified_reward +=  -0.025 * min(max( previous_board.current_player.kore - 1000, 0), 4000)
        
        ## print(modified_reward)


        return observation, modified_reward, done, info
    

def build_env():
    return CustomKoreEnv()

"""
# build Q-learning model
"""

# from keras import models
def create_q_model():
    num_actions  = 10
    return keras.models.Sequential(
        [
            layers.Input(shape=(21, 21, 12)),
            layers.Conv2D(64, 4),
            layers.Activation("linear"),
            layers.Conv2D(128, 8),
            layers.Activation("linear"),
            layers.Flatten(),
            layers.Dense(256),
            layers.Activation("sigmoid"),
            layers.Dense(num_actions),
            layers.Activation("linear")
        ]
    )


# DQN的灵魂：构建两个神经网络
# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
model.load_weights('my_weights_第三阶段.h5')
# model.summary()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()
model_target.load_weights('my_weights_第三阶段.h5')


"""
## Train the model
"""

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# Using huber loss for stability
loss_function = keras.losses.Huber() # 融合了MSE和MAE优点的损失函数


import time
start_time = time.time()

while True:  # Run until solved
    # 初始化环境
    env = build_env()
    config = env.env_configuration
    observation = env.reset()   # 只有在预测的时候才需要增加维度
    # 本时间片的收益
    episode_reward = 0
    
    for timestep in range(1, max_steps_per_episode): 

        frame_count += 1

        # Use epsilon-greedy for exploration 
        if frame_count < epsilon_random_frames or epsilon > random.random():
            # if episode_count % 5 == 1 or episode_count % 5 == 2:
            #     action = 4 #  用defenser策略战斗
            # else:
            action = random.choice(range(num_actions))

        else:
        # Predict action Q-values
        # From environment state
        # Take best action
            observation_tensor = tf.convert_to_tensor(observation)
            observation_tensor = tf.expand_dims(observation_tensor,0)
            actions = model(observation_tensor,training=False)   # 只有在预测的时候才需要增加维度

            # print(actions)
            action = np.argmax(actions)

        # print(action)

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        observation_next, reward, done, info = env.step(action)
        # observation_next_tensor = tf.convert_to_tensor(observation_next)
        # observation_next_tensor = tf.expand_dims(observation_next_tensor,0)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(observation)
        state_next_history.append(observation_next)
        done_history.append(done)
        rewards_history.append(reward)

        # 状态更新
        observation = observation_next

        # Update every fourth frame and once batch size is over batch_size
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample,verbose = 0)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 5: # 只保留最后5次的成绩
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    # model.save('model_save/my_model') # 每训练一轮就保存一下模型，防止数据丢失
    # weights = model.get_weights()
    # with open('weight.pickle', 'wb') as f:
    #     pickle.dump(weights, f)
    # model.save_weights('my_weights_v3')  # 保存模型权重
    model.save_weights('my_weights_第三阶段.h5')
    # model.save_weights('weight/my_weights_v3.h5')

    now = time.time()
    # print("minute:", (now - start_time)/60)
    if now - start_time > 2*60*60:
        break

