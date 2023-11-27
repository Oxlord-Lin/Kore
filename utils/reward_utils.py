# from action_helper.config import GAME_CONFIG, SHIP_COST, SHIPYARD_COST
"""
config
"""
import numpy as np
from kaggle_environments import make

# Read env specification
ENV_SPECIFICATION = make('kore_fleets').specification
SHIP_COST = ENV_SPECIFICATION.configuration.spawnCost.default
SHIPYARD_COST = ENV_SPECIFICATION.configuration.convertCost.default
GAME_CONFIG = {
    'episodeSteps':  ENV_SPECIFICATION.configuration.episodeSteps.default,  # You might want to start with smaller values
    'size': ENV_SPECIFICATION.configuration.size.default,
    'maxLogLength': None
}

# Define your opponent. We'll use the starter bot in the notebook environment for this baseline.
OPPONENT = 'opponent.py'
GAME_AGENTS = [None, OPPONENT]

# Define our parameters
N_FEATURES = 4
ACTION_SIZE = (2,)
DTYPE = np.float64
MAX_OBSERVABLE_KORE = 500
MAX_OBSERVABLE_SHIPS = 200
MAX_ACTION_FLEET_SIZE = 150
MAX_KORE_IN_RESERVE = 40000
WIN_REWARD = 1000



# ————————————————————————————————————————————分割线——————————————————————————————————————————————————————————————#



"""
reward-utils
"""

from kaggle_environments.envs.kore_fleets.helpers import Board
import numpy as np
from math import floor

# Compute weight constants -- See get_board_value's docstring
_max_steps = GAME_CONFIG['episodeSteps']
_end_of_asset_value = floor(.5 * _max_steps)
_weights_assets = np.linspace(start=1, stop=0, num=_end_of_asset_value)
_weights_kore = np.linspace(start=0, stop=1, num=_end_of_asset_value)
WEIGHTS_ASSETS = np.append(_weights_assets, np.zeros(_max_steps - _end_of_asset_value))
WEIGHTS_KORE = np.append(_weights_kore, np.ones(_max_steps - _end_of_asset_value))
WEIGHTS_MAX_SPAWN = {x: (x+3)/4 for x in range(1, 11)}  # Value multiplier of a shipyard as a function of its max spawn
WEIGHTS_KORE_IN_FLEETS = WEIGHTS_KORE * WEIGHTS_ASSETS/2  # Always equal or smaller than either, almost always smaller


def get_board_value(board: Board) -> float:
    """Computes the board value for the current player.

    The board value captures how are we currently performing, compared to the opponent. Each player's partial board
    value assesses the player's situation, taking into account their current kore, ship count, shipyard count
    (including their max spawn) and kore carried by fleets. We then define the board value as the difference between
    player's partial board values.
    Flight plans and the positioning of fleet and shipyards do not flow into the board value (yet).

    To keep things simple, we'll take a weighted sum as the partial board value. We need weighting since
    the importance of each item changes over time. We don't need to have the most kore at the beginning of the game,
    but we do at the end. Ship count won't help us win games in the latter stages, but it is crucial in the beginning.
    Fleets and shipyards will be accounted for proportionally to their kore cost.

    For efficiency, the weight factors are pre-computed at module level. Here is the logic behind the weighting:
    WEIGHTS_KORE: Applied to the player's kore count. Increases linearly from 0 to 1. It reaches one before
        the maximum game length is reached.
    WEIGHTS_ASSETS: Applied to fleets and shipyards. Decreases linearly from 1 to 0 and reaches zero before the maximum
        length. It emphasizes the need of having ships over kore at the beginning of the game.
    WEIGHTS_MAX_SPAWN: Shipyard value is multiplied by its max spawn. This captures the idea that long-held shipyards
        are more valuable.
    WEIGHTS_KORE_IN_FLEETS: Kore in fleets should be valued, too. But its value must be upper-bounded by WEIGHTS_KORE
        (it can never be better to have kore in cargo than home) and it must decrease in time, since it doesn't
        count towards the end kore count.

    Args:
        board: The board for which we want to compute the value.

    Returns:
        The value of the board.
    """
    board_value = 0
    if not board:
        return board_value

    # Get the weights as a function of the current game step
    step = board.step
    weight_kore, weight_assets, weight_cargo = WEIGHTS_KORE[step], WEIGHTS_ASSETS[step], WEIGHTS_KORE_IN_FLEETS[step]

    # Compute the partial board values
    for player in board.players.values():
        player_fleets, player_shipyards = list(player.fleets), list(player.shipyards)

        value_kore = weight_kore * player.kore

        value_fleets = weight_assets * SHIP_COST * (
                sum(fleet.ship_count for fleet in player_fleets)
                + sum(shipyard.ship_count for shipyard in player_shipyards)
        )

        value_shipyards = weight_assets * SHIPYARD_COST * (
            sum(shipyard.max_spawn * WEIGHTS_MAX_SPAWN[shipyard.max_spawn] for shipyard in player_shipyards)
        )

        value_kore_in_cargo = weight_cargo * sum(fleet.kore for fleet in player_fleets)

        # Add (or subtract) the partial values to the total board value. The current player is always us.
        modifier = 1 if player.is_current_player else -1
        board_value += modifier * (value_kore + value_fleets + value_shipyards + value_kore_in_cargo)

    return board_value
