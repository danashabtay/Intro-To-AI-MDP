from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np

from copy import deepcopy
import random

actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']


def get_states(mdp):
    states = []
    for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                if mdp.board[row][col] == "WALL":
                    continue
                else:
                    states.append((row, col, float(mdp.board[row][col])))
    
def get_max_value_and_action(mdp, U, row, col):
    values = []
    max_action = None
    for action in actions: #check for each legal action its E value
        s = []
        for i in range(4):
            p = mdp.transition_function[action][i]
            if p == 0:
                continue
            next_state = mdp.step((row, col), actions[i])
            s.append(p * U[next_state[0]][next_state[1]])   
        values.append([sum(s), action])
        values_list = [j[0] for j in values]
        max_value = max(values_list)
        max_index = values_list.index(max_value)
        max_action = values[max_index][1]

    return (max_value,max_action)               


def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float=10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #
    
    U_final = None
    # TODO:
    # ====== YOUR CODE: ======
    
    U = deepcopy(U_init)
    delta = 1
    
    while delta >= epsilon * ((1-mdp.gamma) / mdp.gamma):
        for state in mdp.terminal_states:
            row = state[0]
            col = state[1]
            U[row][col] = float(mdp.board[row][col])
        U_final = deepcopy(U)
        delta = 0
        states = get_states(mdp)
        for row, col, value in states:
            if(row,col) not in mdp.terminal_states:
                max_val, _ = get_max_value_and_action(mdp,U_final,row,col)
                U[row][col] = float(value + mdp.gamma * max_val)
                delta = max(delta, abs(U[row][col] - U_final[row][col]))
    return U_final
    #raise NotImplementedError
    # ========================
    

def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    
    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 
    raise NotImplementedError
    # ========================
    return policy


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # TODO:
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:

    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
    return optimal_policy



def adp_algorithm(
    sim: Simulator, 
    num_episodes: int,
    num_rows: int = 3, 
    num_cols: int = 4, 
    actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT] 
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid, 
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.
    
    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions. 
    """
    

    transition_probs = None
    reward_matrix = None
    # TODO
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
    return reward_matrix, transition_probs 
