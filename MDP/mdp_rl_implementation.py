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
            if mdp.board[row][col] != "WALL":
                states.append((row, col, float(mdp.board[row][col])))
    return states
    
def get_max_value_and_action(mdp, U, row, col):
    values = []
    max_action = None
    for action in Action: 
        s = []
        for i in range(4):
            p = mdp.transition_function[action][i]
            if p == 0:
                continue
            next_state = mdp.step((row, col), list(Action)[i])
            s.append(p * U[next_state[0]][next_state[1]])   
        values.append([sum(s), action])
        values_list = [i[0] for i in values]
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
    # ====== YOUR CODE: ======
    
    U = deepcopy(U_init)
    delta = 1
    while delta >= epsilon * ((1 - mdp.gamma) / mdp.gamma):
        for state in mdp.terminal_states:  
            row, col = state[0], state[1]
            U[row][col] = float(mdp.board[row][col])
        U_final = deepcopy(U)
        delta = 0
        current_states = get_states(mdp)
        for row, col, value in current_states:
            if (row, col) not in mdp.terminal_states: 
                max_val, _ = get_max_value_and_action(mdp, U_final, row, col)
                U[row][col] = float(value + mdp.gamma * max_val)
                delta = max(delta, abs(U[row][col] - U_final[row][col]))
    return U_final
    # ========================
    

def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    
    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 
    
    
    
    policy = deepcopy(U)
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            _, action = get_max_value_and_action(mdp, U, row, col)
            policy[row][col] = action
    return policy
    
    # ========================
    
def sum_next_steps(mdp, U, state, action):
        next_steps_prob = [(mdp.step(state, move), float(mdp.transition_function[action][index])) for index, move in enumerate(mdp.actions)]
        return sum( P * float(U[next_step[0]][next_step[1]]) for next_step, P in next_steps_prob)



def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # TODO:
    # ====== YOUR CODE: ======
    
    # Initialize U 
    U = np.full((mdp.num_row, mdp.num_col), 0.0, dtype=object)

    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            state = (row, col)
            if mdp.board[row][col] == "WALL":
                U[row][col] = None
            if state in mdp.terminal_states:
                U[row][col] = float(mdp.get_reward(state))
            

    # Evaluate the policy until it converges:
    epsilon = 10 ** (-3)
    while True:
        theta = 0
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                state = (row, col)
                if state in mdp.terminal_states or mdp.board[row][col] == "WALL":
                    continue
                action = Action(policy[row][col])
                reward = float(mdp.get_reward(state))
                v = reward + mdp.gamma * sum_next_steps(mdp, U, state, action)
                diff = np.abs(U[row][col] - v)
                theta = max(theta, diff)
                U[row][col] = v

        if theta < epsilon:
            break

    return U

    # ========================


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:

    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    
    policy = deepcopy(policy_init)

    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            state = (row, col)
            if state in mdp.terminal_states or mdp.board[row][col] == "WALL":
                policy[row][col] = None
  
    while True:
        unchanged = True
        U = policy_evaluation(mdp,policy)
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                state = (row, col)
                if state in mdp.terminal_states or mdp.board[row][col] == "WALL":
                    continue
                
                max_val, max_action = get_max_value_and_action(mdp, U, row, col)
                
                action = Action(policy[row][col])
                sum_steps = sum_next_steps(mdp, U, state, action)
                
                if max_val > sum_steps:
                    policy[row][col] = max_action
                    unchnaged = False
        
        if unchanged == True:
            break
        
    optimal_policy = deepcopy(policy)
    return optimal_policy
    # raise NotImplementedError
    # ========================
    



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
