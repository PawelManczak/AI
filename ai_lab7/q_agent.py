import numpy as np
from rl_base import Agent, Action, State
import os
import random


class QAgent(Agent):

    def __init__(self, n_states, n_actions,
                 name='QAgent', initial_q_value=0.0, q_table=None):
        super().__init__(name)

        # hyperparams
        self.lr = 0.1 # [0.1 = 0.001]
        self.gamma = 0.9 # [0.9 -0.99]
        self.epsilon = 0.7
        self.eps_decrement = 0.00001
        self.eps_min = 0.00001

        self.action_space = [i for i in range(n_actions)]
        self.n_states = n_states
        self.q_table = q_table if q_table is not None else self.init_q_table(initial_q_value)

    def init_q_table(self, initial_q_value=0.):
        q_table = np.full((self.n_states, len(self.action_space)), initial_q_value)
        return q_table

    def update_action_policy(self) -> None:
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_decrement
        else:
            self.epsilon = self.eps_min

    def choose_action(self, state: State) -> Action:

        assert 0 <= state < self.n_states, \
            f"Bad state_idx. Has to be int between 0 and {self.n_states}"

        if np.random.uniform(0, 1) < self.epsilon:  # explore: choose a random action
            action = Action(random.choice(self.action_space))
        else:  # exploit: choose the action with max value (greedy policy)
            action = Action(np.argmax(self.q_table[state, :]))
        return action

    def learn(self, state: State, action: Action, reward: float, new_state: State, done: bool) -> None:
        current_q = self.q_table[state, action]
        new_q = reward + self.gamma * np.max(self.q_table[new_state, :])
        self.q_table[state, action] += self.lr * (new_q - current_q)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    def get_instruction_string(self):
        return [f"Linearly decreasing eps-greedy: eps={self.epsilon:0.4f}"]
