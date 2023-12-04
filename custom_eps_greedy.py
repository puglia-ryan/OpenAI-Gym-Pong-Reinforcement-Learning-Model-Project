import numpy as np
from rl.policy import Policy

class CustomEpsGreedyPolicy(Policy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)