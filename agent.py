#This file is responsible for creating the agent which interracts with the game environment
import numpy as np
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from keras.optimizers import Adam
import memory
from model import CustomModel


class Agent:
    def __init__(self, input_shape, actions, memory_len=5):
        self.input_shape = input_shape
        self.actions = actions
        self.memory = memory.Memory(memory_len)
        self.neural_model = CustomModel(input_shape, actions)
        self.dqAgent = self.create_dqn_agent()
        self.total_memory = []

    def create_dqn_agent(self):
        epsilon_policy = EpsGreedyQPolicy(epsilon=0.1)
        agent = DQNAgent(
            model=self.neural_model.model,
            memory=self.memory,
            policy=epsilon_policy,
            nb_actions=self.actions,
            nb_steps_warmup=100,
            target_model_update=1000
        )
        agent.compile(Adam(lr=0.0001), metrics=["mae"])
        return agent

    def model_summary(self):
        print(self.neural_model.model.summary())

    def preprocess_coord_input(self):
        last_coords = list(self.memory.frames)
        if len(last_coords) < 5:
            return np.zeros(self.input_shape)
        last_coords = np.stack(last_coords, axis=0)
        return last_coords

    def select_move(self):
        input_coords = self.preprocess_coord_input()
        q_values = self.neural_model.model.predict(input_coords[np.newaxis, :, :, :])
        self.total_memory.append(input_coords[np.newaxis, :, :, :])
        return np.argmax(q_values[-1])
