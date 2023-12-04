#This file is responsible for creating the agent which interracts with the game environment
import numpy as np
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from keras.optimizers import Adam

import memory
from model import CustomModel
from memory import Memory

class Agent:
    def __init__(self, input_shape, actions, memory_len=3):
        self.input_shape = input_shape
        self.actions = actions
        self.memory = memory.Memory(memory_len)
        self.neural_model = CustomModel(input_shape, actions)
        self.dqAgent = self.create_dqn_agent()
        self.total_memory = []

    def create_dqn_agent(self):
        agent = DQNAgent(
            model=self.neural_model.model,
            memory=self.memory,
            policy=BoltzmannQPolicy(),
            nb_actions=self.actions,
            nb_steps_warmup=100,
            target_model_update=1000
        )
        agent.compile(Adam(lr=0.0001), metrics=["mae"])
        return agent

    def model_summary(self):
        print(self.neural_model.model.summary())

    def select_move(self):
        last_frames = list(self.memory.frames)
        if len(self.memory.frames) < 3:
            return 0
        last_frames = np.stack(last_frames, axis=0)
        q_values = self.neural_model.model.predict(last_frames[np.newaxis, :, :, :])
        self.total_memory.append(last_frames[np.newaxis, :, :, :])
        return np.argmax(q_values[-1])
