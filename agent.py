#This file is responsible for creating the agent which interracts with the game environment
import numpy as np
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from keras.optimizers import Adam
from model import CustomModel
from memory import create_memory

class Agent:
    def __init__(self, input_shape, actions, memory_len=4):
        self.input_shape = input_shape
        self.actions = actions
        self.neural_model = CustomModel(input_shape, actions)
        self.memory = create_memory()
        self.dqAgent = self.create_dqn_agent()

    def create_dqn_agent(self):
        agent = DQNAgent(
            model=self.neural_model.model,
            memory=self.memory,
            policy=BoltzmannQPolicy(),
            nb_actions=self.actions,
            nb_steps_warmup=100,
            target_model_update=1000
        )
        agent.compile(Adam(lr=0.001), metrics=["mae"])
        return agent

    def model_summary(self):
        print(self.model.model.summary())

    def select_move(self, frame):
        q_values = self.neural_model.model.predict(frame)
        return np.argmax(q_values[-1])
