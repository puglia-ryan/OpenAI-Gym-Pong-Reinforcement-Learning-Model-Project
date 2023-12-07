#This file is responsible for creating the agent which interracts with the game environment
import numpy as np
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from keras.optimizers import Adam
import memory
from model import CustomModel


class Agent:
    def __init__(self, input_shape, actions, memory_len=5):
        self.input_shape = input_shape
        self.actions = actions
        self.memory = memory.Memory.mem
        self.neural_model = CustomModel(input_shape, actions)
        self.dqAgent = self.create_dqn_agent()
        self.total_memory = []

    def create_dqn_agent(self):
        epsilon_policy = LinearAnnealedPolicy(EpsGreedyQPolicy, attr='eps', value_max=1.0, value_min=0.1, value_test=0.05, nb_steps=5000000)
        agent = DQNAgent(
            model=self.neural_model.model,
            memory=self.memory,
            policy=epsilon_policy,
            nb_actions=self.actions,
            nb_steps_warmup=50000,
            gamma=.99,
            target_model_update=1000,
            train_interval=12,
            delta_clip=1
        )
        agent.compile(Adam(lr=0.001), metrics=["mae"])
        return agent



    def model_summary(self):
        print(self.neural_model.model.summary())


    def select_move(self):
        def select_move(self):
            # Get the input coordinates from the agent's memory
            input_coords = np.array(list(self.memory.frames))

            # Use epsilon-greedy policy to decide whether to explore or exploit
            if np.random.rand() < self.dqAgent.policy.epsilon:
                # Explore: Randomly choose an action
                return np.random.randint(self.actions)
            else:
                # Exploit: Choose the action with the highest Q-value
                q_values = self.neural_model.model.predict(input_coords)
                return np.argmax(q_values[-1])
