#This file is responsible for creating the agent which interracts with the game environment
import keras.models
import numpy as np
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.callbacks import ModelIntervalCheckpoint
from keras.optimizers import Adam
from memory import Memory
from model import CustomModel
from keras.models import load_model
from preprocess_frame import Frame_Processor


class Agent:
    def __init__(self, input_shape, actions, mem_lim_len, window_len):
        self.checkpoint_filename = "checkpoint_file.h5f"
        self.checkpoint_callback = ModelIntervalCheckpoint(self.checkpoint_filename, interval=1000)
        self.actions = actions
        self.mem_lim_len = mem_lim_len
        self.winow_len = window_len
        self.memory = Memory(self.mem_lim_len, self.winow_len)
        self.input_shape = input_shape
        self.neural_model = CustomModel((self.input_shape), self.actions).model
        self.try_load_weights()
        self.processor = Frame_Processor()
        self.dqAgent = self.create_dqn_agent()
        self.total_memory = []
        self.try_load_weights()

    def create_dqn_agent(self):
        epsilon_policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.9, value_min=0.1, value_test=0.05, nb_steps=5000000)
        agent = DQNAgent(
            model=self.neural_model,
            memory=self.memory.mem,
            policy=epsilon_policy,
            processor=self.processor,
            nb_actions=self.actions,
            nb_steps_warmup=50000,
            gamma=.99,
            target_model_update=1000,
            train_interval=12,
            delta_clip=1
        )
        agent.compile(Adam(lr=0.00025), metrics=["mae"])
        return agent


    def model_summary(self):
        print(self.neural_model.summary())

    #This function attempts to load a checkpoint if there is one available
    def try_load_weights(self):
        try:
            print("Checkpoint file found")
            self.neural_model.load_weights(self.checkpoint_filename)
        except:
            print("No checkpoint file found")

