#This file is responsible for creating the memory for the agent, since the agent will use multiple frames to analyse its moves
from collections import deque

class Memory():
    def __init__(self, length):
        self.length = length
        self.frames = deque(maxlen=length)
        self.actions = deque(maxlen=length)
        self.rewards = deque(maxlen=length)
        self.done = deque(maxlen=length)

    def add_entry(self, new_frame, new_reward, new_action, new_done):
        self.frames.append(new_frame)
        self.rewards.append(new_reward)
        self.actions.append(new_action)
        self.done.append(new_done)
