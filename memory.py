#This file is responsible for creating the memory for the agent
from rl.memory import SequentialMemory

class Memory():
    def __init__(self, limit_len, win_len):
        #win_len determines how many of the previous frames are fed to the agent for every step/decision
        self.win_len = win_len
        self.mem = SequentialMemory(limit=limit_len, window_length=self.win_len)


