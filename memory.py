#This file is responsible for creating the memory for the agent, since the agent will use multiple frames to analyse its moves
from collections import deque
from rl.memory import SequentialMemory

class Memory():
    def __init__(self, win_len):
        self.win_len = win_len
        self.mem = SequentialMemory(limit=500000, window_length=self.winlen)



