from rl.memory import SequentialMemory

def create_memory(limit=10000, window_length=4):
    return SequentialMemory(limit=limit, window_length=window_length,  ignore_episode_boundaries=True)