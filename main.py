import numpy as np

import game_env
import agent
import time
import matplotlib.pyplot as plt
import random

total_steps = 1500
update_frequency = 4
target_update_frequency = 1500


pg = game_env.pong_game()
pg.reset()

while True:
    r = random.randint(0, 1)
    pg.takeAction(r)