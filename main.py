import numpy as np

import game_env
import agent
import time
import matplotlib.pyplot as plt
import random
import preprocess_frame as ppf
import cv2

total_steps = 100000
target_update_frequency = 1500
step = 0

pg = game_env.pong_game()
pg.reset()
valid_actions = 2
game_agent = agent.Agent((84, 84), valid_actions)
cv2.namedWindow("Pong Game", cv2.WINDOW_NORMAL)

game_agent.load_weights()



while True:
    training_func = game_agent.dqAgent.fit(pg, nb_steps=1000000, callbacks=[game_agent.checkpoint_callback], log_interval=10000, visualize=False)
    game_agent.dqAgent.test(game_env, nb_episodes=1, visualize=True)
    pg.close()
    game_agent.dqAgent.model.summary()
    break


