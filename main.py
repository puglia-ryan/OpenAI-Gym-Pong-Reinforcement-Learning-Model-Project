import numpy as np

import game_env
import agent
import time
import matplotlib.pyplot as plt
import random
import preprocess_frame as ppf
import cv2

total_steps = 1500
update_frequency = 4
target_update_frequency = 1500
np.set_printoptions(threshold=np.inf)
pg = game_env.pong_game()
pg.reset()
valid_actions = 2
game_agent = agent.Agent(4, valid_actions)

for i in range(40):
    action = random.randint(0, 1)
    frame, reward, done, info = pg.takeAction(action)
    processed_frame = ppf.resize_frame(frame)
    cv2.imshow("Pong Game", processed_frame)
    cv2.waitKey(10)

    p1, p2, ball = ppf.get_coords(processed_frame)
    if ball is not None:
        prev_ball = ball
    else:
        if prev_ball[1] < 20:
            ball = [prev_ball[0], prev_ball[1]-1]
            prev_ball = ball
        else:
            ball = [prev_ball[0], prev_ball[1]+1]
            prev_ball = ball
    all_inputs = [p1, p2, ball[0], ball[1]]
    game_agent.memory.add_entry(all_inputs, reward, action, done)
    if done:
        pg.reset()
