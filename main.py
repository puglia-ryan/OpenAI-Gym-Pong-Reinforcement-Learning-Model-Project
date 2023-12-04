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

for i in range(100):
    r = random.randint(0, 1)
    frame, reward, action, info = pg.takeAction(r)
    processed_frame = ppf.resize_frame(frame)
    cv2.imshow("Pong Game", processed_frame)
    cv2.waitKey(10)
    if i > 97:
        print(processed_frame)
        p1, p2, ball = ppf.get_coords(processed_frame)
        print("P1: ", p1)
        print("P2: ", p2)
        print("Ball: ", ball)
        cv2.waitKey(100000)