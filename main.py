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


while True:
    """
    training_func = game_agent.dqAgent.fit(pg, nb_steps=1000000, callbacks=[game_agent.checkpoint_callback], log_interval=10000, visualize=False)
    game_agent.dqAgent.test(game_env, nb_episodes=1, visualize=True)
    pg.close()
    game_agent.model_summary()
    """


    action = game_agent.select_move()
    print(action)
    frame, reward, done, info = pg.takeAction(action)
    processed_frame = ppf.resize_frame(frame)
    cv2.imshow("Pong Game", processed_frame)
    cv2.waitKey(10)

    """
    p1, p2, ball = ppf.get_coords(processed_frame)
    if ball is not None:
        prev_ball = ball
    else:
        if prev_ball[1] < 20:
            ball = [prev_ball[0], prev_ball[1] - 1]
            prev_ball = ball
        else:
            ball = [prev_ball[0], prev_ball[1] + 1]
            prev_ball = ball

    all_inputs = [p1, p2, ball[0], ball[1]]
    """

    #game_agent.memory.add_entry(all_inputs, reward, action, done)
    if step % target_update_frequency == 0:
        game_agent.dqAgent.save_weights("your_weights.h5", overwrite=True)
        game_agent.dqAgent.update_target_model_hard()

    if done:
        pg.reset()
    step += 1
