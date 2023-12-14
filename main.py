import gym
import preprocess_frame

import agent
import matplotlib.pyplot as plt
import random
import tensorflow as tf


processor = preprocess_frame.Frame_Processor()
total_steps = 400000
target_update_frequency = 10000
step = 0

env = gym.make("ALE/Pong-v5", render_mode="human")
env.reset()
valid_actions = env.action_space.n
img_shape = (84, 84)
window_length = 12
input_shape = (window_length, img_shape[0], img_shape[1])
game_agent = agent.Agent(input_shape, valid_actions)
game_agent.model_summary()



#game_agent.dqAgent.test(env, nb_episodes=1, visualize=False)

#Training function
"""
while True:
    training_func = game_agent.dqAgent.fit(env, nb_steps=300000, callbacks=[game_agent.checkpoint_callback], log_interval=5000, visualize=False)
    game_agent.dqAgent.test(env, nb_episodes=1, visualize=False)
    env.close()
    game_agent.model_summary()
    break
"""
"""
env = gym.make("ALE/Pong-v5", render_mode="human")
env.reset()

for i in range(2000):

    observation = preprocess_frame.process_observation(env)
    #observation = processor.process_state_batch(observation)
    action = test_agent.forward(observation)
    observation, reward, done, info = env.step(action)
    observation, reward, done, info = processor.process_step(observation, reward, done, info)
    print(f"Action: {action}", f"Reward: {reward}")
    test_agent.backward(reward, terminal=done)

env.close()
"""