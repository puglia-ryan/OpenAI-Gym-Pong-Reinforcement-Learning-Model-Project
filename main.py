import gym
import numpy as np
import matplotlib.pyplot as plt
import preprocess_frame
from rl.callbacks import Callback
import random

import agent

processor = preprocess_frame.Frame_Processor()
total_steps = 600000
target_update_frequency = 10000
step = 0


def train_model():
    valid_actions = env.action_space.n
    img_shape = (84, 84)
    window_length = 12
    input_shape = (window_length, img_shape[0], img_shape[1])
    game_agent = agent.Agent(input_shape, valid_actions, total_steps, window_length)
    env = gym.make("ALE/Pong-v5")
    env.reset()

    # Training function: This function trains the agent for a given number of steps and saves the weights of the model every given interval
    training_func = game_agent.dqAgent.fit(env, nb_steps=total_steps, callbacks=[game_agent.checkpoint_callback],
                                           log_interval=target_update_frequency, visualize=False)
    env.close()


def test_model():
    env = gym.make("ALE/Pong-v5")
    env.reset()
    ep_num = 200

    class TestCallback(Callback):
        def __init__(self):
            self.episode_rewards = []

        def on_episode_end(self, episode, logs):
            # Collect the total episode reward
            self.episode_rewards.append(logs['episode_reward'])

    test_callback = TestCallback()
    valid_actions = env.action_space.n
    img_shape = (84, 84)
    window_length = 12
    input_shape = (window_length, img_shape[0], img_shape[1])
    game_agent = agent.Agent(input_shape, valid_actions, total_steps, window_length)
    game_agent.dqAgent.test(env, nb_episodes=ep_num, visualize=False, callbacks=[test_callback])
    adjusted_rewards = [element + 21 for element in test_callback.episode_rewards]
    env.reset()

    games_played = 0
    total_reward = 0
    rand_total_reward = []
    while True:
        action = random.randint(0, 5)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            rand_total_reward.append(total_reward)
            if games_played == ep_num - 1 :
                break
            else:
                games_played += 1
                total_reward = 0
                env.reset()

    env.close()

    rand_total_reward = [element + 21 for element in rand_total_reward]
    plt.scatter(np.arange(len(adjusted_rewards)), adjusted_rewards, label="The Agent's scores", color="blue", marker="x")
    plt.axhline(y=np.mean(adjusted_rewards), color="blue", linestyle="--", label="Agent's average score")
    plt.scatter(np.arange(len(rand_total_reward)), rand_total_reward, label="Scores of Random Inputs", color="red", marker="x")
    plt.axhline(y=np.mean(rand_total_reward), color="red", linestyle="--", label="Random Input average score")
    plt.ylim(-1, 22)
    plt.xlabel('Episode Number')
    plt.ylabel('Score')
    plt.title(f"Reinforcement Learning Agent rewards over the course of {(len(adjusted_rewards))} games:")
    plt.legend()
    print(f"Agent Episode Rewards: {adjusted_rewards}\nAverage Agent Reward: {np.mean(adjusted_rewards)}")
    print(f"Random Action Rewards: {rand_total_reward}\nRandom Action Average: {np.mean(rand_total_reward)}")
    plt.show()



def custom_test():
    # env = gym.make("ALE/Pong-v5", render_mode="human")
    observation = env.reset()
    np.set_printoptions(threshold=np.inf)
    processor = preprocess_frame.Frame_Processor()

    for i in range(10000):
        observation = processor.process_observation(observation)
        # observation = processor.process_state_batch(observation)

        action = game_agent.dqAgent.forward(observation)

        observation, reward, done, info = env.step(action)

        observation, reward, done, info = processor.process_step(observation, reward, done, info)

        game_agent.dqAgent.backward(reward, terminal=done)

        print(f"Action: {action}", f"Reward: {reward}")


while True:
    user_input = input("Type 1 to train the model\nType 2 to test the model\nType 3 to end the program")
    if user_input == "1":
        train_model()
        break
    elif user_input == "2":
        test_model()
        break
    else:
        print("Not a valid input")
