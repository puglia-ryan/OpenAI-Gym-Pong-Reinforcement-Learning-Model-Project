import gym
import preprocess_frame

import agent



processor = preprocess_frame.Frame_Processor()
total_steps = 600000
target_update_frequency = 10000
step = 0

env = gym.make("ALE/Pong-v5", render_mode="human")
env.reset()
valid_actions = env.action_space.n
img_shape = (84, 84)
window_length = 12
input_shape = (window_length, img_shape[0], img_shape[1])
game_agent = agent.Agent(input_shape, valid_actions, total_steps, window_length)




#game_agent.dqAgent.test(env, nb_episodes=20, visualize=False)

#Training function: This function trains the agent for a given number of steps and saves the weights of the model every given interval
training_func = game_agent.dqAgent.fit(env, nb_steps=total_steps, callbacks=[game_agent.checkpoint_callback], log_interval=target_update_frequency, visualize=False)
#The test() function let's the agent play a given number of games by using its best predicted strategy. There is no exploration/exploitation
game_agent.dqAgent.test(env, nb_episodes=1, visualize=False)
env.close()
game_agent.model_summary()



"""
env = gym.make("ALE/Pong-v5", render_mode="human")
env.reset()

for i in range(2000):

    observation = preprocess_frame.process_observation(env)
    #observation = processor.process_state_batch(observation)
    action = game_agent.forward(observation)
    observation, reward, done, info = env.step(action)
    observation, reward, done, info = processor.process_step(observation, reward, done, info)
    print(f"Action: {action}", f"Reward: {reward}")
    game_agent.backward(reward, terminal=done)

env.close()
"""