import numpy as np
from preprocess_frame import resize_frame

import game_env
import agent
import memory

total_steps = 1000000
update_frequency = 4
target_update_frequency = 10000
batch_size = 32


# Training loop
pg = game_env.pong_game()
step = 0
total_reward = 0
state = np.zeros((4, 84, 84))
valid_actions = 2
pg.reset()
deepQAgent = agent.Agent((84, 84), valid_actions)

while step < total_steps:
    action = deepQAgent.select_move(state)
    print(action)
    processed_frame, reward, done = pg.takeAction(action)
    total_reward += reward

    # Append processed_frame along the first dimension to maintain the temporal sequence
    new_state = np.append(state[1:], processed_frame[np.newaxis, :, :], axis=0)

    # Add the experience to the memory
    deepQAgent.memory.append(state, action, reward, new_state, done)

    state = new_state
    step += 1

    if step > deepQAgent.dqAgent.nb_steps_warmup and deepQAgent.memory.nb_entries >= deepQAgent.memory.window_length + 2:
        # Train the DQN agent using a batch of experiences from memory
        experiences = deepQAgent.memory.sample(batch_size)
        deepQAgent.dqAgent.train_on_batch(experiences)

    if step % update_frequency == 0:
        deepQAgent.dqAgent.update_target_model_hard()

    if step % target_update_frequency == 0:
        deepQAgent.dqAgent.save_weights("your_weights.h5")

    if done:
        print(f"Step: {step}, Total Reward: {total_reward}")
        pg.reset()
        total_reward = 0

