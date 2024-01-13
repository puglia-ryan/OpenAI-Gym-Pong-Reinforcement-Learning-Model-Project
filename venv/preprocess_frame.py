import cv2
import numpy as np
from rl.core import Processor
from PIL import Image

class Frame_Processor(Processor):
    def process_observation(self, observation):
        shape = (84, 84)

        observation = observation.astype(np.uint8)
        observation = observation[34:34 + 160, :160]

        # If the frame has multiple color channels (e.g., RGB), convert to grayscales
        if len(observation.shape) == 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # We then use a threshhold (144) to convert the image to an array of 0s and 255s
        observation[observation < 144] = 0
        observation[observation >= 144] = 255
        observation = cv2.resize(observation, shape, interpolation=cv2.INTER_NEAREST)

        return observation

    def process_state_batch(self, batch):
        return  batch.astype('float32') / 255

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)




