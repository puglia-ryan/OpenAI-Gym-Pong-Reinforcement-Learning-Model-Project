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
        """
        frame = Image.fromarray(observation)
        frame = frame.resize(shape)
        frame = frame.convert("L")
        frame = np.array("unit8")
        """
        return observation

    def process_state_batch(self, batch):
        return batch / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)

def get_coords(frame):
    paddle1 = []
    paddle2 = []
    ball = None

    for row_index in range(len(frame)):
        for col_index in range(len(frame)):
            if frame[row_index][col_index] == 255:
                if col_index < 12:
                    paddle1.append(row_index)
                elif col_index > 73:
                    paddle2.append(row_index)
                else:
                    ball = [row_index, col_index]
    if len(paddle1) == 0 or len(paddle2) == 0:
        return 42, 42, [42, 42]
    return sum(paddle1)/len(paddle1), sum(paddle2)/len(paddle2), ball


