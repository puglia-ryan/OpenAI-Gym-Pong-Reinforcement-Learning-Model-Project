import cv2
import numpy as np


def resize_frame(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)
    frame = frame[34:34 + 160, :160]

    # If the frame has multiple color channels (e.g., RGB), convert to grayscales
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #We then use a threshhold (144) to convert the image to strictly black and white
    frame[frame < 144] = 0
    frame[frame >= 144] = 255
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    return frame

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

    return sum(paddle1)/len(paddle1), sum(paddle2)/len(paddle2), ball




