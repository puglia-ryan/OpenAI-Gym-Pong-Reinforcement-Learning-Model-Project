import cv2
import gym
import preprocess_frame as ppf


class pong_game(object):

    def __init__(self):
        self.environment = gym.make("ALE/Pong-v5")
        self.environment.seed(0)
        cv2.namedWindow("Pong Game", cv2.WINDOW_NORMAL)
    def reset(self):
        self.frame = self.environment.reset()

    def takeAction(self, action: int):
        # In the Pong documentation, there are 6 actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
        # We only need RIGHT and LEFT (so 2 and 3) for our model. The agent always returns 0 or 1 as a result, as there are only two output layers
        # That's why we add 2 to the action
        frame, reward, done, info = self.environment.step(action + 2)
        processed_frame = ppf.resize_frame(frame)
        cv2.imshow("Pong Game", processed_frame)
        cv2.waitKey(10)
        return processed_frame, reward, done
