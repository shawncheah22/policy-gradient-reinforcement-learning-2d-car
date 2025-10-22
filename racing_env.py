import gymnasium as gym
import numpy as np
from collections import deque
import cv2 # OpenCV for image processing

class CarRacingEnv:
    def __init__(self, n_stack=4, img_size=(84, 84)):
        self.env = gym.make('CarRacing-v3', continuous=True)
        self.img_size = img_size
        self.n_stack = n_stack
        self.frames = deque(maxlen=self.n_stack)

    def _preprocess(self, frame):
        # 1. Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 2. Crop the bottom part (scores/text) and resize
        frame = frame[:84, 6:90] # Crop to 84x84
        frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
        # 3. Normalize pixel values
        return frame / 255.0

    def reset(self):
        """Resets the environment and returns the initial state."""
        frame, _ = self.env.reset()
        processed_frame = self._preprocess(frame)
        # For the first state, we stack the same frame n_stack times
        for _ in range(self.n_stack):
            self.frames.append(processed_frame)
        return np.array(self.frames)

    def step(self, action):
        """Takes an action and returns the next state, reward, and done flag."""
        next_frame, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        processed_frame = self._preprocess(next_frame)
        self.frames.append(processed_frame)
        
        next_state = np.array(self.frames)
        
        return next_state, reward, done
        
    def close(self):
        self.env.close()