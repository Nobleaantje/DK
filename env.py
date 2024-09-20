import gymnasium as gym
import numpy as np
from racer.car_info import CarInfo
from racer.track import Track

DT = 0.016666 # 60 FPS

def clamp(v: float, lo: float, hi: float):
    return max(lo, min(v, hi))

class RacingEnv(gym.Env):
    def __init__(self, render=False):
        super(RacingEnv, self).__init__()

        # Game State logic
        self.frames = 0  # type: int # Number of frames since the start of the game
        self.track = Track()  # Initialize the track
        self.bot = CarInfo(self.track)  # Initialize the bot

        # RL logic
        # Define action and observation space
        # Assuming 2 discrete actions: [throttle, steering]
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        # Define observation space
        # Assuming observation is a vector of game state features
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
    def reset(self):
        # Game logic
        self.frames = 0
        
        # RL logic
        observation = self._get_observation()
        return observation

    def step(self, action):
        # Rl logic
        throttle, steering_command = action

        # Game logic
        self.frames += 1
        throttle = clamp(throttle, -1, 1)
        steering_command = clamp(steering_command, -1, 1)
        self.bot.update(self.frames * DT, DT, throttle, steering_command)
        
        # RL logic
        observation = self._get_observation()
        reward = self._get_reward()
        done = self.game_state.is_done()
        info = self.game_state.get_info()
        return observation, reward, done, info

    # RL logic
    def _get_observation(self):
        # Extract relevant features from the game state
        return np.array(self.game_state.get_features(), dtype=np.float32)

    def _get_reward(self):
        # Define reward logic based on game state
        return self.game_state.get_reward()

# Example usage
if __name__ == "__main__":
    env = RacingEnv(render=True)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with your bot's action logic
        obs, reward, done, info = env.step(action)
        env.render()