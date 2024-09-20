# %%
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from env import RacingEnv  # Ensure this import matches the location of your RacingEnv class

# Create the environment
env = RacingEnv(render=False)

# %%
# Check the environment
check_env(env)

# Wrap the environment in a DummyVecEnv
env = DummyVecEnv([lambda: env])

# Define the model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_racing_env")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Load the model
model = PPO.load("ppo_racing_env")

# Enjoy the trained agent
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()