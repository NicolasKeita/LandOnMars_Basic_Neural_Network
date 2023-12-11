import json

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

from src.TD3_SB3.create_environment import RocketLandingEnv
from stable_baselines3 import HerReplayBuffer, DDPG, TD3, SAC, PPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import gymnasium
import matplotlib

matplotlib.use('Qt5Agg')

# Available strategies (cf paper): future, final, episode
np.set_printoptions(suppress=True)

env = RocketLandingEnv()
# env = gymnasium.make("LunarLander-v2", continuous=True, render_mode="human")
# env = gymnasium.make("LunarLander-v2", continuous=True)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

if False:
    model = TD3.load("./her_bit_env", env=env)
else:
    model = TD3("MlpPolicy",
                env,
                # action_noise=action_noise,
                # learning_rate=0.001,
                # policy_kwargs=dict(net_arch=[256, 516, 256]),
                verbose=2)


def save_model(model):
    params = model.get_parameters()
    policy_net_weights = params.get('policy', None)
    json_path = "policy_net_weights.json"
    with open(json_path, 'w') as json_file:
        for key, value in policy_net_weights.items():
            policy_net_weights[key] = value.tolist()
        policy_net_weights = {key: value for key, value in policy_net_weights.items() if 'critic' not in key}

        json.dump(policy_net_weights, json_file)


save_model(model)
model.learn(1_000_000)
print("Learning finished!")
model.save("./her_bit_env")
save_model(model)
exit(0)
# TODO check feature extractor by stable baseline
vec_env = model.get_env()
obs = vec_env.reset()

print("EVALUATION !")
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print('action after predict:', action)
    action_to_do = np.copy(action[0])
    action_to_do[0] = action_to_do[0] * 90
    action_to_do[1] = action_to_do[1] * 4
    action_to_do = np.round(action_to_do)
    action_to_do = action_to_do.reshape(-1, 2)
    action_to_do = np.squeeze(action_to_do)
    print(action_to_do)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
