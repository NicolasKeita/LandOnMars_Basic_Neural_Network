import json

import numpy as np

from src.PPO_stable_baseline.create_environment import RocketLandingEnv
from stable_baselines3 import PPO

env = RocketLandingEnv()

model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[350]),  verbose=1)
# model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1400_000)
# model.save("model_ppo_stable_baseline")
#https://stable-baselines3.readthedocs.io/en/master/guide/export.html#manual-export

params = model.get_parameters()
policy_net_weights = params.get('policy', None)
json_path = "policy_net_weights.json"
with open(json_path, 'w') as json_file:
    for key, value in policy_net_weights.items():
        policy_net_weights[key] = value.tolist()
    keys_to_remove = ['value_net.weight', 'value_net.bias', 'mlp_extractor.value_net.0.weight',
                      'mlp_extractor.value_net.0.bias', 'log_std', "mlp_extractor.value_net.2.weight", "mlp_extractor.value_net.2.bias"]
    for key in keys_to_remove:
        policy_net_weights.pop(key, None)

    json.dump(policy_net_weights, json_file)

#TODO check feature extractor by stable baseline
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
