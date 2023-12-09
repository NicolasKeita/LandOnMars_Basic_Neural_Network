import json

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

from src.DDPG_HER_stable_baseline.create_environment import RocketLandingEnv
from stable_baselines3 import HerReplayBuffer, DDPG, TD3, SAC, PPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
# import gymnasium

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "final"  # equivalent to GoalSelectionStrategy.FUTURE
np.set_printoptions(suppress = True)

env = RocketLandingEnv()
action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))
# env = gymnasium.wrappers.FlattenObservation(env)
# print(env)

model = TD3(
    "MultiInputPolicy",
    env,
    # action_noise=action_noise,
    # learning_rate=3e-7,
    replay_buffer_class=HerReplayBuffer,
    # policy_kwargs=dict(net_arch=[78]),
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=GoalSelectionStrategy.FINAL,
    ),
    verbose=2,
)



# env = RocketLandingEnv()

# model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[350]),  verbose=1)
# model = PPO("MlpPolicy", env)
# model.learn(total_timesteps=1400_000)

def save_model(model):
    params = model.get_parameters()
    policy_net_weights = params.get('policy', None)
    json_path = "policy_net_weights.json"
    with open(json_path, 'w') as json_file:
        for key, value in policy_net_weights.items():
            policy_net_weights[key] = value.tolist()
        keys_to_remove = ['value_net.weight', 'value_net.bias',
                          'mlp_extractor.value_net.0.weight',
                          'mlp_extractor.value_net.0.bias', 'log_std',
                          "mlp_extractor.value_net.2.weight",
                          "mlp_extractor.value_net.2.bias",
                          "critic_target.qf0.2.bias",
                          "critic_target.qf0.2.weight",
                          "critic_target.qf1.4.bias",
                          "critic_target.qf1.4.weight"
        ]
        for key in keys_to_remove:
            policy_net_weights.pop(key, None)

        json.dump(policy_net_weights, json_file)

# save_model(model)
model.learn(100000)
print("Learning finished!")
exit(0)
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
