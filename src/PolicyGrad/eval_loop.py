import numpy as np
from tqdm import tqdm
from src.create_environment import RocketLandingEnv
from src.graph_handler import display_graph
import tensorflow as tf

n_episodes = 1000

input_dim = 7
output_dim = 720 # or 800, or 720 ** 720 ?

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim, )),
    tf.keras.layers.Dense(output_dim, activation='softmax')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)




def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


# Function to train the policy using Vanilla Policy Gradient
def train_policy(states, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        probs = model(states)
        chosen_probs = tf.reduce_sum(actions * probs, axis=1)
        policy_gradient = -tf.reduce_mean(tf.math.log(chosen_probs + 1e-8) * discounted_rewards)
    grads = tape.gradient(policy_gradient, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def eval_loop(env: RocketLandingEnv):
    for i in range(n_episodes):

        state = env.reset()
        trajectories = []
        trajectory = [(state[0], state[1])]
        states, actions, rewards = [], [], []

        while True:
            # probs = model.predict(np.array(state).reshape(1, -1), verbose=0)[0]
            probs = model.predict(state, verbose=0)

            # action = np.random.choice(output_dim, p=probs)
            action = np.random.choice(output_dim)

            states.append(state)
            actions.append(tf.keras.utils.to_categorical(action, output_dim))

            next_state, reward, done, _ = env.step(action)

            trajectory.append((next_state[0], next_state[1]))
            rewards.append(reward)

            state = next_state
            if done:
                discounted_rewards = compute_discounted_rewards(rewards)
                train_policy(np.vstack(states), np.vstack(actions), discounted_rewards)
                break
        trajectories.append(trajectory)

        display_graph(trajectories, i)

    state = env.reset()
    solution = []
    while True:
        # Forward pass to get the action probabilities
        probs = model.predict(np.array(state).reshape(1, -1))[0]

        # Sample an action from the learned policy
        action = np.argmax(probs)

        solution.append(action)

        # Take the action in the environment
        next_state, _, done, _ = env.step(action)

        state = next_state

        if done:
            print(solution)
            print(env.action_indexes_to_real_action(solution))
            break