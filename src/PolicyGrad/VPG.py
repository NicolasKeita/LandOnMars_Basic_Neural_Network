import numpy as np
import tensorflow as tf
import gymnasium as gym

# Environment
env = gym.make('CartPole-v1')

# Neural Network for Policy
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Define the policy neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(output_dim, activation='softmax')
])

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

# Function to compute discounted rewards
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
        # Forward pass
        probs = model(states)
        chosen_probs = tf.reduce_sum(actions * probs, axis=1)

        # Compute the policy gradient
        policy_gradient = -tf.reduce_mean(tf.math.log(chosen_probs) * discounted_rewards)

    # Compute gradients
    grads = tape.gradient(policy_gradient, model.trainable_variables)

    # Apply the gradients to the model
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    # Collect data
    states, actions, rewards = [], [], []
    state = env.reset()
    while True:
        # Forward pass to get the action probabilities
        probs = model.predict(np.array(state).reshape(1, -1))[0]

        # Sample an action from the distribution
        action = np.random.choice(output_dim, p=probs)

        # Save state, action, and reward for later
        states.append(state)
        actions.append(tf.keras.utils.to_categorical(action, output_dim))

        # Take the action in the environment
        next_state, reward, done, _ = env.step(action)

        # Save reward
        rewards.append(reward)

        state = next_state

        if done:
            # Compute discounted rewards
            discounted_rewards = compute_discounted_rewards(rewards)

            # Train the policy
            train_policy(np.vstack(states), np.vstack(actions), discounted_rewards)

            break
