import os
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define the Q-network
def build_q_network(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_experience(self, experience):
        print("ADDING ! ")
        print(experience)
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        return np.random.choice(self.buffer, batch_size)

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, env, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = build_q_network(state_size, action_size)
        self.target_network = build_q_network(state_size, action_size)
        self.target_network.set_weights(self.q_network.get_weights())

        self.replay_buffer = ReplayBuffer(buffer_size=1000)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def train(self, batch_size=32):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        batch = self.replay_buffer.sample_batch(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Calculate target Q-values using the target network
        target_q_values = self.target_network.predict(np.vstack(next_states))
        targets = rewards + (1 - np.array(dones)) * self.gamma * np.max(target_q_values, axis=1)

        # Update the Q-network
        self.q_network.fit(np.vstack(states), targets, epochs=1, verbose=0)

        # Update the epsilon parameter
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update the target network weights periodically
        if len(self.replay_buffer.buffer) % 10 == 0:
            self.target_network.set_weights(self.q_network.get_weights())

    # TODO rename
    def pre_train(self):
        episodes = 200
        for episode in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                self.replay_buffer.add_experience((state, action, reward, next_state, done))
                self.train()

                total_reward += reward
                state = next_state

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
