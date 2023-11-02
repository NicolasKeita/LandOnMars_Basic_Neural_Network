import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape

# Define the CartPole environment
env = gym.make('CartPole-v1')

# Define the CNN-based Q-learning agent
class QLearningAgent:
    def __init__(self, state_shape, action_space, learning_rate=0.001, discount_factor=0.99):
        self.action_space = action_space
        self.discount_factor = discount_factor

        # Create a CNN model
        self.model = Sequential()
        self.model.add(Reshape((state_shape[0], 1), input_shape=state_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(action_space, activation='linear'))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')

    def select_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_space - 1)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            q_next = np.max(self.model.predict(next_state)[0])
            target = reward + self.discount_factor * q_next
        q_values = self.model.predict(state)
        q_values[0][action] = target
        self.model.fit(state, q_values, epochs=1, verbose=0)

# Initialize the Q-learning agent
state_shape = (4, 1)  # Correct the input shape to match the Reshape layer
action_space = env.action_space.n
agent = QLearningAgent(state_shape, action_space)

# Training the agent (the rest of the code remains the same)

# Test the agent on a new state
new_state = np.random.rand(4, 1)
action = agent.select_action(new_state, epsilon)
print(f"Agent selects action {action} for the new state: {new_state}")
