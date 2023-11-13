import numpy as np

from src.graph_handler import display_graph

class LinearQAgent:
    def __init__(self, env, learning_rate=0.0001, discount_factor=0.99, exploration_prob=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.weights = np.random.rand(env.feature_amount, env.action_space_n)

    def feature_extraction(self, state):
        return np.array(state)
        bias = [self.env.landing_spot[0].x, self.env.landing_spot[0].y,
                self.env.landing_spot[1].x, self.env.landing_spot[1].y
                ]
        return np.concatenate((state, bias))

    def predict_q_value(self, state, action_index):
        features = self.feature_extraction(state)
        q_value = np.dot(features, self.weights.transpose()[action_index])
        return q_value

    def choose_action(self, state) -> int:
        if np.random.rand() < self.exploration_prob:
            return self.env.action_space_sample()
        else:
            q_values = [self.predict_q_value(state, action_index) for action_index in range(self.env.action_space_n)]
            return np.argmax(q_values)

    def update_weights(self, state, chosen_action_index, reward, next_state, done):
        current_q_value = self.predict_q_value(state, chosen_action_index)
        next_state_q_values = [self.predict_q_value(next_state, action_index) for action_index in range(self.env.action_space_n)]
        next_state_best_q_value = np.max(next_state_q_values)
        gradient = self.feature_extraction(state)
        delta = reward + self.discount_factor * next_state_best_q_value - current_q_value
        # clipped_delta = np.clip(delta, -1.0, 1.0)
        # self.weights += self.learning_rate * clipped_delta * gradient[:, np.newaxis]
        print(gradient)
        self.weights[:, chosen_action_index] += self.learning_rate * delta * gradient

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            trajectory = []

            while True:
                action_index = self.choose_action(state)
                next_state, reward, done = self.env.step(action_index)
                trajectory.append((next_state[0], next_state[1]))
                self.update_weights(state, action_index, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    break
            display_graph(trajectory, episode)

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
