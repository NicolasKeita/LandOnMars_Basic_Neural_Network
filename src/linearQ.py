import numpy as np

from src.graph_handler import display_graph


# import gym

class LinearQAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_prob=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Initialize weights for linear function approximation
        # print(env.observation_space)
        # print(env.observation_space.shape)
        # self.weights = np.random.rand(env.observation_space.shape[0] + 1, env.action_space.n)
        self.weights = np.random.rand(env.feature_amount, env.action_space_n)
        # print(env.action_space.n)

    def feature_extraction(self, state):
        # Simple feature extraction for CartPole environment
        # features = np.concatenate((state, [1]))  # Adding a constant feature
        # features = np.concatenate((state, self.env.landing_spot[0].x))
        # features = np.concatenate((state, self.env.landing_spot[0].y))
        # features = np.concatenate((state, self.env.landing_spot[1].x))
        # features = np.concatenate((state, self.env.landing_spot[1].y))
        # TODO this is called adding bias
        features = np.concatenate((state, [self.env.landing_spot[0].x, self.env.landing_spot[0].y]))
        features = np.concatenate((features, [self.env.landing_spot[1].x, self.env.landing_spot[1].y]))
        return features

    def predict_q_value(self, state, action_index):
        features = self.feature_extraction(state)
        q_value = np.dot(features, self.weights.transpose()[action_index])
        #TODO compute complexity
        return q_value

    def choose_action(self, state) -> int:
        if np.random.rand() < self.exploration_prob:
            sample = self.env.action_space_sample()
            if sample == 720:
                raise
            return sample
        else:
            q_values = [self.predict_q_value(state, action_index) for action_index in range(self.env.action_space_n)]
            index = np.argmax(q_values)
            if index == 720:
                raise
            return index

    def update_weights(self, state, chosen_action_index, reward, next_state, done):
        current_q_value = self.predict_q_value(state, chosen_action_index)
        next_state_best_q_value = np.max([self.predict_q_value(next_state, action_index) for action_index in range(self.env.action_space_n)])
        target = reward + (1 - int(done)) * self.discount_factor * next_state_best_q_value

        gradient = self.feature_extraction(state)
        # print(target, current_q_value)
        error = target - current_q_value
        delta = reward + self.discount_factor * next_state_best_q_value - current_q_value
        clipped_delta = np.clip(delta, -1.0, 1.0)
        clipped_delta = delta
        # print(delta)
        # print(self.weights)
        self.weights += self.learning_rate * clipped_delta * gradient[:, np.newaxis]
        return;
        # print(self.weights[0][:3])
        # for feature_index in range(len(self.weights)):
        #     print("Feature: ", feature_index)
        #     print(self.weights[feature_index][:3])
        #     self.weights[feature_index] = self.weights[feature_index] + self.learning_rate * delta * gradient[feature_index]
        #     print(gradient[feature_index])
        #     print(self.weights[feature_index][:3])
            # exit(9)
        # print(self.weights)
        # print("UPDATE DONE")
        # return
        exit(5)
        print(gradient)
        print(gradient[:, np.newaxis])

        print(self.weights)
        print(self.weights.shape)
        print(np.dot(self.learning_rate * (reward + self.discount_factor * next_state_best_q_value - current_q_value), gradient))
        print(self.learning_rate * (reward + self.discount_factor * next_state_best_q_value - current_q_value) * gradient)
        print(self.weights[chosen_action_index] + np.dot(self.learning_rate * (reward + self.discount_factor * next_state_best_q_value - current_q_value), gradient))
        print(self.weights[chosen_action_index] + self.learning_rate * (reward + self.discount_factor * next_state_best_q_value - current_q_value) * gradient)
        # self.weights = self.weights + self.learning_rate * error * features[:, np.newaxis]
        # print(self.weights)
        exit(50)

        self.weights = self.weight + self.learning_rate * error * gradient[:, np.newaxis]

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            # trajectory = []

            while True:
                action_index = self.choose_action(state)
                next_state, reward, done = self.env.step(action_index)
                # trajectory.append((next_state[0], next_state[1]))
                self.update_weights(state, action_index, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    break
            # display_graph(trajectory, episode)

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
