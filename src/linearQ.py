import numpy as np
# import gym

class LinearQAgent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99, exploration_prob=0.1):
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

    def feature_extraction(self, state, action):
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
        features = self.feature_extraction(state, action_index)
        q_value = np.dot(features, self.weights.transpose()[action_index])
        #TODO compute complexity
        return q_value

    def choose_action(self, state) -> int:
        # Epsilon-greedy policy
        if np.random.rand() < self.exploration_prob:
            return self.env.action_space_sample()
            # return self.env.action_space.sample()
        else:
            # q_values = [self.predict_q_value(state, a) for a in range(self.env.action_space.n)]
            # q_values = []
            # for action_index in range(self.env.action_space_n):
            #     q_values.append(self.predict_q_value(state, action_index))
            q_values = [self.predict_q_value(state, action_index) for action_index in range(self.env.action_space_n)]
            best_q_value_index = q_values.index(np.max(q_values))
            return best_q_value_index
            return self.env.action_space[best_q_value_index]
            # print(q_values)
            # print(q_values.index(max(q_values)))
            # print(self.env.action_space[q_values.index(max(q_values))])
            # return np.argmax(q_values)

    def update_weights(self, state, action, reward, next_state, done):
        current_q_value = self.predict_q_value(state, action)
        next_q_value = np.max([self.predict_q_value(next_state, action_index) for action_index in range(self.env.action_space_n)])
        target = reward + (1 - int(done)) * self.discount_factor * next_q_value

        features = self.feature_extraction(state, action)
        print(target, current_q_value)
        error = target - current_q_value

        self.weights += self.learning_rate * error * features[:, np.newaxis]

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                self.update_weights(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    break

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
