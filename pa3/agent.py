'''
Author: Caleb Ehrisman
Course- Advanced AI CSC-549
Assignment - Programming Assignment #3 - Mountain Car

This file contains the code to implement the SARSA(lambda) algorithm.

All functions needed by solely the agent are included as member functions of class Agent
'''
import numpy as np
import fourier_basis as basis

ALPHA = 0.001
GAMMA = 1
EPSILON = 0.5
LAMBDA = 0.9


class Agent:

    def __init__(self, environment, lambda_dec=0.9, order=3):
        """
                init is the constructor for the Agent class.

                :param environment
                :return None
        """
        self.env = environment
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.lambda_decay = LAMBDA
        self.epoch_rewards = []
        self.epoch_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}
        self.epoch_max_pos = []
        self.order = order
        self.basis = basis.FourierBasis(self.env.observation_space, self.env.action_space.n, 3)
        self.lr = basis.FourierBasis.learning_rate(self.basis, self.alpha)
        self.theta = np.zeros([(order + 1) * (order + 1), self.env.action_space.n])
        self.e = np.zeros([(order + 1) * (order + 1), self.env.action_space.n])
        self.q_old = 0

    def create_q_table(self):
        """
                Agent.create_q_table creates the Q table that fits all states


                :return np.array of [x_lim][y_lim][num_actions]
        """

        high = self.env.observation_space.high
        low = self.env.observation_space.low
        num_states = (high - low) * np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1
        num_actions = self.env.action_space.n
        return np.zeros([num_states[0], num_states[1], num_actions])

    def action(self, phi):
        """
                Agent.action determines what action to take based on state

                :param phi:
                :return action
        """
        if np.random.uniform(0, 1) < EPSILON:
            action = self.env.action_space.sample()
        else:
            # disc_state = self.discretized_env_state(state)
            # print(np.dot(phi, self.theta[:, 0]))
            l = np.dot(phi, self.theta[:, 0])
            n = np.dot(phi, self.theta[:, 1])
            r = np.dot(phi, self.theta[:, 2])
            action = np.argmax([l, n, r])
        return action

    def learn(self, env, num_epochs):
        """
            Agent.learn does the actual stepping through and exploring the environment and then updates the Q_table and
            trace table.

            :param env
            :param num_epochs
            :return None
            """
        for i in range(num_epochs):
            curr_state, _ = env.reset()  # reset environment
            self.e = np.zeros([(self.order + 1) * (self.order + 1), self.env.action_space.n])
            self.q_old = 0

            # print(self.theta.shape)
            # print(basis.FourierBasis.get_features(self.basis, curr_state))
            # curr_state = self.discretized_env_state(curr_state)
            phi = basis.FourierBasis.get_features(self.basis, curr_state)  # initial phi based on initial state
            action = self.action(phi)  # initial action

            done = False
            max_pos = -99
            reward_sum = 0

            # While not finished with episode - continue
            while not done:
                next_state, reward, done, null, _ = env.step(action)  # Observe the next state

                next_phi = basis.FourierBasis.get_features(self.basis, next_state)
                next_action = self.action(next_phi)



                self.update(reward, phi, next_phi)  # Update current state based on future state

                # If the environment value state[0] is greater than equal 0.5 then it has reached the terminal state
                if next_state[0] >= max_pos:
                    max_pos = next_state[0]

                action = next_action
                reward_sum += reward
                print(reward_sum)

            #  Append max position data and reward data for evaluation
            self.epoch_max_pos.append(max_pos)
            self.epoch_rewards.append(reward_sum)

            self.terminal_output(i)

        return self.epoch_rewards, self.epoch_max_pos

    def update(self, reward, phi, next_phi):
        """
            Agent.update updates the Q table based on the SARSA algorithm. It also updates the trace table

            :param next_phi:
            :param phi:
            :param reward
            :return None
        """
        action = self.action(next_phi)
        q = np.dot(phi, self.theta)
        q_dot = np.dot(next_phi, self.theta)

        delta = reward + self.gamma * q_dot - q
        self.e[:, action] = self.gamma * self.lambda_decay * self.e[:, action] + phi - self.alpha * self.gamma * self.lambda_decay * np.dot(
            self.e[:, action], phi) * phi
        print(self.theta[:, action])
        self.theta[:, action] = self.theta[:, action] + self.alpha * (delta + q - self.q_old) * self.e[:, action] - self.alpha * (q - self.q_old) * phi
        self.q_old, phi = q_dot, next_phi

    def discretized_env_state(self, state):
        """
            Agent.discretized_env_state takes a given state and discretizes the state to use whole numbers instead of
            integers for easier computations.

            :param state
            :return discrete_state
        """
        min_states = self.env.observation_space.low
        discrete_state = (state - min_states) * np.array([10, 100])
        return np.round(discrete_state, 0).astype(int)

    def terminal_output(self, i):
        # Terminal Output for stats of each epoch
        avg_reward = sum(self.epoch_rewards[-1:]) / len(self.epoch_rewards[-1:])
        self.epoch_rewards_table['ep'].append(i)
        self.epoch_rewards_table['avg'].append(avg_reward)
        self.epoch_rewards_table['min'].append(min(self.epoch_rewards[-1:]))
        self.epoch_rewards_table['max'].append(max(self.epoch_rewards[-1:]))

        print(f"Epoch - {i}\t| avg: {avg_reward:.2f}\t| min: {min(self.epoch_rewards[-1:]):.2f}"
              f"\t| max: {max(self.epoch_rewards[-1:]):.2f}")
