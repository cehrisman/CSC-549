'''
Author: Caleb Ehrisman
Course- Advanced AI CSC-549
Assignment - Programming Assignment #3 - Mountain Car

This file contains the code to implement the SARSA(lambda) algorithm.

All functions needed by solely the agent are included as member functions of class Agent
'''
import numpy as np

DISCOUNT = 0.95
ALPHA = 0.1
GAMMA = 1
EPSILON = 0.5
LAMBDA = 0.9


# EPSILON_DECREMENTER = EPSILON / (EPISODES // 4)


class Agent:

    def __init__(self, environment):
        """
                init is the constructor for the Agent class.

                :param environment
                :return None
        """
        self.E_table = None
        self.env = environment
        self.Q_table = self.create_q_table()
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.lambda_decay = LAMBDA
        self.discount = DISCOUNT
        self.epoch_rewards = []
        self.epoch_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}
        self.epoch_max_pos = []

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

    def create_e_table(self):
        """
                Agent.create_e_table creates the E table that fits all states


                :return np.array of [x_lim][y_lim][num_actions]
        """
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        num_states = (high - low) * np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1
        num_actions = self.env.action_space.n
        return np.zeros([num_states[0], num_states[1], num_actions])

    def action(self, state):
        """
                Agent.action determines what action to take based on state

                :param state
                :return action
        """
        if np.random.uniform(0, 1) < EPSILON:
            action = self.env.action_space.sample()
        else:
            # disc_state = self.discretized_env_state(state)
            action = np.argmax(self.Q_table[state[0], state[1]])
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


            curr_state = self.discretized_env_state(curr_state)
            action = self.action(curr_state)

            self.E_table = self.create_q_table()  # Create trace table. Same size as Q table

            done = False
            max_pos = -99
            reward_sum = 0

            # While not finished with episode - continue
            while not done:
                next_state, reward, done, null, _ = env.step(action)  # Observe the next state
                next_state = self.discretized_env_state(next_state)
                next_action = self.action(next_state)

                self.update(curr_state, action, reward, next_state,
                            next_action)  # Update current state based on future state

                # If the environment value state[0] is greater than equal 0.5 then it has reached the terminal state
                if next_state[0] >= max_pos:
                    max_pos = next_state[0]

                self.E_table *= self.gamma * self.lambda_decay

                curr_state = next_state
                action = next_action
                reward_sum += reward

            #  Append max position data and reward data for evaluation
            self.epoch_max_pos.append(max_pos)
            self.epoch_rewards.append(reward_sum)

            # Terminal Output for stats of each epoch
            avg_reward = sum(self.epoch_rewards[-1:]) / len(self.epoch_rewards[-1:])
            self.epoch_rewards_table['ep'].append(i)
            self.epoch_rewards_table['avg'].append(avg_reward)
            self.epoch_rewards_table['min'].append(min(self.epoch_rewards[-1:]))
            self.epoch_rewards_table['max'].append(max(self.epoch_rewards[-1:]))

            print(f"Epoch - {i}\t| avg: {avg_reward:.2f}\t| min: {min(self.epoch_rewards[-1:]):.2f}"
                  f"\t| max: {max(self.epoch_rewards[-1:]):.2f}")

        return self.epoch_rewards, self.epoch_max_pos

    def update(self, state, action, reward, next_state, next_action):
        """
            Agent.update updates the Q table based on the SARSA algorithm. It also updates the trace table

            :param state
            :param action
            :param reward
            :param next_state
            :param next_action
            :return None
        """
        target = reward + self.gamma * self.Q_table[next_state[0], next_state[1], next_action]

        error = target - self.Q_table[state[0], state[1], action]
        # print(self.E_table[state[0], state[1], action])
        self.E_table[state[0], state[1], action] += 1

        self.Q_table += 0.01 * error * self.E_table

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

    def terminal_output(self):
        pass