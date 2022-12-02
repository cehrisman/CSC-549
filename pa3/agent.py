'''
Author: Caleb Ehrisman
Course- Advanced AI CSC-549
Assignment - Programming Assignment #3 - Mountain Car

This file contains the code to implement the SARSA(lambda) algorithm.

All functions needed by solely the agent are included as member functions of class Agent
'''
import random

import numpy as np
from fourier_basis import FourierBasis
import math
from sarsalambdaFA import SarsaLambdaFA

ALPHA = 0.0001
GAMMA = 1
EPSILON = 0.5
LAMBDA = 0.9


class Agent:

    def __init__(self, environment, order=3, runs=1, gamma=0.001):
        """
                init is the constructor for the Agent class.

                :param environment
                :return None
        """
        self.runs = runs
        self.order = order
        self.env = environment
        self.gamma = gamma
        self.num_actions = self.env.action_space.n
        self.state_dims = self.env.observation_space.shape[0]
        self.epoch_rewards = []
        self.epoch_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}
        self.epoch_max_pos = []

    def learn(self, env, num_epochs):
        """
            Agent.learn does the actual stepping through and exploring the environment and then updates the Q_table and
            trace table.

            :param env
            :param num_epochs
            :return None
            """
        for run in range(0, self.runs):
            fb = FourierBasis(state_space=self.env.observation_space.shape[0], order=self.order)
            learner = SarsaLambdaFA(fa=fb, num_actions=self.num_actions, alpha=0.0001, epsilon=0.8)

            for i in range(num_epochs):

                learner.epsilon *= .99
                learner.lambda_weight = np.zeros(learner.theta.shape)
                state, _ = env.reset()
                state = (state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
                action = learner.action(state)
                done = False
                steps = 0
                reward_sum = 0
                max_pos = 0.5
                while not done:
                    next_state, reward, done, info, _ = env.step(action)

                    if next_state[0] >= max_pos:
                        max_pos = next_state[0]

                    next_state = (next_state - env.observation_space.low) / (
                                env.observation_space.high - env.observation_space.low)


                    next_action = learner.action(next_state)

                    learner.update(state, action, reward, next_state, next_action, done)
                    state = next_state
                    action = next_action
                    reward_sum += reward
                    # print(reward_sum)
                #  Append max position data and reward data for evaluation
                self.epoch_rewards.append(reward_sum)
                self.epoch_max_pos.append(max_pos)

                self.terminal_output(i)

        return self.epoch_rewards, self.epoch_max_pos

    def terminal_output(self, i):
        # Terminal Output for stats of each epoch
        avg_reward = sum(self.epoch_rewards[-2:]) / len(self.epoch_rewards[-2:])
        self.epoch_rewards_table['ep'].append(i)
        self.epoch_rewards_table['avg'].append(avg_reward)
        self.epoch_rewards_table['min'].append(min(self.epoch_rewards))
        self.epoch_rewards_table['max'].append(max(self.epoch_rewards))

        print(f"Epoch - {i}\t| avg: {avg_reward:.2f}\t| min: {min(self.epoch_rewards[-1:]):.2f}"
              f"\t| max: {max(self.epoch_rewards[-1:]):.2f}")
