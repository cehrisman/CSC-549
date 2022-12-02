import numpy as np
import random
import math
import copy


class SarsaLambdaFA:
    def __init__(self, fa, num_actions=None, alpha=0.01, gamma=1.0, lamb=0.9, epsilon=0.5):
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.alpha = alpha

        self.num_actions = num_actions
        self.fa = []

        for i in range(0, self.num_actions):
            self.fa.append(copy.deepcopy(fa))

        self.theta = np.zeros([self.fa[0].coeff.shape[0], num_actions])
        self.lambda_weight = np.zeros(self.theta.shape)

        self.theta[0, :] = 0.0

    def action(self, state):
        """
                Agent.action determines what action to take based on state

                :param state:
                :return action
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return random.randrange(0, self.num_actions)

        best = float("-inf")
        best_actions = []
        for a in range(0, self.num_actions):
            q = self.Q(state, a)
            if math.isclose(q, best):
                best_actions.append(a)
            elif q > best:
                best = q
                best_actions = [a]

        return random.choice(best_actions)

    def Q(self, state, action):
        return np.dot(self.theta[:, action], self.fa[action].get_features(state))

    def max_Q(self, state):

        best = float("-inf")
        best_action = 0
        for a in range(0, self.num_actions):
            q = self.Q(state, a)
            if q > best:
                best = q
                best_action = a
        return best, best_action

    def update(self, state, action, reward, next_state, next_action=None, terminal=False):
        """
            Agent.update updates the Q table based on the SARSA algorithm. It also updates the trace table

            :param prev_action:
            :param action:
            :param next_phi:
            :param phi:
            :param reward
            :return None
        """

        delta = reward - self.Q(state, action)

        if not terminal:
            if next_action is not None:
                delta += self.gamma * self.Q(next_state, next_action)
            else:
                q_dot, next_action = self.max_Q(next_state)
                delta += self.gamma * self.max_Q(q_dot)

        phi = self.fa[action].get_features(state)
        phi_dot = self.fa[next_action].get_features(next_state)

        for a in range(0, self.num_actions):
            self.lambda_weight[:, a] *= self.gamma * self.lamb
            if a == action:
                self.lambda_weight[:, a] += phi
            self.theta[:, a] += self.alpha * delta * self.lambda_weight[:, a]

        return delta
