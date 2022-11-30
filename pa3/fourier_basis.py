import numpy as np
import itertools


class FourierBasis:
    def __init__(self, state_space, action_space, order, max_non_zero=2):
        self.order = order
        self.max_non_zero = min(max_non_zero, state_space.shape[0])
        self.state_dim = state_space.shape[0]
        self.coeff = self.coefficents()

    def coefficents(self):
        coeff = np.array(np.zeros(self.state_dim))

        for i in range(1, self.max_non_zero + 1):
            for indices in itertools.combinations(range(self.state_dim), i):
                for c in itertools.product(range(1, self.order + 1), repeat=i):
                    coef = np.zeros(self.state_dim)
                    coef[list(indices)] = list(c)
                    coeff = np.vstack((coeff, coef))
        return coeff

    def get_features(self, state):
        a = np.cos(np.dot(np.pi * self.coeff, state))
        print(np.array(a).shape)
        return np.cos(np.dot(np.pi * self.coeff, state))

    def get_num_basis(self):
        return len(self.coeff)

    def learning_rate(self, alpha):
        lr = np.linalg.norm(self.coeff, axis=1)
        lr[lr == 0.] = 1.
        lr = alpha/lr
        return lr