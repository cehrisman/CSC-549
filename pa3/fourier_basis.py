import numpy as np
import itertools


class FourierBasis:
    def __init__(self, state_space, order):
        self.order = order
        self.state_dim = state_space
        self.order = [order]*self.state_dim
        self.coeff = self.coefficents()
        self.gradient_factors = np.array([])

        with np.errstate(divide='ignore', invalid='ignore'):
            self.gradient_factors = 1.0 / np.linalg.norm(self.coeff, ord=2, axis=1)
        self.gradient_factors[0] = 1.0


    def coefficents(self):
        coeff = [np.zeros([self.state_dim])]

        for i in range(0, self.state_dim):
            for c in range(0, self.order[i]):
                v = np.zeros(self.state_dim)
                v[i] = c + 1
                coeff.append(v)
        return np.array(coeff)

    def get_features(self, state):
        # print(np.array(a).shape)
        # norm_state = (state - minimum) / (maximum - minimum)
        return np.cos(np.pi * np.dot(self.coeff, state))

    def grad_factors(self):
        return self.gradient_factors
