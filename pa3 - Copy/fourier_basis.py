'''
Author: Caleb Ehrisman
Course- Advanced AI CSC-549
Assignment - Programming Assignment #3 - Mountain Car
This file contains the code to implement the SARSA(lambda) algorithm.
All functions needed by solely the agent are included as member functions of class Agent
'''
import numpy as np
import itertools


class FourierBasis:
    def __init__(self, env, order):
        self.order = order
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.order = [order]*self.state_dim
        # self.coeff = self.coefficients()
        self.coeff = np.array([itr for itr in itertools.product(*[range(0, x + 1) for x in self.order])])
        self.gradient = np.array([])
        with np.errstate(divide='ignore', invalid='ignore'):
            self.gradient = 1.0 / np.linalg.norm(self.coeff, ord=2, axis=1)
            self.gradient[0] = 1.0

    def coefficients(self):
        """
            FourierBasis.coefficients creates the coeffs for the FourierBasis

            :return np.array(coeff)
        """
        coeff = [np.zeros([self.state_dim])]

        for i in range(0, self.state_dim):
            for c in range(0, self.order[i]):
                v = np.zeros(self.state_dim)
                v[i] = c + 1
                coeff.append(v)
        return np.array(coeff)

    def get_features(self, state):
        """
            FourierBasis.get_features gets the feature vector. Usually noted as x in SARSA(LAMBDA)

            :param state

            :return feature_vector
        """
        state = np.array((state - np.array([-2.4, -3, -0.2095, -3])) / np.array([4.8, 6, 0.419, 6]))
        return np.cos(np.pi * np.dot(self.coeff, state))

    def get_grad(self):
        return self.gradient


# import numpy as np
# import sys
# import itertools
#
#
# class FourierBasis:
#     def __init__(self, order: int, dimensions: int):
#         # Instance variables
#         self.coefficients = np.array([])
#         self.gradient_factors = np.array([])
#         self.dimensions = dimensions
#         self.order = [order] * self.dimensions
#
#         # create empty container for coefficient array
#         prods = [range(0, o + 1) for o in self.order]
#         coeffs = [v for v in itertools.product(*prods)]
#         self.coefficients = np.array(coeffs)
#
#         with np.errstate(divide='ignore', invalid='ignore'):
#             self.gradient_factors = 1.0 / np.linalg.norm(self.coefficients, ord=2, axis=1)
#         self.gradient_factors[0] = 1.0  # Overwrite division by zero for function with all-zero coefficients.
#
#     def get_features(self, state_vector: np.ndarray):
#         """
#         Computes basis function values at a given state.
#         """
#
#         # Bounds check state vector
#         if np.min(state_vector) < 0.0 or np.max(state_vector) > 1.0:
#             print('Fourier Basis: Given State Vector ({}) not in range [0.0, 1.0]'.format(state_vector),
#                   file=sys.stderr)
#
#         # Compute the Fourier Basis feature values
#         return np.cos(np.pi * np.dot(self.coefficients, state_vector))
#
#     def getShape(self):
#         return self.coefficients.shape
#
#     def get_grad(self):
#         return self.gradient_factors
#
#     def getGradientFactor(self, function_no):
#         return self.gradient_factors[function_no]
#
#     def length(self):
#         """Return the number of basis functions."""
#         return self.coefficients.shape[0]
