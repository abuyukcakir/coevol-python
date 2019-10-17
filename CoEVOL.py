import networkx as nx
import numpy as np
import scipy as sp
import pandas as pd
import math
import matplotlib.pyplot as plt
from networkx.readwrite.edgelist import read_edgelist

class CoEVOL():
    '''
    Shared Temporal Matrix Factorization with CoEVOL
    as described in Yu et al. 2018 - 'Modeling CoEvolution
    Across Multiple Networks'

    Inputs
    ----------
    A_T_all: Tensor of size (T, s, n, n) where
        T: num of time stamps
        s: num of subjects
        n: matrix size

    Parameters
    -----------
    k: latent dimension size
    theta: decaying factor
    '''

    def __init__(self, A_T_all, k: int = 10, theta: float = 0.3,
                 alpha: float = 0.01, beta: float = 0.01, gamma: float = 0.01):

        self.A = A_T_all
        self.T = self.A.shape[0]
        self.s = self.A.shape[1]
        self.n = self.A.shape[2]
        self.k = k

        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        print('CoEVOL model with:\n Number of time steps={}\n \
        Number of Subjects={}\nMatrix Size={} '.format(self.T, self.s, self.n))

        # We assume V(t) = X*t + Y
        self.U = np.random.rand(self.n, self.k)
        self.X = np.random.rand(self.s, self.n, self.k)
        self.Y = np.random.rand(self.s, self.n, self.k)

        return

    def factorize(self):
        """
        Main loop described in Algorithm 1 in the paper.
        """
        num_iter = 0
        MAX_ITER = 500

        # Originally, lambda is found by Line Search
        lamda = 0.0001

        for i in range(MAX_ITER):
            dJdU, dJdX, dJdY = self.get_derivatives()

            # Update with gradient descent operation
            # Notice lamda is not found by Line Search in this implementation.
            self.U -= np.multiply(lamda, dJdU)

            for r in range(self.s):
                self.X[r] -= np.multiply(lamda, dJdX[r])
                self.Y[r] -= np.multiply(lamda, dJdY[r])

            num_iter += 1

            # Report the reconstruction error at the end of the iteration
            print('Error at the end of iteration {}:'.format(num_iter))
            sum_error = self.get_reconstruction_error()
            print("Total Error: {}".format(sum_error))

        # print('Before nonnegative projection')
        # print(self.X)
        # print(self.Y)

        # Apply nonnegative projection onto X and Y
        # i.e. make negative entries zero (similar to ReLU)
        for r in range(self.s):
            self.X = self.X.clip(min=0)
            self.Y = self.Y.clip(min=0)

        # print('After nonnegative projection')
        # print(self.X)
        # print(self.Y)
        return

    def get_derivatives(self):
        """
        Calculates partial derivatives of the error function
        wrt U, X and Y. Equations described in 2.8, 2.9, and 2.10 in the paper.
        """
        dJdU = np.zeros((self.n, self.k))
        dJdX = np.zeros((self.s, self.n, self.k))
        dJdY = np.zeros((self.s, self.n, self.k))

        for t in range(self.T):
            for r in range(self.s):
                error = self.get_error(t, r)

                # print('aaaaaaa')
                # Inner term for dJdU
                dJdU += np.dot(error, ( -1 * self.X[r] * t - self.Y[r]) )

                dJdX[r] += np.dot(error.T, (-self.U * t))

                dJdY[r] += np.dot(error.T, (-self.U))

                # print('bbbbbbb')

            # print('cccccc')
            dJdU *= math.pow(math.e, (-self.theta * (self.T - t) ))
            dJdU += self.alpha * self.U     # Regularization's effect on U

            # print('11111')

            for r in range(self.s):     # Regularization's effect on X and Y
                # print('222222')
                dJdX[r] *= math.pow(math.e, (-self.theta * (self.T - t) ))
                dJdY[r] *= math.pow(math.e, (-self.theta * (self.T - t) ))

                # print('333333')
                dJdX[r] += self.beta * self.X[r]
                dJdY[r] += self.gamma * self.Y[r]

        return dJdU, dJdX, dJdX

    def get_reconstruction_error(self):
        """
        Get total error as described in Equation 2.4 in the paper.
        """
        error_over_time = np.zeros(self.T)
        for t in range(self.T):
            for r in range(self.s):
                error_over_time[t] += np.linalg.norm( self.get_error(t, r) )

        decayed_error_over_time = self.decay(error_over_time)

        # Regularization errors
        reg_U = 0.5 * self.alpha * np.linalg.norm(self.U)
        reg_X = 0.5 * self.beta * np.linalg.norm(self.X)
        reg_Y = 0.5 * self.gamma * np.linalg.norm(self.Y)

        # print(decayed_error_over_time)
        print(reg_U)
        print(reg_X)
        print(reg_Y)

        sum_error = np.linalg.norm(decayed_error_over_time) + reg_U + reg_X + reg_Y
        return sum_error

    def get_error(self, t, r):
        """
        Returns the error of subject r at time t. Equation 2.7 in the paper.
        """
        error = self.A[t, r] - (self.U.dot( (self.X[r] * t + self.Y[r]).T ) )

        return error

    def decay(self, V):
        '''
        Given a vector of size T (time),
        Multiple all entries with diminishing factors (where
        the latest entry will not be diminished)
        theta > 0
        '''
        T = len(V)
        V_new = np.zeros_like(V)
        for i in range(T):
            V_new[T-1 - i] = V[T-1 - i] * math.pow(math.e, (-i * self.theta)) / 2.0

        print(V_new)

        return V_new

    def get_factors(self):
        """
        Getter for the learned factors U and V (where V is a linear
        combination of X and Y)
        """
        return self.U, self.X, self.Y


# TEST
A = np.random.randint(2, size=(20, 3, 10, 10))
print(np.linalg.norm(A))
print(A)

coevol = CoEVOL(A, k=4)
coevol.factorize()
