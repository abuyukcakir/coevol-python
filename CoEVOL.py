import networkx as nx
import numpy as np
import scipy as sp
import pandas as pd
import math
import matplotlib.pyplot as plt
from networkx.readwrite.edgelist import read_edgelist
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import scipy.sparse as sps

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
                 alpha: float = 0.01, beta: float = 0.01, gamma: float = 0.01,
                 threshold : float = 0.01):

        self.A = A_T_all
        self.T = self.A.shape[0]
        self.s = self.A.shape[1]
        self.n = self.A[0,0].shape[0]
        self.k = k

        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold

        print('CoEVOL model with:\n Number of time steps={}\n \
        Number of Subjects={}\nMatrix Size={} '.format(self.T, self.s, self.n))

        # We assume V(t) = X*t + Y
        self.U = np.random.rand(self.n, self.k)
        self.X = np.random.rand(self.s, self.n, self.k)
        self.Y = np.random.rand(self.s, self.n, self.k)
        self.error = np.empty((self.T, self.s), dtype=object) # sparse error matrices

        return

    def factorize(self):
        """
        Main loop described in Algorithm 1 in the paper.
        """
        num_iter = 0
        MAX_ITER = 500

        # Originally, lambda is found by Line Search
        lamda = 0.00001

        prev_error = math.inf
        cur_error = 0.0

        for i in range(MAX_ITER):
            dJdU, dJdX, dJdY = self.get_derivatives()

            # Update with gradient descent operation
            # Notice lamda is not found by Line Search in this implementation.
            self.U -= np.multiply(lamda, dJdU)

            for r in range(self.s):
                self.X[r] -= np.multiply(lamda, dJdX[r])
                self.Y[r] -= np.multiply(lamda, dJdY[r])

            # Apply nonnegative projection onto X and Y
            # i.e. make negative entries zero (similar to ReLU)
            for r in range(self.s):
                self.X = self.X.clip(min=0)
                self.Y = self.Y.clip(min=0)

            num_iter += 1

            # Report the reconstruction error at the end of the iteration
            print('At the end of iteration {},'.format(num_iter))
            cur_error = self.get_reconstruction_error()
            print("Total Error: {}".format(cur_error))

            if(abs(cur_error - prev_error) < self.threshold):
                # Stopping criterion met.
                print('Convergence is reached. Error is:')
                print(cur_error)
                break
            # Otherwise, continue running
            prev_error = cur_error

        print('Execution completed. Error is:')
        print(cur_error)
        print('Factorization completed for k = {}, theta = {}.'.format(self.k, self.theta))

        return self.U, self.X, self.Y

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
                dJdU += error.dot( -1 * self.X[r] * t - self.Y[r])

                dJdX[r] += error.transpose().dot(-self.U * t)

                dJdY[r] += error.transpose().dot(-self.U)

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
                error_over_time[t] += self.get_rmse(t, r)


        decayed_error_over_time = self.decay(error_over_time)

        # Regularization errors
        reg_U = 0.5 * self.alpha * np.linalg.norm(self.U)
        reg_X = 0.5 * self.beta * np.linalg.norm(self.X)
        reg_Y = 0.5 * self.gamma * np.linalg.norm(self.Y)

        # print(decayed_error_over_time)
        # print(reg_U)
        # print(reg_X)
        # print(reg_Y)

        sum_error = np.linalg.norm(decayed_error_over_time) + reg_U + reg_X + reg_Y
        return sum_error

    def get_rmse(self, t, r):
        '''
        For the given snapshot, return RMSE of the latest iteration
        '''
        # Get the nonzero indices
        nnz_ind, _ = self.error[t, r].nonzero()

        # Calculate RMSE, using the number of nonzero indices
        rmse = sps.linalg.norm(self.error[t, r]) / math.sqrt(len(nnz_ind))
        return rmse

    def get_error(self, t, r):
        """
        Returns the error of subject r at time t. Equation 2.7 in the paper.
        """
        # Get the nonzero indices of the current A matrix. Only for those indices
        # We should add up errors.
        nz_i, nz_j = self.A[t,r].nonzero()

        err = lil_matrix((self.n, self.n))    # Empty sparse matrix NxN
        reconst = self.U.dot( (self.X[r] * t + self.Y[r]).T )

        for k in range(len(nz_i)):  # for each nonzero index
            err[nz_i[k], nz_j[k]] = self.A[t, r][nz_i[k], nz_j[k]] - reconst[nz_i[k], nz_j[k]]
            self.error[t, r] = err

        # error = self.A[t, r] - (self.U.dot( (self.X[r] * t + self.Y[r]).T ) )
        # print(sps.linalg.norm(error))

        return self.error[t, r]

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

        # print(V_new)

        return V_new

    def get_factors(self):
        """
        Getter for the learned factors U and V (where V is a linear
        combination of X and Y)
        """
        return self.U, self.X, self.Y


# TEST
# A = np.empty((20, 3), dtype=object)
# for i in range(20):
#     for j in range(3):
#         A[i, j] = csr_matrix([[0,0,1],[0,1,1],[0,0,0]])

# print(A[0,0].shape)

# coevol = CoEVOL(A, 3, k=4)
# coevol.factorize()
