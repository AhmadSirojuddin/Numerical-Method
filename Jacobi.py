"""
Jacobi Method example

By: Ahmad Sirojuddin
Affiliation: Institut Teknologi Sepuluh Nopember, Indonesia
mail: sirojuddin@its.ac.id
"""

import numpy as np
import warnings

def jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray, tol: float = 1e-6, max_iter: int = 50):
    """
    The following code is the implementation of the Jacobi Method to solve the system of linear equation:
        Ax = b
    :param A: square matrix of size NxN
    :param b: vector of size N
    :param x_init: vector of size N
    :param tol: tolerance
    :param max_iter: maximum iteration
    :return: the solution x and the error
    """

    assert A.ndim == 2, f"The dimension of A must be 2. Current A.ndim = {A.ndim}"
    assert A.shape[0] == A.shape[1], f"A must be a square matrix. Current A.shape = {A.shape}"
    assert b.ndim == 1, f"The dimension of b must be 1. Current b.ndim = {b.ndim}"
    assert x_init.ndim == 1, f"The dimension of x_init must be 1. Current x_init.ndim = {x_init.ndim}"
    N = A.shape[0]
    assert b.size == N, f"The size of b must be {N}. Current b.size = {b.size}"
    assert x_init.size == N, f"The size of x_init must be {N}. Current x_init.size = {x_init.size}"
    assert np.linalg.matrix_rank(A) == N, \
        f"The current version of code only support full rank of matrix A. The current shape of A is {A.shape}," \
        f"and its rank is {np.linalg.matrix_rank(A)}"

    G = A.copy()
    G = G * (-np.eye(N) + 1)
    G = np.concatenate((np.expand_dims(b, axis=1), -G), axis=1)
    G = G / np.expand_dims(np.diag(A), axis=1)
    x = np.concatenate((np.ones(shape=1), x_init), axis=0)
    # print(f'G = {G}')
    # print(f'x = {x}')

    err = np.inf
    iter_th = 0
    while iter_th < max_iter and err > tol:
        iter_th = iter_th + 1
        print(f'iter_th = {iter_th} ---------------------')

        x = G @ x
        x = np.concatenate((np.ones(shape=1), x), axis=0)
        err = np.linalg.norm(A @ x[1:] - b)**2
        print(f'   x = {x[1:]}')
        print(f'   err = {err}')

    if iter_th == max_iter:
        warnings.warn(f'iteration number (={iter_th}) reaches the limit. The iteration may not convergence')
    return x[1:], err


# ----------- Example ----------- #
A = np.array([[10, -1, 1], [1, 10, -1], [1, 1, 10]])
b = np.array([18, 13, -7])
x_init = np.zeros(3)
x_solution, err = jacobi(A, b, x_init, tol=1e-4)
print(f'the solution to the system of linear equation is x = {x_solution} with error = {err}')

