"""
Bisection Method Example

By: Ahmad Sirojuddin
Affiliation: Institut Teknologi Sepuluh Nopember, Indonesia
mail: sirojuddin@its.ac.id
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

x = np.arange(start=-5, stop=6.1, step=0.5)

def my_func(x):
    return np.exp(0.7*x) + 2*x - 7


y = my_func(x)

plt.plot(x, y)
plt.grid(visible=True, axis='both')

def bisection(func, x_low, x_up, iter_lim, tol):
    # ----------- Checking Initial Points ----------- #
    assert x_low < x_up,\
        f'MY ERROR x_low must be smaller than x_up. \nCurrent x_low = {x_low} \nCurrent x_up = {x_up}'
    assert func(x_low) * func(x_up) < 0,\
        f'MY ERROR! The root does not lie between x_low and x_up. ' \
        f'\nfunc({x_low}) = {func(x_low)} \nfunc({x_up}) = {func(x_up)}'

    iter_th = 0
    x_root = x_low + (x_up - x_low) / 2  # update x_root using the bisection method
    y_root = func(x_root)
    print(f'x_low = {x_low:.5f} | x_root = {x_root:.5f} | x_up = {x_up:.5f} | y_root = {y_root:.5f}')
    while iter_th < iter_lim and np.abs(y_root) > tol:
        iter_th += 1
        print(f'iter_th = {iter_th} ---------------------------------------------------------')
        y_low = func(x_low)
        y_up = func(x_up)
        print(f'y_low = {y_low:.5f} | y_root = {y_root:.5f} | y_up = {y_up:.5f}')

        # shrinking the upper and lower bound
        if np.sign(func(x_root)) == np.sign(func(x_low)):
            x_low = x_root
            print(f'Current x_root becomes x_low')
        else:
            x_up = x_root
            print(f'Current x_root becomes x_up')
        x_root = x_low + (x_up - x_low) / 2  # update x_root using the bisection method
        y_root = func(x_root)
        print(f'x_low = {x_low:.5f} | x_root = {x_root:.5f} | x_up = {x_up:.5f} | y_root = {y_root:.5f}')

    if iter_th == iter_lim:
        warnings.warn(f'iteration number (={iter_th}) reaches the limit. The iteration may not convergence')

    return x_root, y_root


x_solution, _ = bisection(func=my_func, x_low=-3, x_up=5, iter_lim=15, tol=1e-3)
print(f'THE SOLUTION TO THE ROOT OF EQUATION PROBLEM: x = {x_solution}')

plt.show()
