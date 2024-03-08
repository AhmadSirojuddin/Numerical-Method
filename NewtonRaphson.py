"""
Newton-Raphson Method example

By: Ahmad Sirojuddin
Affiliation: Institut Teknologi Sepuluh Nopember, Indonesia
mail: sirojuddin@its.ac.id
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

x = np.arange(start=-5, stop=5.1, step=0.5)

def my_func(x):
    return np.exp(0.7*x) + 2*x - 7

def my_func_derivative(x):
    return 0.7*np.exp(0.7*x) + 2


y = my_func(x)

plt.plot(x, y)
plt.grid(visible=True, axis='both')

def newton_raphson(func, func_derivative, x_init, iter_lim, tol):
    iter_th = 0
    y = func(x_init)
    print(f'x_init = {x_init:.5f} | y = {y:.5f}')

    x = x_init
    while iter_th < iter_lim and np.abs(y) > tol:
        iter_th += 1
        print(f'iter_th = {iter_th} ---------------------------------------------------------')

        x = x - func(x) / func_derivative(x)  # update x using the Newton-Raphson method
        y = func(x)
        print(f'x = {x:.5f} | y = {y}')

    if iter_th == iter_lim:
        warnings.warn(f'iteration number (={iter_th}) reaches the limit. The iteration may not convergence')

    return x, y


x_solution, _ = newton_raphson(func=my_func, func_derivative=my_func_derivative, x_init=5., iter_lim=15, tol=1e-3)
print(f'THE SOLUTION TO THE ROOT OF EQUATION PROBLEM: x = {x_solution}')

plt.show()
