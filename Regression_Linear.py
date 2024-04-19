"""
Simulating the linear regression using least-square
Creator: Ahmad Sirojuddin
mail: sirojuddin@its.ac.id
"""

import torch
import matplotlib.pyplot as plt

torch.set_printoptions(precision=3)
torch.manual_seed(123)

# ----------- Generating a toy dataset ----------- #
"""
The relation between input and output is stated as y_train(x_train) = w0 + w1 * x_train + noise
Let's set w0 = -1; w1 = 0.5; and noise is gaussian with zero mean and variance = 0.3
"""
w0 = -1
w1 = 0.5

x = torch.arange(start=-5., end=5, step=1)
print(f'x_train = {x}')
y = w0 + w1 * x + torch.randn_like(x) * torch.sqrt(torch.tensor(0))  # Try to vary the noise variance
print(f'y_train = {y}')

_, plot_linreg = plt.subplots(figsize=(6.4, 5.4))
plot_linreg.scatter(x, y)
plot_linreg.set_ylim()

# ----------- Finding the value of w0 and w1 that best fit the scatter ----------- #
X_train = torch.cat(tensors=(torch.ones_like(x).unsqueeze(dim=1),
                             x.unsqueeze(dim=1)),
                    dim=1)
print(f'X = {X_train}')
y_train = y.unsqueeze(dim=1)
print(f'y_train = {y_train}')

w_vect = torch.inverse(X_train.T @ X_train) @ X_train.T @ y_train
w0_pred = w_vect[0]
w1_pred = w_vect[1]
print(f'w0_pred = {w0_pred} | w1_pred = {w1_pred}')
y_pred = w0_pred + w1_pred * x
plot_linreg.plot(x, y_pred)

# ---------- Plot also the original data without noise ----------- #
y_ori = w0 + w1 * x
plot_linreg.plot(x, y_ori, color=(0, 100 / 255, 0))  # green line

plt.show()
