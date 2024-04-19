"""
Simulating the polynomial regression using least-square
Creator: Ahmad Sirojuddin
mail: sirojuddin@its.ac.id
"""

import torch  # numpy
import matplotlib.pyplot as plt

torch.set_printoptions(precision=3)
torch.manual_seed(123)

# ----------- Generating a toy dataset ----------- #
"""
The relation between input and output is stated as y_train(x_train) = w0 + w1 * x_train + w2 * x_train**2 + w3 * x_train**3 + w4 * x_train**4 + noise
Let's set w0 = 4; w1 = 0.6; and noise is gaussian with zero mean and variance = 0.5
"""
w0 = 32
w1 = 5.6
w2 = -5.4
w3 = 0.1
w4 = 0.1
x = torch.arange(start=-9., end=7.5, step=0.5)
print(f'x_train = {x}')
y = w0 + w1 * x + w2 * x**2 + w3 * x**3 + w4 * x**4 + torch.randn_like(x) * torch.sqrt(torch.tensor(49))
print(f'y_train = {y}')
_, plot_linreg = plt.subplots(figsize=(6.4, 5.4))
plot_linreg.scatter(x, y)

# ----------- Finding the value of w0 until w4 that best fit the scatter ----------- #
X_train = torch.concatenate(tensors=(torch.ones_like(x).unsqueeze(dim=1),
                                     torch.unsqueeze(x, dim=1),
                                     torch.unsqueeze(x**2, dim=1),
                                     torch.unsqueeze(x**3, dim=1),
                                     torch.unsqueeze(x**4, dim=1)),
                            dim=1)
print(f'X_train = {X_train}')
y_train = y.unsqueeze(dim=1)
a_vect = torch.inverse(X_train.T @ X_train) @ X_train.T @ y_train  # The formula for finding w1 and w0 that best fit the scatter
w0_pred = a_vect[0]
w1_pred = a_vect[1]
w2_pred = a_vect[2]
w3_pred = a_vect[3]
w4_pred = a_vect[4]
print(f'w0_pred = {w0_pred} | w1_pred = {w1_pred} | w2_pred = {w2_pred} | w3_pred = {w3_pred} | w4_pred = {w4_pred}')
y_pred = w0_pred + w1_pred * x + w2_pred * x**2 + w3_pred * x**3 + w4_pred * x**4
plot_linreg.plot(x, y_pred)

# ---------- Plot also the original data without noise ----------- #
y_ori = w0 + w1 * x + w2 * x**2 + w3 * x**3 + w4 * x**4
plot_linreg.plot(x, y_ori, color=(0, 100 / 255, 0))

plt.show()
