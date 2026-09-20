import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting import save_axes_centered

df = pd.read_csv("data/real_estate_valuation.csv")

y = df["Y house price of unit area"].to_numpy()
x1 = df["X1 transaction date"].to_numpy()
x2 = df["X2 house age"].to_numpy()
x3 = df["X3 distance to the nearest MRT station"].to_numpy()
x4 = df["X4 number of convenience stores"].to_numpy()
x5 = df["X5 latitude"].to_numpy()
x6 = df["X6 longitude"].to_numpy()

x = np.column_stack([np.ones_like(x1), x1, x2, x3, x4, x5, x6])
z = np.column_stack([x1, x2, x3, x4, x5, x6])

mean_vec = z.mean(axis=0)
std_dev_vec = z.std(axis=0)

z_norm = (z - mean_vec) / std_dev_vec

x_norm = np.column_stack([np.ones_like(x1), z_norm])  
w_norm = np.zeros(x_norm.shape[1])                       

N = len(y)  # Number of data samples
alpha = 1e-2 # Learning rate

"""
Index x-vector like:

x[:,0][1] refers to 2nd element of 1-vector
x[:,1][5] refers to 6th element of x1
x[:,2][3] refers to 4th element of x2
"""

epochs = np.arange(0,500)
loss = np.zeros(len(epochs))

for i in range(len(epochs)):
    y_hat = x_norm @ w_norm
    Ln = y - y_hat    
                           
    loss[i] = (1 / N) * np.sum(Ln**2)   # Loss function

    dL = -(2 / N) * (x_norm.T @ Ln)          # Gradient of loss function
    w_norm -= alpha * dL                     # Gradient descent     

"""Calculating original weights and predicted data"""
w = np.zeros_like(w_norm)
w[0] = w_norm[0] - np.sum(w_norm[1:] * mean_vec / std_dev_vec)
w[1:] = w_norm[1:] / std_dev_vec

y_predicted = x @ w

"""Plotting"""
fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.scatter(x4, y, label='True', color='black')
ax.scatter(x4, y_predicted, label='Predicted', color='dodgerblue')

ax.set_xlabel(r'Number of Convenience Stores')
ax.set_ylabel(r'House Price per Unit Area')
ax.set_xlim(0, x4.max())

ax.grid(True, alpha=0.3, linewidth=0.6)
ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.2)
save_axes_centered(fig, ax, f"norm_multi-number-of-convenience-stores-scatter.pdf")    

plt.show()