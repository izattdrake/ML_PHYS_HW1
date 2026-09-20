import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting import save_axes_centered

df = pd.read_csv("data/real_estate_valuation.csv")

x, y = df["X2 house age"].to_numpy(), df["Y house price of unit area"].to_numpy()
N = len(x) # Number of data samples
lr = 0.002
w0, w1 = 0.0, 0.0

epochs = np.arange(0,1e4)

for i in epochs:
    y_hat = w0 + w1 * x

    Ln = y - y_hat                            # Base term used in loss rate sum 

    dL_w0 = -(2 / N) * np.sum(Ln)             # Loss function derivative wrt w0 
    dL_w1 = -(2 / N) * np.sum(Ln * x)         # Loss function derivative wrt w1

    # Adjust parameters w/ gradient descent
    w0 -= lr * dL_w0            
    w1 -= lr * dL_w1

y_predicted = w0 + w1 * x
Ln = y - y_predicted
loss = (1 / N) * np.sum(Ln**2)
print(f"Final loss: {loss}")
fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.scatter(x, y, label='True', color='black')
ax.scatter(x, y_predicted, label='Predicted', color='dodgerblue')

ax.set_xlabel(r'House Age')
ax.set_ylabel(r'House Price per Unit Area')
ax.set_xlim(0, x.max())

ax.grid(True, alpha=0.3, linewidth=0.6)
ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.2)
save_axes_centered(fig, ax, f"simple-model_scatter.pdf")    

plt.show()