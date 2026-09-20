import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from part_b.plotting import save_axes_centered

df = pd.read_csv("data/real_estate_valuation.csv")

x, y = df["X2 house age"].to_numpy(), df["Y house price of unit area"].to_numpy()
N = len(x) # Number of data samples

learning_rates = [0.002, 0.001, 0.0005, 0.0001, 0.00001]

losses = [[] for _ in range(len(learning_rates))]
epoch = np.arange(0,1e4)

for j in range(len(learning_rates)):
    w0, w1 = 0.0, 0.0

    lr = learning_rates[j]
    for i in epoch:
        y_hat = w0 + w1 * x

        Ln = y - y_hat                            # Base term used in loss rate sum 

        # Loss function
        losses[j].append((1 / N) * np.sum(Ln**2))      # Loss function
        dL_w0 = -(2 / N) * np.sum(Ln)             # Loss function derivative wrt w0 
        dL_w1 = -(2 / N) * np.sum(Ln * x)         # Loss function derivative wrt w1

        # Adjust parameters w/ gradient descent
        w0 -= lr * dL_w0            
        w1 -= lr * dL_w1

fig, ax = plt.subplots(figsize=(6.5, 4.2))

for i in range(len(learning_rates)):
    ax.plot(epoch, losses[i], label=f'lr = {learning_rates[i]}')

ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Loss')
ax.set_xlim(0, epoch.max())

ax.set_yscale('log')

ax.grid(True, alpha=0.3, linewidth=0.6)
ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.2)
save_axes_centered(fig, ax, f"Lr_comparison_simple.pdf")    

plt.show()