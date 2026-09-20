import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting import save_axes_centered

df = pd.read_csv("data/real_estate_valuation.csv")

x3 = df['X3 distance to the nearest MRT station']
x2 = df['X2 house age']

fig, ax = plt.subplots(figsize=(6.5, 4.2))

counts, bins, patches = ax.hist(
    x3, 
    color='dodgerblue',       
    edgecolor='black',      
    alpha=0.8               
)

ax.set_xlabel('Distance to Nearest MRT Station')
ax.set_ylabel('Frequency')

ax.grid(True, alpha=0.3, linewidth=0.6)
save_axes_centered(fig, ax, f"histogram-distance.pdf")    

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(6.5, 4.2))

counts, bins, patches = ax.hist(
    x2, 
    color='dodgerblue',        
    edgecolor='black',      
    alpha=0.78              
)

ax.set_xlabel('House Age')
ax.set_ylabel('Frequency')

ax.grid(True, alpha=0.3, linewidth=0.6)
save_axes_centered(fig, ax, f"histogram-house-age.pdf")    

plt.tight_layout()
plt.show()