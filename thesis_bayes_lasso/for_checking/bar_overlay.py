# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:16:54 2024

@author: ichel
"""

import matplotlib.pyplot as plt
import numpy as np

# Number of categories (lags)
n = 10  # Number of lag points
num_datasets = 6  # Number of datasets (this can be any number)

# Lags (x-values)
lags = np.arange(n)

# Generate `n` decay constants, can be linearly spaced or randomized
taus = np.linspace(2, 10, num_datasets)  # Linear spacing for tau values, or use random values
datasets = [np.exp(-lags / tau) for tau in taus]  # Exponentially decaying data
colors = plt.cm.get_cmap('tab10', num_datasets)  # Color map for `num_datasets` colors
labels = [f'Tau = {tau:.2f}' for tau in taus]  # Labels for each dataset

# Bar width
bar_width = 0.15  # Adjust this width for better spacing

# Create a plot
fig, ax = plt.subplots()

# Plot each dataset with an offset
for i, (data, color) in enumerate(zip(datasets, colors.colors)):
    # Offset the x positions for each dataset
    ax.bar(lags + i * bar_width - (num_datasets - 1) * bar_width / 2,
           data,
           width=bar_width,
           label=labels[i],
           color=color,
           alpha=0.7)  # Set transparency

# Adding labels and title
ax.set_xlabel('Lag')
ax.set_ylabel('Decay Value')
ax.set_title(f'Overlayed Decaying Bar Charts ({num_datasets} Datasets)')

# Adding a legend
ax.legend()

# Show plot
plt.show()