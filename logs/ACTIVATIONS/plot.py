import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 

with open("activation_log_KL_250.json", "r") as f:
    log = json.load(f)

# Determine global min and max for consistent axes
epochs_all = []
kld_min, kld_max = float("inf"), float("-inf")

for im in range(20):
    if im == 10: continue
    ilog = {(int(epoch) + 1): values for epoch, values in log[str(im)].items()}
    epochs = sorted(ilog.keys())
    real = [ilog[epoch][0] for epoch in epochs]
    rand = [ilog[epoch][1] for epoch in epochs]
    epochs_all.extend(epochs)
    kld_min = min(kld_min, min(real + rand))
    kld_max = max(kld_max, max(real + rand))

x_min, x_max = min(epochs_all), max(epochs_all)

# Generate individual plots
for im in range(20):
    if im == 10: continue
    ilog = {(int(epoch) + 1): values for epoch, values in log[str(im)].items()}
    epochs = sorted(ilog.keys())
    real = [ilog[epoch][0] for epoch in epochs]
    rand = [ilog[epoch][1] for epoch in epochs]
    
    plt.figure(figsize=(6, 4))  # Adjust figure size as needed
    plt.plot(epochs, real, label="IMP", color="blue")
    plt.plot(epochs, rand, label="RANDOM", color="orange")
    
    plt.yscale('log')
    plt.xlim(x_min, x_max)
    plt.ylim(kld_min * 0.5, kld_max * 1.5)
    plt.title(f"{(100 * 0.8 ** im):.2f}% Sparsity.")
    
    plt.xlabel("Epoch")
    plt.ylabel("Fitness (LOG)")
    plt.legend()
    #plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    plt.savefig(f"plot_{im:02d}.png")
    plt.close()


"""import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np 

with open("activation_log_KL_250.json", "r") as f:
    log = json.load(f)

# Create a figure for combined plots
fig, axes = plt.subplots(4, 5, figsize=(20, 16), squeeze = False)  # 4 rows, 5 columns grid
axes = axes.flatten()  # Flatten axes for easy indexing

# Plot each IMP iteration
for im in range(20):
    if im == 10: continue
    ilog = {(int(epoch) + 1): values for epoch, values in log[str(im)].items()}
    epochs = sorted(ilog.keys())
    real = [ilog[epoch][0] for epoch in epochs]
    rand = [ilog[epoch][1] for epoch in epochs]

    ax = axes[im]  # Select the appropriate subplot
    ax.plot(epochs, real, label="IMP", color="blue")
    ax.plot(epochs, rand, label="RANDOM", color="orange")

    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    ax.set_title(f"{(100 * 0.8 ** im):.2f}% Sparsity.")

    # Add labels only to the outermost plots
    if im % 5 == 0:  # Left-most column
        ax.set_ylabel("KLD")
    if im >= 15:  # Bottom row
        ax.set_xlabel("Epoch")

    ax.legend()
    
fig.suptitle("Kullback-Leibler Divergence vs. Epoch For VGG IMP with Rewinding.")

# Adjust layout
#plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)


# Save the combined figure
plt.savefig("VGG_REWIND_Divergences.png")
plt.close()"""
