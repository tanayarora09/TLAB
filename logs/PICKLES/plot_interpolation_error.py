import pickle
import numpy as np
import matplotlib.pyplot as plt

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Define file pattern and number of repetitions
file_pattern = "interpolation_STRENGTHEN_WEAK{}_alphas.pickle"
num_reps = 4  # Adjust based on the number of repetitions

# Dictionary to store results
aggregated_data = {}

# Load and aggregate data from multiple experiments
for rep in range(num_reps):
    filename = file_pattern.format(rep)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
    for sparsity, alpha_errors in data.items():
        if sparsity not in aggregated_data:
            aggregated_data[sparsity] = {}
        
        for alpha, error in alpha_errors.items():
            if alpha not in aggregated_data[sparsity]:
                aggregated_data[sparsity][alpha] = []
            aggregated_data[sparsity][alpha].append(1 - error)

# Compute averages and standard deviations
final_results = {}
std_dev_results = {}
for sparsity, alpha_errors in aggregated_data.items():
    final_results[sparsity] = {}
    std_dev_results[sparsity] = {}
    
    for alpha, errors in alpha_errors.items():
        errors = np.array(errors) * 100
        final_results[sparsity][alpha] = np.mean(errors)
        std_dev_results[sparsity][alpha] = np.std(errors)

# Create a colormap for sparsity levels
sparsity_levels = sorted(final_results.keys())
norm = mcolors.Normalize(vmin=min(sparsity_levels), vmax=max(sparsity_levels))
cmap = cm.viridis

plt.figure()
for sparsity in sparsity_levels:
    alphas = sorted(final_results[sparsity].keys())
    mean_errors = np.array([final_results[sparsity][alpha] for alpha in alphas])
    std_devs = np.array([std_dev_results[sparsity][alpha] for alpha in alphas])
    color = cmap(norm(sparsity))
    
    plt.plot(alphas, mean_errors, label=f"{(100 * 0.8**sparsity):.2f}%", color=color)
    plt.fill_between(alphas, mean_errors - std_devs, mean_errors + std_devs, color=color, alpha=0.3)

plt.xlabel("Interpolation Alpha")
plt.ylabel("Error (%)")
#plt.title("Instability as a Function of Alpha over Sparsities")
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label="Sparsity Level")
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width, box.height * 0.9])

# Move legend to the top
plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                fancybox=True, shadow=True, ncol=5, fontsize=7, )
plt.ylim(0, 100)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig("../RESULTS/Interpolation_Error_Colormap_Weak.png")
plt.close()

"""# Define file pattern and number of repetitions
file_pattern = "interpolation_STRENGTHEN_EARLY{}_alphas.pickle"
num_reps = 4  # Adjust based on the number of repetitions

# Dictionary to store results
aggregated_data = {}

# Load and aggregate data from multiple experiments
for rep in range(num_reps):
    filename = file_pattern.format(rep)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
    for sparsity, alpha_errors in data.items():
        if sparsity not in aggregated_data:
            aggregated_data[sparsity] = {}
        
        for alpha, error in alpha_errors.items():
            print(error)
            if alpha not in aggregated_data[sparsity]:
                aggregated_data[sparsity][alpha] = []
            aggregated_data[sparsity][alpha].append(1 - error)

# Compute averages and standard deviations
final_results = {}
std_dev_results = {}
for sparsity, alpha_errors in aggregated_data.items():
    final_results[sparsity] = {}
    std_dev_results[sparsity] = {}
    
    for alpha, errors in alpha_errors.items():
        errors = np.array(errors)
        errors *= 100
        final_results[sparsity][alpha] = np.mean(errors)
        std_dev_results[sparsity][alpha] = np.std(errors)

# Generate separate line plots for each sparsity level
for sparsity in sorted(final_results.keys()):
    plt.figure()
    
    alphas = sorted(final_results[sparsity].keys())
    mean_errors = [final_results[sparsity][alpha] for alpha in alphas]
    std_devs = [std_dev_results[sparsity][alpha] for alpha in alphas]
    
    mean_errors = np.array(mean_errors)
    std_devs = np.array(std_devs)
    
    plt.plot(alphas, mean_errors, label=f"Density: {(100 * 0.8**sparsity):.2f}%")
    plt.fill_between(alphas, mean_errors - std_devs, mean_errors + std_devs, alpha=0.3)
    
    plt.xlabel("Interpolation Alpha")
    plt.ylabel("Instability (%)")
    plt.title(f"Instability for Density: {(100 * 0.8**sparsity):.2f}%")
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", linewidth=0.5)
    
    # Save each plot separately
    plt.savefig(f"../RESULTS/Instability_Sparsity_{sparsity:02d}.png")
    plt.close()"""