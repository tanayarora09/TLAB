import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Filenames containing the data
filenames = [
    "DATA_SENSITIVITY_0_data_fitnesses.json",
    "DATA_SENSITIVITY_1_data_fitnesses.json",
    "DATA_SENSITIVITY_2_data_fitnesses.json"
]

# Dictionary to store standard deviations per batch_count and sparsity
std_dev_percentages = {}

# Load and process each file
for file in filenames:
    with open(file, 'r') as f:
        data = json.load(f)
    
    for batch_count, sparsities in data.items():
        batch_count = int(batch_count)  # Ensure batch_count is an integer
        if batch_count not in std_dev_percentages:
            std_dev_percentages[batch_count] = {}
        
        for sparsity, values in sparsities.items():
            values = np.array(values)  # Shape (2, 4)
            mean_values = np.mean(values, axis=1)  # Mean of each list_of_4_values
            std_values = np.std(values, axis=1)  # Std deviation of each list_of_4_values
            
            avg_std_percentage = (std_values / mean_values) * 100  # Compute percentage per row
            
            if sparsity not in std_dev_percentages[batch_count]:
                std_dev_percentages[batch_count][sparsity] = []
            
            std_dev_percentages[batch_count][sparsity].append(avg_std_percentage)

# Compute final averages and standard deviations per batch_count and sparsity
final_results = {}
std_dev_results = {}
for batch_count, sparsities in std_dev_percentages.items():
    final_results[batch_count] = {}
    std_dev_results[batch_count] = {}
    for sparsity, values in sparsities.items():
        values = np.array(values)  # Convert list to numpy array
        mean_avg_std = np.mean(values)  # Mean of averages
        std_avg_std = np.std(values)  # Std of averages
        
        final_results[batch_count][sparsity] = mean_avg_std
        std_dev_results[batch_count][sparsity] = std_avg_std

# Generate separate line plots for each sparsity level
sparsity_levels = sorted(set(int(s) for b in final_results.values() for s in b.keys()))

for sparsity in sparsity_levels:
    plt.figure()
    
    batch_counts = sorted(final_results.keys())  # Ensure batch counts are sorted numerically
    values = [final_results[batch][str(sparsity)] for batch in batch_counts]
    errors = [std_dev_results[batch][str(sparsity)] for batch in batch_counts]
    
    plt.errorbar([100 * item / 98 for item in batch_counts], values, yerr=errors, fmt='.-')
    
    plt.xlabel("Data Distribution Percentage")
    plt.ylabel("Average STDDEV as a Percentage of Mean Fitness")
    
    # Dynamically setting title with consistent length
    plt.title(f"Density: {(100 * 0.8**sparsity):.2f}%")
    
    # Ensure consistent grid and label formatting
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=10)  # Adjust size if necessary
    
    # Set y-axis locator for consistent tick marks
    ax = plt.gca()  # Get current axis
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Adjust as needed
    
    # Save the figure for each sparsity level
    plt.savefig(f"Data_Sensitivity_{sparsity:02d}.png")
    plt.close()
