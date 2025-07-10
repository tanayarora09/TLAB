import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# --- Helpers ---
def sparsity_to_density(level):
    return 100 * 0.8 ** int(level)

# --- Config ---
COLORS = {
    #"magnitude": "#32CD32",
    #"random": "#DC143C",
    "imp": "#FF8C00",
    "concrete": "#4682B4",
    "grasp_improved": "#8A2BE2",
    "grasp": "#A52A2A",
    "dense": "#808080",
}
TITLES = {
    "imp": "IMP (Frankle et al.)",
    "concrete": "Concrete (Ours)",
    "grasp": "GraSP (Wang et al.)",
    "grasp_improved": "GraSP Magnitude",
    "dense": "Unpruned Network",
    "magnitude": "Magnitude",
    "random": "Random",
}

# --- Load JSON ---
with open("vgg16/gradnorm_concrete_comparison.json", "r") as f:
    raw_data = json.load(f)

# --- Determine Sparsity Levels ---
sparsity_levels = sorted([int(k) for k in raw_data.keys()])[2:]

# --- Plot each phase separately ---
for phase in ["init", "rewind"]:
    # Organize Results
    methods = list(next(iter(raw_data.values()))[phase].keys())
    results = {method: {"x": [], "mean": [], "std": []} for method in methods}

    for level in sparsity_levels:
        entry = raw_data[str(level)][phase]
        for method in methods:
            if method in entry:
                results[method]["x"].append(level)
                results[method]["mean"].append(entry[method]["mean"])
                results[method]["std"].append(entry[method]["std"])

    # --- Plotting ---
    plt.style.use("seaborn-v0_8-poster")
    fig, ax = plt.subplots(figsize=(7, 6))

    for method in COLORS:
        if method not in results:
            continue
        if method == "dense":
            continue
        x = results[method]["x"]
        y = results[method]["mean"]
        yerr = results[method]["std"]
        ax.errorbar(
            x, y, #yerr=yerr, 
            fmt='.', capsize=3,
            elinewidth=1, label=TITLES[method],
            color=COLORS[method], alpha=0.8
        )

    # Plot dense baseline
    if "dense" in results and results["dense"]["mean"]:
        dense_y = results["dense"]["mean"][0]
        dense_std = results["dense"]["std"][0]
        ax.axhline(dense_y, color=COLORS["dense"], linestyle='-', label=TITLES["dense"], alpha=0.8)
        ax.axhspan(dense_y - dense_std, dense_y + dense_std, alpha=0.2, color=COLORS["dense"])

    # --- Formatting ---
    title_font = {'fontname': 'DejaVu Serif', 'fontsize': 14}
    label_font = {'fontname': 'DejaVu Serif', 'fontsize': 12}
    legend_font = {'family': 'DejaVu Serif', 'size': 9}

    ax.set_title(f"VGG-16 ({phase.upper()})", **label_font, pad=20)
    ax.set_ylabel("Gradient Norm", **label_font, labelpad=4)
    ax.set_xlabel("Sparsity (%)", **label_font, labelpad=12)

    ax.set_xticks(sparsity_levels[::2])
    ax.set_xticklabels(
        [f"{100 - sparsity_to_density(lvl):.1f}" for lvl in sparsity_levels][::2],
        fontsize=11
    )

    ax.tick_params(axis='y', labelsize=10)
    #plt.yscale('log')

    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        prop=legend_font,
        frameon=False,
        bbox_transform=fig.transFigure
    )

    plt.subplots_adjust(bottom=0.25)

    for spine_name in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine_name].set_linewidth(1.5)

    ax.minorticks_on()
    ax.yaxis.set_tick_params(which='minor', length=4, color='gray', direction='inout')
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='lightgray', axis='both')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', axis='y')

    # --- Save ---
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/vgg16_gradnorm_concrete_comparison_{phase}.png")
    plt.close()
