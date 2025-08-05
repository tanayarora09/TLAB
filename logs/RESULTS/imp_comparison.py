import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# --- Helpers ---
def sparsity_to_density(level):
    return 100 * 0.8 ** int(level)

is_vgg = False
ctype = "oldkld"

model = "vgg16" if is_vgg else "resnet20"


def get_salient_name(ctype, iterative: bool):
    map = {"loss": "SNIP (Lee et al.)", "gradnorm": "GraSP (Wang et al.)"}
    out = "Saliency (One-Shot)"
    if iterative: out = "Saliency (Iter. 100)"
    if ctype in map.keys(): 
        out = map[ctype]
        if iterative:
            out = out[:out.rindex('(')] + "(Iter. 100)"

    return out

FullTitles = {

    "loss": "Task Loss", 
    "deltaloss": "Δ Loss", 
    "gradnorm": "Gradient Norm (LOG)", 
    "kldlogit": "Parent Logit KLD", 
    "msefeature": "Parent Feature MSE", 
    "gradmatch": "Parent Gradient MSE",
    "oldkld": "Custom Smoothened KLD"

}

# --- Config ---
COLORS = {
    "magnitude": "#32CD32",
    #"random": "#DC143C",
    "imp": "#FF8C00",
    "gradbalance": "#4682B4",
    "multiplier": "blue",
    "salient": "#DC143C",
    "iterative": "darkmagenta",
    "dense": "#808080",
}
TITLES = {
    "imp": "LTR (Frankle et al.)",
    "gradbalance": "CTS | GradBalance",
    "multiplier": "CTS | Multiplier",
    "salient": get_salient_name(ctype, iterative = False),
    "iterative": get_salient_name(ctype, iterative = True),
    "dense": "Unpruned Network",
    "magnitude": "Magnitude",
    "random": "Random",
}

# --- Load JSON ---
with open(f"{model}/comparisons/{ctype}.json", "r") as f:
    raw_data = json.load(f)

# --- Determine Sparsity Levels ---
sparsity_levels = sorted([int(k) for k in raw_data.keys()])

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

        marker = '.'
        if method in ["gradbalance", "multiplier"]: marker = 'X'
        elif "imp" in method: marker = 'v'
        elif "magnitude" in method: marker = 'H'

        additional = {}
        if marker != '.': additional['markersize'] = 8

        """
        if ctype == 'gradnorm':
            y = [1/v for v in y]
            yerr = [1/v for v in yerr]  
        """

        ax.errorbar(
            x, y, yerr=yerr, 
            fmt=marker, capsize=2,
            elinewidth=0.35, label=TITLES[method],
            color=COLORS[method], alpha=0.8, **additional
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
    legend_font = {'family': 'DejaVu Serif', 'size': 8}

    ax.set_title(f"{"VGG-16" if is_vgg else "ResNet-20"} ({phase.upper()})", **label_font, pad=20)
    ax.set_ylabel(FullTitles[ctype], **label_font, labelpad=4)
    ax.set_xlabel("Sparsity (%)", **label_font, labelpad=12)

    ax.set_xticks(sparsity_levels[4::6] if is_vgg else sparsity_levels[::4])
    ax.set_xticklabels(
        [f"{100 - sparsity_to_density(lvl):.3f}" for lvl in (sparsity_levels[4::6] if is_vgg else sparsity_levels[::4])],
        fontsize=11
    )

    ax.tick_params(axis='y', labelsize=10)
    if ctype == "gradnorm": plt.yscale('log')

    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
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
    os.makedirs(f"plots/{model}/comparisons/{ctype}/", exist_ok=True)
    print(f"Saving to plots/{model}/comparisons/{ctype}/{phase}.png")
    plt.savefig(f"plots/{model}/comparisons/{ctype}/{phase}.png", )
    plt.close()
