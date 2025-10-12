import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from matplotlib.ticker import LogFormatterSciNotation, LogLocator

# --- Custom Formatter and Helpers from original script ---
class NegativeLogFormatter:
    def __init__(self, base=10.0):
        self._inner = LogFormatterSciNotation(base=base)

    def __call__(self, x, pos=None):
        if not np.isfinite(x):
            return ""
        if x == 0.0:
            return "0"
        return "-" + self._inner(x, pos)

def sparsity_to_density(level):
    return 100 * 0.8 ** int(level)

def format_func(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(value)}"
    return f"{value:.2f}"

# --- LaTeX and Font Configuration ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    # MODIFICATION: Added fontenc package for small caps support
    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{dejavu}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \newcommand{\textlusc}[1]{{\fontfamily{cmr}\scshape #1}}
    """
})
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["mathtext.it"] = "serif:italic"
plt.rcParams["mathtext.bf"] = "serif:bold"

# --- Main Plotting Function ---
def create_grid_plot(is_vgg, phase):
    model = "vgg16" if is_vgg else "resnet20"
    
    FullTitles = {
        "loss": r"$\mathcal{R}_{\mathcal{L}}$", 
        "deltaloss": r"$\mathcal{R}_{\Delta \mathcal{L}}$", 
        "gradnorm": r"$\mathcal{R}_{- \lVert \nabla_{\theta} \mathcal{L} \rVert_2}$ (log)", 
        "kldlogit": r"$\mathcal{R}_{\text{KL}}$", 
        "msefeature": r"$\mathcal{R}_{\text{feature}}$", 
        "gradmatch": r"$\mathcal{R}_{\text{grad}}$",
    }
    
    COLORS = {"random": "#303030", "magnitude": "darkmagenta", "imp": "#FF8C00", 
              "salient": "#EB0000", "multiplier": "blue", "gradbalance": "#4682B4", "dense": "#808080", }
    
    TITLES = {"imp": "LTR (Frankle et al.)", "gradbalance": r"\textlusc{GradBalance}", "multiplier": r"\textlusc{Lagrange}",
              "salient": "Saliency", "dense": "Unpruned Network", "magnitude": "Magnitude", "random": "Random"}
    
    plot_layout = [
        ["loss", "deltaloss",], 
        ["kldlogit", "gradnorm"],
        ["msefeature", "gradmatch"]
    ]

    # --- Plotting Setup ---
    plt.style.use("seaborn-v0_8-poster")
    fig, axs = plt.subplots(3, 2, figsize=(7, 10), sharex=True, sharey=False) #14, 8 for 2x3

    model_title = "VGG-16" if is_vgg else "ResNet-20"
    fig.suptitle(fr"\textbf{{{model_title}}}", fontsize=18)

    for r, row_ctypes in enumerate(plot_layout):
        for c, ctype in enumerate(row_ctypes):
            ax = axs[r, c]

            # Load data for the specific objective
            with open(f"{model}/comparisons/{ctype}.json", "r") as f:
                raw_data = json.load(f)

            sparsity_levels = sorted([int(k) for k in raw_data.keys()])
            methods = list(next(iter(raw_data.values()))[phase].keys())
            results = {method: {"x": [], "mean": [], "std": []} for method in methods}

            for level in sparsity_levels:
                entry = raw_data[str(level)][phase]
                for method in methods:
                    if method in entry:
                        results[method]["x"].append(level)
                        results[method]["mean"].append(entry[method]["mean"])
                        results[method]["std"].append(entry[method]["std"])

            # Plot data for each method
            for method in COLORS:
                if method not in results or not results[method]["x"]: continue
                if method == "dense": continue
                
                marker = '.'
                if method in ["gradbalance", "multiplier"]: marker = 'X'
                elif "imp" in method: marker = 'v'
                elif "magnitude" in method: marker = 'H'
                
                additional = {'markersize': 6} if marker != '.' else {'markersize': 10}

                ax.errorbar(results[method]["x"], results[method]["mean"], yerr=results[method]["std"],
                            fmt=marker, capsize=0, elinewidth=0.35, label=TITLES.get(method, method),
                            color=COLORS[method], alpha=0.8, **additional)

            # Plot dense baseline
            if "dense" in results and results[method]["mean"]:
                dense_y, dense_std = results[method]["mean"][0], results[method]["std"][0]
                ax.axhline(dense_y, color=COLORS["dense"], linestyle='-', label=TITLES["dense"], alpha=0.8)
                ax.axhspan(dense_y - dense_std, dense_y + dense_std, alpha=0.2, color=COLORS["dense"])

            # --- Subplot Formatting ---
            ax.set_title(FullTitles[ctype], fontsize=14, pad=10)
            
            tick_levels = sparsity_levels[4::6] if is_vgg else sparsity_levels[::4]
            ax.set_xticks(tick_levels)
            ax.set_xticklabels([format_func(sparsity_to_density(lvl)) for lvl in tick_levels], fontsize=10, rotation=45)
            ax.tick_params(axis='y', labelsize=10)
            
            for spine_name in ['left', 'right', 'top', 'bottom']:
                ax.spines[spine_name].set_linewidth(1.5)
            
            ax.minorticks_on()
            ax.yaxis.set_tick_params(which='minor', length=4, color='gray', direction='inout')
            ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='lightgray', axis='both')
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', axis='y')

            # Special formatting for gradnorm plot
            if ctype == "gradnorm":
                ax.set_yscale("log")
                ax.invert_yaxis()
                ax.yaxis.set_major_formatter(NegativeLogFormatter(base=10.0))
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
            
            # Set shared labels only on the outer plots
            if c == 0:
                ax.set_ylabel(r"$\mathcal{R}$", fontsize=12, labelpad=10)
            if r == 2:
                ax.set_xlabel(r"Density (\%)", fontsize=12, labelpad=12)
            
            bbox = ax.get_position()
            fig_w, fig_h = fig.get_size_inches()
            print(f"Axis {r}, {c}: {bbox.width * fig_w:.3f} | {bbox.height * fig_h * (3.850/3.523):.3f}")


    # --- Shared Legend ---
    handles, labels = ax.get_legend_handles_labels()
    legend_font = {'family': 'DejaVu Serif', 'size': 12}
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=7, prop=legend_font, frameon=False)
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.2, hspace=0.3, wspace=0.25)

    # --- Save ---
    output_dir = f"plots/{model}/comparisons"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{phase}.png")
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=1200)
    plt.close()

    
def create_init_overview_plot(is_vgg):
    model = "vgg16" if is_vgg else "resnet20"
    phase = "init" # This plot is only for the init phase
    
    # Titles with \overline for this specific plot
    FullTitles = {
        "loss": r"$\overline{\mathcal{R}_{\mathcal{L}}}$", 
        "gradnorm": r"$\overline{\mathcal{R}_{- \lVert \nabla_{\theta} \mathcal{L} \rVert_2}}$ (log)", 
    }
    
    COLORS = {"random": "#303030", "magnitude": "darkmagenta", "imp": "#FF8C00", 
              "salient": "#EB0000", "multiplier": "blue", "gradbalance": "#4682B4", "dense": "#808080", }
    
    TITLES = {"imp": "LTR (Frankle et al.)", "gradbalance": r"\textlusc{GradBalance}", "multiplier": r"\textlusc{Lagrange}",
              "salient": "Saliency", "dense": "Unpruned Network", "magnitude": "Magnitude", "random": "Random"}
    
    plot_layout = ["loss", "gradnorm"]

    # --- Plotting Setup for 1x2 grid ---
    plt.style.use("seaborn-v0_8-poster")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=False) # 1 row, 2 cols

    model_title = "VGG-16" if is_vgg else "ResNet-20"
    fig.suptitle(fr"\textbf{{{model_title}}}", fontsize=18)

    for c, ctype in enumerate(plot_layout):
        ax = axs[c]

        with open(f"{model}/comparisons/{ctype}.json", "r") as f:
            raw_data = json.load(f)

        sparsity_levels = sorted([int(k) for k in raw_data.keys()])
        methods = list(next(iter(raw_data.values()))[phase].keys())
        results = {method: {"x": [], "mean": [], "std": []} for method in methods}

        for level in sparsity_levels:
            entry = raw_data[str(level)][phase]
            for method in methods:
                if method in entry:
                    results[method]["x"].append(level)
                    results[method]["mean"].append(entry[method]["mean"])
                    results[method]["std"].append(entry[method]["std"])

        for method in COLORS:
            if method not in results or not results[method]["x"]: continue
            if method == "dense": continue
            
            marker = '.'
            if method in ["gradbalance", "multiplier"]: marker = 'X'
            elif "imp" in method: marker = 'v'
            elif "magnitude" in method: marker = 'H'
            
            additional = {'markersize': 6} if marker != '.' else {'markersize': 10}
            ax.errorbar(results[method]["x"], results[method]["mean"], yerr=results[method]["std"],
                        fmt=marker, capsize=0, elinewidth=0.35, label=TITLES.get(method, method),
                        color=COLORS[method], alpha=0.8, **additional)

        if "dense" in results and results[method]["mean"]:
            dense_y, dense_std = results[method]["mean"][0], results[method]["std"][0]
            ax.axhline(dense_y, color=COLORS["dense"], linestyle='-', label=TITLES["dense"], alpha=0.8)
            ax.axhspan(dense_y - dense_std, dense_y + dense_std, alpha=0.2, color=COLORS["dense"])

        # --- Subplot Formatting ---
        ax.set_title(FullTitles[ctype], fontsize=14, pad=10)
        
        tick_levels = sparsity_levels[4::6] if is_vgg else sparsity_levels[::4]
        ax.set_xticks(tick_levels)
        ax.set_xticklabels([format_func(sparsity_to_density(lvl)) for lvl in tick_levels], fontsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=10)
        
        for spine_name in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine_name].set_linewidth(1.5)
        
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which='minor', length=4, color='gray', direction='inout')
        ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='lightgray', axis='both')
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', axis='y')

        if ctype == "gradnorm":
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.yaxis.set_major_formatter(NegativeLogFormatter(base=10.0))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
        
        if c == 0:
            ax.set_ylabel(r"$\mathcal{R}$", fontsize=12, labelpad=10)
        ax.set_xlabel(r"Density (\%)", fontsize=10, labelpad=12)

        bbox = ax.get_position()
        fig_w, fig_h = fig.get_size_inches()
        print(f"Axis {c}: {bbox.width * fig_w:.3f} | {bbox.height * fig_h:.3f}")

    handles, labels = ax.get_legend_handles_labels()
    legend_font = {'family': 'DejaVu Serif', 'size': 8}
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=7, prop=legend_font, frameon=False)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.25, hspace=0.3, wspace=0.3)

    output_dir = f"plots/{model}/comparisons"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"init_overview.png")
    print(f"Saving overview plot to {save_path}")
    plt.savefig(save_path, dpi=1200)
    plt.close()

if __name__ == "__main__":
    vggs = [False, True]
    phases = ["rewind", "init", ]
    
    """for phase in phases:
        for is_vgg in vggs:
            create_grid_plot(is_vgg, phase)"""
    
    for is_vgg in vggs:
        create_grid_plot(is_vgg, "rewind")
        #create_init_overview_plot(is_vgg)