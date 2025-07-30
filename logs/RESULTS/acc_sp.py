import matplotlib
import matplotlib.scale
matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.pyplot as plt 

import numpy as np

from matplotlib.ticker import FuncFormatter

import pickle
import json

def format_func(value: float) -> str:
    """Formats float values, showing integers for whole numbers."""
    if abs(value - round(value)) < 1e-9: return f"{int(value)}"
    return f"{value:.2f}"

COLORS = {"black": "#303030","grey": "#808080", "blue": "#4682B4", "orange": "#FF8C00", "darkorange": "#DB5B00", "green": "#32CD32", "red": "#DC143C", "purple": "#8A2BE2", "brown": "#A52A2A"}

if __name__ == '__main__':
    # Set to True for VGG-16 configuration, False for ResNet-20 configuration
    is_vgg = False 

    OUT_NAME = "rewind_short_others"
    CONCRETE_PREFIX = "concrete/gradbalance/rewind/short/"

    CONCRETE_POSTFIX = "/finetuned"
    CONCRETE_TYPES = list(reversed(["loss", "kldlogit", "deltaloss", "gradmatch", "msefeature", "gradnorm"]))



    # Model-specific configurations
    if is_vgg:
        FINAL_SP = 42
        UNPRUNED_MEAN = 93.83500069379807
        UNPRUNED_STD = 0.11715378860254261
        MODEL_DIR = "vgg16"
        PLOT_TITLE = "VGG-16 (CIFAR10)"
        X_TICKS_STEP = [0,] + list(range(4, FINAL_SP + 1, 6)) # [0, 4, 10, 16, 22, 28, 34, 40]
        Y_LIM = (0, 95.5)
        X_GRID_MINOR = True

    else: # ResNet-20
        FINAL_SP = 32
        UNPRUNED_MEAN = 92.11666782697041
        UNPRUNED_STD = 0.08178697160130831
        MODEL_DIR = "resnet20"
        PLOT_TITLE = "ResNet-20 (CIFAR10) - Saliency Methods"
        X_TICKS_STEP = list(range(0, FINAL_SP + 1, 4)) # [0, 4, 8, 12, 16, 20, 24, 28, 32]
        Y_LIM = (0, 93.5)
        X_GRID_MINOR = True

    CONCRETE_NAMES = {"loss": "Task Loss", "deltaloss": "Δ Loss", "gradnorm": "Gradient Norm", "kldlogit": "Parent Logit KLD", "msefeature": "Parent Feature MSE", "gradmatch": "Parent Gradient MSE"}
    cname = lambda x: CONCRETE_PREFIX + x + CONCRETE_POSTFIX

    names = ["rbt", "imp", "snip", "grasp", "synflow", ]
    names.extend(cname(ctype) for ctype in CONCRETE_TYPES)
    print("Methods included:", names)

    titles = {"rbt": "Random", "mbt": "Magnitude",
              "imp": "LTR (Frankle et al.)",
              "late_imp": "LTR [Late]", 
              "grasp": "GraSP (Wang et al.)", 
              "grasp_improved": "GraSP Magnitude",
              "snip": "SNIP (Lee et al.)",
              "synflow": "SynFlow (Tanaka et al.)",
              }
    titles.update({cname(ctype): f"CTS | {CONCRETE_NAMES[ctype]}" for ctype in CONCRETE_TYPES})
    print("Titles mapping:", titles)

    colors = {"rbt": "black", 
              "grasp": COLORS["green"],
              "synflow": COLORS["green"],
              "imp": COLORS["orange"], 
              "late_imp": COLORS["darkorange"], 
              "snip": COLORS["green"],
              cname("loss"): COLORS["blue"], 
              cname("deltaloss"): COLORS["purple"],
              cname("gradnorm"): COLORS["grey"],
              cname("kldlogit"): COLORS["red"],
              cname("msefeature"): COLORS["grey"],
              cname("gradmatch"): COLORS["grey"],}

    marker_map = {'synflow': '*', 'grasp': 'o', 'snip': 'D',
                  'imp': 'v',
                  cname("loss"): 'X', cname('deltaloss'): 'X', cname('gradnorm'): 's', 
                  cname("kldlogit"): 'X', cname('msefeature'): 'o', cname('gradmatch'): 'D',}

    results = dict()

    for name in names:
        with open(f"{MODEL_DIR}/{name}_acc_val.json", "r") as f:
            result = json.load(f)

        # Sparsity threshold calculation
        # The VGG script used 1e-3, ResNet 1e-2. Standardizing to 1e-2 for consistency if not specified
        epsilon_threshold = 1e-3 if is_vgg else 1e-2
        threshold_value = 100 * 0.8**(FINAL_SP + 1) + epsilon_threshold
        
        sp = sorted([float(key) for key in result.keys() if float(key) > threshold_value], reverse=True)
        acc = [result[f"{key:.4f}"]['mean'] for key in sp]
        std = [result[f"{key:.4f}"]['std'] for key in sp]

        print(f"{name}: {len(sp)} data points")

        results[name] = [sp, acc, std]

    sparsities = sorted([100 * 0.8 ** im for im in range(FINAL_SP + 1)], reverse = True)

    plt.style.use("seaborn-v0_8-poster")
    fig, ax = plt.subplots(figsize = (10, 6))

    sparsity_plots = list(range(FINAL_SP + 1))

    handles, _ = plt.gca().get_legend_handles_labels()

    handles.append(plt.axhline(UNPRUNED_MEAN, color = COLORS["black"], linestyle = '-', label = "Unpruned Network", alpha = 0.8))
    plt.axhspan(UNPRUNED_MEAN - UNPRUNED_STD, UNPRUNED_MEAN + UNPRUNED_STD, alpha = 0.2, color = COLORS["black"])

    for name in names:
        sp, acc, std = results[name]
        spes = list()
        # Adjusted epsilon for matching sparsity values
        match_epsilon = 1e-3 if is_vgg else 1e-2
        spes = [val for val in sparsity_plots if any([abs(100 * 0.8**val - sps) < match_epsilon for sps in sp])]
        print(f"Plotting {name}: {spes} for sparsities {sp}")
        marker = '.-'
        additional = {}
        if name in marker_map.keys(): marker = marker_map[name] + '-'
        if marker in [item + '-' for item in ['D', 's', 'o']]: additional['markersize']= 6
        elif marker == '*-': additional['markersize'] = 14
        plt.plot(spes, acc, marker, color = colors[name], label = titles[name], alpha = 0.7, **additional)
        plt.fill_between(spes, [a + s for a, s in zip(acc, std)], [a - s for a, s in zip(acc, std)], alpha = 0.2, color = colors[name])

    title_font = {'fontname':'DejaVu Serif', 'fontsize':14, 'fontweight':'normal'} 
    label_font =  {'fontname':'DejaVu Serif', 'fontsize':12, 'fontweight':'normal'} 
    legend_font = {'family': 'DejaVu Serif', 'size': 9}

    ax.set_title(PLOT_TITLE, **label_font, pad = 20)
    plt.ylabel("Test Accuracy (%)", **label_font, labelpad = 15)
    plt.xlabel("Sparsity (%)", **label_font, labelpad = 12)

    ax.set_xticks([sparsity_plots[idx] for idx in X_TICKS_STEP])
    if is_vgg:
        ax.set_xticklabels([format_func(100 - sparsities[idx]) for idx in X_TICKS_STEP], fontsize = 11)
    else: # ResNet
        ax.set_xticklabels([f"{(100 - sparsities[idx]):.2f}" for idx in X_TICKS_STEP], fontsize = 11)

    ax.set_ylim(*Y_LIM)
    ax.set_yticks([0,] + np.arange(15,96,20).tolist())
    ax.tick_params(axis = 'y', labelsize = '11')

    ax.minorticks_on()
    ax.yaxis.set_tick_params(which='minor', length=4, color='gray', direction='inout')

    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='lightgray', axis='x')
    if X_GRID_MINOR:
        ax.grid(True, which='minor', linestyle='-', linewidth=0.2, color='lightgray', axis='x')
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='lightgray', axis='y')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', axis='y') 

    for spine_name in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine_name].set_linewidth(1.5)

    legend = ax.legend(loc='lower center', 
                      bbox_to_anchor=(0.5, 0.02),
                      ncol=4, 
                      prop = legend_font, 
                      frameon=False, 
                      bbox_transform=fig.transFigure) 

    plt.subplots_adjust(bottom=0.27) 

    plt.savefig(f"plots/{MODEL_DIR}/accuracy/{OUT_NAME}.png")
    plt.close()