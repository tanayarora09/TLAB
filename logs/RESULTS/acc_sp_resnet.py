import matplotlib
import matplotlib.scale
matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.pyplot as plt 

import numpy as np

from matplotlib.ticker import FuncFormatter

import pickle
import json

def format_func(value: float, tick_number: float) -> str:
    #print(value, abs(value - round(value)) < 1e-9)
    if abs(value - round(value)) < 1e-9: return f"{int(value)}"
    return f"{value:.3f}"

UNPRUNED_MEAN = 92.11666782697041
UNPRUNED_STD = 0.08178697160130831

COLORS = {"black": "#303030","grey": "#808080", "blue": "#4682B4", "orange": "#FF8C00", "green": "#32CD32", "red": "#DC143C", "purple": "#8A2BE2", "brown": "#A52A2A"}

if __name__ == '__main__':

    FINAL_SP = 32

    concrete_prefix = "concrete_rewind/long/"

    concrete_types = list(reversed(["loss", "kldlogit", "gradmatch", ]))#"msefeature", ]))#"gradnorm"]))
    concrete_names = {"loss": "Task Loss", "gradnorm": "Gradient Norm", "kldlogit": "Parent Logit KLD", "msefeature": "Parent Feature MSE", "gradmatch": "Parent Gradient MSE"}
    cname = lambda x: concrete_prefix + x + "/" + x

    names = ["rbt", "snip", "grasp", "synflow", "imp",]  
    names.extend(cname(ctype) for ctype in concrete_types)
    print(names)

    titles = {"rbt": "Random", "mbt": "Magnitude",
              "imp": "IMP (Frankle et al.)", 
              "grasp": "GraSP (Wang et al.)", 
              "grasp_improved": "GraSP Magnitude",
              "snip": "SNIP (Lee et al.)",
              "synflow": "SynFlow (Tanaka et al.)",
              }
    titles.update({cname(ctype): f"CTS | {concrete_names[ctype]}" for ctype in concrete_types})
    print(titles)

    colors = {"rbt": COLORS["grey"], "mbt": COLORS["green"], 
              "grasp": "limegreen",
              "synflow": "lime",
              "imp": COLORS["orange"], 
              "snip": "green",
              cname("loss"): COLORS["red"], 
              cname("gradnorm"): "darkmagenta",
              cname("kldlogit"): COLORS["blue"],
              cname("msefeature"): "cyan",
              cname("gradmatch"): "magenta"}

    results = dict()

    for name in names:

        with open(f"resnet20/{name}_acc_val.json", "r") as f:
            result = json.load(f)

        sp = sorted([float(key) for key in result.keys() if float(key) > 100*0.8**(FINAL_SP+1)+1e-2], reverse = True)
        acc = [result[f"{key:.4f}"]['mean'] for key in sp]
        std = [result[f"{key:.4f}"]['std'] for key in sp]

        results[name] = [sp, acc, std]

        #print(results[name])

    sparsities = sorted([100 * 0.8 ** im for im in range(FINAL_SP + 1)], reverse = True)


    plt.style.use("seaborn-v0_8-poster")
    fig, ax = plt.subplots(figsize = (10, 6))

    sparsity_plots = list(range(FINAL_SP + 1))

    for name in names:
        sp, acc, std = results[name]
        spes = list()
        spes = [val for val in sparsity_plots if any([abs(100 * 0.8**val - sps) < 1e-2 for sps in sp])]
        print(spes, sp, sparsity_plots)
        marker = '.-'
        if name in [cname(ctype) for ctype in concrete_types]: marker = 'X-'
        elif name == "imp": marker = 'v-'
        plt.plot(spes, acc, marker, color = colors[name], label = titles[name], alpha = 0.7)
        plt.fill_between(spes, [a + s for a, s in zip(acc, std)], [a - s for a, s in zip(acc, std)], alpha = 0.2, color = colors[name])

    title_font = {'fontname':'DejaVu Serif', 'fontsize':14, 'fontweight':'normal'} 
    label_font =  {'fontname':'DejaVu Serif', 'fontsize':12, 'fontweight':'normal'} 
    legend_font = {'family': 'DejaVu Serif', 'size': 9}


    ax.set_title("ResNet-20 (CIFAR10) - Saliency Methods", **label_font, pad = 20)
    plt.ylabel("Test Accuracy (%)", **label_font, labelpad = 15)
    plt.xlabel("Sparsity (%)", **label_font, labelpad = 12)


    ax.set_xticks(sparsity_plots[::4])
    ax.set_xticklabels([f"{(100 - val):.2f}" for val in sparsities[::4]], fontsize = 11)

    ax.set_ylim(0, 93.5)
    ax.set_yticks(np.arange(15,96,20))
    ax.tick_params(axis = 'y', labelsize = '11')

    handles, _ = plt.gca().get_legend_handles_labels()

    handles.append(plt.axhline(UNPRUNED_MEAN, color = COLORS["black"], linestyle = '-', label = "Unpruned Network", alpha = 0.8))
    plt.axhspan(UNPRUNED_MEAN - UNPRUNED_STD, UNPRUNED_MEAN + UNPRUNED_STD, alpha = 0.2, color = COLORS["black"])

    ax.minorticks_on()
    ax.yaxis.set_tick_params(which='minor', length=4, color='gray', direction='inout')

    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='lightgray', axis='x')
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='lightgray', axis='y')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', axis='y') # Minor grid for y-axis    

    for spine_name in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine_name].set_linewidth(1.5)

    #print(ax.get_legend_handles_labels())

    legend = ax.legend(loc='lower center',         # Anchors the top-center of the legend box
                    bbox_to_anchor=(0.5, 0.02), # (x, y) relative to the figure. 0.5 is center, -0.2 pushes it down.
                    ncol=3,                     # Number of columns for legend entries (adjust to your number of lines)
                    prop = legend_font,                # Adjust font size if needed
                    frameon=False,              # Removes the box around the legend
                    bbox_transform=fig.transFigure) # Crucial: place relative to the entire figure

    # --- Adjust Subplots to Make Room for Legend ---
    # Increase bottom margin significantly to make space for both xlabel AND legend
    plt.subplots_adjust(bottom=0.27) 

    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_color('gray')
    #ax.spines['bottom'].set_color('gray')

    #box = plt.gca().get_position()
    #plt.gca().set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Move legend to the top
    #plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
    #                fancybox=True, shadow=True, ncol=3, fontsize=7, )
    
    #plt.tight_layout()
    plt.savefig("tmp.png")
    plt.close()

    """
    100:

    """
    """
    for name in names:
        sp, acc, std = results[name]
        #plt.errorbar(sp, acc, color=colors[name], yerr = std, fmt = '.-', label = titles[name])
        plt.plot(sp, acc, color = colors[name], label = titles[name])
        plt.fill_between(sp, [a + s for a, s in zip(acc, std)], [a - s for a, s in zip(acc, std)], alpha = 0.2, color = colors[name])
    

    # Custom logarithmic scale with base 0.8

    #scale = matplotlib.scale.LogScale(ax, base = 0.8)
    plt.xscale("log", base = 0.8)
    plt.xticks(sparsities[::3])#[2:])
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

    # Labels and title
    #plt.title("VGG19 Accuracy Comparison")
    plt.ylim(86.0, 94.0)

    handles, _ = plt.gca().get_legend_handles_labels()

    handles.append(plt.axhline(93.83500069379807, color = 'black', linestyle = '--', label = "Dense Baseline"))
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Density (%)")
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Move legend to the top
    plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                    fancybox=True, shadow=True, ncol=3, fontsize=7, )
    #plt.tight_layout()
    plt.savefig("tmp.png")
    plt.close()
    """