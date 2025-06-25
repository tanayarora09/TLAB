import matplotlib
import matplotlib.scale
matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.pyplot as plt 

from matplotlib.ticker import FuncFormatter

import pickle
import json

def format_func(value: float, tick_number: float) -> str:
    print(value, abs(value - round(value)) < 1e-9)
    if abs(value - round(value)) < 1e-9: return f"{int(value)}"
    return f"{value:.2f}"

if __name__ == '__main__':

    names = ["base", "premg", "ltr", "randsearch", "mgsearch", "strengthen_early", "strengthen_weak", ]#"postmg",]
    titles = {"base": "Random", "mgsearch": "GTS with Magnitude", "premg": "Magnitude at Init", 
              "strengthen_early": "GTS with Lottery (Strong)", "ltr": "IMP with Rewinding", 
              "randsearch": "GTS with Random", "strengthen_weak": "GTS with Lottery (Weak)"}#"postmg": "Magnitude after Training"}

    colors = {"base": "red", "mgsearch": "blue", "premg": "darkolivegreen", 
              "strengthen_early": "darkslategrey", "ltr": "orange", 
              "randsearch": "indigo", "strengthen_weak": "darkcyan"}#,"postmg": "pink"}

    results = dict()

    for name in names:

        with open(f"{name}_acc_test.json", "r") as f:
            result = json.load(f)

        sp = sorted([float(key) for key in result.keys()], reverse = True)
        acc = [result[f"{key:.4f}"]['mean'] for key in sp]
        std = [result[f"{key:.4f}"]['std'] for key in sp]

        results[name] = [sp, acc, std]

    sparsities = sorted([100 * 0.8 ** im for im in range(31)], reverse = True)

    for name in names:
        sp, acc, std = results[name]
        plt.errorbar(sp, acc, color=colors[name], yerr = std, fmt = '.-', label = titles[name])
    

    # Custom logarithmic scale with base 0.8

    #scale = matplotlib.scale.LogScale(ax, base = 0.8)
    plt.xscale("log", base = 0.8)
    plt.xticks([100, 50, 26.2144] + sparsities[2::4][2:])
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

    # Labels and title
    #plt.title("VGG19 Accuracy Comparison")
    plt.ylim(75.0, 95.0)

    handles, _ = plt.gca().get_legend_handles_labels()

    handles.append(plt.axhline(93.6400016148885, color = 'purple', linestyle = '--', label = "Dense Baseline"))
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