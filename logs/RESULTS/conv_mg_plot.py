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

    names = ["rand", "imp", "randsearch", "mgsearch", "strongsearch", ]#"strengthen_weak", ]#"postmg",]
    titles = {"rand": "Random", "mgsearch": "GTS with Magnitude",# "premg": "Magnitude at Init", 
              "strongsearch": "GTS with Lottery (Strong)", "imp": "IMP with Rewinding", 
              "randsearch": "GTS with Random", }#"strengthen_weak": "GTS with Lottery (Weak)"}#"postmg": "Magnitude after Training"}

    colors = {"rand": "red", "mgsearch": "blue", "premg": "darkolivegreen", 
              "strongsearch": "darkslategrey", "imp": "orange", 
              "randsearch": "indigo", "strengthen_weak": "darkcyan"}#,"postmg": "pink"}

    results = dict()

    with open(f"../PICKLES/temporary.json", "r") as f:
        inp = json.load(f)

    for name in names:

        result = inp[name]

        sp = sorted([float(key) for key in result.keys()], reverse = True)
        print(name, sp)
        acc = [result[str(int(key))]['mean'] for key in sp]
        std = [result[str(int(key))]['std'] for key in sp]

        results[name] = [[100 * 0.8**it for it in sp], acc, std]

    sparsities = sorted([100 * 0.8 ** im for im in range(17, 31)], reverse = True)

    for name in names:
        sp, acc, std = results[name]
        plt.errorbar(sp, acc, color=colors[name], yerr = std, fmt = '.-', label = titles[name])
    

    # Custom logarithmic scale with base 0.8

    #scale = matplotlib.scale.LogScale(ax, base = 0.8)
    plt.xscale("log", base = 0.8)
    plt.xticks(sparsities[::2])
    plt.yscale('log')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.grid(True, which = 'both', linestyle = '--', linewidth = 0.5)

    # Labels and title
    #plt.title("VGG19 Accuracy Comparison")
    #plt.ylim(75.0, 95.0)

    handles, _ = plt.gca().get_legend_handles_labels()

    handles.append(plt.axhline(1.0, color = colors['premg'], linestyle = '--', label = "Magnitude Pruning"))
    plt.ylabel("Best Fitness Relative to Magnitude Pruning")
    plt.xlabel("Density (%)")
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Move legend to the top
    plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                    fancybox=True, shadow=True, ncol=3, fontsize=7, )
    #plt.tight_layout()
    plt.savefig("tmp_plot.png")
    plt.close()