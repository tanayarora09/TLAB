import numpy as np
import json
from collections import defaultdict

is_vgg = False
ctype = "loss"
reps = 3

model = "vgg16" if is_vgg else "resnet20"
prefix = f"{ctype}_imp_comparison_{model}"
postfix = "comparison"

output = f"{model}/comparisons/{ctype}.json"

accum = defaultdict(list)

for rep in range(reps):

    with open(f"{prefix}_{rep}_{postfix}.json", 'r') as f:
        data = json.load(f)

        for num_key, num_val in data.items():
            for cat_key, cat_val in num_val.items():
                for metric_key, metric_val in cat_val.items():
                    key = (num_key, cat_key, metric_key)
                    accum[key].append(metric_val)

results = defaultdict(lambda: defaultdict(dict))

for (num_key, cat_key, metric_key), values in accum.items():
    mean = np.mean(values)
    std = np.std(values)  # Use np.std for population standard deviation
    results[num_key][cat_key][metric_key] = {
        "mean": mean,
        "std": std
    }

with open(output, 'w') as f:
    json.dump(results, f, indent = 6)
    print("Saving to", output)