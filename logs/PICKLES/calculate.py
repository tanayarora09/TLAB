import pickle
import json
import numpy as np

value = dict()

def find_best(logs):
	best = 0
	for ep in range(1, 161):
		curr = logs[ep * 391]["val_accuracy"]
		if curr > best: best = curr
	return best

sparsities = int(input("How many total sparsities are calculated?" ))

for sp in range(sparsities):
	
	if (sp == 10) or (sp == 11): continue

	accs = list()

	for idx in range(8):
		print(sp, idx)
		with open(f"RANDOM{idx}_{sp}.0_logs.pickle", "rb") as f:
			logs = pickle.load(f)
		accs.append(100 * logs[150 * 391]['val_accuracy'])

	accs = np.asarray(accs)

	value[0.8**sp * 100] = {"Mean": float(accs.mean()), "Std": float(accs.std())}

with open("Baselines150.json", "w") as f:
	json.dump(value, f, indent = 6)
