import json

json_files = [
    "base_acc_train.json",
    "base_loss_test.json",
    "base_loss_train.json",
    "premg_acc_test.json",
    "premg_acc_train.json",
    "premg_loss_test.json",
    "premg_loss_train.json"
]

for file in json_files:

	with open(file, "r") as f:
		data = json.load(f)

# Convert keys to four decimal places
	new_data = {f"{float(k):.4f}": v for k, v in data.items()}

# Save the updated JSON
	with open(file, "w") as f:
		json.dump(new_data, f, indent=4)

	print(f"Updated: {file}")
