"""
import json 
import numpy as np

output_valA = dict()
output_valL = dict()
output_A = dict()
output_L = dict()

out_name = "resnet20/snip"
prefix_name = "snip_resnet20"

sparsity_indexes = [24, 26, 28, 30, 32] #list(range(2,33,2)) # RESNET
#sparsity_indexes = [26, 28, 30, 34, 38, 42] #list(range(2,43,2)) # VGG

is_adding = True

#sparsity_indexes = list(range(26))

reps = 3

for spidx in sparsity_indexes:

    sp = 0.8 ** spidx * 100

    tmpvalA = list()
    tmpvalL = list()
    tmpA = list()
    tmpL = list()

    for rep in range(reps):
        postfix = "f"
        
        if rep == reps-1:
            postfix = "s"
            rep = 0

        name = f"{prefix_name}_{postfix}_{rep}_{spidx}.json"
        
        with open(name, 'r') as f:
            print(f"Opened {name}")
        #with open(f"imp_vgg16_{rep}_{spidx}.json", 'r') as f:
            result = json.load(f)

        tmpvalA.append(result['val_accuracy'] * 100)
        tmpvalL.append(result['val_loss'] )
        tmpA.append(result['accuracy'] * 100)
        tmpL.append(result['loss'])


    tmpvalA = np.asarray(tmpvalA)
    tmpvalL = np.asarray(tmpvalL)
    tmpA = np.asarray(tmpA)
    tmpL = np.asarray(tmpL)

    key = f"{sp:.4f}"

    output_valA[key] = {"mean": tmpvalA.mean(), "std": tmpvalA.std()}
    output_valL[key] = {"mean": tmpvalL.mean(), "std": tmpvalL.std()}
    output_A[key] = {"mean": tmpA.mean(), "std": tmpA.std()}
    output_L[key] = {"mean": tmpL.mean(), "std": tmpL.std()}

if not is_adding:

    with open(f"{out_name}_acc_train.json", 'w') as f:
        json.dump(output_A, f, indent = 6 )

    with open(f"{out_name}_acc_val.json", 'w') as f:
        print('Writing to', f"{out_name}_acc_val.json")
        json.dump(output_valA, f, indent = 6 )

    with open(f"{out_name}_loss_train.json", 'w') as f:
        json.dump(output_L, f, indent = 6 )

    with open(f"{out_name}_loss_val.json", 'w') as f:
        json.dump(output_valL, f, indent = 6 )

if is_adding:

    with open(f"{out_name}_acc_train.json", 'r') as f:
        output_A.update(json.load(f))

    with open(f"{out_name}_acc_val.json", 'r') as f:
        output_valA.update(json.load(f))

    with open(f"{out_name}_loss_train.json", 'r') as f:
        output_L.update(json.load(f))

    with open(f"{out_name}_loss_val.json", 'r') as f:
        output_valL.update(json.load(f))

    with open(f"{out_name}_acc_train.json", 'w') as f:
        json.dump(output_A, f, indent = 6 )

    with open(f"{out_name}_acc_val.json", 'w') as f:
        print('Writing to', f"{out_name}_acc_val.json")
        json.dump(output_valA, f, indent = 6 )

    with open(f"{out_name}_loss_train.json", 'w') as f:
        json.dump(output_L, f, indent = 6 )

    with open(f"{out_name}_loss_val.json", 'w') as f:
        json.dump(output_valL, f, indent = 6 ) 

"""
import json
import numpy as np
import os
from pathlib import Path

# --- Configuration and Mappings ---
# These maps translate the integer codes from the submission script to directory/file names.
ARCH_MAP = {1: 'vgg16', 0: 'resnet20'}
CONCRETE_MAP = {1: 'gradbalance', 0: 'multiplier'}
SCHEME_MAP = {1: 'init', 0: 'rewind'}
DURATION_MAP = {1: 'short', 0: 'long'}
# This map is based on the CONCRETE_EXPERIMENTS dictionary in concrete_final.py
METHOD_MAP = {
    0: 'loss',
    1: 'gradnorm',
    2: 'kldlogit',
    3: 'msefeature',
    4: 'gradmatch'
}
RUN_TAGS = ['f', 's', 't']
INPUT_DIR = Path("./logs/RESULTS")

# Define the experiment parameter space, matching the submission script
ARCH_CODES = [1, 0]
CONCRETE_CODES = [1, 0]
SCHEME_CODES = [1, 0]
DURATION_CODES = [1, 0]
METHOD_CODES = [0, 1, 2, 3, 4]

# --- Main Collation Loop ---
for arch_code in ARCH_CODES:
    for concrete_code in CONCRETE_CODES:
        for scheme_code in SCHEME_CODES:
            for duration_code in DURATION_CODES:
                for method_code in METHOD_CODES:

                    # --- 1. Construct Paths and Names ---
                    arch_name = ARCH_MAP[arch_code]
                    concrete_name = CONCRETE_MAP[concrete_code]
                    scheme_name = SCHEME_MAP[scheme_code]
                    duration_name = DURATION_MAP[duration_code]
                    method_name = METHOD_MAP[method_code]

                    print(f"--- Processing: {arch_name}/{scheme_name}/{duration_name}/{concrete_code}/{method_name} ---")

                    # Define the final output directory and create it if it doesn't exist
                    output_dir = Path(f"./{arch_name}/{scheme_name}/{duration_name}/{concrete_code}/{method_name}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    finetuned = ['original', 'finetuned']

                    for finetune in finetuned:

                        # Base name for the four output JSON files
                        output_file_base = output_dir / finetune

                        # --- 2. Initialize Data Dictionaries for this Configuration ---
                        output_val_acc = dict()
                        output_val_loss = dict()
                        output_train_acc = dict()
                        output_train_loss = dict()

                        # --- 3. Determine Sparsity Levels ---
                        # VGG has more layers, so it has a different sparsity range
                        sparsity_indexes = range(2, 43, 2) if arch_code == 1 else range(2, 33, 2)

                        # --- 4. Loop Through Each Sparsity Level ---
                        for spidx in sparsity_indexes:
                            sparsity_percent = 0.8 ** spidx

                            # Temp lists to hold results from the 3 replicates (f, s, t)
                            tmp_val_acc, tmp_val_loss = [], []
                            tmp_train_acc, tmp_train_loss = [], []
                            
                            # --- 5. Loop Through Replicates (f, s, t) ---
                            for tag in RUN_TAGS:
                                # Construct the expected input filename based on the logic in the experiment scripts
                                prefix_name = f"{method_name}_{scheme_name}_{duration_name}_{concrete_name}_{arch_name}_concrete_evaluation_{tag}"
                                input_filename = INPUT_DIR / f"{prefix_name}_{spidx:02d}.json"

                                try:
                                    with open(input_filename, 'r') as f:
                                        result = json.load(f)[finetune]
                                    
                                    # Append results from this replicate
                                    tmp_val_acc.append(result['val_accuracy'] * 100)
                                    tmp_val_loss.append(result['val_loss'])
                                    tmp_train_acc.append(result['accuracy'] * 100)
                                    tmp_train_loss.append(result['loss'])

                                except FileNotFoundError:
                                    print(f"  WARNING: File not found, skipping: {input_filename}")
                                except KeyError:
                                    print(f"  WARNING: 'finetuned' key missing in {input_filename}")

                            # --- 6. Aggregate and Store Results ---
                            # If no files were found for this sparsity level, continue
                            if not tmp_val_acc:
                                continue
                            
                            # Convert lists to numpy arrays for stats
                            np_val_acc = np.asarray(tmp_val_acc)
                            np_val_loss = np.asarray(tmp_val_loss)
                            np_train_acc = np.asarray(tmp_train_acc)
                            np_train_loss = np.asarray(tmp_train_loss)
                            
                            # Key for the output dictionary is the sparsity percentage
                            key = f"{sparsity_percent * 100:.4f}"

                            output_val_acc[key] = {"mean": np_val_acc.mean(), "std": np_val_acc.std()}
                            output_val_loss[key] = {"mean": np_val_loss.mean(), "std": np_val_loss.std()}
                            output_train_acc[key] = {"mean": np_train_acc.mean(), "std": np_train_acc.std()}
                            output_train_loss[key] = {"mean": np_train_loss.mean(), "std": np_train_loss.std()}
                        
                        # --- 7. Save the Collated JSON Files for this Configuration ---
                        with open(f"{output_file_base}_acc_train.json", 'w') as f:
                            json.dump(output_train_acc, f, indent=4)
                        
                        with open(f"{output_file_base}_acc_val.json", 'w') as f:
                            json.dump(output_val_acc, f, indent=4)

                        with open(f"{output_file_base}_loss_train.json", 'w') as f:
                            json.dump(output_train_loss, f, indent=4)

                        with open(f"{output_file_base}_loss_val.json", 'w') as f:
                            json.dump(output_val_loss, f, indent=4)

                        print(f"  Successfully saved 4 collated files to {output_dir}\n")

print("--- All configurations processed. ---")
