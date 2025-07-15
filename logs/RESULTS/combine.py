import json 
import numpy as np 
import math

output_valA = dict()
output_valL = dict()
output_A = dict()
output_L = dict()

out_name = "vgg16/loss_concrete_long_finetuned"
prefix_name = "LONG_LOSSCONCRETE_VGG"

sparsity_indexes = [2,4,6,8,10,12,14,16,17,18,19,20,21,22,23,24,25] #0,
#sparsity_indexes = [2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22] # RESNET
#sparsity_indexes = [21, 22]

is_adding = False

#sparsity_indexes = list(range(26))

reps = 3

for spidx in sparsity_indexes:

    sp = 0.8 ** spidx * 100

    tmpvalA = list()
    tmpvalL = list()
    tmpA = list()
    tmpL = list()

    for rep in range(reps):
        name = f"{prefix_name}_{rep}_{spidx}.json"
        with open(name, 'r') as f:
            print(f"Opened {name}")
        #with open(f"imp_vgg16_{rep}_{spidx}.json", 'r') as f:
            result = json.load(f)["results"]["finetuned"]

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