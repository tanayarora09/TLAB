import json 
import numpy as np 

output_valA = dict()
output_valL = dict()
output_A = dict()
output_L = dict()

sparsity_indexes = [2,4,6,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

reps = 4

for spidx in sparsity_indexes:

    sp = 0.8 ** spidx * 100

    tmpvalA = list()
    tmpvalL = list()
    tmpA = list()
    tmpL = list()

    for rep in range(reps):

        with open(f"BASELINE_RBT_{rep}_{spidx}.json", 'r') as f:
        #with open(f"BASELINE_GraSP{["1_0","2_0","2_1"][rep]}_{spidx}.json", 'r') as f:
            result = json.load(f)

        tmpvalA.append(result['val_accuracy'] * 100)
        tmpvalL.append(result['val_loss'] )
        tmpA.append(result['accuracy'] * 100)
        tmpL.append(result['loss'])


    tmpvalA = np.asarray(tmpvalA)
    tmpvalL = np.asarray(tmpvalL)
    tmpA = np.asarray(tmpA)
    tmpL = np.asarray(tmpL)

    output_valA[f"{sp:.4f}"] = { 'mean': tmpvalA.mean() , 'std': tmpvalA.std() }
    output_valL[f"{sp:.4f}"] = { 'mean': tmpvalL.mean() , 'std': tmpvalL.std() }
    output_A[f"{sp:.4f}"] = { 'mean': tmpA.mean() , 'std': tmpA.std() }
    output_L[f"{sp:.4f}"] = { 'mean': tmpL.mean() , 'std': tmpL.std() }

with open("rbt_acc_train.json", 'w') as f:
    json.dump(output_A, f, indent = 6 )

with open("rbt_acc_val.json", 'w') as f:
    json.dump(output_valA, f, indent = 6 )

with open("rbt_loss_train.json", 'w') as f:
    json.dump(output_L, f, indent = 6 )

with open("rbt_loss_val.json", 'w') as f:
    json.dump(output_valL, f, indent = 6 )