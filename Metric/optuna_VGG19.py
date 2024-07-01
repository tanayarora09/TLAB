import keras as K 
import tensorflow as tf

from Helper import plot_logs, logs_to_pickle, create_optimizer_args, get_cifar
from VGG import VGG19

import glob 
import os

import optuna

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    def objective(trial, dt, dv):
        loss = K.losses.CategoricalCrossentropy()

        args = create_optimizer_args(trial)

        name = 'T_' + str(list(args.values())).strip(" ")#f"SGD_optuna_expdecay_losscale_{args}"
        epochs = 95

        lr_sched = K.optimizers.schedules.ExponentialDecay(args["init_lr"], 782, 0.96) #PolynomialDecay(0.01, decay_steps = 1960, end_learning_rate=0.0005, power = 0.5, cycle = True)
        optim = K.optimizers.LossScaleOptimizer(K.optimizers.SGD(lr_sched, momentum=args["momentum"], weight_decay=args["weight_decay"]), dynamic_growth_steps = 490) #K.optimizers.Adam(0.1, weight_decay = 0.0005, clipnorm = 2.0) 
        
        model = VGG19(args)

        model.compile(loss = loss, optimizer = optim, metrics = ["accuracy"])

        logs, epoch = model.train_one(dt, dv, epochs, int(dt.cardinality), name, strategy=strategy)

        if epoch < epochs:
            for fp in glob.glob(f'./WEIGHTS/*{name}*.h5'): 
                print(fp)
                os.remove(fp)
            raise optuna.TrialPruned()

        trial.set_user_attr("LOGS", {key:{k:float(v.numpy()) for k, v in value.items()} for key, value in logs.items()})
        trial.set_user_att("ARGS_NAME", name)

        del model

        return logs[epoch * int(dt.cardinality)]["val_accuracy"]
    
    dt, dv = get_cifar()
    dt, dv = strategy.experimental_distribute_dataset(dt), strategy.experimental_distribute_dataset(dv)

    study = optuna.load_study(study_name = "VGG19_trials_224", storage='sqlite:///resize_tuning_VGG19.db')
    study.optimize(lambda trial: objective(trial, dt, dv), n_trials = 64, gc_after_trial = True)#150, gc_after_trial = True)#120)

    trials = study.get_trials(states = [optuna.trial.TrialState.COMPLETE])

    print(f"Finished {len(trials)} trials.")

    print("Best trial: ")

    trial = study.best_trial

    print("Val_Accuracy: ", trial.value)

    print(" Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    logs_to_pickle(trial.user_attrs["LOGS"], "best_trial_optuna")

    logs = trial.user_attrs["LOGS"]

    plot_logs(logs, 95, "best_optuna")

    with open('PARAMS.txt', 'w') as f:
        f.write(str(trial.params.items()))

    for tri in trials:
        print(trial)    
        plot_logs(tri.user_attrs["LOGS"], 95, tri.user_attrs["ARGS_NAME"])