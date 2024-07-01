import tensorflow as tf
import keras as K 

from Helper import * 
from VGG import VGG19

strat = tf.distribute.MultiWorkerMirroredStrategy()
name = "Resize_71_Params"

with strat.scope():

    dt, dv = get_cifar()
    dt, dv = strat.experimental_distribute_dataset(dt), strat.experimental_distribute_dataset(dv)

    loss = K.losses.CategoricalCrossentropy()
    lr_sched = K.optimizers.schedules.PolynomialDecay(0.03654534632021443, 62560, 0.02814257665, power = 2.6)
    optim = K.optimizers.LossScaleOptimizer(K.optimizers.SGD(lr_sched, 0.6, weight_decay = 1.6e-06), dynamic_growth_steps = 1955) 
    
    model = VGG19()
    model.compile(loss = loss, optimizer = optim, metrics = ["accuracy"])

    model.call(K.ops.zeros((128, 224, 224, 3)), training = False)

    logs = model.train_one(dt, dv, 95, cardinality = int(dt.cardinality), name = name, strategy = strat)

    logs_to_pickle(logs, name)

    plot_logs(logs, 160, name = name)

    model.save_vars_to_file(f"./WEIGHTS/final_model_{name}")