import tensorflow as tf
import keras as K 

from Helper import * 
from VGG import VGG19

strat = tf.distribute.MultiWorkerMirroredStrategy()
name = "Resize_Original_Params_Higher_WD"

with strat.scope():

    dt, dv = get_cifar()
    dt, dv = strat.experimental_distribute_dataset(dt), strat.experimental_distribute_dataset(dv)

    loss = K.losses.CategoricalCrossentropy()
    #lr_sched = K.optimizers.schedules.ExponentialDecay(0.1, 2450, 0.96)#K.optimizers.schedules.PolynomialDecay(0.03654534632021443, 15680, 0.02814257665, power = 0.5)#
    optim = K.optimizers.LossScaleOptimizer(K.optimizers.SGD(0.01, 0.9, weight_decay = 0.00625), dynamic_growth_steps = 1955) 

    model = VGG19()
    model.compile(loss = loss, optimizer = optim, metrics = ["accuracy"])

    model.call(K.ops.zeros((128, 224, 224, 3)), training = False)

    logs = model.train_one(dt, dv, 160, cardinality = int(dt.cardinality), name = name, strategy = strat)

    logs_to_pickle(logs, name)

    plot_logs(logs, 160, name = name)

    model.save_vars_to_file(f"./WEIGHTS/final_model_{name}")