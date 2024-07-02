import tensorflow as tf
import keras as K 
from VGG import VGG19
from Helper import get_cifar


strat = tf.distribute.MultiWorkerMirroredStrategy()
with strat.scope():
  '''loss = K.losses.CategoricalCrossentropy()
  #lr_sched = K.optimizers.schedules.ExponentialDecay(0.1, 2450, 0.96)#K.optimizers.schedules.PolynomialDecay(0.03654534632021443, 15680, 0.02814257665, power = 0.5)#
  optim = K.optimizers.LossScaleOptimizer(K.optimizers.SGD(0.1, 0.9, weight_decay = 1e-06), dynamic_growth_steps = 1955) 
  model = VGG19({"L2_Reg": 0.0005, "drop": 0.4, "num_hidden": 0, "crop_size": 200})
  model.compile(loss = loss, optimizer = optim, metrics = ["accuracy"])
  model.call(K.ops.zeros((128, 224, 224, 3)))
  model.initialize_masks()
  #model._mask_list[0].assign(tf.where(tf.random.uniform(model._mask_list[0].shape) < 0.5, tf.ones(model._mask_list[0].shape), tf.zeros(model._mask_list[0].shape)))
  model.train_one(dt, dv, 1, 391, "tmp", strategy = strat)'''
  dt, dv = get_cifar()
  dt, dv = strat.experimental_distribute_dataset(dt), strat.experimental_distribute_dataset(dv)
  for step, (x, y) in enumerate(dt):
    if step > 2:
        if step % 40 == 0: print(step) 
        continue
    tf.print(strat.experimental_local_results(x)[0][:10, 0, 0 ,0])

  for step, (x, y) in enumerate(dt):
    if step > 2:
        if step % 40 == 0: print(step) 
        continue
    tf.print(strat.experimental_local_results(x)[0][:10, 0, 0 ,0])

  for step, (x, y) in enumerate(dt):
    if step > 2:
        if step % 40 == 0: print(step) 
        continue
    tf.print(strat.experimental_local_results(x)[0][:10, 0, 0 ,0])