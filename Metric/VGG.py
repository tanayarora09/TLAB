import keras as K 
import keras.api.layers as lyr 
from LotteryLayers import * 
import tensorflow as tf
import numpy as np 
from collections import defaultdict, deque
import time
import h5py

class VGG19(K.Model):

    # 160 Epochs, SGD with 0.01 lr (decrease 10x at 80 and 120 epochs), 0.9 momentum, rewind to 100, conv20%
    # batchnorm, weight decay, augmentation
    #0.1 lr diverges to nan

    #FINAL L_R = 0.01, MOMENTUM = 0.9, WEIGHT_DECAY = 0.000125

    def __init__(self, args):
        super().__init__()

        self.L2_RATE = args["L2_Reg"]

        self.data_augmentation = K.Sequential([
            lyr.RandomCrop(args["crop_size"], args["crop_size"]), 
            lyr.RandomZoom(0.2),
            lyr.RandomRotation(0.13),
            lyr.RandomContrast(0.3),
            lyr.Resizing(224, 224)
        ], name = "Data_Augmentation") 
        
        self.conv11 = LotteryConv2D(64, (3,3), padding="same", kernel_initializer='glorot_normal', name="block1_conv1", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.relu11 = lyr.ReLU(name = "block1_relu1")
        self.conv12 = LotteryConv2D(64, (3,3), padding="same", kernel_initializer='glorot_normal', name="block1_conv2", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.bn11 = LotteryBatchNormalization(name = "block1_norm1")
        self.relu12 = lyr.ReLU(name = "block1_relu2")
        self.pool1 = lyr.MaxPooling2D((2, 2), strides=(2,2), name="block1_pool")
        
        self.conv21 = LotteryConv2D(128, (3,3), padding="same", kernel_initializer='glorot_normal', name="block2_conv1", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.relu21 = lyr.ReLU(name = "block2_relu1")
        self.conv22 = LotteryConv2D(128, (3,3), padding="same", kernel_initializer='glorot_normal', name="block2_conv2", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.bn21 = LotteryBatchNormalization(name = "block2_norm1")
        self.relu22 = lyr.ReLU(name = "block2_relu2")
        self.pool2 = lyr.MaxPooling2D((2, 2), strides=(2,2), name="block2_pool")
        
        self.conv31 = LotteryConv2D(256, (3,3), padding="same", kernel_initializer='glorot_normal', name="block3_conv1", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.relu31 = lyr.ReLU(name = "block3_relu1")
        self.conv32 = LotteryConv2D(256, (3,3), padding="same", kernel_initializer='glorot_normal', name="block3_conv2", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.bn31 = LotteryBatchNormalization(name = "block3_norm1")
        self.relu32 = lyr.ReLU(name = "block3_relu2")
        self.conv33 = LotteryConv2D(256, (3,3), padding="same", kernel_initializer='glorot_normal', name="block3_conv3", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.relu33 = lyr.ReLU(name = "block3_relu3")
        self.conv34 = LotteryConv2D(256, (3,3), padding="same", kernel_initializer='glorot_normal', name="block3_conv4", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.bn32 = LotteryBatchNormalization(name = "block3_norm2")
        self.relu34 = lyr.ReLU(name = "block3_relu4")
        self.pool3 = lyr.MaxPooling2D((2, 2), strides=(2,2), name="block3_pool")
        
        self.conv41 = LotteryConv2D(512, (3,3), padding="same", kernel_initializer='glorot_normal', name="block4_conv1", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.relu41 = lyr.ReLU(name = "block4_relu1")
        self.conv42 = LotteryConv2D(512, (3,3), padding="same", kernel_initializer='glorot_normal', name="block4_conv2", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.bn41 = LotteryBatchNormalization(name = "block4_norm1")
        self.relu42 = lyr.ReLU(name = "block4_relu2")
        self.conv43 = LotteryConv2D(512, (3,3), padding="same", kernel_initializer='glorot_normal', name="block4_conv3", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.relu43 = lyr.ReLU(name = "block4_relu3")
        self.conv44 = LotteryConv2D(512, (3,3), padding="same", kernel_initializer='glorot_normal', name="block4_conv4", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.bn42 = LotteryBatchNormalization(name = "block4_norm2")
        self.relu44 = lyr.ReLU(name = "block4_relu4")
        self.pool4 = lyr.MaxPooling2D((2, 2), strides=(2,2), name="block4_pool")

        self.conv51 = LotteryConv2D(512, (3,3), padding="same", kernel_initializer='glorot_normal', name="block5_conv1", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.relu51 = lyr.ReLU(name = "block5_relu1")
        self.conv52 = LotteryConv2D(512, (3,3), padding="same", kernel_initializer='glorot_normal', name="block5_conv2", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.bn51 = LotteryBatchNormalization(name = "block5_norm1")
        self.relu52 = lyr.ReLU(name = "block5_relu2")
        self.conv53 = LotteryConv2D(512, (3,3), padding="same", kernel_initializer='glorot_normal', name="block5_conv3", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.relu53 = lyr.ReLU(name = "block5_relu3")
        self.conv54 = LotteryConv2D(512, (3,3), padding="same", kernel_initializer='glorot_normal', name="block5_conv4", kernel_regularizer=K.regularizers.L2(self.L2_RATE))
        self.bn52 = LotteryBatchNormalization(name = "block5_norm2")
        self.relu54 = lyr.ReLU(name = "block5_relu4")
        self.pool5 = lyr.MaxPooling2D((2, 2), strides=(2,2), name="block5_pool")

        self.avg = lyr.GlobalAveragePooling2D(name="GAP_FC")
        self.FCs = [(LotteryBatchNormalization(name = f"norm_{i}"), LotteryDense(args[f"hidden_{i}"], activation = "relu", name = f"FC_{i}", kernel_regularizer=K.regularizers.L2(self.L2_RATE))) for i in range(args["num_hidden"])]
        self.drop1 = lyr.Dropout(args["drop"], name = "Dropout_1")
        self.bno = lyr.BatchNormalization(name = "norm_out")
        self.out = lyr.Dense(10, activation="softmax", name="FC_OUT", kernel_regularizer=K.regularizers.L2(self.L2_RATE))

        self._mask_list = None
        self._mask_to_kernel = None
        self._rewind = None
        self._init = None
        self._iterations = [] # if first train, save grads at each step every 3rd epoch
        self._grads = None
        self._activation_patterns = None

        self.strategy = None

    def call(self, x, training = False):
        if training: x = self.data_augmentation(x)
        x = self.pool1(self.relu12(self.bn11(self.conv12(self.relu11(self.conv11(x))))))
        x = self.pool2(self.relu22(self.bn21(self.conv22(self.relu21(self.conv21(x))))))
        x = self.pool3(self.relu34(self.bn32(self.conv34(self.relu33(self.conv33(self.relu32(self.bn31(self.conv32(self.relu31(self.conv31(x)))))))))))
        x = self.pool4(self.relu44(self.bn42(self.conv44(self.relu43(self.conv43(self.relu42(self.bn41(self.conv42(self.relu41(self.conv41(x)))))))))))
        x = self.pool5(self.relu54(self.bn52(self.conv54(self.relu53(self.conv53(self.relu52(self.bn51(self.conv52(self.relu51(self.conv51(x)))))))))))
        x = self.avg(x)
        for BN, FC in self.FCs:
            x = FC(BN(x))
        x = self.drop1(x, training = training)
        return self.out(self.bno(x))

    def build(self, input_shape):
        super(VGG19, self).build(input_shape)
        self.initialize_masks()
        self._iterations.append(self._mask_list)

    def functional_rep(self):
        ins = K.Input((224,224,3))
        return K.Model(ins, self.call(ins))

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
            reg_loss = tf.reduce_sum(self.losses)
            total_loss = loss + reg_loss
            scaled_loss = self.optimizer.scale_loss(total_loss)

        weights = self.trainable_variables
        grads = tape.gradient(scaled_loss, weights)
        #save grads
        masked_grads = grads # assert grads same mask as masks

        #if save_grad: self._grads.append(masked_grads)
        #else: {Compute L2 Diff and return}

        self.optimizer.apply_gradients(zip(masked_grads, weights))

        out = self.compute_metrics(x, y, y_pred)

        out.update({'loss': total_loss, 'learning_rate': self.optimizer.learning_rate})

        return out

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(x, y, y_pred)
        reg_loss = tf.reduce_sum(self.losses)
        total_loss = loss + reg_loss
        out = self.compute_metrics(x, y, y_pred)
        out.update({'loss': total_loss})
        return out

    @tf.function
    def distributed_train_step(self, dataset_inputs):
        per_replica_losses = self.strategy.run(self.train_step, args=(dataset_inputs, ))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(self, dataset_inputs):
        per_replica_losses = self.strategy.run(self.test_step, args=(dataset_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    def train_one(self, dt, dv, epochs, cardinality, name, first = True, strategy = None):
        """'''
        LOAD: [load_bool, epochs_trained, logs_of_orig_train]
        '''"""
        if strategy:
            self.strategy = strategy

        logs = defaultdict(dict)
        self.call(K.ops.zeros((1,224,224,3)), training = False) # Force build
        self.save_init(name = name) 

        train_log = {}
        val_log = {}
        best_val_loss = 2**16
        last_update = 0
        save_grad = False 
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}")
            print(f"Learning rate: {self.optimizer.learning_rate.numpy()}")
            if first: save_grad = True if (epoch % 3 == 0) else False  
            #if epoch + 1 == 80 or epoch + 1 == 120:
            #    self.optimizer.learning_rate.assign(self.optimizer.learning_rate / 10)
            start_time = time.time()
            for step, (x, y) in enumerate(dt):
                iteration = int(epoch * cardinality + step + 1)
                train_log = self.distributed_train_step((x, y), save_grad)
                logs[iteration] = train_log
                if iteration % 24 == 0 and iteration != 0:
                  print(f"Status at {(iteration - epoch * cardinality) // 24}/16; Accuracy (accumulative): {(logs[iteration]['accuracy'])}, Loss (step): {(logs[iteration]['loss'])}")
                  if iteration  == 1250: # iteration 5000
                      self.save_rewind(name = name)

            self.reset_metrics()

            for step, (x, y) in enumerate(dv):
                val_log = self.distributed_test_step((x, y))
            
            logs[(epoch + 1) * cardinality].update({f"val_{key}": value for key, value in val_log.items()})

            print(f"EPOCH {epoch + 1}: ", {key: value.numpy() for key, value in logs[(epoch + 1) * cardinality].items()})
            
            if logs[(epoch + 1) * cardinality]["val_loss"] < best_val_loss:
                best_val_loss = logs[(epoch + 1) * cardinality]["val_loss"]
                last_update = epoch
                print(f"\nUPDATING BEST WEIGHTS TO {epoch + 1}\n")
                self.save_vars_to_file(f"./WEIGHTS/best_val_loss_{name}")

            self.reset_metrics()
            
            if (epoch - last_update) == 12:
                return logs, epoch + 1

            print(f"TIME: {time.time() - start_time}")

        return logs, epochs

    def train_IMP(self, dt, dv, epochs, prune_rate, iters, cardinality, name, strategy = None):

        """
        dt : Train_data
        dv: Validation_data
        epochs: Epochs_per_train_cycle
        prune_rate: pruning rate (i.e. if you want to remove 20% of weights per iterations, prune_rate = 0.8)
        iters: iterations to prune. prune_rate ** iters == final_sparsity
        cardinality: cardinality of the train data
        name: name of experiment - will log in output files
        strategy: multi-gpu strategy
        """

        self.call(K.ops.zeros((1,224,224,3)), training = False) # Force build weights if not already
        self.initialize_custom_metrics()
        self.save_init(name) 
        
        logs = defaultdict(dict)
        iter = 0
        step_logs = self.train_one(dt, dv, epochs, cardinality, name, strategy = strategy) # train full model first, so loop ends with train
        logs[iter] = step_logs
        for iter in range(1, iters):
            self.prune_by_rate_mg(prune_rate, iter)
            self.load_weights_from_obj_or_file(name, rewind = True)
            self.train_one(dt, dv, epochs, cardinality, name, strategy = strategy, first = False)
        return logs, self._mask

    def prune_by_rate_mg(self, rate, iter):
        all_weights = K.ops.concatenate([K.ops.reshape(self.mask_to_kernel(m) * m, [-1]) for m in self._mask_list], axis = 0)
        threshold = K.ops.quantile(K.ops.abs(all_weights), 1.0 - rate**iter)
        for m in self._mask_list:
            m.assign(tf.cast(K.ops.abs(self._mask_to_kernel(m)) > threshold, dtype = tf.float32))

    def initialize_custom_metrics(self):
        self._grads = deque(maxlen = 54) # 160 // 3 + 1 (for 0 epoch)
        self._activation_patterns = defaultdict()
        # LOGIC HERE

    def initialize_masks(self):
        if self._mask_list and self._mask_to_kernel: return
        self._mask_list = [var for var in self.variables if var.name[-5:] == "_mask"] # list of masks
        self._mask_to_kernel = {} #points from mask to weight
        for weight in self.trainable_variables:
            if (weight.name != "bias" and weight.path[:6] != "FC_OUT"):
                self._mask_to_kernel[weight.path + "_mask"] = weight 

    def save_rewind(self, name):
        if self._rewind: return
        self.save_vars_to_file(f"./WEIGHTS/rewind_{name}")
        self._rewind = self.return_vars()
    
    def save_init(self, name):
        if self._init: return
        self.save_vars_to_file(f"./WEIGHTS/init_{name}")
        self._init = self.return_vars()

    def return_vars(self):
        return [w.numpy() for w in self.variables]

    def save_vars_to_file(self, name):
      with h5py.File(name + ".h5", 'w') as f:
        for layer in self.layers:
          for weight in layer.weights:
            if weight.name[-5:] == "_mask": continue
            f.create_dataset(weight.path, data=weight.numpy())

    def load_vars_from_obj_or_file(self, rewind = True, name = None):
        if name:
            with h5py.File(name + ".h5", 'r') as f:
                for layer in self.layers:
                    for weight in layer.weights:
                        if weight.path in f:
                            weight_value = f[weight.path][:]
                            try:
                                weight.assign(weight_value)
                            except:
                                print("Error with loading " + weight.path)
            return          
        elif rewind:
            for var, initial in zip(self.variables, self._rewind):
                if var.name[-5:] == "_mask": continue
                var.assign(initial)
            return
        for var, initial in zip(self.variables, self._init):
            if var.name[-5:] == "_mask": continue
            var.assign(initial)

