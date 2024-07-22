import keras as K 
import keras.layers as lyr 
import tensorflow as tf

class ResBlock(K.models.Model):

    def __init__(self, channels, down = False):
        super().__init__()
        self.__channels = channels
        self.__down = down
        self.__strides = [2,1] if down else [1,1]
        self.__kernel = (3,3)
        self.conv1 = lyr.Conv2D(self.__channels, strides = self.__strides[0],
                                kernel_size = self.__kernel, padding = "same")
        self.bn1 = lyr.BatchNormalization()
        self.conv2 = lyr.Conv2D(self.__channels, strides = self.__strides[1],
                                kernel_size = self.__kernel, padding = "same")
        self.bn2 = lyr.BatchNormalization()
        self.merge = lyr.Add()
        if self.__down:
            self.res_conv = lyr.Conv2D(self.__channels, strides = 2, 
                                       kernel_size = (1,1), padding = "same")
            self.res_bn = lyr.BatchNormalization()

    def call(self, ins):
        res = ins
        
        x = self.conv1(ins)
        x = self.bn1(x)
        x = K.ops.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.__down:
            res = self.res_conv(res)
            res = self.res_bn(res)

        x = self.merge([x, res])
        return K.ops.relu(x)

class ResNet18Large(K.models.Model):

    def __init__(self, classes = 10):
        super().__init__()

        self.conv1 = lyr.Conv2D(64, (7,7), strides = 2, padding = "same")
        self.initbn = lyr.BatchNormalization()
        self.pool1 = lyr.MaxPool2D((2, 2), strides = 2, padding = "same")
        self.res_1_1 = ResBlock(64)
        self.res_1_2 = ResBlock(64)
        self.res_2_1 = ResBlock(128, True)
        self.res_2_2 = ResBlock(128)
        self.res_3_1 = ResBlock(256, True)
        self.res_3_2 = ResBlock(256)
        self.res_4_1 = ResBlock(512, True)
        self.res_4_2 = ResBlock(512)
        self.res_blocks = [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, 
                           self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]
        self.avg_pool = lyr.GlobalAveragePooling2D()
        self.flat = lyr.Flatten()
        self.out = lyr.Dense(classes, activation = "softmax")

        self._mask = [K.ops.ones_like(w) for w in self.trainable_variables]
        self._rewind_weights = None

    def call(self, ins):
        x = self.conv1(ins)
        x = self.initbn(x)
        x = K.ops.relu(x)
        x = self.pool1(x)
        for resBlock in self.res_blocks:
            x = resBlock(x)
        x = self.avg_pool(x)
        x = self.flat(x)
        return self.out(x)
    
    def functional_rep(self):
        ins = K.Input((32,32,3))
        return K.Model(ins, self.call(ins))
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training = True)
            loss = self.compute_loss(y = y, y_pred = y_pred)
            loss = self.optimizer.scale_loss(loss)
            
        weights = self.trainable_variables
        grads = [g*m for g,m in zip(tape.gradient(loss, weights), self._mask)]

        self.optimizer.apply_gradients(zip(grads, weights))
        
        return self.compute_metrics(x, y, y_pred)
    
    def prune(self, k):
        all_weights = K.ops.concatenate([ K.ops.reshape(w, [-1]) for w in self.trainable_variables], axis = 0)
        threshold = K.ops.quantile(K.ops.abs(all_weights), k)
        self._mask = [tf.cast(K.ops.abs(w) > threshold, tf.float32) for w in self.trainable_variables]

    def save_rewind_weights(self):
        self._rewind_weights = [w.numpy() for w in self.trainable_variables]

    def rewind_weights(self):
        for var, initial in zip(self.trainable_variables, self._rewind_weights):
            var.assign(initial)

    def save_weights_iter(self, fp, epoch, iter):
        self.save_weights(f"{fp}_epoch{epoch}_iter{iter}.ckpt")


