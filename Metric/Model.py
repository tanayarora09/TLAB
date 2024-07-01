import keras as K
import keras.api.layers as lyr
from keras import ops
import tensorflow as tf

class Lenet_300_100(K.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = lyr.Dense(300, activation = "relu",
                                kernel_initializer = K.initializers.GlorotNormal())
        self.dense2 = lyr.Dense(100, activation = "relu", kernel_initializer = K.initializers.GlorotNormal())
        self.dense3 = lyr.Dense(10, activation = "softmax")

    def call(self, x):
        x = lyr.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training = True)
            loss = self.compute_loss(y = y, y_pred = y_pred)
        
        weights = self.trainable_variables
        gradients = tape.gradient(loss, weights)
        
        self.optimizer.apply(gradients, weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def functionalrep(self):
        input = K.Input((28,28,1))
        return K.Model(input, self.call(input))


class BaseModel(K.Model):
    
    def __init__(self):
        super().__init__()
        self.conv1 = lyr.Conv2D(128, 3, activation='relu') 
        self.conv2 = lyr.Conv2D(128, 3, activation='relu') 
        self.conv3 = lyr.Conv2D(128, 3, activation='relu') 
        self.conv4 = lyr.Conv2D(64, 3, activation='relu') 
        self.pool = lyr.MaxPooling2D() 
        self.dense1 = lyr.Dense(300, activation='relu') 
        self.dense2 = lyr.Dense(200, activation='relu') 
        self.out = lyr.Dense(10, activation = 'softmax') 
        self.bn1 = lyr.BatchNormalization()
        self.bn2 = lyr.BatchNormalization() 
        self.daug = K.Sequential([
            lyr.RandomZoom(0.2),
            lyr.RandomTranslation(0.2, 0.2),
            lyr.Resizing(28, 28)
        ])
        
    
    def call(self, x, training = False):
        if training: x = self.daug(x)
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.pool(x)
        x = self.bn1(x) 
        x = self.conv3(x) 
        x = self.conv4(x) 
        x = self.pool(x) 
        x = self.bn2(x) 
        x = lyr.Flatten()(x) 
        x = self.dense1(x) 
        x = self.dense2(x) 
        return self.out(x)

    #Implement Training
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training = True)
            loss = self.compute_loss(y = y, y_pred = y_pred)
        
        weights = self.trainable_variables
        gradients = tape.gradient(loss, weights)
        
        self.optimizer.apply(gradients, weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}


    #Implement Inference
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training = False)
        loss = self.compute_loss(y = y, y_pred = y_pred)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def functionalrep(self):
        input = K.Input((28,28,1))
        return K.Model(input, self.call(input))