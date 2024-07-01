import tensorflow as tf
import keras as K 
import keras.layers as lyr # type: ignore

class LotteryDense(lyr.Layer):

    def __init__(
        self,
        units,
        kernel_initializer = "glorot_normal",
        kernel_regularizer = None,
        activation = None,
        name = None
        ):
        super(LotteryDense, self).__init__(name = name)
        self.units = units
        self.activation = K.activations.get(activation)
        self.initializer = K.initializers.get(kernel_initializer)
        self.regularizer = K.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units), 
            initializer = self.initializer,
            regularizer = self.regularizer,
            trainable=True,
            name="kernel",
            dtype = tf.float32
        )
        b_init = K.initializers.zeros()
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer=b_init,
            trainable=True,
            name="bias",
            dtype = tf.float32
        )
        m_init = K.initializers.ones()
        self._mask = self.add_weight(
            shape=(input_dim, self.units),
            initializer=m_init,
            trainable=False,
            name= self.kernel.name + "_mask",
            dtype = tf.float32
        )
        self.built = True

    def call(self, inputs):
        masked_kernel = self.kernel*self._mask
        output = inputs @ masked_kernel + self.bias
        if self.activation: output = self.activation(output)

        return output   


class LotteryConv2D(lyr.Layer):

    def __init__(
        self,
        filters,
        kernel_size = (3, 3),
        strides = (1, 1),
        padding = 'same',
        kernel_regularizer = None,
        kernel_initializer = "glorot_normal",
        name = None
    ):
        super(LotteryConv2D, self).__init__(name = name)
        self.kernel_size = kernel_size
        self.n_filters = filters
        self.stride = strides
        self.padding = padding.upper()
        self.initializer = K.initializers.get(kernel_initializer)
        self.regularizer = K.regularizers.get(kernel_regularizer)
    
    def build(self, input_shape):
        
        filter_dims = self.kernel_size + (input_shape[-1], self.n_filters)
        self.kernel = self.add_weight(
            shape = filter_dims,
            initializer = self.initializer,
            regularizer = self.regularizer,
            trainable = "True",
            name = "kernel",
            dtype = tf.float32
        )

        b_init = K.initializers.zeros()
        self.bias = self.add_weight(
            shape=(self.n_filters,),
            initializer=b_init,
            trainable=True,
            name="bias",
            dtype = tf.float32
        )
        m_init = K.initializers.ones()
        self._mask = self.add_weight(
            shape=filter_dims,
            initializer=m_init,
            trainable=False,
            name= self.kernel.name + "_mask",
            dtype = tf.float32
        )
        self.built = True


    def call(self, inputs):
        masked_kernel = self.kernel*self._mask
        conv = K.ops.conv(inputs=inputs, 
                        kernel=masked_kernel, 
                        strides=self.stride,
                        padding=self.padding)
        b_shape = (1, 1, 1, self.n_filters)
        feat_map = conv+K.ops.reshape(self.bias, b_shape) 
        return feat_map


class LotteryBatchNormalization(lyr.Layer):
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        name=None
    ):
        super(LotteryBatchNormalization, self).__init__(name=name)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = K.initializers.get(beta_initializer)
        self.gamma_initializer = K.initializers.get(gamma_initializer)
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer

    def build(self, input_shape):
        dim = input_shape[self.axis]

        self.beta = self.add_weight(
                shape=(dim,),
                initializer=self.beta_initializer,
                trainable=True,
                autocast = False,
                name="beta",
            dtype = tf.float32
        )

        self.gamma = self.add_weight(
            shape=(dim,),
            initializer=self.gamma_initializer,
            trainable=True,
            autocast = False,
            name="gamma",
            dtype = tf.float32
        )

        self.moving_mean = self.add_weight(
            shape=(dim,),
            initializer=self.moving_mean_initializer,
            trainable=False,
            name="moving_mean",
            dtype = tf.float32
        )

        self.moving_variance = self.add_weight(
            shape=(dim,),
            initializer=self.moving_variance_initializer,
            trainable=False,
            name="moving_variance",
            dtype = tf.float32
        )

        self._beta_mask = self.add_weight(
            shape=(dim,),
            initializer=K.initializers.ones(),
            trainable=False,
            name=self.beta.name + "_mask",
            dtype = tf.float32
        )

        self._gamma_mask = self.add_weight(
            shape=(dim,),
            initializer=K.initializers.ones(),
            trainable=False,
            name=self.gamma.name + "_mask",
            dtype = tf.float32
        )
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        self._reduction_axes = reduction_axes
        self.built = True

    def call(self, inputs, training=None):

        if training:
            mean, variance = K.ops.moments(
                inputs,
                axes=self._reduction_axes,
                synchronized=False,
            )
            moving_mean = tf.cast(self.moving_mean, inputs.dtype)
            moving_variance = tf.cast(self.moving_variance, inputs.dtype)
            self.moving_mean.assign(
                moving_mean * self.momentum + mean * (1.0 - self.momentum)
            )
            self.moving_variance.assign(
                moving_variance * self.momentum
                + variance * (1.0 - self.momentum)
            )
        else:
            moving_mean = tf.cast(self.moving_mean, inputs.dtype)
            moving_variance = tf.cast(self.moving_variance, inputs.dtype)
            mean = moving_mean
            variance = moving_variance


        outputs = K.ops.batch_normalization(
            x=inputs,
            mean=mean,
            variance=variance,
            axis=self.axis,
            offset=self.beta * self._beta_mask,
            scale=self.gamma * self._gamma_mask,
            epsilon=self.epsilon,
        )


        return outputs


        