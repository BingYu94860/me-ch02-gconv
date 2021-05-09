import tensorflow as tf
import numpy as np
import scipy.sparse as sp

class BaseDense(tf.keras.layers.Dense):
    def add_weight_kernel(self, shape, name='kernel', trainable=True):
        return self.add_weight(name=name,
                               shape=shape,
                               trainable=trainable,
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint,
                               dtype=self.dtype)

    def add_weight_bias(self, shape, name='bias', trainable=True):
        return self.add_weight(name=name,
                               shape=shape,
                               trainable=trainable,
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint,
                               dtype=self.dtype)

class GraphConvs(BaseDense):
    def __init__(self, units, DADs, name=None, **kwargs):
        super(GraphConvs, self).__init__(units=units, name=name, **kwargs)
        self.DADs = tf.convert_to_tensor(DADs)
        self.num_K = len(DADs)  # K

    def build(self, input_shape):
        self.num_Fin = input_shape[-1]  # Fin
        self.num_Fout = self.units  # Fout
        self.kernel = self.add_weight_kernel([self.num_K, self.num_Fin, self.num_Fout])
        self.bias = self.add_weight_bias([self.num_Fout,]) if self.use_bias else None
        self.built = True

    def call(self, inputs):
        X = tf.tensordot(inputs, self.DADs, axes=([1], [-1]))
        X = tf.tensordot(X, self.kernel, axes=([1, 2], [1, 0]))
        if self.use_bias:
            X = tf.add(X, self.bias)
        outputs = self.activation(X)
        return outputs