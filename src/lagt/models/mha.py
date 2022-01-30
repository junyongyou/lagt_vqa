import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense,
    Layer,
    Input)
from tensorflow.keras import Model


class MultiHeadAttention(Layer):
    """
    This is the standard multi-head attention layer
    """
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(
                f'embedding dimension = {d_model} should be divisible by number of heads = {num_heads}'
            )
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.d_model

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = matmul_qk / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def call(self, inputs):
        query = inputs
        key = inputs
        value = inputs
        batch_size = tf.shape(query)[0]

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention, weights = self.scaled_dot_product_attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.d_model)
        )
        output = self.dense(concat_attention)
        output = tf.reduce_mean(output, axis=1)
        return output


if __name__ == '__main__':
    input_shape = (16, 5 * 256)
    inputs = Input(shape=input_shape)
    local_mha = MultiHeadAttention(d_model=256, num_heads=4)
    x = local_mha(inputs)
    model = Model(inputs=inputs, outputs=x)
    model.summary()

    input_x = np.random.randn(1, 16, 1280)
    out = model(input_x)
    t = 0