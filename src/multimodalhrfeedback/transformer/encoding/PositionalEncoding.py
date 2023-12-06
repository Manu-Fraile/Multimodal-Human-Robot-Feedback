import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, w_model, d_model):
        super(PositionalEncoding, self).__init__()

        self.dropout = tf.keras.layers.Dropout(rate=0.5, name='Encoding_Dropout')
        self.learnable_encoder = tf.random.uniform(shape=[w_model, d_model], minval=-0.02, maxval=0.02)

    def __call__(self, x, *args, **kwargs):
        x = tf.math.add(x, self.learnable_encoder, name='Sum_Encoding_to_Input')
        return self.dropout(x)
