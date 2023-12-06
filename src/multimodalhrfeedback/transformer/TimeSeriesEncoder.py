from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from .attention.MultiHeadAttention import MultiHeadAttention
from .encoding.PositionalEncoding import PositionalEncoding


class TimeSeriesEncoder(tf.keras.Model, ABC):
    def __init__(self, num_layers, m_model, d_model, w_model,  num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.5):
        super(TimeSeriesEncoder, self).__init__()

        self.d_model = d_model
        self.w_model = w_model
        self.m_model = m_model
        self.num_layers = num_layers

        self.pos_encoding = PositionalEncoding(self.w_model, self.d_model)

        self.enc_layers = [TimeSeriesEncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def build_model(self, inp):
        inputs = tf.keras.Input(shape=inp.shape, name='Model_input_X')

        # Project input from m-space to d-space
        x = layers.Dense(self.d_model, input_shape=(self.m_model,), activation='relu',
                         name='Projection_layer_from_m_to_d')(inputs)

        # BUILD TRANSFORMER ARCHITECTURE
        # positional encoding
        pos_out = self.pos_encoding(x)

        enc_out = []
        # encoder layers
        for i in range(self.num_layers):
            if i == 0:
                enc_out = self.enc_layers[i](pos_out)
            else:
                enc_out = self.enc_layers[i](enc_out)

        # CLASSIFICATION BLOCK
        # concatenation layer
        z = layers.Reshape(target_shape=(self.w_model * self.d_model,), name='Concatenation_Layer')(enc_out)

        # linear layer
        y_hat = layers.Dense(3, input_shape=(self.w_model * self.d_model,), activation='softmax',
                             name='Output_Linear_Layer')(z)

        # BUILDING THE MODEL
        model = tf.keras.Model(inputs=inputs, outputs=y_hat, name='El_Transformer_del_Manolens')

        return model


class TimeSeriesEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TimeSeriesEncoderLayer, self).__init__()
        self.d_model = d_model

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-5)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])
