import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint


class LowestLossModelCallback(keras.callbacks.Callback):
    def __init__(self, filepath, verbose=0):
        super(LowestLossModelCallback, self).__init__()
        self.best_model = None
        self.lowest_loss = np.inf
        self.best_accuracy = 0
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs['val_loss']

        if np.less(current_loss, self.lowest_loss):
            self.lowest_loss = current_loss
            self.best_model = self.model
            self.best_accuracy = logs['val_accuracy']

    def on_train_end(self, logs=None):
        filename = self.filepath + '_acc' + str(self.best_accuracy) + '_loss' + str(self.lowest_loss)
        self.best_model.save(filename)
