import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
import tensorflow.keras.losses
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss

from multimodalhrfeedback.cnn import InceptionTimeClassifier
from multimodalhrfeedback.randomforests import RandomForestClassifier
from multimodalhrfeedback.transformer import TimeSeriesEncoder
from multimodalhrfeedback.transformer.callbacks import LowestLossModelCallback
from multimodalhrfeedback.visualisation.Plotter import plot_loss, plot_accuracy

"""
GENERAL PARAMETERS
    :param TRAIN: True trains a model. False loads a model.
    :param MODEL_NAME: When loading model, name of such.
    :param DL_MODEL: transformer, inceptiontime, randomforest
    :param INPUT_X_DATA: Input X for Deep Learning models.
    :param INPUT_Y_DATA: Labels Y for Deep Learning models.
    :param RF_INPUT_X_DATA: Input X for Random Forest Classifier.
    :param RF_INPUT_Y_DATA: Labels Y for Random Forest Classifier.
"""
TRAIN = False
MODEL_NAME = 'randomforest'
DL_MODEL = 'randomforest'
INPUT_X_DATA = '../data/custom-data/input_x.npy'
INPUT_Y_DATA = '../data/custom-data/input_y.npy'
RF_INPUT_X_DATA = '../data/custom-data/RFinput_x.npy'
RF_INPUT_Y_DATA = '../data/custom-data/RFinput_y.npy'

"""
DEEP LEARNING TRAINING PARAMETERS
    :param LEARNING_RATE
    :param EPOCHS
    :param BATCH_SIZE
"""
LEARNING_RATE = 0.0001
EPOCHS = 19
BATCH_SIZE = 128

"""
TIME SERIES ENCODER PARAMETERS
    :param NUM_LAYERS
    :param NUM_HEADS
    :param M: Input dimensionality.
    :param W: Input length.
    :param D: Transformer dimensionality.  
"""
NUM_LAYERS = 1
NUM_HEADS = 16
M = 38
W = 410
D = 64

"""
INCEPTIONTIME PARAMETERS
    :param NUM_CLASSES
"""
NUM_CLASSES = 3


def main():
    now = datetime.now()
    filename_nowdate = now.strftime("%Y%m%d_%H%M%S")

    if (DL_MODEL == 'transformer') or (DL_MODEL == 'inceptiontime'):
        x = np.load(INPUT_X_DATA, allow_pickle=True)
        y = np.load(INPUT_y_DATA, allow_pickle=True)

        x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.6, shuffle=True, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5, shuffle=True, random_state=42)
    else:
        x = np.load(RF_INPUT_X_DATA, allow_pickle=True)
        y = np.load(RF_INPUT_Y_DATA, allow_pickle=True)

        x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.6, shuffle=True, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5, shuffle=True, random_state=42)

    if DL_MODEL == 'transformer':
        filename_nowdate = 'TS_' + filename_nowdate

        if TRAIN:
            encoderTransformer = TimeSeriesEncoder(num_layers=NUM_LAYERS, m_model=M, d_model=D, w_model=W,
                                                   num_heads=NUM_HEADS,
                                                   dff=256, input_vocab_size=30000, maximum_position_encoding=80,
                                                   rate=0.1)

            model = encoderTransformer.build_model(inp=x[0])
            print("\n")
            print(model.summary())
            print("\n")

            print("\n\nTRAINING...")
            optimizer = Adam(learning_rate=LEARNING_RATE)
            loss = CategoricalCrossentropy(from_logits=False)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mse', Precision(), Recall()])

            x_train = tf.convert_to_tensor(x_train)
            y_train = tf.convert_to_tensor(y_train)
            x_val = tf.convert_to_tensor(x_val)
            y_val = tf.convert_to_tensor(y_val)

            filepath = '../results/tsencoder/models/' + filename_nowdate
            extract_lowloss = LowestLossModelCallback(filepath)
            fit_report = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                   validation_data=(x_val, y_val), callbacks=[extract_lowloss])

            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, "../results/tsencoder/plots/" + filename_nowdate)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            plot_loss(fit_report, save=True, filename="../results/tsencoder/plots/" + filename_nowdate + "/loss.png")
            plot_accuracy(fit_report, save=True,
                          filename="../results/tsencoder/plots/" + filename_nowdate + "/accuracy.png")

        else:
            model = tf.keras.models.load_model('../results/final_results/tsencoder/models/' + MODEL_NAME)

        test_loss, test_acc, test_mse, test_prec, test_rec = model.evaluate(x=x_test, y=y_test, batch_size=64)
        f1score = (2 * (test_prec * test_rec)) / (test_prec + test_rec)

        predictions = model.predict(x=x_test, batch_size=64)
        msle = tf.keras.losses.MeanSquaredLogarithmicError()
        prediction_loss = log_loss(y_test, predictions)

        print("\n\n\nPREDICTIONS")
        print(prediction_loss)
        print("\n\n\n")

        print('+++++   TRANSFORMER ON TEST   +++++')
        print('\nThe test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc) +
              '\nThe test mse: ' + str(test_mse) + '\nThe test f1score: ' + str(f1score))

        print('\n\n+++++   TRANSFORMER PREDICTIONS   +++++')
        y_pred = model.predict(x=x_test)
        y_pred_maxed = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)
        print(y_pred_maxed)

        with open('../results/tsencoder/predictions/tsencoder.npy', 'wb') as f:
            np.save(f, y_pred_maxed)

    elif DL_MODEL == 'inceptiontime':
        filename_nowdate = 'IT_' + filename_nowdate

        if TRAIN:
            mcp_save = tf.keras.callbacks.ModelCheckpoint("../results/inceptiontime/models/" + filename_nowdate + '.h5',
                                                          save_best_only=True, monitor='val_loss', mode='min')

            ITclassifier = InceptionTimeClassifier()
            metrics = ['accuracy', 'mse', Precision(), Recall()]
            ITmodel = ITclassifier.build_model(input_shape=(W, M), nb_classes=NUM_CLASSES, learning_rate=LEARNING_RATE,
                                               metrics=metrics)
            ITreport = ITmodel.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                   validation_data=(x_val, y_val), callbacks=mcp_save)

            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, "../results/inceptiontime/plots/" + filename_nowdate)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

            plot_loss(ITreport, save=True, filename="../results/inceptiontime/plots/" + filename_nowdate + "/loss.png")
            plot_accuracy(ITreport, save=True,
                          filename="../results/inceptiontime/plots/" + filename_nowdate + "/accuracy.png")
        else:
            ITmodel = tf.keras.models.load_model('../results/inceptiontime/models/' + MODEL_NAME)

        test_loss, test_acc, test_mse, test_prec, test_rec = ITmodel.evaluate(x=x_test, y=y_test)
        f1score = (2 * (test_prec * test_rec)) / (test_prec + test_rec)

        print('+++++   INCEPTIONTIME ON TEST   +++++')
        print('The test loss: ' + str(test_loss) + '\nThe test accuracy: ' + str(test_acc) +
              '\nThe test mse: ' + str(test_mse) + '\nThe test f1score: ' + str(f1score))

        print('\n\n+++++   INCEPTIONTIME PREDICTIONS   +++++')
        y_pred = ITmodel.predict(x=x_test)
        y_pred_maxed = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)
        print(y_pred_maxed)

        with open('../results/inceptiontime/predictions/inceptiontime.npy', 'wb') as f:
            np.save(f, y_pred_maxed)

    else:
        filename = "../results/rfc/models/randomforest.pickle"
        if TRAIN:
            clf = RandomForestClassifier(max_depth=10000, verbose=2)
            clf.fit(x_train, y_train, save=True, filename=filename)
        else:
            clf = RandomForestClassifier(model=pickle.load(open(filename, "rb")))

        accuracy = clf.accuracy(x_test, y_test)
        y_pred = clf.predictions(x_test)

        print('\n\n+++++   RANDOM FORESTS PREDICTIONS   +++++')
        print(y_pred)

        cce_loss = clf.log_loss(y_test, y_pred)
        f1score = clf.f1score(y_test, y_pred)

        print('The test accuracy: ' + str(accuracy) + '\nThe test f1score: ' + str(f1score) + '\nThe test loss: ' + str(
            cce_loss))

        with open('../results/rfc/predictions/randomforest.npy', 'wb') as f:
            np.save(f, y_pred)


if __name__ == "__main__":
    main()
