import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import shutil
import os
from tensorflow.keras.layers import MaxPooling1D, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras import regularizers

import h5py

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


train_file = 'MA0035_4_s5000_train.h5'
test_file = 'MA0035_4_s5000_test.h5'

h5_train = h5py.File(train_file, 'r')
h5_test = h5py.File(test_file, 'r')

train_data = h5_train['data'][:90000]
train_binlabels = h5_train['binlabels'][:90000]

val_data = h5_train['data'][-10000:]
val_binlabels = h5_train['binlabels'][-10000:]

test_data = h5_test['data'][:]
test_binlabels = h5_test['binlabels'][:]

print(train_data.shape)
train_data_T = np.transpose(train_data, axes=(0, 2, 1))
print(train_data_T.shape)

print(test_data.shape)
test_data_T = np.transpose(test_data, axes=(0, 2, 1))
print(test_data_T.shape)

print(val_data.shape)
val_data_T = np.transpose(val_data, axes=(0, 2, 1))
print(val_data_T.shape)

def train_model(hyperparameters, log_dir):
    board_log = log_dir + '/board'
    hparams_log = log_dir + '/hparams.json'

    my_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=board_log),
        tf.keras.callbacks.EarlyStopping('val_loss', patience=200, verbose=1, mode='min', restore_best_weights=True), #Restore best weights option
    ]

    with open(hparams_log, 'w') as hparam_file:
        json.dump(hyperparameters, hparam_file)

    model = tf.keras.models.Sequential(
        [
            Conv1D(320, 26, activation='relu', input_shape=(1000, 4), padding='valid'), #Play with kernel_size
            MaxPooling1D(13, strides=13),
            Dropout(0.2),
            Flatten(),
            Dense(925, activation=tf.keras.activations.relu),
            Dense(919, activation=tf.keras.activations.relu),
            Dense(1, activation=tf.keras.activations.sigmoid)
        ]
    )
    with tf.device('/GPU:0'):
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
            loss = 'binary_crossentropy',
            metrics=['acc', tf.keras.metrics.AUC(name='auc')]
        )

        print(model.summary())

        history = model.fit(train_data_T, train_binlabels,
                epochs=hyperparameters['epochs'], 
                validation_data=(val_data_T, val_binlabels),
                batch_size=hyperparameters['batch_size'],
                callbacks=my_callbacks)

    return model, history


def plot_and_save_roc(false_positive_rate, true_positive_rate, log_dir):
    roc = plt.figure(0)

    plt.plot(false_positive_rate, true_positive_rate, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    roc.savefig(log_dir + '/roc.pdf')
    pickle.dump(roc, open((log_dir + '/pickle/roc.fig.pickle'), 'wb'))
    plt.close(roc)  


def plot_and_save_loss(history, log_dir):
    loss = plt.figure(1)
    history_loss = history.history['loss']
    history_val_loss = history.history['val_loss']
    epochs = range(len(history_loss))

    plt.plot(epochs, history_loss, 'ko', label='Training loss')
    plt.plot(epochs, history_val_loss, 'b', label='Validation loss')
    plt.ylim(0.0)
    plt.title('Training and validation loss')
    plt.legend()

    loss.savefig(log_dir + '/loss.pdf')
    pickle.dump(loss, open((log_dir + '/pickle/loss.fig.pickle'), 'wb'))
    plt.close(loss)

def plot_and_save_accuracy(history, log_dir):
    accuracy = plt.figure(2)
    history_accuracy = history.history['acc']
    history_val_accuracy = history.history['val_acc']
    epochs = range(len(history_accuracy))

    plt.plot(epochs, history_accuracy, 'ko', label='Training accuracy')
    plt.plot(epochs, history_val_accuracy, 'b', label='Validation accuracy')
    plt.ylim([0.0, 1.05])
    plt.title('Training and validation accuracy')
    plt.legend()

    accuracy.savefig(log_dir + '/accuracy.pdf')
    pickle.dump(accuracy, open((log_dir + '/pickle/accuracy.fig.pickle'), 'wb'))
    plt.close(accuracy)


def plot_and_save_auc(history, log_dir):
    auc = plt.figure(3)

    history_auc = history.history['auc']
    history_val_auc = history.history['val_auc']
    epochs = range(len(history_auc))

    plt.plot(epochs, history_auc, 'ko', label='Training auc')
    plt.plot(epochs, history_val_auc, 'b', label='Validation auc')
    plt.ylim([0.5, 1.05])
    plt.title('Training and validation auc')
    plt.ylim(0.46, 1.0)
    plt.legend()

    auc.savefig(log_dir + '/auc.pdf')
    pickle.dump(auc, open((log_dir + '/pickle/auc.fig.pickle'), 'wb'))
    plt.close(auc)


import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 


L0_DROPOUT_RATES = [0.0] #0.5
L1_DROPOUT_RATES = [0.25] #0.4
# L1_HIDDEN_UNITS = [32, 254, 512] #128
# BATCH_SIZE = [32, 64, 128, 254, 512]
# LEARNING_RATE = [0.000001 ,0.000005, 0.000007] #0.000007

# L0_DROPOUT_RATES = [0.0]
# L1_DROPOUT_RATES = [0.0]
L1_HIDDEN_UNITS = [64]
BATCH_SIZE = [32, 64, 128, 254, 512, 1024]
LEARNING_RATE = [0.000003]

for l1_hidden_units in L1_HIDDEN_UNITS:
    for learning_rate in LEARNING_RATE:
        for l0_dropout_rate in L0_DROPOUT_RATES:
            for l1_dropout_rate in L1_DROPOUT_RATES:
                # log_dir = './logs/1_layer_dropout_test_batch/' + 'l1hu' + str(l1_hidden_units) + '_l0dr' + str(l0_dropout_rate) + '_l1dr' + str(l1_dropout_rate) + 'lr' + str(learning_rate)
                # log_dir = './logs/Convoltuion' + 'l1hu' + str(l1_hidden_units) + '_l0dr' + str(l0_dropout_rate) + '_l1dr' + str(l1_dropout_rate) + 'lr' + str(learning_rate)
                
                log_dir = './logs/Convoltuion' + 'test1'

                try:
                    shutil.rmtree(log_dir)
                except OSError as e:
                    print(e)

                try:
                    os.makedirs(log_dir)
                    os.makedirs(log_dir + '/pickle')
                except OSError as e:
                    print(e)

                hparams = {
                    'c1_masks': 200,
                    'c2_masks': 150,
                    'c3_masks': 50,
                    'c4_masks': 50,
                    'c5_masks': 30,
                    'l0_dropout_rate': l0_dropout_rate,
                    'l1_dropout_rate': l1_dropout_rate,
                    'l1_hidden_units': l1_hidden_units,
                    'learning_rate': learning_rate,
                    'batch_size': BATCH_SIZE[3],
                    'epochs': 1500,
                }

                model, history = train_model(hparams, log_dir)

                model.evaluate(test_data_T, test_binlabels)
                yhat = model.predict(test_data_T)
                fpr, tpr, _ = roc_curve(test_binlabels, yhat)
                roc_auc = roc_auc_score(test_binlabels, yhat)

                print('saving')
                plot_and_save_roc(fpr, tpr, log_dir)
                plot_and_save_loss(history, log_dir)
                plot_and_save_accuracy(history, log_dir)
                plot_and_save_auc(history, log_dir)
                model.save(log_dir + '/model')
                print('saved')





