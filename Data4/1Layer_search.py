import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle
import json
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import BinaryCrossentropy
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 

train_file = 'MA0035_4_s5000_train.h5'
test_file = 'MA0035_4_s5000_test.h5'

h5_train = h5py.File(train_file, 'r')
h5_test = h5py.File(test_file, 'r')

train_data = h5_train['data'][:90000]
train_binlabels = h5_train['binlabels'][:90000]
train_data_T = np.transpose(train_data, axes=(0, 2, 1))

val_data = h5_train['data'][-10000:]
val_binlabels = h5_train['binlabels'][-10000:]
val_data_T = np.transpose(val_data, axes=(0, 2, 1))

test_data = h5_test['data'][:]
test_binlabels = h5_test['binlabels'][:]
test_data_T = np.transpose(test_data, axes=(0, 2, 1))

train_data = tf.cast(train_data, dtype=tf.float32)
train_labels = tf.cast(train_binlabels, dtype=tf.float32)

def plot_and_save_roc(false_positive_rate, true_positive_rate, roc_auc, log_dir, hidden_units):
    roc = plt.figure(0)

    plt.plot(false_positive_rate, true_positive_rate, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(log_dir + '/roc' + '(' + str(hidden_units) + ')' + '.pdf')
    pickle.dump(roc, open((log_dir + '/pickle/roc.fig.pickle'), 'wb'))
    plt.close(roc)

def plot_and_save_loss(history, log_dir, hidden_units):
    loss = plt.figure(1)
    history_loss = history.history['loss']
    history_val_loss = history.history['val_loss']
    epochs = range(len(history_loss))

    plt.plot(epochs, history_loss, 'ko', label='Training loss')
    plt.plot(epochs, history_val_loss, 'b', label='Validation loss')
    plt.ylim([0.0, 1.5])
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(log_dir + '/loss' + '(' + str(hidden_units) + ')' + '.pdf')
    pickle.dump(loss, open((log_dir + '/pickle/loss.fig.pickle'), 'wb'))
    plt.close(loss)

def plot_and_save_auc(history, log_dir, hidden_units):
    auc = plt.figure(2)

    history_auc = history.history['auc']
    history_val_auc = history.history['val_auc']
    epochs = range(len(history_auc))

    plt.plot(epochs, history_auc, 'ko', label='Training auc')
    plt.plot(epochs, history_val_auc, 'b', label='Validation auc')
    plt.title('Training and validation auc')
    plt.ylim(0.46, 1.0)
    plt.legend()

    plt.savefig(log_dir + '/auc' + '(' + str(hidden_units) + ')' + '.pdf')
    pickle.dump(auc, open((log_dir + '/pickle/auc.fig.pickle'), 'wb'))
    plt.close(auc)

def train_model(hyperparameters, log_dir):
    board_log = log_dir + '/board'
    hparams_log = log_dir + '/hparams.json'

    my_callbacks = [
        TensorBoard(log_dir=board_log),
        EarlyStopping('val_loss', patience=35, mode='min')
    ]

    with open(hparams_log, 'w') as hparam_file:
        json.dump(hyperparameters, hparam_file)

    if hyperparameters['optimizer'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hyperparameters['learning_rate'])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate'])

    model = tf.keras.models.Sequential(
        [
            Flatten(input_shape=(1000, 4)),
            Dropout(hyperparameters['l0_dr']),
            Dense(hyperparameters['l1_hu'], activation=tf.keras.activations.relu, kernel_regularizer='l2'),
            Dropout(hyperparameters['l1_dr']),
            Dense(1, activation=tf.keras.activations.sigmoid),
        ]
    )

    with tf.device('/GPU:0'):
        model.compile(
            optimizer = optimizer,
            loss = tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name='auc')],
        )

        history = model.fit(train_data_T, train_binlabels,
                epochs=hyperparameters['epochs'], 
                validation_data=(val_data_T, val_binlabels),
                batch_size=hyperparameters['batch_size'],
                callbacks=my_callbacks,
                shuffle=True)

    return model, history

OPTIMIZERS = ['rmsprop', 'adam']
learning_rate = 0.0000025
# dr = [0.00, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60]
dr = [0.00]

for optimizer in OPTIMIZERS:
    for i in range(1000, 10000, 500):
        for j in dr:
            log_dir = 'logs/log1layer_l2v2/' + str(optimizer) + '/lr='+ str(learning_rate) + '/' + str(i) + '[' + str(j) + ']'

            hparams = {
                'l0_dr': 0.1,
                'l1_hu': i,
                'l1_dr': j,
                'learning_rate': learning_rate,
                'batch_size': 1024,
                'epochs': 1000,
                'optimizer': optimizer,
            }

            try:
                os.makedirs(log_dir)
                os.makedirs(log_dir + '/pickle')
            except OSError as e:
                print(e)

            model, history = train_model(hparams, log_dir)

            yhat = model.predict(test_data_T)
            fpr, tpr, _ = roc_curve(test_binlabels, yhat)
            roc_auc = roc_auc_score(test_binlabels, yhat)

            plot_and_save_roc(fpr, tpr, roc_auc, log_dir, i)
            plot_and_save_loss(history, log_dir, i)
            plot_and_save_auc(history, log_dir, i)

        # model.save(log_dir + '/model')  #Too heavy for long logs