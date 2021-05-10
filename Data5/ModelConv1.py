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


train_file = 'GATA_train_3000_100.h5'
test_file = 'GATA_test_3000_010.h5'

h5_train = h5py.File(train_file, 'r')
h5_test = h5py.File(test_file, 'r')


def get_model(hyperparameters, log_dir):
    hparams_log = log_dir + '/hparams.json'

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

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
        loss = 'binary_crossentropy',
        metrics=['acc', tf.keras.metrics.AUC(name='auc')]
    )

    return model


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


def plot_and_save_loss(history_loss, history_val_loss, log_dir):
    loss = plt.figure(1)
    epochs = range(len(history_loss))

    plt.plot(epochs, history_loss, 'ko', label='Training loss')
    plt.plot(epochs, history_val_loss, 'b', label='Validation loss')
    plt.ylim(0.0)
    plt.title('Training and validation loss')
    plt.legend()

    loss.savefig(log_dir + '/loss.pdf')
    pickle.dump(loss, open((log_dir + '/pickle/loss.fig.pickle'), 'wb'))
    plt.close(loss)

def plot_and_save_accuracy(history_accuracy, history_val_accuracy, log_dir):
    accuracy = plt.figure(2)

    epochs = range(len(history_accuracy))

    plt.plot(epochs, history_accuracy, 'ko', label='Training accuracy')
    plt.plot(epochs, history_val_accuracy, 'b', label='Validation accuracy')
    plt.ylim([0.0, 1.05])
    plt.title('Training and validation accuracy')
    plt.legend()

    accuracy.savefig(log_dir + '/accuracy.pdf')
    pickle.dump(accuracy, open((log_dir + '/pickle/accuracy.fig.pickle'), 'wb'))
    plt.close(accuracy)


def plot_and_save_auc(history_auc, history_val_auc, log_dir):
    auc = plt.figure(3)

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
    'learning_rate': 0.000003,
    'batch_size': 254,
    'epochs': 1500,
}

model = get_model(hparams, log_dir)



print(model.summary())

my_callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir + '/board'),
    tf.keras.callbacks.EarlyStopping('val_loss', patience=200, verbose=1, mode='min', restore_best_weights=True), #Restore best weights option
]

train_loss = None
train_val_loss = None
train_auc = None
train_val_auc = None
train_acc = None
train_val_acc = None
curr = 0
for i in range(100000, 1000001, 100000):
    prev = curr
    curr = i
    val_threshold = curr - 10000

    train_data = h5_train['data'][prev : val_threshold]
    train_binlabels = h5_train['binlabels'][prev : val_threshold]
    train_binlabels = [label[0] for label in train_binlabels]

    val_data = h5_train['data'][val_threshold : curr]
    val_binlabels = h5_train['binlabels'][val_threshold : curr]
    val_binlabels = [label[0] for label in val_binlabels]

    train_data = tf.cast(train_data, dtype=tf.float32)
    train_binlabels = tf.cast(train_binlabels, dtype=tf.float32)

    val_data = tf.cast(val_data, dtype=tf.float32)
    val_binlabels = tf.cast(val_binlabels, dtype=tf.float32)

    print(train_data.shape)
    train_data_T = np.transpose(train_data, axes=(0, 2, 1))
    print(train_data_T.shape)

    print(val_data.shape)
    val_data_T = np.transpose(val_data, axes=(0, 2, 1))
    print(val_data_T.shape)

    history = model.fit(train_data_T, train_binlabels,
            epochs=hparams['epochs'], 
            validation_data=(val_data_T, val_binlabels),
            batch_size=hparams['batch_size'],
            callbacks=my_callbacks)

    train_loss.append(history.history['loss'])
    train_val_loss.append(history.history['val_loss'])
    train_acc.append(history.history['acc'])
    train_val_acc.append(history.history['val_acc'])
    train_auc.append(history.history['auc'])
    train_val_auc.append(history.history['val_auc'])




test_data = h5_test['data'][:]
test_binlabels = h5_test['binlabels'][:]
test_binlabels = [label[0] for label in test_binlabels]
test_data = tf.cast(test_data, dtype=tf.float32)
test_binlabels = tf.cast(test_binlabels, dtype=tf.float32)


print(test_data.shape)
test_data_T = np.transpose(test_data, axes=(0, 2, 1))
print(test_data_T.shape)

model.evaluate(test_data_T, test_binlabels)
yhat = model.predict(test_data_T)
fpr, tpr, _ = roc_curve(test_binlabels, yhat)
roc_auc = roc_auc_score(test_binlabels, yhat)

print('saving')
plot_and_save_roc(fpr, tpr, log_dir)
plot_and_save_loss(train_loss, train_val_loss, log_dir)
plot_and_save_accuracy(train_acc, train_val_acc, log_dir)
plot_and_save_auc(train_auc, train_val_auc, log_dir)
model.save(log_dir + '/model')
print('saved')





