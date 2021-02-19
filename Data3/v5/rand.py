import h5py
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.version.VERSION)

file_predictions = 'predictions.txt'
train_file = 'MA0035_4_m5_train.h5'
test_file = 'MA0034_4_m5_test.h5'


h5_train = h5py.File(train_file, 'r')


train_data = h5_train['data']
train_binlabels = h5_train['binlabels']

train_data = train_data[:2000]
train_binlabels = train_binlabels[:2000]

train_data = tf.cast(train_data, dtype=tf.float32)
train_binlabels = tf.cast(train_binlabels, dtype=tf.float32)


print(train_data.shape)
print(train_binlabels.shape)
print(train_binlabels[0])


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)
    ]
)


model.compile(
    optimizer = tf.optimizers.Adam(),
    loss = tf.keras.losses.mse,
    metrics=['accuracy']
)


print(train_data.shape)
print(train_data[0])
train_data_reshaped = tf.reshape(train_data, [2000, 4000])
print(train_data_reshaped.shape)
print(train_data_reshaped[0])
test_data_reshaped = train_data_reshaped[-1000:]
train_data_reshaped = train_data_reshaped[:1000]
test_binlabels_reshaped = train_binlabels[-1000:]
train_binlabels_reshaped = train_binlabels[:1000]


history = model.fit(train_data_reshaped, train_binlabels_reshaped, epochs=100, verbose=1)


print(model.summary())


# i = 0
# print(train_data[i].numpy())
# print(train_binlabels[i].numpy()) 
# print(train_data_reshaped.shape)


# print(test_data_reshaped[0].shape)


print(test_data_reshaped[0])
model.predict(test_data_reshaped[0])



