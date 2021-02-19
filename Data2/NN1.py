import h5py
import tensorflow as tf
from tensorflow import keras

train_file = 'GATA_train.h5'
h5_train = h5py.File(train_file, 'r')

train_data = h5_train['data']
train_labels = h5_train['labels']
print(train_data.shape)

train_data = train_data[:1000]
train_labels = train_labels[:1000]

train_data = tf.cast(train_data, dtype=tf.float32)
train_labels = tf.cast(train_labels, dtype=tf.float32)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(17, activation=tf.keras.activations.linear)
    ]
)

model.compile(
    optimizer = tf.optimizers.Adam(),
    loss = tf.keras.losses.mse
)
print('train data shape: ', train_data.shape)
print('train labels shape ', train_labels.shape)

# model.build(train_data)

# print(model.summary())

model.fit(train_data, train_labels)
# model.summary()
