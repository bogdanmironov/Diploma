import h5py
import tensorflow as tf
from tensorflow import keras

file_predictions = "predictions.txt"
train_file = 'MA0035_4_m5_train.h5'
h5_train = h5py.File(train_file, 'r')

train_data = h5_train['data']
train_labels = h5_train['labels']

print(train_data.shape)

val_data = train_data[-1000:]
val_labels = train_labels[-1000:]
val_labels = [label[0] for label in val_labels]

val_data = tf.cast(val_data, dtype=tf.float32)
val_labels = tf.cast(val_labels, dtype=tf.float32)

train_data = train_data[:1000]
train_labels = train_labels[:1000]
train_labels = [label[0] for label in train_labels]

train_data = tf.cast(train_data, dtype=tf.float32)
train_labels = tf.cast(train_labels, dtype=tf.float32)

model = tf.keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=(4, 1000)),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
    ]
)

model.compile(
    optimizer = tf.optimizers.Adam(learning_rate=0.00012),
    loss = tf.keras.losses.mse,
    metrics= ['accuracy']
)
print('train data shape: ', train_data.shape)
print('train labels shape ', train_labels.shape)

# model.build(train_data)

# print(model.summary())

history = model.fit(train_data, train_labels, epochs=50, validation_data = (val_data, val_labels))

test_file = h5py.File('MA0035_4_m5_test.h5', 'r')
test_data = test_file['data']
test_labels = test_file['labels']

print(test_data.shape)

test_data = test_data[:1000]
test_labels = test_labels[:1000]
test_labels = [label[0] for label in test_labels]

print("test data:", end = "  ")
print(test_data[0].shape)
print("test label: ", end = " ")
print(test_labels[0])

test_data = tf.cast(test_data, dtype=tf.float32)
test_labels = tf.cast(test_labels, dtype=tf.float32)

eval_history = model.evaluate(test_data, test_labels)
# f = open(file_predictions, "w")

# print(model.summary())
# print(model.predict(test_data[1]))
# print(test_labels[1])

# for i in range(1000):
#     print(i)
#     f.write("prediction - " + model.predict(test_data[i]) + "; label - " + test_labels[i])

# f.close()
#model.predict(test_data) SAVE THAT!   LOSS IN EVALUATE IS NOT PERCISE ENOUGH
#TRY BINLABES; EASIER... probably

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# model.summary()
