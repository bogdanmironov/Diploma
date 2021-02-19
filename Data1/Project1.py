import h5py                                    
import tensorflow as tf
import numpy as np
from tensorflow import keras                                                                                           

h5 = h5py.File("MA0035_4_subst_train.h5")
data = h5["data"]
# data = data[:]  Copies into                        
labels = h5["labels"]                     
# labels = labels[:]
print("type is", type(data[0][0]))

data = data[:1000]
labels = labels[:1000]

data = tf.cast(data, dtype=tf.float32)
labels = tf.cast(labels, dtype=tf.float32)

# labels = [x / 100 for x in labels]
# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(4, 1000)), 
model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(128, 3, activation='relu'),
                                    tf.keras.layers.Dense(32), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(500)]) #Smaller number
                                    #Softmax doesnt work; You need to guess a number!!! RESEARCH! #500

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'mean_squared_error', #New loss!! mean square error
            #   metrics=['accuracy']
              )

print(model.summary())
model.fit(data, labels)


h5_test = h5py.File("MA0035_4_subst_test.h5")

test_data = h5_test['data']
test_labels = h5_test['labels']

# model.evaluate(test_data, test_labels)       