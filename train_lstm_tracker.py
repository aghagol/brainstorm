import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
map_fn = tf.map_fn

def generate_batch(seq_length, batch_size):
    CURVE_ORDER = 3
    x = np.empty((num_bits, batch_size, 2))
    y = np.empty((num_bits, batch_size, 2))
    for batch_number in batch_size:
        xloc = np.arange(seq_length).reshape(seq_length,1)-(seq_length/2)
        yloc = np.hstack([x**p for p in range(CURVE_ORDER+1)]).dot(np.random.randn(CURVE_ORDER+1,1))
        x[:,batch_number,0] = xloc
        x[:,batch_number,1] = yloc
    y = np.roll(x,-1,axis=0) #future locations
    return x, y

INPUT_SIZE    = 2       # 2 bits per timestep
RNN_HIDDEN    = 20      # dimention of hidden variable?
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01
USE_LSTM = True

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)
cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
batch_size    = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)
predicted_outputs = map_fn(final_projection, rnn_outputs)
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)
# session = tf.Session()
session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

for epoch in range(10):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. error and accuracy on their
        # own do not trigger the backprop.
        x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
    epoch_error /= ITERATIONS_PER_EPOCH
    valid_accuracy = session.run(accuracy, {
        inputs:  valid_x,
        outputs: valid_y,
    })
    print("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0))
