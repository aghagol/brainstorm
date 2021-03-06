import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
map_fn = tf.map_fn
import matplotlib.pyplot as plt
###################################### helper functions
def generate_batch(seq_length,batch_size,curve_order,curve_step,hist_depth):
  x = np.empty((seq_length, batch_size, 2*hist_depth))
  for batch_number in range(batch_size):
    xloc = (np.arange(seq_length).reshape(seq_length,1)-(seq_length/2))*curve_step
    yloc = np.hstack([xloc**p for p in range(curve_order+1)]).dot(np.random.randn(curve_order+1,1))
    x[:,batch_number,0] = xloc.squeeze()
    x[:,batch_number,1] = yloc.squeeze()
  for level_down in range(1,hist_depth):
    x[:,:,2*level_down:2*level_down+2] = np.roll(x[:,:,:2],-1*level_down,axis=0)
  y = np.roll(x[:,:,:2],-hist_depth,axis=0) #future locations
  return x[:-1*hist_depth,:,:], y[:-1*hist_depth,:,:]
###################################### data parameters (for training and validation)
CURVE_ORDER = 2
CURVE_STEP = .1
SEQ_LENGTH = 20
HIST_DEPTH = 5
###################################### model parameters
USE_LSTM = True
INPUT_SIZE = 2*HIST_DEPTH
OUTPUT_SIZE = 2
RNN_HIDDEN = 20
###################################### training parameters
BATCH_SIZE = 100
LEARNING_RATE = .01
N_EPOCHS = 1000
ITERATIONS_PER_EPOCH = 100
###################################### tf graph build-up
inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)
cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
batch_size = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)
final_projection = lambda x: layers.fully_connected(x, num_outputs=OUTPUT_SIZE, activation_fn=None)
predicted_outputs = map_fn(final_projection, rnn_outputs)
error = tf.reduce_mean((predicted_outputs-outputs)**2)
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
###################################### training
valid_x,valid_y = generate_batch(SEQ_LENGTH,100,CURVE_ORDER,CURVE_STEP,HIST_DEPTH)
session = tf.Session()
session.run(tf.initialize_all_variables())
for epoch in range(N_EPOCHS):
  ###################################### plot a sample network output
  sample_inp = valid_x[:,:1,:]
  sample_out = session.run(predicted_outputs,{inputs:sample_inp})
  sample_lab = valid_y[:,:1,:]
  plt.clf()
  for level_down in range(HIST_DEPTH):
    plt.plot(sample_inp[0,0,0+2*level_down],sample_inp[0,0,1+2*level_down],'k.')
  plt.plot(sample_lab[:-1,0,0],sample_lab[:-1,0,1],'k.')
  plt.plot(sample_out[:-1,0,0],sample_out[:-1,0,1],'ro',mfc='None')
  plt.pause(1)
  ###################################### update the network
  epoch_error = 0
  for _ in range(ITERATIONS_PER_EPOCH):
    x,y = generate_batch(SEQ_LENGTH,BATCH_SIZE,CURVE_ORDER,CURVE_STEP,HIST_DEPTH)
    epoch_error += session.run([error,train_fn],{inputs:x,outputs:y})[0]
  epoch_error /= ITERATIONS_PER_EPOCH
  valid_error = session.run(error,{inputs:valid_x,outputs: valid_y})
  print("Epoch %d, train error: %.10f, valid error: %.10f"%(epoch,epoch_error,valid_error))
###################################### save the trained net
