import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
map_fn = tf.map_fn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
###################################### helper functions
def gen_data_batch(seq_length,batch_size,curve_order,curve_step,hist_depth):
  x = np.empty((seq_length,batch_size,3*hist_depth))
  for sample_number in range(batch_size):
    xloc = np.sort(np.random.rand(seq_length)-.5).reshape(seq_length,1)*seq_length/2*curve_step
    yloc = np.hstack([xloc**p for p in range(curve_order+1)]).dot(np.random.randn(curve_order+1,1))
    x[:,sample_number,0] = xloc.squeeze()
    x[:,sample_number,1] = yloc.squeeze()
  x[:,:,2] = np.sqrt(((np.roll(x[:,:,:2],-1,axis=0)-x[:,:,:2])**2).sum(axis=2))
  for level_down in range(1,hist_depth):
    x[:,:,3*level_down:3*(level_down+1)] = np.roll(x[:,:,:3],-1*level_down,axis=0)
  y = np.roll(x[:,:,:2],-hist_depth,axis=0) #future locations
  return x[:-1*hist_depth,:,:], y[:-1*hist_depth,:,:]
###################################### data parameters (for training and validation)
CURVE_ORDER = 2
CURVE_STEP = .5
SEQ_LENGTH = 30
HIST_DEPTH = 2
###################################### model parameters
USE_LSTM = True
INPUT_SIZE = 3*HIST_DEPTH
OUTPUT_SIZE = 2
RNN_HIDDEN = 111
###################################### training parameters
BATCH_SIZE = 100
LEARNING_RATE = .001
N_EPOCHS = 10000
ITERATIONS_PER_EPOCH = 100
###################################### tf graph build-up
inputs  = tf.placeholder(tf.float32, (None,None,INPUT_SIZE), name='inputs')  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None,None,OUTPUT_SIZE)) # (time, batch, out)
cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
initial_state = cell.zero_state(tf.shape(inputs)[1], tf.float32)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)
final_projection = lambda x: layers.fully_connected(x, num_outputs=OUTPUT_SIZE, activation_fn=None)
predicted_outputs = map_fn(final_projection, rnn_outputs)
prediction = tf.identity(predicted_outputs, name='prediction') #dummy tensor
error = tf.reduce_mean((predicted_outputs-outputs)**2)
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
###################################### training
valid_x,valid_y = gen_data_batch(SEQ_LENGTH,100,CURVE_ORDER,CURVE_STEP,HIST_DEPTH)
saver = tf.train.Saver()
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  for epoch in range(N_EPOCHS):
    ###################################### plot a sample network output
    sample_inp = valid_x[:,:1,:]
    sample_out = session.run("prediction:0",{"inputs:0":sample_inp})
    sample_lab = valid_y[:,:1,:]
    plt.clf()
    for level_down in range(HIST_DEPTH):
      plt.plot(sample_inp[0,0,0+3*level_down],sample_inp[0,0,1+3*level_down],'k.')
    plt.plot(sample_lab[:-1,0,0],sample_lab[:-1,0,1],'k.')
    plt.plot(sample_out[:-1,0,0],sample_out[:-1,0,1],'ro',mfc='None')
    plt.plot(sample_out[:-1,0,0],sample_out[:-1,0,1],'g')
    plt.pause(1)
    ###################################### update the network
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
      x,y = gen_data_batch(SEQ_LENGTH,BATCH_SIZE,CURVE_ORDER,CURVE_STEP,HIST_DEPTH)
      epoch_error += session.run([error,train_fn],{inputs:x,outputs:y})[0]
    epoch_error /= ITERATIONS_PER_EPOCH
    valid_error = session.run(error,{inputs:valid_x,outputs:valid_y})
    print("Epoch %d, train error: %.10f, valid error: %.10f"%(epoch,epoch_error,valid_error))
  ###################################### save the trained net
  saver.save(session,'rnntracker')