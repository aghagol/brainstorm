import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
map_fn = tf.map_fn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
VERBOSE = True
###################################### helper functions
def gen_data_batch(seq_length,batch_size,curve_order,curve_step,obs_depth,point_dim):
  x = np.zeros((seq_length,batch_size,point_dim*obs_depth))
  for sample_number in range(batch_size):
    xloc = np.sort(np.random.rand(seq_length)-.5).reshape(seq_length,1)*seq_length*curve_step
    yloc = np.hstack([xloc**p for p in range(curve_order+1)]).dot(np.random.randn(curve_order+1,1))
    x[:,sample_number,0] = xloc.squeeze()
    x[:,sample_number,1] = yloc.squeeze()
  dt_next = np.sqrt(((np.roll(x[:,:,:2],-1,axis=0)-x[:,:,:2])**2).sum(axis=2))
  x[:,:,point_dim-1] = dt_next
  if point_dim==5:
    dt_prev = np.sqrt(((np.roll(x[:,:,:2],+1,axis=0)-x[:,:,:2])**2).sum(axis=2))
    x[:,:,2] = (np.roll(x[:,:,0],+1,axis=0)-x[:,:,0]) /dt_prev *dt_next #linear (naive) estimation using previous observation
    x[:,:,3] = (np.roll(x[:,:,1],+1,axis=0)-x[:,:,1]) /dt_prev *dt_next #linear (naive) estimation using previous observation
  for step_back in range(1,obs_depth):
    x[:,:,point_dim*step_back:point_dim*(step_back+1)] = np.roll(x[:,:,:point_dim],-step_back,axis=0)
  y = np.roll(x[:,:,:2],-obs_depth,axis=0) #future locations
  return x[1:-obs_depth,:,:], y[1:-obs_depth,:,:]
###################################### data parameters (for training and validation)
CURVE_ORDER = 3
CURVE_STEP = .5
SEQ_LENGTH = 30
POINT_DIM = 5 #number of dimensions per point (x,y,t,...)
OBS_DEPTH = 1 #observation depth (how many points back in time are observed?)
###################################### model parameters
USE_LSTM = True
INPUT_SIZE = POINT_DIM*OBS_DEPTH
OUTPUT_SIZE = 2
RNN_HIDDEN = 25
NUM_LAYERS = 2
###################################### training parameters
BATCH_SIZE = 100
LEARNING_RATE = .001
N_EPOCHS = 100000
ITERATIONS_PER_EPOCH = 100
###################################### tf graph build-up
# create place holders for input, output and some run-time parameters
inputs  = tf.placeholder(tf.float32, (None,None,INPUT_SIZE), name='inputs')  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None,None,OUTPUT_SIZE)) # (time, batch, out)
out_keep_prob = tf.placeholder(tf.float32, name='out_keep_prob')
# make a list of LSTM layers
cells_list = [tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True) for _ in range(NUM_LAYERS)]
# cells_with_dropout = [tf.contrib.rnn.DropoutWrapper(cell=each_cell, output_keep_prob=out_keep_prob) for each_cell in cells_list]
cells_with_dropout = cells_list
# build a deep LSTM network
cell = tf.contrib.rnn.MultiRNNCell(cells=cells_with_dropout, state_is_tuple=True)
initial_state = cell.zero_state(tf.shape(inputs)[1], tf.float32)
rnn_output, rnn_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)
fc1 = lambda x: layers.fully_connected(x, num_outputs=OUTPUT_SIZE, activation_fn=None)
fc1_output = map_fn(fc1, rnn_output)
prediction = tf.identity(fc1_output, name='prediction') #dummy tensor
# add cost layer for optimization
error = tf.reduce_mean((fc1_output-outputs)**2)
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
###################################### training
valid_x,valid_y = gen_data_batch(SEQ_LENGTH,100,CURVE_ORDER,CURVE_STEP,OBS_DEPTH,POINT_DIM)
saver = tf.train.Saver()
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  try:
    for epoch in range(N_EPOCHS):
      ###################################### plot a sample network output
      if VERBOSE:
        sample_inp = valid_x[:,:1,:]
        sample_out = session.run("prediction:0",{"inputs:0":valid_x,"out_keep_prob:0":1})[:,:1,:]
        sample_lab = valid_y[:,:1,:]
        plt.clf()
        for step_back in range(OBS_DEPTH):
          plt.plot(sample_inp[0,0,POINT_DIM*step_back],sample_inp[0,0,1+POINT_DIM*step_back],'b.')
        plt.plot(sample_lab[:-1,0,0],sample_lab[:-1,0,1],'k.')
        plt.plot(sample_out[:-1,0,0],sample_out[:-1,0,1],'ro',mfc='None')
        plt.plot(sample_out[:-1,0,0],sample_out[:-1,0,1],'g')
        plt.pause(1)
      ###################################### update the network
      epoch_error = 0
      for _ in range(ITERATIONS_PER_EPOCH):
        x,y = gen_data_batch(SEQ_LENGTH,BATCH_SIZE,CURVE_ORDER,CURVE_STEP,OBS_DEPTH,POINT_DIM)
        epoch_error += session.run([error,train_fn],{inputs:x,outputs:y,out_keep_prob:1})[0]
      epoch_error /= ITERATIONS_PER_EPOCH
      valid_error = session.run(error,{inputs:valid_x,outputs:valid_y,out_keep_prob:1})
      print("Epoch %d, train error: %.10f, valid error: %.10f"%(epoch,epoch_error,valid_error))
    ###################################### save the trained net
    saver.save(session,'rnntracker-polyn%d'%(CURVE_ORDER))
  except KeyboardInterrupt:
    print('\nSaving results in rnntracker-polyn%d.tmp*'%(CURVE_ORDER))
    saver.save(session,'rnntracker-polyn%d.tmp'%(CURVE_ORDER))