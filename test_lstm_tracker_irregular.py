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
SEQ_LENGTH = 25
HIST_DEPTH = 5
###################################### model parameters
USE_LSTM = True
INPUT_SIZE = 3*HIST_DEPTH
OUTPUT_SIZE = 2
RNN_HIDDEN = 111
###################################### training parameters
BATCH_SIZE = 100
LEARNING_RATE = .01
N_EPOCHS = 10
ITERATIONS_PER_EPOCH = 100
###################################### training
valid_x,valid_y = gen_data_batch(SEQ_LENGTH,100,CURVE_ORDER,CURVE_STEP,HIST_DEPTH)
sample_inp = valid_x[:,:1,:]
sample_lab = valid_y[:,:1,:]
saver = tf.train.import_meta_graph("my-model.meta")
with tf.Session() as session:
  saver.restore(session, "my-model")
  sample_out = session.run("prediction:0",{"inputs:0":sample_inp})
for level_down in range(HIST_DEPTH):
  plt.plot(sample_inp[0,0,0+3*level_down],sample_inp[0,0,1+3*level_down],'k.')
plt.plot(sample_lab[:-1,0,0],sample_lab[:-1,0,1],'k.')
plt.plot(sample_out[:-1,0,0],sample_out[:-1,0,1],'ro',mfc='None')
plt.plot(sample_out[:-1,0,0],sample_out[:-1,0,1],'g')
plt.show()
